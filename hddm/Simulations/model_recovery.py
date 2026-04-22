"""Model recovery for PainReward HDDM models.

Tests whether the data-generating model can be identified by DIC:
  1. For each source version, draw true group parameters from its empirical posterior
  2. Simulate a dataset
  3. Fit all candidate models to that dataset
  4. Record which model wins (lowest DIC) per rep
  5. Output a confusion matrix: rows = true model, cols = winning model

By default tests versions 9, 10, 11, 19 (the theoretically meaningful behavioural
models). Override with --versions.

Usage:
    python model_recovery.py [--versions 9 10 19] [--n-reps 10]
                              [--n-chains 4] [--samples 2000] [--burn 200]

Env vars: PROJECT_DIR, MODEL_DIR
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import arviz as az
import hddm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from model_specs import MODEL_BASE_NAME, get_param_list, get_formula_terms, build_model

PROJECT_DIR = Path(os.getenv("PROJECT_DIR", str(Path(__file__).resolve().parent.parent.parent))).resolve()
MODEL_DIR   = Path(os.getenv("MODEL_DIR",   str(PROJECT_DIR / "derivatives" / "hddm" / "models"))).resolve()
INPUT_CSV   = PROJECT_DIR / "derivatives" / "behav" / "hddm_ready.csv"

# Default candidate models for model recovery
DEFAULT_VERSIONS = [9, 10, 11, 19]


# ---------------------------------------------------------------------------
# Posterior helpers (same logic as param_recovery.py)
# ---------------------------------------------------------------------------

def _random_draw(idata, param: str, rng: np.random.Generator) -> float:
    if param not in idata.posterior:
        raise KeyError(
            f"'{param}' not in posterior. "
            f"Available: {list(idata.posterior.data_vars)}"
        )
    da    = idata.posterior[param]
    chain = int(rng.integers(da.sizes["chain"]))
    draw  = int(rng.integers(da.sizes["draw"]))
    return float(da.isel(chain=chain, draw=draw))


def extract_group_params(idata, param_list: list, rng: np.random.Generator) -> dict:
    return {p: _random_draw(idata, p, rng) for p in param_list}


def extract_group_sds(idata, param_list: list, rng: np.random.Generator) -> dict:
    out = {}
    for p in param_list:
        sd_name = f"{p}_std"
        out[p] = _random_draw(idata, sd_name, rng) if sd_name in idata.posterior else 0.0
    return out


def sample_true_subjects(mu: dict, sd: dict, subjects: list,
                          rng: np.random.Generator) -> dict:
    true = {}
    for s in subjects:
        pars = {}
        for p in mu:
            sigma = sd.get(p, 0.0)
            pars[p] = float(rng.normal(mu[p], sigma)) if sigma > 0 else float(mu[p])
        true[s] = pars
    return true


# ---------------------------------------------------------------------------
# Simulation (mirrors param_recovery.py)
# ---------------------------------------------------------------------------

def simulate_dataset(true_individuals: dict, raw_df: pd.DataFrame,
                     source_version: int) -> pd.DataFrame:
    formula_terms = get_formula_terms(source_version)
    regressed_dvs  = set(formula_terms.keys())
    rows = []
    df = raw_df.copy()
    df["subj_idx"] = df["subj_idx"].astype(str).str.replace(r"^\D+", "", regex=True).astype(int)

    for _, tr in df.iterrows():
        subj = int(tr["subj_idx"])
        pars = true_individuals[subj]
        par_dict = {}

        for dv, terms in formula_terms.items():
            val = 0.0
            for param_name, cols in terms:
                x = 1.0 if len(cols) == 0 else float(np.prod([tr[c] for c in cols]))
                val += pars[param_name] * x
            par_dict[dv] = val

        for p in ("a", "v", "t", "z"):
            if p not in regressed_dvs and p in pars:
                par_dict[p] = float(pars[p])

        trial_df, _ = hddm.generate.gen_rand_data(par_dict, size=1, subjs=1)
        for col in df.columns:
            if col not in ("rt", "response"):
                trial_df[col] = tr[col]
        rows.append(trial_df)

    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------------------
# DIC fitting
# ---------------------------------------------------------------------------

def fit_for_dic(sim_df: pd.DataFrame, version: int,
                samples: int, burn: int, seed: int) -> float:
    np.random.seed(seed)
    mdl = build_model(sim_df, version)
    mdl.find_starting_values()
    mdl.sample(samples, burn=burn, db="ram", dbname=f"ram_mr_{seed}")
    return float(mdl.dic)


def load_empirical(version: int, n_chains: int, model_dir: Path) -> az.InferenceData:
    model_name = f"{MODEL_BASE_NAME}mod_{version}"
    paths = [model_dir / f"{model_name}_{i}.nc" for i in range(n_chains)]
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(f"Posterior not found: {p}")
    return az.concat([az.from_netcdf(str(p)) for p in paths], dim="chain")


# ---------------------------------------------------------------------------
# Checkpointing helpers
# ---------------------------------------------------------------------------

def _atomic_csv(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def _read_or_empty(path: Path, cols: list) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=cols)
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=cols)
    keep = [c for c in cols if c in df.columns]
    return df[keep] if keep else pd.DataFrame(columns=cols)


# ---------------------------------------------------------------------------
# Confusion matrix plot
# ---------------------------------------------------------------------------

def plot_confusion(records: list, versions: list, out_dir: Path) -> None:
    df = pd.DataFrame(records)
    # Count wins per (true_version, winning_version)
    counts = (df.groupby(["true_version", "winning_version"])
                .size()
                .reset_index(name="n"))

    total_per_true = df.groupby("true_version").size().rename("total")
    counts = counts.join(total_per_true, on="true_version")
    counts["proportion"] = counts["n"] / counts["total"]

    mat = counts.pivot(index="true_version", columns="winning_version",
                       values="proportion").reindex(index=versions, columns=versions).fillna(0)

    fig, ax = plt.subplots(figsize=(len(versions) * 1.4 + 1, len(versions) * 1.4 + 0.5))
    sns.heatmap(mat, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1,
                linewidths=0.5, ax=ax, cbar_kws={"label": "Proportion of reps"})
    ax.set_xlabel("Winning model (lowest DIC)")
    ax.set_ylabel("True (generating) model")
    ax.set_title("Model recovery confusion matrix")
    fig.tight_layout()
    out = out_dir / "model_recovery_confusion.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Model recovery for PainReward HDDM")
    parser.add_argument("--versions",  type=int, nargs="+", default=DEFAULT_VERSIONS,
                        help="Source (and candidate) model versions to test")
    parser.add_argument("--n-reps",    type=int, default=10)
    parser.add_argument("--n-chains",  type=int, default=4)
    parser.add_argument("--samples",   type=int, default=2000)
    parser.add_argument("--burn",      type=int, default=200)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    args = parser.parse_args()

    out_dir = PROJECT_DIR / "derivatives" / "hddm" / "model_recovery"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input data not found: {INPUT_CSV}")
    raw_df = pd.read_csv(INPUT_CSV)
    raw_df["subj_idx"] = raw_df["subj_idx"].astype(str).str.replace(r"^\D+", "", regex=True).astype(int)
    subjects = sorted(raw_df["subj_idx"].unique())
    print(f"Versions to test: {args.versions} | Subjects: {len(subjects)} | Reps: {args.n_reps}")

    # Load all empirical posteriors upfront
    empiricals = {}
    for v in args.versions:
        print(f"Loading posterior v{v}...")
        empiricals[v] = load_empirical(v, args.n_chains, args.model_dir)

    # Checkpointing
    rec_cols    = ["rep", "true_version", "winning_version"] + [f"dic_{v}" for v in args.versions]
    partial_csv = out_dir / "partial_model_recovery.csv"
    partial     = _read_or_empty(partial_csv, rec_cols)

    complete_reps = set()
    if not partial.empty:
        n_expected   = len(args.versions)  # one row per true_version per rep
        rep_counts   = partial.groupby("rep")["true_version"].count()
        complete_reps = set(rep_counts[rep_counts >= n_expected].index)

    incomplete = set(partial["rep"].unique()) - complete_reps if not partial.empty else set()
    if incomplete:
        partial = partial[~partial["rep"].isin(incomplete)]

    start_rep = (max(complete_reps) + 1) if complete_reps else 0
    records   = partial.to_dict("records")
    print(f"Resuming from rep {start_rep} (completed: {sorted(complete_reps)})")

    for rep in tqdm(range(start_rep, args.n_reps), desc="model-recovery", unit="rep"):
        for true_v in args.versions:
            param_list = get_param_list(true_v)
            idata      = empiricals[true_v]
            rng_mu     = np.random.default_rng(rep * 1000 + true_v)
            rng_sd     = np.random.default_rng(rep * 1000 + true_v + 500)
            rng_subj   = np.random.default_rng(rep * 1000 + true_v + 700)

            mu  = extract_group_params(idata, param_list, rng_mu)
            sd  = extract_group_sds(idata,   param_list, rng_sd)
            true_individuals = sample_true_subjects(mu, sd, subjects, rng_subj)
            sim_df = simulate_dataset(true_individuals, raw_df, true_v)

            # Fit all candidate models and collect DIC
            dic_vals = {}
            for cand_v in args.versions:
                seed = rep * 100000 + true_v * 1000 + cand_v
                print(f"  rep={rep} true_v={true_v} cand_v={cand_v} ...", flush=True)
                dic_vals[cand_v] = fit_for_dic(sim_df, cand_v, args.samples, args.burn, seed)
                print(f"    DIC={dic_vals[cand_v]:.1f}")

            winning_v = min(dic_vals, key=dic_vals.get)
            row = {"rep": rep, "true_version": true_v, "winning_version": winning_v}
            row.update({f"dic_{v}": dic_vals[v] for v in args.versions})
            records.append(row)
            _atomic_csv(pd.DataFrame(records), partial_csv)

    df_final = pd.DataFrame(records)
    df_final.to_csv(out_dir / "model_recovery_results.csv", index=False)
    print(f"Results saved to {out_dir / 'model_recovery_results.csv'}")

    plot_confusion(records, args.versions, out_dir)

    # Summary table: proportion correct per true model
    df_final["correct"] = df_final["true_version"] == df_final["winning_version"]
    summary = df_final.groupby("true_version")["correct"].mean().reset_index()
    summary.columns = ["true_version", "p_correct"]
    print("\nModel recovery summary:")
    print(summary.to_string(index=False))
    summary.to_csv(out_dir / "model_recovery_summary.csv", index=False)
    print("Done.")


if __name__ == "__main__":
    main()
