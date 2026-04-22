"""Parameter recovery for PainReward HDDM models.

For a given model version:
  1. Load empirical posterior (.nc files from hddm_fit.py)
  2. For each rep: draw true group-level params from posterior, sample true
     individual params, simulate a dataset, refit, compare true vs recovered
  3. Save scatter plots (group-level and individual-level) with R², RMSE

Usage:
    python param_recovery.py --version 9 [--n-reps 20] [--n-chains 4]
                              [--samples 2000] [--burn 200]

Env vars: PROJECT_DIR, MODEL_DIR
"""
import argparse
import os
import re
import sys
import warnings
from pathlib import Path

import arviz as az
import hddm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats as st
from tqdm.auto import trange

warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model_specs import MODEL_BASE_NAME, get_param_list, get_formula_terms, build_model

PROJECT_DIR = Path(os.getenv("PROJECT_DIR", str(Path(__file__).resolve().parent.parent.parent))).resolve()
MODEL_DIR   = Path(os.getenv("MODEL_DIR",   str(PROJECT_DIR / "derivatives" / "hddm" / "models"))).resolve()
INPUT_CSV   = PROJECT_DIR / "derivatives" / "behav" / "hddm_ready.csv"


# ---------------------------------------------------------------------------
# Posterior extraction
# ---------------------------------------------------------------------------

def _random_draw(idata, param: str, rng: np.random.Generator) -> float:
    """Draw one posterior sample for a scalar group-level parameter."""
    if param not in idata.posterior:
        raise KeyError(
            f"Parameter '{param}' not found in posterior. "
            f"Available: {list(idata.posterior.data_vars)}"
        )
    da = idata.posterior[param]
    chain = int(rng.integers(da.sizes["chain"]))
    draw  = int(rng.integers(da.sizes["draw"]))
    return float(da.isel(chain=chain, draw=draw))


def extract_group_params(idata, param_list: list, rng: np.random.Generator) -> dict:
    """Draw one posterior sample for each group-level mean parameter."""
    return {p: _random_draw(idata, p, rng) for p in param_list}


def extract_group_sds(idata, param_list: list, rng: np.random.Generator) -> dict:
    """Draw one posterior sample for each group-level SD parameter."""
    out = {}
    for p in param_list:
        sd_name = f"{p}_std"
        if sd_name in idata.posterior:
            out[p] = _random_draw(idata, sd_name, rng)
        else:
            out[p] = 0.0
    return out


def sample_true_subjects(mu: dict, sd: dict, subjects: list,
                          rng: np.random.Generator) -> dict:
    """Draw per-subject true parameters from Normal(mu, sd)."""
    true = {}
    for s in subjects:
        pars = {}
        for p in mu:
            sigma = sd.get(p, 0.0)
            pars[p] = float(rng.normal(mu[p], sigma)) if sigma > 0 else float(mu[p])
        true[s] = pars
    return true


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def simulate_dataset(true_individuals: dict, raw_df: pd.DataFrame,
                     version: int) -> pd.DataFrame:
    """Generate simulated HDDM-compatible data using true individual parameters.

    For regressor models, trial-level values of regressed DVs are computed from
    the formula. Non-regressed DVs use the subject-level constant.
    For plain HDDM (version 0), all params are subject-level constants.
    """
    formula_terms = get_formula_terms(version)   # {dv: [(param_name, predictor_cols)]}
    regressed_dvs  = set(formula_terms.keys())

    rows = []
    df = raw_df.copy()
    df["subj_idx"] = df["subj_idx"].astype(str).str.extract(r'(\d+)')[0].astype(int)

    for _, tr in df.iterrows():
        subj = int(tr["subj_idx"])
        pars = true_individuals[subj]

        par_dict = {}

        # Compute trial-level values for each regressed DV
        for dv, terms in formula_terms.items():
            val = 0.0
            for param_name, cols in terms:
                if param_name not in pars:
                    raise KeyError(
                        f"True param '{param_name}' missing for subject {subj}. "
                        f"Available: {list(pars.keys())}"
                    )
                if len(cols) == 0:
                    x = 1.0  # intercept
                else:
                    x = float(np.prod([tr[c] for c in cols]))
                val += pars[param_name] * x
            par_dict[dv] = val

        # Non-regressed parameters use subject constant
        for p in ("a", "v", "t", "z"):
            if p not in regressed_dvs and p in pars:
                par_dict[p] = float(pars[p])

        trial_df, _ = hddm.generate.gen_rand_data(par_dict, size=1, subjs=1)

        # Carry forward all predictor columns from the original trial
        for col in df.columns:
            if col not in ("rt", "response"):
                trial_df[col] = tr[col]

        rows.append(trial_df)

    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------------------
# Model fitting and extraction
# ---------------------------------------------------------------------------

def refit_model(sim_df: pd.DataFrame, version: int,
                samples: int, burn: int, seed: int):
    np.random.seed(seed)
    mdl = build_model(sim_df, version)
    mdl.find_starting_values()
    mdl.sample(samples, burn=burn, db="ram", dbname=f"ram_{seed}")
    return mdl


def extract_group_means(mdl, param_list: list) -> dict:
    """Extract posterior mean for each group-level parameter from a fitted model."""
    means = {}
    for p in param_list:
        if p in mdl.nodes_db.index:
            means[p] = mdl.nodes_db.loc[p, "node"].trace().mean()
        else:
            # Parameter may be stored under a slightly different name; warn rather than crash
            candidates = [n for n in mdl.nodes_db.index if n == p or n.startswith(p + "_")]
            if candidates:
                means[p] = mdl.nodes_db.loc[candidates[0], "node"].trace().mean()
            else:
                raise KeyError(
                    f"Cannot find '{p}' in fitted model nodes. "
                    f"Available: {sorted(mdl.nodes_db.index.tolist())}"
                )
    return means


def extract_individual_means(mdl, param_list: list) -> dict:
    """Return {(subj_int, param_name): posterior_mean} for individual nodes."""
    # HDDM node naming: "v_painlevel_subj.3", "a_subj.3", "t_subj.3"
    # Interaction nodes may look like "v_painlevel:moneylevel_subj.3"
    pattern = re.compile(r"^(.+)_subj[\.(](\d+)\)?$")
    out = {}
    for node_name in mdl.nodes_db.index:
        m = pattern.match(node_name)
        if not m:
            continue
        base, subj_str = m.group(1), m.group(2)
        if base in param_list:
            out[(int(subj_str), base)] = mdl.nodes_db.loc[node_name, "node"].trace().mean()
    return out


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
# Plotting
# ---------------------------------------------------------------------------

def _regress_stats(x, y) -> dict:
    x, y = np.asarray(x), np.asarray(y)
    ok = len(x) >= 3 and np.isfinite(x).all() and np.isfinite(y).all() and np.std(x) > 0
    if not ok:
        return {"ok": False, "N": len(x), "R2": np.nan, "p": np.nan, "RMSE": np.nan}
    res  = st.linregress(x, y)
    rmse = float(np.sqrt(np.mean((y - (res.intercept + res.slope * x)) ** 2)))
    return {"ok": True, "N": len(x), "R2": float(res.rvalue ** 2),
            "p": float(res.pvalue), "RMSE": rmse}


def _annotate(ax, x, y) -> None:
    s = _regress_stats(x, y)
    if not s["ok"]:
        return
    p_txt = "p<.001" if s["p"] < 1e-3 else f"p={s['p']:.3f}"
    ax.text(0.97, 0.03, f"R²={s['R2']:.2f}\n{p_txt}\nRMSE={s['RMSE']:.3f}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.9))


def _identity_scatter(data, **_):
    ax = plt.gca()
    x, y = data["true"].values, data["recovered"].values
    ax.scatter(x, y, s=18, alpha=0.7)
    lo = min(ax.get_xlim()[0], ax.get_ylim()[0])
    hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([lo, hi], [lo, hi], "--k", lw=1)
    ax.set_xlabel("true value")
    ax.set_ylabel("recovered (posterior mean)")
    _annotate(ax, x, y)


def save_scatter(records: list, level: str, version: int, out_dir: Path) -> None:
    df = pd.DataFrame(records).replace([np.inf, -np.inf], np.nan).dropna(subset=["true", "recovered"])
    if df.empty:
        return
    params = sorted(df["parameter"].unique())
    col_wrap = min(3, len(params))
    sns.set_style("white")
    g = sns.FacetGrid(df, col="parameter", col_order=params,
                      col_wrap=col_wrap, sharex=False, sharey=False,
                      height=3.2, despine=True)
    g.set_titles("{col_name}")
    g.map_dataframe(_identity_scatter)
    g.figure.suptitle(f"Parameter recovery ({level}) — model v{version}")
    g.tight_layout()
    out = out_dir / f"param_recovery_{level}_v{version}.png"
    g.savefig(out, dpi=300, bbox_inches="tight")
    plt.close("all")
    print(f"  Saved {out.name}")

    # Stats CSV
    rows = []
    for p in params:
        sub = df[df["parameter"] == p]
        s = _regress_stats(sub["true"], sub["recovered"])
        rows.append({"parameter": p, "R2": s["R2"], "p": s["p"], "RMSE": s["RMSE"], "N": s["N"]})
    pd.DataFrame(rows).to_csv(out_dir / f"param_recovery_{level}_stats_v{version}.csv", index=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Parameter recovery for PainReward HDDM")
    parser.add_argument("--version",   type=int, required=True)
    parser.add_argument("--n-reps",    type=int, default=20)
    parser.add_argument("--n-chains",  type=int, default=4,
                        help="Number of posterior chains to load for drawing true params")
    parser.add_argument("--samples",   type=int, default=2000)
    parser.add_argument("--burn",      type=int, default=200)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    args = parser.parse_args()

    model_name = f"{MODEL_BASE_NAME}mod_{args.version}"
    param_list = get_param_list(args.version)
    print(f"Model v{args.version} | parameters: {param_list}")

    out_dir = PROJECT_DIR / "derivatives" / "hddm" / "param_recovery" / f"v{args.version}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load empirical posterior chains and concatenate
    nc_paths = [args.model_dir / f"{model_name}_{i}.nc" for i in range(args.n_chains)]
    for p in nc_paths:
        if not p.exists():
            raise FileNotFoundError(f"Posterior file not found: {p}")
    empirical = az.concat([az.from_netcdf(str(p)) for p in nc_paths], dim="chain")
    print(f"Loaded empirical posterior: {empirical.posterior.dims}")

    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input data not found: {INPUT_CSV}")
    raw_df = pd.read_csv(INPUT_CSV)
    raw_df["subj_idx"] = raw_df["subj_idx"].astype(str).str.extract(r'(\d+)')[0].astype(int)
    subjects = sorted(raw_df["subj_idx"].unique())
    print(f"Subjects: {len(subjects)}, trials: {len(raw_df)}")

    # Checkpointing
    grp_cols  = ["rep", "parameter", "true", "recovered"]
    ind_cols  = ["rep", "subj", "parameter", "true", "recovered"]
    grp_csv   = out_dir / f"partial_group_v{args.version}.csv"
    ind_csv   = out_dir / f"partial_indiv_v{args.version}.csv"

    grp_partial = _read_or_empty(grp_csv, grp_cols)
    ind_partial = _read_or_empty(ind_csv, ind_cols)

    counts        = grp_partial.groupby("rep")["parameter"].count()
    complete_reps = set(counts[counts >= len(param_list)].index.tolist())
    incomplete    = set(grp_partial["rep"].unique()) - complete_reps
    if incomplete:
        grp_partial = grp_partial[~grp_partial["rep"].isin(incomplete)]
        ind_partial = ind_partial[~ind_partial["rep"].isin(incomplete)]

    start_rep     = (max(complete_reps) + 1) if complete_reps else 0
    grp_records   = grp_partial.to_dict("records")
    ind_records   = ind_partial.to_dict("records")
    print(f"Resuming from rep {start_rep} (completed: {sorted(complete_reps)})")

    for rep in trange(start_rep, args.n_reps, desc="param-recovery", unit="rep"):
        try:
            rng = np.random.default_rng(rep)
            mu  = extract_group_params(empirical, param_list, np.random.default_rng(rep))
            sd  = extract_group_sds(empirical,   param_list, np.random.default_rng(rep + 10000))

            true_individuals = sample_true_subjects(mu, sd, subjects,
                                                    np.random.default_rng(rep + 20000))
            sim_df = simulate_dataset(true_individuals, raw_df, args.version)

            mdl = refit_model(sim_df, args.version, args.samples, args.burn,
                              seed=30000 + rep)

            # Group-level
            recovered_group = extract_group_means(mdl, param_list)
            for p in param_list:
                grp_records.append(dict(rep=rep, parameter=p,
                                        true=mu[p], recovered=recovered_group[p]))

            # Individual-level
            recovered_indiv = extract_individual_means(mdl, param_list)
            for (subj, param), rec_val in recovered_indiv.items():
                true_val = true_individuals[subj].get(param)
                if true_val is not None:
                    ind_records.append(dict(rep=rep, subj=subj, parameter=param,
                                            true=true_val, recovered=rec_val))
        finally:
            _atomic_csv(pd.DataFrame(grp_records), grp_csv)
            _atomic_csv(pd.DataFrame(ind_records), ind_csv)

    # Final CSVs
    pd.DataFrame(grp_records).to_csv(out_dir / f"group_recovery_v{args.version}.csv",  index=False)
    pd.DataFrame(ind_records).to_csv(out_dir / f"indiv_recovery_v{args.version}.csv", index=False)

    save_scatter(grp_records, "group",      args.version, out_dir)
    save_scatter(ind_records, "individual", args.version, out_dir)
    print("Done.")


if __name__ == "__main__":
    main()
