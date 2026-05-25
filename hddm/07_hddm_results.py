# hddm_results.py — load fitted HDDM chains, compute MAP estimates, plot diagnostics
#
# Merges the logic from DDM_EEG_load.py and MAP_estimates.py.
#
# Run: python hddm_results.py --version 9 [--model-dir PATH] [--fig-dir PATH]
# Env vars: PROJECT_DIR, MODEL_DIR, FIG_DIR

import os
import re
import warnings
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl

import numpy as np
import pandas as pd
import pickle
import shutil
import tempfile
import seaborn as sns
import scipy.stats as stats

try:
    import kabuki
    import hddm
except ModuleNotFoundError:
    kabuki = None
    hddm = None

from model_specs import REQUIRED_COLS, get_formula_terms

warnings.simplefilter(action="ignore", category=FutureWarning)

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
try:
    import numba
    numba.config.CACHE_ENABLE = False
except ModuleNotFoundError:
    numba = None

PROJECT_DIR    = Path(os.getenv("PROJECT_DIR", str(Path(__file__).resolve().parent.parent.parent))).resolve()
BASE_MODEL_DIR = Path(os.getenv("MODEL_DIR",
                      str(PROJECT_DIR / "derivatives" / "hddm" / "models"))).resolve()
FIG_DIR_ROOT   = Path(os.getenv("FIG_DIR",
                      str(PROJECT_DIR / "derivatives" / "hddm" / "figures"))).resolve()
INPUT_CSV      = PROJECT_DIR / "derivatives" / "behav" / "hddm_ready.csv"

MODEL_BASE_NAME = "painreward_behavioural_data_"
N_CHAINS = 4

# Output CSV name for trial-level contributions, keyed by version.
CONTRIBUTION_CSV = {
    9:  "v_pain_money.csv",
    10: "a_pain_money.csv",
    11: "t_pain_money.csv",
    12: "v_pain_money_interaction.csv",
    17: "v_pain_money_rp.csv",
    18: "a_pain_money_rp.csv",
    19: "v_a_pain_money.csv",
    20: "v_a_pain_money_rp.csv",
    21: "v_pain_money_z_interaction.csv",
}

# Derived posterior ratios to append to the MAP table: (label, numerator_node, denominator_node).
DERIVED_RATIOS = {
    9:  [("delta", "v_moneylevel", "v_painlevel")],
    10: [("delta", "a_moneylevel", "a_painlevel")],
}

MODEL_LABELS = {
    0:  "Null\n(a, v, t)",
    1:  "v ~ sv_pain\n(intercept)",
    2:  "v ~ sv_pain\n(no intercept)",
    3:  "a ~ sv_pain",
    9:  "v ~ pain\n+ money",
    10: "a ~ pain\n+ money",
    11: "t ~ pain\n+ money",
    12: "v ~ pain\n× money",
    17: "v ~ pain+money\n+rp+interactions",
    18: "a ~ pain+money\n+rp+interactions",
    19: "v+a ~ pain\n+ money",
    20: "v+a ~ pain+money\n+rp+interactions",
    21: "v ~ z-pain\n× z-money",
}

MODEL_TABLE_SPECS = {
    0: {
        "equation": "a, v, t, z",
        "description": "Null/intercept-only HDDM with threshold, drift, non-decision time, and bias estimated.",
        "n_parameters": 4,
        "n_free_parameters": 4,
        "n_fixed_parameters": 0,
    },
    1: {
        "equation": "v = beta0 + beta1 * sv_pain_para; a, t estimated; z fixed",
        "description": "Drift rate varies with subjective pain value, with an intercept.",
        "n_parameters": 5,
        "n_free_parameters": 4,
        "n_fixed_parameters": 1,
    },
    2: {
        "equation": "v = beta1 * sv_pain_para; a, t estimated; z fixed",
        "description": "Drift rate varies with subjective pain value, without an intercept.",
        "n_parameters": 4,
        "n_free_parameters": 3,
        "n_fixed_parameters": 1,
    },
    3: {
        "equation": "a = beta0 + beta1 * sv_pain_para; v, t estimated; z fixed",
        "description": "Decision threshold varies with subjective pain value.",
        "n_parameters": 5,
        "n_free_parameters": 4,
        "n_fixed_parameters": 1,
    },
    9: {
        "equation": "v = beta0 + beta1 * painlevel + beta2 * moneylevel; a, t estimated; z fixed",
        "description": "Drift rate varies additively with objective pain and money levels.",
        "n_parameters": 6,
        "n_free_parameters": 5,
        "n_fixed_parameters": 1,
    },
    10: {
        "equation": "a = beta0 + beta1 * painlevel + beta2 * moneylevel; v, t estimated; z fixed",
        "description": "Decision threshold varies additively with objective pain and money levels.",
        "n_parameters": 6,
        "n_free_parameters": 5,
        "n_fixed_parameters": 1,
    },
    11: {
        "equation": "t = beta0 + beta1 * painlevel + beta2 * moneylevel; a, v estimated; z fixed",
        "description": "Non-decision time varies additively with objective pain and money levels.",
        "n_parameters": 6,
        "n_free_parameters": 5,
        "n_fixed_parameters": 1,
    },
    12: {
        "equation": "v = beta0 + beta1 * painlevel + beta2 * moneylevel + beta3 * painlevel:moneylevel; a, t estimated; z fixed",
        "description": "Drift rate varies with pain, money, and their interaction.",
        "n_parameters": 7,
        "n_free_parameters": 6,
        "n_fixed_parameters": 1,
    },
    17: {
        "equation": "v = beta0 + beta1 * pain_z + beta2 * money_z + beta3 * rp_z + beta4 * pain_z:rp_z + beta5 * money_z:rp_z; a, t estimated; z fixed",
        "description": "Drift rate varies with standardized pain, money, response-locked potential, and pain/money by RP interactions.",
        "n_parameters": 9,
        "n_free_parameters": 8,
        "n_fixed_parameters": 1,
    },
    18: {
        "equation": "a = beta0 + beta1 * pain_z + beta2 * money_z + beta3 * rp_z + beta4 * pain_z:rp_z + beta5 * money_z:rp_z; v, t estimated; z fixed",
        "description": "Decision threshold varies with standardized pain, money, response-locked potential, and pain/money by RP interactions.",
        "n_parameters": 9,
        "n_free_parameters": 8,
        "n_fixed_parameters": 1,
    },
    19: {
        "equation": "v = beta0 + beta1 * pain_z + beta2 * money_z; a = gamma0 + gamma1 * pain_z + gamma2 * money_z; t estimated; z fixed",
        "description": "Drift rate and threshold both vary additively with standardized pain and money.",
        "n_parameters": 8,
        "n_free_parameters": 7,
        "n_fixed_parameters": 1,
    },
    20: {
        "equation": "v = beta0 + beta1 * pain_z + beta2 * money_z + beta3 * rp_z + beta4 * pain_z:rp_z + beta5 * money_z:rp_z; a = gamma0 + gamma1 * pain_z + gamma2 * money_z + gamma3 * rp_z + gamma4 * pain_z:rp_z + gamma5 * money_z:rp_z; t estimated; z fixed",
        "description": "Drift rate and threshold both vary with standardized pain, money, response-locked potential, and pain/money by RP interactions.",
        "n_parameters": 14,
        "n_free_parameters": 13,
        "n_fixed_parameters": 1,
    },
    21: {
        "equation": "v = beta0 + beta1 * pain_z + beta2 * money_z + beta3 * pain_z:money_z; a, t estimated; z fixed",
        "description": "Model 12 variant: drift rate varies with within-subject z-scored pain, z-scored money, and their interaction.",
        "n_parameters": 7,
        "n_free_parameters": 6,
        "n_fixed_parameters": 1,
    },
}

VERSION_PARAMS = {
    0:  ["a", "t", "v"],
    1:  ["a", "t", "v_Intercept", "v_sv_pain_para"],
    2:  ["a", "t", "v_sv_pain_para"],
    3:  ["v", "t", "a_Intercept", "a_sv_pain_para"],
    9:  ["a", "t", "v_Intercept", "v_painlevel", "v_moneylevel"],
    10: ["v", "t", "a_Intercept", "a_painlevel", "a_moneylevel"],
    11: ["a", "v", "t_Intercept", "t_painlevel", "t_moneylevel"],
    12: ["a", "t", "v_Intercept", "v_painlevel", "v_moneylevel", "v_painlevel:moneylevel"],
    17: ["a", "t", "v_Intercept", "v_pain_z", "v_money_z", "v_rp_z",
         "v_pain_z:rp_z", "v_money_z:rp_z"],
    18: ["v", "t", "a_Intercept", "a_pain_z", "a_money_z", "a_rp_z",
         "a_pain_z:rp_z", "a_money_z:rp_z"],
    19: ["t", "v_Intercept", "v_pain_z", "v_money_z",
         "a_Intercept", "a_pain_z", "a_money_z"],
    20: ["t",
         "v_Intercept", "v_pain_z", "v_money_z", "v_rp_z", "v_pain_z:rp_z", "v_money_z:rp_z",
         "a_Intercept", "a_pain_z", "a_money_z", "a_rp_z", "a_pain_z:rp_z", "a_money_z:rp_z"],
    21: ["a", "t", "v_Intercept", "v_pain_z", "v_money_z", "v_pain_z:money_z"],
}


def _sanitize_filename(fname):
    safe = re.sub(r'[:\(\)\[\],]', '_', fname)
    return re.sub(r'_+', '_', safe)


def load_chains(version: int, model_dir: Path, n_chains: int = N_CHAINS) -> list:
    """Load all pkl chains and return list of individual models."""
    model_name = MODEL_BASE_NAME + f"mod_{version}"
    models = []
    for i in range(n_chains):
        path = model_dir / f"{model_name}_{i}.pkl"
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        with open(path, "rb") as fh:
            models.append(pickle.load(fh))
    return models


def map_estimates(combined, param_names: list, version: int) -> pd.DataFrame:
    """Compute MAP (posterior mean) and 95% HDI for group-level parameters.

    Appends derived ratio parameters (e.g. delta = money/pain) if defined for this version.
    """
    rows = []
    stats_df = combined.gen_stats()
    available = set(stats_df.index)

    for name in param_names:
        if name not in available:
            raise KeyError(
                f"Parameter '{name}' not found in model stats. "
                f"Available: {sorted(available)}"
            )
        trace = combined.nodes_db.node[name].trace()
        rows.append({
            "parameter": name,
            "MAP":       trace.mean(),
            "HDI_2.5":   stats.mstats.mquantiles(trace, [0.025])[0],
            "HDI_97.5":  stats.mstats.mquantiles(trace, [0.975])[0],
        })

    for label, num_node, den_node in DERIVED_RATIOS.get(version, []):
        num = combined.nodes_db.node[num_node].trace()
        den = combined.nodes_db.node[den_node].trace()
        ratio = num / den
        rows.append({
            "parameter": label,
            "MAP":       ratio.mean(),
            "HDI_2.5":   stats.mstats.mquantiles(ratio, [0.025])[0],
            "HDI_97.5":  stats.mstats.mquantiles(ratio, [0.975])[0],
        })

    return pd.DataFrame(rows)


def plot_diagnostics(models: list, combined, version: int, diag_dir: Path) -> None:
    """Generate diagnostic figures and stats: GR, DIC, PPC, posteriors, per-param KDEs."""
    diag_dir.mkdir(parents=True, exist_ok=True)

    # Gelman-Rubin convergence
    gr = hddm.analyze.gelman_rubin(models)
    with open(diag_dir / "gelman_rubin.txt", "w") as f:
        for param, val in gr.items():
            f.write(f"{param}: {val}\n")

    # DIC
    (diag_dir / "DIC.txt").write_text(f"DIC: {combined.dic}\n")

    # Full stats table
    combined.gen_stats().to_csv(diag_dir / "results.csv")

    # Posterior predictive check
    n_subj = len(combined.data.subj_idx.unique())
    combined.plot_posterior_predictive(
        samples=10, bins=100,
        figsize=(6, max(4, n_subj / 3.0 * 1.5)),
        save=True, path=str(diag_dir), format="pdf",
    )

    # All parameter posteriors — write to /tmp first, then rename, to avoid
    # colon/special-char filenames failing on exFAT/NTFS host filesystems.
    with tempfile.TemporaryDirectory() as tmp:
        mpl.rcParams.update({"font.size": 6})
        combined.plot_posteriors(save=True, path=tmp, format="pdf")
        mpl.rcParams.update({"font.size": 12})
        for src in Path(tmp).iterdir():
            if src.suffix in (".pdf", ".csv"):
                shutil.move(str(src), str(diag_dir / _sanitize_filename(src.name)))

    # Per-parameter vertical KDE plots for group-level params
    kde_dir = diag_dir / "group_param_vertical_kdes"
    kde_dir.mkdir(exist_ok=True)

    for param in VERSION_PARAMS.get(version, []):
        try:
            tr = combined.nodes_db.node[param].trace()
        except Exception:
            print(f"  Skipping missing parameter: {param}")
            continue

        tr = np.asarray(tr, dtype=float).ravel()
        tr = tr[np.isfinite(tr)]
        if tr.size < 2:
            print(f"  Skipping parameter with too few samples: {param}")
            continue

        ci_low, ci_high = np.percentile(tr, [2.5, 97.5])

        fig, ax = plt.subplots(figsize=(5, 8))
        sns.kdeplot(y=tr, fill=True, ax=ax)
        ax.axhline(ci_low,    linestyle=":", linewidth=2)
        ax.axhline(ci_high,   linestyle=":", linewidth=2)
        ax.axhline(tr.mean(), linestyle="--", linewidth=1.8)
        ax.set_title(param,       fontsize=26, pad=12)
        ax.set_xlabel("Density",  fontsize=25, labelpad=10)
        ax.set_ylabel("Value",    fontsize=25)
        ax.tick_params(axis="both", labelsize=25, width=1.2)
        for side in ["top", "right"]:
            ax.spines[side].set_visible(False)
        plt.tight_layout()
        fig.savefig(kde_dir / f"{_sanitize_filename(param)}_kde.pdf", bbox_inches="tight")
        plt.close(fig)

    # Rename files with characters that are unsafe for some filesystems
    for f in diag_dir.iterdir():
        if f.suffix in (".pdf", ".csv"):
            safe = _sanitize_filename(f.name)
            if safe != f.name:
                f.rename(diag_dir / safe)


def compute_trial_contributions(combined, data: pd.DataFrame, version: int, diag_dir: Path) -> None:
    """Compute trial-level parameter estimates (posterior mean × trial covariates) and save CSV.

    Uses get_formula_terms() from model_specs so no per-version branching is needed here.
    For each regression term the subject-level posterior mean is multiplied by the trial
    covariate(s); contributions are summed into a *_full_trial column.
    """
    formula_terms = get_formula_terms(version)
    if not formula_terms:
        return  # version 0 is a plain HDDM with no regression terms

    data_out = data.copy()

    for dv, terms in formula_terms.items():
        for param_name, _ in terms:
            data_out[f"{param_name}_subj"]    = np.nan
            data_out[f"{param_name}_contrib"] = np.nan
        data_out[f"{dv}_full_trial"] = np.nan

        for subj in data["subj_idx"].unique():
            mask      = data_out["subj_idx"] == subj
            subj_data = data_out.loc[mask]
            running   = np.zeros(mask.sum())

            for param_name, predictor_cols in terms:
                node_key = f"{param_name}_subj.{subj}"
                beta = combined.nodes_db.loc[node_key, "node"].trace().mean()
                data_out.loc[mask, f"{param_name}_subj"] = beta

                if not predictor_cols:
                    contrib = np.full(mask.sum(), beta)
                else:
                    cov = subj_data[predictor_cols[0]].values.copy().astype(float)
                    for col in predictor_cols[1:]:
                        cov *= subj_data[col].values
                    contrib = beta * cov

                data_out.loc[mask, f"{param_name}_contrib"] = contrib
                running += contrib

            data_out.loc[mask, f"{dv}_full_trial"] = running

    csv_name = CONTRIBUTION_CSV.get(version, f"contributions_v{version}.csv")
    out_path = diag_dir / csv_name
    data_out.to_csv(out_path, index=False)
    print(f"  Saved trial contributions → {out_path}")


def _load_and_clean_data(version: int) -> pd.DataFrame:
    """Load hddm_ready.csv and apply the same cleaning as hddm_fit.py."""
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}\nRun hddm_prep.py first.")
    df = pd.read_csv(INPUT_CSV)
    if version in [17, 18, 20]:
        df = df[df["badtrial"] == 0]
    df = df.dropna(subset=REQUIRED_COLS[version])
    return df


def run_version(version: int, model_dir: Path, fig_dir: Path) -> None:
    if kabuki is None or hddm is None:
        raise ModuleNotFoundError("Processing a fitted HDDM version requires kabuki and hddm.")

    print(f"\n=== Version {version} ===")
    models   = load_chains(version, model_dir)
    combined = kabuki.utils.concat_models(models)

    print("DIC:", combined.dic)
    try:
        print("BPIC:", combined.mc.BPIC)
    except AttributeError:
        pass

    diag_dir = fig_dir / (MODEL_BASE_NAME + f"mod_{version}") / "diagnostics"

    # Plots + stats (creates diag_dir)
    plot_diagnostics(models, combined, version, diag_dir)

    # MAP table (includes derived ratios such as delta for versions 9/10)
    params = VERSION_PARAMS.get(version, [])
    df_map = map_estimates(combined, params, version)
    print(df_map.to_string(index=False))
    out_path = diag_dir / f"group_level_MAP_table_m{version}.csv"
    df_map.to_csv(out_path, index=False)
    print(f"Saved → {out_path}")

    # Trial-level parameter contributions (needs raw data)
    if version in CONTRIBUTION_CSV:
        data = _load_and_clean_data(version)
        compute_trial_contributions(combined, data, version, diag_dir)


def plot_model_comparison(fig_dir: Path) -> None:
    """Read DIC.txt from each model's diagnostics dir and plot a DIC comparison bar chart."""
    records = []
    for version, label in MODEL_LABELS.items():
        dic_path = fig_dir / (MODEL_BASE_NAME + f"mod_{version}") / "diagnostics" / "DIC.txt"
        if not dic_path.exists():
            print(f"  Skipping v{version}: DIC file not found ({dic_path})")
            continue
        text = dic_path.read_text().strip()
        dic_val = float(text.split(":")[1].strip())
        table_spec = MODEL_TABLE_SPECS[version]
        records.append({
            "version": version,
            "model_name": f"Model {version}",
            "label": label,
            "equation": table_spec["equation"],
            "description": table_spec["description"],
            "n_parameters": table_spec["n_parameters"],
            "n_free_parameters": table_spec["n_free_parameters"],
            "n_fixed_parameters": table_spec["n_fixed_parameters"],
            "DIC": dic_val,
            "fit_statistic": dic_val,
            "fit_statistic_name": "DIC",
        })

    if not records:
        print("No DIC files found — run individual versions first.")
        return

    df = pd.DataFrame(records).sort_values("version")
    best_idx = df["DIC"].idxmin()

    fig, ax = plt.subplots(figsize=(max(8, len(df) * 1.2), 6))
    colors = ["#d62728" if i == best_idx else "#4878d0" for i in df.index]
    bars = ax.bar(range(len(df)), df["DIC"], color=colors, edgecolor="white", linewidth=0.8)

    # Annotate bars with ΔDIC relative to best
    best_dic = df.loc[best_idx, "DIC"]
    for bar, (_, row) in zip(bars, df.iterrows()):
        delta = row["DIC"] - best_dic
        label_txt = "best" if delta == 0 else f"+{delta:.0f}"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + abs(best_dic) * 0.002,
                label_txt, ha="center", va="bottom", fontsize=9)

    ax.set_xticks(range(len(df)))
    ax.set_xticklabels([f"v{r['version']}\n{r['label']}" for _, r in df.iterrows()],
                       fontsize=8)
    ax.set_ylabel("DIC (lower = better)", fontsize=12)
    ax.set_title("HDDM Model Comparison (DIC)", fontsize=14)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)

    plt.tight_layout()
    out = fig_dir / "model_comparison_DIC.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Model comparison figure saved → {out}")

    csv_out = fig_dir / "model_comparison_DIC.csv"
    df[["version", "label", "fit_statistic"]].rename(
        columns={"fit_statistic": "DIC"}
    ).to_csv(csv_out, index=False)
    print(f"Model comparison table saved  → {csv_out}")

    table = df[[
        "model_name",
        "equation",
        "description",
        "n_parameters",
        "n_free_parameters",
        "n_fixed_parameters",
        "fit_statistic",
        "fit_statistic_name",
    ]].copy()
    table.insert(0, "dic_rank", table["fit_statistic"].rank(method="min").astype(int))
    table = table.sort_values("dic_rank")
    table.insert(0, "winning_model", table["fit_statistic"] == table["fit_statistic"].min())

    table_out = fig_dir / "model_comparison_table.csv"
    table.to_csv(table_out, index=False)
    print(f"Full model comparison table saved → {table_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load HDDM chains and compute MAP estimates")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--version", type=int,
                       help="Model version number to process (e.g. 9, 10, 19)")
    group.add_argument("--compare", action="store_true",
                       help="Read saved DIC files and produce model comparison figure")
    parser.add_argument("--model-dir", type=Path, default=BASE_MODEL_DIR)
    parser.add_argument("--fig-dir",   type=Path, default=FIG_DIR_ROOT)
    args = parser.parse_args()

    if args.compare:
        plot_model_comparison(args.fig_dir)
    else:
        run_version(args.version, args.model_dir, args.fig_dir)
