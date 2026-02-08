# groupplots for rp analysis
# Vero

import mne
import pandas as pd
import numpy as np
from os.path import join as opj
import matplotlib.pyplot as plt
from bids import BIDSLayout
import seaborn as sns
import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from pathlib import Path
import scipy.stats as stats

# -----------------------
# Directories (cluster/container friendly)
# -----------------------
PROJECT_DIR = Path(os.getenv("PROJECT_DIR", "/workspace"))
DATA_DIR    = Path(os.getenv("DATA_DIR", PROJECT_DIR / "EEG" / "PainReward_sub-001-050" / "painrewardeegdata"))
OUT_DIR     = Path(os.getenv("OUT_DIR", DATA_DIR / "derivatives"))   
HDDM_DIR    = Path(os.getenv("HDDM_DIR", OUT_DIR / "derivatives" / "hddm"))

basepath = DATA_DIR
layout = BIDSLayout(basepath)

# -----------------------
# CONFIG
# -----------------------
version = 1                 # 1=v, 2=a
stats_subdir = "Zscoring"   

if version == 1:
    out_prefix = "v1"
    pred1_col, pred2_col = "v_painlevel_subj", "v_moneylevel_subj"
    analysis_dir = OUT_DIR / "erps_massuni_sv_cuelong" / "v1_rp_drift_joint"
elif version == 2:
    out_prefix = "v2"
    pred1_col, pred2_col = "a_painlevel_subj", "a_moneylevel_subj"
    analysis_dir = OUT_DIR / "erps_massuni_sv_cuelong" / "v2_rp_boundary_joint"
else:
    raise ValueError("version must be 1 or 2")

rp_dir = analysis_dir / stats_subdir
if not rp_dir.exists():
    raise FileNotFoundError(f"Cannot find rp_dir: {rp_dir}\nCheck OUT_DIR")

# figures output folder 
outfigpath = OUT_DIR / "figures" / "erps_massuni_sv_cuelong" / analysis_dir.name / stats_subdir
outfigpath.mkdir(parents=True, exist_ok=True)

# -----------------------
# Load analysis outputs
# -----------------------
subj_csv  = rp_dir / f"{out_prefix}_subject_level_rp_means_by_bin.csv"
group_csv = rp_dir / f"{out_prefix}_group_regress_rp_on_ddmparam_by_bin.csv"

if not subj_csv.exists():
    raise FileNotFoundError(f"Missing subject file: {subj_csv}")
if not group_csv.exists():
    raise FileNotFoundError(f"Missing group file: {group_csv}")

subj_df  = pd.read_csv(subj_csv)
group_df = pd.read_csv(group_csv)

electrode_sets = sorted(subj_df["set"].unique())

def bin_label(tmin, tmax):
    return f"{tmin:.1f}–{tmax:.1f}s"

subj_df["bin_label"]  = subj_df.apply(lambda r: bin_label(r.bin_tmin, r.bin_tmax), axis=1)
group_df["bin_label"] = group_df.apply(lambda r: bin_label(r.bin_tmin, r.bin_tmax), axis=1)

# -------------------------
# Waveforms (grand average and SEM)
# -------------------------
for set_name in electrode_sets:
    wf_npy = rp_dir / f"{out_prefix}_{set_name}__rp_subject_waveforms.npy"
    t_npy  = rp_dir / f"{out_prefix}_{set_name}__rp_times.npy"
    if not (wf_npy.exists() and t_npy.exists()):
        print(f"Missing waveform files for {set_name}: {wf_npy.name} / {t_npy.name}")
        continue

    wfs = np.load(wf_npy)        # (n_subj, n_times) in Volts
    times = np.load(t_npy)       # seconds

    mean = np.nanmean(wfs, axis=0) * 1e6
    sem  = stats.sem(wfs, axis=0, nan_policy="omit") * 1e6

    fig, ax = plt.subplots(figsize=(4.5, 3.0))
    ax.plot(times * 1000, mean, linewidth=2)
    ax.fill_between(times * 1000, mean - sem, mean + sem, alpha=0.25)
    ax.axvline(0, linestyle="--", color="gray")
    ax.axhline(0, linestyle="--", color="gray")
    ax.set_xlabel("Time from response (ms)")
    ax.set_ylabel("Amplitude (µV)")
    ax.set_title(f"{out_prefix.upper()} RP waveform: {set_name}")
    fig.tight_layout()
    fig.savefig(outfigpath / f"{out_prefix}_{set_name}__rp_grand_average.svg", dpi=600, bbox_inches="tight")
    plt.close(fig)

# -------------------------
# Group betas (bars per bin) + p + FDR + stars
# -------------------------
has_fdr = "p_fdr_bh" in group_df.columns

for set_name in electrode_sets:
    sdf_set = group_df[group_df["set"] == set_name].copy()
    if sdf_set.empty:
        continue

    for predictor in [pred1_col, pred2_col]:
        sdf = sdf_set[sdf_set["predictor"] == predictor].copy()
        if sdf.empty:
            continue

        bins_sorted = (
            sdf[["bin_tmin", "bin_tmax", "bin_label"]]
            .drop_duplicates()
            .sort_values(["bin_tmin", "bin_tmax"])
        )
        sdf = sdf.merge(bins_sorted, on=["bin_tmin", "bin_tmax", "bin_label"], how="right")

        x = np.arange(len(bins_sorted))
        betas = sdf["beta_z"].to_numpy(float)
        pvals = sdf["p"].to_numpy(float)
        pfdr  = sdf["p_fdr_bh"].to_numpy(float) if has_fdr else np.full_like(pvals, np.nan)

        fig, ax = plt.subplots(figsize=(6.2, 3.4))
        ax.bar(x, betas, width=0.6)
        ax.axhline(0, linestyle="--", color="gray")
        ax.set_xticks(x)
        ax.set_xticklabels(bins_sorted["bin_label"].to_list())
        ax.set_ylabel("β (standardized)")
        ax.set_title(f"{out_prefix.upper()} RP_mean_uV ~ {predictor}\n{set_name} (joint + RT)")

        for i in range(len(betas)):
            if not np.isfinite(betas[i]):
                continue
            txt = f"p={pvals[i]:.3f}"
            if np.isfinite(pfdr[i]):
                txt += f"\nFDR={pfdr[i]:.3f}"
            ax.text(x[i], betas[i], txt, ha="center", va="bottom", fontsize=8)

            # star criterion: prefer FDR, else raw p
            sig = (np.isfinite(pfdr[i]) and pfdr[i] < 0.05) if has_fdr else (np.isfinite(pvals[i]) and pvals[i] < 0.05)
            if sig:
                ax.text(x[i], betas[i], "*", ha="center", va="bottom", fontsize=16)

        fig.tight_layout()
        fig.savefig(outfigpath / f"{out_prefix}_{set_name}__betas_{predictor}.svg", dpi=600, bbox_inches="tight")
        plt.close(fig)

# -------------------------
# Partial regression scatter (unique effect in joint+RT model)
# regression line + 95% CI band
# -------------------------
def residualize(y, X):
    """Residuals of y after OLS on X (X should include intercept column)."""
    y = np.asarray(y, float)
    X = np.asarray(X, float)

    keep = np.isfinite(y) & np.isfinite(X).all(axis=1)
    if keep.sum() < 8:
        out = np.full_like(y, np.nan)
        return out

    yk, Xk = y[keep], X[keep]
    beta, *_ = np.linalg.lstsq(Xk, yk, rcond=None)
    resid = yk - Xk @ beta

    out = np.full_like(y, np.nan)
    out[keep] = resid
    return out

if "rt_subj" not in subj_df.columns:
    raise ValueError("rt_subj column missing")

for set_name in electrode_sets:
    sdf_set = subj_df[subj_df["set"] == set_name].copy()
    if sdf_set.empty:
        continue

    for (tmin, tmax), sdf_bin in sdf_set.groupby(["bin_tmin", "bin_tmax"]):
        y  = sdf_bin["rp_mean_uV"].to_numpy(float)
        rt = sdf_bin["rt_subj"].to_numpy(float)

        for target_pred, other_pred in [(pred1_col, pred2_col), (pred2_col, pred1_col)]:
            if target_pred not in sdf_bin.columns or other_pred not in sdf_bin.columns:
                continue

            x       = sdf_bin[target_pred].to_numpy(float)
            x_other = sdf_bin[other_pred].to_numpy(float)

            # build design matrices (intercept + z(other) + z(rt))
            Z = np.column_stack([
                np.ones(len(y)),
                stats.zscore(x_other, nan_policy="omit"),
                stats.zscore(rt, nan_policy="omit"),
            ])

            y_resid = residualize(y, Z)
            x_resid = residualize(x, Z)

            keep = np.isfinite(x_resid) & np.isfinite(y_resid)
            if keep.sum() < 8:
                continue

            df_plot = pd.DataFrame({"x_resid": x_resid[keep], "y_resid": y_resid[keep]})

            fig, ax = plt.subplots(figsize=(3.9, 3.5))
            sns.regplot(
                data=df_plot, x="x_resid", y="y_resid", ax=ax, ci=95,
                scatter_kws={"s": 40, "alpha": 0.85},
                line_kws={"linewidth": 2},
            )

            # correlation annotation
            r, p = stats.pearsonr(df_plot["x_resid"], df_plot["y_resid"])

            ax.axhline(0, linestyle="--", color="gray", linewidth=1)
            ax.axvline(0, linestyle="--", color="gray", linewidth=1)
            ax.set_xlabel(f"{target_pred} (unique residual)")
            ax.set_ylabel("RP mean (µV) (unique residual)")
            ax.set_title(
                f"{out_prefix.upper()} partial effect (joint+RT)\n"
                f"{set_name} {bin_label(tmin,tmax)}  r={r:.2f}, p={p:.3f}"
            )

            fig.tight_layout()
            fig.savefig(
                outfigpath / f"{out_prefix}_{set_name}__partial_{target_pred}__{tmin:.2f}_{tmax:.2f}.svg",
                dpi=600, bbox_inches="tight"
            )
            plt.close(fig)

print(f"\nSaved figures to:\n  {outfigpath}\n")