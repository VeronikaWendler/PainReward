"""
06_eeg_rp_ddm_regression.py — Two-stage RP ~ DDM regression (model 19)

Stage 1 (already run): behavioral DDM with v + a ~ pain_z + money_z (model 19).
Stage 2 (this script): for each participant, extract mean pre-response RP
amplitude from response-locked epochs, then run a group-level OLS regression
predicting RP amplitude from participant-level DDM coefficients, with median
RT as a nuisance covariate.

Inputs
------
  derivatives/hddm/figures/painreward_behavioural_data_mod_19/diagnostics/
      v_a_pain_money.csv
  derivatives/{p}/eeg/erps_decisionresp/{p}_decision_resp_singletrials-epo.fif

Outputs
-------
  derivatives/eeg/erps_massuni_sv_cuelong/mod_19/
      subject_level_rp_means.csv
      group_regress_rp_on_ddm.csv

Run: python 06_eeg_rp_ddm_regression.py [--project-dir PATH]
"""

import argparse
import os
import warnings
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from mne.stats import fdr_correction
from scipy import stats

warnings.simplefilter(action="ignore", category=FutureWarning)

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_DIR = Path(os.getenv("PROJECT_DIR",
                   str(Path(__file__).resolve().parent.parent.parent))).resolve()
DERIV_DIR   = PROJECT_DIR / "derivatives"

DDM_CSV = (DERIV_DIR / "hddm" / "figures"
           / "painreward_behavioural_data_mod_19" / "diagnostics"
           / "v_a_pain_money.csv")
OUT_DIR = DERIV_DIR / "eeg" / "erps_massuni_sv_cuelong" / "mod_19"

# ─────────────────────────────────────────────────────────────────────────────
# PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
RP_WINDOW = (-0.5, -0.1)   # seconds pre-response

ROI_CHS = ("Cz", "CPz", "CP2", "CP1", "C2", "C1", "FC1", "FC2", "FCz")

ELECTRODE_SETS = {
    "roi_CzCPzCP2CP1C2C1FC1FC2FCz": ROI_CHS,
    "ch_Cz":  ("Cz",),
    "ch_CPz": ("CPz",),
    "ch_CP2": ("CP2",),
    "ch_CP1": ("CP1",),
    "ch_C2":  ("C2",),
    "ch_C1":  ("C1",),
    "ch_FC1": ("FC1",),
    "ch_FC2": ("FC2",),
    "ch_FCz": ("FCz",),
}

PREDICTORS    = ["v_pain_z_subj", "v_money_z_subj", "a_pain_z_subj", "a_money_z_subj"]
MIN_GOOD_TRIALS = 8

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _z(x):
    return stats.zscore(np.asarray(x, float), nan_policy="omit")


def _ols_with_t(X, y):
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    keep = np.isfinite(y) & np.isfinite(X).all(axis=1)
    X, y = X[keep], y[keep]
    n, p = X.shape
    df = n - p
    if df <= 0 or n <= p + 1:
        nan_arr = np.full(p, np.nan)
        return nan_arr, nan_arr, nan_arr, nan_arr, nan_arr, nan_arr
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid     = y - X @ beta
    s2        = (resid @ resid) / df
    XtX_inv   = np.linalg.inv(X.T @ X)
    se        = np.sqrt(np.diag(XtX_inv) * s2)
    tvals     = beta / se
    pvals     = 2 * stats.t.sf(np.abs(tvals), df)
    tcrit     = stats.t.ppf(0.975, df)
    return beta, se, tvals, pvals, beta - tcrit * se, beta + tcrit * se


def _group_regress(y, X_pred, pred_names, cov_dict=None):
    """OLS across subjects: y ~ intercept + z(predictors) + z(covariates).

    Returns a list of result dicts, one per predictor.
    """
    y      = np.asarray(y, float)
    X_pred = np.asarray(X_pred, float)
    if X_pred.ndim == 1:
        X_pred = X_pred[:, None]

    keep = np.isfinite(y) & np.isfinite(X_pred).all(axis=1)
    cov_arrays = []
    if cov_dict:
        for v in cov_dict.values():
            v = np.asarray(v, float)
            keep &= np.isfinite(v)
            cov_arrays.append(v)

    y, X_pred = y[keep], X_pred[keep]
    cov_arrays = [c[keep] for c in cov_arrays]
    n = len(y)

    if n < (X_pred.shape[1] + len(cov_arrays) + 5):
        return [{"predictor": nm, "beta_z": np.nan, "se": np.nan,
                 "t": np.nan, "p": np.nan,
                 "ci_low": np.nan, "ci_high": np.nan, "n_subj": n}
                for nm in pred_names]

    X_cols = [np.ones(n)]
    for j in range(X_pred.shape[1]):
        X_cols.append(_z(X_pred[:, j]))
    for c in cov_arrays:
        X_cols.append(_z(c))

    beta, se, tvals, pvals, ci_low, ci_high = _ols_with_t(np.column_stack(X_cols), y)
    return [
        {"predictor": nm,
         "beta_z":  float(beta[j + 1]),
         "se":      float(se[j + 1]),
         "t":       float(tvals[j + 1]),
         "p":       float(pvals[j + 1]),
         "ci_low":  float(ci_low[j + 1]),
         "ci_high": float(ci_high[j + 1]),
         "n_subj":  int(n)}
        for j, nm in enumerate(pred_names)
    ]


def _bh_fdr(pvals):
    pvals = np.asarray(pvals, float)
    out   = np.full_like(pvals, np.nan)
    keep  = np.isfinite(pvals)
    if keep.sum() == 0:
        return out
    _, p_adj  = fdr_correction(pvals[keep], alpha=0.05, method="indep")
    out[keep] = p_adj
    return out


def _extract_mean_rp(epo, window):
    """Return {set_name: mean_µV} for good trials, or None if too few trials."""
    if epo.metadata is not None and "badtrial" in epo.metadata.columns:
        good_idx = np.where(epo.metadata["badtrial"].to_numpy() == 0)[0]
        epo = epo[good_idx]

    if len(epo) < MIN_GOOD_TRIALS:
        return None, len(epo)

    tmin, tmax = window
    tidx = np.where((epo.times >= tmin) & (epo.times <= tmax))[0]
    if len(tidx) < 3:
        raise ValueError(f"Too few time samples in window {window}")

    results = {}
    for set_name, chs in ELECTRODE_SETS.items():
        present = [c for c in chs if c in epo.ch_names]
        if not present:
            continue
        data = epo.copy().pick(present).get_data()  # (n_trials, n_ch, n_times)
        results[set_name] = float(data[:, :, tidx].mean()) * 1e6  # V → µV
    return results, len(epo)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    if not DDM_CSV.exists():
        raise FileNotFoundError(
            f"Model 19 results not found: {DDM_CSV}\n"
            "Run:  python hddm/07_hddm_results.py --version 19"
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Subject-level DDM coefficients ────────────────────────────────────────
    # Coefficients are constant within participant — take the first row per subject.
    ddm = pd.read_csv(DDM_CSV)
    missing_cols = [c for c in PREDICTORS if c not in ddm.columns]
    if missing_cols:
        raise KeyError(f"Missing columns in DDM CSV: {missing_cols}")

    subj_ddm = (
        ddm.groupby("participant")[PREDICTORS + ["rt"]]
        .agg({**{p: "first" for p in PREDICTORS}, "rt": "median"})
        .rename(columns={"rt": "rt_median"})
        .reset_index()
    )
    participants = sorted(subj_ddm["participant"].tolist())
    print(f"Participants in DDM results: {len(participants)}")

    # ── Per-subject RP extraction ─────────────────────────────────────────────
    subj_rows = []
    included, skipped = [], []

    for pa in participants:
        epo_path = (DERIV_DIR / pa / "eeg" / "erps_decisionresp"
                    / f"{pa}_decision_resp_singletrials-epo.fif")
        if not epo_path.exists():
            print(f"  {pa}: epoch file missing — skipping")
            skipped.append(pa)
            continue

        epo = mne.read_epochs(str(epo_path), preload=True, verbose="ERROR")
        rp_by_set, n_good = _extract_mean_rp(epo, RP_WINDOW)

        if rp_by_set is None:
            print(f"  {pa}: only {n_good} good trials — skipping")
            skipped.append(pa)
            continue

        ddm_row = subj_ddm.loc[subj_ddm["participant"] == pa].iloc[0]
        for set_name, rp_mean in rp_by_set.items():
            row = {"participant": pa, "set": set_name,
                   "rp_mean_uV": rp_mean, "n_good_trials": n_good,
                   "rt_median": ddm_row["rt_median"]}
            for pred in PREDICTORS:
                row[pred] = ddm_row[pred]
            subj_rows.append(row)

        included.append(pa)
        print(f"  {pa}: {n_good} good trials")

    print(f"\nIncluded: {len(included)}  Skipped: {len(skipped)}")
    if skipped:
        print(f"  Skipped: {skipped}")

    if not subj_rows:
        raise RuntimeError("No subjects processed — cannot run group regression.")

    subj_df = pd.DataFrame(subj_rows)
    subj_df.to_csv(OUT_DIR / "subject_level_rp_means.csv", index=False)
    print(f"\nSubject-level data → {OUT_DIR / 'subject_level_rp_means.csv'}")

    # ── Group-level regression ────────────────────────────────────────────────
    group_rows = []
    for set_name, sdf in subj_df.groupby("set"):
        res = _group_regress(
            y        = sdf["rp_mean_uV"].to_numpy(float),
            X_pred   = sdf[PREDICTORS].to_numpy(float),
            pred_names = PREDICTORS,
            cov_dict = {"rt_median": sdf["rt_median"].to_numpy(float)},
        )
        for rr in res:
            group_rows.append({"set": set_name, **rr})

    group_df = pd.DataFrame(group_rows)

    # BH-FDR across electrode sets within each predictor
    group_df["p_fdr_bh"] = np.nan
    for pred in group_df["predictor"].unique():
        idx = group_df["predictor"] == pred
        group_df.loc[idx, "p_fdr_bh"] = _bh_fdr(
            group_df.loc[idx, "p"].to_numpy(float)
        )

    group_df.to_csv(OUT_DIR / "group_regress_rp_on_ddm.csv", index=False)
    print(f"Group regression       → {OUT_DIR / 'group_regress_rp_on_ddm.csv'}")

    roi_results = group_df[group_df["set"] == "roi_CzCPzCP2CP1C2C1FC1FC2FCz"]
    print("\nROI results:")
    print(roi_results[["predictor", "beta_z", "se", "t", "p", "p_fdr_bh", "n_subj"]].to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Two-stage RP ~ DDM regression (model 19 coefficients)"
    )
    parser.add_argument("--project-dir", type=Path, default=None,
                        help="Project root (default: inferred from script location)")
    args = parser.parse_args()

    if args.project_dir is not None:
        PROJECT_DIR = args.project_dir.resolve()
        DERIV_DIR   = PROJECT_DIR / "derivatives"
        DDM_CSV     = (DERIV_DIR / "hddm" / "figures"
                       / "painreward_behavioural_data_mod_19" / "diagnostics"
                       / "v_a_pain_money.csv")
        OUT_DIR     = DERIV_DIR / "eeg" / "erps_massuni_sv_cuelong" / "mod_19"

    main()
