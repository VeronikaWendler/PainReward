# -*- coding: utf-8 -*-
"""
Step 3: Cross-phase time-time generalization (train time x test time heatmaps)

1) Binary classifier (low vs high money; drop middle level 60)
   - chance = 0.5
   - score = AUC (default) or accuracy
   - plots score - 0.5
   - provides no-control and "control pain" (TEST-EEG residualization)

2) Regression variants
   - Primary regression with Pearson r (chance = 0)
   - heatmaps from regression predictions:
    Ridge + AUC scorer on binary low/high target (chance = 0.5; plot -0.5)

3) Pain control versions for each approach
   - No control (baseline)
   - Control pain (in two ways):
       A) label residualization (for regression-r): residualize decision money labels by pain + pain^2
          and mean-center passive money labels (intercept-only).
       B) EEG residualization (for classifier / and for regression-based AUC): residualize TEST EEG by pain + pain^2

4) for speed rn:
   - 20 ms steps RESAMPLE_SFREQ = 50 Hz
   - Fit on ALL training trials for cross-phase (no CV)
   - trialsnum merge

5) Stats + significance
   - For every analysis, saves:
       subj matrices
       group mean
       cluster-based permutation test on the 2D grid (TFCE)
       p_map + cluster table CSV
       summary JSON

OUTPUTS per analysis (in OUT_DIR/tag/):
- NPZ: <tag>_timegen_group_results.npz
- CSV: <tag>_cluster_table.csv
- JSON: <tag>_summary.json
- PNG: figs/<tag>_heatmap.png
"""

from __future__ import annotations

import os
import sys
import json
import re
from pathlib import Path
from dataclasses import dataclass
from typing import Callable, Optional, Literal

import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score

from mne.decoding import GeneralizingEstimator
from mne.stats import permutation_cluster_1samp_test, combine_adjacency
from tqdm.auto import tqdm
from scipy import stats

def logprint(*args):
    print(*args, flush=True)
    sys.stdout.flush()


# =============================================================================
# SUBJECT SELECTION 
# =============================================================================

#   RUN_MODE="all"    -> run all subjects /
#   RUN_MODE="subset" -> run first N_SUBJECTS subjects (sorted)
#   RUN_MODE="list"   -> run explicit SUBJECT_LIST
RUN_MODE = "subset"     # "all" | "subset" | "list"
N_SUBJECTS = 5          # used only if RUN_MODE=="subset"
SUBJECT_LIST = [        # used only if RUN_MODE=="list"
    # "sub-001", "sub-002", "sub-003", "sub-004", "sub-005"
]

# =============================================================================
# debug controls
# =============================================================================
# 5000+ (but debug faster with 500-2000)
N_PERM_DEFAULT = 2000

# =============================================================================
# Paths
# =============================================================================
DATA_DIR_STR = os.getenv("DATA_DIR", "").strip()
OUT_DIR_STR = os.getenv("OUT_DIR", "").strip()
if DATA_DIR_STR == "":
    raise RuntimeError("DATA_DIR env var not set")

RAW_DIR = Path(DATA_DIR_STR).expanduser()
DERIV_DIR = RAW_DIR / "derivatives"

if OUT_DIR_STR != "":
    OUT_BASE = Path(OUT_DIR_STR).expanduser()
    OUT_DIR = OUT_BASE / "statistics" / "mvpa_crossphase_step3_timegen"
else:
    OUT_DIR = DERIV_DIR / "statistics" / "mvpa_crossphase_step3_timegen"

OUT_DIR.mkdir(parents=True, exist_ok=True)
DEBUG_DIR = OUT_DIR / "debug"
DEBUG_DIR.mkdir(parents=True, exist_ok=True)

# ---- PASSIVE ----
PASS_EPO_DIR = Path("eeg") / "erps_passive"
PASS_EPO_SUFFIX = "_passive_cues_singletrials-epo.fif"
PASS_BEH_SUFFIX = "_task-passive_beh.tsv"

# ---- DECISION ----
DEC_EPO_DIR = Path("eeg") / "erps"
DEC_EPO_SUFFIX = "_decision_cues_singletrials-epo.fif"
DEC_BEH_SUFFIX = "_task-decision_beh.tsv"


# =============================================================================
# Parameters
# =============================================================================
RANDOM_STATE = 23
ALPHA_CLUSTER = 0.05

# 20 ms steps
RESAMPLE_SFREQ = 50  # 50 Hz => 20 ms per sample

TMIN_STAT = 0.0
TMAX_STAT = 1.0

# Cross-phase strictness control: fit on ALL training trials (no CV)
CROSSPHASE_FIT_FULL_TRAIN = True

# Metadata keys
KEY_BLOCK = "blocks.thisN"
KEY_TRIAL = "trials.thisN"
KEY_TRIALNUM = "trialsnum"

# Passive columns
COL_COND = "condition"      # 'm' or 'p'
COL_LEVEL = "level"         # 20/40/60/80/100
COND_MONEY = "m"
COND_PAIN = "p"

# Decision columns
DEC_MONEY_COL_CANDIDATES = ["moneystim"]
DEC_PAIN_COL_CANDIDATES = ["painstim"]

LEVEL_CODE_TO_LEVEL = {1: 20, 2: 40, 3: 60, 4: 80, 5: 100}
LEVELS_ALL = np.array([20, 40, 60, 80, 100], dtype=int)

# Binary low/high scheme (drop 60)
BIN_KEEP_LEVELS = np.array([20, 40, 80, 100], dtype=int)
BIN_LOW_LEVELS = {20, 40}
BIN_HIGH_LEVELS = {80, 100}
BIN_CHANCE = 0.5

# Regression-r chance
CHANCE_R = 0.0

# Label-permutation inference
N_PERM_LABEL = 500        # 200–1000 
CLUSTER_FORMING_P = 0.01  # for cluster-mass (if TFCE not used)
USE_TFCE = True           


# =============================================================================
# Utilities
# =============================================================================
def list_subjects(deriv_dir: Path) -> list[str]:
    return sorted([
        p.name for p in deriv_dir.iterdir()
        if p.is_dir() and p.name.startswith("sub-")
    ])


def select_subjects(all_subs: list[str]) -> list[str]:
    """Select subjects based on RUN_MODE / N_SUBJECTS / SUBJECT_LIST."""
    if RUN_MODE == "all":
        subs = all_subs
    elif RUN_MODE == "subset":
        subs = all_subs[:int(N_SUBJECTS)]
    elif RUN_MODE == "list":
        wanted = list(SUBJECT_LIST)
        subs = [s for s in wanted if s in all_subs]
        missing = [s for s in wanted if s not in all_subs]
        if len(missing) > 0:
            logprint("WARNING: requested subjects not found:", missing)
    else:
        raise ValueError("RUN_MODE must be one of: 'all', 'subset', 'list'")

    if len(subs) == 0:
        raise RuntimeError("No subjects selected to run (check RUN_MODE / N_SUBJECTS / SUBJECT_LIST).")

    logprint(f"Subject selection: RUN_MODE={RUN_MODE} | n={len(subs)}")
    logprint("Subjects:", subs)
    return subs


def _coerce_int_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("Int64")


def load_epochs(sub: str, phase: str) -> mne.Epochs:
    if phase == "passive":
        epo_path = DERIV_DIR / sub / PASS_EPO_DIR / f"{sub}{PASS_EPO_SUFFIX}"
    elif phase == "decision":
        epo_path = DERIV_DIR / sub / DEC_EPO_DIR / f"{sub}{DEC_EPO_SUFFIX}"
    else:
        raise ValueError("phase must be 'passive' or 'decision'")

    if not epo_path.exists():
        raise FileNotFoundError(f"Missing epochs for {sub} ({phase}): {epo_path}")

    return mne.read_epochs(epo_path, preload=True, verbose="ERROR")


def load_beh(sub: str, phase: str) -> pd.DataFrame:
    if phase == "passive":
        beh_path = RAW_DIR / sub / "eeg" / f"{sub}{PASS_BEH_SUFFIX}"
    elif phase == "decision":
        beh_path = RAW_DIR / sub / "eeg" / f"{sub}{DEC_BEH_SUFFIX}"
    else:
        raise ValueError("phase must be 'passive' or 'decision'")

    if not beh_path.exists():
        raise FileNotFoundError(f"Missing beh.tsv for {sub} ({phase}): {beh_path}")

    beh = pd.read_csv(beh_path, sep="\t")


    beh = beh.reset_index(drop=True)
    beh = ensure_trialsnum_in_beh(beh, sub=sub, phase=phase)
    logprint(sub, phase, "beh trialsnum range:",
             int(beh[KEY_TRIALNUM].min()), int(beh[KEY_TRIALNUM].max()))


    return beh


def ensure_trialsnum_in_beh(beh: pd.DataFrame, *, sub: str, phase: str) -> pd.DataFrame:
    if KEY_TRIALNUM in beh.columns:
        return beh

    beh = beh.copy().reset_index(drop=True)

    # robust: sequential id after whatever filtering you did in load_beh()
    beh[KEY_TRIALNUM] = np.arange(1, len(beh) + 1, dtype=int)

    # optional sanity checks
    if beh[KEY_TRIALNUM].duplicated().any():
        raise ValueError(f"{sub} {phase}: trialsnum still has duplicates unexpectedly.")

    return beh



def merge_beh_into_epochs(epo: mne.Epochs, beh: pd.DataFrame, sub: str, phase: str) -> mne.Epochs:
    """
    CRITICAL: If epochs have trialsnum, beh MUST have trialsnum, otherwise we refuse fallback merges.
    This prevents silent misalignment.
    """
    if epo.metadata is None:
        raise ValueError(f"{sub} {phase}: epochs has no metadata; cannot merge beh.")
    md = epo.metadata.reset_index(drop=True).copy()

    if (KEY_TRIALNUM in md.columns) and (KEY_TRIALNUM in beh.columns):
        if len(md) != len(beh):
            raise ValueError(f"{sub} {phase}: len mismatch md={len(md)} vs beh={len(beh)} (sequential trialsnum would misalign).")


    # ---- Mandatory trialsnum merge if epochs contain trialsnum ----
    if (KEY_TRIALNUM in md.columns) and (KEY_TRIALNUM not in beh.columns):
        raise ValueError(
            f"{sub} {phase}: epochs have {KEY_TRIALNUM} but beh does not — refusing fallback merge."
        )

    # ---- trialsnum merge ----    
    if (KEY_TRIALNUM in md.columns) and (KEY_TRIALNUM in beh.columns):
        merged = md.merge(
            beh,
            on=KEY_TRIALNUM,
            how="left",
            validate="1:1",
            indicator=True,
            suffixes=("", "_beh"),
        )
        n_missing = int((merged["_merge"] == "left_only").sum())
        logprint(f"{sub} {phase} merge: {n_missing}/{len(merged)} rows have NO beh match")
        merged = merged.drop(columns=["_merge"])
        epo.metadata = merged
        return epo

    # ---- key merge (block/trial) ----
    can_key_merge = (
        (KEY_BLOCK in md.columns) and (KEY_TRIAL in md.columns) and
        (KEY_BLOCK in beh.columns) and (KEY_TRIAL in beh.columns)
    )
    if can_key_merge:
        md2 = md.copy()
        beh2 = beh.copy()
        md2[KEY_BLOCK] = _coerce_int_series(md2[KEY_BLOCK])
        md2[KEY_TRIAL] = _coerce_int_series(md2[KEY_TRIAL])
        beh2[KEY_BLOCK] = _coerce_int_series(beh2[KEY_BLOCK])
        beh2[KEY_TRIAL] = _coerce_int_series(beh2[KEY_TRIAL])
        merged = md2.merge(beh2, on=[KEY_BLOCK, KEY_TRIAL], how="left", validate="1:1")
        epo.metadata = merged
        return epo

    # ---- order merge (last resort) ----
    if len(md) == len(beh):
        merged = md.copy()
        for c in beh.columns:
            if c not in merged.columns:
                merged[c] = beh[c].to_numpy()
        epo.metadata = merged
        return epo

    md.head(30).to_csv(DEBUG_DIR / f"{sub}_{phase}_epo_md_head.csv", index=False)
    beh.head(30).to_csv(DEBUG_DIR / f"{sub}_{phase}_beh_head.csv", index=False)
    raise ValueError(f"{sub} {phase}: cannot merge beh into epochs; see debug heads.")


def drop_badtrials(epo: mne.Epochs) -> mne.Epochs:
    if epo.metadata is None:
        return epo
    if "badtrial" in epo.metadata.columns:
        bad = epo.metadata["badtrial"].fillna(0).astype(int).to_numpy()
        if bad.sum() > 0:
            epo = epo.copy()[bad == 0]
    return epo


def ensure_resampled(epo: mne.Epochs) -> mne.Epochs:
    if RESAMPLE_SFREQ is not None:
        epo = epo.copy().resample(RESAMPLE_SFREQ, npad="auto")
    return epo


def pick_first_existing_col(df: pd.DataFrame, candidates: list[str], *, label: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"Could not find {label} column in metadata. Tried: {candidates}")


def parse_stim_code_to_level(series: pd.Series, prefix: str) -> np.ndarray:
    s = series.astype(str).str.strip().str.lower()
    pat = rf"^{re.escape(prefix)}\s*([1-5])$"
    codes = s.str.extract(pat, expand=False)
    if codes.isna().any():
        bad = s[codes.isna()].unique()[:10]
        raise ValueError(f"Unexpected {prefix} stim codes (examples): {bad}")
    codes_int = codes.astype(int).to_numpy()
    levels = np.array([LEVEL_CODE_TO_LEVEL[int(c)] for c in codes_int], dtype=int)
    return levels


def filter_levels(X: np.ndarray, levels: np.ndarray, keep_levels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return X_f, levels_f, keep_mask."""
    keep_mask = np.isin(levels, keep_levels)
    return X[keep_mask], levels[keep_mask], keep_mask


def to_binary_low_high(levels: np.ndarray) -> np.ndarray:
    """
    Map {20,40} -> 0, {80,100} -> 1. (Assumes 60 already dropped.)
    """
    y = np.full(levels.shape, -1, dtype=int)
    for i, v in enumerate(levels.astype(int).tolist()):
        if v in BIN_LOW_LEVELS:
            y[i] = 0
        elif v in BIN_HIGH_LEVELS:
            y[i] = 1
        else:
            y[i] = -1
    if np.any(y < 0):
        bad = np.unique(levels[y < 0])
        raise ValueError(f"Binary mapping saw unexpected levels (did you forget to drop 60?): {bad}")
    return y


def residualize_X_by_nuisance(
    X: np.ndarray,                 # (n_trials, n_ch, n_t)
    nuisance_levels: np.ndarray,    # (n_trials,)
    *,
    model: str = "pain+pain2",      # "pain2" or "pain+pain2"
) -> np.ndarray:
    """
    Residualize nuisance from EEG features (typically apply on TEST EEG).
    """
    n_trials, n_ch, n_t = X.shape
    z = nuisance_levels.astype(float)

    if model == "pain2":
        D = np.c_[np.ones(n_trials), (z ** 2)]
    elif model == "pain+pain2":
        D = np.c_[np.ones(n_trials), z, (z ** 2)]
    else:
        raise ValueError("model must be 'pain2' or 'pain+pain2'")

    X_flat = X.reshape(n_trials, -1)
    beta, *_ = np.linalg.lstsq(D, X_flat, rcond=None)
    R = X_flat - (D @ beta)
    return R.reshape(n_trials, n_ch, n_t)


def residualize_y_by_nuisance(
    y: np.ndarray,
    nuisance_levels: Optional[np.ndarray],
    *,
    model: str = "intercept",  # "intercept", "pain2", "pain+pain2"
) -> np.ndarray:
    """
    Residualize labels instead of EEG.

    - For passive (no pain nuisance): use model="intercept" -> mean-center y.
    - For decision (control pain): use model="pain+pain2" (or "pain2") with nuisance_levels=pain.
    """
    y = y.astype(float).ravel()
    n = y.size

    if model == "intercept":
        D = np.ones((n, 1), dtype=float)
    else:
        if nuisance_levels is None:
            raise ValueError("nuisance_levels required for model != 'intercept'")
        z = nuisance_levels.astype(float).ravel()
        if z.size != n:
            raise ValueError("nuisance_levels length mismatch for label residualization")

        if model == "pain2":
            D = np.c_[np.ones(n), (z ** 2)]
        elif model == "pain+pain2":
            D = np.c_[np.ones(n), z, (z ** 2)]
        else:
            raise ValueError("model must be 'intercept', 'pain2', or 'pain+pain2'")

    beta, *_ = np.linalg.lstsq(D, y, rcond=None)
    y_hat = D @ beta
    return (y - y_hat).astype(float)

def subject_timegen_crossphase_with_nulls(
    timegen: GeneralizingEstimator,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    n_perm_label: int,
    permute: Literal["train", "test", "both"] = "train",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns
    -------
    mat_real : (n_train_t, n_test_t)
    mats_null: (n_perm_label, n_train_t, n_test_t)

    MVPA: permute labels and refit each perm.
    """
    rng = np.random.default_rng(RANDOM_STATE)

    # Real
    timegen.fit(X_train, y_train)
    mat_real = timegen.score(X_test, y_test)

    # Nulls
    mats_null = []
    for _ in range(n_perm_label):
        if permute == "train":
            ytr = rng.permutation(y_train)
            yte = y_test
        elif permute == "test":
            ytr = y_train
            yte = rng.permutation(y_test)
        elif permute == "both":
            ytr = rng.permutation(y_train)
            yte = rng.permutation(y_test)
        else:
            raise ValueError("permute must be 'train', 'test', or 'both'")

        tg = timegen  # safe because we refit below; if you want, clone via sklearn.base.clone
        tg.fit(X_train, ytr)
        mats_null.append(tg.score(X_test, yte))

    return mat_real, np.asarray(mats_null)

# =============================================================================
# Selectors per phase
# =============================================================================
def prepare_passive(epo: mne.Epochs, label: str):
    if epo.metadata is None:
        raise ValueError("No metadata.")
    md = epo.metadata.reset_index(drop=True)

    cond = md[COL_COND].astype(str).str.lower().to_numpy()
    keep = (cond == (COND_MONEY if label == "money" else COND_PAIN))
    epo_f = epo.copy()[keep]
    md_f = epo_f.metadata.reset_index(drop=True)

    levels_f = md_f[COL_LEVEL].to_numpy(dtype=int)
    X = epo_f.get_data()
    return X, levels_f, md_f


def prepare_decision(epo: mne.Epochs, label: str):
    if epo.metadata is None:
        raise ValueError("No metadata.")
    md = epo.metadata.reset_index(drop=True)

    if label == "money":
        col = pick_first_existing_col(md, DEC_MONEY_COL_CANDIDATES, label="decision money (moneystim)")
        levels = parse_stim_code_to_level(md[col], prefix="m")
    elif label == "pain":
        col = pick_first_existing_col(md, DEC_PAIN_COL_CANDIDATES, label="decision pain (painstim)")
        levels = parse_stim_code_to_level(md[col], prefix="p")
    else:
        raise ValueError("label must be 'money' or 'pain'")

    keep = np.isin(levels, LEVELS_ALL)
    epo_f = epo.copy()[keep]
    md_f = epo_f.metadata.reset_index(drop=True)

    # recompute after filtering
    if label == "money":
        levels_f = parse_stim_code_to_level(md_f[col], prefix="m")
    else:
        levels_f = parse_stim_code_to_level(md_f[col], prefix="p")

    X = epo_f.get_data()
    return X, levels_f, md_f


# =============================================================================
# Scorers (compatible with GeneralizingEstimator)
# =============================================================================
def make_corr_scorer() -> Callable:
    def _corr_scorer(estimator, X, y_true) -> float:
        y_pred = estimator.predict(X)
        y_true = np.asarray(y_true, dtype=float).ravel()
        y_pred = np.asarray(y_pred, dtype=float).ravel()
        if y_true.size < 3:
            return 0.0
        if np.std(y_true) < 1e-12 or np.std(y_pred) < 1e-12:
            return 0.0
        r = np.corrcoef(y_true, y_pred)[0, 1]
        return 0.0 if np.isnan(r) else float(r)
    return _corr_scorer


def make_auc_scorer() -> Callable:
    def _auc_scorer(estimator, X, y_true) -> float:
        y_true = np.asarray(y_true, dtype=int).ravel()
        if y_true.size < 3:
            return 0.5
        # Need both classes
        if len(np.unique(y_true)) < 2:
            return 0.5

        # Try predict_proba -> decision_function fallback
        if hasattr(estimator, "predict_proba"):
            s = estimator.predict_proba(X)[:, 1]
        elif hasattr(estimator, "decision_function"):
            s = estimator.decision_function(X)
        else:
            # last resort: hard predictions (AUC not meaningful; return 0.5)
            return 0.5

        try:
            return float(roc_auc_score(y_true, s))
        except Exception:
            return 0.5
    return _auc_scorer


def make_acc_scorer() -> Callable:
    def _acc_scorer(estimator, X, y_true) -> float:
        y_true = np.asarray(y_true, dtype=int).ravel()
        if y_true.size < 1:
            return 0.0
        y_pred = estimator.predict(X).astype(int).ravel()
        # if y_pred are floats (e.g., regression), threshold at 0.5
        if y_pred.dtype.kind in ("f",):
            y_pred = (y_pred >= 0.5).astype(int)
        try:
            return float(accuracy_score(y_true, y_pred))
        except Exception:
            return 0.0
    return _acc_scorer


def make_regression_auc_scorer(threshold: float = 60.0) -> Callable:
    """
    For a Ridge model: turn continuous predictions into an AUC score against binary labels.
    This gives you an "accuracy-like" heatmap from regression.
    """
    def _reg_auc_scorer(estimator, X, y_true) -> float:
        y_true = np.asarray(y_true, dtype=int).ravel()
        if y_true.size < 3 or len(np.unique(y_true)) < 2:
            return 0.5
        y_pred = np.asarray(estimator.predict(X), dtype=float).ravel()
        try:
            return float(roc_auc_score(y_true, y_pred))
        except Exception:
            return 0.5
    return _reg_auc_scorer


# =============================================================================
# Estimators
# =============================================================================
def make_timegen_estimator_regression(score: Literal["r", "auc_from_reg"]):
    if score == "r":
        reg = make_pipeline(
            StandardScaler(),
            Ridge(alpha=1.0, random_state=RANDOM_STATE),
        )
        return GeneralizingEstimator(reg, scoring=make_corr_scorer(), n_jobs=1)

    if score == "auc_from_reg":
        reg = make_pipeline(
            StandardScaler(),
            Ridge(alpha=1.0, random_state=RANDOM_STATE),
        )
        # scorer expects binary y_true (0/1)
        return GeneralizingEstimator(reg, scoring=make_regression_auc_scorer(), n_jobs=1)

    raise ValueError("Unknown regression score")


def make_timegen_estimator_classifier(score: Literal["auc", "acc"]):
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            solver="lbfgs",
            max_iter=5000,
            random_state=RANDOM_STATE,
        )
    )
    if score == "auc":
        return GeneralizingEstimator(clf, scoring=make_auc_scorer(), n_jobs=1)
    if score == "acc":
        return GeneralizingEstimator(clf, scoring=make_acc_scorer(), n_jobs=1)
    raise ValueError("Unknown classifier score")


# =============================================================================
# Cross-phase time×time per subject (fit full train)
# =============================================================================
def subject_timegen_crossphase(
    timegen: GeneralizingEstimator,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    shuffle_train: bool,
    shuffle_test: bool,
) -> np.ndarray:
    rng = np.random.default_rng(RANDOM_STATE)
    y_tr = rng.permutation(y_train) if shuffle_train else y_train
    y_te = rng.permutation(y_test) if shuffle_test else y_test

    if not CROSSPHASE_FIT_FULL_TRAIN:
        raise RuntimeError("This script is optimized for cross-phase full-train fitting. Keep CROSSPHASE_FIT_FULL_TRAIN=True.")

    timegen.fit(X_train, y_tr)
    mat = timegen.score(X_test, y_te)
    return mat



def group_cluster_timegen_labelperm(
    mats_real: np.ndarray,          # (n_subj, n_tr, n_te)
    mats_null: np.ndarray,          # (n_subj, n_perm, n_tr, n_te)
    *,
    chance: float,
    tail: int,
    alpha_cluster: float,
    use_tfce: bool,
):
    """
    MVPA-style inference:
    - Compute observed group statistic map from real data
    - Build permutation distribution by, for each perm index:
        take one permuted map per subject, compute group statistic
    - Use cluster correction via max cluster stat (handled by MNE permutation_cluster_1samp_test
      when we feed the permuted samples as X and let it permute signs?).
    
    Here we do it in a standard/transparent way:
    - Build X_obs = real - chance
    - Build X_perm[k] = perm_mean - chance
    - use MNEs cluster test on the real maps but estimate corrected p-values
      from the label-perm distribution of max cluster stats.

    Returns p_map corrected (FWER) and cluster table info.
    """
    X_obs = mats_real - chance
    n_subj, n_tr, n_te = X_obs.shape
    adjacency = combine_adjacency(n_tr, n_te)

    # ---- cluster-forming threshold ----
    threshold = None
    if use_tfce:
        threshold = dict(start=0.0, step=0.2)  
    else:
        # cluster-forming threshold at p=CLUSTER_FORMING_P
        df = n_subj - 1
        if tail == 0:
            t_thr = stats.t.ppf(1 - CLUSTER_FORMING_P / 2, df)
        else:
            t_thr = stats.t.ppf(1 - CLUSTER_FORMING_P, df)
        threshold = t_thr

    # ---- get observed clusters using MNE, but with n_permutations=0 (no internal permutation) ----
    # MNE doesn't have n_permutations=0, so we do 1 permutation and ignore its pvals,
    # using it only to get clusters and T_obs.
    T_obs, clusters, cluster_pv_dummy, _ = permutation_cluster_1samp_test(
        X_obs.reshape(n_subj, -1),
        n_permutations=1,
        threshold=threshold,
        tail=tail,
        adjacency=adjacency,
        out_type="mask",
        n_jobs=1,
        seed=RANDOM_STATE,
    )
    T_obs = T_obs.reshape(n_tr, n_te)

    # ---- compute observed cluster stats (cluster mass) from T_obs and clusters ----
    obs_cluster_stats = []
    for cl in clusters:
        if cl is None:
            obs_cluster_stats.append(0.0)
            continue
        m = np.asarray(cl, dtype=bool).ravel()
        if not m.any():
            obs_cluster_stats.append(0.0)
            continue
        # cluster mass = sum of T within cluster (common in MVPA)
        obs_cluster_stats.append(float(np.sum(T_obs.ravel()[m])))

    obs_cluster_stats = np.asarray(obs_cluster_stats, dtype=float)

    # ---- build permutation distribution of max cluster stat ----
    # For each perm: take one perm map per subject, compute group t-map, find max cluster mass
    n_perm = mats_null.shape[1]
    max_stats = np.zeros(n_perm, dtype=float)

    for k in range(n_perm):
        Xk = mats_null[:, k, :, :] - chance  # (n_subj, n_tr, n_te)

        Tk, cl_k, _, _ = permutation_cluster_1samp_test(
            Xk.reshape(n_subj, -1),
            n_permutations=1,
            threshold=threshold,
            tail=tail,
            adjacency=adjacency,
            out_type="mask",
            n_jobs=1,
            seed=RANDOM_STATE + k + 1,
        )
        Tk = Tk.reshape(n_tr, n_te)

        # max cluster mass for this perm
        best = 0.0
        for cl in cl_k:
            if cl is None:
                continue
            m = np.asarray(cl, dtype=bool).ravel()
            if not m.any():
                continue
            stat = float(np.sum(Tk.ravel()[m]))
            if stat > best:
                best = stat
        max_stats[k] = best

    # ---- corrected p-value per observed cluster using max-stat ----
    # p = proportion of perm max >= observed cluster stat
    cluster_pv = np.ones_like(obs_cluster_stats, dtype=float)
    for i, s in enumerate(obs_cluster_stats):
        if s <= 0:
            cluster_pv[i] = 1.0
        else:
            cluster_pv[i] = (np.sum(max_stats >= s) + 1.0) / (n_perm + 1.0)

    # ---- build corrected p_map by assigning each cluster its corrected p ----
    p_map = np.ones((n_tr, n_te), dtype=float)
    for cl, p in zip(clusters, cluster_pv):
        if cl is None:
            continue
        m = np.asarray(cl, dtype=bool).ravel()
        if m.size == n_tr * n_te and m.any():
            p_map.ravel()[m] = np.minimum(p_map.ravel()[m], float(p))

    return dict(
        T_obs=T_obs,
        clusters=clusters,
        cluster_pv=cluster_pv,
        p_map=p_map,
        thresh_used=("tfce" if use_tfce else f"t@p<{CLUSTER_FORMING_P}"),
        max_stats=max_stats,
    )

# =============================================================================
# Group stats on 2D grid
# =============================================================================
def group_cluster_timegen(mats_by_subj: np.ndarray, *, chance: float, tail: int, n_perm: int):
    """
    mats_by_subj: (n_subj, n_train_t, n_test_t)
    chance: subtract this before stats
    tail:
      1 -> positive clusters (metric > chance)
      0 -> two-sided
     -1 -> negative clusters
    """
    X = mats_by_subj - chance
    n_subj, n_tr, n_te = X.shape

    adjacency = combine_adjacency(n_tr, n_te)
    X_flat = X.reshape(n_subj, n_tr * n_te)

    tfce_thresh = dict(start=0.0, step=0.2)
    try:
        T_obs, clusters, cluster_pv, H0 = permutation_cluster_1samp_test(
            X_flat,
            n_permutations=n_perm,
            threshold=tfce_thresh,
            tail=tail,
            adjacency=adjacency,
            out_type="mask",
            n_jobs=1,
            seed=RANDOM_STATE,
        )
        thresh_used = "tfce"
    except Exception:
        T_obs, clusters, cluster_pv, H0 = permutation_cluster_1samp_test(
            X_flat,
            n_permutations=n_perm,
            threshold=None,
            tail=tail,
            adjacency=adjacency,
            out_type="mask",
            n_jobs=1,
            seed=RANDOM_STATE,
        )
        thresh_used = "threshold=None"

    p_map = np.ones((n_tr * n_te,), dtype=float)
    for cl, p in zip(clusters, cluster_pv):
        if cl is None:
            continue
        m = np.asarray(cl, dtype=bool).ravel()
        if m.size == p_map.size and m.any():
            p_map[m] = np.minimum(p_map[m], float(p))

    return dict(
        T_obs=T_obs.reshape(n_tr, n_te),
        clusters=clusters,
        cluster_pv=np.asarray(cluster_pv, dtype=float),
        p_map=p_map.reshape(n_tr, n_te),
        thresh_used=thresh_used,
    )


# =============================================================================
# Plotting + saving
# =============================================================================
def plot_timegen_heatmap(
    mean_mat: np.ndarray,
    times_train: np.ndarray,
    times_test: np.ndarray,
    p_map: np.ndarray,
    *,
    chance: float,
    title: str,
    out_path: Path,
    alpha: float,
    cbar_label: str,
):
    data = mean_mat - chance

    fig, ax = plt.subplots(figsize=(6.0, 5.2))
    im = ax.imshow(
        data,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=[times_test[0], times_test[-1], times_train[0], times_train[-1]],
    )
    cbar = fig.colorbar(im, ax=ax, shrink=0.9, pad=0.02)
    cbar.set_label(cbar_label)

    sig = (p_map < alpha)
    if np.any(sig):
        ax.contour(
            sig.astype(float),
            levels=[0.5],
            linewidths=1.5,
            origin="lower",
            extent=[times_test[0], times_test[-1], times_train[0], times_train[-1]],
        )

    ax.axvline(0, linewidth=1)
    ax.axhline(0, linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Test time (s)")
    ax.set_ylabel("Train time (s)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def save_cluster_table_2d(out_csv: Path, times_train: np.ndarray, times_test: np.ndarray, stats_out: dict):
    rows = []
    clusters = stats_out["clusters"]
    cluster_pv = stats_out["cluster_pv"]

    n_tr = len(times_train)
    n_te = len(times_test)

    for i, (cl, p) in enumerate(zip(clusters, cluster_pv)):
        if cl is None:
            continue
        m = np.asarray(cl, dtype=bool).ravel()
        if m.size != n_tr * n_te or (not m.any()):
            continue
        m2 = m.reshape(n_tr, n_te)

        tr_inds = np.where(np.any(m2, axis=1))[0]
        te_inds = np.where(np.any(m2, axis=0))[0]

        rows.append(dict(
            cluster=i,
            p_value=float(p),
            train_t_start_s=float(times_train[tr_inds[0]]),
            train_t_end_s=float(times_train[tr_inds[-1]]),
            test_t_start_s=float(times_test[te_inds[0]]),
            test_t_end_s=float(times_test[te_inds[-1]]),
            n_cells=int(m.sum()),
        ))

    df = pd.DataFrame(rows).sort_values("p_value") if len(rows) else pd.DataFrame(
        columns=["cluster", "p_value", "train_t_start_s", "train_t_end_s",
                 "test_t_start_s", "test_t_end_s", "n_cells"]
    )
    df.to_csv(out_csv, index=False)


# =============================================================================
# Analysis configuration
# =============================================================================
MetricKind = Literal["clf_auc", "clf_acc", "reg_r", "reg_auc"]

@dataclass
class AnalysisCfg:
    tag: str
    metric: MetricKind
    train_phase: str
    train_label: str
    test_phase: str
    test_label: str
    # binary drop-60 selection
    binary_drop_middle: bool
    # pain control switches
    control_mode: Literal["none", "eeg_resid_test", "label_resid"]  # label_resid only meaningful for reg_r here
    resid_model: str  # "pain+pain2" or "pain2" (used by EEG or label control)
    shuffle_train: bool = False
    shuffle_test: bool = False


def _metric_meta(metric: MetricKind) -> tuple[float, int, str]:
    """
    Returns: (chance, tail, cbar_label)
    """
    if metric in ("clf_auc", "clf_acc", "reg_auc"):
        # chance 0.5, one-sided positive clusters
        return BIN_CHANCE, 1, "Score − 0.5"
    if metric == "reg_r":
        # r chance 0, two-sided
        return CHANCE_R, 0, "Pearson r (pred, true)"
    raise ValueError("Unknown metric")


# =============================================================================
# runner
# =============================================================================
def run_one_analysis(
    cfg: AnalysisCfg,
    *,
    subjects: list[str],
    n_perm: int,   # kept for backward-compat; not used by label-perm inference below
):
    out_dir = OUT_DIR / cfg.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    figs_dir = out_dir / "figs"
    figs_dir.mkdir(exist_ok=True)

    chance, tail, cbar_label = _metric_meta(cfg.metric)

    included, skipped = [], []
    mats = []
    nulls = [] 
    times_train = None
    times_test = None

    pbar = tqdm(subjects, desc=cfg.tag, unit="sub", dynamic_ncols=True)
    for sub in pbar:
        try:
            # ---- load train/test epochs + merge beh ----
            epo_tr = ensure_resampled(drop_badtrials(
                merge_beh_into_epochs(load_epochs(sub, cfg.train_phase),
                                      load_beh(sub, cfg.train_phase),
                                      sub=sub, phase=cfg.train_phase)
            ))
            epo_te = ensure_resampled(drop_badtrials(
                merge_beh_into_epochs(load_epochs(sub, cfg.test_phase),
                                      load_beh(sub, cfg.test_phase),
                                      sub=sub, phase=cfg.test_phase)
            ))

            # ---- select train/test trials + labels (levels) ----
            if cfg.train_phase == "passive":
                Xtr, tr_levels, md_tr = prepare_passive(epo_tr, cfg.train_label)
            else:
                Xtr, tr_levels, md_tr = prepare_decision(epo_tr, cfg.train_label)

            if cfg.test_phase == "passive":
                Xte, te_levels, md_te = prepare_passive(epo_te, cfg.test_label)
            else:
                Xte, te_levels, md_te = prepare_decision(epo_te, cfg.test_label)

            # ---- nuisance arrays (decision only) ----
            pain_levels_te = None
            money_levels_te = None
            if cfg.test_phase == "decision":
                pain_levels_te = parse_stim_code_to_level(
                    md_te[pick_first_existing_col(md_te, DEC_PAIN_COL_CANDIDATES, label="painstim")],
                    "p"
                )
                money_levels_te = parse_stim_code_to_level(
                    md_te[pick_first_existing_col(md_te, DEC_MONEY_COL_CANDIDATES, label="moneystim")],
                    "m"
                )

            # ---- optional binary filtering (drop 60) ----
            if cfg.binary_drop_middle:
                Xtr, tr_levels, _ = filter_levels(Xtr, tr_levels, BIN_KEEP_LEVELS)
                Xte, te_levels, keep_mask_te = filter_levels(Xte, te_levels, BIN_KEEP_LEVELS)

                if pain_levels_te is not None:
                    pain_levels_te = pain_levels_te[keep_mask_te]
                if money_levels_te is not None:
                    money_levels_te = money_levels_te[keep_mask_te]

                if len(tr_levels) < 10 or len(te_levels) < 10:
                    raise ValueError("Too few trials after binary filtering (drop 60).")

            # ---- pain control: EEG residualization on TEST ----
            if cfg.control_mode == "eeg_resid_test":
                if cfg.test_phase != "decision":
                    raise ValueError("EEG residualization control is implemented for decision TEST only.")
                if pain_levels_te is None:
                    raise ValueError("Missing pain levels for test residualization.")
                Xte = residualize_X_by_nuisance(Xte, pain_levels_te, model=cfg.resid_model)

            # ---- build y + timegen estimator ----
            if cfg.metric in ("clf_auc", "clf_acc"):
                if not cfg.binary_drop_middle:
                    raise ValueError("clf_* metrics require binary_drop_middle=True (drop 60).")
                ytr = to_binary_low_high(tr_levels)
                yte = to_binary_low_high(te_levels)
                timegen = make_timegen_estimator_classifier("auc" if cfg.metric == "clf_auc" else "acc")

            elif cfg.metric == "reg_r":
                ytr = tr_levels.astype(float)
                yte = te_levels.astype(float)

                if cfg.control_mode == "label_resid":
                    if cfg.test_phase != "decision":
                        raise ValueError("label_resid control is implemented for decision TEST only.")
                    if pain_levels_te is None:
                        raise ValueError("Missing pain levels for label residualization.")

                    ytr = residualize_y_by_nuisance(ytr, None, model="intercept")
                    yte = residualize_y_by_nuisance(yte, pain_levels_te, model=cfg.resid_model)

                timegen = make_timegen_estimator_regression("r")

            elif cfg.metric == "reg_auc":
                if not cfg.binary_drop_middle:
                    raise ValueError("reg_auc requires binary_drop_middle=True (drop 60).")

                ytr = tr_levels.astype(float)
                yte = to_binary_low_high(te_levels).astype(int)

                if cfg.control_mode == "label_resid":
                    raise ValueError("label_resid is not supported for reg_auc (use eeg_resid_test).")

                timegen = make_timegen_estimator_regression("auc_from_reg")

            else:
                raise ValueError("Unknown metric")

            # ---- time axes ----
            if times_train is None:
                times_train = epo_tr.times.copy()
                times_test = epo_te.times.copy()
            else:
                if len(epo_tr.times) != len(times_train) or np.max(np.abs(epo_tr.times - times_train)) > 1e-9:
                    raise RuntimeError("Train time axis mismatch across subjects.")
                if len(epo_te.times) != len(times_test) or np.max(np.abs(epo_te.times - times_test)) > 1e-9:
                    raise RuntimeError("Test time axis mismatch across subjects.")

            # ---- compute subject matrix + nulls (LABEL PERMUTATION; MVPA standard) ----
            mat_real, mats_null = subject_timegen_crossphase_with_nulls(
                timegen,
                Xtr, np.asarray(ytr),
                Xte, np.asarray(yte),
                n_perm_label=N_PERM_LABEL,
                permute="train",
            )

            mats.append(mat_real)
            nulls.append(mats_null)
            included.append(sub)

            pbar.set_postfix_str(
                f"{sub} | tr={len(ytr)} te={len(yte)} | mean={np.mean(mat_real):.3f}"
            )

        except Exception as e:
            skipped.append((sub, str(e)))
            logprint(f"Skipped {sub}: {e}")

    if len(mats) < max(5, min(8, len(subjects))):
        raise RuntimeError(f"{cfg.tag}: too few subjects for group stats (n={len(mats)}). Skipped={len(skipped)}")

    mats = np.stack(mats, axis=0)       # (n_subj, n_tr, n_te)
    nulls = np.stack(nulls, axis=0)     # (n_subj, n_perm_label, n_tr, n_te)

    # ---- restrict to stats window ----
    tr_mask = (times_train >= TMIN_STAT) & (times_train <= TMAX_STAT)
    te_mask = (times_test >= TMIN_STAT) & (times_test <= TMAX_STAT)
    tr_times_stat = times_train[tr_mask]
    te_times_stat = times_test[te_mask]

    mats_stat = mats[:, tr_mask][:, :, te_mask]
    nulls_stat = nulls[:, :, tr_mask][:, :, :, te_mask]

    # ---- MVPA-style group stats (label permutation null) ----
    stats_out = group_cluster_timegen_labelperm(
        mats_real=mats_stat,
        mats_null=nulls_stat,
        chance=chance,
        tail=tail,
        alpha_cluster=ALPHA_CLUSTER,
        use_tfce=USE_TFCE,
    )

    mean_mat = np.mean(mats_stat, axis=0)
    sem_mat = np.std(mats_stat, axis=0, ddof=1) / np.sqrt(mats_stat.shape[0])

    # ---- save ----
    np.savez(
        out_dir / f"{cfg.tag}_timegen_group_results.npz",
        subj_mats=mats,
        subj_nulls=nulls,  # can be big; remove if storage is an issue
        subj_mats_stat=mats_stat,
        subj_nulls_stat=nulls_stat,
        times_train=times_train,
        times_test=times_test,
        times_train_stat=tr_times_stat,
        times_test_stat=te_times_stat,
        included=np.array(included, dtype=object),
        skipped=np.array(skipped, dtype=object),
        metric=cfg.metric,
        chance=float(chance),
        mean_mat=mean_mat,
        sem_mat=sem_mat,
        T_obs=stats_out["T_obs"],
        p_map=stats_out["p_map"],
        cluster_pv=stats_out["cluster_pv"],
        thresh_used=stats_out["thresh_used"],
        alpha_cluster=float(ALPHA_CLUSTER),
        resample_sfreq=int(RESAMPLE_SFREQ),
        fit_full_train=bool(CROSSPHASE_FIT_FULL_TRAIN),
        # permutation meta
        max_stats=stats_out["max_stats"],
        n_perm_label=int(N_PERM_LABEL),
        permute_labels="train",
        use_tfce=bool(USE_TFCE),
        cluster_forming_p=float(CLUSTER_FORMING_P),
    )

    save_cluster_table_2d(
        out_csv=out_dir / f"{cfg.tag}_cluster_table.csv",
        times_train=tr_times_stat,
        times_test=te_times_stat,
        stats_out=stats_out,
    )

    meta = dict(
        tag=cfg.tag,
        metric=cfg.metric,
        n_subjects=int(mats_stat.shape[0]),
        n_train_times=int(mats_stat.shape[1]),
        n_test_times=int(mats_stat.shape[2]),
        chance=float(chance),
        alpha=float(ALPHA_CLUSTER),
        tail=int(tail),
        thresh_used=stats_out["thresh_used"],
        min_cluster_p=float(np.min(stats_out["cluster_pv"])) if len(stats_out["cluster_pv"]) else 1.0,
        included_subjects=included,
        train_phase=cfg.train_phase,
        train_label=cfg.train_label,
        test_phase=cfg.test_phase,
        test_label=cfg.test_label,
        binary_drop_middle=bool(cfg.binary_drop_middle),
        control_mode=cfg.control_mode,
        resid_model=cfg.resid_model,
        resample_sfreq=int(RESAMPLE_SFREQ),
        tmin_stat=float(TMIN_STAT),
        tmax_stat=float(TMAX_STAT),
        fit_full_train=bool(CROSSPHASE_FIT_FULL_TRAIN),
        # permutation meta
        n_perm_label=int(N_PERM_LABEL),
        permute_labels="train",
        use_tfce=bool(USE_TFCE),
        cluster_forming_p=float(CLUSTER_FORMING_P),
    )
    with open(out_dir / f"{cfg.tag}_summary.json", "w") as f:
        json.dump(meta, f, indent=2)

    plot_timegen_heatmap(
        mean_mat=mean_mat,
        times_train=tr_times_stat,
        times_test=te_times_stat,
        p_map=stats_out["p_map"],
        chance=chance,
        title=f"{cfg.tag} ({cfg.metric})",
        out_path=figs_dir / f"{cfg.tag}_heatmap.png",
        alpha=ALPHA_CLUSTER,
        cbar_label=cbar_label,
    )

    logprint(
        f"DONE {cfg.tag}: included={len(included)} | skipped={len(skipped)} | "
        f"min cluster p={(float(np.min(stats_out['cluster_pv'])) if len(stats_out['cluster_pv']) else 1.0):.6f}"
    )


# =============================================================================
# Main
# =============================================================================
def main():
    mne.set_log_level("WARNING")

    all_subs = list_subjects(DERIV_DIR)
    subjects = select_subjects(all_subs)
    n_perm = int(N_PERM_DEFAULT)


    # -------------------------------------------------------------------------
    analyses: list[AnalysisCfg] = []

    # (A) classifier low vs high money (drop 60), score=AUC, plot AUC-0.5
    analyses.append(AnalysisCfg(
        tag="CLF_AUC_trainPASS_money__testDEC_money__binLowHigh_drop60__NOCTRL__50Hz",
        metric="clf_auc",
        train_phase="passive", train_label="money",
        test_phase="decision", test_label="money",
        binary_drop_middle=True,
        control_mode="none",
        resid_model="pain+pain2",
    ))
    analyses.append(AnalysisCfg(
        tag="CLF_AUC_trainPASS_money__testDEC_money__binLowHigh_drop60__CTRLpain_EEGresidTest_painPlusPain2__50Hz",
        metric="clf_auc",
        train_phase="passive", train_label="money",
        test_phase="decision", test_label="money",
        binary_drop_middle=True,
        control_mode="eeg_resid_test",
        resid_model="pain+pain2",
    ))

    # (B) REGRESSION: Pearson r on 5 levels (chance=0), baseline
    analyses.append(AnalysisCfg(
        tag="REG_r_trainPASS_money__testDEC_money__5level__NOCTRL__50Hz",
        metric="reg_r",
        train_phase="passive", train_label="money",
        test_phase="decision", test_label="money",
        binary_drop_middle=False,
        control_mode="none",
        resid_model="pain+pain2",
    ))
    # (B-control preferred) REGRESSION r with LABEL residualization (controls pain without touching EEG)
    analyses.append(AnalysisCfg(
        tag="REG_r_trainPASS_money__testDEC_money__5level__CTRLpain_LABELresid_painPlusPain2__50Hz",
        metric="reg_r",
        train_phase="passive", train_label="money",
        test_phase="decision", test_label="money",
        binary_drop_middle=False,
        control_mode="label_resid",
        resid_model="pain+pain2",
    ))
    # (B-control alternative) REGRESSION r with TEST-EEG residualization (sometimes more conservative)
    analyses.append(AnalysisCfg(
        tag="REG_r_trainPASS_money__testDEC_money__5level__CTRLpain_EEGresidTest_painPlusPain2__50Hz",
        metric="reg_r",
        train_phase="passive", train_label="money",
        test_phase="decision", test_label="money",
        binary_drop_middle=False,
        control_mode="eeg_resid_test",
        resid_model="pain+pain2",
    ))

    # (C) REGRESSION -> "accuracy-like" heatmap: Ridge predictions scored as AUC on binary low/high
    # chance=0.5, plot AUC-0.5 (this answers your "regression but still want accuracy-chance" request)
    analyses.append(AnalysisCfg(
        tag="REG_AUC_trainPASS_money__testDEC_money__binLowHigh_drop60__NOCTRL__50Hz",
        metric="reg_auc",
        train_phase="passive", train_label="money",
        test_phase="decision", test_label="money",
        binary_drop_middle=True,
        control_mode="none",
        resid_model="pain+pain2",
    ))
    analyses.append(AnalysisCfg(
        tag="REG_AUC_trainPASS_money__testDEC_money__binLowHigh_drop60__CTRLpain_EEGresidTest_painPlusPain2__50Hz",
        metric="reg_auc",
        train_phase="passive", train_label="money",
        test_phase="decision", test_label="money",
        binary_drop_middle=True,
        control_mode="eeg_resid_test",
        resid_model="pain+pain2",
    ))

    # -------------------------------------------------------------------------
    # RUN
    # -------------------------------------------------------------------------
    for cfg in analyses:
        run_one_analysis(cfg, subjects=subjects, n_perm=n_perm)


if __name__ == "__main__":
    main()
