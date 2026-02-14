# -*- coding: utf-8 -*-
"""
Step 3: Cross-phase time-time generalization (train time x test time heatmaps)

Robust merge logic:
- epochs.metadata is the master (left table)
- attach beh via trialsnum if trustworthy, else block/trial, else order merge
- allow length mismatches (epochs can be fewer due to rejection; beh can be fewer due to filtering)
- never silently misalign: we log and dump debug heads when needed
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
RUN_MODE = "subset"     # "all" | "subset" | "list"
N_SUBJECTS = 5
SUBJECT_LIST = []

# =============================================================================
# debug controls
# =============================================================================
N_PERM_DEFAULT = 1000

# If True: refuse merges that look dangerous.
# If False: do best-effort left-join on epochs and keep going, but log warnings.
STRICT_MERGE = False

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

RESAMPLE_SFREQ = 50  # 50 Hz => 20 ms per sample

TMIN_STAT = 0.0
TMAX_STAT = 1.0

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

CHANCE_R = 0.0

N_PERM_LABEL = 500
CLUSTER_FORMING_P = 0.01
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
        raise RuntimeError("No subjects selected to run.")

    logprint(f"Subject selection: RUN_MODE={RUN_MODE} | n={len(subs)}")
    logprint("Subjects:", subs)
    return subs


def _coerce_int_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("Int64")


def _ensure_participant_id_column(df: pd.DataFrame, sub: str) -> pd.DataFrame:
    """
    Make sure participant_id exists as a *column* (not index).
    """
    df = df.copy()
    if df.index.name == "participant_id":
        df = df.reset_index()
    if "participant_id" not in df.columns:
        df["participant_id"] = sub
    return df


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


def ensure_trialsnum_in_beh(beh: pd.DataFrame, *, sub: str, phase: str) -> pd.DataFrame:
    """
    If trialsnum exists, validate it. If duplicated / missing / non-numeric, rebuild sequentially.
    If missing, create sequentially.
    """
    beh = beh.copy().reset_index(drop=True)

    if KEY_TRIALNUM in beh.columns:
        beh[KEY_TRIALNUM] = pd.to_numeric(beh[KEY_TRIALNUM], errors="coerce").astype("Int64")
        bad = beh[KEY_TRIALNUM].isna().any() or beh[KEY_TRIALNUM].duplicated().any()
        if bad:
            logprint(f"{sub} {phase}: beh trialsnum invalid or duplicated -> rebuilding 1..N")
            beh = beh.drop(columns=[KEY_TRIALNUM])
        else:
            return beh

    beh[KEY_TRIALNUM] = np.arange(1, len(beh) + 1, dtype=int)
    return beh


def load_beh(sub: str, phase: str) -> pd.DataFrame:
    if phase == "passive":
        beh_path = RAW_DIR / sub / "eeg" / f"{sub}{PASS_BEH_SUFFIX}"
    elif phase == "decision":
        beh_path = RAW_DIR / sub / "eeg" / f"{sub}{DEC_BEH_SUFFIX}"
    else:
        raise ValueError("phase must be 'passive' or 'decision'")

    if not beh_path.exists():
        raise FileNotFoundError(f"Missing beh.tsv for {sub} ({phase}): {beh_path}")

    beh = pd.read_csv(beh_path, sep="\t").reset_index(drop=True)
    beh = _ensure_participant_id_column(beh, sub)

    # Mirror the basic preprocessing filter (important!):
    if "fixcross.started" in beh.columns:
        beh = beh[~beh["fixcross.started"].isna()].copy()

    beh = ensure_trialsnum_in_beh(beh, sub=sub, phase=phase)

    logprint(sub, phase, "beh n=", len(beh),
             "| trialsnum range:",
             int(pd.to_numeric(beh[KEY_TRIALNUM]).min()),
             int(pd.to_numeric(beh[KEY_TRIALNUM]).max()))
    return beh


def merge_beh_into_epochs(epo: mne.Epochs, beh: pd.DataFrame, sub: str, phase: str) -> mne.Epochs:
    """
    Robust merge:
    - Keep epochs as master (left join).
    - Prefer trialsnum.
    - Allow length mismatches (common after rejection).
    - Validate uniqueness where possible; if not, degrade safely with logging.
    """
    if epo.metadata is None:
        raise ValueError(f"{sub} {phase}: epochs has no metadata; cannot merge beh.")

    md = epo.metadata.copy()
    md = md.reset_index(drop=True)
    md = _ensure_participant_id_column(md, sub)

    beh = beh.copy().reset_index(drop=True)
    beh = _ensure_participant_id_column(beh, sub)

    # --- ensure trialsnum numeric when present ---
    if KEY_TRIALNUM in md.columns:
        md[KEY_TRIALNUM] = pd.to_numeric(md[KEY_TRIALNUM], errors="coerce").astype("Int64")
    if KEY_TRIALNUM in beh.columns:
        beh[KEY_TRIALNUM] = pd.to_numeric(beh[KEY_TRIALNUM], errors="coerce").astype("Int64")

    # 1) trialsnum merge (best)
    if (KEY_TRIALNUM in md.columns) and (KEY_TRIALNUM in beh.columns):
        # detect duplicates
        md_dup = md[KEY_TRIALNUM].duplicated().any()
        beh_dup = beh[KEY_TRIALNUM].duplicated().any()

        if md_dup:
            # this should not happen; dump and error
            md.head(50).to_csv(DEBUG_DIR / f"{sub}_{phase}_md_trialsnum_dups_head.csv", index=False)
            raise ValueError(f"{sub} {phase}: epochs metadata trialsnum has duplicates (unexpected).")

        # If beh has duplicates, attempt to collapse by keeping first occurrence per trialsnum
        if beh_dup:
            logprint(f"{sub} {phase}: beh trialsnum has duplicates -> keeping first per trialsnum for merge")
            beh = beh.sort_values(KEY_TRIALNUM).drop_duplicates(subset=[KEY_TRIALNUM], keep="first").reset_index(drop=True)

        # left join on md (epochs), allow md longer/shorter than beh
        merged = md.merge(
            beh,
            on=KEY_TRIALNUM,
            how="left",
            validate="1:1" if not beh_dup else "1:1",  # after drop_duplicates it is safe
            indicator=True,
            suffixes=("", "_beh"),
        )
        n_miss = int((merged["_merge"] == "left_only").sum())
        if n_miss > 0:
            logprint(f"{sub} {phase}: trialsnum merge missing beh for {n_miss}/{len(merged)} epochs (left_only).")
            if STRICT_MERGE:
                merged.head(50).to_csv(DEBUG_DIR / f"{sub}_{phase}_merge_leftonly_head.csv", index=False)
                raise ValueError(f"{sub} {phase}: missing beh rows after trialsnum merge in STRICT_MERGE mode.")
        merged = merged.drop(columns=["_merge"])
        epo.metadata = merged
        return epo

    # 2) block/trial merge
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

        merged = md2.merge(
            beh2,
            on=[KEY_BLOCK, KEY_TRIAL],
            how="left",
            validate="1:1",
            indicator=True,
            suffixes=("", "_beh"),
        )
        n_miss = int((merged["_merge"] == "left_only").sum())
        if n_miss > 0:
            logprint(f"{sub} {phase}: block/trial merge missing beh for {n_miss}/{len(merged)} epochs.")
            if STRICT_MERGE:
                merged.head(50).to_csv(DEBUG_DIR / f"{sub}_{phase}_merge_blocktrial_leftonly_head.csv", index=False)
                raise ValueError(f"{sub} {phase}: missing beh rows after block/trial merge in STRICT_MERGE mode.")
        merged = merged.drop(columns=["_merge"])
        epo.metadata = merged
        return epo

    # 3) order merge (last resort)
    if len(md) == len(beh):
        merged = md.copy()
        for c in beh.columns:
            if c not in merged.columns:
                merged[c] = beh[c].to_numpy()
        epo.metadata = merged
        return epo

    # fail with debug dumps
    md.head(50).to_csv(DEBUG_DIR / f"{sub}_{phase}_epo_md_head.csv", index=False)
    beh.head(50).to_csv(DEBUG_DIR / f"{sub}_{phase}_beh_head.csv", index=False)
    raise ValueError(f"{sub} {phase}: cannot merge beh into epochs (no compatible keys). See debug heads.")


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
    keep_mask = np.isin(levels, keep_levels)
    return X[keep_mask], levels[keep_mask], keep_mask


def to_binary_low_high(levels: np.ndarray) -> np.ndarray:
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
        raise ValueError(f"Binary mapping saw unexpected levels: {bad}")
    return y


def residualize_X_by_nuisance(
    X: np.ndarray,
    nuisance_levels: np.ndarray,
    *,
    model: str = "pain+pain2",
) -> np.ndarray:
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
    model: str = "intercept",
) -> np.ndarray:
    y = y.astype(float).ravel()
    n = y.size

    if model == "intercept":
        D = np.ones((n, 1), dtype=float)
    else:
        if nuisance_levels is None:
            raise ValueError("nuisance_levels required for model != 'intercept'")
        z = nuisance_levels.astype(float).ravel()
        if z.size != n:
            raise ValueError("nuisance_levels length mismatch")

        if model == "pain2":
            D = np.c_[np.ones(n), (z ** 2)]
        elif model == "pain+pain2":
            D = np.c_[np.ones(n), z, (z ** 2)]
        else:
            raise ValueError("model must be 'intercept', 'pain2', or 'pain+pain2'")

    beta, *_ = np.linalg.lstsq(D, y, rcond=None)
    y_hat = D @ beta
    return (y - y_hat).astype(float)


from sklearn.base import clone

def subject_timegen_crossphase_with_nulls_fast(
    timegen: GeneralizingEstimator,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    n_perm_label: int,
    permute: Literal["test", "both"] = "test",
) -> tuple[np.ndarray, np.ndarray]:

    rng = np.random.default_rng(RANDOM_STATE)

    # IMPORTANT: clone so we don't carry state between subjects
    tg = clone(timegen)

    # Fit ONCE
    tg.fit(X_train, y_train)
    mat_real = tg.score(X_test, y_test)

    mats_null = np.empty((n_perm_label, mat_real.shape[0], mat_real.shape[1]), dtype=float)

    if permute not in ("test", "both"):
        raise ValueError("permute must be 'test' or 'both'")

    # Nulls: shuffle test labels (fast)
    for k in range(n_perm_label):
        yte = rng.permutation(y_test)

        # optional: if you REALLY want "both", you can also shuffle y_train
        # but we still don't refit => it's basically the same practical effect as "test"
        # (kept only so you can keep the same interface)
        mats_null[k] = tg.score(X_test, yte)

    return mat_real, mats_null


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
        col = pick_first_existing_col(md, DEC_MONEY_COL_CANDIDATES, label="decision money")
        levels = parse_stim_code_to_level(md[col], prefix="m")
    elif label == "pain":
        col = pick_first_existing_col(md, DEC_PAIN_COL_CANDIDATES, label="decision pain")
        levels = parse_stim_code_to_level(md[col], prefix="p")
    else:
        raise ValueError("label must be 'money' or 'pain'")

    keep = np.isin(levels, LEVELS_ALL)
    epo_f = epo.copy()[keep]
    md_f = epo_f.metadata.reset_index(drop=True)

    if label == "money":
        levels_f = parse_stim_code_to_level(md_f[col], prefix="m")
    else:
        levels_f = parse_stim_code_to_level(md_f[col], prefix="p")

    X = epo_f.get_data()
    return X, levels_f, md_f


# =============================================================================
# Scorers
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
        if len(np.unique(y_true)) < 2:
            return 0.5

        if hasattr(estimator, "predict_proba"):
            s = estimator.predict_proba(X)[:, 1]
        elif hasattr(estimator, "decision_function"):
            s = estimator.decision_function(X)
        else:
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
        if y_pred.dtype.kind in ("f",):
            y_pred = (y_pred >= 0.5).astype(int)
        try:
            return float(accuracy_score(y_true, y_pred))
        except Exception:
            return 0.0
    return _acc_scorer


def make_regression_auc_scorer() -> Callable:
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
    reg = make_pipeline(StandardScaler(), Ridge(alpha=1.0, random_state=RANDOM_STATE))
    if score == "r":
        return GeneralizingEstimator(reg, scoring=make_corr_scorer(), n_jobs=1)
    if score == "auc_from_reg":
        return GeneralizingEstimator(reg, scoring=make_regression_auc_scorer(), n_jobs=1)
    raise ValueError("Unknown regression score")


def make_timegen_estimator_classifier(score: Literal["auc", "acc"]):
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(solver="lbfgs", max_iter=5000, random_state=RANDOM_STATE),
    )
    if score == "auc":
        return GeneralizingEstimator(clf, scoring=make_auc_scorer(), n_jobs=1)
    if score == "acc":
        return GeneralizingEstimator(clf, scoring=make_acc_scorer(), n_jobs=1)
    raise ValueError("Unknown classifier score")


# =============================================================================
# Stats helpers
# =============================================================================
def group_cluster_timegen_labelperm(
    mats_real: np.ndarray,
    mats_null: np.ndarray,
    *,
    chance: float,
    tail: int,
    alpha_cluster: float,
    use_tfce: bool,
):
    X_obs = mats_real - chance
    n_subj, n_tr, n_te = X_obs.shape
    adjacency = combine_adjacency(n_tr, n_te)

    if use_tfce:
        threshold = dict(start=0.0, step=0.2)
    else:
        df = n_subj - 1
        if tail == 0:
            threshold = stats.t.ppf(1 - CLUSTER_FORMING_P / 2, df)
        else:
            threshold = stats.t.ppf(1 - CLUSTER_FORMING_P, df)

    T_obs, clusters, _, _ = permutation_cluster_1samp_test(
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

    obs_cluster_stats = []
    for cl in clusters:
        if cl is None:
            obs_cluster_stats.append(0.0)
            continue
        m = np.asarray(cl, dtype=bool).ravel()
        obs_cluster_stats.append(float(np.sum(T_obs.ravel()[m])) if m.any() else 0.0)
    obs_cluster_stats = np.asarray(obs_cluster_stats, dtype=float)

    n_perm = mats_null.shape[1]
    max_stats = np.zeros(n_perm, dtype=float)

    for k in range(n_perm):
        Xk = mats_null[:, k, :, :] - chance
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

        best = 0.0
        for cl in cl_k:
            if cl is None:
                continue
            m = np.asarray(cl, dtype=bool).ravel()
            if m.any():
                best = max(best, float(np.sum(Tk.ravel()[m])))
        max_stats[k] = best

    cluster_pv = np.ones_like(obs_cluster_stats, dtype=float)
    for i, s in enumerate(obs_cluster_stats):
        cluster_pv[i] = 1.0 if s <= 0 else (np.sum(max_stats >= s) + 1.0) / (n_perm + 1.0)

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
    binary_drop_middle: bool
    control_mode: Literal["none", "eeg_resid_test", "label_resid"]
    resid_model: str
    shuffle_train: bool = False
    shuffle_test: bool = False


def _metric_meta(metric: MetricKind) -> tuple[float, int, str]:
    if metric in ("clf_auc", "clf_acc", "reg_auc"):
        return BIN_CHANCE, 1, "Score − 0.5"
    if metric == "reg_r":
        return CHANCE_R, 0, "Pearson r (pred, true)"
    raise ValueError("Unknown metric")


# =============================================================================
# runner
# =============================================================================
def run_one_analysis(cfg: AnalysisCfg, *, subjects: list[str], n_perm: int):
    out_dir = OUT_DIR / cfg.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    figs_dir = out_dir / "figs"
    figs_dir.mkdir(exist_ok=True)

    chance, tail, cbar_label = _metric_meta(cfg.metric)

    included, skipped = [], []
    mats, nulls = [], []
    times_train = None
    times_test = None

    pbar = tqdm(subjects, desc=cfg.tag, unit="sub", dynamic_ncols=True)
    for sub in pbar:
        try:
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

            if cfg.train_phase == "passive":
                Xtr, tr_levels, md_tr = prepare_passive(epo_tr, cfg.train_label)
            else:
                Xtr, tr_levels, md_tr = prepare_decision(epo_tr, cfg.train_label)

            if cfg.test_phase == "passive":
                Xte, te_levels, md_te = prepare_passive(epo_te, cfg.test_label)
            else:
                Xte, te_levels, md_te = prepare_decision(epo_te, cfg.test_label)

            pain_levels_te = None
            money_levels_te = None
            if cfg.test_phase == "decision":
                pain_levels_te = parse_stim_code_to_level(
                    md_te[pick_first_existing_col(md_te, DEC_PAIN_COL_CANDIDATES, label="painstim")], "p"
                )
                money_levels_te = parse_stim_code_to_level(
                    md_te[pick_first_existing_col(md_te, DEC_MONEY_COL_CANDIDATES, label="moneystim")], "m"
                )

            if cfg.binary_drop_middle:
                Xtr, tr_levels, _ = filter_levels(Xtr, tr_levels, BIN_KEEP_LEVELS)
                Xte, te_levels, keep_mask_te = filter_levels(Xte, te_levels, BIN_KEEP_LEVELS)
                if pain_levels_te is not None:
                    pain_levels_te = pain_levels_te[keep_mask_te]
                if money_levels_te is not None:
                    money_levels_te = money_levels_te[keep_mask_te]

                if len(tr_levels) < 10 or len(te_levels) < 10:
                    raise ValueError("Too few trials after binary filtering (drop 60).")

            if cfg.control_mode == "eeg_resid_test":
                if cfg.test_phase != "decision":
                    raise ValueError("EEG residualization control is for decision TEST only.")
                if pain_levels_te is None:
                    raise ValueError("Missing pain levels for residualization.")
                Xte = residualize_X_by_nuisance(Xte, pain_levels_te, model=cfg.resid_model)

            if cfg.metric in ("clf_auc", "clf_acc"):
                if not cfg.binary_drop_middle:
                    raise ValueError("clf_* metrics require binary_drop_middle=True.")
                ytr = to_binary_low_high(tr_levels)
                yte = to_binary_low_high(te_levels)
                timegen = make_timegen_estimator_classifier("auc" if cfg.metric == "clf_auc" else "acc")

            elif cfg.metric == "reg_r":
                ytr = tr_levels.astype(float)
                yte = te_levels.astype(float)
                if cfg.control_mode == "label_resid":
                    if cfg.test_phase != "decision":
                        raise ValueError("label_resid control is for decision TEST only.")
                    if pain_levels_te is None:
                        raise ValueError("Missing pain levels for label residualization.")
                    ytr = residualize_y_by_nuisance(ytr, None, model="intercept")
                    yte = residualize_y_by_nuisance(yte, pain_levels_te, model=cfg.resid_model)
                timegen = make_timegen_estimator_regression("r")

            elif cfg.metric == "reg_auc":
                if not cfg.binary_drop_middle:
                    raise ValueError("reg_auc requires binary_drop_middle=True.")
                ytr = tr_levels.astype(float)
                yte = to_binary_low_high(te_levels).astype(int)
                if cfg.control_mode == "label_resid":
                    raise ValueError("label_resid not supported for reg_auc (use eeg_resid_test).")
                timegen = make_timegen_estimator_regression("auc_from_reg")

            else:
                raise ValueError("Unknown metric")

            if times_train is None:
                times_train = epo_tr.times.copy()
                times_test = epo_te.times.copy()
            else:
                if len(epo_tr.times) != len(times_train) or np.max(np.abs(epo_tr.times - times_train)) > 1e-9:
                    raise RuntimeError("Train time axis mismatch across subjects.")
                if len(epo_te.times) != len(times_test) or np.max(np.abs(epo_te.times - times_test)) > 1e-9:
                    raise RuntimeError("Test time axis mismatch across subjects.")

            mat_real, mats_null = subject_timegen_crossphase_with_nulls_fast(
                timegen,
                Xtr, np.asarray(ytr),
                Xte, np.asarray(yte),
                n_perm_label=N_PERM_LABEL,
                permute="test",
                )


            mats.append(mat_real)
            nulls.append(mats_null)
            included.append(sub)

            pbar.set_postfix_str(f"{sub} | tr={len(ytr)} te={len(yte)} | mean={np.mean(mat_real):.3f}")

        except Exception as e:
            skipped.append((sub, str(e)))
            logprint(f"Skipped {sub}: {e}")

    if len(mats) < max(5, min(8, len(subjects))):
        raise RuntimeError(f"{cfg.tag}: too few subjects for group stats (n={len(mats)}). Skipped={len(skipped)}")

    mats = np.stack(mats, axis=0)
    nulls = np.stack(nulls, axis=0)

    tr_mask = (times_train >= TMIN_STAT) & (times_train <= TMAX_STAT)
    te_mask = (times_test >= TMIN_STAT) & (times_test <= TMAX_STAT)
    tr_times_stat = times_train[tr_mask]
    te_times_stat = times_test[te_mask]

    mats_stat = mats[:, tr_mask][:, :, te_mask]
    nulls_stat = nulls[:, :, tr_mask][:, :, :, te_mask]

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

    np.savez(
        out_dir / f"{cfg.tag}_timegen_group_results.npz",
        subj_mats=mats,
        subj_nulls=nulls,
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
        out_path=(out_dir / "figs" / f"{cfg.tag}_heatmap.png"),
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

    analyses: list[AnalysisCfg] = []

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

    analyses.append(AnalysisCfg(
        tag="REG_r_trainPASS_money__testDEC_money__5level__NOCTRL__50Hz",
        metric="reg_r",
        train_phase="passive", train_label="money",
        test_phase="decision", test_label="money",
        binary_drop_middle=False,
        control_mode="none",
        resid_model="pain+pain2",
    ))
    analyses.append(AnalysisCfg(
        tag="REG_r_trainPASS_money__testDEC_money__5level__CTRLpain_LABELresid_painPlusPain2__50Hz",
        metric="reg_r",
        train_phase="passive", train_label="money",
        test_phase="decision", test_label="money",
        binary_drop_middle=False,
        control_mode="label_resid",
        resid_model="pain+pain2",
    ))
    analyses.append(AnalysisCfg(
        tag="REG_r_trainPASS_money__testDEC_money__5level__CTRLpain_EEGresidTest_painPlusPain2__50Hz",
        metric="reg_r",
        train_phase="passive", train_label="money",
        test_phase="decision", test_label="money",
        binary_drop_middle=False,
        control_mode="eeg_resid_test",
        resid_model="pain+pain2",
    ))

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

    for cfg in analyses:
        run_one_analysis(cfg, subjects=subjects, n_perm=n_perm)


if __name__ == "__main__":
    main()
