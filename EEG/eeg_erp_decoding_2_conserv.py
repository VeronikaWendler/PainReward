# -*- coding: utf-8 -*-
"""
Step 1 (Passive): Time-resolved decoding of money level (and pain as control)

- Loads passive epochs from derivatives
- Loads passive beh.tsv from raw painrewardeegdata
- Merges beh columns (condition, level, blocks.thisN, trials.thisN) into epochs.metadata
- Runs time-resolved decoding (AUC) for:
    money-only: low (20/40) vs high (80/100), drop 60
    pain-only : same (positive control)
- Group-level cluster permutation test over time on (AUC - 0.5)

Optional (recommended once):
- Decode stimulus type: money vs pain (sanity check that pipeline works)

Outputs:
  derivatives/statistics/mvpa_passive_step1/
"""
# libraries

from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.linear_model import Ridge


from mne.decoding import SlidingEstimator, cross_val_multiscore
from mne.stats import permutation_cluster_1samp_test
from scipy import stats
from tqdm.auto import tqdm
import json


# -----------------------------
# Paths 
# -----------------------------

DATA_DIR_STR = os.getenv("DATA_DIR", "").strip()
OUT_DIR_STR = os.getenv("OUT_DIR", "").strip()

if DATA_DIR_STR == "":
    raise RuntimeError("DATA_DIR env var not set")

RAW_DIR = Path(DATA_DIR_STR).expanduser()
DERIV_DIR = RAW_DIR / "derivatives"

# Output folder
if OUT_DIR_STR != "":
    OUT_BASE = Path(OUT_DIR_STR).expanduser()
    OUT_DIR = OUT_BASE / "statistics" / "mvpa_passive_step1_conserv"
else:
    OUT_DIR = DERIV_DIR / "statistics" / "mvpa_passive_step1_conserv"

OUT_DIR.mkdir(parents=True, exist_ok=True)
DEBUG_DIR = OUT_DIR / "debug"
DEBUG_DIR.mkdir(parents=True, exist_ok=True)

# epochs/beh locations relative to subject folder
EPO_DIR = Path("eeg") / "erps_passive"
EPO_SUFFIX = "_passive_cues_singletrials-epo.fif"
BEH_SUFFIX = "_task-passive_beh.tsv"

# -----------------------------
# Decoding params
# -----------------------------
RANDOM_STATE = 23
N_SPLITS = 5
N_PERM = 5000
ALPHA_CLUSTER = 0.05
CHANCE = 0.5
KEY_TRIALNUM = "trialsnum"

# downsample for speed
RESAMPLE_SFREQ = 256  # set None to keep original

# extra validation
RUN_SHUFFLE_CONTROL = True  # should return ~chance
RUN_STIMTYPE_SANITY = True  # set True once: decode money vs pain across all passive trials

# Passive beh columns
COL_COND = "condition"   # 'p' or 'm'
COL_LEVEL = "level"      # 20/40/60/80/100

KEY_BLOCK = "blocks.thisN"   # 0..4 
KEY_TRIAL = "trials.thisN"   # 0..39 within each block

COND_MONEY = "m"
COND_PAIN = "p"

TMIN_STAT = 0.0
TMAX_STAT = 0.8

RUN_BINARY = False          # AUC + balanced accuracy (low vs high; drops 60)
RUN_REGRESSION = True     # ridge regression decoding (all levels; keeps 60)
RUN_STIMTYPE = False       # sanity check (money vs pain)
RUN_SHUFFLE = True        # shuffle controls (for whichever analyses you turned on)

# -----------------------------
# output folders

OUT_DIR_BIN = OUT_DIR / "binary_lowhigh_auc_bacc"
OUT_DIR_REG = OUT_DIR / "regression_ridgecorr"
OUT_DIR_STIM = OUT_DIR / "stimtype"

for _d in [OUT_DIR_BIN, OUT_DIR_REG, OUT_DIR_STIM]:
    _d.mkdir(parents=True, exist_ok=True)

DEBUG_DIR_BIN = OUT_DIR_BIN / "debug"
DEBUG_DIR_REG = OUT_DIR_REG / "debug"
DEBUG_DIR_STIM = OUT_DIR_STIM / "debug"
for _d in [DEBUG_DIR_BIN, DEBUG_DIR_REG, DEBUG_DIR_STIM]:
    _d.mkdir(parents=True, exist_ok=True)

def list_subjects(deriv_dir: Path) -> list[str]:
    return sorted([p.name for p in deriv_dir.iterdir()
                   if p.is_dir() and p.name.startswith("sub-")])


def load_passive_epochs(sub: str) -> mne.Epochs:
    epo_path = DERIV_DIR / sub / EPO_DIR / f"{sub}{EPO_SUFFIX}"
    if not epo_path.exists():
        raise FileNotFoundError(f"Missing epochs for {sub}: {epo_path}")
    return mne.read_epochs(epo_path, preload=True, verbose="ERROR")


def load_passive_beh(sub: str) -> pd.DataFrame:
    beh_path = RAW_DIR / sub / "eeg" / f"{sub}{BEH_SUFFIX}"
    if not beh_path.exists():
        raise FileNotFoundError(f"Missing beh.tsv for {sub}: {beh_path}")

    beh = pd.read_csv(beh_path, sep="\t")

    if "fixcross.started" in beh.columns:
        beh = beh[~beh["fixcross.started"].isna()].copy()

    beh = beh.reset_index(drop=True)
    beh[KEY_TRIALNUM] = np.arange(1, len(beh) + 1)
    return beh

def select_trials_money_or_pain_regression(
    epo: mne.Epochs,
    which: str,
    drop_level60: bool = False,   # IMPORTANT: for regression I'd keep 60 by default
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Regression version:
      y = continuous level (20/40/60/80/100)
    """
    if epo.metadata is None:
        raise ValueError("Epochs has no metadata.")

    md = epo.metadata.reset_index(drop=True)
    for col in [COL_COND, COL_LEVEL]:
        if col not in md.columns:
            raise ValueError(f"Missing metadata column '{col}'. Have: {md.columns.tolist()}")

    cond = md[COL_COND].astype(str).str.lower().to_numpy()
    if which == "money":
        keep = (cond == COND_MONEY)
    elif which == "pain":
        keep = (cond == COND_PAIN)
    else:
        raise ValueError("which must be 'money' or 'pain'")

    epo_f = epo.copy()[keep]
    md_f = epo_f.metadata.reset_index(drop=True)

    levels = md_f[COL_LEVEL].to_numpy(dtype=float)

    if drop_level60:
        keep2 = ~np.isin(levels, [60])
        epo_f = epo_f.copy()[keep2]
        md_f = epo_f.metadata.reset_index(drop=True)
        levels = md_f[COL_LEVEL].to_numpy(dtype=float)

    if len(epo_f) < 10:
        raise ValueError(f"Too few trials after filtering for {which}. n={len(epo_f)}")

    # y is continuous / ordinal numeric
    y = levels.astype(float)

    X = epo_f.get_data()
    times = epo_f.times.copy()
    return X, y, times, md_f

def _corr_scorer(estimator, X, y_true) -> float:
    """Pearson r between y_true and model predictions. Returns 0 if undefined."""
    y_pred = estimator.predict(X)
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()

    if y_true.size < 3:
        return 0.0
    if np.std(y_true) < 1e-12 or np.std(y_pred) < 1e-12:
        return 0.0

    r = np.corrcoef(y_true, y_pred)[0, 1]
    if np.isnan(r):
        return 0.0
    return float(r)

def subject_decode_regression(
    X: np.ndarray,
    y: np.ndarray,                      # continuous (levels)
    shuffle: bool = False,
    groups: np.ndarray | None = None,
):
    rng = np.random.default_rng(RANDOM_STATE)
    y_use = rng.permutation(y) if shuffle else y

    reg = make_pipeline(
        StandardScaler(),
        Ridge(alpha=1.0, random_state=RANDOM_STATE),
    )

    time_decod = SlidingEstimator(reg, scoring=_corr_scorer)

    cv = None
    if groups is not None:
        groups = np.asarray(groups)
        ok = ~pd.isna(groups)
        if ok.sum() == len(groups):
            n_groups = len(np.unique(groups))
            if n_groups >= 2:
                n_splits = min(N_SPLITS, n_groups)
                cv = GroupKFold(n_splits=n_splits)

    if cv is None:
        # For regression, plain KFold is typical; but to keep it simple, keep StratifiedKFold out.
        # We'll do KFold-like splits using np.arange (balanced is not essential for regression)
        from sklearn.model_selection import KFold
        cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    scores = cross_val_multiscore(
        time_decod,
        X,
        y_use,
        cv=cv,
        groups=groups if isinstance(cv, GroupKFold) else None,
        n_jobs=1
    )

    cv_used = "GroupKFold" if isinstance(cv, GroupKFold) else "KFold"
    return scores.mean(axis=0), cv_used

def _coerce_int_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("Int64")


def merge_beh_into_epochs(epo: mne.Epochs, beh: pd.DataFrame, sub: str) -> mne.Epochs:
    if epo.metadata is None:
        raise ValueError(f"{sub}: epochs has no metadata at all; cannot merge beh.")

    md = epo.metadata.reset_index(drop=True).copy()

    if (COL_COND in md.columns) and (COL_LEVEL in md.columns):
        print(f"{sub}: epochs.metadata already contains {COL_COND}+{COL_LEVEL} (no merge needed)")
        return epo

    if (KEY_TRIALNUM in md.columns) and (KEY_TRIALNUM in beh.columns):
        cols_to_add = [KEY_TRIALNUM, COL_COND, COL_LEVEL]

        for extra in [KEY_BLOCK, KEY_TRIAL]:
            if extra in beh.columns:
                cols_to_add.append(extra)

        merged = md.merge(
            beh[cols_to_add],
            on=KEY_TRIALNUM,
            how="left",
            validate="1:1",
        )

        if merged[COL_COND].isna().any() or merged[COL_LEVEL].isna().any():
            md.head(50).to_csv(DEBUG_DIR / f"{sub}_epo_md_head.csv", index=False)
            beh.head(50).to_csv(DEBUG_DIR / f"{sub}_beh_head.csv", index=False)
            raise ValueError(f"{sub}: trialsnum-merge produced unlabeled epochs.")

        epo.metadata = merged
        print(f"{sub}: merged beh into epochs using '{KEY_TRIALNUM}'")
        return epo

    for col in [COL_COND, COL_LEVEL]:
        if col not in beh.columns:
            beh.head(30).to_csv(DEBUG_DIR / f"{sub}_beh_head.csv", index=False)
            raise ValueError(f"{sub}: beh.tsv missing required column '{col}'. Found: {list(beh.columns)}")

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

        beh_small = beh2[[KEY_BLOCK, KEY_TRIAL, COL_COND, COL_LEVEL]].copy()

        merged = md2.merge(
            beh_small,
            on=[KEY_BLOCK, KEY_TRIAL],
            how="left",
            validate="1:1"
        )

        if merged[COL_COND].isna().any() or merged[COL_LEVEL].isna().any():
            md.head(30).to_csv(DEBUG_DIR / f"{sub}_epo_md_head.csv", index=False)
            beh.head(30).to_csv(DEBUG_DIR / f"{sub}_beh_head.csv", index=False)
            n_bad = int(merged[COL_COND].isna().sum())
            raise ValueError(
                f"{sub}: key-merge produced {n_bad} unlabeled epochs. "
                "Likely keys don't align between events-derived metadata and beh.tsv."
            )

        epo.metadata = merged
        print(f"{sub}: merged beh into epochs using KEYS ({KEY_BLOCK}, {KEY_TRIAL})")
        return epo

    # ---- final fallback: order merge only if lengths match ----
    if len(md) == len(beh):
        merged = md.copy()
        merged[COL_COND] = beh[COL_COND].to_numpy()
        merged[COL_LEVEL] = beh[COL_LEVEL].to_numpy()

        # carry block/trial if possible
        for extra in [KEY_BLOCK, KEY_TRIAL]:
            if extra in beh.columns:
                merged[extra] = beh[extra].to_numpy()

        epo.metadata = merged
        print(f"{sub}: merged beh into epochs by ORDER (len match: {len(md)})")
        return epo

    md.head(50).to_csv(DEBUG_DIR / f"{sub}_epo_md_head.csv", index=False)
    beh.head(50).to_csv(DEBUG_DIR / f"{sub}_beh_head.csv", index=False)

    raise ValueError(
        f"{sub}: cannot merge beh into epochs.\n"
        f"- epochs n={len(md)}; beh n={len(beh)} (not equal, so order-based merge not possible)\n"
        f"- keys '{KEY_BLOCK}'/'{KEY_TRIAL}' not present in BOTH epochs.metadata and beh.tsv\n"
        "See debug CSVs in OUT_DIR/debug for columns present in each."
    )



def make_binary_labels(level: np.ndarray) -> np.ndarray:
    """
    level values expected: 20/40/60/80/100
    returns y: 0=low (20,40), 1=high (80,100)
    Assumes level 60 already removed.
    """
    level = np.asarray(level, dtype=float)
    y = np.full(len(level), -1, dtype=int)
    y[np.isin(level, [20, 40])] = 0
    y[np.isin(level, [80, 100])] = 1
    return y


def select_trials_money_or_pain(epo: mne.Epochs, which: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """
    which: "money" or "pain"
    Returns X, y, times, md_used
    """
    if epo.metadata is None:
        raise ValueError("Epochs has no metadata.")

    md = epo.metadata.reset_index(drop=True)

    for col in [COL_COND, COL_LEVEL]:
        if col not in md.columns:
            raise ValueError(f"Missing metadata column '{col}'. Have: {md.columns.tolist()}")

    cond = md[COL_COND].astype(str).str.lower().to_numpy()
    if which == "money":
        keep = (cond == COND_MONEY)
    elif which == "pain":
        keep = (cond == COND_PAIN)
    else:
        raise ValueError("which must be 'money' or 'pain'")

    epo_f = epo.copy()[keep]
    md_f = epo_f.metadata.reset_index(drop=True)

    # drop level 60
    levels = md_f[COL_LEVEL].to_numpy(dtype=float)
    keep2 = ~np.isin(levels, [60])
    epo_f = epo_f.copy()[keep2]
    md_f = epo_f.metadata.reset_index(drop=True)

    if len(epo_f) < 10:
        raise ValueError(f"Too few trials after filtering for {which}. n={len(epo_f)}")

    y = make_binary_labels(md_f[COL_LEVEL].to_numpy(dtype=float))
    if np.any(y < 0):
        raise ValueError(f"Unlabeled trials exist. Levels seen: {np.unique(md_f[COL_LEVEL])}")

    X = epo_f.get_data()  # (n_trials, n_chans, n_times)
    times = epo_f.times.copy()
    return X, y, times, md_f


def select_trials_stimtype(epo: mne.Epochs) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Sanity check: decode stimulus type across all passive trials.
    labels: 0 = pain (p), 1 = money (m)
    """
    if epo.metadata is None:
        raise ValueError("Epochs has no metadata.")

    md = epo.metadata.reset_index(drop=True)
    if COL_COND not in md.columns:
        raise ValueError(f"Missing metadata column '{COL_COND}'.")

    cond = md[COL_COND].astype(str).str.lower().to_numpy()
    keep = np.isin(cond, [COND_PAIN, COND_MONEY])
    epo_f = epo.copy()[keep]
    md_f = epo_f.metadata.reset_index(drop=True)

    cond_f = md_f[COL_COND].astype(str).str.lower().to_numpy()
    y = np.full(len(cond_f), -1, dtype=int)
    y[cond_f == COND_PAIN] = 0
    y[cond_f == COND_MONEY] = 1

    if np.any(y < 0):
        raise ValueError(f"Unexpected condition labels: {np.unique(cond_f)}")

    X = epo_f.get_data()
    times = epo_f.times.copy()
    return X, y, times, md_f


def subject_decode(
    X: np.ndarray,
    y: np.ndarray,
    shuffle: bool = False,
    groups: np.ndarray | None = None,
    metric: str = "roc_auc",  # "roc_auc" or "balanced_accuracy" or "accuracy"
):
    rng = np.random.default_rng(RANDOM_STATE)
    y_use = rng.permutation(y) if shuffle else y

    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            solver="liblinear",
            max_iter=2000,
            random_state=RANDOM_STATE,
            class_weight="balanced",
        )
    )

    time_decod = SlidingEstimator(clf, scoring=metric)

    cv = None
    if groups is not None:
        groups = np.asarray(groups)
        ok = ~pd.isna(groups)
        if ok.sum() == len(groups):
            n_groups = len(np.unique(groups))
            if n_groups >= 2:
                n_splits = min(N_SPLITS, n_groups)
                cv = GroupKFold(n_splits=n_splits)

    if cv is None:
        cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    scores = cross_val_multiscore(
        time_decod,
        X,
        y_use,
        cv=cv,
        groups=groups if isinstance(cv, GroupKFold) else None,
        n_jobs=1
    )
    cv_used = "GroupKFold" if isinstance(cv, GroupKFold) else "StratifiedKFold"
    return scores.mean(axis=0), cv_used


def _cluster_to_bool_mask(cl, n_times: int) -> np.ndarray | None:
    if isinstance(cl, (tuple, list)):
        if len(cl) == 0:
            return None
        if len(cl) == 1:
            return _cluster_to_bool_mask(cl[0], n_times)
        return _cluster_to_bool_mask(cl[0], n_times)

    if isinstance(cl, slice):
        m = np.zeros(n_times, dtype=bool)
        m[cl] = True
        return m

    arr = np.asarray(cl)

    if arr.dtype == bool:
        arr = arr.squeeze()
        if arr.ndim == 1 and arr.size == n_times:
            return arr
        arr = arr.ravel()
        if arr.size == n_times:
            return arr
        return None
    try:
        idx = arr.astype(int).ravel()
    except Exception:
        return None

    if idx.size == 0:
        return None
    if np.any(idx < 0) or np.any(idx >= n_times):
        return None

    m = np.zeros(n_times, dtype=bool)
    m[idx] = True
    return m


def group_cluster(scores_by_subj: np.ndarray, times: np.ndarray):
    """
    Cluster permutation test on (Score - chance) across time.
    Tries TFCE first (threshold dict). Falls back if not supported.
    """
    X = scores_by_subj - CHANCE  # (n_subj, n_times)

    # TFCE settings
    tfce_thresh = dict(start=0.0, step=0.2)

    try:
        T_obs, clusters, cluster_pv, H0 = permutation_cluster_1samp_test(
            X,
            n_permutations=N_PERM,
            threshold=tfce_thresh,   # TFCE
            tail=1,
            out_type="mask",
            n_jobs=1,
            seed=RANDOM_STATE,
            buffer_size=None,
        )
        t_thresh_used = "tfce"
    except Exception as e:
        try:
            T_obs, clusters, cluster_pv, H0 = permutation_cluster_1samp_test(
                X,
                n_permutations=N_PERM,
                threshold=None,
                tail=1,
                out_type="mask",
                n_jobs=1,
                seed=RANDOM_STATE,
                buffer_size=None,
            )
            t_thresh_used = "threshold=None"
        except Exception:
            # original parametric cluster-forming threshold
            p_form = 0.01
            t_thresh = stats.t.ppf(1 - p_form / 2, df=X.shape[0] - 1)
            T_obs, clusters, cluster_pv, H0 = permutation_cluster_1samp_test(
                X,
                n_permutations=N_PERM,
                threshold=t_thresh,
                tail=1,
                out_type="mask",
                n_jobs=1,
                seed=RANDOM_STATE,
                buffer_size=None,
            )
            t_thresh_used = float(t_thresh)

    # p_map = np.ones(len(times), dtype=float)
    # for cl, p in zip(clusters, cluster_pv):
    #     p_map[cl] = np.minimum(p_map[cl], p)


    p_map = np.ones(len(times), dtype=float)
    clusters_fixed = []
    
    for cl, p in zip(clusters, cluster_pv):
        m = _cluster_to_bool_mask(cl, len(times))
        clusters_fixed.append(m)  
        if m is None or not m.any():
            continue
        p_map[m] = np.minimum(p_map[m], p)
    
    return dict(
        T_obs=T_obs,
        clusters=clusters_fixed,  
        cluster_pv=cluster_pv,
        p_map=p_map,
        t_thresh=t_thresh_used,
    )
    

def plot_group(scores_by_subj: np.ndarray, times: np.ndarray, p_map: np.ndarray, title: str, out_png: Path):
    mean = scores_by_subj.mean(axis=0)
    sem = scores_by_subj.std(axis=0, ddof=1) / np.sqrt(scores_by_subj.shape[0])

    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(times, mean, linewidth=2)
    ax.fill_between(times, mean - sem, mean + sem, alpha=0.25)

    ax.axhline(CHANCE, linestyle="--", linewidth=1)
    ax.axvline(0, linestyle="--", linewidth=1)

    sig = p_map < ALPHA_CLUSTER
    if np.any(sig):
        ax.fill_between(times, CHANCE - 0.02, CHANCE - 0.01, where=sig, alpha=0.9)

    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Decoding (AUC)")
    ax.set_ylim(0.35, 0.85)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)

def plot_group_metric(scores_by_subj, times, p_map, title, out_png, ylabel, chance=0.5, ylim=(0.35, 0.85)):
    mean = scores_by_subj.mean(axis=0)
    sem = scores_by_subj.std(axis=0, ddof=1) / np.sqrt(scores_by_subj.shape[0])

    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(times, mean, linewidth=2)
    ax.fill_between(times, mean - sem, mean + sem, alpha=0.25)

    ax.axhline(chance, linestyle="--", linewidth=1)
    ax.axvline(0, linestyle="--", linewidth=1)

    sig = p_map < ALPHA_CLUSTER
    if np.any(sig):
        ax.fill_between(times, chance - 0.02, chance - 0.01, where=sig, alpha=0.9)

    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)

def group_cluster_metric(scores_by_subj: np.ndarray, times: np.ndarray, *, chance: float, tail: int = 1):
    """
    Cluster permutation test on (scores - chance) across time.
    tail=1 tests for scores > chance.
    """
    X = scores_by_subj - chance  # (n_subj, n_times)

    tfce_thresh = dict(start=0.0, step=0.2)

    try:
        T_obs, clusters, cluster_pv, H0 = permutation_cluster_1samp_test(
            X,
            n_permutations=N_PERM,
            threshold=tfce_thresh,
            tail=tail,
            out_type="mask",
            n_jobs=1,
            seed=RANDOM_STATE,
            buffer_size=None,
        )
        t_thresh_used = "tfce"
    except Exception:
        T_obs, clusters, cluster_pv, H0 = permutation_cluster_1samp_test(
            X,
            n_permutations=N_PERM,
            threshold=None,
            tail=tail,
            out_type="mask",
            n_jobs=1,
            seed=RANDOM_STATE,
            buffer_size=None,
        )
        t_thresh_used = "threshold=None"

    p_map = np.ones(len(times), dtype=float)
    clusters_fixed = []
    for cl, p in zip(clusters, cluster_pv):
        m = _cluster_to_bool_mask(cl, len(times))
        clusters_fixed.append(m)
        if m is None or not m.any():
            continue
        p_map[m] = np.minimum(p_map[m], p)

    return dict(
        T_obs=T_obs,
        clusters=clusters_fixed,
        cluster_pv=cluster_pv,
        p_map=p_map,
        t_thresh=t_thresh_used,
    )


def save_trial_counts(md_used: pd.DataFrame, sub: str, which: str):
    """
    is my labeling sane? artifact
    For stimtype sanity, we just count conditions.
    """
    if which == "stimtype":
        tab = (
            md_used
            .assign(condition=md_used[COL_COND].astype(str).str.lower())
            .groupby(["condition"])
            .size()
            .reset_index(name="n")
            .sort_values(["condition"])
        )
    else:
        tab = (
            md_used
            .assign(condition=md_used[COL_COND].astype(str).str.lower())
            .groupby(["condition", COL_LEVEL])
            .size()
            .reset_index(name="n")
            .sort_values(["condition", COL_LEVEL])
        )

    tab.to_csv(OUT_DIR / f"{sub}_passive_{which}_trial_counts.csv", index=False)


def alignment_sanity_check(md: pd.DataFrame, sub: str, log):
    """
    Quick checks to catch obvious misalignment after merge.
    Saves first 20 merged rows for visual inspection.
    Warns on suspicious temporal structure (e.g., condition almost monotonic).
    """
    preview_path = DEBUG_DIR / f"{sub}_merged_preview20.csv"
    md.head(20).to_csv(preview_path, index=False)

    if COL_COND not in md.columns:
        log(f"{sub}: alignment check skipped (no '{COL_COND}' in metadata).")
        return

    cond = md[COL_COND].astype(str).str.lower().to_numpy()
    keep = np.isin(cond, [COND_MONEY, COND_PAIN])
    cond = cond[keep]

    if len(cond) < 20:
        log(f"{sub}: alignment check skipped (too few labeled trials)")
        return

    # how often does condition change from one trial to the next?
    switches = np.mean(cond[1:] != cond[:-1])

    half = len(cond) // 2
    early_m = np.mean(cond[:half] == COND_MONEY)
    late_m = np.mean(cond[half:] == COND_MONEY)
    diff = abs(early_m - late_m)

    if switches < 0.05:
        log(f"{sub}: WARNING condition switch-rate is very low ({switches:.3f}). "
            f"Could be real block structure, but also can indicate misalignment. "
            f"Saved {preview_path.name}")

    if diff > 0.70:
        log(f"{sub}: WARNING strong early/late condition split (|early_m-late_m|={diff:.2f}). "
            f"Could be block design, but double-check merge. "
            f"Saved {preview_path.name}")

def save_group_summaries(
    *,
    tag: str,
    scores_all: np.ndarray,
    times: np.ndarray,
    included: list[str],
    stats_out: dict,
    out_dir: Path,
    alpha: float = 0.05,
    chance: float = CHANCE,   # <-- ADD THIS
):
    """
    Save summary stats that are useful for reports + later analyses.
      - {tag}_grand_mean_sem.csv
      - {tag}_sig_timepoints.csv
      - {tag}_cluster_table.csv
      - {tag}_summary.json
    """
    mean = scores_all.mean(axis=0)
    sem = scores_all.std(axis=0, ddof=1) / np.sqrt(scores_all.shape[0])

    n = scores_all.shape[0]
    tcrit = stats.t.ppf(0.975, df=n - 1)  # 95% CI (two-sided)
    ci95_low = mean - tcrit * sem
    ci95_high = mean + tcrit * sem

    p_map = np.asarray(stats_out["p_map"]).squeeze()
    if p_map.ndim != 1:
        p_map = p_map.ravel()
    if p_map.size == 1:
        p_map = np.full(times.shape, float(p_map))
    if p_map.size != times.size:
        raise RuntimeError(f"{tag}: p_map length mismatch: {p_map.size} vs times {times.size}")

    # ---- timecourse summary ----
    df_tc = pd.DataFrame({
        "time_s": times,
        "mean_score": mean,
        "sem_score": sem,
        "mean_above_chance": mean - chance,
        "T_obs": stats_out["T_obs"],
        "p_map": p_map,
        "sig": p_map < alpha,
        "ci95_low_score": ci95_low,
        "ci95_high_score": ci95_high,
    })
    
    df_tc.to_csv(out_dir / f"{tag}_grand_mean_sem.csv", index=False)

    # ---- peak stats ----
    peak_idx = int(np.argmax(mean))
    peak_score= float(mean[peak_idx])
    peak_time = float(times[peak_idx])

    peak_t_idx = int(np.argmax(stats_out["T_obs"]))
    peak_t = float(stats_out["T_obs"][peak_t_idx])
    peak_t_time = float(times[peak_t_idx])

    # ---- cluster table ----
    clusters = stats_out["clusters"]
    cluster_pv = stats_out["cluster_pv"]

    rows = []
    for i, (mask, p) in enumerate(zip(clusters, cluster_pv)):
        mask = _cluster_to_bool_mask(mask, scores_all.shape[1])
        if mask is None:
            continue
        if not mask.any():
            continue


        t_start = float(times[np.where(mask)[0][0]])
        t_end = float(times[np.where(mask)[0][-1]])
        dur_ms = (t_end - t_start) * 1000.0

        # effect metrics
        eff = (scores_all[:, mask] - chance)
        mean_eff = float(eff.mean())
        # sign is based on mean effect (positive=above chance)
        sign = "pos" if mean_eff >= 0 else "neg"

        # cluster mass: sum of T_obs within cluster (common reporting stat)
        cl_mass = float(stats_out["T_obs"][mask].sum())
        cl_max_t = float(stats_out["T_obs"][mask].max())
        cl_min_t = float(stats_out["T_obs"][mask].min())

        rows.append({
            "cluster": i,
            "p_value": float(p),
            "sign": sign,
            "t_start_s": t_start,
            "t_end_s": t_end,
            "duration_ms": dur_ms,
            "cluster_mass_sumT": cl_mass,
            "cluster_maxT": cl_max_t,
            "cluster_minT": cl_min_t,
            "cluster_mean_effect_score_minus_chance": mean_eff,
        })

    df_cl = pd.DataFrame(rows).sort_values("p_value") if len(rows) else pd.DataFrame(
        columns=[
            "cluster","p_value","sign","t_start_s","t_end_s","duration_ms",
            "cluster_mass_sumT","cluster_maxT","cluster_minT",
            "cluster_mean_effect_score_minus_chance"
        ]
    )
    df_cl.to_csv(out_dir / f"{tag}_cluster_table.csv", index=False)

    # ---- simple significant timepoints file (nice for later overlays) ----
    df_sig = df_tc.loc[:, ["time_s", "p_map", "sig"]].copy()
    df_sig.to_csv(out_dir / f"{tag}_sig_timepoints.csv", index=False)

    # ---- metadata JSON ----
    meta = {
        "tag": tag,
        "n_subjects": int(scores_all.shape[0]),
        "n_times": int(scores_all.shape[1]),
        "chance_score": float(chance),
        "alpha_cluster": float(alpha),
        "tfce_or_threshold": stats_out.get("t_thresh", None),
        "peak_score": peak_score,
        "peak_time_s": peak_time,
        "peak_T_obs": peak_t,
        "peak_T_time_s": peak_t_time,
        "n_clusters": int(len(cluster_pv)),
        "min_cluster_p": float(np.min(cluster_pv)) if len(cluster_pv) else 1.0,
        "included_subjects": included,
    }
    with open(out_dir / f"{tag}_summary.json", "w") as f:
        json.dump(meta, f, indent=2)

def append_subject_summary(
    records: list[dict],
    *,
    sub: str,
    which: str,
    shuffle: bool,
    md_used: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray | None,
    cv_used: str,
):
    rec = {
        "subject": sub,
        "which": which,
        "shuffle": bool(shuffle),
        "n_trials": int(len(y)),
        "n_low": int(np.sum(y == 0)) if which in ("money","pain") else None,
        "n_high": int(np.sum(y == 1)) if which in ("money","pain") else None,
        "n_classes": int(len(np.unique(y))),
        "has_blocks": bool(groups is not None),
        "n_blocks": int(len(np.unique(groups))) if groups is not None else None,
        "cv": cv_used,
    }
    records.append(rec)


def save_trial_counts_to(md_used: pd.DataFrame, sub: str, which: str, out_dir: Path):
    """Same as save_trial_counts but writes into out_dir."""
    if which == "stimtype":
        tab = (
            md_used.assign(condition=md_used[COL_COND].astype(str).str.lower())
            .groupby(["condition"])
            .size()
            .reset_index(name="n")
            .sort_values(["condition"])
        )
    else:
        tab = (
            md_used.assign(condition=md_used[COL_COND].astype(str).str.lower())
            .groupby(["condition", COL_LEVEL])
            .size()
            .reset_index(name="n")
            .sort_values(["condition", COL_LEVEL])
        )

    tab.to_csv(out_dir / f"{sub}_passive_{which}_trial_counts.csv", index=False)


def alignment_sanity_check_to(md: pd.DataFrame, sub: str, log, debug_dir: Path):
    """Same as alignment_sanity_check but writes into debug_dir."""
    preview_path = debug_dir / f"{sub}_merged_preview20.csv"
    md.head(20).to_csv(preview_path, index=False)

    if COL_COND not in md.columns:
        log(f"{sub}: alignment check skipped (no '{COL_COND}' in metadata).")
        return

    cond = md[COL_COND].astype(str).str.lower().to_numpy()
    keep = np.isin(cond, [COND_MONEY, COND_PAIN])
    cond = cond[keep]

    if len(cond) < 20:
        log(f"{sub}: alignment check skipped (too few labeled trials)")
        return

    switches = np.mean(cond[1:] != cond[:-1])
    half = len(cond) // 2
    early_m = np.mean(cond[:half] == COND_MONEY)
    late_m = np.mean(cond[half:] == COND_MONEY)
    diff = abs(early_m - late_m)

    if switches < 0.05:
        log(f"{sub}: WARNING low condition switch-rate ({switches:.3f}). Check {preview_path.name}")

    if diff > 0.70:
        log(f"{sub}: WARNING strong early/late split (|early_m-late_m|={diff:.2f}). Check {preview_path.name}")




def run(which: str, shuffle: bool = False):
    try:
        from tqdm.auto import tqdm
    except Exception:
        tqdm = None

    def log(msg: str):
        if tqdm is not None:
            try:
                tqdm.write(msg)
                return
            except Exception:
                pass
        print(msg, flush=True)

    subs = list_subjects(DERIV_DIR)
    scores_all_auc, scores_all_bacc, included, skipped = [], [], [], []

    times_ref = None

    # subject-level summary rows
    subj_records: list[dict] = []

    pbar = subs
    if tqdm is not None:
        pbar = tqdm(
            subs,
            desc=f"{which}{'_shuf' if shuffle else ''}",
            unit="sub",
            dynamic_ncols=True,
            leave=True,
        )

    for sub in pbar:
        if tqdm is not None:
            try:
                pbar.set_postfix_str(sub)
            except Exception:
                pass

        try:
            epo = load_passive_epochs(sub)
            beh = load_passive_beh(sub)
            epo = merge_beh_into_epochs(epo, beh, sub=sub)

            if epo.metadata is None:
                raise RuntimeError(f"{sub}: metadata is None after merge (should never happen).")

            if "badtrial" in epo.metadata.columns:
                n_bad = int(epo.metadata["badtrial"].fillna(0).astype(int).sum())
                if n_bad > 0:
                    epo = epo.copy()[epo.metadata["badtrial"].fillna(0).astype(int) == 0]
                    log(f"{sub}: dropped bad trials for MVPA: {n_bad} removed, {len(epo)} kept")
            else:
                log(f"{sub}: WARNING no 'badtrial' column found in epochs.metadata (not dropping trials)")

            md = epo.metadata.reset_index(drop=True)
            alignment_sanity_check(md, sub=sub, log=log)

            if md[COL_COND].isna().any() or md[COL_LEVEL].isna().any():
                n_nan_cond = int(md[COL_COND].isna().sum())
                n_nan_level = int(md[COL_LEVEL].isna().sum())
                md.head(50).to_csv(DEBUG_DIR / f"{sub}_md_aftermerge_head.csv", index=False)
                raise RuntimeError(
                    f"{sub}: NaNs in merged labels. "
                    f"{COL_COND} NaNs={n_nan_cond}, {COL_LEVEL} NaNs={n_nan_level}. "
                    f"Saved {sub}_md_aftermerge_head.csv"
                )

            cond_vals = set(md[COL_COND].astype(str).str.lower().unique())
            bad_cond = cond_vals - {COND_MONEY, COND_PAIN}
            if bad_cond:
                md.head(50).to_csv(DEBUG_DIR / f"{sub}_md_aftermerge_head.csv", index=False)
                raise RuntimeError(f"{sub}: unexpected condition labels found: {bad_cond}")

            log(f"{sub}: n_epochs={len(epo)} n_beh={len(beh)}")

            # ---------- resample ----------
            if RESAMPLE_SFREQ is not None:
                epo = epo.copy().resample(RESAMPLE_SFREQ, npad="auto")

            # ---------- select trials ----------
            if which in ("money", "pain"):
                X, y, times, md_used = select_trials_money_or_pain(epo, which=which)
            elif which == "stimtype":
                X, y, times, md_used = select_trials_stimtype(epo)
            else:
                raise ValueError("which must be 'money', 'pain', or 'stimtype'")

            save_trial_counts(md_used, sub=sub, which=which)

            # ---------- groups for block-wise CV ----------
            groups = None
            if KEY_BLOCK in md_used.columns:
                groups = md_used[KEY_BLOCK].to_numpy()

            # ---------- time-axis consistency ----------
            if times_ref is None:
                times_ref = times
            else:
                if len(times) != len(times_ref) or np.max(np.abs(times - times_ref)) > 1e-9:
                    raise RuntimeError("Time axis mismatch across subjects.")

            # ---------- decode ----------
            scores_auc, cv_used = subject_decode(X, y, shuffle=shuffle, groups=groups, metric="roc_auc")
            scores_bacc, _      = subject_decode(X, y, shuffle=shuffle, groups=groups, metric="balanced_accuracy")

            scores_all_auc.append(scores_auc)
            scores_all_bacc.append(scores_bacc)
            included.append(sub)


            append_subject_summary(
                subj_records,
                sub=sub,
                which=which,
                shuffle=shuffle,
                md_used=md_used,
                y=y,
                groups=groups,
                cv_used=cv_used,
            )

            if tqdm is not None:
                try:
                    pbar.set_postfix_str(f"{sub} | trials={len(y)} | {cv_used}")
                except Exception:
                    pass

            log(f"Included {sub} ({which}{'_shuf' if shuffle else ''}): trials={len(y)} | cv={cv_used}")

        except Exception as e:
            skipped.append((sub, str(e)))
            log(f"Skipped {sub} ({which}{'_shuf' if shuffle else ''}): {e}")

    # ---------- group stats ----------
    if len(scores_all_auc) < 8:
        raise RuntimeError(f"Too few subjects included for group stats ({which}): n={len(scores_all_auc)}")
    
    scores_all_auc = np.stack(scores_all_auc, axis=0)    # (n_subj, n_times)
    scores_all_bacc = np.stack(scores_all_bacc, axis=0)  # (n_subj, n_times)
    times = times_ref
    
    time_mask = (times >= TMIN_STAT) & (times <= TMAX_STAT)
    times_stat = times[time_mask]
    
    # ---- AUC stats + outputs (0-0.8s only) ----
    auc_stat = scores_all_auc[:, time_mask]
    stats_auc = group_cluster(auc_stat, times_stat)
    tag_auc = f"passive_{which}_auc" + ("_shuffle" if shuffle else "")
    
    np.savez(
        OUT_DIR / f"{tag_auc}_group_results.npz",
        scores_by_subj=scores_all_auc,
        times=times,
        included=np.array(included, dtype=object),
        skipped=np.array(skipped, dtype=object),
        T_obs=stats_auc["T_obs"],
        p_map=stats_auc["p_map"],
        cluster_pv=stats_auc["cluster_pv"],
        t_thresh=stats_auc["t_thresh"],
        chance=CHANCE,
        resample_sfreq=RESAMPLE_SFREQ if RESAMPLE_SFREQ is not None else -1,
        tmin_stat=TMIN_STAT,
        tmax_stat=TMAX_STAT,
    )
    
    pd.DataFrame(scores_all_auc, index=included, columns=np.round(times, 6)).to_csv(
        OUT_DIR / f"{tag_auc}_scores_by_subject.csv"
    )
    
    plot_group_metric(
        auc_stat,
        times_stat,
        stats_auc["p_map"],
        title=f"Passive {which} ({'SHUFFLED' if shuffle else 'REAL'}): AUC",
        out_png=OUT_DIR / f"{tag_auc}_group_plot.png",
        ylabel="Decoding (AUC)",
        chance=0.5,
        ylim=(0.35, 0.85),
    )
    
    save_group_summaries(
        tag=tag_auc,
        scores_all=auc_stat,
        times=times_stat,
        included=included,
        stats_out=stats_auc,
        out_dir=OUT_DIR,
        alpha=ALPHA_CLUSTER,
    )
    
    # ---- Balanced accuracy stats + outputs (0-0.8s only) ----
    bacc_stat = scores_all_bacc[:, time_mask]
    stats_bacc = group_cluster(bacc_stat, times_stat)
    tag_bacc = f"passive_{which}_bacc" + ("_shuffle" if shuffle else "")
    
    np.savez(
        OUT_DIR / f"{tag_bacc}_group_results.npz",
        scores_by_subj=scores_all_bacc,
        times=times,
        included=np.array(included, dtype=object),
        skipped=np.array(skipped, dtype=object),
        T_obs=stats_bacc["T_obs"],
        p_map=stats_bacc["p_map"],
        cluster_pv=stats_bacc["cluster_pv"],
        t_thresh=stats_bacc["t_thresh"],
        chance=CHANCE,
        resample_sfreq=RESAMPLE_SFREQ if RESAMPLE_SFREQ is not None else -1,
        tmin_stat=TMIN_STAT,
        tmax_stat=TMAX_STAT,
    )
    
    pd.DataFrame(scores_all_bacc, index=included, columns=np.round(times, 6)).to_csv(
        OUT_DIR / f"{tag_bacc}_scores_by_subject.csv"
    )
    
    plot_group_metric(
        bacc_stat,
        times_stat,
        stats_bacc["p_map"],
        title=f"Passive {which} ({'SHUFFLED' if shuffle else 'REAL'}): Balanced accuracy",
        out_png=OUT_DIR / f"{tag_bacc}_group_plot.png",
        ylabel="Decoding (balanced accuracy)",
        chance=0.5,
        ylim=(0.35, 0.85),
    )
    
    save_group_summaries(
        tag=tag_bacc,
        scores_all=bacc_stat,
        times=times_stat,
        included=included,
        stats_out=stats_bacc,
        out_dir=OUT_DIR,
        alpha=ALPHA_CLUSTER,
    )
    
    # subject summary (keep as you had)
    pd.DataFrame(subj_records).to_csv(OUT_DIR / f"passive_{which}" + ("_shuffle" if shuffle else "") + "_subject_summary.csv", index=False)
    
    min_p_auc = float(np.min(stats_auc["cluster_pv"])) if len(stats_auc["cluster_pv"]) else 1.0
    min_p_bacc = float(np.min(stats_bacc["cluster_pv"])) if len(stats_bacc["cluster_pv"]) else 1.0
    log(f"\nFinished {which}{'_shuffle' if shuffle else ''}: included n={len(included)}, min cluster p AUC={min_p_auc:.6f}, bAcc={min_p_bacc:.6f}")



def run_regression(which: str, shuffle: bool = False):
    """
    Regression decoding of level (20/40/60/80/100) using Ridge.
    Score = Pearson r (corr) between predicted and true level, per timepoint.
    """
    REG_CHANCE = 0.0
    OUT = OUT_DIR_REG
    DBG = DEBUG_DIR_REG

    try:
        from tqdm.auto import tqdm
    except Exception:
        tqdm = None

    def log(msg: str):
        if tqdm is not None:
            try:
                tqdm.write(msg)
                return
            except Exception:
                pass
        print(msg, flush=True)

    subs = list_subjects(DERIV_DIR)

    scores_all_r = []
    included, skipped = [], []
    times_ref = None

    subj_records: list[dict] = []

    pbar = subs
    if tqdm is not None:
        pbar = tqdm(
            subs,
            desc=f"REG {which}{'_shuf' if shuffle else ''}",
            unit="sub",
            dynamic_ncols=True,
            leave=True,
        )

    for sub in pbar:
        try:
            epo = load_passive_epochs(sub)
            beh = load_passive_beh(sub)
            epo = merge_beh_into_epochs(epo, beh, sub=sub)

            if epo.metadata is None:
                raise RuntimeError(f"{sub}: metadata is None after merge.")

            # drop bad trials if present
            if "badtrial" in epo.metadata.columns:
                n_bad = int(epo.metadata["badtrial"].fillna(0).astype(int).sum())
                if n_bad > 0:
                    epo = epo.copy()[epo.metadata["badtrial"].fillna(0).astype(int) == 0]
                    log(f"{sub}: dropped bad trials for REG: {n_bad} removed, {len(epo)} kept")

            md = epo.metadata.reset_index(drop=True)
            alignment_sanity_check_to(md, sub=sub, log=log, debug_dir=DBG)

            # ---------- resample ----------
            if RESAMPLE_SFREQ is not None:
                epo = epo.copy().resample(RESAMPLE_SFREQ, npad="auto")

            # ---------- select trials (REGRESSION keeps level 60 by default) ----------
            if which in ("money", "pain"):
                X, y, times, md_used = select_trials_money_or_pain_regression(
                    epo, which=which, drop_level60=False
                )
            else:
                raise ValueError("run_regression only supports 'money' or 'pain'")

            save_trial_counts_to(md_used, sub=sub, which=f"{which}_reg", out_dir=OUT)

            # groups for block-wise CV
            groups = None
            if KEY_BLOCK in md_used.columns:
                groups = md_used[KEY_BLOCK].to_numpy()

            # time axis consistency
            if times_ref is None:
                times_ref = times
            else:
                if len(times) != len(times_ref) or np.max(np.abs(times - times_ref)) > 1e-9:
                    raise RuntimeError("Time axis mismatch across subjects.")

            # decode
            scores_r, cv_used = subject_decode_regression(X, y, shuffle=shuffle, groups=groups)

            scores_all_r.append(scores_r)
            included.append(sub)

            subj_records.append({
                "subject": sub,
                "which": which,
                "shuffle": bool(shuffle),
                "n_trials": int(len(y)),
                "y_min": float(np.min(y)),
                "y_max": float(np.max(y)),
                "has_blocks": bool(groups is not None),
                "n_blocks": int(len(np.unique(groups))) if groups is not None else None,
                "cv": cv_used,
            })

            log(f"Included {sub} REG({which}{'_shuf' if shuffle else ''}): trials={len(y)} | cv={cv_used}")

        except Exception as e:
            skipped.append((sub, str(e)))
            log(f"Skipped {sub} REG({which}{'_shuf' if shuffle else ''}): {e}")

    if len(scores_all_r) < 8:
        raise RuntimeError(f"Too few subjects included for group stats (REG {which}): n={len(scores_all_r)}")

    scores_all_r = np.stack(scores_all_r, axis=0)
    times = times_ref

    # stats window
    time_mask = (times >= TMIN_STAT) & (times <= TMAX_STAT)
    times_stat = times[time_mask]
    r_stat = scores_all_r[:, time_mask]

    # cluster test against chance=0
    stats_r = group_cluster_metric(r_stat, times_stat, chance=REG_CHANCE, tail=0)

    tag = f"passive_{which}_ridgecorr" + ("_shuffle" if shuffle else "")

    np.savez(
        OUT / f"{tag}_group_results.npz",
        scores_by_subj=scores_all_r,
        times=times,
        included=np.array(included, dtype=object),
        skipped=np.array(skipped, dtype=object),
        T_obs=stats_r["T_obs"],
        p_map=stats_r["p_map"],
        cluster_pv=stats_r["cluster_pv"],
        t_thresh=stats_r["t_thresh"],
        chance=REG_CHANCE,
        resample_sfreq=RESAMPLE_SFREQ if RESAMPLE_SFREQ is not None else -1,
        tmin_stat=TMIN_STAT,
        tmax_stat=TMAX_STAT,
    )

    pd.DataFrame(scores_all_r, index=included, columns=np.round(times, 6)).to_csv(
        OUT / f"{tag}_scores_by_subject.csv"
    )

    plot_group_metric(
        r_stat,
        times_stat,
        stats_r["p_map"],
        title=f"Passive {which} ({'SHUFFLED' if shuffle else 'REAL'}): Ridge regression (corr r)",
        out_png=OUT / f"{tag}_group_plot.png",
        ylabel="Decoding (corr r)",
        chance=REG_CHANCE,
        ylim=(-0.10, 0.40),
    )

    save_group_summaries(
        tag=tag,
        scores_all=r_stat,
        times=times_stat,
        included=included,
        stats_out=stats_r,
        out_dir=OUT,
        alpha=ALPHA_CLUSTER,
        chance=REG_CHANCE,
    )

    pd.DataFrame(subj_records).to_csv(
        OUT / f"{tag}_subject_summary.csv", index=False
    )

    min_p = float(np.min(stats_r["cluster_pv"])) if len(stats_r["cluster_pv"]) else 1.0
    log(f"\nFinished REG {which}{'_shuffle' if shuffle else ''}: included n={len(included)}, min cluster p={min_p:.6f}")


def main():
    mne.set_log_level("WARNING")

    # -------------------------
    if RUN_BINARY:
        # temporarily redirect global OUT_DIR/DEBUG_DIR for binary outputs
        global OUT_DIR, DEBUG_DIR
        OUT_DIR, DEBUG_DIR = OUT_DIR_BIN, DEBUG_DIR_BIN

        run("money", shuffle=False)
        run("pain", shuffle=False)

        if RUN_STIMTYPE:
            run("stimtype", shuffle=False)

        if RUN_SHUFFLE:
            run("money", shuffle=True)
            run("pain", shuffle=True)
            if RUN_STIMTYPE:
                run("stimtype", shuffle=True)

    # -------------------------
    # REGRESSION (new)
    # -------------------------
    if RUN_REGRESSION:
        run_regression("money", shuffle=False)
        run_regression("pain", shuffle=False)

        if RUN_SHUFFLE:
            run_regression("money", shuffle=True)
            run_regression("pain", shuffle=True)

if __name__ == "__main__":
    main()
