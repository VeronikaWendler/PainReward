# -*- coding: utf-8 -*-
"""
Decision phase: Time-resolved decoding of money- and pain-cue levels (decision phase)

- Loads decision epochs from derivatives
- Loads decision beh.tsv from raw painrewardeegdata
- merge on trialsnum
- Extracts cue levels from decision metadata columns:
    - moneystim: m1..m5 -> 20/40/60/80/100
    - painstim : p1..p5 -> 20/40/60/80/100
- Runs time-resolved decoding for:
    BINARY: low (20/40) vs high (80/100), drops 60
      - money
      - pain
      - optional ctrlOther versions: residualize EEG by the OTHER cue level
    REGRESSION: ridge regression decoding (all levels; keeps 60)
      - money
      - pain
      - optional ctrlOther versions: residualize EEG by the OTHER cue level

- Group-level cluster permutation test over time on (score - chance)
    - Binary AUC/bAcc chance = 0.5
    - Regression corr-r chance = 0.0

Outputs:
  derivatives/statistics/mvpa_decision_step1_conserv/
    binary_lowhigh_auc_bacc/
    regression_ridgecorr/
    debug/

"""

from __future__ import annotations
import os
from pathlib import Path
import json
import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedKFold, GroupKFold

from mne.decoding import SlidingEstimator, cross_val_multiscore
from mne.stats import permutation_cluster_1samp_test
from scipy import stats

from tqdm.auto import tqdm


# -----------------------------
# Paths

DATA_DIR_STR = os.getenv("DATA_DIR", "").strip()
OUT_DIR_STR = os.getenv("OUT_DIR", "").strip()

if DATA_DIR_STR == "":
    raise RuntimeError("DATA_DIR env var not set")

RAW_DIR = Path(DATA_DIR_STR).expanduser()
DERIV_DIR = RAW_DIR / "derivatives"

# Output folder
if OUT_DIR_STR != "":
    OUT_BASE = Path(OUT_DIR_STR).expanduser()
    OUT_DIR = OUT_BASE / "statistics" / "mvpa_decision_step1_conserv"
else:
    OUT_DIR = DERIV_DIR / "statistics" / "mvpa_decision_step1_conserv"

OUT_DIR.mkdir(parents=True, exist_ok=True)
DEBUG_DIR = OUT_DIR / "debug"
DEBUG_DIR.mkdir(parents=True, exist_ok=True)

# epochs/beh locations relative to subject folder
EPO_DIR = Path("eeg") / "erps"
EPO_SUFFIX = "_decision_cues_singletrials-epo.fif"
BEH_SUFFIX = "_task-decision_beh.tsv"


# -----------------------------
# Decoding params
# -----------------------------
RANDOM_STATE = 23
N_SPLITS = 5
N_PERM = 5000
ALPHA_CLUSTER = 0.05

KEY_TRIALNUM = "trialsnum"
KEY_BLOCK = "blocks.thisN"
KEY_TRIAL = "trials.thisN"

# downsample for speed
RESAMPLE_SFREQ = 256  # set None to keep original

# decision metadata columns
DEC_MONEY_COL = "moneystim"  # e.g., m1..m5
DEC_PAIN_COL = "painstim"    # e.g., p1..p5

# stats window
TMIN_STAT = 0.0
TMAX_STAT = 0.8

# Run flags
RUN_BINARY = True          # AUC + balanced accuracy (low vs high; drops 60)
RUN_REGRESSION = True      # ridge regression decoding (all levels; keeps 60)
RUN_SHUFFLE = True
RUN_CONTROL_BY_OTHER = True  # residualize EEG by other cue level

# Binary settings
CHANCE_BIN = 0.5
BIN_KEEP_LEVELS = np.array([20, 40, 80, 100], dtype=int)

# Regression settings
CHANCE_REG = 0.0
LEVELS_ALL = np.array([20, 40, 60, 80, 100], dtype=int)

LEVEL_CODE_TO_LEVEL = {1: 20, 2: 40, 3: 60, 4: 80, 5: 100}


# -----------------------------
# output folders
# -----------------------------
OUT_DIR_BIN = OUT_DIR / "binary_lowhigh_auc_bacc"
OUT_DIR_REG = OUT_DIR / "regression_ridgecorr"

for _d in [OUT_DIR_BIN, OUT_DIR_REG]:
    _d.mkdir(parents=True, exist_ok=True)

DEBUG_DIR_BIN = OUT_DIR_BIN / "debug"
DEBUG_DIR_REG = OUT_DIR_REG / "debug"
for _d in [DEBUG_DIR_BIN, DEBUG_DIR_REG]:
    _d.mkdir(parents=True, exist_ok=True)


# =====================================================================
# IO
# =====================================================================

def list_subjects(deriv_dir: Path) -> list[str]:
    return sorted([p.name for p in deriv_dir.iterdir()
                   if p.is_dir() and p.name.startswith("sub-")])


def load_decision_epochs(sub: str) -> mne.Epochs:
    epo_path = DERIV_DIR / sub / EPO_DIR / f"{sub}{EPO_SUFFIX}"
    if not epo_path.exists():
        raise FileNotFoundError(f"Missing decision epochs for {sub}: {epo_path}")
    return mne.read_epochs(epo_path, preload=True, verbose="ERROR")


def load_decision_beh(sub: str) -> pd.DataFrame:
    beh_path = RAW_DIR / sub / "eeg" / f"{sub}{BEH_SUFFIX}"
    if not beh_path.exists():
        raise FileNotFoundError(f"Missing decision beh.tsv for {sub}: {beh_path}")

    beh = pd.read_csv(beh_path, sep="\t")

    if "fixcross.started" in beh.columns:
        beh = beh[~beh["fixcross.started"].isna()].copy()

    beh = beh.reset_index(drop=True)
    beh[KEY_TRIALNUM] = np.arange(1, len(beh) + 1)
    return beh


def _coerce_int_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("Int64")


def merge_beh_into_epochs_decision(epo: mne.Epochs, beh: pd.DataFrame, sub: str, debug_dir: Path) -> mne.Epochs:
    """
    Decision merge: if epochs.metadata already contains moneystim/painstim, skip.
    Otherwise merge using trialsnum, else block+trial, else order if lengths match.
    """
    if epo.metadata is None:
        raise ValueError(f"{sub}: epochs has no metadata at all; cannot merge beh.")

    md = epo.metadata.reset_index(drop=True).copy()

    if (DEC_MONEY_COL in md.columns) and (DEC_PAIN_COL in md.columns):
        tqdm.write(f"{sub}: epochs.metadata already contains {DEC_MONEY_COL}+{DEC_PAIN_COL} (no merge needed)")
        return epo

    # ---- trialsnum merge ----
    if (KEY_TRIALNUM in md.columns) and (KEY_TRIALNUM in beh.columns):
        merged = md.merge(
            beh,
            on=KEY_TRIALNUM,
            how="left",
            validate="1:1",
        )
        if merged[DEC_MONEY_COL].isna().any() or merged[DEC_PAIN_COL].isna().any():
            md.head(50).to_csv(debug_dir / f"{sub}_epo_md_head.csv", index=False)
            beh.head(50).to_csv(debug_dir / f"{sub}_beh_head.csv", index=False)
            raise ValueError(f"{sub}: trialsnum-merge produced unlabeled epochs (missing moneystim/painstim).")
        epo.metadata = merged
        tqdm.write(f"{sub}: merged beh into epochs using '{KEY_TRIALNUM}'")
        return epo

    # ---- key merge ----
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
        )

        if merged[DEC_MONEY_COL].isna().any() or merged[DEC_PAIN_COL].isna().any():
            md.head(50).to_csv(debug_dir / f"{sub}_epo_md_head.csv", index=False)
            beh.head(50).to_csv(debug_dir / f"{sub}_beh_head.csv", index=False)
            n_bad = int(merged[DEC_MONEY_COL].isna().sum() + merged[DEC_PAIN_COL].isna().sum())
            raise ValueError(
                f"{sub}: key-merge produced unlabeled epochs (missing moneystim/painstim). "
                f"n_missing_total={n_bad}."
            )

        epo.metadata = merged
        tqdm.write(f"{sub}: merged beh into epochs using KEYS ({KEY_BLOCK}, {KEY_TRIAL})")
        return epo

    # ---- final fallback: order merge ----
    if len(md) == len(beh):
        merged = md.copy()
        for c in beh.columns:
            if c not in merged.columns:
                merged[c] = beh[c].to_numpy()

        if merged[DEC_MONEY_COL].isna().any() or merged[DEC_PAIN_COL].isna().any():
            md.head(50).to_csv(debug_dir / f"{sub}_epo_md_head.csv", index=False)
            beh.head(50).to_csv(debug_dir / f"{sub}_beh_head.csv", index=False)
            raise ValueError(f"{sub}: order-merge produced missing moneystim/painstim.")
        epo.metadata = merged
        tqdm.write(f"{sub}: merged beh into epochs by ORDER (len match: {len(md)})")
        return epo

    md.head(50).to_csv(debug_dir / f"{sub}_epo_md_head.csv", index=False)
    beh.head(50).to_csv(debug_dir / f"{sub}_beh_head.csv", index=False)

    raise ValueError(
        f"{sub}: cannot merge beh into epochs.\n"
        f"- epochs n={len(md)}; beh n={len(beh)} (not equal, so order merge not possible)\n"
        f"- keys '{KEY_TRIALNUM}' or '{KEY_BLOCK}/{KEY_TRIAL}' not present in BOTH epochs.metadata and beh.tsv\n"
        f"See debug CSVs in {debug_dir}"
    )


# =====================================================================
# Label parsing + selection
# =====================================================================

def parse_stim_code_to_level(series: pd.Series, prefix: str) -> np.ndarray:
    """
    Parse e.g. 'm1'..'m5' or 'p1'..'p5' to 20/40/60/80/100
    """
    s = series.astype(str).str.strip().str.lower()
    codes = s.str.extract(rf"^{prefix}\s*([1-5])$", expand=False)
    if codes.isna().any():
        bad = s[codes.isna()].unique()[:10]
        raise ValueError(f"Unexpected '{prefix}' stim codes (examples): {bad}")
    codes_int = codes.astype(int).to_numpy()
    return np.array([LEVEL_CODE_TO_LEVEL[int(c)] for c in codes_int], dtype=int)


def make_binary_labels(level: np.ndarray) -> np.ndarray:
    """
    level values expected: 20/40/80/100
    returns y: 0=low (20,40), 1=high (80,100)
    """
    level = np.asarray(level, dtype=float)
    y = np.full(len(level), -1, dtype=int)
    y[np.isin(level, [20, 40])] = 0
    y[np.isin(level, [80, 100])] = 1
    return y


def decision_alignment_sanity_check(md: pd.DataFrame, sub: str, log, debug_dir: Path):
    """
    Very similar spirit to passive alignment check:
    - save first 20 rows for inspection
    - ensure labels parse and look plausible
    """
    preview_path = debug_dir / f"{sub}_merged_preview20.csv"
    md.head(20).to_csv(preview_path, index=False)

    if (DEC_MONEY_COL not in md.columns) or (DEC_PAIN_COL not in md.columns):
        log(f"{sub}: alignment check: missing {DEC_MONEY_COL}/{DEC_PAIN_COL} (merge likely failed). Saved {preview_path.name}")
        return

    try:
        m = parse_stim_code_to_level(md[DEC_MONEY_COL], "m")
        p = parse_stim_code_to_level(md[DEC_PAIN_COL], "p")
    except Exception as e:
        log(f"{sub}: alignment check: stim parsing failed: {e}. Saved {preview_path.name}")
        return

    if len(m) >= 20:
        r = np.corrcoef(m, p)[0, 1]
        if np.isfinite(r) and abs(r) > 0.95:
            log(f"{sub}: WARNING |corr(money,pain)|={abs(r):.2f} extremely high; maybe design, but check merge. "
                f"Saved {preview_path.name}")


def save_decision_trial_counts(out_dir: Path, sub: str, tag: str, which: str, levels: np.ndarray, other_levels: np.ndarray):
    """
    Analog of save_trial_counts: saves decoded level distribution (and other-level distribution).
    """
    df = pd.DataFrame({
        "level_decoded": levels.astype(int),
        "level_other": other_levels.astype(int),
    })
    tab1 = df.groupby("level_decoded").size().reset_index(name="n").sort_values("level_decoded")
    tab2 = df.groupby("level_other").size().reset_index(name="n").sort_values("level_other")

    tab1.to_csv(out_dir / f"{sub}_decision_{tag}_{which}_counts_decoded.csv", index=False)
    tab2.to_csv(out_dir / f"{sub}_decision_{tag}_{which}_counts_other.csv", index=False)


# =====================================================================
# Optional residualization (control-by-other)
# =====================================================================

def residualize_X_by_nuisance(X: np.ndarray, nuisance_levels: np.ndarray, model: str = "quad") -> np.ndarray:
    """
    Residualize EEG features with respect to nuisance_levels across trials.
    Model:
      - "lin": intercept + nuis
      - "quad": intercept + nuis + nuis^2
    """
    X = np.asarray(X, dtype=float)
    nuis = np.asarray(nuisance_levels, dtype=float).ravel()

    if X.shape[0] != nuis.shape[0]:
        raise ValueError(f"Residualization mismatch: X trials={X.shape[0]} nuis={nuis.shape[0]}")

    if model == "lin":
        A = np.column_stack([np.ones_like(nuis), nuis])
    elif model == "quad":
        A = np.column_stack([np.ones_like(nuis), nuis, nuis**2])
    else:
        raise ValueError("model must be 'lin' or 'quad'")

    n_trials, n_chans, n_times = X.shape
    Y = X.reshape(n_trials, -1)

    beta, *_ = np.linalg.lstsq(A, Y, rcond=None)
    Y_hat = A @ beta
    Y_resid = Y - Y_hat

    return Y_resid.reshape(n_trials, n_chans, n_times)


def select_trials_decision_binary(epo: mne.Epochs, which: str, control_resid_by_other: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Decision BINARY selection:
      - keep only 20/40/80/100 (drop 60)
      - y = 0 low, 1 high
      - optionally residualize EEG by other cue level
    Returns X, y, times, md_used, levels_used, other_levels_used
    """
    if epo.metadata is None:
        raise ValueError("Epochs has no metadata.")

    md = epo.metadata.reset_index(drop=True)

    for col in [DEC_MONEY_COL, DEC_PAIN_COL]:
        if col not in md.columns:
            raise ValueError(f"Missing metadata column '{col}'. Have: {md.columns.tolist()}")

    money_levels = parse_stim_code_to_level(md[DEC_MONEY_COL], "m")
    pain_levels = parse_stim_code_to_level(md[DEC_PAIN_COL], "p")

    levels = money_levels if which == "money" else pain_levels
    other = pain_levels if which == "money" else money_levels

    keep = np.isin(levels, BIN_KEEP_LEVELS)
    epo_f = epo.copy()[keep]
    md_f = epo_f.metadata.reset_index(drop=True)

    levels_f = levels[keep]
    other_f = other[keep]

    if len(epo_f) < 10:
        raise ValueError(f"Too few trials after filtering for decision {which} binary. n={len(epo_f)}")

    X = epo_f.get_data()
    times = epo_f.times.copy()

    if control_resid_by_other:
        X = residualize_X_by_nuisance(X, nuisance_levels=other_f, model="quad")

    y = make_binary_labels(levels_f.astype(float))
    if np.any(y < 0):
        raise ValueError(f"Unlabeled trials exist after filtering. Levels seen: {np.unique(levels_f)}")

    return X, y, times, md_f, levels_f, other_f


def select_trials_decision_regression(epo: mne.Epochs, which: str, control_resid_by_other: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Decision REGRESSION selection:
      - keep all 20/40/60/80/100
      - y = numeric level
      - optionally residualize EEG by other cue level
    Returns X, y, times, md_used, levels_used, other_levels_used
    """
    if epo.metadata is None:
        raise ValueError("Epochs has no metadata.")

    md = epo.metadata.reset_index(drop=True)

    for col in [DEC_MONEY_COL, DEC_PAIN_COL]:
        if col not in md.columns:
            raise ValueError(f"Missing metadata column '{col}'. Have: {md.columns.tolist()}")

    money_levels = parse_stim_code_to_level(md[DEC_MONEY_COL], "m")
    pain_levels = parse_stim_code_to_level(md[DEC_PAIN_COL], "p")

    levels = money_levels if which == "money" else pain_levels
    other = pain_levels if which == "money" else money_levels

    keep = np.isin(levels, LEVELS_ALL)
    epo_f = epo.copy()[keep]
    md_f = epo_f.metadata.reset_index(drop=True)

    levels_f = levels[keep]
    other_f = other[keep]

    if len(epo_f) < 10:
        raise ValueError(f"Too few trials after filtering for decision {which} regression. n={len(epo_f)}")

    X = epo_f.get_data()
    times = epo_f.times.copy()

    if control_resid_by_other:
        X = residualize_X_by_nuisance(X, nuisance_levels=other_f, model="quad")

    y = levels_f.astype(float)
    return X, y, times, md_f, levels_f, other_f


# =====================================================================
# Decoders
# =====================================================================

def _corr_scorer(estimator, X, y_true) -> float:
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


def subject_decode_binary(X: np.ndarray, y: np.ndarray, shuffle: bool, groups: np.ndarray | None, metric: str):
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
        time_decod, X, y_use,
        cv=cv,
        groups=groups if isinstance(cv, GroupKFold) else None,
        n_jobs=1
    )
    cv_used = "GroupKFold" if isinstance(cv, GroupKFold) else "StratifiedKFold"
    return scores.mean(axis=0), cv_used


def subject_decode_regression(X: np.ndarray, y: np.ndarray, shuffle: bool, groups: np.ndarray | None):
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
        from sklearn.model_selection import KFold
        cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    scores = cross_val_multiscore(
        time_decod, X, y_use,
        cv=cv,
        groups=groups if isinstance(cv, GroupKFold) else None,
        n_jobs=1
    )
    cv_used = "GroupKFold" if isinstance(cv, GroupKFold) else "KFold"
    return scores.mean(axis=0), cv_used


# =====================================================================
# Stats + plotting
# =====================================================================

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


def group_cluster_metric(scores_by_subj: np.ndarray, times: np.ndarray, *, chance: float, tail: int):
    """
    Passive-style TFCE->None->param threshold fallback, with p_map construction.
    """
    X = scores_by_subj - chance

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
        try:
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
        except Exception:
            p_form = 0.01
            t_thresh = stats.t.ppf(1 - p_form / 2, df=X.shape[0] - 1)
            T_obs, clusters, cluster_pv, H0 = permutation_cluster_1samp_test(
                X,
                n_permutations=N_PERM,
                threshold=t_thresh,
                tail=tail,
                out_type="mask",
                n_jobs=1,
                seed=RANDOM_STATE,
                buffer_size=None,
            )
            t_thresh_used = float(t_thresh)

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


def plot_group_metric(scores_by_subj, times, p_map, title, out_png, ylabel, chance, ylim):
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


def save_group_summaries(
    *,
    tag: str,
    scores_all: np.ndarray,
    times: np.ndarray,
    included: list[str],
    stats_out: dict,
    out_dir: Path,
    alpha: float,
    chance: float,
):
    mean = scores_all.mean(axis=0)
    sem = scores_all.std(axis=0, ddof=1) / np.sqrt(scores_all.shape[0])

    n = scores_all.shape[0]
    tcrit = stats.t.ppf(0.975, df=n - 1)
    ci95_low = mean - tcrit * sem
    ci95_high = mean + tcrit * sem

    p_map = np.asarray(stats_out["p_map"]).squeeze()
    if p_map.ndim != 1:
        p_map = p_map.ravel()
    if p_map.size == 1:
        p_map = np.full(times.shape, float(p_map))
    if p_map.size != times.size:
        raise RuntimeError(f"{tag}: p_map length mismatch: {p_map.size} vs times {times.size}")

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

    peak_idx = int(np.argmax(mean))
    peak_score = float(mean[peak_idx])
    peak_time = float(times[peak_idx])

    peak_t_idx = int(np.argmax(stats_out["T_obs"]))
    peak_t = float(stats_out["T_obs"][peak_t_idx])
    peak_t_time = float(times[peak_t_idx])

    clusters = stats_out["clusters"]
    cluster_pv = stats_out["cluster_pv"]

    rows = []
    for i, (mask, p) in enumerate(zip(clusters, cluster_pv)):
        mask = _cluster_to_bool_mask(mask, scores_all.shape[1])
        if mask is None or not mask.any():
            continue

        t_start = float(times[np.where(mask)[0][0]])
        t_end = float(times[np.where(mask)[0][-1]])
        dur_ms = (t_end - t_start) * 1000.0

        eff = (scores_all[:, mask] - chance)
        mean_eff = float(eff.mean())
        sign = "pos" if mean_eff >= 0 else "neg"

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
            "cluster", "p_value", "sign", "t_start_s", "t_end_s", "duration_ms",
            "cluster_mass_sumT", "cluster_maxT", "cluster_minT",
            "cluster_mean_effect_score_minus_chance"
        ]
    )
    df_cl.to_csv(out_dir / f"{tag}_cluster_table.csv", index=False)

    df_sig = df_tc.loc[:, ["time_s", "p_map", "sig"]].copy()
    df_sig.to_csv(out_dir / f"{tag}_sig_timepoints.csv", index=False)

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


def append_subject_summary(records: list[dict], *, sub: str, which: str, shuffle: bool, tag: str,
                           n_trials: int, groups: np.ndarray | None, cv_used: str,
                           extra: dict | None = None):
    rec = {
        "subject": sub,
        "which": which,
        "tag": tag,
        "shuffle": bool(shuffle),
        "n_trials": int(n_trials),
        "has_blocks": bool(groups is not None),
        "n_blocks": int(len(np.unique(groups))) if groups is not None else None,
        "cv": cv_used,
    }
    if extra:
        rec.update(extra)
    records.append(rec)


# =====================================================================
# run 
# =====================================================================

def run_binary(which: str, shuffle: bool, tag: str, control_by_other: bool):
    """
    Passive-like run() but for decision binary, producing AUC and bAcc group outputs.
    tag: "raw" or "ctrlOther"
    """
    OUT = OUT_DIR_BIN
    DBG = DEBUG_DIR_BIN

    def log(msg: str):
        try:
            tqdm.write(msg)
        except Exception:
            print(msg, flush=True)

    subs = list_subjects(DERIV_DIR)
    scores_all_auc, scores_all_bacc, included, skipped = [], [], [], []
    times_ref = None

    subj_records: list[dict] = []

    pbar = tqdm(
        subs,
        desc=f"BIN {which} {tag}{'_shuf' if shuffle else ''}",
        unit="sub",
        dynamic_ncols=True,
        leave=True,
    )

    for sub in pbar:
        try:
            pbar.set_postfix_str(sub)

            epo = load_decision_epochs(sub)
            beh = load_decision_beh(sub)
            epo = merge_beh_into_epochs_decision(epo, beh, sub=sub, debug_dir=DBG)

            if epo.metadata is None:
                raise RuntimeError(f"{sub}: metadata is None after merge (should never happen).")

            # drop bad trials if present
            if "badtrial" in epo.metadata.columns:
                n_bad = int(epo.metadata["badtrial"].fillna(0).astype(int).sum())
                if n_bad > 0:
                    epo = epo.copy()[epo.metadata["badtrial"].fillna(0).astype(int) == 0]
                    log(f"{sub}: dropped bad trials for BIN: {n_bad} removed, {len(epo)} kept")
            else:
                log(f"{sub}: WARNING no 'badtrial' column found in epochs.metadata (not dropping trials)")

            md = epo.metadata.reset_index(drop=True)
            decision_alignment_sanity_check(md, sub=sub, log=log, debug_dir=DBG)

            if md[DEC_MONEY_COL].isna().any() or md[DEC_PAIN_COL].isna().any():
                md.head(50).to_csv(DBG / f"{sub}_md_aftermerge_head.csv", index=False)
                raise RuntimeError(f"{sub}: NaNs in merged decision labels. Saved {sub}_md_aftermerge_head.csv")

            log(f"{sub}: n_epochs={len(epo)} n_beh={len(beh)}")

            # resample
            if RESAMPLE_SFREQ is not None:
                epo = epo.copy().resample(RESAMPLE_SFREQ, npad="auto")

            # select
            X, y, times, md_used, levels_f, other_f = select_trials_decision_binary(
                epo, which=which, control_resid_by_other=control_by_other
            )

            save_decision_trial_counts(OUT, sub=sub, tag=tag, which=which, levels=levels_f, other_levels=other_f)

            # groups for block-wise CV
            groups = None
            if KEY_BLOCK in md_used.columns:
                groups = md_used[KEY_BLOCK].to_numpy()

            # time-axis consistency
            if times_ref is None:
                times_ref = times
            else:
                if len(times) != len(times_ref) or np.max(np.abs(times - times_ref)) > 1e-9:
                    raise RuntimeError("Time axis mismatch across subjects.")

            # decode
            scores_auc, cv_used = subject_decode_binary(X, y, shuffle=shuffle, groups=groups, metric="roc_auc")
            scores_bacc, _ = subject_decode_binary(X, y, shuffle=shuffle, groups=groups, metric="balanced_accuracy")

            scores_all_auc.append(scores_auc)
            scores_all_bacc.append(scores_bacc)
            included.append(sub)

            append_subject_summary(
                subj_records,
                sub=sub, which=which, shuffle=shuffle, tag=tag,
                n_trials=len(y), groups=groups, cv_used=cv_used,
                extra=dict(n_low=int(np.sum(y == 0)), n_high=int(np.sum(y == 1)), control_by_other=bool(control_by_other))
            )

            pbar.set_postfix_str(f"{sub} | trials={len(y)} | {cv_used}")
            log(f"Included {sub} BIN({which},{tag}{'_shuf' if shuffle else ''}): trials={len(y)} | cv={cv_used}")

        except Exception as e:
            skipped.append((sub, str(e)))
            log(f"Skipped {sub} BIN({which},{tag}{'_shuf' if shuffle else ''}): {e}")

    # group stats
    if len(scores_all_auc) < 8:
        raise RuntimeError(f"Too few subjects included for group stats BIN({which},{tag}): n={len(scores_all_auc)}")

    scores_all_auc = np.stack(scores_all_auc, axis=0)
    scores_all_bacc = np.stack(scores_all_bacc, axis=0)
    times = times_ref

    time_mask = (times >= TMIN_STAT) & (times <= TMAX_STAT)
    times_stat = times[time_mask]

    # AUC
    auc_stat = scores_all_auc[:, time_mask]
    stats_auc = group_cluster_metric(auc_stat, times_stat, chance=CHANCE_BIN, tail=1)
    tag_auc = f"decision_{tag}_{which}_auc" + ("_shuffle" if shuffle else "")

    np.savez(
        OUT / f"{tag_auc}_group_results.npz",
        scores_by_subj=scores_all_auc,
        times=times,
        included=np.array(included, dtype=object),
        skipped=np.array(skipped, dtype=object),
        T_obs=stats_auc["T_obs"],
        p_map=stats_auc["p_map"],
        cluster_pv=stats_auc["cluster_pv"],
        t_thresh=stats_auc["t_thresh"],
        chance=CHANCE_BIN,
        resample_sfreq=RESAMPLE_SFREQ if RESAMPLE_SFREQ is not None else -1,
        tmin_stat=TMIN_STAT,
        tmax_stat=TMAX_STAT,
    )

    pd.DataFrame(scores_all_auc, index=included, columns=np.round(times, 6)).to_csv(
        OUT / f"{tag_auc}_scores_by_subject.csv"
    )

    plot_group_metric(
        auc_stat,
        times_stat,
        stats_auc["p_map"],
        title=f"Decision {which} ({tag.upper()} | {'SHUFFLED' if shuffle else 'REAL'}): AUC",
        out_png=OUT / f"{tag_auc}_group_plot.png",
        ylabel="Decoding (AUC)",
        chance=CHANCE_BIN,
        ylim=(0.35, 0.85),
    )

    save_group_summaries(
        tag=tag_auc,
        scores_all=auc_stat,
        times=times_stat,
        included=included,
        stats_out=stats_auc,
        out_dir=OUT,
        alpha=ALPHA_CLUSTER,
        chance=CHANCE_BIN,
    )

    # bAcc
    bacc_stat = scores_all_bacc[:, time_mask]
    stats_bacc = group_cluster_metric(bacc_stat, times_stat, chance=CHANCE_BIN, tail=1)
    tag_bacc = f"decision_{tag}_{which}_bacc" + ("_shuffle" if shuffle else "")

    np.savez(
        OUT / f"{tag_bacc}_group_results.npz",
        scores_by_subj=scores_all_bacc,
        times=times,
        included=np.array(included, dtype=object),
        skipped=np.array(skipped, dtype=object),
        T_obs=stats_bacc["T_obs"],
        p_map=stats_bacc["p_map"],
        cluster_pv=stats_bacc["cluster_pv"],
        t_thresh=stats_bacc["t_thresh"],
        chance=CHANCE_BIN,
        resample_sfreq=RESAMPLE_SFREQ if RESAMPLE_SFREQ is not None else -1,
        tmin_stat=TMIN_STAT,
        tmax_stat=TMAX_STAT,
    )

    pd.DataFrame(scores_all_bacc, index=included, columns=np.round(times, 6)).to_csv(
        OUT / f"{tag_bacc}_scores_by_subject.csv"
    )

    plot_group_metric(
        bacc_stat,
        times_stat,
        stats_bacc["p_map"],
        title=f"Decision {which} ({tag.upper()} | {'SHUFFLED' if shuffle else 'REAL'}): Balanced accuracy",
        out_png=OUT / f"{tag_bacc}_group_plot.png",
        ylabel="Decoding (balanced accuracy)",
        chance=CHANCE_BIN,
        ylim=(0.35, 0.85),
    )

    save_group_summaries(
        tag=tag_bacc,
        scores_all=bacc_stat,
        times=times_stat,
        included=included,
        stats_out=stats_bacc,
        out_dir=OUT,
        alpha=ALPHA_CLUSTER,
        chance=CHANCE_BIN,
    )

    # subject summary
    suffix = "_shuffle" if shuffle else ""
    fname = f"decision_{tag}_{which}{suffix}_subject_summary.csv"
    pd.DataFrame(subj_records).to_csv(OUT / fname, index=False)

    min_p_auc = float(np.min(stats_auc["cluster_pv"])) if len(stats_auc["cluster_pv"]) else 1.0
    min_p_bacc = float(np.min(stats_bacc["cluster_pv"])) if len(stats_bacc["cluster_pv"]) else 1.0
    log(f"\nFinished BIN {which} ({tag}{'_shuffle' if shuffle else ''}): included n={len(included)}, "
        f"min cluster p AUC={min_p_auc:.6f}, bAcc={min_p_bacc:.6f}")


def run_regression(which: str, shuffle: bool, tag: str, control_by_other: bool):
    """
    run_regression() but for decision regression ridgecorr.
    """
    OUT = OUT_DIR_REG
    DBG = DEBUG_DIR_REG

    def log(msg: str):
        try:
            tqdm.write(msg)
        except Exception:
            print(msg, flush=True)

    subs = list_subjects(DERIV_DIR)
    scores_all_r = []
    included, skipped = [], []
    times_ref = None

    subj_records: list[dict] = []

    pbar = tqdm(
        subs,
        desc=f"REG {which} {tag}{'_shuf' if shuffle else ''}",
        unit="sub",
        dynamic_ncols=True,
        leave=True,
    )

    for sub in pbar:
        try:
            pbar.set_postfix_str(sub)

            epo = load_decision_epochs(sub)
            beh = load_decision_beh(sub)
            epo = merge_beh_into_epochs_decision(epo, beh, sub=sub, debug_dir=DBG)

            if epo.metadata is None:
                raise RuntimeError(f"{sub}: metadata is None after merge.")

            if "badtrial" in epo.metadata.columns:
                n_bad = int(epo.metadata["badtrial"].fillna(0).astype(int).sum())
                if n_bad > 0:
                    epo = epo.copy()[epo.metadata["badtrial"].fillna(0).astype(int) == 0]
                    log(f"{sub}: dropped bad trials for REG: {n_bad} removed, {len(epo)} kept")
            else:
                log(f"{sub}: WARNING no 'badtrial' column found in epochs.metadata (not dropping trials)")

            md = epo.metadata.reset_index(drop=True)
            decision_alignment_sanity_check(md, sub=sub, log=log, debug_dir=DBG)

            if md[DEC_MONEY_COL].isna().any() or md[DEC_PAIN_COL].isna().any():
                md.head(50).to_csv(DBG / f"{sub}_md_aftermerge_head.csv", index=False)
                raise RuntimeError(f"{sub}: NaNs in merged decision labels. Saved {sub}_md_aftermerge_head.csv")

            # resample
            if RESAMPLE_SFREQ is not None:
                epo = epo.copy().resample(RESAMPLE_SFREQ, npad="auto")

            # select
            X, y, times, md_used, levels_f, other_f = select_trials_decision_regression(
                epo, which=which, control_resid_by_other=control_by_other
            )

            save_decision_trial_counts(OUT, sub=sub, tag=tag, which=which, levels=levels_f, other_levels=other_f)

            groups = None
            if KEY_BLOCK in md_used.columns:
                groups = md_used[KEY_BLOCK].to_numpy()

            # time axis
            if times_ref is None:
                times_ref = times
            else:
                if len(times) != len(times_ref) or np.max(np.abs(times - times_ref)) > 1e-9:
                    raise RuntimeError("Time axis mismatch across subjects.")

            # decode
            scores_r, cv_used = subject_decode_regression(X, y, shuffle=shuffle, groups=groups)

            scores_all_r.append(scores_r)
            included.append(sub)

            append_subject_summary(
                subj_records,
                sub=sub, which=which, shuffle=shuffle, tag=tag,
                n_trials=len(y), groups=groups, cv_used=cv_used,
                extra=dict(y_min=float(np.min(y)), y_max=float(np.max(y)), control_by_other=bool(control_by_other))
            )

            pbar.set_postfix_str(f"{sub} | trials={len(y)} | {cv_used}")
            log(f"Included {sub} REG({which},{tag}{'_shuf' if shuffle else ''}): trials={len(y)} | cv={cv_used}")

        except Exception as e:
            skipped.append((sub, str(e)))
            log(f"Skipped {sub} REG({which},{tag}{'_shuf' if shuffle else ''}): {e}")

    if len(scores_all_r) < 8:
        raise RuntimeError(f"Too few subjects included for group stats REG({which},{tag}): n={len(scores_all_r)}")

    scores_all_r = np.stack(scores_all_r, axis=0)
    times = times_ref

    time_mask = (times >= TMIN_STAT) & (times <= TMAX_STAT)
    times_stat = times[time_mask]
    r_stat = scores_all_r[:, time_mask]

    # regression: two-sided by default (tail=0)
    stats_r = group_cluster_metric(r_stat, times_stat, chance=CHANCE_REG, tail=0)

    out_tag = f"decision_{tag}_{which}_ridgecorr" + ("_shuffle" if shuffle else "")

    np.savez(
        OUT / f"{out_tag}_group_results.npz",
        scores_by_subj=scores_all_r,
        times=times,
        included=np.array(included, dtype=object),
        skipped=np.array(skipped, dtype=object),
        T_obs=stats_r["T_obs"],
        p_map=stats_r["p_map"],
        cluster_pv=stats_r["cluster_pv"],
        t_thresh=stats_r["t_thresh"],
        chance=CHANCE_REG,
        resample_sfreq=RESAMPLE_SFREQ if RESAMPLE_SFREQ is not None else -1,
        tmin_stat=TMIN_STAT,
        tmax_stat=TMAX_STAT,
    )

    pd.DataFrame(scores_all_r, index=included, columns=np.round(times, 6)).to_csv(
        OUT / f"{out_tag}_scores_by_subject.csv"
    )

    plot_group_metric(
        r_stat,
        times_stat,
        stats_r["p_map"],
        title=f"Decision {which} ({tag.upper()} | {'SHUFFLED' if shuffle else 'REAL'}): Ridge regression (corr r)",
        out_png=OUT / f"{out_tag}_group_plot.png",
        ylabel="Decoding (corr r)",
        chance=CHANCE_REG,
        ylim=(-0.10, 0.40),
    )

    save_group_summaries(
        tag=out_tag,
        scores_all=r_stat,
        times=times_stat,
        included=included,
        stats_out=stats_r,
        out_dir=OUT,
        alpha=ALPHA_CLUSTER,
        chance=CHANCE_REG,
    )

    pd.DataFrame(subj_records).to_csv(
        OUT / f"{out_tag}_subject_summary.csv",
        index=False
    )

    min_p = float(np.min(stats_r["cluster_pv"])) if len(stats_r["cluster_pv"]) else 1.0
    log(f"\nFinished REG {which} ({tag}{'_shuffle' if shuffle else ''}): included n={len(included)}, min cluster p={min_p:.6f}")


# =====================================================================
# Main
# =====================================================================

def main():
    mne.set_log_level("WARNING")

    # -------------------------
    # BINARY
    # -------------------------
    if RUN_BINARY:
        run_binary("money", shuffle=False, tag="raw", control_by_other=False)
        run_binary("pain",  shuffle=False, tag="raw", control_by_other=False)

        if RUN_SHUFFLE:
            run_binary("money", shuffle=True, tag="raw", control_by_other=False)
            run_binary("pain",  shuffle=True, tag="raw", control_by_other=False)

        if RUN_CONTROL_BY_OTHER:
            run_binary("money", shuffle=False, tag="ctrlOther", control_by_other=True)
            run_binary("pain",  shuffle=False, tag="ctrlOther", control_by_other=True)

            if RUN_SHUFFLE:
                run_binary("money", shuffle=True, tag="ctrlOther", control_by_other=True)
                run_binary("pain",  shuffle=True, tag="ctrlOther", control_by_other=True)

    # -------------------------
    # REGRESSION
    # -------------------------
    if RUN_REGRESSION:
        run_regression("money", shuffle=False, tag="raw", control_by_other=False)
        run_regression("pain",  shuffle=False, tag="raw", control_by_other=False)

        if RUN_SHUFFLE:
            run_regression("money", shuffle=True, tag="raw", control_by_other=False)
            run_regression("pain",  shuffle=True, tag="raw", control_by_other=False)

        if RUN_CONTROL_BY_OTHER:
            run_regression("money", shuffle=False, tag="ctrlOther", control_by_other=True)
            run_regression("pain",  shuffle=False, tag="ctrlOther", control_by_other=True)

            if RUN_SHUFFLE:
                run_regression("money", shuffle=True, tag="ctrlOther", control_by_other=True)
                run_regression("pain",  shuffle=True, tag="ctrlOther", control_by_other=True)


if __name__ == "__main__":
    main()
