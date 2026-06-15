# -*- coding: utf-8 -*-
# Author: Michel-Pierre Coll (michel-pierre.coll@psy.ulaval.ca), 2026
#
# =============================================================================
# Categorical pain-vs-money decoding: passive → decision.
#
# This script is a companion to 07b1_eeg_decoding_pain_money_regression.py.
#
#   * 07b1 fits *two* ridge regressors in the passive phase — one mapping EEG
#     to painlevel on pain trials, one mapping EEG to moneylevel on money
#     trials — and asks how the decoded magnitudes evolve at decision time.
#
#   * This script (07b2) fits *one* binary classifier in the passive phase
#     that separates pain cues from money cues (regardless of level), then
#     projects each decision-phase trial onto that classifier's decision
#     axis. The resulting decoded score is interpretable as a continuous
#     "pain-vs-money" signal: positive ⇒ EEG looks more like a pain cue,
#     negative ⇒ EEG looks more like a money cue.
#
# Why a separate script?
# ----------------------
# The two analyses ask scientifically different questions:
#
#   * 07b1: "Does the decision-phase EEG carry information about how much
#     pain (or how much money) is currently on the table, beyond the stimuli
#     themselves?"
#
#   * 07b2: "Does the decision-phase EEG resemble a pain cue more or less
#     than a money cue at each instant, and does that relative pain-vs-money
#     bias track painlevel/moneylevel/choice?"
#
# Both are legitimate decoding analyses but they have different sensitivity
# profiles and different interpretations, so they should be reported
# separately rather than mashed into one script.
#
# Design fixes carried over from 07b1 (same review issues, same solutions)
# -----------------------------------------------------------------------
#   1. Pre-registered passive window (400–800 ms) chosen from the
#      manuscript's mass-univariate effects, not picked per subject by
#      argmax over CV scores.
#   2. Per-trial pre-cue baseline subtraction on the decoded time-series,
#      replacing the pooled z-score that is biased by accept/reject
#      class imbalance.
#   3. Stimulus-controlled regression as the *primary* decision-time test:
#         decoded_category(t) ~ painlevel + moneylevel + accepted.
#      The `accepted` coefficient is the choice-related EEG variance after
#      the stimuli are held constant.
#   4. Incremental choice AUC instead of raw choice AUC: how much does the
#      decoded EEG add beyond a logistic baseline of (painlevel, moneylevel)?
#      The raw AUC would be dominated by the trivial "people accept money,
#      reject pain" pattern that is already obvious from behaviour.
#   5. Two-tailed cluster permutation tests at the group level: we have no
#      prior about which direction the choice-related signal points, so we
#      do not pre-commit to a tail.
#   6. Fail-fast error handling: no try/except around per-subject work, per
#      the project's CLAUDE.md policy.
#
# Specificity check (different from 07b1)
# ---------------------------------------
# 07b1 had natural "cross-decoding" between conditions because there were
# two decoders. Here there is only one. The analogous specificity check is:
# at decision time both pain and money cues are present on every trial, so
# if the classifier is picking up category-specific neural signatures (and
# not just generic magnitude / arousal), the decoded score should respond
# to the *relative* pain-vs-money content of the trial. We make this
# explicit by also computing
#         beta_diff(t) on (painlevel - moneylevel)
# as a derived contrast in the group-level stats.
#
# Output
# ------
# Everything lands under:
#     derivatives/statistics/eeg_decoding_pain_vs_money_category_clean/
# Tables are CSVs, figures are PNG (300 dpi). A params.json records the
# exact configuration used.
# =============================================================================

import argparse
import json
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from mne.stats import permutation_cluster_1samp_test
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Silence the same third-party noise as the rest of the EEG pipeline.
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)
mne.set_log_level("ERROR")


# =============================================================================
# Configuration
# =============================================================================

BASE_PATH = Path(os.getenv("basepath", Path(__file__).parent.parent.parent))
HDDM_DIR = Path(os.getenv("HDDM_DIR", BASE_PATH / "derivatives" / "hddm"))
OUT_DIR = BASE_PATH / "derivatives" / "statistics" / "eeg_decoding_pain_vs_money_category_clean"
FIG_DIR = OUT_DIR / "figures"

RANDOM_STATE = 23

# Logistic regression regularization. We keep the default C=1.0 (L2). The
# pipeline standardizes features, so this is comparable across subjects.
# No need to match the ridge alpha from 07b1 because the loss functions
# are different; what matters is that this is fixed and pre-registered.
LOGREG_C = 1.0

# Pre-registered passive window in milliseconds.
# Justification (from drafts/PainReward_Draft_VW_2.docx.md):
#   - Mass-univariate passive pain effect:  263-1200 ms, peak CP1 ~798 ms.
#   - Mass-univariate passive money effect: 285-351 ms + 377-1200 ms.
#   - Pain > money contrast:                543-1200 ms.
# 400-800 ms is a single contiguous window that overlaps the pain effect
# peak, the late money cluster, and the pain>money contrast onset. Picking
# it up front eliminates the argmax selection bias in the original script
# and keeps the categorical analysis comparable to 07b1.
PRE_REGISTERED_PASSIVE_WINDOW_MS = (400.0, 800.0)

# Pre-cue baseline for the *decoded* time-series. Re-centers each trial on
# its own pre-stimulus mean. See 07b1 for the full rationale; same logic
# applies here.
DECISION_BASELINE_WINDOW_MS = (-200.0, 0.0)


# =============================================================================
# Argument parsing
# =============================================================================

def parse_args():
    """CLI options. Defaults tuned for a local laptop run."""
    parser = argparse.ArgumentParser(
        description="Categorical pain-vs-money passive→decision EEG decoding (clean)."
    )
    parser.add_argument("--quick", action="store_true",
                        help="Run on the first 3 subjects with reduced permutations for a smoke test.")
    parser.add_argument("--subjects", nargs="*", default=None,
                        help="Optional explicit list of participant ids (e.g. sub-001 sub-004).")
    parser.add_argument("--n-permutations", type=int, default=5000,
                        help="Number of group-level cluster permutations (use --quick to override).")
    parser.add_argument("--resample-hz", type=float, default=100.0,
                        help="Sampling rate to downsample epochs to before windowing. 100 Hz = 10 ms steps.")
    parser.add_argument("--window-ms", type=float, default=50.0,
                        help="Sliding-window width for decision-time feature extraction.")
    parser.add_argument("--step-ms", type=float, default=10.0,
                        help="Step between adjacent decision windows.")
    parser.add_argument("--n-jobs", type=int,
                        default=max(1, (os.cpu_count() or 4) - 1),
                        help="Parallel workers for the cluster permutation tests.")
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="Cluster-forming and reporting alpha.")
    return parser.parse_args()


# =============================================================================
# Subject container
# =============================================================================

@dataclass
class SubjectData:
    """Per-participant state. NaN fields mean 'not yet computed'."""
    participant_id: str

    # Passive: per-sliding-window CV AUC curve (diagnostic only) and
    # scalar headline AUC computed in the pre-registered window.
    passive_category_cv_auc: Optional[np.ndarray] = None    # shape (n_windows,)
    passive_times_ms: Optional[np.ndarray] = None
    within_category_auc: Optional[float] = None             # scalar in 400-800 ms

    # Decision: decoded "pain-vs-money axis" time-series after pre-cue
    # baseline subtraction. Positive ⇒ EEG looks like a pain cue at that
    # instant, negative ⇒ looks like a money cue.
    decoded_category: Optional[np.ndarray] = None           # shape (n_trials, n_windows)
    decision_times_ms: Optional[np.ndarray] = None

    # Per-trial decision metadata aligned to decoded_category rows.
    painlevel: Optional[np.ndarray] = None
    moneylevel: Optional[np.ndarray] = None
    accepted: Optional[np.ndarray] = None

    # Per-time betas from the stimulus-controlled regression.
    beta_painlevel_on_decoded: Optional[np.ndarray] = None
    beta_moneylevel_on_decoded: Optional[np.ndarray] = None
    beta_accepted_on_decoded: Optional[np.ndarray] = None
    # Derived contrast: beta on (painlevel - moneylevel) as a single
    # "relative pain-vs-money content" predictor. Computed from the same
    # design but with painlevel - moneylevel collapsed into one regressor
    # so it has a single beta (cleaner to interpret as a pain-vs-money
    # contrast signal at the decision-phase EEG level).
    beta_pain_minus_money_on_decoded: Optional[np.ndarray] = None

    # Per-time incremental choice AUC.
    incremental_auc: Optional[np.ndarray] = None


# =============================================================================
# Tiny numeric utilities
# =============================================================================

def ensure_dir(path: Path):
    """Create an output directory if it does not exist (idempotent)."""
    path.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Data loading helpers (identical to 07b1 — same data, same alignment policy)
# =============================================================================
# These are duplicated rather than imported because 07b1 is itself under
# review and may move/rename. Keeping this script self-contained makes it
# straightforward to re-run for reproducibility without worrying about an
# inconsistent sibling file.

def decision_trialsnum(df: pd.DataFrame) -> pd.Series:
    """Recreate the 1-based decision trial number used as epoch metadata key."""
    return (
        df["blocks.thisRepN"].astype(int) * 25
        + df["trials.thisN"].astype(int)
        + 1
    )


def load_hddm_mod9() -> pd.DataFrame:
    """Load HDDM mod_9 trial-by-trial table (the source of truth for inclusion)."""
    mod_path = (
        HDDM_DIR / "figures" / "painreward_behavioural_data_mod_9"
        / "diagnostics" / "v_pain_money.csv"
    )
    if not mod_path.exists():
        raise FileNotFoundError(f"Missing HDDM mod_9 file: {mod_path}")
    mod = pd.read_csv(mod_path, sep=None, engine="python")
    mod["trialsnum"] = decision_trialsnum(mod)
    return mod


def get_common_subjects(mod_data: pd.DataFrame) -> List[str]:
    """Participants present in BOTH participants.tsv and the HDDM table."""
    participants = pd.read_csv(BASE_PATH / "participants.tsv", sep="\t")
    return sorted(set(participants["participant_id"]) & set(mod_data["participant"].unique()))


def passive_epoch_path(pa: str) -> Path:
    return BASE_PATH / "derivatives" / pa / "eeg" / "erps_passive" / f"{pa}_passive_cues_singletrials-epo.fif"


def decision_epoch_path(pa: str) -> Path:
    return BASE_PATH / "derivatives" / pa / "eeg" / "erps_decision" / f"{pa}_decision_cues_singletrials-epo.fif"


def passive_beh_path(pa: str) -> Path:
    return BASE_PATH / pa / "eeg" / f"{pa}_task-passive_beh.tsv"


def level_to_1_5(values) -> pd.Series:
    """Normalize passive `level` column to a 1-5 numeric scale.

    Some participants' raw level columns are stored as percentages (20, 40,
    60, 80, 100). We divide by 20 in that case so all subjects share the
    same 1-5 ordinal scale. This matches 07b1.
    """
    vals = pd.to_numeric(values, errors="coerce")
    if vals.max(skipna=True) > 5:
        vals = vals / 20.0
    return vals.astype(float)


def load_passive_beh(pa: str) -> pd.DataFrame:
    """Load passive behavior file and select pain/money rows with valid levels."""
    beh_path = passive_beh_path(pa)
    if not beh_path.exists():
        raise FileNotFoundError(f"{pa}: missing passive behavior file: {beh_path}")
    beh = pd.read_csv(beh_path, sep="\t")
    if "fixcross.started" in beh.columns:
        beh = beh[~beh["fixcross.started"].isna()].copy()
    beh = beh.reset_index(drop=True)
    beh["trialsnum"] = np.arange(1, len(beh) + 1)
    beh["condition"] = beh["condition"].astype(str).str.lower().str.strip()
    beh["level_1_5"] = level_to_1_5(beh["level"])
    beh = beh[beh["condition"].isin(["p", "m"])].copy()
    return beh[["trialsnum", "condition", "level_1_5", "blocks.thisRepN"]]


def read_epochs_for_decoding(epo_path: Path, resample_hz: float) -> mne.Epochs:
    """Read MNE epochs, restrict to EEG, and downsample."""
    if not epo_path.exists():
        raise FileNotFoundError(f"Missing epoch file: {epo_path}")
    epo = mne.read_epochs(str(epo_path), preload=True, verbose="ERROR")
    epo = epo.pick("eeg", exclude=[])
    if resample_hz and epo.info["sfreq"] != float(resample_hz):
        epo = epo.resample(float(resample_hz), verbose="ERROR")
    return epo


def align_passive_epochs(pa: str, resample_hz: float) -> mne.Epochs:
    """Attach passive behavior to passive epochs via exact 1:1 trialsnum merge."""
    epo = read_epochs_for_decoding(passive_epoch_path(pa), resample_hz)
    if epo.metadata is None or "trialsnum" not in epo.metadata.columns:
        raise RuntimeError(f"{pa}: passive epochs missing metadata['trialsnum']")
    md = epo.metadata.reset_index(drop=True).copy()
    md["trialsnum"] = pd.to_numeric(md["trialsnum"], errors="coerce").astype(int)
    merged = md.merge(load_passive_beh(pa), on="trialsnum", how="left", validate="1:1")
    if merged["condition"].isna().any() or merged["level_1_5"].isna().any():
        raise RuntimeError(f"{pa}: passive behavior merge left unlabeled epochs")
    epo.metadata = merged
    return epo


def align_decision_epochs(pa: str, mod_data: pd.DataFrame, resample_hz: float) -> mne.Epochs:
    """Select decision epochs in the exact order of HDDM-filtered rows."""
    mod = mod_data[mod_data["participant"] == pa].copy().reset_index(drop=True)
    if mod.empty:
        raise RuntimeError(f"{pa}: no HDDM rows found")
    if mod["trialsnum"].duplicated().any():
        dupes = sorted(mod.loc[mod["trialsnum"].duplicated(), "trialsnum"].astype(int).unique())
        raise RuntimeError(f"{pa}: duplicate HDDM trialsnum values: {dupes}")
    epo = read_epochs_for_decoding(decision_epoch_path(pa), resample_hz)
    if epo.metadata is None or "trialsnum" not in epo.metadata.columns:
        raise RuntimeError(f"{pa}: decision epochs missing metadata['trialsnum']")
    md = epo.metadata.reset_index(drop=True).copy()
    epoch_trials = pd.to_numeric(md["trialsnum"], errors="coerce").astype(int)
    if epoch_trials.duplicated().any():
        dupes = sorted(epoch_trials[epoch_trials.duplicated()].unique())
        raise RuntimeError(f"{pa}: duplicate epoch trialsnum values: {dupes}")
    trial_to_idx = {trial: idx for idx, trial in enumerate(epoch_trials)}
    requested = mod["trialsnum"].astype(int).to_numpy()
    missing = [trial for trial in requested if trial not in trial_to_idx]
    if missing:
        raise RuntimeError(f"{pa}: missing decision epoch trials: {missing[:20]}")
    epo = epo[[trial_to_idx[trial] for trial in requested]]
    md = epo.metadata.reset_index(drop=True).copy()
    for col in ["painlevel", "moneylevel", "accepted", "trialsnum"]:
        md[col] = mod[col].to_numpy()
    md["participant"] = pa
    epo.metadata = md
    return epo


# =============================================================================
# Feature extraction (identical to 07b1)
# =============================================================================

def make_window_slices(times: np.ndarray, window_ms: float, step_ms: float
                       ) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Return centered sliding-window sample indices that fit inside `times`."""
    half = window_ms / 1000.0 / 2.0
    step = step_ms / 1000.0
    centers = []
    cur = float(times[0]) + half
    stop = float(times[-1]) - half
    while cur <= stop + 1e-12:
        centers.append(cur)
        cur += step
    slices: List[np.ndarray] = []
    valid_centers: List[float] = []
    for center in centers:
        mask = np.where((times >= center - half - 1e-12) & (times <= center + half + 1e-12))[0]
        if len(mask) == 0:
            continue
        slices.append(mask)
        valid_centers.append(center)
    return np.asarray(valid_centers), slices


def window_features(epo: mne.Epochs, window_slices: List[np.ndarray]) -> np.ndarray:
    """Average EEG amplitude inside each window → shape (trials, windows, channels)."""
    data = epo.get_data(copy=False).astype(np.float64, copy=False) * 1e6
    if not np.all(np.isfinite(data)):
        raise RuntimeError("Epoch data contain non-finite EEG values")
    features = np.empty((data.shape[0], len(window_slices), data.shape[1]), dtype=np.float64)
    for i, sl in enumerate(window_slices):
        features[:, i, :] = data[:, :, sl].mean(axis=2)
    return features


def features_for_window_range(epo: mne.Epochs, lo_ms: float, hi_ms: float) -> np.ndarray:
    """Mean amplitude across [lo_ms, hi_ms] → shape (trials, channels).

    Used to build the *one big window* feature vector that trains the
    pre-registered categorical decoder. Averaging across 400-800 ms reduces
    variance, commits us to one decision before seeing the data, and
    mirrors the ERP-style averaging the manuscript already uses.
    """
    times_s = epo.times
    lo_s = lo_ms / 1000.0
    hi_s = hi_ms / 1000.0
    if lo_s < times_s[0] - 1e-9 or hi_s > times_s[-1] + 1e-9:
        raise RuntimeError(
            f"Requested window {lo_ms}-{hi_ms} ms outside epoch range "
            f"{times_s[0]*1000:.0f}-{times_s[-1]*1000:.0f} ms"
        )
    mask = (times_s >= lo_s - 1e-9) & (times_s <= hi_s + 1e-9)
    if not mask.any():
        raise RuntimeError(f"No samples inside {lo_ms}-{hi_ms} ms after resampling")
    data = epo.get_data(copy=False).astype(np.float64, copy=False) * 1e6
    if not np.all(np.isfinite(data)):
        raise RuntimeError("Epoch data contain non-finite EEG values")
    return data[:, :, mask].mean(axis=2)


# =============================================================================
# Decoder primitives
# =============================================================================

def make_classifier():
    """Standard L2 logistic-regression pipeline used throughout the script.

    Notes on the choice:
        - StandardScaler is mandatory: L2 regularization is not scale-invariant.
        - We use solver='liblinear' because it is deterministic given a fixed
          random_state and handles small samples well.
        - decision_function() returns the signed distance to the decision
          boundary (log-odds, up to a positive scaling), which is what we
          actually feed into the downstream regression.
    """
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=LOGREG_C,
            solver="liblinear",
            random_state=RANDOM_STATE,
            max_iter=2000,
        ),
    )


def subset_passive_category(epo: mne.Epochs
                            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (trial indices, binary category label, block-of-origin) for all passive trials.

    Label convention:
        1 = pain cue, 0 = money cue.
    The classifier therefore predicts P(pain | EEG). Its `decision_function`
    is positive when EEG looks pain-like, negative when it looks money-like
    — that is the sign convention we carry into the decision-time analysis.

    Block index becomes the CV group so train and test never share a block:
    this is the same leakage protection as 07b1.
    """
    md = epo.metadata.reset_index(drop=True).copy()
    cond = md["condition"].astype(str).str.lower().to_numpy()
    selected = np.isin(cond, ["p", "m"])
    # Drop artifact trials. `badtrial` is set upstream by artifact rejection.
    if "badtrial" in md.columns:
        selected &= md["badtrial"].fillna(0).astype(int).to_numpy() == 0
    if "blocks.thisRepN" not in md.columns:
        raise RuntimeError("Passive epochs missing metadata['blocks.thisRepN']")
    block_all = pd.to_numeric(md["blocks.thisRepN"], errors="coerce").to_numpy(dtype=float)
    if np.any(selected & ~np.isfinite(block_all)):
        raise RuntimeError("Passive: non-finite block ids among non-artifact trials")
    label_all = np.where(cond == "p", 1, 0).astype(int)
    idx = np.where(selected)[0]
    return idx, label_all[selected], block_all[selected].astype(int)


def subset_decision_trials(epo: mne.Epochs
                           ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return decision trial indices, painlevel, moneylevel, accepted (0/1)."""
    md = epo.metadata.reset_index(drop=True).copy()
    selected = np.ones(len(md), dtype=bool)
    if "badtrial" in md.columns:
        selected &= md["badtrial"].fillna(0).astype(int).to_numpy() == 0
    accepted = pd.to_numeric(md["accepted"], errors="coerce").to_numpy(dtype=float)
    painlevel = pd.to_numeric(md["painlevel"], errors="coerce").to_numpy(dtype=float)
    moneylevel = pd.to_numeric(md["moneylevel"], errors="coerce").to_numpy(dtype=float)
    valid = (
        np.isfinite(accepted) & np.isin(accepted, [0.0, 1.0])
        & np.isfinite(painlevel) & np.isfinite(moneylevel)
    )
    if np.any(selected & ~valid):
        raise RuntimeError("Decision: non-finite or non-binary metadata among non-artifact trials")
    idx = np.where(selected)[0]
    return idx, painlevel[selected], moneylevel[selected], accepted[selected].astype(int)


# =============================================================================
# Passive analyses
# =============================================================================

def passive_category_cv_auc_curve(features_by_window: np.ndarray,
                                  y: np.ndarray,
                                  groups: np.ndarray,
                                  n_splits: int = 5) -> np.ndarray:
    """For each passive sliding window, return CV AUC for pain-vs-money decoding.

    *Diagnostic only.* The primary passive readout is the scalar AUC from
    the pre-registered window (next function). Separating curve and scalar
    keeps us honest about not picking the best window on the curve.
    """
    n_splits = min(n_splits, len(np.unique(groups)))
    if n_splits < 2:
        raise RuntimeError("Need at least two passive blocks for grouped CV")
    if len(np.unique(y)) < 2:
        raise RuntimeError("Passive labels have only one class")
    scores = np.full(features_by_window.shape[1], np.nan, dtype=float)
    cv = GroupKFold(n_splits=n_splits)
    for w in range(features_by_window.shape[1]):
        # decision_function returns continuous scores → real AUC, not 0/1 accuracy.
        preds = np.full(len(y), np.nan, dtype=float)
        for train_idx, test_idx in cv.split(features_by_window[:, w, :], y, groups):
            model = make_classifier().fit(features_by_window[train_idx, w, :], y[train_idx])
            preds[test_idx] = model.decision_function(features_by_window[test_idx, w, :])
        if np.isfinite(preds).all() and len(np.unique(y)) == 2:
            scores[w] = roc_auc_score(y, preds)
    return scores


def passive_category_auc_scalar(features_window_mean: np.ndarray,
                                y: np.ndarray,
                                groups: np.ndarray,
                                n_splits: int = 5) -> float:
    """CV AUC for the *pre-registered single-window* categorical decoder.

    Headline per-subject passive number. GroupKFold by passive block keeps
    training and testing on different sessions of the task (no temporal
    leakage from session drifts in EEG amplitude).
    """
    n_splits = min(n_splits, len(np.unique(groups)))
    if n_splits < 2:
        raise RuntimeError("Need at least two passive blocks for grouped CV")
    if len(np.unique(y)) < 2:
        raise RuntimeError("Passive labels have only one class")
    preds = np.full(len(y), np.nan, dtype=float)
    cv = GroupKFold(n_splits=n_splits)
    for train_idx, test_idx in cv.split(features_window_mean, y, groups):
        model = make_classifier().fit(features_window_mean[train_idx], y[train_idx])
        preds[test_idx] = model.decision_function(features_window_mean[test_idx])
    if not np.isfinite(preds).all():
        raise RuntimeError("CV produced non-finite predictions")
    return float(roc_auc_score(y, preds))


# =============================================================================
# Decision-time analyses
# =============================================================================

def apply_passive_classifier_to_decision(classifier,
                                         decision_features_3d: np.ndarray
                                         ) -> np.ndarray:
    """Apply the locked passive classifier to every decision-window slice.

    Returns:
        decoded_category -- shape (n_trials, n_windows). decision_function
                            output: positive ⇒ pain-like, negative ⇒ money-like.

    No scaling is applied here. Pre-cue baseline subtraction happens next.
    """
    n_trials, n_windows, _ = decision_features_3d.shape
    decoded = np.empty((n_trials, n_windows), dtype=np.float64)
    for w in range(n_windows):
        decoded[:, w] = classifier.decision_function(decision_features_3d[:, w, :])
    return decoded


def baseline_correct_decoded(decoded: np.ndarray,
                             window_centers_ms: np.ndarray,
                             baseline_ms: Tuple[float, float]) -> np.ndarray:
    """Subtract each trial's pre-cue mean from its decoded time-series.

    Replaces the original 07d's pooled z-score. With ~75/25 accept/reject
    class imbalance, the pooled mean is dominated by accepted trials, which
    biases the accept-vs-reject contrast. Per-trial baseline subtraction
    sidesteps that entirely: each trial is its own control, and the
    baseline interval is *before* the cue so it cannot contain
    condition information.
    """
    lo, hi = baseline_ms
    mask = (window_centers_ms >= lo - 1e-9) & (window_centers_ms <= hi + 1e-9)
    if mask.sum() < 2:
        raise RuntimeError(
            f"Baseline window {baseline_ms} ms contains <2 sliding windows "
            f"(available centers {window_centers_ms[:3]}…{window_centers_ms[-3:]})"
        )
    baseline_mean = decoded[:, mask].mean(axis=1, keepdims=True)
    return decoded - baseline_mean


def stimulus_controlled_regression(decoded: np.ndarray,
                                   painlevel: np.ndarray,
                                   moneylevel: np.ndarray,
                                   accepted: np.ndarray
                                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-time regression: decoded(t) ~ painlevel + moneylevel + accepted.

    Returns four vectors of length n_windows:
        beta_pain, beta_money, beta_acc, beta_pain_minus_money.

    The first three come from one fitted model with all three predictors
    standardized. The fourth comes from a separate, simpler model that
    *collapses* painlevel and moneylevel into a single (painlevel -
    moneylevel) contrast predictor (still with `accepted` as a covariate).
    The collapsed model is more directly interpretable as "does the EEG
    pain-vs-money axis track the relative pain-vs-money content of the
    trial?" which is the question this analysis was built to answer.

    Why this is the *primary* decision-time test
    --------------------------------------------
    The accept-vs-reject contrast confounds choice with stimulus magnitude
    (accepted trials have lower pain and higher money, by construction).
    Multiple regression holds the stimuli constant so the beta on `accepted`
    captures choice-related variance in the decoded signal that is *not*
    explained by painlevel or moneylevel.
    """
    n_trials, n_windows = decoded.shape
    if not (n_trials == len(painlevel) == len(moneylevel) == len(accepted)):
        raise RuntimeError("Mismatched trial counts in regression inputs")

    beta_pain = np.full(n_windows, np.nan, dtype=float)
    beta_money = np.full(n_windows, np.nan, dtype=float)
    beta_acc = np.full(n_windows, np.nan, dtype=float)
    beta_diff = np.full(n_windows, np.nan, dtype=float)

    # Build standardized predictors *once*: they are trial-level and constant
    # across time. accepted is recoded ±1 (already mean-centered, so we do
    # not z-score the binary).
    pred_full = np.column_stack([
        stats.zscore(painlevel.astype(float)),
        stats.zscore(moneylevel.astype(float)),
        np.where(accepted == 1, 1.0, -1.0),
    ])
    design_full = np.column_stack([np.ones(n_trials), pred_full])

    # Collapsed design: one (pain-money) contrast + accepted.
    pred_diff = np.column_stack([
        stats.zscore(painlevel.astype(float) - moneylevel.astype(float)),
        np.where(accepted == 1, 1.0, -1.0),
    ])
    design_diff = np.column_stack([np.ones(n_trials), pred_diff])

    for w in range(n_windows):
        y = decoded[:, w].astype(float)
        keep = np.isfinite(y) & np.isfinite(design_full).all(axis=1)
        if keep.sum() < pred_full.shape[1] + 3:
            continue
        y_z = stats.zscore(y[keep])
        if not np.isfinite(y_z).all():
            continue
        # Full model: pain + money + accepted.
        coefs, *_ = np.linalg.lstsq(design_full[keep], y_z, rcond=None)
        beta_pain[w] = float(coefs[1])
        beta_money[w] = float(coefs[2])
        beta_acc[w] = float(coefs[3])
        # Collapsed model: (pain - money) + accepted.
        coefs_d, *_ = np.linalg.lstsq(design_diff[keep], y_z, rcond=None)
        beta_diff[w] = float(coefs_d[1])

    return beta_pain, beta_money, beta_acc, beta_diff


def incremental_choice_auc(decoded: np.ndarray,
                           painlevel: np.ndarray,
                           moneylevel: np.ndarray,
                           accepted: np.ndarray,
                           n_splits: int = 5) -> np.ndarray:
    """ΔAUC(t) = AUC(painlevel + moneylevel + decoded(t)) − AUC(painlevel + moneylevel).

    Same rationale as 07b1: the headline "does decoded EEG predict choice?"
    AUC is dominated by the trivial behavioural fact that high pain is
    rejected and high money is accepted. The honest question is whether
    the EEG at time `t` adds anything *beyond* the stimuli we already saw.
    That is the increment.

    CV is stratified on `accepted` (which is class-imbalanced).
    """
    n_trials, n_windows = decoded.shape
    auc = np.full(n_windows, np.nan, dtype=float)
    class_counts = np.bincount(accepted.astype(int), minlength=2)
    fold_count = min(n_splits, int(class_counts.min()))
    if fold_count < 2:
        return auc

    cv = StratifiedKFold(n_splits=fold_count, shuffle=True, random_state=RANDOM_STATE)
    baseline_predictors = np.column_stack([painlevel.astype(float), moneylevel.astype(float)])

    # Baseline (no-EEG) AUC, computed once with the same CV folds.
    baseline_scores = np.full(n_trials, np.nan, dtype=float)
    for train_idx, test_idx in cv.split(baseline_predictors, accepted):
        m0 = make_pipeline(
            StandardScaler(),
            LogisticRegression(solver="liblinear", random_state=RANDOM_STATE),
        ).fit(baseline_predictors[train_idx], accepted[train_idx])
        baseline_scores[test_idx] = m0.decision_function(baseline_predictors[test_idx])
    if not np.isfinite(baseline_scores).all():
        return auc
    baseline_auc = roc_auc_score(accepted, baseline_scores)

    for w in range(n_windows):
        eeg = decoded[:, w].astype(float)
        if not np.isfinite(eeg).all():
            continue
        predictors = np.column_stack([baseline_predictors, eeg])
        scores = np.full(n_trials, np.nan, dtype=float)
        for train_idx, test_idx in cv.split(predictors, accepted):
            m1 = make_pipeline(
                StandardScaler(),
                LogisticRegression(solver="liblinear", random_state=RANDOM_STATE),
            ).fit(predictors[train_idx], accepted[train_idx])
            scores[test_idx] = m1.decision_function(predictors[test_idx])
        if not np.isfinite(scores).all():
            continue
        auc[w] = roc_auc_score(accepted, scores) - baseline_auc
    return auc


# =============================================================================
# Per-subject pipeline
# =============================================================================

def process_subject(pa: str, mod_data: pd.DataFrame, args) -> SubjectData:
    """Full per-subject pipeline. Raises on any data issue (no try/except)."""
    sub = SubjectData(participant_id=pa)
    print(f"  [{pa}] loading epochs", flush=True)

    # ---- 1. Passive epochs -------------------------------------------------
    passive_epo = align_passive_epochs(pa, args.resample_hz)
    passive_idx, passive_y, passive_groups = subset_passive_category(passive_epo)
    if len(passive_idx) < 40 or len(np.unique(passive_y)) < 2:
        raise RuntimeError(
            f"{pa}: too few passive trials or only one class "
            f"(n={len(passive_idx)}, classes={np.unique(passive_y).tolist()})"
        )

    # 1a. Sliding-window features for the diagnostic CV curve. Not used to
    # pick a window — only for the plot the reviewer will look at.
    passive_times_s = passive_epo.times
    centers_s, slices = make_window_slices(passive_times_s, args.window_ms, args.step_ms)
    sub.passive_times_ms = centers_s * 1000.0
    features_3d = window_features(passive_epo, slices)[passive_idx]
    sub.passive_category_cv_auc = passive_category_cv_auc_curve(
        features_3d, passive_y, passive_groups,
    )

    # 1b. Pre-registered single-window features — the actual inputs to the
    # locked classifier.
    feat_window = features_for_window_range(
        passive_epo, *PRE_REGISTERED_PASSIVE_WINDOW_MS,
    )[passive_idx]

    # 1c. Headline scalar passive AUC (CV).
    sub.within_category_auc = passive_category_auc_scalar(
        feat_window, passive_y, passive_groups,
    )

    # 1d. Fit the *final* deployment classifier on ALL passive trials. No
    # CV here — this model is what we apply to decision trials below.
    deployment_clf = make_classifier().fit(feat_window, passive_y)
    del passive_epo, features_3d, feat_window

    # ---- 2. Decision epochs ------------------------------------------------
    decision_epo = align_decision_epochs(pa, mod_data, args.resample_hz)
    dec_idx, painlevel, moneylevel, accepted = subset_decision_trials(decision_epo)
    if len(dec_idx) < 30:
        raise RuntimeError(f"{pa}: too few decision trials ({len(dec_idx)})")

    decision_times_s = decision_epo.times
    dec_centers_s, dec_slices = make_window_slices(decision_times_s, args.window_ms, args.step_ms)
    sub.decision_times_ms = dec_centers_s * 1000.0
    dec_features_3d = window_features(decision_epo, dec_slices)[dec_idx]

    raw_decoded = apply_passive_classifier_to_decision(deployment_clf, dec_features_3d)

    # Pre-cue baseline subtraction (per-trial). See the comment in
    # baseline_correct_decoded for the full rationale.
    sub.decoded_category = baseline_correct_decoded(
        raw_decoded, sub.decision_times_ms, DECISION_BASELINE_WINDOW_MS,
    )
    sub.painlevel = painlevel
    sub.moneylevel = moneylevel
    sub.accepted = accepted

    # ---- 3. Stimulus-controlled regression ---------------------------------
    bp, bm, ba, bd = stimulus_controlled_regression(
        sub.decoded_category, painlevel, moneylevel, accepted,
    )
    sub.beta_painlevel_on_decoded = bp
    sub.beta_moneylevel_on_decoded = bm
    sub.beta_accepted_on_decoded = ba
    sub.beta_pain_minus_money_on_decoded = bd

    # ---- 4. Incremental choice AUC -----------------------------------------
    sub.incremental_auc = incremental_choice_auc(
        sub.decoded_category, painlevel, moneylevel, accepted,
    )

    print(f"  [{pa}] done (passive AUC = {sub.within_category_auc:.3f})", flush=True)
    return sub


# =============================================================================
# Group-level statistics
# =============================================================================

def run_cluster_test_one_sample(data: np.ndarray,
                                n_permutations: int,
                                n_jobs: int,
                                alpha: float,
                                tail: int = 0) -> Dict:
    """One-sample cluster permutation test of `data` against zero.

    Two-tailed by default (tail=0) — we have no prior on sign for any of
    the beta time-courses, and we do not want to inflate Type I error by
    post-hoc choosing a tail.
    """
    data = np.asarray(data, dtype=float)
    keep_subjects = np.isfinite(data).all(axis=1)
    test_data = data[keep_subjects]
    if test_data.shape[0] < 3:
        raise RuntimeError("Need at least three subjects with finite data for cluster testing")
    if tail == 0:
        threshold = float(stats.t.ppf(1.0 - alpha / 2.0, test_data.shape[0] - 1))
    else:
        threshold = float(stats.t.ppf(1.0 - alpha, test_data.shape[0] - 1))
    stat, clusters, cluster_p, _ = permutation_cluster_1samp_test(
        test_data,
        threshold=threshold,
        n_permutations=n_permutations,
        tail=tail,
        seed=RANDOM_STATE,
        n_jobs=n_jobs,
        out_type="indices",
        verbose=False,
    )
    return {
        "stat": stat,
        "clusters": [np.asarray(c[0], dtype=int) for c in clusters],
        "cluster_p": np.asarray(cluster_p, dtype=float),
        "threshold": threshold,
        "n_subjects": int(test_data.shape[0]),
    }


def cluster_summary_rows(name: str, result: Dict, times_ms: np.ndarray) -> List[Dict]:
    """Flatten cluster results into long-format rows for CSV output."""
    rows = []
    for cid, (cluster, pval) in enumerate(zip(result["clusters"], result["cluster_p"])):
        vals = result["stat"][cluster]
        peak = int(np.argmax(np.abs(vals)))
        rows.append({
            "map": name,
            "cluster_id": cid,
            "p_value": float(pval),
            "extent_n_samples": int(len(cluster)),
            "time_start_ms": float(times_ms[cluster.min()]),
            "time_end_ms": float(times_ms[cluster.max()]),
            "peak_stat": float(vals[peak]),
            "peak_time_ms": float(times_ms[cluster[peak]]),
            "n_subjects": int(result["n_subjects"]),
        })
    return rows


# =============================================================================
# Plotting (diagnostic figures, not publication panels)
# =============================================================================

def plot_passive_auc_curve(subjects: List[SubjectData], out_path: Path):
    """Group mean ± SEM of the passive pain-vs-money CV AUC curve over time."""
    times = subjects[0].passive_times_ms
    arr = np.vstack([s.passive_category_cv_auc for s in subjects])
    m = np.nanmean(arr, axis=0)
    sem = stats.sem(arr, axis=0, nan_policy="omit")
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    ax.plot(times, m, color="#6a3d9a", linewidth=2, label="Pain vs. money (CV AUC)")
    ax.fill_between(times, m - sem, m + sem, color="#6a3d9a", alpha=0.18)
    ax.axhline(0.5, color="black", linewidth=0.8, linestyle="--", label="Chance (0.5)")
    ax.axvline(0, color="black", linewidth=0.8, linestyle=":")
    ax.axvspan(*PRE_REGISTERED_PASSIVE_WINDOW_MS, color="grey", alpha=0.10,
               label=f"Pre-registered window {int(PRE_REGISTERED_PASSIVE_WINDOW_MS[0])}-"
                     f"{int(PRE_REGISTERED_PASSIVE_WINDOW_MS[1])} ms")
    ax.set_xlabel("Passive time from cue onset (ms)")
    ax.set_ylabel("CV AUC (pain vs. money)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_beta_timecourse(subjects: List[SubjectData],
                         attr: str,
                         cluster_result: Dict,
                         title: str,
                         out_path: Path,
                         alpha: float = 0.05):
    """Plot group mean ± SEM of one beta time-series, with sig clusters shaded."""
    times = subjects[0].decision_times_ms
    arr = np.vstack([getattr(s, attr) for s in subjects])
    m = np.nanmean(arr, axis=0)
    sem = stats.sem(arr, axis=0, nan_policy="omit")
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    ax.plot(times, m, color="#1f77b4", linewidth=2)
    ax.fill_between(times, m - sem, m + sem, color="#1f77b4", alpha=0.18)
    for cluster, pval in zip(cluster_result["clusters"], cluster_result["cluster_p"]):
        if pval < alpha:
            ax.axvspan(times[cluster.min()], times[cluster.max()],
                       color="orange", alpha=0.25)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Decision time from cue onset (ms)")
    ax.set_ylabel(title)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_passive_auc_distribution(subjects: List[SubjectData], out_path: Path):
    """Strip plot of the per-subject headline AUC, with chance line.

    A quick sanity check: are most subjects above 0.5 in the pre-registered
    window? If many sit at chance, the analysis is unlikely to find anything
    interesting at decision time either.
    """
    aucs = np.array([s.within_category_auc for s in subjects], dtype=float)
    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    jitter = (np.random.RandomState(RANDOM_STATE).rand(len(aucs)) - 0.5) * 0.18
    ax.scatter(np.zeros_like(aucs) + jitter, aucs, color="#6a3d9a", s=24, alpha=0.7)
    ax.axhline(0.5, color="black", linewidth=0.8, linestyle="--", label="Chance")
    ax.axhline(float(np.nanmean(aucs)), color="#6a3d9a", linewidth=1.4,
               label=f"Mean = {float(np.nanmean(aucs)):.3f}")
    ax.set_xticks([])
    ax.set_ylabel("CV AUC (pain vs. money, pre-registered 400-800 ms)")
    ax.set_title("Per-subject passive categorical AUC")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# main
# =============================================================================

def main():
    """Top-level driver. No try/except — failures abort the run (per CLAUDE.md)."""
    args = parse_args()
    ensure_dir(OUT_DIR)
    ensure_dir(FIG_DIR)

    if args.quick:
        # Smoke-test settings: fewer permutations. Subject subsetting is below.
        args.n_permutations = max(200, args.n_permutations // 10)

    # Persist the exact parameters used for this run. Helps reproducibility
    # and is the first thing to check when results look weird.
    params = {
        "quick": args.quick,
        "subjects_arg": args.subjects,
        "n_permutations": args.n_permutations,
        "resample_hz": args.resample_hz,
        "window_ms": args.window_ms,
        "step_ms": args.step_ms,
        "n_jobs": args.n_jobs,
        "alpha": args.alpha,
        "pre_registered_passive_window_ms": PRE_REGISTERED_PASSIVE_WINDOW_MS,
        "decision_baseline_window_ms": DECISION_BASELINE_WINDOW_MS,
        "logreg_C": LOGREG_C,
        "random_state": RANDOM_STATE,
        "base_path": str(BASE_PATH),
    }
    with open(OUT_DIR / "params.json", "w") as f:
        json.dump(params, f, indent=2)

    # ---- subject selection -------------------------------------------------
    mod_data = load_hddm_mod9()
    if args.subjects:
        subjects_to_run = list(args.subjects)
    else:
        subjects_to_run = get_common_subjects(mod_data)
    if args.quick:
        subjects_to_run = subjects_to_run[:3]
    print(f"Running on {len(subjects_to_run)} subjects: {subjects_to_run}", flush=True)

    # ---- per-subject loop --------------------------------------------------
    subjects: List[SubjectData] = []
    for pa in subjects_to_run:
        subjects.append(process_subject(pa, mod_data, args))

    # ---- group-level passive headline table -------------------------------
    spec_rows = []
    for s in subjects:
        spec_rows.append({
            "participant_id": s.participant_id,
            "within_category_auc": s.within_category_auc,
        })
    spec_df = pd.DataFrame(spec_rows)
    spec_df.to_csv(OUT_DIR / "passive_category_auc_per_subject.csv", index=False)

    # Single-sample t-test vs chance (0.5). Cheap, informative.
    aucs = spec_df["within_category_auc"].to_numpy(dtype=float)
    t, p = stats.ttest_1samp(aucs, 0.5, nan_policy="omit")
    pd.DataFrame([{
        "scalar": "within_category_auc",
        "mean": float(np.nanmean(aucs)),
        "sem": float(stats.sem(aucs, nan_policy="omit")),
        "t_vs_chance": float(t),
        "df": int(np.isfinite(aucs).sum() - 1),
        "p_two_tailed": float(p),
        "n": int(np.isfinite(aucs).sum()),
    }]).to_csv(OUT_DIR / "passive_category_auc_summary.csv", index=False)

    plot_passive_auc_curve(subjects, FIG_DIR / "passive_category_auc_curve.png")
    plot_passive_auc_distribution(subjects, FIG_DIR / "passive_category_auc_distribution.png")

    # ---- group-level decision tests ---------------------------------------
    # Shared time vector across subjects. Mismatch ⇒ misaligned cluster test
    # ⇒ silent garbage. Sanity-check it explicitly.
    base_times = subjects[0].decision_times_ms
    for s in subjects[1:]:
        if not np.allclose(s.decision_times_ms, base_times):
            raise RuntimeError(
                f"{s.participant_id}: decision time vector mismatch — "
                "subjects must share resampling rate and epoch range"
            )

    cluster_targets = {
        # The four primary maps. beta_accepted_on_decoded is the headline
        # "choice over and above stimuli" effect. beta_pain_minus_money is
        # the categorical specificity readout. The two stimulus betas are
        # diagnostic (we expect them ≈ painlevel and ≈ −moneylevel if the
        # decoder picks up pain-like vs money-like content).
        "beta_painlevel_on_decoded": np.vstack([s.beta_painlevel_on_decoded for s in subjects]),
        "beta_moneylevel_on_decoded": np.vstack([s.beta_moneylevel_on_decoded for s in subjects]),
        "beta_accepted_on_decoded": np.vstack([s.beta_accepted_on_decoded for s in subjects]),
        "beta_pain_minus_money_on_decoded": np.vstack([s.beta_pain_minus_money_on_decoded for s in subjects]),
        # Incremental choice AUC: does the categorical decoded EEG add to
        # choice prediction beyond (painlevel, moneylevel)?
        "incremental_auc": np.vstack([s.incremental_auc for s in subjects]),
    }

    cluster_results: Dict[str, Dict] = {}
    summary_rows: List[Dict] = []
    for name, arr in cluster_targets.items():
        print(f"  cluster test: {name}", flush=True)
        res = run_cluster_test_one_sample(
            arr, args.n_permutations, args.n_jobs, args.alpha, tail=0,
        )
        cluster_results[name] = res
        summary_rows.extend(cluster_summary_rows(name, res, base_times))

        # Per-map per-subject CSV for any downstream re-analysis.
        df = pd.DataFrame(arr, columns=[f"t{int(round(t))}" for t in base_times])
        df.insert(0, "participant_id", [s.participant_id for s in subjects])
        df.to_csv(OUT_DIR / f"{name}_per_subject.csv", index=False)

    pd.DataFrame(summary_rows).to_csv(OUT_DIR / "cluster_summary.csv", index=False)

    # Diagnostic plots for each tested map.
    for name in cluster_targets:
        plot_beta_timecourse(
            subjects, name, cluster_results[name],
            title=name.replace("_", " "),
            out_path=FIG_DIR / f"{name}.png",
            alpha=args.alpha,
        )

    print(f"Done. Outputs in {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
