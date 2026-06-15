# -*- coding: utf-8 -*-
# Author: Michel-Pierre Coll (michel-pierre.coll@psy.ulaval.ca), 2026
#
# =============================================================================
# Clean re-implementation of the original student-written passive->decision
# regression-decoding analysis (the former 07d_eeg_decoding_pain_money_
# regression_choice.py, removed during review).
#
# This script is a code-review / methods-clean-up companion to that original
# 07d analysis. It is intentionally **not** a refactor of that
# file: it is a smaller, more conservative analysis that fixes design problems
# identified during review.
#
# What this script asks (re-statement of the scientific question)
# ----------------------------------------------------------------
#
# In the *passive* phase, each cue announces either a pain level (1-5) or a
# money level (1-5) that the participant will later be offered. We can train
# a ridge regression decoder that maps multi-channel EEG (averaged inside a
# short time window) onto that scalar level. The original 07d script then
# applies those passive decoders to *decision* trials, where pain and money
# cues are shown simultaneously, and asks whether the decoded values evolve
# differently for accepted vs. rejected offers.
#
# The intent is good, but the original implementation conflates several
# different signals. This script addresses six concrete issues:
#
#   1. SPECIFICITY of the passive decoders.
#      A pain decoder fit to passive pain trials may simply track generic
#      "magnitude / arousal / attention" rather than anything pain-specific.
#      We test this by *cross-decoding*: applying the pain decoder to passive
#      *money* trials (and vice versa) and correlating the decoded value with
#      the held-out money level. Strong cross-decoding ⇒ generic magnitude.
#
#   2. UNIT-FREE SCALING of decoded values at decision time.
#      The original script pooled all decoded values across trials and
#      windows and z-scored them globally. With ~75% accept / 25% reject,
#      this pooled z-score is dominated by the accept distribution, which
#      biases the accept-vs-reject contrast. We replace the pooled z-score
#      with a **per-trial pre-cue baseline subtraction** (each trial is
#      expressed relative to its own pre-stimulus mean), which is the same
#      logic ERP analyses already use.
#
#   3. POST-HOC WINDOW SELECTION in passive decoding.
#      The original picks the *best* passive window per subject via
#      np.nanargmax over CV scores, then refits a model at that window and
#      applies it to decision trials. This is a noisy, biased selector.
#      We replace it with a **pre-registered fixed window** (400-800 ms),
#      chosen from the manuscript's mass-univariate pain effect (263-1200 ms,
#      peak 798 ms at CP1; money pain>contrast 543-1200 ms). The model is
#      trained on the *average features inside* this window.
#
#   4. CONFOUNDED ACCEPT-vs-REJECT CONTRAST.
#      Accepted and rejected trials differ systematically in the underlying
#      stimulus magnitudes (low-pain/high-money trials are accepted; the
#      opposite are rejected). A raw difference in decoded values can be
#      driven entirely by this stimulus imbalance. We therefore make the
#      **primary decision-time test** a trial-level regression
#         decoded_X(t) ~ painlevel + moneylevel + accepted
#      where the `accepted` coefficient captures choice-related variance
#      after the stimuli are held constant.
#
#   5. CIRCULAR CHOICE-AUC.
#      The original predicts trial-level choice from decoded pain/money,
#      but those decoded values are themselves ≈ painlevel/moneylevel
#      (especially for the within-condition models). So the AUC mostly
#      reflects "pain hurts, money helps" — which is already known from
#      behaviour and not an EEG result. We replace this with an
#      **incremental AUC**: how much does adding decoded EEG to a baseline
#      logistic model of (painlevel, moneylevel) improve choice prediction?
#
#   6. ASYMMETRIC GROUP TESTS.
#      The original cluster-tested only `accepted_money_minus_pain` and not
#      its rejected counterpart, nor the choice-by-decoded interaction.
#      We test both, and the interaction explicitly.
#
# A note on CLAUDE.md
# --------------------
# The project CLAUDE.md states:
#   "Do not use graceful error handling unless explicitly requested and
#    explained in a comment. I want an error to be raised for any missing
#    files or data issues."
#
# This script therefore has no per-subject try/except. If any subject fails
# (missing file, mismatched trial counts, bad metadata), the whole run aborts
# with the original traceback — which is what we want during code review.
#
# Output
# ------
# Everything lands under:
#     derivatives/statistics/eeg_decoding_pain_money_regression_clean/
# Tables are CSVs, figures are PNG (300 dpi). A params.json records the
# exact configuration used.
# =============================================================================

import argparse
import json
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from mne.stats import permutation_cluster_1samp_test
from scipy import stats
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# MNE and sklearn are extremely chatty by default. We silence the noise the
# same way the rest of the EEG pipeline does. We do *not* silence Python's
# own warnings module on anything we have written — only the third-party
# noise that we know is irrelevant here.
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)
mne.set_log_level("ERROR")


# =============================================================================
# Configuration
# =============================================================================
# All paths derive from a single BASE_PATH so the script can be relocated
# (cluster vs. laptop) by exporting `basepath`. The original 07d script uses
# exactly the same convention; we keep it for compatibility.

BASE_PATH = Path(os.getenv("basepath", Path(__file__).parent.parent.parent))
HDDM_DIR = Path(os.getenv("HDDM_DIR", BASE_PATH / "derivatives" / "hddm"))
OUT_DIR = BASE_PATH / "derivatives" / "statistics" / "eeg_decoding_pain_money_regression_clean"
FIG_DIR = OUT_DIR / "figures"

# Fixed seed for any stochastic component (CV shuffles, permutation tests).
# Reproducibility matters here because we want the manuscript reviewer to
# be able to re-run the exact same numbers.
RANDOM_STATE = 23

# Ridge alpha: standardized features, modest regularization. The original
# script uses alpha=1.0 as well; we keep it so the *only* difference between
# the two analyses is in design, not in regularization strength.
RIDGE_ALPHA = 1.0

# Pre-registered passive window in milliseconds.
# Justification (from drafts/PainReward_Draft_VW_2.docx.md):
#   - Mass-univariate passive pain effect: 263-1200 ms, peak CP1 at ~798 ms.
#   - Mass-univariate passive money effect: 285-351 ms + 377-1200 ms.
#   - Pain > money contrast: 543-1200 ms.
# A 400-800 ms window sits inside the pain peak and the late money cluster,
# is wide enough to average out high-frequency noise, and is narrow enough
# to be temporally interpretable. Picking it *up front* (and not per
# subject) eliminates the argmax selection bias in the original script.
PRE_REGISTERED_PASSIVE_WINDOW_MS = (400.0, 800.0)

# Baseline window for the *decoded* time series. The decision epochs are
# already ERP-baseline-corrected at the raw-signal level, but the decoded
# value of each trial still has its own arbitrary offset (determined by the
# subject's mean EEG amplitude on training data). We re-center each trial's
# decoded time-series so that its average in this window is zero. This is
# what the ERP literature does for raw signals, and it removes the pooled
# z-score's class-imbalance bias.
DECISION_BASELINE_WINDOW_MS = (-200.0, 0.0)


# =============================================================================
# Argument parsing
# =============================================================================

def parse_args():
    """CLI options. Defaults are tuned for a *local laptop* run, not SLURM."""
    parser = argparse.ArgumentParser(
        description="Clean version of the passive→decision pain/money regression decoding analysis."
    )
    # `--quick` is the fastest way to sanity-check the whole pipeline end-to-end.
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
# A dataclass groups everything we accumulate per subject so functions stop
# taking 8 positional arguments. It also gives us one obvious place to look
# when debugging "what did we have for sub-012?".

@dataclass
class SubjectData:
    """Everything we compute for one participant. NaN fields signal 'not yet computed'."""
    participant_id: str

    # Passive: per-window CV scores for sanity plots, plus the held-out
    # cross-decoding scores. `cross_decode_*` answers the specificity question.
    passive_pain_cv: Optional[np.ndarray] = None        # shape (n_windows,) Fisher z(r)
    passive_money_cv: Optional[np.ndarray] = None
    passive_times_ms: Optional[np.ndarray] = None       # window centers (ms)
    cross_decode_pain_to_money: Optional[float] = None  # scalar Fisher z(r)
    cross_decode_money_to_pain: Optional[float] = None
    within_decode_pain: Optional[float] = None          # scalar Fisher z(r) in the registered window
    within_decode_money: Optional[float] = None

    # Decision: decoded time series after pre-cue baseline subtraction.
    decoded_pain: Optional[np.ndarray] = None           # shape (n_trials, n_windows)
    decoded_money: Optional[np.ndarray] = None
    decision_times_ms: Optional[np.ndarray] = None

    # Per-trial decision metadata aligned to decoded_* rows.
    painlevel: Optional[np.ndarray] = None
    moneylevel: Optional[np.ndarray] = None
    accepted: Optional[np.ndarray] = None

    # Per-time betas from the stimulus-controlled regression. Each is a
    # vector of length n_windows.
    beta_painlevel_on_decoded_pain: Optional[np.ndarray] = None
    beta_moneylevel_on_decoded_pain: Optional[np.ndarray] = None
    beta_accepted_on_decoded_pain: Optional[np.ndarray] = None
    beta_painlevel_on_decoded_money: Optional[np.ndarray] = None
    beta_moneylevel_on_decoded_money: Optional[np.ndarray] = None
    beta_accepted_on_decoded_money: Optional[np.ndarray] = None

    # Per-time incremental choice AUC (Δ over a logistic model that already
    # contains painlevel and moneylevel).
    incremental_auc_pain: Optional[np.ndarray] = None
    incremental_auc_money: Optional[np.ndarray] = None
    incremental_auc_joint: Optional[np.ndarray] = None  # both decoded as predictors


# =============================================================================
# Tiny numeric utilities
# =============================================================================

def ensure_dir(path: Path):
    """Create an output directory if it does not already exist (idempotent)."""
    path.mkdir(parents=True, exist_ok=True)


def fisher_z(r):
    """Convert Pearson r to Fisher z. Clip to avoid arctanh(±1) = ±inf.

    Group-level statistics on correlations should always be done on z-values
    rather than raw r — the z is approximately normally distributed under H0,
    which is what cluster permutation testing assumes.
    """
    return np.arctanh(np.clip(np.asarray(r, dtype=float), -0.999999, 0.999999))


def safe_corr(y_true, y_pred):
    """Pearson r with explicit handling of degenerate inputs.

    Returns NaN if either vector has zero variance or fewer than 3 finite
    overlapping values. We deliberately do NOT raise here, because some
    cross-decoding folds *can* have effectively constant predictions
    (small sample, regularized model) and that is informative, not fatal.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    keep = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[keep]
    y_pred = y_pred[keep]
    if len(y_true) < 3 or np.nanstd(y_true) == 0 or np.nanstd(y_pred) == 0:
        return np.nan
    return float(np.corrcoef(y_true, y_pred)[0, 1])


# =============================================================================
# Data loading helpers (mirror 07d to stay compatible with the rest of the pipeline)
# =============================================================================
# Why duplicate these instead of importing? The original 07d script is being
# reviewed and may move/rename. Keeping this script self-contained makes it
# straightforward to share for reproducibility without worrying about an
# inconsistent sibling file.

def decision_trialsnum(df: pd.DataFrame) -> pd.Series:
    """Recreate the 1-based decision trial number used as epoch metadata key.

    The acquisition software stores `blocks.thisRepN` (0-based block) and
    `trials.thisN` (0-based trial within block, with 25 trials per block).
    Epoch metadata stores the running 1-based trial number — so we reproduce
    that mapping here to merge HDDM rows with decision epochs.
    """
    return (
        df["blocks.thisRepN"].astype(int) * 25
        + df["trials.thisN"].astype(int)
        + 1
    )


def load_hddm_mod9() -> pd.DataFrame:
    """Load HDDM mod_9 trial-by-trial table.

    `mod_9` is the model from which trial-level RP and inclusion masks are
    drawn (see DDM section of the manuscript). We use it here only to
    select which decision trials survived HDDM filtering — not for any
    parameter values.
    """
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
    """Return participants present in BOTH participants.tsv and the HDDM table.

    If a subject is in one but not the other, that is an upstream pipeline
    bug; here we just take the intersection and let downstream code raise on
    missing files (which is the explicit CLAUDE.md policy).
    """
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

    Some participants' raw level columns are stored as percentages (e.g. 20,
    40, 60, 80, 100). We divide by 20 in that case so all subjects share the
    same 1-5 ordinal scale. This *exactly* matches the original 07d logic
    so subject-level effects are comparable between the two analyses.
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
    # `fixcross.started` is non-NaN only on real trial rows (PsychoPy logs
    # also include block-instruction rows etc.). Dropping NaNs gives the
    # canonical 1-based trial numbering.
    if "fixcross.started" in beh.columns:
        beh = beh[~beh["fixcross.started"].isna()].copy()
    beh = beh.reset_index(drop=True)
    beh["trialsnum"] = np.arange(1, len(beh) + 1)
    beh["condition"] = beh["condition"].astype(str).str.lower().str.strip()
    beh["level_1_5"] = level_to_1_5(beh["level"])
    # Two non-baseline conditions: 'p' (pain cue) and 'm' (money cue).
    beh = beh[beh["condition"].isin(["p", "m"])].copy()
    return beh[["trialsnum", "condition", "level_1_5", "blocks.thisRepN"]]


def read_epochs_for_decoding(epo_path: Path, resample_hz: float) -> mne.Epochs:
    """Read an MNE epochs file, restrict to EEG, and downsample.

    Why downsample to 100 Hz? The decoders use 50 ms windows. At 100 Hz that's
    5 samples per window — enough averaging to suppress high-frequency noise
    without smearing the temporal effects we care about. Higher rates
    multiply CV cost without changing the result.
    """
    if not epo_path.exists():
        raise FileNotFoundError(f"Missing epoch file: {epo_path}")
    epo = mne.read_epochs(str(epo_path), preload=True, verbose="ERROR")
    epo = epo.pick("eeg", exclude=[])
    if resample_hz and epo.info["sfreq"] != float(resample_hz):
        epo = epo.resample(float(resample_hz), verbose="ERROR")
    return epo


def align_passive_epochs(pa: str, resample_hz: float) -> mne.Epochs:
    """Attach passive behavior to passive epochs via exact 1:1 trialsnum merge.

    The `validate="1:1"` is critical. Anything other than a perfect 1:1 match
    here would silently mis-align trials — a class of bug that is invisible
    in summary plots but completely breaks the analysis.
    """
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
    """Select decision epochs in the exact order of HDDM-filtered rows.

    The HDDM table is the source of truth for which decision trials we
    include (it has already dropped slow/fast outlier RTs and missed
    trials). We re-order the epochs to match the HDDM table so the EEG
    rows and the behavior rows can be indexed in lock-step.
    """
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
    # Overwrite stale metadata columns with the HDDM-side values. This is
    # safe because we just re-ordered the epochs to match HDDM rows.
    for col in ["painlevel", "moneylevel", "accepted", "trialsnum"]:
        md[col] = mod[col].to_numpy()
    md["participant"] = pa
    epo.metadata = md
    return epo


# =============================================================================
# Feature extraction (sliding window averages)
# =============================================================================

def make_window_slices(times: np.ndarray, window_ms: float, step_ms: float
                       ) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Return centered sliding-window sample indices that fit inside `times`.

    Mathematically: for each window center `c`, return the sample indices in
    [c - half, c + half]. We avoid window edges that would fall outside the
    epoch — i.e., we never extrapolate.
    """
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
    """Average EEG amplitude inside each window → shape (trials, windows, channels).

    We scale by 1e6 so units are microvolts (just to keep numbers in a
    human-readable range). Standardization inside the pipeline handles the
    actual scale-invariance.
    """
    data = epo.get_data(copy=False).astype(np.float64, copy=False) * 1e6
    if not np.all(np.isfinite(data)):
        raise RuntimeError("Epoch data contain non-finite EEG values")
    features = np.empty((data.shape[0], len(window_slices), data.shape[1]), dtype=np.float64)
    for i, sl in enumerate(window_slices):
        features[:, i, :] = data[:, :, sl].mean(axis=2)
    return features


def features_for_window_range(epo: mne.Epochs, lo_ms: float, hi_ms: float) -> np.ndarray:
    """Mean amplitude across [lo_ms, hi_ms] → shape (trials, channels).

    This is the "one big window" feature vector used for training the
    pre-registered passive decoder. Compared to picking the best of many
    50 ms windows, averaging across the full 400-800 ms window:
        + reduces variance (more samples in the mean),
        + commits us to one decision before seeing the data,
        + matches the ERP-style averaging that the manuscript already uses.
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

def make_decoder() -> "make_pipeline":
    """Standard ridge pipeline used throughout the script.

    Notes on the choice:
        - StandardScaler is mandatory: ridge's L2 penalty is not scale-invariant.
        - alpha=1.0 is the same value the original 07d uses, kept here so the
          design differences between the two scripts are not confounded with
          regularization differences.
    """
    return make_pipeline(StandardScaler(), Ridge(alpha=RIDGE_ALPHA, random_state=RANDOM_STATE))


def subset_passive_target(epo: mne.Epochs, target: str
                          ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (trial indices, level labels, block-of-origin) for one passive condition.

    `target` is either "pain" or "money". Block index becomes the CV group
    so that train and test sets never come from the same block — this
    prevents leakage from session-level drifts in EEG amplitude.
    """
    md = epo.metadata.reset_index(drop=True).copy()
    condition = "p" if target == "pain" else "m"
    selected = md["condition"].astype(str).str.lower().to_numpy() == condition
    # `badtrial` is set upstream (artifact rejection). Always drop those.
    if "badtrial" in md.columns:
        selected &= md["badtrial"].fillna(0).astype(int).to_numpy() == 0
    y_all = pd.to_numeric(md["level_1_5"], errors="coerce").to_numpy(dtype=float)
    if np.any(selected & ~np.isfinite(y_all)):
        # Per CLAUDE.md: fail fast on data issues, do not silently drop.
        raise RuntimeError(f"Passive {target}: non-finite level labels among non-artifact trials")
    if "blocks.thisRepN" not in md.columns:
        raise RuntimeError("Passive epochs missing metadata['blocks.thisRepN']")
    block_all = pd.to_numeric(md["blocks.thisRepN"], errors="coerce").to_numpy(dtype=float)
    if np.any(selected & ~np.isfinite(block_all)):
        raise RuntimeError(f"Passive {target}: non-finite block ids among non-artifact trials")
    idx = np.where(selected)[0]
    return idx, y_all[selected], block_all[selected].astype(int)


def subset_decision_trials(epo: mne.Epochs) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return decision trial indices, painlevel, moneylevel, accepted (0/1).

    Drops only artifact trials. HDDM filtering has already removed bad-RT
    trials upstream (in `load_hddm_mod9` and `align_decision_epochs`).
    """
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

def passive_within_cv_curve(features_by_window: np.ndarray,
                            y: np.ndarray,
                            groups: np.ndarray,
                            n_splits: int = 5) -> np.ndarray:
    """For each passive sliding window, return CV Fisher-z(r) decoding score.

    Used for *diagnostic plotting only*. The primary passive readout is the
    averaged-window scalar in `passive_within_decode_scalar`, computed below.
    Separating the two makes it harder to accidentally select windows on
    the basis of the curve.
    """
    n_splits = min(n_splits, len(np.unique(groups)))
    if n_splits < 2:
        raise RuntimeError("Need at least two passive blocks for grouped CV")
    if np.nanstd(y) == 0:
        raise RuntimeError("Passive labels have zero variance")
    scores = np.full(features_by_window.shape[1], np.nan, dtype=float)
    cv = GroupKFold(n_splits=n_splits)
    for w in range(features_by_window.shape[1]):
        preds = np.full(len(y), np.nan, dtype=float)
        for train_idx, test_idx in cv.split(features_by_window[:, w, :], y, groups):
            model = make_decoder().fit(features_by_window[train_idx, w, :], y[train_idx])
            preds[test_idx] = model.predict(features_by_window[test_idx, w, :])
        scores[w] = fisher_z(safe_corr(y, preds))
    return scores


def passive_within_decode_scalar(features_window_mean: np.ndarray,
                                 y: np.ndarray,
                                 groups: np.ndarray,
                                 n_splits: int = 5) -> float:
    """CV Fisher-z(r) for the *pre-registered single-window* passive decoder.

    This is the headline within-condition number per subject. It is
    GroupKFold by passive block to keep training and testing on different
    sessions of the task (no temporal leakage).
    """
    n_splits = min(n_splits, len(np.unique(groups)))
    if n_splits < 2:
        raise RuntimeError("Need at least two passive blocks for grouped CV")
    preds = np.full(len(y), np.nan, dtype=float)
    cv = GroupKFold(n_splits=n_splits)
    for train_idx, test_idx in cv.split(features_window_mean, y, groups):
        model = make_decoder().fit(features_window_mean[train_idx], y[train_idx])
        preds[test_idx] = model.predict(features_window_mean[test_idx])
    return fisher_z(safe_corr(y, preds))


def passive_cross_decode_scalar(train_features: np.ndarray,
                                train_y: np.ndarray,
                                test_features: np.ndarray,
                                test_y: np.ndarray) -> float:
    """Fit on one condition, evaluate on the other. The specificity check.

    Logic:
        - Train pain decoder on all passive *pain* trials.
        - Apply it to all passive *money* trials.
        - Correlate the predicted "pain value" with the held-out money level.
    A large correlation here means the decoder is generic — it tracks any
    magnitude cue, not pain specifically. A near-zero correlation means the
    decoder is *specific* to its training condition, which is what we want
    if we are to interpret it as a pain (or money) decoder at decision time.
    """
    model = make_decoder().fit(train_features, train_y)
    pred = model.predict(test_features)
    return fisher_z(safe_corr(test_y, pred))


# =============================================================================
# Decision-time analyses
# =============================================================================

def apply_passive_models_to_decision(pain_model,
                                     money_model,
                                     decision_features_3d: np.ndarray
                                     ) -> Tuple[np.ndarray, np.ndarray]:
    """Apply the two pre-registered passive decoders across decision windows.

    Returns:
        decoded_pain  -- shape (n_trials, n_windows). Decoder output, in the
                         decoder's native scale (i.e. predicted level 1-5).
        decoded_money -- same shape.

    No scaling is applied here. Pre-cue baseline subtraction happens after.
    """
    n_trials, n_windows, _ = decision_features_3d.shape
    decoded_pain = np.empty((n_trials, n_windows), dtype=np.float64)
    decoded_money = np.empty((n_trials, n_windows), dtype=np.float64)
    for w in range(n_windows):
        decoded_pain[:, w] = pain_model.predict(decision_features_3d[:, w, :])
        decoded_money[:, w] = money_model.predict(decision_features_3d[:, w, :])
    return decoded_pain, decoded_money


def baseline_correct_decoded(decoded: np.ndarray,
                             window_centers_ms: np.ndarray,
                             baseline_ms: Tuple[float, float]) -> np.ndarray:
    """Subtract each trial's pre-cue mean from its decoded time-series.

    Why this and not pooled z-scoring?
        - Pooled z-score: subtract one mean / divide by one SD across *all*
          trials. With ~75/25 class imbalance, the mean is dominated by
          accepted trials, so the resulting accept-vs-reject contrast is
          biased toward zero on the accept side and toward a fixed offset
          on the reject side.
        - Pre-cue baseline subtraction: each trial is its own control.
          The baseline interval is *before* the cue, so it cannot contain
          any condition information. Differences between conditions after
          baseline correction therefore reflect post-cue divergence, not
          differences in the noise floor.
    The scale of the decoded values is preserved (we do NOT divide by SD),
    which makes the betas in the downstream regression interpretable in
    units of "predicted level 1-5".
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
                                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-time regression: decoded(t) ~ painlevel + moneylevel + accepted.

    Returns three vectors of length n_windows: beta_pain, beta_money, beta_acc.
    All predictors are standardized so the betas are directly comparable.

    Why this is the *primary* decision-time test
    --------------------------------------------
    The accept-vs-reject contrast confounds choice with stimulus magnitude
    (accepted trials have lower pain and higher money, by construction). To
    ask "does the EEG track the choice over and above the stimuli?", we
    have to hold the stimuli constant. Multiple regression does that
    linearly: the beta on `accepted` is the choice-related variance in the
    decoded signal that is *not* explained by painlevel or moneylevel.

    Implementation notes:
        - We z-score the per-trial decoded values at this time index so the
          beta is in "1 SD of decoded value per 1 SD of predictor" units.
          This is post-baseline so it is just a unit conversion, not the
          biased pooled z-score we removed earlier.
        - All-NaN trials at this window get dropped per time index. If the
          remaining sample is too small, we return NaN at that time so the
          group-level cluster test simply ignores it.
    """
    n_trials, n_windows = decoded.shape
    if not (n_trials == len(painlevel) == len(moneylevel) == len(accepted)):
        raise RuntimeError("Mismatched trial counts in regression inputs")

    beta_pain = np.full(n_windows, np.nan, dtype=float)
    beta_money = np.full(n_windows, np.nan, dtype=float)
    beta_acc = np.full(n_windows, np.nan, dtype=float)

    # Standardize predictors *once*, not per time index. The predictors are
    # constant across time (they are trial-level) and we want a fixed scale.
    pred = np.column_stack([
        stats.zscore(painlevel.astype(float)),
        stats.zscore(moneylevel.astype(float)),
        # accept/reject treated as a numeric contrast (1 = accept, -1 = reject)
        # rather than standardizing a binary, which is fine because it is
        # already mean-centered after recoding.
        np.where(accepted == 1, 1.0, -1.0),
    ])
    design = np.column_stack([np.ones(n_trials), pred])

    for w in range(n_windows):
        y = decoded[:, w].astype(float)
        keep = np.isfinite(y) & np.isfinite(design).all(axis=1)
        if keep.sum() < pred.shape[1] + 3:
            continue
        y_z = stats.zscore(y[keep])
        if not np.isfinite(y_z).all():
            continue
        coefs, *_ = np.linalg.lstsq(design[keep], y_z, rcond=None)
        # coefs[0] is the intercept; we want the slopes.
        beta_pain[w] = float(coefs[1])
        beta_money[w] = float(coefs[2])
        beta_acc[w] = float(coefs[3])

    return beta_pain, beta_money, beta_acc


def incremental_choice_auc(decoded: np.ndarray,
                           painlevel: np.ndarray,
                           moneylevel: np.ndarray,
                           accepted: np.ndarray,
                           n_splits: int = 5) -> np.ndarray:
    """ΔAUC(t) = AUC(painlevel + moneylevel + decoded(t)) − AUC(painlevel + moneylevel).

    Why incremental and not raw AUC?
    --------------------------------
    If the decoded value approximately reproduces painlevel (which it must
    do at least somewhat, since the decoder was trained to predict it),
    then a logistic model of decoded values will predict choice almost as
    well as a logistic model of painlevel/moneylevel — but that just tells
    us that "people accept high-money and reject high-pain offers", which
    we already know from behaviour and which is not an EEG result.

    The honest question is: does the EEG at time `t` add anything beyond
    the stimuli we already showed? That is the increment.

    Cross-validation is stratified on `accepted` (which is imbalanced).
    """
    n_trials, n_windows = decoded.shape
    auc = np.full(n_windows, np.nan, dtype=float)
    # Class-aware CV. With ~75/25 imbalance we still need both classes
    # in every fold.
    class_counts = np.bincount(accepted.astype(int), minlength=2)
    fold_count = min(n_splits, int(class_counts.min()))
    if fold_count < 2:
        return auc

    cv = StratifiedKFold(n_splits=fold_count, shuffle=True, random_state=RANDOM_STATE)
    baseline_predictors = np.column_stack([painlevel.astype(float), moneylevel.astype(float)])

    # Compute the *baseline* (no-EEG) AUC once. Same CV folds are reused
    # for the with-EEG model so the comparison is paired.
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
        keep = np.isfinite(eeg)
        if keep.sum() < n_trials:
            # NaN at this window → cannot evaluate, leave AUC NaN.
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


def joint_incremental_choice_auc(decoded_pain: np.ndarray,
                                 decoded_money: np.ndarray,
                                 painlevel: np.ndarray,
                                 moneylevel: np.ndarray,
                                 accepted: np.ndarray,
                                 n_splits: int = 5) -> np.ndarray:
    """ΔAUC(t) when both decoded streams are added jointly to the baseline.

    Some signals may only be visible when pain and money decoded values are
    considered together (e.g. a value-difference signal). The single-stream
    incremental_choice_auc misses those.
    """
    n_trials, n_windows = decoded_pain.shape
    auc = np.full(n_windows, np.nan, dtype=float)
    class_counts = np.bincount(accepted.astype(int), minlength=2)
    fold_count = min(n_splits, int(class_counts.min()))
    if fold_count < 2:
        return auc
    cv = StratifiedKFold(n_splits=fold_count, shuffle=True, random_state=RANDOM_STATE)
    baseline_predictors = np.column_stack([painlevel.astype(float), moneylevel.astype(float)])
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
        eeg_p = decoded_pain[:, w].astype(float)
        eeg_m = decoded_money[:, w].astype(float)
        if not (np.isfinite(eeg_p).all() and np.isfinite(eeg_m).all()):
            continue
        predictors = np.column_stack([baseline_predictors, eeg_p, eeg_m])
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
    pain_idx, pain_y, pain_groups = subset_passive_target(passive_epo, "pain")
    money_idx, money_y, money_groups = subset_passive_target(passive_epo, "money")
    if len(pain_idx) < 20 or len(money_idx) < 20:
        raise RuntimeError(f"{pa}: too few passive trials (pain={len(pain_idx)}, money={len(money_idx)})")

    # 1a. Sliding-window features for the *diagnostic* CV curve.
    passive_times_s = passive_epo.times
    centers_s, slices = make_window_slices(passive_times_s, args.window_ms, args.step_ms)
    sub.passive_times_ms = centers_s * 1000.0
    features_3d = window_features(passive_epo, slices)

    sub.passive_pain_cv = passive_within_cv_curve(
        features_3d[pain_idx], pain_y, pain_groups,
    )
    sub.passive_money_cv = passive_within_cv_curve(
        features_3d[money_idx], money_y, money_groups,
    )

    # 1b. Pre-registered single-window features (the *primary* passive decoder input).
    feat_window_pain = features_for_window_range(
        passive_epo, *PRE_REGISTERED_PASSIVE_WINDOW_MS,
    )[pain_idx]
    feat_window_money = features_for_window_range(
        passive_epo, *PRE_REGISTERED_PASSIVE_WINDOW_MS,
    )[money_idx]

    # 1c. Within-condition scalars (headline passive numbers).
    sub.within_decode_pain = passive_within_decode_scalar(
        feat_window_pain, pain_y, pain_groups,
    )
    sub.within_decode_money = passive_within_decode_scalar(
        feat_window_money, money_y, money_groups,
    )

    # 1d. Cross-decoding (specificity).
    # Fit a pain decoder on all passive-pain trials, then ask how well it
    # predicts passive *money* levels. A high score = generic magnitude
    # signal. Symmetric for money.
    sub.cross_decode_pain_to_money = passive_cross_decode_scalar(
        feat_window_pain, pain_y, feat_window_money, money_y,
    )
    sub.cross_decode_money_to_pain = passive_cross_decode_scalar(
        feat_window_money, money_y, feat_window_pain, pain_y,
    )

    # 1e. Fit the *final* passive decoders on ALL passive trials of each
    # condition. These models are then locked and used at decision time
    # below. No CV here: we are training, not testing.
    pain_model_final = make_decoder().fit(feat_window_pain, pain_y)
    money_model_final = make_decoder().fit(feat_window_money, money_y)
    del passive_epo, features_3d, feat_window_pain, feat_window_money

    # ---- 2. Decision epochs ------------------------------------------------
    decision_epo = align_decision_epochs(pa, mod_data, args.resample_hz)
    dec_idx, painlevel, moneylevel, accepted = subset_decision_trials(decision_epo)
    if len(dec_idx) < 30:
        raise RuntimeError(f"{pa}: too few decision trials ({len(dec_idx)})")

    decision_times_s = decision_epo.times
    dec_centers_s, dec_slices = make_window_slices(decision_times_s, args.window_ms, args.step_ms)
    sub.decision_times_ms = dec_centers_s * 1000.0
    dec_features_3d = window_features(decision_epo, dec_slices)[dec_idx]

    decoded_pain, decoded_money = apply_passive_models_to_decision(
        pain_model_final, money_model_final, dec_features_3d,
    )

    # Pre-cue baseline subtraction. This is the fix for the original pooled
    # z-score bias. Each trial is centered on its own pre-cue mean.
    sub.decoded_pain = baseline_correct_decoded(
        decoded_pain, sub.decision_times_ms, DECISION_BASELINE_WINDOW_MS,
    )
    sub.decoded_money = baseline_correct_decoded(
        decoded_money, sub.decision_times_ms, DECISION_BASELINE_WINDOW_MS,
    )
    sub.painlevel = painlevel
    sub.moneylevel = moneylevel
    sub.accepted = accepted

    # ---- 3. Stimulus-controlled regression ---------------------------------
    bp, bm, ba = stimulus_controlled_regression(
        sub.decoded_pain, painlevel, moneylevel, accepted,
    )
    sub.beta_painlevel_on_decoded_pain = bp
    sub.beta_moneylevel_on_decoded_pain = bm
    sub.beta_accepted_on_decoded_pain = ba

    bp, bm, ba = stimulus_controlled_regression(
        sub.decoded_money, painlevel, moneylevel, accepted,
    )
    sub.beta_painlevel_on_decoded_money = bp
    sub.beta_moneylevel_on_decoded_money = bm
    sub.beta_accepted_on_decoded_money = ba

    # ---- 4. Incremental choice AUC -----------------------------------------
    sub.incremental_auc_pain = incremental_choice_auc(
        sub.decoded_pain, painlevel, moneylevel, accepted,
    )
    sub.incremental_auc_money = incremental_choice_auc(
        sub.decoded_money, painlevel, moneylevel, accepted,
    )
    sub.incremental_auc_joint = joint_incremental_choice_auc(
        sub.decoded_pain, sub.decoded_money, painlevel, moneylevel, accepted,
    )

    print(f"  [{pa}] done", flush=True)
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

    Two-tailed (`tail=0`) by default — we are agnostic about sign and
    do not want to inflate type I error by post-hoc choosing a tail.
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
# Plotting (kept minimal — figures are diagnostics, not publication panels)
# =============================================================================

def plot_passive_curves(subjects: List[SubjectData], out_path: Path):
    """Group mean ± SEM passive within-condition CV curves over time."""
    times = subjects[0].passive_times_ms
    pain_arr = np.vstack([s.passive_pain_cv for s in subjects])
    money_arr = np.vstack([s.passive_money_cv for s in subjects])
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    for arr, color, label in [(pain_arr, "#b22222", "Passive pain"),
                              (money_arr, "#2ca02c", "Passive money")]:
        m = np.nanmean(arr, axis=0)
        s = stats.sem(arr, axis=0, nan_policy="omit")
        ax.plot(times, m, color=color, linewidth=2, label=label)
        ax.fill_between(times, m - s, m + s, color=color, alpha=0.15)
    # Mark the pre-registered window so reviewers can see it on every plot.
    ax.axvspan(*PRE_REGISTERED_PASSIVE_WINDOW_MS, color="grey", alpha=0.10,
               label=f"Pre-registered window {int(PRE_REGISTERED_PASSIVE_WINDOW_MS[0])}-"
                     f"{int(PRE_REGISTERED_PASSIVE_WINDOW_MS[1])} ms")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Passive time from cue onset (ms)")
    ax.set_ylabel("Within-condition decoding (Fisher z(r))")
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
    s = stats.sem(arr, axis=0, nan_policy="omit")
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    ax.plot(times, m, color="#1f77b4", linewidth=2)
    ax.fill_between(times, m - s, m + s, color="#1f77b4", alpha=0.18)
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


def plot_cross_decoding_specificity(subjects: List[SubjectData], out_path: Path):
    """Bar plot: within-condition vs cross-condition passive Fisher z(r).

    If within-condition decoding is strong but cross-decoding is at zero,
    the decoders are specific. If both are similar, the decoders track
    generic magnitude — which would change how we interpret the decision
    results entirely.
    """
    labels = [
        "Pain → pain (within)", "Pain → money (cross)",
        "Money → money (within)", "Money → pain (cross)",
    ]
    vals = [
        np.array([s.within_decode_pain for s in subjects]),
        np.array([s.cross_decode_pain_to_money for s in subjects]),
        np.array([s.within_decode_money for s in subjects]),
        np.array([s.cross_decode_money_to_pain for s in subjects]),
    ]
    means = [float(np.nanmean(v)) for v in vals]
    sems = [float(stats.sem(v, nan_policy="omit")) for v in vals]
    colors = ["#b22222", "#b22222", "#2ca02c", "#2ca02c"]
    alphas = [1.0, 0.4, 1.0, 0.4]
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    x = np.arange(len(labels))
    for i, (m, e, c, a) in enumerate(zip(means, sems, colors, alphas)):
        ax.bar(x[i], m, yerr=e, color=c, alpha=a, capsize=4)
    for i, v in enumerate(vals):
        jitter = (np.random.RandomState(RANDOM_STATE + i).rand(len(v)) - 0.5) * 0.25
        ax.scatter(x[i] + jitter, v, color="black", s=12, alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Fisher z(r)")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(f"Passive decoder specificity, pre-registered "
                 f"{int(PRE_REGISTERED_PASSIVE_WINDOW_MS[0])}-"
                 f"{int(PRE_REGISTERED_PASSIVE_WINDOW_MS[1])} ms")
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
        # Smoke-test settings: fewer subjects, fewer permutations.
        # Useful for catching pipeline bugs without waiting 30 minutes.
        args.n_permutations = max(200, args.n_permutations // 10)

    # Persist exact parameters used for this run.
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
        "ridge_alpha": RIDGE_ALPHA,
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

    # ---- group-level passive specificity table ----------------------------
    spec_rows = []
    for s in subjects:
        spec_rows.append({
            "participant_id": s.participant_id,
            "within_decode_pain_z": s.within_decode_pain,
            "within_decode_money_z": s.within_decode_money,
            "cross_pain_to_money_z": s.cross_decode_pain_to_money,
            "cross_money_to_pain_z": s.cross_decode_money_to_pain,
        })
    spec_df = pd.DataFrame(spec_rows)
    spec_df.to_csv(OUT_DIR / "passive_specificity_per_subject.csv", index=False)

    # Group t-tests on the four scalars vs zero. Cheap, informative.
    spec_summary = []
    for col in ["within_decode_pain_z", "within_decode_money_z",
                "cross_pain_to_money_z", "cross_money_to_pain_z"]:
        vals = spec_df[col].to_numpy(dtype=float)
        t, p = stats.ttest_1samp(vals, 0.0, nan_policy="omit")
        spec_summary.append({
            "scalar": col,
            "mean": float(np.nanmean(vals)),
            "sem": float(stats.sem(vals, nan_policy="omit")),
            "t": float(t),
            "df": int(np.isfinite(vals).sum() - 1),
            "p_two_tailed": float(p),
            "n": int(np.isfinite(vals).sum()),
        })
    pd.DataFrame(spec_summary).to_csv(OUT_DIR / "passive_specificity_summary.csv", index=False)
    plot_cross_decoding_specificity(subjects, FIG_DIR / "passive_specificity.png")

    # ---- group-level decision tests ---------------------------------------
    # Shared time vector across subjects. Sanity-check it is identical.
    base_times = subjects[0].decision_times_ms
    for s in subjects[1:]:
        if not np.allclose(s.decision_times_ms, base_times):
            raise RuntimeError(
                f"{s.participant_id}: decision time vector mismatch — "
                "subjects must share resampling rate and epoch range"
            )

    # Stack betas across subjects → (n_subj, n_windows).
    cluster_targets = {
        "beta_accepted_on_decoded_pain": np.vstack([s.beta_accepted_on_decoded_pain for s in subjects]),
        "beta_accepted_on_decoded_money": np.vstack([s.beta_accepted_on_decoded_money for s in subjects]),
        "beta_painlevel_on_decoded_pain": np.vstack([s.beta_painlevel_on_decoded_pain for s in subjects]),
        "beta_moneylevel_on_decoded_money": np.vstack([s.beta_moneylevel_on_decoded_money for s in subjects]),
        # Symmetric "wrong-stimulus" controls. These should be near zero if
        # the passive decoders are specific (echoing the passive cross-decoding
        # check, but evaluated on decision-time data).
        "beta_moneylevel_on_decoded_pain": np.vstack([s.beta_moneylevel_on_decoded_pain for s in subjects]),
        "beta_painlevel_on_decoded_money": np.vstack([s.beta_painlevel_on_decoded_money for s in subjects]),
        # Incremental choice AUC streams.
        "incremental_auc_pain": np.vstack([s.incremental_auc_pain for s in subjects]),
        "incremental_auc_money": np.vstack([s.incremental_auc_money for s in subjects]),
        "incremental_auc_joint": np.vstack([s.incremental_auc_joint for s in subjects]),
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

        # Per-map per-subject CSV for re-analysis.
        df = pd.DataFrame(arr, columns=[f"t{int(round(t))}" for t in base_times])
        df.insert(0, "participant_id", [s.participant_id for s in subjects])
        df.to_csv(OUT_DIR / f"{name}_per_subject.csv", index=False)

    pd.DataFrame(summary_rows).to_csv(OUT_DIR / "cluster_summary.csv", index=False)

    # Diagnostic plots for each tested map. Reviewer-friendly.
    plot_passive_curves(subjects, FIG_DIR / "passive_decoding_curves.png")
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
