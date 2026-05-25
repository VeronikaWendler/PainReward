"""
Shared helpers for EEG decoding scripts (Stage 1).

This module is intentionally small: it owns the cross-cutting pieces (subject
list, epoch loading, LDA pipeline factory, Haufe transform, cluster permutation,
plotting) used by 07a_eeg_decoding_passive.py and, later, 07b_eeg_decoding_cross.py.

Per the project CLAUDE.md, helpers do not silently swallow failures: missing
files, missing metadata columns, or empty contrasts raise.
"""

from __future__ import annotations

import json
import os
import platform
import socket
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import sklearn
from mne.decoding import SlidingEstimator, cross_val_multiscore
from mne.stats import permutation_cluster_1samp_test
from scipy import stats as scipy_stats
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import Ridge
from sklearn.metrics import make_scorer
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)
mne.set_log_level("ERROR")

RANDOM_STATE = 23

BASE_PATH = Path(os.getenv("basepath", Path(__file__).parent.parent.parent))
BEHAV_EXCLUSIONS_CSV = BASE_PATH / "derivatives" / "behav" / "behav_cleaned_with_exclusions.csv"


def load_subject_list(behav_csv: Path = BEHAV_EXCLUSIONS_CSV) -> List[str]:
    """Return the post-exclusion subject list (sub-XXX strings, sorted).

    Drives subject inclusion for all Stage 1 analyses. Reads the cleaned
    behavioural CSV so whichever exclusions the behavioural pipeline applied
    propagate to the decoding analyses without us re-implementing them.
    """
    if not behav_csv.exists():
        raise FileNotFoundError(f"Behavioural exclusions CSV not found: {behav_csv}")
    df = pd.read_csv(behav_csv)
    if "participant" not in df.columns:
        raise ValueError(
            f"{behav_csv} missing 'participant' column. Got: {list(df.columns)[:30]}"
        )
    subs = sorted(df["participant"].dropna().astype(str).unique().tolist())
    if not subs:
        raise ValueError(f"No subjects found in {behav_csv}")
    return subs


PASSIVE_TRIAL_TYPES = [f"rew{i}" for i in range(1, 6)] + [f"shk{i}" for i in range(1, 6)]


def _parse_trial_type(trial_type: str) -> Tuple[str, int]:
    """Map a passive event code to (attribute, level).

    rew{1..5} -> ('money', 1..5);  shk{1..5} -> ('pain', 1..5).
    Raises if the code is not a known passive cue.
    """
    if trial_type not in PASSIVE_TRIAL_TYPES:
        raise ValueError(f"Unknown passive trial_type: {trial_type!r}")
    attribute = "pain" if trial_type.startswith("shk") else "money"
    level = int(trial_type[3:])
    return attribute, level


def load_passive_epochs(
    subject: str,
    base_path: Path = BASE_PATH,
    resample_hz: float = 250.0,
) -> mne.Epochs:
    """Load a subject's passive single-trial epochs.

    Returns epochs with attached metadata DataFrame containing:
      - trial_type (str): event code (rew1..rew5, shk1..shk5)
      - attribute (str): 'pain' or 'money'
      - level (int): 1..5
      - badtrial (int): 0/1 from upstream rejection (kept as-is, not dropped here)

    Epochs are downsampled to `resample_hz` before return.
    """
    fif = (
        base_path / "derivatives" / subject / "eeg" / "erps_passive"
        / f"{subject}_passive_cues_singletrials-epo.fif"
    )
    if not fif.exists():
        raise FileNotFoundError(f"{subject}: passive epoch file missing: {fif}")

    epochs = mne.read_epochs(str(fif), preload=True, verbose="ERROR")

    # Build metadata from epochs.events or event_id. Epoch files saved by 04_…
    # already carry metadata; if it's there we trust it, otherwise we derive
    # from event_id.
    md = epochs.metadata.copy() if epochs.metadata is not None else pd.DataFrame(
        {"trial_type": [list(epochs.event_id.keys())[list(epochs.event_id.values()).index(e)]
                        for e in epochs.events[:, 2]]}
    )

    if "trial_type" not in md.columns:
        raise ValueError(
            f"{subject}: passive epoch metadata missing 'trial_type'. "
            f"Columns: {list(md.columns)}"
        )

    parsed = md["trial_type"].astype(str).map(_parse_trial_type)
    md["attribute"] = parsed.map(lambda t: t[0])
    md["level"] = parsed.map(lambda t: t[1]).astype(int)
    if "badtrial" not in md.columns:
        md["badtrial"] = 0
    epochs.metadata = md.reset_index(drop=True)

    # Drop trials flagged as bad upstream (same convention as 05a_…_passive.py).
    good_mask = epochs.metadata["badtrial"].fillna(0).astype(int).to_numpy() == 0
    n_dropped = int((~good_mask).sum())
    if n_dropped:
        print(f"  {subject}: dropping {n_dropped} bad passive epoch(s)", flush=True)
        epochs = epochs[good_mask]

    if resample_hz and abs(epochs.info["sfreq"] - resample_hz) > 1e-6:
        epochs = epochs.resample(resample_hz, npad="auto", verbose="ERROR")

    return epochs


def bin_epochs_data(
    X: np.ndarray,
    times_ms: np.ndarray,
    window_ms: float,
    step_ms: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Average X over overlapping sliding windows.

    Each output bin is the mean of `window_ms` worth of consecutive samples;
    consecutive bins are offset by `step_ms`. Bin centers are returned.

    Parameters
    ----------
    X        : (..., n_times) — any leading axes are preserved (e.g. trials, channels).
    times_ms : (n_times,) — sample times in ms.
    window_ms, step_ms : positive floats.

    Returns
    -------
    X_binned       : (..., n_bins) — mean within each window.
    bin_centers_ms : (n_bins,) — midpoint of each window in ms.

    Notes
    -----
    Set `window_ms` equal to one sample's duration (e.g. 4 ms at 250 Hz) and
    `step_ms` the same to recover the unbinned single-sample behavior.
    """
    if X.shape[-1] != len(times_ms):
        raise ValueError(
            f"X has {X.shape[-1]} samples but times_ms has {len(times_ms)}."
        )
    if window_ms <= 0 or step_ms <= 0:
        raise ValueError("window_ms and step_ms must be positive.")
    dt_ms = float(np.median(np.diff(times_ms)))
    win_samples = max(1, int(round(window_ms / dt_ms)))
    step_samples = max(1, int(round(step_ms / dt_ms)))
    n_times = X.shape[-1]
    if win_samples > n_times:
        raise ValueError(
            f"window_ms={window_ms} ({win_samples} samples) exceeds n_times={n_times}."
        )
    starts = np.arange(0, n_times - win_samples + 1, step_samples)
    X_binned = np.stack(
        [X[..., s:s + win_samples].mean(axis=-1) for s in starts], axis=-1
    )
    centers = np.array([times_ms[s:s + win_samples].mean() for s in starts])
    return X_binned, centers


def concat_window_features(
    X: np.ndarray,
    times_ms: np.ndarray,
    window_ms: float,
    step_ms: float,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Slide a window over X and concatenate within-window samples as features.

    Unlike `bin_epochs_data` which averages samples inside each window into a
    single feature per channel, this function preserves all samples and stacks
    them as additional features. The resulting feature vector at every bin has
    shape (n_channels * win_samples,) with channels-blocked, win-fastest layout
    so callers can reshape Haufe patterns back to (n_channels, win_samples).

    Parameters
    ----------
    X : (n_trials, n_channels, n_times) array — must be 3-D.
    times_ms : (n_times,) sample times in ms.
    window_ms, step_ms : positive floats.

    Returns
    -------
    X_concat       : (n_trials, n_channels * win_samples, n_bins)
    bin_centers_ms : (n_bins,) midpoint of each window in ms.
    win_samples    : int — number of within-window samples (callers need this
                     to reshape Haufe pattern outputs back to channel topographies).
    """
    if X.ndim != 3:
        raise ValueError(
            f"concat_window_features requires 3-D X (trials, channels, n_times); "
            f"got shape {X.shape}"
        )
    if X.shape[-1] != len(times_ms):
        raise ValueError(
            f"X has {X.shape[-1]} samples but times_ms has {len(times_ms)}."
        )
    if window_ms <= 0 or step_ms <= 0:
        raise ValueError("window_ms and step_ms must be positive.")
    dt_ms = float(np.median(np.diff(times_ms)))
    win_samples = max(1, int(round(window_ms / dt_ms)))
    step_samples = max(1, int(round(step_ms / dt_ms)))
    n_trials, n_chan, n_times = X.shape
    if win_samples > n_times:
        raise ValueError(
            f"window_ms={window_ms} ({win_samples} samples) exceeds n_times={n_times}."
        )
    starts = np.arange(0, n_times - win_samples + 1, step_samples)
    X_concat = np.stack(
        [X[:, :, s:s + win_samples].reshape(n_trials, n_chan * win_samples)
         for s in starts],
        axis=-1,
    )
    centers = np.array([times_ms[s:s + win_samples].mean() for s in starts])
    return X_concat, centers, win_samples


def make_lda_pipeline() -> Pipeline:
    """StandardScaler + shrinkage LDA. Returns a fresh Pipeline each call.

    `lsqr` solver supports `shrinkage="auto"` (Ledoit-Wolf) and is the
    standard MNE-Python recommendation for EEG decoding.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("lda", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
    ])


def cv_auc_timecourse(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    n_jobs: int = 1,
    random_state: int = RANDOM_STATE,
) -> np.ndarray:
    """5-fold stratified CV ROC-AUC at every timepoint.

    Parameters
    ----------
    X : (n_trials, n_channels, n_times) array
    y : (n_trials,) binary label vector (0/1)

    Returns
    -------
    auc : (n_times,) array of mean cross-validated AUC across folds.
    """
    if X.ndim != 3:
        raise ValueError(f"X must be 3-D (trials, channels, times); got shape {X.shape}")
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X has {X.shape[0]} trials but y has {y.shape[0]}")
    if len(np.unique(y)) != 2:
        raise ValueError(f"cv_auc_timecourse expects a binary y; got classes {np.unique(y)}")

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    est = SlidingEstimator(make_lda_pipeline(), scoring="roc_auc", n_jobs=n_jobs,
                           verbose="ERROR")
    scores = cross_val_multiscore(est, X, y, cv=cv, n_jobs=1, verbose="ERROR")
    # cross_val_multiscore returns (n_splits, n_times)
    return scores.mean(axis=0)


def haufe_pattern_timecourse(
    X: np.ndarray,
    y: np.ndarray,
    win_samples: Optional[int] = None,
) -> np.ndarray:
    """Per-timepoint Haufe activation pattern from a shrinkage-LDA fit.

    Refits LDA on the full data at each timepoint (no CV) — patterns are
    descriptive topographies of what the decoder learned, not generalization
    estimates. Pattern is `cov(X(t)) @ w(t)` (Haufe et al. 2014, NeuroImage).

    Parameters
    ----------
    X : (n_trials, n_features, n_times) array. With `win_samples=None`,
        `n_features == n_channels`. When the caller has used
        `concat_window_features`, `n_features == n_channels * win_samples`
        and the function reduces the within-window axis by averaging so the
        returned topography is still per-channel.
    y : (n_trials,) binary label vector.
    win_samples : int or None. If int, features are assumed to be laid out as
        (n_channels, win_samples).ravel() per bin and the within-window axis
        is averaged out before returning the pattern.

    Returns
    -------
    patterns : (n_channels, n_times) array
    """
    if X.ndim != 3:
        raise ValueError(f"X must be 3-D; got shape {X.shape}")
    n_trials, n_feat, n_times = X.shape
    if win_samples is None:
        n_chan = n_feat
    else:
        if n_feat % win_samples != 0:
            raise ValueError(
                f"n_features={n_feat} not divisible by win_samples={win_samples}"
            )
        n_chan = n_feat // win_samples
    patterns = np.empty((n_chan, n_times), dtype=float)
    for t in range(n_times):
        Xt = X[:, :, t]                      # (n_trials, n_features)
        # Standardize the same way the decoding pipeline does so the pattern
        # is in the same feature space.
        scaler = StandardScaler().fit(Xt)
        Xt_z = scaler.transform(Xt)
        lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(Xt_z, y)
        w = lda.coef_.ravel()                # (n_features,)
        cov = np.cov(Xt_z, rowvar=False)     # (n_features, n_features)
        # Undo z-scoring on the pattern so it has interpretable µV units.
        pattern_z = cov @ w
        scale = scaler.scale_                # std per feature
        pattern_native = pattern_z * scale
        if win_samples is None:
            patterns[:, t] = pattern_native
        else:
            patterns[:, t] = pattern_native.reshape(n_chan, win_samples).mean(axis=1)
    return patterns


# ----------------------------------------------------------------------
# Ridge-regression analogues for level decoding (ordinal level 1..5).
# ----------------------------------------------------------------------

def make_ridge_pipeline(alpha: float = 1.0) -> Pipeline:
    """StandardScaler + Ridge. Returns a fresh Pipeline each call.

    Alpha=1.0 is the same regularization used in the archived old/07d2 script
    so per-subject patterns are comparable across code-review iterations.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=alpha)),
    ])


def _pearson_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Pearson correlation between truth and prediction.

    Returns 0.0 if the prediction has zero variance (otherwise pearsonr
    returns NaN and breaks downstream cluster perm).
    """
    if np.std(y_pred) == 0 or np.std(y_true) == 0:
        return 0.0
    r = float(np.corrcoef(y_true, y_pred)[0, 1])
    return r if np.isfinite(r) else 0.0


pearson_scorer = make_scorer(_pearson_score, greater_is_better=True)


def cv_pearson_r_timecourse(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    n_jobs: int = 1,
    random_state: int = RANDOM_STATE,
    alpha: float = 1.0,
) -> np.ndarray:
    """5-fold CV Pearson r at every timepoint (Ridge regression).

    Parameters
    ----------
    X : (n_trials, n_channels, n_times) array
    y : (n_trials,) continuous/ordinal target (e.g. level 1..5)

    Returns
    -------
    r : (n_times,) array of mean cross-validated Pearson r across folds.
    """
    if X.ndim != 3:
        raise ValueError(f"X must be 3-D (trials, channels, times); got {X.shape}")
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X has {X.shape[0]} trials but y has {y.shape[0]}")
    if np.std(y) == 0:
        raise ValueError("y has zero variance; cannot run regression decoding.")

    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    est = SlidingEstimator(make_ridge_pipeline(alpha=alpha), scoring=pearson_scorer,
                           n_jobs=n_jobs, verbose="ERROR")
    scores = cross_val_multiscore(est, X, y, cv=cv, n_jobs=1, verbose="ERROR")
    return scores.mean(axis=0)


def haufe_pattern_timecourse_ridge(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float = 1.0,
    win_samples: Optional[int] = None,
) -> np.ndarray:
    """Per-timepoint Haufe activation pattern from a Ridge fit.

    Same Haufe transform as the LDA version (`cov(X(t)) @ w(t)`), but uses
    Ridge weights instead of LDA's. For ordinal/continuous y. See
    `haufe_pattern_timecourse` for the `win_samples` semantics.
    """
    if X.ndim != 3:
        raise ValueError(f"X must be 3-D; got shape {X.shape}")
    n_trials, n_feat, n_times = X.shape
    if win_samples is None:
        n_chan = n_feat
    else:
        if n_feat % win_samples != 0:
            raise ValueError(
                f"n_features={n_feat} not divisible by win_samples={win_samples}"
            )
        n_chan = n_feat // win_samples
    patterns = np.empty((n_chan, n_times), dtype=float)
    for t in range(n_times):
        Xt = X[:, :, t]
        scaler = StandardScaler().fit(Xt)
        Xt_z = scaler.transform(Xt)
        ridge = Ridge(alpha=alpha).fit(Xt_z, y)
        w = ridge.coef_.ravel()
        cov = np.cov(Xt_z, rowvar=False)
        pattern_z = cov @ w
        scale = scaler.scale_
        pattern_native = pattern_z * scale
        if win_samples is None:
            patterns[:, t] = pattern_native
        else:
            patterns[:, t] = pattern_native.reshape(n_chan, win_samples).mean(axis=1)
    return patterns


@dataclass
class ClusterResult:
    cluster_pvals: np.ndarray         # (n_clusters,)
    cluster_pvals_fdr: Optional[np.ndarray]  # filled by fdr_correct_clusters
    cluster_inds: List[np.ndarray]    # list of timepoint-index arrays
    t_obs: np.ndarray                 # (n_times,) per-timepoint t-statistic


def cluster_perm_1samp_vs_chance(
    scores: np.ndarray,
    chance: float = 0.5,
    n_permutations: int = 1000,
    tail: int = 1,
    alpha: float = 0.05,
    n_jobs: int = 1,
    random_state: int = RANDOM_STATE,
) -> ClusterResult:
    """One-sample cluster permutation against `chance`.

    `scores` is (n_subjects, n_times). `tail=1` for AUC vs chance (directional).
    """
    if scores.ndim != 2:
        raise ValueError(f"scores must be 2-D (subjects x times); got {scores.shape}")
    data = scores - chance
    threshold = None  # MNE picks a t-threshold from `tail` and `alpha`
    t_obs, clusters, cluster_p, _ = permutation_cluster_1samp_test(
        data,
        n_permutations=n_permutations,
        threshold=threshold,
        tail=tail,
        out_type="indices",
        n_jobs=n_jobs,
        seed=random_state,
        verbose="ERROR",
    )
    # `clusters` is a list of (array_of_indices,) tuples — unwrap.
    inds = [c[0] if isinstance(c, tuple) else c for c in clusters]
    return ClusterResult(
        cluster_pvals=np.asarray(cluster_p, dtype=float),
        cluster_pvals_fdr=None,
        cluster_inds=inds,
        t_obs=t_obs,
    )


def fdr_correct_clusters(results: Sequence[ClusterResult]) -> None:
    """In-place Benjamini-Hochberg FDR across a family of cluster results.

    Pools all clusters across all results, computes FDR-adjusted p-values,
    and fills `cluster_pvals_fdr` on each result.
    """
    pooled = np.concatenate([r.cluster_pvals for r in results]) if results else np.array([])
    if pooled.size == 0:
        for r in results:
            r.cluster_pvals_fdr = np.array([], dtype=float)
        return
    # BH-FDR
    order = np.argsort(pooled)
    ranked = pooled[order]
    n = len(ranked)
    adj = ranked * n / (np.arange(1, n + 1))
    # Enforce monotonicity from the back
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj_full = np.empty_like(pooled)
    adj_full[order] = np.clip(adj, 0, 1)
    # Split back to each result
    i = 0
    for r in results:
        k = len(r.cluster_pvals)
        r.cluster_pvals_fdr = adj_full[i:i + k]
        i += k


def save_params(out_dir: Path, argv: list, extra: Optional[dict] = None) -> None:
    """Dump CLI args + environment info to params.json for reproducibility."""
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "argv": list(argv),
        "random_state": RANDOM_STATE,
        "datetime_utc": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "mne": mne.__version__,
        "sklearn": sklearn.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    if extra:
        payload.update(extra)
    (out_dir / "params.json").write_text(json.dumps(payload, indent=2))


def find_peak_window(t_obs: np.ndarray, times_ms: np.ndarray,
                     min_width_ms: float = 50.0) -> Tuple[float, float]:
    """Return (tmin_ms, tmax_ms) centered on the argmax of t_obs, at least
    `min_width_ms` wide. Used for the Haufe topomap window."""
    if t_obs.shape != times_ms.shape:
        raise ValueError(f"t_obs and times_ms shape mismatch: {t_obs.shape} vs {times_ms.shape}")
    peak_idx = int(np.nanargmax(t_obs))
    peak_t = float(times_ms[peak_idx])
    half = min_width_ms / 2.0
    return peak_t - half, peak_t + half


def plot_decoding_timecourse(
    times_ms: np.ndarray,
    scores: np.ndarray,
    sig_mask: np.ndarray,
    title: str,
    chance: float = 0.5,
    ylabel: str = "ROC-AUC",
    out_path: Optional[Path] = None,
) -> plt.Figure:
    """Group-mean decoding time course with 95% CI and shaded significant cluster(s).

    scores : (n_subjects, n_times) array.
    sig_mask : (n_times,) bool, True where significant.
    """
    mean = scores.mean(axis=0)
    sem = scores.std(axis=0, ddof=1) / np.sqrt(scores.shape[0])
    ci95 = 1.96 * sem

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axhline(chance, color="grey", linestyle="--", linewidth=1)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.fill_between(times_ms, mean - ci95, mean + ci95, alpha=0.25)
    ax.plot(times_ms, mean, linewidth=2)
    # Shade significant timepoints
    if sig_mask.any():
        ylo, yhi = ax.get_ylim()
        ax.fill_between(times_ms, ylo, yhi, where=sig_mask, alpha=0.15, color="C1")
        ax.set_ylim(ylo, yhi)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300)
    return fig


def plot_topomap_peak(
    pattern: np.ndarray,
    info: mne.Info,
    title: str,
    out_path: Optional[Path] = None,
) -> plt.Figure:
    """Group-mean Haufe pattern over the peak window as an MNE topomap.

    pattern : (n_channels,) array, group-mean activation pattern averaged over time.
    info : an mne.Info matching the channel order in `pattern`.
    """
    fig, ax = plt.subplots(figsize=(4, 4))
    mne.viz.plot_topomap(pattern, info, axes=ax, show=False, contours=4)
    ax.set_title(title)
    fig.tight_layout()
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300)
    return fig


HDDM_READY_CSV = BASE_PATH / "derivatives" / "behav" / "hddm_ready.csv"


def load_decision_epochs(
    subject: str,
    base_path: Path = BASE_PATH,
    resample_hz: float = 250.0,
    hddm_ready_csv: Path = HDDM_READY_CSV,
) -> mne.Epochs:
    """Load a subject's decision-cue single-trial epochs with merged HDDM metadata.

    Returns epochs with attached metadata DataFrame containing (in addition to
    whatever upstream columns exist) the trial-level behavioural fields:
      - painlevel (float, 1..5)
      - moneylevel (float, 1..5)
      - accepted (int, 0/1)
      - rt (float, seconds)
      - badtrial (int, 0/1, taken from the HDDM CSV — overrides any upstream flag)

    Trials present in the epochs but absent from the HDDM CSV are dropped.
    Epochs are then downsampled to `resample_hz`.

    The HDDM CSV's `trial_seq` column aligns with the epoch metadata's `trialsnum`.
    """
    fif = (
        base_path / "derivatives" / subject / "eeg" / "erps_decision"
        / f"{subject}_decision_cues_singletrials-epo.fif"
    )
    if not fif.exists():
        raise FileNotFoundError(f"{subject}: decision epoch file missing: {fif}")
    if not hddm_ready_csv.exists():
        raise FileNotFoundError(f"hddm_ready.csv missing: {hddm_ready_csv}")

    epochs = mne.read_epochs(str(fif), preload=True, verbose="ERROR")
    if epochs.metadata is None or "trialsnum" not in epochs.metadata.columns:
        raise ValueError(
            f"{subject}: decision epoch metadata missing 'trialsnum'. "
            f"Got: {None if epochs.metadata is None else list(epochs.metadata.columns)}"
        )

    hddm = pd.read_csv(hddm_ready_csv)
    sub_hddm = hddm[hddm["participant"] == subject].copy()
    if sub_hddm.empty:
        raise ValueError(f"{subject}: no rows in hddm_ready.csv (participant column).")
    required = ["trial_seq", "painlevel", "moneylevel", "accepted", "rt", "badtrial"]
    missing = [c for c in required if c not in sub_hddm.columns]
    if missing:
        raise ValueError(f"hddm_ready.csv missing columns: {missing}")
    sub_hddm = sub_hddm[required].copy()
    sub_hddm["trial_seq"] = sub_hddm["trial_seq"].astype(int)
    if sub_hddm["trial_seq"].duplicated().any():
        dupes = sorted(sub_hddm.loc[sub_hddm["trial_seq"].duplicated(), "trial_seq"].unique())
        raise ValueError(f"{subject}: duplicate trial_seq in HDDM CSV: {dupes[:10]}")

    md = epochs.metadata.copy()
    md["trialsnum"] = md["trialsnum"].astype(int)
    keep_mask = md["trialsnum"].isin(set(sub_hddm["trial_seq"].tolist())).to_numpy()
    if keep_mask.sum() == 0:
        raise ValueError(
            f"{subject}: no epoch trialsnum overlaps with HDDM trial_seq. "
            f"Epoch range: {md['trialsnum'].min()}..{md['trialsnum'].max()}; "
            f"HDDM range: {sub_hddm['trial_seq'].min()}..{sub_hddm['trial_seq'].max()}"
        )
    if keep_mask.sum() < len(md):
        dropped = int(len(md) - keep_mask.sum())
        print(f"  {subject}: dropping {dropped} decision epoch(s) without HDDM match",
              flush=True)
        epochs = epochs[keep_mask]
        md = md.loc[keep_mask].reset_index(drop=True)

    merged = md.merge(
        sub_hddm.rename(columns={"trial_seq": "trialsnum"}),
        on="trialsnum",
        how="left",
        validate="1:1",
        suffixes=("", "_hddm"),
    )
    # Prefer the HDDM badtrial flag (overrides any upstream one).
    if "badtrial_hddm" in merged.columns:
        merged["badtrial"] = merged["badtrial_hddm"].astype(int)
        merged = merged.drop(columns=["badtrial_hddm"])
    epochs.metadata = merged.reset_index(drop=True)

    # Drop trials flagged as bad (same convention as 05b_…_decision.py).
    good_mask = epochs.metadata["badtrial"].fillna(0).astype(int).to_numpy() == 0
    n_dropped = int((~good_mask).sum())
    if n_dropped:
        print(f"  {subject}: dropping {n_dropped} bad decision epoch(s)", flush=True)
        epochs = epochs[good_mask]

    if resample_hz and abs(epochs.info["sfreq"] - resample_hz) > 1e-6:
        epochs = epochs.resample(resample_hz, npad="auto", verbose="ERROR")
    return epochs


from mne.decoding import GeneralizingEstimator


def make_lda_generalizing(n_jobs: int = 1) -> "GeneralizingEstimator":
    """LDA inside a temporal-generalization estimator (no scoring set here — we
    extract decision_function values per trial in `generalize_decision_function`)."""
    return GeneralizingEstimator(
        make_lda_pipeline(),
        scoring=None,
        n_jobs=n_jobs,
        verbose="ERROR",
    )


def generalize_decision_function(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    n_jobs: int = 1,
) -> np.ndarray:
    """Fit a temporal-generalization LDA on (X_train, y_train) and return the
    signed decision function for every test trial × (passive_time, decision_time).

    Parameters
    ----------
    X_train : (n_train_trials, n_channels, n_passive_times)
    y_train : (n_train_trials,) binary 0/1
    X_test  : (n_test_trials, n_channels, n_decision_times)

    Returns
    -------
    df : (n_test_trials, n_passive_times, n_decision_times) float32
        LDA `decision_function` at every (train_t, test_t). The sign convention
        follows sklearn's LDA: positive ⇒ class 1.
    """
    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError(
            f"channel mismatch: train n_chan={X_train.shape[1]} vs test n_chan={X_test.shape[1]}"
        )
    if len(np.unique(y_train)) != 2:
        raise ValueError(f"y_train must be binary; got classes {np.unique(y_train)}")
    est = make_lda_generalizing(n_jobs=n_jobs)
    est.fit(X_train, y_train)
    # MNE's GeneralizingEstimator exposes `decision_function` when the underlying
    # estimator does. Output shape is (n_test_trials, n_train_times, n_test_times).
    df = est.decision_function(X_test)
    return df.astype(np.float32)


def generalize_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    alpha: float = 1.0,
    n_jobs: int = 1,
) -> np.ndarray:
    """Regression analogue of `generalize_decision_function`.

    Fits a temporal-generalization Ridge on (X_train, y_train) and returns
    Ridge predictions for every test trial × (train_time, test_time).

    Parameters
    ----------
    X_train : (n_train_trials, n_channels, n_train_times)
    y_train : (n_train_trials,) continuous/ordinal target (e.g. level 1..5)
    X_test  : (n_test_trials, n_channels, n_test_times)

    Returns
    -------
    pred : (n_test_trials, n_train_times, n_test_times) float32
        Ridge `predict` output at every (train_t, test_t). Predicted values
        live on the same scale as y_train (≈ 1..5 for level decoding).
    """
    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError(
            f"channel mismatch: train n_chan={X_train.shape[1]} vs test n_chan={X_test.shape[1]}"
        )
    if np.std(y_train) == 0:
        raise ValueError("y_train has zero variance; cannot fit regression.")
    est = GeneralizingEstimator(make_ridge_pipeline(alpha=alpha), scoring=None,
                                n_jobs=n_jobs, verbose="ERROR")
    est.fit(X_train, y_train)
    pred = est.predict(X_test)
    return pred.astype(np.float32)


def cluster_perm_2d_1samp(
    data: np.ndarray,
    null: float = 0.0,
    n_permutations: int = 1000,
    tail: int = 0,
    alpha: float = 0.05,
    n_jobs: int = 1,
    random_state: int = RANDOM_STATE,
) -> ClusterResult:
    """One-sample 2-D cluster permutation (subjects × time × time) against `null`.

    `data` has shape (n_subjects, n_rows, n_cols). `tail=0` for two-tailed,
    `tail=1` for one-tailed (above null). Returns a `ClusterResult` whose
    `t_obs` is 2-D and `cluster_inds` are tuples of 2-D index arrays.
    """
    if data.ndim != 3:
        raise ValueError(f"data must be 3-D (subjects x rows x cols); got {data.shape}")
    centered = data - null
    t_obs, clusters, cluster_p, _ = permutation_cluster_1samp_test(
        centered,
        n_permutations=n_permutations,
        threshold=None,
        tail=tail,
        out_type="indices",
        n_jobs=n_jobs,
        seed=random_state,
        verbose="ERROR",
    )
    inds = list(clusters)  # already a list; each cluster is a tuple of index arrays
    return ClusterResult(
        cluster_pvals=np.asarray(cluster_p, dtype=float),
        cluster_pvals_fdr=None,
        cluster_inds=inds,
        t_obs=t_obs,
    )


def fisher_z_spearman_matrix(
    evidence: np.ndarray,
    x: np.ndarray,
) -> np.ndarray:
    """Per-(passive_time, decision_time) Spearman correlation between decoded
    evidence and a per-trial scalar `x`, Fisher-z-transformed.

    Parameters
    ----------
    evidence : (n_trials, n_passive_times, n_decision_times)
    x        : (n_trials,) scalar per trial

    Returns
    -------
    z : (n_passive_times, n_decision_times) Fisher-z-transformed Spearman rho.
    """
    if evidence.ndim != 3:
        raise ValueError(f"evidence must be 3-D; got {evidence.shape}")
    if evidence.shape[0] != x.shape[0]:
        raise ValueError(
            f"evidence has {evidence.shape[0]} trials but x has {x.shape[0]}"
        )
    n_trials, n_pt, n_dt = evidence.shape
    out = np.empty((n_pt, n_dt), dtype=float)
    # Rank x once.
    x_ranks = scipy_stats.rankdata(x)
    for i in range(n_pt):
        # Vectorise across decision times by ranking evidence per (i, t).
        ev = evidence[:, i, :]                       # (n_trials, n_dt)
        ev_ranks = scipy_stats.rankdata(ev, axis=0)  # (n_trials, n_dt)
        # Pearson correlation between rank vectors = Spearman rho.
        xr = x_ranks - x_ranks.mean()
        er = ev_ranks - ev_ranks.mean(axis=0)
        num = (xr[:, None] * er).sum(axis=0)
        den = np.sqrt((xr ** 2).sum() * (er ** 2).sum(axis=0))
        rho = num / np.where(den == 0, 1.0, den)
        rho = np.clip(rho, -0.9999, 0.9999)
        out[i] = np.arctanh(rho)                     # Fisher-z
    return out


def fisher_z_pearson_matrix(
    predictions: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """Per-(passive_time, decision_time) Pearson correlation between predicted
    level and the actual per-trial target `y`, Fisher-z-transformed.

    Used by 07b for the level decoders' group-level inference: at each
    (train_t, test_t) cell, correlate Ridge predicted level with the actual
    decision-trial level across trials.

    Parameters
    ----------
    predictions : (n_trials, n_passive_times, n_decision_times)
    y           : (n_trials,) actual target value per trial

    Returns
    -------
    z : (n_passive_times, n_decision_times) Fisher-z-transformed Pearson r.
    """
    if predictions.ndim != 3:
        raise ValueError(f"predictions must be 3-D; got {predictions.shape}")
    if predictions.shape[0] != y.shape[0]:
        raise ValueError(
            f"predictions has {predictions.shape[0]} trials but y has {y.shape[0]}"
        )
    n_trials, n_pt, n_dt = predictions.shape
    out = np.empty((n_pt, n_dt), dtype=float)
    yc = y - y.mean()
    y_ss = float((yc ** 2).sum())
    for i in range(n_pt):
        p = predictions[:, i, :]                  # (n_trials, n_dt)
        pc = p - p.mean(axis=0)
        num = (yc[:, None] * pc).sum(axis=0)
        den = np.sqrt(y_ss * (pc ** 2).sum(axis=0))
        r = num / np.where(den == 0, 1.0, den)
        r = np.clip(r, -0.9999, 0.9999)
        out[i] = np.arctanh(r)
    return out


def plot_generalization_matrix(
    matrix: np.ndarray,
    passive_times_ms: np.ndarray,
    decision_times_ms: np.ndarray,
    sig_mask: Optional[np.ndarray],
    title: str,
    cbar_label: str = "AUC",
    centered_value: float = 0.5,
    out_path: Optional[Path] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> plt.Figure:
    """King & Dehaene-style temporal generalization heatmap.

    `matrix` is (n_passive_times, n_decision_times). `sig_mask` (same shape)
    outlines significant cells with a thin contour.
    """
    if matrix.shape != (len(passive_times_ms), len(decision_times_ms)):
        raise ValueError(
            f"matrix shape {matrix.shape} doesn't match "
            f"({len(passive_times_ms)},{len(decision_times_ms)})"
        )
    if vmin is None or vmax is None:
        span = float(np.nanmax(np.abs(matrix - centered_value)))
        vmin = centered_value - span
        vmax = centered_value + span
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(
        matrix,
        origin="lower",
        aspect="auto",
        cmap="RdBu_r",
        vmin=vmin, vmax=vmax,
        extent=[decision_times_ms[0], decision_times_ms[-1],
                passive_times_ms[0],  passive_times_ms[-1]],
    )
    if sig_mask is not None and sig_mask.any():
        ax.contour(
            decision_times_ms, passive_times_ms,
            sig_mask.astype(float),
            levels=[0.5], colors="k", linewidths=0.8,
        )
    ax.axvline(0, color="k", linewidth=0.6)
    ax.axhline(0, color="k", linewidth=0.6)
    ax.set_xlabel("Decision time (ms)")
    ax.set_ylabel("Passive train time (ms)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label=cbar_label)
    fig.tight_layout()
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300)
    return fig


def plot_sanity_levels(
    times_ms: np.ndarray,
    per_level: "Dict[int, np.ndarray]",
    title: str,
    ylabel: str = "Decoded evidence (LDA decision fn)",
    out_path: Optional[Path] = None,
) -> plt.Figure:
    """Decoded evidence averaged over decision trials, faceted by integer level.

    per_level : dict {level: (n_subjects, n_decision_times)} array.
    """
    import matplotlib.cm as cm
    levels = sorted(per_level.keys())
    colors = cm.viridis(np.linspace(0, 1, len(levels)))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axhline(0, color="grey", linestyle="--", linewidth=1)
    ax.axvline(0, color="black", linewidth=0.8)
    for lv, color in zip(levels, colors):
        m = per_level[lv].mean(axis=0)
        sem = per_level[lv].std(axis=0, ddof=1) / np.sqrt(per_level[lv].shape[0])
        ax.fill_between(times_ms, m - 1.96 * sem, m + 1.96 * sem, alpha=0.2, color=color)
        ax.plot(times_ms, m, color=color, linewidth=2, label=f"level {lv}")
    ax.set_xlabel("Decision time (ms)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300)
    return fig
