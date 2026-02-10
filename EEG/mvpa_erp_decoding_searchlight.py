# -*- coding: utf-8 -*-
"""
Step 2 (Passive): Spatio-temporal SEARCHLIGHT decoding (accuracy + regression)

Like Step 1, but searchlight (channel neighborhoods + temporal neighborhoods).

Analyses:
- Binary (accuracy):
    * money low (20/40) vs high (80/100), drop 60
    * pain  low (20/40) vs high (80/100), drop 60 (control)
- Stimulus type (accuracy):
    * pain vs money across all passive trials (sanity)
- Regression (Ridge; scoring = Pearson r):
    * money levels (20/40/60/80/100), keep 60
    * pain  levels (20/40/60/80/100), keep 60 (control)

Multiple comparisons:
- Within each analysis, group-level spatio-temporal cluster permutation test
  using channel×time adjacency (TFCE if available). This controls family-wise error
  across all channel×time tests within that analysis.

Outputs (per analysis):
- npz with subject maps, T_obs, p_map, cluster p-values, parameters
- csv: subject-level mean timecourses (optional)
- figures: topomaps at key timepoints with sig masks
- json summary + cluster table

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
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold, cross_val_score
from scipy import stats

from mne.channels import find_ch_adjacency
from mne.stats import combine_adjacency
from mne.stats import spatio_temporal_cluster_1samp_test

from tqdm.auto import tqdm


# =============================================================================
# Paths (Step-1 style)
# =============================================================================
DATA_DIR_STR = os.getenv("DATA_DIR", "").strip()
OUT_DIR_STR = os.getenv("OUT_DIR", "").strip()
if DATA_DIR_STR == "":
    raise RuntimeError("DATA_DIR env var not set")

RAW_DIR = Path(DATA_DIR_STR).expanduser()
DERIV_DIR = RAW_DIR / "derivatives"

if OUT_DIR_STR != "":
    OUT_BASE = Path(OUT_DIR_STR).expanduser()
    OUT_DIR = OUT_BASE / "statistics" / "mvpa_passive_step2_searchlight_all"
else:
    OUT_DIR = DERIV_DIR / "statistics" / "mvpa_passive_step2_searchlight_all"

OUT_DIR.mkdir(parents=True, exist_ok=True)
DEBUG_DIR = OUT_DIR / "debug"
DEBUG_DIR.mkdir(parents=True, exist_ok=True)

EPO_DIR = Path("eeg") / "erps_passive"
EPO_SUFFIX = "_passive_cues_singletrials-epo.fif"
BEH_SUFFIX = "_task-passive_beh.tsv"


# =============================================================================
# Parameters
# =============================================================================
RANDOM_STATE = 23
N_SPLITS = 5
N_PERM = 5000
ALPHA_CLUSTER = 0.05

# Resample like Step 1 (recommended for speed + uniform dt)
RESAMPLE_SFREQ = 256  # None to keep original

# Searchlight radii (researcher-ish defaults)
SPATIAL_RADIUS_M = 0.03     # 3 cm
TEMPORAL_RADIUS_MS = 20     # 20 ms radius around each center timepoint

# Stats time window (like Step 1)
TMIN_STAT = 0.0
TMAX_STAT = 0.8

# Columns / codes
COL_COND = "condition"   # 'p' or 'm'
COL_LEVEL = "level"      # 20/40/60/80/100
KEY_BLOCK = "blocks.thisN"
KEY_TRIAL = "trials.thisN"
KEY_TRIALNUM = "trialsnum"
COND_MONEY = "m"
COND_PAIN = "p"

# Which analyses to run
RUN_BINARY = True
RUN_STIMTYPE = True
RUN_REGRESSION = True
RUN_SHUFFLE = True  # for each analysis run a shuffled-label control

CHANCE_ACC = 0.5
CHANCE_R = 0.0  # for correlation score


# =============================================================================
# Helpers: loading / subject list
# =============================================================================
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


def _coerce_int_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("Int64")


def merge_beh_into_epochs(epo: mne.Epochs, beh: pd.DataFrame, sub: str) -> mne.Epochs:
    """Same spirit as your Step 1 merge (trialsnum > keys > order if len matches)."""
    if epo.metadata is None:
        raise ValueError(f"{sub}: epochs has no metadata at all; cannot merge beh.")

    md = epo.metadata.reset_index(drop=True).copy()

    # already merged?
    if (COL_COND in md.columns) and (COL_LEVEL in md.columns):
        return epo

    # 1) trialsnum
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
        return epo

    # required cols exist in beh?
    for col in [COL_COND, COL_LEVEL]:
        if col not in beh.columns:
            beh.head(50).to_csv(DEBUG_DIR / f"{sub}_beh_head.csv", index=False)
            raise ValueError(f"{sub}: beh.tsv missing '{col}'")

    # 2) key merge
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
        merged = md2.merge(beh_small, on=[KEY_BLOCK, KEY_TRIAL], how="left", validate="1:1")

        if merged[COL_COND].isna().any() or merged[COL_LEVEL].isna().any():
            md.head(50).to_csv(DEBUG_DIR / f"{sub}_epo_md_head.csv", index=False)
            beh.head(50).to_csv(DEBUG_DIR / f"{sub}_beh_head.csv", index=False)
            raise ValueError(f"{sub}: key-merge produced unlabeled epochs.")
        epo.metadata = merged
        return epo

    # 3) order merge if possible
    if len(md) == len(beh):
        merged = md.copy()
        merged[COL_COND] = beh[COL_COND].to_numpy()
        merged[COL_LEVEL] = beh[COL_LEVEL].to_numpy()
        for extra in [KEY_BLOCK, KEY_TRIAL]:
            if extra in beh.columns:
                merged[extra] = beh[extra].to_numpy()
        epo.metadata = merged
        return epo

    md.head(50).to_csv(DEBUG_DIR / f"{sub}_epo_md_head.csv", index=False)
    beh.head(50).to_csv(DEBUG_DIR / f"{sub}_beh_head.csv", index=False)
    raise ValueError(f"{sub}: cannot merge beh into epochs; see debug CSVs.")


def alignment_sanity_check(md: pd.DataFrame, sub: str, log, debug_dir: Path):
    """Same idea as your Step 1 sanity check."""
    preview_path = debug_dir / f"{sub}_merged_preview20.csv"
    md.head(20).to_csv(preview_path, index=False)

    if COL_COND not in md.columns:
        log(f"{sub}: alignment check skipped (no '{COL_COND}').")
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
        log(f"{sub}: WARNING very low condition switch-rate ({switches:.3f}). Check {preview_path.name}")
    if diff > 0.70:
        log(f"{sub}: WARNING strong early/late split (|early_m-late_m|={diff:.2f}). Check {preview_path.name}")


# =============================================================================
# Trial selection
# =============================================================================
def make_binary_labels(level: np.ndarray) -> np.ndarray:
    """0=low (20/40), 1=high (80/100). Level 60 should be removed first."""
    level = np.asarray(level, dtype=float)
    y = np.full(len(level), -1, dtype=int)
    y[np.isin(level, [20, 40])] = 0
    y[np.isin(level, [80, 100])] = 1
    return y


def select_trials_binary(epo: mne.Epochs, which: str):
    """Binary low vs high; drops 60. which in {'money','pain'}."""
    if epo.metadata is None:
        raise ValueError("Epochs has no metadata.")
    md = epo.metadata.reset_index(drop=True)

    for col in [COL_COND, COL_LEVEL]:
        if col not in md.columns:
            raise ValueError(f"Missing metadata column '{col}'")

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
    keep2 = ~np.isin(levels, [60])
    epo_f = epo_f.copy()[keep2]
    md_f = epo_f.metadata.reset_index(drop=True)

    if len(epo_f) < 10:
        raise ValueError(f"Too few trials after filtering for {which}. n={len(epo_f)}")

    y = make_binary_labels(md_f[COL_LEVEL].to_numpy(dtype=float))
    if np.any(y < 0):
        raise ValueError(f"Unlabeled trials exist. Levels seen: {np.unique(md_f[COL_LEVEL])}")

    X = epo_f.get_data()  # (n_trials, n_ch, n_t)
    times = epo_f.times.copy()
    return X, y, times, md_f, epo_f.info


def select_trials_stimtype(epo: mne.Epochs):
    """Stimulus type: 0=pain, 1=money across all passive trials."""
    if epo.metadata is None:
        raise ValueError("Epochs has no metadata.")
    md = epo.metadata.reset_index(drop=True)

    if COL_COND not in md.columns:
        raise ValueError(f"Missing metadata column '{COL_COND}'")

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
    return X, y, times, md_f, epo_f.info


def select_trials_regression(epo: mne.Epochs, which: str, drop_level60: bool = False):
    """Regression: y=continuous level (default keeps 60)."""
    if epo.metadata is None:
        raise ValueError("Epochs has no metadata.")
    md = epo.metadata.reset_index(drop=True)

    for col in [COL_COND, COL_LEVEL]:
        if col not in md.columns:
            raise ValueError(f"Missing metadata column '{col}'")

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

    y = levels.astype(float)
    X = epo_f.get_data()
    times = epo_f.times.copy()
    return X, y, times, md_f, epo_f.info


# =============================================================================
# SEARCHLIGHT structure
# =============================================================================
def _get_channel_positions(info: mne.Info, picks: list[int]) -> tuple[np.ndarray, list[int]]:
    """Return (pos Nx3, kept_picks) for channels with valid xyz loc."""
    pos = []
    kept = []
    for pi in picks:
        xyz = np.array(info["chs"][pi]["loc"][:3], dtype=float)
        if np.all(np.isfinite(xyz)) and np.linalg.norm(xyz) > 0:
            pos.append(xyz)
            kept.append(pi)
    if len(pos) == 0:
        raise RuntimeError("No valid electrode positions found (ch['loc'][:3]).")
    return np.vstack(pos), kept


def build_spatiotemporal_searchlight_patches(
    info: mne.Info,
    times: np.ndarray,
    spatial_radius_m: float,
    temporal_radius_ms: float,
):
    """
    Returns:
      centers: list[(ci, ti)] indices into kept-channels and times
      patches: list[(chan_inds, time_inds)] per center
      kept_picks: channel picks into original info/X
      kept_info: mne.Info for kept channels (for adjacency/topos)
      half_win_samp: temporal half window in samples
    """
    picks = mne.pick_types(info, eeg=True, meg=False, eog=False, stim=False, exclude=[])

    ch_pos, kept_picks = _get_channel_positions(info, list(picks))
    kept_info = mne.pick_info(info, kept_picks)

    # distance matrix (Nch x Nch)
    d = np.linalg.norm(ch_pos[:, None, :] - ch_pos[None, :, :], axis=2)

    dt = float(np.median(np.diff(times)))
    if not np.allclose(np.diff(times), dt, atol=1e-9):
        raise RuntimeError("Times are not uniformly spaced. Resample should fix this.")
    sfreq = 1.0 / dt

    half_win_samp = int(np.round((temporal_radius_ms / 1000.0) * sfreq))
    half_win_samp = max(0, half_win_samp)

    n_ch = d.shape[0]
    n_t = len(times)

    # channel neighborhoods
    chan_nbrs = []
    for ci in range(n_ch):
        nbr = np.where(d[ci, :] <= spatial_radius_m)[0]
        if nbr.size == 0:
            nbr = np.array([ci], dtype=int)
        chan_nbrs.append(nbr.astype(int))

    # time neighborhoods
    time_nbrs = []
    for ti in range(n_t):
        t0 = max(0, ti - half_win_samp)
        t1 = min(n_t, ti + half_win_samp + 1)
        time_nbrs.append(np.arange(t0, t1, dtype=int))

    centers = []
    patches = []
    for ci in range(n_ch):
        for ti in range(n_t):
            centers.append((ci, ti))
            patches.append((chan_nbrs[ci], time_nbrs[ti]))

    return centers, patches, kept_picks, kept_info, half_win_samp


def _flatten_patch(X: np.ndarray, chan_inds: np.ndarray, time_inds: np.ndarray) -> np.ndarray:
    """X: (n_trials, n_ch, n_t) -> (n_trials, n_features)."""
    Xp = X[:, chan_inds, :][:, :, time_inds]  # (n_trials, n_chan_patch, n_time_patch)
    return Xp.reshape(Xp.shape[0], -1)


# =============================================================================
# Subject-level searchlight scoring
# =============================================================================
def _corr_scorer(estimator, X, y_true) -> float:
    """Pearson r between y_true and predictions. Returns 0 if undefined."""
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


def _make_cv(groups: np.ndarray | None, y: np.ndarray, *, is_regression: bool):
    """Prefer GroupKFold if blocks exist; else StratifiedKFold for classification, KFold for regression."""
    if groups is not None:
        groups = np.asarray(groups)
        ok = ~pd.isna(groups)
        if ok.sum() == len(groups):
            n_groups = len(np.unique(groups))
            if n_groups >= 2:
                n_splits = min(N_SPLITS, n_groups)
                return GroupKFold(n_splits=n_splits)

    if is_regression:
        return KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    return StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)


def subject_searchlight(
    X_kept: np.ndarray,
    y: np.ndarray,
    centers: list[tuple[int, int]],
    patches: list[tuple[np.ndarray, np.ndarray]],
    *,
    mode: str,  # "acc" or "ridgecorr"
    groups: np.ndarray | None,
    shuffle: bool,
):
    """
    Returns:
      score_map: (n_ch_kept, n_t)
    """
    rng = np.random.default_rng(RANDOM_STATE)
    y_use = rng.permutation(y) if shuffle else y

    if mode == "acc":
        est = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                solver="liblinear",
                max_iter=2000,
                random_state=RANDOM_STATE,
                class_weight="balanced",
            ),
        )
        cv = _make_cv(groups, y_use, is_regression=False)
        scoring = "accuracy"
        needs_groups = isinstance(cv, GroupKFold)

    elif mode == "ridgecorr":
        est = make_pipeline(
            StandardScaler(),
            Ridge(alpha=1.0, random_state=RANDOM_STATE),
        )
        cv = _make_cv(groups, y_use, is_regression=True)
        scoring = _corr_scorer
        needs_groups = isinstance(cv, GroupKFold)

    else:
        raise ValueError("mode must be 'acc' or 'ridgecorr'")

    n_centers = len(centers)
    scores = np.full(n_centers, np.nan, dtype=float)

    for i, (chan_inds, time_inds) in enumerate(patches):
        Xi = _flatten_patch(X_kept, chan_inds, time_inds)

        # safety checks (degenerate matrices)
        if not np.all(np.isfinite(Xi)):
            continue
        if Xi.shape[1] < 2:
            continue
        if np.nanstd(Xi) < 1e-12:
            continue

        sc = cross_val_score(
            est,
            Xi,
            y_use,
            cv=cv,
            groups=groups if needs_groups else None,
            scoring=scoring,
            n_jobs=1,
        )
        scores[i] = float(np.nanmean(sc))

    # centers were built (ci loop then ti loop), so reshape to (n_ch, n_t)
    ch_ids = [c[0] for c in centers]
    t_ids = [c[1] for c in centers]
    n_ch = max(ch_ids) + 1
    n_t = max(t_ids) + 1
    return scores.reshape(n_ch, n_t)


# =============================================================================
# Group-level spatio-temporal cluster test (multiple comparisons)
# =============================================================================
def group_cluster_spatiotemporal(
    maps_by_subj: np.ndarray,  # (n_subj, n_ch, n_t)
    times: np.ndarray,
    info_kept: mne.Info,
    *,
    chance: float,
    tail: int,
):
    """
    Uses channel×time adjacency and spatio-temporal cluster permutation.
    Returns dict with p_map (n_times_stat, n_ch), T_obs (n_times_stat, n_ch), clusters, cluster_pv.
    """
    tmask = (times >= TMIN_STAT) & (times <= TMAX_STAT)
    times_stat = times[tmask]

    # test against chance
    X = maps_by_subj[:, :, tmask] - chance  # (n_subj, n_ch, n_tstat)

    # adjacency
    ch_adj, _ = find_ch_adjacency(info_kept, ch_type="eeg")
    adjacency = combine_adjacency(len(times_stat), ch_adj)

    # MNE expects (n_subj, n_times, n_ch)
    X_mne = X.transpose(0, 2, 1)

    tfce_thresh = dict(start=0.0, step=0.2)
    try:
        T_obs, clusters, cluster_pv, H0 = spatio_temporal_cluster_1samp_test(
            X_mne,
            adjacency=adjacency,
            n_permutations=N_PERM,
            threshold=tfce_thresh,
            tail=tail,
            n_jobs=1,
            seed=RANDOM_STATE,
            buffer_size=None,
            out_type="mask",
        )
        thresh_used = "tfce"
    except Exception:
        T_obs, clusters, cluster_pv, H0 = spatio_temporal_cluster_1samp_test(
            X_mne,
            adjacency=adjacency,
            n_permutations=N_PERM,
            threshold=None,
            tail=tail,
            n_jobs=1,
            seed=RANDOM_STATE,
            buffer_size=None,
            out_type="mask",
        )
        thresh_used = "threshold=None"

    # p_map (n_times_stat, n_ch)
    p_map = np.ones_like(T_obs, dtype=float)
    for cl, p in zip(clusters, cluster_pv):
        if cl is None:
            continue
        if np.any(cl):
            p_map[cl] = np.minimum(p_map[cl], p)

    return dict(
        times_stat=times_stat,
        T_obs=T_obs,
        clusters=clusters,
        cluster_pv=cluster_pv,
        p_map=p_map,
        thresh_used=thresh_used,
    )


# =============================================================================
# Plotting (topomaps with sig masks)
# =============================================================================
def plot_topomap_timepoint(
    info: mne.Info,
    data_ch: np.ndarray,
    mask: np.ndarray | None,
    title: str,
    out_path: Path,
    vlim: tuple[float, float] | None = None,
):
    fig, ax = plt.subplots(figsize=(2.2, 2.2))
    mne.viz.plot_topomap(
        data_ch,
        pos=info,
        mask=mask,
        show=False,
        axes=ax,
        contours=0,
        sensors=False,
        outlines="head",
        extrapolate="head",
        vlim=vlim,
        mask_params=dict(marker="o", markerfacecolor="w", markeredgecolor="k",
                         linewidth=0, markersize=3),
    )
    ax.set_title(title, pad=0.1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_channelmean_timecourse(mean_map: np.ndarray, times: np.ndarray, title: str,
                               out_path: Path, chance: float, ylabel: str, ylim: tuple[float, float]):
    """Mean across channels as a compact summary plot."""
    m = np.nanmean(mean_map, axis=0)
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(times, m, linewidth=2)
    ax.axhline(chance, linestyle="--", linewidth=1)
    ax.axvline(0, linestyle="--", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

def plot_sig_channel_fraction(
    p_map: np.ndarray,      # (n_times, n_ch)
    times: np.ndarray,      # (n_times,)
    alpha: float,
    title: str,
    out_path: Path,
):
    sig = (p_map < alpha)
    frac = sig.mean(axis=1)  # fraction of channels significant at each time

    fig, ax = plt.subplots(figsize=(7, 2.5))
    ax.plot(times, frac, linewidth=2)
    ax.axvline(0, linestyle="--", linewidth=1)

    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frac. sig channels")
    ax.set_ylim(-0.02, 1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

# =============================================================================
# Saving cluster table / summary
# =============================================================================
def save_cluster_table_and_summary(
    *,
    tag: str,
    out_dir: Path,
    stats_out: dict,
    times_stat: np.ndarray,
    info_kept: mne.Info,
    maps_stat: np.ndarray,   # (n_subj, n_ch, n_tstat)
    chance: float,
    included: list[str],
):
    # cluster table
    rows = []
    T_obs = stats_out["T_obs"]          # (n_tstat, n_ch)
    cluster_pv = stats_out["cluster_pv"]
    clusters = stats_out["clusters"]    # list of boolean masks (n_tstat, n_ch)

    for i, (cl_mask, p) in enumerate(zip(clusters, cluster_pv)):
        if cl_mask is None or not np.any(cl_mask):
            continue

        # time span
        t_inds = np.where(np.any(cl_mask, axis=1))[0]
        t_start = float(times_stat[t_inds[0]])
        t_end = float(times_stat[t_inds[-1]])
        dur_ms = (t_end - t_start) * 1000.0

        # channel count in cluster (unique channels)
        ch_inds = np.where(np.any(cl_mask, axis=0))[0]
        n_ch = int(len(ch_inds))

        # mass stats
        mass_sumT = float(T_obs[cl_mask].sum())
        maxT = float(T_obs[cl_mask].max())
        minT = float(T_obs[cl_mask].min())

        # mean effect size in cluster (score - chance)
        # maps_stat: (n_subj, n_ch, n_tstat) -> transpose to (n_subj, n_tstat, n_ch)
        eff = maps_stat.transpose(0, 2, 1) - chance
        mean_eff = float(eff[:, cl_mask].mean())
        sign = "pos" if mean_eff >= 0 else "neg"

        rows.append(dict(
            cluster=i,
            p_value=float(p),
            sign=sign,
            t_start_s=t_start,
            t_end_s=t_end,
            duration_ms=dur_ms,
            n_channels=n_ch,
            cluster_mass_sumT=mass_sumT,
            cluster_maxT=maxT,
            cluster_minT=minT,
            cluster_mean_effect_minus_chance=mean_eff,
        ))

    df_cl = pd.DataFrame(rows).sort_values("p_value") if len(rows) else pd.DataFrame(
        columns=["cluster","p_value","sign","t_start_s","t_end_s","duration_ms",
                 "n_channels","cluster_mass_sumT","cluster_maxT","cluster_minT",
                 "cluster_mean_effect_minus_chance"]
    )
    df_cl.to_csv(out_dir / f"{tag}_cluster_table.csv", index=False)

    meta = dict(
        tag=tag,
        n_subjects=int(maps_stat.shape[0]),
        n_channels=int(maps_stat.shape[1]),
        n_times_stat=int(maps_stat.shape[2]),
        chance=float(chance),
        alpha_cluster=float(ALPHA_CLUSTER),
        thresh_used=stats_out.get("thresh_used", None),
        n_clusters=int(len(cluster_pv)),
        min_cluster_p=float(np.min(cluster_pv)) if len(cluster_pv) else 1.0,
        included_subjects=included,
        spatial_radius_m=float(SPATIAL_RADIUS_M),
        temporal_radius_ms=float(TEMPORAL_RADIUS_MS),
        tmin_stat=float(TMIN_STAT),
        tmax_stat=float(TMAX_STAT),
    )
    with open(out_dir / f"{tag}_summary.json", "w") as f:
        json.dump(meta, f, indent=2)


# =============================================================================
# Core runner
# =============================================================================
def _prepare_subject_epochs(sub: str, *, debug_dir: Path, log):
    epo = load_passive_epochs(sub)
    beh = load_passive_beh(sub)
    epo = merge_beh_into_epochs(epo, beh, sub=sub)

    if epo.metadata is None:
        raise RuntimeError(f"{sub}: metadata is None after merge (should never happen).")

    # alignment sanity
    md = epo.metadata.reset_index(drop=True)
    alignment_sanity_check(md, sub=sub, log=log, debug_dir=debug_dir)

    # drop bad trials
    if "badtrial" in epo.metadata.columns:
        bad = epo.metadata["badtrial"].fillna(0).astype(int).to_numpy()
        if bad.sum() > 0:
            epo = epo.copy()[bad == 0]

    # resample
    if RESAMPLE_SFREQ is not None:
        epo = epo.copy().resample(RESAMPLE_SFREQ, npad="auto")

    return epo


def run_searchlight_analysis(
    *,
    analysis_name: str,    # e.g., "binary_money"
    mode: str,             # "acc" or "ridgecorr"
    selector_fn,           # function(epo)->(X,y,times,md,info)
    chance: float,
    tail: int,
    out_dir: Path,
    debug_dir: Path,
    shuffle: bool,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)

    def log(msg: str):
        tqdm.write(msg)

    subs = list_subjects(DERIV_DIR)
    included, skipped = [], []
    subj_records = []

    maps = []
    times_ref = None
    centers = patches = None
    kept_picks = None
    kept_info = None
    half_win_samp = None

    pbar = tqdm(subs, desc=f"{analysis_name}{'_shuf' if shuffle else ''}", unit="sub", dynamic_ncols=True)
    for sub in pbar:
        try:
            epo = _prepare_subject_epochs(sub, debug_dir=debug_dir, log=log)

            X, y, times, md_used, info = selector_fn(epo)

            # groups for block-wise CV
            groups = None
            if KEY_BLOCK in md_used.columns:
                groups = md_used[KEY_BLOCK].to_numpy()

            # build patches once
            if times_ref is None:
                times_ref = times

                centers, patches, kept_picks, kept_info, half_win_samp = build_spatiotemporal_searchlight_patches(
                    info, times, SPATIAL_RADIUS_M, TEMPORAL_RADIUS_MS
                )
                log(f"{sub}: searchlight template built | kept_ch={len(kept_picks)} "
                    f"| n_times={len(times)} | half_win={half_win_samp} samples @ sfreq≈{RESAMPLE_SFREQ}")

            else:
                if len(times) != len(times_ref) or np.max(np.abs(times - times_ref)) > 1e-9:
                    raise RuntimeError("Time axis mismatch across subjects.")

            # align channels to kept_picks
            X_kept = X[:, kept_picks, :]

            # run subject searchlight
            score_map = subject_searchlight(
                X_kept,
                y,
                centers,
                patches,
                mode=mode,
                groups=groups,
                shuffle=shuffle,
            )

            maps.append(score_map)
            included.append(sub)

            rec = dict(
                subject=sub,
                shuffle=bool(shuffle),
                n_trials=int(len(y)),
                has_blocks=bool(groups is not None),
                n_blocks=int(len(np.unique(groups))) if groups is not None else None,
            )
            # add class counts if classification
            if mode == "acc":
                rec["n0"] = int(np.sum(np.asarray(y) == 0))
                rec["n1"] = int(np.sum(np.asarray(y) == 1))
            else:
                rec["y_min"] = float(np.nanmin(y))
                rec["y_max"] = float(np.nanmax(y))
            subj_records.append(rec)

            pbar.set_postfix_str(f"{sub} | trials={len(y)} | map_mean={np.nanmean(score_map):.3f}")

        except Exception as e:
            skipped.append((sub, str(e)))
            log(f"Skipped {sub}: {e}")

    if len(maps) < 8:
        raise RuntimeError(f"{analysis_name}: too few subjects included for group stats (n={len(maps)})")

    maps = np.stack(maps, axis=0)   # (n_subj, n_ch, n_t)
    times = times_ref

    # group cluster stats (FWER corrected via cluster permutation)
    stats_out = group_cluster_spatiotemporal(
        maps,
        times,
        kept_info,
        chance=chance,
        tail=tail,
    )
    times_stat = stats_out["times_stat"]

    # slice maps to stat window for some outputs
    tmask = (times >= TMIN_STAT) & (times <= TMAX_STAT)
    maps_stat = maps[:, :, tmask]  # (n_subj, n_ch, n_tstat)

    tag = f"{analysis_name}" + ("_shuffle" if shuffle else "")

    # save npz
    np.savez(
        out_dir / f"{tag}_group_results.npz",
        maps_by_subj=maps,
        times=times,
        ch_names=np.array(kept_info["ch_names"], dtype=object),
        included=np.array(included, dtype=object),
        skipped=np.array(skipped, dtype=object),
        T_obs=stats_out["T_obs"],
        p_map=stats_out["p_map"],
        cluster_pv=stats_out["cluster_pv"],
        thresh_used=stats_out["thresh_used"],
        chance=chance,
        spatial_radius_m=SPATIAL_RADIUS_M,
        temporal_radius_ms=TEMPORAL_RADIUS_MS,
        resample_sfreq=RESAMPLE_SFREQ if RESAMPLE_SFREQ is not None else -1,
        tmin_stat=TMIN_STAT,
        tmax_stat=TMAX_STAT,
        alpha_cluster=ALPHA_CLUSTER,
    )

    # subject summary
    pd.DataFrame(subj_records).to_csv(out_dir / f"{tag}_subject_summary.csv", index=False)

    # compact channel-mean timecourse plot
    mean_map = np.nanmean(maps, axis=0)  # (n_ch, n_t)
    ylabel = "Accuracy" if mode == "acc" else "Corr r"
    ylim = (0.35, 0.85) if mode == "acc" else (-0.10, 0.40)
    plot_channelmean_timecourse(
        mean_map,
        times,
        title=f"{analysis_name} ({'SHUFFLED' if shuffle else 'REAL'}) | channel-mean",
        out_path=out_dir / f"{tag}_channelmean_timecourse.png",
        chance=chance,
        ylabel=ylabel,
        ylim=ylim,
    )

    # topomaps at standard timepoints (masked by p_map at that time)
    figs_dir = out_dir / "figs_topo"
    figs_dir.mkdir(exist_ok=True)

    plot_times = [0.2, 0.4, 0.6, 0.8]
    p_map = stats_out["p_map"]  # (n_tstat, n_ch)
    mean_stat = np.nanmean(maps_stat, axis=0)  # (n_ch, n_tstat)

    plot_sig_channel_fraction(
        p_map=p_map,
        times=times_stat,
        alpha=ALPHA_CLUSTER,
        title=f"{analysis_name} ({'SHUFFLED' if shuffle else 'REAL'}): fraction sig channels",
        out_path=out_dir / f"{tag}_sig_fraction_timecourse.png",
    )


    for t in plot_times:
        ti = int(np.argmin(np.abs(times_stat - t)))
        mask = (p_map[ti, :] < ALPHA_CLUSTER)

        vlim = (0.45, 0.75) if mode == "acc" else (-0.05, 0.25)
        plot_topomap_timepoint(
            kept_info,
            data_ch=mean_stat[:, ti],
            mask=mask,
            title=f"{analysis_name} @ {int(t*1000)} ms",
            out_path=figs_dir / f"{tag}_topo_{int(t*1000)}ms.png",
            vlim=vlim,
        )

    # cluster table + json summary
    save_cluster_table_and_summary(
        tag=tag,
        out_dir=out_dir,
        stats_out=stats_out,
        times_stat=times_stat,
        info_kept=kept_info,
        maps_stat=maps_stat,
        chance=chance,
        included=included,
    )

    min_p = float(np.min(stats_out["cluster_pv"])) if len(stats_out["cluster_pv"]) else 1.0
    log(f"\nDONE {tag}: included={len(included)} | min cluster p={min_p:.6f} | thresh={stats_out['thresh_used']}")


# =============================================================================
# Define selectors as closures (keeps runner generic)
# =============================================================================
def selector_binary_money(epo):
    X, y, times, md, info = select_trials_binary(epo, "money")
    return X, y, times, md, info

def selector_binary_pain(epo):
    X, y, times, md, info = select_trials_binary(epo, "pain")
    return X, y, times, md, info

def selector_stimtype(epo):
    X, y, times, md, info = select_trials_stimtype(epo)
    return X, y, times, md, info

def selector_reg_money(epo):
    X, y, times, md, info = select_trials_regression(epo, "money", drop_level60=False)
    return X, y, times, md, info

def selector_reg_pain(epo):
    X, y, times, md, info = select_trials_regression(epo, "pain", drop_level60=False)
    return X, y, times, md, info


# =============================================================================
# Main
# =============================================================================
def main():
    mne.set_log_level("WARNING")

    # folder layout like Step 1
    OUT_BIN = OUT_DIR / "binary_accuracy"
    OUT_STIM = OUT_DIR / "stimtype_accuracy"
    OUT_REG = OUT_DIR / "regression_ridgecorr"

    DBG_BIN = OUT_BIN / "debug"
    DBG_STIM = OUT_STIM / "debug"
    DBG_REG = OUT_REG / "debug"

    if RUN_BINARY:
        run_searchlight_analysis(
            analysis_name="passive_money_binary_acc",
            mode="acc",
            selector_fn=selector_binary_money,
            chance=CHANCE_ACC,
            tail=1,  # test > chance
            out_dir=OUT_BIN,
            debug_dir=DBG_BIN,
            shuffle=False,
        )
        run_searchlight_analysis(
            analysis_name="passive_pain_binary_acc",
            mode="acc",
            selector_fn=selector_binary_pain,
            chance=CHANCE_ACC,
            tail=1,
            out_dir=OUT_BIN,
            debug_dir=DBG_BIN,
            shuffle=False,
        )
        if RUN_SHUFFLE:
            run_searchlight_analysis(
                analysis_name="passive_money_binary_acc",
                mode="acc",
                selector_fn=selector_binary_money,
                chance=CHANCE_ACC,
                tail=1,
                out_dir=OUT_BIN,
                debug_dir=DBG_BIN,
                shuffle=True,
            )
            run_searchlight_analysis(
                analysis_name="passive_pain_binary_acc",
                mode="acc",
                selector_fn=selector_binary_pain,
                chance=CHANCE_ACC,
                tail=1,
                out_dir=OUT_BIN,
                debug_dir=DBG_BIN,
                shuffle=True,
            )

    if RUN_STIMTYPE:
        run_searchlight_analysis(
            analysis_name="passive_stimtype_acc",
            mode="acc",
            selector_fn=selector_stimtype,
            chance=CHANCE_ACC,
            tail=1,
            out_dir=OUT_STIM,
            debug_dir=DBG_STIM,
            shuffle=False,
        )
        if RUN_SHUFFLE:
            run_searchlight_analysis(
                analysis_name="passive_stimtype_acc",
                mode="acc",
                selector_fn=selector_stimtype,
                chance=CHANCE_ACC,
                tail=1,
                out_dir=OUT_STIM,
                debug_dir=DBG_STIM,
                shuffle=True,
            )

    if RUN_REGRESSION:
        # For correlation, strictly speaking you might want two-sided (tail=0)
        # because r can be negative. If you only care about positive decoding:
        # keep tail=1. Here we do two-sided by using tail=0, like your time-resolved reg.
        run_searchlight_analysis(
            analysis_name="passive_money_ridgecorr",
            mode="ridgecorr",
            selector_fn=selector_reg_money,
            chance=CHANCE_R,
            tail=0,  # two-sided test around 0
            out_dir=OUT_REG,
            debug_dir=DBG_REG,
            shuffle=False,
        )
        run_searchlight_analysis(
            analysis_name="passive_pain_ridgecorr",
            mode="ridgecorr",
            selector_fn=selector_reg_pain,
            chance=CHANCE_R,
            tail=0,
            out_dir=OUT_REG,
            debug_dir=DBG_REG,
            shuffle=False,
        )
        if RUN_SHUFFLE:
            run_searchlight_analysis(
                analysis_name="passive_money_ridgecorr",
                mode="ridgecorr",
                selector_fn=selector_reg_money,
                chance=CHANCE_R,
                tail=0,
                out_dir=OUT_REG,
                debug_dir=DBG_REG,
                shuffle=True,
            )
            run_searchlight_analysis(
                analysis_name="passive_pain_ridgecorr",
                mode="ridgecorr",
                selector_fn=selector_reg_pain,
                chance=CHANCE_R,
                tail=0,
                out_dir=OUT_REG,
                debug_dir=DBG_REG,
                shuffle=True,
            )


if __name__ == "__main__":
    main()
