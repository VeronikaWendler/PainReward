# -*- coding: utf-8 -*-

#- Searchlight maps (channel neighborhoods + temporal neighborhoods) for decision money/pain binary
#  Metrics: balanced accuracy + AUC
#  Conditions: raw + ctrlOther (residualize by other cue level, lin)
#  Shuffle controls (optional)
#  Group spatio-temporal TFCE cluster stats (channel×time)
#  Topomaps bundle like passive searchlight

#- Temporal generalization heatmaps (train-time × test-time) for analyses
#  Score shown as (score - chance), red=above, blue=below
#  Saves per-subject matrices + group mean matrix
#  Also extracts diagonal timecourse for cluster stats over time

#- Cross-label generalization (money->pain, pain->money)
#  Diagonal + heatmap
#  raw + optional ctrlOtherTrain + shuffle

#- Money vs Pain difference maps (money - pain) spatio-temporal stats + topos


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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.base import clone
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, accuracy_score

from mne.channels import find_ch_adjacency
from mne.stats import (
    combine_adjacency,
    spatio_temporal_cluster_1samp_test,
    permutation_cluster_1samp_test,
)
from mne.decoding import GeneralizingEstimator
from tqdm.auto import tqdm


# =============================================================================
# Paths 
# =============================================================================
DATA_DIR_STR = os.getenv("DATA_DIR", "").strip()
OUT_DIR_STR  = os.getenv("OUT_DIR", "").strip()
if DATA_DIR_STR == "":
    raise RuntimeError("DATA_DIR env var not set")

RAW_DIR   = Path(DATA_DIR_STR).expanduser()
DERIV_DIR = RAW_DIR / "derivatives"

if OUT_DIR_STR != "":
    OUT_BASE = Path(OUT_DIR_STR).expanduser()
    OUT_DIR = OUT_BASE / "statistics" / "mvpa_decision_step2_searchlight"
else:
    OUT_DIR = DERIV_DIR / "statistics" / "mvpa_decision_step2_searchlight"

OUT_DIR.mkdir(parents=True, exist_ok=True)
DEBUG_DIR = OUT_DIR / "debug"
DEBUG_DIR.mkdir(exist_ok=True)


# =============================================================================
# Decision epochs/beh
# =============================================================================
EPO_DIR    = Path("eeg") / "erps"
EPO_SUFFIX = "_decision_cues_singletrials-epo.fif"
BEH_SUFFIX = "_task-decision_beh.tsv"

DEC_MONEY_COL = "moneystim"
DEC_PAIN_COL  = "painstim"

KEY_TRIALNUM = "trialsnum"
KEY_BLOCK    = "blocks.thisN"
KEY_TRIAL    = "trials.thisN"


# =============================================================================
# Parameters
# =============================================================================
RANDOM_STATE  = 23
N_SPLITS      = 5
N_PERM        = 5000
ALPHA_CLUSTER = 0.05

RESAMPLE_SFREQ = 256

# Searchlight radii (same as passive default)
SPATIAL_RADIUS_M   = 0.03   # 3 cm
TEMPORAL_RADIUS_MS = 20     # 20 ms

# Stats time window
TMIN_STAT = 0.0
TMAX_STAT = 0.8

# Binary decision levels
BIN_KEEP_LEVELS = np.array([20, 40, 80, 100], dtype=int)
CHANCE = 0.5

# Heatmap decimation (temporal generalization)
HEATMAP_DECIM = 2

# Timepoints for topomaps (requested)
TOPO_TIMES_S = [0.2, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
# extra auto timepoints where significance exists
N_EXTRA_SIG_TOPO_TIMES = 4

# What to run
RUN_RAW          = False
RUN_CTRLOTHER    = True
RUN_SHUFFLE      = True

RUN_WITHINLABEL_HEATMAPS = True
RUN_CROSS_LABEL          = True
RUN_DIFF_MONEY_MINUS_PAIN = True

# Timecourse difference stats for diag (money - pain)
RUN_DIAG_DIFF_STATS = True


# =============================================================================
# Logging helper
# =============================================================================
def log_print(msg: str):
    try:
        tqdm.write(msg)
    except Exception:
        print(msg, flush=True)


# =============================================================================
# IO
# =============================================================================
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
    - if epochs.metadata contains DEC_MONEY_COL+DEC_PAIN_COL, skip
    - else merge via trialsnum; else block+trial; else order if lengths match
    """
    if epo.metadata is None:
        raise ValueError(f"{sub}: epochs has no metadata; cannot merge beh.")

    md = epo.metadata.reset_index(drop=True).copy()

    if (DEC_MONEY_COL in md.columns) and (DEC_PAIN_COL in md.columns):
        return epo

    # trialsnum merge
    if (KEY_TRIALNUM in md.columns) and (KEY_TRIALNUM in beh.columns):
        merged = md.merge(beh, on=KEY_TRIALNUM, how="left", validate="1:1")
        if merged[DEC_MONEY_COL].isna().any() or merged[DEC_PAIN_COL].isna().any():
            md.head(50).to_csv(debug_dir / f"{sub}_epo_md_head.csv", index=False)
            beh.head(50).to_csv(debug_dir / f"{sub}_beh_head.csv", index=False)
            raise ValueError(f"{sub}: trialsnum merge produced unlabeled epochs.")
        epo.metadata = merged
        return epo

    # key merge
    can_key_merge = (
        (KEY_BLOCK in md.columns) and (KEY_TRIAL in md.columns) and
        (KEY_BLOCK in beh.columns) and (KEY_TRIAL in beh.columns)
    )
    if can_key_merge:
        md2  = md.copy()
        beh2 = beh.copy()
        md2[KEY_BLOCK]  = _coerce_int_series(md2[KEY_BLOCK])
        md2[KEY_TRIAL]  = _coerce_int_series(md2[KEY_TRIAL])
        beh2[KEY_BLOCK] = _coerce_int_series(beh2[KEY_BLOCK])
        beh2[KEY_TRIAL] = _coerce_int_series(beh2[KEY_TRIAL])

        merged = md2.merge(beh2, on=[KEY_BLOCK, KEY_TRIAL], how="left", validate="1:1")
        if merged[DEC_MONEY_COL].isna().any() or merged[DEC_PAIN_COL].isna().any():
            md.head(50).to_csv(debug_dir / f"{sub}_epo_md_head.csv", index=False)
            beh.head(50).to_csv(debug_dir / f"{sub}_beh_head.csv", index=False)
            raise ValueError(f"{sub}: key merge produced unlabeled epochs.")
        epo.metadata = merged
        return epo

    # order merge
    if len(md) == len(beh):
        merged = md.copy()
        for c in beh.columns:
            if c not in merged.columns:
                merged[c] = beh[c].to_numpy()
        if merged[DEC_MONEY_COL].isna().any() or merged[DEC_PAIN_COL].isna().any():
            raise ValueError(f"{sub}: order merge produced unlabeled epochs.")
        epo.metadata = merged
        return epo

    md.head(50).to_csv(debug_dir / f"{sub}_epo_md_head.csv", index=False)
    beh.head(50).to_csv(debug_dir / f"{sub}_beh_head.csv", index=False)
    raise ValueError(f"{sub}: cannot merge beh into epochs. See debug CSVs.")


# =============================================================================
# Labels + residualization (control-by-other)
# =============================================================================
LEVEL_CODE_TO_LEVEL = {1: 20, 2: 40, 3: 60, 4: 80, 5: 100}

def parse_stim_code_to_level(series: pd.Series, prefix: str) -> np.ndarray:
    s = series.astype(str).str.strip().str.lower()
    codes = s.str.extract(rf"^{prefix}\s*([1-5])$", expand=False)
    if codes.isna().any():
        bad = s[codes.isna()].unique()[:10]
        raise ValueError(f"Unexpected '{prefix}' stim codes (examples): {bad}")
    codes_int = codes.astype(int).to_numpy()
    return np.array([LEVEL_CODE_TO_LEVEL[int(c)] for c in codes_int], dtype=int)

def make_binary_labels(level: np.ndarray) -> np.ndarray:
    level = np.asarray(level, dtype=float)
    y = np.full(len(level), -1, dtype=int)
    y[np.isin(level, [20, 40])] = 0
    y[np.isin(level, [80, 100])] = 1
    return y

def _design_matrix_from_nuisances(n: np.ndarray) -> np.ndarray:
    n = np.asarray(n, dtype=float).ravel()
    mu = np.nanmean(n)
    sd = np.nanstd(n)
    if not np.isfinite(sd) or sd < 1e-12:
        nz = np.zeros_like(n)
    else:
        nz = (n - mu) / sd
    return np.column_stack([np.ones_like(nz), nz])

def residualize_X_by_other_level(X: np.ndarray, other_level: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    n_trials = X.shape[0]
    A = _design_matrix_from_nuisances(other_level)
    Y = X.reshape(n_trials, -1)
    beta, *_ = np.linalg.lstsq(A, Y, rcond=None)
    Y_hat = A @ beta
    Y_res = Y - Y_hat
    return Y_res.reshape(X.shape)

def _get_levels_from_epochs(epo: mne.Epochs) -> tuple[np.ndarray, np.ndarray]:
    if epo.metadata is None:
        raise ValueError("Epochs has no metadata.")
    md = epo.metadata.reset_index(drop=True)
    for col in [DEC_MONEY_COL, DEC_PAIN_COL]:
        if col not in md.columns:
            raise ValueError(f"Missing metadata column '{col}'.")
    money_levels = parse_stim_code_to_level(md[DEC_MONEY_COL], "m")
    pain_levels  = parse_stim_code_to_level(md[DEC_PAIN_COL],  "p")
    return money_levels, pain_levels

def select_trials_decision_binary(epo: mne.Epochs, which: str, control_resid_by_other: bool):
    money_levels, pain_levels = _get_levels_from_epochs(epo)
    levels = money_levels if which == "money" else pain_levels
    other  = pain_levels  if which == "money" else money_levels

    keep = np.isin(levels, BIN_KEEP_LEVELS)
    epo_f = epo.copy()[keep]
    md_f  = epo_f.metadata.reset_index(drop=True)

    levels_f = levels[keep]
    other_f  = other[keep]

    if len(epo_f) < 10:
        raise ValueError(f"Too few trials after filtering for {which} binary. n={len(epo_f)}")

    X = epo_f.get_data()
    times = epo_f.times.copy()

    if control_resid_by_other:
        X = residualize_X_by_other_level(X, other_f.astype(float))

    y = make_binary_labels(levels_f.astype(float))
    if np.any(y < 0):
        raise ValueError("Binary labels contain -1 (unexpected after filtering).")

    info = epo_f.info
    return X, y, times, md_f, info


# =============================================================================
# Searchlight patches (same as passive)
# =============================================================================
def _get_channel_positions(info: mne.Info, picks: list[int]) -> tuple[np.ndarray, list[int]]:
    pos, kept = [], []
    for pi in picks:
        xyz = np.array(info["chs"][pi]["loc"][:3], dtype=float)
        if np.all(np.isfinite(xyz)) and np.linalg.norm(xyz) > 0:
            pos.append(xyz)
            kept.append(pi)
    if len(pos) == 0:
        raise RuntimeError("No valid electrode positions found (ch['loc'][:3]).")
    return np.vstack(pos), kept

def build_spatiotemporal_searchlight_patches(info: mne.Info, times: np.ndarray,
                                             spatial_radius_m: float, temporal_radius_ms: float):
    picks = mne.pick_types(info, eeg=True, meg=False, eog=False, stim=False, exclude=[])
    ch_pos, kept_picks = _get_channel_positions(info, list(picks))
    kept_info = mne.pick_info(info, kept_picks)

    d = np.linalg.norm(ch_pos[:, None, :] - ch_pos[None, :, :], axis=2)

    dt = float(np.median(np.diff(times)))
    if not np.allclose(np.diff(times), dt, atol=1e-9):
        raise RuntimeError("Times are not uniformly spaced. Resample should fix this.")
    sfreq = 1.0 / dt
    half_win_samp = int(np.round((temporal_radius_ms / 1000.0) * sfreq))
    half_win_samp = max(0, half_win_samp)

    n_ch = d.shape[0]
    n_t  = len(times)

    chan_nbrs = []
    for ci in range(n_ch):
        nbr = np.where(d[ci, :] <= spatial_radius_m)[0]
        if nbr.size == 0:
            nbr = np.array([ci], dtype=int)
        chan_nbrs.append(nbr.astype(int))

    time_nbrs = []
    for ti in range(n_t):
        t0 = max(0, ti - half_win_samp)
        t1 = min(n_t, ti + half_win_samp + 1)
        time_nbrs.append(np.arange(t0, t1, dtype=int))

    centers, patches = [], []
    for ci in range(n_ch):
        for ti in range(n_t):
            centers.append((ci, ti))
            patches.append((chan_nbrs[ci], time_nbrs[ti]))

    return centers, patches, kept_picks, kept_info, half_win_samp

def _flatten_patch(X: np.ndarray, chan_inds: np.ndarray, time_inds: np.ndarray) -> np.ndarray:
    Xp = X[:, chan_inds, :][:, :, time_inds]
    return Xp.reshape(Xp.shape[0], -1)


# =============================================================================
# CV + scoring
# =============================================================================
def _make_cv(groups: np.ndarray | None, y: np.ndarray):
    if groups is not None:
        groups = np.asarray(groups)
        ok = ~pd.isna(groups)
        if ok.sum() == len(groups):
            n_groups = len(np.unique(groups))
            if n_groups >= 2:
                n_splits = min(N_SPLITS, n_groups)
                return GroupKFold(n_splits=n_splits), True
    return StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE), False

def _score_auc(est, X2d, y_true):
    if hasattr(est, "predict_proba"):
        s = est.predict_proba(X2d)[:, 1]
    elif hasattr(est, "decision_function"):
        s = est.decision_function(X2d)
    else:
        s = est.predict(X2d)
    return float(roc_auc_score(y_true, s))

def _score_metric(metric: str, est, X2d, y_true):
    if metric == "balanced_accuracy":
        yhat = est.predict(X2d)
        return float(balanced_accuracy_score(y_true, yhat))
    if metric == "roc_auc":
        return _score_auc(est, X2d, y_true)
    if metric == "accuracy":
        yhat = est.predict(X2d)
        return float(accuracy_score(y_true, yhat))
    raise ValueError(metric)


# =============================================================================
# Subject-level searchlight: metric-selectable
# =============================================================================
def subject_searchlight_metric(X_kept, y, centers, patches, *, metric: str, groups, shuffle: bool):
    rng = np.random.default_rng(RANDOM_STATE)
    y_use = rng.permutation(y) if shuffle else y

    pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            solver="liblinear",
            max_iter=2000,
            random_state=RANDOM_STATE,
            class_weight="balanced",
        )
    )

    cv, needs_groups = _make_cv(groups, y_use)
    splits = list(cv.split(X_kept, y_use, groups=groups if needs_groups else None))

    scores = np.full(len(centers), np.nan, float)

    for i, (chan_inds, time_inds) in enumerate(patches):
        Xi = _flatten_patch(X_kept, chan_inds, time_inds)
        if (not np.all(np.isfinite(Xi))) or Xi.shape[1] < 2 or np.nanstd(Xi) < 1e-12:
            continue

        fold_scores = []
        for tr, te in splits:
            est = clone(pipe)
            est.fit(Xi[tr], y_use[tr])
            fold_scores.append(_score_metric(metric, est, Xi[te], y_use[te]))
        scores[i] = float(np.mean(fold_scores))

    ch_ids = [c[0] for c in centers]
    t_ids  = [c[1] for c in centers]
    n_ch = max(ch_ids) + 1
    n_t  = max(t_ids) + 1
    return scores.reshape(n_ch, n_t)


# =============================================================================
# Group-level spatio-temporal cluster stats (TFCE)
# =============================================================================
def group_cluster_spatiotemporal(maps_by_subj, times, info_kept, *, chance: float, tail: int):
    tmask = (times >= TMIN_STAT) & (times <= TMAX_STAT)
    times_stat = times[tmask]
    X = maps_by_subj[:, :, tmask] - chance  # (n_subj, n_ch, n_tstat)

    ch_adj, _ = find_ch_adjacency(info_kept, ch_type="eeg")
    adjacency = combine_adjacency(len(times_stat), ch_adj)

    # MNE expects (n_subj, n_times, n_ch)
    X_mne = X.transpose(0, 2, 1)

    tfce_thresh = dict(start=0.0, step=0.2)
    try:
        T_obs, clusters, cluster_pv, H0 = spatio_temporal_cluster_1samp_test(
            X_mne, adjacency=adjacency,
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
            X_mne, adjacency=adjacency,
            n_permutations=N_PERM,
            threshold=None,
            tail=tail,
            n_jobs=1,
            seed=RANDOM_STATE,
            buffer_size=None,
            out_type="mask",
        )
        thresh_used = "threshold=None"

    p_map = np.ones_like(T_obs, float)  # (n_tstat, n_ch)
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
# Group-level 1D TFCE cluster stats (diagonal timecourses)
# =============================================================================
def group_cluster_time_1d_tfce(data_by_subj, times, *, chance: float, tail: int):
    """
    data_by_subj: (n_subj, n_times)
    tests (data - chance) over time with TFCE correction.
    Returns T_obs (n_times), clusters list, cluster_pv, and p_time (n_times) with min p per sample.
    """
    X = data_by_subj - chance
    tfce_thresh = dict(start=0.0, step=0.2)
    try:
        T_obs, clusters, cluster_pv, H0 = permutation_cluster_1samp_test(
            X,
            n_permutations=N_PERM,
            threshold=tfce_thresh,
            tail=tail,
            seed=RANDOM_STATE,
            n_jobs=1,
            out_type="mask",
        )
        thresh_used = "tfce"
    except Exception:
        T_obs, clusters, cluster_pv, H0 = permutation_cluster_1samp_test(
            X,
            n_permutations=N_PERM,
            threshold=None,
            tail=tail,
            seed=RANDOM_STATE,
            n_jobs=1,
            out_type="mask",
        )
        thresh_used = "threshold=None"

    p_time = np.ones_like(T_obs, float)
    for cl, p in zip(clusters, cluster_pv):
        if cl is None or not np.any(cl):
            continue
        p_time[cl] = np.minimum(p_time[cl], p)

    return dict(
        times=times,
        T_obs=T_obs,
        clusters=clusters,
        cluster_pv=cluster_pv,
        p_time=p_time,
        thresh_used=thresh_used,
    )


# =============================================================================
# Plotting utilities
# =============================================================================
def _robust_vlim(x, lo=5, hi=95, symmetric=False):
    x = np.asarray(x).ravel()
    x = x[np.isfinite(x)]
    if x.size == 0:
        return None
    v0 = float(np.percentile(x, lo))
    v1 = float(np.percentile(x, hi))
    if symmetric:
        m = max(abs(v0), abs(v1))
        return (-m, m)
    if np.isclose(v0, v1):
        eps = 1e-6
        v0, v1 = v0 - eps, v1 + eps
    return (v0, v1)

def p_to_signed_logp(p, sign, eps: float = 1e-300):
    p = np.asarray(p, float)
    sign = np.asarray(sign, float)
    p = np.clip(p, eps, 1.0)
    return np.sign(sign) * (-np.log10(p))

def top_percent_mask(values, top_pct=20.0):
    v = np.abs(np.asarray(values, float))
    thr = np.nanpercentile(v, 100.0 - top_pct)
    return v >= thr

def choose_plot_times(times_stat, p_map, requested_times, alpha, n_extra=4):
    """
    Always include requested_times (snapped to nearest available).
    Additionally include up to n_extra times within significant window(s) where any-channel p<alpha.
    """
    times_stat = np.asarray(times_stat, float)
    req = []
    for t in requested_times:
        ti = int(np.argmin(np.abs(times_stat - float(t))))
        req.append(float(times_stat[ti]))
    req = sorted(set(req))

    sig_any = (p_map < alpha).any(axis=1)
    extra = []
    if np.any(sig_any) and n_extra > 0:
        sig_inds = np.where(sig_any)[0]
        if sig_inds.size > 0:
            # pick roughly evenly spaced indices across the sig range
            pick = np.linspace(sig_inds.min(), sig_inds.max(), num=min(n_extra, sig_inds.size))
            pick = np.unique(np.round(pick).astype(int))
            extra = [float(times_stat[i]) for i in pick]

    out = sorted(set(req + extra))
    return out

def plot_topomap_field(info, data_ch, *, mask, title, out_path, vlim=None, cmap=None,
                       sensors=True, contours=6, colorbar_label=None):
    fig, ax = plt.subplots(figsize=(3.0, 2.8))
    im, cn = mne.viz.plot_topomap(
        data_ch, pos=info, mask=mask, show=False, axes=ax,
        contours=contours, sensors=sensors,
        outlines="head", extrapolate="head",
        vlim=vlim, cmap=cmap,
        mask_params=dict(marker="o", markerfacecolor="w", markeredgecolor="k",
                         linewidth=0, markersize=3),
    )
    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.05)
    if colorbar_label:
        cbar.set_label(colorbar_label)
    ax.set_title(title, pad=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_wta_topomap(info, data_ch, *, wta_mask, title, out_path, vlim=None, colorbar_label=None):
    plot_topomap_field(info, data_ch, mask=wta_mask, title=title, out_path=out_path,
                       vlim=vlim, contours=6, sensors=True, colorbar_label=colorbar_label)

def plot_topos_bundle(figs_dir, tag, analysis_name, kept_info, times_stat, chance,
                      mean_stat, T_obs, p_map, requested_times, alpha, n_extra):
    figs_dir.mkdir(exist_ok=True)

    effect = mean_stat - chance  # (n_ch, n_tstat)
    eff_vlim = _robust_vlim(effect, lo=5, hi=95, symmetric=False)
    t_vlim   = _robust_vlim(T_obs,  lo=5, hi=95, symmetric=True)

    signedlogp = p_to_signed_logp(p_map, T_obs)
    slogp_vlim = _robust_vlim(signedlogp, lo=5, hi=95, symmetric=True)

    plot_times = choose_plot_times(times_stat, p_map, requested_times, alpha, n_extra=n_extra)

    for t in plot_times:
        ti = int(np.argmin(np.abs(times_stat - t)))
        t_ms = int(round(times_stat[ti] * 1000))
        sig_mask = (p_map[ti, :] < alpha)

        plot_topomap_field(
            kept_info, effect[:, ti], mask=sig_mask,
            title=f"{analysis_name} effect (mean−chance) {t_ms} ms",
            out_path=figs_dir / f"{tag}_topo_effect_{t_ms}ms.png",
            vlim=eff_vlim, cmap=None, sensors=True, contours=6,
            colorbar_label="score − chance",
        )
        plot_topomap_field(
            kept_info, T_obs[ti, :], mask=sig_mask,
            title=f"{analysis_name} T {t_ms} ms",
            out_path=figs_dir / f"{tag}_topo_T_{t_ms}ms.png",
            vlim=t_vlim, cmap="RdBu_r", sensors=True, contours=6,
            colorbar_label="t value",
        )
        plot_topomap_field(
            kept_info, signedlogp[ti, :], mask=sig_mask,
            title=f"{analysis_name} signed −log10(p) {t_ms} ms",
            out_path=figs_dir / f"{tag}_topo_signedlogp_{t_ms}ms.png",
            vlim=slogp_vlim, cmap=None, sensors=True, contours=6,
            colorbar_label="signed −log10(p)",
        )

        wta20 = top_percent_mask(T_obs[ti, :], top_pct=20.0)
        wta20_sig = wta20 & sig_mask

        plot_wta_topomap(
            kept_info, T_obs[ti, :], wta_mask=wta20,
            title=f"{analysis_name} WTA top20% |T| {t_ms} ms",
            out_path=figs_dir / f"{tag}_topo_T_WTA20_{t_ms}ms.png",
            vlim=t_vlim, colorbar_label="t value",
        )
        plot_wta_topomap(
            kept_info, T_obs[ti, :], wta_mask=wta20_sig,
            title=f"{analysis_name} WTA top20% |T| (sig) {t_ms} ms",
            out_path=figs_dir / f"{tag}_topo_T_WTA20_sig_{t_ms}ms.png",
            vlim=t_vlim, colorbar_label="t value",
        )

def plot_channelmean_timecourse_with_sem(maps_by_subj, times, *, title, out_path,
                                        chance, ylabel, ylim):
    """
    SEM shading.
    maps_by_subj: (n_subj, n_ch, n_t) OR (n_subj, n_t)
    """
    X = np.asarray(maps_by_subj, float)
    if X.ndim == 3:
        subj_tc = np.nanmean(X, axis=1)  # (n_subj, n_t)
    elif X.ndim == 2:
        subj_tc = X
    else:
        raise ValueError("maps_by_subj must be (n_subj,n_ch,n_t) or (n_subj,n_t)")

    mean = subj_tc.mean(axis=0)
    sem  = subj_tc.std(axis=0, ddof=1) / np.sqrt(subj_tc.shape[0])

    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(times, mean, linewidth=2)
    ax.fill_between(times, mean - sem, mean + sem, alpha=0.20, label="SEM")
    ax.axhline(chance, linestyle="--", linewidth=1)
    ax.axvline(0, linestyle="--", linewidth=1)

    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    ax.legend(loc="best", frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

def plot_time_sig_bar_1d(times, p_time, *, alpha, out_png, title):
    sig = (p_time < alpha)
    fig, ax = plt.subplots(figsize=(7, 1.4))
    ax.fill_between(times, 0, 1, where=sig, alpha=0.9)
    ax.axvline(0, linestyle="--", linewidth=1)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)

def plot_time_sig_bar_spatiotemporal(times_stat, p_map, *, alpha, out_png, title):
    sig_any = (p_map < alpha).any(axis=1)  # (n_tstat,)
    fig, ax = plt.subplots(figsize=(7, 1.4))
    ax.fill_between(times_stat, 0, 1, where=sig_any, alpha=0.9)
    ax.axvline(0, linestyle="--", linewidth=1)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)

def plot_heatmap_score_minus_chance(M, times_hm, *, title, out_png, chance):
    """
    M: (n_train, n_test)
    plots (M - chance) with symmetric vlim; red=above chance
    """
    D = M - chance
    v = np.nanmax(np.abs(D))
    if not np.isfinite(v) or v < 1e-6:
        v = 0.01
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.imshow(
        D, origin="lower", aspect="auto",
        extent=[times_hm[0], times_hm[-1], times_hm[0], times_hm[-1]],
        vmin=-v, vmax=v, cmap="RdBu_r"
    )
    ax.set_title(title)
    ax.set_xlabel("Test time (s)")
    ax.set_ylabel("Train time (s)")
    ax.axhline(0, linestyle="--", linewidth=0.8)
    ax.axvline(0, linestyle="--", linewidth=0.8)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Score − chance (red = above)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


# =============================================================================
# Saving cluster table + summary
# =============================================================================
def save_cluster_table_and_summary(tag, out_dir, stats_out, times_stat, maps_stat, chance, included,
                                  spatial_radius_m, temporal_radius_ms):
    rows = []
    T_obs = stats_out["T_obs"]        # (n_tstat, n_ch)
    cluster_pv = stats_out["cluster_pv"]
    clusters   = stats_out["clusters"]

    for i, (cl_mask, p) in enumerate(zip(clusters, cluster_pv)):
        if cl_mask is None or not np.any(cl_mask):
            continue
        t_inds = np.where(np.any(cl_mask, axis=1))[0]
        t_start = float(times_stat[t_inds[0]])
        t_end   = float(times_stat[t_inds[-1]])
        dur_ms  = (t_end - t_start) * 1000.0
        ch_inds = np.where(np.any(cl_mask, axis=0))[0]
        n_ch = int(len(ch_inds))

        mass_sumT = float(T_obs[cl_mask].sum())
        maxT = float(T_obs[cl_mask].max())
        minT = float(T_obs[cl_mask].min())

        eff = maps_stat.transpose(0, 2, 1) - chance  # (n_subj, n_tstat, n_ch)
        mean_eff = float(eff[:, cl_mask].mean())
        sign = "pos" if mean_eff >= 0 else "neg"

        rows.append(dict(
            cluster=i, p_value=float(p), sign=sign,
            t_start_s=t_start, t_end_s=t_end, duration_ms=dur_ms,
            n_channels=n_ch,
            cluster_mass_sumT=mass_sumT,
            cluster_maxT=maxT,
            cluster_minT=minT,
            cluster_mean_effect_minus_chance=mean_eff
        ))

    df_cl = pd.DataFrame(rows).sort_values("p_value") if rows else pd.DataFrame(
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
        spatial_radius_m=float(spatial_radius_m),
        temporal_radius_ms=float(temporal_radius_ms),
        tmin_stat=float(TMIN_STAT),
        tmax_stat=float(TMAX_STAT),
    )
    with open(out_dir / f"{tag}_summary.json", "w") as f:
        json.dump(meta, f, indent=2)

def save_timecourse_cluster_table(tag, out_dir, stats_1d, *, chance, included):
    """
    Save TFCE cluster table for 1D timecourse stats.
    """
    rows = []
    T_obs = stats_1d["T_obs"]
    clusters = stats_1d["clusters"]
    cluster_pv = stats_1d["cluster_pv"]
    times = stats_1d["times"]

    for i, (cl, p) in enumerate(zip(clusters, cluster_pv)):
        if cl is None or not np.any(cl):
            continue
        t_inds = np.where(cl)[0]
        t_start = float(times[t_inds[0]])
        t_end   = float(times[t_inds[-1]])
        dur_ms  = (t_end - t_start) * 1000.0
        mass_sumT = float(T_obs[cl].sum())
        maxT = float(T_obs[cl].max())
        minT = float(T_obs[cl].min())
        rows.append(dict(
            cluster=i,
            p_value=float(p),
            t_start_s=t_start,
            t_end_s=t_end,
            duration_ms=dur_ms,
            cluster_mass_sumT=mass_sumT,
            cluster_maxT=maxT,
            cluster_minT=minT,
        ))

    df = pd.DataFrame(rows).sort_values("p_value") if rows else pd.DataFrame(
        columns=["cluster","p_value","t_start_s","t_end_s","duration_ms",
                 "cluster_mass_sumT","cluster_maxT","cluster_minT"]
    )
    df.to_csv(out_dir / f"{tag}_timecourse_cluster_table.csv", index=False)

    meta = dict(
        tag=tag,
        n_subjects=int(len(included)),
        n_times=int(len(times)),
        chance=float(chance),
        alpha=float(ALPHA_CLUSTER),
        thresh_used=stats_1d.get("thresh_used", None),
        n_clusters=int(len(cluster_pv)),
        min_cluster_p=float(np.min(cluster_pv)) if len(cluster_pv) else 1.0,
        included_subjects=included,
    )
    with open(out_dir / f"{tag}_timecourse_summary.json", "w") as f:
        json.dump(meta, f, indent=2)


# =============================================================================
# Temporal generalization (within-label) using MNE GeneralizingEstimator
# =============================================================================
def subject_temporal_generalization_matrix(X, y, groups, *, metric: str, shuffle: bool, time_idx: np.ndarray):
    rng = np.random.default_rng(RANDOM_STATE)
    y_use = rng.permutation(y) if shuffle else y

    pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            solver="liblinear",
            max_iter=2000,
            random_state=RANDOM_STATE,
            class_weight="balanced",
        )
    )

    Xs = X[:, :, time_idx]

    cv, needs_groups = _make_cv(groups, y_use)
    splits = list(cv.split(Xs, y_use, groups=groups if needs_groups else None))

    fold_mats = []
    for tr, te in splits:
        gen = GeneralizingEstimator(pipe, scoring=metric, n_jobs=1)
        gen.fit(Xs[tr], y_use[tr])
        M = gen.score(Xs[te], y_use[te])  # (n_train, n_test)
        fold_mats.append(M)

    M_mean = np.mean(np.stack(fold_mats, axis=0), axis=0)
    diag = np.diag(M_mean).copy()
    return M_mean, diag


# =============================================================================
# Cross-label generalization (money->pain, pain->money)
# =============================================================================
def select_trials_crossgen_binary(epo: mne.Epochs, control_by_other_train: bool, train: str):
    money_levels, pain_levels = _get_levels_from_epochs(epo)

    keep = np.isin(money_levels, BIN_KEEP_LEVELS) & np.isin(pain_levels, BIN_KEEP_LEVELS)
    epo_f = epo.copy()[keep]
    md_f = epo_f.metadata.reset_index(drop=True)

    m = money_levels[keep]
    p = pain_levels[keep]

    y_money = make_binary_labels(m.astype(float))
    y_pain  = make_binary_labels(p.astype(float))

    X = epo_f.get_data()
    times = epo_f.times.copy()

    if control_by_other_train:
        nuis_other = p if train == "money" else m
        X = residualize_X_by_other_level(X, nuis_other.astype(float))

    if train == "money":
        return X, y_money, y_pain, times, md_f
    else:
        return X, y_pain, y_money, times, md_f

def crossgen_temporal_generalization_matrix(X, y_train, y_test, groups, *,
                                            metric: str, shuffle_train: bool, time_idx: np.ndarray):
    rng = np.random.default_rng(RANDOM_STATE)
    y_tr_use = rng.permutation(y_train) if shuffle_train else y_train

    pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            solver="liblinear",
            max_iter=2000,
            random_state=RANDOM_STATE,
            class_weight="balanced",
        )
    )

    cv, needs_groups = _make_cv(groups, y_tr_use)
    splits = list(cv.split(X, y_tr_use, groups=groups if needs_groups else None))

    T = np.asarray(time_idx, int)
    Xs = X[:, :, T]
    fold_mats = []

    for tr, te in splits:
        Xtr = Xs[tr]
        Xte = Xs[te]
        ytr = y_tr_use[tr]
        yte = y_test[te]

        nT = Xtr.shape[2]
        M = np.zeros((nT, nT), float)

        for i in range(nT):
            est = clone(pipe)
            est.fit(Xtr[:, :, i], ytr)
            for j in range(nT):
                M[i, j] = _score_metric(metric, est, Xte[:, :, j], yte)

        fold_mats.append(M)

    M_mean = np.mean(np.stack(fold_mats, axis=0), axis=0)
    diag = np.diag(M_mean).copy()
    return M_mean, diag


# =============================================================================
# Core runner: decision searchlight for ONE analysis + ONE metric
# =============================================================================
def run_decision_searchlight_one(*, analysis_name: str, which: str, control_by_other: bool,
                                 metric: str, shuffle: bool, out_dir: Path):
    """
    metric: "balanced_accuracy" or "roc_auc"
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    dbg = out_dir / "debug"
    dbg.mkdir(exist_ok=True)

    subs = list_subjects(DERIV_DIR)
    included, skipped = [], []
    maps = []
    times_ref = None
    centers = patches = None
    kept_picks = None
    kept_info = None

    pbar = tqdm(subs, desc=f"{analysis_name}_{metric}{'_shuf' if shuffle else ''}",
                unit="sub", dynamic_ncols=True, leave=True)

    for sub in pbar:
        try:
            epo = load_decision_epochs(sub)
            beh = load_decision_beh(sub)
            epo = merge_beh_into_epochs_decision(epo, beh, sub=sub, debug_dir=dbg)

            if epo.metadata is None:
                raise RuntimeError("metadata None after merge")

            if "badtrial" in epo.metadata.columns:
                bad = epo.metadata["badtrial"].fillna(0).astype(int).to_numpy()
                if bad.sum() > 0:
                    epo = epo.copy()[bad == 0]

            if RESAMPLE_SFREQ is not None:
                epo = epo.copy().resample(RESAMPLE_SFREQ, npad="auto")

            X, y, times, md_used, info = select_trials_decision_binary(
                epo, which=which, control_resid_by_other=control_by_other
            )
            groups = md_used[KEY_BLOCK].to_numpy() if (KEY_BLOCK in md_used.columns) else None

            if times_ref is None:
                times_ref = times
                centers, patches, kept_picks, kept_info, half_win = build_spatiotemporal_searchlight_patches(
                    info, times, SPATIAL_RADIUS_M, TEMPORAL_RADIUS_MS
                )
                log_print(f"{sub}: built searchlight template | kept_ch={len(kept_picks)} | n_times={len(times)}")
            else:
                if len(times) != len(times_ref) or np.max(np.abs(times - times_ref)) > 1e-9:
                    raise RuntimeError("Time axis mismatch across subjects.")

            X_kept = X[:, kept_picks, :]

            score_map = subject_searchlight_metric(
                X_kept, y, centers, patches,
                metric=metric, groups=groups, shuffle=shuffle
            )

            maps.append(score_map)
            included.append(sub)
            pbar.set_postfix_str(f"{sub} | trials={len(y)} | mean={np.nanmean(score_map):.3f}")

        except Exception as e:
            skipped.append((sub, str(e)))
            log_print(f"Skipped {sub}: {e}")

    if len(maps) < 8:
        raise RuntimeError(f"{analysis_name}: too few subjects included (n={len(maps)})")

    maps = np.stack(maps, axis=0)   # (n_subj, n_ch, n_t)
    times = times_ref

    tail = 1  # decoding > chance
    stats_out = group_cluster_spatiotemporal(
        maps, times, kept_info, chance=CHANCE, tail=tail
    )
    times_stat = stats_out["times_stat"]

    tmask = (times >= TMIN_STAT) & (times <= TMAX_STAT)
    maps_stat = maps[:, :, tmask]
    mean_stat = np.nanmean(maps_stat, axis=0)  # (n_ch, n_tstat)

    tag = f"{analysis_name}_{metric}" + ("_shuffle" if shuffle else "")

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
        chance=CHANCE,
        metric=metric,
        spatial_radius_m=SPATIAL_RADIUS_M,
        temporal_radius_ms=TEMPORAL_RADIUS_MS,
        resample_sfreq=RESAMPLE_SFREQ if RESAMPLE_SFREQ is not None else -1,
        tmin_stat=TMIN_STAT,
        tmax_stat=TMAX_STAT,
        alpha_cluster=ALPHA_CLUSTER,
        which=which,
        control_by_other=bool(control_by_other),
        shuffle=bool(shuffle),
    )

    ylabel = "bAcc" if metric == "balanced_accuracy" else "AUC"
    ylim = (0.35, 0.85)

    plot_channelmean_timecourse_with_sem(
        maps, times,
        title=f"{analysis_name} | {ylabel} | {'SHUFFLED' if shuffle else 'REAL'}",
        out_path=out_dir / f"{tag}_channelmean_timecourse_SEM.png",
        chance=CHANCE, ylabel=ylabel, ylim=ylim
    )

    plot_time_sig_bar_spatiotemporal(
        times_stat, stats_out["p_map"],
        alpha=ALPHA_CLUSTER,
        out_png=out_dir / f"{tag}_sigbar_anychannel_TFCE.png",
        title=f"{analysis_name} | {ylabel}: any-channel sig (TFCE p≤{ALPHA_CLUSTER})"
    )

    figs_dir = out_dir / "figs_topo"
    plot_topos_bundle(
        figs_dir=figs_dir,
        tag=tag,
        analysis_name=f"{analysis_name} | {ylabel} | {'SHUFFLED' if shuffle else 'REAL'}",
        kept_info=kept_info,
        times_stat=times_stat,
        chance=CHANCE,
        mean_stat=mean_stat,
        T_obs=stats_out["T_obs"],
        p_map=stats_out["p_map"],
        requested_times=TOPO_TIMES_S,
        alpha=ALPHA_CLUSTER,
        n_extra=N_EXTRA_SIG_TOPO_TIMES,
    )

    save_cluster_table_and_summary(
        tag=tag,
        out_dir=out_dir,
        stats_out=stats_out,
        times_stat=times_stat,
        maps_stat=maps_stat,
        chance=CHANCE,
        included=included,
        spatial_radius_m=SPATIAL_RADIUS_M,
        temporal_radius_ms=TEMPORAL_RADIUS_MS,
    )

    min_p = float(np.min(stats_out["cluster_pv"])) if len(stats_out["cluster_pv"]) else 1.0
    log_print(f"DONE {tag}: included={len(included)} | min cluster p={min_p:.6f} | thresh={stats_out['thresh_used']}")

    return dict(
        tag=tag,
        maps=maps,
        times=times,
        kept_info=kept_info,
        stats_out=stats_out,
        included=included,
        skipped=skipped,
    )


# =============================================================================
# Temporal generalization runner (within-label) + diagonal stats vs chance
# =============================================================================
def run_withinlabel_heatmaps(*, tag_prefix: str, which: str, control_by_other: bool,
                             shuffle: bool, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    dbg = out_dir / "debug"
    dbg.mkdir(exist_ok=True)

    subs = list_subjects(DERIV_DIR)
    included, skipped = [], []
    times_ref = None

    mats_bacc, mats_auc = [], []
    diag_bacc, diag_auc = [], []

    pbar = tqdm(subs, desc=f"{tag_prefix}_within_heatmaps{'_shuf' if shuffle else ''}",
                unit="sub", dynamic_ncols=True, leave=True)

    for sub in pbar:
        try:
            epo = load_decision_epochs(sub)
            beh = load_decision_beh(sub)
            epo = merge_beh_into_epochs_decision(epo, beh, sub=sub, debug_dir=dbg)

            if "badtrial" in epo.metadata.columns:
                bad = epo.metadata["badtrial"].fillna(0).astype(int).to_numpy()
                if bad.sum() > 0:
                    epo = epo.copy()[bad == 0]

            if RESAMPLE_SFREQ is not None:
                epo = epo.copy().resample(RESAMPLE_SFREQ, npad="auto")

            X, y, times, md_used, info = select_trials_decision_binary(
                epo, which=which, control_resid_by_other=control_by_other
            )
            groups = md_used[KEY_BLOCK].to_numpy() if (KEY_BLOCK in md_used.columns) else None

            if times_ref is None:
                times_ref = times
            else:
                if len(times) != len(times_ref) or np.max(np.abs(times - times_ref)) > 1e-9:
                    raise RuntimeError("Time axis mismatch across subjects.")

            tmask = (times >= TMIN_STAT) & (times <= TMAX_STAT)
            idx = np.where(tmask)[0][::max(1, int(HEATMAP_DECIM))]
            times_hm = times[idx]

            M_bacc, d_bacc = subject_temporal_generalization_matrix(
                X, y, groups, metric="balanced_accuracy",
                shuffle=shuffle, time_idx=idx
            )
            M_auc, d_auc = subject_temporal_generalization_matrix(
                X, y, groups, metric="roc_auc",
                shuffle=shuffle, time_idx=idx
            )

            mats_bacc.append(M_bacc); mats_auc.append(M_auc)
            diag_bacc.append(d_bacc); diag_auc.append(d_auc)
            included.append(sub)

        except Exception as e:
            skipped.append((sub, str(e)))
            log_print(f"Skipped {sub} within-heatmaps: {e}")

    if len(included) < 8:
        raise RuntimeError(f"withinlabel heatmaps: too few subjects (n={len(included)})")

    mats_bacc = np.stack(mats_bacc, axis=0)  # (n_sub, nT, nT)
    mats_auc  = np.stack(mats_auc,  axis=0)
    diag_bacc = np.stack(diag_bacc, axis=0) # (n_sub, nT)
    diag_auc  = np.stack(diag_auc,  axis=0)

    M_bacc_mean = mats_bacc.mean(axis=0)
    M_auc_mean  = mats_auc.mean(axis=0)

    suffix = "_shuffle" if shuffle else ""
    out_tag = f"{tag_prefix}{suffix}"

    # diagonal TFCE stats vs chance (1D)
    stats_diag_bacc = group_cluster_time_1d_tfce(diag_bacc, times_hm, chance=CHANCE, tail=1)
    stats_diag_auc  = group_cluster_time_1d_tfce(diag_auc,  times_hm, chance=CHANCE, tail=1)

    np.savez(
        out_dir / f"{out_tag}_withinlabel_heatmaps.npz",
        mats_bacc=mats_bacc,
        mats_auc=mats_auc,
        mean_bacc=M_bacc_mean,
        mean_auc=M_auc_mean,
        diag_bacc=diag_bacc,
        diag_auc=diag_auc,
        times=times_hm,
        included=np.array(included, dtype=object),
        skipped=np.array(skipped, dtype=object),
        chance=CHANCE,
        decim=int(HEATMAP_DECIM),
        which=which,
        control_by_other=bool(control_by_other),
        shuffle=bool(shuffle),
        # diag stats
        diag_bacc_T=stats_diag_bacc["T_obs"],
        diag_bacc_p=stats_diag_bacc["p_time"],
        diag_bacc_thresh=stats_diag_bacc["thresh_used"],
        diag_auc_T=stats_diag_auc["T_obs"],
        diag_auc_p=stats_diag_auc["p_time"],
        diag_auc_thresh=stats_diag_auc["thresh_used"],
    )

    plot_heatmap_score_minus_chance(
        M_bacc_mean, times_hm,
        title=f"{tag_prefix} within-label bAcc heatmap (score−chance)",
        out_png=out_dir / f"{out_tag}_within_bacc_heatmap.png",
        chance=CHANCE
    )
    plot_heatmap_score_minus_chance(
        M_auc_mean, times_hm,
        title=f"{tag_prefix} within-label AUC heatmap (score−chance)",
        out_png=out_dir / f"{out_tag}_within_auc_heatmap.png",
        chance=CHANCE
    )

    # diagonal timecourses + SEM + TFCE sigbar
    plot_channelmean_timecourse_with_sem(
        diag_bacc, times_hm,
        title=f"{tag_prefix} diagonal bAcc (SEM) | {'SHUF' if shuffle else 'REAL'}",
        out_path=out_dir / f"{out_tag}_diag_bacc_timecourse.png",
        chance=CHANCE, ylabel="bAcc", ylim=(0.35, 0.85)
    )
    plot_time_sig_bar_1d(
        times_hm, stats_diag_bacc["p_time"],
        alpha=ALPHA_CLUSTER,
        out_png=out_dir / f"{out_tag}_diag_bacc_sigbar_TFCE.png",
        title=f"{tag_prefix} diag bAcc sig over time (TFCE p≤{ALPHA_CLUSTER})"
    )
    save_timecourse_cluster_table(f"{out_tag}_diag_bacc", out_dir, stats_diag_bacc, chance=CHANCE, included=included)

    plot_channelmean_timecourse_with_sem(
        diag_auc, times_hm,
        title=f"{tag_prefix} diagonal AUC (SEM) | {'SHUF' if shuffle else 'REAL'}",
        out_path=out_dir / f"{out_tag}_diag_auc_timecourse.png",
        chance=CHANCE, ylabel="AUC", ylim=(0.35, 0.85)
    )
    plot_time_sig_bar_1d(
        times_hm, stats_diag_auc["p_time"],
        alpha=ALPHA_CLUSTER,
        out_png=out_dir / f"{out_tag}_diag_auc_sigbar_TFCE.png",
        title=f"{tag_prefix} diag AUC sig over time (TFCE p≤{ALPHA_CLUSTER})"
    )
    save_timecourse_cluster_table(f"{out_tag}_diag_auc", out_dir, stats_diag_auc, chance=CHANCE, included=included)

    return dict(
        out_tag=out_tag,
        included=included,
        times_hm=times_hm,
        diag_bacc=diag_bacc,
        diag_auc=diag_auc,
        stats_diag_bacc=stats_diag_bacc,
        stats_diag_auc=stats_diag_auc,
    )


# =============================================================================
# Cross-label heatmaps 
# =============================================================================
def run_crosslabel_heatmaps(*, tag_prefix: str, train: str, control_by_other_train: bool,
                            shuffle: bool, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    dbg = out_dir / "debug"
    dbg.mkdir(exist_ok=True)

    direction = f"{train}_to_{'pain' if train=='money' else 'money'}"

    subs = list_subjects(DERIV_DIR)
    included, skipped = [], []
    times_ref = None

    mats_bacc, mats_auc = [], []
    diag_bacc, diag_auc = [], []

    pbar = tqdm(subs, desc=f"{tag_prefix}_{direction}{'_shuf' if shuffle else ''}",
                unit="sub", dynamic_ncols=True, leave=True)

    for sub in pbar:
        try:
            epo = load_decision_epochs(sub)
            beh = load_decision_beh(sub)
            epo = merge_beh_into_epochs_decision(epo, beh, sub=sub, debug_dir=dbg)

            if "badtrial" in epo.metadata.columns:
                bad = epo.metadata["badtrial"].fillna(0).astype(int).to_numpy()
                if bad.sum() > 0:
                    epo = epo.copy()[bad == 0]

            if RESAMPLE_SFREQ is not None:
                epo = epo.copy().resample(RESAMPLE_SFREQ, npad="auto")

            X, y_tr, y_te, times, md_used = select_trials_crossgen_binary(
                epo, control_by_other_train=control_by_other_train, train=train
            )
            groups = md_used[KEY_BLOCK].to_numpy() if (KEY_BLOCK in md_used.columns) else None

            if times_ref is None:
                times_ref = times
            else:
                if len(times) != len(times_ref) or np.max(np.abs(times - times_ref)) > 1e-9:
                    raise RuntimeError("Time axis mismatch across subjects.")

            tmask = (times >= TMIN_STAT) & (times <= TMAX_STAT)
            idx = np.where(tmask)[0][::max(1, int(HEATMAP_DECIM))]
            times_hm = times[idx]

            M_bacc, d_bacc = crossgen_temporal_generalization_matrix(
                X, y_tr, y_te, groups,
                metric="balanced_accuracy",
                shuffle_train=shuffle,
                time_idx=idx
            )
            M_auc, d_auc = crossgen_temporal_generalization_matrix(
                X, y_tr, y_te, groups,
                metric="roc_auc",
                shuffle_train=shuffle,
                time_idx=idx
            )

            mats_bacc.append(M_bacc); mats_auc.append(M_auc)
            diag_bacc.append(d_bacc); diag_auc.append(d_auc)
            included.append(sub)

        except Exception as e:
            skipped.append((sub, str(e)))
            log_print(f"Skipped {sub} crosslabel heatmaps: {e}")

    if len(included) < 8:
        raise RuntimeError(f"crosslabel heatmaps: too few subjects (n={len(included)})")

    mats_bacc = np.stack(mats_bacc, axis=0)
    mats_auc  = np.stack(mats_auc,  axis=0)
    diag_bacc = np.stack(diag_bacc, axis=0)
    diag_auc  = np.stack(diag_auc,  axis=0)

    M_bacc_mean = mats_bacc.mean(axis=0)
    M_auc_mean  = mats_auc.mean(axis=0)

    suffix = "_shuffle" if shuffle else ""
    ctrl = "ctrlOtherTrain" if control_by_other_train else "raw"
    out_tag = f"{tag_prefix}_{ctrl}_{direction}{suffix}"

    np.savez(
        out_dir / f"{out_tag}_crosslabel_heatmaps.npz",
        mats_bacc=mats_bacc,
        mats_auc=mats_auc,
        mean_bacc=M_bacc_mean,
        mean_auc=M_auc_mean,
        diag_bacc=diag_bacc,
        diag_auc=diag_auc,
        times=times_hm,
        included=np.array(included, dtype=object),
        skipped=np.array(skipped, dtype=object),
        chance=CHANCE,
        decim=int(HEATMAP_DECIM),
        direction=direction,
        train=train,
        control_by_other_train=bool(control_by_other_train),
        shuffle=bool(shuffle),
    )

    plot_heatmap_score_minus_chance(
        M_bacc_mean, times_hm,
        title=f"{direction} bAcc heatmap (score−chance) | {ctrl} | {'SHUF' if shuffle else 'REAL'}",
        out_png=out_dir / f"{out_tag}_bacc_heatmap.png",
        chance=CHANCE
    )
    plot_heatmap_score_minus_chance(
        M_auc_mean, times_hm,
        title=f"{direction} AUC heatmap (score−chance) | {ctrl} | {'SHUF' if shuffle else 'REAL'}",
        out_png=out_dir / f"{out_tag}_auc_heatmap.png",
        chance=CHANCE
    )


# =============================================================================
# Difference maps: money - pain (paired)
# PLUS: special masking requirement (money>pain AND both money/pain significant vs chance)
# =============================================================================
def run_money_minus_pain_difference(*, tag_prefix: str, control_by_other: bool, metric: str, shuffle: bool, out_dir: Path):
    """
    Paired difference (money_map - pain_map) tested vs 0 over channel×time.
    Also computes money vs chance and pain vs chance TFCE maps.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    dbg = out_dir / "debug"
    dbg.mkdir(exist_ok=True)

    subs = list_subjects(DERIV_DIR)
    included, skipped = [], []
    diffs = []
    maps_money = []
    maps_pain = []

    times_ref = None
    kept_info = None
    kept_picks = None
    centers = patches = None

    pbar = tqdm(subs, desc=f"{tag_prefix}_money_minus_pain_{metric}{'_shuf' if shuffle else ''}",
                unit="sub", dynamic_ncols=True, leave=True)

    for sub in pbar:
        try:
            epo = load_decision_epochs(sub)
            beh = load_decision_beh(sub)
            epo = merge_beh_into_epochs_decision(epo, beh, sub=sub, debug_dir=dbg)

            if "badtrial" in epo.metadata.columns:
                bad = epo.metadata["badtrial"].fillna(0).astype(int).to_numpy()
                if bad.sum() > 0:
                    epo = epo.copy()[bad == 0]

            if RESAMPLE_SFREQ is not None:
                epo = epo.copy().resample(RESAMPLE_SFREQ, npad="auto")

            # money
            X_m, y_m, times, md_m, info = select_trials_decision_binary(
                epo, which="money", control_resid_by_other=control_by_other
            )
            groups_m = md_m[KEY_BLOCK].to_numpy() if (KEY_BLOCK in md_m.columns) else None

            # pain
            X_p, y_p, times2, md_p, info2 = select_trials_decision_binary(
                epo, which="pain", control_resid_by_other=control_by_other
            )
            groups_p = md_p[KEY_BLOCK].to_numpy() if (KEY_BLOCK in md_p.columns) else None

            if times_ref is None:
                times_ref = times
                centers, patches, kept_picks, kept_info, half_win = build_spatiotemporal_searchlight_patches(
                    info, times, SPATIAL_RADIUS_M, TEMPORAL_RADIUS_MS
                )
            else:
                if len(times) != len(times_ref) or np.max(np.abs(times - times_ref)) > 1e-9:
                    raise RuntimeError("Time axis mismatch across subjects.")
                if len(times2) != len(times_ref) or np.max(np.abs(times2 - times_ref)) > 1e-9:
                    raise RuntimeError("Time axis mismatch (pain vs ref).")

            Xm = X_m[:, kept_picks, :]
            Xp = X_p[:, kept_picks, :]

            map_m = subject_searchlight_metric(Xm, y_m, centers, patches, metric=metric, groups=groups_m, shuffle=shuffle)
            map_p = subject_searchlight_metric(Xp, y_p, centers, patches, metric=metric, groups=groups_p, shuffle=shuffle)

            maps_money.append(map_m)
            maps_pain.append(map_p)
            diffs.append(map_m - map_p)
            included.append(sub)

        except Exception as e:
            skipped.append((sub, str(e)))
            log_print(f"Skipped {sub} diff: {e}")

    if len(included) < 8:
        raise RuntimeError(f"diff maps: too few subjects (n={len(included)})")

    diffs = np.stack(diffs, axis=0)
    maps_money = np.stack(maps_money, axis=0)
    maps_pain  = np.stack(maps_pain,  axis=0)
    times = times_ref

    # Stats:
    # 1) DIFF vs 0 (two-sided)
    stats_diff = group_cluster_spatiotemporal(
        diffs, times, kept_info, chance=0.0, tail=0
    )
    # 2) money vs chance (one-sided > chance)
    stats_money = group_cluster_spatiotemporal(
        maps_money, times, kept_info, chance=CHANCE, tail=1
    )
    # 3) pain vs chance (one-sided > chance)
    stats_pain = group_cluster_spatiotemporal(
        maps_pain, times, kept_info, chance=CHANCE, tail=1
    )

    times_stat = stats_diff["times_stat"]
    tmask = (times >= TMIN_STAT) & (times <= TMAX_STAT)
    diffs_stat = diffs[:, :, tmask]
    money_stat = maps_money[:, :, tmask]
    pain_stat  = maps_pain[:, :, tmask]

    mean_diff = np.nanmean(diffs_stat, axis=0)   # (n_ch, n_tstat)
    mean_money = np.nanmean(money_stat, axis=0)
    mean_pain  = np.nanmean(pain_stat, axis=0)

    suffix = "_shuffle" if shuffle else ""
    ctrl = "ctrlOther" if control_by_other else "raw"
    tag = f"{tag_prefix}_{ctrl}_money_minus_pain_{metric}{suffix}"

    np.savez(
        out_dir / f"{tag}_group_results.npz",
        diffs_by_subj=diffs,
        money_by_subj=maps_money,
        pain_by_subj=maps_pain,
        times=times,
        ch_names=np.array(kept_info["ch_names"], dtype=object),
        included=np.array(included, dtype=object),
        skipped=np.array(skipped, dtype=object),
        # diff stats
        diff_T=stats_diff["T_obs"],
        diff_p=stats_diff["p_map"],
        diff_cluster_pv=stats_diff["cluster_pv"],
        diff_thresh_used=stats_diff["thresh_used"],
        # money stats
        money_T=stats_money["T_obs"],
        money_p=stats_money["p_map"],
        money_cluster_pv=stats_money["cluster_pv"],
        money_thresh_used=stats_money["thresh_used"],
        # pain stats
        pain_T=stats_pain["T_obs"],
        pain_p=stats_pain["p_map"],
        pain_cluster_pv=stats_pain["cluster_pv"],
        pain_thresh_used=stats_pain["thresh_used"],
        metric=metric,
        control_by_other=bool(control_by_other),
        shuffle=bool(shuffle),
        alpha_cluster=ALPHA_CLUSTER,
    )

    # --- Topos:
    figs_dir = out_dir / "figs_topo"
    figs_dir.mkdir(exist_ok=True)

    # Standard DIFF topos (mask = diff sig)
    plot_topos_bundle(
        figs_dir=figs_dir / "diff_sig",
        tag=tag + "_diffsig",
        analysis_name=f"DIFF money−pain | {metric} | {ctrl} | {'SHUF' if shuffle else 'REAL'} (mask=diff TFCE)",
        kept_info=kept_info,
        times_stat=times_stat,
        chance=0.0,
        mean_stat=mean_diff,
        T_obs=stats_diff["T_obs"],
        p_map=stats_diff["p_map"],
        requested_times=TOPO_TIMES_S,
        alpha=ALPHA_CLUSTER,
        n_extra=N_EXTRA_SIG_TOPO_TIMES,
    )

    # Special “money better & both individually sig” mask
    # Align money/pain p-maps to diff times_stat (they share times_stat by construction)
    p_money = stats_money["p_map"]
    p_pain  = stats_pain["p_map"]
    p_diff  = stats_diff["p_map"]

    # We'll create a derived p_map for masking but keep T as diff-T for visualization
    # mask = (money>pain) & (money sig) & (pain sig)
    money_better = (mean_diff > 0.0).T  # (n_tstat, n_ch)
    both_sig = (p_money < ALPHA_CLUSTER) & (p_pain < ALPHA_CLUSTER)
    mask_special = money_better & both_sig

    # To reuse plot_topos_bundle, we pass p_map=1 for non-masked entries:
    p_map_special = np.ones_like(p_diff)
    p_map_special[mask_special] = 0.0  # mark as "sig" for plotting

    plot_topos_bundle(
        figs_dir=figs_dir / "money_better_AND_bothSig",
        tag=tag + "_moneyBetter_bothSig",
        analysis_name=f"DIFF money−pain | {metric} | {ctrl} | {'SHUF' if shuffle else 'REAL'} (mask=money>pain & moneySig & painSig)",
        kept_info=kept_info,
        times_stat=times_stat,
        chance=0.0,
        mean_stat=mean_diff,
        T_obs=stats_diff["T_obs"],
        p_map=p_map_special,
        requested_times=TOPO_TIMES_S,
        alpha=ALPHA_CLUSTER,
        n_extra=N_EXTRA_SIG_TOPO_TIMES,
    )

    # Also: strict version additionally requiring DIFF sig (optional but useful)
    mask_special_and_diff = mask_special & (p_diff < ALPHA_CLUSTER)
    p_map_special_and_diff = np.ones_like(p_diff)
    p_map_special_and_diff[mask_special_and_diff] = 0.0

    plot_topos_bundle(
        figs_dir=figs_dir / "money_better_AND_bothSig_AND_diffSig",
        tag=tag + "_moneyBetter_bothSig_diffSig",
        analysis_name=f"DIFF money−pain | {metric} | {ctrl} | {'SHUF' if shuffle else 'REAL'} (mask=money>pain & moneySig & painSig & diffSig)",
        kept_info=kept_info,
        times_stat=times_stat,
        chance=0.0,
        mean_stat=mean_diff,
        T_obs=stats_diff["T_obs"],
        p_map=p_map_special_and_diff,
        requested_times=TOPO_TIMES_S,
        alpha=ALPHA_CLUSTER,
        n_extra=N_EXTRA_SIG_TOPO_TIMES,
    )

    # Save cluster tables for DIFF (primary)
    save_cluster_table_and_summary(
        tag=tag + "_diff",
        out_dir=out_dir,
        stats_out=stats_diff,
        times_stat=times_stat,
        maps_stat=diffs_stat,
        chance=0.0,
        included=included,
        spatial_radius_m=SPATIAL_RADIUS_M,
        temporal_radius_ms=TEMPORAL_RADIUS_MS,
    )

    # Sig bar for DIFF any-channel
    plot_time_sig_bar_spatiotemporal(
        times_stat, stats_diff["p_map"],
        alpha=ALPHA_CLUSTER,
        out_png=out_dir / f"{tag}_diff_sigbar_anychannel_TFCE.png",
        title=f"DIFF money−pain | any-channel sig (TFCE p≤{ALPHA_CLUSTER})"
    )

    min_p = float(np.min(stats_diff["cluster_pv"])) if len(stats_diff["cluster_pv"]) else 1.0
    log_print(f"DONE {tag}: included={len(included)} | min DIFF cluster p={min_p:.6f}")


# =============================================================================
# DIAGONAL DIFFERENCE STATS (money diag - pain diag) + plots
# =============================================================================
def run_diag_money_minus_pain_stats(*, tag_prefix: str,
                                   money_res: dict, pain_res: dict,
                                   out_dir: Path):
    """
    money_res/pain_res: returns from run_withinlabel_heatmaps() for same ctrl/shuffle.
    Produces:
    - diff timecourse (money_diag - pain_diag) with SEM
    - TFCE cluster stats over time vs 0
    - also masks where each individually > chance TFCE (optional plots saved already)
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    times = money_res["times_hm"]
    if len(times) != len(pain_res["times_hm"]) or np.max(np.abs(times - pain_res["times_hm"])) > 1e-9:
        raise RuntimeError("money/pain diagonal time axes mismatch")

    included_m = money_res["included"]
    included_p = pain_res["included"]
    included = sorted(set(included_m).intersection(set(included_p)))
    if len(included) < 8:
        raise RuntimeError(f"diag diff: too few overlapping subjects (n={len(included)})")

    # Re-index to intersection (keep order stable by included list above)
    idx_m = [included_m.index(s) for s in included]
    idx_p = [included_p.index(s) for s in included]

    money_bacc = money_res["diag_bacc"][idx_m, :]
    pain_bacc  = pain_res["diag_bacc"][idx_p, :]
    money_auc  = money_res["diag_auc"][idx_m, :]
    pain_auc   = pain_res["diag_auc"][idx_p, :]

    # Differences
    diff_bacc = money_bacc - pain_bacc
    diff_auc  = money_auc  - pain_auc

    # Stats vs 0 (two-sided, because could flip)
    stats_diff_bacc = group_cluster_time_1d_tfce(diff_bacc, times, chance=0.0, tail=0)
    stats_diff_auc  = group_cluster_time_1d_tfce(diff_auc,  times, chance=0.0, tail=0)

    # Save
    np.savez(
        out_dir / f"{tag_prefix}_diag_money_minus_pain_stats.npz",
        included=np.array(included, dtype=object),
        times=times,
        diff_bacc=diff_bacc,
        diff_auc=diff_auc,
        # stats bacc
        bacc_T=stats_diff_bacc["T_obs"],
        bacc_p=stats_diff_bacc["p_time"],
        bacc_cluster_pv=stats_diff_bacc["cluster_pv"],
        bacc_thresh=stats_diff_bacc["thresh_used"],
        # stats auc
        auc_T=stats_diff_auc["T_obs"],
        auc_p=stats_diff_auc["p_time"],
        auc_cluster_pv=stats_diff_auc["cluster_pv"],
        auc_thresh=stats_diff_auc["thresh_used"],
        alpha=ALPHA_CLUSTER,
    )

    # Plot diff timecourse SEM
    plot_channelmean_timecourse_with_sem(
        diff_bacc, times,
        title=f"{tag_prefix} DIAG (money − pain) bAcc (SEM)",
        out_path=out_dir / f"{tag_prefix}_diag_diff_bacc_timecourse.png",
        chance=0.0, ylabel="bAcc diff", ylim=(-0.20, 0.20)
    )
    plot_time_sig_bar_1d(
        times, stats_diff_bacc["p_time"],
        alpha=ALPHA_CLUSTER,
        out_png=out_dir / f"{tag_prefix}_diag_diff_bacc_sigbar_TFCE.png",
        title=f"{tag_prefix} DIAG (money−pain) bAcc diff sig (TFCE p≤{ALPHA_CLUSTER})"
    )
    save_timecourse_cluster_table(f"{tag_prefix}_diag_diff_bacc", out_dir, stats_diff_bacc, chance=0.0, included=included)

    plot_channelmean_timecourse_with_sem(
        diff_auc, times,
        title=f"{tag_prefix} DIAG (money − pain) AUC (SEM)",
        out_path=out_dir / f"{tag_prefix}_diag_diff_auc_timecourse.png",
        chance=0.0, ylabel="AUC diff", ylim=(-0.20, 0.20)
    )
    plot_time_sig_bar_1d(
        times, stats_diff_auc["p_time"],
        alpha=ALPHA_CLUSTER,
        out_png=out_dir / f"{tag_prefix}_diag_diff_auc_sigbar_TFCE.png",
        title=f"{tag_prefix} DIAG (money−pain) AUC diff sig (TFCE p≤{ALPHA_CLUSTER})"
    )
    save_timecourse_cluster_table(f"{tag_prefix}_diag_diff_auc", out_dir, stats_diff_auc, chance=0.0, included=included)

    minp_b = float(np.min(stats_diff_bacc["cluster_pv"])) if len(stats_diff_bacc["cluster_pv"]) else 1.0
    minp_a = float(np.min(stats_diff_auc["cluster_pv"])) if len(stats_diff_auc["cluster_pv"]) else 1.0
    log_print(f"DONE {tag_prefix} diag money−pain: minp bAcc={minp_b:.6f} | minp AUC={minp_a:.6f}")


# =============================================================================
# Main
# =============================================================================
def main():
    mne.set_log_level("WARNING")
    log_print(f"\n=== mvpa_decision_step2_searchlight_binaryonly START ===")
    log_print(f"DATA_DIR: {RAW_DIR}")
    log_print(f"OUT_DIR:  {OUT_DIR}")
    log_print(f"RESAMPLE_SFREQ: {RESAMPLE_SFREQ}")
    log_print(f"N_PERM: {N_PERM} | ALPHA_CLUSTER: {ALPHA_CLUSTER} | STATS WINDOW: [{TMIN_STAT},{TMAX_STAT}] s")
    log_print(f"Searchlight radii: spatial={SPATIAL_RADIUS_M} m | temporal={TEMPORAL_RADIUS_MS} ms")
    log_print(f"Heatmap decim: {HEATMAP_DECIM}")
    log_print(f"Topo times: {TOPO_TIMES_S} (+{N_EXTRA_SIG_TOPO_TIMES} auto if sig)\n")

    OUT_SL = OUT_DIR / "searchlight_maps"
    OUT_HM = OUT_DIR / "temporal_generalization_heatmaps"
    OUT_XG = OUT_DIR / "cross_label_generalization"
    OUT_DF = OUT_DIR / "money_minus_pain_difference"
    OUT_DD = OUT_DIR / "diag_money_minus_pain_difference"

    for d in [OUT_SL, OUT_HM, OUT_XG, OUT_DF, OUT_DD]:
        d.mkdir(exist_ok=True, parents=True)

    def run_pair(analysis_name, which, ctrl, shuffle, out_subdir):
        run_decision_searchlight_one(
            analysis_name=analysis_name, which=which, control_by_other=ctrl,
            metric="balanced_accuracy", shuffle=shuffle,
            out_dir=out_subdir
        )
        run_decision_searchlight_one(
            analysis_name=analysis_name, which=which, control_by_other=ctrl,
            metric="roc_auc", shuffle=shuffle,
            out_dir=out_subdir
        )

    # -------------------------
    # RAW
    if RUN_RAW:
        for shuffle in ([False, True] if RUN_SHUFFLE else [False]):
            run_pair("decision_money_binary_raw", "money", False, shuffle, OUT_SL / "raw")
            run_pair("decision_pain_binary_raw",  "pain",  False, shuffle, OUT_SL / "raw")

            money_hm_res = pain_hm_res = None
            if RUN_WITHINLABEL_HEATMAPS:
                money_hm_res = run_withinlabel_heatmaps(
                    tag_prefix="decision_money_raw", which="money", control_by_other=False,
                    shuffle=shuffle, out_dir=OUT_HM / "raw"
                )
                pain_hm_res = run_withinlabel_heatmaps(
                    tag_prefix="decision_pain_raw", which="pain", control_by_other=False,
                    shuffle=shuffle, out_dir=OUT_HM / "raw"
                )

            if RUN_DIAG_DIFF_STATS and (money_hm_res is not None) and (pain_hm_res is not None):
                run_diag_money_minus_pain_stats(
                    tag_prefix=f"raw{'_shuffle' if shuffle else ''}",
                    money_res=money_hm_res,
                    pain_res=pain_hm_res,
                    out_dir=OUT_DD / "raw"
                )

            if RUN_CROSS_LABEL:
                run_crosslabel_heatmaps(
                    tag_prefix="xgen", train="money", control_by_other_train=False,
                    shuffle=shuffle, out_dir=OUT_XG / "raw"
                )
                run_crosslabel_heatmaps(
                    tag_prefix="xgen", train="pain", control_by_other_train=False,
                    shuffle=shuffle, out_dir=OUT_XG / "raw"
                )

            if RUN_DIFF_MONEY_MINUS_PAIN:
                run_money_minus_pain_difference(
                    tag_prefix="diff", control_by_other=False,
                    metric="balanced_accuracy", shuffle=shuffle,
                    out_dir=OUT_DF / "raw"
                )
                run_money_minus_pain_difference(
                    tag_prefix="diff", control_by_other=False,
                    metric="roc_auc", shuffle=shuffle,
                    out_dir=OUT_DF / "raw"
                )

    # -------------------------
    # CTRLOTHER (residualize by other cue)
    if RUN_CTRLOTHER:
        for shuffle in ([False, True] if RUN_SHUFFLE else [False]):
            run_pair("decision_money_binary_ctrlOther", "money", True, shuffle, OUT_SL / "ctrlOther")
            run_pair("decision_pain_binary_ctrlOther",  "pain",  True, shuffle, OUT_SL / "ctrlOther")

            money_hm_res = pain_hm_res = None
            if RUN_WITHINLABEL_HEATMAPS:
                money_hm_res = run_withinlabel_heatmaps(
                    tag_prefix="decision_money_ctrlOther", which="money", control_by_other=True,
                    shuffle=shuffle, out_dir=OUT_HM / "ctrlOther"
                )
                pain_hm_res = run_withinlabel_heatmaps(
                    tag_prefix="decision_pain_ctrlOther", which="pain", control_by_other=True,
                    shuffle=shuffle, out_dir=OUT_HM / "ctrlOther"
                )

            if RUN_DIAG_DIFF_STATS and (money_hm_res is not None) and (pain_hm_res is not None):
                run_diag_money_minus_pain_stats(
                    tag_prefix=f"ctrlOther{'_shuffle' if shuffle else ''}",
                    money_res=money_hm_res,
                    pain_res=pain_hm_res,
                    out_dir=OUT_DD / "ctrlOther"
                )

            if RUN_CROSS_LABEL:
                run_crosslabel_heatmaps(
                    tag_prefix="xgen", train="money", control_by_other_train=True,
                    shuffle=shuffle, out_dir=OUT_XG / "ctrlOtherTrain"
                )
                run_crosslabel_heatmaps(
                    tag_prefix="xgen", train="pain", control_by_other_train=True,
                    shuffle=shuffle, out_dir=OUT_XG / "ctrlOtherTrain"
                )

            if RUN_DIFF_MONEY_MINUS_PAIN:
                run_money_minus_pain_difference(
                    tag_prefix="diff", control_by_other=True,
                    metric="balanced_accuracy", shuffle=shuffle,
                    out_dir=OUT_DF / "ctrlOther"
                )
                run_money_minus_pain_difference(
                    tag_prefix="diff", control_by_other=True,
                    metric="roc_auc", shuffle=shuffle,
                    out_dir=OUT_DF / "ctrlOther"
                )

    log_print(f"\n=== mvpa_decision_step2_searchlight_binaryonly DONE ===\n")


if __name__ == "__main__":
    main()
