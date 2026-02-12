# -*- coding: utf-8 -*-
"""
Passive RSA (Step-1 style): graded value geometry for money and pain

- Loads passive epochs from derivatives
- Loads passive beh.tsv from raw painrewardeegdata
- Merges beh columns into epochs.metadata
- Drops bad trials (if badtrial column exists)
- Resamples (optional)
- Runs temporal RSA (20 ms windows, 10 ms step) for:
    (A) Money trials: model RDM = |level_i - level_j| (includes 60)
    (B) Pain trials : same (control)
  Neural RDM at each time window:
    distance = 1 - correlation between trial patterns (channels x time-window)
  RSA score:
    Spearman rho(neural_RDM_upper, model_RDM_upper)

- Group-level cluster permutation test over time on RSA (rho - 0.0)
- Saves extensive outputs for further analysis
- Optional shuffle control (permute labels)

Outputs:
  derivatives/statistics/rsa_passive_step1/
"""

from __future__ import annotations

import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt

from scipy import stats
from scipy.spatial.distance import pdist
from mne.stats import permutation_cluster_1samp_test
from tqdm.auto import tqdm


# -----------------------------
# Paths
# -----------------------------
DATA_DIR_STR = os.getenv("DATA_DIR", "").strip()
OUT_DIR_STR = os.getenv("OUT_DIR", "").strip()

if DATA_DIR_STR == "":
    raise RuntimeError("DATA_DIR env var not set")

RAW_DIR = Path(DATA_DIR_STR).expanduser()
DERIV_DIR = RAW_DIR / "derivatives"

if OUT_DIR_STR != "":
    OUT_BASE = Path(OUT_DIR_STR).expanduser()
    OUT_DIR = OUT_BASE / "statistics" / "rsa_passive_step1"
else:
    OUT_DIR = DERIV_DIR / "statistics" / "rsa_passive_step1"

OUT_DIR.mkdir(parents=True, exist_ok=True)
DEBUG_DIR = OUT_DIR / "debug"
DEBUG_DIR.mkdir(parents=True, exist_ok=True)

# epochs/beh locations relative to subject folder
EPO_DIR = Path("eeg") / "erps_passive"
EPO_SUFFIX = "_passive_cues_singletrials-epo.fif"
BEH_SUFFIX = "_task-passive_beh.tsv"

# -----------------------------
# RSA params
# -----------------------------
RANDOM_STATE = 23
N_PERM = 5000
ALPHA_CLUSTER = 0.05

# resample for speed (set None to keep original)
RESAMPLE_SFREQ = 256

# stats time window (match Step-1)
TMIN_STAT = 0.0
TMAX_STAT = 0.8

# RSA windows
WIN_MS = 20.0       # window length in ms
STEP_MS = 10.0      # step in ms (overlap)

# chance for RSA rho
CHANCE_RSA = 0.0

# metadata keys
KEY_TRIALNUM = "trialsnum"
COL_COND = "condition"   # 'p' or 'm'
COL_LEVEL = "level"      # 20/40/60/80/100
KEY_BLOCK = "blocks.thisN"
KEY_TRIAL = "trials.thisN"

COND_MONEY = "m"
COND_PAIN = "p"

# Controls
RUN_SHUFFLE = True
RUN_RDM_DEBUG_FIGS = True   # saves example neural/model RDM diagnostics (per condition)


# -----------------------------
# Utility: listing/loading
# -----------------------------
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


def merge_beh_into_epochs(epo: mne.Epochs, beh: pd.DataFrame, sub: str, debug_dir: Path) -> mne.Epochs:
    """Same logic as your Step-1: try trialsnum merge, then (block,trial), then order."""
    if epo.metadata is None:
        raise ValueError(f"{sub}: epochs has no metadata; cannot merge beh.")

    md = epo.metadata.reset_index(drop=True).copy()

    if (COL_COND in md.columns) and (COL_LEVEL in md.columns):
        return epo

    # 1) trialsnum merge
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
            md.head(50).to_csv(debug_dir / f"{sub}_epo_md_head.csv", index=False)
            beh.head(50).to_csv(debug_dir / f"{sub}_beh_head.csv", index=False)
            raise ValueError(f"{sub}: trialsnum-merge produced unlabeled epochs.")
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

        beh_small = beh2[[KEY_BLOCK, KEY_TRIAL, COL_COND, COL_LEVEL]].copy()
        merged = md2.merge(
            beh_small,
            on=[KEY_BLOCK, KEY_TRIAL],
            how="left",
            validate="1:1"
        )
        if merged[COL_COND].isna().any() or merged[COL_LEVEL].isna().any():
            md.head(50).to_csv(debug_dir / f"{sub}_epo_md_head.csv", index=False)
            beh.head(50).to_csv(debug_dir / f"{sub}_beh_head.csv", index=False)
            n_bad = int(merged[COL_COND].isna().sum())
            raise ValueError(f"{sub}: key-merge produced {n_bad} unlabeled epochs.")
        epo.metadata = merged
        return epo

    # 3) order fallback if lengths match
    if len(md) == len(beh):
        merged = md.copy()
        merged[COL_COND] = beh[COL_COND].to_numpy()
        merged[COL_LEVEL] = beh[COL_LEVEL].to_numpy()
        for extra in [KEY_BLOCK, KEY_TRIAL]:
            if extra in beh.columns:
                merged[extra] = beh[extra].to_numpy()
        epo.metadata = merged
        return epo

    md.head(50).to_csv(debug_dir / f"{sub}_epo_md_head.csv", index=False)
    beh.head(50).to_csv(debug_dir / f"{sub}_beh_head.csv", index=False)
    raise ValueError(f"{sub}: cannot merge beh into epochs (see debug CSVs).")


def alignment_sanity_check(md: pd.DataFrame, sub: str, debug_dir: Path, log):
    """Lightweight sanity check similar to your Step-1."""
    preview_path = debug_dir / f"{sub}_merged_preview20.csv"
    md.head(20).to_csv(preview_path, index=False)

    if COL_COND not in md.columns:
        return

    cond = md[COL_COND].astype(str).str.lower().to_numpy()
    keep = np.isin(cond, [COND_MONEY, COND_PAIN])
    cond = cond[keep]
    if len(cond) < 20:
        return

    switches = np.mean(cond[1:] != cond[:-1])
    half = len(cond) // 2
    early_m = np.mean(cond[:half] == COND_MONEY)
    late_m = np.mean(cond[half:] == COND_MONEY)
    diff = abs(early_m - late_m)

    if switches < 0.05:
        log(f"{sub}: WARNING low condition switch-rate ({switches:.3f}) — check {preview_path.name}")
    if diff > 0.70:
        log(f"{sub}: WARNING strong early/late split (|early_m-late_m|={diff:.2f}) — check {preview_path.name}")


def save_trial_counts(md_used: pd.DataFrame, sub: str, which: str, out_dir: Path):
    tab = (
        md_used.assign(condition=md_used[COL_COND].astype(str).str.lower())
        .groupby(["condition", COL_LEVEL]).size()
        .reset_index(name="n")
        .sort_values(["condition", COL_LEVEL])
    )
    tab.to_csv(out_dir / f"{sub}_{which}_trial_counts.csv", index=False)


# -----------------------------
# RSA core
# -----------------------------
def make_time_windows(times: np.ndarray, win_ms: float, step_ms: float) -> list[tuple[int, int, float]]:
    """Return list of (i_start, i_stop, t_center) for overlapping windows."""
    sfreq = 1.0 / np.median(np.diff(times))
    win_samp = int(np.round((win_ms / 1000.0) * sfreq))
    step_samp = int(np.round((step_ms / 1000.0) * sfreq))
    win_samp = max(win_samp, 2)
    step_samp = max(step_samp, 1)

    windows = []
    start = 0
    while start + win_samp <= len(times):
        stop = start + win_samp
        t_center = float(times[start:stop].mean())
        windows.append((start, stop, t_center))
        start += step_samp
    return windows


def _rank_vec(x: np.ndarray) -> np.ndarray:
    """Rank-transform with average ranks; robust for RSA comparisons."""
    x = np.asarray(x, dtype=float).ravel()
    # stats.rankdata handles ties appropriately
    return stats.rankdata(x, method="average")


def rsa_timecourse(
    X: np.ndarray,          # (n_trials, n_ch, n_times)
    levels: np.ndarray,     # (n_trials,)
    times: np.ndarray,
    *,
    windows: list[tuple[int, int, float]],
    shuffle_labels: bool,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute RSA rho per window.
    Neural distance = pdist(patterns, metric='correlation') -> 1 - corr
    Model distance  = pdist(levels, metric='cityblock') -> |diff|
    RSA = Spearman(neural_vec, model_vec) implemented as corr(rank(neural), rank(model)).
    Returns:
      rhos: (n_windows,)
      t_centers: (n_windows,)
    """
    levels = np.asarray(levels, dtype=float).ravel()
    if shuffle_labels:
        levels_use = rng.permutation(levels)
    else:
        levels_use = levels

    # Model vector (upper triangle condensed form)
    model_vec = pdist(levels_use.reshape(-1, 1), metric="cityblock")
    model_r = _rank_vec(model_vec)

    rhos = np.zeros(len(windows), dtype=float)
    t_centers = np.zeros(len(windows), dtype=float)

    for wi, (i0, i1, tc) in enumerate(windows):
        t_centers[wi] = tc
        # patterns: trials x features
        pat = X[:, :, i0:i1].reshape(X.shape[0], -1)

        # neural condensed distances
        # correlation metric internally normalizes (mean-center) each pattern
        neural_vec = pdist(pat, metric="correlation")

        # handle degenerate cases
        if not np.all(np.isfinite(neural_vec)) or np.nanstd(neural_vec) < 1e-12:
            rhos[wi] = 0.0
            continue

        neural_r = _rank_vec(neural_vec)

        # Pearson corr between ranks == Spearman rho
        rr = np.corrcoef(neural_r, model_r)[0, 1]
        if np.isnan(rr):
            rr = 0.0
        rhos[wi] = float(rr)

    return rhos, t_centers


# -----------------------------
# Stats + saving
# -----------------------------
def _cluster_to_bool_mask(cl, n_times: int) -> np.ndarray | None:
    arr = np.asarray(cl)
    if arr.dtype == bool:
        arr = arr.squeeze()
        if arr.ndim == 1 and arr.size == n_times:
            return arr
        arr = arr.ravel()
        if arr.size == n_times:
            return arr
        return None
    # integer indices form
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


def group_cluster_timecourse(scores_by_subj: np.ndarray, times: np.ndarray, *, chance: float, tail: int = 1) -> dict:
    """
    Cluster permutation test on (scores - chance) across time.
    """
    X = scores_by_subj - chance  # (n_subj, n_times)

    tfce_thresh = dict(start=0.0, step=0.2)
    try:
        T_obs, clusters, cluster_pv, _ = permutation_cluster_1samp_test(
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
        T_obs, clusters, cluster_pv, _ = permutation_cluster_1samp_test(
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


def plot_timecourse(scores_by_subj: np.ndarray, times: np.ndarray, p_map: np.ndarray,
                   *, title: str, ylabel: str, chance: float, out_png: Path,
                   ylim: tuple[float, float] | None = None):
    mean = scores_by_subj.mean(axis=0)
    sem = scores_by_subj.std(axis=0, ddof=1) / np.sqrt(scores_by_subj.shape[0])

    fig, ax = plt.subplots(figsize=(7.5, 3.2))
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
    if ylim is not None:
        ax.set_ylim(*ylim)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def plot_overlay(money_scores: np.ndarray, pain_scores: np.ndarray,
                 times: np.ndarray,
                 money_p: np.ndarray, pain_p: np.ndarray,
                 *, out_png: Path):
    m_mean = money_scores.mean(axis=0)
    m_sem = money_scores.std(axis=0, ddof=1) / np.sqrt(money_scores.shape[0])

    p_mean = pain_scores.mean(axis=0)
    p_sem = pain_scores.std(axis=0, ddof=1) / np.sqrt(pain_scores.shape[0])

    fig, ax = plt.subplots(figsize=(7.8, 3.4))
    ax.plot(times, m_mean, linewidth=2, label="Money RSA")
    ax.fill_between(times, m_mean - m_sem, m_mean + m_sem, alpha=0.20)

    ax.plot(times, p_mean, linewidth=2, label="Pain RSA")
    ax.fill_between(times, p_mean - p_sem, p_mean + p_sem, alpha=0.20)

    ax.axhline(CHANCE_RSA, linestyle="--", linewidth=1)
    ax.axvline(0, linestyle="--", linewidth=1)

    # significance bars (stacked)
    m_sig = money_p < ALPHA_CLUSTER
    p_sig = pain_p < ALPHA_CLUSTER
    if np.any(m_sig):
        ax.fill_between(times, CHANCE_RSA - 0.030, CHANCE_RSA - 0.022, where=m_sig, alpha=0.9)
    if np.any(p_sig):
        ax.fill_between(times, CHANCE_RSA - 0.018, CHANCE_RSA - 0.010, where=p_sig, alpha=0.9)

    ax.set_title("Passive RSA: Money vs Pain (graded geometry)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("RSA (Spearman ρ)")
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def save_group_bundle(*, tag: str, scores: np.ndarray, times: np.ndarray,
                      included: list[str], skipped: list[tuple[str, str]],
                      stats_out: dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save main NPZ
    np.savez(
        out_dir / f"{tag}_group_results.npz",
        scores_by_subj=scores,
        times=times,
        included=np.array(included, dtype=object),
        skipped=np.array(skipped, dtype=object),
        T_obs=stats_out["T_obs"],
        p_map=stats_out["p_map"],
        cluster_pv=stats_out["cluster_pv"],
        t_thresh=stats_out["t_thresh"],
        chance=CHANCE_RSA,
        resample_sfreq=RESAMPLE_SFREQ if RESAMPLE_SFREQ is not None else -1,
        tmin_stat=TMIN_STAT,
        tmax_stat=TMAX_STAT,
        win_ms=WIN_MS,
        step_ms=STEP_MS,
    )

    # scores-by-subject CSV
    pd.DataFrame(scores, index=included, columns=np.round(times, 6)).to_csv(
        out_dir / f"{tag}_scores_by_subject.csv"
    )

    # timecourse summary + cluster table
    mean = scores.mean(axis=0)
    sem = scores.std(axis=0, ddof=1) / np.sqrt(scores.shape[0])
    n = scores.shape[0]
    tcrit = stats.t.ppf(0.975, df=n - 1)
    ci95_low = mean - tcrit * sem
    ci95_high = mean + tcrit * sem

    p_map = np.asarray(stats_out["p_map"]).ravel()
    df_tc = pd.DataFrame({
        "time_s": times,
        "mean_rho": mean,
        "sem_rho": sem,
        "ci95_low": ci95_low,
        "ci95_high": ci95_high,
        "T_obs": stats_out["T_obs"],
        "p_map": p_map,
        "sig": p_map < ALPHA_CLUSTER,
        "chance": CHANCE_RSA,
    })
    df_tc.to_csv(out_dir / f"{tag}_grand_mean_sem.csv", index=False)

    rows = []
    for i, (mask, p) in enumerate(zip(stats_out["clusters"], stats_out["cluster_pv"])):
        mask = _cluster_to_bool_mask(mask, len(times))
        if mask is None or not mask.any():
            continue
        t_start = float(times[np.where(mask)[0][0]])
        t_end = float(times[np.where(mask)[0][-1]])
        dur_ms = (t_end - t_start) * 1000.0
        eff = scores[:, mask] - CHANCE_RSA
        mean_eff = float(eff.mean())
        rows.append({
            "cluster": i,
            "p_value": float(p),
            "t_start_s": t_start,
            "t_end_s": t_end,
            "duration_ms": dur_ms,
            "cluster_mass_sumT": float(stats_out["T_obs"][mask].sum()),
            "cluster_maxT": float(stats_out["T_obs"][mask].max()),
            "mean_effect_rho_minus_chance": mean_eff,
        })
    df_cl = pd.DataFrame(rows).sort_values("p_value") if len(rows) else pd.DataFrame(
        columns=["cluster","p_value","t_start_s","t_end_s","duration_ms","cluster_mass_sumT","cluster_maxT","mean_effect_rho_minus_chance"]
    )
    df_cl.to_csv(out_dir / f"{tag}_cluster_table.csv", index=False)

    # JSON meta
    peak_idx = int(np.argmax(mean))
    meta = {
        "tag": tag,
        "n_subjects": int(scores.shape[0]),
        "n_times": int(scores.shape[1]),
        "chance": float(CHANCE_RSA),
        "alpha_cluster": float(ALPHA_CLUSTER),
        "tfce_or_threshold": stats_out.get("t_thresh", None),
        "peak_rho": float(mean[peak_idx]),
        "peak_time_s": float(times[peak_idx]),
        "min_cluster_p": float(np.min(stats_out["cluster_pv"])) if len(stats_out["cluster_pv"]) else 1.0,
        "included_subjects": included,
        "win_ms": float(WIN_MS),
        "step_ms": float(STEP_MS),
        "tmin_stat": float(TMIN_STAT),
        "tmax_stat": float(TMAX_STAT),
    }
    with open(out_dir / f"{tag}_summary.json", "w") as f:
        json.dump(meta, f, indent=2)


# -----------------------------
# Main run
# -----------------------------
def run_condition(which: str, *, shuffle: bool, out_dir: Path) -> tuple[np.ndarray, np.ndarray, list[str], list[tuple[str, str]], dict]:
    """
    which: "money" or "pain"
    Returns: scores (n_subj,n_times), times_stat, included, skipped, stats_out
    """
    rng = np.random.default_rng(RANDOM_STATE + (1 if shuffle else 0) + (7 if which == "pain" else 0))

    def log(msg: str):
        try:
            tqdm.write(msg)
        except Exception:
            print(msg, flush=True)

    subs = list_subjects(DERIV_DIR)
    included, skipped = [], []
    scores_list = []
    times_ref = None

    for sub in tqdm(subs, desc=f"RSA {which}{'_shuf' if shuffle else ''}", unit="sub", dynamic_ncols=True):
        try:
            epo = load_passive_epochs(sub)
            beh = load_passive_beh(sub)
            epo = merge_beh_into_epochs(epo, beh, sub=sub, debug_dir=DEBUG_DIR)

            if epo.metadata is None:
                raise RuntimeError("metadata is None after merge (unexpected).")

            # drop bad trials
            if "badtrial" in epo.metadata.columns:
                bad = epo.metadata["badtrial"].fillna(0).astype(int).to_numpy()
                if bad.sum() > 0:
                    epo = epo.copy()[bad == 0]

            md = epo.metadata.reset_index(drop=True)
            alignment_sanity_check(md, sub=sub, debug_dir=DEBUG_DIR, log=log)

            # check labels sane
            if md[COL_COND].isna().any() or md[COL_LEVEL].isna().any():
                md.head(50).to_csv(DEBUG_DIR / f"{sub}_md_aftermerge_head.csv", index=False)
                raise RuntimeError("NaNs in merged labels (see debug head).")

            # resample
            if RESAMPLE_SFREQ is not None:
                epo = epo.copy().resample(RESAMPLE_SFREQ, npad="auto")

            md = epo.metadata.reset_index(drop=True)
            cond = md[COL_COND].astype(str).str.lower().to_numpy()

            if which == "money":
                keep = (cond == COND_MONEY)
            elif which == "pain":
                keep = (cond == COND_PAIN)
            else:
                raise ValueError("which must be 'money' or 'pain'")

            epo_f = epo.copy()[keep]
            md_f = epo_f.metadata.reset_index(drop=True)

            if len(epo_f) < 10:
                raise RuntimeError(f"Too few trials after filtering ({len(epo_f)}).")

            levels = md_f[COL_LEVEL].to_numpy(dtype=float)
            if not np.all(np.isfinite(levels)):
                raise RuntimeError("Non-finite levels after filtering.")

            X = epo_f.get_data()     # (n_trials, n_ch, n_times)
            times = epo_f.times.copy()

            # windows (based on this subject's times)
            windows = make_time_windows(times, WIN_MS, STEP_MS)

            # time axis consistency
            t_centers = np.array([w[2] for w in windows], dtype=float)
            if times_ref is None:
                times_ref = t_centers
            else:
                if len(t_centers) != len(times_ref) or np.max(np.abs(t_centers - times_ref)) > 1e-9:
                    raise RuntimeError("RSA window time-axis mismatch across subjects.")

            # RSA timecourse
            rhos, _ = rsa_timecourse(
                X, levels, times,
                windows=windows,
                shuffle_labels=shuffle,
                rng=rng
            )

            # Save subject artifacts
            included.append(sub)
            scores_list.append(rhos)

            # trial counts file
            save_trial_counts(md_f, sub=sub, which=f"rsa_{which}", out_dir=out_dir)

            # One-time debug figs per condition (first included subject only)
            if RUN_RDM_DEBUG_FIGS and len(included) == 1 and not shuffle:
                # pick a mid-late window inside stats window if possible
                target_t = 0.4
                wi = int(np.argmin(np.abs(times_ref - target_t)))
                i0, i1, tc = windows[wi]
                pat = X[:, :, i0:i1].reshape(X.shape[0], -1)
                neural_vec = pdist(pat, metric="correlation")
                model_vec = pdist(levels.reshape(-1, 1), metric="cityblock")

                # quick visualize as squareform-ish without importing squareform (avoid extra deps)
                n = len(levels)
                def vec_to_mat(v):
                    M = np.zeros((n, n), dtype=float)
                    iu = np.triu_indices(n, 1)
                    M[iu] = v
                    M[(iu[1], iu[0])] = v
                    return M

                neural_mat = vec_to_mat(neural_vec)
                model_mat = vec_to_mat(model_vec)

                fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
                axes[0].imshow(model_mat, aspect="auto")
                axes[0].set_title(f"{which} model RDM |Δlevel|")
                axes[1].imshow(neural_mat, aspect="auto")
                axes[1].set_title(f"{which} neural RDM (1-corr)\n~{tc:.3f}s")
                for ax in axes:
                    ax.set_xlabel("trial")
                    ax.set_ylabel("trial")
                fig.tight_layout()
                fig.savefig(DEBUG_DIR / f"debug_{which}_rdm_example_{sub}.png", dpi=200)
                plt.close(fig)

        except Exception as e:
            skipped.append((sub, str(e)))
            log(f"Skipped {sub} RSA {which}{'_shuf' if shuffle else ''}: {e}")

    if len(scores_list) < 8:
        raise RuntimeError(f"Too few subjects for group stats RSA {which}: n={len(scores_list)}")

    scores = np.stack(scores_list, axis=0)  # (n_subj, n_windows)
    times = times_ref

    # stats window
    mask = (times >= TMIN_STAT) & (times <= TMAX_STAT)
    times_stat = times[mask]
    scores_stat = scores[:, mask]

    stats_out = group_cluster_timecourse(scores_stat, times_stat, chance=CHANCE_RSA, tail=1)

    return scores_stat, times_stat, included, skipped, stats_out


def main():
    mne.set_log_level("WARNING")

    # output subfolders
    money_dir = OUT_DIR / "money"
    pain_dir = OUT_DIR / "pain"
    money_dir.mkdir(parents=True, exist_ok=True)
    pain_dir.mkdir(parents=True, exist_ok=True)

    # ----- REAL -----
    money_scores, times_stat, money_incl, money_skip, money_stats = run_condition(
        "money", shuffle=False, out_dir=money_dir
    )
    pain_scores, times_stat2, pain_incl, pain_skip, pain_stats = run_condition(
        "pain", shuffle=False, out_dir=pain_dir
    )

    if len(times_stat2) != len(times_stat) or np.max(np.abs(times_stat2 - times_stat)) > 1e-9:
        raise RuntimeError("Money vs pain RSA time axes differ (unexpected).")

    # save group bundles
    save_group_bundle(
        tag="passive_money_rsa",
        scores=money_scores,
        times=times_stat,
        included=money_incl,
        skipped=money_skip,
        stats_out=money_stats,
        out_dir=money_dir,
    )
    save_group_bundle(
        tag="passive_pain_rsa",
        scores=pain_scores,
        times=times_stat,
        included=pain_incl,
        skipped=pain_skip,
        stats_out=pain_stats,
        out_dir=pain_dir,
    )

    # plots
    plot_timecourse(
        money_scores, times_stat, money_stats["p_map"],
        title="Passive RSA (Money): graded value geometry",
        ylabel="RSA (Spearman ρ)",
        chance=CHANCE_RSA,
        out_png=money_dir / "passive_money_rsa_group_plot.png",
        ylim=(-0.10, 0.40),
    )
    plot_timecourse(
        pain_scores, times_stat, pain_stats["p_map"],
        title="Passive RSA (Pain): graded value geometry (control)",
        ylabel="RSA (Spearman ρ)",
        chance=CHANCE_RSA,
        out_png=pain_dir / "passive_pain_rsa_group_plot.png",
        ylim=(-0.10, 0.40),
    )

    plot_overlay(
        money_scores, pain_scores,
        times_stat,
        money_stats["p_map"], pain_stats["p_map"],
        out_png=OUT_DIR / "passive_rsa_money_vs_pain_overlay.png"
    )

    # ----- SHUFFLE CONTROL -----
    if RUN_SHUFFLE:
        money_shuf_dir = OUT_DIR / "money_shuffle"
        pain_shuf_dir = OUT_DIR / "pain_shuffle"
        money_shuf_dir.mkdir(parents=True, exist_ok=True)
        pain_shuf_dir.mkdir(parents=True, exist_ok=True)

        money_s, t_s, incl_s, skip_s, st_s = run_condition("money", shuffle=True, out_dir=money_shuf_dir)
        pain_s, t_s2, incl_p, skip_p, st_p = run_condition("pain", shuffle=True, out_dir=pain_shuf_dir)

        save_group_bundle(
            tag="passive_money_rsa_shuffle",
            scores=money_s,
            times=t_s,
            included=incl_s,
            skipped=skip_s,
            stats_out=st_s,
            out_dir=money_shuf_dir,
        )
        save_group_bundle(
            tag="passive_pain_rsa_shuffle",
            scores=pain_s,
            times=t_s2,
            included=incl_p,
            skipped=skip_p,
            stats_out=st_p,
            out_dir=pain_shuf_dir,
        )

        plot_timecourse(
            money_s, t_s, st_s["p_map"],
            title="Passive RSA (Money) — SHUFFLE control",
            ylabel="RSA (Spearman ρ)",
            chance=CHANCE_RSA,
            out_png=money_shuf_dir / "passive_money_rsa_shuffle_group_plot.png",
            ylim=(-0.10, 0.40),
        )
        plot_timecourse(
            pain_s, t_s2, st_p["p_map"],
            title="Passive RSA (Pain) — SHUFFLE control",
            ylabel="RSA (Spearman ρ)",
            chance=CHANCE_RSA,
            out_png=pain_shuf_dir / "passive_pain_rsa_shuffle_group_plot.png",
            ylim=(-0.10, 0.40),
        )

    print("\nDONE: Passive RSA saved to:", str(OUT_DIR))


if __name__ == "__main__":
    main()


