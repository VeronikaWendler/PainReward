# -*- coding: utf-8 -*-

# Vero
# Decision phase: Time-resolved decoding of money- and pain-cue levels (decision phase)
#
# KEEP ONLY:
# 1) Standard binary (low=20/40 vs high=80/100; drops 60)
#    - money: AUC + balanced accuracy
#    - pain : AUC + balanced accuracy
#    - ctrlOtherRT only  (other cue + RT controlled together)
#    - shuffle controls
#    - group TFCE cluster permutation tests over time + time plots
#
# 2) Cross-label generalization, binary only
#    (A) Time-resolved diagonal:
#        - Train on money labels, test on pain labels  (money->pain)
#        - Train on pain labels, test on money labels  (pain->money)
#        - balanced accuracy + AUC
#
#    (B) Cross-label temporal generalization heatmaps:
#        - money->pain and pain->money
#        - binary only
#        - ONLY for ctrlOtherRT and its shuffle
#
# IMPORTANT CONTROL RULE:
#    whenever we control for the other cue, we ALWAYS also control for RT
#    -> ctrlOther == other cue + RT
#
# OUTPUTS
# derivatives/statistics/mvpa_decision_step1_conserv2/
#   binary_lowhigh_auc_bacc/
#   cross_generalization/
#      binary/
#   debug/

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
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.base import clone

from mne.decoding import SlidingEstimator, cross_val_multiscore
from mne.stats import permutation_cluster_1samp_test
from scipy import stats

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

# Output folder
if OUT_DIR_STR != "":
    OUT_BASE = Path(OUT_DIR_STR).expanduser()
    OUT_DIR = OUT_BASE / "statistics" / "mvpa_time_decoding_decision_phase"
else:
    OUT_DIR = DERIV_DIR / "statistics" / "mvpa_time_decoding_decision_phase"

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

# downsample for speed (~10 ms)
RESAMPLE_SFREQ = 100  # 10 ms step

# decision metadata columns
DEC_MONEY_COL = "moneystim"
DEC_PAIN_COL = "painstim"
RT_COL = "choice_resp.rt"

# stats window
TMIN_STAT = 0.0
TMAX_STAT = 1.0

# Run flags
RUN_BINARY = True
RUN_SHUFFLE = True
RUN_CONTROL_BY_OTHER = True
RUN_CROSS_GENERALIZATION = True

# Binary settings
CHANCE_BIN = 0.5
BIN_KEEP_LEVELS = np.array([20, 40, 80, 100], dtype=int)

# Heatmap controls (temporal generalization)
# At 100 Hz, decim=2 -> 20 ms grid for heatmaps
HEATMAP_DECIM = 2

LEVEL_CODE_TO_LEVEL = {1: 20, 2: 40, 3: 60, 4: 80, 5: 100}


# -----------------------------
# output folders
# -----------------------------
OUT_DIR_BIN = OUT_DIR / "binary_lowhigh_auc_bacc"

OUT_DIR_XGEN = OUT_DIR / "cross_generalization"
OUT_DIR_XGEN_BIN = OUT_DIR_XGEN / "binary"

for _d in [OUT_DIR_BIN, OUT_DIR_XGEN_BIN]:
    _d.mkdir(parents=True, exist_ok=True)

DEBUG_DIR_BIN = OUT_DIR_BIN / "debug"
DEBUG_DIR_XGEN_BIN = OUT_DIR_XGEN_BIN / "debug"
for _d in [DEBUG_DIR_BIN, DEBUG_DIR_XGEN_BIN]:
    _d.mkdir(parents=True, exist_ok=True)


# =====================================================================
# Logging + reporting helpers
# =====================================================================

def log_print(msg: str):
    """Always print (plays nicely with tqdm)."""
    try:
        tqdm.write(msg)
    except Exception:
        print(msg, flush=True)


def _fmt_float(x, nd=4):
    try:
        if x is None:
            return "NA"
        if isinstance(x, str):
            return x
        if np.isnan(float(x)):
            return "NA"
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)


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


def summarize_group_results_text(
    *,
    analysis_name: str,
    metric_name: str,
    tag: str,
    which: str,
    shuffle: bool,
    control: str,
    nuisance_model: str,
    out_dir: Path,
    scores_all: np.ndarray,
    times: np.ndarray,
    chance: float,
    stats_out: dict,
    alpha: float,
    included: list[str],
    skipped: list[tuple[str, str]],
    trial_stats: dict,
):
    n_sub = int(scores_all.shape[0])
    n_time = int(scores_all.shape[1])

    mean = scores_all.mean(axis=0)
    sem = scores_all.std(axis=0, ddof=1) / np.sqrt(max(1, n_sub))

    peak_idx = int(np.argmax(mean))
    peak_score = float(mean[peak_idx])
    peak_time = float(times[peak_idx])

    T_obs = np.asarray(stats_out["T_obs"]).ravel()
    peak_t_idx = int(np.argmax(T_obs))
    peak_T = float(T_obs[peak_t_idx])
    peak_T_time = float(times[peak_t_idx])

    cluster_pv = np.asarray(stats_out.get("cluster_pv", [])).ravel()
    min_cluster_p = float(np.min(cluster_pv)) if cluster_pv.size else 1.0

    n_sub_total = int(trial_stats.get("n_sub_total", n_sub + len(skipped)))
    n_sub_skipped = int(len(skipped))

    total_trials_in = int(trial_stats.get("total_trials_in", -1))
    total_trials_after_badtrial = int(trial_stats.get("total_trials_after_badtrial", -1))
    total_trials_after_level = int(trial_stats.get("total_trials_after_level", -1))

    n_trials_per_sub = np.asarray(trial_stats.get("n_trials_per_sub", []), dtype=float)
    n_trials_mean = float(np.mean(n_trials_per_sub)) if n_trials_per_sub.size else np.nan
    n_trials_sd = float(np.std(n_trials_per_sub, ddof=1)) if n_trials_per_sub.size > 1 else np.nan
    n_trials_min = float(np.min(n_trials_per_sub)) if n_trials_per_sub.size else np.nan
    n_trials_max = float(np.max(n_trials_per_sub)) if n_trials_per_sub.size else np.nan

    p_map = np.asarray(stats_out["p_map"]).ravel()
    sig_mask = p_map < alpha
    sig_any = bool(np.any(sig_mask))
    if sig_any:
        sig_times = times[sig_mask]
        sig_start = float(sig_times[0])
        sig_end = float(sig_times[-1])
        sig_dur_ms = (sig_end - sig_start) * 1000.0
        mean_in_sig = float(np.mean(mean[sig_mask]))
        mean_above_chance_in_sig = float(np.mean(mean[sig_mask] - chance))
    else:
        sig_start = sig_end = sig_dur_ms = np.nan
        mean_in_sig = np.nan
        mean_above_chance_in_sig = np.nan

    lines = []
    lines.append("=" * 78)
    lines.append(f"FINISHED: {analysis_name}")
    lines.append("-" * 78)
    lines.append(f"Metric:               {metric_name}")
    lines.append(f"Decoded variable:     {which}")
    lines.append(f"Tag:                  {tag}")
    lines.append(f"Shuffle:              {shuffle}")
    lines.append(f"Control:              {control}")
    lines.append(f"Nuisance model:       {nuisance_model}")
    lines.append("")
    lines.append("DATA / EXCLUSIONS")
    lines.append(f"  Subjects total:      {n_sub_total}")
    lines.append(f"  Subjects included:   {n_sub}")
    lines.append(f"  Subjects skipped:    {n_sub_skipped}")
    if n_sub_skipped:
        lines.append("  Skipped subjects (first 10):")
        for s, reason in skipped[:10]:
            lines.append(f"    - {s}: {reason}")
        if len(skipped) > 10:
            lines.append(f"    ... (+{len(skipped)-10} more)")
    lines.append("")
    lines.append("  Trial accounting (summed across included subjects):")
    if total_trials_in >= 0:
        lines.append(f"    total trials in (post-merge):        {total_trials_in}")
        lines.append(f"    after dropping badtrial:             {total_trials_after_badtrial}")
        lines.append(f"    after level-filtering / RT valid:    {total_trials_after_level}")
    lines.append(
        f"  Trials per included subject: mean={_fmt_float(n_trials_mean,2)} "
        f"sd={_fmt_float(n_trials_sd,2)} min={_fmt_float(n_trials_min,0)} max={_fmt_float(n_trials_max,0)}"
    )
    lines.append("")
    lines.append("ANALYSIS SETTINGS")
    lines.append(f"  Chance level:         {_fmt_float(chance,3)}")
    lines.append(f"  N permutations:       {N_PERM}")
    lines.append(f"  Alpha cluster:        {alpha}")
    lines.append(f"  Cluster threshold:    {stats_out.get('t_thresh', 'NA')}")
    lines.append(f"  Time window (stats):  [{_fmt_float(times[0],3)}, {_fmt_float(times[-1],3)}] s")
    lines.append("")
    lines.append("RESULTS (group mean over subjects)")
    lines.append(f"  Peak mean score:      {_fmt_float(peak_score,4)} at t={_fmt_float(peak_time,4)} s")
    lines.append(f"  Peak T_obs:           {_fmt_float(peak_T,4)} at t={_fmt_float(peak_T_time,4)} s")
    lines.append(f"  Min cluster p-value:  {_fmt_float(min_cluster_p,6)}")
    if sig_any:
        lines.append(f"  Significant timepoints (p<{alpha}): YES")
        lines.append(
            f"    span:               {_fmt_float(sig_start,4)} .. {_fmt_float(sig_end,4)} s "
            f"(dur {_fmt_float(sig_dur_ms,1)} ms)"
        )
        lines.append(
            f"    mean score in span: {_fmt_float(mean_in_sig,4)} "
            f"(above chance: {_fmt_float(mean_above_chance_in_sig,4)})"
        )
    else:
        lines.append(f"  Significant timepoints (p<{alpha}): NO")
    lines.append("")

    clusters = stats_out.get("clusters", [])
    if np.asarray(stats_out.get("cluster_pv", [])).size and len(clusters):
        rows = []
        for i, (mask, p) in enumerate(zip(clusters, stats_out["cluster_pv"])):
            m = _cluster_to_bool_mask(mask, n_time)
            if m is None or not m.any():
                continue
            i0 = int(np.where(m)[0][0])
            i1 = int(np.where(m)[0][-1])
            t0 = float(times[i0])
            t1 = float(times[i1])
            dur = (t1 - t0) * 1000.0
            eff = float(np.mean(scores_all[:, m] - chance))
            rows.append((float(p), i, t0, t1, dur, eff))
        rows.sort(key=lambda x: x[0])

        lines.append("CLUSTERS (sorted by p-value; max 5 shown)")
        for (p, idx, t0, t1, dur, eff) in rows[:5]:
            lines.append(
                f"  cluster {idx}: p={_fmt_float(p,6)} | {t0:.4f}-{t1:.4f}s "
                f"({dur:.1f} ms) | mean(score-chance)={_fmt_float(eff,4)}"
            )
        lines.append("")
    lines.append("=" * 78)

    safe_name = f"{analysis_name}_{metric_name}".replace(" ", "_").replace("/", "_")
    out_txt = out_dir / f"{safe_name}_GROUP_REPORT.txt"
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    for ln in lines:
        log_print(ln)

    out_json = out_dir / f"{safe_name}_GROUP_REPORT.json"
    payload = dict(
        analysis_name=analysis_name,
        metric_name=metric_name,
        which=which,
        tag=tag,
        shuffle=bool(shuffle),
        control=control,
        nuisance_model=nuisance_model,
        n_subjects_total=int(trial_stats.get("n_sub_total", n_sub + len(skipped))),
        n_subjects_included=n_sub,
        n_subjects_skipped=int(len(skipped)),
        included_subjects=included,
        skipped_subjects=skipped,
        trial_stats=trial_stats,
        chance=float(chance),
        n_perm=int(N_PERM),
        alpha=float(alpha),
        cluster_threshold=str(stats_out.get("t_thresh", None)),
        peak_mean_score=float(peak_score),
        peak_mean_time_s=float(peak_time),
        peak_T_obs=float(peak_T),
        peak_T_time_s=float(peak_T_time),
        min_cluster_p=float(np.min(np.asarray(stats_out.get("cluster_pv", [1.0])))),
        sig_any=bool(sig_any),
        sig_start_s=None if not sig_any else float(sig_start),
        sig_end_s=None if not sig_any else float(sig_end),
        sig_duration_ms=None if not sig_any else float(sig_dur_ms),
        mean_in_sig=None if not sig_any else float(mean_in_sig),
        mean_above_chance_in_sig=None if not sig_any else float(mean_above_chance_in_sig),
    )
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


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
        if RT_COL not in md.columns and RT_COL in beh.columns and KEY_TRIALNUM in md.columns:
            merged = md.merge(beh[[KEY_TRIALNUM, RT_COL]], on=KEY_TRIALNUM, how="left", validate="1:1")
            epo.metadata = merged
            log_print(f"{sub}: epochs.metadata already had cue labels; merged RT via '{KEY_TRIALNUM}'")
        else:
            log_print(f"{sub}: epochs.metadata already contains {DEC_MONEY_COL}+{DEC_PAIN_COL} (no merge needed)")
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
        log_print(f"{sub}: merged beh into epochs using '{KEY_TRIALNUM}'")
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
        log_print(f"{sub}: merged beh into epochs using KEYS ({KEY_BLOCK}, {KEY_TRIAL})")
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
        log_print(f"{sub}: merged beh into epochs by ORDER (len match: {len(md)})")
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
    """Parse e.g. 'm1'..'m5' or 'p1'..'p5' to 20/40/60/80/100"""
    s = series.astype(str).str.strip().str.lower()
    codes = s.str.extract(rf"^{prefix}\s*([1-5])$", expand=False)
    if codes.isna().any():
        bad = s[codes.isna()].unique()[:10]
        raise ValueError(f"Unexpected '{prefix}' stim codes (examples): {bad}")
    codes_int = codes.astype(int).to_numpy()
    return np.array([LEVEL_CODE_TO_LEVEL[int(c)] for c in codes_int], dtype=int)


def make_binary_labels(level: np.ndarray) -> np.ndarray:
    """level values expected: 20/40/80/100; returns y: 0=low, 1=high"""
    level = np.asarray(level, dtype=float)
    y = np.full(len(level), -1, dtype=int)
    y[np.isin(level, [20, 40])] = 0
    y[np.isin(level, [80, 100])] = 1
    return y


def decision_alignment_sanity_check(md: pd.DataFrame, sub: str, debug_dir: Path):
    preview_path = debug_dir / f"{sub}_merged_preview20.csv"
    md.head(20).to_csv(preview_path, index=False)

    if (DEC_MONEY_COL not in md.columns) or (DEC_PAIN_COL not in md.columns):
        log_print(f"{sub}: alignment check: missing {DEC_MONEY_COL}/{DEC_PAIN_COL}. Saved {preview_path.name}")
        return

    try:
        m = parse_stim_code_to_level(md[DEC_MONEY_COL], "m")
        p = parse_stim_code_to_level(md[DEC_PAIN_COL], "p")
    except Exception as e:
        log_print(f"{sub}: alignment check: stim parsing failed: {e}. Saved {preview_path.name}")
        return

    if len(m) >= 20:
        r = np.corrcoef(m, p)[0, 1]
        if np.isfinite(r) and abs(r) > 0.95:
            log_print(f"{sub}: WARNING |corr(money,pain)|={abs(r):.2f} extremely high; check merge. Saved {preview_path.name}")


def save_decision_trial_counts(
    out_dir: Path,
    sub: str,
    tag: str,
    which: str,
    levels: np.ndarray,
    other_levels: np.ndarray,
    rt: np.ndarray,
):
    df = pd.DataFrame({
        "level_decoded": levels.astype(int),
        "level_other": other_levels.astype(int),
        "rt": rt.astype(float),
    })
    tab1 = df.groupby("level_decoded").size().reset_index(name="n").sort_values("level_decoded")
    tab2 = df.groupby("level_other").size().reset_index(name="n").sort_values("level_other")
    tab1.to_csv(out_dir / f"{sub}_decision_{tag}_{which}_counts_decoded.csv", index=False)
    tab2.to_csv(out_dir / f"{sub}_decision_{tag}_{which}_counts_other.csv", index=False)


# =====================================================================
# Residualization helpers (control-by-other + RT)
# =====================================================================

def _design_matrix_from_nuisances(
    nuisances: list[np.ndarray],
    models: str | list[str] = "lin",
    zscore: bool = True
) -> np.ndarray:
    """
    Build design matrix with intercept + nuisance terms.

    models:
      - if str: applied to ALL nuisances ("lin" or "quad")
      - if list[str]: per-nuisance specification, len(models) == len(nuisances)

    "lin"  -> add nuisance
    "quad" -> add nuisance + nuisance^2
    """
    if len(nuisances) == 0:
        raise ValueError("No nuisances provided.")

    if isinstance(models, str):
        models_use = [models] * len(nuisances)
    else:
        models_use = list(models)
        if len(models_use) != len(nuisances):
            raise ValueError(f"models length {len(models_use)} must match nuisances length {len(nuisances)}")

    cols = [np.ones_like(np.asarray(nuisances[0]).ravel(), dtype=float)]

    for n, model in zip(nuisances, models_use):
        n = np.asarray(n, dtype=float).ravel()
        if zscore:
            mu = np.nanmean(n)
            sd = np.nanstd(n)
            if not np.isfinite(sd) or sd < 1e-12:
                n_z = np.zeros_like(n)
            else:
                n_z = (n - mu) / sd
        else:
            n_z = n

        cols.append(n_z)

        if model == "quad":
            cols.append(n_z ** 2)
        elif model == "lin":
            pass
        else:
            raise ValueError("models entries must be 'lin' or 'quad'")

    return np.column_stack(cols)


def residualize_X_by_nuisances(
    X: np.ndarray,
    nuisances: list[np.ndarray],
    models: str | list[str] = "lin"
) -> np.ndarray:
    """
    Residualize EEG features w.r.t. nuisance vector(s) across trials.

    Here ctrlOther means:
      - other cue level (lin)
      - RT (lin)
    """
    X = np.asarray(X, dtype=float)
    if len(nuisances) == 0:
        return X

    n_trials = X.shape[0]
    for i, n in enumerate(nuisances):
        if np.asarray(n).shape[0] != n_trials:
            raise ValueError(
                f"Residualization mismatch: nuisance[{i}] len={np.asarray(n).shape[0]} vs X trials={n_trials}"
            )

    A = _design_matrix_from_nuisances(nuisances, models=models, zscore=True)

    Y = X.reshape(n_trials, -1)
    beta, *_ = np.linalg.lstsq(A, Y, rcond=None)
    Y_hat = A @ beta
    Y_resid = Y - Y_hat
    return Y_resid.reshape(X.shape)


# =====================================================================
# Trial selection (standard)
# =====================================================================

def _get_levels_and_rt_from_epochs(epo: mne.Epochs) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if epo.metadata is None:
        raise ValueError("Epochs has no metadata.")
    md = epo.metadata.reset_index(drop=True)

    for col in [DEC_MONEY_COL, DEC_PAIN_COL]:
        if col not in md.columns:
            raise ValueError(f"Missing metadata column '{col}'. Have: {md.columns.tolist()}")

    if RT_COL not in md.columns:
        raise ValueError(f"Missing RT column '{RT_COL}' in metadata.")

    money_levels = parse_stim_code_to_level(md[DEC_MONEY_COL], "m")
    pain_levels = parse_stim_code_to_level(md[DEC_PAIN_COL], "p")
    rt = pd.to_numeric(md[RT_COL], errors="coerce").to_numpy(dtype=float)

    return money_levels, pain_levels, rt


def select_trials_decision_binary(epo: mne.Epochs, which: str, control_resid_by_other: bool):
    """
    BINARY:
      - keep 20/40/80/100 (drop 60)
      - y = 0/1
      - drop trials with invalid RT
      - if control_resid_by_other: residualize by [other cue, RT] (lin, lin)
    Returns: X, y, times, md_f, levels_f, other_f, rt_f, info(dict)
    """
    money_levels, pain_levels, rt = _get_levels_and_rt_from_epochs(epo)

    levels = money_levels if which == "money" else pain_levels
    other = pain_levels if which == "money" else money_levels

    n_in = int(len(epo))
    keep = np.isin(levels, BIN_KEEP_LEVELS) & np.isfinite(rt)
    n_after_level = int(keep.sum())

    epo_f = epo.copy()[keep]
    md_f = epo_f.metadata.reset_index(drop=True)

    levels_f = levels[keep]
    other_f = other[keep]
    rt_f = rt[keep]

    if len(epo_f) < 10:
        raise ValueError(f"Too few trials after filtering for decision {which} binary. n={len(epo_f)}")

    X = epo_f.get_data()
    times = epo_f.times.copy()

    if control_resid_by_other:
        X = residualize_X_by_nuisances(
            X,
            nuisances=[other_f.astype(float), rt_f.astype(float)],
            models=["lin", "lin"],
        )

    y = make_binary_labels(levels_f.astype(float))
    if np.any(y < 0):
        raise ValueError(f"Unlabeled trials exist after filtering. Levels seen: {np.unique(levels_f)}")

    info = dict(
        n_in=n_in,
        n_after_level=n_after_level,
        n_after_nan=int(len(y)),
        n_drop_level=int(n_in - n_after_level),
        n_drop_nan=0,
        n_low=int(np.sum(y == 0)),
        n_high=int(np.sum(y == 1)),
    )
    return X, y, times, md_f, levels_f, other_f, rt_f, info


# =====================================================================
# Trial selection (cross-generalization)
# =====================================================================

def select_trials_crossgen_binary(epo: mne.Epochs, control_by_other_train: bool, train: str):
    """
    Cross-label binary:
      - keep trials where BOTH money and pain are in BIN_KEEP_LEVELS
      - drop trials with invalid RT
      - if control_by_other_train: residualize by [other-of-train, RT] (lin, lin)
    """
    money_levels, pain_levels, rt = _get_levels_and_rt_from_epochs(epo)

    n_in = int(len(epo))
    keep = (
        np.isin(money_levels, BIN_KEEP_LEVELS)
        & np.isin(pain_levels, BIN_KEEP_LEVELS)
        & np.isfinite(rt)
    )
    n_after_level = int(keep.sum())

    epo_f = epo.copy()[keep]
    md_f = epo_f.metadata.reset_index(drop=True)

    m = money_levels[keep]
    p = pain_levels[keep]
    rt_f = rt[keep]

    if len(epo_f) < 10:
        raise ValueError(f"Too few trials after filtering for crossgen binary. n={len(epo_f)}")

    y_money = make_binary_labels(m.astype(float))
    y_pain = make_binary_labels(p.astype(float))
    if np.any(y_money < 0) or np.any(y_pain < 0):
        raise ValueError("Crossgen binary: unlabeled trials exist after filtering (should not happen).")

    X = epo_f.get_data()
    times = epo_f.times.copy()

    if control_by_other_train:
        nuis_other = p if train == "money" else m
        X = residualize_X_by_nuisances(
            X,
            nuisances=[nuis_other.astype(float), rt_f.astype(float)],
            models=["lin", "lin"],
        )

    info = dict(
        n_in=n_in,
        n_after_level=n_after_level,
        n_after_nan=int(len(epo_f)),
        n_drop_level=int(n_in - n_after_level),
        n_drop_nan=0,
    )

    if train == "money":
        return X, y_money, y_pain, times, md_f, m, p, rt_f, info
    else:
        return X, y_pain, y_money, times, md_f, m, p, rt_f, info


# =====================================================================
# Decoders (standard)
# =====================================================================

def _make_cv(y: np.ndarray, groups: np.ndarray | None) -> tuple[object, str]:
    cv = None
    if groups is not None:
        groups = np.asarray(groups)
        ok = ~pd.isna(groups)
        if ok.sum() == len(groups):
            n_groups = len(np.unique(groups))
            if n_groups >= 2:
                n_splits = min(N_SPLITS, n_groups)
                cv = GroupKFold(n_splits=n_splits)
                return cv, "GroupKFold"

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    return cv, "StratifiedKFold"


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

    cv, cv_used = _make_cv(y_use, groups)

    scores = cross_val_multiscore(
        time_decod, X, y_use,
        cv=cv,
        groups=groups if isinstance(cv, GroupKFold) else None,
        n_jobs=1
    )
    return scores.mean(axis=0), cv_used


# =====================================================================
# Cross-generalization decoding (custom)
# =====================================================================

def _score_binary_auc(est, X2d, y_true):
    if hasattr(est, "predict_proba"):
        s = est.predict_proba(X2d)[:, 1]
    elif hasattr(est, "decision_function"):
        s = est.decision_function(X2d)
    else:
        s = est.predict(X2d)
    return float(roc_auc_score(y_true, s))


def _score_metric(metric: str, est, X2d, y_true):
    if metric == "accuracy":
        yhat = est.predict(X2d)
        return float(accuracy_score(y_true, yhat))
    if metric == "balanced_accuracy":
        yhat = est.predict(X2d)
        return float(balanced_accuracy_score(y_true, yhat))
    if metric == "roc_auc":
        return _score_binary_auc(est, X2d, y_true)
    raise ValueError(f"Unknown metric: {metric}")


def crossgen_time_resolved(
    X: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    groups: np.ndarray | None,
    *,
    estimator_pipeline,
    metric: str,
    shuffle_train: bool,
) -> tuple[np.ndarray, str]:
    """
    Cross-label generalization at SAME TIMEPOINTS (diagonal time series):
      Fit on (X_train[:,:,t], y_train) and evaluate on (X_test[:,:,t], y_test)
    Returns mean over folds: scores[t]
    """
    rng = np.random.default_rng(RANDOM_STATE)
    y_tr_use = rng.permutation(y_train) if shuffle_train else y_train

    cv, cv_used = _make_cv(y_tr_use, groups)
    splits = list(cv.split(X, y_tr_use, groups=groups if isinstance(cv, GroupKFold) else None))

    n_times = X.shape[2]
    fold_scores = np.zeros((len(splits), n_times), dtype=float)

    for fi, (tr, te) in enumerate(splits):
        Xtr = X[tr]
        Xte = X[te]
        ytr = y_tr_use[tr]
        yte = y_test[te]

        for t in range(n_times):
            est = clone(estimator_pipeline)
            est.fit(Xtr[:, :, t], ytr)
            fold_scores[fi, t] = _score_metric(metric, est, Xte[:, :, t], yte)

    return fold_scores.mean(axis=0), cv_used


def crossgen_temporal_generalization_matrix(
    X: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    groups: np.ndarray | None,
    *,
    estimator_pipeline,
    metric: str,
    shuffle_train: bool,
    time_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, str]:
    """
    Cross-label temporal generalization matrix:
      For each train-time i and test-time j:
        fit on X_train[:,:,i] with y_train
        test on X_test[:,:,j] with y_test
    Returns:
      M (nT x nT) averaged over folds,
      diag (nT,),
      cv_used
    """
    rng = np.random.default_rng(RANDOM_STATE)
    y_tr_use = rng.permutation(y_train) if shuffle_train else y_train

    cv, cv_used = _make_cv(y_tr_use, groups)
    splits = list(cv.split(X, y_tr_use, groups=groups if isinstance(cv, GroupKFold) else None))

    T = np.asarray(time_idx, dtype=int)
    nT = T.size
    fold_mats = np.zeros((len(splits), nT, nT), dtype=float)

    for fi, (tr, te) in enumerate(splits):
        Xtr = X[tr]
        Xte = X[te]
        ytr = y_tr_use[tr]
        yte = y_test[te]

        XtrT = Xtr[:, :, T]
        XteT = Xte[:, :, T]

        for ii in range(nT):
            est = clone(estimator_pipeline)
            est.fit(XtrT[:, :, ii], ytr)
            for jj in range(nT):
                fold_mats[fi, ii, jj] = _score_metric(metric, est, XteT[:, :, jj], yte)

    M = fold_mats.mean(axis=0)
    diag = np.diag(M).copy()
    return M, diag, cv_used


# =====================================================================
# Stats + plotting
# =====================================================================

def group_cluster_metric(scores_by_subj: np.ndarray, times: np.ndarray, *, chance: float, tail: int):
    """
    TFCE ONLY. No fallback procedures.
    """
    X = scores_by_subj - chance
    tfce_thresh = dict(start=0.0, step=0.2)

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

    p_map = np.ones(len(times), dtype=float)
    clusters_fixed = []
    for cl, p in zip(clusters, cluster_pv):
        m = _cluster_to_bool_mask(cl, len(times))
        clusters_fixed.append(m)
        if m is None or not m.any():
            continue
        p_map[m] = np.minimum(p_map[m], p)

    return dict(T_obs=T_obs, clusters=clusters_fixed, cluster_pv=cluster_pv, p_map=p_map, t_thresh=t_thresh_used)


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


def plot_heatmap(M: np.ndarray, times: np.ndarray, title: str, out_png: Path, vmin: float, vmax: float, chance: float):
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.imshow(
        M, origin="lower", aspect="auto",
        extent=[times[0], times[-1], times[0], times[-1]],
        vmin=vmin, vmax=vmax
    )
    ax.set_title(title)
    ax.set_xlabel("Test time (s)")
    ax.set_ylabel("Train time (s)")
    ax.axhline(0, linestyle="--", linewidth=0.8)
    ax.axvline(0, linestyle="--", linewidth=0.8)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(f"Score (chance={chance})")
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
# RUN: Standard Binary
# =====================================================================

def run_binary(which: str, shuffle: bool, tag: str, control_by_other: bool):
    OUT = OUT_DIR_BIN
    DBG = DEBUG_DIR_BIN

    analysis_name = f"Decision BINARY {which} ({tag})"
    log_print(f"\n>>> START: {analysis_name} | shuffle={shuffle} | control_by_other={control_by_other}\n")

    subs = list_subjects(DERIV_DIR)
    scores_all_auc, scores_all_bacc, included, skipped = [], [], [], []
    times_ref = None
    subj_records: list[dict] = []

    trial_stats = dict(
        n_sub_total=len(subs),
        total_trials_in=0,
        total_trials_after_badtrial=0,
        total_trials_after_level=0,
        total_trials_after_nan=0,
        n_trials_per_sub=[],
    )

    pbar = tqdm(subs, desc=f"BIN {which} {tag}{'_shuf' if shuffle else ''}", unit="sub", dynamic_ncols=True, leave=True)

    for sub in pbar:
        try:
            epo = load_decision_epochs(sub)
            beh = load_decision_beh(sub)
            epo = merge_beh_into_epochs_decision(epo, beh, sub=sub, debug_dir=DBG)

            if epo.metadata is None:
                raise RuntimeError(f"{sub}: metadata is None after merge (should never happen).")

            n0 = int(len(epo))
            trial_stats["total_trials_in"] += n0

            if "badtrial" in epo.metadata.columns:
                bad = epo.metadata["badtrial"].fillna(0).astype(int).to_numpy()
                n_bad = int(bad.sum())
                if n_bad > 0:
                    epo = epo.copy()[bad == 0]
            else:
                n_bad = 0
                log_print(f"{sub}: WARNING no 'badtrial' column found in epochs.metadata (not dropping trials)")

            n1 = int(len(epo))
            trial_stats["total_trials_after_badtrial"] += n1

            md = epo.metadata.reset_index(drop=True)
            decision_alignment_sanity_check(md, sub=sub, debug_dir=DBG)

            if md[DEC_MONEY_COL].isna().any() or md[DEC_PAIN_COL].isna().any():
                md.head(50).to_csv(DBG / f"{sub}_md_aftermerge_head.csv", index=False)
                raise RuntimeError(f"{sub}: NaNs in merged decision labels. Saved {sub}_md_aftermerge_head.csv")

            if RT_COL not in md.columns:
                md.head(50).to_csv(DBG / f"{sub}_md_aftermerge_head.csv", index=False)
                raise RuntimeError(f"{sub}: Missing RT column '{RT_COL}' after merge. Saved {sub}_md_aftermerge_head.csv")

            if RESAMPLE_SFREQ is not None:
                epo = epo.copy().resample(RESAMPLE_SFREQ, npad="auto")

            X, y, times, md_used, levels_f, other_f, rt_f, info = select_trials_decision_binary(
                epo, which=which, control_resid_by_other=control_by_other
            )

            trial_stats["total_trials_after_level"] += int(info["n_after_level"])
            trial_stats["total_trials_after_nan"] += int(info["n_after_nan"])
            trial_stats["n_trials_per_sub"].append(int(len(y)))

            save_decision_trial_counts(OUT, sub=sub, tag=tag, which=which, levels=levels_f, other_levels=other_f, rt=rt_f)
            groups = md_used[KEY_BLOCK].to_numpy() if (KEY_BLOCK in md_used.columns) else None

            if times_ref is None:
                times_ref = times
            else:
                if len(times) != len(times_ref) or np.max(np.abs(times - times_ref)) > 1e-9:
                    raise RuntimeError("Time axis mismatch across subjects.")

            scores_auc, cv_used = subject_decode_binary(X, y, shuffle=shuffle, groups=groups, metric="roc_auc")
            scores_bacc, _ = subject_decode_binary(X, y, shuffle=shuffle, groups=groups, metric="balanced_accuracy")

            scores_all_auc.append(scores_auc)
            scores_all_bacc.append(scores_bacc)
            included.append(sub)

            append_subject_summary(
                subj_records,
                sub=sub, which=which, shuffle=shuffle, tag=tag,
                n_trials=len(y), groups=groups, cv_used=cv_used,
                extra=dict(
                    n_low=int(np.sum(y == 0)),
                    n_high=int(np.sum(y == 1)),
                    control_by_other=bool(control_by_other),
                    control_rt=bool(control_by_other),
                    other_lin=bool(control_by_other),
                    rt_lin=bool(control_by_other),
                    n_badtrial_dropped=int(n_bad),
                    n_drop_level=int(info["n_drop_level"]),
                    n_drop_nan=int(info["n_drop_nan"]),
                )
            )

        except Exception as e:
            skipped.append((sub, str(e)))
            log_print(f"Skipped {sub} BIN({which},{tag}{'_shuf' if shuffle else ''}): {e}")

    if len(scores_all_auc) < 8:
        raise RuntimeError(f"Too few subjects included for group stats BIN({which},{tag}): n={len(scores_all_auc)}")

    scores_all_auc = np.stack(scores_all_auc, axis=0)
    scores_all_bacc = np.stack(scores_all_bacc, axis=0)
    times = times_ref

    time_mask = (times >= TMIN_STAT) & (times <= TMAX_STAT)
    times_stat = times[time_mask]

    nuisance_model = "other=lin + rt=lin"
    control_label = "ctrlOtherRT"

    # AUC (tail=1)
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
        control_rt=True,
        other_lin=True,
        rt_lin=True,
    )

    pd.DataFrame(scores_all_auc, index=included, columns=np.round(times, 6)).to_csv(
        OUT / f"{tag_auc}_scores_by_subject.csv"
    )

    plot_group_metric(
        auc_stat, times_stat, stats_auc["p_map"],
        title=f"Decision {which} ({tag.upper()} | {'SHUFFLED' if shuffle else 'REAL'}): AUC",
        out_png=OUT / f"{tag_auc}_group_plot.png",
        ylabel="Decoding (AUC)",
        chance=CHANCE_BIN,
        ylim=(0.35, 0.85),
    )

    save_group_summaries(
        tag=tag_auc, scores_all=auc_stat, times=times_stat, included=included,
        stats_out=stats_auc, out_dir=OUT, alpha=ALPHA_CLUSTER, chance=CHANCE_BIN,
    )

    summarize_group_results_text(
        analysis_name=analysis_name,
        metric_name="AUC",
        tag=tag,
        which=which,
        shuffle=shuffle,
        control=control_label,
        nuisance_model=nuisance_model,
        out_dir=OUT,
        scores_all=auc_stat,
        times=times_stat,
        chance=CHANCE_BIN,
        stats_out=stats_auc,
        alpha=ALPHA_CLUSTER,
        included=included,
        skipped=skipped,
        trial_stats=trial_stats,
    )

    # bAcc (tail=1)
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
        control_rt=True,
        other_lin=True,
        rt_lin=True,
    )

    pd.DataFrame(scores_all_bacc, index=included, columns=np.round(times, 6)).to_csv(
        OUT / f"{tag_bacc}_scores_by_subject.csv"
    )

    plot_group_metric(
        bacc_stat, times_stat, stats_bacc["p_map"],
        title=f"Decision {which} ({tag.upper()} | {'SHUFFLED' if shuffle else 'REAL'}): Balanced accuracy",
        out_png=OUT / f"{tag_bacc}_group_plot.png",
        ylabel="Decoding (balanced accuracy)",
        chance=CHANCE_BIN,
        ylim=(0.35, 0.85),
    )

    save_group_summaries(
        tag=tag_bacc, scores_all=bacc_stat, times=times_stat, included=included,
        stats_out=stats_bacc, out_dir=OUT, alpha=ALPHA_CLUSTER, chance=CHANCE_BIN,
    )

    summarize_group_results_text(
        analysis_name=analysis_name,
        metric_name="BalancedAccuracy",
        tag=tag,
        which=which,
        shuffle=shuffle,
        control=control_label,
        nuisance_model=nuisance_model,
        out_dir=OUT,
        scores_all=bacc_stat,
        times=times_stat,
        chance=CHANCE_BIN,
        stats_out=stats_bacc,
        alpha=ALPHA_CLUSTER,
        included=included,
        skipped=skipped,
        trial_stats=trial_stats,
    )

    suffix = "_shuffle" if shuffle else ""
    fname = f"decision_{tag}_{which}{suffix}_subject_summary.csv"
    pd.DataFrame(subj_records).to_csv(OUT / fname, index=False)

    log_print(f"\n<<< DONE: {analysis_name} | included_subs={len(included)} | skipped_subs={len(skipped)}\n")


# =====================================================================
# RUN: Cross-generalization + heatmaps
# =====================================================================

def run_crossgen_binary(train: str, shuffle: bool, tag: str, control_by_other_train: bool):
    """
    train: "money" or "pain"
    Evaluates diagonal + heatmap for:
      - balanced_accuracy
      - roc_auc
    Output in OUT_DIR_XGEN_BIN

    ctrlOtherTrain now means:
      - other-of-train cue (lin)
      - RT (lin)
    """
    OUT = OUT_DIR_XGEN_BIN
    DBG = DEBUG_DIR_XGEN_BIN

    direction = f"{train}_to_{'pain' if train=='money' else 'money'}"
    analysis_name = f"Decision XGEN BIN {direction} ({tag})"
    log_print(f"\n>>> START: {analysis_name} | shuffle={shuffle} | control_by_other_train={control_by_other_train}\n")

    subs = list_subjects(DERIV_DIR)
    included, skipped = [], []
    times_ref = None

    diag_bacc_all = []
    diag_auc_all = []
    mats_bacc = []
    mats_auc = []

    trial_stats = dict(
        n_sub_total=len(subs),
        total_trials_in=0,
        total_trials_after_badtrial=0,
        total_trials_after_level=0,
        total_trials_after_nan=0,
        n_trials_per_sub=[],
    )

    clf_bin = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            solver="liblinear",
            max_iter=2000,
            random_state=RANDOM_STATE,
            class_weight="balanced",
        )
    )

    pbar = tqdm(subs, desc=f"XGEN BIN {direction} {tag}{'_shuf' if shuffle else ''}",
                unit="sub", dynamic_ncols=True, leave=True)

    for sub in pbar:
        try:
            epo = load_decision_epochs(sub)
            beh = load_decision_beh(sub)
            epo = merge_beh_into_epochs_decision(epo, beh, sub=sub, debug_dir=DBG)

            if epo.metadata is None:
                raise RuntimeError(f"{sub}: metadata is None after merge.")

            n0 = int(len(epo))
            trial_stats["total_trials_in"] += n0

            if "badtrial" in epo.metadata.columns:
                bad = epo.metadata["badtrial"].fillna(0).astype(int).to_numpy()
                n_bad = int(bad.sum())
                if n_bad > 0:
                    epo = epo.copy()[bad == 0]

            n1 = int(len(epo))
            trial_stats["total_trials_after_badtrial"] += n1

            if RESAMPLE_SFREQ is not None:
                epo = epo.copy().resample(RESAMPLE_SFREQ, npad="auto")

            X, y_tr, y_te, times, md_used, m_levels, p_levels, rt_f, info = select_trials_crossgen_binary(
                epo, control_by_other_train=control_by_other_train, train=train
            )

            trial_stats["total_trials_after_level"] += int(info["n_after_level"])
            trial_stats["total_trials_after_nan"] += int(info["n_after_nan"])
            trial_stats["n_trials_per_sub"].append(int(len(y_tr)))

            groups = md_used[KEY_BLOCK].to_numpy() if (KEY_BLOCK in md_used.columns) else None

            if times_ref is None:
                times_ref = times
            else:
                if len(times) != len(times_ref) or np.max(np.abs(times - times_ref)) > 1e-9:
                    raise RuntimeError("Time axis mismatch across subjects.")

            diag_bacc, _ = crossgen_time_resolved(
                X, y_tr, y_te, groups,
                estimator_pipeline=clf_bin,
                metric="balanced_accuracy",
                shuffle_train=shuffle,
            )
            diag_auc, _ = crossgen_time_resolved(
                X, y_tr, y_te, groups,
                estimator_pipeline=clf_bin,
                metric="roc_auc",
                shuffle_train=shuffle,
            )

            # heatmaps ONLY for ctrlOtherRT / its shuffle (raw dropped entirely anyway)
            time_mask = (times >= TMIN_STAT) & (times <= TMAX_STAT)
            idx = np.where(time_mask)[0][::max(1, int(HEATMAP_DECIM))]

            M_bacc, _, _ = crossgen_temporal_generalization_matrix(
                X, y_tr, y_te, groups,
                estimator_pipeline=clf_bin,
                metric="balanced_accuracy",
                shuffle_train=shuffle,
                time_idx=idx,
            )
            M_auc, _, _ = crossgen_temporal_generalization_matrix(
                X, y_tr, y_te, groups,
                estimator_pipeline=clf_bin,
                metric="roc_auc",
                shuffle_train=shuffle,
                time_idx=idx,
            )

            diag_bacc_all.append(diag_bacc)
            diag_auc_all.append(diag_auc)
            mats_bacc.append(M_bacc)
            mats_auc.append(M_auc)
            included.append(sub)

        except Exception as e:
            skipped.append((sub, str(e)))
            log_print(f"Skipped {sub} XGEN BIN({train},{tag}{'_shuf' if shuffle else ''}): {e}")

    if len(included) < 8:
        raise RuntimeError(f"Too few subjects included for XGEN BIN({train},{tag}): n={len(included)}")

    times = times_ref
    diag_bacc_all = np.stack(diag_bacc_all, axis=0)
    diag_auc_all = np.stack(diag_auc_all, axis=0)

    time_mask = (times >= TMIN_STAT) & (times <= TMAX_STAT)
    times_stat = times[time_mask]
    bacc_stat = diag_bacc_all[:, time_mask]
    auc_stat = diag_auc_all[:, time_mask]

    stats_bacc = group_cluster_metric(bacc_stat, times_stat, chance=CHANCE_BIN, tail=1)
    stats_auc = group_cluster_metric(auc_stat, times_stat, chance=CHANCE_BIN, tail=1)

    suffix = "_shuffle" if shuffle else ""
    ctrl = "ctrlOtherTrainRT"

    tag_bacc = f"xgen_{ctrl}_{direction}_bacc{suffix}"
    tag_auc = f"xgen_{ctrl}_{direction}_auc{suffix}"

    np.savez(
        OUT / f"{tag_bacc}_diag_group_results.npz",
        scores_by_subj=diag_bacc_all,
        times=times,
        included=np.array(included, dtype=object),
        skipped=np.array(skipped, dtype=object),
        T_obs=stats_bacc["T_obs"],
        p_map=stats_bacc["p_map"],
        cluster_pv=stats_bacc["cluster_pv"],
        t_thresh=stats_bacc["t_thresh"],
        chance=CHANCE_BIN,
        control_rt=True,
        other_lin=True,
        rt_lin=True,
    )
    np.savez(
        OUT / f"{tag_auc}_diag_group_results.npz",
        scores_by_subj=diag_auc_all,
        times=times,
        included=np.array(included, dtype=object),
        skipped=np.array(skipped, dtype=object),
        T_obs=stats_auc["T_obs"],
        p_map=stats_auc["p_map"],
        cluster_pv=stats_auc["cluster_pv"],
        t_thresh=stats_auc["t_thresh"],
        chance=CHANCE_BIN,
        control_rt=True,
        other_lin=True,
        rt_lin=True,
    )

    plot_group_metric(
        bacc_stat, times_stat, stats_bacc["p_map"],
        title=f"XGEN BIN {direction} ({ctrl.upper()} | {'SHUFFLED' if shuffle else 'REAL'}): bAcc (diag)",
        out_png=OUT / f"{tag_bacc}_diag_plot.png",
        ylabel="Cross-gen (balanced accuracy)",
        chance=CHANCE_BIN,
        ylim=(0.35, 0.85),
    )
    plot_group_metric(
        auc_stat, times_stat, stats_auc["p_map"],
        title=f"XGEN BIN {direction} ({ctrl.upper()} | {'SHUFFLED' if shuffle else 'REAL'}): AUC (diag)",
        out_png=OUT / f"{tag_auc}_diag_plot.png",
        ylabel="Cross-gen (AUC)",
        chance=CHANCE_BIN,
        ylim=(0.35, 0.85),
    )

    save_group_summaries(
        tag=tag_bacc + "_diag",
        scores_all=bacc_stat,
        times=times_stat,
        included=included,
        stats_out=stats_bacc,
        out_dir=OUT,
        alpha=ALPHA_CLUSTER,
        chance=CHANCE_BIN,
    )
    save_group_summaries(
        tag=tag_auc + "_diag",
        scores_all=auc_stat,
        times=times_stat,
        included=included,
        stats_out=stats_auc,
        out_dir=OUT,
        alpha=ALPHA_CLUSTER,
        chance=CHANCE_BIN,
    )

    nuisance_model = "other(train)=lin + rt=lin"
    control_label = "ctrlOtherTrainRT"

    summarize_group_results_text(
        analysis_name=analysis_name,
        metric_name="BalancedAccuracy_Diag",
        tag=tag,
        which=f"{direction}",
        shuffle=shuffle,
        control=control_label,
        nuisance_model=nuisance_model,
        out_dir=OUT,
        scores_all=bacc_stat,
        times=times_stat,
        chance=CHANCE_BIN,
        stats_out=stats_bacc,
        alpha=ALPHA_CLUSTER,
        included=included,
        skipped=skipped,
        trial_stats=trial_stats,
    )
    summarize_group_results_text(
        analysis_name=analysis_name,
        metric_name="AUC_Diag",
        tag=tag,
        which=f"{direction}",
        shuffle=shuffle,
        control=control_label,
        nuisance_model=nuisance_model,
        out_dir=OUT,
        scores_all=auc_stat,
        times=times_stat,
        chance=CHANCE_BIN,
        stats_out=stats_auc,
        alpha=ALPHA_CLUSTER,
        included=included,
        skipped=skipped,
        trial_stats=trial_stats,
    )

    # heatmaps
    mats_bacc = np.stack(mats_bacc, axis=0)
    mats_auc = np.stack(mats_auc, axis=0)
    M_bacc_mean = mats_bacc.mean(axis=0)
    M_auc_mean = mats_auc.mean(axis=0)

    idx = np.where(time_mask)[0][::max(1, int(HEATMAP_DECIM))]
    times_hm = times[idx]

    np.savez(
        OUT / f"{tag_bacc}_heatmaps.npz",
        mats_by_subj=mats_bacc,
        mean_mat=M_bacc_mean,
        times=times_hm,
        included=np.array(included, dtype=object),
        chance=CHANCE_BIN,
        decim=int(HEATMAP_DECIM),
        control_rt=True,
        other_lin=True,
        rt_lin=True,
    )
    np.savez(
        OUT / f"{tag_auc}_heatmaps.npz",
        mats_by_subj=mats_auc,
        mean_mat=M_auc_mean,
        times=times_hm,
        included=np.array(included, dtype=object),
        chance=CHANCE_BIN,
        decim=int(HEATMAP_DECIM),
        control_rt=True,
        other_lin=True,
        rt_lin=True,
    )

    plot_heatmap(
        M_bacc_mean, times_hm,
        title=f"XGEN BIN {direction} ({ctrl.upper()} | {'SHUFFLED' if shuffle else 'REAL'}): bAcc heatmap",
        out_png=OUT / f"{tag_bacc}_heatmap.png",
        vmin=0.35, vmax=0.85, chance=CHANCE_BIN
    )
    plot_heatmap(
        M_auc_mean, times_hm,
        title=f"XGEN BIN {direction} ({ctrl.upper()} | {'SHUFFLED' if shuffle else 'REAL'}): AUC heatmap",
        out_png=OUT / f"{tag_auc}_heatmap.png",
        vmin=0.35, vmax=0.85, chance=CHANCE_BIN
    )

    log_print(f"\n<<< DONE: {analysis_name} | included_subs={len(included)} | skipped_subs={len(skipped)}\n")


# =====================================================================
# Main
# =====================================================================

def main():
    mne.set_log_level("WARNING")
    log_print(f"\n=== mvpa_decision_step1_conserv2 START ===")
    log_print(f"DATA_DIR: {RAW_DIR}")
    log_print(f"OUT_DIR:  {OUT_DIR}")
    log_print(f"RESAMPLE_SFREQ: {RESAMPLE_SFREQ}")
    log_print(f"N_PERM: {N_PERM} | ALPHA_CLUSTER: {ALPHA_CLUSTER} | STATS WINDOW: [{TMIN_STAT},{TMAX_STAT}] s")
    log_print("Control model (when enabled): other cue = lin + RT = lin")
    log_print("TFCE only")
    log_print("RAW analyses removed; only ctrlOtherRT / ctrlOtherTrainRT kept\n")

    # -------------------------
    # Binary (standard) - ctrlOtherRT only
    if RUN_BINARY and RUN_CONTROL_BY_OTHER:
        run_binary("money", shuffle=False, tag="ctrlOtherRT", control_by_other=True)
        run_binary("pain",  shuffle=False, tag="ctrlOtherRT", control_by_other=True)

        if RUN_SHUFFLE:
            run_binary("money", shuffle=True, tag="ctrlOtherRT", control_by_other=True)
            run_binary("pain",  shuffle=True, tag="ctrlOtherRT", control_by_other=True)

    # -------------------------
    # Cross-generalization (binary only) - ctrlOtherTrainRT only
    if RUN_CROSS_GENERALIZATION and RUN_CONTROL_BY_OTHER:
        run_crossgen_binary(train="money", shuffle=False, tag="ctrlOtherTrainRT", control_by_other_train=True)
        run_crossgen_binary(train="pain",  shuffle=False, tag="ctrlOtherTrainRT", control_by_other_train=True)

        if RUN_SHUFFLE:
            run_crossgen_binary(train="money", shuffle=True, tag="ctrlOtherTrainRT", control_by_other_train=True)
            run_crossgen_binary(train="pain",  shuffle=True, tag="ctrlOtherTrainRT", control_by_other_train=True)

    log_print(f"\n=== mvpa_decision_step1_conserv2 DONE ===\n")


if __name__ == "__main__":
    main()