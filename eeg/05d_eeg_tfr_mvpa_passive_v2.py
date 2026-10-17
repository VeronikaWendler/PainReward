#!/usr/bin/env python3
"""
05d_eeg_tfr_mvpa_passive_v2.py

Redesigned passive-phase spectral MVPA for PainReward.

Uses the EXISTING single-trial TFR files. It does not recompute TFRs.

PRIMARY family: ordinal magnitude decoding (levels 1..5)
- pain and money separately
- StandardScaler -> Ridge(alpha=1)
- leave-one-passive-block-out GroupKFold
- pooled out-of-fold Pearson r
- frequency-resolved 4..45 Hz x time decoding
- separate 2D cluster permutation for pain and money
- Holm correction across the two domain omnibus tests
- canonical-band temporal cluster tests with Holm across 4 bands
  separately within pain and within money

SECONDARY family: high-vs-low decoding
- low = levels 1,2
- high = levels 4,5
- level 3 excluded
- StandardScaler -> RidgeClassifier(alpha=1)
- leave-one-passive-block-out GroupKFold
- pooled out-of-fold ROC-AUC
- same frequency-resolved and canonical-band inferential structure

High-vs-low does NOT use more total trials than the ordinal analysis:
it discards level 3 (~20%). It is simpler and gives more observations per
binary class than per single level, so it can be a useful complementary
secondary analysis.

Formal inference is restricted to cue-present time 0..1.5 s. Pre-cue scores
are retained for leakage/QC diagnostics. Late clusters are flagged as
boundary-sensitive because Morlet temporal support extends around the nominal
centre time near cue offset.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from mne.stats import combine_adjacency, permutation_cluster_1samp_test
from scipy import stats
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 23
BASE_PATH = Path(os.getenv("basepath", Path(__file__).parent.parent.parent))
PARTICIPANTS_TSV = BASE_PATH / "participants.tsv"

OUT_ROOT = BASE_PATH / "derivatives" / "statistics" / "tfr_mvpa_passive_v2"
SCORE_DIR = OUT_ROOT / "participant_scores"
FIG_DIR = OUT_ROOT / "figures"

ATTRIBUTES = {"pain": "p", "money": "m"}

BANDS: Dict[str, Tuple[float, float]] = {
    "theta": (4.0, 7.0),
    "alpha": (8.0, 13.0),
    "beta": (14.0, 30.0),
    "low_gamma": (31.0, 45.0),
}

RIDGE_ALPHA = 1.0
CLUSTER_FORMING_ALPHA = 0.05
INFER_TMIN = 0.0
INFER_TMAX = 1.5
CONSERVATIVE_THETA_SAFE_TMAX = 1.10


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PainReward passive spectral MVPA v2")
    p.add_argument("--subjects", nargs="*", default=None)
    p.add_argument("--n-permutations", type=int, default=5000)
    p.add_argument(
        "--n-jobs",
        type=int,
        default=max(1, int(os.getenv("SLURM_CPUS_PER_TASK", "1"))),
    )
    p.add_argument("--overwrite-scores", action="store_true")
    p.add_argument("--group-only", action="store_true")
    return p.parse_args()


def tfr_paths(sub: str) -> Dict[str, Path]:
    d = BASE_PATH / "derivatives" / sub / "eeg" / "tfr"
    stem = f"{sub}_passive_cue"
    return {
        "power": d / f"{stem}_power.npy",
        "times": d / f"{stem}_times.npy",
        "freqs": d / f"{stem}_freqs.npy",
        "metadata": d / f"{stem}_metadata.csv",
        "channels": d / f"{stem}_channels.json",
    }


def score_path(sub: str) -> Path:
    return SCORE_DIR / f"{sub}_passive_tfr_mvpa_v2.npz"


def load_subject_list(explicit: Sequence[str] | None) -> List[str]:
    if explicit:
        subs = sorted(dict.fromkeys(str(x) for x in explicit))
    else:
        if not PARTICIPANTS_TSV.exists():
            raise FileNotFoundError(f"participants.tsv missing: {PARTICIPANTS_TSV}")
        df = pd.read_csv(PARTICIPANTS_TSV, sep="\t")
        if "participant_id" not in df.columns:
            raise KeyError(
                f"{PARTICIPANTS_TSV} missing participant_id; columns={list(df.columns)}"
            )
        subs = sorted(df["participant_id"].dropna().astype(str).unique().tolist())

    if not subs:
        raise RuntimeError("No participants selected.")

    for sub in subs:
        missing = [str(p) for p in tfr_paths(sub).values() if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"{sub}: missing required TFR inputs:\n" + "\n".join(missing)
            )
    return subs


def make_regressor() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=RIDGE_ALPHA)),
    ])


def make_classifier() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("ridge_classifier", RidgeClassifier(alpha=RIDGE_ALPHA)),
    ])


def make_group_splits(groups: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
    unique = np.unique(groups)
    if len(unique) < 2:
        raise RuntimeError(f"Need >=2 passive blocks, got {unique.tolist()}")
    cv = GroupKFold(n_splits=len(unique))
    dummy = np.zeros(len(groups))
    return [(tr, te) for tr, te in cv.split(np.zeros((len(groups), 1)), dummy, groups)]


def safe_corr(y: np.ndarray, pred: np.ndarray) -> float:
    if np.std(y) == 0 or np.std(pred) == 0:
        return 0.0
    r = float(np.corrcoef(y, pred)[0, 1])
    return r if np.isfinite(r) else 0.0


def regression_timecourse(
    X: np.ndarray, y: np.ndarray, splits: List[Tuple[np.ndarray, np.ndarray]]
) -> np.ndarray:
    out = np.empty(X.shape[2], dtype=np.float32)
    for ti in range(X.shape[2]):
        Xt = np.asarray(X[:, :, ti], dtype=np.float64)
        pred = np.full(len(y), np.nan, dtype=float)
        for train, test in splits:
            if np.std(y[train]) == 0:
                raise RuntimeError("Regression training fold has zero y variance.")
            model = make_regressor()
            model.fit(Xt[train], y[train])
            pred[test] = model.predict(Xt[test])
        if not np.isfinite(pred).all():
            raise RuntimeError("Regression produced non-finite OOF predictions.")
        out[ti] = safe_corr(y, pred)
    return out


def classification_timecourse(
    X: np.ndarray, y: np.ndarray, splits: List[Tuple[np.ndarray, np.ndarray]]
) -> np.ndarray:
    if set(np.unique(y).tolist()) != {0, 1}:
        raise RuntimeError(f"High/low labels must be 0/1, got {np.unique(y)}")
    out = np.empty(X.shape[2], dtype=np.float32)
    for ti in range(X.shape[2]):
        Xt = np.asarray(X[:, :, ti], dtype=np.float64)
        decision = np.full(len(y), np.nan, dtype=float)
        for train, test in splits:
            if len(np.unique(y[train])) < 2:
                raise RuntimeError("Classification training fold contains one class.")
            model = make_classifier()
            model.fit(Xt[train], y[train])
            decision[test] = model.decision_function(Xt[test])
        if not np.isfinite(decision).all():
            raise RuntimeError("Classification produced non-finite OOF scores.")
        out[ti] = float(roc_auc_score(y, decision))
    return out


def decode_frequency_regression(X, y, splits, fi):
    return regression_timecourse(X[:, :, fi, :], y, splits)


def decode_frequency_classification(X, y, splits, fi):
    return classification_timecourse(X[:, :, fi, :], y, splits)


def process_subject(sub: str, n_jobs: int, overwrite: bool) -> None:
    out = score_path(sub)
    if out.exists() and not overwrite:
        print(f"[{sub}] cached score file exists -> skip", flush=True)
        return

    paths = tfr_paths(sub)
    power = np.load(paths["power"], mmap_mode="r")
    times = np.load(paths["times"]).astype(float)
    freqs = np.load(paths["freqs"]).astype(float)
    meta = pd.read_csv(paths["metadata"])

    if power.ndim != 4:
        raise RuntimeError(f"{sub}: expected 4D power; got {power.shape}")
    if len(meta) != power.shape[0]:
        raise RuntimeError(f"{sub}: metadata/power trial mismatch")
    if len(freqs) != power.shape[2] or len(times) != power.shape[3]:
        raise RuntimeError(f"{sub}: coordinate lengths do not match power")

    required = ["condition", "level_1_5", "blocks.thisRepN"]
    missing = [c for c in required if c not in meta.columns]
    if missing:
        raise KeyError(f"{sub}: metadata missing {missing}")

    cond = meta["condition"].astype(str).str.lower().str.strip().to_numpy()
    level = pd.to_numeric(meta["level_1_5"], errors="coerce").to_numpy(float)
    groups = pd.to_numeric(meta["blocks.thisRepN"], errors="coerce").to_numpy(float)

    if not np.isfinite(level).all() or not np.isfinite(groups).all():
        raise RuntimeError(f"{sub}: non-finite level or block")
    groups = groups.astype(int)

    n_attr = len(ATTRIBUTES)
    n_freq = len(freqs)
    n_time = len(times)
    n_band = len(BANDS)

    regression_freq_r = np.empty((n_attr, n_freq, n_time), dtype=np.float32)
    regression_band_r = np.empty((n_attr, n_band, n_time), dtype=np.float32)
    highlow_freq_auc = np.empty((n_attr, n_freq, n_time), dtype=np.float32)
    highlow_band_auc = np.empty((n_attr, n_band, n_time), dtype=np.float32)

    n_trials_reg = np.empty(n_attr, dtype=int)
    n_trials_highlow = np.empty(n_attr, dtype=int)
    n_blocks_reg = np.empty(n_attr, dtype=int)
    n_blocks_highlow = np.empty(n_attr, dtype=int)

    print(
        f"[{sub}] trials={len(meta)}, channels={power.shape[1]}, "
        f"freqs={n_freq}, times={n_time}",
        flush=True,
    )

    for ai, (attribute, code) in enumerate(ATTRIBUTES.items()):
        attr_mask = cond == code
        X = np.asarray(power[attr_mask], dtype=np.float32)
        y_reg = level[attr_mask].astype(float)
        g_reg = groups[attr_mask]

        if len(np.unique(y_reg)) < 5:
            raise RuntimeError(
                f"{sub}/{attribute}: not all levels 1..5 retained; got {np.unique(y_reg)}"
            )

        reg_splits = make_group_splits(g_reg)
        n_trials_reg[ai] = len(y_reg)
        n_blocks_reg[ai] = len(np.unique(g_reg))

        reg_rows = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(decode_frequency_regression)(X, y_reg, reg_splits, fi)
            for fi in range(n_freq)
        )
        regression_freq_r[ai] = np.stack(reg_rows, axis=0)

        hl_mask_local = np.isin(y_reg, [1.0, 2.0, 4.0, 5.0])
        X_hl = X[hl_mask_local]
        levels_hl = y_reg[hl_mask_local]
        y_hl = (levels_hl >= 4.0).astype(int)
        g_hl = g_reg[hl_mask_local]
        hl_splits = make_group_splits(g_hl)

        n_trials_highlow[ai] = len(y_hl)
        n_blocks_highlow[ai] = len(np.unique(g_hl))

        cls_rows = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(decode_frequency_classification)(X_hl, y_hl, hl_splits, fi)
            for fi in range(n_freq)
        )
        highlow_freq_auc[ai] = np.stack(cls_rows, axis=0)

        for bi, (band, (lo, hi)) in enumerate(BANDS.items()):
            fmask = (freqs >= lo) & (freqs <= hi)
            if not fmask.any():
                raise RuntimeError(f"{sub}: no frequencies for {band} {lo}-{hi}")
            X_band = X[:, :, fmask, :].mean(axis=2, dtype=np.float64).astype(np.float32)
            regression_band_r[ai, bi] = regression_timecourse(
                X_band, y_reg, reg_splits
            )
            highlow_band_auc[ai, bi] = classification_timecourse(
                X_band[hl_mask_local], y_hl, hl_splits
            )

    tmp = out.with_suffix(out.suffix + ".tmp")
    with open(tmp, "wb") as f:
        np.savez_compressed(
            f,
            participant=np.array(sub),
            times=times,
            freqs=freqs,
            attribute_names=np.array(list(ATTRIBUTES.keys())),
            band_names=np.array(list(BANDS.keys())),
            regression_freq_r=regression_freq_r,
            regression_band_r=regression_band_r,
            highlow_freq_auc=highlow_freq_auc,
            highlow_band_auc=highlow_band_auc,
            n_trials_reg=n_trials_reg,
            n_trials_highlow=n_trials_highlow,
            n_blocks_reg=n_blocks_reg,
            n_blocks_highlow=n_blocks_highlow,
        )
    os.replace(tmp, out)

    print(
        f"[{sub}] DONE | ordinal trials pain/money={n_trials_reg.tolist()} | "
        f"highlow trials={n_trials_highlow.tolist()}",
        flush=True,
    )


def holm_adjust(pvals: Sequence[float]) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    if m == 0:
        return np.array([], dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    adj_ranked = np.maximum.accumulate((m - np.arange(m)) * ranked)
    adj_ranked = np.clip(adj_ranked, 0.0, 1.0)
    out = np.empty(m, dtype=float)
    out[order] = adj_ranked
    return out


def fisher_z_array(r: np.ndarray) -> np.ndarray:
    return np.arctanh(np.clip(r, -0.999999, 0.999999))


def cluster_threshold(n_subjects: int) -> float:
    return float(stats.t.ppf(
        1.0 - CLUSTER_FORMING_ALPHA, df=n_subjects - 1
    ))


def run_tf_cluster(data, freqs, times, n_permutations, n_jobs):
    tmask = (times >= INFER_TMIN - 1e-12) & (times <= INFER_TMAX + 1e-12)
    times_inf = times[tmask]
    x = data[:, :, tmask]
    if not np.isfinite(x).all():
        raise RuntimeError("Non-finite data entering TF cluster test")

    n_sub, n_freq, n_time = x.shape
    adjacency = combine_adjacency(n_freq, n_time)
    flat = x.reshape(n_sub, n_freq * n_time)

    t_obs, clusters, pvals, _ = permutation_cluster_1samp_test(
        flat,
        threshold=cluster_threshold(n_sub),
        n_permutations=n_permutations,
        tail=1,
        adjacency=adjacency,
        seed=RANDOM_STATE,
        n_jobs=n_jobs,
        out_type="indices",
        verbose=False,
    )
    cluster_idx = [np.asarray(c[0], dtype=int) for c in clusters]
    return t_obs.reshape(n_freq, n_time), cluster_idx, np.asarray(pvals), times_inf


def run_time_cluster(data, times, n_permutations, n_jobs):
    tmask = (times >= INFER_TMIN - 1e-12) & (times <= INFER_TMAX + 1e-12)
    times_inf = times[tmask]
    x = data[:, tmask]
    if not np.isfinite(x).all():
        raise RuntimeError("Non-finite data entering temporal cluster test")

    n_sub = x.shape[0]
    t_obs, clusters, pvals, _ = permutation_cluster_1samp_test(
        x,
        threshold=cluster_threshold(n_sub),
        n_permutations=n_permutations,
        tail=1,
        seed=RANDOM_STATE,
        n_jobs=n_jobs,
        out_type="indices",
        verbose=False,
    )
    cluster_idx = [np.asarray(c[0], dtype=int) for c in clusters]
    return t_obs, cluster_idx, np.asarray(pvals), times_inf


def summarize_tf_cluster(flat_idx, tmap, freqs, times_inf):
    n_time = len(times_inf)
    fi, ti = np.unravel_index(flat_idx, (len(freqs), n_time))
    stat_flat = tmap.reshape(-1)
    vals = stat_flat[flat_idx]
    peak_local = int(np.argmax(vals))
    peak_flat = int(flat_idx[peak_local])
    pfi, pti = np.unravel_index(peak_flat, (len(freqs), n_time))
    return {
        "freq_min_hz": float(freqs[fi].min()),
        "freq_max_hz": float(freqs[fi].max()),
        "time_start_s": float(times_inf[ti].min()),
        "time_end_s": float(times_inf[ti].max()),
        "peak_freq_hz": float(freqs[pfi]),
        "peak_time_s": float(times_inf[pti]),
        "peak_t": float(vals[peak_local]),
        "cluster_mass": float(vals.sum()),
        "n_points": int(len(flat_idx)),
    }


def load_group_scores(subjects):
    arrays = []
    ref_times = ref_freqs = None
    attr_names = band_names = None
    counts = []

    for sub in subjects:
        p = score_path(sub)
        if not p.exists():
            raise FileNotFoundError(f"Missing participant decoding cache: {p}")
        z = np.load(p, allow_pickle=False)

        times = z["times"].astype(float)
        freqs = z["freqs"].astype(float)
        attrs = z["attribute_names"].astype(str).tolist()
        bands = z["band_names"].astype(str).tolist()

        if ref_times is None:
            ref_times, ref_freqs = times, freqs
            attr_names, band_names = attrs, bands
        else:
            if not np.allclose(times, ref_times):
                raise RuntimeError(f"{sub}: time grid differs")
            if not np.array_equal(freqs, ref_freqs):
                raise RuntimeError(f"{sub}: frequency grid differs")
            if attrs != attr_names or bands != band_names:
                raise RuntimeError(f"{sub}: label ordering differs")

        arrays.append({
            "reg_freq": z["regression_freq_r"],
            "reg_band": z["regression_band_r"],
            "hl_freq": z["highlow_freq_auc"],
            "hl_band": z["highlow_band_auc"],
        })

        for ai, attr in enumerate(attrs):
            counts.append({
                "participant": sub,
                "attribute": attr,
                "n_trials_ordinal": int(z["n_trials_reg"][ai]),
                "n_trials_highlow": int(z["n_trials_highlow"][ai]),
                "n_blocks_ordinal": int(z["n_blocks_reg"][ai]),
                "n_blocks_highlow": int(z["n_blocks_highlow"][ai]),
            })

    stack = {
        k: np.stack([x[k] for x in arrays], axis=0)
        for k in ["reg_freq", "reg_band", "hl_freq", "hl_band"]
    }
    return stack, ref_times, ref_freqs, attr_names, band_names, pd.DataFrame(counts)


def analyze_tf_family(
    family, raw, times, freqs, attr_names, n_permutations, n_jobs
):
    if family == "ordinal":
        test = fisher_z_array(raw)
        score_name = "r"
        chance = 0.0
    else:
        test = raw - 0.5
        score_name = "auc"
        chance = 0.5

    domain_rows = []
    cluster_rows = []
    tmask = (times >= INFER_TMIN - 1e-12) & (times <= INFER_TMAX + 1e-12)

    for ai, attr in enumerate(attr_names):
        tmap, clusters, pvals, times_inf = run_tf_cluster(
            test[:, ai], freqs, times, n_permutations, n_jobs
        )
        pmin = float(np.min(pvals)) if len(pvals) else 1.0
        domain_rows.append({
            "family": family,
            "attribute": attr,
            "omnibus_p_within_surface": pmin,
        })

        for ci, (idx, p) in enumerate(zip(clusters, pvals)):
            row = {
                "family": family,
                "attribute": attr,
                "cluster_id": ci,
                "cluster_p_within_surface": float(p),
            }
            row.update(summarize_tf_cluster(idx, tmap, freqs, times_inf))
            row["late_boundary_sensitive"] = bool(
                row["time_end_s"] > CONSERVATIVE_THETA_SAFE_TMAX
            )
            cluster_rows.append(row)

    domain_df = pd.DataFrame(domain_rows)
    domain_df["omnibus_p_holm_across_pain_money"] = holm_adjust(
        domain_df["omnibus_p_within_surface"].to_numpy()
    )
    domain_df["domain_significant_holm_0p05"] = (
        domain_df["omnibus_p_holm_across_pain_money"] < 0.05
    )

    cluster_df = pd.DataFrame(cluster_rows)
    if len(cluster_df):
        cluster_df = cluster_df.merge(
            domain_df[[
                "family", "attribute",
                "omnibus_p_holm_across_pain_money",
                "domain_significant_holm_0p05",
            ]],
            on=["family", "attribute"],
            how="left",
            validate="many_to_one",
        )
        cluster_df["cluster_significant_final"] = (
            (cluster_df["cluster_p_within_surface"] < 0.05)
            & cluster_df["domain_significant_holm_0p05"]
        )

    surface_rows = []
    for ai, attr in enumerate(attr_names):
        arr = raw[:, ai][:, :, tmask]
        mean = np.mean(arr, axis=0)
        sem = stats.sem(arr, axis=0)
        for fi, f in enumerate(freqs):
            for ti, t in enumerate(times[tmask]):
                surface_rows.append({
                    "family": family,
                    "attribute": attr,
                    "frequency_hz": float(f),
                    "time_s": float(t),
                    f"mean_{score_name}": float(mean[fi, ti]),
                    f"sem_{score_name}": float(sem[fi, ti]),
                    "chance": chance,
                })

    return domain_df, cluster_df, pd.DataFrame(surface_rows)


def analyze_band_family(
    family, raw, times, attr_names, band_names, n_permutations, n_jobs
):
    if family == "ordinal":
        test = fisher_z_array(raw)
        score_name = "r"
        chance = 0.0
    else:
        test = raw - 0.5
        score_name = "auc"
        chance = 0.5

    omnibus_rows = []
    cluster_rows = []

    for ai, attr in enumerate(attr_names):
        for bi, band in enumerate(band_names):
            tmap, clusters, pvals, times_inf = run_time_cluster(
                test[:, ai, bi], times, n_permutations, n_jobs
            )
            pmin = float(np.min(pvals)) if len(pvals) else 1.0
            omnibus_rows.append({
                "family": family,
                "attribute": attr,
                "band": band,
                "band_omnibus_p_within_time": pmin,
            })

            for ci, (idx, p) in enumerate(zip(clusters, pvals)):
                vals = tmap[idx]
                peak_local = int(np.argmax(vals))
                peak_idx = int(idx[peak_local])
                cluster_rows.append({
                    "family": family,
                    "attribute": attr,
                    "band": band,
                    "cluster_id": ci,
                    "cluster_p_within_band_time": float(p),
                    "time_start_s": float(times_inf[idx].min()),
                    "time_end_s": float(times_inf[idx].max()),
                    "peak_time_s": float(times_inf[peak_idx]),
                    "peak_t": float(vals[peak_local]),
                    "cluster_mass": float(vals.sum()),
                    "n_time_bins": int(len(idx)),
                    "late_boundary_sensitive": bool(
                        float(times_inf[idx].max()) > CONSERVATIVE_THETA_SAFE_TMAX
                    ),
                })

    omnibus_df = pd.DataFrame(omnibus_rows)
    omnibus_df["band_omnibus_p_holm_within_attribute"] = np.nan
    for attr in attr_names:
        idx = omnibus_df["attribute"] == attr
        omnibus_df.loc[idx, "band_omnibus_p_holm_within_attribute"] = holm_adjust(
            omnibus_df.loc[idx, "band_omnibus_p_within_time"].to_numpy()
        )
    omnibus_df["band_significant_holm_0p05"] = (
        omnibus_df["band_omnibus_p_holm_within_attribute"] < 0.05
    )

    cluster_df = pd.DataFrame(cluster_rows)
    if len(cluster_df):
        cluster_df = cluster_df.merge(
            omnibus_df[[
                "family", "attribute", "band",
                "band_omnibus_p_holm_within_attribute",
                "band_significant_holm_0p05",
            ]],
            on=["family", "attribute", "band"],
            how="left",
            validate="many_to_one",
        )
        cluster_df["cluster_significant_final"] = (
            (cluster_df["cluster_p_within_band_time"] < 0.05)
            & cluster_df["band_significant_holm_0p05"]
        )

    curve_rows = []
    for ai, attr in enumerate(attr_names):
        for bi, band in enumerate(band_names):
            arr = raw[:, ai, bi]
            mean = np.mean(arr, axis=0)
            sem = stats.sem(arr, axis=0)
            for ti, t in enumerate(times):
                curve_rows.append({
                    "family": family,
                    "attribute": attr,
                    "band": band,
                    "time_s": float(t),
                    f"mean_{score_name}": float(mean[ti]),
                    f"sem_{score_name}": float(sem[ti]),
                    "chance": chance,
                })

    return omnibus_df, cluster_df, pd.DataFrame(curve_rows)


def plot_tf_heatmaps(
    family, raw, times, freqs, attr_names, domain_df, cluster_df
):
    tmask = (times >= INFER_TMIN - 1e-12) & (times <= INFER_TMAX + 1e-12)
    tplot = times[tmask]
    for ai, attr in enumerate(attr_names):
        mean = np.mean(raw[:, ai][:, :, tmask], axis=0)
        label = (
            "Mean cross-validated Pearson r"
            if family == "ordinal"
            else "Mean cross-validated ROC-AUC"
        )
        metric = "ordinal 1–5" if family == "ordinal" else "high vs low"

        fig, ax = plt.subplots(figsize=(9.0, 5.3))
        im = ax.pcolormesh(tplot, freqs, mean, shading="auto")
        ax.axvline(0.0, linewidth=1.0)
        ax.set_xlabel("Time from cue (s)")
        ax.set_ylabel("Frequency (Hz)")
        drow = domain_df[
            (domain_df["family"] == family)
            & (domain_df["attribute"] == attr)
        ].iloc[0]
        ax.set_title(
            f"Passive {attr} — {metric} spectral decoding\n"
            f"domain Holm p={drow['omnibus_p_holm_across_pain_money']:.4g}"
        )
        fig.colorbar(im, ax=ax, label=label)

        if len(cluster_df):
            cdf = cluster_df[
                (cluster_df["family"] == family)
                & (cluster_df["attribute"] == attr)
                & (cluster_df["cluster_significant_final"] == True)
            ]
            for _, row in cdf.iterrows():
                ax.axvspan(row["time_start_s"], row["time_end_s"], alpha=0.08)

        fig.tight_layout()
        fig.savefig(FIG_DIR / f"{family}_{attr}_freq_time.png", dpi=220)
        plt.close(fig)


def plot_band_curves(
    family, raw, times, attr_names, band_names, omnibus_df, cluster_df
):
    for ai, attr in enumerate(attr_names):
        for bi, band in enumerate(band_names):
            arr = raw[:, ai, bi]
            mean = np.mean(arr, axis=0)
            sem = stats.sem(arr, axis=0)

            if family == "ordinal":
                chance, ylabel, metric = 0.0, "Cross-validated Pearson r", "ordinal 1–5"
            else:
                chance, ylabel, metric = 0.5, "Cross-validated ROC-AUC", "high vs low"

            fig, ax = plt.subplots(figsize=(8.5, 4.8))
            ax.plot(times, mean, linewidth=2.0)
            ax.fill_between(times, mean - sem, mean + sem, alpha=0.2)
            ax.axhline(chance, linewidth=1.0)
            ax.axvline(0.0, linewidth=1.0)
            ax.axvspan(-0.7, -0.5, alpha=0.06)

            orow = omnibus_df[
                (omnibus_df["family"] == family)
                & (omnibus_df["attribute"] == attr)
                & (omnibus_df["band"] == band)
            ].iloc[0]

            ax.set_title(
                f"Passive {attr} — {band} — {metric}\n"
                f"Holm across 4 bands p="
                f"{orow['band_omnibus_p_holm_within_attribute']:.4g}"
            )
            ax.set_xlabel("Time from cue (s)")
            ax.set_ylabel(ylabel)

            if len(cluster_df):
                sig = cluster_df[
                    (cluster_df["family"] == family)
                    & (cluster_df["attribute"] == attr)
                    & (cluster_df["band"] == band)
                    & (cluster_df["cluster_significant_final"] == True)
                ]
                if len(sig):
                    ymin, ymax = ax.get_ylim()
                    ymark = ymin + 0.04 * (ymax - ymin)
                    for _, row in sig.iterrows():
                        ax.plot(
                            [row["time_start_s"], row["time_end_s"]],
                            [ymark, ymark],
                            linewidth=5.0,
                            solid_capstyle="butt",
                        )

            fig.tight_layout()
            fig.savefig(
                FIG_DIR / f"{family}_{attr}_{band}_timecourse.png",
                dpi=220,
            )
            plt.close(fig)


def run_group(subjects, args):
    scores, times, freqs, attr_names, band_names, counts_df = load_group_scores(subjects)

    counts_df.to_csv(OUT_ROOT / "participant_trial_counts.csv", index=False)

    np.savez_compressed(
        OUT_ROOT / "group_score_arrays.npz",
        subjects=np.array(subjects),
        times=times,
        freqs=freqs,
        attribute_names=np.array(attr_names),
        band_names=np.array(band_names),
        regression_freq_r=scores["reg_freq"],
        regression_band_r=scores["reg_band"],
        highlow_freq_auc=scores["hl_freq"],
        highlow_band_auc=scores["hl_band"],
    )

    all_domain = []
    all_tf_clusters = []
    all_tf_long = []
    all_band_omnibus = []
    all_band_clusters = []
    all_band_curves = []

    for family, freq_key, band_key in [
        ("ordinal", "reg_freq", "reg_band"),
        ("highlow", "hl_freq", "hl_band"),
    ]:
        print(f"\n===== {family.upper()} FREQUENCY x TIME =====", flush=True)
        domain_df, tf_cluster_df, tf_long_df = analyze_tf_family(
            family,
            scores[freq_key],
            times,
            freqs,
            attr_names,
            args.n_permutations,
            args.n_jobs,
        )
        all_domain.append(domain_df)
        all_tf_clusters.append(tf_cluster_df)
        all_tf_long.append(tf_long_df)
        plot_tf_heatmaps(
            family, scores[freq_key], times, freqs,
            attr_names, domain_df, tf_cluster_df
        )

        print(domain_df.to_string(index=False), flush=True)
        if len(tf_cluster_df):
            sig = tf_cluster_df[tf_cluster_df["cluster_significant_final"] == True]
            print("\nSignificant TF clusters:", flush=True)
            if len(sig):
                cols = [
                    "attribute", "freq_min_hz", "freq_max_hz",
                    "time_start_s", "time_end_s",
                    "peak_freq_hz", "peak_time_s",
                    "peak_t", "cluster_p_within_surface",
                    "omnibus_p_holm_across_pain_money",
                ]
                print(sig[cols].to_string(index=False), flush=True)
            else:
                print("None", flush=True)

        print(f"\n===== {family.upper()} CANONICAL BANDS =====", flush=True)
        band_omnibus_df, band_cluster_df, band_curve_df = analyze_band_family(
            family,
            scores[band_key],
            times,
            attr_names,
            band_names,
            args.n_permutations,
            args.n_jobs,
        )
        all_band_omnibus.append(band_omnibus_df)
        all_band_clusters.append(band_cluster_df)
        all_band_curves.append(band_curve_df)
        plot_band_curves(
            family, scores[band_key], times, attr_names, band_names,
            band_omnibus_df, band_cluster_df
        )

        print(band_omnibus_df.to_string(index=False), flush=True)
        if len(band_cluster_df):
            sig = band_cluster_df[band_cluster_df["cluster_significant_final"] == True]
            print("\nSignificant band clusters:", flush=True)
            if len(sig):
                cols = [
                    "attribute", "band", "time_start_s", "time_end_s",
                    "peak_time_s", "peak_t",
                    "cluster_p_within_band_time",
                    "band_omnibus_p_holm_within_attribute",
                ]
                print(sig[cols].to_string(index=False), flush=True)
            else:
                print("None", flush=True)

    pd.concat(all_domain, ignore_index=True).to_csv(
        OUT_ROOT / "freq_time_domain_omnibus_holm.csv", index=False
    )
    nonempty = [x for x in all_tf_clusters if len(x)]
    (pd.concat(nonempty, ignore_index=True) if nonempty else pd.DataFrame()).to_csv(
        OUT_ROOT / "freq_time_clusters.csv", index=False
    )
    pd.concat(all_tf_long, ignore_index=True).to_csv(
        OUT_ROOT / "freq_time_group_surface.csv", index=False
    )

    pd.concat(all_band_omnibus, ignore_index=True).to_csv(
        OUT_ROOT / "band_omnibus_holm.csv", index=False
    )
    nonempty = [x for x in all_band_clusters if len(x)]
    (pd.concat(nonempty, ignore_index=True) if nonempty else pd.DataFrame()).to_csv(
        OUT_ROOT / "band_clusters.csv", index=False
    )
    pd.concat(all_band_curves, ignore_index=True).to_csv(
        OUT_ROOT / "band_group_timecourses.csv", index=False
    )

    params = {
        "analysis": "PainReward passive spectral MVPA v2",
        "n_subjects": len(subjects),
        "subjects": list(subjects),
        "participant_source": str(PARTICIPANTS_TSV),
        "attributes": list(ATTRIBUTES.keys()),
        "frequencies_hz": freqs.tolist(),
        "bands_hz": {k: list(v) for k, v in BANDS.items()},
        "saved_score_time_s": [float(times.min()), float(times.max())],
        "formal_inference_time_s": [INFER_TMIN, INFER_TMAX],
        "pre_cue_policy": "saved for QC, excluded from formal cluster inference",
        "ordinal_primary": {
            "target": "level 1..5",
            "model": "StandardScaler -> Ridge(alpha=1)",
            "cv": "leave-one-passive-block-out GroupKFold",
            "score": "Pearson r from pooled OOF predictions",
            "group_transform": "Fisher z(r)",
        },
        "highlow_secondary": {
            "low": [1, 2],
            "excluded": [3],
            "high": [4, 5],
            "model": "StandardScaler -> RidgeClassifier(alpha=1)",
            "cv": "leave-one-passive-block-out GroupKFold",
            "score": "ROC-AUC from pooled OOF decision scores",
            "group_null": "AUC - 0.5 = 0",
        },
        "frequency_resolved_inference": (
            "separate one-sided positive 2D frequency x time cluster permutation "
            "for pain and money; Holm gate across the two domain omnibus p-values"
        ),
        "band_inference": (
            "one-sided positive temporal cluster permutation inside each band; "
            "Holm across theta/alpha/beta/low_gamma separately within pain and money"
        ),
        "cluster_forming_alpha_one_sided": CLUSTER_FORMING_ALPHA,
        "n_permutations": args.n_permutations,
        "random_state": RANDOM_STATE,
        "old_8map_global_max_analysis": "retained separately as robustness analysis",
        "late_time_warning": (
            "cue duration ~1.5 s; late nominal TFR centres have Morlet support "
            "extending around the centre. Clusters ending after "
            f"{CONSERVATIVE_THETA_SAFE_TMAX:.2f}s are flagged boundary-sensitive."
        ),
    }
    with open(OUT_ROOT / "params.json", "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)

    print("\n" + "=" * 78, flush=True)
    print("PASSIVE SPECTRAL MVPA V2 COMPLETE", flush=True)
    print(f"Subjects: {len(subjects)}", flush=True)
    print(f"Outputs: {OUT_ROOT}", flush=True)
    print("=" * 78, flush=True)


def main():
    args = parse_args()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    SCORE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    subjects = load_subject_list(args.subjects)

    print("=" * 78, flush=True)
    print("PAINREWARD PASSIVE SPECTRAL MVPA V2", flush=True)
    print(f"Subjects: {len(subjects)}", flush=True)
    print("PRIMARY: level 1..5 Ridge regression", flush=True)
    print("SECONDARY: low(1,2) vs high(4,5), level 3 excluded", flush=True)
    print(f"CPUs: {args.n_jobs}", flush=True)
    print(f"Permutations: {args.n_permutations}", flush=True)
    print("=" * 78, flush=True)

    if not args.group_only:
        for i, sub in enumerate(subjects, 1):
            print(f"\n[{i:02d}/{len(subjects)}] {sub}", flush=True)
            process_subject(sub, args.n_jobs, args.overwrite_scores)

    run_group(subjects, args)


if __name__ == "__main__":
    main()
