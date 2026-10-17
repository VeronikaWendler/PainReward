#!/usr/bin/env python3
"""
05c_eeg_tfr_decode_passive_within.py

Primary within-passive TFR magnitude decoding for PainReward.

Scientific question
-------------------
At what times, and in which pre-defined frequency bands, does the spatial
pattern of baseline-normalised EEG power carry information about the level
(1..5) of:
    1) pain cues, and
    2) money cues
during the passive phase?

This script deliberately does ONLY the first-stage within-domain analysis.
It does NOT perform pain<->money cross-decoding, passive->decision transfer,
choice analyses, or HSSM/DDM analyses.

Input per participant
---------------------
derivatives/<sub>/eeg/tfr/
    <sub>_passive_cue_power.npy
    <sub>_passive_cue_times.npy
    <sub>_passive_cue_freqs.npy
    <sub>_passive_cue_metadata.csv
    <sub>_passive_cue_channels.json

Input power has shape:
    trial x channel x frequency x time

Pre-defined bands
-----------------
theta       4-7 Hz
alpha       8-13 Hz
beta        14-30 Hz
low_gamma   31-45 Hz

Decoding
--------
For each participant, attribute, band and time bin:
    - average already-normalised log power across frequencies in the band
    - features = all EEG channels at that time bin
    - target = cue level 1..5
    - StandardScaler -> Ridge(alpha=1)
    - leave-one-passive-block-out GroupKFold
    - concatenate out-of-fold predictions across all held-out blocks
    - score = Pearson r(predicted level, true level)
    - Fisher-z transform participant r for group inference

Group inference
---------------
The 8 pre-defined maps (2 attributes x 4 bands) are tested jointly:
    - one-sided test for above-chance decoding (Fisher-z(r) > 0)
    - cluster-forming threshold p=.05
    - sign-flip permutation across participants
    - maximum cluster mass is taken across ALL 8 maps on every permutation
      => family-wise error correction jointly across time and all 8 primary maps

Subject inclusion
-----------------
By default, use all participant IDs listed in:
    participants.tsv

The analysis then fails fast if any selected participant is missing one of the
required passive-TFR files. This keeps the passive-only decoding independent
of legacy HDDM/decision-pipeline exclusions. Use --subjects only for an
explicit diagnostic or sensitivity subset.

Outputs
-------
derivatives/statistics/tfr_decoding_passive_within/
    participant_scores_long.csv
    participant_summary.csv
    group_timecourses.csv
    cluster_results_fwer.csv
    params.json
    figures/<attribute>_<band>_decoding.png

Fail-fast policy: missing/malformed input aborts the run.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 23
BASE_PATH = Path(os.getenv("basepath", Path(__file__).parent.parent.parent))
OUT_DIR = BASE_PATH / "derivatives" / "statistics" / "tfr_decoding_passive_within"
FIG_DIR = OUT_DIR / "figures"
PARTICIPANTS_TSV = BASE_PATH / "participants.tsv"

RIDGE_ALPHA = 1.0
CLUSTER_FORMING_ALPHA = 0.05
DEFAULT_N_PERMUTATIONS = 10000

BANDS: Dict[str, Tuple[float, float]] = {
    "theta": (4.0, 7.0),
    "alpha": (8.0, 13.0),
    "beta": (14.0, 30.0),
    "low_gamma": (31.0, 45.0),
}
ATTRIBUTES = {
    "pain": "p",
    "money": "m",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Within-passive TFR magnitude decoding.")
    p.add_argument(
        "--subjects",
        nargs="*",
        default=None,
        help="Optional explicit participant ids. Default: established post-exclusion list.",
    )
    p.add_argument(
        "--n-permutations",
        type=int,
        default=DEFAULT_N_PERMUTATIONS,
        help="Joint max-cluster sign-flip permutations (default: 10000).",
    )
    p.add_argument(
        "--ridge-alpha",
        type=float,
        default=RIDGE_ALPHA,
        help="Fixed Ridge regularisation alpha (default: 1.0).",
    )
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


def load_subject_list(explicit: List[str] | None) -> List[str]:
    if explicit:
        subs = sorted(dict.fromkeys(explicit))
    else:
        if not PARTICIPANTS_TSV.exists():
            raise FileNotFoundError(
                f"participants.tsv missing: {PARTICIPANTS_TSV}"
            )

        df = pd.read_csv(PARTICIPANTS_TSV, sep="\t")
        if "participant_id" not in df.columns:
            raise KeyError(
                f"{PARTICIPANTS_TSV} lacks required 'participant_id' column. "
                f"Columns: {list(df.columns)}"
            )

        subs = sorted(
            df["participant_id"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

    if not subs:
        raise RuntimeError("No participants selected for decoding.")

    missing = []
    for sub in subs:
        paths = tfr_paths(sub)
        for key, path in paths.items():
            if not path.exists():
                missing.append(f"{sub} [{key}]: {path}")

    if missing:
        raise FileNotFoundError(
            "Missing TFR input file(s):\n" + "\n".join(missing[:100])
        )

    return subs

def make_model(alpha: float) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=alpha)),
    ])


def safe_pearson_r(y: np.ndarray, pred: np.ndarray) -> float:
    if len(y) != len(pred):
        raise ValueError("y/pred length mismatch")
    if np.std(y) == 0 or np.std(pred) == 0:
        return 0.0
    r = float(np.corrcoef(y, pred)[0, 1])
    return r if np.isfinite(r) else 0.0


def fisher_z(r: float) -> float:
    return float(np.arctanh(np.clip(r, -0.999999, 0.999999)))


def oof_r_timecourse(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """
    X: trials x channels x time
    y: trials
    groups: passive block id

    Returns one Pearson-r value per time bin, using pooled out-of-fold
    predictions across leave-one-block-out folds.
    """
    if X.ndim != 3:
        raise ValueError(f"Expected X=(trial, channel, time); got {X.shape}")
    if not (len(X) == len(y) == len(groups)):
        raise ValueError("X/y/groups trial counts differ.")
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise RuntimeError("Non-finite decoding input.")
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        raise RuntimeError(f"Need >=2 passive blocks, got {unique_groups}")
    if np.std(y) == 0:
        raise RuntimeError("Target level has zero variance.")

    cv = GroupKFold(n_splits=len(unique_groups))
    n_times = X.shape[2]
    scores = np.empty(n_times, dtype=float)

    for ti in range(n_times):
        Xt = X[:, :, ti]
        pred = np.full(len(y), np.nan, dtype=float)

        for train_idx, test_idx in cv.split(Xt, y, groups=groups):
            if np.std(y[train_idx]) == 0:
                raise RuntimeError("A CV training fold has zero target variance.")
            model = make_model(alpha)
            model.fit(Xt[train_idx], y[train_idx])
            pred[test_idx] = model.predict(Xt[test_idx])

        if not np.isfinite(pred).all():
            raise RuntimeError("Non-finite/outstanding OOF predictions.")
        scores[ti] = safe_pearson_r(y, pred)

    return scores


def contiguous_clusters(tvals: np.ndarray, threshold: float) -> List[np.ndarray]:
    above = np.asarray(tvals > threshold, dtype=bool)
    idx = np.flatnonzero(above)
    if len(idx) == 0:
        return []
    split_points = np.where(np.diff(idx) > 1)[0] + 1
    return [c for c in np.split(idx, split_points) if len(c)]


def one_sample_t_positive(data: np.ndarray) -> np.ndarray:
    """
    data: subject x map x time, tested against 0.
    """
    n = data.shape[0]
    mean = data.mean(axis=0)
    sd = data.std(axis=0, ddof=1)
    se = sd / np.sqrt(n)
    tvals = np.divide(mean, se, out=np.zeros_like(mean), where=se > 0)
    return tvals


def joint_max_cluster_test(
    z_data: np.ndarray,
    map_names: List[Tuple[str, str]],
    times: np.ndarray,
    n_permutations: int,
    seed: int = RANDOM_STATE,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Joint FWER correction across all 8 maps and time.

    z_data shape: subject x map x time
    """
    if z_data.ndim != 3:
        raise ValueError(f"Expected z_data=(subject,map,time), got {z_data.shape}")
    n_sub, n_maps, n_times = z_data.shape
    if n_maps != len(map_names):
        raise ValueError("Map-name count mismatch.")
    if n_times != len(times):
        raise ValueError("Time-coordinate mismatch.")
    if n_sub < 3:
        raise RuntimeError("Need >=3 participants for group inference.")

    threshold = float(stats.t.ppf(1.0 - CLUSTER_FORMING_ALPHA, df=n_sub - 1))
    obs_t = one_sample_t_positive(z_data)

    observed = []
    for mi, (attribute, band) in enumerate(map_names):
        for ci, cluster in enumerate(contiguous_clusters(obs_t[mi], threshold)):
            mass = float(obs_t[mi, cluster].sum())
            peak_local = int(np.argmax(obs_t[mi, cluster]))
            peak_idx = int(cluster[peak_local])
            observed.append({
                "map_index": mi,
                "attribute": attribute,
                "band": band,
                "cluster_id": ci,
                "start_index": int(cluster[0]),
                "end_index": int(cluster[-1]),
                "time_start_s": float(times[cluster[0]]),
                "time_end_s": float(times[cluster[-1]]),
                "peak_time_s": float(times[peak_idx]),
                "peak_t": float(obs_t[mi, peak_idx]),
                "cluster_mass": mass,
                "n_time_bins": int(len(cluster)),
            })

    rng = np.random.default_rng(seed)
    null_max = np.zeros(n_permutations, dtype=float)

    for pi in range(n_permutations):
        signs = rng.choice(np.array([-1.0, 1.0]), size=n_sub)
        perm_t = one_sample_t_positive(z_data * signs[:, None, None])
        max_mass = 0.0
        for mi in range(n_maps):
            for cluster in contiguous_clusters(perm_t[mi], threshold):
                mass = float(perm_t[mi, cluster].sum())
                if mass > max_mass:
                    max_mass = mass
        null_max[pi] = max_mass

    rows = []
    for row in observed:
        p_fwer = (1.0 + float(np.sum(null_max >= row["cluster_mass"]))) / (n_permutations + 1.0)
        row = dict(row)
        row["p_fwer_8maps_time"] = p_fwer
        row["significant_fwer_0p05"] = bool(p_fwer < 0.05)
        row["cluster_forming_t"] = threshold
        row["n_subjects"] = n_sub
        row["n_permutations"] = n_permutations
        rows.append(row)

    cols = [
        "map_index", "attribute", "band", "cluster_id",
        "time_start_s", "time_end_s", "peak_time_s", "peak_t",
        "cluster_mass", "n_time_bins", "p_fwer_8maps_time",
        "significant_fwer_0p05", "cluster_forming_t",
        "n_subjects", "n_permutations", "start_index", "end_index",
    ]
    return obs_t, pd.DataFrame(rows, columns=cols)


def plot_curve(
    attribute: str,
    band: str,
    times: np.ndarray,
    raw_r: np.ndarray,
    cluster_df: pd.DataFrame,
    outpath: Path,
) -> None:
    mean_r = np.mean(raw_r, axis=0)
    sem_r = stats.sem(raw_r, axis=0, ddof=1)

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(times, mean_r, linewidth=2.0)
    ax.fill_between(times, mean_r - sem_r, mean_r + sem_r, alpha=0.2)
    ax.axhline(0.0, linewidth=1.0)
    ax.axvline(0.0, linewidth=1.0)
    ax.axvspan(-0.7, -0.5, alpha=0.08)

    sig = cluster_df[
        (cluster_df["attribute"] == attribute)
        & (cluster_df["band"] == band)
        & (cluster_df["significant_fwer_0p05"] == True)
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

    ax.set_xlabel("Time from cue (s)")
    ax.set_ylabel("Cross-validated Pearson r")
    ax.set_title(f"Passive {attribute} magnitude decoding — {band}")
    ax.text(
        0.01, 0.98,
        "Shading: ±SEM\nBottom bars: joint FWER p<.05",
        transform=ax.transAxes,
        ha="left", va="top", fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    subjects = load_subject_list(args.subjects)
    print("=" * 76, flush=True)
    print("PASSIVE TFR WITHIN-DOMAIN MAGNITUDE DECODING", flush=True)
    print(f"Subjects: {len(subjects)}", flush=True)
    print(f"Subject source: {'--subjects' if args.subjects else PARTICIPANTS_TSV}", flush=True)
    print(f"Ridge alpha: {args.ridge_alpha}", flush=True)
    print(f"Joint permutations: {args.n_permutations}", flush=True)
    print("=" * 76, flush=True)

    reference_times = None
    reference_freqs = None
    reference_channels = None

    map_names = [(attribute, band) for attribute in ATTRIBUTES for band in BANDS]
    raw_scores: Dict[Tuple[str, str], List[np.ndarray]] = {k: [] for k in map_names}
    long_rows: List[pd.DataFrame] = []
    summary_rows = []

    for si, sub in enumerate(subjects, start=1):
        paths = tfr_paths(sub)
        power = np.load(paths["power"], mmap_mode="r")
        times = np.load(paths["times"]).astype(float)
        freqs = np.load(paths["freqs"]).astype(float)
        meta = pd.read_csv(paths["metadata"])
        with open(paths["channels"], "r", encoding="utf-8") as f:
            channels = json.load(f)

        if power.ndim != 4:
            raise RuntimeError(f"{sub}: expected 4D power, got {power.shape}")
        if len(meta) != power.shape[0]:
            raise RuntimeError(f"{sub}: metadata/power trial mismatch.")
        if len(channels) != power.shape[1] or len(freqs) != power.shape[2] or len(times) != power.shape[3]:
            raise RuntimeError(f"{sub}: coordinate lengths mismatch power shape {power.shape}")

        required_md = ["condition", "level_1_5", "blocks.thisRepN"]
        missing_md = [c for c in required_md if c not in meta.columns]
        if missing_md:
            raise KeyError(f"{sub}: metadata missing {missing_md}")

        if reference_times is None:
            reference_times = times
            reference_freqs = freqs
            reference_channels = channels
        else:
            if channels != reference_channels:
                raise RuntimeError(f"{sub}: channel order differs from first participant.")
            if not np.array_equal(freqs, reference_freqs):
                raise RuntimeError(f"{sub}: frequency grid differs from first participant.")
            if not np.allclose(times, reference_times, atol=1e-12, rtol=0):
                raise RuntimeError(f"{sub}: time grid differs from first participant.")

        cond = meta["condition"].astype(str).str.lower().str.strip().to_numpy()
        level = pd.to_numeric(meta["level_1_5"], errors="coerce").to_numpy(dtype=float)
        blocks = pd.to_numeric(meta["blocks.thisRepN"], errors="coerce").to_numpy(dtype=float)

        if np.any(~np.isfinite(level)) or np.any(~np.isfinite(blocks)):
            raise RuntimeError(f"{sub}: non-finite level/block metadata.")
        blocks = blocks.astype(int)

        print(
            f"[{si:02d}/{len(subjects)}] {sub}: trials={len(meta)}, "
            f"blocks={sorted(np.unique(blocks).tolist())}",
            flush=True,
        )

        # Average frequency only AFTER frequency-wise TFR normalisation.
        for band, (f_lo, f_hi) in BANDS.items():
            fmask = (freqs >= f_lo) & (freqs <= f_hi)
            if not fmask.any():
                raise RuntimeError(f"{sub}: no frequency bins for {band} {f_lo}-{f_hi} Hz")

            # trials x channels x time
            band_power = np.asarray(power[:, :, fmask, :], dtype=np.float32).mean(
                axis=2, dtype=np.float64
            )

            for attribute, code in ATTRIBUTES.items():
                mask = cond == code
                X = band_power[mask]
                y = level[mask]
                g = blocks[mask]

                if len(X) < 20:
                    raise RuntimeError(f"{sub}/{attribute}/{band}: too few trials ({len(X)})")
                if len(np.unique(g)) < 2:
                    raise RuntimeError(f"{sub}/{attribute}/{band}: fewer than two passive blocks")
                if len(np.unique(y)) < 5:
                    raise RuntimeError(
                        f"{sub}/{attribute}/{band}: not all five magnitude levels represented: "
                        f"{sorted(np.unique(y).tolist())}"
                    )

                r = oof_r_timecourse(X, y, g, alpha=args.ridge_alpha)
                raw_scores[(attribute, band)].append(r)
                z = np.array([fisher_z(v) for v in r], dtype=float)

                long_rows.append(pd.DataFrame({
                    "participant": sub,
                    "attribute": attribute,
                    "band": band,
                    "time_s": times,
                    "r": r,
                    "fisher_z_r": z,
                    "n_trials": len(X),
                    "n_blocks": len(np.unique(g)),
                }))

                summary_rows.append({
                    "participant": sub,
                    "attribute": attribute,
                    "band": band,
                    "n_trials": len(X),
                    "n_blocks": len(np.unique(g)),
                    "mean_r_across_time": float(np.mean(r)),
                    "max_r": float(np.max(r)),
                    "max_r_time_s": float(times[np.argmax(r)]),
                })

            del band_power

    participant_long = pd.concat(long_rows, ignore_index=True)
    participant_long.to_csv(OUT_DIR / "participant_scores_long.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(OUT_DIR / "participant_summary.csv", index=False)

    raw_tensor = np.stack(
        [np.stack(raw_scores[name], axis=0) for name in map_names],
        axis=1,
    )  # subject x map x time
    z_tensor = np.arctanh(np.clip(raw_tensor, -0.999999, 0.999999))

    obs_t, cluster_df = joint_max_cluster_test(
        z_tensor,
        map_names=map_names,
        times=reference_times,
        n_permutations=args.n_permutations,
    )
    cluster_df.to_csv(OUT_DIR / "cluster_results_fwer.csv", index=False)

    group_rows = []
    for mi, (attribute, band) in enumerate(map_names):
        arr = raw_tensor[:, mi, :]
        zarr = z_tensor[:, mi, :]
        mean_r = arr.mean(axis=0)
        sem_r = stats.sem(arr, axis=0, ddof=1)
        mean_z = zarr.mean(axis=0)

        for ti, t in enumerate(reference_times):
            group_rows.append({
                "attribute": attribute,
                "band": band,
                "time_s": float(t),
                "mean_r": float(mean_r[ti]),
                "sem_r": float(sem_r[ti]),
                "mean_fisher_z_r": float(mean_z[ti]),
                "t_fisher_z": float(obs_t[mi, ti]),
                "n_subjects": len(subjects),
            })

        plot_curve(
            attribute,
            band,
            reference_times,
            arr,
            cluster_df,
            FIG_DIR / f"{attribute}_{band}_decoding.png",
        )

    pd.DataFrame(group_rows).to_csv(OUT_DIR / "group_timecourses.csv", index=False)

    params = {
        "analysis": "within-passive TFR magnitude decoding",
        "basepath": str(BASE_PATH),
        "subject_source": str(PARTICIPANTS_TSV) if not args.subjects else "explicit --subjects",
        "subjects": subjects,
        "n_subjects": len(subjects),
        "attributes": ATTRIBUTES,
        "bands_hz": {k: list(v) for k, v in BANDS.items()},
        "feature_definition": "mean normalised log power across frequencies within band; all EEG channels",
        "time_bins": "use existing 20-ms TFR bins without additional temporal smoothing",
        "model": "StandardScaler -> Ridge",
        "ridge_alpha": args.ridge_alpha,
        "cv": "leave-one-passive-block-out GroupKFold",
        "score": "Pearson r on pooled out-of-fold predictions",
        "group_transform": "Fisher z(r)",
        "group_test": (
            "one-sided positive cluster-mass sign-flip permutation; max cluster mass "
            "jointly across 2 attributes x 4 bands x all time bins"
        ),
        "cluster_forming_alpha_one_sided": CLUSTER_FORMING_ALPHA,
        "n_permutations": args.n_permutations,
        "random_state": RANDOM_STATE,
        "multiple_comparison_control": "joint FWER across all 8 maps and time",
        "interpretation_warning": (
            "Passive cue level covaries with visual fill magnitude. Within-domain decoding alone "
            "does not establish valuation coding; cross-domain direction and scalp patterns are "
            "needed for interpretation."
        ),
    }
    with open(OUT_DIR / "params.json", "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)

    print("\n" + "=" * 76, flush=True)
    print("PASSIVE TFR DECODING COMPLETE", flush=True)
    print(f"Participants: {len(subjects)}", flush=True)
    print(f"Outputs: {OUT_DIR}", flush=True)

    if cluster_df.empty:
        print("No supra-threshold temporal clusters in any of the 8 maps.", flush=True)
    else:
        sig = cluster_df[cluster_df["significant_fwer_0p05"] == True].copy()
        print(f"Observed clusters: {len(cluster_df)}", flush=True)
        print(f"Joint-FWER significant clusters: {len(sig)}", flush=True)
        if len(sig):
            display_cols = [
                "attribute", "band", "time_start_s", "time_end_s",
                "peak_time_s", "peak_t", "p_fwer_8maps_time"
            ]
            print(sig[display_cols].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
