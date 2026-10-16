"""
Group-level QC for the prepared passive cue TFR files.

This script performs NO decoding and NO inferential tests. It verifies that
all participant outputs have consistent dimensions/channels/frequencies,
checks baseline centering, summarizes trial counts and low-gamma tails, and
creates simple grand-average TFR heatmaps for selected channels.

Input
-----
derivatives/<sub>/eeg/tfr/<sub>_passive_cue_*

Outputs
-------
derivatives/tfr/qc/
    passive_tfr_qc_summary.csv
    passive_tfr_condition_counts.csv
    passive_tfr_grandmean.npy
    passive_tfr_grandmean_<channel>.png
    passive_tfr_qc_params.json
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_PATH = Path(os.getenv("basepath", Path(__file__).parent.parent.parent))
QC_DIR = BASE_PATH / "derivatives" / "tfr" / "qc"

BASELINE = (-0.7, -0.5)
POSTCUE_GAMMA_WINDOW = (0.0, 1.0)
LOW_GAMMA = (31.0, 45.0)
PLOT_CHANNELS = ["FCz", "Cz", "CPz", "Pz", "Oz"]


def participants() -> list[str]:
    subs = sorted(p.name for p in BASE_PATH.iterdir() if p.is_dir() and p.name.startswith("sub-"))
    if not subs:
        raise RuntimeError(f"No sub-* directories found under {BASE_PATH}")
    return subs


def subject_paths(sub: str) -> dict[str, Path]:
    d = BASE_PATH / "derivatives" / sub / "eeg" / "tfr"
    stem = f"{sub}_passive_cue"
    return {
        "power": d / f"{stem}_power.npy",
        "times": d / f"{stem}_times.npy",
        "freqs": d / f"{stem}_freqs.npy",
        "metadata": d / f"{stem}_metadata.csv",
        "channels": d / f"{stem}_channels.json",
        "params": d / f"{stem}_params.json",
        "summary": d / f"{stem}_prep_summary.json",
    }


def robust_high_tail_count(values: np.ndarray, threshold_mad: float = 8.0) -> int:
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med)))
    if mad == 0.0:
        return 0
    robust_z = 0.6744897501960817 * (values - med) / mad
    return int(np.sum(robust_z > threshold_mad))


def main() -> None:
    QC_DIR.mkdir(parents=True, exist_ok=True)

    reference_times = None
    reference_freqs = None
    reference_channels = None
    grand_sum = None
    n_subjects = 0

    qc_rows = []
    count_rows = []

    for sub in participants():
        paths = subject_paths(sub)
        missing = [str(p) for p in paths.values() if not p.exists()]
        if missing:
            raise FileNotFoundError(f"{sub}: missing TFR output(s):\n" + "\n".join(missing))

        power = np.load(paths["power"], mmap_mode="r")
        times = np.load(paths["times"])
        freqs = np.load(paths["freqs"])
        meta = pd.read_csv(paths["metadata"])
        with open(paths["channels"], "r", encoding="utf-8") as f:
            channels = json.load(f)

        if power.ndim != 4:
            raise RuntimeError(f"{sub}: expected 4D power array, got {power.shape}")
        n_trials, n_channels, n_freqs, n_times = power.shape
        if len(meta) != n_trials:
            raise RuntimeError(f"{sub}: metadata rows {len(meta)} != power trials {n_trials}")
        if len(channels) != n_channels or len(freqs) != n_freqs or len(times) != n_times:
            raise RuntimeError(f"{sub}: coordinate lengths do not match power shape {power.shape}")
        if not np.isfinite(np.asarray(power)).all():
            raise RuntimeError(f"{sub}: non-finite TFR values")

        if reference_times is None:
            reference_times = times
            reference_freqs = freqs
            reference_channels = channels
            grand_sum = np.zeros((n_channels, n_freqs, n_times), dtype=np.float64)
        else:
            if channels != reference_channels:
                raise RuntimeError(f"{sub}: channel order differs from first participant")
            if not np.array_equal(freqs, reference_freqs):
                raise RuntimeError(f"{sub}: frequency grid differs from first participant")
            if not np.allclose(times, reference_times, atol=1e-12, rtol=0):
                raise RuntimeError(f"{sub}: time grid differs from first participant")

        baseline_mask = (times >= BASELINE[0] - 1e-12) & (times <= BASELINE[1] + 1e-12)
        gamma_f = (freqs >= LOW_GAMMA[0]) & (freqs <= LOW_GAMMA[1])
        gamma_t = (times >= POSTCUE_GAMMA_WINDOW[0]) & (times <= POSTCUE_GAMMA_WINDOW[1])
        if not baseline_mask.any() or not gamma_f.any() or not gamma_t.any():
            raise RuntimeError(f"{sub}: QC masks are empty")

        # Equal participant weighting for the group QC average.
        subj_mean = np.asarray(power).mean(axis=0, dtype=np.float64)
        grand_sum += subj_mean
        n_subjects += 1

        baseline_mean = float(np.asarray(power[..., baseline_mask]).mean())
        gamma_trial_mean = np.asarray(power[:, :, gamma_f, :][:, :, :, gamma_t]).mean(axis=(1, 2, 3))

        qc_rows.append(
            {
                "participant": sub,
                "n_trials": n_trials,
                "n_channels": n_channels,
                "n_freqs": n_freqs,
                "n_times": n_times,
                "baseline_grand_mean_logpower": baseline_mean,
                "low_gamma_trial_median": float(np.median(gamma_trial_mean)),
                "low_gamma_trial_p95": float(np.quantile(gamma_trial_mean, 0.95)),
                "low_gamma_trial_max": float(np.max(gamma_trial_mean)),
                "low_gamma_n_extreme_gt8mad": robust_high_tail_count(gamma_trial_mean),
            }
        )

        counts = (
            meta.groupby(["condition", "level_1_5"], dropna=False)
            .size()
            .rename("n_trials")
            .reset_index()
        )
        counts.insert(0, "participant", sub)
        count_rows.append(counts)

        print(f"[{sub}] QC OK: shape={power.shape}, baseline mean={baseline_mean:.6f}")

    if n_subjects == 0:
        raise RuntimeError("No participant TFR outputs found")

    grand = (grand_sum / n_subjects).astype(np.float32)
    np.save(QC_DIR / "passive_tfr_grandmean.npy", grand, allow_pickle=False)

    qc_df = pd.DataFrame(qc_rows)
    qc_df.to_csv(QC_DIR / "passive_tfr_qc_summary.csv", index=False)
    pd.concat(count_rows, ignore_index=True).to_csv(
        QC_DIR / "passive_tfr_condition_counts.csv", index=False
    )

    # A large baseline offset here would indicate a normalisation/coordinate bug,
    # not a scientific effect. Keep the tolerance loose enough for 20-ms rebinning.
    max_abs_baseline = float(qc_df["baseline_grand_mean_logpower"].abs().max())
    if max_abs_baseline > 0.10:
        raise RuntimeError(
            f"Baseline QC failed: max absolute participant baseline mean = {max_abs_baseline:.4f}"
        )

    for channel in PLOT_CHANNELS:
        if channel not in reference_channels:
            continue
        c = reference_channels.index(channel)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        im = ax.imshow(
            grand[c],
            origin="lower",
            aspect="auto",
            extent=[reference_times[0], reference_times[-1], reference_freqs[0], reference_freqs[-1]],
        )
        ax.axvline(0.0, linewidth=1)
        ax.set_xlabel("Time from cue (s)")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_title(f"Passive grand-average normalised power — {channel}")
        fig.colorbar(im, ax=ax, label="mean-log-ratio power")
        fig.tight_layout()
        fig.savefig(QC_DIR / f"passive_tfr_grandmean_{channel}.png", dpi=200)
        plt.close(fig)

    with open(QC_DIR / "passive_tfr_qc_params.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "n_subjects": n_subjects,
                "baseline_s": list(BASELINE),
                "low_gamma_hz": list(LOW_GAMMA),
                "low_gamma_postcue_window_s": list(POSTCUE_GAMMA_WINDOW),
                "gamma_extreme_qc_rule": "robust z > 8 MAD-equivalent; QC only, no automatic exclusion",
                "plot_channels": PLOT_CHANNELS,
            },
            f,
            indent=2,
        )

    print("\nGROUP TFR QC COMPLETE")
    print("=" * 72)
    print(f"participants: {n_subjects}")
    print(f"max |baseline mean|: {max_abs_baseline:.6f}")
    print(f"outputs: {QC_DIR}")


if __name__ == "__main__":
    main()
