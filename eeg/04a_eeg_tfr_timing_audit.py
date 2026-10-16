"""
TFR timing audit for the PainReward project.

Purpose
-------
Before computing cue-locked time-frequency representations, verify that the
planned pre-cue baseline lies inside the fixation period on every real trial.
This script is intentionally independent of EEG outcomes: it reads only the
PsychoPy behavioural timing columns and writes timing/QC tables.

Cluster paths follow the rest of the repository:
    basepath = $basepath (set to /data inside the MNE container)

Outputs
-------
derivatives/tfr/timing/
    tfr_timing_trials.csv
    tfr_timing_summary_by_participant.csv
    tfr_timing_summary_global.csv
    tfr_timing_params.json

The default planned baseline is [-0.7, -0.5] s relative to cue onset. With
n_cycles = frequency / 2, the longest Morlet half-support is ~0.398 s, so the
baseline ending at -0.5 s remains separated from cue onset by more than that
conservative half-support estimate.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


BASE_PATH = Path(os.getenv("basepath", Path(__file__).parent.parent.parent))
OUT_DIR = BASE_PATH / "derivatives" / "tfr" / "timing"

# These are the values used by 04b_eeg_tfr_prep.py. Keep them synchronized.
BASELINE_TMIN = -0.7
BASELINE_TMAX = -0.5
LOWEST_FREQ_HZ = 4.0
N_CYCLES_FACTOR = 0.5  # n_cycles = factor * frequency
MORLET_N_SIGMAS = 5.0  # MNE Morlet wavelet support is approximately +/- 5 sigma_t


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit pre-cue fixation timing for TFR baseline selection.")
    parser.add_argument(
        "--require-safe",
        action="store_true",
        help=(
            "Fail if any passive trial has less fixation before the cue than "
            f"required by the planned baseline start ({abs(BASELINE_TMIN):.3f} s)."
        ),
    )
    return parser.parse_args()


def participants() -> list[str]:
    subs = sorted(p.name for p in BASE_PATH.iterdir() if p.is_dir() and p.name.startswith("sub-"))
    if not subs:
        raise RuntimeError(f"No sub-* directories found under basepath: {BASE_PATH}")
    return subs


def task_beh_path(participant: str, task: str) -> Path:
    return BASE_PATH / participant / "eeg" / f"{participant}_task-{task}_beh.tsv"


def read_task_timing(participant: str, task: str) -> pd.DataFrame:
    path = task_beh_path(participant, task)
    if not path.exists():
        raise FileNotFoundError(f"{participant}: missing {task} behaviour file: {path}")

    df = pd.read_csv(path, sep="\t")

    if task == "passive":
        cue_col = "cue.started"
    elif task == "decision":
        # This is the same cue-onset column used by 03_eeg_preprocess.py.
        cue_col = "cue_right1.started"
    else:
        raise ValueError(f"Unsupported task: {task}")

    required = ["fixcross.started", cue_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"{participant} [{task}]: missing timing columns: {missing}")

    # Same definition of a real trial used elsewhere in the PainReward pipeline.
    df = df[~df["fixcross.started"].isna()].copy().reset_index(drop=True)
    if df.empty:
        raise RuntimeError(f"{participant} [{task}]: no real trial rows after fixation filtering")

    fix = pd.to_numeric(df["fixcross.started"], errors="coerce")
    cue = pd.to_numeric(df[cue_col], errors="coerce")
    if fix.isna().any() or cue.isna().any():
        raise RuntimeError(f"{participant} [{task}]: non-numeric fixation/cue onset on real trial rows")

    duration = cue - fix
    if (duration <= 0).any():
        bad = np.where(duration.to_numpy() <= 0)[0][:10].tolist()
        raise RuntimeError(f"{participant} [{task}]: non-positive fixation duration on trial rows {bad}")

    out = pd.DataFrame(
        {
            "participant": participant,
            "task": task,
            "trial_seq": np.arange(1, len(df) + 1, dtype=int),
            "fixation_duration_s": duration.to_numpy(dtype=float),
        }
    )
    out["planned_baseline_tmin_s"] = BASELINE_TMIN
    out["planned_baseline_tmax_s"] = BASELINE_TMAX
    out["baseline_start_after_fixation_onset_s"] = out["fixation_duration_s"] + BASELINE_TMIN
    out["baseline_fully_inside_fixation"] = out["fixation_duration_s"] >= abs(BASELINE_TMIN)
    return out


def morlet_half_support_s(freq_hz: float) -> float:
    """Conservative half-support estimate: 5 * sigma_t for MNE's Morlet wavelet."""
    n_cycles = N_CYCLES_FACTOR * freq_hz
    sigma_t = n_cycles / (2.0 * np.pi * freq_hz)
    return MORLET_N_SIGMAS * sigma_t


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    half_support = morlet_half_support_s(LOWEST_FREQ_HZ)
    cue_gap = abs(BASELINE_TMAX)
    if cue_gap <= half_support:
        raise RuntimeError(
            "Planned baseline ends too close to cue onset for the configured lowest-frequency "
            f"wavelet: gap={cue_gap:.3f}s, conservative half-support={half_support:.3f}s"
        )

    rows = []
    for sub in participants():
        rows.append(read_task_timing(sub, "passive"))
        rows.append(read_task_timing(sub, "decision"))

    trials = pd.concat(rows, ignore_index=True)

    summary_by_participant = (
        trials.groupby(["participant", "task"], as_index=False)
        .agg(
            n_trials=("trial_seq", "size"),
            min_fixation_s=("fixation_duration_s", "min"),
            p01_fixation_s=("fixation_duration_s", lambda x: float(np.quantile(x, 0.01))),
            p05_fixation_s=("fixation_duration_s", lambda x: float(np.quantile(x, 0.05))),
            median_fixation_s=("fixation_duration_s", "median"),
            n_baseline_unsafe=("baseline_fully_inside_fixation", lambda x: int((~x).sum())),
            prop_baseline_safe=("baseline_fully_inside_fixation", "mean"),
        )
    )

    summary_global = (
        trials.groupby("task", as_index=False)
        .agg(
            n_participants=("participant", "nunique"),
            n_trials=("trial_seq", "size"),
            min_fixation_s=("fixation_duration_s", "min"),
            p01_fixation_s=("fixation_duration_s", lambda x: float(np.quantile(x, 0.01))),
            p05_fixation_s=("fixation_duration_s", lambda x: float(np.quantile(x, 0.05))),
            median_fixation_s=("fixation_duration_s", "median"),
            n_baseline_unsafe=("baseline_fully_inside_fixation", lambda x: int((~x).sum())),
            prop_baseline_safe=("baseline_fully_inside_fixation", "mean"),
        )
    )

    trials.to_csv(OUT_DIR / "tfr_timing_trials.csv", index=False)
    summary_by_participant.to_csv(OUT_DIR / "tfr_timing_summary_by_participant.csv", index=False)
    summary_global.to_csv(OUT_DIR / "tfr_timing_summary_global.csv", index=False)

    params = {
        "basepath": str(BASE_PATH),
        "planned_baseline_s": [BASELINE_TMIN, BASELINE_TMAX],
        "lowest_frequency_hz": LOWEST_FREQ_HZ,
        "n_cycles_rule": "n_cycles = frequency * 0.5",
        "morlet_half_support_estimate_s": half_support,
        "baseline_end_to_cue_gap_s": cue_gap,
        "criterion": "baseline start must occur at or after fixation onset on each trial",
    }
    with open(OUT_DIR / "tfr_timing_params.json", "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)

    print("\nTFR TIMING AUDIT")
    print("=" * 72)
    print(f"basepath: {BASE_PATH}")
    print(f"planned baseline: [{BASELINE_TMIN:.3f}, {BASELINE_TMAX:.3f}] s")
    print(f"conservative lowest-frequency Morlet half-support: {half_support:.3f} s")
    print(summary_global.to_string(index=False))
    print(f"\nOutputs: {OUT_DIR}")

    passive_unsafe = int(
        summary_global.loc[summary_global["task"] == "passive", "n_baseline_unsafe"].iloc[0]
    )
    if args.require_safe and passive_unsafe > 0:
        offenders = summary_by_participant[
            (summary_by_participant["task"] == "passive")
            & (summary_by_participant["n_baseline_unsafe"] > 0)
        ]
        raise RuntimeError(
            f"Planned passive TFR baseline is not fully inside fixation for {passive_unsafe} trial(s).\n"
            f"Affected participants:\n{offenders.to_string(index=False)}\n"
            "Inspect the timing CSVs before launching the expensive TFR jobs."
        )


if __name__ == "__main__":
    main()
