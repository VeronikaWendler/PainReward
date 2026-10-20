#!/usr/bin/env python3
"""
05r_eeg_tfr_response_locked_mvpa_full.py
========================================

Proper response-locked pain/money spectral MVPA for the PainReward project.

The completed response-locked TFRs span -0.8..+0.2 s relative to the response.
This analysis uses the broad PRE-RESPONSE interval frozen by the behavior-only
05q audit rather than forcing every participant/trial into one tiny common
post-cue interval.

At response-relative time tau, a trial is allowed to enter the decoder only if
its CURRENT trial cue has already occurred:

    event_RT + tau >= 0

Thus very fast trials can be absent at early response-relative times without
forcing all other trials/participants to be discarded.

Primary analysis
----------------
For pain and money separately, at each frequency x response-time point:

    65-channel power -> StandardScaler -> Ridge(alpha=1)
                     -> leave-one-decision-block-out
                     -> pooled OOF Pearson r

RAW decoding is primary.

RT sensitivity
--------------
The same decoder is repeated after removing from EEG FEATURES, using TRAINING
FOLDS ONLY, variance associated with:

    log(RT)
    log(RT)^2

This is a sensitivity analysis. It is not described as an "RT-independent"
neural signal because RT is itself downstream of the decision process.

Motor / choice specificity
--------------------------
If 05q established adequate behavioral support, magnitude decoding is repeated
while response outcome/identity is fixed. The exact fixed-response definition
(accepted==1 or a literal response_event_type) is determined by 05q BEFORE any
new neural results are inspected.

This asks whether graded pain/money information survives when a simple
accept-vs-reject / response-identity difference cannot drive the decoder.

Inference
---------
- full 4..45 Hz x response-time surface;
- theta 4-7, alpha 8-13, beta 14-30, low-gamma 31-45;
- one-sided positive group cluster tests;
- Holm across pain/money surface omnibus tests;
- Holm across four canonical bands separately within each attribute;
- direct Fisher-z(RT-adjusted) - Fisher-z(raw) > 0 comparison.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from response_locked_representation_common import (
    ATTRIBUTES,
    CANONICAL_BANDS,
    MVPA_OUTPUT_ROOT,
    cue_has_occurred_mask,
    fisher_z,
    leave_one_block_out_score,
    load_design_plan,
    load_response_axes,
    load_response_metadata,
    response_tfr_paths,
    summarize_band_family,
    summarize_frequency_time_family,
)


PARTICIPANT_SCORE_DIR = MVPA_OUTPUT_ROOT / "participant_scores"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Proper full response-locked PainReward TFR MVPA."
    )
    parser.add_argument("--n-permutations", type=int, default=5000)
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=max(1, int(os.getenv("SLURM_CPUS_PER_TASK", "1"))),
    )
    parser.add_argument("--overwrite-scores", action="store_true")
    parser.add_argument("--group-only", action="store_true")
    return parser.parse_args()


def participant_score_path(participant: str) -> Path:
    return PARTICIPANT_SCORE_DIR / f"{participant}_response_full_mvpa.npz"


def fixed_response_mask_from_plan(metadata: pd.DataFrame, motor_plan: dict) -> np.ndarray:
    kind = motor_plan["kind"]
    value = motor_plan["value"]

    if kind == "accepted_equals_1":
        return metadata["accepted"].to_numpy(int) == int(value)

    if kind == "response_event_type":
        return metadata["response_event_type"].astype(str).to_numpy() == str(value)

    raise RuntimeError(f"Unknown motor-control kind in design plan: {kind!r}")


def band_features(current_power: np.ndarray, frequencies: np.ndarray, band_name: str) -> np.ndarray:
    low_hz, high_hz = CANONICAL_BANDS[band_name]
    frequency_indices = np.where(
        (frequencies >= low_hz) & (frequencies <= high_hz)
    )[0]
    if len(frequency_indices) == 0:
        raise RuntimeError(f"No frequencies found for band {band_name}")

    # np.take makes the axis explicit and avoids NumPy advanced-indexing
    # dimension reordering.
    return np.take(
        current_power,
        frequency_indices,
        axis=2,
    ).mean(axis=2)


def decode_one_participant(participant: str, plan: dict, overwrite_scores: bool) -> None:
    output_path = participant_score_path(participant)
    if output_path.exists() and not overwrite_scores:
        print(f"    {participant}: score file exists; skipping", flush=True)
        return

    paths = response_tfr_paths(participant)
    power = np.load(paths["power"], mmap_mode="r")
    times, frequencies = load_response_axes(participant)
    metadata = load_response_metadata(participant)

    if power.ndim != 4:
        raise RuntimeError(f"{participant}: expected 4D TFR, got {power.shape}")
    if power.shape[0] != len(metadata):
        raise RuntimeError(
            f"{participant}: TFR trials {power.shape[0]} != metadata rows {len(metadata)}"
        )
    if power.shape[2] != len(frequencies):
        raise RuntimeError(f"{participant}: TFR frequency dimension mismatch")
    if power.shape[3] != len(times):
        raise RuntimeError(f"{participant}: TFR time dimension mismatch")

    primary_tmin, primary_tmax = map(float, plan["primary_formal_window_s"])
    primary_time_mask = (
        (times >= primary_tmin - 1e-12)
        & (times <= primary_tmax + 1e-12)
    )
    primary_time_indices = np.where(primary_time_mask)[0]
    primary_times = times[primary_time_mask]

    if len(primary_times) == 0:
        raise RuntimeError(f"{participant}: no bins in primary formal window")

    pain_levels = metadata["painlevel"].to_numpy(float)
    money_levels = metadata["moneylevel"].to_numpy(float)
    block_ids = metadata["block"].to_numpy(int)
    reaction_time = metadata["behavioral_rt_s"].to_numpy(float)

    targets = {
        "pain": pain_levels,
        "money": money_levels,
    }

    n_attributes = len(ATTRIBUTES)
    n_frequencies = len(frequencies)
    n_times = len(primary_times)
    band_names = list(CANONICAL_BANDS)

    raw_frequency_scores = np.full(
        (n_attributes, n_frequencies, n_times),
        np.nan,
        dtype=np.float32,
    )
    rt_frequency_scores = np.full_like(raw_frequency_scores, np.nan)
    raw_band_scores = np.full(
        (n_attributes, len(band_names), n_times),
        np.nan,
        dtype=np.float32,
    )
    rt_band_scores = np.full_like(raw_band_scores, np.nan)
    primary_trial_counts = np.zeros(n_times, dtype=int)

    print(f"    {participant}: primary full response-locked decoding", flush=True)

    for local_time_index, global_time_index in enumerate(primary_time_indices):
        response_time_s = float(times[global_time_index])
        trial_mask = cue_has_occurred_mask(metadata, response_time_s)

        primary_trial_counts[local_time_index] = int(trial_mask.sum())
        current_blocks = block_ids[trial_mask]
        current_rt = reaction_time[trial_mask]

        # Shape: trial x channel x frequency.
        current_power = np.asarray(
            power[trial_mask, :, :, global_time_index],
            dtype=float,
        )

        for attribute_index, attribute_name in enumerate(ATTRIBUTES):
            target = targets[attribute_name][trial_mask]

            for frequency_index in range(n_frequencies):
                features = current_power[:, :, frequency_index]

                raw_frequency_scores[
                    attribute_index,
                    frequency_index,
                    local_time_index,
                ] = leave_one_block_out_score(
                    features=features,
                    target=target,
                    block_ids=current_blocks,
                    reaction_time=current_rt,
                    remove_rt_related_variance=False,
                )

                rt_frequency_scores[
                    attribute_index,
                    frequency_index,
                    local_time_index,
                ] = leave_one_block_out_score(
                    features=features,
                    target=target,
                    block_ids=current_blocks,
                    reaction_time=current_rt,
                    remove_rt_related_variance=True,
                )

            for band_index, band_name in enumerate(band_names):
                features = band_features(
                    current_power=current_power,
                    frequencies=frequencies,
                    band_name=band_name,
                )

                raw_band_scores[
                    attribute_index,
                    band_index,
                    local_time_index,
                ] = leave_one_block_out_score(
                    features=features,
                    target=target,
                    block_ids=current_blocks,
                    reaction_time=current_rt,
                    remove_rt_related_variance=False,
                )

                rt_band_scores[
                    attribute_index,
                    band_index,
                    local_time_index,
                ] = leave_one_block_out_score(
                    features=features,
                    target=target,
                    block_ids=current_blocks,
                    reaction_time=current_rt,
                    remove_rt_related_variance=True,
                )

    for name, values in {
        "raw frequency": raw_frequency_scores,
        "RT-adjusted frequency": rt_frequency_scores,
        "raw band": raw_band_scores,
        "RT-adjusted band": rt_band_scores,
    }.items():
        if not np.isfinite(values).all():
            raise RuntimeError(
                f"{participant}: {name} scores contain non-finite values. "
                "The 05q support audit and decoder do not agree."
            )

    # ----------------------------------------------------------------------
    # Fixed-response motor/choice-specificity analysis.
    # ----------------------------------------------------------------------
    motor_plan = plan["motor_control"]
    motor_enabled_for_participant = bool(
        motor_plan["enabled"]
        and participant in motor_plan["participants"]
    )

    if motor_enabled_for_participant:
        motor_tmin, motor_tmax = map(float, motor_plan["formal_window_s"])
        motor_time_mask = (
            (times >= motor_tmin - 1e-12)
            & (times <= motor_tmax + 1e-12)
        )
        motor_time_indices = np.where(motor_time_mask)[0]
        motor_times = times[motor_time_mask]
        fixed_response_mask = fixed_response_mask_from_plan(metadata, motor_plan)

        motor_raw_frequency_scores = np.full(
            (n_attributes, n_frequencies, len(motor_times)),
            np.nan,
            dtype=np.float32,
        )
        motor_rt_frequency_scores = np.full_like(
            motor_raw_frequency_scores,
            np.nan,
        )
        motor_raw_band_scores = np.full(
            (n_attributes, len(band_names), len(motor_times)),
            np.nan,
            dtype=np.float32,
        )
        motor_rt_band_scores = np.full_like(
            motor_raw_band_scores,
            np.nan,
        )
        motor_trial_counts = np.zeros(len(motor_times), dtype=int)

        print(
            f"    {participant}: fixed-response specificity decoding "
            f"({motor_plan['kind']}={motor_plan['value']})",
            flush=True,
        )

        for local_time_index, global_time_index in enumerate(motor_time_indices):
            response_time_s = float(times[global_time_index])
            trial_mask = (
                cue_has_occurred_mask(metadata, response_time_s)
                & fixed_response_mask
            )

            motor_trial_counts[local_time_index] = int(trial_mask.sum())
            current_blocks = block_ids[trial_mask]
            current_rt = reaction_time[trial_mask]
            current_power = np.asarray(
                power[trial_mask, :, :, global_time_index],
                dtype=float,
            )

            for attribute_index, attribute_name in enumerate(ATTRIBUTES):
                target = targets[attribute_name][trial_mask]

                for frequency_index in range(n_frequencies):
                    features = current_power[:, :, frequency_index]

                    motor_raw_frequency_scores[
                        attribute_index,
                        frequency_index,
                        local_time_index,
                    ] = leave_one_block_out_score(
                        features=features,
                        target=target,
                        block_ids=current_blocks,
                        reaction_time=current_rt,
                        remove_rt_related_variance=False,
                    )

                    motor_rt_frequency_scores[
                        attribute_index,
                        frequency_index,
                        local_time_index,
                    ] = leave_one_block_out_score(
                        features=features,
                        target=target,
                        block_ids=current_blocks,
                        reaction_time=current_rt,
                        remove_rt_related_variance=True,
                    )

                for band_index, band_name in enumerate(band_names):
                    features = band_features(
                        current_power=current_power,
                        frequencies=frequencies,
                        band_name=band_name,
                    )

                    motor_raw_band_scores[
                        attribute_index,
                        band_index,
                        local_time_index,
                    ] = leave_one_block_out_score(
                        features=features,
                        target=target,
                        block_ids=current_blocks,
                        reaction_time=current_rt,
                        remove_rt_related_variance=False,
                    )

                    motor_rt_band_scores[
                        attribute_index,
                        band_index,
                        local_time_index,
                    ] = leave_one_block_out_score(
                        features=features,
                        target=target,
                        block_ids=current_blocks,
                        reaction_time=current_rt,
                        remove_rt_related_variance=True,
                    )

        for name, values in {
            "motor raw frequency": motor_raw_frequency_scores,
            "motor RT-adjusted frequency": motor_rt_frequency_scores,
            "motor raw band": motor_raw_band_scores,
            "motor RT-adjusted band": motor_rt_band_scores,
        }.items():
            if not np.isfinite(values).all():
                raise RuntimeError(
                    f"{participant}: {name} scores contain non-finite values "
                    "despite inclusion by the 05q motor-support audit."
                )
    else:
        motor_times = np.asarray([], dtype=float)
        motor_trial_counts = np.asarray([], dtype=int)
        motor_raw_frequency_scores = np.empty(
            (n_attributes, n_frequencies, 0),
            dtype=np.float32,
        )
        motor_rt_frequency_scores = motor_raw_frequency_scores.copy()
        motor_raw_band_scores = np.empty(
            (n_attributes, len(band_names), 0),
            dtype=np.float32,
        )
        motor_rt_band_scores = motor_raw_band_scores.copy()

    np.savez_compressed(
        output_path,
        primary_times=primary_times,
        freqs=frequencies,
        band_names=np.asarray(band_names),
        attribute_names=np.asarray(ATTRIBUTES),
        primary_trial_counts=primary_trial_counts,
        raw_freq_r=raw_frequency_scores,
        rtres_freq_r=rt_frequency_scores,
        raw_band_r=raw_band_scores,
        rtres_band_r=rt_band_scores,
        motor_enabled=np.asarray(motor_enabled_for_participant),
        motor_times=motor_times,
        motor_trial_counts=motor_trial_counts,
        motor_raw_freq_r=motor_raw_frequency_scores,
        motor_rtres_freq_r=motor_rt_frequency_scores,
        motor_raw_band_r=motor_raw_band_scores,
        motor_rtres_band_r=motor_rt_band_scores,
    )

    print(
        f"    {participant}: primary scores saved; current-trial counts "
        f"{primary_trial_counts.min()}..{primary_trial_counts.max()}",
        flush=True,
    )
    if motor_enabled_for_participant:
        print(
            f"    {participant}: fixed-response scores saved; trial counts "
            f"{motor_trial_counts.min()}..{motor_trial_counts.max()}",
            flush=True,
        )


def concatenate_nonempty_tables(tables: list[pd.DataFrame]) -> pd.DataFrame:
    nonempty = [table for table in tables if len(table)]
    return pd.concat(nonempty, ignore_index=True) if nonempty else pd.DataFrame()


def run_group_analysis(plan: dict, n_permutations: int, n_jobs: int) -> None:
    primary_participants = plan["primary_participants"]
    participant_results = [
        np.load(participant_score_path(participant), allow_pickle=False)
        for participant in primary_participants
    ]

    times = participant_results[0]["primary_times"].astype(float)
    frequencies = participant_results[0]["freqs"].astype(float)
    band_names = participant_results[0]["band_names"].astype(str).tolist()

    for participant, result in zip(primary_participants, participant_results):
        if not np.allclose(result["primary_times"].astype(float), times, atol=1e-12):
            raise RuntimeError(f"{participant}: primary response-time grid differs")
        if not np.array_equal(result["freqs"].astype(float), frequencies):
            raise RuntimeError(f"{participant}: frequency grid differs")

    raw_frequency = np.stack([result["raw_freq_r"] for result in participant_results], axis=0)
    rt_frequency = np.stack([result["rtres_freq_r"] for result in participant_results], axis=0)
    raw_band = np.stack([result["raw_band_r"] for result in participant_results], axis=0)
    rt_band = np.stack([result["rtres_band_r"] for result in participant_results], axis=0)

    rt_minus_raw_frequency = fisher_z(rt_frequency) - fisher_z(raw_frequency)
    rt_minus_raw_band = fisher_z(rt_band) - fisher_z(raw_band)

    np.savez_compressed(
        MVPA_OUTPUT_ROOT / "group_score_arrays.npz",
        participants=np.asarray(primary_participants),
        times=times,
        freqs=frequencies,
        band_names=np.asarray(band_names),
        attribute_names=np.asarray(ATTRIBUTES),
        raw_freq_r=raw_frequency,
        rtres_freq_r=rt_frequency,
        raw_band_r=raw_band,
        rtres_band_r=rt_band,
        rt_minus_raw_freq_fisher_z=rt_minus_raw_frequency,
        rt_minus_raw_band_fisher_z=rt_minus_raw_band,
    )

    surface_omnibus_tables = []
    surface_cluster_tables = []
    band_omnibus_tables = []
    band_cluster_tables = []

    for family_name, frequency_values, band_values, transform_correlations in [
        ("raw_response_locked_full", raw_frequency, raw_band, True),
        ("rt_adjusted_response_locked_full", rt_frequency, rt_band, True),
        ("rt_adjusted_minus_raw_fisher_z", rt_minus_raw_frequency, rt_minus_raw_band, False),
    ]:
        surface_omnibus, surface_clusters = summarize_frequency_time_family(
            group_values=frequency_values,
            family_name=family_name,
            frequencies=frequencies,
            times=times,
            n_permutations=n_permutations,
            n_jobs=n_jobs,
            fisher_transform_correlations=transform_correlations,
        )
        band_omnibus, band_clusters = summarize_band_family(
            group_values=band_values,
            family_name=family_name,
            times=times,
            band_names=band_names,
            n_permutations=n_permutations,
            n_jobs=n_jobs,
            fisher_transform_correlations=transform_correlations,
        )
        surface_omnibus_tables.append(surface_omnibus)
        surface_cluster_tables.append(surface_clusters)
        band_omnibus_tables.append(band_omnibus)
        band_cluster_tables.append(band_clusters)

    surface_omnibus = pd.concat(surface_omnibus_tables, ignore_index=True)
    surface_clusters = concatenate_nonempty_tables(surface_cluster_tables)
    band_omnibus = pd.concat(band_omnibus_tables, ignore_index=True)
    band_clusters = concatenate_nonempty_tables(band_cluster_tables)

    surface_omnibus.to_csv(MVPA_OUTPUT_ROOT / "surface_omnibus_holm.csv", index=False)
    surface_clusters.to_csv(MVPA_OUTPUT_ROOT / "surface_significant_clusters.csv", index=False)
    band_omnibus.to_csv(MVPA_OUTPUT_ROOT / "band_omnibus_holm.csv", index=False)
    band_clusters.to_csv(MVPA_OUTPUT_ROOT / "band_significant_clusters.csv", index=False)

    trial_count_rows = []
    for participant, result in zip(primary_participants, participant_results):
        for time_point, trial_count in zip(times, result["primary_trial_counts"]):
            trial_count_rows.append({
                "participant": participant,
                "response_time_s": float(time_point),
                "n_trials": int(trial_count),
            })
    pd.DataFrame(trial_count_rows).to_csv(
        MVPA_OUTPUT_ROOT / "primary_trial_counts_by_participant_time.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # Fixed-response specificity group inference.
    # ------------------------------------------------------------------
    motor_plan = plan["motor_control"]
    motor_surface_omnibus = pd.DataFrame()
    motor_surface_clusters = pd.DataFrame()
    motor_band_omnibus = pd.DataFrame()
    motor_band_clusters = pd.DataFrame()

    if motor_plan["enabled"]:
        motor_participants = motor_plan["participants"]
        motor_results = [
            np.load(participant_score_path(participant), allow_pickle=False)
            for participant in motor_participants
        ]
        motor_times = motor_results[0]["motor_times"].astype(float)

        for participant, result in zip(motor_participants, motor_results):
            if not np.allclose(result["motor_times"].astype(float), motor_times, atol=1e-12):
                raise RuntimeError(f"{participant}: motor response-time grid differs")

        motor_raw_frequency = np.stack(
            [result["motor_raw_freq_r"] for result in motor_results],
            axis=0,
        )
        motor_rt_frequency = np.stack(
            [result["motor_rtres_freq_r"] for result in motor_results],
            axis=0,
        )
        motor_raw_band = np.stack(
            [result["motor_raw_band_r"] for result in motor_results],
            axis=0,
        )
        motor_rt_band = np.stack(
            [result["motor_rtres_band_r"] for result in motor_results],
            axis=0,
        )

        motor_surface_tables = []
        motor_surface_cluster_tables = []
        motor_band_tables = []
        motor_band_cluster_tables = []

        for family_name, frequency_values, band_values in [
            ("fixed_response_raw", motor_raw_frequency, motor_raw_band),
            ("fixed_response_rt_adjusted", motor_rt_frequency, motor_rt_band),
        ]:
            current_surface_omnibus, current_surface_clusters = summarize_frequency_time_family(
                group_values=frequency_values,
                family_name=family_name,
                frequencies=frequencies,
                times=motor_times,
                n_permutations=n_permutations,
                n_jobs=n_jobs,
                fisher_transform_correlations=True,
            )
            current_band_omnibus, current_band_clusters = summarize_band_family(
                group_values=band_values,
                family_name=family_name,
                times=motor_times,
                band_names=band_names,
                n_permutations=n_permutations,
                n_jobs=n_jobs,
                fisher_transform_correlations=True,
            )
            motor_surface_tables.append(current_surface_omnibus)
            motor_surface_cluster_tables.append(current_surface_clusters)
            motor_band_tables.append(current_band_omnibus)
            motor_band_cluster_tables.append(current_band_clusters)

        motor_surface_omnibus = pd.concat(motor_surface_tables, ignore_index=True)
        motor_surface_clusters = concatenate_nonempty_tables(motor_surface_cluster_tables)
        motor_band_omnibus = pd.concat(motor_band_tables, ignore_index=True)
        motor_band_clusters = concatenate_nonempty_tables(motor_band_cluster_tables)

        motor_surface_omnibus.to_csv(
            MVPA_OUTPUT_ROOT / "motor_specificity_surface_omnibus_holm.csv",
            index=False,
        )
        motor_surface_clusters.to_csv(
            MVPA_OUTPUT_ROOT / "motor_specificity_surface_significant_clusters.csv",
            index=False,
        )
        motor_band_omnibus.to_csv(
            MVPA_OUTPUT_ROOT / "motor_specificity_band_omnibus_holm.csv",
            index=False,
        )
        motor_band_clusters.to_csv(
            MVPA_OUTPUT_ROOT / "motor_specificity_band_significant_clusters.csv",
            index=False,
        )

    params = {
        "analysis": "proper broad response-locked pain/money spectral MVPA",
        "design_plan": plan,
        "decoder": "StandardScaler -> Ridge(alpha=1)",
        "cv": "leave-one-decision-block-out GroupKFold",
        "score": "pooled OOF Pearson r",
        "trial_availability": "event_rt_s + response_relative_time_s >= 0",
        "primary": "raw response-locked magnitude decoding",
        "rt_sensitivity": (
            "training-fold-only EEG feature residualisation for log(RT), log(RT)^2"
        ),
        "motor_specificity": motor_plan,
        "n_permutations": n_permutations,
    }
    (MVPA_OUTPUT_ROOT / "params.json").write_text(json.dumps(params, indent=2))

    print("\nPRIMARY / RT-ADJUSTED RESPONSE-LOCKED FREQUENCY x TIME", flush=True)
    print(surface_omnibus.to_string(index=False), flush=True)
    if len(surface_clusters):
        print("\nSIGNIFICANT RESPONSE-LOCKED SURFACE CLUSTERS", flush=True)
        print(surface_clusters.to_string(index=False), flush=True)

    print("\nCANONICAL BANDS", flush=True)
    print(band_omnibus.to_string(index=False), flush=True)
    if len(band_clusters):
        print("\nSIGNIFICANT BAND CLUSTERS", flush=True)
        print(band_clusters.to_string(index=False), flush=True)

    if motor_plan["enabled"]:
        print("\nFIXED-RESPONSE MOTOR/CHOICE-SPECIFICITY FREQUENCY x TIME", flush=True)
        print(motor_surface_omnibus.to_string(index=False), flush=True)
        if len(motor_surface_clusters):
            print("\nSIGNIFICANT FIXED-RESPONSE SURFACE CLUSTERS", flush=True)
            print(motor_surface_clusters.to_string(index=False), flush=True)

    print(f"\nOutputs: {MVPA_OUTPUT_ROOT}", flush=True)


def main() -> None:
    args = parse_args()
    MVPA_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    PARTICIPANT_SCORE_DIR.mkdir(parents=True, exist_ok=True)

    plan = load_design_plan()
    primary_participants = plan["primary_participants"]
    motor_participants = plan["motor_control"]["participants"]
    participants_to_score = sorted(set(primary_participants) | set(motor_participants))

    print("=" * 78, flush=True)
    print("PAINREWARD PROPER BROAD RESPONSE-LOCKED TFR MVPA", flush=True)
    print(f"Primary formal window: {plan['primary_formal_window_s']}", flush=True)
    print(f"Primary fixed participant sample: {len(primary_participants)}", flush=True)
    print(f"Motor specificity enabled: {plan['motor_control']['enabled']}", flush=True)
    print(f"Permutations: {args.n_permutations}", flush=True)
    print("=" * 78, flush=True)

    if not args.group_only:
        for participant_index, participant in enumerate(participants_to_score, start=1):
            print(
                f"\n[{participant_index}/{len(participants_to_score)}] {participant}",
                flush=True,
            )
            decode_one_participant(
                participant=participant,
                plan=plan,
                overwrite_scores=args.overwrite_scores,
            )

    missing = [
        participant
        for participant in primary_participants
        if not participant_score_path(participant).exists()
    ]
    if missing:
        raise RuntimeError(
            "Missing primary participant MVPA score files: " + ", ".join(missing)
        )

    run_group_analysis(
        plan=plan,
        n_permutations=args.n_permutations,
        n_jobs=args.n_jobs,
    )

    print("\n" + "=" * 78, flush=True)
    print("PROPER BROAD RESPONSE-LOCKED MVPA COMPLETE", flush=True)
    print("=" * 78, flush=True)


if __name__ == "__main__":
    main()
