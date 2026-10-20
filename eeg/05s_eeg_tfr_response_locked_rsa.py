#!/usr/bin/env python3
"""
05s_eeg_tfr_response_locked_rsa.py
==================================

Response-locked multiple-regression RSA for PainReward pain/money spectral
representations.

The MVPA asks whether graded magnitude can be predicted. This RSA asks whether
the distributed neural GEOMETRY contains graded pain and money structure after
accounting for the other attribute, reaction time, and choice/response
structure.

At each participant x frequency x response-relative time point:

1. Keep only trials whose CURRENT decision cue has already occurred:

       event_RT + response_time >= 0

2. Standardize each EEG channel across the currently available trials.

3. Neural RDM:
       Euclidean distance across the 65-channel spectral-power pattern.

4. Model RDMs:
       pain distance   = |pain_i - pain_j|
       money distance  = |money_i - money_j|
       RT distance     = |logRT_i - logRT_j|
       choice mismatch = accepted_i != accepted_j

   If the behavior-only 05q audit established that response-event identity
   contains non-redundant structure beyond accepted/rejected outcome, an
   additional response-event mismatch RDM is included.

5. Rank-transform neural and model RDM vectors, z-standardize model RDMs, then
   fit one simultaneous OLS multiple-regression RSA:

       neural_RDM ~ pain_RDM + money_RDM + RT_RDM + choice_RDM
                    [+ response_event_RDM]

The participant-level pain and money coefficients are the inferential targets.
No within-participant p-values are computed from pairwise RDM entries because
those entries are not independent.

Group inference mirrors MVPA:
- one-sided positive 4..45 Hz x response-time cluster tests;
- Holm across pain/money surface omnibus tests;
- canonical-band temporal tests, Holm across four bands within attribute.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import pdist

from response_locked_representation_common import (
    ATTRIBUTES,
    CANONICAL_BANDS,
    RSA_OUTPUT_ROOT,
    cue_has_occurred_mask,
    load_design_plan,
    load_response_axes,
    load_response_metadata,
    response_tfr_paths,
    summarize_band_family,
    summarize_frequency_time_family,
)


PARTICIPANT_SCORE_DIR = RSA_OUTPUT_ROOT / "participant_scores"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Response-locked multiple-regression RSA for PainReward."
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
    return PARTICIPANT_SCORE_DIR / f"{participant}_response_rsa.npz"


def rank_standardize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    ranked = stats.rankdata(values, method="average")
    standard_deviation = ranked.std()
    if standard_deviation <= 1e-12:
        raise RuntimeError("RDM vector has no variance")
    return (ranked - ranked.mean()) / standard_deviation


def neural_rdm_from_features(features: np.ndarray) -> np.ndarray:
    """
    Euclidean trial-pattern dissimilarity after channel-wise standardization.

    Channel standardization prevents intrinsically high-variance electrodes
    from dominating the RDM while preserving distributed multichannel pattern
    and amplitude information.
    """

    features = np.asarray(features, dtype=float)
    channel_mean = features.mean(axis=0, keepdims=True)
    channel_sd = features.std(axis=0, keepdims=True)
    channel_sd[channel_sd <= 1e-12] = 1.0
    standardized_features = (features - channel_mean) / channel_sd

    return pdist(
        standardized_features,
        metric="euclidean",
    ) / np.sqrt(standardized_features.shape[1])


def model_rdm_design(
    metadata: pd.DataFrame,
    include_response_event_mismatch: bool,
) -> tuple[list[str], np.ndarray]:
    pain = metadata["painlevel"].to_numpy(float)
    money = metadata["moneylevel"].to_numpy(float)
    reaction_time = metadata["behavioral_rt_s"].to_numpy(float)
    accepted = metadata["accepted"].to_numpy(int)

    if np.any(reaction_time <= 0):
        raise RuntimeError("Non-positive RT encountered in RSA")

    log_rt = np.log(reaction_time)

    predictor_names = [
        "pain",
        "money",
        "log_rt",
        "choice",
    ]
    predictor_vectors = [
        pdist(pain[:, None], metric="cityblock"),
        pdist(money[:, None], metric="cityblock"),
        pdist(log_rt[:, None], metric="cityblock"),
        pdist(accepted[:, None], metric="hamming"),
    ]

    if include_response_event_mismatch:
        response_event_codes, _ = pd.factorize(
            metadata["response_event_type"].astype(str),
            sort=True,
        )
        predictor_names.append("response_event")
        predictor_vectors.append(
            pdist(response_event_codes[:, None], metric="hamming")
        )

    standardized_predictors = [
        rank_standardize(vector)
        for vector in predictor_vectors
    ]

    design_matrix = np.column_stack(
        [
            np.ones(len(standardized_predictors[0])),
            *standardized_predictors,
        ]
    )

    if np.linalg.matrix_rank(design_matrix) < design_matrix.shape[1]:
        raise RuntimeError(
            "RSA model-RDM design is rank deficient. Inspect the 05q "
            "accepted/response-event mapping rather than silently dropping a control."
        )

    return predictor_names, design_matrix


def rsa_coefficients(
    features: np.ndarray,
    metadata: pd.DataFrame,
    include_response_event_mismatch: bool,
) -> dict[str, float]:
    neural_rdm = rank_standardize(
        neural_rdm_from_features(features)
    )

    predictor_names, design_matrix = model_rdm_design(
        metadata=metadata,
        include_response_event_mismatch=include_response_event_mismatch,
    )

    coefficients, *_ = np.linalg.lstsq(
        design_matrix,
        neural_rdm,
        rcond=None,
    )

    return {
        predictor_name: float(coefficients[predictor_index + 1])
        for predictor_index, predictor_name in enumerate(predictor_names)
    }


def band_features(current_power: np.ndarray, frequencies: np.ndarray, band_name: str) -> np.ndarray:
    low_hz, high_hz = CANONICAL_BANDS[band_name]
    frequency_indices = np.where(
        (frequencies >= low_hz) & (frequencies <= high_hz)
    )[0]
    if len(frequency_indices) == 0:
        raise RuntimeError(f"No frequencies found for band {band_name}")
    return np.take(current_power, frequency_indices, axis=2).mean(axis=2)


def process_one_participant(participant: str, plan: dict, overwrite_scores: bool) -> None:
    output_path = participant_score_path(participant)
    if output_path.exists() and not overwrite_scores:
        print(f"    {participant}: RSA score file exists; skipping", flush=True)
        return

    paths = response_tfr_paths(participant)
    power = np.load(paths["power"], mmap_mode="r")
    times, frequencies = load_response_axes(participant)
    metadata = load_response_metadata(participant)

    if power.ndim != 4:
        raise RuntimeError(f"{participant}: expected 4D TFR, got {power.shape}")
    if power.shape[0] != len(metadata):
        raise RuntimeError(f"{participant}: TFR/metadata trial mismatch")
    if power.shape[2] != len(frequencies) or power.shape[3] != len(times):
        raise RuntimeError(f"{participant}: TFR axis mismatch")

    formal_tmin, formal_tmax = map(float, plan["primary_formal_window_s"])
    formal_time_mask = (
        (times >= formal_tmin - 1e-12)
        & (times <= formal_tmax + 1e-12)
    )
    formal_time_indices = np.where(formal_time_mask)[0]
    formal_times = times[formal_time_mask]

    if len(formal_times) == 0:
        raise RuntimeError(f"{participant}: no RSA bins in formal window")

    include_response_event_mismatch = bool(
        plan["rsa_controls"]["include_response_event_mismatch"]
    )

    n_frequencies = len(frequencies)
    n_times = len(formal_times)
    band_names = list(CANONICAL_BANDS)

    # Attribute axis order is pain, money.
    frequency_betas = np.full(
        (2, n_frequencies, n_times),
        np.nan,
        dtype=np.float32,
    )
    band_betas = np.full(
        (2, len(band_names), n_times),
        np.nan,
        dtype=np.float32,
    )
    trial_counts = np.zeros(n_times, dtype=int)

    for local_time_index, global_time_index in enumerate(formal_time_indices):
        response_time_s = float(times[global_time_index])
        trial_mask = cue_has_occurred_mask(metadata, response_time_s)
        current_metadata = metadata.loc[trial_mask].reset_index(drop=True)
        trial_counts[local_time_index] = int(len(current_metadata))

        current_power = np.asarray(
            power[trial_mask, :, :, global_time_index],
            dtype=float,
        )

        for frequency_index in range(n_frequencies):
            features = current_power[:, :, frequency_index]
            coefficients = rsa_coefficients(
                features=features,
                metadata=current_metadata,
                include_response_event_mismatch=include_response_event_mismatch,
            )
            frequency_betas[0, frequency_index, local_time_index] = coefficients["pain"]
            frequency_betas[1, frequency_index, local_time_index] = coefficients["money"]

        for band_index, band_name in enumerate(band_names):
            features = band_features(
                current_power=current_power,
                frequencies=frequencies,
                band_name=band_name,
            )
            coefficients = rsa_coefficients(
                features=features,
                metadata=current_metadata,
                include_response_event_mismatch=include_response_event_mismatch,
            )
            band_betas[0, band_index, local_time_index] = coefficients["pain"]
            band_betas[1, band_index, local_time_index] = coefficients["money"]

    if not np.isfinite(frequency_betas).all():
        raise RuntimeError(f"{participant}: non-finite frequency RSA coefficients")
    if not np.isfinite(band_betas).all():
        raise RuntimeError(f"{participant}: non-finite band RSA coefficients")

    np.savez_compressed(
        output_path,
        times=formal_times,
        freqs=frequencies,
        band_names=np.asarray(band_names),
        attribute_names=np.asarray(ATTRIBUTES),
        frequency_betas=frequency_betas,
        band_betas=band_betas,
        trial_counts=trial_counts,
        include_response_event_mismatch=np.asarray(include_response_event_mismatch),
    )

    print(
        f"    {participant}: RSA saved; current-trial counts "
        f"{trial_counts.min()}..{trial_counts.max()}",
        flush=True,
    )


def run_group_analysis(plan: dict, n_permutations: int, n_jobs: int) -> None:
    participants = plan["primary_participants"]
    participant_results = [
        np.load(participant_score_path(participant), allow_pickle=False)
        for participant in participants
    ]

    times = participant_results[0]["times"].astype(float)
    frequencies = participant_results[0]["freqs"].astype(float)
    band_names = participant_results[0]["band_names"].astype(str).tolist()

    for participant, result in zip(participants, participant_results):
        if not np.allclose(result["times"].astype(float), times, atol=1e-12):
            raise RuntimeError(f"{participant}: RSA time grid differs")
        if not np.array_equal(result["freqs"].astype(float), frequencies):
            raise RuntimeError(f"{participant}: RSA frequency grid differs")

    frequency_betas = np.stack(
        [result["frequency_betas"] for result in participant_results],
        axis=0,
    )
    band_betas = np.stack(
        [result["band_betas"] for result in participant_results],
        axis=0,
    )

    np.savez_compressed(
        RSA_OUTPUT_ROOT / "group_rsa_arrays.npz",
        participants=np.asarray(participants),
        times=times,
        freqs=frequencies,
        band_names=np.asarray(band_names),
        attribute_names=np.asarray(ATTRIBUTES),
        frequency_betas=frequency_betas,
        band_betas=band_betas,
    )

    surface_omnibus, surface_clusters = summarize_frequency_time_family(
        group_values=frequency_betas,
        family_name="multiple_regression_rsa_unique_geometry",
        frequencies=frequencies,
        times=times,
        n_permutations=n_permutations,
        n_jobs=n_jobs,
        fisher_transform_correlations=False,
    )
    band_omnibus, band_clusters = summarize_band_family(
        group_values=band_betas,
        family_name="multiple_regression_rsa_unique_geometry",
        times=times,
        band_names=band_names,
        n_permutations=n_permutations,
        n_jobs=n_jobs,
        fisher_transform_correlations=False,
    )

    surface_omnibus.to_csv(RSA_OUTPUT_ROOT / "surface_omnibus_holm.csv", index=False)
    surface_clusters.to_csv(RSA_OUTPUT_ROOT / "surface_significant_clusters.csv", index=False)
    band_omnibus.to_csv(RSA_OUTPUT_ROOT / "band_omnibus_holm.csv", index=False)
    band_clusters.to_csv(RSA_OUTPUT_ROOT / "band_significant_clusters.csv", index=False)

    trial_count_rows = []
    for participant, result in zip(participants, participant_results):
        for time_point, trial_count in zip(times, result["trial_counts"]):
            trial_count_rows.append({
                "participant": participant,
                "response_time_s": float(time_point),
                "n_trials": int(trial_count),
            })
    pd.DataFrame(trial_count_rows).to_csv(
        RSA_OUTPUT_ROOT / "trial_counts_by_participant_time.csv",
        index=False,
    )

    params = {
        "analysis": "response-locked multiple-regression RSA",
        "design_plan": plan,
        "neural_rdm": (
            "Euclidean distance across channel-wise standardized multichannel "
            "spectral-power features"
        ),
        "model_rdms": [
            "absolute pain-level distance",
            "absolute money-level distance",
            "absolute log-RT distance",
            "accepted/rejected mismatch",
        ] + (
            ["response-event mismatch"]
            if plan["rsa_controls"]["include_response_event_mismatch"]
            else []
        ),
        "rdm_transform": "rank transform then z-standardize; simultaneous OLS",
        "group_targets": [
            "unique pain geometry coefficient",
            "unique money geometry coefficient",
        ],
        "trial_availability": "event_rt_s + response_relative_time_s >= 0",
        "n_permutations": n_permutations,
    }
    (RSA_OUTPUT_ROOT / "params.json").write_text(json.dumps(params, indent=2))

    print("\nRESPONSE-LOCKED MULTIPLE-REGRESSION RSA FREQUENCY x TIME", flush=True)
    print(surface_omnibus.to_string(index=False), flush=True)
    if len(surface_clusters):
        print("\nSIGNIFICANT RSA SURFACE CLUSTERS", flush=True)
        print(surface_clusters.to_string(index=False), flush=True)

    print("\nRESPONSE-LOCKED RSA CANONICAL BANDS", flush=True)
    print(band_omnibus.to_string(index=False), flush=True)
    if len(band_clusters):
        print("\nSIGNIFICANT RSA BAND CLUSTERS", flush=True)
        print(band_clusters.to_string(index=False), flush=True)

    print(f"\nOutputs: {RSA_OUTPUT_ROOT}", flush=True)


def main() -> None:
    args = parse_args()
    RSA_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    PARTICIPANT_SCORE_DIR.mkdir(parents=True, exist_ok=True)

    plan = load_design_plan()
    participants = plan["primary_participants"]

    print("=" * 78, flush=True)
    print("PAINREWARD RESPONSE-LOCKED MULTIPLE-REGRESSION RSA", flush=True)
    print(f"Formal response window: {plan['primary_formal_window_s']}", flush=True)
    print(f"Fixed participant sample: {len(participants)}", flush=True)
    print(
        "Controls: other attribute + logRT distance + choice mismatch"
        + (
            " + response-event mismatch"
            if plan["rsa_controls"]["include_response_event_mismatch"]
            else ""
        ),
        flush=True,
    )
    print(f"Permutations: {args.n_permutations}", flush=True)
    print("=" * 78, flush=True)

    if not args.group_only:
        for participant_index, participant in enumerate(participants, start=1):
            print(
                f"\n[{participant_index}/{len(participants)}] {participant}",
                flush=True,
            )
            process_one_participant(
                participant=participant,
                plan=plan,
                overwrite_scores=args.overwrite_scores,
            )

    missing = [
        participant
        for participant in participants
        if not participant_score_path(participant).exists()
    ]
    if missing:
        raise RuntimeError(
            "Missing RSA participant files: " + ", ".join(missing)
        )

    run_group_analysis(
        plan=plan,
        n_permutations=args.n_permutations,
        n_jobs=args.n_jobs,
    )

    print("\n" + "=" * 78, flush=True)
    print("RESPONSE-LOCKED RSA COMPLETE", flush=True)
    print("=" * 78, flush=True)


if __name__ == "__main__":
    main()
