#!/usr/bin/env python3
"""
05o_eeg_voltage_transfer_passive_to_decision.py
================================================

Same-time passive -> decision SIGNED-VOLTAGE transfer.

Scientific question
-------------------
Does the multichannel signed-voltage pattern that represents pain or money
magnitude during isolated passive cue viewing reappear during active choice?

This analysis is deliberately matched across phases:

    - exact good passive trials from the prepared passive TFR metadata
    - exact good decision trials from the prepared decision TFR metadata
    - cleaned broadband EEG
    - EEG channels only
    - voltage baseline copied from TFR preparation (normally -0.7..-0.5 s)
    - same saved time range (normally -0.8..+1.5 s)
    - same 20-ms bins
    - same channel order required
    - StandardScaler -> Ridge(alpha=1)
    - train on passive, freeze, test on independent decision phase
    - Pearson r between transferred prediction and true decision level

No temporal-generalisation search is performed: passive time t is applied only
to decision time t.

RT sensitivity
--------------
The passive voltage decoder remains frozen.

For each time bin, a decision-block-cross-fitted partial correlation controls:

    log(RT)
    log(RT)^2
    whether the response has occurred by the end of the current 20-ms bin

RT adjustment is a sensitivity analysis, because RT is downstream of the
decision process.

Group inference
---------------
0.00-1.10 s, one-sided positive temporal cluster tests for transfer.
Pain and money are tested separately and Holm-corrected across the two
attribute-level omnibus p-values within each family.

Outputs
-------
derivatives/statistics/voltage_transfer_passive_to_decision/
    participant_scores/
    group_score_arrays.npz
    domain_omnibus_holm.csv
    significant_clusters.csv
    group_timecourses.csv
    participant_trial_counts.csv
    params.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import mne
import numpy as np
import pandas as pd
from mne.stats import permutation_cluster_1samp_test
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# =============================================================================
# Configuration
# =============================================================================

mne.set_log_level("ERROR")

BASE_PATH = Path(os.getenv("basepath", Path(__file__).parent.parent.parent))
SOURCE_BIDS = Path(os.getenv("SOURCE_BIDS", BASE_PATH))
PARTICIPANTS_TSV = BASE_PATH / "participants.tsv"

OUTPUT_ROOT = (
    BASE_PATH
    / "derivatives"
    / "statistics"
    / "voltage_transfer_passive_to_decision"
)
PARTICIPANT_SCORE_DIR = OUTPUT_ROOT / "participant_scores"

ATTRIBUTES = {"pain": "p", "money": "m"}

RIDGE_ALPHA = 1.0
RANDOM_STATE = 23
CLUSTER_FORMING_ALPHA = 0.05
INFERENCE_TMIN = 0.0
INFERENCE_TMAX = 1.10

CHANNELS_TO_DROP = ["HEOGL", "HEOGR", "VEOGL", "STI 014", "Status"]


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Same-time passive-to-decision voltage transfer."
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


# =============================================================================
# Paths
# =============================================================================

def participant_score_path(participant: str) -> Path:
    return (
        PARTICIPANT_SCORE_DIR
        / f"{participant}_passive_to_decision_voltage.npz"
    )


def phase_reference_paths(participant: str, phase: str) -> Dict[str, Path]:
    """
    Use total-TFR metadata/params only as a trial/time-grid reference.
    Voltage itself is always reconstructed from cleaned broadband raw.
    """

    if phase == "passive":
        directory = (
            BASE_PATH / "derivatives" / participant / "eeg" / "tfr"
        )
        stem = f"{participant}_passive_cue"
    elif phase == "decision":
        directory = (
            BASE_PATH / "derivatives" / participant / "eeg" / "tfr_decision"
        )
        stem = f"{participant}_decision_cue"
    else:
        raise ValueError(f"Unknown phase: {phase}")

    return {
        "times": directory / f"{stem}_times.npy",
        "metadata": directory / f"{stem}_metadata.csv",
        "channels": directory / f"{stem}_channels.json",
        "params": directory / f"{stem}_params.json",
    }


def cleaned_raw_path(participant: str, phase: str) -> Path:
    if phase not in {"passive", "decision"}:
        raise ValueError(f"Unknown phase: {phase}")

    return (
        SOURCE_BIDS
        / "derivatives"
        / participant
        / "eeg"
        / f"{participant}_{phase}_cleaned-raw.fif"
    )


def load_participant_list() -> List[str]:
    table = pd.read_csv(PARTICIPANTS_TSV, sep="\t")
    participants = sorted(
        table["participant_id"].dropna().astype(str).unique().tolist()
    )

    if not participants:
        raise RuntimeError("No participants found in participants.tsv")

    for participant in participants:
        missing = []
        for phase in ("passive", "decision"):
            for path in phase_reference_paths(participant, phase).values():
                if not path.exists():
                    missing.append(str(path))
            raw_path = cleaned_raw_path(participant, phase)
            if not raw_path.exists():
                missing.append(str(raw_path))

        if missing:
            raise FileNotFoundError(
                f"{participant}: missing voltage-transfer inputs:\n"
                + "\n".join(missing)
            )

    return participants


# =============================================================================
# Voltage reconstruction
# =============================================================================

def make_time_bins(
    sample_times: np.ndarray,
    start_s: float,
    end_s: float,
    bin_width_s: float,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Create the same non-overlapping bins used by the TFR preparation."""

    edges = np.arange(
        start_s,
        end_s + bin_width_s * 0.5,
        bin_width_s,
    )

    if edges[-1] < end_s - 1e-9:
        edges = np.append(edges, end_s)

    if abs(edges[-1] - end_s) > 1e-9:
        edges = np.append(edges[edges < end_s], end_s)

    centers = []
    masks = []

    for bin_index in range(len(edges) - 1):
        low = float(edges[bin_index])
        high = float(edges[bin_index + 1])

        if bin_index == len(edges) - 2:
            mask = np.where(
                (sample_times >= low - 1e-12)
                & (sample_times <= high + 1e-12)
            )[0]
        else:
            mask = np.where(
                (sample_times >= low - 1e-12)
                & (sample_times < high - 1e-12)
            )[0]

        if not len(mask):
            raise RuntimeError(
                f"No voltage samples in {low:.3f}..{high:.3f} s"
            )

        centers.append((low + high) / 2.0)
        masks.append(mask)

    return np.asarray(centers, dtype=float), masks


def load_phase_voltage(participant: str, phase: str):
    """
    Reconstruct signed-voltage features using the exact TFR-approved trial set.
    """

    reference_paths = phase_reference_paths(participant, phase)

    metadata = pd.read_csv(reference_paths["metadata"])
    reference_times = np.load(reference_paths["times"]).astype(float)

    with open(reference_paths["channels"], "r", encoding="utf-8") as file_handle:
        reference_channels = json.load(file_handle)

    with open(reference_paths["params"], "r", encoding="utf-8") as file_handle:
        params = json.load(file_handle)

    required_metadata = ["sample"]
    missing_metadata = [
        column for column in required_metadata if column not in metadata.columns
    ]
    if missing_metadata:
        raise KeyError(
            f"{participant}/{phase}: metadata missing {missing_metadata}"
        )

    required_params = [
        "epoch_resample_hz",
        "baseline_s",
        "saved_time_range_s",
        "time_bin_s",
    ]
    missing_params = [
        key for key in required_params if key not in params
    ]
    if missing_params:
        raise KeyError(
            f"{participant}/{phase}: params missing {missing_params}"
        )

    raw = mne.io.read_raw_fif(
        str(cleaned_raw_path(participant, phase)),
        preload=True,
        verbose="ERROR",
    )

    channels_to_drop = [
        channel for channel in CHANNELS_TO_DROP if channel in raw.ch_names
    ]
    if channels_to_drop:
        raw.drop_channels(channels_to_drop)

    raw.pick("eeg", exclude=[])

    if list(raw.ch_names) != list(reference_channels):
        raise RuntimeError(
            f"{participant}/{phase}: cleaned-raw channel order differs from TFR"
        )

    event_samples = pd.to_numeric(
        metadata["sample"],
        errors="raise",
    ).to_numpy(int)

    events = np.column_stack(
        [
            event_samples,
            np.zeros(len(metadata), dtype=int),
            np.ones(len(metadata), dtype=int),
        ]
    )

    epoch_start_s, epoch_end_s = map(
        float,
        params["saved_time_range_s"],
    )

    epochs = mne.Epochs(
        raw,
        events=events,
        event_id={"cue": 1},
        tmin=epoch_start_s,
        tmax=epoch_end_s,
        baseline=None,
        metadata=metadata.copy(),
        preload=True,
        reject=None,
        reject_by_annotation=True,
        verbose="ERROR",
    )

    if len(epochs) != len(metadata):
        raise RuntimeError(
            f"{participant}/{phase}: retained {len(epochs)} of "
            f"{len(metadata)} already-approved trials"
        )

    epochs.resample(
        float(params["epoch_resample_hz"]),
        npad="auto",
        verbose="ERROR",
    )

    voltage = epochs.get_data(copy=True).astype(np.float64)
    sample_times = epochs.times.astype(float)

    baseline_start_s, baseline_end_s = map(
        float,
        params["baseline_s"],
    )
    baseline_mask = (
        (sample_times >= baseline_start_s - 1e-12)
        & (sample_times <= baseline_end_s + 1e-12)
    )

    if not baseline_mask.any():
        raise RuntimeError(
            f"{participant}/{phase}: no voltage samples in baseline"
        )

    # Preserve signed voltage; only subtract the pre-cue channel-wise baseline.
    voltage -= voltage[..., baseline_mask].mean(axis=-1, keepdims=True)

    time_bin_s = float(params["time_bin_s"])
    bin_centers, bin_masks = make_time_bins(
        sample_times=sample_times,
        start_s=epoch_start_s,
        end_s=epoch_end_s,
        bin_width_s=time_bin_s,
    )

    if not np.allclose(bin_centers, reference_times):
        raise RuntimeError(
            f"{participant}/{phase}: voltage/TFR time bins differ"
        )

    binned_voltage = np.stack(
        [voltage[..., mask].mean(axis=-1) for mask in bin_masks],
        axis=-1,
    ).astype(np.float32)

    return (
        binned_voltage,
        bin_centers,
        epochs.metadata.reset_index(drop=True).copy(),
        list(epochs.ch_names),
        time_bin_s,
    )


# =============================================================================
# Transfer scoring
# =============================================================================

def make_ridge_decoder():
    return make_pipeline(
        StandardScaler(),
        Ridge(alpha=RIDGE_ALPHA),
    )


def safe_correlation(true_values: np.ndarray, predictions: np.ndarray) -> float:
    true_values = np.asarray(true_values, dtype=float)
    predictions = np.asarray(predictions, dtype=float)

    if not np.isfinite(true_values).all() or not np.isfinite(predictions).all():
        raise RuntimeError("Non-finite transfer values")

    if np.std(true_values) == 0 or np.std(predictions) == 0:
        return 0.0

    return float(np.corrcoef(true_values, predictions)[0, 1])


def nuisance_matrix(
    reaction_time: np.ndarray,
    response_has_occurred: np.ndarray,
) -> np.ndarray:
    log_rt = np.log(np.asarray(reaction_time, dtype=float))
    return np.column_stack(
        [
            log_rt,
            log_rt**2,
            np.asarray(response_has_occurred, dtype=float),
        ]
    )


def standardize_nuisance_from_training_block(
    train_nuisance: np.ndarray,
    test_nuisance: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    mean = train_nuisance.mean(axis=0)
    sd = train_nuisance.std(axis=0)
    varying = sd > 1e-12

    if not np.any(varying):
        return (
            np.ones((len(train_nuisance), 1)),
            np.ones((len(test_nuisance), 1)),
        )

    train_standardized = (train_nuisance[:, varying] - mean[varying]) / sd[varying]
    test_standardized = (test_nuisance[:, varying] - mean[varying]) / sd[varying]

    return (
        np.column_stack([np.ones(len(train_standardized)), train_standardized]),
        np.column_stack([np.ones(len(test_standardized)), test_standardized]),
    )


def crossfit_rt_adjusted_transfer_correlation(
    true_values: np.ndarray,
    transferred_predictions: np.ndarray,
    decision_blocks: np.ndarray,
    reaction_time: np.ndarray,
    response_has_occurred: np.ndarray,
) -> float:
    """
    Block-cross-fitted partial correlation. The passive decoder stays frozen.
    """

    nuisance = nuisance_matrix(
        reaction_time,
        response_has_occurred,
    )

    target_residuals = np.full(len(true_values), np.nan, dtype=float)
    prediction_residuals = np.full(len(true_values), np.nan, dtype=float)

    for heldout_block in np.unique(decision_blocks):
        train = decision_blocks != heldout_block
        test = decision_blocks == heldout_block

        train_design, test_design = standardize_nuisance_from_training_block(
            nuisance[train],
            nuisance[test],
        )

        target_coefficients, *_ = np.linalg.lstsq(
            train_design,
            true_values[train],
            rcond=None,
        )
        prediction_coefficients, *_ = np.linalg.lstsq(
            train_design,
            transferred_predictions[train],
            rcond=None,
        )

        target_residuals[test] = (
            true_values[test] - test_design @ target_coefficients
        )
        prediction_residuals[test] = (
            transferred_predictions[test]
            - test_design @ prediction_coefficients
        )

    return safe_correlation(target_residuals, prediction_residuals)


def process_participant(participant: str, overwrite: bool) -> None:
    output_path = participant_score_path(participant)

    if output_path.exists() and not overwrite:
        print(f"[{participant}] existing voltage-transfer scores -> SKIP", flush=True)
        return

    (
        passive_voltage,
        passive_times,
        passive_metadata,
        passive_channels,
        passive_bin_width,
    ) = load_phase_voltage(participant, "passive")

    (
        decision_voltage,
        decision_times,
        decision_metadata,
        decision_channels,
        decision_bin_width,
    ) = load_phase_voltage(participant, "decision")

    if not np.allclose(passive_times, decision_times):
        raise RuntimeError(f"{participant}: passive/decision voltage times differ")

    if passive_channels != decision_channels:
        raise RuntimeError(f"{participant}: passive/decision channel order differs")

    if not np.isclose(passive_bin_width, decision_bin_width):
        raise RuntimeError(f"{participant}: passive/decision bin widths differ")

    required_passive = ["condition", "level_1_5"]
    required_decision = ["painlevel", "moneylevel", "block", "rt"]

    for column in required_passive:
        if column not in passive_metadata.columns:
            raise KeyError(f"{participant}: passive metadata missing {column}")

    for column in required_decision:
        if column not in decision_metadata.columns:
            raise KeyError(f"{participant}: decision metadata missing {column}")

    passive_condition = (
        passive_metadata["condition"].astype(str).str.lower().str.strip().to_numpy()
    )
    passive_level = pd.to_numeric(
        passive_metadata["level_1_5"],
        errors="raise",
    ).to_numpy(float)

    decision_targets = {
        "pain": pd.to_numeric(
            decision_metadata["painlevel"],
            errors="raise",
        ).to_numpy(float),
        "money": pd.to_numeric(
            decision_metadata["moneylevel"],
            errors="raise",
        ).to_numpy(float),
    }

    decision_blocks = pd.to_numeric(
        decision_metadata["block"],
        errors="raise",
    ).to_numpy(int)

    reaction_time = pd.to_numeric(
        decision_metadata["rt"],
        errors="raise",
    ).to_numpy(float)

    # For 20-ms signed-voltage bins, the response-overlap indicator simply asks
    # whether the response has occurred by the END of the current bin.
    response_has_occurred = np.stack(
        [
            reaction_time <= (time_point + decision_bin_width / 2.0)
            for time_point in decision_times
        ],
        axis=1,
    )

    raw_r = np.empty((len(ATTRIBUTES), len(passive_times)), dtype=np.float32)
    rt_adjusted_r = np.empty_like(raw_r)
    passive_counts = np.empty(len(ATTRIBUTES), dtype=int)

    for attribute_index, (attribute_name, passive_code) in enumerate(
        ATTRIBUTES.items()
    ):
        passive_mask = passive_condition == passive_code
        passive_target = passive_level[passive_mask]
        decision_target = decision_targets[attribute_name]

        if len(np.unique(passive_target)) < 5:
            raise RuntimeError(
                f"{participant}/{attribute_name}: passive levels are "
                f"{np.unique(passive_target)}"
            )

        passive_counts[attribute_index] = int(passive_mask.sum())

        print(
            f"    {participant}: transferring passive {attribute_name} "
            f"signed-voltage code",
            flush=True,
        )

        for time_index in range(len(passive_times)):
            decoder = make_ridge_decoder()
            decoder.fit(
                passive_voltage[passive_mask, :, time_index],
                passive_target,
            )

            transferred_predictions = decoder.predict(
                decision_voltage[:, :, time_index]
            )

            raw_r[attribute_index, time_index] = safe_correlation(
                decision_target,
                transferred_predictions,
            )

            rt_adjusted_r[
                attribute_index,
                time_index,
            ] = crossfit_rt_adjusted_transfer_correlation(
                true_values=decision_target,
                transferred_predictions=transferred_predictions,
                decision_blocks=decision_blocks,
                reaction_time=reaction_time,
                response_has_occurred=response_has_occurred[:, time_index],
            )

    np.savez_compressed(
        output_path,
        participant=np.asarray(participant),
        times=passive_times,
        channels=np.asarray(passive_channels),
        attribute_names=np.asarray(list(ATTRIBUTES)),
        raw_r=raw_r,
        rt_adjusted_r=rt_adjusted_r,
        passive_trial_counts=passive_counts,
        decision_trial_count=np.asarray(len(decision_metadata)),
        response_has_occurred_fraction=response_has_occurred.mean(axis=0).astype(
            np.float32
        ),
    )


# =============================================================================
# Group statistics
# =============================================================================

def fisher_z(values: np.ndarray) -> np.ndarray:
    return np.arctanh(np.clip(values, -0.999999, 0.999999))


def holm_correct(p_values: Sequence[float]) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=float)
    n_tests = len(p_values)

    order = np.argsort(p_values)
    ranked = p_values[order]
    adjusted_ranked = np.maximum.accumulate(
        (n_tests - np.arange(n_tests)) * ranked
    )
    adjusted_ranked = np.clip(adjusted_ranked, 0.0, 1.0)

    adjusted = np.empty(n_tests, dtype=float)
    adjusted[order] = adjusted_ranked
    return adjusted


def temporal_cluster_test(
    participant_scores: np.ndarray,
    times: np.ndarray,
    n_permutations: int,
    n_jobs: int,
):
    time_mask = (
        (times >= INFERENCE_TMIN - 1e-12)
        & (times <= INFERENCE_TMAX + 1e-12)
    )
    inference_times = times[time_mask]
    scores = fisher_z(participant_scores[:, time_mask])
    n_participants = scores.shape[0]

    threshold = float(
        stats.t.ppf(
            1.0 - CLUSTER_FORMING_ALPHA,
            df=n_participants - 1,
        )
    )

    t_values, clusters, cluster_p_values, _ = permutation_cluster_1samp_test(
        scores,
        threshold=threshold,
        n_permutations=n_permutations,
        tail=1,
        seed=RANDOM_STATE,
        n_jobs=n_jobs,
        out_type="indices",
        verbose=False,
    )

    cluster_indices = [
        np.asarray(cluster[0], dtype=int) for cluster in clusters
    ]

    return (
        t_values,
        cluster_indices,
        np.asarray(cluster_p_values, dtype=float),
        inference_times,
    )


def summarize_family(
    group_scores: np.ndarray,
    family_name: str,
    times: np.ndarray,
    n_permutations: int,
    n_jobs: int,
):
    stored_results = {}
    p_values = []

    for attribute_index, attribute_name in enumerate(ATTRIBUTES):
        result = temporal_cluster_test(
            participant_scores=group_scores[:, attribute_index],
            times=times,
            n_permutations=n_permutations,
            n_jobs=n_jobs,
        )
        minimum_p = float(result[2].min()) if len(result[2]) else 1.0
        p_values.append(minimum_p)
        stored_results[attribute_name] = result

    adjusted_p_values = holm_correct(p_values)

    omnibus_rows = []
    cluster_rows = []

    for attribute_index, attribute_name in enumerate(ATTRIBUTES):
        adjusted_p = float(adjusted_p_values[attribute_index])

        omnibus_rows.append(
            {
                "family": family_name,
                "attribute": attribute_name,
                "omnibus_p_within_time": p_values[attribute_index],
                "omnibus_p_holm_across_pain_money": adjusted_p,
                "domain_significant_holm_0p05": bool(adjusted_p < 0.05),
            }
        )

        if adjusted_p >= 0.05:
            continue

        t_values, clusters, cluster_p_values, inference_times = stored_results[
            attribute_name
        ]

        for cluster_id, (indices, cluster_p) in enumerate(
            zip(clusters, cluster_p_values)
        ):
            if cluster_p >= 0.05:
                continue

            cluster_t_values = t_values[indices]
            peak_position = int(np.argmax(cluster_t_values))

            cluster_rows.append(
                {
                    "family": family_name,
                    "attribute": attribute_name,
                    "cluster_id": cluster_id,
                    "cluster_p_within_time": float(cluster_p),
                    "time_start_s": float(inference_times[indices].min()),
                    "time_end_s": float(inference_times[indices].max()),
                    "peak_time_s": float(
                        inference_times[indices[peak_position]]
                    ),
                    "peak_t": float(cluster_t_values[peak_position]),
                    "omnibus_p_holm_across_pain_money": adjusted_p,
                }
            )

    return pd.DataFrame(omnibus_rows), pd.DataFrame(cluster_rows)


def run_group_analysis(
    participants: Sequence[str],
    n_permutations: int,
    n_jobs: int,
) -> None:
    participant_results = [
        np.load(participant_score_path(participant), allow_pickle=False)
        for participant in participants
    ]

    times = participant_results[0]["times"].astype(float)
    channels = participant_results[0]["channels"].astype(str)

    for participant, result in zip(participants, participant_results):
        if not np.allclose(result["times"].astype(float), times):
            raise RuntimeError(f"{participant}: voltage-transfer time mismatch")
        if not np.array_equal(result["channels"].astype(str), channels):
            raise RuntimeError(f"{participant}: voltage-transfer channel mismatch")

    raw_scores = np.stack(
        [result["raw_r"] for result in participant_results],
        axis=0,
    )
    rt_adjusted_scores = np.stack(
        [result["rt_adjusted_r"] for result in participant_results],
        axis=0,
    )

    np.savez_compressed(
        OUTPUT_ROOT / "group_score_arrays.npz",
        subjects=np.asarray(participants),
        times=times,
        channels=channels,
        attribute_names=np.asarray(list(ATTRIBUTES)),
        raw_r=raw_scores,
        rt_adjusted_r=rt_adjusted_scores,
    )

    omnibus_tables = []
    cluster_tables = []
    timecourse_rows = []

    for family_name, family_scores in [
        ("raw", raw_scores),
        ("rt_adjusted", rt_adjusted_scores),
    ]:
        omnibus, clusters = summarize_family(
            group_scores=family_scores,
            family_name=family_name,
            times=times,
            n_permutations=n_permutations,
            n_jobs=n_jobs,
        )
        omnibus_tables.append(omnibus)
        cluster_tables.append(clusters)

        mean_scores = family_scores.mean(axis=0)
        sem_scores = stats.sem(family_scores, axis=0)

        for attribute_index, attribute_name in enumerate(ATTRIBUTES):
            for time_index, time_value in enumerate(times):
                timecourse_rows.append(
                    {
                        "family": family_name,
                        "attribute": attribute_name,
                        "time_s": float(time_value),
                        "mean_transfer_r": float(
                            mean_scores[attribute_index, time_index]
                        ),
                        "sem_transfer_r": float(
                            sem_scores[attribute_index, time_index]
                        ),
                        "chance": 0.0,
                    }
                )

    omnibus_results = pd.concat(omnibus_tables, ignore_index=True)
    cluster_results = (
        pd.concat(cluster_tables, ignore_index=True)
        if any(len(table) for table in cluster_tables)
        else pd.DataFrame()
    )

    omnibus_results.to_csv(
        OUTPUT_ROOT / "domain_omnibus_holm.csv",
        index=False,
    )
    cluster_results.to_csv(
        OUTPUT_ROOT / "significant_clusters.csv",
        index=False,
    )
    pd.DataFrame(timecourse_rows).to_csv(
        OUTPUT_ROOT / "group_timecourses.csv",
        index=False,
    )

    trial_count_rows = []
    for participant, result in zip(participants, participant_results):
        passive_counts = result["passive_trial_counts"]
        for attribute_index, attribute_name in enumerate(ATTRIBUTES):
            trial_count_rows.append(
                {
                    "participant": participant,
                    "attribute": attribute_name,
                    "n_passive_training_trials": int(
                        passive_counts[attribute_index]
                    ),
                    "n_decision_test_trials": int(
                        result["decision_trial_count"]
                    ),
                }
            )

    pd.DataFrame(trial_count_rows).to_csv(
        OUTPUT_ROOT / "participant_trial_counts.csv",
        index=False,
    )

    params = {
        "question": (
            "same-time reuse of passive signed-voltage magnitude codes during "
            "active decision making"
        ),
        "signal": "baseline-corrected signed multichannel EEG voltage",
        "trial_sets": (
            "exact passive and decision good-trial sets from total-TFR metadata"
        ),
        "decoder": "StandardScaler -> Ridge(alpha=1), fit on all passive trials",
        "decision_test_set": "independent decision phase",
        "score": "Pearson r between transferred prediction and true decision level",
        "rt_sensitivity": (
            "decision-block-cross-fitted partial correlation controlling "
            "log(RT), log(RT)^2, and response-has-occurred-by-bin-end; passive "
            "decoder itself is unchanged"
        ),
        "formal_inference_s": [INFERENCE_TMIN, INFERENCE_TMAX],
        "inference": (
            "one-sided positive temporal cluster permutation separately pain/money; "
            "Holm across pain/money within family"
        ),
        "temporal_generalization": "not performed",
        "n_permutations": n_permutations,
        "rt_note": (
            "RT adjustment is sensitivity only because RT is downstream of the "
            "decision process."
        ),
    }
    (OUTPUT_ROOT / "params.json").write_text(
        json.dumps(params, indent=2)
    )

    print("\n" + "=" * 82, flush=True)
    print("PASSIVE -> DECISION VOLTAGE TRANSFER: GROUP RESULTS", flush=True)
    print("=" * 82, flush=True)

    for family_name in ("raw", "rt_adjusted"):
        print(f"\n{family_name.upper()}", flush=True)
        print(
            omnibus_results[
                omnibus_results["family"] == family_name
            ].to_string(index=False),
            flush=True,
        )

        if len(cluster_results):
            family_clusters = cluster_results[
                cluster_results["family"] == family_name
            ]
        else:
            family_clusters = cluster_results

        print("\nSignificant clusters:", flush=True)
        print(
            family_clusters.to_string(index=False)
            if len(family_clusters)
            else "None",
            flush=True,
        )

    print(f"\nOutputs: {OUTPUT_ROOT}", flush=True)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = parse_args()

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    PARTICIPANT_SCORE_DIR.mkdir(parents=True, exist_ok=True)

    participants = load_participant_list()

    print("=" * 82, flush=True)
    print("PASSIVE -> DECISION SAME-TIME SIGNED-VOLTAGE TRANSFER", flush=True)
    print(f"Participants: {len(participants)}", flush=True)
    print("No temporal-generalisation search.", flush=True)
    print("=" * 82, flush=True)

    if not args.group_only:
        for participant_index, participant in enumerate(participants, start=1):
            if (
                participant_score_path(participant).exists()
                and not args.overwrite_scores
            ):
                print(
                    f"[{participant_index}/{len(participants)}] "
                    f"{participant}: SKIP existing scores",
                    flush=True,
                )
                continue

            print(
                f"\n[{participant_index}/{len(participants)}] {participant}",
                flush=True,
            )
            process_participant(
                participant=participant,
                overwrite=args.overwrite_scores,
            )
            print(f"[{participant}] DONE", flush=True)

    print("\nRunning group-level voltage-transfer inference...", flush=True)
    run_group_analysis(
        participants=participants,
        n_permutations=args.n_permutations,
        n_jobs=args.n_jobs,
    )

    print("\nPASSIVE -> DECISION VOLTAGE TRANSFER COMPLETE", flush=True)


if __name__ == "__main__":
    main()
