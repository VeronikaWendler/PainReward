#!/usr/bin/env python3
"""
05n_eeg_tfr_transfer_passive_to_decision.py
===========================================

Same-time, same-frequency passive -> decision TFR transfer.

Scientific question
-------------------
Do the multichannel spectral patterns that encode magnitude during isolated
PASSIVE cue viewing reappear while the participant is actively deciding
between pain and money?

For each participant and attribute:

    passive pain trials  -> train pain Ridge decoder
    passive money trials -> train money Ridge decoder

The fitted passive decoder is then frozen and applied to DECISION trials at the
SAME frequency and SAME time bin.

This is deliberately NOT temporal-generalisation. We are testing direct reuse
of the passive representation, not searching for a later/earlier decision-time
match after seeing the native-decision results.

Primary signal
--------------
TOTAL POWER -> TOTAL POWER

Secondary mechanistic signal
----------------------------
EVOKED-REMOVED passive TFR -> EVOKED-REMOVED decision TFR

The evoked-removed analysis is useful but must be interpreted carefully:
passive evoked removal subtracts a domain-specific generic waveform because
pain and money are shown on separate passive trials, whereas every decision
trial contains both attributes and therefore uses one generic decision
waveform. It is an induced-enriched sensitivity analysis, not a perfect
identity transform across phases.

Decoder
-------
StandardScaler -> Ridge(alpha=1)

The model is fit using ALL good passive trials for the relevant attribute.
That is valid here because the decision phase is an entirely independent test
set. No decision magnitude labels are used to fit the neural decoder.

Participant score
-----------------
Pearson correlation between:
    transferred passive-decoder prediction on decision trials
and:
    the true decision pain or money level.

RT sensitivity
--------------
We do NOT alter the passive decoder for the RT analysis.

Instead, after applying the frozen passive decoder to decision trials, we
compute a leakage-safe block-cross-fitted partial correlation controlling for:

    log(RT)
    log(RT)^2
    response/wavelet overlap

For each held-out decision block, nuisance regressions are estimated only on
the other blocks. This preserves the passive neural feature geometry while
asking whether transferred predictions still track magnitude beyond simple
response-timing structure.

RT adjustment remains a sensitivity analysis because RT is downstream of the
decision process.

Group inference
---------------
Formal window: 0.00-1.10 s.

Frequency x time:
    one-sided cluster permutation for positive transfer,
    separately pain and money,
    Holm correction across pain/money within each predefined family.

Canonical bands:
    theta 4-7 Hz
    alpha 8-13 Hz
    beta 14-30 Hz
    low gamma 31-45 Hz

    one-sided temporal cluster permutation,
    Holm across four bands within each attribute/family.

Direct total-vs-evoked-removed comparison:
    Fisher-z(total transfer r) - Fisher-z(evoked-removed transfer r)
    two-sided frequency x time cluster permutation,
    Holm across pain/money.

Outputs
-------
derivatives/statistics/tfr_transfer_passive_to_decision/
    participant_scores/
    group_score_arrays.npz
    surface_omnibus_holm.csv
    surface_significant_clusters.csv
    band_omnibus_holm.csv
    band_significant_clusters.csv
    total_vs_evoked_removed_omnibus_holm.csv
    total_vs_evoked_removed_clusters.csv
    participant_trial_counts.csv
    params.json
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from mne.stats import combine_adjacency, permutation_cluster_1samp_test
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# =============================================================================
# Configuration
# =============================================================================

BASE_PATH = Path(os.getenv("basepath", Path(__file__).parent.parent.parent))
PARTICIPANTS_TSV = BASE_PATH / "participants.tsv"

OUTPUT_ROOT = (
    BASE_PATH
    / "derivatives"
    / "statistics"
    / "tfr_transfer_passive_to_decision"
)
PARTICIPANT_SCORE_DIR = OUTPUT_ROOT / "participant_scores"

ATTRIBUTES = {"pain": "p", "money": "m"}

CANONICAL_BANDS: Dict[str, Tuple[float, float]] = {
    "theta": (4.0, 7.0),
    "alpha": (8.0, 13.0),
    "beta": (14.0, 30.0),
    "low_gamma": (31.0, 45.0),
}

RIDGE_ALPHA = 1.0
RANDOM_STATE = 23
CLUSTER_FORMING_ALPHA = 0.05
INFERENCE_TMIN = 0.0
INFERENCE_TMAX = 1.10

# Same conservative Morlet half-support used in the native decision TFR decoder.
MORLET_HALF_SUPPORT_S = 5.0 * (0.5 / (2.0 * np.pi))


@dataclass(frozen=True)
class SignalDefinition:
    """Where to find one matched passive/decision TFR representation."""

    name: str
    passive_directory: str
    passive_power_suffix: str
    decision_directory: str
    decision_power_suffix: str


SIGNALS = [
    SignalDefinition(
        name="total",
        passive_directory="tfr",
        passive_power_suffix="power.npy",
        decision_directory="tfr_decision",
        decision_power_suffix="power.npy",
    ),
    SignalDefinition(
        name="evoked_removed",
        passive_directory="tfr_evoked_removed",
        passive_power_suffix="power_evoked_removed.npy",
        decision_directory="tfr_decision_evoked_removed",
        decision_power_suffix="power_evoked_removed.npy",
    ),
]


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Same-time passive-to-decision spectral transfer."
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
# Paths and data loading
# =============================================================================

def participant_score_path(participant: str) -> Path:
    return PARTICIPANT_SCORE_DIR / f"{participant}_passive_to_decision_tfr.npz"


def phase_tfr_paths(
    participant: str,
    phase: str,
    signal: SignalDefinition,
) -> Dict[str, Path]:
    """Return the TFR files for passive or decision phase."""

    if phase == "passive":
        directory = (
            BASE_PATH
            / "derivatives"
            / participant
            / "eeg"
            / signal.passive_directory
        )
        stem = f"{participant}_passive_cue"
        power_suffix = signal.passive_power_suffix
    elif phase == "decision":
        directory = (
            BASE_PATH
            / "derivatives"
            / participant
            / "eeg"
            / signal.decision_directory
        )
        stem = f"{participant}_decision_cue"
        power_suffix = signal.decision_power_suffix
    else:
        raise ValueError(f"Unknown phase: {phase}")

    return {
        "power": directory / f"{stem}_{power_suffix}",
        "times": directory / f"{stem}_times.npy",
        "freqs": directory / f"{stem}_freqs.npy",
        "metadata": directory / f"{stem}_metadata.csv",
        "channels": directory / f"{stem}_channels.json",
    }


def load_participant_list() -> List[str]:
    table = pd.read_csv(PARTICIPANTS_TSV, sep="\t")
    participants = sorted(
        table["participant_id"].dropna().astype(str).unique().tolist()
    )

    if not participants:
        raise RuntimeError("No participants found in participants.tsv")

    for participant in participants:
        missing = []
        for signal in SIGNALS:
            for phase in ("passive", "decision"):
                for path in phase_tfr_paths(participant, phase, signal).values():
                    if not path.exists():
                        missing.append(str(path))

        if missing:
            raise FileNotFoundError(
                f"{participant}: missing passive-to-decision transfer inputs:\n"
                + "\n".join(missing)
            )

    return participants


def load_tfr_dataset(
    participant: str,
    phase: str,
    signal: SignalDefinition,
):
    """Load one phase/signal TFR plus metadata and coordinates."""

    paths = phase_tfr_paths(participant, phase, signal)

    power = np.load(paths["power"], mmap_mode="r")
    times = np.load(paths["times"]).astype(float)
    freqs = np.load(paths["freqs"]).astype(float)
    metadata = pd.read_csv(paths["metadata"])

    with open(paths["channels"], "r", encoding="utf-8") as file_handle:
        channels = json.load(file_handle)

    if power.ndim != 4:
        raise RuntimeError(
            f"{participant}/{phase}/{signal.name}: expected 4D power, got {power.shape}"
        )

    if len(metadata) != power.shape[0]:
        raise RuntimeError(
            f"{participant}/{phase}/{signal.name}: metadata/power trial mismatch"
        )

    if len(channels) != power.shape[1]:
        raise RuntimeError(
            f"{participant}/{phase}/{signal.name}: channel-count mismatch"
        )

    if len(freqs) != power.shape[2] or len(times) != power.shape[3]:
        raise RuntimeError(
            f"{participant}/{phase}/{signal.name}: coordinate/power mismatch"
        )

    return power, times, freqs, metadata, channels


def validate_cross_phase_grids(
    participant: str,
    signal_name: str,
    passive_times: np.ndarray,
    decision_times: np.ndarray,
    passive_freqs: np.ndarray,
    decision_freqs: np.ndarray,
    passive_channels: Sequence[str],
    decision_channels: Sequence[str],
) -> None:
    """Fail fast unless same-time/same-frequency transfer is actually possible."""

    if not np.allclose(passive_times, decision_times):
        raise RuntimeError(
            f"{participant}/{signal_name}: passive and decision time grids differ"
        )

    if not np.array_equal(passive_freqs, decision_freqs):
        raise RuntimeError(
            f"{participant}/{signal_name}: passive and decision frequency grids differ"
        )

    if list(passive_channels) != list(decision_channels):
        raise RuntimeError(
            f"{participant}/{signal_name}: passive and decision channel order differs"
        )


# =============================================================================
# Decoder and scoring helpers
# =============================================================================

def make_ridge_decoder():
    """Fixed decoder used throughout the project."""

    return make_pipeline(
        StandardScaler(),
        Ridge(alpha=RIDGE_ALPHA),
    )


def safe_correlation(true_values: np.ndarray, predictions: np.ndarray) -> float:
    """Pearson r with explicit checks for degenerate vectors."""

    true_values = np.asarray(true_values, dtype=float)
    predictions = np.asarray(predictions, dtype=float)

    if not np.isfinite(true_values).all() or not np.isfinite(predictions).all():
        raise RuntimeError("Non-finite values supplied to correlation")

    if np.std(true_values) == 0 or np.std(predictions) == 0:
        return 0.0

    return float(np.corrcoef(true_values, predictions)[0, 1])


def nuisance_matrix(
    reaction_time: np.ndarray,
    response_overlap: np.ndarray,
) -> np.ndarray:
    """RT nuisance variables for the transfer sensitivity analysis."""

    log_rt = np.log(np.asarray(reaction_time, dtype=float))
    return np.column_stack(
        [
            log_rt,
            log_rt**2,
            np.asarray(response_overlap, dtype=float),
        ]
    )


def standardize_nuisance_from_training_block(
    train_nuisance: np.ndarray,
    test_nuisance: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Standardize nuisance columns using training blocks only."""

    mean = train_nuisance.mean(axis=0)
    sd = train_nuisance.std(axis=0)
    varying = sd > 1e-12

    if not np.any(varying):
        train_design = np.ones((len(train_nuisance), 1))
        test_design = np.ones((len(test_nuisance), 1))
        return train_design, test_design

    train_standardized = (train_nuisance[:, varying] - mean[varying]) / sd[varying]
    test_standardized = (test_nuisance[:, varying] - mean[varying]) / sd[varying]

    train_design = np.column_stack([np.ones(len(train_standardized)), train_standardized])
    test_design = np.column_stack([np.ones(len(test_standardized)), test_standardized])

    return train_design, test_design


def crossfit_rt_adjusted_transfer_correlation(
    true_values: np.ndarray,
    transferred_predictions: np.ndarray,
    decision_blocks: np.ndarray,
    reaction_time: np.ndarray,
    response_overlap: np.ndarray,
) -> float:
    """
    Leakage-safe block-cross-fitted partial correlation.

    For each held-out decision block:
        prediction ~ RT nuisances   fitted on other blocks
        target     ~ RT nuisances   fitted on other blocks

    Residuals are concatenated across held-out blocks and correlated.

    The passive neural decoder is never altered.
    """

    true_values = np.asarray(true_values, dtype=float)
    transferred_predictions = np.asarray(transferred_predictions, dtype=float)
    decision_blocks = np.asarray(decision_blocks, dtype=int)

    nuisance = nuisance_matrix(reaction_time, response_overlap)

    target_residuals = np.full(len(true_values), np.nan, dtype=float)
    prediction_residuals = np.full(len(true_values), np.nan, dtype=float)

    for heldout_block in np.unique(decision_blocks):
        train = decision_blocks != heldout_block
        test = decision_blocks == heldout_block

        if not train.any() or not test.any():
            raise RuntimeError(f"Invalid decision block split: {heldout_block}")

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

    if not np.isfinite(target_residuals).all():
        raise RuntimeError("Non-finite RT-adjusted target residuals")

    if not np.isfinite(prediction_residuals).all():
        raise RuntimeError("Non-finite RT-adjusted prediction residuals")

    return safe_correlation(target_residuals, prediction_residuals)


def transfer_one_time_point(
    passive_features: np.ndarray,
    passive_target: np.ndarray,
    decision_features: np.ndarray,
    decision_target: np.ndarray,
    decision_blocks: np.ndarray,
    reaction_time: np.ndarray,
    response_overlap: np.ndarray,
) -> Tuple[float, float]:
    """Fit on passive trials, freeze model, and score on decision trials."""

    decoder = make_ridge_decoder()
    decoder.fit(passive_features, passive_target)

    transferred_predictions = decoder.predict(decision_features)

    raw_r = safe_correlation(
        decision_target,
        transferred_predictions,
    )

    rt_adjusted_r = crossfit_rt_adjusted_transfer_correlation(
        true_values=decision_target,
        transferred_predictions=transferred_predictions,
        decision_blocks=decision_blocks,
        reaction_time=reaction_time,
        response_overlap=response_overlap,
    )

    return raw_r, rt_adjusted_r


def transfer_one_frequency(
    passive_power,
    decision_power,
    passive_trial_mask: np.ndarray,
    passive_target: np.ndarray,
    decision_target: np.ndarray,
    decision_blocks: np.ndarray,
    reaction_time: np.ndarray,
    response_overlap_by_time: np.ndarray,
    frequency_index: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Transfer all time bins at one frequency."""

    passive_frequency_data = np.asarray(
        passive_power[passive_trial_mask, :, frequency_index, :],
        dtype=np.float32,
    )
    decision_frequency_data = np.asarray(
        decision_power[:, :, frequency_index, :],
        dtype=np.float32,
    )

    n_times = passive_frequency_data.shape[-1]
    raw_scores = np.empty(n_times, dtype=np.float32)
    rt_adjusted_scores = np.empty(n_times, dtype=np.float32)

    for time_index in range(n_times):
        raw_r, rt_r = transfer_one_time_point(
            passive_features=passive_frequency_data[:, :, time_index],
            passive_target=passive_target,
            decision_features=decision_frequency_data[:, :, time_index],
            decision_target=decision_target,
            decision_blocks=decision_blocks,
            reaction_time=reaction_time,
            response_overlap=response_overlap_by_time[:, time_index],
        )
        raw_scores[time_index] = raw_r
        rt_adjusted_scores[time_index] = rt_r

    return raw_scores, rt_adjusted_scores


def compute_transfer_for_signal(
    participant: str,
    signal: SignalDefinition,
    n_jobs: int,
):
    """Compute pain and money transfer for one TFR signal representation."""

    (
        passive_power,
        passive_times,
        passive_freqs,
        passive_metadata,
        passive_channels,
    ) = load_tfr_dataset(participant, "passive", signal)

    (
        decision_power,
        decision_times,
        decision_freqs,
        decision_metadata,
        decision_channels,
    ) = load_tfr_dataset(participant, "decision", signal)

    validate_cross_phase_grids(
        participant=participant,
        signal_name=signal.name,
        passive_times=passive_times,
        decision_times=decision_times,
        passive_freqs=passive_freqs,
        decision_freqs=decision_freqs,
        passive_channels=passive_channels,
        decision_channels=decision_channels,
    )

    required_passive = ["condition", "level_1_5"]
    required_decision = ["painlevel", "moneylevel", "block", "rt"]

    missing_passive = [
        column for column in required_passive if column not in passive_metadata.columns
    ]
    missing_decision = [
        column for column in required_decision if column not in decision_metadata.columns
    ]

    if missing_passive:
        raise KeyError(
            f"{participant}/{signal.name}: passive metadata missing {missing_passive}"
        )
    if missing_decision:
        raise KeyError(
            f"{participant}/{signal.name}: decision metadata missing {missing_decision}"
        )

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

    if not np.isfinite(passive_level).all():
        raise RuntimeError(f"{participant}/{signal.name}: bad passive levels")
    if not np.isfinite(reaction_time).all():
        raise RuntimeError(f"{participant}/{signal.name}: bad decision RT")

    response_overlap_by_time = np.stack(
        [
            reaction_time <= (time_point + MORLET_HALF_SUPPORT_S)
            for time_point in decision_times
        ],
        axis=1,
    )

    n_attributes = len(ATTRIBUTES)
    n_frequencies = len(passive_freqs)
    n_times = len(passive_times)
    band_names = list(CANONICAL_BANDS)

    raw_frequency_r = np.empty(
        (n_attributes, n_frequencies, n_times),
        dtype=np.float32,
    )
    rt_adjusted_frequency_r = np.empty_like(raw_frequency_r)

    raw_band_r = np.empty(
        (n_attributes, len(band_names), n_times),
        dtype=np.float32,
    )
    rt_adjusted_band_r = np.empty_like(raw_band_r)

    passive_counts = np.empty(n_attributes, dtype=int)

    for attribute_index, (attribute_name, passive_code) in enumerate(
        ATTRIBUTES.items()
    ):
        passive_mask = passive_condition == passive_code
        passive_target = passive_level[passive_mask]
        decision_target = decision_targets[attribute_name]

        if len(np.unique(passive_target)) < 5:
            raise RuntimeError(
                f"{participant}/{signal.name}/{attribute_name}: "
                f"passive levels are {np.unique(passive_target)}"
            )

        passive_counts[attribute_index] = int(passive_mask.sum())

        print(
            f"    {participant}: {signal.name} {attribute_name} "
            f"same-time frequency transfer",
            flush=True,
        )

        frequency_results = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(transfer_one_frequency)(
                passive_power=passive_power,
                decision_power=decision_power,
                passive_trial_mask=passive_mask,
                passive_target=passive_target,
                decision_target=decision_target,
                decision_blocks=decision_blocks,
                reaction_time=reaction_time,
                response_overlap_by_time=response_overlap_by_time,
                frequency_index=frequency_index,
            )
            for frequency_index in range(n_frequencies)
        )

        raw_frequency_r[attribute_index] = np.stack(
            [result[0] for result in frequency_results],
            axis=0,
        )
        rt_adjusted_frequency_r[attribute_index] = np.stack(
            [result[1] for result in frequency_results],
            axis=0,
        )

        print(
            f"    {participant}: {signal.name} {attribute_name} "
            f"canonical-band transfer",
            flush=True,
        )

        for band_index, band_name in enumerate(band_names):
            low_hz, high_hz = CANONICAL_BANDS[band_name]
            frequency_mask = (
                (passive_freqs >= low_hz) & (passive_freqs <= high_hz)
            )

            passive_band = np.asarray(
                passive_power[passive_mask, :, frequency_mask, :],
                dtype=np.float32,
            ).mean(axis=2)
            decision_band = np.asarray(
                decision_power[:, :, frequency_mask, :],
                dtype=np.float32,
            ).mean(axis=2)

            for time_index in range(n_times):
                raw_r, rt_r = transfer_one_time_point(
                    passive_features=passive_band[:, :, time_index],
                    passive_target=passive_target,
                    decision_features=decision_band[:, :, time_index],
                    decision_target=decision_target,
                    decision_blocks=decision_blocks,
                    reaction_time=reaction_time,
                    response_overlap=response_overlap_by_time[:, time_index],
                )
                raw_band_r[attribute_index, band_index, time_index] = raw_r
                rt_adjusted_band_r[
                    attribute_index, band_index, time_index
                ] = rt_r

    return {
        "times": passive_times,
        "freqs": passive_freqs,
        "channels": np.asarray(passive_channels),
        "band_names": np.asarray(band_names),
        "raw_freq_r": raw_frequency_r,
        "rt_adjusted_freq_r": rt_adjusted_frequency_r,
        "raw_band_r": raw_band_r,
        "rt_adjusted_band_r": rt_adjusted_band_r,
        "passive_counts": passive_counts,
        "decision_count": np.asarray(len(decision_metadata)),
        "response_overlap_fraction": response_overlap_by_time.mean(axis=0).astype(
            np.float32
        ),
    }


# =============================================================================
# Participant processing
# =============================================================================

def process_participant(participant: str, n_jobs: int, overwrite: bool) -> None:
    output_path = participant_score_path(participant)

    if output_path.exists() and not overwrite:
        print(f"[{participant}] existing transfer score file -> SKIP", flush=True)
        return

    signal_results = {}

    for signal in SIGNALS:
        print(
            f"  {participant}: loading {signal.name} passive/decision TFRs",
            flush=True,
        )
        signal_results[signal.name] = compute_transfer_for_signal(
            participant=participant,
            signal=signal,
            n_jobs=n_jobs,
        )

    total = signal_results["total"]
    evoked_removed = signal_results["evoked_removed"]

    for key in ("times", "freqs", "channels", "band_names"):
        if key in ("times", "freqs"):
            matches = np.allclose(total[key], evoked_removed[key])
        else:
            matches = np.array_equal(total[key], evoked_removed[key])

        if not matches:
            raise RuntimeError(
                f"{participant}: total and evoked-removed transfer grids differ for {key}"
            )

    np.savez_compressed(
        output_path,
        participant=np.asarray(participant),
        times=total["times"],
        freqs=total["freqs"],
        channels=total["channels"],
        band_names=total["band_names"],
        attribute_names=np.asarray(list(ATTRIBUTES)),
        total_raw_freq_r=total["raw_freq_r"],
        total_rt_adjusted_freq_r=total["rt_adjusted_freq_r"],
        total_raw_band_r=total["raw_band_r"],
        total_rt_adjusted_band_r=total["rt_adjusted_band_r"],
        evoked_removed_raw_freq_r=evoked_removed["raw_freq_r"],
        evoked_removed_rt_adjusted_freq_r=evoked_removed["rt_adjusted_freq_r"],
        evoked_removed_raw_band_r=evoked_removed["raw_band_r"],
        evoked_removed_rt_adjusted_band_r=evoked_removed["rt_adjusted_band_r"],
        passive_trial_counts=total["passive_counts"],
        decision_trial_count=total["decision_count"],
        response_overlap_fraction=total["response_overlap_fraction"],
    )


# =============================================================================
# Group statistics
# =============================================================================

def fisher_z(correlation_values: np.ndarray) -> np.ndarray:
    return np.arctanh(np.clip(correlation_values, -0.999999, 0.999999))


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


def positive_cluster_threshold(n_participants: int) -> float:
    return float(
        stats.t.ppf(
            1.0 - CLUSTER_FORMING_ALPHA,
            df=n_participants - 1,
        )
    )


def two_sided_cluster_threshold(n_participants: int) -> float:
    return float(
        stats.t.ppf(
            1.0 - CLUSTER_FORMING_ALPHA / 2.0,
            df=n_participants - 1,
        )
    )


def frequency_time_cluster_test(
    participant_scores: np.ndarray,
    frequencies: np.ndarray,
    times: np.ndarray,
    n_permutations: int,
    n_jobs: int,
    two_sided: bool = False,
):
    time_mask = (
        (times >= INFERENCE_TMIN - 1e-12)
        & (times <= INFERENCE_TMAX + 1e-12)
    )
    inference_times = times[time_mask]

    scores = participant_scores[:, :, time_mask]
    n_participants, n_frequencies, n_times = scores.shape

    threshold = (
        two_sided_cluster_threshold(n_participants)
        if two_sided
        else positive_cluster_threshold(n_participants)
    )
    tail = 0 if two_sided else 1

    t_values, clusters, cluster_p_values, _ = permutation_cluster_1samp_test(
        scores.reshape(n_participants, n_frequencies * n_times),
        threshold=threshold,
        n_permutations=n_permutations,
        tail=tail,
        adjacency=combine_adjacency(n_frequencies, n_times),
        seed=RANDOM_STATE,
        n_jobs=n_jobs,
        out_type="indices",
        verbose=False,
    )

    flattened_clusters = [
        np.asarray(cluster[0], dtype=int) for cluster in clusters
    ]

    return (
        t_values.reshape(n_frequencies, n_times),
        flattened_clusters,
        np.asarray(cluster_p_values, dtype=float),
        inference_times,
    )


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
    scores = participant_scores[:, time_mask]
    n_participants = scores.shape[0]

    t_values, clusters, cluster_p_values, _ = permutation_cluster_1samp_test(
        scores,
        threshold=positive_cluster_threshold(n_participants),
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


def summarize_transfer_surface(
    group_scores: np.ndarray,
    family_name: str,
    frequencies: np.ndarray,
    times: np.ndarray,
    n_permutations: int,
    n_jobs: int,
):
    """
    group_scores:
        participant x attribute x frequency x time

    Input correlations are Fisher-z transformed before group inference.
    """

    transformed = fisher_z(group_scores)

    stored_results = {}
    p_values = []

    for attribute_index, attribute_name in enumerate(ATTRIBUTES):
        result = frequency_time_cluster_test(
            participant_scores=transformed[:, attribute_index],
            frequencies=frequencies,
            times=times,
            n_permutations=n_permutations,
            n_jobs=n_jobs,
            two_sided=False,
        )

        minimum_p = (
            float(result[2].min()) if len(result[2]) else 1.0
        )
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
                "omnibus_p_within_surface": p_values[attribute_index],
                "omnibus_p_holm_across_pain_money": adjusted_p,
                "domain_significant_holm_0p05": bool(adjusted_p < 0.05),
            }
        )

        if adjusted_p >= 0.05:
            continue

        t_values, clusters, cluster_p_values, inference_times = stored_results[
            attribute_name
        ]
        n_frequencies, n_times = t_values.shape

        for cluster_id, (flat_indices, cluster_p) in enumerate(
            zip(clusters, cluster_p_values)
        ):
            if cluster_p >= 0.05:
                continue

            frequency_indices, time_indices = np.unravel_index(
                flat_indices,
                (n_frequencies, n_times),
            )
            cluster_t_values = t_values.reshape(-1)[flat_indices]
            peak_position = int(np.argmax(cluster_t_values))
            peak_flat = int(flat_indices[peak_position])
            peak_frequency_index, peak_time_index = np.unravel_index(
                peak_flat,
                (n_frequencies, n_times),
            )

            cluster_rows.append(
                {
                    "family": family_name,
                    "attribute": attribute_name,
                    "cluster_id": cluster_id,
                    "cluster_p_within_surface": float(cluster_p),
                    "freq_min_hz": float(frequencies[frequency_indices].min()),
                    "freq_max_hz": float(frequencies[frequency_indices].max()),
                    "time_start_s": float(inference_times[time_indices].min()),
                    "time_end_s": float(inference_times[time_indices].max()),
                    "peak_freq_hz": float(frequencies[peak_frequency_index]),
                    "peak_time_s": float(inference_times[peak_time_index]),
                    "peak_t": float(cluster_t_values[peak_position]),
                    "omnibus_p_holm_across_pain_money": adjusted_p,
                }
            )

    return pd.DataFrame(omnibus_rows), pd.DataFrame(cluster_rows)


def summarize_transfer_bands(
    group_band_scores: np.ndarray,
    family_name: str,
    times: np.ndarray,
    band_names: Sequence[str],
    n_permutations: int,
    n_jobs: int,
):
    """
    group_band_scores:
        participant x attribute x band x time
    """

    transformed = fisher_z(group_band_scores)
    omnibus_rows = []
    cluster_rows = []

    for attribute_index, attribute_name in enumerate(ATTRIBUTES):
        stored_results = []
        band_p_values = []

        for band_index, band_name in enumerate(band_names):
            result = temporal_cluster_test(
                participant_scores=transformed[
                    :, attribute_index, band_index
                ],
                times=times,
                n_permutations=n_permutations,
                n_jobs=n_jobs,
            )
            minimum_p = (
                float(result[2].min()) if len(result[2]) else 1.0
            )
            band_p_values.append(minimum_p)
            stored_results.append((band_name, result))

        adjusted_p_values = holm_correct(band_p_values)

        for band_index, (band_name, result) in enumerate(stored_results):
            adjusted_p = float(adjusted_p_values[band_index])

            omnibus_rows.append(
                {
                    "family": family_name,
                    "attribute": attribute_name,
                    "band": band_name,
                    "band_omnibus_p_within_time": band_p_values[band_index],
                    "band_omnibus_p_holm_within_attribute": adjusted_p,
                    "band_significant_holm_0p05": bool(adjusted_p < 0.05),
                }
            )

            if adjusted_p >= 0.05:
                continue

            t_values, clusters, cluster_p_values, inference_times = result

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
                        "band": band_name,
                        "cluster_id": cluster_id,
                        "cluster_p_within_band_time": float(cluster_p),
                        "time_start_s": float(inference_times[indices].min()),
                        "time_end_s": float(inference_times[indices].max()),
                        "peak_time_s": float(
                            inference_times[indices[peak_position]]
                        ),
                        "peak_t": float(cluster_t_values[peak_position]),
                        "band_omnibus_p_holm_within_attribute": adjusted_p,
                    }
                )

    return pd.DataFrame(omnibus_rows), pd.DataFrame(cluster_rows)


def summarize_total_vs_evoked_removed(
    total_scores: np.ndarray,
    evoked_removed_scores: np.ndarray,
    family_name: str,
    frequencies: np.ndarray,
    times: np.ndarray,
    n_permutations: int,
    n_jobs: int,
):
    """
    Direct two-sided paired test:
        Fisher-z(total transfer r) - Fisher-z(evoked-removed transfer r)
    """

    differences = fisher_z(total_scores) - fisher_z(evoked_removed_scores)

    stored_results = {}
    p_values = []

    for attribute_index, attribute_name in enumerate(ATTRIBUTES):
        result = frequency_time_cluster_test(
            participant_scores=differences[:, attribute_index],
            frequencies=frequencies,
            times=times,
            n_permutations=n_permutations,
            n_jobs=n_jobs,
            two_sided=True,
        )
        minimum_p = (
            float(result[2].min()) if len(result[2]) else 1.0
        )
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
                "omnibus_p_within_surface": p_values[attribute_index],
                "omnibus_p_holm_across_pain_money": adjusted_p,
                "significant_holm_0p05": bool(adjusted_p < 0.05),
            }
        )

        t_values, clusters, cluster_p_values, inference_times = stored_results[
            attribute_name
        ]
        n_frequencies, n_times = t_values.shape

        for cluster_id, (flat_indices, cluster_p) in enumerate(
            zip(clusters, cluster_p_values)
        ):
            frequency_indices, time_indices = np.unravel_index(
                flat_indices,
                (n_frequencies, n_times),
            )
            cluster_t_values = t_values.reshape(-1)[flat_indices]
            peak_position = int(np.argmax(np.abs(cluster_t_values)))
            peak_flat = int(flat_indices[peak_position])
            peak_frequency_index, peak_time_index = np.unravel_index(
                peak_flat,
                (n_frequencies, n_times),
            )

            cluster_rows.append(
                {
                    "family": family_name,
                    "attribute": attribute_name,
                    "cluster_id": cluster_id,
                    "direction": (
                        "total_gt_evoked_removed"
                        if cluster_t_values.mean() > 0
                        else "evoked_removed_gt_total"
                    ),
                    "cluster_p_within_surface": float(cluster_p),
                    "omnibus_p_holm_across_pain_money": adjusted_p,
                    "cluster_significant_final": bool(
                        cluster_p < 0.05 and adjusted_p < 0.05
                    ),
                    "freq_min_hz": float(frequencies[frequency_indices].min()),
                    "freq_max_hz": float(frequencies[frequency_indices].max()),
                    "time_start_s": float(inference_times[time_indices].min()),
                    "time_end_s": float(inference_times[time_indices].max()),
                    "peak_freq_hz": float(frequencies[peak_frequency_index]),
                    "peak_time_s": float(inference_times[peak_time_index]),
                    "peak_t": float(cluster_t_values[peak_position]),
                }
            )

    return pd.DataFrame(omnibus_rows), pd.DataFrame(cluster_rows), differences


# =============================================================================
# Group aggregation
# =============================================================================

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
    freqs = participant_results[0]["freqs"].astype(float)
    channels = participant_results[0]["channels"].astype(str)
    band_names = participant_results[0]["band_names"].astype(str).tolist()

    for participant, result in zip(participants, participant_results):
        if not np.allclose(result["times"].astype(float), times):
            raise RuntimeError(f"{participant}: transfer time grid mismatch")
        if not np.array_equal(result["freqs"].astype(float), freqs):
            raise RuntimeError(f"{participant}: transfer frequency grid mismatch")
        if not np.array_equal(result["channels"].astype(str), channels):
            raise RuntimeError(f"{participant}: transfer channel order mismatch")

    array_keys = [
        "total_raw_freq_r",
        "total_rt_adjusted_freq_r",
        "total_raw_band_r",
        "total_rt_adjusted_band_r",
        "evoked_removed_raw_freq_r",
        "evoked_removed_rt_adjusted_freq_r",
        "evoked_removed_raw_band_r",
        "evoked_removed_rt_adjusted_band_r",
    ]

    group_arrays = {
        key: np.stack([result[key] for result in participant_results], axis=0)
        for key in array_keys
    }

    np.savez_compressed(
        OUTPUT_ROOT / "group_score_arrays.npz",
        subjects=np.asarray(participants),
        times=times,
        freqs=freqs,
        channels=channels,
        band_names=np.asarray(band_names),
        attribute_names=np.asarray(list(ATTRIBUTES)),
        **group_arrays,
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

    surface_omnibus_tables = []
    surface_cluster_tables = []
    band_omnibus_tables = []
    band_cluster_tables = []

    transfer_families = [
        (
            "total_raw",
            group_arrays["total_raw_freq_r"],
            group_arrays["total_raw_band_r"],
        ),
        (
            "total_rt_adjusted",
            group_arrays["total_rt_adjusted_freq_r"],
            group_arrays["total_rt_adjusted_band_r"],
        ),
        (
            "evoked_removed_raw",
            group_arrays["evoked_removed_raw_freq_r"],
            group_arrays["evoked_removed_raw_band_r"],
        ),
        (
            "evoked_removed_rt_adjusted",
            group_arrays["evoked_removed_rt_adjusted_freq_r"],
            group_arrays["evoked_removed_rt_adjusted_band_r"],
        ),
    ]

    for family_name, frequency_scores, band_scores in transfer_families:
        surface_omnibus, surface_clusters = summarize_transfer_surface(
            group_scores=frequency_scores,
            family_name=family_name,
            frequencies=freqs,
            times=times,
            n_permutations=n_permutations,
            n_jobs=n_jobs,
        )
        band_omnibus, band_clusters = summarize_transfer_bands(
            group_band_scores=band_scores,
            family_name=family_name,
            times=times,
            band_names=band_names,
            n_permutations=n_permutations,
            n_jobs=n_jobs,
        )

        surface_omnibus_tables.append(surface_omnibus)
        surface_cluster_tables.append(surface_clusters)
        band_omnibus_tables.append(band_omnibus)
        band_cluster_tables.append(band_clusters)

    surface_omnibus = pd.concat(surface_omnibus_tables, ignore_index=True)
    surface_clusters = (
        pd.concat(surface_cluster_tables, ignore_index=True)
        if any(len(table) for table in surface_cluster_tables)
        else pd.DataFrame()
    )
    band_omnibus = pd.concat(band_omnibus_tables, ignore_index=True)
    band_clusters = (
        pd.concat(band_cluster_tables, ignore_index=True)
        if any(len(table) for table in band_cluster_tables)
        else pd.DataFrame()
    )

    surface_omnibus.to_csv(
        OUTPUT_ROOT / "surface_omnibus_holm.csv",
        index=False,
    )
    surface_clusters.to_csv(
        OUTPUT_ROOT / "surface_significant_clusters.csv",
        index=False,
    )
    band_omnibus.to_csv(
        OUTPUT_ROOT / "band_omnibus_holm.csv",
        index=False,
    )
    band_clusters.to_csv(
        OUTPUT_ROOT / "band_significant_clusters.csv",
        index=False,
    )

    paired_omnibus_tables = []
    paired_cluster_tables = []
    paired_arrays = {}

    paired_families = [
        (
            "raw",
            group_arrays["total_raw_freq_r"],
            group_arrays["evoked_removed_raw_freq_r"],
        ),
        (
            "rt_adjusted",
            group_arrays["total_rt_adjusted_freq_r"],
            group_arrays["evoked_removed_rt_adjusted_freq_r"],
        ),
    ]

    for family_name, total_scores, evoked_removed_scores in paired_families:
        omnibus, clusters, differences = summarize_total_vs_evoked_removed(
            total_scores=total_scores,
            evoked_removed_scores=evoked_removed_scores,
            family_name=family_name,
            frequencies=freqs,
            times=times,
            n_permutations=n_permutations,
            n_jobs=n_jobs,
        )
        paired_omnibus_tables.append(omnibus)
        paired_cluster_tables.append(clusters)
        paired_arrays[f"{family_name}_fisherz_total_minus_evoked_removed"] = (
            differences
        )

    paired_omnibus = pd.concat(paired_omnibus_tables, ignore_index=True)
    paired_clusters = (
        pd.concat(paired_cluster_tables, ignore_index=True)
        if any(len(table) for table in paired_cluster_tables)
        else pd.DataFrame()
    )

    paired_omnibus.to_csv(
        OUTPUT_ROOT / "total_vs_evoked_removed_omnibus_holm.csv",
        index=False,
    )
    paired_clusters.to_csv(
        OUTPUT_ROOT / "total_vs_evoked_removed_clusters.csv",
        index=False,
    )
    np.savez_compressed(
        OUTPUT_ROOT / "total_vs_evoked_removed_difference_arrays.npz",
        subjects=np.asarray(participants),
        times=times,
        freqs=freqs,
        attribute_names=np.asarray(list(ATTRIBUTES)),
        **paired_arrays,
    )

    params = {
        "primary_question": (
            "same-time, same-frequency reuse of passive spectral magnitude "
            "representations during active decision making"
        ),
        "primary_signal": "total passive TFR -> total decision TFR",
        "secondary_signal": (
            "passive evoked-removed TFR -> decision evoked-removed TFR"
        ),
        "decoder": "StandardScaler -> Ridge(alpha=1), fit on all passive trials",
        "decision_test_set": "independent decision phase",
        "score": "Pearson r between transferred prediction and true decision level",
        "rt_sensitivity": (
            "decision-block-cross-fitted partial correlation controlling "
            "log(RT), log(RT)^2, and response/wavelet overlap; passive decoder "
            "itself is unchanged"
        ),
        "morlet_half_support_s": MORLET_HALF_SUPPORT_S,
        "formal_inference_s": [INFERENCE_TMIN, INFERENCE_TMAX],
        "surface_inference": (
            "one-sided positive frequency x time cluster permutation; "
            "Holm across pain/money within each predefined family"
        ),
        "band_inference": (
            "one-sided temporal cluster permutation; Holm across four canonical "
            "bands within attribute/family"
        ),
        "paired_total_vs_evoked_removed": (
            "two-sided Fisher-z transfer-score difference; Holm across pain/money"
        ),
        "temporal_generalization": "not performed",
        "n_permutations": n_permutations,
        "important_rt_note": (
            "RT adjustment is sensitivity only because RT is downstream of the "
            "decision process."
        ),
        "important_evoked_removed_note": (
            "passive and decision generic-evoked subtraction differ structurally "
            "because passive cues are domain-specific trials while decision trials "
            "contain both attributes."
        ),
    }
    (OUTPUT_ROOT / "params.json").write_text(
        json.dumps(params, indent=2)
    )

    print("\n" + "=" * 90, flush=True)
    print("PASSIVE -> DECISION TFR TRANSFER: GROUP RESULTS", flush=True)
    print("=" * 90, flush=True)

    for family_name, _, _ in transfer_families:
        print(f"\n{family_name.upper()} FREQUENCY x TIME", flush=True)
        print(
            surface_omnibus[
                surface_omnibus["family"] == family_name
            ].to_string(index=False),
            flush=True,
        )

        significant = (
            surface_clusters[
                surface_clusters["family"] == family_name
            ]
            if len(surface_clusters)
            else surface_clusters
        )
        if len(significant):
            print("\nSignificant surface clusters:", flush=True)
            print(significant.to_string(index=False), flush=True)

    print("\nDIRECT TOTAL VS EVOKED-REMOVED TRANSFER COMPARISON", flush=True)
    print(paired_omnibus.to_string(index=False), flush=True)

    if len(paired_clusters):
        significant_paired = paired_clusters[
            paired_clusters["cluster_significant_final"] == True
        ]
    else:
        significant_paired = paired_clusters

    print("\nSignificant paired-difference clusters:", flush=True)
    print(
        significant_paired.to_string(index=False)
        if len(significant_paired)
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

    print("=" * 90, flush=True)
    print("PASSIVE -> DECISION SAME-TIME TFR TRANSFER", flush=True)
    print(f"Participants: {len(participants)}", flush=True)
    print("Primary: total power -> total power", flush=True)
    print("Secondary: evoked-removed -> evoked-removed", flush=True)
    print("No temporal-generalisation search.", flush=True)
    print("=" * 90, flush=True)

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
                n_jobs=args.n_jobs,
                overwrite=args.overwrite_scores,
            )
            print(f"[{participant}] DONE", flush=True)

    print("\nRunning group-level transfer inference...", flush=True)
    run_group_analysis(
        participants=participants,
        n_permutations=args.n_permutations,
        n_jobs=args.n_jobs,
    )

    print("\nPASSIVE -> DECISION TFR TRANSFER COMPLETE", flush=True)


if __name__ == "__main__":
    main()
