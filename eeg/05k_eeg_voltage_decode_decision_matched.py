#!/usr/bin/env python3
"""
05k_eeg_voltage_decode_decision_matched.py
==========================================

Matched decision-phase VOLTAGE decoding for the PainReward project.

Why this analysis?
------------------
The existing decision-phase ERP/mass-univariate analysis and the newer
decision-phase TFR decoding use different analysis frameworks. To test whether
pain and money may be expressed in different electrophysiological formats, we
need a more apples-to-apples comparison.

This script therefore uses:

    - the SAME participants as the decision TFR analysis,
    - the SAME good trials as the decision TFR analysis,
    - the SAME 20-ms time bins,
    - the SAME leave-one-block-out cross-validation,
    - the SAME StandardScaler -> Ridge(alpha=1) decoder,
    - the SAME pooled out-of-fold Pearson-r score,
    - the SAME 0-1.10 s group inference window,
    - the SAME pain-vs-money Holm correction logic.

The only important change is the neural feature:

    decision TFR analysis: multichannel spectral power
    this analysis:         multichannel signed EEG voltage

RT sensitivity
--------------
Two analysis families are saved:

1. RAW voltage decoding (primary).
2. RT-RESIDUALIZED voltage decoding (sensitivity analysis).

For the RT sensitivity analysis, EEG channel features are residualized within
each training fold for:

    - log(RT)
    - log(RT)^2
    - whether the response has occurred by the end of the current 20-ms bin

The nuisance model is fitted ONLY on the training fold and then applied to the
held-out block. This avoids leakage.

RT adjustment is deliberately treated as a sensitivity analysis because RT is
downstream of the decision process and can itself contain genuine decision-
related neural variance.

Inputs
------
Prepared total-power decision TFR metadata:
    derivatives/<sub>/eeg/tfr_decision/
        <sub>_decision_cue_metadata.csv
        <sub>_decision_cue_times.npy
        <sub>_decision_cue_params.json

Cleaned decision EEG:
    SOURCE_BIDS/derivatives/<sub>/eeg/
        <sub>_decision_cleaned-raw.fif

Outputs
-------
derivatives/statistics/voltage_mvpa_decision_matched/
    participant_scores/
    group_score_arrays.npz
    domain_omnibus_holm.csv
    significant_clusters.csv
    group_timecourses.csv
    participant_rt_summary.csv
    response_overlap_by_time.csv
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
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# =============================================================================
# Project paths and constants
# =============================================================================

mne.set_log_level("ERROR")

BASE_PATH = Path(
    os.getenv("basepath", Path(__file__).parent.parent.parent)
)

SOURCE_BIDS = Path(
    os.getenv("SOURCE_BIDS", BASE_PATH)
)

PARTICIPANTS_TSV = BASE_PATH / "participants.tsv"

OUTPUT_ROOT = (
    BASE_PATH
    / "derivatives"
    / "statistics"
    / "voltage_mvpa_decision_matched"
)

PARTICIPANT_SCORE_DIR = OUTPUT_ROOT / "participant_scores"

ATTRIBUTES = ["pain", "money"]

RIDGE_ALPHA = 1.0
RANDOM_STATE = 23
CLUSTER_FORMING_ALPHA = 0.05

INFERENCE_TMIN = 0.0
INFERENCE_TMAX = 1.10

CHANNELS_TO_DROP = [
    "HEOGL",
    "HEOGR",
    "VEOGL",
    "STI 014",
    "Status",
]


# =============================================================================
# Command-line arguments
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Matched decision-phase voltage decoding."
    )

    parser.add_argument(
        "--n-permutations",
        type=int,
        default=5000,
        help="Number of group-level cluster permutations.",
    )

    parser.add_argument(
        "--n-jobs",
        type=int,
        default=max(
            1,
            int(os.getenv("SLURM_CPUS_PER_TASK", "1")),
        ),
        help="Parallel jobs used by MNE group-level permutation tests.",
    )

    parser.add_argument(
        "--overwrite-scores",
        action="store_true",
        help="Recompute participant scores even if they already exist.",
    )

    parser.add_argument(
        "--group-only",
        action="store_true",
        help="Skip participant decoding and recompute group statistics only.",
    )

    return parser.parse_args()


# =============================================================================
# Input/output paths
# =============================================================================

def decision_tfr_paths(participant: str) -> Dict[str, Path]:
    """Files from the already-completed total-power decision TFR preparation."""

    tfr_dir = (
        BASE_PATH
        / "derivatives"
        / participant
        / "eeg"
        / "tfr_decision"
    )

    stem = f"{participant}_decision_cue"

    return {
        "times": tfr_dir / f"{stem}_times.npy",
        "metadata": tfr_dir / f"{stem}_metadata.csv",
        "params": tfr_dir / f"{stem}_params.json",
    }


def decision_raw_path(participant: str) -> Path:
    """Cleaned broadband decision EEG."""

    return (
        SOURCE_BIDS
        / "derivatives"
        / participant
        / "eeg"
        / f"{participant}_decision_cleaned-raw.fif"
    )


def participant_score_path(participant: str) -> Path:
    """Saved participant-level voltage decoding scores."""

    return (
        PARTICIPANT_SCORE_DIR
        / f"{participant}_decision_voltage_mvpa.npz"
    )


def load_participant_list() -> List[str]:
    """Read participants.tsv and fail fast if required inputs are missing."""

    participant_table = pd.read_csv(
        PARTICIPANTS_TSV,
        sep="\t",
    )

    participants = sorted(
        participant_table["participant_id"]
        .dropna()
        .astype(str)
        .unique()
    )

    for participant in participants:
        required = list(
            decision_tfr_paths(participant).values()
        ) + [
            decision_raw_path(participant),
        ]

        missing = [
            str(path)
            for path in required
            if not path.exists()
        ]

        if missing:
            raise FileNotFoundError(
                f"{participant}: missing required input(s):\n"
                + "\n".join(missing)
            )

    return participants


# =============================================================================
# Time-binning helper
# =============================================================================

def make_time_bins(
    sample_times: np.ndarray,
    start_s: float,
    end_s: float,
    bin_width_s: float,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Reproduce the same non-overlapping time-bin construction used by the TFR
    preparation.
    """

    edges = np.arange(
        start_s,
        end_s + bin_width_s * 0.5,
        bin_width_s,
    )

    if edges[-1] < end_s - 1e-9:
        edges = np.append(
            edges,
            end_s,
        )

    if abs(edges[-1] - end_s) > 1e-9:
        edges = np.append(
            edges[edges < end_s],
            end_s,
        )

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

        if len(mask) == 0:
            raise RuntimeError(
                f"No EEG samples found in time bin {low:.3f}..{high:.3f} s."
            )

        centers.append(
            (low + high) / 2.0
        )

        masks.append(mask)

    return (
        np.asarray(centers, dtype=float),
        masks,
    )


# =============================================================================
# Load and bin signed voltage
# =============================================================================

def load_binned_voltage(
    participant: str,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, List[str], float]:
    """
    Load the same good trials used by the TFR analysis and create
    baseline-corrected 20-ms signed-voltage features.

    Returns
    -------
    voltage
        trial x channel x time

    times
        time-bin centers

    metadata
        exact good-trial metadata copied from the TFR preparation

    channels
        EEG channel names

    bin_width_s
        width of each voltage bin
    """

    tfr_paths = decision_tfr_paths(participant)

    metadata = pd.read_csv(
        tfr_paths["metadata"]
    )

    required_metadata = [
        "sample",
        "painlevel",
        "moneylevel",
        "block",
        "rt",
        "trialsnum",
    ]

    missing_metadata = [
        column
        for column in required_metadata
        if column not in metadata.columns
    ]

    if missing_metadata:
        raise KeyError(
            f"{participant}: TFR metadata missing {missing_metadata}"
        )

    with open(
        tfr_paths["params"],
        "r",
        encoding="utf-8",
    ) as file_handle:
        tfr_params = json.load(file_handle)

    required_params = [
        "epoch_resample_hz",
        "baseline_s",
        "saved_time_range_s",
        "time_bin_s",
    ]

    missing_params = [
        key
        for key in required_params
        if key not in tfr_params
    ]

    if missing_params:
        raise KeyError(
            f"{participant}: decision TFR params missing {missing_params}"
        )

    epoch_resample_hz = float(
        tfr_params["epoch_resample_hz"]
    )

    baseline_start_s, baseline_end_s = map(
        float,
        tfr_params["baseline_s"],
    )

    epoch_start_s, epoch_end_s = map(
        float,
        tfr_params["saved_time_range_s"],
    )

    bin_width_s = float(
        tfr_params["time_bin_s"]
    )

    raw = mne.io.read_raw_fif(
        str(decision_raw_path(participant)),
        preload=True,
        verbose="ERROR",
    )

    channels_to_drop = [
        channel
        for channel in CHANNELS_TO_DROP
        if channel in raw.ch_names
    ]

    if channels_to_drop:
        raw.drop_channels(
            channels_to_drop
        )

    raw.pick(
        "eeg",
        exclude=[],
    )

    event_samples = pd.to_numeric(
        metadata["sample"],
        errors="raise",
    ).to_numpy(dtype=int)

    events = np.column_stack(
        [
            event_samples,
            np.zeros(
                len(metadata),
                dtype=int,
            ),
            np.ones(
                len(metadata),
                dtype=int,
            ),
        ]
    )

    epochs = mne.Epochs(
        raw,
        events=events,
        event_id={"decision_cue": 1},
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
            f"{participant}: voltage epoching retained {len(epochs)} "
            f"of {len(metadata)} already-approved TFR trials."
        )

    epochs.resample(
        epoch_resample_hz,
        npad="auto",
        verbose="ERROR",
    )

    voltage = epochs.get_data(
        copy=True
    ).astype(
        np.float64,
        copy=False,
    )

    sample_times = epochs.times.astype(
        float
    )

    baseline_mask = (
        (sample_times >= baseline_start_s - 1e-12)
        & (sample_times <= baseline_end_s + 1e-12)
    )

    if not baseline_mask.any():
        raise RuntimeError(
            f"{participant}: no voltage samples inside baseline "
            f"{baseline_start_s}..{baseline_end_s} s."
        )

    # Signed voltage baseline correction. We deliberately do NOT square or take
    # absolute values: preserving polarity is the entire point of this analysis.
    voltage -= voltage[
        ...,
        baseline_mask,
    ].mean(
        axis=-1,
        keepdims=True,
    )

    bin_centers, bin_masks = make_time_bins(
        sample_times=sample_times,
        start_s=epoch_start_s,
        end_s=epoch_end_s,
        bin_width_s=bin_width_s,
    )

    binned_voltage = np.stack(
        [
            voltage[
                ...,
                bin_mask,
            ].mean(
                axis=-1
            )
            for bin_mask in bin_masks
        ],
        axis=-1,
    ).astype(
        np.float32
    )

    # Ensure this really is the exact same nominal time grid used by the
    # completed decision TFR analysis.
    reference_tfr_times = np.load(
        tfr_paths["times"]
    ).astype(float)

    if not np.allclose(
        bin_centers,
        reference_tfr_times,
        atol=1e-12,
        rtol=0.0,
    ):
        raise RuntimeError(
            f"{participant}: voltage and TFR time-bin centers do not match."
        )

    return (
        binned_voltage,
        bin_centers,
        epochs.metadata.reset_index(
            drop=True
        ).copy(),
        list(epochs.ch_names),
        bin_width_s,
    )


# =============================================================================
# Decoder and RT-sensitivity helpers
# =============================================================================

def make_ridge_decoder():
    """StandardScaler -> Ridge(alpha=1), fitted inside each CV fold."""

    return make_pipeline(
        StandardScaler(),
        Ridge(
            alpha=RIDGE_ALPHA
        ),
    )


def pearson_correlation(
    true_values: np.ndarray,
    predicted_values: np.ndarray,
) -> float:
    """Return pooled out-of-fold Pearson r."""

    if (
        np.std(true_values) == 0
        or np.std(predicted_values) == 0
    ):
        return np.nan

    return float(
        np.corrcoef(
            true_values,
            predicted_values,
        )[0, 1]
    )


def make_rt_nuisance_matrix(
    reaction_time: np.ndarray,
    response_has_occurred: np.ndarray,
) -> np.ndarray:
    """
    Nuisance matrix for the RT sensitivity analysis.

    Columns:
        log(RT)
        log(RT)^2
        response-has-occurred indicator
    """

    log_rt = np.log(
        np.asarray(
            reaction_time,
            dtype=float,
        )
    )

    return np.column_stack(
        [
            log_rt,
            log_rt ** 2,
            np.asarray(
                response_has_occurred,
                dtype=float,
            ),
        ]
    )


def residualize_features_using_training_fold(
    train_features: np.ndarray,
    test_features: np.ndarray,
    train_nuisance_raw: np.ndarray,
    test_nuisance_raw: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove nuisance-related EEG variance without test-set leakage.

    The nuisance model is estimated only from the training fold.
    """

    nuisance_mean = train_nuisance_raw.mean(
        axis=0
    )

    nuisance_sd = train_nuisance_raw.std(
        axis=0
    )

    varying_columns = (
        nuisance_sd > 1e-12
    )

    if not np.any(
        varying_columns
    ):
        return (
            train_features,
            test_features,
        )

    train_nuisance = (
        train_nuisance_raw[
            :,
            varying_columns,
        ]
        - nuisance_mean[
            varying_columns
        ]
    ) / nuisance_sd[
        varying_columns
    ]

    test_nuisance = (
        test_nuisance_raw[
            :,
            varying_columns,
        ]
        - nuisance_mean[
            varying_columns
        ]
    ) / nuisance_sd[
        varying_columns
    ]

    train_design = np.column_stack(
        [
            np.ones(
                len(train_nuisance)
            ),
            train_nuisance,
        ]
    )

    test_design = np.column_stack(
        [
            np.ones(
                len(test_nuisance)
            ),
            test_nuisance,
        ]
    )

    nuisance_coefficients, *_ = np.linalg.lstsq(
        train_design,
        train_features,
        rcond=None,
    )

    residualized_train = (
        train_features
        - train_design @ nuisance_coefficients
    )

    residualized_test = (
        test_features
        - test_design @ nuisance_coefficients
    )

    return (
        residualized_train,
        residualized_test,
    )


def leave_one_block_out_score(
    features: np.ndarray,
    target: np.ndarray,
    block_ids: np.ndarray,
    reaction_time: np.ndarray,
    response_has_occurred: np.ndarray,
    remove_rt_related_variance: bool,
) -> float:
    """Decode one target at one time point with leave-one-block-out CV."""

    predictions = np.full(
        len(target),
        np.nan,
        dtype=float,
    )

    unique_blocks = np.unique(
        block_ids
    )

    if len(unique_blocks) < 2:
        raise RuntimeError(
            "Need at least two decision blocks for cross-validation."
        )

    cross_validation = GroupKFold(
        n_splits=len(unique_blocks)
    )

    for train_indices, test_indices in cross_validation.split(
        features,
        target,
        groups=block_ids,
    ):
        train_features = np.asarray(
            features[
                train_indices
            ],
            dtype=float,
        )

        test_features = np.asarray(
            features[
                test_indices
            ],
            dtype=float,
        )

        if remove_rt_related_variance:
            train_nuisance = make_rt_nuisance_matrix(
                reaction_time[
                    train_indices
                ],
                response_has_occurred[
                    train_indices
                ],
            )

            test_nuisance = make_rt_nuisance_matrix(
                reaction_time[
                    test_indices
                ],
                response_has_occurred[
                    test_indices
                ],
            )

            (
                train_features,
                test_features,
            ) = residualize_features_using_training_fold(
                train_features=train_features,
                test_features=test_features,
                train_nuisance_raw=train_nuisance,
                test_nuisance_raw=test_nuisance,
            )

        decoder = make_ridge_decoder()

        decoder.fit(
            train_features,
            target[
                train_indices
            ],
        )

        predictions[
            test_indices
        ] = decoder.predict(
            test_features
        )

    if not np.isfinite(
        predictions
    ).all():
        raise RuntimeError(
            "Non-finite out-of-fold predictions."
        )

    return pearson_correlation(
        true_values=target,
        predicted_values=predictions,
    )


# =============================================================================
# Participant-level decoding
# =============================================================================

def decode_one_participant(
    participant: str,
) -> None:
    """Run raw and RT-residualized voltage decoding for pain and money."""

    (
        voltage,
        times,
        metadata,
        channels,
        bin_width_s,
    ) = load_binned_voltage(
        participant
    )

    pain_levels = pd.to_numeric(
        metadata["painlevel"],
        errors="raise",
    ).to_numpy(
        dtype=float
    )

    money_levels = pd.to_numeric(
        metadata["moneylevel"],
        errors="raise",
    ).to_numpy(
        dtype=float
    )

    block_ids = pd.to_numeric(
        metadata["block"],
        errors="raise",
    ).to_numpy(
        dtype=int
    )

    reaction_time = pd.to_numeric(
        metadata["rt"],
        errors="raise",
    ).to_numpy(
        dtype=float
    )

    if not (
        np.isfinite(
            pain_levels
        ).all()
        and np.isfinite(
            money_levels
        ).all()
        and np.isfinite(
            reaction_time
        ).all()
    ):
        raise RuntimeError(
            f"{participant}: non-finite pain, money, or RT values."
        )

    # For signed voltage there is no Morlet temporal support. We simply ask
    # whether the response has happened by the END of the current 20-ms bin.
    response_has_occurred = np.stack(
        [
            reaction_time
            <= (
                time_point
                + bin_width_s / 2.0
            )
            for time_point in times
        ],
        axis=1,
    )

    targets = {
        "pain": pain_levels,
        "money": money_levels,
    }

    raw_scores = np.empty(
        (
            len(ATTRIBUTES),
            len(times),
        ),
        dtype=np.float32,
    )

    rt_residualized_scores = np.empty_like(
        raw_scores
    )

    for attribute_index, attribute_name in enumerate(
        ATTRIBUTES
    ):
        target = targets[
            attribute_name
        ]

        print(
            f"    {participant}: decoding {attribute_name} "
            f"from signed voltage",
            flush=True,
        )

        for time_index in range(
            len(times)
        ):
            channel_features = np.asarray(
                voltage[
                    :,
                    :,
                    time_index,
                ],
                dtype=float,
            )

            response_now = response_has_occurred[
                :,
                time_index,
            ]

            raw_scores[
                attribute_index,
                time_index,
            ] = leave_one_block_out_score(
                features=channel_features,
                target=target,
                block_ids=block_ids,
                reaction_time=reaction_time,
                response_has_occurred=response_now,
                remove_rt_related_variance=False,
            )

            rt_residualized_scores[
                attribute_index,
                time_index,
            ] = leave_one_block_out_score(
                features=channel_features,
                target=target,
                block_ids=block_ids,
                reaction_time=reaction_time,
                response_has_occurred=response_now,
                remove_rt_related_variance=True,
            )

    np.savez_compressed(
        participant_score_path(
            participant
        ),
        participant=np.asarray(
            participant
        ),
        times=times,
        channels=np.asarray(
            channels
        ),
        attribute_names=np.asarray(
            ATTRIBUTES
        ),
        raw_r=raw_scores,
        rtres_r=rt_residualized_scores,
        n_trials=np.asarray(
            len(metadata)
        ),
        rt_min=np.asarray(
            reaction_time.min()
        ),
        rt_p05=np.asarray(
            np.quantile(
                reaction_time,
                0.05,
            )
        ),
        rt_median=np.asarray(
            np.median(
                reaction_time
            )
        ),
        response_overlap_fraction=(
            response_has_occurred
            .mean(
                axis=0
            )
            .astype(
                np.float32
            )
        ),
    )


# =============================================================================
# Group inference helpers
# =============================================================================

def holm_correct(
    p_values: Sequence[float],
) -> np.ndarray:
    """Holm correction for a small family of p-values."""

    p_values = np.asarray(
        p_values,
        dtype=float,
    )

    n_tests = len(
        p_values
    )

    order = np.argsort(
        p_values
    )

    ranked = p_values[
        order
    ]

    adjusted_ranked = np.maximum.accumulate(
        (
            n_tests
            - np.arange(
                n_tests
            )
        )
        * ranked
    )

    adjusted_ranked = np.clip(
        adjusted_ranked,
        0.0,
        1.0,
    )

    adjusted = np.empty(
        n_tests,
        dtype=float,
    )

    adjusted[
        order
    ] = adjusted_ranked

    return adjusted


def fisher_z(
    correlation_values: np.ndarray,
) -> np.ndarray:
    """Fisher-z transform correlations before group inference."""

    return np.arctanh(
        np.clip(
            correlation_values,
            -0.999999,
            0.999999,
        )
    )


def temporal_cluster_test(
    participant_scores: np.ndarray,
    times: np.ndarray,
    n_permutations: int,
    n_jobs: int,
):
    """
    One-sided cluster permutation test for above-zero decoding.

    participant_scores shape:
        participant x time
    """

    time_mask = (
        (times >= INFERENCE_TMIN - 1e-12)
        & (times <= INFERENCE_TMAX + 1e-12)
    )

    inference_times = times[
        time_mask
    ]

    fisher_scores = fisher_z(
        participant_scores[
            :,
            time_mask,
        ]
    )

    n_participants = fisher_scores.shape[
        0
    ]

    threshold = float(
        stats.t.ppf(
            1.0
            - CLUSTER_FORMING_ALPHA,
            df=n_participants - 1,
        )
    )

    (
        t_values,
        clusters,
        cluster_p_values,
        _,
    ) = permutation_cluster_1samp_test(
        fisher_scores,
        threshold=threshold,
        n_permutations=n_permutations,
        tail=1,
        seed=RANDOM_STATE,
        n_jobs=n_jobs,
        out_type="indices",
        verbose=False,
    )

    cluster_indices = [
        np.asarray(
            cluster[0],
            dtype=int,
        )
        for cluster in clusters
    ]

    return (
        t_values,
        cluster_indices,
        np.asarray(
            cluster_p_values,
            dtype=float,
        ),
        inference_times,
    )


def summarize_family(
    group_scores: np.ndarray,
    family_name: str,
    times: np.ndarray,
    n_permutations: int,
    n_jobs: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Test pain and money separately, then Holm-correct the two domain-level
    omnibus p-values.
    """

    stored_results = {}
    domain_p_values = []

    for attribute_index, attribute_name in enumerate(
        ATTRIBUTES
    ):
        result = temporal_cluster_test(
            participant_scores=group_scores[
                :,
                attribute_index,
            ],
            times=times,
            n_permutations=n_permutations,
            n_jobs=n_jobs,
        )

        (
            t_values,
            clusters,
            cluster_p_values,
            inference_times,
        ) = result

        if len(
            cluster_p_values
        ):
            minimum_p = float(
                cluster_p_values.min()
            )
        else:
            minimum_p = 1.0

        domain_p_values.append(
            minimum_p
        )

        stored_results[
            attribute_name
        ] = result

    holm_adjusted = holm_correct(
        domain_p_values
    )

    omnibus_rows = []
    cluster_rows = []

    for attribute_index, attribute_name in enumerate(
        ATTRIBUTES
    ):
        adjusted_p = float(
            holm_adjusted[
                attribute_index
            ]
        )

        omnibus_rows.append(
            {
                "family":
                    family_name,
                "attribute":
                    attribute_name,
                "omnibus_p_within_time":
                    domain_p_values[
                        attribute_index
                    ],
                "omnibus_p_holm_across_pain_money":
                    adjusted_p,
                "domain_significant_holm_0p05":
                    bool(
                        adjusted_p < 0.05
                    ),
            }
        )

        (
            t_values,
            clusters,
            cluster_p_values,
            inference_times,
        ) = stored_results[
            attribute_name
        ]

        if adjusted_p >= 0.05:
            continue

        for cluster_id, (
            cluster_indices,
            cluster_p,
        ) in enumerate(
            zip(
                clusters,
                cluster_p_values,
            )
        ):
            if cluster_p >= 0.05:
                continue

            cluster_t_values = t_values[
                cluster_indices
            ]

            peak_position = int(
                np.argmax(
                    cluster_t_values
                )
            )

            cluster_rows.append(
                {
                    "family":
                        family_name,
                    "attribute":
                        attribute_name,
                    "cluster_id":
                        cluster_id,
                    "cluster_p_within_time":
                        float(
                            cluster_p
                        ),
                    "time_start_s":
                        float(
                            inference_times[
                                cluster_indices
                            ].min()
                        ),
                    "time_end_s":
                        float(
                            inference_times[
                                cluster_indices
                            ].max()
                        ),
                    "peak_time_s":
                        float(
                            inference_times[
                                cluster_indices[
                                    peak_position
                                ]
                            ]
                        ),
                    "peak_t":
                        float(
                            cluster_t_values[
                                peak_position
                            ]
                        ),
                    "omnibus_p_holm_across_pain_money":
                        adjusted_p,
                }
            )

    return (
        pd.DataFrame(
            omnibus_rows
        ),
        pd.DataFrame(
            cluster_rows
        ),
    )


# =============================================================================
# Group analysis
# =============================================================================

def run_group_analysis(
    participants: Sequence[str],
    n_permutations: int,
    n_jobs: int,
) -> None:
    """Load participant scores, run inference, and save readable outputs."""

    participant_results = [
        np.load(
            participant_score_path(
                participant
            ),
            allow_pickle=False,
        )
        for participant in participants
    ]

    reference_times = participant_results[
        0
    ]["times"].astype(
        float
    )

    reference_channels = participant_results[
        0
    ]["channels"].astype(
        str
    )

    for participant, result in zip(
        participants,
        participant_results,
    ):
        if not np.allclose(
            result["times"].astype(float),
            reference_times,
        ):
            raise RuntimeError(
                f"{participant}: voltage time grid mismatch."
            )

        if not np.array_equal(
            result["channels"].astype(str),
            reference_channels,
        ):
            raise RuntimeError(
                f"{participant}: voltage channel order mismatch."
            )

    raw_scores = np.stack(
        [
            result["raw_r"]
            for result in participant_results
        ],
        axis=0,
    )

    rt_residualized_scores = np.stack(
        [
            result["rtres_r"]
            for result in participant_results
        ],
        axis=0,
    )

    response_overlap_fraction = np.stack(
        [
            result[
                "response_overlap_fraction"
            ]
            for result in participant_results
        ],
        axis=0,
    )

    np.savez_compressed(
        OUTPUT_ROOT
        / "group_score_arrays.npz",
        subjects=np.asarray(
            participants
        ),
        times=reference_times,
        channels=reference_channels,
        attribute_names=np.asarray(
            ATTRIBUTES
        ),
        raw_r=raw_scores,
        rtres_r=rt_residualized_scores,
        response_overlap_fraction=response_overlap_fraction,
    )

    omnibus_tables = []
    cluster_tables = []
    timecourse_rows = []

    for family_name, family_scores in [
        (
            "raw",
            raw_scores,
        ),
        (
            "rt_residualized",
            rt_residualized_scores,
        ),
    ]:
        (
            omnibus_table,
            cluster_table,
        ) = summarize_family(
            group_scores=family_scores,
            family_name=family_name,
            times=reference_times,
            n_permutations=n_permutations,
            n_jobs=n_jobs,
        )

        omnibus_tables.append(
            omnibus_table
        )

        cluster_tables.append(
            cluster_table
        )

        family_mean = family_scores.mean(
            axis=0
        )

        family_sem = stats.sem(
            family_scores,
            axis=0,
        )

        for attribute_index, attribute_name in enumerate(
            ATTRIBUTES
        ):
            for time_index, time_value in enumerate(
                reference_times
            ):
                timecourse_rows.append(
                    {
                        "family":
                            family_name,
                        "attribute":
                            attribute_name,
                        "time_s":
                            float(
                                time_value
                            ),
                        "mean_r":
                            float(
                                family_mean[
                                    attribute_index,
                                    time_index,
                                ]
                            ),
                        "sem_r":
                            float(
                                family_sem[
                                    attribute_index,
                                    time_index,
                                ]
                            ),
                        "chance":
                            0.0,
                    }
                )

    omnibus_results = pd.concat(
        omnibus_tables,
        ignore_index=True,
    )

    cluster_results = (
        pd.concat(
            cluster_tables,
            ignore_index=True,
        )
        if any(
            len(table)
            for table in cluster_tables
        )
        else pd.DataFrame()
    )

    omnibus_results.to_csv(
        OUTPUT_ROOT
        / "domain_omnibus_holm.csv",
        index=False,
    )

    cluster_results.to_csv(
        OUTPUT_ROOT
        / "significant_clusters.csv",
        index=False,
    )

    pd.DataFrame(
        timecourse_rows
    ).to_csv(
        OUTPUT_ROOT
        / "group_timecourses.csv",
        index=False,
    )

    participant_rt_rows = []

    for participant, result in zip(
        participants,
        participant_results,
    ):
        participant_rt_rows.append(
            {
                "participant":
                    participant,
                "n_trials":
                    int(
                        result["n_trials"]
                    ),
                "rt_min_s":
                    float(
                        result["rt_min"]
                    ),
                "rt_p05_s":
                    float(
                        result["rt_p05"]
                    ),
                "rt_median_s":
                    float(
                        result["rt_median"]
                    ),
            }
        )

    pd.DataFrame(
        participant_rt_rows
    ).to_csv(
        OUTPUT_ROOT
        / "participant_rt_summary.csv",
        index=False,
    )

    pd.DataFrame(
        {
            "time_s":
                reference_times,
            "mean_fraction_trials_response_has_occurred":
                response_overlap_fraction.mean(
                    axis=0
                ),
            "sem_fraction_trials_response_has_occurred":
                stats.sem(
                    response_overlap_fraction,
                    axis=0,
                ),
        }
    ).to_csv(
        OUTPUT_ROOT
        / "response_overlap_by_time.csv",
        index=False,
    )

    params = {
        "scientific_question":
            (
                "Does signed multichannel decision-phase EEG voltage carry "
                "pain or money magnitude information under the same CV and "
                "scoring framework used for decision TFR decoding?"
            ),
        "signal":
            "baseline-corrected signed EEG voltage",
        "trial_set":
            (
                "exact same good decision trials as total-power decision TFR"
            ),
        "baseline":
            (
                "copied from each participant's decision TFR params "
                "(normally -0.7..-0.5 s)"
            ),
        "time_bins":
            (
                "copied from decision TFR params and checked against "
                "decision TFR time grid"
            ),
        "decoder":
            "StandardScaler -> Ridge(alpha=1)",
        "cross_validation":
            "leave-one-decision-block-out",
        "score":
            "pooled out-of-fold Pearson r",
        "primary_family":
            "raw voltage",
        "rt_sensitivity":
            (
                "training-fold-only feature residualization for log(RT), "
                "log(RT)^2, and response-has-occurred-by-bin-end"
            ),
        "formal_inference_s":
            [
                INFERENCE_TMIN,
                INFERENCE_TMAX,
            ],
        "n_permutations":
            n_permutations,
        "multiplicity":
            (
                "one-sided temporal cluster test separately pain/money; "
                "Holm across pain/money within raw and RT-residualized families"
            ),
        "interpretation_note":
            (
                "RT adjustment is a sensitivity analysis because RT is "
                "downstream of the decision process."
            ),
    }

    (
        OUTPUT_ROOT
        / "params.json"
    ).write_text(
        json.dumps(
            params,
            indent=2,
        )
    )

    print(
        "\nPRIMARY RAW VOLTAGE",
        flush=True,
    )

    print(
        omnibus_results[
            omnibus_results["family"]
            == "raw"
        ].to_string(
            index=False
        ),
        flush=True,
    )

    print(
        "\nRT-RESIDUALIZED VOLTAGE",
        flush=True,
    )

    print(
        omnibus_results[
            omnibus_results["family"]
            == "rt_residualized"
        ].to_string(
            index=False
        ),
        flush=True,
    )

    if len(
        cluster_results
    ):
        print(
            "\nSIGNIFICANT VOLTAGE CLUSTERS",
            flush=True,
        )

        print(
            cluster_results.to_string(
                index=False
            ),
            flush=True,
        )


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = parse_args()

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    PARTICIPANT_SCORE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    participants = load_participant_list()

    print(
        "=" * 76,
        flush=True,
    )
    print(
        "MATCHED DECISION-PHASE VOLTAGE DECODING",
        flush=True,
    )
    print(
        f"Participants: {len(participants)}",
        flush=True,
    )
    print(
        "=" * 76,
        flush=True,
    )

    if not args.group_only:
        for participant_index, participant in enumerate(
            participants,
            start=1,
        ):
            score_file = participant_score_path(
                participant
            )

            if (
                score_file.exists()
                and not args.overwrite_scores
            ):
                print(
                    f"[{participant_index}/{len(participants)}] "
                    f"{participant}: SKIP existing scores",
                    flush=True,
                )
                continue

            print(
                f"\n[{participant_index}/{len(participants)}] "
                f"{participant}",
                flush=True,
            )

            decode_one_participant(
                participant
            )

            print(
                f"[{participant}] DONE",
                flush=True,
            )

    print(
        "\nRunning group-level voltage inference...",
        flush=True,
    )

    run_group_analysis(
        participants=participants,
        n_permutations=args.n_permutations,
        n_jobs=args.n_jobs,
    )

    print(
        "\nMATCHED DECISION-PHASE VOLTAGE DECODING COMPLETE",
        flush=True,
    )


if __name__ == "__main__":
    main()
