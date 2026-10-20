#!/usr/bin/env python3
"""
05u_eeg_voltage_decode_decision_interaction.py
==============================================

Cue-locked decision-phase SIGNED-VOLTAGE MVPA of the centered Pain x Money
interaction.

Purpose
-------
This is the signed-voltage counterpart of the completed cue-locked TFR
interaction analysis.

The scientific question is:

    Does cue-locked multichannel signed EEG voltage contain information about
    the JOINT Pain x Money combination beyond additive pain and money main
    effects?

Interaction target
------------------
Pain and money use the known 1-5 factorial midpoint:

    pain_centered  = painlevel  - 3
    money_centered = moneylevel - 3

    interaction = pain_centered * money_centered

This is an objective factorial interaction. It is NOT yet a subjective-value
estimate.

Why "signed-voltage MVPA" rather than a conventional averaged ERP?
-------------------------------------------------------------------
The interaction target varies trial by trial. Therefore the apples-to-apples
counterpart of the TFR MVPA is single-trial, cue-locked signed voltage across
the 65 EEG channels.

Voltage is:
    - baseline corrected,
    - averaged into the exact same 20-ms bins as the decision TFR,
    - kept signed (not squared or absolute-valued).

This is often described as ERP-like / signed-voltage MVPA. A conventional
condition-averaged ERP would throw away the trialwise information needed by
the decoder.

Primary interaction-specific analysis
-------------------------------------
Within each leave-one-block-out fold:

    1. residualise the interaction target against centered Pain + Money using
       training-fold nuisance estimates;
    2. residualise each EEG channel feature against centered Pain + Money using
       training-fold nuisance estimates;
    3. fit StandardScaler -> Ridge(alpha=1) to the residualised training data;
    4. transform the held-out block with the training-fold nuisance models;
    5. pool held-out predictions and score against the held-out conditional
       interaction target using Pearson r.

This makes the test explicitly conditional on the two additive main effects.

RT sensitivity
--------------
The secondary RT-adjusted analysis uses the SAME conditional interaction target
but additionally removes from EEG features, within each training fold:

    - log(RT)
    - log(RT)^2
    - whether the behavioral response has occurred by the END of the current
      20-ms voltage bin

RT is not removed from the interaction target. RT remains a downstream
behavioral variable, so RT adjustment is a sensitivity analysis rather than
the primary estimand.

Direct RT-adjusted minus primary comparison
-------------------------------------------
At group level the script additionally tests:

    Fisher-z(r_RT-adjusted) - Fisher-z(r_primary)

with the same one-sided temporal cluster procedure.

This prevents "one p-value is below .05 and the other is not" from being
misinterpreted as evidence that RT adjustment itself improved decoding.

Inputs
------
Uses the SAME already-approved trials as the existing cue-locked decision TFR:

    derivatives/<sub>/eeg/tfr_decision/
        <sub>_decision_cue_metadata.csv
        <sub>_decision_cue_times.npy
        <sub>_decision_cue_params.json

and cleaned broadband decision EEG:

    SOURCE_BIDS/derivatives/<sub>/eeg/
        <sub>_decision_cleaned-raw.fif

No new trial exclusions are introduced.

Inference
---------
    formal window: 0.00 to 1.10 s after decision cue
    one-sided temporal cluster permutation
    5000 permutations by default

There is one interaction target, so there is no pain-vs-money Holm correction.

Outputs
-------
derivatives/statistics/voltage_mvpa_decision_interaction/
    participant_scores/
    group_score_arrays.npz
    temporal_omnibus.csv
    significant_clusters.csv
    group_timecourses.csv
    participant_interaction_target_summary.csv
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
# Paths and frozen analysis constants
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
    / "voltage_mvpa_decision_interaction"
)

PARTICIPANT_SCORE_DIR = OUTPUT_ROOT / "participant_scores"

PAIN_DESIGN_CENTER = 3.0
MONEY_DESIGN_CENTER = 3.0

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
        description=(
            "Cue-locked signed-voltage decoding of the centered "
            "Pain x Money interaction."
        )
    )

    parser.add_argument(
        "--n-permutations",
        type=int,
        default=5000,
        help="Number of group-level temporal-cluster permutations.",
    )

    parser.add_argument(
        "--n-jobs",
        type=int,
        default=max(
            1,
            int(os.getenv("SLURM_CPUS_PER_TASK", "1")),
        ),
        help="Parallel jobs used by MNE group inference.",
    )

    parser.add_argument(
        "--overwrite-scores",
        action="store_true",
        help="Recompute participant scores even when saved scores exist.",
    )

    parser.add_argument(
        "--group-only",
        action="store_true",
        help="Skip participant decoding and rerun group inference only.",
    )

    return parser.parse_args()


# =============================================================================
# Input/output paths
# =============================================================================

def decision_tfr_paths(
    participant: str,
) -> Dict[str, Path]:
    """
    Existing decision-TFR metadata defines the exact accepted trial set and
    nominal 20-ms time grid used for the matched signed-voltage analysis.
    """

    tfr_directory = (
        BASE_PATH
        / "derivatives"
        / participant
        / "eeg"
        / "tfr_decision"
    )

    stem = f"{participant}_decision_cue"

    return {
        "times":
            tfr_directory
            / f"{stem}_times.npy",
        "metadata":
            tfr_directory
            / f"{stem}_metadata.csv",
        "params":
            tfr_directory
            / f"{stem}_params.json",
    }


def decision_raw_path(
    participant: str,
) -> Path:
    """Cleaned broadband decision EEG."""

    return (
        SOURCE_BIDS
        / "derivatives"
        / participant
        / "eeg"
        / f"{participant}_decision_cleaned-raw.fif"
    )


def participant_score_path(
    participant: str,
) -> Path:
    return (
        PARTICIPANT_SCORE_DIR
        / f"{participant}_decision_interaction_voltage_mvpa.npz"
    )


def load_participant_list() -> List[str]:
    participant_table = pd.read_csv(
        PARTICIPANTS_TSV,
        sep="\t",
    )

    participants = sorted(
        participant_table[
            "participant_id"
        ]
        .dropna()
        .astype(str)
        .unique()
    )

    for participant in participants:
        required_paths = list(
            decision_tfr_paths(
                participant
            ).values()
        ) + [
            decision_raw_path(
                participant
            )
        ]

        missing_paths = [
            str(path)
            for path in required_paths
            if not path.exists()
        ]

        if missing_paths:
            raise FileNotFoundError(
                f"{participant}: missing required input(s):\n"
                + "\n".join(
                    missing_paths
                )
            )

    return participants


# =============================================================================
# Exact 20-ms signed-voltage construction
# =============================================================================

def make_time_bins(
    sample_times: np.ndarray,
    start_s: float,
    end_s: float,
    bin_width_s: float,
) -> Tuple[
    np.ndarray,
    List[np.ndarray],
]:
    """Reproduce the non-overlapping time bins used by decision TFR prep."""

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

    if abs(
        edges[-1]
        - end_s
    ) > 1e-9:
        edges = np.append(
            edges[
                edges
                < end_s
            ],
            end_s,
        )

    centers: List[float] = []
    masks: List[np.ndarray] = []

    for bin_index in range(
        len(edges)
        - 1
    ):
        low = float(
            edges[
                bin_index
            ]
        )

        high = float(
            edges[
                bin_index
                + 1
            ]
        )

        if bin_index == len(edges) - 2:
            sample_indices = np.where(
                (
                    sample_times
                    >= low - 1e-12
                )
                & (
                    sample_times
                    <= high + 1e-12
                )
            )[0]
        else:
            sample_indices = np.where(
                (
                    sample_times
                    >= low - 1e-12
                )
                & (
                    sample_times
                    < high - 1e-12
                )
            )[0]

        if len(
            sample_indices
        ) == 0:
            raise RuntimeError(
                "No EEG samples found in voltage bin "
                f"{low:.3f}..{high:.3f} s."
            )

        centers.append(
            (
                low
                + high
            )
            / 2.0
        )

        masks.append(
            sample_indices
        )

    return (
        np.asarray(
            centers,
            dtype=float,
        ),
        masks,
    )


def load_binned_voltage(
    participant: str,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    pd.DataFrame,
    List[str],
    float,
]:
    """
    Create baseline-corrected signed-voltage features on the exact decision-TFR
    trial set and time grid.

    Returns
    -------
    binned_voltage
        trial x channel x time

    bin_centers
        nominal centers of the 20-ms bins

    metadata
        exact approved TFR trial metadata

    channel_names
        EEG channels

    bin_width_s
        nominal bin width
    """

    tfr_paths = decision_tfr_paths(
        participant
    )

    metadata = pd.read_csv(
        tfr_paths[
            "metadata"
        ]
    )

    required_metadata_columns = [
        "sample",
        "painlevel",
        "moneylevel",
        "block",
        "rt",
        "trialsnum",
    ]

    missing_metadata_columns = [
        column
        for column in required_metadata_columns
        if column not in metadata.columns
    ]

    if missing_metadata_columns:
        raise KeyError(
            f"{participant}: decision metadata missing "
            f"{missing_metadata_columns}"
        )

    with open(
        tfr_paths[
            "params"
        ],
        "r",
        encoding="utf-8",
    ) as file_handle:
        tfr_params = json.load(
            file_handle
        )

    required_parameter_names = [
        "epoch_resample_hz",
        "baseline_s",
        "saved_time_range_s",
        "time_bin_s",
    ]

    missing_parameter_names = [
        parameter_name
        for parameter_name in required_parameter_names
        if parameter_name not in tfr_params
    ]

    if missing_parameter_names:
        raise KeyError(
            f"{participant}: decision TFR params missing "
            f"{missing_parameter_names}"
        )

    epoch_resample_hz = float(
        tfr_params[
            "epoch_resample_hz"
        ]
    )

    (
        baseline_start_s,
        baseline_end_s,
    ) = map(
        float,
        tfr_params[
            "baseline_s"
        ],
    )

    (
        epoch_start_s,
        epoch_end_s,
    ) = map(
        float,
        tfr_params[
            "saved_time_range_s"
        ],
    )

    bin_width_s = float(
        tfr_params[
            "time_bin_s"
        ]
    )

    raw = mne.io.read_raw_fif(
        str(
            decision_raw_path(
                participant
            )
        ),
        preload=True,
        verbose="ERROR",
    )

    channels_to_drop = [
        channel_name
        for channel_name in CHANNELS_TO_DROP
        if channel_name in raw.ch_names
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
        metadata[
            "sample"
        ],
        errors="raise",
    ).to_numpy(
        dtype=int
    )

    events = np.column_stack(
        [
            event_samples,
            np.zeros(
                len(
                    metadata
                ),
                dtype=int,
            ),
            np.ones(
                len(
                    metadata
                ),
                dtype=int,
            ),
        ]
    )

    epochs = mne.Epochs(
        raw,
        events=events,
        event_id={
            "decision_cue":
                1
        },
        tmin=epoch_start_s,
        tmax=epoch_end_s,
        baseline=None,
        metadata=
            metadata.copy(),
        preload=True,
        reject=None,
        reject_by_annotation=True,
        verbose="ERROR",
    )

    if len(
        epochs
    ) != len(
        metadata
    ):
        raise RuntimeError(
            f"{participant}: voltage epoching retained {len(epochs)} of "
            f"{len(metadata)} already-approved decision-TFR trials."
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
        (
            sample_times
            >= baseline_start_s - 1e-12
        )
        & (
            sample_times
            <= baseline_end_s + 1e-12
        )
    )

    if not baseline_mask.any():
        raise RuntimeError(
            f"{participant}: no signed-voltage samples inside baseline "
            f"{baseline_start_s}..{baseline_end_s} s."
        )

    # Preserve voltage polarity. The only transform here is baseline subtraction.
    voltage -= voltage[
        ...,
        baseline_mask,
    ].mean(
        axis=-1,
        keepdims=True,
    )

    (
        bin_centers,
        bin_masks,
    ) = make_time_bins(
        sample_times=
            sample_times,
        start_s=
            epoch_start_s,
        end_s=
            epoch_end_s,
        bin_width_s=
            bin_width_s,
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

    reference_tfr_times = np.load(
        tfr_paths[
            "times"
        ]
    ).astype(
        float
    )

    if not np.allclose(
        bin_centers,
        reference_tfr_times,
        atol=1e-12,
        rtol=0.0,
    ):
        raise RuntimeError(
            f"{participant}: signed-voltage and TFR time-bin centers differ."
        )

    return (
        binned_voltage,
        bin_centers,
        epochs.metadata.reset_index(
            drop=True
        ).copy(),
        list(
            epochs.ch_names
        ),
        bin_width_s,
    )


# =============================================================================
# Interaction target and leakage-safe nuisance residualisation
# =============================================================================

def centered_design_variables(
    pain_levels: np.ndarray,
    money_levels: np.ndarray,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Return centered Pain, centered Money, and their centered product.

    The known design midpoint (3) is used rather than the retained-trial mean.
    """

    pain_centered = (
        np.asarray(
            pain_levels,
            dtype=float,
        )
        - PAIN_DESIGN_CENTER
    )

    money_centered = (
        np.asarray(
            money_levels,
            dtype=float,
        )
        - MONEY_DESIGN_CENTER
    )

    interaction_target = (
        pain_centered
        * money_centered
    )

    return (
        pain_centered,
        money_centered,
        interaction_target,
    )


def fit_nuisance_residualizer(
    nuisance_train_raw: np.ndarray,
    values_train: np.ndarray,
) -> dict:
    """
    Fit one linear nuisance model on the TRAINING fold only.

    `values_train` can be either a 1-D target or trial x channel features.
    """

    nuisance_train_raw = np.asarray(
        nuisance_train_raw,
        dtype=float,
    )

    values_train = np.asarray(
        values_train,
        dtype=float,
    )

    nuisance_mean = nuisance_train_raw.mean(
        axis=0
    )

    nuisance_sd = nuisance_train_raw.std(
        axis=0
    )

    varying_columns = (
        nuisance_sd
        > 1e-12
    )

    if np.any(
        varying_columns
    ):
        standardized_nuisance = (
            nuisance_train_raw[
                :,
                varying_columns,
            ]
            - nuisance_mean[
                varying_columns
            ]
        ) / nuisance_sd[
            varying_columns
        ]

        design_matrix = np.column_stack(
            [
                np.ones(
                    len(
                        standardized_nuisance
                    )
                ),
                standardized_nuisance,
            ]
        )
    else:
        design_matrix = np.ones(
            (
                len(
                    nuisance_train_raw
                ),
                1,
            )
        )

    coefficients, *_ = np.linalg.lstsq(
        design_matrix,
        values_train,
        rcond=None,
    )

    return {
        "mean":
            nuisance_mean,
        "sd":
            nuisance_sd,
        "varying":
            varying_columns,
        "coefficients":
            coefficients,
    }


def apply_nuisance_residualizer(
    nuisance_raw: np.ndarray,
    values: np.ndarray,
    fitted_residualizer: dict,
) -> np.ndarray:
    """Apply a training-fold nuisance model to train or held-out values."""

    nuisance_raw = np.asarray(
        nuisance_raw,
        dtype=float,
    )

    values = np.asarray(
        values,
        dtype=float,
    )

    varying_columns = fitted_residualizer[
        "varying"
    ]

    if np.any(
        varying_columns
    ):
        standardized_nuisance = (
            nuisance_raw[
                :,
                varying_columns,
            ]
            - fitted_residualizer[
                "mean"
            ][
                varying_columns
            ]
        ) / fitted_residualizer[
            "sd"
        ][
            varying_columns
        ]

        design_matrix = np.column_stack(
            [
                np.ones(
                    len(
                        standardized_nuisance
                    )
                ),
                standardized_nuisance,
            ]
        )
    else:
        design_matrix = np.ones(
            (
                len(
                    nuisance_raw
                ),
                1,
            )
        )

    nuisance_prediction = (
        design_matrix
        @ fitted_residualizer[
            "coefficients"
        ]
    )

    return (
        values
        - nuisance_prediction
    )


def main_effect_nuisance_matrix(
    pain_centered: np.ndarray,
    money_centered: np.ndarray,
) -> np.ndarray:
    """Additive Pain and Money main effects."""

    return np.column_stack(
        [
            np.asarray(
                pain_centered,
                dtype=float,
            ),
            np.asarray(
                money_centered,
                dtype=float,
            ),
        ]
    )


def feature_nuisance_matrix(
    pain_centered: np.ndarray,
    money_centered: np.ndarray,
    reaction_time: np.ndarray,
    response_has_occurred: np.ndarray,
    include_rt_controls: bool,
) -> np.ndarray:
    """
    EEG nuisance model.

    Primary:
        centered Pain + centered Money

    RT sensitivity:
        centered Pain + centered Money
        + log(RT) + log(RT)^2 + response-has-occurred
    """

    nuisance_columns = [
        np.asarray(
            pain_centered,
            dtype=float,
        ),
        np.asarray(
            money_centered,
            dtype=float,
        ),
    ]

    if include_rt_controls:
        log_rt = np.log(
            np.asarray(
                reaction_time,
                dtype=float,
            )
        )

        nuisance_columns.extend(
            [
                log_rt,
                log_rt ** 2,
                np.asarray(
                    response_has_occurred,
                    dtype=float,
                ),
            ]
        )

    return np.column_stack(
        nuisance_columns
    )


# =============================================================================
# Leave-one-block-out conditional interaction decoding
# =============================================================================

def make_ridge_decoder():
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
    true_values = np.asarray(
        true_values,
        dtype=float,
    )

    predicted_values = np.asarray(
        predicted_values,
        dtype=float,
    )

    if (
        np.std(
            true_values
        ) <= 1e-12
        or np.std(
            predicted_values
        ) <= 1e-12
    ):
        return np.nan

    return float(
        np.corrcoef(
            true_values,
            predicted_values,
        )[0, 1]
    )


def leave_one_block_out_interaction_score(
    features: np.ndarray,
    pain_centered: np.ndarray,
    money_centered: np.ndarray,
    interaction_target: np.ndarray,
    block_ids: np.ndarray,
    reaction_time: np.ndarray,
    response_has_occurred: np.ndarray,
    include_rt_controls: bool,
) -> float:
    """
    Decode Pain x Money conditional on additive Pain and Money main effects.

    All nuisance fitting happens inside the training fold.
    """

    predictions = np.full(
        len(
            interaction_target
        ),
        np.nan,
        dtype=float,
    )

    heldout_conditional_target = np.full(
        len(
            interaction_target
        ),
        np.nan,
        dtype=float,
    )

    unique_blocks = np.unique(
        block_ids
    )

    if len(
        unique_blocks
    ) < 2:
        raise RuntimeError(
            "Need at least two decision blocks for leave-one-block-out CV."
        )

    cross_validation = GroupKFold(
        n_splits=len(
            unique_blocks
        )
    )

    target_nuisance = main_effect_nuisance_matrix(
        pain_centered=
            pain_centered,
        money_centered=
            money_centered,
    )

    all_feature_nuisance = feature_nuisance_matrix(
        pain_centered=
            pain_centered,
        money_centered=
            money_centered,
        reaction_time=
            reaction_time,
        response_has_occurred=
            response_has_occurred,
        include_rt_controls=
            include_rt_controls,
    )

    for (
        train_indices,
        test_indices,
    ) in cross_validation.split(
        features,
        interaction_target,
        groups=block_ids,
    ):
        # --------------------------------------------------------------
        # Conditional target: interaction beyond additive main effects.
        # --------------------------------------------------------------

        fitted_target_nuisance = fit_nuisance_residualizer(
            nuisance_train_raw=
                target_nuisance[
                    train_indices
                ],
            values_train=
                interaction_target[
                    train_indices
                ],
        )

        train_target = apply_nuisance_residualizer(
            nuisance_raw=
                target_nuisance[
                    train_indices
                ],
            values=
                interaction_target[
                    train_indices
                ],
            fitted_residualizer=
                fitted_target_nuisance,
        )

        test_target = apply_nuisance_residualizer(
            nuisance_raw=
                target_nuisance[
                    test_indices
                ],
            values=
                interaction_target[
                    test_indices
                ],
            fitted_residualizer=
                fitted_target_nuisance,
        )

        # --------------------------------------------------------------
        # EEG features: additive main effects always removed; RT-related
        # feature variance removed only in the sensitivity family.
        # --------------------------------------------------------------

        fitted_feature_nuisance = fit_nuisance_residualizer(
            nuisance_train_raw=
                all_feature_nuisance[
                    train_indices
                ],
            values_train=
                features[
                    train_indices
                ],
        )

        train_features = apply_nuisance_residualizer(
            nuisance_raw=
                all_feature_nuisance[
                    train_indices
                ],
            values=
                features[
                    train_indices
                ],
            fitted_residualizer=
                fitted_feature_nuisance,
        )

        test_features = apply_nuisance_residualizer(
            nuisance_raw=
                all_feature_nuisance[
                    test_indices
                ],
            values=
                features[
                    test_indices
                ],
            fitted_residualizer=
                fitted_feature_nuisance,
        )

        decoder = make_ridge_decoder()

        decoder.fit(
            train_features,
            train_target,
        )

        predictions[
            test_indices
        ] = decoder.predict(
            test_features
        )

        heldout_conditional_target[
            test_indices
        ] = test_target

    if not (
        np.isfinite(
            predictions
        ).all()
        and np.isfinite(
            heldout_conditional_target
        ).all()
    ):
        raise RuntimeError(
            "Cross-validation produced non-finite predictions or targets."
        )

    return pearson_correlation(
        true_values=
            heldout_conditional_target,
        predicted_values=
            predictions,
    )


def cross_fitted_conditional_interaction_target(
    pain_centered: np.ndarray,
    money_centered: np.ndarray,
    interaction_target: np.ndarray,
    block_ids: np.ndarray,
) -> np.ndarray:
    """Held-out conditional interaction target for diagnostics only."""

    conditional_target = np.full(
        len(
            interaction_target
        ),
        np.nan,
        dtype=float,
    )

    target_nuisance = main_effect_nuisance_matrix(
        pain_centered=
            pain_centered,
        money_centered=
            money_centered,
    )

    unique_blocks = np.unique(
        block_ids
    )

    cross_validation = GroupKFold(
        n_splits=len(
            unique_blocks
        )
    )

    for (
        train_indices,
        test_indices,
    ) in cross_validation.split(
        target_nuisance,
        interaction_target,
        groups=block_ids,
    ):
        fitted_target_nuisance = fit_nuisance_residualizer(
            nuisance_train_raw=
                target_nuisance[
                    train_indices
                ],
            values_train=
                interaction_target[
                    train_indices
                ],
        )

        conditional_target[
            test_indices
        ] = apply_nuisance_residualizer(
            nuisance_raw=
                target_nuisance[
                    test_indices
                ],
            values=
                interaction_target[
                    test_indices
                ],
            fitted_residualizer=
                fitted_target_nuisance,
        )

    if not np.isfinite(
        conditional_target
    ).all():
        raise RuntimeError(
            "Conditional interaction target contains non-finite values."
        )

    return conditional_target


# =============================================================================
# Participant-level signed-voltage decoding
# =============================================================================

def decode_one_participant(
    participant: str,
) -> None:
    (
        voltage,
        times,
        metadata,
        channel_names,
        bin_width_s,
    ) = load_binned_voltage(
        participant
    )

    pain_levels = pd.to_numeric(
        metadata[
            "painlevel"
        ],
        errors="raise",
    ).to_numpy(
        dtype=float
    )

    money_levels = pd.to_numeric(
        metadata[
            "moneylevel"
        ],
        errors="raise",
    ).to_numpy(
        dtype=float
    )

    block_ids = pd.to_numeric(
        metadata[
            "block"
        ],
        errors="raise",
    ).to_numpy(
        dtype=int
    )

    reaction_time = pd.to_numeric(
        metadata[
            "rt"
        ],
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

    if np.any(
        reaction_time
        <= 0
    ):
        raise RuntimeError(
            f"{participant}: non-positive RT encountered."
        )

    (
        pain_centered,
        money_centered,
        interaction_target,
    ) = centered_design_variables(
        pain_levels=
            pain_levels,
        money_levels=
            money_levels,
    )

    conditional_interaction_target = (
        cross_fitted_conditional_interaction_target(
            pain_centered=
                pain_centered,
            money_centered=
                money_centered,
            interaction_target=
                interaction_target,
            block_ids=
                block_ids,
        )
    )

    # Voltage has no Morlet support. The response indicator switches on when
    # the response has occurred by the end of the current 20-ms bin.
    response_has_occurred = np.stack(
        [
            reaction_time
            <= (
                time_point
                + bin_width_s
                / 2.0
            )
            for time_point
            in times
        ],
        axis=1,
    )

    primary_scores = np.empty(
        len(
            times
        ),
        dtype=np.float32,
    )

    rt_adjusted_scores = np.empty_like(
        primary_scores
    )

    print(
        f"    {participant}: decoding centered Pain x Money "
        "from signed voltage",
        flush=True,
    )

    for time_index in range(
        len(
            times
        )
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

        primary_scores[
            time_index
        ] = leave_one_block_out_interaction_score(
            features=
                channel_features,
            pain_centered=
                pain_centered,
            money_centered=
                money_centered,
            interaction_target=
                interaction_target,
            block_ids=
                block_ids,
            reaction_time=
                reaction_time,
            response_has_occurred=
                response_now,
            include_rt_controls=
                False,
        )

        rt_adjusted_scores[
            time_index
        ] = leave_one_block_out_interaction_score(
            features=
                channel_features,
            pain_centered=
                pain_centered,
            money_centered=
                money_centered,
            interaction_target=
                interaction_target,
            block_ids=
                block_ids,
            reaction_time=
                reaction_time,
            response_has_occurred=
                response_now,
            include_rt_controls=
                True,
        )

    np.savez_compressed(
        participant_score_path(
            participant
        ),
        participant=
            np.asarray(
                participant
            ),
        times=
            times,
        channels=
            np.asarray(
                channel_names
            ),
        primary_r=
            primary_scores,
        rt_adjusted_r=
            rt_adjusted_scores,
        n_trials=
            np.asarray(
                len(
                    metadata
                )
            ),
        rt_min=
            np.asarray(
                reaction_time.min()
            ),
        rt_p05=
            np.asarray(
                np.quantile(
                    reaction_time,
                    0.05,
                )
            ),
        rt_median=
            np.asarray(
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
        corr_raw_interaction_with_pain=
            np.asarray(
                pearson_correlation(
                    interaction_target,
                    pain_centered,
                )
            ),
        corr_raw_interaction_with_money=
            np.asarray(
                pearson_correlation(
                    interaction_target,
                    money_centered,
                )
            ),
        corr_conditional_interaction_with_pain=
            np.asarray(
                pearson_correlation(
                    conditional_interaction_target,
                    pain_centered,
                )
            ),
        corr_conditional_interaction_with_money=
            np.asarray(
                pearson_correlation(
                    conditional_interaction_target,
                    money_centered,
                )
            ),
        corr_raw_with_conditional_interaction=
            np.asarray(
                pearson_correlation(
                    interaction_target,
                    conditional_interaction_target,
                )
            ),
    )


# =============================================================================
# Group-level temporal cluster inference
# =============================================================================

def fisher_z(
    correlation_values: np.ndarray,
) -> np.ndarray:
    return np.arctanh(
        np.clip(
            correlation_values,
            -0.999999,
            0.999999,
        )
    )


def one_sided_temporal_cluster_test(
    participant_values: np.ndarray,
    times: np.ndarray,
    n_permutations: int,
    n_jobs: int,
    values_are_correlations: bool,
):
    """
    One-sided test for positive decoding / positive RT-adjusted-minus-primary
    difference.
    """

    time_mask = (
        (
            times
            >= INFERENCE_TMIN - 1e-12
        )
        & (
            times
            <= INFERENCE_TMAX + 1e-12
        )
    )

    inference_times = times[
        time_mask
    ]

    values = participant_values[
        :,
        time_mask,
    ]

    if values_are_correlations:
        values = fisher_z(
            values
        )

    n_participants = values.shape[
        0
    ]

    threshold = float(
        stats.t.ppf(
            1.0
            - CLUSTER_FORMING_ALPHA,
            df=
                n_participants
                - 1,
        )
    )

    (
        t_values,
        clusters,
        cluster_p_values,
        _,
    ) = permutation_cluster_1samp_test(
        values,
        threshold=
            threshold,
        n_permutations=
            n_permutations,
        tail=1,
        seed=
            RANDOM_STATE,
        n_jobs=
            n_jobs,
        out_type=
            "indices",
        verbose=False,
    )

    cluster_indices = [
        np.asarray(
            cluster[
                0
            ],
            dtype=int,
        )
        for cluster in clusters
    ]

    return (
        np.asarray(
            t_values,
            dtype=float,
        ),
        cluster_indices,
        np.asarray(
            cluster_p_values,
            dtype=float,
        ),
        inference_times,
    )


def summarize_temporal_family(
    participant_values: np.ndarray,
    family_name: str,
    times: np.ndarray,
    n_permutations: int,
    n_jobs: int,
    values_are_correlations: bool,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    (
        t_values,
        clusters,
        cluster_p_values,
        inference_times,
    ) = one_sided_temporal_cluster_test(
        participant_values=
            participant_values,
        times=
            times,
        n_permutations=
            n_permutations,
        n_jobs=
            n_jobs,
        values_are_correlations=
            values_are_correlations,
    )

    minimum_cluster_p = (
        float(
            cluster_p_values.min()
        )
        if len(
            cluster_p_values
        )
        else 1.0
    )

    omnibus_table = pd.DataFrame(
        [
            {
                "family":
                    family_name,
                "target":
                    "centered_pain_x_money",
                "omnibus_p_within_time":
                    minimum_cluster_p,
                "temporal_significant_0p05":
                    bool(
                        minimum_cluster_p
                        < 0.05
                    ),
                "n_participants":
                    participant_values.shape[
                        0
                    ],
                "formal_tmin_s":
                    INFERENCE_TMIN,
                "formal_tmax_s":
                    INFERENCE_TMAX,
            }
        ]
    )

    cluster_rows = []

    if minimum_cluster_p < 0.05:
        for (
            cluster_id,
            (
                cluster_indices,
                cluster_p,
            ),
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
                    "target":
                        "centered_pain_x_money",
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
                }
            )

    return (
        omnibus_table,
        pd.DataFrame(
            cluster_rows
        ),
    )


# =============================================================================
# Group analysis and outputs
# =============================================================================

def run_group_analysis(
    participants: Sequence[str],
    n_permutations: int,
    n_jobs: int,
) -> None:
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
    ][
        "times"
    ].astype(
        float
    )

    reference_channels = participant_results[
        0
    ][
        "channels"
    ].astype(
        str
    )

    for (
        participant,
        participant_result,
    ) in zip(
        participants,
        participant_results,
    ):
        if not np.allclose(
            participant_result[
                "times"
            ].astype(
                float
            ),
            reference_times,
        ):
            raise RuntimeError(
                f"{participant}: voltage time-grid mismatch."
            )

        if not np.array_equal(
            participant_result[
                "channels"
            ].astype(
                str
            ),
            reference_channels,
        ):
            raise RuntimeError(
                f"{participant}: voltage channel-order mismatch."
            )

    primary_scores = np.stack(
        [
            participant_result[
                "primary_r"
            ]
            for participant_result
            in participant_results
        ],
        axis=0,
    )

    rt_adjusted_scores = np.stack(
        [
            participant_result[
                "rt_adjusted_r"
            ]
            for participant_result
            in participant_results
        ],
        axis=0,
    )

    rt_adjusted_minus_primary = (
        fisher_z(
            rt_adjusted_scores
        )
        - fisher_z(
            primary_scores
        )
    )

    response_overlap_fraction = np.stack(
        [
            participant_result[
                "response_overlap_fraction"
            ]
            for participant_result
            in participant_results
        ],
        axis=0,
    )

    np.savez_compressed(
        OUTPUT_ROOT
        / "group_score_arrays.npz",
        subjects=
            np.asarray(
                participants
            ),
        times=
            reference_times,
        channels=
            reference_channels,
        primary_r=
            primary_scores,
        rt_adjusted_r=
            rt_adjusted_scores,
        rt_adjusted_minus_primary_fisher_z=
            rt_adjusted_minus_primary,
        response_overlap_fraction=
            response_overlap_fraction,
    )

    omnibus_tables = []
    cluster_tables = []

    analysis_families = [
        (
            "main_effect_controlled_interaction",
            primary_scores,
            True,
        ),
        (
            "main_effect_plus_rt_controlled_interaction",
            rt_adjusted_scores,
            True,
        ),
        (
            "rt_adjusted_minus_primary_fisher_z",
            rt_adjusted_minus_primary,
            False,
        ),
    ]

    for (
        family_name,
        family_values,
        values_are_correlations,
    ) in analysis_families:
        (
            omnibus_table,
            cluster_table,
        ) = summarize_temporal_family(
            participant_values=
                family_values,
            family_name=
                family_name,
            times=
                reference_times,
            n_permutations=
                n_permutations,
            n_jobs=
                n_jobs,
            values_are_correlations=
                values_are_correlations,
        )

        omnibus_tables.append(
            omnibus_table
        )

        cluster_tables.append(
            cluster_table
        )

    omnibus_results = pd.concat(
        omnibus_tables,
        ignore_index=True,
    )

    significant_clusters = (
        pd.concat(
            cluster_tables,
            ignore_index=True,
        )
        if any(
            len(
                table
            )
            for table in cluster_tables
        )
        else pd.DataFrame()
    )

    omnibus_results.to_csv(
        OUTPUT_ROOT
        / "temporal_omnibus.csv",
        index=False,
    )

    significant_clusters.to_csv(
        OUTPUT_ROOT
        / "significant_clusters.csv",
        index=False,
    )

    timecourse_rows = []

    for (
        family_name,
        family_values,
    ) in [
        (
            "main_effect_controlled_interaction",
            primary_scores,
        ),
        (
            "main_effect_plus_rt_controlled_interaction",
            rt_adjusted_scores,
        ),
        (
            "rt_adjusted_minus_primary_fisher_z",
            rt_adjusted_minus_primary,
        ),
    ]:
        group_mean = family_values.mean(
            axis=0
        )

        group_sem = stats.sem(
            family_values,
            axis=0,
        )

        for (
            time_index,
            time_value,
        ) in enumerate(
            reference_times
        ):
            timecourse_rows.append(
                {
                    "family":
                        family_name,
                    "time_s":
                        float(
                            time_value
                        ),
                    "mean_score":
                        float(
                            group_mean[
                                time_index
                            ]
                        ),
                    "sem_score":
                        float(
                            group_sem[
                                time_index
                            ]
                        ),
                    "chance":
                        0.0,
                }
            )

    pd.DataFrame(
        timecourse_rows
    ).to_csv(
        OUTPUT_ROOT
        / "group_timecourses.csv",
        index=False,
    )

    target_summary_rows = []
    rt_summary_rows = []

    for (
        participant,
        participant_result,
    ) in zip(
        participants,
        participant_results,
    ):
        target_summary_rows.append(
            {
                "participant":
                    participant,
                "corr_raw_interaction_with_pain":
                    float(
                        participant_result[
                            "corr_raw_interaction_with_pain"
                        ]
                    ),
                "corr_raw_interaction_with_money":
                    float(
                        participant_result[
                            "corr_raw_interaction_with_money"
                        ]
                    ),
                "corr_conditional_interaction_with_pain":
                    float(
                        participant_result[
                            "corr_conditional_interaction_with_pain"
                        ]
                    ),
                "corr_conditional_interaction_with_money":
                    float(
                        participant_result[
                            "corr_conditional_interaction_with_money"
                        ]
                    ),
                "corr_raw_with_conditional_interaction":
                    float(
                        participant_result[
                            "corr_raw_with_conditional_interaction"
                        ]
                    ),
            }
        )

        rt_summary_rows.append(
            {
                "participant":
                    participant,
                "n_trials":
                    int(
                        participant_result[
                            "n_trials"
                        ]
                    ),
                "rt_min_s":
                    float(
                        participant_result[
                            "rt_min"
                        ]
                    ),
                "rt_p05_s":
                    float(
                        participant_result[
                            "rt_p05"
                        ]
                    ),
                "rt_median_s":
                    float(
                        participant_result[
                            "rt_median"
                        ]
                    ),
            }
        )

    pd.DataFrame(
        target_summary_rows
    ).to_csv(
        OUTPUT_ROOT
        / "participant_interaction_target_summary.csv",
        index=False,
    )

    pd.DataFrame(
        rt_summary_rows
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

    parameter_record = {
        "scientific_question":
            (
                "Does cue-locked signed-voltage EEG contain Pain x Money "
                "interaction information beyond additive pain and money "
                "main effects?"
            ),
        "interaction_definition":
            "(painlevel - 3) * (moneylevel - 3)",
        "neural_feature":
            (
                "baseline-corrected signed EEG voltage, averaged into the "
                "same 20-ms bins as decision TFR"
            ),
        "primary_analysis":
            (
                "leave-one-block-out interaction decoding after training-fold "
                "residualisation of BOTH EEG features and interaction target "
                "against centered pain and centered money"
            ),
        "rt_sensitivity":
            (
                "same conditional interaction target; EEG features additionally "
                "residualised within training fold for log(RT), log(RT)^2, and "
                "response-has-occurred by end of current 20-ms voltage bin"
            ),
        "direct_rt_comparison":
            (
                "one-sided temporal cluster test of Fisher-z(RT-adjusted r) "
                "minus Fisher-z(primary r)"
            ),
        "decoder":
            "StandardScaler -> Ridge(alpha=1)",
        "score":
            (
                "pooled held-out Pearson r against fold-specific conditional "
                "interaction target"
            ),
        "pain_design_center":
            PAIN_DESIGN_CENTER,
        "money_design_center":
            MONEY_DESIGN_CENTER,
        "formal_inference_s": [
            INFERENCE_TMIN,
            INFERENCE_TMAX,
        ],
        "n_permutations":
            n_permutations,
        "trial_set_note":
            (
                "Uses existing cue-locked decision TFR trial sets unchanged. "
                "No new trial exclusions are introduced."
            ),
        "interpretation_note":
            (
                "A significant centered-product result is evidence for joint/"
                "non-additive Pain x Money information in signed voltage, not "
                "yet evidence for a participant-specific subjective-value "
                "representation."
            ),
        "posthoc_context_note":
            (
                "This signed-voltage analysis was motivated after observing "
                "the cue-locked TFR interaction result and should therefore be "
                "described as a cross-format follow-up / convergence test, "
                "not as an independent preregistered replication."
            ),
    }

    (
        OUTPUT_ROOT
        / "params.json"
    ).write_text(
        json.dumps(
            parameter_record,
            indent=2,
        )
    )

    print(
        "\nCUE-LOCKED PAIN x MONEY INTERACTION: SIGNED VOLTAGE",
        flush=True,
    )

    print(
        omnibus_results.to_string(
            index=False
        ),
        flush=True,
    )

    if len(
        significant_clusters
    ):
        print(
            "\nSIGNIFICANT TEMPORAL CLUSTERS",
            flush=True,
        )

        print(
            significant_clusters.to_string(
                index=False
            ),
            flush=True,
        )

    print(
        f"\nOutputs: {OUTPUT_ROOT}",
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
        "=" * 78,
        flush=True,
    )

    print(
        "PAINREWARD CUE-LOCKED SIGNED-VOLTAGE PAIN x MONEY INTERACTION",
        flush=True,
    )

    print(
        f"Participants: {len(participants)}",
        flush=True,
    )

    print(
        "Interaction: (pain - 3) * (money - 3)",
        flush=True,
    )

    print(
        "Primary: interaction conditional on additive Pain + Money main effects",
        flush=True,
    )

    print(
        "Sensitivity: same target + RT-related EEG feature residualisation",
        flush=True,
    )

    print(
        f"Formal inference: {INFERENCE_TMIN:.2f}..{INFERENCE_TMAX:.2f} s",
        flush=True,
    )

    print(
        f"Output: {OUTPUT_ROOT}",
        flush=True,
    )

    print(
        "=" * 78,
        flush=True,
    )

    if not args.group_only:
        for (
            participant_index,
            participant,
        ) in enumerate(
            participants,
            start=1,
        ):
            score_path = participant_score_path(
                participant
            )

            if (
                score_path.exists()
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

            decode_one_participant(
                participant
            )

            print(
                f"[{participant}] DONE",
                flush=True,
            )

    print(
        "\nRunning group-level signed-voltage interaction inference...",
        flush=True,
    )

    run_group_analysis(
        participants=
            participants,
        n_permutations=
            args.n_permutations,
        n_jobs=
            args.n_jobs,
    )

    print(
        "\nCUE-LOCKED SIGNED-VOLTAGE PAIN x MONEY INTERACTION COMPLETE",
        flush=True,
    )


if __name__ == "__main__":
    main()
