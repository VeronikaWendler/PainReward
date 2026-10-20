#!/usr/bin/env python3
"""
05p_eeg_tfr_decode_decision_response_locked.py
==============================================

Response-locked decision-phase TFR decoding for the PainReward project.

Scientific question
-------------------
When decision EEG is aligned to the actual response, does the multichannel
spectral-power pattern contain information about:

    1. pain magnitude (levels 1-5), and
    2. money magnitude (levels 1-5)?

The main motivation is to test whether a magnitude representation that is weak
under cue locking becomes detectable when neural activity is aligned to the
response. This is tested symmetrically for pain and money.

Primary analysis
----------------
RAW response-locked decoding.

For each participant, frequency, and response-relative time bin:

    multichannel TFR power -> target level (1-5)

using:

    StandardScaler -> Ridge(alpha=1)

Cross-validation is leave-one-decision-block-out. Predictions from all held-out
blocks are concatenated and scored with Pearson r against the true target.

RT sensitivity
--------------
The complete decoding analysis is repeated after removing EEG feature variance
associated with:

    - log(RT)
    - log(RT)^2

The nuisance model is estimated on TRAINING BLOCKS ONLY and then applied to the
training and held-out block. This prevents held-out EEG from contributing to
the nuisance fit.

There is deliberately NO response-overlap nuisance term here. In a genuinely
response-locked analysis every trial is already at the same response-relative
latency.

RT adjustment remains a sensitivity analysis rather than the primary analysis
because RT is itself an outcome of the decision process.

Direct RAW-vs-RT-adjusted comparison
------------------------------------
In addition to testing RAW and RT-adjusted decoding separately, the script
directly tests:

    Fisher-z(r_RT-adjusted) - Fisher-z(r_RAW) > 0

at the group level.

This avoids interpreting "significant after RT adjustment but not before" as
itself evidence that RT adjustment significantly improved decoding.

Frequency analyses
------------------
Primary frequency-resolved surface:
    4-45 Hz x response-relative time

Secondary canonical bands:
    theta       4-7 Hz
    alpha       8-13 Hz
    beta       14-30 Hz
    low gamma  31-45 Hz

The frequency grid and band definitions match the completed cue-locked
decision TFR decoding.

Timing inference window
-----------------------
The response TFR files contain -0.8..+0.2 s relative to the response, but the
formal inferential interval is chosen BEFORE looking at decoding values.

For every retained trial, TFR preparation saved the cue time relative
to the response. A nominal response-relative TFR centre tau counts as post-cue
when:

    event_RT + tau >= 0

The formal inference start is the first response-time bin at which at least
95% of retained trials WITHIN EVERY PARTICIPANT have nominal TFR centres after
cue onset. The formal end is the last available bin at or before response
onset.

The stricter full-Morlet-support-after-cue quantity is retained as timing QC
but does not define the formal window, because that rule is stricter than the
completed cue-locked analysis and is infeasible for the fastest responses.

The TFR values outside that interval are retained in participant score files
for plotting/QC but are not used for cluster inference.

Group inference
---------------
Frequency x time:
    - one-sided positive cluster permutation test for RAW decoding
    - one-sided positive cluster permutation test for RT-adjusted decoding
    - one-sided positive cluster permutation test for RT-adjusted minus RAW
    - pain and money tested separately
    - Holm correction across pain/money within each family

Canonical bands:
    - one-sided positive temporal cluster tests
    - Holm across theta/alpha/beta/low_gamma separately within each attribute
      and family

Significant clusters are labelled:
    - fully_pre_response:
          cluster end <= -Morlet half-support
    - peri_response:
          otherwise

This prevents a nominally pre-response TFR centre whose wavelet extends through
the button press from being described as wholly pre-response.

Inputs
------
derivatives/<sub>/eeg/tfr_decision_response/
    <sub>_decision_response_power.npy
    <sub>_decision_response_times.npy
    <sub>_decision_response_freqs.npy
    <sub>_decision_response_metadata.csv

Outputs
-------
derivatives/statistics/tfr_mvpa_decision_response_locked/
    participant_scores/
    group_score_arrays.npz
    timing_inference_window.json
    response_timing_coverage_by_time.csv

    surface_omnibus_holm.csv
    surface_significant_clusters.csv
    band_omnibus_holm.csv
    band_significant_clusters.csv

    rt_adjustment_difference_surface_omnibus_holm.csv
    rt_adjustment_difference_surface_clusters.csv
    rt_adjustment_difference_band_omnibus_holm.csv
    rt_adjustment_difference_band_clusters.csv

    participant_rt_summary.csv
    params.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from mne.stats import combine_adjacency, permutation_cluster_1samp_test
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# =============================================================================
# Project paths and fixed analysis choices
# =============================================================================

BASE_PATH = Path(
    os.getenv(
        "basepath",
        Path(__file__).parent.parent.parent,
    )
)

PARTICIPANTS_TSV = BASE_PATH / "participants.tsv"

OUTPUT_ROOT = (
    BASE_PATH
    / "derivatives"
    / "statistics"
    / "tfr_mvpa_decision_response_locked"
)

PARTICIPANT_SCORE_DIR = (
    OUTPUT_ROOT
    / "participant_scores"
)

ATTRIBUTES = [
    "pain",
    "money",
]

CANONICAL_BANDS: Dict[str, Tuple[float, float]] = {
    "theta": (4.0, 7.0),
    "alpha": (8.0, 13.0),
    "beta": (14.0, 30.0),
    "low_gamma": (31.0, 45.0),
}

RIDGE_ALPHA = 1.0
RANDOM_STATE = 23
CLUSTER_FORMING_ALPHA = 0.05

# Response onset is the upper bound of formal inference.
INFERENCE_TMAX_TARGET_S = 0.0

# Timing-only criterion used to define the formal response-relative window.
#
# At every formally tested response-relative time bin, at least 95% of the
# retained trials WITHIN EVERY PARTICIPANT must have their nominal TFR centre
# after cue onset. This criterion is determined from event timing only, before
# any neural decoding results are inspected.
TIMING_COVERAGE_THRESHOLD = 0.95

# Same conservative project-wide Morlet half-support used during TFR prep:
# n_cycles = f/2 -> 5*sigma_t = 5*(0.5/(2*pi)) ~= 0.398 s.
MORLET_HALF_SUPPORT_S = (
    5.0
    * (
        0.5
        / (
            2.0
            * np.pi
        )
    )
)


# =============================================================================
# Command-line arguments
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Response-locked decision TFR decoding for pain and money."
        )
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
            int(
                os.getenv(
                    "SLURM_CPUS_PER_TASK",
                    "1",
                )
            ),
        ),
        help="Parallel jobs used by MNE permutation tests.",
    )

    parser.add_argument(
        "--overwrite-scores",
        action="store_true",
        help="Recompute participant decoding scores if they already exist.",
    )

    parser.add_argument(
        "--group-only",
        action="store_true",
        help="Skip participant decoding and recompute group inference only.",
    )

    return parser.parse_args()


# =============================================================================
# Input/output paths
# =============================================================================

def response_tfr_paths(
    participant: str,
) -> Dict[str, Path]:
    tfr_directory = (
        BASE_PATH
        / "derivatives"
        / participant
        / "eeg"
        / "tfr_decision_response"
    )

    stem = (
        f"{participant}_decision_response"
    )

    return {
        "power":
            tfr_directory
            / f"{stem}_power.npy",
        "times":
            tfr_directory
            / f"{stem}_times.npy",
        "freqs":
            tfr_directory
            / f"{stem}_freqs.npy",
        "metadata":
            tfr_directory
            / f"{stem}_metadata.csv",
        "params":
            tfr_directory
            / f"{stem}_params.json",
        "summary":
            tfr_directory
            / f"{stem}_prep_summary.json",
    }


def participant_score_path(
    participant: str,
) -> Path:
    return (
        PARTICIPANT_SCORE_DIR
        / f"{participant}_response_locked_mvpa.npz"
    )


# =============================================================================
# Participant discovery and input checks
# =============================================================================

def load_participant_list() -> List[str]:
    if PARTICIPANTS_TSV.exists():
        participants_table = pd.read_csv(
            PARTICIPANTS_TSV,
            sep="\t",
        )

        if "participant_id" not in participants_table.columns:
            raise KeyError(
                f"{PARTICIPANTS_TSV} has no participant_id column."
            )

        candidates = sorted(
            participants_table[
                "participant_id"
            ]
            .astype(str)
            .tolist()
        )
    else:
        candidates = sorted(
            directory.name
            for directory in BASE_PATH.iterdir()
            if directory.name.startswith("sub-")
            and directory.is_dir()
        )

    participants = []

    for participant in candidates:
        paths = response_tfr_paths(
            participant
        )

        required = [
            paths["power"],
            paths["times"],
            paths["freqs"],
            paths["metadata"],
        ]

        if all(
            path.exists()
            for path in required
        ):
            participants.append(
                participant
            )

    if not participants:
        raise RuntimeError(
            "No participants have a complete response-locked TFR input set."
        )

    return participants


# =============================================================================
# Timing-only definition of formal inference window
# =============================================================================

def derive_response_inference_window(
    participants: Sequence[str],
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Define the response-relative formal inference interval using timing only.

    Why nominal TFR centres?
    ------------------------
    The response-locked TFR is intentionally matched to the completed
    cue-locked TFR analysis. In that cue-locked analysis, a nominal centre at
    cue onset was analysed even though the symmetric Morlet kernel extends
    around that centre.

    We therefore do NOT require the entire Morlet kernel to lie after cue
    onset. That criterion is too strict for fast-response trials and would make
    the response-locked analysis impossible.

    Instead, for a response-relative TFR centre tau, a trial counts as
    post-cue when:

        event_RT + tau >= 0

    Equivalently, the cue itself occurred at response-relative time:

        cue_time_relative_to_response = -event_RT

    Formal inference requires at least TIMING_COVERAGE_THRESHOLD of retained
    trials WITHIN EVERY PARTICIPANT to have a nominal centre after cue onset.

    The end of formal inference is the last available bin at or before the
    response. No neural decoding values enter this timing decision.

    We additionally retain the stricter full-Morlet-support coverage as a
    descriptive QC measure only.

    Returns
    -------
    inference_tmin
    inference_tmax
    reference_times
    minimum_participant_nominal_center_coverage_by_time
    """

    reference_times = None

    nominal_postcue_start_values_by_participant = []
    full_support_postcue_start_values_by_participant = []
    timing_rows = []

    for participant in participants:
        paths = response_tfr_paths(
            participant
        )

        times = np.load(
            paths["times"]
        ).astype(
            float
        )

        if reference_times is None:
            reference_times = times
        elif not np.allclose(
            times,
            reference_times,
            atol=1e-12,
            rtol=0.0,
        ):
            raise RuntimeError(
                f"{participant}: response TFR time grid differs "
                f"from the first participant."
            )

        metadata = pd.read_csv(
            paths["metadata"]
        )

        required_columns = [
            "cue_time_relative_to_response_s",
            "earliest_center_full_morlet_support_after_cue_s",
            "event_rt_s",
        ]

        missing_columns = [
            column
            for column in required_columns
            if column not in metadata.columns
        ]

        if missing_columns:
            raise KeyError(
                f"{participant}: response metadata lacks "
                f"{missing_columns}."
            )

        nominal_postcue_start = pd.to_numeric(
            metadata[
                "cue_time_relative_to_response_s"
            ],
            errors="coerce",
        ).to_numpy(
            dtype=float
        )

        full_support_postcue_start = pd.to_numeric(
            metadata[
                "earliest_center_full_morlet_support_after_cue_s"
            ],
            errors="coerce",
        ).to_numpy(
            dtype=float
        )

        event_rt = pd.to_numeric(
            metadata[
                "event_rt_s"
            ],
            errors="coerce",
        ).to_numpy(
            dtype=float
        )

        if not (
            np.isfinite(
                nominal_postcue_start
            ).all()
            and np.isfinite(
                full_support_postcue_start
            ).all()
            and np.isfinite(
                event_rt
            ).all()
        ):
            raise RuntimeError(
                f"{participant}: non-finite timing values."
            )

        # Internal consistency:
        # cue-relative-to-response should equal -event RT.
        maximum_identity_error = float(
            np.max(
                np.abs(
                    nominal_postcue_start
                    + event_rt
                )
            )
        )

        if maximum_identity_error > 1e-9:
            raise RuntimeError(
                f"{participant}: cue/response timing identity failed; "
                f"max |-RT - cue_time| = {maximum_identity_error:.3e}."
            )

        nominal_postcue_start_values_by_participant.append(
            nominal_postcue_start
        )

        full_support_postcue_start_values_by_participant.append(
            full_support_postcue_start
        )

        timing_rows.append(
            {
                "participant":
                    participant,
                "n_trials":
                    int(
                        len(
                            event_rt
                        )
                    ),
                "event_rt_min_s":
                    float(
                        np.min(
                            event_rt
                        )
                    ),
                "event_rt_p05_s":
                    float(
                        np.quantile(
                            event_rt,
                            0.05,
                        )
                    ),
                "event_rt_median_s":
                    float(
                        np.median(
                            event_rt
                        )
                    ),
                "nominal_center_95pct_postcue_start_s":
                    float(
                        np.quantile(
                            nominal_postcue_start,
                            TIMING_COVERAGE_THRESHOLD,
                        )
                    ),
                "full_support_95pct_postcue_start_s":
                    float(
                        np.quantile(
                            full_support_postcue_start,
                            TIMING_COVERAGE_THRESHOLD,
                        )
                    ),
            }
        )

    if reference_times is None:
        raise RuntimeError(
            "No response-time grids were loaded."
        )

    # -------------------------------------------------------------------------
    # Coverage of nominal TFR centres after cue onset.
    # -------------------------------------------------------------------------

    nominal_center_coverage_by_participant = np.stack(
        [
            np.asarray(
                [
                    np.mean(
                        participant_start_values
                        <= time_point + 1e-12
                    )
                    for time_point in reference_times
                ],
                dtype=float,
            )
            for participant_start_values
            in nominal_postcue_start_values_by_participant
        ],
        axis=0,
    )

    minimum_participant_nominal_center_coverage = (
        nominal_center_coverage_by_participant.min(
            axis=0
        )
    )

    pooled_nominal_start_values = np.concatenate(
        nominal_postcue_start_values_by_participant
    )

    pooled_nominal_center_coverage = np.asarray(
        [
            np.mean(
                pooled_nominal_start_values
                <= time_point + 1e-12
            )
            for time_point in reference_times
        ],
        dtype=float,
    )

    # -------------------------------------------------------------------------
    # Stricter full-Morlet-support coverage, saved only as QC.
    # -------------------------------------------------------------------------

    full_support_coverage_by_participant = np.stack(
        [
            np.asarray(
                [
                    np.mean(
                        participant_start_values
                        <= time_point + 1e-12
                    )
                    for time_point in reference_times
                ],
                dtype=float,
            )
            for participant_start_values
            in full_support_postcue_start_values_by_participant
        ],
        axis=0,
    )

    minimum_participant_full_support_coverage = (
        full_support_coverage_by_participant.min(
            axis=0
        )
    )

    pooled_full_support_start_values = np.concatenate(
        full_support_postcue_start_values_by_participant
    )

    pooled_full_support_coverage = np.asarray(
        [
            np.mean(
                pooled_full_support_start_values
                <= time_point + 1e-12
            )
            for time_point in reference_times
        ],
        dtype=float,
    )

    # -------------------------------------------------------------------------
    # Freeze formal inference window from timing only.
    # -------------------------------------------------------------------------

    pre_response_mask = (
        reference_times
        <= INFERENCE_TMAX_TARGET_S + 1e-12
    )

    meets_nominal_timing_criterion = (
        minimum_participant_nominal_center_coverage
        >= TIMING_COVERAGE_THRESHOLD - 1e-12
    )

    valid_inference_mask = (
        pre_response_mask
        & meets_nominal_timing_criterion
    )

    if not np.any(
        valid_inference_mask
    ):
        maximum_pre_response_coverage = float(
            minimum_participant_nominal_center_coverage[
                pre_response_mask
            ].max()
        )

        raise RuntimeError(
            "No pre-response TFR bin reaches the pre-specified "
            f"{TIMING_COVERAGE_THRESHOLD:.0%} within-participant "
            "nominal-centre-after-cue criterion. "
            "Maximum minimum-participant coverage before response = "
            f"{maximum_pre_response_coverage:.3f}. "
            "Do not relax this based on EEG results."
        )

    valid_inference_times = reference_times[
        valid_inference_mask
    ]

    inference_tmin = float(
        valid_inference_times[
            0
        ]
    )

    inference_tmax = float(
        valid_inference_times[
            -1
        ]
    )

    # -------------------------------------------------------------------------
    # Save timing diagnostics before any decoding begins.
    # -------------------------------------------------------------------------

    coverage_table = pd.DataFrame(
        {
            "response_time_s":
                reference_times,
            "pooled_fraction_nominal_centers_after_cue":
                pooled_nominal_center_coverage,
            "minimum_participant_fraction_nominal_centers_after_cue":
                minimum_participant_nominal_center_coverage,
            "meets_95pct_nominal_center_within_every_participant":
                meets_nominal_timing_criterion,
            "pooled_fraction_full_morlet_support_after_cue":
                pooled_full_support_coverage,
            "minimum_participant_fraction_full_morlet_support_after_cue":
                minimum_participant_full_support_coverage,
            "inside_formal_inference_window":
                valid_inference_mask,
            "fully_pre_response_under_conservative_morlet_support":
                (
                    reference_times
                    <= -MORLET_HALF_SUPPORT_S + 1e-12
                ),
        }
    )

    coverage_table.to_csv(
        OUTPUT_ROOT
        / "response_timing_coverage_by_time.csv",
        index=False,
    )

    pd.DataFrame(
        timing_rows
    ).to_csv(
        OUTPUT_ROOT
        / "response_timing_summary_by_participant.csv",
        index=False,
    )

    n_formal_bins_fully_pre_response = int(
        np.sum(
            valid_inference_mask
            & (
                reference_times
                <= -MORLET_HALF_SUPPORT_S + 1e-12
            )
        )
    )

    timing_record = {
        "selection_basis":
            (
                "timing only; no neural decoding values used"
            ),
        "formal_window_policy":
            (
                "first through last pre-response bins for which at least "
                f"{TIMING_COVERAGE_THRESHOLD:.0%} of retained trials within "
                "every participant have nominal TFR centres after cue onset"
            ),
        "nominal_center_definition":
            (
                "event_RT + response_relative_time >= 0"
            ),
        "full_morlet_support_policy":
            (
                "saved as descriptive QC only; not used to define the formal "
                "window because that criterion is stricter than the completed "
                "cue-locked analysis and is infeasible for fast-response trials"
            ),
        "n_participants":
            int(
                len(
                    participants
                )
            ),
        "n_trials_total":
            int(
                len(
                    pooled_nominal_start_values
                )
            ),
        "timing_coverage_threshold":
            TIMING_COVERAGE_THRESHOLD,
        "morlet_half_support_s":
            MORLET_HALF_SUPPORT_S,
        "formal_inference_tmin_s":
            inference_tmin,
        "formal_inference_tmax_s":
            inference_tmax,
        "n_formal_bins":
            int(
                np.sum(
                    valid_inference_mask
                )
            ),
        "n_formal_bins_fully_pre_response_under_conservative_support":
            n_formal_bins_fully_pre_response,
        "saved_response_time_min_s":
            float(
                reference_times.min()
            ),
        "saved_response_time_max_s":
            float(
                reference_times.max()
            ),
        "fully_pre_response_interpretation_requires_center_at_or_before_s":
            -MORLET_HALF_SUPPORT_S,
    }

    (
        OUTPUT_ROOT
        / "timing_inference_window.json"
    ).write_text(
        json.dumps(
            timing_record,
            indent=2,
        )
    )

    print(
        "\nTIMING-ONLY FORMAL INFERENCE WINDOW",
        flush=True,
    )

    print(
        f"  criterion: >= "
        f"{TIMING_COVERAGE_THRESHOLD:.0%} nominal-centre-after-cue "
        f"coverage within EVERY participant",
        flush=True,
    )

    print(
        f"  snapped formal window: "
        f"{inference_tmin:.3f} .. {inference_tmax:.3f} s "
        f"relative to response",
        flush=True,
    )

    print(
        f"  formal bins: "
        f"{int(np.sum(valid_inference_mask))}",
        flush=True,
    )

    print(
        f"  conservative Morlet half-support: "
        f"{MORLET_HALF_SUPPORT_S:.3f} s",
        flush=True,
    )

    print(
        f"  formal bins fully pre-response under that support: "
        f"{n_formal_bins_fully_pre_response}",
        flush=True,
    )

    if n_formal_bins_fully_pre_response == 0:
        print(
            "  NOTE: all formal bins are peri-response under the conservative "
            "Morlet-support definition; do not describe significant clusters "
            "as wholly pre-response.",
            flush=True,
        )

    return (
        inference_tmin,
        inference_tmax,
        reference_times,
        minimum_participant_nominal_center_coverage,
    )


# =============================================================================
# Decoder and scoring helpers
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
    if (
        np.std(
            true_values
        ) == 0
        or np.std(
            predicted_values
        ) == 0
    ):
        return np.nan

    return float(
        np.corrcoef(
            true_values,
            predicted_values,
        )[
            0,
            1,
        ]
    )


# =============================================================================
# Leakage-safe RT sensitivity
# =============================================================================

def make_rt_nuisance_matrix(
    reaction_time: np.ndarray,
) -> np.ndarray:
    """
    Response-locked RT nuisance matrix.

    Columns:
        1. log(RT)
        2. log(RT)^2

    No response-overlap term is included because every trial is already aligned
    to the actual response.
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
        ]
    )


def residualize_features_using_training_fold(
    train_features: np.ndarray,
    test_features: np.ndarray,
    train_nuisance_raw: np.ndarray,
    test_nuisance_raw: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Regress nuisance variables from EEG features without test-fold leakage.
    """

    nuisance_mean = train_nuisance_raw.mean(
        axis=0
    )

    nuisance_sd = train_nuisance_raw.std(
        axis=0
    )

    varying_columns = (
        nuisance_sd
        > 1e-12
    )

    if not np.any(
        varying_columns
    ):
        return (
            train_features,
            test_features,
        )

    standardized_train_nuisance = (
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

    standardized_test_nuisance = (
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
                len(
                    standardized_train_nuisance
                )
            ),
            standardized_train_nuisance,
        ]
    )

    test_design = np.column_stack(
        [
            np.ones(
                len(
                    standardized_test_nuisance
                )
            ),
            standardized_test_nuisance,
        ]
    )

    nuisance_coefficients, *_ = np.linalg.lstsq(
        train_design,
        train_features,
        rcond=None,
    )

    residualized_train = (
        train_features
        - train_design
        @ nuisance_coefficients
    )

    residualized_test = (
        test_features
        - test_design
        @ nuisance_coefficients
    )

    return (
        residualized_train,
        residualized_test,
    )


# =============================================================================
# Leave-one-decision-block-out decoding
# =============================================================================

def leave_one_block_out_score(
    features: np.ndarray,
    target: np.ndarray,
    block_ids: np.ndarray,
    reaction_time: np.ndarray,
    remove_rt_related_variance: bool,
) -> float:
    """
    Decode one target from one trial x channel feature matrix.
    """

    predictions = np.full(
        len(
            target
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
            "Need at least two decision blocks for GroupKFold."
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
                ]
            )

            test_nuisance = make_rt_nuisance_matrix(
                reaction_time[
                    test_indices
                ]
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
            "Cross-validation produced non-finite predictions."
        )

    score = pearson_correlation(
        true_values=target,
        predicted_values=predictions,
    )

    if not np.isfinite(
        score
    ):
        raise RuntimeError(
            "Decoding produced a non-finite Pearson correlation."
        )

    return score


# =============================================================================
# Participant-level decoding
# =============================================================================

def decode_one_participant(
    participant: str,
    overwrite_scores: bool,
) -> None:
    score_path = participant_score_path(
        participant
    )

    if (
        score_path.exists()
        and not overwrite_scores
    ):
        print(
            f"    {participant}: score file exists; skipping",
            flush=True,
        )
        return

    paths = response_tfr_paths(
        participant
    )

    power = np.load(
        paths["power"],
        mmap_mode="r",
    )

    times = np.load(
        paths["times"]
    ).astype(
        float
    )

    frequencies = np.load(
        paths["freqs"]
    ).astype(
        float
    )

    metadata = pd.read_csv(
        paths["metadata"]
    )

    if power.ndim != 4:
        raise RuntimeError(
            f"{participant}: expected 4D power array, got {power.shape}."
        )

    if power.shape[0] != len(
        metadata
    ):
        raise RuntimeError(
            f"{participant}: power trials {power.shape[0]} != "
            f"metadata rows {len(metadata)}."
        )

    if power.shape[2] != len(
        frequencies
    ):
        raise RuntimeError(
            f"{participant}: frequency dimension mismatch."
        )

    if power.shape[3] != len(
        times
    ):
        raise RuntimeError(
            f"{participant}: time dimension mismatch."
        )

    required_columns = [
        "painlevel",
        "moneylevel",
        "block",
        "behavioral_rt_s",
        "event_rt_s",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in metadata.columns
    ]

    if missing_columns:
        raise KeyError(
            f"{participant}: response metadata missing "
            f"{missing_columns}."
        )

    pain_levels = pd.to_numeric(
        metadata[
            "painlevel"
        ],
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    money_levels = pd.to_numeric(
        metadata[
            "moneylevel"
        ],
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    block_ids = pd.to_numeric(
        metadata[
            "block"
        ],
        errors="coerce",
    ).to_numpy(
        dtype=int
    )

    # Keep the behavioural RT for exact comparability with the earlier
    # cue-locked RT-sensitivity analysis.
    reaction_time = pd.to_numeric(
        metadata[
            "behavioral_rt_s"
        ],
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    event_rt = pd.to_numeric(
        metadata[
            "event_rt_s"
        ],
        errors="coerce",
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
        and np.isfinite(
            event_rt
        ).all()
    ):
        raise RuntimeError(
            f"{participant}: non-finite pain, money, or RT metadata."
        )

    if np.any(
        reaction_time
        <= 0
    ):
        raise RuntimeError(
            f"{participant}: RT must be positive for log(RT)."
        )

    targets = {
        "pain":
            pain_levels,
        "money":
            money_levels,
    }

    n_attributes = len(
        ATTRIBUTES
    )

    n_frequencies = len(
        frequencies
    )

    n_times = len(
        times
    )

    raw_frequency_scores = np.empty(
        (
            n_attributes,
            n_frequencies,
            n_times,
        ),
        dtype=np.float32,
    )

    rt_residualized_frequency_scores = np.empty_like(
        raw_frequency_scores
    )

    # -------------------------------------------------------------------------
    # Full frequency x response-time surface
    # -------------------------------------------------------------------------

    for (
        attribute_index,
        attribute_name,
    ) in enumerate(
        ATTRIBUTES
    ):
        target = targets[
            attribute_name
        ]

        print(
            f"    {participant}: {attribute_name} "
            f"frequency x response-time decoding",
            flush=True,
        )

        for frequency_index in range(
            n_frequencies
        ):
            for time_index in range(
                n_times
            ):
                channel_features = np.asarray(
                    power[
                        :,
                        :,
                        frequency_index,
                        time_index,
                    ],
                    dtype=float,
                )

                raw_frequency_scores[
                    attribute_index,
                    frequency_index,
                    time_index,
                ] = leave_one_block_out_score(
                    features=channel_features,
                    target=target,
                    block_ids=block_ids,
                    reaction_time=reaction_time,
                    remove_rt_related_variance=False,
                )

                rt_residualized_frequency_scores[
                    attribute_index,
                    frequency_index,
                    time_index,
                ] = leave_one_block_out_score(
                    features=channel_features,
                    target=target,
                    block_ids=block_ids,
                    reaction_time=reaction_time,
                    remove_rt_related_variance=True,
                )

    # -------------------------------------------------------------------------
    # Canonical-band response-time courses
    # -------------------------------------------------------------------------

    band_names = list(
        CANONICAL_BANDS
    )

    raw_band_scores = np.empty(
        (
            n_attributes,
            len(
                band_names
            ),
            n_times,
        ),
        dtype=np.float32,
    )

    rt_residualized_band_scores = np.empty_like(
        raw_band_scores
    )

    for (
        attribute_index,
        attribute_name,
    ) in enumerate(
        ATTRIBUTES
    ):
        target = targets[
            attribute_name
        ]

        print(
            f"    {participant}: {attribute_name} canonical bands",
            flush=True,
        )

        for (
            band_index,
            band_name,
        ) in enumerate(
            band_names
        ):
            band_low, band_high = CANONICAL_BANDS[
                band_name
            ]

            frequency_mask = (
                (
                    frequencies
                    >= band_low
                )
                & (
                    frequencies
                    <= band_high
                )
            )

            if not frequency_mask.any():
                raise RuntimeError(
                    f"{participant}: no frequencies in band {band_name}."
                )

            for time_index in range(
                n_times
            ):
                # Index the frequency dimension after selecting the current
                # time so NumPy cannot reorder dimensions via advanced indexing.
                current_time_power = np.asarray(
                    power[
                        :,
                        :,
                        :,
                        time_index,
                    ],
                    dtype=float,
                )

                channel_features = (
                    current_time_power[
                        :,
                        :,
                        frequency_mask,
                    ]
                    .mean(
                        axis=2
                    )
                )

                raw_band_scores[
                    attribute_index,
                    band_index,
                    time_index,
                ] = leave_one_block_out_score(
                    features=channel_features,
                    target=target,
                    block_ids=block_ids,
                    reaction_time=reaction_time,
                    remove_rt_related_variance=False,
                )

                rt_residualized_band_scores[
                    attribute_index,
                    band_index,
                    time_index,
                ] = leave_one_block_out_score(
                    features=channel_features,
                    target=target,
                    block_ids=block_ids,
                    reaction_time=reaction_time,
                    remove_rt_related_variance=True,
                )

    np.savez_compressed(
        score_path,
        raw_freq_r=
            raw_frequency_scores,
        rtres_freq_r=
            rt_residualized_frequency_scores,
        raw_band_r=
            raw_band_scores,
        rtres_band_r=
            rt_residualized_band_scores,
        times=
            times,
        freqs=
            frequencies,
        band_names=
            np.asarray(
                band_names
            ),
        attribute_names=
            np.asarray(
                ATTRIBUTES
            ),
        n_trials=
            np.asarray(
                len(
                    metadata
                )
            ),
        behavioral_rt_min=
            np.asarray(
                reaction_time.min()
            ),
        behavioral_rt_p05=
            np.asarray(
                np.quantile(
                    reaction_time,
                    0.05,
                )
            ),
        behavioral_rt_median=
            np.asarray(
                np.median(
                    reaction_time
                )
            ),
        behavioral_rt_p95=
            np.asarray(
                np.quantile(
                    reaction_time,
                    0.95,
                )
            ),
        event_rt_min=
            np.asarray(
                event_rt.min()
            ),
        event_rt_median=
            np.asarray(
                np.median(
                    event_rt
                )
            ),
        max_abs_behavior_event_rt_difference=
            np.asarray(
                np.max(
                    np.abs(
                        reaction_time
                        - event_rt
                    )
                )
            ),
    )

    print(
        f"    {participant}: scores saved",
        flush=True,
    )


# =============================================================================
# Multiple-comparison and transform helpers
# =============================================================================

def holm_correct(
    p_values: Sequence[float],
) -> np.ndarray:
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
    clipped = np.clip(
        correlation_values,
        -0.999999,
        0.999999,
    )

    return np.arctanh(
        clipped
    )


def one_sided_t_threshold(
    n_participants: int,
) -> float:
    return float(
        stats.t.ppf(
            1.0
            - CLUSTER_FORMING_ALPHA,
            df=
                n_participants
                - 1,
        )
    )


# =============================================================================
# Generic group cluster tests
# =============================================================================

def frequency_time_cluster_test(
    participant_values: np.ndarray,
    frequencies: np.ndarray,
    times: np.ndarray,
    inference_tmin: float,
    inference_tmax: float,
    n_permutations: int,
    n_jobs: int,
    fisher_transform_correlations: bool,
):
    """
    One-sided frequency x time cluster permutation test against zero.

    participant_values:
        participant x frequency x time

    When fisher_transform_correlations=True, values are individual-participant
    Pearson correlations and are Fisher-z transformed before group testing.

    When False, values are already on the group-test scale, e.g.:
        Fisher-z(r_RT) - Fisher-z(r_RAW)
    """

    time_mask = (
        (
            times
            >= inference_tmin - 1e-12
        )
        & (
            times
            <= inference_tmax + 1e-12
        )
    )

    inference_times = times[
        time_mask
    ]

    values = np.asarray(
        participant_values[
            :,
            :,
            time_mask,
        ],
        dtype=float,
    )

    if fisher_transform_correlations:
        values = fisher_z(
            values
        )

    if not np.isfinite(
        values
    ).all():
        raise RuntimeError(
            "Non-finite values entered frequency x time group inference."
        )

    (
        n_participants,
        n_frequencies,
        n_times,
    ) = values.shape

    t_values, clusters, cluster_p_values, _ = (
        permutation_cluster_1samp_test(
            values.reshape(
                n_participants,
                n_frequencies
                * n_times,
            ),
            threshold=one_sided_t_threshold(
                n_participants
            ),
            n_permutations=n_permutations,
            tail=1,
            adjacency=combine_adjacency(
                n_frequencies,
                n_times,
            ),
            seed=RANDOM_STATE,
            n_jobs=n_jobs,
            out_type="indices",
            verbose=False,
        )
    )

    reshaped_t_values = t_values.reshape(
        n_frequencies,
        n_times,
    )

    flattened_cluster_indices = [
        np.asarray(
            cluster[
                0
            ],
            dtype=int,
        )
        for cluster in clusters
    ]

    return (
        reshaped_t_values,
        flattened_cluster_indices,
        np.asarray(
            cluster_p_values,
            dtype=float,
        ),
        inference_times,
    )


def temporal_cluster_test(
    participant_values: np.ndarray,
    times: np.ndarray,
    inference_tmin: float,
    inference_tmax: float,
    n_permutations: int,
    n_jobs: int,
    fisher_transform_correlations: bool,
):
    """
    One-sided temporal cluster permutation test against zero.
    """

    time_mask = (
        (
            times
            >= inference_tmin - 1e-12
        )
        & (
            times
            <= inference_tmax + 1e-12
        )
    )

    inference_times = times[
        time_mask
    ]

    values = np.asarray(
        participant_values[
            :,
            time_mask,
        ],
        dtype=float,
    )

    if fisher_transform_correlations:
        values = fisher_z(
            values
        )

    if not np.isfinite(
        values
    ).all():
        raise RuntimeError(
            "Non-finite values entered temporal group inference."
        )

    n_participants = values.shape[
        0
    ]

    t_values, clusters, cluster_p_values, _ = (
        permutation_cluster_1samp_test(
            values,
            threshold=one_sided_t_threshold(
                n_participants
            ),
            n_permutations=n_permutations,
            tail=1,
            seed=RANDOM_STATE,
            n_jobs=n_jobs,
            out_type="indices",
            verbose=False,
        )
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
        t_values,
        cluster_indices,
        np.asarray(
            cluster_p_values,
            dtype=float,
        ),
        inference_times,
    )


def response_timing_label(
    cluster_end_s: float,
) -> str:
    if (
        cluster_end_s
        <= -MORLET_HALF_SUPPORT_S + 1e-12
    ):
        return (
            "fully_pre_response"
        )

    return (
        "peri_response"
    )


# =============================================================================
# Frequency x time summary
# =============================================================================

def summarize_frequency_time_family(
    group_values: np.ndarray,
    family_name: str,
    frequencies: np.ndarray,
    times: np.ndarray,
    inference_tmin: float,
    inference_tmax: float,
    n_permutations: int,
    n_jobs: int,
    fisher_transform_correlations: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    stored_results = {}
    domain_omnibus_p_values = []

    for (
        attribute_index,
        attribute_name,
    ) in enumerate(
        ATTRIBUTES
    ):
        result = frequency_time_cluster_test(
            participant_values=
                group_values[
                    :,
                    attribute_index,
                ],
            frequencies=
                frequencies,
            times=
                times,
            inference_tmin=
                inference_tmin,
            inference_tmax=
                inference_tmax,
            n_permutations=
                n_permutations,
            n_jobs=
                n_jobs,
            fisher_transform_correlations=
                fisher_transform_correlations,
        )

        (
            _,
            _,
            cluster_p_values,
            _,
        ) = result

        minimum_cluster_p = (
            float(
                cluster_p_values.min()
            )
            if len(
                cluster_p_values
            )
            else 1.0
        )

        domain_omnibus_p_values.append(
            minimum_cluster_p
        )

        stored_results[
            attribute_name
        ] = result

    holm_adjusted_domain_p = holm_correct(
        domain_omnibus_p_values
    )

    omnibus_rows = []
    cluster_rows = []

    for (
        attribute_index,
        attribute_name,
    ) in enumerate(
        ATTRIBUTES
    ):
        adjusted_domain_p = float(
            holm_adjusted_domain_p[
                attribute_index
            ]
        )

        omnibus_rows.append(
            {
                "family":
                    family_name,
                "attribute":
                    attribute_name,
                "omnibus_p_within_surface":
                    domain_omnibus_p_values[
                        attribute_index
                    ],
                "omnibus_p_holm_across_pain_money":
                    adjusted_domain_p,
                "domain_significant_holm_0p05":
                    bool(
                        adjusted_domain_p
                        < 0.05
                    ),
                "formal_inference_tmin_s":
                    inference_tmin,
                "formal_inference_tmax_s":
                    inference_tmax,
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

        n_frequencies, n_times = t_values.shape

        if adjusted_domain_p >= 0.05:
            continue

        for (
            cluster_id,
            (
                flat_cluster_indices,
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

            (
                frequency_indices,
                time_indices,
            ) = np.unravel_index(
                flat_cluster_indices,
                (
                    n_frequencies,
                    n_times,
                ),
            )

            cluster_t_values = (
                t_values
                .reshape(
                    -1
                )[
                    flat_cluster_indices
                ]
            )

            peak_position = int(
                np.argmax(
                    cluster_t_values
                )
            )

            peak_flat_index = int(
                flat_cluster_indices[
                    peak_position
                ]
            )

            (
                peak_frequency_index,
                peak_time_index,
            ) = np.unravel_index(
                peak_flat_index,
                (
                    n_frequencies,
                    n_times,
                ),
            )

            cluster_start = float(
                inference_times[
                    time_indices
                ].min()
            )

            cluster_end = float(
                inference_times[
                    time_indices
                ].max()
            )

            cluster_rows.append(
                {
                    "family":
                        family_name,
                    "attribute":
                        attribute_name,
                    "cluster_id":
                        cluster_id,
                    "cluster_p_within_surface":
                        float(
                            cluster_p
                        ),
                    "freq_min_hz":
                        float(
                            frequencies[
                                frequency_indices
                            ].min()
                        ),
                    "freq_max_hz":
                        float(
                            frequencies[
                                frequency_indices
                            ].max()
                        ),
                    "time_start_s":
                        cluster_start,
                    "time_end_s":
                        cluster_end,
                    "peak_freq_hz":
                        float(
                            frequencies[
                                peak_frequency_index
                            ]
                        ),
                    "peak_time_s":
                        float(
                            inference_times[
                                peak_time_index
                            ]
                        ),
                    "peak_t":
                        float(
                            cluster_t_values[
                                peak_position
                            ]
                        ),
                    "response_timing_label":
                        response_timing_label(
                            cluster_end
                        ),
                    "omnibus_p_holm_across_pain_money":
                        adjusted_domain_p,
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
# Canonical-band summary
# =============================================================================

def summarize_band_family(
    group_values: np.ndarray,
    family_name: str,
    times: np.ndarray,
    band_names: Sequence[str],
    inference_tmin: float,
    inference_tmax: float,
    n_permutations: int,
    n_jobs: int,
    fisher_transform_correlations: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    omnibus_rows = []
    cluster_rows = []

    for (
        attribute_index,
        attribute_name,
    ) in enumerate(
        ATTRIBUTES
    ):
        stored_band_results = []
        band_omnibus_p_values = []

        for (
            band_index,
            band_name,
        ) in enumerate(
            band_names
        ):
            result = temporal_cluster_test(
                participant_values=
                    group_values[
                        :,
                        attribute_index,
                        band_index,
                    ],
                times=
                    times,
                inference_tmin=
                    inference_tmin,
                inference_tmax=
                    inference_tmax,
                n_permutations=
                    n_permutations,
                n_jobs=
                    n_jobs,
                fisher_transform_correlations=
                    fisher_transform_correlations,
            )

            (
                _,
                _,
                cluster_p_values,
                _,
            ) = result

            minimum_cluster_p = (
                float(
                    cluster_p_values.min()
                )
                if len(
                    cluster_p_values
                )
                else 1.0
            )

            band_omnibus_p_values.append(
                minimum_cluster_p
            )

            stored_band_results.append(
                (
                    band_name,
                    *result,
                )
            )

        holm_adjusted_band_p = holm_correct(
            band_omnibus_p_values
        )

        for (
            band_index,
            stored_result,
        ) in enumerate(
            stored_band_results
        ):
            (
                band_name,
                t_values,
                clusters,
                cluster_p_values,
                inference_times,
            ) = stored_result

            adjusted_band_p = float(
                holm_adjusted_band_p[
                    band_index
                ]
            )

            omnibus_rows.append(
                {
                    "family":
                        family_name,
                    "attribute":
                        attribute_name,
                    "band":
                        band_name,
                    "band_omnibus_p_within_time":
                        band_omnibus_p_values[
                            band_index
                        ],
                    "band_omnibus_p_holm_within_attribute":
                        adjusted_band_p,
                    "band_significant_holm_0p05":
                        bool(
                            adjusted_band_p
                            < 0.05
                        ),
                    "formal_inference_tmin_s":
                        inference_tmin,
                    "formal_inference_tmax_s":
                        inference_tmax,
                }
            )

            if adjusted_band_p >= 0.05:
                continue

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

                cluster_start = float(
                    inference_times[
                        cluster_indices
                    ].min()
                )

                cluster_end = float(
                    inference_times[
                        cluster_indices
                    ].max()
                )

                cluster_rows.append(
                    {
                        "family":
                            family_name,
                        "attribute":
                            attribute_name,
                        "band":
                            band_name,
                        "cluster_id":
                            cluster_id,
                        "cluster_p_within_band_time":
                            float(
                                cluster_p
                            ),
                        "time_start_s":
                            cluster_start,
                        "time_end_s":
                            cluster_end,
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
                        "response_timing_label":
                            response_timing_label(
                                cluster_end
                            ),
                        "band_omnibus_p_holm_within_attribute":
                            adjusted_band_p,
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
# Group-level analysis
# =============================================================================

def run_group_analysis(
    participants: Sequence[str],
    inference_tmin: float,
    inference_tmax: float,
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

    times = participant_results[
        0
    ][
        "times"
    ].astype(
        float
    )

    frequencies = participant_results[
        0
    ][
        "freqs"
    ].astype(
        float
    )

    band_names = (
        participant_results[
            0
        ][
            "band_names"
        ]
        .astype(
            str
        )
        .tolist()
    )

    for (
        participant,
        result,
    ) in zip(
        participants,
        participant_results,
    ):
        if not np.allclose(
            result[
                "times"
            ].astype(
                float
            ),
            times,
            atol=1e-12,
            rtol=0.0,
        ):
            raise RuntimeError(
                f"{participant}: score time grid differs."
            )

        if not np.array_equal(
            result[
                "freqs"
            ].astype(
                float
            ),
            frequencies,
        ):
            raise RuntimeError(
                f"{participant}: score frequency grid differs."
            )

    raw_frequency_scores = np.stack(
        [
            result[
                "raw_freq_r"
            ]
            for result in participant_results
        ],
        axis=0,
    )

    rt_residualized_frequency_scores = np.stack(
        [
            result[
                "rtres_freq_r"
            ]
            for result in participant_results
        ],
        axis=0,
    )

    raw_band_scores = np.stack(
        [
            result[
                "raw_band_r"
            ]
            for result in participant_results
        ],
        axis=0,
    )

    rt_residualized_band_scores = np.stack(
        [
            result[
                "rtres_band_r"
            ]
            for result in participant_results
        ],
        axis=0,
    )

    # Direct within-participant comparison on the proper correlation scale.
    rt_minus_raw_frequency = (
        fisher_z(
            rt_residualized_frequency_scores
        )
        - fisher_z(
            raw_frequency_scores
        )
    )

    rt_minus_raw_band = (
        fisher_z(
            rt_residualized_band_scores
        )
        - fisher_z(
            raw_band_scores
        )
    )

    np.savez_compressed(
        OUTPUT_ROOT
        / "group_score_arrays.npz",
        subjects=
            np.asarray(
                participants
            ),
        times=
            times,
        freqs=
            frequencies,
        band_names=
            np.asarray(
                band_names
            ),
        attribute_names=
            np.asarray(
                ATTRIBUTES
            ),
        raw_freq_r=
            raw_frequency_scores,
        rtres_freq_r=
            rt_residualized_frequency_scores,
        raw_band_r=
            raw_band_scores,
        rtres_band_r=
            rt_residualized_band_scores,
        rt_minus_raw_freq_fisher_z=
            rt_minus_raw_frequency,
        rt_minus_raw_band_fisher_z=
            rt_minus_raw_band,
    )

    # -------------------------------------------------------------------------
    # RAW and RT-adjusted absolute decoding families
    # -------------------------------------------------------------------------

    surface_omnibus_tables = []
    surface_cluster_tables = []
    band_omnibus_tables = []
    band_cluster_tables = []

    absolute_families = [
        (
            "raw_response_locked",
            raw_frequency_scores,
            raw_band_scores,
        ),
        (
            "rt_adjusted_response_locked",
            rt_residualized_frequency_scores,
            rt_residualized_band_scores,
        ),
    ]

    for (
        family_name,
        frequency_values,
        band_values,
    ) in absolute_families:
        (
            surface_omnibus,
            surface_clusters,
        ) = summarize_frequency_time_family(
            group_values=
                frequency_values,
            family_name=
                family_name,
            frequencies=
                frequencies,
            times=
                times,
            inference_tmin=
                inference_tmin,
            inference_tmax=
                inference_tmax,
            n_permutations=
                n_permutations,
            n_jobs=
                n_jobs,
            fisher_transform_correlations=
                True,
        )

        (
            band_omnibus,
            band_clusters,
        ) = summarize_band_family(
            group_values=
                band_values,
            family_name=
                family_name,
            times=
                times,
            band_names=
                band_names,
            inference_tmin=
                inference_tmin,
            inference_tmax=
                inference_tmax,
            n_permutations=
                n_permutations,
            n_jobs=
                n_jobs,
            fisher_transform_correlations=
                True,
        )

        surface_omnibus_tables.append(
            surface_omnibus
        )

        surface_cluster_tables.append(
            surface_clusters
        )

        band_omnibus_tables.append(
            band_omnibus
        )

        band_cluster_tables.append(
            band_clusters
        )

    absolute_surface_omnibus = pd.concat(
        surface_omnibus_tables,
        ignore_index=True,
    )

    absolute_surface_clusters = (
        pd.concat(
            surface_cluster_tables,
            ignore_index=True,
        )
        if any(
            len(
                table
            )
            for table in surface_cluster_tables
        )
        else pd.DataFrame()
    )

    absolute_band_omnibus = pd.concat(
        band_omnibus_tables,
        ignore_index=True,
    )

    absolute_band_clusters = (
        pd.concat(
            band_cluster_tables,
            ignore_index=True,
        )
        if any(
            len(
                table
            )
            for table in band_cluster_tables
        )
        else pd.DataFrame()
    )

    absolute_surface_omnibus.to_csv(
        OUTPUT_ROOT
        / "surface_omnibus_holm.csv",
        index=False,
    )

    absolute_surface_clusters.to_csv(
        OUTPUT_ROOT
        / "surface_significant_clusters.csv",
        index=False,
    )

    absolute_band_omnibus.to_csv(
        OUTPUT_ROOT
        / "band_omnibus_holm.csv",
        index=False,
    )

    absolute_band_clusters.to_csv(
        OUTPUT_ROOT
        / "band_significant_clusters.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Direct RT-adjusted > RAW comparison
    # -------------------------------------------------------------------------

    (
        difference_surface_omnibus,
        difference_surface_clusters,
    ) = summarize_frequency_time_family(
        group_values=
            rt_minus_raw_frequency,
        family_name=
            "rt_adjusted_minus_raw_fisher_z",
        frequencies=
            frequencies,
        times=
            times,
        inference_tmin=
            inference_tmin,
        inference_tmax=
            inference_tmax,
        n_permutations=
            n_permutations,
        n_jobs=
            n_jobs,
        fisher_transform_correlations=
            False,
    )

    (
        difference_band_omnibus,
        difference_band_clusters,
    ) = summarize_band_family(
        group_values=
            rt_minus_raw_band,
        family_name=
            "rt_adjusted_minus_raw_fisher_z",
        times=
            times,
        band_names=
            band_names,
        inference_tmin=
            inference_tmin,
        inference_tmax=
            inference_tmax,
        n_permutations=
            n_permutations,
        n_jobs=
            n_jobs,
        fisher_transform_correlations=
            False,
    )

    difference_surface_omnibus.to_csv(
        OUTPUT_ROOT
        / "rt_adjustment_difference_surface_omnibus_holm.csv",
        index=False,
    )

    difference_surface_clusters.to_csv(
        OUTPUT_ROOT
        / "rt_adjustment_difference_surface_clusters.csv",
        index=False,
    )

    difference_band_omnibus.to_csv(
        OUTPUT_ROOT
        / "rt_adjustment_difference_band_omnibus_holm.csv",
        index=False,
    )

    difference_band_clusters.to_csv(
        OUTPUT_ROOT
        / "rt_adjustment_difference_band_clusters.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Descriptive RT summary
    # -------------------------------------------------------------------------

    participant_rt_rows = []

    for (
        participant,
        result,
    ) in zip(
        participants,
        participant_results,
    ):
        participant_rt_rows.append(
            {
                "participant":
                    participant,
                "n_trials":
                    int(
                        result[
                            "n_trials"
                        ]
                    ),
                "behavioral_rt_min_s":
                    float(
                        result[
                            "behavioral_rt_min"
                        ]
                    ),
                "behavioral_rt_p05_s":
                    float(
                        result[
                            "behavioral_rt_p05"
                        ]
                    ),
                "behavioral_rt_median_s":
                    float(
                        result[
                            "behavioral_rt_median"
                        ]
                    ),
                "behavioral_rt_p95_s":
                    float(
                        result[
                            "behavioral_rt_p95"
                        ]
                    ),
                "event_rt_min_s":
                    float(
                        result[
                            "event_rt_min"
                        ]
                    ),
                "event_rt_median_s":
                    float(
                        result[
                            "event_rt_median"
                        ]
                    ),
                "max_abs_behavior_event_rt_difference_s":
                    float(
                        result[
                            "max_abs_behavior_event_rt_difference"
                        ]
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

    parameter_record = {
        "analysis":
            "response-locked decision TFR magnitude decoding",
        "attributes":
            ATTRIBUTES,
        "primary_analysis":
            "raw response-locked leave-one-block-out ordinal decoding",
        "decoder":
            "StandardScaler -> Ridge(alpha=1)",
        "ridge_alpha":
            RIDGE_ALPHA,
        "score":
            "pooled out-of-fold Pearson r",
        "rt_sensitivity":
            (
                "training-fold-only EEG feature residualisation for "
                "log(RT) and log(RT)^2"
            ),
        "rt_variable":
            (
                "behavioral RT, retained for exact comparability with the "
                "earlier cue-locked sensitivity analysis"
            ),
        "response_overlap_nuisance":
            (
                "not used because data are genuinely response-locked"
            ),
        "direct_rt_adjustment_test":
            (
                "one-sided cluster test of "
                "Fisher-z(r_RT-adjusted) - Fisher-z(r_RAW) > 0"
            ),
        "canonical_bands_hz":
            {
                name:
                    list(
                        limits
                    )
                for (
                    name,
                    limits,
                ) in CANONICAL_BANDS.items()
            },
        "morlet_half_support_s":
            MORLET_HALF_SUPPORT_S,
        "formal_inference_s":
            [
                inference_tmin,
                inference_tmax,
            ],
        "formal_inference_selection":
            (
                "timing-only >=95% nominal-centre-after-cue coverage within every "
                "participant; last qualifying response-time bin at/before "
                "response onset as end"
            ),
        "cluster_forming_alpha_one_sided":
            CLUSTER_FORMING_ALPHA,
        "n_permutations":
            n_permutations,
        "random_state":
            RANDOM_STATE,
        "interpretation_note":
            (
                "RT adjustment is a sensitivity analysis because RT is "
                "downstream of the decision process. A positive RT-adjustment "
                "difference indicates that removing RT-associated feature "
                "variance increased recoverable magnitude information; it "
                "does not make the residualized EEG a causally RT-independent "
                "signal."
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

    # -------------------------------------------------------------------------
    # Concise cluster-log summary
    # -------------------------------------------------------------------------

    print(
        "\nPRIMARY RAW RESPONSE-LOCKED FREQUENCY x TIME",
        flush=True,
    )

    print(
        absolute_surface_omnibus[
            absolute_surface_omnibus[
                "family"
            ]
            == "raw_response_locked"
        ].to_string(
            index=False
        ),
        flush=True,
    )

    print(
        "\nRT-ADJUSTED RESPONSE-LOCKED FREQUENCY x TIME",
        flush=True,
    )

    print(
        absolute_surface_omnibus[
            absolute_surface_omnibus[
                "family"
            ]
            == "rt_adjusted_response_locked"
        ].to_string(
            index=False
        ),
        flush=True,
    )

    if len(
        absolute_surface_clusters
    ):
        print(
            "\nSIGNIFICANT ABSOLUTE FREQUENCY x TIME CLUSTERS",
            flush=True,
        )

        print(
            absolute_surface_clusters.to_string(
                index=False
            ),
            flush=True,
        )

    print(
        "\nDIRECT RT-ADJUSTED > RAW FREQUENCY x TIME",
        flush=True,
    )

    print(
        difference_surface_omnibus.to_string(
            index=False
        ),
        flush=True,
    )

    if len(
        difference_surface_clusters
    ):
        print(
            "\nSIGNIFICANT RT-ADJUSTMENT DIFFERENCE CLUSTERS",
            flush=True,
        )

        print(
            difference_surface_clusters.to_string(
                index=False
            ),
            flush=True,
        )

    print(
        "\nPRIMARY RAW RESPONSE-LOCKED CANONICAL BANDS",
        flush=True,
    )

    print(
        absolute_band_omnibus[
            absolute_band_omnibus[
                "family"
            ]
            == "raw_response_locked"
        ].to_string(
            index=False
        ),
        flush=True,
    )

    print(
        "\nRT-ADJUSTED RESPONSE-LOCKED CANONICAL BANDS",
        flush=True,
    )

    print(
        absolute_band_omnibus[
            absolute_band_omnibus[
                "family"
            ]
            == "rt_adjusted_response_locked"
        ].to_string(
            index=False
        ),
        flush=True,
    )

    print(
        "\nDIRECT RT-ADJUSTED > RAW CANONICAL BANDS",
        flush=True,
    )

    print(
        difference_band_omnibus.to_string(
            index=False
        ),
        flush=True,
    )

    if len(
        absolute_band_clusters
    ):
        print(
            "\nSIGNIFICANT ABSOLUTE BAND CLUSTERS",
            flush=True,
        )

        print(
            absolute_band_clusters.to_string(
                index=False
            ),
            flush=True,
        )

    if len(
        difference_band_clusters
    ):
        print(
            "\nSIGNIFICANT RT-ADJUSTMENT BAND DIFFERENCE CLUSTERS",
            flush=True,
        )

        print(
            difference_band_clusters.to_string(
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
        "PAINREWARD RESPONSE-LOCKED DECISION TFR DECODING",
        flush=True,
    )
    print(
        f"Participants with complete inputs: {len(participants)}",
        flush=True,
    )
    print(
        "PRIMARY: raw ordinal pain + money decoding",
        flush=True,
    )
    print(
        "SENSITIVITY: training-fold-only log(RT) + log(RT)^2 residualisation",
        flush=True,
    )
    print(
        "DIRECT TEST: Fisher-z(RT-adjusted) - Fisher-z(raw) > 0",
        flush=True,
    )
    print(
        f"Permutations: {args.n_permutations}",
        flush=True,
    )
    print(
        "=" * 78,
        flush=True,
    )

    (
        inference_tmin,
        inference_tmax,
        _,
        _,
    ) = derive_response_inference_window(
        participants
    )

    if not args.group_only:
        for (
            participant_index,
            participant,
        ) in enumerate(
            participants,
            start=1,
        ):
            print(
                f"\n[{participant_index}/{len(participants)}] "
                f"{participant}",
                flush=True,
            )

            decode_one_participant(
                participant=
                    participant,
                overwrite_scores=
                    args.overwrite_scores,
            )

    missing_score_files = [
        participant
        for participant in participants
        if not participant_score_path(
            participant
        ).exists()
    ]

    if missing_score_files:
        raise RuntimeError(
            "Missing participant score files before group inference: "
            + ", ".join(
                missing_score_files
            )
        )

    run_group_analysis(
        participants=
            participants,
        inference_tmin=
            inference_tmin,
        inference_tmax=
            inference_tmax,
        n_permutations=
            args.n_permutations,
        n_jobs=
            args.n_jobs,
    )

    print(
        "\n"
        + "=" * 78,
        flush=True,
    )
    print(
        "RESPONSE-LOCKED DECODING COMPLETE",
        flush=True,
    )
    print(
        "=" * 78,
        flush=True,
    )


if __name__ == "__main__":
    main()
