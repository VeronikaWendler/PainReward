#!/usr/bin/env python3
"""
05l_eeg_tfr_decode_decision_evoked_removed.py
=====================================

Native decision-phase decoding of cross-fitted evoked-removed / induced-enriched TFR power.

Signal being decoded
--------------------
The input is NOT total power. It is the cross-fitted generic-evoked-removed /
induced-enriched TFR created by 04f_eeg_tfr_prep_decision_evoked_removed.py.

The decoding, CV, scoring, RT sensitivity analysis, and group inference are
otherwise deliberately matched to the completed total-power decision decoder
(05j).

Scientific question
-------------------
While participants are actively deciding between pain and money, does the
multichannel spectral-power pattern contain information about:

    1. pain magnitude (levels 1-5), and
    2. money magnitude (levels 1-5)?

The decision task is a balanced 5 x 5 factorial design. Pain and money levels
are therefore decoded separately without residualising one target against the
other.

Primary analysis
----------------
For each participant, frequency, and time bin:

    multichannel TFR power -> target level (1-5)

using:

    StandardScaler -> Ridge(alpha=1)

Cross-validation is leave-one-decision-block-out. Predictions from all held-out
blocks are concatenated and scored with Pearson r against the true target.

RT sensitivity analysis
-----------------------
The same decoding is repeated after removing from the EEG features, within each
training fold only, variance associated with:

    - log(RT)
    - log(RT)^2
    - whether the response occurs before the conservative end of the Morlet
      wavelet support for that time bin

This is a sensitivity analysis, NOT a replacement for the primary analysis.
RT is downstream of the decision process, so removing RT-related variance can
also remove genuine decision-related neural activity.

Group inference
---------------
Two kinds of analyses are performed:

1. Frequency x time surfaces (4-45 Hz)
2. Canonical-band time courses:
       theta      4-7 Hz
       alpha      8-13 Hz
       beta      14-30 Hz
       low gamma 31-45 Hz

Formal inference is restricted to 0-1.10 s after cue onset.

For frequency x time surfaces:
    - one-sided cluster permutation test for above-zero decoding
    - pain and money tested separately
    - Holm correction across the two domain-level omnibus p-values

For canonical bands:
    - one-sided temporal cluster permutation test
    - Holm correction across the four bands separately within pain and money

Outputs
-------
derivatives/statistics/tfr_mvpa_decision_evoked_removed/
    participant_scores/
    group_score_arrays.npz
    surface_omnibus_holm.csv
    surface_significant_clusters.csv
    band_omnibus_holm.csv
    band_significant_clusters.csv
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

import numpy as np
import pandas as pd
from mne.stats import combine_adjacency, permutation_cluster_1samp_test
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# =============================================================================
# Project paths and analysis constants
# =============================================================================

BASE_PATH = Path(
    os.getenv("basepath", Path(__file__).parent.parent.parent)
)

PARTICIPANTS_TSV = BASE_PATH / "participants.tsv"

OUTPUT_ROOT = (
    BASE_PATH
    / "derivatives"
    / "statistics"
    / "tfr_mvpa_decision_evoked_removed"
)

PARTICIPANT_SCORE_DIR = OUTPUT_ROOT / "participant_scores"


ATTRIBUTES = ["pain", "money"]

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

# With n_cycles = frequency / 2, the conservative 5-sigma half-support is
# approximately constant across frequencies.
MORLET_HALF_SUPPORT_S = 5.0 * (0.5 / (2.0 * np.pi))


# =============================================================================
# Command-line arguments
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Decision-phase decoding of cross-fitted evoked-removed / induced-enriched TFR power."
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
        help="Recompute participant decoding scores even if they already exist.",
    )

    parser.add_argument(
        "--group-only",
        action="store_true",
        help="Skip participant decoding and recompute group statistics only.",
    )

    return parser.parse_args()


# =============================================================================
# File locations
# =============================================================================

def decision_tfr_paths(participant: str) -> Dict[str, Path]:
    """Return all decision-TFR input files for one participant."""

    tfr_dir = (
        BASE_PATH
        / "derivatives"
        / participant
        / "eeg"
        / "tfr_decision_evoked_removed"
    )

    stem = f"{participant}_decision_cue"

    return {
        "power": tfr_dir / f"{stem}_power_evoked_removed.npy",
        "times": tfr_dir / f"{stem}_times.npy",
        "freqs": tfr_dir / f"{stem}_freqs.npy",
        "metadata": tfr_dir / f"{stem}_metadata.csv",
    }


def participant_score_path(participant: str) -> Path:
    """Output NPZ containing all decoding scores for one participant."""

    return (
        PARTICIPANT_SCORE_DIR
        / f"{participant}_decision_evoked_removed_mvpa.npz"
    )


def load_participant_list() -> List[str]:
    """
    Read the participant IDs from participants.tsv and verify that every
    participant has a complete set of decision-TFR inputs.
    """

    participant_table = pd.read_csv(PARTICIPANTS_TSV, sep="\t")

    participants = sorted(
        participant_table["participant_id"]
        .dropna()
        .astype(str)
        .unique()
    )

    for participant in participants:
        required_paths = decision_tfr_paths(participant)

        missing = [
            str(path)
            for path in required_paths.values()
            if not path.exists()
        ]

        if missing:
            raise FileNotFoundError(
                f"{participant}: missing decision-TFR input(s):\n"
                + "\n".join(missing)
            )

    return participants


# =============================================================================
# Decoder and scoring helpers
# =============================================================================

def make_ridge_decoder():
    """
    Standardise channel features inside each training fold and fit a
    Ridge regression decoder.
    """

    return make_pipeline(
        StandardScaler(),
        Ridge(alpha=RIDGE_ALPHA),
    )


def pearson_correlation(
    true_values: np.ndarray,
    predicted_values: np.ndarray,
) -> float:
    """Return Pearson r, or NaN if either vector is constant."""

    if np.std(true_values) == 0 or np.std(predicted_values) == 0:
        return np.nan

    return float(
        np.corrcoef(true_values, predicted_values)[0, 1]
    )


# =============================================================================
# RT-sensitivity helpers
# =============================================================================

def make_rt_nuisance_matrix(
    reaction_time: np.ndarray,
    response_overlap: np.ndarray,
) -> np.ndarray:
    """
    Construct nuisance regressors used in the RT-sensitivity analysis.

    Columns:
        1. log(RT)
        2. log(RT)^2
        3. response-overlap indicator
    """

    log_rt = np.log(
        np.asarray(reaction_time, dtype=float)
    )

    return np.column_stack(
        [
            log_rt,
            log_rt ** 2,
            np.asarray(response_overlap, dtype=float),
        ]
    )


def residualize_features_using_training_fold(
    train_features: np.ndarray,
    test_features: np.ndarray,
    train_nuisance_raw: np.ndarray,
    test_nuisance_raw: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove nuisance-related variance from EEG features without leakage.

    The nuisance model is fitted ONLY on the training fold. Its coefficients
    are then applied to both training and held-out test trials.

    This prevents test-trial EEG from contributing to the nuisance model.
    """

    nuisance_mean = train_nuisance_raw.mean(axis=0)
    nuisance_sd = train_nuisance_raw.std(axis=0)

    varying_columns = nuisance_sd > 1e-12

    if not np.any(varying_columns):
        return train_features, test_features

    standardized_train_nuisance = (
        train_nuisance_raw[:, varying_columns]
        - nuisance_mean[varying_columns]
    ) / nuisance_sd[varying_columns]

    standardized_test_nuisance = (
        test_nuisance_raw[:, varying_columns]
        - nuisance_mean[varying_columns]
    ) / nuisance_sd[varying_columns]

    train_design = np.column_stack(
        [
            np.ones(len(standardized_train_nuisance)),
            standardized_train_nuisance,
        ]
    )

    test_design = np.column_stack(
        [
            np.ones(len(standardized_test_nuisance)),
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
        - train_design @ nuisance_coefficients
    )

    residualized_test = (
        test_features
        - test_design @ nuisance_coefficients
    )

    return residualized_train, residualized_test


# =============================================================================
# Leave-one-block-out decoding
# =============================================================================

def leave_one_block_out_score(
    features: np.ndarray,
    target: np.ndarray,
    block_ids: np.ndarray,
    reaction_time: np.ndarray,
    response_overlap: np.ndarray,
    remove_rt_related_variance: bool = False,
) -> float:
    """
    Decode a target using leave-one-decision-block-out cross-validation.

    Parameters
    ----------
    features
        Trial x channel matrix for one frequency/time point.

    target
        Pain level or money level (1-5).

    block_ids
        Decision-task block number for each trial.

    reaction_time
        Trial-wise RT in seconds.

    response_overlap
        Boolean/0-1 indicator specifying whether the response occurs before
        the conservative end of the wavelet support for the current time bin.

    remove_rt_related_variance
        If True, perform leakage-safe RT nuisance residualisation inside each
        CV fold before fitting the Ridge decoder.
    """

    predictions = np.full(
        len(target),
        np.nan,
        dtype=float,
    )

    unique_blocks = np.unique(block_ids)

    cross_validation = GroupKFold(
        n_splits=len(unique_blocks)
    )

    for train_indices, test_indices in cross_validation.split(
        features,
        target,
        groups=block_ids,
    ):
        train_features = np.asarray(
            features[train_indices],
            dtype=float,
        )

        test_features = np.asarray(
            features[test_indices],
            dtype=float,
        )

        if remove_rt_related_variance:
            train_nuisance = make_rt_nuisance_matrix(
                reaction_time[train_indices],
                response_overlap[train_indices],
            )

            test_nuisance = make_rt_nuisance_matrix(
                reaction_time[test_indices],
                response_overlap[test_indices],
            )

            train_features, test_features = (
                residualize_features_using_training_fold(
                    train_features=train_features,
                    test_features=test_features,
                    train_nuisance_raw=train_nuisance,
                    test_nuisance_raw=test_nuisance,
                )
            )

        decoder = make_ridge_decoder()

        decoder.fit(
            train_features,
            target[train_indices],
        )

        predictions[test_indices] = decoder.predict(
            test_features
        )

    if not np.isfinite(predictions).all():
        raise RuntimeError(
            "Cross-validation produced non-finite predictions."
        )

    return pearson_correlation(
        true_values=target,
        predicted_values=predictions,
    )


# =============================================================================
# Participant-level decoding
# =============================================================================

def decode_one_participant(participant: str) -> None:
    """
    Run the complete native decision-phase decoding for one participant.

    Saves:
        - raw frequency x time decoding
        - RT-residualized frequency x time decoding
        - raw canonical-band decoding
        - RT-residualized canonical-band decoding
    """

    paths = decision_tfr_paths(participant)

    power = np.load(
        paths["power"],
        mmap_mode="r",
    )

    times = np.load(
        paths["times"]
    ).astype(float)

    frequencies = np.load(
        paths["freqs"]
    ).astype(float)

    metadata = pd.read_csv(
        paths["metadata"]
    )

    pain_levels = pd.to_numeric(
        metadata["painlevel"],
        errors="coerce",
    ).to_numpy(dtype=float)

    money_levels = pd.to_numeric(
        metadata["moneylevel"],
        errors="coerce",
    ).to_numpy(dtype=float)

    block_ids = pd.to_numeric(
        metadata["block"],
        errors="coerce",
    ).to_numpy(dtype=int)

    reaction_time = pd.to_numeric(
        metadata["rt"],
        errors="coerce",
    ).to_numpy(dtype=float)

    if not (
        np.isfinite(pain_levels).all()
        and np.isfinite(money_levels).all()
        and np.isfinite(reaction_time).all()
    ):
        raise RuntimeError(
            f"{participant}: non-finite pain, money, or RT metadata."
        )

    # -------------------------------------------------------------------------
    # Response-overlap matrix
    #
    # For each trial and nominal TFR time bin, mark whether the participant's
    # response occurred before the conservative end of the wavelet support.
    #
    # Shape: trials x time
    # -------------------------------------------------------------------------

    response_overlap = np.stack(
        [
            reaction_time
            <= (time_point + MORLET_HALF_SUPPORT_S)
            for time_point in times
        ],
        axis=1,
    )

    targets = {
        "pain": pain_levels,
        "money": money_levels,
    }

    n_attributes = len(ATTRIBUTES)
    n_frequencies = len(frequencies)
    n_times = len(times)

    raw_frequency_scores = np.empty(
        (n_attributes, n_frequencies, n_times),
        dtype=np.float32,
    )

    rt_residualized_frequency_scores = np.empty_like(
        raw_frequency_scores
    )

    # -------------------------------------------------------------------------
    # Frequency-resolved decoding
    # -------------------------------------------------------------------------

    for attribute_index, attribute_name in enumerate(ATTRIBUTES):
        target = targets[attribute_name]

        print(
            f"    {participant}: decoding {attribute_name} "
            f"(frequency-resolved)",
            flush=True,
        )

        for frequency_index in range(n_frequencies):
            for time_index in range(n_times):
                channel_features = np.asarray(
                    power[
                        :,
                        :,
                        frequency_index,
                        time_index,
                    ],
                    dtype=float,
                )

                overlap_now = response_overlap[
                    :,
                    time_index,
                ]

                raw_frequency_scores[
                    attribute_index,
                    frequency_index,
                    time_index,
                ] = leave_one_block_out_score(
                    features=channel_features,
                    target=target,
                    block_ids=block_ids,
                    reaction_time=reaction_time,
                    response_overlap=overlap_now,
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
                    response_overlap=overlap_now,
                    remove_rt_related_variance=True,
                )

    # -------------------------------------------------------------------------
    # Canonical-band decoding
    # -------------------------------------------------------------------------

    band_names = list(CANONICAL_BANDS)

    raw_band_scores = np.empty(
        (
            n_attributes,
            len(band_names),
            n_times,
        ),
        dtype=np.float32,
    )

    rt_residualized_band_scores = np.empty_like(
        raw_band_scores
    )

    for attribute_index, attribute_name in enumerate(ATTRIBUTES):
        target = targets[attribute_name]

        print(
            f"    {participant}: decoding {attribute_name} "
            f"(canonical bands)",
            flush=True,
        )

        for band_index, band_name in enumerate(band_names):
            band_low, band_high = CANONICAL_BANDS[
                band_name
            ]

            frequency_mask = (
                (frequencies >= band_low)
                & (frequencies <= band_high)
            )

            for time_index in range(n_times):
                # Average baseline-normalised power across all frequencies
                # inside the predefined band, leaving channels as features.
                channel_features = np.asarray(
                    power[
                        :,
                        :,
                        frequency_mask,
                        time_index,
                    ],
                    dtype=float,
                ).mean(axis=2)

                overlap_now = response_overlap[
                    :,
                    time_index,
                ]

                raw_band_scores[
                    attribute_index,
                    band_index,
                    time_index,
                ] = leave_one_block_out_score(
                    features=channel_features,
                    target=target,
                    block_ids=block_ids,
                    reaction_time=reaction_time,
                    response_overlap=overlap_now,
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
                    response_overlap=overlap_now,
                    remove_rt_related_variance=True,
                )

    np.savez_compressed(
        participant_score_path(participant),
        raw_freq_r=raw_frequency_scores,
        rtres_freq_r=rt_residualized_frequency_scores,
        raw_band_r=raw_band_scores,
        rtres_band_r=rt_residualized_band_scores,
        times=times,
        freqs=frequencies,
        band_names=np.asarray(band_names),
        attribute_names=np.asarray(ATTRIBUTES),
        n_trials=np.asarray(len(metadata)),
        rt_min=np.asarray(reaction_time.min()),
        rt_p05=np.asarray(
            np.quantile(reaction_time, 0.05)
        ),
        rt_median=np.asarray(
            np.median(reaction_time)
        ),
        response_overlap_fraction=(
            response_overlap
            .mean(axis=0)
            .astype(np.float32)
        ),
    )


# =============================================================================
# Multiple-comparison helpers
# =============================================================================

def holm_correct(p_values: Sequence[float]) -> np.ndarray:
    """Holm correction for a small family of p-values."""

    p_values = np.asarray(
        p_values,
        dtype=float,
    )

    n_tests = len(p_values)

    order = np.argsort(p_values)
    ranked = p_values[order]

    adjusted_ranked = np.maximum.accumulate(
        (n_tests - np.arange(n_tests))
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

    adjusted[order] = adjusted_ranked

    return adjusted


def fisher_z(correlation_values: np.ndarray) -> np.ndarray:
    """Fisher-z transform while protecting against exactly +/-1."""

    clipped = np.clip(
        correlation_values,
        -0.999999,
        0.999999,
    )

    return np.arctanh(clipped)


def one_sided_t_threshold(
    n_participants: int,
) -> float:
    """Cluster-forming t threshold for a one-sided alpha=.05 test."""

    return float(
        stats.t.ppf(
            1.0 - CLUSTER_FORMING_ALPHA,
            df=n_participants - 1,
        )
    )


# =============================================================================
# Group-level cluster tests
# =============================================================================

def frequency_time_cluster_test(
    participant_scores: np.ndarray,
    frequencies: np.ndarray,
    times: np.ndarray,
    n_permutations: int,
    n_jobs: int,
):
    """
    One-sided frequency x time cluster permutation test against zero.

    participant_scores shape:
        participant x frequency x time
    """

    time_mask = (
        (times >= INFERENCE_TMIN)
        & (times <= INFERENCE_TMAX)
    )

    inference_times = times[time_mask]

    fisher_scores = fisher_z(
        participant_scores[
            :,
            :,
            time_mask,
        ]
    )

    (
        n_participants,
        n_frequencies,
        n_times,
    ) = fisher_scores.shape

    t_values, clusters, cluster_p_values, _ = (
        permutation_cluster_1samp_test(
            fisher_scores.reshape(
                n_participants,
                n_frequencies * n_times,
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
        np.asarray(cluster[0], dtype=int)
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
    participant_scores: np.ndarray,
    times: np.ndarray,
    n_permutations: int,
    n_jobs: int,
):
    """
    One-sided temporal cluster permutation test against zero.

    participant_scores shape:
        participant x time
    """

    time_mask = (
        (times >= INFERENCE_TMIN)
        & (times <= INFERENCE_TMAX)
    )

    inference_times = times[time_mask]

    fisher_scores = fisher_z(
        participant_scores[
            :,
            time_mask,
        ]
    )

    n_participants = fisher_scores.shape[0]

    t_values, clusters, cluster_p_values, _ = (
        permutation_cluster_1samp_test(
            fisher_scores,
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
        np.asarray(cluster[0], dtype=int)
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


# =============================================================================
# Summarise frequency x time inference
# =============================================================================

def summarize_frequency_time_family(
    group_scores: np.ndarray,
    family_name: str,
    frequencies: np.ndarray,
    times: np.ndarray,
    n_permutations: int,
    n_jobs: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run and summarise the pain and money frequency x time tests for one family.

    family_name is either:
        "raw"
        "rt_residualized"
    """

    stored_results = {}
    domain_omnibus_p_values = []

    for attribute_index, attribute_name in enumerate(ATTRIBUTES):
        result = frequency_time_cluster_test(
            participant_scores=group_scores[
                :,
                attribute_index,
            ],
            frequencies=frequencies,
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

        if len(cluster_p_values):
            minimum_cluster_p = float(
                cluster_p_values.min()
            )
        else:
            minimum_cluster_p = 1.0

        domain_omnibus_p_values.append(
            minimum_cluster_p
        )

        stored_results[attribute_name] = result

    holm_adjusted_domain_p = holm_correct(
        domain_omnibus_p_values
    )

    omnibus_rows = []
    cluster_rows = []

    for attribute_index, attribute_name in enumerate(ATTRIBUTES):
        adjusted_domain_p = float(
            holm_adjusted_domain_p[
                attribute_index
            ]
        )

        omnibus_rows.append(
            {
                "family": family_name,
                "attribute": attribute_name,
                "omnibus_p_within_surface":
                    domain_omnibus_p_values[
                        attribute_index
                    ],
                "omnibus_p_holm_across_pain_money":
                    adjusted_domain_p,
                "domain_significant_holm_0p05":
                    bool(adjusted_domain_p < 0.05),
            }
        )

        (
            t_values,
            clusters,
            cluster_p_values,
            inference_times,
        ) = stored_results[attribute_name]

        n_frequencies, n_times = t_values.shape

        # Only report individual clusters when the domain-level Holm-corrected
        # omnibus test is itself significant.
        if adjusted_domain_p >= 0.05:
            continue

        for cluster_id, (
            flat_cluster_indices,
            cluster_p,
        ) in enumerate(
            zip(
                clusters,
                cluster_p_values,
            )
        ):
            if cluster_p >= 0.05:
                continue

            frequency_indices, time_indices = (
                np.unravel_index(
                    flat_cluster_indices,
                    (
                        n_frequencies,
                        n_times,
                    ),
                )
            )

            cluster_t_values = (
                t_values
                .reshape(-1)[
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

            peak_frequency_index, peak_time_index = (
                np.unravel_index(
                    peak_flat_index,
                    (
                        n_frequencies,
                        n_times,
                    ),
                )
            )

            cluster_rows.append(
                {
                    "family": family_name,
                    "attribute": attribute_name,
                    "cluster_id": cluster_id,
                    "cluster_p_within_surface":
                        float(cluster_p),
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
                        float(
                            inference_times[
                                time_indices
                            ].min()
                        ),
                    "time_end_s":
                        float(
                            inference_times[
                                time_indices
                            ].max()
                        ),
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
                    "omnibus_p_holm_across_pain_money":
                        adjusted_domain_p,
                }
            )

    return (
        pd.DataFrame(omnibus_rows),
        pd.DataFrame(cluster_rows),
    )


# =============================================================================
# Summarise canonical-band inference
# =============================================================================

def summarize_band_family(
    group_band_scores: np.ndarray,
    family_name: str,
    times: np.ndarray,
    band_names: Sequence[str],
    n_permutations: int,
    n_jobs: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run canonical-band temporal cluster tests.

    Holm correction is applied across the four bands separately within
    pain and within money.
    """

    omnibus_rows = []
    cluster_rows = []

    for attribute_index, attribute_name in enumerate(ATTRIBUTES):
        stored_band_results = []
        band_omnibus_p_values = []

        for band_index, band_name in enumerate(band_names):
            result = temporal_cluster_test(
                participant_scores=group_band_scores[
                    :,
                    attribute_index,
                    band_index,
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

            if len(cluster_p_values):
                minimum_cluster_p = float(
                    cluster_p_values.min()
                )
            else:
                minimum_cluster_p = 1.0

            band_omnibus_p_values.append(
                minimum_cluster_p
            )

            stored_band_results.append(
                (
                    band_name,
                    t_values,
                    clusters,
                    cluster_p_values,
                    inference_times,
                )
            )

        holm_adjusted_band_p = holm_correct(
            band_omnibus_p_values
        )

        for band_index, (
            band_name,
            t_values,
            clusters,
            cluster_p_values,
            inference_times,
        ) in enumerate(stored_band_results):

            adjusted_band_p = float(
                holm_adjusted_band_p[
                    band_index
                ]
            )

            omnibus_rows.append(
                {
                    "family": family_name,
                    "attribute": attribute_name,
                    "band": band_name,
                    "band_omnibus_p_within_time":
                        band_omnibus_p_values[
                            band_index
                        ],
                    "band_omnibus_p_holm_within_attribute":
                        adjusted_band_p,
                    "band_significant_holm_0p05":
                        bool(adjusted_band_p < 0.05),
                }
            )

            if adjusted_band_p >= 0.05:
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
                        "family": family_name,
                        "attribute": attribute_name,
                        "band": band_name,
                        "cluster_id": cluster_id,
                        "cluster_p_within_band_time":
                            float(cluster_p),
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
                        "band_omnibus_p_holm_within_attribute":
                            adjusted_band_p,
                    }
                )

    return (
        pd.DataFrame(omnibus_rows),
        pd.DataFrame(cluster_rows),
    )


# =============================================================================
# Group-level analysis
# =============================================================================

def run_group_analysis(
    participants: Sequence[str],
    n_permutations: int,
    n_jobs: int,
) -> None:
    """Load all participant scores, run group statistics, and save outputs."""

    participant_results = [
        np.load(
            participant_score_path(participant),
            allow_pickle=False,
        )
        for participant in participants
    ]

    times = participant_results[0][
        "times"
    ].astype(float)

    frequencies = participant_results[0][
        "freqs"
    ].astype(float)

    band_names = (
        participant_results[0][
            "band_names"
        ]
        .astype(str)
        .tolist()
    )

    raw_frequency_scores = np.stack(
        [
            result["raw_freq_r"]
            for result in participant_results
        ],
        axis=0,
    )

    rt_residualized_frequency_scores = np.stack(
        [
            result["rtres_freq_r"]
            for result in participant_results
        ],
        axis=0,
    )

    raw_band_scores = np.stack(
        [
            result["raw_band_r"]
            for result in participant_results
        ],
        axis=0,
    )

    rt_residualized_band_scores = np.stack(
        [
            result["rtres_band_r"]
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
        OUTPUT_ROOT / "group_score_arrays.npz",
        subjects=np.asarray(participants),
        times=times,
        freqs=frequencies,
        band_names=np.asarray(band_names),
        attribute_names=np.asarray(ATTRIBUTES),
        raw_freq_r=raw_frequency_scores,
        rtres_freq_r=rt_residualized_frequency_scores,
        raw_band_r=raw_band_scores,
        rtres_band_r=rt_residualized_band_scores,
        response_overlap_fraction=response_overlap_fraction,
    )

    surface_omnibus_tables = []
    surface_cluster_tables = []
    band_omnibus_tables = []
    band_cluster_tables = []

    analysis_families = [
        (
            "raw",
            raw_frequency_scores,
            raw_band_scores,
        ),
        (
            "rt_residualized",
            rt_residualized_frequency_scores,
            rt_residualized_band_scores,
        ),
    ]

    for (
        family_name,
        frequency_scores,
        band_scores,
    ) in analysis_families:

        surface_omnibus, surface_clusters = (
            summarize_frequency_time_family(
                group_scores=frequency_scores,
                family_name=family_name,
                frequencies=frequencies,
                times=times,
                n_permutations=n_permutations,
                n_jobs=n_jobs,
            )
        )

        band_omnibus, band_clusters = (
            summarize_band_family(
                group_band_scores=band_scores,
                family_name=family_name,
                times=times,
                band_names=band_names,
                n_permutations=n_permutations,
                n_jobs=n_jobs,
            )
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

    surface_omnibus = pd.concat(
        surface_omnibus_tables,
        ignore_index=True,
    )

    surface_clusters = (
        pd.concat(
            surface_cluster_tables,
            ignore_index=True,
        )
        if any(
            len(table)
            for table in surface_cluster_tables
        )
        else pd.DataFrame()
    )

    band_omnibus = pd.concat(
        band_omnibus_tables,
        ignore_index=True,
    )

    band_clusters = (
        pd.concat(
            band_cluster_tables,
            ignore_index=True,
        )
        if any(
            len(table)
            for table in band_cluster_tables
        )
        else pd.DataFrame()
    )

    surface_omnibus.to_csv(
        OUTPUT_ROOT
        / "surface_omnibus_holm.csv",
        index=False,
    )

    surface_clusters.to_csv(
        OUTPUT_ROOT
        / "surface_significant_clusters.csv",
        index=False,
    )

    band_omnibus.to_csv(
        OUTPUT_ROOT
        / "band_omnibus_holm.csv",
        index=False,
    )

    band_clusters.to_csv(
        OUTPUT_ROOT
        / "band_significant_clusters.csv",
        index=False,
    )

    # Participant-level RT summary is descriptive only.
    participant_rt_rows = []

    for participant, result in zip(
        participants,
        participant_results,
    ):
        participant_rt_rows.append(
            {
                "participant": participant,
                "n_trials":
                    int(result["n_trials"]),
                "rt_min_s":
                    float(result["rt_min"]),
                "rt_p05_s":
                    float(result["rt_p05"]),
                "rt_median_s":
                    float(result["rt_median"]),
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
            "time_s": times,
            "mean_fraction_trials_response_overlaps_wavelet":
                response_overlap_fraction.mean(
                    axis=0
                ),
            "sem_fraction_trials_response_overlaps_wavelet":
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
        "signal":
            "cross-fitted generic-evoked-removed / induced-enriched TFR",
        "primary_analysis":
            "raw leave-one-block-out ordinal pain/money decoding",
        "decoder":
            "StandardScaler -> Ridge(alpha=1)",
        "score":
            "pooled out-of-fold Pearson r",
        "rt_sensitivity":
            (
                "training-fold-only EEG feature residualisation for "
                "log(RT), log(RT)^2, and response-before-wavelet-end"
            ),
        "morlet_half_support_s":
            MORLET_HALF_SUPPORT_S,
        "formal_inference_s":
            [
                INFERENCE_TMIN,
                INFERENCE_TMAX,
            ],
        "n_permutations":
            n_permutations,
        "note":
            (
                "RT control is a sensitivity analysis only because RT is "
                "downstream of the decision process and removing RT-related "
                "variance may also remove genuine decision-related variance."
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

    # Console summary for the cluster log.
    print(
        "\nPRIMARY RAW FREQUENCY x TIME",
        flush=True,
    )

    print(
        surface_omnibus[
            surface_omnibus["family"]
            == "raw"
        ].to_string(index=False),
        flush=True,
    )

    print(
        "\nRT-RESIDUALIZED FREQUENCY x TIME",
        flush=True,
    )

    print(
        surface_omnibus[
            surface_omnibus["family"]
            == "rt_residualized"
        ].to_string(index=False),
        flush=True,
    )

    if len(surface_clusters):
        print(
            "\nSIGNIFICANT FREQUENCY x TIME CLUSTERS",
            flush=True,
        )

        print(
            surface_clusters.to_string(
                index=False
            ),
            flush=True,
        )

    print(
        "\nPRIMARY RAW CANONICAL BANDS",
        flush=True,
    )

    print(
        band_omnibus[
            band_omnibus["family"]
            == "raw"
        ].to_string(index=False),
        flush=True,
    )

    print(
        "\nRT-RESIDUALIZED CANONICAL BANDS",
        flush=True,
    )

    print(
        band_omnibus[
            band_omnibus["family"]
            == "rt_residualized"
        ].to_string(index=False),
        flush=True,
    )

    if len(band_clusters):
        print(
            "\nSIGNIFICANT BAND CLUSTERS",
            flush=True,
        )

        print(
            band_clusters.to_string(
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
        "=" * 72,
        flush=True,
    )
    print(
        "DECISION-PHASE EVOKED-REMOVED TFR DECODING",
        flush=True,
    )
    print(
        f"Participants: {len(participants)}",
        flush=True,
    )
    print(
        f"Output: {OUTPUT_ROOT}",
        flush=True,
    )
    print(
        "=" * 72,
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
        "\nRunning group-level inference...",
        flush=True,
    )

    run_group_analysis(
        participants=participants,
        n_permutations=args.n_permutations,
        n_jobs=args.n_jobs,
    )

    print(
        "\nDECISION-PHASE TFR DECODING COMPLETE",
        flush=True,
    )


if __name__ == "__main__":
    main()
