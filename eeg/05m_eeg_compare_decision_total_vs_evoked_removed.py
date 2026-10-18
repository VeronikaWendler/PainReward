#!/usr/bin/env python3
"""
05m_eeg_compare_decision_total_vs_evoked_removed.py
===================================================

Direct paired comparison of decision-phase decoding from:

    TOTAL TFR power
        versus
    CROSS-FITTED EVOKED-REMOVED / INDUCED-ENRICHED TFR power

Why is this necessary?
----------------------
A result such as:

    total power significant
    evoked-removed power not significant

does NOT itself prove that the two decoding strengths differ.

This script therefore directly tests participant-wise decoding-score
differences:

    Fisher-z(r_total) - Fisher-z(r_evoked_removed)

for both:

    1. the RAW decoding family, and
    2. the RT-RESIDUALIZED sensitivity family.

Inference
---------
For each family and each attribute (pain, money):

    - two-sided 2D frequency x time cluster permutation test
    - formal window: 0-1.10 s
    - Holm correction across pain and money

Interpretation
--------------
Positive difference:
    total-power decoding > evoked-removed decoding

Negative difference:
    evoked-removed decoding > total-power decoding

This is the appropriate mechanistic test of whether decoding depends on the
generic phase-locked decision-cue response.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from mne.stats import combine_adjacency, permutation_cluster_1samp_test
from scipy import stats


# =============================================================================
# Paths and constants
# =============================================================================

BASE_PATH = Path(
    os.getenv("basepath", Path(__file__).parent.parent.parent)
)

TOTAL_RESULTS = (
    BASE_PATH
    / "derivatives"
    / "statistics"
    / "tfr_mvpa_decision_native"
    / "group_score_arrays.npz"
)

EVOKED_REMOVED_RESULTS = (
    BASE_PATH
    / "derivatives"
    / "statistics"
    / "tfr_mvpa_decision_evoked_removed"
    / "group_score_arrays.npz"
)

OUTPUT_ROOT = (
    BASE_PATH
    / "derivatives"
    / "statistics"
    / "tfr_decision_total_vs_evoked_removed"
)

INFERENCE_TMIN = 0.0
INFERENCE_TMAX = 1.10

CLUSTER_FORMING_ALPHA = 0.05
RANDOM_STATE = 23

ATTRIBUTES = [
    "pain",
    "money",
]


# =============================================================================
# Command-line arguments
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Paired total-vs-evoked-removed decision TFR decoding comparison."
        )
    )

    parser.add_argument(
        "--n-permutations",
        type=int,
        default=5000,
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
    )

    return parser.parse_args()


# =============================================================================
# Statistical helpers
# =============================================================================

def holm_correct(
    p_values: Sequence[float],
) -> np.ndarray:
    """Holm correction across pain and money."""

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
    """Fisher-z transform participant decoding correlations."""

    return np.arctanh(
        np.clip(
            correlation_values,
            -0.999999,
            0.999999,
        )
    )


def paired_frequency_time_cluster_test(
    participant_differences: np.ndarray,
    frequencies: np.ndarray,
    times: np.ndarray,
    n_permutations: int,
    n_jobs: int,
):
    """
    Two-sided frequency x time cluster test against zero.

    participant_differences shape:
        participant x frequency x time
    """

    time_mask = (
        (times >= INFERENCE_TMIN - 1e-12)
        & (times <= INFERENCE_TMAX + 1e-12)
    )

    inference_times = times[
        time_mask
    ]

    differences = participant_differences[
        :,
        :,
        time_mask,
    ]

    (
        n_participants,
        n_frequencies,
        n_times,
    ) = differences.shape

    threshold = float(
        stats.t.ppf(
            1.0
            - CLUSTER_FORMING_ALPHA / 2.0,
            df=n_participants - 1,
        )
    )

    (
        t_values,
        clusters,
        cluster_p_values,
        _,
    ) = permutation_cluster_1samp_test(
        differences.reshape(
            n_participants,
            n_frequencies * n_times,
        ),
        threshold=threshold,
        n_permutations=n_permutations,
        tail=0,
        adjacency=combine_adjacency(
            n_frequencies,
            n_times,
        ),
        seed=RANDOM_STATE,
        n_jobs=n_jobs,
        out_type="indices",
        verbose=False,
    )

    reshaped_t_values = t_values.reshape(
        n_frequencies,
        n_times,
    )

    flattened_cluster_indices = [
        np.asarray(
            cluster[0],
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


# =============================================================================
# Input validation
# =============================================================================

def verify_matching_results(
    total_results,
    evoked_removed_results,
) -> None:
    """Fail fast unless both analyses contain exactly matching grids/subjects."""

    for key in [
        "subjects",
        "times",
        "freqs",
        "attribute_names",
    ]:
        total_value = total_results[
            key
        ]

        evoked_removed_value = evoked_removed_results[
            key
        ]

        if total_value.dtype.kind in "f":
            matches = np.allclose(
                total_value,
                evoked_removed_value,
            )
        else:
            matches = np.array_equal(
                total_value,
                evoked_removed_value,
            )

        if not matches:
            raise RuntimeError(
                f"Total and evoked-removed results differ for {key}."
            )


# =============================================================================
# Main paired comparison
# =============================================================================

def main() -> None:
    args = parse_args()

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not TOTAL_RESULTS.exists():
        raise FileNotFoundError(
            TOTAL_RESULTS
        )

    if not EVOKED_REMOVED_RESULTS.exists():
        raise FileNotFoundError(
            EVOKED_REMOVED_RESULTS
        )

    total_results = np.load(
        TOTAL_RESULTS,
        allow_pickle=False,
    )

    evoked_removed_results = np.load(
        EVOKED_REMOVED_RESULTS,
        allow_pickle=False,
    )

    verify_matching_results(
        total_results=total_results,
        evoked_removed_results=evoked_removed_results,
    )

    subjects = total_results[
        "subjects"
    ].astype(
        str
    )

    times = total_results[
        "times"
    ].astype(
        float
    )

    frequencies = total_results[
        "freqs"
    ].astype(
        float
    )

    attribute_names = total_results[
        "attribute_names"
    ].astype(
        str
    ).tolist()

    if attribute_names != ATTRIBUTES:
        raise RuntimeError(
            f"Unexpected attribute order: {attribute_names}"
        )

    analysis_families = [
        (
            "raw",
            "raw_freq_r",
        ),
        (
            "rt_residualized",
            "rtres_freq_r",
        ),
    ]

    omnibus_rows = []
    cluster_rows = []

    saved_difference_arrays = {
        "subjects":
            subjects,
        "times":
            times,
        "freqs":
            frequencies,
        "attribute_names":
            np.asarray(
                ATTRIBUTES
            ),
    }

    for family_name, score_key in analysis_families:
        total_scores = total_results[
            score_key
        ]

        evoked_removed_scores = evoked_removed_results[
            score_key
        ]

        participant_differences = (
            fisher_z(
                total_scores
            )
            - fisher_z(
                evoked_removed_scores
            )
        )

        saved_difference_arrays[
            f"{family_name}_fisherz_total_minus_evoked_removed"
        ] = participant_differences

        attribute_results = []

        for attribute_index, attribute_name in enumerate(
            ATTRIBUTES
        ):
            (
                t_values,
                clusters,
                cluster_p_values,
                inference_times,
            ) = paired_frequency_time_cluster_test(
                participant_differences=participant_differences[
                    :,
                    attribute_index,
                ],
                frequencies=frequencies,
                times=times,
                n_permutations=args.n_permutations,
                n_jobs=args.n_jobs,
            )

            if len(
                cluster_p_values
            ):
                minimum_p = float(
                    cluster_p_values.min()
                )
            else:
                minimum_p = 1.0

            attribute_results.append(
                (
                    attribute_name,
                    minimum_p,
                    t_values,
                    clusters,
                    cluster_p_values,
                    inference_times,
                )
            )

        holm_adjusted = holm_correct(
            [
                result[
                    1
                ]
                for result in attribute_results
            ]
        )

        for attribute_index, (
            attribute_name,
            minimum_p,
            t_values,
            clusters,
            cluster_p_values,
            inference_times,
        ) in enumerate(
            attribute_results
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
                    "omnibus_p_within_surface":
                        minimum_p,
                    "omnibus_p_holm_across_pain_money":
                        adjusted_p,
                    "significant_holm_0p05":
                        bool(
                            adjusted_p < 0.05
                        ),
                }
            )

            n_frequencies, n_times = t_values.shape

            for cluster_id, (
                flat_cluster_indices,
                cluster_p,
            ) in enumerate(
                zip(
                    clusters,
                    cluster_p_values,
                )
            ):
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
                        np.abs(
                            cluster_t_values
                        )
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

                mean_cluster_t = float(
                    cluster_t_values.mean()
                )

                cluster_rows.append(
                    {
                        "family":
                            family_name,
                        "attribute":
                            attribute_name,
                        "cluster_id":
                            cluster_id,
                        "direction":
                            (
                                "total_gt_evoked_removed"
                                if mean_cluster_t > 0
                                else "evoked_removed_gt_total"
                            ),
                        "cluster_p_within_surface":
                            float(
                                cluster_p
                            ),
                        "omnibus_p_holm_across_pain_money":
                            adjusted_p,
                        "cluster_significant_final":
                            bool(
                                cluster_p < 0.05
                                and adjusted_p < 0.05
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
                        "cluster_mass_signed":
                            float(
                                cluster_t_values.sum()
                            ),
                    }
                )

    omnibus_table = pd.DataFrame(
        omnibus_rows
    )

    cluster_table = pd.DataFrame(
        cluster_rows
    )

    omnibus_table.to_csv(
        OUTPUT_ROOT
        / "domain_omnibus_holm.csv",
        index=False,
    )

    cluster_table.to_csv(
        OUTPUT_ROOT
        / "paired_difference_clusters.csv",
        index=False,
    )

    np.savez_compressed(
        OUTPUT_ROOT
        / "paired_difference_arrays.npz",
        **saved_difference_arrays,
    )

    params = {
        "comparison":
            (
                "decision total power minus cross-fitted generic-evoked-removed "
                "/ induced-enriched power"
            ),
        "families":
            [
                "raw",
                "rt_residualized",
            ],
        "score_difference":
            "Fisher-z(r_total) - Fisher-z(r_evoked_removed)",
        "formal_inference_s":
            [
                INFERENCE_TMIN,
                INFERENCE_TMAX,
            ],
        "test":
            (
                "two-sided paired 2D frequency x time cluster permutation "
                "separately pain and money"
            ),
        "multiplicity":
            "Holm across pain and money within each family",
        "n_permutations":
            args.n_permutations,
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
        "\nPAIRED TOTAL VS EVOKED-REMOVED OMNIBUS TESTS",
        flush=True,
    )

    print(
        omnibus_table.to_string(
            index=False
        ),
        flush=True,
    )

    significant_clusters = (
        cluster_table[
            cluster_table[
                "cluster_significant_final"
            ]
            == True
        ]
        if len(
            cluster_table
        )
        else cluster_table
    )

    print(
        "\nSIGNIFICANT PAIRED-DIFFERENCE CLUSTERS",
        flush=True,
    )

    if len(
        significant_clusters
    ):
        print(
            significant_clusters.to_string(
                index=False
            ),
            flush=True,
        )
    else:
        print(
            "None",
            flush=True,
        )

    print(
        "\nDECISION TOTAL VS EVOKED-REMOVED COMPARISON COMPLETE",
        flush=True,
    )


if __name__ == "__main__":
    main()
