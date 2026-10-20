#!/usr/bin/env python3
"""
05t_eeg_tfr_decode_decision_interaction.py
==========================================

Cue-locked decision-phase TFR test of the Pain x Money interaction.

Scientific question
-------------------
During choice, does the multichannel spectral-power pattern contain information
about the JOINT combination of pain and money beyond their additive main
effects?

The objective interaction is defined from the balanced 1-5 factorial design as

    pain_centered  = painlevel  - 3
    money_centered = moneylevel - 3
    interaction    = pain_centered * money_centered

The centered product is the conventional linear Pain x Money interaction term.

Important interpretation
------------------------
This is NOT yet a subjective-value model.

A significant result would mean that decision-period EEG contains information
about the joint/non-additive Pain x Money combination. It would NOT by itself
show that the brain represents a participant-specific subjective trade-off.

Primary interaction-specific analysis
-------------------------------------
To make the interaction test conditional on the two additive main effects,
leave-one-block-out cross-validation performs Frisch-Waugh-Lovell-style
residualisation inside each training fold:

    1. residualise the interaction target against centered Pain + centered Money;
    2. residualise each EEG channel feature against centered Pain + centered Money;
    3. fit StandardScaler -> Ridge(alpha=1) on the residualised training data;
    4. apply the training-fold nuisance models and Ridge model to the held-out
       block;
    5. concatenate held-out predictions and score them against the held-out
       residualised interaction target using Pearson r.

This protects the interaction test against small imbalances caused by rejected
EEG trials even though the experimental 5 x 5 design itself is balanced.

RT sensitivity analysis
-----------------------
The target remains the same interaction residual (Pain x Money beyond Pain and
Money). EEG features are additionally residualised, inside the training fold,
for:

    - log(RT)
    - log(RT)^2
    - whether the response occurs before the conservative end of the Morlet
      support for the current cue-locked time bin

This matches the RT-sensitivity logic used in the existing native decision TFR
decoder.

RT adjustment is a sensitivity analysis, not the primary analysis, because RT
is downstream of the choice process and may itself contain meaningful decision
variance.

Direct RT-adjusted minus primary comparison
-------------------------------------------
At group level the script also tests:

    Fisher-z(RT-adjusted r) - Fisher-z(primary r)

with a one-sided cluster test. This prevents interpreting "one p-value crossed
.05 and the other did not" as evidence that RT adjustment genuinely improved
decoding.

Data
----
Uses the EXISTING cue-locked decision TFR files:

    derivatives/<sub>/eeg/tfr_decision/

No new TFR calculation and no new trial exclusions are introduced. This keeps
the analysis directly comparable with the already completed cue-locked pain and
money decoding.

Inference
---------
Frequency x time:
    4-45 Hz x 0-1.10 s
    one-sided cluster-permutation test
    5000 permutations by default

Canonical bands:
    theta      4-7 Hz
    alpha      8-13 Hz
    beta      14-30 Hz
    low gamma 31-45 Hz

Holm correction is applied across the four canonical bands within each analysis
family. There is only one interaction target, so there is no pain-vs-money Holm
correction for the full frequency x time surface.

Outputs
-------
derivatives/statistics/tfr_mvpa_decision_interaction/
    participant_scores/
    group_score_arrays.npz
    surface_omnibus.csv
    surface_significant_clusters.csv
    band_omnibus_holm.csv
    band_significant_clusters.csv
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

import numpy as np
import pandas as pd
from mne.stats import combine_adjacency, permutation_cluster_1samp_test
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# =============================================================================
# Project paths and frozen analysis constants
# =============================================================================

BASE_PATH = Path(
    os.getenv("basepath", Path(__file__).parent.parent.parent)
)

PARTICIPANTS_TSV = BASE_PATH / "participants.tsv"

OUTPUT_ROOT = (
    BASE_PATH
    / "derivatives"
    / "statistics"
    / "tfr_mvpa_decision_interaction"
)

PARTICIPANT_SCORE_DIR = OUTPUT_ROOT / "participant_scores"


PAIN_DESIGN_CENTER = 3.0
MONEY_DESIGN_CENTER = 3.0

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

# Existing TFR prep used n_cycles = frequency / 2.
# The conservative 5-sigma half-support is therefore approximately constant.
MORLET_HALF_SUPPORT_S = 5.0 * (0.5 / (2.0 * np.pi))


# =============================================================================
# Command-line arguments
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Cue-locked TFR decoding of the centered Pain x Money interaction."
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
        help="Skip participant decoding and rerun group inference only.",
    )

    return parser.parse_args()


# =============================================================================
# File locations
# =============================================================================

def decision_tfr_paths(participant: str) -> Dict[str, Path]:
    """Return cue-locked decision TFR files for one participant."""

    tfr_dir = (
        BASE_PATH
        / "derivatives"
        / participant
        / "eeg"
        / "tfr_decision"
    )

    stem = f"{participant}_decision_cue"

    return {
        "power": tfr_dir / f"{stem}_power.npy",
        "times": tfr_dir / f"{stem}_times.npy",
        "freqs": tfr_dir / f"{stem}_freqs.npy",
        "metadata": tfr_dir / f"{stem}_metadata.csv",
    }


def participant_score_path(participant: str) -> Path:
    return (
        PARTICIPANT_SCORE_DIR
        / f"{participant}_decision_interaction_mvpa.npz"
    )


def load_participant_list() -> List[str]:
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
        missing = [
            str(path)
            for path in decision_tfr_paths(participant).values()
            if not path.exists()
        ]

        if missing:
            raise FileNotFoundError(
                f"{participant}: missing cue-locked decision TFR input(s):\n"
                + "\n".join(missing)
            )

    return participants


# =============================================================================
# Basic helpers
# =============================================================================

def make_ridge_decoder():
    return make_pipeline(
        StandardScaler(),
        Ridge(alpha=RIDGE_ALPHA),
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
        np.std(true_values) <= 1e-12
        or np.std(predicted_values) <= 1e-12
    ):
        return np.nan

    return float(
        np.corrcoef(
            true_values,
            predicted_values,
        )[0, 1]
    )


def fisher_z(correlations: np.ndarray) -> np.ndarray:
    clipped = np.clip(
        correlations,
        -0.999999,
        0.999999,
    )

    return np.arctanh(
        clipped
    )


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


def one_sided_t_threshold(
    n_participants: int,
) -> float:
    return float(
        stats.t.ppf(
            1.0
            - CLUSTER_FORMING_ALPHA,
            df=n_participants - 1,
        )
    )


# =============================================================================
# Interaction target
# =============================================================================

def centered_design_variables(
    pain_levels: np.ndarray,
    money_levels: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return centered Pain, centered Money, and their centered product.

    Design centering uses the known factorial midpoint (3), not each
    participant's observed mean. This preserves the intended 5 x 5 factorial
    coding even if EEG artifact rejection removed a few trials.
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

    interaction = (
        pain_centered
        * money_centered
    )

    return (
        pain_centered,
        money_centered,
        interaction,
    )


# =============================================================================
# Leakage-safe nuisance residualisation
# =============================================================================

def fit_nuisance_residualizer(
    nuisance_train_raw: np.ndarray,
    values_train: np.ndarray,
):
    """
    Fit a linear nuisance model on the training fold only.

    The nuisance columns are standardized from the training fold. Constant
    nuisance columns are omitted automatically.

    values_train can be:
        trial x channel features
        or
        a one-dimensional target vector
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
        standardized_train = (
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

        train_design = np.column_stack(
            [
                np.ones(
                    len(
                        standardized_train
                    )
                ),
                standardized_train,
            ]
        )
    else:
        train_design = np.ones(
            (
                len(
                    nuisance_train_raw
                ),
                1,
            )
        )

    coefficients, *_ = np.linalg.lstsq(
        train_design,
        values_train,
        rcond=None,
    )

    return {
        "mean": nuisance_mean,
        "sd": nuisance_sd,
        "varying": varying_columns,
        "coefficients": coefficients,
    }


def apply_nuisance_residualizer(
    nuisance_raw: np.ndarray,
    values: np.ndarray,
    fitted_residualizer: dict,
) -> np.ndarray:
    """
    Apply a nuisance model fitted on the training fold.
    """

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

        design = np.column_stack(
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
        design = np.ones(
            (
                len(
                    nuisance_raw
                ),
                1,
            )
        )

    fitted_values = (
        design
        @ fitted_residualizer[
            "coefficients"
        ]
    )

    return (
        values
        - fitted_values
    )


# =============================================================================
# Nuisance matrices
# =============================================================================

def main_effect_nuisance_matrix(
    pain_centered: np.ndarray,
    money_centered: np.ndarray,
) -> np.ndarray:
    """
    Additive Pain and Money main effects.

    These are always controlled in the interaction-specific analysis.
    """

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
    response_overlap: np.ndarray,
    include_rt_controls: bool,
) -> np.ndarray:
    """
    EEG nuisance matrix.

    Primary analysis:
        centered Pain + centered Money

    RT sensitivity:
        centered Pain + centered Money
        + log(RT) + log(RT)^2 + response/wavelet-overlap indicator
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
                    response_overlap,
                    dtype=float,
                ),
            ]
        )

    return np.column_stack(
        nuisance_columns
    )


# =============================================================================
# Leave-one-block-out interaction decoding
# =============================================================================

def leave_one_block_out_interaction_score(
    features: np.ndarray,
    pain_centered: np.ndarray,
    money_centered: np.ndarray,
    interaction_target: np.ndarray,
    block_ids: np.ndarray,
    reaction_time: np.ndarray,
    response_overlap: np.ndarray,
    include_rt_controls: bool,
) -> float:
    """
    Decode the Pain x Money interaction conditional on additive main effects.

    The interaction target is residualised against centered Pain and centered
    Money within each training fold.

    EEG features are residualised against the same two main effects. The RT
    sensitivity version additionally removes RT-related feature variance.

    The held-out residualised target and held-out predictions are pooled across
    blocks and scored with Pearson r.
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
        interaction_target,
        groups=block_ids,
    ):
        # ------------------------------------------------------------------
        # Conditional interaction target:
        # interaction residual after additive Pain + Money main effects.
        # ------------------------------------------------------------------

        target_nuisance = main_effect_nuisance_matrix(
            pain_centered=pain_centered,
            money_centered=money_centered,
        )

        fitted_target_nuisance = fit_nuisance_residualizer(
            nuisance_train_raw=target_nuisance[
                train_indices
            ],
            values_train=interaction_target[
                train_indices
            ],
        )

        train_target = apply_nuisance_residualizer(
            nuisance_raw=target_nuisance[
                train_indices
            ],
            values=interaction_target[
                train_indices
            ],
            fitted_residualizer=fitted_target_nuisance,
        )

        test_target = apply_nuisance_residualizer(
            nuisance_raw=target_nuisance[
                test_indices
            ],
            values=interaction_target[
                test_indices
            ],
            fitted_residualizer=fitted_target_nuisance,
        )

        # ------------------------------------------------------------------
        # Interaction-specific EEG:
        # remove additive main effects, and optionally RT-related variance.
        # ------------------------------------------------------------------

        all_feature_nuisance = feature_nuisance_matrix(
            pain_centered=pain_centered,
            money_centered=money_centered,
            reaction_time=reaction_time,
            response_overlap=response_overlap,
            include_rt_controls=include_rt_controls,
        )

        fitted_feature_nuisance = fit_nuisance_residualizer(
            nuisance_train_raw=all_feature_nuisance[
                train_indices
            ],
            values_train=features[
                train_indices
            ],
        )

        train_features = apply_nuisance_residualizer(
            nuisance_raw=all_feature_nuisance[
                train_indices
            ],
            values=features[
                train_indices
            ],
            fitted_residualizer=fitted_feature_nuisance,
        )

        test_features = apply_nuisance_residualizer(
            nuisance_raw=all_feature_nuisance[
                test_indices
            ],
            values=features[
                test_indices
            ],
            fitted_residualizer=fitted_feature_nuisance,
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
            "Cross-validation produced non-finite interaction predictions "
            "or conditional targets."
        )

    return pearson_correlation(
        true_values=heldout_conditional_target,
        predicted_values=predictions,
    )


def cross_fitted_conditional_interaction_target(
    pain_centered: np.ndarray,
    money_centered: np.ndarray,
    interaction_target: np.ndarray,
    block_ids: np.ndarray,
) -> np.ndarray:
    """
    Produce one held-out conditional interaction value per trial.

    This is used only for design diagnostics and descriptive correlations.
    """

    conditional_target = np.full(
        len(
            interaction_target
        ),
        np.nan,
        dtype=float,
    )

    target_nuisance = main_effect_nuisance_matrix(
        pain_centered=pain_centered,
        money_centered=money_centered,
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
            nuisance_train_raw=target_nuisance[
                train_indices
            ],
            values_train=interaction_target[
                train_indices
            ],
        )

        conditional_target[
            test_indices
        ] = apply_nuisance_residualizer(
            nuisance_raw=target_nuisance[
                test_indices
            ],
            values=interaction_target[
                test_indices
            ],
            fitted_residualizer=fitted_target_nuisance,
        )

    if not np.isfinite(
        conditional_target
    ).all():
        raise RuntimeError(
            "Conditional interaction target contains non-finite values."
        )

    return conditional_target


# =============================================================================
# Participant-level decoding
# =============================================================================

def decode_one_participant(
    participant: str,
) -> None:
    paths = decision_tfr_paths(
        participant
    )

    power = np.load(
        paths[
            "power"
        ],
        mmap_mode="r",
    )

    times = np.load(
        paths[
            "times"
        ]
    ).astype(
        float
    )

    frequencies = np.load(
        paths[
            "freqs"
        ]
    ).astype(
        float
    )

    metadata = pd.read_csv(
        paths[
            "metadata"
        ]
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

    reaction_time = pd.to_numeric(
        metadata[
            "rt"
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
    ):
        raise RuntimeError(
            f"{participant}: non-finite pain, money, or RT metadata."
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
        pain_levels=pain_levels,
        money_levels=money_levels,
    )

    conditional_interaction_target = (
        cross_fitted_conditional_interaction_target(
            pain_centered=pain_centered,
            money_centered=money_centered,
            interaction_target=interaction_target,
            block_ids=block_ids,
        )
    )

    response_overlap = np.stack(
        [
            reaction_time
            <= (
                time_point
                + MORLET_HALF_SUPPORT_S
            )
            for time_point
            in times
        ],
        axis=1,
    )

    n_frequencies = len(
        frequencies
    )

    n_times = len(
        times
    )

    primary_frequency_scores = np.empty(
        (
            n_frequencies,
            n_times,
        ),
        dtype=np.float32,
    )

    rt_adjusted_frequency_scores = np.empty_like(
        primary_frequency_scores
    )

    print(
        f"    {participant}: frequency-resolved centered interaction",
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

            overlap_now = response_overlap[
                :,
                time_index,
            ]

            primary_frequency_scores[
                frequency_index,
                time_index,
            ] = leave_one_block_out_interaction_score(
                features=channel_features,
                pain_centered=pain_centered,
                money_centered=money_centered,
                interaction_target=interaction_target,
                block_ids=block_ids,
                reaction_time=reaction_time,
                response_overlap=overlap_now,
                include_rt_controls=False,
            )

            rt_adjusted_frequency_scores[
                frequency_index,
                time_index,
            ] = leave_one_block_out_interaction_score(
                features=channel_features,
                pain_centered=pain_centered,
                money_centered=money_centered,
                interaction_target=interaction_target,
                block_ids=block_ids,
                reaction_time=reaction_time,
                response_overlap=overlap_now,
                include_rt_controls=True,
            )

    band_names = list(
        CANONICAL_BANDS
    )

    primary_band_scores = np.empty(
        (
            len(
                band_names
            ),
            n_times,
        ),
        dtype=np.float32,
    )

    rt_adjusted_band_scores = np.empty_like(
        primary_band_scores
    )

    print(
        f"    {participant}: canonical-band centered interaction",
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

        for time_index in range(
            n_times
        ):
            channel_features = np.asarray(
                power[
                    :,
                    :,
                    frequency_mask,
                    time_index,
                ],
                dtype=float,
            ).mean(
                axis=2
            )

            overlap_now = response_overlap[
                :,
                time_index,
            ]

            primary_band_scores[
                band_index,
                time_index,
            ] = leave_one_block_out_interaction_score(
                features=channel_features,
                pain_centered=pain_centered,
                money_centered=money_centered,
                interaction_target=interaction_target,
                block_ids=block_ids,
                reaction_time=reaction_time,
                response_overlap=overlap_now,
                include_rt_controls=False,
            )

            rt_adjusted_band_scores[
                band_index,
                time_index,
            ] = leave_one_block_out_interaction_score(
                features=channel_features,
                pain_centered=pain_centered,
                money_centered=money_centered,
                interaction_target=interaction_target,
                block_ids=block_ids,
                reaction_time=reaction_time,
                response_overlap=overlap_now,
                include_rt_controls=True,
            )

    np.savez_compressed(
        participant_score_path(
            participant
        ),
        primary_freq_r=
            primary_frequency_scores,
        rt_adjusted_freq_r=
            rt_adjusted_frequency_scores,
        primary_band_r=
            primary_band_scores,
        rt_adjusted_band_r=
            rt_adjusted_band_scores,
        times=times,
        freqs=frequencies,
        band_names=np.asarray(
            band_names
        ),
        n_trials=np.asarray(
            len(
                metadata
            )
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
            response_overlap
            .mean(
                axis=0
            )
            .astype(
                np.float32
            )
        ),
        corr_raw_interaction_with_pain=np.asarray(
            pearson_correlation(
                interaction_target,
                pain_centered,
            )
        ),
        corr_raw_interaction_with_money=np.asarray(
            pearson_correlation(
                interaction_target,
                money_centered,
            )
        ),
        corr_conditional_interaction_with_pain=np.asarray(
            pearson_correlation(
                conditional_interaction_target,
                pain_centered,
            )
        ),
        corr_conditional_interaction_with_money=np.asarray(
            pearson_correlation(
                conditional_interaction_target,
                money_centered,
            )
        ),
        corr_raw_with_conditional_interaction=np.asarray(
            pearson_correlation(
                interaction_target,
                conditional_interaction_target,
            )
        ),
    )


# =============================================================================
# Cluster-permutation inference
# =============================================================================

def frequency_time_cluster_test(
    participant_values: np.ndarray,
    frequencies: np.ndarray,
    times: np.ndarray,
    n_permutations: int,
    n_jobs: int,
    values_are_correlations: bool,
):
    """
    One-sided cluster test against zero.

    participant_values:
        participant x frequency x time

    Correlation scores are Fisher-z transformed before inference.
    Fisher-z differences are already on the appropriate scale.
    """

    time_mask = (
        (
            times
            >= INFERENCE_TMIN
        )
        & (
            times
            <= INFERENCE_TMAX
        )
    )

    inference_times = times[
        time_mask
    ]

    values = participant_values[
        :,
        :,
        time_mask,
    ]

    if values_are_correlations:
        values = fisher_z(
            values
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

    return (
        t_values.reshape(
            n_frequencies,
            n_times,
        ),
        [
            np.asarray(
                cluster[
                    0
                ],
                dtype=int,
            )
            for cluster
            in clusters
        ],
        np.asarray(
            cluster_p_values,
            dtype=float,
        ),
        inference_times,
    )


def temporal_cluster_test(
    participant_values: np.ndarray,
    times: np.ndarray,
    n_permutations: int,
    n_jobs: int,
    values_are_correlations: bool,
):
    time_mask = (
        (
            times
            >= INFERENCE_TMIN
        )
        & (
            times
            <= INFERENCE_TMAX
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

    return (
        t_values,
        [
            np.asarray(
                cluster[
                    0
                ],
                dtype=int,
            )
            for cluster
            in clusters
        ],
        np.asarray(
            cluster_p_values,
            dtype=float,
        ),
        inference_times,
    )


# =============================================================================
# Result summarisation
# =============================================================================

def summarize_surface_family(
    participant_values: np.ndarray,
    family_name: str,
    frequencies: np.ndarray,
    times: np.ndarray,
    n_permutations: int,
    n_jobs: int,
    values_are_correlations: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    (
        t_values,
        clusters,
        cluster_p_values,
        inference_times,
    ) = frequency_time_cluster_test(
        participant_values=
            participant_values,
        frequencies=frequencies,
        times=times,
        n_permutations=
            n_permutations,
        n_jobs=n_jobs,
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

    omnibus = pd.DataFrame(
        [
            {
                "family":
                    family_name,
                "target":
                    "centered_pain_x_money",
                "omnibus_p_within_surface":
                    minimum_cluster_p,
                "surface_significant_0p05":
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

            flattened_t_values = (
                t_values
                .reshape(
                    -1
                )[
                    flat_cluster_indices
                ]
            )

            peak_position = int(
                np.argmax(
                    flattened_t_values
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

            cluster_rows.append(
                {
                    "family":
                        family_name,
                    "target":
                        "centered_pain_x_money",
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
                            flattened_t_values[
                                peak_position
                            ]
                        ),
                }
            )

    return (
        omnibus,
        pd.DataFrame(
            cluster_rows
        ),
    )


def summarize_band_family(
    participant_values: np.ndarray,
    family_name: str,
    times: np.ndarray,
    band_names: Sequence[str],
    n_permutations: int,
    n_jobs: int,
    values_are_correlations: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    stored_results = []
    band_omnibus_p_values = []

    for band_index, band_name in enumerate(
        band_names
    ):
        result = temporal_cluster_test(
            participant_values=
                participant_values[
                    :,
                    band_index,
                ],
            times=times,
            n_permutations=
                n_permutations,
            n_jobs=n_jobs,
            values_are_correlations=
                values_are_correlations,
        )

        (
            t_values,
            clusters,
            cluster_p_values,
            inference_times,
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

        stored_results.append(
            (
                band_name,
                t_values,
                clusters,
                cluster_p_values,
                inference_times,
            )
        )

    holm_adjusted = holm_correct(
        band_omnibus_p_values
    )

    omnibus_rows = []
    cluster_rows = []

    for band_index, (
        band_name,
        t_values,
        clusters,
        cluster_p_values,
        inference_times,
    ) in enumerate(
        stored_results
    ):
        adjusted_p = float(
            holm_adjusted[
                band_index
            ]
        )

        omnibus_rows.append(
            {
                "family":
                    family_name,
                "target":
                    "centered_pain_x_money",
                "band":
                    band_name,
                "band_omnibus_p_within_time":
                    band_omnibus_p_values[
                        band_index
                    ],
                "band_omnibus_p_holm_within_family":
                    adjusted_p,
                "band_significant_holm_0p05":
                    bool(
                        adjusted_p
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
        )

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
                    "target":
                        "centered_pain_x_money",
                    "band":
                        band_name,
                    "cluster_id":
                        cluster_id,
                    "cluster_p_within_band_time":
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
                    "band_omnibus_p_holm_within_family":
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
    participant_results = [
        np.load(
            participant_score_path(
                participant
            ),
            allow_pickle=False,
        )
        for participant
        in participants
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

    primary_frequency_scores = np.stack(
        [
            result[
                "primary_freq_r"
            ]
            for result
            in participant_results
        ],
        axis=0,
    )

    rt_adjusted_frequency_scores = np.stack(
        [
            result[
                "rt_adjusted_freq_r"
            ]
            for result
            in participant_results
        ],
        axis=0,
    )

    primary_band_scores = np.stack(
        [
            result[
                "primary_band_r"
            ]
            for result
            in participant_results
        ],
        axis=0,
    )

    rt_adjusted_band_scores = np.stack(
        [
            result[
                "rt_adjusted_band_r"
            ]
            for result
            in participant_results
        ],
        axis=0,
    )

    frequency_difference = (
        fisher_z(
            rt_adjusted_frequency_scores
        )
        - fisher_z(
            primary_frequency_scores
        )
    )

    band_difference = (
        fisher_z(
            rt_adjusted_band_scores
        )
        - fisher_z(
            primary_band_scores
        )
    )

    response_overlap_fraction = np.stack(
        [
            result[
                "response_overlap_fraction"
            ]
            for result
            in participant_results
        ],
        axis=0,
    )

    np.savez_compressed(
        OUTPUT_ROOT
        / "group_score_arrays.npz",
        subjects=np.asarray(
            participants
        ),
        times=times,
        freqs=frequencies,
        band_names=np.asarray(
            band_names
        ),
        primary_freq_r=
            primary_frequency_scores,
        rt_adjusted_freq_r=
            rt_adjusted_frequency_scores,
        rt_adjusted_minus_primary_fisher_z_freq=
            frequency_difference,
        primary_band_r=
            primary_band_scores,
        rt_adjusted_band_r=
            rt_adjusted_band_scores,
        rt_adjusted_minus_primary_fisher_z_band=
            band_difference,
        response_overlap_fraction=
            response_overlap_fraction,
    )

    surface_omnibus_tables = []
    surface_cluster_tables = []
    band_omnibus_tables = []
    band_cluster_tables = []

    analysis_families = [
        (
            "main_effect_controlled_interaction",
            primary_frequency_scores,
            primary_band_scores,
            True,
        ),
        (
            "main_effect_plus_rt_controlled_interaction",
            rt_adjusted_frequency_scores,
            rt_adjusted_band_scores,
            True,
        ),
        (
            "rt_adjusted_minus_primary_fisher_z",
            frequency_difference,
            band_difference,
            False,
        ),
    ]

    for (
        family_name,
        frequency_values,
        band_values,
        values_are_correlations,
    ) in analysis_families:
        (
            surface_omnibus,
            surface_clusters,
        ) = summarize_surface_family(
            participant_values=
                frequency_values,
            family_name=
                family_name,
            frequencies=
                frequencies,
            times=times,
            n_permutations=
                n_permutations,
            n_jobs=n_jobs,
            values_are_correlations=
                values_are_correlations,
        )

        (
            band_omnibus,
            band_clusters,
        ) = summarize_band_family(
            participant_values=
                band_values,
            family_name=
                family_name,
            times=times,
            band_names=
                band_names,
            n_permutations=
                n_permutations,
            n_jobs=n_jobs,
            values_are_correlations=
                values_are_correlations,
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

    band_omnibus = pd.concat(
        band_omnibus_tables,
        ignore_index=True,
    )

    surface_clusters = (
        pd.concat(
            surface_cluster_tables,
            ignore_index=True,
        )
        if any(
            len(
                table
            )
            for table
            in surface_cluster_tables
        )
        else pd.DataFrame()
    )

    band_clusters = (
        pd.concat(
            band_cluster_tables,
            ignore_index=True,
        )
        if any(
            len(
                table
            )
            for table
            in band_cluster_tables
        )
        else pd.DataFrame()
    )

    surface_omnibus.to_csv(
        OUTPUT_ROOT
        / "surface_omnibus.csv",
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

    target_rows = []
    rt_rows = []

    for (
        participant,
        result,
    ) in zip(
        participants,
        participant_results,
    ):
        target_rows.append(
            {
                "participant":
                    participant,
                "corr_raw_interaction_with_pain":
                    float(
                        result[
                            "corr_raw_interaction_with_pain"
                        ]
                    ),
                "corr_raw_interaction_with_money":
                    float(
                        result[
                            "corr_raw_interaction_with_money"
                        ]
                    ),
                "corr_conditional_interaction_with_pain":
                    float(
                        result[
                            "corr_conditional_interaction_with_pain"
                        ]
                    ),
                "corr_conditional_interaction_with_money":
                    float(
                        result[
                            "corr_conditional_interaction_with_money"
                        ]
                    ),
                "corr_raw_with_conditional_interaction":
                    float(
                        result[
                            "corr_raw_with_conditional_interaction"
                        ]
                    ),
            }
        )

        rt_rows.append(
            {
                "participant":
                    participant,
                "n_trials":
                    int(
                        result[
                            "n_trials"
                        ]
                    ),
                "rt_min_s":
                    float(
                        result[
                            "rt_min"
                        ]
                    ),
                "rt_p05_s":
                    float(
                        result[
                            "rt_p05"
                        ]
                    ),
                "rt_median_s":
                    float(
                        result[
                            "rt_median"
                        ]
                    ),
            }
        )

    pd.DataFrame(
        target_rows
    ).to_csv(
        OUTPUT_ROOT
        / "participant_interaction_target_summary.csv",
        index=False,
    )

    pd.DataFrame(
        rt_rows
    ).to_csv(
        OUTPUT_ROOT
        / "participant_rt_summary.csv",
        index=False,
    )

    pd.DataFrame(
        {
            "time_s":
                times,
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
        "scientific_question":
            (
                "Does cue-locked decision TFR contain Pain x Money interaction "
                "information beyond additive pain and money main effects?"
            ),
        "interaction_definition":
            "(painlevel - 3) * (moneylevel - 3)",
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
                "response-before-wavelet-end"
            ),
        "direct_rt_comparison":
            (
                "one-sided cluster test of Fisher-z(RT-adjusted r) minus "
                "Fisher-z(primary r)"
            ),
        "decoder":
            "StandardScaler -> Ridge(alpha=1)",
        "score":
            "pooled held-out Pearson r against fold-specific conditional interaction target",
        "pain_design_center":
            PAIN_DESIGN_CENTER,
        "money_design_center":
            MONEY_DESIGN_CENTER,
        "morlet_half_support_s":
            MORLET_HALF_SUPPORT_S,
        "formal_inference_s": [
            INFERENCE_TMIN,
            INFERENCE_TMAX,
        ],
        "canonical_bands_hz":
            CANONICAL_BANDS,
        "n_permutations":
            n_permutations,
        "trial_set_note":
            (
                "Uses existing cue-locked decision TFR trial sets unchanged. "
                "The response-locked 250-ms RT exclusion is NOT retroactively "
                "introduced here."
            ),
        "interpretation_note":
            (
                "A significant centered-product result is evidence for joint/"
                "non-additive Pain x Money information, not yet evidence for a "
                "participant-specific subjective-value representation."
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
        "\nCUE-LOCKED PAIN x MONEY INTERACTION: FREQUENCY x TIME",
        flush=True,
    )

    print(
        surface_omnibus.to_string(
            index=False
        ),
        flush=True,
    )

    if len(
        surface_clusters
    ):
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
        "\nCUE-LOCKED PAIN x MONEY INTERACTION: CANONICAL BANDS",
        flush=True,
    )

    print(
        band_omnibus.to_string(
            index=False
        ),
        flush=True,
    )

    if len(
        band_clusters
    ):
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
        "=" * 78,
        flush=True,
    )
    print(
        "PAINREWARD CUE-LOCKED TFR: CENTERED PAIN x MONEY INTERACTION",
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
        "Sensitivity: same analysis + RT-related EEG feature residualisation",
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
            output_file = participant_score_path(
                participant
            )

            if (
                output_file.exists()
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
        "\nRunning group-level interaction inference...",
        flush=True,
    )

    run_group_analysis(
        participants=participants,
        n_permutations=
            args.n_permutations,
        n_jobs=
            args.n_jobs,
    )

    print(
        "\nCUE-LOCKED PAIN x MONEY INTERACTION ANALYSIS COMPLETE",
        flush=True,
    )


if __name__ == "__main__":
    main()
