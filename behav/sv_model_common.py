#!/usr/bin/env python3
"""
sv_model_common.py
==================

Shared utilities for the PainReward subjective-value modeling pipeline.

The behavioral models form a 2 x 2 comparison:

                              Independent                 Interactive
Linear pain              M - k*P                 M - k*P + gamma*M*P
Quadratic pain            M - k*P^2               M - k*P^2 + gamma*M*P^2

The implementation uses centered/scaled design features so priors are comparable
across linear and quadratic families.

Primary participant inclusion rule
----------------------------------
Keep participants only when:

    0.05 < acceptance_rate < 0.95

Thus participants who accept at least 95% OR at most 5% of valid decision
trials are excluded before model fitting.

The choice model is:

    logit P(accept_it)
        = intercept_i + sensitivity_i * decision_value_it

Money's coefficient inside decision value is fixed to +1. This anchors the
subjective-value scale. Participant sensitivity remains free and positive.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, Sequence, Tuple

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm


# =============================================================================
# Project paths
# =============================================================================

BASE_PATH = Path(
    os.getenv(
        "basepath",
        Path(__file__).parent.parent.parent,
    )
)

BEHAVIOR_ROOT = Path(
    os.getenv(
        "SOURCE_BIDS",
        BASE_PATH,
    )
)

OUTPUT_ROOT = (
    BASE_PATH
    / "derivatives"
    / "sv_modeling_v3"
)

MODEL_OUTPUT_ROOT = (
    OUTPUT_ROOT
    / "model_fits"
)

OOF_OUTPUT_ROOT = (
    OUTPUT_ROOT
    / "oof_subjective_value"
)

INCLUSION_MANIFEST = (
    OUTPUT_ROOT
    / "participant_inclusion_symmetric_5_95.csv"
)


# =============================================================================
# Frozen participant-inclusion rule
# =============================================================================

MIN_ACCEPTANCE_RATE = 0.05
MAX_ACCEPTANCE_RATE = 0.95

PRIMARY_INCLUSION_COLUMN = (
    "include_primary_gt5_lt95pct"
)


# =============================================================================
# Factorial design coding
# =============================================================================

DESIGN_LEVELS = np.arange(
    1.0,
    6.0,
)

DESIGN_LINEAR_MEAN = float(
    DESIGN_LEVELS.mean()
)

DESIGN_LINEAR_SD = float(
    DESIGN_LEVELS.std(
        ddof=0
    )
)

DESIGN_QUADRATIC_VALUES = (
    DESIGN_LEVELS ** 2
)

DESIGN_QUADRATIC_MEAN = float(
    DESIGN_QUADRATIC_VALUES.mean()
)

DESIGN_QUADRATIC_SD = float(
    DESIGN_QUADRATIC_VALUES.std(
        ddof=0
    )
)


MODEL_DEFINITIONS: Dict[str, dict] = {
    "linear_independent": {
        "pain_shape":
            "linear",
        "interactive":
            False,
        "display_formula":
            "DV = M - k*P",
    },
    "quadratic_independent": {
        "pain_shape":
            "quadratic",
        "interactive":
            False,
        "display_formula":
            "DV = M - k*P^2",
    },
    "linear_interactive": {
        "pain_shape":
            "linear",
        "interactive":
            True,
        "display_formula":
            "DV = M - k*P + gamma*M*P",
    },
    "quadratic_interactive": {
        "pain_shape":
            "quadratic",
        "interactive":
            True,
        "display_formula":
            "DV = M - k*P^2 + gamma*M*P^2",
    },
}


# =============================================================================
# Behavioral file discovery and loading
# =============================================================================

def participant_ids() -> list[str]:
    participants = sorted(
        path.name
        for path in BEHAVIOR_ROOT.glob(
            "sub-*"
        )
        if path.exists()
    )

    if not participants:
        raise RuntimeError(
            f"No sub-* participant folders found under {BEHAVIOR_ROOT}"
        )

    return participants


def find_decision_behavior_file(
    participant: str,
) -> Path:
    participant_root = (
        BEHAVIOR_ROOT
        / participant
    )

    direct_candidates = [
        participant_root
        / "beh"
        / f"{participant}_task-decision_beh.tsv",

        participant_root
        / "eeg"
        / f"{participant}_task-decision_beh.tsv",

        participant_root
        / f"{participant}_task-decision_beh.tsv",
    ]

    existing_direct = [
        path
        for path in direct_candidates
        if path.exists()
    ]

    if len(
        existing_direct
    ) == 1:
        return existing_direct[
            0
        ]

    recursive_candidates = sorted(
        path
        for path in participant_root.glob(
            "**/*decision*beh*.tsv"
        )
        if path.is_file()
    )

    unique_candidates = []

    for path in (
        existing_direct
        + recursive_candidates
    ):
        if path not in unique_candidates:
            unique_candidates.append(
                path
            )

    if len(
        unique_candidates
    ) != 1:
        raise RuntimeError(
            f"{participant}: expected exactly one decision behavior TSV, "
            f"found {len(unique_candidates)}:\n"
            + "\n".join(
                str(path)
                for path in unique_candidates
            )
        )

    return unique_candidates[
        0
    ]


def load_participant_behavior(
    participant: str,
) -> pd.DataFrame:
    behavior_path = find_decision_behavior_file(
        participant
    )

    behavior = pd.read_csv(
        behavior_path,
        sep="\t",
    )

    required_columns = [
        "painlevel",
        "moneylevel",
        "accepted",
        "block",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in behavior.columns
    ]

    if missing_columns:
        raise KeyError(
            f"{participant}: {behavior_path} is missing columns "
            f"{missing_columns}"
        )

    table = pd.DataFrame(
        {
            "participant":
                participant,
            "painlevel":
                pd.to_numeric(
                    behavior[
                        "painlevel"
                    ],
                    errors="coerce",
                ),
            "moneylevel":
                pd.to_numeric(
                    behavior[
                        "moneylevel"
                    ],
                    errors="coerce",
                ),
            "accepted":
                pd.to_numeric(
                    behavior[
                        "accepted"
                    ],
                    errors="coerce",
                ),
            "block":
                pd.to_numeric(
                    behavior[
                        "block"
                    ],
                    errors="coerce",
                ),
        }
    )

    if "trialsnum" in behavior.columns:
        table[
            "trialsnum"
        ] = pd.to_numeric(
            behavior[
                "trialsnum"
            ],
            errors="coerce",
        )
    else:
        table[
            "trialsnum"
        ] = np.arange(
            1,
            len(
                table
            )
            + 1,
            dtype=int,
        )

    if "choice_resp.rt" in behavior.columns:
        table[
            "rt"
        ] = pd.to_numeric(
            behavior[
                "choice_resp.rt"
            ],
            errors="coerce",
        )
    elif "rt" in behavior.columns:
        table[
            "rt"
        ] = pd.to_numeric(
            behavior[
                "rt"
            ],
            errors="coerce",
        )
    else:
        table[
            "rt"
        ] = np.nan

    valid_choice = (
        table[
            "painlevel"
        ].isin(
            DESIGN_LEVELS
        )
        & table[
            "moneylevel"
        ].isin(
            DESIGN_LEVELS
        )
        & table[
            "accepted"
        ].isin(
            [
                0.0,
                1.0,
            ]
        )
        & table[
            "block"
        ].notna()
    )

    table = table.loc[
        valid_choice
    ].copy()

    table[
        "accepted"
    ] = table[
        "accepted"
    ].astype(
        int
    )

    table[
        "block"
    ] = table[
        "block"
    ].astype(
        int
    )

    table[
        "trialsnum"
    ] = table[
        "trialsnum"
    ].astype(
        int
    )

    return table.reset_index(
        drop=True
    )


def load_inclusion_manifest() -> pd.DataFrame:
    if not INCLUSION_MANIFEST.exists():
        raise FileNotFoundError(
            f"Missing inclusion manifest: {INCLUSION_MANIFEST}\n"
            "Run 06a_acceptance_inclusion_audit.py first."
        )

    manifest = pd.read_csv(
        INCLUSION_MANIFEST
    )

    if PRIMARY_INCLUSION_COLUMN not in manifest.columns:
        raise KeyError(
            f"Inclusion manifest has no column "
            f"{PRIMARY_INCLUSION_COLUMN}"
        )

    return manifest


def included_participants() -> list[str]:
    manifest = load_inclusion_manifest()

    keep = (
        manifest[
            PRIMARY_INCLUSION_COLUMN
        ]
        .astype(
            bool
        )
    )

    participants = (
        manifest.loc[
            keep,
            "participant",
        ]
        .astype(
            str
        )
        .tolist()
    )

    if not participants:
        raise RuntimeError(
            "No participants pass the frozen symmetric 5%-95% rule."
        )

    return participants


def load_included_behavior() -> Tuple[
    pd.DataFrame,
    list[str],
]:
    participants = included_participants()

    tables = [
        load_participant_behavior(
            participant
        )
        for participant
        in participants
    ]

    data = pd.concat(
        tables,
        ignore_index=True,
    )

    data = add_design_features(
        data
    )

    return (
        data,
        participants,
    )


# =============================================================================
# Design features
# =============================================================================

def add_design_features(
    data: pd.DataFrame,
) -> pd.DataFrame:
    data = data.copy()

    data[
        "money_scaled"
    ] = (
        data[
            "moneylevel"
        ]
        - DESIGN_LINEAR_MEAN
    ) / DESIGN_LINEAR_SD

    data[
        "pain_linear_scaled"
    ] = (
        data[
            "painlevel"
        ]
        - DESIGN_LINEAR_MEAN
    ) / DESIGN_LINEAR_SD

    data[
        "pain_quadratic_scaled"
    ] = (
        data[
            "painlevel"
        ] ** 2
        - DESIGN_QUADRATIC_MEAN
    ) / DESIGN_QUADRATIC_SD

    return data


def pain_feature_column(
    model_name: str,
) -> str:
    definition = MODEL_DEFINITIONS[
        model_name
    ]

    if definition[
        "pain_shape"
    ] == "linear":
        return (
            "pain_linear_scaled"
        )

    if definition[
        "pain_shape"
    ] == "quadratic":
        return (
            "pain_quadratic_scaled"
        )

    raise ValueError(
        f"Unknown pain shape: "
        f"{definition['pain_shape']}"
    )


# =============================================================================
# Hierarchical Bayesian model
# =============================================================================

def hierarchical_normal(
    name: str,
    population_mean_prior_sd: float,
    population_sd_prior_sd: float,
):
    """
    Non-centered participant-level normal hierarchy.
    """

    population_mean = pm.Normal(
        f"mu_{name}",
        mu=0.0,
        sigma=
            population_mean_prior_sd,
    )

    population_sd = pm.HalfNormal(
        f"sigma_{name}",
        sigma=
            population_sd_prior_sd,
    )

    participant_z = pm.Normal(
        f"{name}_z",
        mu=0.0,
        sigma=1.0,
        dims="participant",
    )

    participant_value = pm.Deterministic(
        name,
        population_mean
        + population_sd
        * participant_z,
        dims="participant",
    )

    return participant_value


def build_subjective_value_model(
    data: pd.DataFrame,
    model_name: str,
    participant_names: Sequence[str] | None = None,
) -> pm.Model:
    if model_name not in MODEL_DEFINITIONS:
        raise ValueError(
            f"Unknown model: {model_name}"
        )

    if participant_names is None:
        participant_names = sorted(
            data[
                "participant"
            ]
            .astype(
                str
            )
            .unique()
            .tolist()
        )

    participant_names = list(
        participant_names
    )

    participant_lookup = {
        participant:
            index
        for index, participant in enumerate(
            participant_names
        )
    }

    unknown_participants = sorted(
        set(
            data[
                "participant"
            ].astype(
                str
            )
        )
        - set(
            participant_names
        )
    )

    if unknown_participants:
        raise RuntimeError(
            "Behavior contains participants absent from model coordinates: "
            + ", ".join(
                unknown_participants
            )
        )

    participant_index = (
        data[
            "participant"
        ]
        .astype(
            str
        )
        .map(
            participant_lookup
        )
        .to_numpy(
            dtype=int
        )
    )

    money_feature = data[
        "money_scaled"
    ].to_numpy(
        dtype=float
    )

    pain_feature = data[
        pain_feature_column(
            model_name
        )
    ].to_numpy(
        dtype=float
    )

    observed_acceptance = data[
        "accepted"
    ].to_numpy(
        dtype=int
    )

    coordinates = {
        "participant":
            participant_names,
        "trial":
            np.arange(
                len(
                    data
                )
            ),
    }

    definition = MODEL_DEFINITIONS[
        model_name
    ]

    with pm.Model(
        coords=coordinates
    ) as model:
        participant_idx = pm.Data(
            "participant_idx",
            participant_index,
            dims="trial",
        )

        money_data = pm.Data(
            "money_feature",
            money_feature,
            dims="trial",
        )

        pain_data = pm.Data(
            "pain_feature",
            pain_feature,
            dims="trial",
        )

        # --------------------------------------------------------------
        # General participant acceptance bias.
        # --------------------------------------------------------------

        intercept = hierarchical_normal(
            name=
                "intercept",
            population_mean_prior_sd=
                2.0,
            population_sd_prior_sd=
                1.5,
        )

        # --------------------------------------------------------------
        # Positive participant choice sensitivity.
        #
        # A hierarchy on log sensitivity keeps the parameter positive.
        # --------------------------------------------------------------

        log_sensitivity = hierarchical_normal(
            name=
                "log_sensitivity",
            population_mean_prior_sd=
                1.0,
            population_sd_prior_sd=
                0.75,
        )

        sensitivity = pm.Deterministic(
            "sensitivity",
            pm.math.exp(
                log_sensitivity
            ),
            dims="participant",
        )

        # --------------------------------------------------------------
        # Positive pain weight relative to money.
        #
        # Money has fixed coefficient +1 inside decision value, which
        # identifies the relative pain weight.
        # --------------------------------------------------------------

        log_pain_weight = hierarchical_normal(
            name=
                "log_pain_weight",
            population_mean_prior_sd=
                1.0,
            population_sd_prior_sd=
                0.75,
        )

        pain_weight = pm.Deterministic(
            "pain_weight",
            pm.math.exp(
                log_pain_weight
            ),
            dims="participant",
        )

        decision_value = (
            money_data
            - pain_weight[
                participant_idx
            ]
            * pain_data
        )

        # --------------------------------------------------------------
        # Optional centered/scaled Pain x Money interaction.
        # --------------------------------------------------------------

        if definition[
            "interactive"
        ]:
            interaction_weight = hierarchical_normal(
                name=
                    "interaction_weight",
                population_mean_prior_sd=
                    1.0,
                population_sd_prior_sd=
                    0.75,
            )

            decision_value = (
                decision_value
                + interaction_weight[
                    participant_idx
                ]
                * money_data
                * pain_data
            )

        logit_accept = (
            intercept[
                participant_idx
            ]
            + sensitivity[
                participant_idx
            ]
            * decision_value
        )

        pm.Bernoulli(
            "accepted",
            logit_p=
                logit_accept,
            observed=
                observed_acceptance,
            dims="trial",
        )

    return model


# =============================================================================
# Posterior extraction
# =============================================================================

def stacked_participant_samples(
    idata: az.InferenceData,
    variable_name: str,
) -> np.ndarray:
    """
    Return posterior samples as participant x sample.
    """

    return (
        idata.posterior[
            variable_name
        ]
        .stack(
            sample=(
                "chain",
                "draw",
            )
        )
        .transpose(
            "participant",
            "sample",
        )
        .values
    )


def posterior_participant_parameter_table(
    idata: az.InferenceData,
    model_name: str,
) -> pd.DataFrame:
    participant_names = (
        idata.posterior.coords[
            "participant"
        ]
        .values
        .astype(
            str
        )
        .tolist()
    )

    parameter_names = [
        "intercept",
        "sensitivity",
        "pain_weight",
    ]

    if MODEL_DEFINITIONS[
        model_name
    ][
        "interactive"
    ]:
        parameter_names.append(
            "interaction_weight"
        )

    rows = []

    for participant_index, participant in enumerate(
        participant_names
    ):
        row = {
            "participant":
                participant,
        }

        for parameter_name in parameter_names:
            samples = stacked_participant_samples(
                idata,
                parameter_name,
            )[
                participant_index
            ]

            hdi = az.hdi(
                samples,
                hdi_prob=0.94,
            )

            row[
                f"{parameter_name}_mean"
            ] = float(
                samples.mean()
            )

            row[
                f"{parameter_name}_sd"
            ] = float(
                samples.std(
                    ddof=1
                )
            )

            row[
                f"{parameter_name}_hdi3"
            ] = float(
                hdi[
                    0
                ]
            )

            row[
                f"{parameter_name}_hdi97"
            ] = float(
                hdi[
                    1
                ]
            )

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


def trial_posterior_summary(
    idata: az.InferenceData,
    data: pd.DataFrame,
    model_name: str,
    chunk_size: int = 256,
) -> pd.DataFrame:
    """
    Compute posterior summaries of trialwise decision value and p(accept)
    without storing a trial x posterior-sample deterministic array in the
    inference object.
    """

    participant_names = (
        idata.posterior.coords[
            "participant"
        ]
        .values
        .astype(
            str
        )
        .tolist()
    )

    participant_lookup = {
        participant:
            index
        for index, participant in enumerate(
            participant_names
        )
    }

    participant_index = (
        data[
            "participant"
        ]
        .astype(
            str
        )
        .map(
            participant_lookup
        )
        .to_numpy(
            dtype=int
        )
    )

    money_feature = data[
        "money_scaled"
    ].to_numpy(
        dtype=float
    )

    pain_feature = data[
        pain_feature_column(
            model_name
        )
    ].to_numpy(
        dtype=float
    )

    intercept = stacked_participant_samples(
        idata,
        "intercept",
    )

    sensitivity = stacked_participant_samples(
        idata,
        "sensitivity",
    )

    pain_weight = stacked_participant_samples(
        idata,
        "pain_weight",
    )

    if MODEL_DEFINITIONS[
        model_name
    ][
        "interactive"
    ]:
        interaction_weight = stacked_participant_samples(
            idata,
            "interaction_weight",
        )
    else:
        interaction_weight = None

    n_trials = len(
        data
    )

    decision_value_mean = np.empty(
        n_trials,
        dtype=float,
    )

    decision_value_sd = np.empty(
        n_trials,
        dtype=float,
    )

    decision_value_q03 = np.empty(
        n_trials,
        dtype=float,
    )

    decision_value_q97 = np.empty(
        n_trials,
        dtype=float,
    )

    probability_accept_mean = np.empty(
        n_trials,
        dtype=float,
    )

    for start in range(
        0,
        n_trials,
        chunk_size,
    ):
        stop = min(
            start
            + chunk_size,
            n_trials,
        )

        indices = np.arange(
            start,
            stop,
        )

        participant_chunk = participant_index[
            indices
        ]

        money_chunk = money_feature[
            indices
        ][
            :,
            None,
        ]

        pain_chunk = pain_feature[
            indices
        ][
            :,
            None,
        ]

        decision_value_samples = (
            money_chunk
            - pain_weight[
                participant_chunk,
                :
            ]
            * pain_chunk
        )

        if interaction_weight is not None:
            decision_value_samples = (
                decision_value_samples
                + interaction_weight[
                    participant_chunk,
                    :
                ]
                * money_chunk
                * pain_chunk
            )

        logit_samples = (
            intercept[
                participant_chunk,
                :
            ]
            + sensitivity[
                participant_chunk,
                :
            ]
            * decision_value_samples
        )

        probability_samples = (
            1.0
            / (
                1.0
                + np.exp(
                    -np.clip(
                        logit_samples,
                        -40.0,
                        40.0,
                    )
                )
            )
        )

        decision_value_mean[
            indices
        ] = decision_value_samples.mean(
            axis=1
        )

        decision_value_sd[
            indices
        ] = decision_value_samples.std(
            axis=1,
            ddof=1,
        )

        decision_value_q03[
            indices
        ] = np.quantile(
            decision_value_samples,
            0.03,
            axis=1,
        )

        decision_value_q97[
            indices
        ] = np.quantile(
            decision_value_samples,
            0.97,
            axis=1,
        )

        probability_accept_mean[
            indices
        ] = probability_samples.mean(
            axis=1
        )

    output_columns = [
        "participant",
        "trialsnum",
        "block",
        "painlevel",
        "moneylevel",
        "accepted",
    ]

    if "rt" in data.columns:
        output_columns.append(
            "rt"
        )

    output = data[
        output_columns
    ].copy()

    output[
        "decision_value_posterior_mean"
    ] = decision_value_mean

    output[
        "decision_value_posterior_sd"
    ] = decision_value_sd

    output[
        "decision_value_q03"
    ] = decision_value_q03

    output[
        "decision_value_q97"
    ] = decision_value_q97

    output[
        "posterior_mean_p_accept"
    ] = probability_accept_mean

    return output


def posterior_cell_summary(
    trial_summary: pd.DataFrame,
) -> pd.DataFrame:
    return (
        trial_summary
        .groupby(
            [
                "painlevel",
                "moneylevel",
            ],
            as_index=False,
        )
        .agg(
            n_trials=(
                "accepted",
                "size",
            ),
            observed_acceptance_rate=(
                "accepted",
                "mean",
            ),
            predicted_acceptance_rate=(
                "posterior_mean_p_accept",
                "mean",
            ),
            mean_decision_value=(
                "decision_value_posterior_mean",
                "mean",
            ),
        )
    )
