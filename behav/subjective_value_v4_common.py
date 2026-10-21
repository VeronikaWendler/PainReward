#!/usr/bin/env python3
"""
subjective_value_v4_common.py
=============================

Shared code for the definitive PainReward subjective-value analysis.

Scientific model family
-----------------------
The model comparison is an adapted version of the pain-money value-integration
framework used by Park et al. (2011, Journal of Neuroscience).

We compare a 2 x 2 model family:

                           Independent                  Interactive
Linear pain               additive                     additive + M x P
Nonlinear pain            additive with P^alpha        additive + M x P^alpha

Money is kept linear in all four models. This is a deliberate parsimonious
constraint: the present scientific uncertainty concerns the shape of pain
valuation and whether pain and money interact. Park et al. allowed nonlinear
transformations of both attributes; here we avoid adding a second exponent that
is not needed for the current question and would be weakly identified by only
five money levels.

Choice model
------------
For participant i and trial t:

    eta_it =
        intercept_i
        + money_slope_i * M_z
        - pain_slope_i * P_z(alpha_i)
        + interaction_slope_i * M_z * P_z(alpha_i)

where the final term is absent in independent models and alpha_i = 1 in linear
models.

    P(accept_it) = logistic(eta_it)

There is NO separate inverse-temperature parameter. The coefficients entering
the Bernoulli likelihood are estimated directly. This avoids the redundant
products that caused severe posterior geometry problems in the previous
parameterization.

Relative subjective value
-------------------------
Utility scale is arbitrary in stochastic choice. We normalize each participant's
decision value by their positive money slope:

    k_i     = pain_slope_i / money_slope_i
    gamma_i = interaction_slope_i / money_slope_i

    SV_it =
        M_z
        - k_i * P_z(alpha_i)
        + gamma_i * M_z * P_z(alpha_i)

This normalized SV is the primary later EEG target. The unnormalized
choice-evidence term on the logit scale is also saved.

Model comparison
----------------
Models are compared by exact leave-one-decision-block-out predictive log score,
not PSIS-LOO. Each model is fit five times, each time excluding one complete
decision block. This:
  * avoids the unstable Pareto-k problem seen previously;
  * evaluates genuinely held-out choices;
  * naturally produces leakage-safe subjective-value targets for EEG.

Participant sample
------------------
The frozen participant manifest is read from the previous audit. It implements:

    keep if rounded acceptance percentage is between 5% and 95%, inclusive.

Current frozen N is expected to be 35.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


# =============================================================================
# Paths and frozen sample
# =============================================================================

BASE_PATH = Path(
    os.getenv(
        "basepath",
        "/data",
    )
)

SOURCE_BIDS = Path(
    os.getenv(
        "SOURCE_BIDS",
        "/source",
    )
)

OUTPUT_ROOT = (
    BASE_PATH
    / "derivatives"
    / "sv_modeling_v4"
)

CV_OUTPUT_ROOT = (
    OUTPUT_ROOT
    / "block_cv"
)

FULL_FIT_ROOT = (
    OUTPUT_ROOT
    / "selected_full_fit"
)

INCLUSION_MANIFEST = (
    BASE_PATH
    / "derivatives"
    / "sv_modeling_v3"
    / "participant_inclusion_symmetric_5_95.csv"
)

PRIMARY_INCLUSION_COLUMN = (
    "include_primary_gt5_lt95pct"
)


# =============================================================================
# Design constants
# =============================================================================

DESIGN_LEVELS = np.arange(
    1.0,
    6.0,
    dtype=float,
)

MONEY_MEAN = float(
    DESIGN_LEVELS.mean()
)

MONEY_SD = float(
    DESIGN_LEVELS.std(
        ddof=0
    )
)


MODEL_DEFINITIONS = {
    "linear_independent": {
        "nonlinear_pain":
            False,
        "interactive":
            False,
        "formula":
            "SV = M - k*P",
    },
    "linear_interactive": {
        "nonlinear_pain":
            False,
        "interactive":
            True,
        "formula":
            "SV = M - k*P + gamma*M*P",
    },
    "nonlinear_independent": {
        "nonlinear_pain":
            True,
        "interactive":
            False,
        "formula":
            "SV = M - k*P^alpha",
    },
    "nonlinear_interactive": {
        "nonlinear_pain":
            True,
        "interactive":
            True,
        "formula":
            "SV = M - k*P^alpha + gamma*M*P^alpha",
    },
}


# =============================================================================
# Behavioral data loading
# =============================================================================

def read_frozen_participants() -> list[str]:
    if not INCLUSION_MANIFEST.exists():
        raise FileNotFoundError(
            f"Frozen inclusion manifest not found: {INCLUSION_MANIFEST}"
        )

    manifest = pd.read_csv(
        INCLUSION_MANIFEST
    )

    if PRIMARY_INCLUSION_COLUMN not in manifest.columns:
        raise KeyError(
            f"Manifest lacks {PRIMARY_INCLUSION_COLUMN}"
        )

    include = (
        manifest[
            PRIMARY_INCLUSION_COLUMN
        ]
        .astype(
            bool
        )
    )

    participants = (
        manifest.loc[
            include,
            "participant",
        ]
        .astype(
            str
        )
        .tolist()
    )

    if not participants:
        raise RuntimeError(
            "Frozen inclusion manifest contains no included participants."
        )

    return participants


def find_decision_behavior_file(
    participant: str,
) -> Path:
    participant_root = (
        SOURCE_BIDS
        / participant
    )

    direct_candidates = [
        participant_root
        / "eeg"
        / f"{participant}_task-decision_beh.tsv",

        participant_root
        / "beh"
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

    recursive = sorted(
        path
        for path in participant_root.glob(
            "**/*decision*beh*.tsv"
        )
        if path.is_file()
    )

    unique = []

    for path in (
        existing_direct
        + recursive
    ):
        if path not in unique:
            unique.append(
                path
            )

    if len(
        unique
    ) != 1:
        raise RuntimeError(
            f"{participant}: expected one decision behavior TSV, found "
            f"{len(unique)}:\n"
            + "\n".join(
                str(path)
                for path in unique
            )
        )

    return unique[
        0
    ]


def parse_stimulus_level(
    values: pd.Series,
    prefix: str,
) -> pd.Series:
    numeric = pd.to_numeric(
        values,
        errors="coerce",
    )

    parsed = numeric.copy()

    missing = parsed.isna()

    if missing.any():
        labels = (
            values.astype(
                str
            )
            .str.strip()
            .str.strip(
                "\"'"
            )
            .str.lower()
        )

        extracted = labels.str.extract(
            rf"^{prefix}\s*([1-5])$",
            expand=False,
        )

        parsed.loc[
            missing
        ] = pd.to_numeric(
            extracted.loc[
                missing
            ],
            errors="coerce",
        )

    return parsed


def load_participant_behavior(
    participant: str,
) -> pd.DataFrame:
    path = find_decision_behavior_file(
        participant
    )

    behavior = pd.read_csv(
        path,
        sep="\t",
    )

    if "accepted" not in behavior.columns:
        raise KeyError(
            f"{participant}: no accepted column in {path}"
        )

    # Pain.
    if "painlevel" in behavior.columns:
        pain = pd.to_numeric(
            behavior[
                "painlevel"
            ],
            errors="coerce",
        )
    elif "painstim" in behavior.columns:
        pain = parse_stimulus_level(
            behavior[
                "painstim"
            ],
            "p",
        )
    else:
        raise KeyError(
            f"{participant}: cannot recover pain level."
        )

    # Money.
    if "moneylevel" in behavior.columns:
        money = pd.to_numeric(
            behavior[
                "moneylevel"
            ],
            errors="coerce",
        )
    elif "moneystim" in behavior.columns:
        money = parse_stimulus_level(
            behavior[
                "moneystim"
            ],
            "m",
        )
    else:
        raise KeyError(
            f"{participant}: cannot recover money level."
        )

    # Trial number.
    if "trialsnum" in behavior.columns:
        trial_number = pd.to_numeric(
            behavior[
                "trialsnum"
            ],
            errors="coerce",
        )
    elif (
        "blocks.thisRepN" in behavior.columns
        and "trials.thisN" in behavior.columns
    ):
        trial_number = (
            pd.to_numeric(
                behavior[
                    "blocks.thisRepN"
                ],
                errors="coerce",
            )
            * 25
            + pd.to_numeric(
                behavior[
                    "trials.thisN"
                ],
                errors="coerce",
            )
            + 1
        )
    elif len(
        behavior
    ) == 125:
        trial_number = pd.Series(
            np.arange(
                1,
                126,
                dtype=float,
            ),
            index=
                behavior.index,
        )
    else:
        raise KeyError(
            f"{participant}: cannot recover trial number."
        )

    # Block.
    if "block" in behavior.columns:
        block = pd.to_numeric(
            behavior[
                "block"
            ],
            errors="coerce",
        )
    elif "blocks.thisRepN" in behavior.columns:
        block = pd.to_numeric(
            behavior[
                "blocks.thisRepN"
            ],
            errors="coerce",
        )
    elif "blocks.thisN" in behavior.columns:
        block = pd.to_numeric(
            behavior[
                "blocks.thisN"
            ],
            errors="coerce",
        )
    else:
        block = (
            (
                trial_number
                - 1
            )
            // 25
        )

    accepted = pd.to_numeric(
        behavior[
            "accepted"
        ],
        errors="coerce",
    )

    if "choice_resp.rt" in behavior.columns:
        rt = pd.to_numeric(
            behavior[
                "choice_resp.rt"
            ],
            errors="coerce",
        )
    elif "rt" in behavior.columns:
        rt = pd.to_numeric(
            behavior[
                "rt"
            ],
            errors="coerce",
        )
    else:
        rt = pd.Series(
            np.nan,
            index=
                behavior.index,
        )

    table = pd.DataFrame(
        {
            "participant":
                participant,
            "trialsnum":
                trial_number,
            "block":
                block,
            "painlevel":
                pain,
            "moneylevel":
                money,
            "accepted":
                accepted,
            "rt":
                rt,
        }
    )

    valid = (
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
            "trialsnum"
        ].notna()
        & table[
            "block"
        ].notna()
    )

    table = table.loc[
        valid
    ].copy()

    table[
        "trialsnum"
    ] = table[
        "trialsnum"
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
        "accepted"
    ] = table[
        "accepted"
    ].astype(
        int
    )

    return table.reset_index(
        drop=True
    )


def load_all_behavior() -> tuple[
    pd.DataFrame,
    list[str],
]:
    participants = read_frozen_participants()

    tables = [
        load_participant_behavior(
            participant
        )
        for participant in participants
    ]

    data = pd.concat(
        tables,
        ignore_index=True,
    )

    data[
        "money_z"
    ] = (
        data[
            "moneylevel"
        ]
        - MONEY_MEAN
    ) / MONEY_SD

    # Linear pain predictor is precomputed. Nonlinear pain is calculated
    # inside the model because alpha differs by posterior sample/participant.
    data[
        "pain_linear_z"
    ] = (
        data[
            "painlevel"
        ]
        - MONEY_MEAN
    ) / MONEY_SD

    return (
        data,
        participants,
    )


# =============================================================================
# PyMC model
# =============================================================================

def hierarchical_normal(
    pm,
    name: str,
    population_mean_sd: float,
    population_sd_scale: float,
):
    """
    Non-centered normal participant hierarchy.
    """

    population_mean = pm.Normal(
        f"mu_{name}",
        mu=0.0,
        sigma=
            population_mean_sd,
    )

    population_sd = pm.HalfNormal(
        f"sigma_{name}",
        sigma=
            population_sd_scale,
    )

    participant_z = pm.Normal(
        f"{name}_z",
        mu=0.0,
        sigma=1.0,
        dims="participant",
    )

    return pm.Deterministic(
        name,
        population_mean
        + population_sd
        * participant_z,
        dims="participant",
    )


def build_model(
    data: pd.DataFrame,
    model_name: str,
    participant_names: Sequence[str],
):
    """
    Build one hierarchical choice model.

    Slopes that must respect known task valence are parameterized on the log
    scale:
        money_slope > 0
        pain_slope  > 0

    The interaction slope is signed.
    """

    import pymc as pm
    import pytensor.tensor as pt

    definition = MODEL_DEFINITIONS[
        model_name
    ]

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

    money_z = data[
        "money_z"
    ].to_numpy(
        dtype=float
    )

    pain_raw = data[
        "painlevel"
    ].to_numpy(
        dtype=float
    )

    pain_linear_z = data[
        "pain_linear_z"
    ].to_numpy(
        dtype=float
    )

    accepted = data[
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
        "design_level":
            np.arange(
                5
            ),
    }

    with pm.Model(
        coords=coordinates
    ) as model:
        participant_idx = pm.Data(
            "participant_idx",
            participant_index,
            dims="trial",
        )

        money_data = pm.Data(
            "money_z_data",
            money_z,
            dims="trial",
        )

        pain_raw_data = pm.Data(
            "pain_raw_data",
            pain_raw,
            dims="trial",
        )

        pain_linear_data = pm.Data(
            "pain_linear_z_data",
            pain_linear_z,
            dims="trial",
        )

        intercept = hierarchical_normal(
            pm=
                pm,
            name=
                "intercept",
            population_mean_sd=
                1.5,
            population_sd_scale=
                1.0,
        )

        log_money_slope = hierarchical_normal(
            pm=
                pm,
            name=
                "log_money_slope",
            population_mean_sd=
                0.75,
            population_sd_scale=
                0.50,
        )

        money_slope = pm.Deterministic(
            "money_slope",
            pt.exp(
                log_money_slope
            ),
            dims="participant",
        )

        log_pain_slope = hierarchical_normal(
            pm=
                pm,
            name=
                "log_pain_slope",
            population_mean_sd=
                0.75,
            population_sd_scale=
                0.50,
        )

        pain_slope = pm.Deterministic(
            "pain_slope",
            pt.exp(
                log_pain_slope
            ),
            dims="participant",
        )

        if definition[
            "nonlinear_pain"
        ]:
            # alpha = 1 corresponds to linear pain.
            # Hierarchical log-alpha keeps alpha positive while allowing
            # concave (<1) and convex (>1) pain valuation.
            log_pain_exponent = hierarchical_normal(
                pm=
                    pm,
                name=
                    "log_pain_exponent",
                population_mean_sd=
                    0.50,
                population_sd_scale=
                    0.35,
            )

            pain_exponent = pm.Deterministic(
                "pain_exponent",
                pt.exp(
                    log_pain_exponent
                ),
                dims="participant",
            )

            participant_alpha = pain_exponent[
                participant_idx
            ]

            transformed_pain = (
                pain_raw_data
                ** participant_alpha
            )

            design_levels = pt.as_tensor_variable(
                DESIGN_LEVELS
            )

            transformed_design = (
                design_levels[
                    None,
                    :
                ]
                ** pain_exponent[
                    :,
                    None,
                ]
            )

            transformed_design_mean = (
                transformed_design.mean(
                    axis=1
                )
            )

            transformed_design_sd = (
                transformed_design.std(
                    axis=1
                )
            )

            pain_feature = (
                transformed_pain
                - transformed_design_mean[
                    participant_idx
                ]
            ) / transformed_design_sd[
                participant_idx
            ]

        else:
            pain_feature = (
                pain_linear_data
            )

        linear_predictor = (
            intercept[
                participant_idx
            ]
            + money_slope[
                participant_idx
            ]
            * money_data
            - pain_slope[
                participant_idx
            ]
            * pain_feature
        )

        if definition[
            "interactive"
        ]:
            interaction_slope = hierarchical_normal(
                pm=
                    pm,
                name=
                    "interaction_slope",
                population_mean_sd=
                    0.75,
                population_sd_scale=
                    0.50,
            )

            linear_predictor = (
                linear_predictor
                + interaction_slope[
                    participant_idx
                ]
                * money_data
                * pain_feature
            )

        pm.Bernoulli(
            "accepted",
            logit_p=
                linear_predictor,
            observed=
                accepted,
            dims="trial",
        )

    return model


# =============================================================================
# Posterior extraction and held-out prediction
# =============================================================================

def posterior_samples(
    idata,
    variable_name: str,
) -> np.ndarray:
    """
    Return participant x posterior-sample array.
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


def stable_logmeanexp(
    values: np.ndarray,
    axis: int,
) -> np.ndarray:
    maximum = np.max(
        values,
        axis=axis,
        keepdims=True,
    )

    result = (
        maximum
        + np.log(
            np.mean(
                np.exp(
                    values
                    - maximum
                ),
                axis=axis,
                keepdims=True,
            )
        )
    )

    return np.squeeze(
        result,
        axis=axis,
    )


def pain_feature_from_alpha(
    pain_levels: np.ndarray,
    alpha_samples: np.ndarray,
) -> np.ndarray:
    """
    Standardized P^alpha for held-out trials.

    pain_levels:
        n_trials

    alpha_samples:
        n_trials x n_samples
    """

    transformed_trial = (
        pain_levels[
            :,
            None,
        ]
        ** alpha_samples
    )

    transformed_design = (
        DESIGN_LEVELS[
            None,
            :,
            None,
        ]
        ** alpha_samples[
            :,
            None,
            :
        ]
    )

    design_mean = transformed_design.mean(
        axis=1
    )

    design_sd = transformed_design.std(
        axis=1,
        ddof=0,
    )

    return (
        transformed_trial
        - design_mean
    ) / design_sd


def summarize_heldout_trials(
    idata,
    heldout_data: pd.DataFrame,
    model_name: str,
    participant_names: Sequence[str],
    chunk_size: int = 128,
) -> pd.DataFrame:
    """
    Calculate exact held-out predictive log density and leakage-safe subjective
    value summaries from posterior draws obtained without the held-out block.
    """

    definition = MODEL_DEFINITIONS[
        model_name
    ]

    participant_lookup = {
        participant:
            index
        for index, participant in enumerate(
            participant_names
        )
    }

    participant_index = (
        heldout_data[
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

    money_z = heldout_data[
        "money_z"
    ].to_numpy(
        dtype=float
    )

    pain_raw = heldout_data[
        "painlevel"
    ].to_numpy(
        dtype=float
    )

    pain_linear_z = heldout_data[
        "pain_linear_z"
    ].to_numpy(
        dtype=float
    )

    accepted = heldout_data[
        "accepted"
    ].to_numpy(
        dtype=int
    )

    intercept = posterior_samples(
        idata,
        "intercept",
    )

    money_slope = posterior_samples(
        idata,
        "money_slope",
    )

    pain_slope = posterior_samples(
        idata,
        "pain_slope",
    )

    if definition[
        "interactive"
    ]:
        interaction_slope = posterior_samples(
            idata,
            "interaction_slope",
        )
    else:
        interaction_slope = None

    if definition[
        "nonlinear_pain"
    ]:
        pain_exponent = posterior_samples(
            idata,
            "pain_exponent",
        )
    else:
        pain_exponent = None

    n_trials = len(
        heldout_data
    )

    pointwise_lpd = np.empty(
        n_trials,
        dtype=float,
    )

    posterior_mean_p_accept = np.empty(
        n_trials,
        dtype=float,
    )

    subjective_value_mean = np.empty(
        n_trials,
        dtype=float,
    )

    subjective_value_sd = np.empty(
        n_trials,
        dtype=float,
    )

    subjective_value_q03 = np.empty(
        n_trials,
        dtype=float,
    )

    subjective_value_q97 = np.empty(
        n_trials,
        dtype=float,
    )

    evidence_mean = np.empty(
        n_trials,
        dtype=float,
    )

    evidence_sd = np.empty(
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

        rows = np.arange(
            start,
            stop,
        )

        pidx = participant_index[
            rows
        ]

        money_chunk = money_z[
            rows
        ][
            :,
            None,
        ]

        if pain_exponent is None:
            pain_chunk = pain_linear_z[
                rows
            ][
                :,
                None,
            ]
        else:
            alpha_chunk = pain_exponent[
                pidx,
                :
            ]

            pain_chunk = pain_feature_from_alpha(
                pain_levels=
                    pain_raw[
                        rows
                    ],
                alpha_samples=
                    alpha_chunk,
            )

        money_slope_chunk = money_slope[
            pidx,
            :
        ]

        pain_slope_chunk = pain_slope[
            pidx,
            :
        ]

        # Unnormalized latent choice evidence excluding general response bias.
        evidence = (
            money_slope_chunk
            * money_chunk
            - pain_slope_chunk
            * pain_chunk
        )

        # Relative subjective value after normalizing utility scale by money.
        subjective_value = (
            money_chunk
            - (
                pain_slope_chunk
                / money_slope_chunk
            )
            * pain_chunk
        )

        if interaction_slope is not None:
            interaction_chunk = interaction_slope[
                pidx,
                :
            ]

            evidence = (
                evidence
                + interaction_chunk
                * money_chunk
                * pain_chunk
            )

            subjective_value = (
                subjective_value
                + (
                    interaction_chunk
                    / money_slope_chunk
                )
                * money_chunk
                * pain_chunk
            )

        eta = (
            intercept[
                pidx,
                :
            ]
            + evidence
        )

        y = accepted[
            rows
        ][
            :,
            None,
        ]

        # Stable Bernoulli log likelihood from logits.
        log_likelihood_samples = np.where(
            y == 1,
            -np.logaddexp(
                0.0,
                -eta,
            ),
            -np.logaddexp(
                0.0,
                eta,
            ),
        )

        pointwise_lpd[
            rows
        ] = stable_logmeanexp(
            log_likelihood_samples,
            axis=1,
        )

        probability = (
            1.0
            / (
                1.0
                + np.exp(
                    -np.clip(
                        eta,
                        -40.0,
                        40.0,
                    )
                )
            )
        )

        posterior_mean_p_accept[
            rows
        ] = probability.mean(
            axis=1
        )

        subjective_value_mean[
            rows
        ] = subjective_value.mean(
            axis=1
        )

        subjective_value_sd[
            rows
        ] = subjective_value.std(
            axis=1,
            ddof=1,
        )

        subjective_value_q03[
            rows
        ] = np.quantile(
            subjective_value,
            0.03,
            axis=1,
        )

        subjective_value_q97[
            rows
        ] = np.quantile(
            subjective_value,
            0.97,
            axis=1,
        )

        evidence_mean[
            rows
        ] = evidence.mean(
            axis=1
        )

        evidence_sd[
            rows
        ] = evidence.std(
            axis=1,
            ddof=1,
        )

    output = heldout_data[
        [
            "participant",
            "trialsnum",
            "block",
            "painlevel",
            "moneylevel",
            "accepted",
            "rt",
        ]
    ].copy()

    output[
        "model"
    ] = model_name

    output[
        "heldout_log_predictive_density"
    ] = pointwise_lpd

    output[
        "posterior_mean_p_accept"
    ] = posterior_mean_p_accept

    output[
        "subjective_value_normalized_mean"
    ] = subjective_value_mean

    output[
        "subjective_value_normalized_sd"
    ] = subjective_value_sd

    output[
        "subjective_value_normalized_q03"
    ] = subjective_value_q03

    output[
        "subjective_value_normalized_q97"
    ] = subjective_value_q97

    output[
        "choice_evidence_logit_mean"
    ] = evidence_mean

    output[
        "choice_evidence_logit_sd"
    ] = evidence_sd

    return output


# =============================================================================
# Diagnostics and participant parameter summaries
# =============================================================================

def fit_diagnostics(
    idata,
    model_name: str,
    heldout_block: int,
) -> dict:
    import arviz as az

    parameter_names = [
        "intercept",
        "money_slope",
        "pain_slope",
    ]

    if MODEL_DEFINITIONS[
        model_name
    ][
        "interactive"
    ]:
        parameter_names.append(
            "interaction_slope"
        )

    if MODEL_DEFINITIONS[
        model_name
    ][
        "nonlinear_pain"
    ]:
        parameter_names.append(
            "pain_exponent"
        )

    summary = az.summary(
        idata,
        var_names=
            parameter_names,
    )

    rhat = (
        summary[
            "r_hat"
        ]
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
    )

    return {
        "model":
            model_name,
        "heldout_block":
            int(
                heldout_block
            ),
        "max_rhat":
            float(
                np.nanmax(
                    rhat
                )
            ),
        "min_bulk_ess":
            float(
                np.nanmin(
                    summary[
                        "ess_bulk"
                    ]
                )
            ),
        "n_divergences":
            int(
                idata.sample_stats[
                    "diverging"
                ]
                .sum()
                .values
            ),
    }


def participant_parameter_means(
    idata,
    model_name: str,
    heldout_block: int,
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

    money_slope = posterior_samples(
        idata,
        "money_slope",
    )

    pain_slope = posterior_samples(
        idata,
        "pain_slope",
    )

    k = (
        pain_slope
        / money_slope
    )

    rows = []

    if MODEL_DEFINITIONS[
        model_name
    ][
        "interactive"
    ]:
        interaction_slope = posterior_samples(
            idata,
            "interaction_slope",
        )

        gamma = (
            interaction_slope
            / money_slope
        )
    else:
        interaction_slope = None
        gamma = None

    if MODEL_DEFINITIONS[
        model_name
    ][
        "nonlinear_pain"
    ]:
        alpha = posterior_samples(
            idata,
            "pain_exponent",
        )
    else:
        alpha = None

    for index, participant in enumerate(
        participant_names
    ):
        row = {
            "model":
                model_name,
            "heldout_block":
                int(
                    heldout_block
                ),
            "participant":
                participant,
            "money_slope_mean":
                float(
                    money_slope[
                        index
                    ].mean()
                ),
            "pain_slope_mean":
                float(
                    pain_slope[
                        index
                    ].mean()
                ),
            "relative_pain_weight_k_mean":
                float(
                    k[
                        index
                    ].mean()
                ),
        }

        if gamma is not None:
            row[
                "interaction_slope_mean"
            ] = float(
                interaction_slope[
                    index
                ].mean()
            )

            row[
                "relative_interaction_gamma_mean"
            ] = float(
                gamma[
                    index
                ].mean()
            )

        if alpha is not None:
            row[
                "pain_exponent_alpha_mean"
            ] = float(
                alpha[
                    index
                ].mean()
            )

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )
