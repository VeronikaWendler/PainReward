#!/usr/bin/env python3
"""
06b_fit_subjective_value_models.py
==================================

Fit and compare four hierarchical Bayesian pain-money trade-off models using
the frozen symmetric 5%-95% participant sample.

Models
------
1. linear_independent
2. quadratic_independent
3. linear_interactive
4. quadratic_interactive

Comparison
----------
Primary model comparison: PSIS-LOO expected log predictive density.

This script also saves participant-level posterior parameters and model-implied
trial/cell acceptance summaries.

IMPORTANT
---------
The full-data posterior decision values saved here are DESCRIPTIVE ONLY.
Do not use them as EEG targets.

After choosing the behavioral model, run 06d to refit that selected model with
each decision block held out and generate leakage-safe out-of-fold subjective
values for EEG decoding.
"""

from __future__ import annotations

import argparse
import json

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

from sv_model_common import (
    MODEL_DEFINITIONS,
    MODEL_OUTPUT_ROOT,
    OUTPUT_ROOT,
    build_subjective_value_model,
    load_included_behavior,
    posterior_cell_summary,
    posterior_participant_parameter_table,
    trial_posterior_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--draws",
        type=int,
        default=2000,
    )

    parser.add_argument(
        "--tune",
        type=int,
        default=2000,
    )

    parser.add_argument(
        "--chains",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--cores",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--target-accept",
        type=float,
        default=0.95,
    )

    parser.add_argument(
        "--random-seed",
        type=int,
        default=20261020,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    MODEL_OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    data, participants = load_included_behavior()

    data.to_csv(
        OUTPUT_ROOT
        / "behavioral_trials_included.csv",
        index=False,
    )

    model_definition_record = {
        "participant_inclusion_rule":
            "0.05 < acceptance_rate < 0.95",
        "models":
            MODEL_DEFINITIONS,
        "choice_rule":
            (
                "logit P(accept_it) = intercept_i "
                "+ sensitivity_i * decision_value_it"
            ),
        "scale_identification":
            (
                "money coefficient inside decision value fixed to +1; "
                "participant sensitivity positive and separately estimated"
            ),
        "predictor_scaling":
            (
                "money, linear pain, and quadratic pain centered/scaled using "
                "constants from equally weighted levels 1..5"
            ),
        "model_comparison":
            "PSIS-LOO; higher elpd_loo is better",
        "eeg_rule":
            (
                "Full-data decision values are descriptive only. "
                "Generate block-held-out decision values after model selection."
            ),
    }

    (
        OUTPUT_ROOT
        / "model_definitions.json"
    ).write_text(
        json.dumps(
            model_definition_record,
            indent=2,
        )
    )

    print(
        "=" * 82
    )
    print(
        "PAINREWARD HIERARCHICAL SUBJECTIVE-VALUE MODEL COMPARISON"
    )
    print(
        f"Participants: {len(participants)}"
    )
    print(
        f"Trials:       {len(data)}"
    )
    print(
        "Models: linear/quadratic pain x independent/interactive integration"
    )
    print(
        "=" * 82
    )

    inference_data = {}
    diagnostics = []

    for model_index, model_name in enumerate(
        MODEL_DEFINITIONS,
        start=1,
    ):
        print(
            f"\n[{model_index}/{len(MODEL_DEFINITIONS)}] "
            f"Fitting {model_name}: "
            f"{MODEL_DEFINITIONS[model_name]['display_formula']}",
            flush=True,
        )

        model = build_subjective_value_model(
            data=
                data,
            model_name=
                model_name,
            participant_names=
                participants,
        )

        with model:
            idata = pm.sample(
                draws=
                    args.draws,
                tune=
                    args.tune,
                chains=
                    args.chains,
                cores=
                    min(
                        args.cores,
                        args.chains,
                    ),
                target_accept=
                    args.target_accept,
                random_seed=
                    args.random_seed
                    + model_index,
                return_inferencedata=
                    True,
                idata_kwargs={
                    "log_likelihood":
                        True,
                },
                progressbar=
                    True,
            )

        inference_data[
            model_name
        ] = idata

        idata.to_netcdf(
            MODEL_OUTPUT_ROOT
            / f"{model_name}.nc"
        )

        parameter_variables = [
            "mu_intercept",
            "sigma_intercept",
            "mu_log_sensitivity",
            "sigma_log_sensitivity",
            "mu_log_pain_weight",
            "sigma_log_pain_weight",
            "intercept",
            "sensitivity",
            "pain_weight",
        ]

        if MODEL_DEFINITIONS[
            model_name
        ][
            "interactive"
        ]:
            parameter_variables.extend(
                [
                    "mu_interaction_weight",
                    "sigma_interaction_weight",
                    "interaction_weight",
                ]
            )

        parameter_summary = az.summary(
            idata,
            var_names=
                parameter_variables,
            hdi_prob=
                0.94,
        )

        parameter_summary.to_csv(
            MODEL_OUTPUT_ROOT
            / f"{model_name}_summary.csv"
        )

        participant_parameters = (
            posterior_participant_parameter_table(
                idata=
                    idata,
                model_name=
                    model_name,
            )
        )

        participant_parameters.to_csv(
            MODEL_OUTPUT_ROOT
            / f"{model_name}_participant_parameters.csv",
            index=False,
        )

        trial_summary = trial_posterior_summary(
            idata=
                idata,
            data=
                data,
            model_name=
                model_name,
        )

        trial_summary.to_csv(
            MODEL_OUTPUT_ROOT
            / f"{model_name}_trial_posterior_summary.csv",
            index=False,
        )

        posterior_cell_summary(
            trial_summary
        ).to_csv(
            MODEL_OUTPUT_ROOT
            / f"{model_name}_pain_money_cell_summary.csv",
            index=False,
        )

        loo_result = az.loo(
            idata,
            pointwise=True,
        )

        rhat_values = (
            parameter_summary[
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

        ess_values = parameter_summary[
            "ess_bulk"
        ]

        pareto_k = np.asarray(
            loo_result.pareto_k
        )

        diagnostics.append(
            {
                "model":
                    model_name,
                "elpd_loo":
                    float(
                        loo_result.elpd_loo
                    ),
                "se_elpd_loo":
                    float(
                        loo_result.se
                    ),
                "p_loo":
                    float(
                        loo_result.p_loo
                    ),
                "max_pareto_k":
                    float(
                        np.nanmax(
                            pareto_k
                        )
                    ),
                "n_pareto_k_gt_0p7":
                    int(
                        np.sum(
                            pareto_k
                            > 0.7
                        )
                    ),
                "max_rhat":
                    float(
                        np.nanmax(
                            rhat_values
                        )
                    ),
                "min_bulk_ess":
                    float(
                        np.nanmin(
                            ess_values
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
        )

        print(
            f"    elpd_loo={loo_result.elpd_loo:.2f}; "
            f"max R-hat={diagnostics[-1]['max_rhat']:.3f}; "
            f"divergences={diagnostics[-1]['n_divergences']}; "
            f"Pareto-k>0.7={diagnostics[-1]['n_pareto_k_gt_0p7']}",
            flush=True,
        )

    print(
        "\nComparing models with PSIS-LOO...",
        flush=True,
    )

    comparison = az.compare(
        inference_data,
        ic="loo",
        method="stacking",
        scale="log",
    )

    comparison = (
        comparison
        .reset_index()
        .rename(
            columns={
                "index":
                    "model",
            }
        )
    )

    comparison.to_csv(
        OUTPUT_ROOT
        / "model_comparison_loo.csv",
        index=False,
    )

    diagnostics_table = (
        pd.DataFrame(
            diagnostics
        )
        .sort_values(
            "elpd_loo",
            ascending=False,
        )
    )

    diagnostics_table.to_csv(
        OUTPUT_ROOT
        / "model_fit_diagnostics.csv",
        index=False,
    )

    print(
        "\nPSIS-LOO MODEL COMPARISON"
    )
    print(
        comparison.to_string(
            index=False
        )
    )

    print(
        "\nFIT DIAGNOSTICS"
    )
    print(
        diagnostics_table.to_string(
            index=False
        )
    )

    print(
        f"\nOutputs: {OUTPUT_ROOT}"
    )

    print(
        "\nSTOP HERE AND INSPECT MODEL COMPARISON/DIAGNOSTICS BEFORE "
        "GENERATING EEG TARGETS."
    )


if __name__ == "__main__":
    main()
