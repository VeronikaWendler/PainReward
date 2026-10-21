#!/usr/bin/env python3
"""
08a_compare_subjective_value_models_regularized_blockcv.py
==============================================

Regularized behavioral model comparison for PainReward subjective value.

Four literature-motivated models are compared using exact leave-one-block-out
cross-validation:

    linear_independent
    linear_interactive
    nonlinear_independent
    nonlinear_interactive

Each held-out decision block is never used to fit the parameters that generate
its predictions or its subjective-value target.

Primary model-selection score:
    sum of held-out trial log predictive densities.

The script also writes leakage-safe out-of-fold subjective values for EVERY
candidate model. Do not choose the EEG target until the model comparison and
sampler diagnostics have been inspected.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import pymc as pm

from subjective_value_v5_common import (
    CV_OUTPUT_ROOT,
    MODEL_DEFINITIONS,
    OUTPUT_ROOT,
    build_model,
    fit_diagnostics,
    load_all_behavior,
    participant_parameter_means,
    summarize_heldout_trials,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--draws",
        type=int,
        default=1500,
    )

    parser.add_argument(
        "--tune",
        type=int,
        default=1500,
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
        default=0.99,
    )

    parser.add_argument(
        "--random-seed",
        type=int,
        default=20261021,
    )

    parser.add_argument(
        "--save-fold-netcdf",
        action="store_true",
        help=(
            "Save every block-CV posterior. Off by default because the OOF "
            "summaries contain what is needed for model comparison and EEG."
        ),
    )

    return parser.parse_args()


def model_comparison_table(
    all_oof: pd.DataFrame,
    diagnostics: pd.DataFrame,
) -> pd.DataFrame:
    model_scores = (
        all_oof
        .groupby(
            "model",
            as_index=False,
        )
        .agg(
            heldout_elpd=(
                "heldout_log_predictive_density",
                "sum",
            ),
            mean_log_score=(
                "heldout_log_predictive_density",
                "mean",
            ),
            n_heldout_trials=(
                "heldout_log_predictive_density",
                "size",
            ),
        )
    )

    diagnostic_summary = (
        diagnostics
        .groupby(
            "model",
            as_index=False,
        )
        .agg(
            max_rhat=(
                "max_rhat",
                "max",
            ),
            min_bulk_ess=(
                "min_bulk_ess",
                "min",
            ),
            total_divergences=(
                "n_divergences",
                "sum",
            ),
            max_divergences_single_fold=(
                "n_divergences",
                "max",
            ),
        )
    )

    model_scores = model_scores.merge(
        diagnostic_summary,
        on="model",
        how="left",
        validate="one_to_one",
    )

    best_model = (
        model_scores
        .sort_values(
            "heldout_elpd",
            ascending=False,
        )
        .iloc[
            0
        ][
            "model"
        ]
    )

    key_columns = [
        "participant",
        "trialsnum",
    ]

    best_pointwise = (
        all_oof.loc[
            all_oof[
                "model"
            ]
            == best_model,
            key_columns
            + [
                "heldout_log_predictive_density",
            ],
        ]
        .rename(
            columns={
                "heldout_log_predictive_density":
                    "best_lpd",
            }
        )
    )

    comparison_rows = []

    for _, score_row in model_scores.iterrows():
        model_name = str(
            score_row[
                "model"
            ]
        )

        model_pointwise = (
            all_oof.loc[
                all_oof[
                    "model"
                ]
                == model_name,
                key_columns
                + [
                    "heldout_log_predictive_density",
                ],
            ]
            .rename(
                columns={
                    "heldout_log_predictive_density":
                        "model_lpd",
                }
            )
        )

        paired = best_pointwise.merge(
            model_pointwise,
            on=key_columns,
            how="inner",
            validate="one_to_one",
        )

        if len(
            paired
        ) != int(
            score_row[
                "n_heldout_trials"
            ]
        ):
            raise RuntimeError(
                f"Pointwise comparison mismatch for {model_name}."
            )

        difference = (
            paired[
                "best_lpd"
            ]
            - paired[
                "model_lpd"
            ]
        )

        if model_name == best_model:
            elpd_difference = 0.0
            se_difference = 0.0
        else:
            elpd_difference = float(
                difference.sum()
            )

            se_difference = float(
                np.sqrt(
                    len(
                        difference
                    )
                    * np.var(
                        difference,
                        ddof=1,
                    )
                )
            )

        row = score_row.to_dict()

        row[
            "rank"
        ] = None

        row[
            "elpd_difference_from_best"
        ] = elpd_difference

        row[
            "se_elpd_difference"
        ] = se_difference

        comparison_rows.append(
            row
        )

    comparison = pd.DataFrame(
        comparison_rows
    ).sort_values(
        "heldout_elpd",
        ascending=False,
    ).reset_index(
        drop=True
    )

    comparison[
        "rank"
    ] = np.arange(
        len(
            comparison
        ),
        dtype=int,
    )

    comparison[
        "best_model_by_heldout_elpd"
    ] = (
        comparison[
            "rank"
        ]
        == 0
    )

    return comparison


def main() -> None:
    args = parse_args()

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    CV_OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    data, participants = load_all_behavior()

    blocks = sorted(
        data[
            "block"
        ]
        .unique()
        .tolist()
    )

    if len(
        blocks
    ) != 5:
        raise RuntimeError(
            f"Expected five decision blocks, found {blocks}"
        )

    print(
        "=" * 88
    )
    print(
        "PAINREWARD SUBJECTIVE-VALUE V5: REGULARIZED EXACT BLOCK CV"
    )
    print(
        f"Participants: {len(participants)}"
    )
    print(
        f"Trials:       {len(data)}"
    )
    print(
        f"Blocks:       {blocks}"
    )
    print(
        "Models:       linear/nonlinear pain x independent/interactive integration"
    )
    print(
        "=" * 88
    )

    metadata = {
        "participant_sample":
            "Frozen rounded 5%-95% acceptance sample",
        "n_participants":
            len(
                participants
            ),
        "n_trials":
            len(
                data
            ),
        "blocks":
            blocks,
        "models":
            MODEL_DEFINITIONS,
        "selection_score":
            (
                "sum of exact held-out trial log predictive densities across "
                "five leave-one-block-out fits"
            ),
        "primary_subjective_value_target":
            "subjective_value_normalized_mean",
        "normalization":
            (
                "participant utility divided by positive money slope; "
                "no separate inverse-temperature parameter"
            ),
    }

    (
        OUTPUT_ROOT
        / "analysis_definition.json"
    ).write_text(
        json.dumps(
            metadata,
            indent=2,
        )
    )

    all_oof_tables = []
    all_diagnostic_rows = []
    all_parameter_tables = []

    total_fits = (
        len(
            MODEL_DEFINITIONS
        )
        * len(
            blocks
        )
    )

    fit_counter = 0

    for model_index, model_name in enumerate(
        MODEL_DEFINITIONS,
        start=1,
    ):
        model_oof_tables = []

        print(
            "\n"
            + "=" * 88
        )
        print(
            f"MODEL {model_index}/{len(MODEL_DEFINITIONS)}: {model_name}"
        )
        print(
            MODEL_DEFINITIONS[
                model_name
            ][
                "formula"
            ]
        )
        print(
            "=" * 88
        )

        for fold_index, heldout_block in enumerate(
            blocks,
            start=1,
        ):
            fit_counter += 1

            train_data = data.loc[
                data[
                    "block"
                ]
                != heldout_block
            ].copy()

            heldout_data = data.loc[
                data[
                    "block"
                ]
                == heldout_block
            ].copy()

            print(
                f"\nFIT {fit_counter}/{total_fits} | "
                f"{model_name} | hold out block {heldout_block}",
                flush=True,
            )
            print(
                f"Training trials={len(train_data)}, "
                f"held-out trials={len(heldout_data)}",
                flush=True,
            )

            model = build_model(
                data=
                    train_data,
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
                        + 100
                        * model_index
                        + fold_index,
                    return_inferencedata=
                        True,
                    progressbar=
                        True,
                )

            diagnostics = fit_diagnostics(
                idata=
                    idata,
                model_name=
                    model_name,
                heldout_block=
                    heldout_block,
            )

            all_diagnostic_rows.append(
                diagnostics
            )

            print(
                f"Diagnostics: max R-hat={diagnostics['max_rhat']:.3f}; "
                f"min bulk ESS={diagnostics['min_bulk_ess']:.0f}; "
                f"divergences={diagnostics['n_divergences']}",
                flush=True,
            )

            heldout_summary = summarize_heldout_trials(
                idata=
                    idata,
                heldout_data=
                    heldout_data,
                model_name=
                    model_name,
                participant_names=
                    participants,
            )

            model_oof_tables.append(
                heldout_summary
            )

            all_oof_tables.append(
                heldout_summary
            )

            all_parameter_tables.append(
                participant_parameter_means(
                    idata=
                        idata,
                    model_name=
                        model_name,
                    heldout_block=
                        heldout_block,
                )
            )

            if args.save_fold_netcdf:
                idata.to_netcdf(
                    CV_OUTPUT_ROOT
                    / (
                        f"{model_name}_"
                        f"holdout_block_{heldout_block}.nc"
                    )
                )

        model_oof = pd.concat(
            model_oof_tables,
            ignore_index=True,
        ).sort_values(
            [
                "participant",
                "trialsnum",
            ]
        )

        expected_rows = len(
            data
        )

        if len(
            model_oof
        ) != expected_rows:
            raise RuntimeError(
                f"{model_name}: expected {expected_rows} OOF rows, "
                f"got {len(model_oof)}."
            )

        if model_oof.duplicated(
            subset=[
                "participant",
                "trialsnum",
            ]
        ).any():
            raise RuntimeError(
                f"{model_name}: duplicate participant/trial OOF rows."
            )

        model_oof.to_csv(
            CV_OUTPUT_ROOT
            / f"{model_name}_oof_subjective_values.csv",
            index=False,
        )

        print(
            f"\n{model_name} held-out ELPD = "
            f"{model_oof['heldout_log_predictive_density'].sum():.2f}",
            flush=True,
        )

    all_oof = pd.concat(
        all_oof_tables,
        ignore_index=True,
    )

    diagnostics_table = pd.DataFrame(
        all_diagnostic_rows
    )

    participant_parameters = pd.concat(
        all_parameter_tables,
        ignore_index=True,
    )

    diagnostics_table.to_csv(
        OUTPUT_ROOT
        / "block_cv_fit_diagnostics.csv",
        index=False,
    )

    participant_parameters.to_csv(
        OUTPUT_ROOT
        / "block_cv_participant_parameter_means.csv",
        index=False,
    )

    comparison = model_comparison_table(
        all_oof=
            all_oof,
        diagnostics=
            diagnostics_table,
    )

    comparison.to_csv(
        OUTPUT_ROOT
        / "model_comparison_block_cv.csv",
        index=False,
    )

    print(
        "\n"
        + "=" * 88
    )
    print(
        "EXACT LEAVE-ONE-BLOCK-OUT MODEL COMPARISON"
    )
    print(
        "=" * 88
    )
    print(
        comparison.to_string(
            index=False
        )
    )

    print(
        "\nBLOCK-CV FIT DIAGNOSTICS"
    )
    print(
        diagnostics_table.to_string(
            index=False
        )
    )

    candidate = str(
        comparison.loc[
            0,
            "model",
        ]
    )

    candidate_record = {
        "candidate_best_model_by_heldout_elpd":
            candidate,
        "formula":
            MODEL_DEFINITIONS[
                candidate
            ][
                "formula"
            ],
        "IMPORTANT":
            (
                "Inspect model-comparison uncertainty and sampler diagnostics "
                "before declaring this the selected model."
            ),
        "candidate_oof_eeg_target_file":
            str(
                CV_OUTPUT_ROOT
                / f"{candidate}_oof_subjective_values.csv"
            ),
        "primary_eeg_target_column":
            "subjective_value_normalized_mean",
    }

    (
        OUTPUT_ROOT
        / "candidate_best_model.json"
    ).write_text(
        json.dumps(
            candidate_record,
            indent=2,
        )
    )

    print(
        f"\nCandidate best model by held-out predictive score: {candidate}"
    )
    print(
        "DO NOT yet call it selected if diagnostics are poor or the "
        "comparison is effectively tied."
    )
    print(
        f"\nOutputs: {OUTPUT_ROOT}"
    )


if __name__ == "__main__":
    main()
