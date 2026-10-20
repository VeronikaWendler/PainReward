#!/usr/bin/env python3
"""
06d_generate_oof_subjective_value_targets.py
=============================================

Generate leakage-safe participant-specific subjective decision values for EEG
decoding.

For each decision block:
    1. remove that block from the behavioral data;
    2. fit ONLY the selected subjective-value model to all remaining blocks;
    3. use the training posterior to compute decision value for held-out trials;
    4. never use held-out choices when estimating their subjective-value target.

The final CSV contains one held-out subjective-value estimate per included
participant/trial.

This is the file to merge with EEG metadata later.
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
    OOF_OUTPUT_ROOT,
    OUTPUT_ROOT,
    build_subjective_value_model,
    load_included_behavior,
    trial_posterior_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        choices=list(
            MODEL_DEFINITIONS
        ),
        default=None,
        help=(
            "Optional override. Otherwise read selected_model.json."
        ),
    )

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
        default=0.95,
    )

    parser.add_argument(
        "--random-seed",
        type=int,
        default=20261030,
    )

    return parser.parse_args()


def selected_model_name(
    command_line_model: str | None,
) -> str:
    if command_line_model is not None:
        return command_line_model

    selection_path = (
        OUTPUT_ROOT
        / "selected_model.json"
    )

    if not selection_path.exists():
        raise FileNotFoundError(
            f"Missing {selection_path}. "
            "Run 06c_select_subjective_value_model.py first."
        )

    selection = json.loads(
        selection_path.read_text()
    )

    return str(
        selection[
            "selected_model"
        ]
    )


def main() -> None:
    args = parse_args()

    OOF_OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    data, participants = load_included_behavior()

    model_name = selected_model_name(
        args.model
    )

    blocks = sorted(
        data[
            "block"
        ]
        .unique()
        .tolist()
    )

    if len(
        blocks
    ) < 2:
        raise RuntimeError(
            "Need at least two decision blocks for out-of-fold targets."
        )

    print(
        "=" * 82
    )
    print(
        "OUT-OF-FOLD SUBJECTIVE-VALUE TARGET GENERATION"
    )
    print(
        f"Model:        {model_name}"
    )
    print(
        f"Formula:      "
        f"{MODEL_DEFINITIONS[model_name]['display_formula']}"
    )
    print(
        f"Participants: {len(participants)}"
    )
    print(
        f"Blocks:       {blocks}"
    )
    print(
        "=" * 82
    )

    heldout_tables = []
    fold_diagnostics = []

    for fold_index, heldout_block in enumerate(
        blocks,
        start=1,
    ):
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

        if len(
            heldout_data
        ) == 0:
            continue

        print(
            f"\n[{fold_index}/{len(blocks)}] "
            f"Hold out block {heldout_block}",
            flush=True,
        )
        print(
            f"    training trials: {len(train_data)}",
            flush=True,
        )
        print(
            f"    held-out trials: {len(heldout_data)}",
            flush=True,
        )

        model = build_subjective_value_model(
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
                    + fold_index,
                return_inferencedata=
                    True,
                idata_kwargs={
                    "log_likelihood":
                        True,
                },
                progressbar=
                    True,
            )

        fold_netcdf = (
            OOF_OUTPUT_ROOT
            / f"{model_name}_holdout_block_{heldout_block}.nc"
        )

        idata.to_netcdf(
            fold_netcdf
        )

        heldout_summary = trial_posterior_summary(
            idata=
                idata,
            data=
                heldout_data,
            model_name=
                model_name,
        )

        heldout_summary[
            "heldout_block"
        ] = heldout_block

        heldout_summary[
            "model"
        ] = model_name

        heldout_tables.append(
            heldout_summary
        )

        parameter_summary = az.summary(
            idata,
            var_names=[
                "intercept",
                "sensitivity",
                "pain_weight",
            ]
            + (
                [
                    "interaction_weight",
                ]
                if MODEL_DEFINITIONS[
                    model_name
                ][
                    "interactive"
                ]
                else []
            ),
            hdi_prob=0.94,
        )

        n_divergences = int(
            idata.sample_stats[
                "diverging"
            ]
            .sum()
            .values
        )

        max_rhat = float(
            np.nanmax(
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
        )

        min_bulk_ess = float(
            np.nanmin(
                parameter_summary[
                    "ess_bulk"
                ]
            )
        )

        fold_diagnostics.append(
            {
                "heldout_block":
                    heldout_block,
                "n_training_trials":
                    len(
                        train_data
                    ),
                "n_heldout_trials":
                    len(
                        heldout_data
                    ),
                "n_divergences":
                    n_divergences,
                "max_rhat":
                    max_rhat,
                "min_bulk_ess":
                    min_bulk_ess,
            }
        )

        print(
            f"    diagnostics: divergences={n_divergences}; "
            f"max R-hat={max_rhat:.3f}; "
            f"min bulk ESS={min_bulk_ess:.1f}",
            flush=True,
        )

    if not heldout_tables:
        raise RuntimeError(
            "No held-out target tables were generated."
        )

    oof_targets = pd.concat(
        heldout_tables,
        ignore_index=True,
    )

    duplicate_keys = oof_targets.duplicated(
        subset=[
            "participant",
            "trialsnum",
        ],
        keep=False,
    )

    if duplicate_keys.any():
        raise RuntimeError(
            "Out-of-fold target table contains duplicate participant/trial IDs."
        )

    expected_rows = len(
        data
    )

    if len(
        oof_targets
    ) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} held-out trial targets but generated "
            f"{len(oof_targets)}."
        )

    oof_targets = oof_targets.sort_values(
        [
            "participant",
            "trialsnum",
        ]
    ).reset_index(
        drop=True
    )

    output_path = (
        OOF_OUTPUT_ROOT
        / f"{model_name}_oof_subjective_value_targets.csv"
    )

    oof_targets.to_csv(
        output_path,
        index=False,
    )

    diagnostics_table = pd.DataFrame(
        fold_diagnostics
    )

    diagnostics_table.to_csv(
        OOF_OUTPUT_ROOT
        / f"{model_name}_oof_fit_diagnostics.csv",
        index=False,
    )

    summary = {
        "model":
            model_name,
        "formula":
            MODEL_DEFINITIONS[
                model_name
            ][
                "display_formula"
            ],
        "n_participants":
            len(
                participants
            ),
        "n_trials":
            len(
                oof_targets
            ),
        "blocks":
            blocks,
        "target_column_for_eeg":
            "decision_value_posterior_mean",
        "target_uncertainty_column":
            "decision_value_posterior_sd",
        "leakage_rule":
            (
                "Each trial's subjective-value target was generated from a "
                "Bayesian model fit that excluded that trial's entire decision "
                "block and therefore did not use the held-out trial's choice."
            ),
    }

    (
        OOF_OUTPUT_ROOT
        / f"{model_name}_oof_target_summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
        )
    )

    print(
        "\nOUT-OF-FOLD TARGET GENERATION COMPLETE"
    )
    print(
        diagnostics_table.to_string(
            index=False
        )
    )
    print(
        f"\nEEG target file: {output_path}"
    )
    print(
        "\nUse decision_value_posterior_mean as the primary continuous "
        "subjective-value target in the later EEG decoder."
    )


if __name__ == "__main__":
    main()
