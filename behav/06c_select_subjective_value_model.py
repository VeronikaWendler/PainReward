#!/usr/bin/env python3
"""
06c_select_subjective_value_model.py
====================================

Select the behavioral model to use for out-of-fold subjective-value targets.

Default rule
------------
Choose the PSIS-LOO rank-0 model.

The script also reports whether the runner-up is within one reported standard
error of the top model. That does not automatically change the winner; it is a
warning that model-selection uncertainty should be acknowledged.

You can override the automatic choice with:

    --model quadratic_independent
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from sv_model_common import (
    MODEL_DEFINITIONS,
    OUTPUT_ROOT,
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
            "Optional manual selection. If omitted, choose PSIS-LOO rank 0."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    comparison_path = (
        OUTPUT_ROOT
        / "model_comparison_loo.csv"
    )

    if not comparison_path.exists():
        raise FileNotFoundError(
            f"Missing model comparison: {comparison_path}"
        )

    comparison = pd.read_csv(
        comparison_path
    )

    comparison = comparison.sort_values(
        "rank"
    ).reset_index(
        drop=True
    )

    automatic_winner = str(
        comparison.loc[
            0,
            "model",
        ]
    )

    selected_model = (
        args.model
        if args.model is not None
        else automatic_winner
    )

    if len(
        comparison
    ) > 1:
        runner_up = comparison.loc[
            1
        ]

        runner_up_diff = float(
            runner_up[
                "elpd_diff"
            ]
        )

        runner_up_dse = float(
            runner_up[
                "dse"
            ]
        )

        near_equivalent = bool(
            runner_up_diff
            <= runner_up_dse
        )
    else:
        runner_up = None
        runner_up_diff = None
        runner_up_dse = None
        near_equivalent = False

    selection_record = {
        "selected_model":
            selected_model,
        "automatic_psis_loo_winner":
            automatic_winner,
        "manual_override":
            args.model is not None,
        "selected_formula":
            MODEL_DEFINITIONS[
                selected_model
            ][
                "display_formula"
            ],
        "runner_up_model":
            (
                None
                if runner_up is None
                else str(
                    runner_up[
                        "model"
                    ]
                )
            ),
        "runner_up_elpd_diff":
            runner_up_diff,
        "runner_up_dse":
            runner_up_dse,
        "runner_up_within_one_dse":
            near_equivalent,
        "selection_rule":
            (
                "PSIS-LOO rank 0 unless explicitly overridden"
            ),
    }

    output_path = (
        OUTPUT_ROOT
        / "selected_model.json"
    )

    output_path.write_text(
        json.dumps(
            selection_record,
            indent=2,
        )
    )

    print(
        "=" * 78
    )
    print(
        "SUBJECTIVE-VALUE MODEL SELECTION"
    )
    print(
        "=" * 78
    )
    print(
        comparison.to_string(
            index=False
        )
    )
    print(
        f"\nAutomatic PSIS-LOO winner: {automatic_winner}"
    )
    print(
        f"Selected model:             {selected_model}"
    )
    print(
        f"Formula:                    "
        f"{MODEL_DEFINITIONS[selected_model]['display_formula']}"
    )

    if runner_up is not None:
        print(
            f"Runner-up:                  "
            f"{runner_up['model']}"
        )
        print(
            f"Runner-up elpd diff:        "
            f"{runner_up_diff:.3f}"
        )
        print(
            f"Runner-up dSE:              "
            f"{runner_up_dse:.3f}"
        )
        print(
            f"Within one dSE:             "
            f"{near_equivalent}"
        )

    if near_equivalent:
        print(
            "\nNOTE: the top two models are close relative to model-comparison "
            "uncertainty. The selected model can still be used for the "
            "predeclared winner-takes-target step, but this uncertainty should "
            "be reported."
        )

    print(
        f"\nSelection file: {output_path}"
    )


if __name__ == "__main__":
    main()
