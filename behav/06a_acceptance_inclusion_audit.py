#!/usr/bin/env python3
"""
06a_acceptance_inclusion_audit.py
=================================

Freeze the participant sample for subjective-value modeling before fitting any
Bayesian model.

PRIMARY RULE
------------
Compute each participant's acceptance percentage from the full behavioral
decision task, round it to the nearest whole percentage point, then exclude if:

    rounded acceptance percent < 5
    OR
    rounded acceptance percent > 95

Thus:
    95.2% -> 95% -> KEEP
    96.0% -> 96% -> EXCLUDE
     4.8% ->  5% -> KEEP
     4.0% ->  4% -> EXCLUDE

IMPORTANT
---------
This audit reads only the full behavioral decision-task `accepted` responses.
It does not use EEG-retained trials or neural results.

The existing manifest column name is retained for compatibility with the rest
of the subjective-value pipeline:

    include_primary_gt5_lt95pct

Its meaning after this patch is the rounded-percentage rule above.
"""

from __future__ import annotations

import json
import math

import pandas as pd

from sv_model_common import (
    OUTPUT_ROOT,
    PRIMARY_INCLUSION_COLUMN,
    load_raw_decision_behavior,
    participant_ids,
)


def round_percentage_to_nearest_integer(
    percentage: float,
) -> int:
    """
    Conventional half-up rounding for a non-negative percentage.

    With 125 trials the possible percentages occur in 0.8% steps, so exact
    x.5 ties cannot arise, but using half-up rounding makes the intended rule
    explicit rather than relying on Python's banker's rounding.
    """

    return int(
        math.floor(
            percentage
            + 0.5
        )
    )


def main() -> None:
    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []

    for participant in participant_ids():
        behavior, behavior_path = load_raw_decision_behavior(
            participant
        )

        if "accepted" not in behavior.columns:
            raise KeyError(
                f"{participant}: {behavior_path} has no `accepted` column."
            )

        accepted = pd.to_numeric(
            behavior[
                "accepted"
            ],
            errors="coerce",
        )

        valid_acceptance = accepted.isin(
            [
                0.0,
                1.0,
            ]
        )

        accepted = accepted.loc[
            valid_acceptance
        ].astype(
            int
        )

        if len(
            accepted
        ) == 0:
            raise RuntimeError(
                f"{participant}: no valid binary accepted responses in "
                f"{behavior_path}"
            )

        n_trials = int(
            len(
                accepted
            )
        )

        n_accept = int(
            accepted.sum()
        )

        n_reject = int(
            n_trials
            - n_accept
        )

        acceptance_rate = float(
            n_accept
            / n_trials
        )

        acceptance_percent = float(
            100.0
            * acceptance_rate
        )

        acceptance_percent_rounded = (
            round_percentage_to_nearest_integer(
                acceptance_percent
            )
        )

        include_primary = bool(
            (
                acceptance_percent_rounded
                >= 5
            )
            and (
                acceptance_percent_rounded
                <= 95
            )
        )

        if acceptance_percent_rounded < 5:
            exclusion_reason = (
                "rounded_acceptance_percent_lt_5"
            )
        elif acceptance_percent_rounded > 95:
            exclusion_reason = (
                "rounded_acceptance_percent_gt_95"
            )
        else:
            exclusion_reason = ""

        rows.append(
            {
                "participant":
                    participant,
                "behavior_file":
                    str(
                        behavior_path
                    ),
                "n_valid_choice_trials":
                    n_trials,
                "n_accept":
                    n_accept,
                "n_reject":
                    n_reject,
                "acceptance_rate":
                    acceptance_rate,
                "acceptance_percent":
                    acceptance_percent,
                "acceptance_percent_rounded":
                    acceptance_percent_rounded,
                PRIMARY_INCLUSION_COLUMN:
                    include_primary,
                "exclusion_reason":
                    exclusion_reason,
            }
        )

    audit = pd.DataFrame(
        rows
    ).sort_values(
        "participant"
    )

    manifest_path = (
        OUTPUT_ROOT
        / "participant_inclusion_symmetric_5_95.csv"
    )

    audit.to_csv(
        manifest_path,
        index=False,
    )

    included = audit.loc[
        audit[
            PRIMARY_INCLUSION_COLUMN
        ],
    ]

    excluded = audit.loc[
        ~audit[
            PRIMARY_INCLUSION_COLUMN
        ],
    ]

    summary = {
        "primary_rule":
            (
                "round acceptance percent to nearest whole percent; "
                "exclude if rounded percent <5 or >95"
            ),
        "keep_rule":
            "5 <= rounded_acceptance_percent <= 95",
        "n_total":
            int(
                len(
                    audit
                )
            ),
        "n_included":
            int(
                len(
                    included
                )
            ),
        "n_excluded":
            int(
                len(
                    excluded
                )
            ),
        "source":
            "full behavioral decision task accepted responses only",
    }

    (
        OUTPUT_ROOT
        / "acceptance_inclusion_summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
        )
    )

    print(
        "=" * 88
    )
    print(
        "PAINREWARD SUBJECTIVE-VALUE PARTICIPANT INCLUSION AUDIT"
    )
    print(
        "PRIMARY RULE: round acceptance %; exclude if rounded % <5 or >95"
    )
    print(
        "Behavior only — full decision task"
    )
    print(
        "=" * 88
    )

    print(
        "\nALL PARTICIPANTS"
    )

    print(
        audit[
            [
                "participant",
                "n_valid_choice_trials",
                "n_accept",
                "n_reject",
                "acceptance_percent",
                "acceptance_percent_rounded",
                PRIMARY_INCLUSION_COLUMN,
                "exclusion_reason",
            ]
        ].to_string(
            index=False,
            formatters={
                "acceptance_percent":
                    lambda value:
                    f"{value:6.2f}",
            },
        )
    )

    print(
        "\nSUMMARY"
    )
    print(
        f"  total:    {len(audit)}"
    )
    print(
        f"  included: {len(included)}"
    )
    print(
        f"  excluded: {len(excluded)}"
    )

    print(
        "\nEXCLUDED PARTICIPANTS"
    )

    if len(
        excluded
    ):
        print(
            excluded[
                [
                    "participant",
                    "n_accept",
                    "n_reject",
                    "acceptance_percent",
                    "acceptance_percent_rounded",
                    "exclusion_reason",
                ]
            ].to_string(
                index=False,
                formatters={
                    "acceptance_percent":
                        lambda value:
                        f"{value:6.2f}",
                },
            )
        )
    else:
        print(
            "None"
        )

    print(
        f"\nFrozen inclusion manifest: {manifest_path}"
    )


if __name__ == "__main__":
    main()
