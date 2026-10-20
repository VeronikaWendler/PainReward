#!/usr/bin/env python3
"""
06a_acceptance_inclusion_audit.py
=================================

Freeze the participant sample for subjective-value modeling before fitting
any Bayesian models.

PRIMARY RULE
------------
Keep only participants satisfying:

    0.05 < acceptance_rate < 0.95

Therefore exclude participants who:
    - accept <= 5% of valid offers, OR
    - accept >= 95% of valid offers.

The acceptance rate is computed from the full behavioral decision task, not
from EEG-retained trials and not from neural results.
"""

from __future__ import annotations

import json

import pandas as pd

from sv_model_common import (
    MAX_ACCEPTANCE_RATE,
    MIN_ACCEPTANCE_RATE,
    OUTPUT_ROOT,
    PRIMARY_INCLUSION_COLUMN,
    load_participant_behavior,
    participant_ids,
)


def main() -> None:
    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []

    for participant in participant_ids():
        behavior = load_participant_behavior(
            participant
        )

        n_trials = int(
            len(
                behavior
            )
        )

        n_accept = int(
            behavior[
                "accepted"
            ].sum()
        )

        n_reject = int(
            n_trials
            - n_accept
        )

        acceptance_rate = float(
            n_accept
            / n_trials
        )

        include_primary = bool(
            (
                acceptance_rate
                > MIN_ACCEPTANCE_RATE
            )
            and (
                acceptance_rate
                < MAX_ACCEPTANCE_RATE
            )
        )

        rows.append(
            {
                "participant":
                    participant,
                "n_valid_choice_trials":
                    n_trials,
                "n_accept":
                    n_accept,
                "n_reject":
                    n_reject,
                "acceptance_rate":
                    acceptance_rate,
                "acceptance_percent":
                    100.0
                    * acceptance_rate,
                PRIMARY_INCLUSION_COLUMN:
                    include_primary,
                "exclusion_reason":
                    (
                        ""
                        if include_primary
                        else (
                            "acceptance_rate_lte_0.05"
                            if acceptance_rate
                            <= MIN_ACCEPTANCE_RATE
                            else "acceptance_rate_gte_0.95"
                        )
                    ),
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
            "0.05 < acceptance_rate < 0.95",
        "exclude_low":
            "acceptance_rate <= 0.05",
        "exclude_high":
            "acceptance_rate >= 0.95",
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
            "full behavioral decision task",
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
        "=" * 82
    )
    print(
        "PAINREWARD SUBJECTIVE-VALUE PARTICIPANT INCLUSION AUDIT"
    )
    print(
        "PRIMARY RULE: 0.05 < acceptance rate < 0.95"
    )
    print(
        "Behavior only — full decision task"
    )
    print(
        "=" * 82
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
