#!/usr/bin/env python3
"""
Behavior/timing-only audit for the proper response-locked pain/money pipeline.

No EEG power is read. This freezes:
- the broad formal response window;
- the fixed participant sample;
- the fixed-response motor/choice-specificity plan;
- the RSA nuisance structure.
"""

from __future__ import annotations

import json
from collections import Counter

import numpy as np
import pandas as pd

from response_locked_representation_common import (
    AUDIT_OUTPUT_ROOT,
    CANDIDATE_TMIN_S,
    CANDIDATE_TMAX_S,
    MOTOR_MIN_BLOCKS,
    MOTOR_MIN_GROUP_PARTICIPANTS,
    MOTOR_MIN_TRIALS,
    MOTOR_MIN_UNIQUE_LEVELS,
    PRIMARY_MIN_BLOCKS,
    PRIMARY_MIN_GROUP_PARTICIPANTS,
    PRIMARY_MIN_TRIALS,
    PRIMARY_MIN_UNIQUE_LEVELS,
    candidate_pre_response_mask,
    load_participant_list,
    load_response_axes,
    load_response_metadata,
    support_is_valid,
    trial_support_summary,
)


def main() -> None:
    AUDIT_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    participants = load_participant_list()

    reference_times = None
    reference_frequencies = None
    metadata_by_participant = {}
    participant_summary_rows = []
    response_mapping_rows = []

    print("=" * 78, flush=True)
    print("PAINREWARD RESPONSE-LOCKED REPRESENTATION DESIGN AUDIT", flush=True)
    print("BEHAVIOR/TIMING ONLY — NO EEG POWER VALUES ARE READ", flush=True)
    print(f"Participants: {len(participants)}", flush=True)
    print("=" * 78, flush=True)

    for participant in participants:
        metadata = load_response_metadata(participant)
        metadata_by_participant[participant] = metadata
        times, frequencies = load_response_axes(participant)

        if reference_times is None:
            reference_times = times
            reference_frequencies = frequencies
        else:
            if not np.allclose(times, reference_times, atol=1e-12, rtol=0.0):
                raise RuntimeError(f"{participant}: response time grid differs")
            if not np.array_equal(frequencies, reference_frequencies):
                raise RuntimeError(f"{participant}: frequency grid differs")

        participant_summary_rows.append({
            "participant": participant,
            "n_trials": int(len(metadata)),
            "accept_n": int((metadata["accepted"] == 1).sum()),
            "reject_n": int((metadata["accepted"] == 0).sum()),
            "accept_rate": float(metadata["accepted"].mean()),
            "event_rt_min_s": float(metadata["event_rt_s"].min()),
            "event_rt_p05_s": float(metadata["event_rt_s"].quantile(0.05)),
            "event_rt_median_s": float(metadata["event_rt_s"].median()),
            "event_rt_p95_s": float(metadata["event_rt_s"].quantile(0.95)),
        })

        mapping = pd.crosstab(
            metadata["accepted"],
            metadata["response_event_type"],
            dropna=False,
        )
        for accepted_value in mapping.index:
            for event_name in mapping.columns:
                response_mapping_rows.append({
                    "participant": participant,
                    "accepted": int(accepted_value),
                    "response_event_type": str(event_name),
                    "n_trials": int(mapping.loc[accepted_value, event_name]),
                })

    if reference_times is None:
        raise RuntimeError("No participant time grids loaded")

    candidate_times = reference_times[candidate_pre_response_mask(reference_times)]

    # ------------------------------------------------------------------
    # Primary support: current trial only.
    # ------------------------------------------------------------------
    primary_validity = np.zeros((len(participants), len(candidate_times)), dtype=bool)
    primary_rows = []

    for participant_index, participant in enumerate(participants):
        metadata = metadata_by_participant[participant]
        for time_index, time_point in enumerate(candidate_times):
            summary = trial_support_summary(metadata, float(time_point))
            valid = support_is_valid(
                summary,
                PRIMARY_MIN_TRIALS,
                PRIMARY_MIN_UNIQUE_LEVELS,
                PRIMARY_MIN_BLOCKS,
            )
            primary_validity[participant_index, time_index] = valid
            primary_rows.append({
                "participant": participant,
                "response_time_s": float(time_point),
                **summary,
                "valid_primary_support": bool(valid),
            })

    n_valid_by_time = primary_validity.sum(axis=0)
    qualifying_indices = np.where(n_valid_by_time >= PRIMARY_MIN_GROUP_PARTICIPANTS)[0]
    if len(qualifying_indices) == 0:
        raise RuntimeError(
            "No response time reaches the primary behavior-only support rule. "
            "Inspect the audit; do not change thresholds based on EEG results."
        )

    primary_start_index = int(qualifying_indices[0])
    primary_tmin = float(candidate_times[primary_start_index])
    primary_tmax = float(candidate_times[-1])
    primary_participant_mask = primary_validity[:, primary_start_index]
    primary_participants = [
        participant
        for participant, keep in zip(participants, primary_participant_mask)
        if keep
    ]

    # ------------------------------------------------------------------
    # What does response identity mean in this dataset?
    # ------------------------------------------------------------------
    all_metadata = pd.concat(
        [metadata_by_participant[p].assign(participant=p) for p in participants],
        ignore_index=True,
    )

    accept_events = sorted(
        all_metadata.loc[all_metadata["accepted"] == 1, "response_event_type"]
        .astype(str).unique().tolist()
    )
    reject_events = sorted(
        all_metadata.loc[all_metadata["accepted"] == 0, "response_event_type"]
        .astype(str).unique().tolist()
    )

    if len(accept_events) == 1:
        motor_kind = "accepted_equals_1"
        motor_value = 1
        motor_event = accept_events[0]
        motor_rationale = (
            "accepted==1 maps to a single EEG response-event label across the retained data; "
            "accepted-only therefore fixes both decision outcome and response-event identity."
        )
    else:
        event_counts = Counter(all_metadata["response_event_type"].astype(str))
        motor_event = event_counts.most_common(1)[0][0]
        motor_kind = "response_event_type"
        motor_value = motor_event
        motor_rationale = (
            "accepted==1 does not map to one EEG response-event label, so the motor control "
            "conditions directly on the globally most common response-event type."
        )

    # ------------------------------------------------------------------
    # Motor/fixed-response support inside the primary interval.
    # ------------------------------------------------------------------
    primary_times = candidate_times[candidate_times >= primary_tmin - 1e-12]
    motor_validity = np.zeros((len(primary_participants), len(primary_times)), dtype=bool)
    motor_rows = []

    for participant_index, participant in enumerate(primary_participants):
        metadata = metadata_by_participant[participant]
        if motor_kind == "accepted_equals_1":
            fixed_response_mask = metadata["accepted"].to_numpy(int) == 1
        else:
            fixed_response_mask = metadata["response_event_type"].astype(str).to_numpy() == str(motor_value)

        for time_index, time_point in enumerate(primary_times):
            summary = trial_support_summary(
                metadata,
                float(time_point),
                additional_mask=fixed_response_mask,
            )
            valid = support_is_valid(
                summary,
                MOTOR_MIN_TRIALS,
                MOTOR_MIN_UNIQUE_LEVELS,
                MOTOR_MIN_BLOCKS,
            )
            motor_validity[participant_index, time_index] = valid
            motor_rows.append({
                "participant": participant,
                "response_time_s": float(time_point),
                **summary,
                "valid_motor_support": bool(valid),
            })

    motor_n_by_time = motor_validity.sum(axis=0)
    motor_qualifying = np.where(motor_n_by_time >= MOTOR_MIN_GROUP_PARTICIPANTS)[0]
    if len(motor_qualifying):
        motor_start_index = int(motor_qualifying[0])
        motor_tmin = float(primary_times[motor_start_index])
        motor_tmax = float(primary_times[-1])
        motor_participant_mask = motor_validity[:, motor_start_index]
        motor_participants = [
            participant
            for participant, keep in zip(primary_participants, motor_participant_mask)
            if keep
        ]
        motor_enabled = len(motor_participants) >= MOTOR_MIN_GROUP_PARTICIPANTS
    else:
        motor_tmin = None
        motor_tmax = None
        motor_participants = []
        motor_enabled = False

    # ------------------------------------------------------------------
    # Audit accept/reject support for possible later cross-response work.
    # We do not automatically promote it to a formal EEG analysis.
    # ------------------------------------------------------------------
    accept_reject_rows = []
    for participant in participants:
        metadata = metadata_by_participant[participant]
        for accepted_value, label in [(1, "accepted"), (0, "rejected")]:
            subset = metadata.loc[metadata["accepted"] == accepted_value]
            accept_reject_rows.append({
                "participant": participant,
                "response_subset": label,
                "n_trials_total": int(len(subset)),
                "pain_levels_total": int(subset["painlevel"].nunique()),
                "money_levels_total": int(subset["moneylevel"].nunique()),
                "blocks_total": int(subset["block"].nunique()),
                "pain_min": float(subset["painlevel"].min()) if len(subset) else np.nan,
                "pain_max": float(subset["painlevel"].max()) if len(subset) else np.nan,
                "money_min": float(subset["moneylevel"].min()) if len(subset) else np.nan,
                "money_max": float(subset["moneylevel"].max()) if len(subset) else np.nan,
            })

    mapping_unique = (
        all_metadata[["accepted", "response_event_type"]]
        .drop_duplicates()
        .sort_values(["accepted", "response_event_type"])
    )
    include_response_event_rdm = bool(
        len(mapping_unique) > 2
        or all_metadata["response_event_type"].nunique() > all_metadata["accepted"].nunique()
    )

    # Save diagnostics.
    participant_summary = pd.DataFrame(participant_summary_rows)
    participant_summary.to_csv(AUDIT_OUTPUT_ROOT / "participant_behavior_summary.csv", index=False)
    pd.DataFrame(response_mapping_rows).to_csv(
        AUDIT_OUTPUT_ROOT / "accepted_by_response_event_type.csv", index=False
    )
    pd.DataFrame(primary_rows).to_csv(
        AUDIT_OUTPUT_ROOT / "primary_support_by_participant_time.csv", index=False
    )
    pd.DataFrame({
        "response_time_s": candidate_times,
        "n_participants_with_primary_support": n_valid_by_time,
    }).to_csv(AUDIT_OUTPUT_ROOT / "primary_group_support_by_time.csv", index=False)
    pd.DataFrame(motor_rows).to_csv(
        AUDIT_OUTPUT_ROOT / "motor_support_by_participant_time.csv", index=False
    )
    pd.DataFrame({
        "response_time_s": primary_times,
        "n_participants_with_motor_support": motor_n_by_time,
    }).to_csv(AUDIT_OUTPUT_ROOT / "motor_group_support_by_time.csv", index=False)
    pd.DataFrame(accept_reject_rows).to_csv(
        AUDIT_OUTPUT_ROOT / "accept_reject_total_support.csv", index=False
    )

    plan = {
        "analysis_name": "PainReward proper response-locked representation analysis",
        "design_selection_basis": "behavior and event timing only; no EEG values used",
        "candidate_response_window_s": [CANDIDATE_TMIN_S, CANDIDATE_TMAX_S],
        "current_trial_availability_rule": "event_rt_s + response_relative_time_s >= 0",
        "primary_support_rule": {
            "minimum_trials_per_participant": PRIMARY_MIN_TRIALS,
            "minimum_unique_pain_levels": PRIMARY_MIN_UNIQUE_LEVELS,
            "minimum_unique_money_levels": PRIMARY_MIN_UNIQUE_LEVELS,
            "minimum_blocks": PRIMARY_MIN_BLOCKS,
            "minimum_group_participants": PRIMARY_MIN_GROUP_PARTICIPANTS,
        },
        "primary_formal_window_s": [primary_tmin, primary_tmax],
        "primary_participants": primary_participants,
        "primary_n_participants": len(primary_participants),
        "motor_control": {
            "enabled": motor_enabled,
            "kind": motor_kind,
            "value": motor_value,
            "event_label_if_known": motor_event,
            "rationale": motor_rationale,
            "support_rule": {
                "minimum_trials_per_participant": MOTOR_MIN_TRIALS,
                "minimum_unique_pain_levels": MOTOR_MIN_UNIQUE_LEVELS,
                "minimum_unique_money_levels": MOTOR_MIN_UNIQUE_LEVELS,
                "minimum_blocks": MOTOR_MIN_BLOCKS,
                "minimum_group_participants": MOTOR_MIN_GROUP_PARTICIPANTS,
            },
            "formal_window_s": [motor_tmin, motor_tmax] if motor_enabled else None,
            "participants": motor_participants,
            "n_participants": len(motor_participants),
        },
        "rsa_controls": {
            "pain_distance": True,
            "money_distance": True,
            "log_rt_distance": True,
            "choice_mismatch_accepted": True,
            "include_response_event_mismatch": include_response_event_rdm,
        },
        "accept_response_event_labels": accept_events,
        "reject_response_event_labels": reject_events,
        "response_event_vs_accepted_unique_pairs": mapping_unique.to_dict(orient="records"),
    }

    with open(AUDIT_OUTPUT_ROOT / "response_locked_design_plan.json", "w", encoding="utf-8") as handle:
        json.dump(plan, handle, indent=2)

    print("\nPRIMARY RESPONSE-LOCKED MVPA/RSA PLAN", flush=True)
    print(f"  candidate window: {CANDIDATE_TMIN_S:.3f}..{CANDIDATE_TMAX_S:.3f} s", flush=True)
    print(
        f"  support: >= {PRIMARY_MIN_TRIALS} trials, >= {PRIMARY_MIN_UNIQUE_LEVELS} pain levels, "
        f">= {PRIMARY_MIN_UNIQUE_LEVELS} money levels, >= {PRIMARY_MIN_BLOCKS} blocks",
        flush=True,
    )
    print(f"  minimum group N: {PRIMARY_MIN_GROUP_PARTICIPANTS}", flush=True)
    print(f"  formal window: {primary_tmin:.3f}..{primary_tmax:.3f} s", flush=True)
    print(f"  fixed primary participant sample: {len(primary_participants)}", flush=True)
    print("  participants: " + ", ".join(primary_participants), flush=True)

    print("\nCHOICE / RESPONSE IDENTITY", flush=True)
    print(f"  accepted==1 event labels: {accept_events}", flush=True)
    print(f"  accepted==0 event labels: {reject_events}", flush=True)
    print(f"  motor-control definition: {motor_kind} = {motor_value}", flush=True)
    print(f"  rationale: {motor_rationale}", flush=True)

    print("\nMOTOR/CHOICE-SPECIFICITY PLAN", flush=True)
    print(f"  enabled: {motor_enabled}", flush=True)
    if motor_enabled:
        print(f"  formal window: {motor_tmin:.3f}..{motor_tmax:.3f} s", flush=True)
        print(f"  fixed motor-control participant sample: {len(motor_participants)}", flush=True)

    print("\nOVERALL ACCEPTANCE", flush=True)
    print(participant_summary[["accept_rate", "accept_n", "reject_n"]].describe().to_string(), flush=True)
    print(f"\nRSA response-event mismatch control included: {include_response_event_rdm}", flush=True)
    print(f"\nOutputs: {AUDIT_OUTPUT_ROOT}", flush=True)


if __name__ == "__main__":
    main()
