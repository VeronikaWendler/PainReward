#!/usr/bin/env python3
"""Cue-locked signed-voltage MVPA of participant-specific subjective value.

Primary inferential analysis
----------------------------
Predict nonlinear-interactive, participant-specific normalized SV while
conditioning BOTH EEG features and SV on linear Pain + Money main effects.
Behavioral and EEG folds are aligned: outer block k uses SV labels from one
behavioral fit that excluded block k.

Complementary analysis
----------------------
Decode the full SV target without removing objective main effects.

RT sensitivity
--------------
Keep the same Pain+Money-conditioned SV target, but additionally remove from
EEG features (training fold only): log(RT), log(RT)^2, and whether the response
has occurred by the end of the current 20-ms voltage bin. RT is never removed
from the SV target.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats

from subjective_value_mvpa_common import (
    align_participant_targets,
    centered_main_effects,
    decode_aligned_outerfold,
    load_outerfold_targets,
    participant_list_from_targets,
    pearson_r,
)


def _load_existing_voltage_module():
    path = Path(__file__).with_name("05u_eeg_voltage_decode_decision_interaction.py")
    spec = importlib.util.spec_from_file_location("pr_voltage_interaction_base", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load existing voltage helper script: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = _load_existing_voltage_module()
BASE_PATH = Path(os.getenv("basepath", Path(__file__).parent.parent.parent))
MODEL_NAME = "nonlinear_interactive"
TARGET_COLUMN = "subjective_value_normalized_mean"

OUTPUT_ROOT = (
    BASE_PATH / "derivatives" / "statistics" / "voltage_mvpa_decision_subjective_value"
)
PARTICIPANT_SCORE_DIR = OUTPUT_ROOT / "participant_scores"
PARTICIPANT_PREDICTION_DIR = OUTPUT_ROOT / "participant_predictions"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-permutations", type=int, default=5000)
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=max(1, int(os.getenv("SLURM_CPUS_PER_TASK", "1"))),
    )
    parser.add_argument("--overwrite-scores", action="store_true")
    parser.add_argument("--group-only", action="store_true")
    return parser.parse_args()


def participant_score_path(participant: str) -> Path:
    return PARTICIPANT_SCORE_DIR / f"{participant}_decision_sv_voltage_mvpa.npz"


def participant_prediction_path(participant: str) -> Path:
    return PARTICIPANT_PREDICTION_DIR / f"{participant}_decision_sv_voltage_predictions.npz"


def validate_inputs(participants: Sequence[str]) -> None:
    for participant in participants:
        required = list(BASE.decision_tfr_paths(participant).values()) + [
            BASE.decision_raw_path(participant)
        ]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(
                f"{participant}: missing required signed-voltage input(s):\n"
                + "\n".join(missing)
            )


def _replace_target_label(table: pd.DataFrame) -> pd.DataFrame:
    if "target" in table.columns:
        table = table.copy()
        table["target"] = "participant_specific_subjective_value"
    return table


def decode_one_participant(participant: str, targets: pd.DataFrame) -> None:
    voltage, times, metadata, channel_names, bin_width_s = BASE.load_binned_voltage(participant)
    metadata = metadata.reset_index(drop=True).copy()

    pain = pd.to_numeric(metadata["painlevel"], errors="raise").to_numpy(float)
    money = pd.to_numeric(metadata["moneylevel"], errors="raise").to_numpy(float)
    blocks = pd.to_numeric(metadata["block"], errors="raise").to_numpy(int)
    rt = pd.to_numeric(metadata["rt"], errors="raise").to_numpy(float)
    trialsnum = pd.to_numeric(metadata["trialsnum"], errors="raise").to_numpy(int)

    if np.any(rt <= 0) or not np.isfinite(rt).all():
        raise RuntimeError(f"{participant}: invalid RT values.")

    pain_c, money_c = centered_main_effects(pain, money)
    sv_by_fold, sv_sd_by_fold = align_participant_targets(
        participant=participant,
        metadata=metadata,
        targets=targets,
        target_column=TARGET_COLUMN,
    )

    response_has_occurred = np.stack(
        [rt <= (time_point + bin_width_s / 2.0) for time_point in times],
        axis=1,
    )

    n_times = len(times)
    n_trials = len(metadata)

    full_r = np.empty(n_times, dtype=np.float32)
    conditional_r = np.empty(n_times, dtype=np.float32)
    conditional_rt_r = np.empty(n_times, dtype=np.float32)

    full_predictions = np.empty((n_trials, n_times), dtype=np.float32)
    conditional_predictions = np.empty_like(full_predictions)
    conditional_rt_predictions = np.empty_like(full_predictions)

    full_target = None
    conditional_target = None

    print(f"    {participant}: decoding participant-specific SV from signed voltage", flush=True)

    for time_index in range(n_times):
        features = np.asarray(voltage[:, :, time_index], dtype=float)
        response_now = response_has_occurred[:, time_index]

        score, predictions, target = decode_aligned_outerfold(
            features=features,
            pain_centered=pain_c,
            money_centered=money_c,
            block_ids=blocks,
            reaction_time=rt,
            response_overlap=response_now,
            sv_by_outer_fold=sv_by_fold,
            conditional_main_effects=False,
            include_rt_controls=False,
        )
        full_r[time_index] = score
        full_predictions[:, time_index] = predictions
        if full_target is None:
            full_target = target
        elif not np.allclose(full_target, target):
            raise RuntimeError(f"{participant}: full held-out SV target changed across time.")

        score, predictions, target = decode_aligned_outerfold(
            features=features,
            pain_centered=pain_c,
            money_centered=money_c,
            block_ids=blocks,
            reaction_time=rt,
            response_overlap=response_now,
            sv_by_outer_fold=sv_by_fold,
            conditional_main_effects=True,
            include_rt_controls=False,
        )
        conditional_r[time_index] = score
        conditional_predictions[:, time_index] = predictions
        if conditional_target is None:
            conditional_target = target
        elif not np.allclose(conditional_target, target):
            raise RuntimeError(f"{participant}: conditional SV target changed across time.")

        score, predictions, target = decode_aligned_outerfold(
            features=features,
            pain_centered=pain_c,
            money_centered=money_c,
            block_ids=blocks,
            reaction_time=rt,
            response_overlap=response_now,
            sv_by_outer_fold=sv_by_fold,
            conditional_main_effects=True,
            include_rt_controls=True,
        )
        conditional_rt_r[time_index] = score
        conditional_rt_predictions[:, time_index] = predictions
        if not np.allclose(conditional_target, target):
            raise RuntimeError(
                f"{participant}: RT sensitivity changed the conditional SV target."
            )

    assert full_target is not None and conditional_target is not None

    # OOF posterior uncertainty corresponding to each trial's own held-out block.
    oof_sv_sd = np.full(n_trials, np.nan, dtype=float)
    for outer_block, sd_values in sv_sd_by_fold.items():
        oof_sv_sd[blocks == outer_block] = sd_values[blocks == outer_block]

    np.savez_compressed(
        participant_score_path(participant),
        participant=np.asarray(participant),
        times=times,
        channels=np.asarray(channel_names),
        full_sv_r=full_r,
        conditional_sv_r=conditional_r,
        conditional_sv_rt_adjusted_r=conditional_rt_r,
        n_trials=np.asarray(n_trials),
        rt_min=np.asarray(rt.min()),
        rt_p05=np.asarray(np.quantile(rt, 0.05)),
        rt_median=np.asarray(np.median(rt)),
        response_overlap_fraction=response_has_occurred.mean(axis=0).astype(np.float32),
        full_target_sd=np.asarray(np.std(full_target, ddof=1)),
        conditional_target_sd=np.asarray(np.std(conditional_target, ddof=1)),
        corr_full_sv_with_pain=np.asarray(pearson_r(full_target, pain_c)),
        corr_full_sv_with_money=np.asarray(pearson_r(full_target, money_c)),
        corr_conditional_sv_with_pain=np.asarray(pearson_r(conditional_target, pain_c)),
        corr_conditional_sv_with_money=np.asarray(pearson_r(conditional_target, money_c)),
    )

    np.savez_compressed(
        participant_prediction_path(participant),
        participant=np.asarray(participant),
        trialsnum=trialsnum,
        block=blocks,
        painlevel=pain,
        moneylevel=money,
        rt=rt,
        times=times,
        full_sv_target_oof=full_target.astype(np.float32),
        conditional_sv_target_oof=conditional_target.astype(np.float32),
        behavioral_sv_posterior_sd_oof=oof_sv_sd.astype(np.float32),
        full_sv_prediction_oof=full_predictions,
        conditional_sv_prediction_oof=conditional_predictions,
        conditional_sv_rt_adjusted_prediction_oof=conditional_rt_predictions,
    )


def run_group_analysis(
    participants: Sequence[str],
    n_permutations: int,
    n_jobs: int,
) -> None:
    results = [np.load(participant_score_path(p), allow_pickle=False) for p in participants]
    times = results[0]["times"].astype(float)
    channels = results[0]["channels"].astype(str)

    for participant, result in zip(participants, results):
        if not np.allclose(result["times"].astype(float), times):
            raise RuntimeError(f"{participant}: voltage time-grid mismatch.")
        if not np.array_equal(result["channels"].astype(str), channels):
            raise RuntimeError(f"{participant}: voltage channel-order mismatch.")

    full = np.stack([r["full_sv_r"] for r in results])
    conditional = np.stack([r["conditional_sv_r"] for r in results])
    conditional_rt = np.stack([r["conditional_sv_rt_adjusted_r"] for r in results])
    rt_minus_primary = BASE.fisher_z(conditional_rt) - BASE.fisher_z(conditional)
    overlap = np.stack([r["response_overlap_fraction"] for r in results])

    np.savez_compressed(
        OUTPUT_ROOT / "group_score_arrays.npz",
        subjects=np.asarray(participants),
        times=times,
        channels=channels,
        full_sv_r=full,
        conditional_sv_r=conditional,
        conditional_sv_rt_adjusted_r=conditional_rt,
        rt_adjusted_minus_primary_fisher_z=rt_minus_primary,
        response_overlap_fraction=overlap,
    )

    families = [
        ("full_subjective_value", full, True),
        ("pain_money_controlled_subjective_value", conditional, True),
        ("pain_money_plus_rt_controlled_subjective_value", conditional_rt, True),
        ("rt_adjusted_minus_primary_fisher_z", rt_minus_primary, False),
    ]

    omnibus_tables = []
    cluster_tables = []
    timecourse_rows = []

    for family_name, values, are_correlations in families:
        omnibus, clusters = BASE.summarize_temporal_family(
            participant_values=values,
            family_name=family_name,
            times=times,
            n_permutations=n_permutations,
            n_jobs=n_jobs,
            values_are_correlations=are_correlations,
        )
        omnibus_tables.append(_replace_target_label(omnibus))
        cluster_tables.append(_replace_target_label(clusters))

        group_mean = values.mean(axis=0)
        group_sem = stats.sem(values, axis=0)
        for idx, time_value in enumerate(times):
            timecourse_rows.append(
                {
                    "family": family_name,
                    "time_s": float(time_value),
                    "mean_score": float(group_mean[idx]),
                    "sem_score": float(group_sem[idx]),
                    "chance": 0.0,
                }
            )

    omnibus = pd.concat(omnibus_tables, ignore_index=True)
    clusters = (
        pd.concat(cluster_tables, ignore_index=True)
        if any(len(x) for x in cluster_tables)
        else pd.DataFrame()
    )
    omnibus.to_csv(OUTPUT_ROOT / "temporal_omnibus.csv", index=False)
    clusters.to_csv(OUTPUT_ROOT / "significant_clusters.csv", index=False)
    pd.DataFrame(timecourse_rows).to_csv(OUTPUT_ROOT / "group_timecourses.csv", index=False)

    target_rows = []
    rt_rows = []
    for participant, result in zip(participants, results):
        target_rows.append(
            {
                "participant": participant,
                "full_target_sd": float(result["full_target_sd"]),
                "conditional_target_sd": float(result["conditional_target_sd"]),
                "corr_full_sv_with_pain": float(result["corr_full_sv_with_pain"]),
                "corr_full_sv_with_money": float(result["corr_full_sv_with_money"]),
                "corr_conditional_sv_with_pain": float(result["corr_conditional_sv_with_pain"]),
                "corr_conditional_sv_with_money": float(result["corr_conditional_sv_with_money"]),
            }
        )
        rt_rows.append(
            {
                "participant": participant,
                "n_trials": int(result["n_trials"]),
                "rt_min_s": float(result["rt_min"]),
                "rt_p05_s": float(result["rt_p05"]),
                "rt_median_s": float(result["rt_median"]),
            }
        )

    pd.DataFrame(target_rows).to_csv(OUTPUT_ROOT / "participant_sv_target_summary.csv", index=False)
    pd.DataFrame(rt_rows).to_csv(OUTPUT_ROOT / "participant_rt_summary.csv", index=False)
    pd.DataFrame(
        {
            "time_s": times,
            "mean_fraction_trials_response_has_occurred": overlap.mean(axis=0),
            "sem_fraction_trials_response_has_occurred": stats.sem(overlap, axis=0),
        }
    ).to_csv(OUTPUT_ROOT / "response_overlap_by_time.csv", index=False)

    params = {
        "scientific_question": (
            "Does cue-locked signed-voltage EEG encode participant-specific behavioral "
            "subjective value beyond additive linear Pain and Money main effects?"
        ),
        "behavioral_model": MODEL_NAME,
        "target_column": TARGET_COLUMN,
        "behavioral_eeg_fold_alignment": (
            "outer EEG block k uses all-trial SV labels from one behavioral fit excluding block k"
        ),
        "primary_analysis": (
            "SV target and EEG features residualised within each outer training fold against "
            "centered linear Pain + Money; StandardScaler -> Ridge(alpha=1); pooled held-out r"
        ),
        "complementary_analysis": "full SV decoding without Pain/Money residualisation",
        "rt_sensitivity": (
            "same Pain+Money-conditioned SV target; EEG additionally residualised for log(RT), "
            "log(RT)^2, and response-has-occurred by the end of the 20-ms bin"
        ),
        "rt_target_rule": "RT is never residualised from the SV target",
        "formal_inference_s": [BASE.INFERENCE_TMIN, BASE.INFERENCE_TMAX],
        "n_permutations": n_permutations,
        "interpretation": (
            "A positive conditional result supports SV-related information beyond additive linear "
            "attributes; it does not uniquely separate pain curvature from Pain x Money interaction."
        ),
        "prediction_files": str(PARTICIPANT_PREDICTION_DIR),
    }
    (OUTPUT_ROOT / "params.json").write_text(json.dumps(params, indent=2))

    print("\nSUBJECTIVE-VALUE SIGNED-VOLTAGE MVPA", flush=True)
    print(omnibus.to_string(index=False), flush=True)
    if len(clusters):
        print("\nSIGNIFICANT TEMPORAL CLUSTERS", flush=True)
        print(clusters.to_string(index=False), flush=True)
    print(f"\nOutputs: {OUTPUT_ROOT}", flush=True)


def main() -> None:
    args = parse_args()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    PARTICIPANT_SCORE_DIR.mkdir(parents=True, exist_ok=True)
    PARTICIPANT_PREDICTION_DIR.mkdir(parents=True, exist_ok=True)

    targets = load_outerfold_targets(BASE_PATH, MODEL_NAME)
    participants = participant_list_from_targets(targets)
    validate_inputs(participants)

    print("=" * 88)
    print("DECISION SUBJECTIVE VALUE: SIGNED-VOLTAGE MVPA")
    print(f"Participants: {len(participants)}")
    print(f"Behavioral model: {MODEL_NAME}")
    print("Primary: conditional on linear Pain + Money")
    print("Sensitivity: same target + RT-related EEG residualisation")
    print("=" * 88, flush=True)

    if not args.group_only:
        for participant in participants:
            score_path = participant_score_path(participant)
            pred_path = participant_prediction_path(participant)
            if score_path.exists() and pred_path.exists() and not args.overwrite_scores:
                print(f"    {participant}: existing SV voltage outputs found; skipping", flush=True)
                continue
            decode_one_participant(participant, targets)

    missing = [p for p in participants if not participant_score_path(p).exists()]
    if missing:
        raise RuntimeError(f"Missing participant score files before group analysis: {missing}")

    run_group_analysis(participants, args.n_permutations, args.n_jobs)


if __name__ == "__main__":
    main()
