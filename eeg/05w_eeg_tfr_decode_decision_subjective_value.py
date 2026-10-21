#!/usr/bin/env python3
"""Cue-locked TFR MVPA of participant-specific subjective value.

This is the TFR counterpart of 05v.  The primary test conditions both the
participant-specific nonlinear-interactive SV target and multichannel power on
linear Pain + Money main effects.  A secondary sensitivity analysis additionally
removes RT-related variance from EEG features only.

The frequency/time grid, canonical bands, Morlet response-overlap definition,
cluster inference and trial set are inherited from the existing 05t objective
Pain x Money interaction analysis.
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


def _load_existing_tfr_module():
    path = Path(__file__).with_name("05t_eeg_tfr_decode_decision_interaction.py")
    spec = importlib.util.spec_from_file_location("pr_tfr_interaction_base", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load existing TFR helper script: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = _load_existing_tfr_module()
BASE_PATH = Path(os.getenv("basepath", Path(__file__).parent.parent.parent))
MODEL_NAME = "nonlinear_interactive"
TARGET_COLUMN = "subjective_value_normalized_mean"

OUTPUT_ROOT = BASE_PATH / "derivatives" / "statistics" / "tfr_mvpa_decision_subjective_value"
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
    parser.add_argument(
        "--no-save-frequency-predictions",
        action="store_true",
        help="Do not save trial x frequency x time held-out neural predictions.",
    )
    return parser.parse_args()


def participant_score_path(participant: str) -> Path:
    return PARTICIPANT_SCORE_DIR / f"{participant}_decision_sv_tfr_mvpa.npz"


def participant_prediction_path(participant: str) -> Path:
    return PARTICIPANT_PREDICTION_DIR / f"{participant}_decision_sv_tfr_predictions.npz"


def validate_inputs(participants: Sequence[str]) -> None:
    for participant in participants:
        missing = [
            str(path)
            for path in BASE.decision_tfr_paths(participant).values()
            if not path.exists()
        ]
        if missing:
            raise FileNotFoundError(
                f"{participant}: missing cue-locked decision TFR input(s):\n"
                + "\n".join(missing)
            )


def _replace_target_label(table: pd.DataFrame) -> pd.DataFrame:
    if "target" in table.columns:
        table = table.copy()
        table["target"] = "participant_specific_subjective_value"
    return table


def decode_one_participant(
    participant: str,
    targets: pd.DataFrame,
    save_frequency_predictions: bool,
) -> None:
    paths = BASE.decision_tfr_paths(participant)
    power = np.load(paths["power"], mmap_mode="r")
    times = np.load(paths["times"]).astype(float)
    frequencies = np.load(paths["freqs"]).astype(float)
    metadata = pd.read_csv(paths["metadata"]).reset_index(drop=True)

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

    response_overlap = np.stack(
        [rt <= (time_point + BASE.MORLET_HALF_SUPPORT_S) for time_point in times],
        axis=1,
    )

    n_trials = len(metadata)
    n_freqs = len(frequencies)
    n_times = len(times)

    full_freq_r = np.empty((n_freqs, n_times), dtype=np.float32)
    conditional_freq_r = np.empty_like(full_freq_r)
    conditional_rt_freq_r = np.empty_like(full_freq_r)

    if save_frequency_predictions:
        full_predictions = np.empty((n_trials, n_freqs, n_times), dtype=np.float32)
        conditional_predictions = np.empty_like(full_predictions)
        conditional_rt_predictions = np.empty_like(full_predictions)
    else:
        full_predictions = conditional_predictions = conditional_rt_predictions = None

    full_target = None
    conditional_target = None

    print(f"    {participant}: frequency-resolved participant-specific SV", flush=True)

    for frequency_index in range(n_freqs):
        for time_index in range(n_times):
            features = np.asarray(power[:, :, frequency_index, time_index], dtype=float)
            overlap_now = response_overlap[:, time_index]

            score, predictions, target = decode_aligned_outerfold(
                features=features,
                pain_centered=pain_c,
                money_centered=money_c,
                block_ids=blocks,
                reaction_time=rt,
                response_overlap=overlap_now,
                sv_by_outer_fold=sv_by_fold,
                conditional_main_effects=False,
                include_rt_controls=False,
            )
            full_freq_r[frequency_index, time_index] = score
            if full_predictions is not None:
                full_predictions[:, frequency_index, time_index] = predictions
            if full_target is None:
                full_target = target

            score, predictions, target = decode_aligned_outerfold(
                features=features,
                pain_centered=pain_c,
                money_centered=money_c,
                block_ids=blocks,
                reaction_time=rt,
                response_overlap=overlap_now,
                sv_by_outer_fold=sv_by_fold,
                conditional_main_effects=True,
                include_rt_controls=False,
            )
            conditional_freq_r[frequency_index, time_index] = score
            if conditional_predictions is not None:
                conditional_predictions[:, frequency_index, time_index] = predictions
            if conditional_target is None:
                conditional_target = target

            score, predictions, target_rt = decode_aligned_outerfold(
                features=features,
                pain_centered=pain_c,
                money_centered=money_c,
                block_ids=blocks,
                reaction_time=rt,
                response_overlap=overlap_now,
                sv_by_outer_fold=sv_by_fold,
                conditional_main_effects=True,
                include_rt_controls=True,
            )
            conditional_rt_freq_r[frequency_index, time_index] = score
            if conditional_rt_predictions is not None:
                conditional_rt_predictions[:, frequency_index, time_index] = predictions

            if not np.allclose(conditional_target, target_rt):
                raise RuntimeError(f"{participant}: RT adjustment changed the conditional SV target.")

    assert full_target is not None and conditional_target is not None

    band_names = list(BASE.CANONICAL_BANDS)
    full_band_r = np.empty((len(band_names), n_times), dtype=np.float32)
    conditional_band_r = np.empty_like(full_band_r)
    conditional_rt_band_r = np.empty_like(full_band_r)

    print(f"    {participant}: canonical-band participant-specific SV", flush=True)

    for band_index, band_name in enumerate(band_names):
        low, high = BASE.CANONICAL_BANDS[band_name]
        freq_mask = (frequencies >= low) & (frequencies <= high)
        if not freq_mask.any():
            raise RuntimeError(f"{participant}: no frequencies for band {band_name}.")

        for time_index in range(n_times):
            features = np.asarray(power[:, :, freq_mask, time_index], dtype=float).mean(axis=2)
            overlap_now = response_overlap[:, time_index]

            full_band_r[band_index, time_index] = decode_aligned_outerfold(
                features, pain_c, money_c, blocks, rt, overlap_now, sv_by_fold, False, False
            )[0]
            conditional_band_r[band_index, time_index] = decode_aligned_outerfold(
                features, pain_c, money_c, blocks, rt, overlap_now, sv_by_fold, True, False
            )[0]
            conditional_rt_band_r[band_index, time_index] = decode_aligned_outerfold(
                features, pain_c, money_c, blocks, rt, overlap_now, sv_by_fold, True, True
            )[0]

    oof_sv_sd = np.full(n_trials, np.nan, dtype=float)
    for outer_block, sd_values in sv_sd_by_fold.items():
        oof_sv_sd[blocks == outer_block] = sd_values[blocks == outer_block]

    np.savez_compressed(
        participant_score_path(participant),
        participant=np.asarray(participant),
        times=times,
        freqs=frequencies,
        band_names=np.asarray(band_names),
        full_freq_r=full_freq_r,
        conditional_freq_r=conditional_freq_r,
        conditional_rt_freq_r=conditional_rt_freq_r,
        full_band_r=full_band_r,
        conditional_band_r=conditional_band_r,
        conditional_rt_band_r=conditional_rt_band_r,
        n_trials=np.asarray(n_trials),
        rt_min=np.asarray(rt.min()),
        rt_p05=np.asarray(np.quantile(rt, 0.05)),
        rt_median=np.asarray(np.median(rt)),
        response_overlap_fraction=response_overlap.mean(axis=0).astype(np.float32),
        full_target_sd=np.asarray(np.std(full_target, ddof=1)),
        conditional_target_sd=np.asarray(np.std(conditional_target, ddof=1)),
        corr_full_sv_with_pain=np.asarray(pearson_r(full_target, pain_c)),
        corr_full_sv_with_money=np.asarray(pearson_r(full_target, money_c)),
        corr_conditional_sv_with_pain=np.asarray(pearson_r(conditional_target, pain_c)),
        corr_conditional_sv_with_money=np.asarray(pearson_r(conditional_target, money_c)),
    )

    prediction_payload = dict(
        participant=np.asarray(participant),
        trialsnum=trialsnum,
        block=blocks,
        painlevel=pain,
        moneylevel=money,
        rt=rt,
        times=times,
        freqs=frequencies,
        full_sv_target_oof=full_target.astype(np.float32),
        conditional_sv_target_oof=conditional_target.astype(np.float32),
        behavioral_sv_posterior_sd_oof=oof_sv_sd.astype(np.float32),
    )
    if full_predictions is not None:
        prediction_payload.update(
            full_sv_prediction_oof=full_predictions,
            conditional_sv_prediction_oof=conditional_predictions,
            conditional_sv_rt_adjusted_prediction_oof=conditional_rt_predictions,
        )
    np.savez_compressed(participant_prediction_path(participant), **prediction_payload)


def run_group_analysis(
    participants: Sequence[str],
    n_permutations: int,
    n_jobs: int,
) -> None:
    results = [np.load(participant_score_path(p), allow_pickle=False) for p in participants]
    times = results[0]["times"].astype(float)
    freqs = results[0]["freqs"].astype(float)
    band_names = results[0]["band_names"].astype(str).tolist()

    full_freq = np.stack([r["full_freq_r"] for r in results])
    conditional_freq = np.stack([r["conditional_freq_r"] for r in results])
    conditional_rt_freq = np.stack([r["conditional_rt_freq_r"] for r in results])
    full_band = np.stack([r["full_band_r"] for r in results])
    conditional_band = np.stack([r["conditional_band_r"] for r in results])
    conditional_rt_band = np.stack([r["conditional_rt_band_r"] for r in results])

    freq_difference = BASE.fisher_z(conditional_rt_freq) - BASE.fisher_z(conditional_freq)
    band_difference = BASE.fisher_z(conditional_rt_band) - BASE.fisher_z(conditional_band)
    overlap = np.stack([r["response_overlap_fraction"] for r in results])

    np.savez_compressed(
        OUTPUT_ROOT / "group_score_arrays.npz",
        subjects=np.asarray(participants),
        times=times,
        freqs=freqs,
        band_names=np.asarray(band_names),
        full_freq_r=full_freq,
        conditional_freq_r=conditional_freq,
        conditional_rt_freq_r=conditional_rt_freq,
        rt_adjusted_minus_primary_fisher_z_freq=freq_difference,
        full_band_r=full_band,
        conditional_band_r=conditional_band,
        conditional_rt_band_r=conditional_rt_band,
        rt_adjusted_minus_primary_fisher_z_band=band_difference,
        response_overlap_fraction=overlap,
    )

    families = [
        ("full_subjective_value", full_freq, full_band, True),
        ("pain_money_controlled_subjective_value", conditional_freq, conditional_band, True),
        (
            "pain_money_plus_rt_controlled_subjective_value",
            conditional_rt_freq,
            conditional_rt_band,
            True,
        ),
        ("rt_adjusted_minus_primary_fisher_z", freq_difference, band_difference, False),
    ]

    surface_omnibus_tables = []
    surface_cluster_tables = []
    band_omnibus_tables = []
    band_cluster_tables = []

    for family_name, freq_values, band_values, are_correlations in families:
        surface_omnibus, surface_clusters = BASE.summarize_surface_family(
            participant_values=freq_values,
            family_name=family_name,
            frequencies=freqs,
            times=times,
            n_permutations=n_permutations,
            n_jobs=n_jobs,
            values_are_correlations=are_correlations,
        )
        band_omnibus, band_clusters = BASE.summarize_band_family(
            participant_values=band_values,
            family_name=family_name,
            times=times,
            band_names=band_names,
            n_permutations=n_permutations,
            n_jobs=n_jobs,
            values_are_correlations=are_correlations,
        )
        surface_omnibus_tables.append(_replace_target_label(surface_omnibus))
        surface_cluster_tables.append(_replace_target_label(surface_clusters))
        band_omnibus_tables.append(_replace_target_label(band_omnibus))
        band_cluster_tables.append(_replace_target_label(band_clusters))

    surface_omnibus = pd.concat(surface_omnibus_tables, ignore_index=True)
    band_omnibus = pd.concat(band_omnibus_tables, ignore_index=True)
    surface_clusters = (
        pd.concat(surface_cluster_tables, ignore_index=True)
        if any(len(x) for x in surface_cluster_tables)
        else pd.DataFrame()
    )
    band_clusters = (
        pd.concat(band_cluster_tables, ignore_index=True)
        if any(len(x) for x in band_cluster_tables)
        else pd.DataFrame()
    )

    surface_omnibus.to_csv(OUTPUT_ROOT / "surface_omnibus.csv", index=False)
    surface_clusters.to_csv(OUTPUT_ROOT / "surface_significant_clusters.csv", index=False)
    band_omnibus.to_csv(OUTPUT_ROOT / "band_omnibus_holm.csv", index=False)
    band_clusters.to_csv(OUTPUT_ROOT / "band_significant_clusters.csv", index=False)

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
            "mean_fraction_trials_response_overlaps_wavelet": overlap.mean(axis=0),
            "sem_fraction_trials_response_overlaps_wavelet": stats.sem(overlap, axis=0),
        }
    ).to_csv(OUTPUT_ROOT / "response_overlap_by_time.csv", index=False)

    params = {
        "scientific_question": (
            "Does cue-locked decision TFR encode participant-specific behavioral subjective "
            "value beyond additive linear Pain and Money main effects?"
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
            "log(RT)^2, and response-before-wavelet-end"
        ),
        "rt_target_rule": "RT is never residualised from the SV target",
        "morlet_half_support_s": BASE.MORLET_HALF_SUPPORT_S,
        "formal_inference_s": [BASE.INFERENCE_TMIN, BASE.INFERENCE_TMAX],
        "canonical_bands_hz": BASE.CANONICAL_BANDS,
        "n_permutations": n_permutations,
        "interpretation": (
            "A positive conditional result supports SV-related information beyond additive linear "
            "attributes; it does not uniquely separate pain curvature from Pain x Money interaction."
        ),
        "prediction_files": str(PARTICIPANT_PREDICTION_DIR),
    }
    (OUTPUT_ROOT / "params.json").write_text(json.dumps(params, indent=2))

    print("\nSUBJECTIVE-VALUE TFR MVPA: FULL FREQUENCY x TIME", flush=True)
    print(surface_omnibus.to_string(index=False), flush=True)
    if len(surface_clusters):
        print("\nSIGNIFICANT FREQUENCY x TIME CLUSTERS", flush=True)
        print(surface_clusters.to_string(index=False), flush=True)
    print("\nSUBJECTIVE-VALUE TFR MVPA: CANONICAL BANDS", flush=True)
    print(band_omnibus.to_string(index=False), flush=True)
    if len(band_clusters):
        print("\nSIGNIFICANT BAND CLUSTERS", flush=True)
        print(band_clusters.to_string(index=False), flush=True)
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
    print("DECISION SUBJECTIVE VALUE: TFR MVPA")
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
                print(f"    {participant}: existing SV TFR outputs found; skipping", flush=True)
                continue
            decode_one_participant(
                participant,
                targets,
                save_frequency_predictions=not args.no_save_frequency_predictions,
            )

    missing = [p for p in participants if not participant_score_path(p).exists()]
    if missing:
        raise RuntimeError(f"Missing participant score files before group analysis: {missing}")

    run_group_analysis(participants, args.n_permutations, args.n_jobs)


if __name__ == "__main__":
    main()
