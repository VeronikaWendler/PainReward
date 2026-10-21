#!/usr/bin/env python3
"""Export fold-aligned nonlinear-interactive SV targets for EEG MVPA.

For each of the five outer decision blocks, refit the already-selected
nonlinear_interactive behavioral model on the other four blocks and calculate
SV summaries for *all* trials from that one fit.  The EEG decoder can then use
that same behavioral fit to define its labels on both the four EEG training
blocks and the held-out EEG block.

This is not a new model comparison.  It is a target-export step for strict
alignment of the behavioral and EEG outer folds.
"""

from __future__ import annotations

import argparse
import json

import pandas as pd
import pymc as pm

from subjective_value_v5_common import (
    CV_OUTPUT_ROOT,
    MODEL_DEFINITIONS,
    build_model,
    fit_diagnostics,
    load_all_behavior,
    summarize_heldout_trials,
)

MODEL_NAME = "nonlinear_interactive"
OUTPUT_DIR = CV_OUTPUT_ROOT / "eeg_outerfold_targets"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--draws", type=int, default=1500)
    parser.add_argument("--tune", type=int, default=1500)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--cores", type=int, default=4)
    parser.add_argument("--target-accept", type=float, default=0.99)
    parser.add_argument("--random-seed", type=int, default=20261021)
    parser.add_argument("--save-fold-netcdf", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data, participants = load_all_behavior()
    blocks = sorted(data["block"].unique().tolist())
    if len(blocks) != 5:
        raise RuntimeError(f"Expected five decision blocks, found {blocks}")

    print("=" * 88)
    print("EEG OUTER-FOLD SUBJECTIVE-VALUE TARGET EXPORT")
    print(f"Model:        {MODEL_NAME}")
    print(f"Formula:      {MODEL_DEFINITIONS[MODEL_NAME]['formula']}")
    print(f"Participants: {len(participants)}")
    print(f"Trials:       {len(data)}")
    print(f"Blocks:       {blocks}")
    print("=" * 88, flush=True)

    all_tables = []
    diagnostics = []

    for fold_index, heldout_block in enumerate(blocks, start=1):
        train = data.loc[data["block"] != heldout_block].copy()
        print(
            f"\nOUTER FOLD {fold_index}/{len(blocks)} | hold out block {heldout_block} | "
            f"training n={len(train)}",
            flush=True,
        )

        model = build_model(
            data=train,
            model_name=MODEL_NAME,
            participant_names=participants,
        )

        with model:
            idata = pm.sample(
                draws=args.draws,
                tune=args.tune,
                chains=args.chains,
                cores=min(args.cores, args.chains),
                target_accept=args.target_accept,
                random_seed=args.random_seed + fold_index,
                return_inferencedata=True,
                progressbar=True,
            )

        diag = fit_diagnostics(
            idata=idata,
            model_name=MODEL_NAME,
            heldout_block=heldout_block,
        )
        diagnostics.append(diag)
        print(
            f"Diagnostics: max R-hat={diag['max_rhat']:.4f}; "
            f"min bulk ESS={diag['min_bulk_ess']:.0f}; "
            f"divergences={diag['n_divergences']}",
            flush=True,
        )

        # summarize_heldout_trials is also a general posterior value-summary
        # function. Here we deliberately pass all trials so the same outer-fold
        # behavioral fit defines labels for both EEG train and EEG test trials.
        summary = summarize_heldout_trials(
            idata=idata,
            heldout_data=data,
            model_name=MODEL_NAME,
            participant_names=participants,
        )

        keep = [
            "participant",
            "trialsnum",
            "block",
            "painlevel",
            "moneylevel",
            "accepted",
            "rt",
            "subjective_value_normalized_mean",
            "subjective_value_normalized_sd",
            "subjective_value_normalized_q03",
            "subjective_value_normalized_q97",
            "choice_evidence_logit_mean",
            "choice_evidence_logit_sd",
        ]
        summary = summary[keep].copy()
        summary["outer_heldout_block"] = int(heldout_block)
        summary["is_outer_test"] = summary["block"].astype(int) == int(heldout_block)

        per_fold_path = OUTPUT_DIR / f"{MODEL_NAME}_outer_holdout_block_{heldout_block}.csv"
        summary.to_csv(per_fold_path, index=False)
        all_tables.append(summary)

        if args.save_fold_netcdf:
            idata.to_netcdf(
                OUTPUT_DIR / f"{MODEL_NAME}_outer_holdout_block_{heldout_block}.nc"
            )

    combined = pd.concat(all_tables, ignore_index=True).sort_values(
        ["outer_heldout_block", "participant", "trialsnum"]
    )

    expected = len(data) * len(blocks)
    if len(combined) != expected:
        raise RuntimeError(f"Expected {expected} combined rows, got {len(combined)}")

    combined_path = OUTPUT_DIR / f"{MODEL_NAME}_outerfold_alltrial_subjective_values.csv"
    combined.to_csv(combined_path, index=False)
    pd.DataFrame(diagnostics).to_csv(
        OUTPUT_DIR / f"{MODEL_NAME}_outerfold_fit_diagnostics.csv",
        index=False,
    )

    definition = {
        "purpose": "fold-aligned behavioral target export for EEG MVPA",
        "model": MODEL_NAME,
        "formula": MODEL_DEFINITIONS[MODEL_NAME]["formula"],
        "primary_target": "subjective_value_normalized_mean",
        "outer_fold_rule": (
            "for EEG outer block k, all behavioral labels used by that EEG fit are "
            "computed from one behavioral model fit excluding block k"
        ),
        "primary_neural_control": "linear pain + linear money main effects",
        "rt_rule": (
            "RT is not removed from SV; RT-related variance is removed from EEG only "
            "in the secondary sensitivity analysis"
        ),
        "combined_target_file": str(combined_path),
    }
    (OUTPUT_DIR / "analysis_definition.json").write_text(json.dumps(definition, indent=2))

    print(f"\nWrote aligned EEG targets: {combined_path}", flush=True)


if __name__ == "__main__":
    main()
