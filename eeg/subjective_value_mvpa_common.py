#!/usr/bin/env python3
"""Shared helpers for participant-specific subjective-value EEG decoding.

The core design is an outer leave-one-decision-block-out analysis.  For each
outer block, behavioral SV labels for *all* trials come from one behavioral fit
that excluded that same outer block.  EEG nuisance models and the Ridge decoder
are then fit on the four EEG training blocks and evaluated only on the held-out
EEG block.

Primary conditional estimand
-----------------------------
Both SV and EEG are residualised against linear Pain + Money main effects using
training-fold coefficients only.  The RT sensitivity keeps the same conditional
SV target but additionally residualises EEG against log(RT), log(RT)^2 and a
response-overlap indicator.  RT is never removed from the SV target.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PAIN_DESIGN_CENTER = 3.0
MONEY_DESIGN_CENTER = 3.0
RIDGE_ALPHA = 1.0


def target_file(base_path: Path, model_name: str = "nonlinear_interactive") -> Path:
    return (
        Path(base_path)
        / "derivatives"
        / "sv_modeling_v5"
        / "block_cv"
        / "eeg_outerfold_targets"
        / f"{model_name}_outerfold_alltrial_subjective_values.csv"
    )


def load_outerfold_targets(
    base_path: Path,
    model_name: str = "nonlinear_interactive",
) -> pd.DataFrame:
    path = target_file(base_path, model_name)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing aligned behavioral EEG target file: {path}\n"
            "Run behav/08c_export_selected_sv_outerfold_targets.py first."
        )

    table = pd.read_csv(path)
    required = {
        "participant",
        "trialsnum",
        "block",
        "painlevel",
        "moneylevel",
        "outer_heldout_block",
        "subjective_value_normalized_mean",
        "subjective_value_normalized_sd",
    }
    missing = sorted(required.difference(table.columns))
    if missing:
        raise KeyError(f"SV outer-fold target file missing columns: {missing}")

    table["participant"] = table["participant"].astype(str)
    for column in ["trialsnum", "block", "outer_heldout_block"]:
        table[column] = pd.to_numeric(table[column], errors="raise").astype(int)
    for column in [
        "painlevel",
        "moneylevel",
        "subjective_value_normalized_mean",
        "subjective_value_normalized_sd",
    ]:
        table[column] = pd.to_numeric(table[column], errors="raise").astype(float)

    if table.duplicated(
        subset=["outer_heldout_block", "participant", "trialsnum"]
    ).any():
        raise RuntimeError("Duplicate outer-fold SV participant/trial rows detected.")

    return table


def participant_list_from_targets(targets: pd.DataFrame) -> list[str]:
    return sorted(targets["participant"].dropna().astype(str).unique().tolist())


def _numeric_metadata(metadata: pd.DataFrame) -> pd.DataFrame:
    required = ["trialsnum", "block", "painlevel", "moneylevel", "rt"]
    missing = [column for column in required if column not in metadata.columns]
    if missing:
        raise KeyError(f"Decision EEG metadata missing columns: {missing}")

    out = metadata.copy()
    for column in ["trialsnum", "block"]:
        out[column] = pd.to_numeric(out[column], errors="raise").astype(int)
    for column in ["painlevel", "moneylevel", "rt"]:
        out[column] = pd.to_numeric(out[column], errors="raise").astype(float)
    return out


def align_participant_targets(
    participant: str,
    metadata: pd.DataFrame,
    targets: pd.DataFrame,
    target_column: str = "subjective_value_normalized_mean",
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """Return all-trial target arrays for every aligned outer behavioral fold."""

    metadata = _numeric_metadata(metadata).reset_index(drop=True)
    subject_targets = targets.loc[targets["participant"] == participant].copy()
    if subject_targets.empty:
        raise RuntimeError(f"{participant}: no behavioral SV target rows.")

    outer_blocks = sorted(subject_targets["outer_heldout_block"].unique().tolist())
    eeg_blocks = sorted(metadata["block"].unique().tolist())
    if outer_blocks != eeg_blocks:
        raise RuntimeError(
            f"{participant}: outer behavioral blocks {outer_blocks} != EEG blocks {eeg_blocks}."
        )

    values_by_fold: Dict[int, np.ndarray] = {}
    sd_by_fold: Dict[int, np.ndarray] = {}

    left = metadata[["trialsnum", "block", "painlevel", "moneylevel"]].copy()
    left = left.rename(
        columns={
            "block": "eeg_block",
            "painlevel": "eeg_painlevel",
            "moneylevel": "eeg_moneylevel",
        }
    )

    for outer_block in outer_blocks:
        fold = subject_targets.loc[
            subject_targets["outer_heldout_block"] == outer_block,
            [
                "trialsnum",
                "block",
                "painlevel",
                "moneylevel",
                target_column,
                "subjective_value_normalized_sd",
            ],
        ].copy()

        merged = left.merge(
            fold,
            on="trialsnum",
            how="left",
            validate="one_to_one",
        )

        if merged[target_column].isna().any():
            missing_trials = merged.loc[merged[target_column].isna(), "trialsnum"].tolist()
            raise RuntimeError(
                f"{participant}: missing outer-fold {outer_block} SV for EEG trials {missing_trials}."
            )

        if not np.array_equal(
            merged["eeg_block"].to_numpy(dtype=int),
            merged["block"].to_numpy(dtype=int),
        ):
            raise RuntimeError(f"{participant}: block mismatch in outer fold {outer_block}.")

        if not np.allclose(
            merged["eeg_painlevel"].to_numpy(dtype=float),
            merged["painlevel"].to_numpy(dtype=float),
        ):
            raise RuntimeError(f"{participant}: pain mismatch in outer fold {outer_block}.")

        if not np.allclose(
            merged["eeg_moneylevel"].to_numpy(dtype=float),
            merged["moneylevel"].to_numpy(dtype=float),
        ):
            raise RuntimeError(f"{participant}: money mismatch in outer fold {outer_block}.")

        values_by_fold[int(outer_block)] = merged[target_column].to_numpy(dtype=float)
        sd_by_fold[int(outer_block)] = merged["subjective_value_normalized_sd"].to_numpy(dtype=float)

    return values_by_fold, sd_by_fold


def centered_main_effects(
    pain_levels: np.ndarray,
    money_levels: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    pain = np.asarray(pain_levels, dtype=float) - PAIN_DESIGN_CENTER
    money = np.asarray(money_levels, dtype=float) - MONEY_DESIGN_CENTER
    return pain, money


def fit_nuisance_residualizer(
    nuisance_train_raw: np.ndarray,
    values_train: np.ndarray,
) -> dict:
    nuisance = np.asarray(nuisance_train_raw, dtype=float)
    values = np.asarray(values_train, dtype=float)

    if nuisance.ndim == 1:
        nuisance = nuisance[:, None]

    mean = nuisance.mean(axis=0)
    sd = nuisance.std(axis=0, ddof=0)
    varying = sd > 1e-12

    standardized = np.zeros_like(nuisance, dtype=float)
    if varying.any():
        standardized[:, varying] = (
            nuisance[:, varying] - mean[varying]
        ) / sd[varying]

    design = np.column_stack([np.ones(len(nuisance)), standardized[:, varying]])
    coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)

    return {
        "mean": mean,
        "sd": sd,
        "varying": varying,
        "coefficients": coefficients,
    }


def apply_nuisance_residualizer(
    nuisance_raw: np.ndarray,
    values: np.ndarray,
    fitted: dict,
) -> np.ndarray:
    nuisance = np.asarray(nuisance_raw, dtype=float)
    values = np.asarray(values, dtype=float)
    if nuisance.ndim == 1:
        nuisance = nuisance[:, None]

    varying = fitted["varying"]
    standardized = np.zeros_like(nuisance, dtype=float)
    if varying.any():
        standardized[:, varying] = (
            nuisance[:, varying] - fitted["mean"][varying]
        ) / fitted["sd"][varying]

    design = np.column_stack([np.ones(len(nuisance)), standardized[:, varying]])
    return values - design @ fitted["coefficients"]


def main_effect_nuisance(
    pain_centered: np.ndarray,
    money_centered: np.ndarray,
) -> np.ndarray:
    return np.column_stack([pain_centered, money_centered])


def feature_nuisance(
    pain_centered: np.ndarray,
    money_centered: np.ndarray,
    reaction_time: np.ndarray,
    response_overlap: np.ndarray,
    include_rt_controls: bool,
) -> np.ndarray:
    columns = [
        np.asarray(pain_centered, dtype=float),
        np.asarray(money_centered, dtype=float),
    ]
    if include_rt_controls:
        log_rt = np.log(np.asarray(reaction_time, dtype=float))
        columns.extend(
            [
                log_rt,
                log_rt ** 2,
                np.asarray(response_overlap, dtype=float),
            ]
        )
    return np.column_stack(columns)


def make_ridge_decoder():
    return make_pipeline(StandardScaler(), Ridge(alpha=RIDGE_ALPHA))


def pearson_r(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if np.std(y_true) <= 1e-12 or np.std(y_pred) <= 1e-12:
        return np.nan
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def decode_aligned_outerfold(
    features: np.ndarray,
    pain_centered: np.ndarray,
    money_centered: np.ndarray,
    block_ids: np.ndarray,
    reaction_time: np.ndarray,
    response_overlap: np.ndarray,
    sv_by_outer_fold: Dict[int, np.ndarray],
    conditional_main_effects: bool,
    include_rt_controls: bool,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Decode SV using behavioral and EEG folds aligned on the same held-out block."""

    if include_rt_controls and not conditional_main_effects:
        raise ValueError("RT sensitivity is defined only for the conditional SV analysis.")

    features = np.asarray(features, dtype=float)
    block_ids = np.asarray(block_ids, dtype=int)
    predictions = np.full(len(block_ids), np.nan, dtype=float)
    heldout_target = np.full(len(block_ids), np.nan, dtype=float)

    pain_money_nuisance = main_effect_nuisance(pain_centered, money_centered)
    all_feature_nuisance = feature_nuisance(
        pain_centered,
        money_centered,
        reaction_time,
        response_overlap,
        include_rt_controls=include_rt_controls,
    )

    for outer_block in sorted(np.unique(block_ids).tolist()):
        if int(outer_block) not in sv_by_outer_fold:
            raise RuntimeError(f"Missing behavioral target set for outer block {outer_block}.")

        train = block_ids != outer_block
        test = block_ids == outer_block
        sv_all = np.asarray(sv_by_outer_fold[int(outer_block)], dtype=float)
        if len(sv_all) != len(block_ids):
            raise RuntimeError("Outer-fold SV vector length does not match EEG trial count.")

        if conditional_main_effects:
            target_fit = fit_nuisance_residualizer(
                pain_money_nuisance[train], sv_all[train]
            )
            y_train = apply_nuisance_residualizer(
                pain_money_nuisance[train], sv_all[train], target_fit
            )
            y_test = apply_nuisance_residualizer(
                pain_money_nuisance[test], sv_all[test], target_fit
            )

            feature_fit = fit_nuisance_residualizer(
                all_feature_nuisance[train], features[train]
            )
            x_train = apply_nuisance_residualizer(
                all_feature_nuisance[train], features[train], feature_fit
            )
            x_test = apply_nuisance_residualizer(
                all_feature_nuisance[test], features[test], feature_fit
            )
        else:
            y_train = sv_all[train]
            y_test = sv_all[test]
            x_train = features[train]
            x_test = features[test]

        decoder = make_ridge_decoder()
        decoder.fit(x_train, y_train)
        predictions[test] = decoder.predict(x_test)
        heldout_target[test] = y_test

    if not np.isfinite(predictions).all() or not np.isfinite(heldout_target).all():
        raise RuntimeError("Non-finite outer-fold SV predictions/targets encountered.")

    return pearson_r(heldout_target, predictions), predictions, heldout_target
