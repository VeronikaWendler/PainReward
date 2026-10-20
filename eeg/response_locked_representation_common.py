#!/usr/bin/env python3
"""Shared helpers for PainReward response-locked representation analyses."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from mne.stats import combine_adjacency, permutation_cluster_1samp_test
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BASE_PATH = Path(os.getenv("basepath", Path(__file__).parent.parent.parent))
PARTICIPANTS_TSV = BASE_PATH / "participants.tsv"

AUDIT_OUTPUT_ROOT = BASE_PATH / "derivatives" / "statistics" / "tfr_response_representation_design_audit"
MVPA_OUTPUT_ROOT = BASE_PATH / "derivatives" / "statistics" / "tfr_mvpa_decision_response_full"
RSA_OUTPUT_ROOT = BASE_PATH / "derivatives" / "statistics" / "tfr_rsa_decision_response"

ATTRIBUTES = ["pain", "money"]
CANONICAL_BANDS: Dict[str, Tuple[float, float]] = {
    "theta": (4.0, 7.0),
    "alpha": (8.0, 13.0),
    "beta": (14.0, 30.0),
    "low_gamma": (31.0, 45.0),
}

RIDGE_ALPHA = 1.0
RANDOM_STATE = 23
CLUSTER_FORMING_ALPHA = 0.05
CANDIDATE_TMIN_S = -0.800
CANDIDATE_TMAX_S = 0.000

# Behavior-only support rules, frozen before any new response-locked EEG results.
PRIMARY_MIN_TRIALS = 30
PRIMARY_MIN_UNIQUE_LEVELS = 4
PRIMARY_MIN_BLOCKS = 4
PRIMARY_MIN_GROUP_PARTICIPANTS = 48

MOTOR_MIN_TRIALS = 20
MOTOR_MIN_UNIQUE_LEVELS = 3
MOTOR_MIN_BLOCKS = 3
MOTOR_MIN_GROUP_PARTICIPANTS = 30


def response_tfr_paths(participant: str) -> Dict[str, Path]:
    directory = BASE_PATH / "derivatives" / participant / "eeg" / "tfr_decision_response"
    stem = f"{participant}_decision_response"
    return {
        "power": directory / f"{stem}_power.npy",
        "times": directory / f"{stem}_times.npy",
        "freqs": directory / f"{stem}_freqs.npy",
        "metadata": directory / f"{stem}_metadata.csv",
        "params": directory / f"{stem}_params.json",
    }


def load_participant_list() -> List[str]:
    if PARTICIPANTS_TSV.exists():
        table = pd.read_csv(PARTICIPANTS_TSV, sep="\t")
        if "participant_id" not in table.columns:
            raise KeyError(f"{PARTICIPANTS_TSV} has no participant_id column")
        candidates = sorted(table["participant_id"].astype(str).tolist())
    else:
        candidates = sorted(
            p.name for p in BASE_PATH.iterdir()
            if p.is_dir() and p.name.startswith("sub-")
        )

    participants = []
    for participant in candidates:
        paths = response_tfr_paths(participant)
        if all(paths[key].exists() for key in ["power", "times", "freqs", "metadata"]):
            participants.append(participant)
    if not participants:
        raise RuntimeError("No complete response-locked TFR datasets found")
    return participants


def decision_behavior_path(participant: str) -> Path:
    return (
        BASE_PATH
        / participant
        / "eeg"
        / f"{participant}_task-decision_beh.tsv"
    )


def load_response_metadata(participant: str) -> pd.DataFrame:
    """
    Load response-TFR metadata and add the original physical response key.

    Important:
    the TFR power array and metadata must remain row-aligned. Therefore the
    single anomalous `resm` row is not physically dropped here. Instead we
    create `valid_response_analysis`, and all response-locked trial masks
    exclude that row.

    Behavior-only diagnostics established:
        res+ = accepted outcome
        res- = rejected outcome
        resm = one anomalous near-deadline trial

    They also showed that acceptance does NOT fix physical motor key:
    accepted trials can be made with either `m` or `n`.
    """
    metadata = pd.read_csv(
        response_tfr_paths(participant)["metadata"]
    )

    required = [
        "trialsnum",
        "painlevel",
        "moneylevel",
        "block",
        "behavioral_rt_s",
        "event_rt_s",
        "response_event_type",
        "cue_time_relative_to_response_s",
        "accepted",
    ]

    missing = [
        column
        for column in required
        if column not in metadata.columns
    ]

    if missing:
        raise KeyError(
            f"{participant}: response metadata missing {missing}"
        )

    behavior_path = decision_behavior_path(
        participant
    )

    if not behavior_path.exists():
        raise FileNotFoundError(
            behavior_path
        )

    behavior = pd.read_csv(
        behavior_path,
        sep="\t",
    )

    behavior = (
        behavior[
            ~behavior["fixcross.started"].isna()
        ]
        .copy()
        .reset_index(drop=True)
    )

    behavior["trialsnum"] = np.arange(
        1,
        len(behavior) + 1,
        dtype=int,
    )

    behavior_required = [
        "trialsnum",
        "choice_resp.keys",
        "acceptkey",
        "acceptside",
    ]

    behavior_missing = [
        column
        for column in behavior_required
        if column not in behavior.columns
    ]

    if behavior_missing:
        raise KeyError(
            f"{participant}: behavior file missing {behavior_missing}"
        )

    metadata = metadata.merge(
        behavior[behavior_required],
        on="trialsnum",
        how="left",
        validate="one_to_one",
    )

    metadata = metadata.rename(
        columns={
            "choice_resp.keys": "physical_response_key",
        }
    )

    numeric = [
        "trialsnum",
        "painlevel",
        "moneylevel",
        "block",
        "behavioral_rt_s",
        "event_rt_s",
        "cue_time_relative_to_response_s",
        "accepted",
    ]

    for column in numeric:
        metadata[column] = pd.to_numeric(
            metadata[column],
            errors="coerce",
        )

    if not np.isfinite(
        metadata[numeric].to_numpy(float)
    ).all():
        raise RuntimeError(
            f"{participant}: non-finite required metadata"
        )

    metadata["accepted"] = metadata["accepted"].astype(int)
    metadata["block"] = metadata["block"].astype(int)
    metadata["response_event_type"] = (
        metadata["response_event_type"].astype(str)
    )
    metadata["physical_response_key"] = (
        metadata["physical_response_key"].astype(str)
    )

    # Genuine button-response trials for the response-locked analysis.
    metadata["valid_response_analysis"] = (
        metadata["response_event_type"].isin(["res+", "res-"])
    )

    valid_rows = metadata["valid_response_analysis"]

    if not metadata.loc[
        valid_rows,
        "physical_response_key",
    ].isin(["m", "n"]).all():
        bad = metadata.loc[
            valid_rows
            & ~metadata["physical_response_key"].isin(["m", "n"]),
            [
                "trialsnum",
                "response_event_type",
                "physical_response_key",
            ],
        ]
        raise RuntimeError(
            f"{participant}: unexpected physical keys on valid responses:\n{bad}"
        )

    event_implied_acceptance = (
        metadata.loc[
            valid_rows,
            "response_event_type",
        ]
        == "res+"
    ).astype(int)

    if not np.array_equal(
        event_implied_acceptance.to_numpy(int),
        metadata.loc[
            valid_rows,
            "accepted",
        ].to_numpy(int),
    ):
        raise RuntimeError(
            f"{participant}: res+/res- disagrees with accepted flag"
        )

    timing_error = np.max(
        np.abs(
            metadata[
                "cue_time_relative_to_response_s"
            ].to_numpy(float)
            + metadata[
                "event_rt_s"
            ].to_numpy(float)
        )
    )

    if timing_error > 1e-9:
        raise RuntimeError(
            f"{participant}: cue/response timing identity failed: "
            f"{timing_error:.3e}"
        )

    return metadata


def load_response_axes(participant: str) -> Tuple[np.ndarray, np.ndarray]:
    paths = response_tfr_paths(participant)
    return np.load(paths["times"]).astype(float), np.load(paths["freqs"]).astype(float)


def candidate_pre_response_mask(times: np.ndarray) -> np.ndarray:
    return (times >= CANDIDATE_TMIN_S - 1e-12) & (times <= CANDIDATE_TMAX_S + 1e-12)


def cue_has_occurred_mask(
    metadata: pd.DataFrame,
    response_relative_time_s: float,
) -> np.ndarray:
    """
    Current-trial availability plus valid-response filtering.

    A trial contributes at response-relative time tau only if:
        event_RT + tau >= 0

    The single anomalous `resm` trial is excluded through
    `valid_response_analysis`.
    """
    mask = (
        metadata["event_rt_s"].to_numpy(float)
        + float(response_relative_time_s)
        >= -1e-12
    )

    if "valid_response_analysis" in metadata.columns:
        mask = (
            mask
            & metadata[
                "valid_response_analysis"
            ].to_numpy(bool)
        )

    return mask


def trial_support_summary(
    metadata: pd.DataFrame,
    response_relative_time_s: float,
    additional_mask: np.ndarray | None = None,
) -> Dict[str, int]:
    mask = cue_has_occurred_mask(metadata, response_relative_time_s)
    if additional_mask is not None:
        additional_mask = np.asarray(additional_mask, dtype=bool)
        if len(additional_mask) != len(metadata):
            raise ValueError("additional_mask length differs from metadata")
        mask &= additional_mask
    selected = metadata.loc[mask]
    return {
        "n_trials": int(len(selected)),
        "n_pain_levels": int(selected["painlevel"].nunique()),
        "n_money_levels": int(selected["moneylevel"].nunique()),
        "n_blocks": int(selected["block"].nunique()),
    }


def support_is_valid(summary: Dict[str, int], minimum_trials: int, minimum_unique_levels: int, minimum_blocks: int) -> bool:
    return bool(
        summary["n_trials"] >= minimum_trials
        and summary["n_pain_levels"] >= minimum_unique_levels
        and summary["n_money_levels"] >= minimum_unique_levels
        and summary["n_blocks"] >= minimum_blocks
    )


def make_rt_nuisance_matrix(reaction_time: np.ndarray) -> np.ndarray:
    reaction_time = np.asarray(reaction_time, float)
    if np.any(reaction_time <= 0):
        raise RuntimeError("RT must be positive")
    log_rt = np.log(reaction_time)
    return np.column_stack([log_rt, log_rt ** 2])


def residualize_features_using_training_fold(
    train_features: np.ndarray,
    test_features: np.ndarray,
    train_nuisance_raw: np.ndarray,
    test_nuisance_raw: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    mean = train_nuisance_raw.mean(axis=0)
    sd = train_nuisance_raw.std(axis=0)
    varying = sd > 1e-12
    if not np.any(varying):
        return train_features, test_features
    train_nuisance = (train_nuisance_raw[:, varying] - mean[varying]) / sd[varying]
    test_nuisance = (test_nuisance_raw[:, varying] - mean[varying]) / sd[varying]
    train_design = np.column_stack([np.ones(len(train_nuisance)), train_nuisance])
    test_design = np.column_stack([np.ones(len(test_nuisance)), test_nuisance])
    coefficients, *_ = np.linalg.lstsq(train_design, train_features, rcond=None)
    return (
        train_features - train_design @ coefficients,
        test_features - test_design @ coefficients,
    )


def leave_one_block_out_score(
    features: np.ndarray,
    target: np.ndarray,
    block_ids: np.ndarray,
    reaction_time: np.ndarray,
    remove_rt_related_variance: bool,
) -> float:
    features = np.asarray(features, float)
    target = np.asarray(target, float)
    block_ids = np.asarray(block_ids, int)
    reaction_time = np.asarray(reaction_time, float)
    unique_blocks = np.unique(block_ids)
    if len(unique_blocks) < 2:
        return np.nan
    predictions = np.full(len(target), np.nan)
    cv = GroupKFold(n_splits=len(unique_blocks))
    for train_index, test_index in cv.split(features, target, groups=block_ids):
        train_features = np.asarray(features[train_index], float)
        test_features = np.asarray(features[test_index], float)
        if remove_rt_related_variance:
            train_features, test_features = residualize_features_using_training_fold(
                train_features,
                test_features,
                make_rt_nuisance_matrix(reaction_time[train_index]),
                make_rt_nuisance_matrix(reaction_time[test_index]),
            )
        decoder = make_pipeline(StandardScaler(), Ridge(alpha=RIDGE_ALPHA))
        decoder.fit(train_features, target[train_index])
        predictions[test_index] = decoder.predict(test_features)
    if not np.isfinite(predictions).all() or np.std(predictions) <= 1e-12 or np.std(target) <= 1e-12:
        return np.nan
    return float(np.corrcoef(target, predictions)[0, 1])


def fisher_z(values: np.ndarray) -> np.ndarray:
    return np.arctanh(np.clip(np.asarray(values, float), -0.999999, 0.999999))


def holm_correct(p_values: Sequence[float]) -> np.ndarray:
    p_values = np.asarray(p_values, float)
    n = len(p_values)
    order = np.argsort(p_values)
    ranked = p_values[order]
    adjusted_ranked = np.maximum.accumulate((n - np.arange(n)) * ranked)
    adjusted_ranked = np.clip(adjusted_ranked, 0, 1)
    adjusted = np.empty(n)
    adjusted[order] = adjusted_ranked
    return adjusted


def _one_sided_t_threshold(n_participants: int) -> float:
    return float(stats.t.ppf(1.0 - CLUSTER_FORMING_ALPHA, df=n_participants - 1))


def frequency_time_cluster_test(
    participant_values: np.ndarray,
    frequencies: np.ndarray,
    times: np.ndarray,
    n_permutations: int,
    n_jobs: int,
    fisher_transform_correlations: bool,
):
    values = np.asarray(participant_values, float)
    if fisher_transform_correlations:
        values = fisher_z(values)
    if not np.isfinite(values).all():
        raise RuntimeError("Non-finite values entered frequency x time group inference")
    n_participants, n_frequencies, n_times = values.shape
    t_values, clusters, p_values, _ = permutation_cluster_1samp_test(
        values.reshape(n_participants, n_frequencies * n_times),
        threshold=_one_sided_t_threshold(n_participants),
        n_permutations=n_permutations,
        tail=1,
        adjacency=combine_adjacency(n_frequencies, n_times),
        seed=RANDOM_STATE,
        n_jobs=n_jobs,
        out_type="indices",
        verbose=False,
    )
    return (
        t_values.reshape(n_frequencies, n_times),
        [np.asarray(cluster[0], int) for cluster in clusters],
        np.asarray(p_values, float),
    )


def temporal_cluster_test(
    participant_values: np.ndarray,
    times: np.ndarray,
    n_permutations: int,
    n_jobs: int,
    fisher_transform_correlations: bool,
):
    values = np.asarray(participant_values, float)
    if fisher_transform_correlations:
        values = fisher_z(values)
    if not np.isfinite(values).all():
        raise RuntimeError("Non-finite values entered temporal group inference")
    n_participants = values.shape[0]
    t_values, clusters, p_values, _ = permutation_cluster_1samp_test(
        values,
        threshold=_one_sided_t_threshold(n_participants),
        n_permutations=n_permutations,
        tail=1,
        seed=RANDOM_STATE,
        n_jobs=n_jobs,
        out_type="indices",
        verbose=False,
    )
    return t_values, [np.asarray(cluster[0], int) for cluster in clusters], np.asarray(p_values, float)


def summarize_frequency_time_family(
    group_values: np.ndarray,
    family_name: str,
    frequencies: np.ndarray,
    times: np.ndarray,
    n_permutations: int,
    n_jobs: int,
    fisher_transform_correlations: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    stored = {}
    omnibus_ps = []
    for attribute_index, attribute_name in enumerate(ATTRIBUTES):
        result = frequency_time_cluster_test(
            group_values[:, attribute_index], frequencies, times,
            n_permutations, n_jobs, fisher_transform_correlations,
        )
        p_values = result[2]
        omnibus_ps.append(float(p_values.min()) if len(p_values) else 1.0)
        stored[attribute_name] = result
    adjusted = holm_correct(omnibus_ps)
    omnibus_rows, cluster_rows = [], []
    for attribute_index, attribute_name in enumerate(ATTRIBUTES):
        adjusted_p = float(adjusted[attribute_index])
        omnibus_rows.append({
            "family": family_name,
            "attribute": attribute_name,
            "omnibus_p_within_surface": omnibus_ps[attribute_index],
            "omnibus_p_holm_across_pain_money": adjusted_p,
            "domain_significant_holm_0p05": adjusted_p < 0.05,
            "n_participants": int(group_values.shape[0]),
            "formal_tmin_s": float(times.min()),
            "formal_tmax_s": float(times.max()),
        })
        if adjusted_p >= 0.05:
            continue
        t_values, clusters, p_values = stored[attribute_name]
        n_frequencies, n_times = t_values.shape
        for cluster_id, (flat_indices, cluster_p) in enumerate(zip(clusters, p_values)):
            if cluster_p >= 0.05:
                continue
            frequency_indices, time_indices = np.unravel_index(flat_indices, (n_frequencies, n_times))
            cluster_t = t_values.reshape(-1)[flat_indices]
            peak_position = int(np.argmax(cluster_t))
            peak_frequency_index, peak_time_index = np.unravel_index(
                int(flat_indices[peak_position]), (n_frequencies, n_times)
            )
            cluster_rows.append({
                "family": family_name,
                "attribute": attribute_name,
                "cluster_id": cluster_id,
                "cluster_p_within_surface": float(cluster_p),
                "freq_min_hz": float(frequencies[frequency_indices].min()),
                "freq_max_hz": float(frequencies[frequency_indices].max()),
                "time_start_s": float(times[time_indices].min()),
                "time_end_s": float(times[time_indices].max()),
                "peak_freq_hz": float(frequencies[peak_frequency_index]),
                "peak_time_s": float(times[peak_time_index]),
                "peak_t": float(cluster_t[peak_position]),
                "omnibus_p_holm_across_pain_money": adjusted_p,
            })
    return pd.DataFrame(omnibus_rows), pd.DataFrame(cluster_rows)


def summarize_band_family(
    group_values: np.ndarray,
    family_name: str,
    times: np.ndarray,
    band_names: Sequence[str],
    n_permutations: int,
    n_jobs: int,
    fisher_transform_correlations: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    omnibus_rows, cluster_rows = [], []
    for attribute_index, attribute_name in enumerate(ATTRIBUTES):
        stored, raw_ps = [], []
        for band_index, band_name in enumerate(band_names):
            result = temporal_cluster_test(
                group_values[:, attribute_index, band_index], times,
                n_permutations, n_jobs, fisher_transform_correlations,
            )
            p_values = result[2]
            raw_ps.append(float(p_values.min()) if len(p_values) else 1.0)
            stored.append((band_name, *result))
        adjusted = holm_correct(raw_ps)
        for band_index, result in enumerate(stored):
            band_name, t_values, clusters, p_values = result
            adjusted_p = float(adjusted[band_index])
            omnibus_rows.append({
                "family": family_name,
                "attribute": attribute_name,
                "band": band_name,
                "band_omnibus_p_within_time": raw_ps[band_index],
                "band_omnibus_p_holm_within_attribute": adjusted_p,
                "band_significant_holm_0p05": adjusted_p < 0.05,
                "n_participants": int(group_values.shape[0]),
                "formal_tmin_s": float(times.min()),
                "formal_tmax_s": float(times.max()),
            })
            if adjusted_p >= 0.05:
                continue
            for cluster_id, (indices, cluster_p) in enumerate(zip(clusters, p_values)):
                if cluster_p >= 0.05:
                    continue
                cluster_t = t_values[indices]
                peak_position = int(np.argmax(cluster_t))
                cluster_rows.append({
                    "family": family_name,
                    "attribute": attribute_name,
                    "band": band_name,
                    "cluster_id": cluster_id,
                    "cluster_p_within_band_time": float(cluster_p),
                    "time_start_s": float(times[indices].min()),
                    "time_end_s": float(times[indices].max()),
                    "peak_time_s": float(times[indices[peak_position]]),
                    "peak_t": float(cluster_t[peak_position]),
                    "band_omnibus_p_holm_within_attribute": adjusted_p,
                })
    return pd.DataFrame(omnibus_rows), pd.DataFrame(cluster_rows)


def load_design_plan() -> Dict:
    path = AUDIT_OUTPUT_ROOT / "response_locked_design_plan.json"
    if not path.exists():
        raise FileNotFoundError(f"Design plan not found: {path}. Run 05q first.")
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)
