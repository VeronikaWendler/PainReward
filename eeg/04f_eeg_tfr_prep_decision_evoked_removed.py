#!/usr/bin/env python3
"""
04f_eeg_tfr_prep_decision_evoked_removed.py
===========================================

Create cross-fitted EVOKED-REMOVED / INDUCED-ENRICHED decision-phase TFRs.

Scientific purpose
------------------
The total-power TFR contains both phase-locked evoked activity and
non-phase-locked/induced activity. To ask whether a decision-phase spectral
representation depends on the generic cue-evoked response, this script removes
a cross-fitted generic decision-cue waveform before recomputing the TFR.

For each participant and each held-out decision block b:

    template_b =
        mean EEG waveform from ALL good decision trials in blocks != b

    residual trial in block b =
        original trial - template_b

Important safeguards
--------------------
- The held-out block never contributes to its own template.
- Pain level is NOT used.
- Money level is NOT used.
- Choice is NOT used.
- RT is NOT used.
- The exact same good trials as the completed total-power decision TFR are used.
- TFR parameters are copied from the completed total-power decision TFR.
- Output time/frequency/channel grids are checked against the total-power TFR.

This removes the generic phase-locked decision-cue response without target-label
leakage. It does NOT guarantee removal of every level-specific phase-locked
component, so the result is called "evoked-removed / induced-enriched", NOT
"pure induced activity".

Inputs
------
Cleaned decision EEG:
    SOURCE_BIDS/derivatives/<sub>/eeg/
        <sub>_decision_cleaned-raw.fif

Completed total-power decision TFR:
    derivatives/<sub>/eeg/tfr_decision/
        <sub>_decision_cue_metadata.csv
        <sub>_decision_cue_times.npy
        <sub>_decision_cue_freqs.npy
        <sub>_decision_cue_channels.json
        <sub>_decision_cue_params.json

Outputs
-------
derivatives/<sub>/eeg/tfr_decision_evoked_removed/
    <sub>_decision_cue_power_evoked_removed.npy
    <sub>_decision_cue_times.npy
    <sub>_decision_cue_freqs.npy
    <sub>_decision_cue_metadata.csv
    <sub>_decision_cue_channels.json
    <sub>_decision_cue_params.json
    <sub>_decision_cue_prep_summary.json
    <sub>_decision_cue_evoked_removal_diagnostics.csv
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import mne
import numpy as np
import pandas as pd
from mne.time_frequency import tfr_array_morlet


# =============================================================================
# Project paths and constants
# =============================================================================

mne.set_log_level("ERROR")

BASE_PATH = Path(
    os.getenv("basepath", Path(__file__).parent.parent.parent)
)

SOURCE_BIDS = Path(
    os.getenv("SOURCE_BIDS", BASE_PATH)
)

CHANNELS_TO_DROP = [
    "HEOGL",
    "HEOGR",
    "VEOGL",
    "STI 014",
    "Status",
]

LOG_FLOOR = np.finfo(
    np.float64
).tiny


# =============================================================================
# Command-line arguments
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare cross-fitted evoked-removed decision-phase TFR power."
        )
    )

    parser.add_argument(
        "--subject",
        required=True,
        help="Participant ID, e.g. sub-001.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite a complete existing output set.",
    )

    parser.add_argument(
        "--n-jobs",
        type=int,
        default=max(
            1,
            int(
                os.getenv(
                    "SLURM_CPUS_PER_TASK",
                    "1",
                )
            ),
        ),
        help="Parallel jobs passed to the MNE Morlet computation.",
    )

    return parser.parse_args()


# =============================================================================
# File locations
# =============================================================================

def decision_raw_path(
    participant: str,
) -> Path:
    """Cleaned broadband decision EEG."""

    return (
        SOURCE_BIDS
        / "derivatives"
        / participant
        / "eeg"
        / f"{participant}_decision_cleaned-raw.fif"
    )


def total_tfr_paths(
    participant: str,
) -> Dict[str, Path]:
    """Files from the already-completed total-power decision TFR."""

    tfr_dir = (
        BASE_PATH
        / "derivatives"
        / participant
        / "eeg"
        / "tfr_decision"
    )

    stem = f"{participant}_decision_cue"

    return {
        "power":
            tfr_dir
            / f"{stem}_power.npy",
        "times":
            tfr_dir
            / f"{stem}_times.npy",
        "freqs":
            tfr_dir
            / f"{stem}_freqs.npy",
        "metadata":
            tfr_dir
            / f"{stem}_metadata.csv",
        "channels":
            tfr_dir
            / f"{stem}_channels.json",
        "params":
            tfr_dir
            / f"{stem}_params.json",
    }


def output_paths(
    participant: str,
) -> Dict[str, Path]:
    """Output files for the evoked-removed decision TFR."""

    output_dir = (
        BASE_PATH
        / "derivatives"
        / participant
        / "eeg"
        / "tfr_decision_evoked_removed"
    )

    stem = f"{participant}_decision_cue"

    return {
        "directory":
            output_dir,
        "power":
            output_dir
            / f"{stem}_power_evoked_removed.npy",
        "times":
            output_dir
            / f"{stem}_times.npy",
        "freqs":
            output_dir
            / f"{stem}_freqs.npy",
        "metadata":
            output_dir
            / f"{stem}_metadata.csv",
        "channels":
            output_dir
            / f"{stem}_channels.json",
        "params":
            output_dir
            / f"{stem}_params.json",
        "summary":
            output_dir
            / f"{stem}_prep_summary.json",
        "diagnostics":
            output_dir
            / f"{stem}_evoked_removal_diagnostics.csv",
    }


# =============================================================================
# Utility functions
# =============================================================================

def atomic_save_npy(
    path: Path,
    array: np.ndarray,
) -> None:
    """Write a NumPy array atomically."""

    temporary_path = Path(
        str(path) + ".tmp"
    )

    with open(
        temporary_path,
        "wb",
    ) as file_handle:
        np.save(
            file_handle,
            array,
            allow_pickle=False,
        )

    os.replace(
        temporary_path,
        path,
    )


def make_time_bins(
    sample_times: np.ndarray,
    start_s: float,
    end_s: float,
    bin_width_s: float,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Reproduce the total-power decision-TFR time-bin construction."""

    edges = np.arange(
        start_s,
        end_s + bin_width_s * 0.5,
        bin_width_s,
    )

    if edges[-1] < end_s - 1e-9:
        edges = np.append(
            edges,
            end_s,
        )

    if abs(
        edges[-1] - end_s
    ) > 1e-9:
        edges = np.append(
            edges[
                edges < end_s
            ],
            end_s,
        )

    centers = []
    masks = []

    for bin_index in range(
        len(edges) - 1
    ):
        low = float(
            edges[
                bin_index
            ]
        )

        high = float(
            edges[
                bin_index + 1
            ]
        )

        if bin_index == len(edges) - 2:
            mask = np.where(
                (sample_times >= low - 1e-12)
                & (sample_times <= high + 1e-12)
            )[0]
        else:
            mask = np.where(
                (sample_times >= low - 1e-12)
                & (sample_times < high - 1e-12)
            )[0]

        if len(
            mask
        ) == 0:
            raise RuntimeError(
                f"No TFR samples inside {low:.3f}..{high:.3f} s."
            )

        centers.append(
            (low + high) / 2.0
        )

        masks.append(
            mask
        )

    return (
        np.asarray(
            centers,
            dtype=float,
        ),
        masks,
    )


# =============================================================================
# Load exact total-TFR metadata and parameters
# =============================================================================

def load_total_tfr_reference(
    participant: str,
):
    """
    Load the exact good-trial metadata and TFR parameter choices from the
    completed total-power decision analysis.
    """

    paths = total_tfr_paths(
        participant
    )

    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(
                f"{participant}: missing total-TFR {name}: {path}"
            )

    metadata = pd.read_csv(
        paths["metadata"]
    )

    required_columns = [
        "sample",
        "trialsnum",
        "painlevel",
        "moneylevel",
        "block",
        "rt",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in metadata.columns
    ]

    if missing_columns:
        raise KeyError(
            f"{participant}: total-TFR metadata missing {missing_columns}"
        )

    with open(
        paths["params"],
        "r",
        encoding="utf-8",
    ) as file_handle:
        params = json.load(
            file_handle
        )

    required_params = [
        "working_epoch_s",
        "epoch_resample_hz",
        "frequencies_hz",
        "tfr_decim",
        "baseline_s",
        "saved_time_range_s",
        "time_bin_s",
    ]

    missing_params = [
        key
        for key in required_params
        if key not in params
    ]

    if missing_params:
        raise KeyError(
            f"{participant}: total-TFR params missing {missing_params}"
        )

    reference_times = np.load(
        paths["times"]
    ).astype(
        float
    )

    reference_frequencies = np.load(
        paths["freqs"]
    ).astype(
        float
    )

    with open(
        paths["channels"],
        "r",
        encoding="utf-8",
    ) as file_handle:
        reference_channels = json.load(
            file_handle
        )

    return (
        metadata,
        params,
        reference_times,
        reference_frequencies,
        reference_channels,
    )


# =============================================================================
# Recreate long broadband epochs for the exact same good trials
# =============================================================================

def load_long_epochs(
    participant: str,
    metadata: pd.DataFrame,
    params: Dict,
) -> mne.Epochs:
    """Create the same long broadband epochs used for total-power TFR."""

    raw_path = decision_raw_path(
        participant
    )

    if not raw_path.exists():
        raise FileNotFoundError(
            raw_path
        )

    raw = mne.io.read_raw_fif(
        str(raw_path),
        preload=True,
        verbose="ERROR",
    )

    channels_to_drop = [
        channel
        for channel in CHANNELS_TO_DROP
        if channel in raw.ch_names
    ]

    if channels_to_drop:
        raw.drop_channels(
            channels_to_drop
        )

    raw.pick(
        "eeg",
        exclude=[],
    )

    event_samples = pd.to_numeric(
        metadata["sample"],
        errors="raise",
    ).to_numpy(
        dtype=int
    )

    events = np.column_stack(
        [
            event_samples,
            np.zeros(
                len(metadata),
                dtype=int,
            ),
            np.ones(
                len(metadata),
                dtype=int,
            ),
        ]
    )

    epoch_start_s, epoch_end_s = map(
        float,
        params[
            "working_epoch_s"
        ],
    )

    epochs = mne.Epochs(
        raw,
        events=events,
        event_id={"decision_cue": 1},
        tmin=epoch_start_s,
        tmax=epoch_end_s,
        baseline=None,
        metadata=metadata.copy(),
        preload=True,
        reject=None,
        reject_by_annotation=True,
        verbose="ERROR",
    )

    if len(
        epochs
    ) != len(
        metadata
    ):
        raise RuntimeError(
            f"{participant}: long epoching retained {len(epochs)} "
            f"of {len(metadata)} total-TFR good trials."
        )

    epochs.resample(
        float(
            params[
                "epoch_resample_hz"
            ]
        ),
        npad="auto",
        verbose="ERROR",
    )

    return epochs


# =============================================================================
# Cross-fitted generic decision-evoked removal
# =============================================================================

def crossfit_remove_generic_evoked(
    broadband_epochs: np.ndarray,
    metadata: pd.DataFrame,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Remove the generic decision-cue evoked waveform without label leakage.

    For each held-out decision block:
        template = mean waveform from all OTHER blocks
        residual = held-out trial - template

    Pain level, money level, choice, and RT are never used.
    """

    block_ids = pd.to_numeric(
        metadata["block"],
        errors="raise",
    ).to_numpy(
        dtype=int
    )

    unique_blocks = np.unique(
        block_ids
    )

    if len(
        unique_blocks
    ) < 2:
        raise RuntimeError(
            "Need at least two decision blocks for cross-fitted evoked removal."
        )

    residual_epochs = np.empty_like(
        broadband_epochs,
        dtype=np.float64,
    )

    assigned = np.zeros(
        len(metadata),
        dtype=bool,
    )

    diagnostics = []

    for heldout_block in unique_blocks:
        training_mask = (
            block_ids
            != heldout_block
        )

        heldout_mask = (
            block_ids
            == heldout_block
        )

        if not training_mask.any():
            raise RuntimeError(
                f"Block {heldout_block}: no training trials."
            )

        if not heldout_mask.any():
            raise RuntimeError(
                f"Block {heldout_block}: no held-out trials."
            )

        generic_evoked_template = broadband_epochs[
            training_mask
        ].mean(
            axis=0
        )

        residual_epochs[
            heldout_mask
        ] = (
            broadband_epochs[
                heldout_mask
            ]
            - generic_evoked_template
        )

        assigned[
            heldout_mask
        ] = True

        rms_before = float(
            np.sqrt(
                np.mean(
                    broadband_epochs[
                        heldout_mask
                    ] ** 2
                )
            )
        )

        rms_after = float(
            np.sqrt(
                np.mean(
                    residual_epochs[
                        heldout_mask
                    ] ** 2
                )
            )
        )

        diagnostics.append(
            {
                "heldout_block":
                    int(
                        heldout_block
                    ),
                "n_template_trials":
                    int(
                        training_mask.sum()
                    ),
                "n_heldout_trials":
                    int(
                        heldout_mask.sum()
                    ),
                "rms_before":
                    rms_before,
                "rms_after":
                    rms_after,
            }
        )

    if not assigned.all():
        raise RuntimeError(
            "Some trials were not assigned an evoked-removed waveform."
        )

    return (
        residual_epochs,
        pd.DataFrame(
            diagnostics
        ),
    )


# =============================================================================
# TFR computation
# =============================================================================

def compute_evoked_removed_tfr(
    participant: str,
    epochs: mne.Epochs,
    residual_epochs: np.ndarray,
    params: Dict,
    reference_times: np.ndarray,
    reference_frequencies: np.ndarray,
    reference_channels: List[str],
    n_jobs: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute TFR power from residual waveforms using the exact total-TFR
    parameter choices.
    """

    frequencies = np.asarray(
        params[
            "frequencies_hz"
        ],
        dtype=float,
    )

    if not np.array_equal(
        frequencies,
        reference_frequencies,
    ):
        raise RuntimeError(
            f"{participant}: frequency list in params does not match total TFR."
        )

    if list(
        epochs.ch_names
    ) != list(
        reference_channels
    ):
        raise RuntimeError(
            f"{participant}: channel order does not match total TFR."
        )

    tfr_decimation = int(
        params[
            "tfr_decim"
        ]
    )

    tfr_sample_times = epochs.times[
        ::tfr_decimation
    ].astype(
        float
    )

    baseline_start_s, baseline_end_s = map(
        float,
        params[
            "baseline_s"
        ],
    )

    baseline_mask = (
        (tfr_sample_times >= baseline_start_s - 1e-12)
        & (tfr_sample_times <= baseline_end_s + 1e-12)
    )

    if not baseline_mask.any():
        raise RuntimeError(
            f"{participant}: no TFR samples inside baseline."
        )

    save_start_s, save_end_s = map(
        float,
        params[
            "saved_time_range_s"
        ],
    )

    bin_width_s = float(
        params[
            "time_bin_s"
        ]
    )

    bin_centers, bin_masks = make_time_bins(
        sample_times=tfr_sample_times,
        start_s=save_start_s,
        end_s=save_end_s,
        bin_width_s=bin_width_s,
    )

    if not np.allclose(
        bin_centers,
        reference_times,
        atol=1e-12,
        rtol=0.0,
    ):
        raise RuntimeError(
            f"{participant}: evoked-removed and total TFR time grids differ."
        )

    # The completed total-TFR params file does not necessarily carry this
    # implementation detail, so use the same chunk size as the total prep.
    frequency_chunk_size = 6

    final_power = np.empty(
        (
            len(epochs),
            len(epochs.ch_names),
            len(frequencies),
            len(bin_centers),
        ),
        dtype=np.float32,
    )

    print(
        f"[{participant}] evoked-removed TFR "
        f"trials={len(epochs)} channels={len(epochs.ch_names)} "
        f"freqs={len(frequencies)} bins={len(bin_centers)}",
        flush=True,
    )

    for start_index in range(
        0,
        len(frequencies),
        frequency_chunk_size,
    ):
        end_index = min(
            start_index
            + frequency_chunk_size,
            len(frequencies),
        )

        frequency_chunk = frequencies[
            start_index:end_index
        ]

        print(
            f"[{participant}] "
            f"{frequency_chunk[0]:.0f}-"
            f"{frequency_chunk[-1]:.0f} Hz",
            flush=True,
        )

        power = tfr_array_morlet(
            residual_epochs,
            sfreq=float(
                epochs.info[
                    "sfreq"
                ]
            ),
            freqs=frequency_chunk,
            n_cycles=frequency_chunk * 0.5,
            output="power",
            use_fft=True,
            zero_mean=True,
            decim=tfr_decimation,
            n_jobs=max(
                1,
                n_jobs,
            ),
            verbose=False,
        )

        log_power = np.log(
            np.maximum(
                power,
                LOG_FLOOR,
            )
        )

        normalized_log_power = (
            log_power
            - log_power[
                ...,
                baseline_mask,
            ].mean(
                axis=-1,
                keepdims=True,
            )
        )

        for bin_index, sample_indices in enumerate(
            bin_masks
        ):
            final_power[
                :,
                :,
                start_index:end_index,
                bin_index,
            ] = normalized_log_power[
                ...,
                sample_indices,
            ].mean(
                axis=-1
            ).astype(
                np.float32
            )

        del (
            power,
            log_power,
            normalized_log_power,
        )

    if not np.isfinite(
        final_power
    ).all():
        raise RuntimeError(
            f"{participant}: non-finite evoked-removed TFR power."
        )

    return (
        final_power,
        bin_centers,
        frequencies,
    )


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = parse_args()
    participant = args.subject

    (
        metadata,
        params,
        reference_times,
        reference_frequencies,
        reference_channels,
    ) = load_total_tfr_reference(
        participant
    )

    outputs = output_paths(
        participant
    )

    outputs[
        "directory"
    ].mkdir(
        parents=True,
        exist_ok=True,
    )

    output_files = [
        path
        for key, path in outputs.items()
        if key != "directory"
    ]

    n_existing = sum(
        path.exists()
        for path in output_files
    )

    if (
        n_existing
        == len(
            output_files
        )
        and not args.overwrite
    ):
        print(
            f"[{participant}] complete evoked-removed outputs already exist -> SKIP",
            flush=True,
        )
        return

    if (
        n_existing > 0
        and not args.overwrite
    ):
        raise FileExistsError(
            f"{participant}: partial evoked-removed output set exists. "
            "Inspect it or rerun with --overwrite."
        )

    epochs = load_long_epochs(
        participant=participant,
        metadata=metadata,
        params=params,
    )

    broadband_data = epochs.get_data(
        copy=False
    ).astype(
        np.float64,
        copy=False,
    )

    (
        residual_epochs,
        diagnostics,
    ) = crossfit_remove_generic_evoked(
        broadband_epochs=broadband_data,
        metadata=epochs.metadata.reset_index(
            drop=True
        ),
    )

    diagnostics.to_csv(
        outputs[
            "diagnostics"
        ],
        index=False,
    )

    (
        evoked_removed_power,
        output_times,
        output_frequencies,
    ) = compute_evoked_removed_tfr(
        participant=participant,
        epochs=epochs,
        residual_epochs=residual_epochs,
        params=params,
        reference_times=reference_times,
        reference_frequencies=reference_frequencies,
        reference_channels=reference_channels,
        n_jobs=args.n_jobs,
    )

    atomic_save_npy(
        outputs[
            "power"
        ],
        evoked_removed_power,
    )

    atomic_save_npy(
        outputs[
            "times"
        ],
        output_times.astype(
            float
        ),
    )

    atomic_save_npy(
        outputs[
            "freqs"
        ],
        output_frequencies.astype(
            float
        ),
    )

    epochs.metadata.reset_index(
        drop=True
    ).to_csv(
        outputs[
            "metadata"
        ],
        index=False,
    )

    outputs[
        "channels"
    ].write_text(
        json.dumps(
            list(
                epochs.ch_names
            ),
            indent=2,
        )
    )

    output_params = {
        "signal":
            "cross-fitted generic decision-evoked removed; induced-enriched",
        "trial_set":
            "exact same good trials as total-power decision TFR",
        "evoked_removal":
            (
                "for held-out block b, subtract mean decision-cue waveform "
                "estimated from all good trials in blocks != b"
            ),
        "labels_used_for_evoked_removal":
            "none",
        "important_limitation":
            (
                "generic evoked removal does not guarantee removal of every "
                "pain-level-specific or money-level-specific phase-locked component"
            ),
        "working_epoch_s":
            params[
                "working_epoch_s"
            ],
        "epoch_resample_hz":
            params[
                "epoch_resample_hz"
            ],
        "frequencies_hz":
            output_frequencies.tolist(),
        "n_cycles_rule":
            "n_cycles = frequency * 0.5",
        "tfr_decim":
            params[
                "tfr_decim"
            ],
        "baseline_s":
            params[
                "baseline_s"
            ],
        "baseline_normalization":
            "log(power) - mean(log(power_baseline))",
        "saved_time_range_s":
            params[
                "saved_time_range_s"
            ],
        "time_bin_s":
            params[
                "time_bin_s"
            ],
        "data_shape_order":
            [
                "trial",
                "channel",
                "frequency",
                "time",
            ],
    }

    outputs[
        "params"
    ].write_text(
        json.dumps(
            output_params,
            indent=2,
        )
    )

    summary = {
        "participant":
            participant,
        "n_trials":
            int(
                len(
                    epochs
                )
            ),
        "n_channels":
            int(
                len(
                    epochs.ch_names
                )
            ),
        "shape":
            list(
                evoked_removed_power.shape
            ),
        "mean_rms_before":
            float(
                diagnostics[
                    "rms_before"
                ].mean()
            ),
        "mean_rms_after":
            float(
                diagnostics[
                    "rms_after"
                ].mean()
            ),
    }

    outputs[
        "summary"
    ].write_text(
        json.dumps(
            summary,
            indent=2,
        )
    )

    print(
        f"[{participant}] DONE",
        flush=True,
    )


if __name__ == "__main__":
    main()
