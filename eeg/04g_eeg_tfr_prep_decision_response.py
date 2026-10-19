#!/usr/bin/env python3
"""
04g_eeg_tfr_prep_decision_response.py
=====================================

Construct genuinely RESPONSE-LOCKED single-trial TFR power for the PainReward
decision task.

This script is deliberately matched as closely as possible to the completed
cue-locked decision TFR pipeline. The key change is the temporal reference:

    cue-locked TFR:       t = 0 is the decision cue ("off+")
    response-locked TFR:  t = 0 is the actual EEG response event
                          ("res+", "res-", or "resm")

Scientific purpose
------------------
The response-locked TFR is intended for a later MVPA analysis asking whether
pain- and money-magnitude representations are organised relative to the time
of the response. This is especially useful for distinguishing:

    1. a neural representation that is smeared in cue-locked time because its
       latency covaries with RT,

from

    2. a representation that only becomes detectable after statistical
       removal of RT-related variance.

IMPORTANT: this script performs NO decoding and NO inferential testing.

Inputs
------
1. Cleaned broadband decision EEG:
       SOURCE_BIDS/derivatives/<sub>/eeg/
           <sub>_decision_cleaned-raw.fif

2. Completed cue-locked decision TFR metadata/parameters:
       BASE_PATH/derivatives/<sub>/eeg/tfr_decision/
           <sub>_decision_cue_metadata.csv
           <sub>_decision_cue_params.json

   These files provide:
       - the exact GOOD trial set already used in the decision TFR analyses;
       - cue event sample;
       - pain level;
       - money level;
       - decision block;
       - behavioural RT;
       - choice/accepted;
       - the exact spectral parameters already used.

3. Decision events TSV:
       SOURCE_BIDS/<sub>/eeg/<sub>_task-decision_events.tsv

   The actual response markers res+/res-/resm are used as response time zero.

Core TFR choices
----------------
Inherited from the COMPLETED cue-locked decision TFR:
    - frequency grid (normally 4-45 Hz, 1-Hz steps)
    - n_cycles = frequency / 2
    - epoch resampling rate
    - Morlet decimation
    - frequency chunk size
    - 20-ms power-envelope bins
    - cue-relative spectral baseline (normally -0.7..-0.5 s)

Response-specific choices:
    - actual EEG response event is t = 0
    - saved response-relative range: -0.8..+0.2 s
    - a LONG response-locked working epoch is created automatically so that:
          a) the original PRE-CUE baseline is present for every retained trial;
          b) Morlet support is available around the saved response interval.
    - NO response-relative voltage baseline is applied.

Why keep the cue-relative TFR baseline?
---------------------------------------
We do NOT use the old ERP baseline of -0.8..-0.7 s relative to response.
That interval can already contain decision-related activity.

Instead, every response-locked trial is normalised to the SAME pre-cue
baseline used in the cue-locked TFR analysis.

For trial i with cue-to-response interval RT_event_i:

    response-relative baseline =
        [-RT_event_i - 0.7, -RT_event_i - 0.5] s

This means response-locked and cue-locked power remain directly comparable.

Timing safeguards
-----------------
The script:
    - requires 125 response events in chronological order;
    - merges them to the already-approved cue-locked trial metadata by the
      original trial number;
    - checks that response event sample > cue event sample;
    - compares event-derived cue-to-response latency with behavioural RT;
    - writes trial-level timing diagnostics;
    - records the earliest response-relative centre whose full conservative
      Morlet support lies after cue onset.

The later decoder should use these timing diagnostics to define its formal
response-relative inference window BEFORE inspecting neural decoding results.

Output
------
BASE_PATH/derivatives/<sub>/eeg/tfr_decision_response/

    <sub>_decision_response_power.npy
        trial x channel x frequency x response-time-bin

    <sub>_decision_response_times.npy
        response-relative bin centres, seconds

    <sub>_decision_response_freqs.npy

    <sub>_decision_response_metadata.csv
        exact good trials + cue/response timing diagnostics

    <sub>_decision_response_channels.json
    <sub>_decision_response_params.json
    <sub>_decision_response_prep_summary.json
    <sub>_decision_response_timing_qc.csv
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple

import mne
import numpy as np
import pandas as pd
from mne.time_frequency import tfr_array_morlet


# =============================================================================
# Project configuration
# =============================================================================

mne.set_log_level("ERROR")

BASE_PATH = Path(
    os.getenv(
        "basepath",
        Path(__file__).parent.parent.parent,
    )
)

SOURCE_BIDS = Path(
    os.getenv(
        "SOURCE_BIDS",
        BASE_PATH,
    )
)

CHANNELS_TO_DROP = [
    "HEOGL",
    "HEOGR",
    "VEOGL",
    "STI 014",
    "Status",
]

RESPONSE_EVENT_TYPES = [
    "res+",
    "res-",
    "resm",
]

EXPECTED_DECISION_TRIALS = 125

# Keep the response window identical to the old response-locked ERP analysis.
RESPONSE_SAVE_TMIN = -0.8
RESPONSE_SAVE_TMAX = 0.2

# Conservative estimate used throughout this project:
# MNE Morlet support ~= +/- 5 sigma_t.
MORLET_N_SIGMAS = 5.0
N_CYCLES_FACTOR = 0.5

# Extra working-epoch padding beyond the conservative Morlet support.
EXTRA_EDGE_PADDING_S = 0.05

# Fail loudly if behavioural RT and the actual EEG response marker differ by
# more than this amount on any retained trial. If this triggers, inspect the
# event definitions rather than silently proceeding with uncertain alignment.
MAX_BEHAVIOURAL_EVENT_RT_DIFFERENCE_S = 0.100

LOG_FLOOR = np.finfo(np.float64).tiny


# =============================================================================
# Command-line arguments
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare response-locked single-trial decision TFR power."
        )
    )

    parser.add_argument(
        "--subject",
        required=True,
        help="Participant ID, for example sub-001.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing complete response-locked TFR output set.",
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
        help="Parallel jobs passed to MNE Morlet computation.",
    )

    return parser.parse_args()


# =============================================================================
# Paths
# =============================================================================

def decision_raw_path(
    participant: str,
) -> Path:
    return (
        SOURCE_BIDS
        / "derivatives"
        / participant
        / "eeg"
        / f"{participant}_decision_cleaned-raw.fif"
    )


def decision_events_path(
    participant: str,
) -> Path:
    canonical = (
        SOURCE_BIDS
        / participant
        / "eeg"
        / f"{participant}_task-decision_events.tsv"
    )

    if canonical.exists():
        return canonical

    matches = sorted(
        (
            SOURCE_BIDS
            / participant
            / "eeg"
        ).glob(
            "*task-decision*events.tsv"
        )
    )

    if len(matches) != 1:
        raise FileNotFoundError(
            f"{participant}: expected exactly one decision events TSV; "
            f"found {matches}"
        )

    return matches[0]


def cue_locked_reference_paths(
    participant: str,
) -> Dict[str, Path]:
    directory = (
        BASE_PATH
        / "derivatives"
        / participant
        / "eeg"
        / "tfr_decision"
    )

    stem = f"{participant}_decision_cue"

    return {
        "metadata":
            directory
            / f"{stem}_metadata.csv",
        "params":
            directory
            / f"{stem}_params.json",
        "channels":
            directory
            / f"{stem}_channels.json",
        "freqs":
            directory
            / f"{stem}_freqs.npy",
    }


def output_paths(
    participant: str,
) -> Dict[str, Path]:
    directory = (
        BASE_PATH
        / "derivatives"
        / participant
        / "eeg"
        / "tfr_decision_response"
    )

    stem = f"{participant}_decision_response"

    return {
        "directory": directory,
        "power":
            directory
            / f"{stem}_power.npy",
        "times":
            directory
            / f"{stem}_times.npy",
        "freqs":
            directory
            / f"{stem}_freqs.npy",
        "metadata":
            directory
            / f"{stem}_metadata.csv",
        "channels":
            directory
            / f"{stem}_channels.json",
        "params":
            directory
            / f"{stem}_params.json",
        "summary":
            directory
            / f"{stem}_prep_summary.json",
        "timing_qc":
            directory
            / f"{stem}_timing_qc.csv",
    }


# =============================================================================
# Small utilities
# =============================================================================

def atomic_save_numpy(
    path: Path,
    array: np.ndarray,
) -> None:
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


def conservative_morlet_half_support_s(
    frequency_hz: float,
) -> float:
    """
    Conservative project-wide Morlet half-support estimate.

    n_cycles = frequency * 0.5

    sigma_t = n_cycles / (2*pi*frequency)

    support ~= +/- 5*sigma_t
    """

    n_cycles = N_CYCLES_FACTOR * frequency_hz
    sigma_t = (
        n_cycles
        / (
            2.0
            * np.pi
            * frequency_hz
        )
    )

    return (
        MORLET_N_SIGMAS
        * sigma_t
    )


def make_time_bins(
    times: np.ndarray,
    start_s: float,
    end_s: float,
    bin_width_s: float,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Create the same non-overlapping power-envelope bins used by the cue TFRs.
    """

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
        edges[-1]
        - end_s
    ) > 1e-9:
        edges = np.append(
            edges[
                edges
                < end_s
            ],
            end_s,
        )

    centres = []
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
            indices = np.where(
                (
                    times
                    >= low - 1e-12
                )
                & (
                    times
                    <= high + 1e-12
                )
            )[0]
        else:
            indices = np.where(
                (
                    times
                    >= low - 1e-12
                )
                & (
                    times
                    < high - 1e-12
                )
            )[0]

        if len(indices) == 0:
            raise RuntimeError(
                f"No TFR samples in response-time bin "
                f"{low:.3f}..{high:.3f} s"
            )

        centres.append(
            (low + high) / 2.0
        )
        masks.append(
            indices
        )

    return (
        np.asarray(
            centres,
            dtype=float,
        ),
        masks,
    )


# =============================================================================
# Load the completed cue-locked decision TFR definition
# =============================================================================

def load_cue_locked_reference(
    participant: str,
):
    """
    Load the exact good-trial set and exact TFR parameters already approved in
    the cue-locked decision analysis.
    """

    paths = cue_locked_reference_paths(
        participant
    )

    missing = [
        str(path)
        for path in paths.values()
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            f"{participant}: missing completed cue-locked decision TFR "
            f"reference files:\n"
            + "\n".join(
                missing
            )
        )

    metadata = pd.read_csv(
        paths["metadata"]
    )

    with open(
        paths["params"],
        "r",
        encoding="utf-8",
    ) as file_handle:
        parameters = json.load(
            file_handle
        )

    with open(
        paths["channels"],
        "r",
        encoding="utf-8",
    ) as file_handle:
        reference_channels = json.load(
            file_handle
        )

    reference_frequencies = np.load(
        paths["freqs"]
    ).astype(
        float
    )

    required_metadata_columns = [
        "trialsnum",
        "sample",
        "painlevel",
        "moneylevel",
        "block",
        "rt",
    ]

    missing_metadata_columns = [
        column
        for column in required_metadata_columns
        if column not in metadata.columns
    ]

    if missing_metadata_columns:
        raise KeyError(
            f"{participant}: cue-locked decision metadata missing "
            f"{missing_metadata_columns}"
        )

    required_parameter_keys = [
        "epoch_resample_hz",
        "frequencies_hz",
        "tfr_decim",
        "baseline_s",
        "time_bin_s",
        "frequency_chunk_size",
    ]

    missing_parameter_keys = [
        key
        for key in required_parameter_keys
        if key not in parameters
    ]

    if missing_parameter_keys:
        raise KeyError(
            f"{participant}: cue-locked decision params missing "
            f"{missing_parameter_keys}"
        )

    parameter_frequencies = np.asarray(
        parameters[
            "frequencies_hz"
        ],
        dtype=float,
    )

    if not np.array_equal(
        parameter_frequencies,
        reference_frequencies,
    ):
        raise RuntimeError(
            f"{participant}: frequency grid in params and NPY differs"
        )

    metadata = metadata.copy()

    metadata["trialsnum"] = pd.to_numeric(
        metadata["trialsnum"],
        errors="raise",
    ).astype(
        int
    )

    metadata["sample"] = pd.to_numeric(
        metadata["sample"],
        errors="raise",
    ).astype(
        int
    )

    metadata["rt"] = pd.to_numeric(
        metadata["rt"],
        errors="raise",
    ).astype(
        float
    )

    if metadata["trialsnum"].duplicated().any():
        raise RuntimeError(
            f"{participant}: duplicate trialsnum values in cue metadata"
        )

    if not np.isfinite(
        metadata["rt"].to_numpy(
            float
        )
    ).all():
        raise RuntimeError(
            f"{participant}: non-finite RT survived cue-TFR trial selection"
        )

    # Make the temporal meaning explicit before merging response events.
    metadata = metadata.rename(
        columns={
            "sample":
                "cue_sample",
            "rt":
                "behavioral_rt_s",
        }
    )

    return (
        metadata,
        parameters,
        reference_channels,
        reference_frequencies,
    )


# =============================================================================
# Actual response events
# =============================================================================

def load_response_events(
    participant: str,
) -> pd.DataFrame:
    """
    Load the actual EEG response markers and assign original trial number by
    chronological order, matching the established response-locked ERP pipeline.
    """

    path = decision_events_path(
        participant
    )

    events = pd.read_csv(
        path,
        sep="\t",
    )

    required_columns = [
        "trial_type",
        "sample",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in events.columns
    ]

    if missing_columns:
        raise KeyError(
            f"{participant}: decision events TSV missing "
            f"{missing_columns}"
        )

    response_events = events[
        events[
            "trial_type"
        ].astype(
            str
        ).isin(
            RESPONSE_EVENT_TYPES
        )
    ].copy()

    response_events = (
        response_events
        .sort_values(
            "sample"
        )
        .reset_index(
            drop=True
        )
    )

    if len(
        response_events
    ) != EXPECTED_DECISION_TRIALS:
        counts = (
            response_events[
                "trial_type"
            ]
            .astype(
                str
            )
            .value_counts(
                dropna=False
            )
            .to_dict()
        )

        raise RuntimeError(
            f"{participant}: expected "
            f"{EXPECTED_DECISION_TRIALS} response events across "
            f"{RESPONSE_EVENT_TYPES}, got {len(response_events)}; "
            f"counts={counts}"
        )

    response_events[
        "response_sample"
    ] = pd.to_numeric(
        response_events[
            "sample"
        ],
        errors="raise",
    ).astype(
        int
    )

    response_events[
        "response_event_type"
    ] = (
        response_events[
            "trial_type"
        ]
        .astype(
            str
        )
    )

    response_events[
        "trialsnum"
    ] = np.arange(
        1,
        EXPECTED_DECISION_TRIALS + 1,
        dtype=int,
    )

    return response_events[
        [
            "trialsnum",
            "response_sample",
            "response_event_type",
        ]
    ]


# =============================================================================
# Trial timing and alignment
# =============================================================================

def build_response_locked_metadata(
    participant: str,
    cue_metadata: pd.DataFrame,
    raw_sampling_rate_hz: float,
    cue_baseline_s: Tuple[float, float],
    morlet_half_support_s: float,
) -> pd.DataFrame:
    """
    Merge the actual response event to the exact retained cue-TFR trial set.
    """

    response_events = load_response_events(
        participant
    )

    metadata = cue_metadata.merge(
        response_events,
        on="trialsnum",
        how="left",
        validate="one_to_one",
    )

    if metadata[
        [
            "response_sample",
            "response_event_type",
        ]
    ].isna().any().any():
        raise RuntimeError(
            f"{participant}: retained cue-TFR trials failed to match "
            f"response events"
        )

    metadata[
        "response_sample"
    ] = pd.to_numeric(
        metadata[
            "response_sample"
        ],
        errors="raise",
    ).astype(
        int
    )

    sample_difference = (
        metadata[
            "response_sample"
        ].to_numpy(
            int
        )
        - metadata[
            "cue_sample"
        ].to_numpy(
            int
        )
    )

    if np.any(
        sample_difference
        <= 0
    ):
        bad_trials = metadata.loc[
            sample_difference
            <= 0,
            "trialsnum",
        ].tolist()

        raise RuntimeError(
            f"{participant}: response event does not follow cue on "
            f"trial(s) {bad_trials[:10]}"
        )

    metadata[
        "event_rt_s"
    ] = (
        sample_difference
        / float(
            raw_sampling_rate_hz
        )
    )

    metadata[
        "rt_alignment_error_s"
    ] = (
        metadata[
            "event_rt_s"
        ]
        - metadata[
            "behavioral_rt_s"
        ]
    )

    absolute_error = np.abs(
        metadata[
            "rt_alignment_error_s"
        ].to_numpy(
            float
        )
    )

    if np.any(
        absolute_error
        > MAX_BEHAVIOURAL_EVENT_RT_DIFFERENCE_S
    ):
        worst_index = int(
            np.argmax(
                absolute_error
            )
        )

        worst = metadata.iloc[
            worst_index
        ]

        raise RuntimeError(
            f"{participant}: behavioural RT and EEG response-event latency "
            f"differ by more than "
            f"{MAX_BEHAVIOURAL_EVENT_RT_DIFFERENCE_S:.3f} s. "
            f"Worst trial={int(worst['trialsnum'])}, "
            f"behavioural RT={float(worst['behavioral_rt_s']):.4f}, "
            f"event RT={float(worst['event_rt_s']):.4f}, "
            f"difference={float(worst['rt_alignment_error_s']):.4f}. "
            f"Inspect trial/event alignment before continuing."
        )

    baseline_start_cue_s = float(
        cue_baseline_s[
            0
        ]
    )
    baseline_end_cue_s = float(
        cue_baseline_s[
            1
        ]
    )

    # Convert the fixed cue-relative baseline into response-relative time.
    metadata[
        "baseline_response_tmin_s"
    ] = (
        baseline_start_cue_s
        - metadata[
            "event_rt_s"
        ]
    )

    metadata[
        "baseline_response_tmax_s"
    ] = (
        baseline_end_cue_s
        - metadata[
            "event_rt_s"
        ]
    )

    metadata[
        "cue_time_relative_to_response_s"
    ] = (
        -metadata[
            "event_rt_s"
        ]
    )

    # A response-relative TFR centre is fully post-cue only if:
    #
    #     centre - half_support >= -RT
    #
    # therefore:
    #
    #     centre >= half_support - RT
    metadata[
        "earliest_center_full_morlet_support_after_cue_s"
    ] = (
        morlet_half_support_s
        - metadata[
            "event_rt_s"
        ]
    )

    return metadata


# =============================================================================
# Working response-locked epochs
# =============================================================================

def determine_working_epoch(
    metadata: pd.DataFrame,
    morlet_half_support_s: float,
) -> Tuple[float, float]:
    """
    Choose a response-locked working epoch long enough for:
      1. every trial's original pre-cue TFR baseline;
      2. the entire saved response-relative window;
      3. conservative Morlet support around both.
    """

    earliest_baseline_start = float(
        metadata[
            "baseline_response_tmin_s"
        ].min()
    )

    earliest_needed_centre = min(
        earliest_baseline_start,
        RESPONSE_SAVE_TMIN,
    )

    latest_needed_centre = max(
        float(
            metadata[
                "baseline_response_tmax_s"
            ].max()
        ),
        RESPONSE_SAVE_TMAX,
    )

    working_tmin = (
        earliest_needed_centre
        - morlet_half_support_s
        - EXTRA_EDGE_PADDING_S
    )

    working_tmax = (
        latest_needed_centre
        + morlet_half_support_s
        + EXTRA_EDGE_PADDING_S
    )

    # Millisecond rounding makes the saved parameter file easier to inspect.
    working_tmin = (
        math.floor(
            working_tmin
            * 1000.0
        )
        / 1000.0
    )

    working_tmax = (
        math.ceil(
            working_tmax
            * 1000.0
        )
        / 1000.0
    )

    return (
        working_tmin,
        working_tmax,
    )


def create_response_locked_epochs(
    participant: str,
    metadata: pd.DataFrame,
    reference_channels: List[str],
    epoch_resample_hz: float,
    working_tmin: float,
    working_tmax: float,
):
    """
    Epoch CLEANED BROADBAND raw around actual response markers.
    """

    raw_path = decision_raw_path(
        participant
    )

    if not raw_path.exists():
        raise FileNotFoundError(
            raw_path
        )

    raw = mne.io.read_raw_fif(
        str(
            raw_path
        ),
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

    if list(
        raw.ch_names
    ) != list(
        reference_channels
    ):
        raise RuntimeError(
            f"{participant}: response-locked raw channel order differs "
            f"from completed cue-locked TFR channel order"
        )

    events = np.column_stack(
        [
            metadata[
                "response_sample"
            ].to_numpy(
                int
            ),
            np.zeros(
                len(
                    metadata
                ),
                dtype=int,
            ),
            np.ones(
                len(
                    metadata
                ),
                dtype=int,
            ),
        ]
    )

    epochs = mne.Epochs(
        raw,
        events=events,
        event_id={
            "response":
                1
        },
        tmin=working_tmin,
        tmax=working_tmax,
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
            f"{participant}: long response epoching retained "
            f"{len(epochs)}/{len(metadata)} already-approved trials. "
            f"Inspect annotations/working epoch before continuing."
        )

    epochs.resample(
        epoch_resample_hz,
        npad="auto",
        verbose="ERROR",
    )

    return epochs


# =============================================================================
# TFR computation
# =============================================================================

def compute_response_locked_tfr(
    participant: str,
    epochs: mne.Epochs,
    metadata: pd.DataFrame,
    frequencies_hz: np.ndarray,
    tfr_decim: int,
    time_bin_s: float,
    frequency_chunk_size: int,
    n_jobs: int,
):
    """
    Compute total power and apply each trial's cue-relative pre-cue baseline.
    """

    data = epochs.get_data(
        copy=False
    ).astype(
        np.float64,
        copy=False,
    )

    morlet_times = epochs.times[
        ::tfr_decim
    ]

    response_bin_centres, response_bin_masks = make_time_bins(
        times=morlet_times,
        start_s=RESPONSE_SAVE_TMIN,
        end_s=RESPONSE_SAVE_TMAX,
        bin_width_s=time_bin_s,
    )

    output = np.empty(
        (
            len(
                epochs
            ),
            len(
                epochs.ch_names
            ),
            len(
                frequencies_hz
            ),
            len(
                response_bin_centres
            ),
        ),
        dtype=np.float32,
    )

    # Build a DIFFERENT baseline mask for every trial because the baseline is
    # fixed relative to the cue while the TFR itself is response-locked.
    trial_baseline_masks = []

    for trial_index, row in metadata.iterrows():
        baseline_start = float(
            row[
                "baseline_response_tmin_s"
            ]
        )
        baseline_end = float(
            row[
                "baseline_response_tmax_s"
            ]
        )

        baseline_indices = np.where(
            (
                morlet_times
                >= baseline_start - 1e-12
            )
            & (
                morlet_times
                <= baseline_end + 1e-12
            )
        )[0]

        if len(
            baseline_indices
        ) == 0:
            raise RuntimeError(
                f"{participant}: trial "
                f"{int(row['trialsnum'])} has no response-relative "
                f"TFR samples in its cue baseline "
                f"{baseline_start:.3f}..{baseline_end:.3f} s"
            )

        trial_baseline_masks.append(
            baseline_indices
        )

    print(
        f"[{participant}] response TFR: "
        f"trials={len(epochs)}, "
        f"channels={len(epochs.ch_names)}, "
        f"freqs={len(frequencies_hz)}, "
        f"bins={len(response_bin_centres)}",
        flush=True,
    )

    for frequency_start in range(
        0,
        len(
            frequencies_hz
        ),
        frequency_chunk_size,
    ):
        frequency_end = min(
            frequency_start
            + frequency_chunk_size,
            len(
                frequencies_hz
            ),
        )

        chunk_frequencies = frequencies_hz[
            frequency_start:
            frequency_end
        ]

        print(
            f"[{participant}] "
            f"{chunk_frequencies[0]:.0f}-"
            f"{chunk_frequencies[-1]:.0f} Hz",
            flush=True,
        )

        power = tfr_array_morlet(
            data,
            sfreq=float(
                epochs.info[
                    "sfreq"
                ]
            ),
            freqs=chunk_frequencies,
            n_cycles=(
                chunk_frequencies
                * N_CYCLES_FACTOR
            ),
            output="power",
            use_fft=True,
            zero_mean=True,
            decim=tfr_decim,
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

        # Baseline EACH trial to the same cue-relative pre-cue baseline that was
        # used in the cue-locked TFR pipeline.
        for trial_index, baseline_indices in enumerate(
            trial_baseline_masks
        ):
            baseline_mean = (
                log_power[
                    trial_index,
                    ...,
                    baseline_indices
                ]
                .mean(
                    axis=-1,
                    keepdims=True,
                )
            )

            log_power[
                trial_index
            ] -= baseline_mean

        # Only after log-ratio normalisation do we make the same 20-ms
        # power-envelope bins as in the cue-locked TFR.
        for bin_index, time_indices in enumerate(
            response_bin_masks
        ):
            output[
                :,
                :,
                frequency_start:
                frequency_end,
                bin_index,
            ] = (
                log_power[
                    ...,
                    time_indices
                ]
                .mean(
                    axis=-1,
                )
                .astype(
                    np.float32
                )
            )

        del power
        del log_power

    return (
        output,
        response_bin_centres,
    )


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = parse_args()
    participant = args.subject

    paths = output_paths(
        participant
    )

    paths[
        "directory"
    ].mkdir(
        parents=True,
        exist_ok=True,
    )

    output_files = [
        path
        for key, path in paths.items()
        if key != "directory"
    ]

    existing_outputs = [
        path
        for path in output_files
        if path.exists()
    ]

    if existing_outputs and not args.overwrite:
        raise FileExistsError(
            f"{participant}: response-locked TFR outputs already exist. "
            f"Use --overwrite to replace them.\n"
            + "\n".join(
                str(path)
                for path in existing_outputs
            )
        )

    (
        cue_metadata,
        cue_parameters,
        reference_channels,
        frequencies_hz,
    ) = load_cue_locked_reference(
        participant
    )

    raw_path = decision_raw_path(
        participant
    )

    if not raw_path.exists():
        raise FileNotFoundError(
            raw_path
        )

    # Open once without preload to obtain the native EEG sampling rate used by
    # event sample numbers.
    raw_header = mne.io.read_raw_fif(
        str(
            raw_path
        ),
        preload=False,
        verbose="ERROR",
    )

    native_sampling_rate_hz = float(
        raw_header.info[
            "sfreq"
        ]
    )

    cue_baseline_s = tuple(
        map(
            float,
            cue_parameters[
                "baseline_s"
            ],
        )
    )

    lowest_frequency_hz = float(
        np.min(
            frequencies_hz
        )
    )

    morlet_half_support_s = conservative_morlet_half_support_s(
        lowest_frequency_hz
    )

    metadata = build_response_locked_metadata(
        participant=participant,
        cue_metadata=cue_metadata,
        raw_sampling_rate_hz=native_sampling_rate_hz,
        cue_baseline_s=cue_baseline_s,
        morlet_half_support_s=morlet_half_support_s,
    )

    working_tmin, working_tmax = determine_working_epoch(
        metadata=metadata,
        morlet_half_support_s=morlet_half_support_s,
    )

    epoch_resample_hz = float(
        cue_parameters[
            "epoch_resample_hz"
        ]
    )

    tfr_decim = int(
        cue_parameters[
            "tfr_decim"
        ]
    )

    time_bin_s = float(
        cue_parameters[
            "time_bin_s"
        ]
    )

    frequency_chunk_size = int(
        cue_parameters[
            "frequency_chunk_size"
        ]
    )

    print(
        "=" * 78,
        flush=True,
    )
    print(
        f"[{participant}] RESPONSE-LOCKED DECISION TFR",
        flush=True,
    )
    print(
        f"Actual response markers: {RESPONSE_EVENT_TYPES}",
        flush=True,
    )
    print(
        f"Good trials inherited from cue TFR: {len(metadata)}",
        flush=True,
    )
    print(
        f"Behaviour/event RT max abs difference: "
        f"{np.abs(metadata['rt_alignment_error_s']).max():.6f} s",
        flush=True,
    )
    print(
        f"Working epoch: {working_tmin:.3f}..{working_tmax:.3f} s",
        flush=True,
    )
    print(
        f"Saved response range: "
        f"{RESPONSE_SAVE_TMIN:.3f}..{RESPONSE_SAVE_TMAX:.3f} s",
        flush=True,
    )
    print(
        f"Cue-relative power baseline: "
        f"{cue_baseline_s[0]:.3f}..{cue_baseline_s[1]:.3f} s",
        flush=True,
    )
    print(
        f"Conservative Morlet half-support: "
        f"{morlet_half_support_s:.3f} s",
        flush=True,
    )
    print(
        "=" * 78,
        flush=True,
    )

    epochs = create_response_locked_epochs(
        participant=participant,
        metadata=metadata,
        reference_channels=reference_channels,
        epoch_resample_hz=epoch_resample_hz,
        working_tmin=working_tmin,
        working_tmax=working_tmax,
    )

    power, response_times = compute_response_locked_tfr(
        participant=participant,
        epochs=epochs,
        metadata=metadata,
        frequencies_hz=frequencies_hz,
        tfr_decim=tfr_decim,
        time_bin_s=time_bin_s,
        frequency_chunk_size=frequency_chunk_size,
        n_jobs=args.n_jobs,
    )

    # Trial-level timing QC is deliberately saved separately as well as in
    # metadata so the response-window audit can be performed without loading
    # any neural data.
    timing_qc_columns = [
        "trialsnum",
        "cue_sample",
        "response_sample",
        "response_event_type",
        "behavioral_rt_s",
        "event_rt_s",
        "rt_alignment_error_s",
        "cue_time_relative_to_response_s",
        "baseline_response_tmin_s",
        "baseline_response_tmax_s",
        "earliest_center_full_morlet_support_after_cue_s",
    ]

    metadata[
        timing_qc_columns
    ].to_csv(
        paths[
            "timing_qc"
        ],
        index=False,
    )

    atomic_save_numpy(
        paths[
            "power"
        ],
        power,
    )

    atomic_save_numpy(
        paths[
            "times"
        ],
        response_times.astype(
            float
        ),
    )

    atomic_save_numpy(
        paths[
            "freqs"
        ],
        frequencies_hz.astype(
            float
        ),
    )

    metadata.to_csv(
        paths[
            "metadata"
        ],
        index=False,
    )

    paths[
        "channels"
    ].write_text(
        json.dumps(
            list(
                epochs.ch_names
            ),
            indent=2,
        )
    )

    parameters = {
        "signal":
            "response-locked total TFR power",
        "temporal_reference":
            "actual EEG response marker",
        "response_event_types":
            RESPONSE_EVENT_TYPES,
        "cue_locked_reference_directory":
            str(
                cue_locked_reference_paths(
                    participant
                )[
                    "metadata"
                ].parent
            ),
        "trial_selection":
            (
                "exact good trial set inherited from completed cue-locked "
                "decision TFR"
            ),
        "working_epoch_s":
            [
                working_tmin,
                working_tmax,
            ],
        "saved_response_time_range_s":
            [
                RESPONSE_SAVE_TMIN,
                RESPONSE_SAVE_TMAX,
            ],
        "epoch_resample_hz":
            epoch_resample_hz,
        "frequencies_hz":
            frequencies_hz.tolist(),
        "n_cycles_rule":
            "n_cycles = frequency * 0.5",
        "tfr_decim":
            tfr_decim,
        "time_bin_s":
            time_bin_s,
        "frequency_chunk_size":
            frequency_chunk_size,
        "power_transform":
            (
                "log(power) - mean(log(power in each trial's original "
                "cue-relative baseline))"
            ),
        "cue_relative_baseline_s":
            [
                cue_baseline_s[
                    0
                ],
                cue_baseline_s[
                    1
                ],
            ],
        "response_relative_baseline":
            (
                "trial-specific: cue-relative baseline shifted by "
                "-event_rt_s"
            ),
        "voltage_baseline":
            None,
        "morlet_half_support_s_conservative":
            morlet_half_support_s,
        "morlet_support_definition":
            "5 * sigma_t at lowest frequency",
        "rt_alignment_tolerance_s":
            MAX_BEHAVIOURAL_EVENT_RT_DIFFERENCE_S,
        "data_shape_order":
            [
                "trial",
                "channel",
                "frequency",
                "response_time",
            ],
        "important_inference_note":
            (
                "Later response-locked decoding must define the formal "
                "response-time inference range from timing diagnostics before "
                "inspecting neural effects. Response-locking does not itself "
                "statistically control RT."
            ),
    }

    paths[
        "params"
    ].write_text(
        json.dumps(
            parameters,
            indent=2,
        )
    )

    event_rt = metadata[
        "event_rt_s"
    ].to_numpy(
        float
    )

    safe_start_per_trial = metadata[
        "earliest_center_full_morlet_support_after_cue_s"
    ].to_numpy(
        float
    )

    summary = {
        "participant":
            participant,
        "n_response_tfr_trials":
            int(
                len(
                    metadata
                )
            ),
        "n_channels":
            int(
                len(
                    epochs.ch_names
                )
            ),
        "n_frequencies":
            int(
                len(
                    frequencies_hz
                )
            ),
        "n_response_time_bins":
            int(
                len(
                    response_times
                )
            ),
        "event_rt_min_s":
            float(
                np.min(
                    event_rt
                )
            ),
        "event_rt_p01_s":
            float(
                np.quantile(
                    event_rt,
                    0.01,
                )
            ),
        "event_rt_p05_s":
            float(
                np.quantile(
                    event_rt,
                    0.05,
                )
            ),
        "event_rt_median_s":
            float(
                np.median(
                    event_rt
                )
            ),
        "event_rt_p95_s":
            float(
                np.quantile(
                    event_rt,
                    0.95,
                )
            ),
        "event_rt_max_s":
            float(
                np.max(
                    event_rt
                )
            ),
        "max_abs_behavior_event_rt_difference_s":
            float(
                np.max(
                    np.abs(
                        metadata[
                            "rt_alignment_error_s"
                        ].to_numpy(
                            float
                        )
                    )
                )
            ),
        "earliest_response_center_full_support_postcue_all_trials_s":
            float(
                np.max(
                    safe_start_per_trial
                )
            ),
        "earliest_response_center_full_support_postcue_95pct_trials_s":
            float(
                np.quantile(
                    safe_start_per_trial,
                    0.95,
                )
            ),
        "response_save_tmin_s":
            RESPONSE_SAVE_TMIN,
        "response_save_tmax_s":
            RESPONSE_SAVE_TMAX,
        "working_tmin_s":
            working_tmin,
        "working_tmax_s":
            working_tmax,
    }

    paths[
        "summary"
    ].write_text(
        json.dumps(
            summary,
            indent=2,
        )
    )

    print(
        f"[{participant}] saved response-locked TFR "
        f"shape={power.shape}",
        flush=True,
    )
    print(
        f"[{participant}] DONE",
        flush=True,
    )


if __name__ == "__main__":
    main()
