"""
Single-trial cue-locked TFR preparation for the PainReward passive task.

This is the spectral-preparation stage only. It does NOT perform decoding,
statistics, band selection, or any outcome-driven feature selection.

Input
-----
1. Cleaned passive raw FIF from eeg/03_eeg_preprocess.py:
       derivatives/<sub>/eeg/<sub>_passive_cleaned-raw.fif
2. Passive BIDS events TSV, used for exact EEG sample identity:
       <sub>/eeg/<sub>_task-passive_events.tsv
3. Passive behaviour TSV, merged by chronological trial number so condition,
   1-5 level and block are preserved for later MVPA.

The TFR branch is intentionally self-contained and does NOT require
eeg/04_eeg_erp_prep.py to have been run. To preserve the existing ERP
artifact policy, this script recreates the passive `badtrial` flag from a
30-Hz-low-passed copy of the cleaned raw using the same -0.2..1.0 s window
and 150-uV peak-to-peak rejection threshold as 04_eeg_erp_prep.py. The
actual TFR is always computed from the original cleaned 0.1-100 Hz raw, not
from that low-passed copy.

Core TFR choices
----------------
- 4-45 Hz in 1-Hz steps
- Morlet n_cycles = frequency / 2
- long working epoch: -2 to +2 s (no voltage baseline)
- resample epochs to 256 Hz before TFR
- compute power in small frequency chunks for memory safety
- TFR convolution output decimated by 2 (128-Hz temporal grid)
- mean-log-ratio baseline per trial x channel x frequency:
      log(power) - mean(log(power_baseline))
- only AFTER baseline normalisation, average into 20-ms time bins
- save -0.8 to +1.5 s
- float32 output

Output per participant
----------------------
derivatives/<sub>/eeg/tfr/
    <sub>_passive_cue_power.npy       # trials x channels x frequencies x time bins
    <sub>_passive_cue_times.npy       # seconds, bin centres
    <sub>_passive_cue_freqs.npy       # Hz
    <sub>_passive_cue_metadata.csv    # good trials only; retains original trialsnum
    <sub>_passive_cue_channels.json
    <sub>_passive_cue_params.json
    <sub>_passive_cue_prep_summary.json

The .npy format is deliberate: it is transparent, version-stable, and can be
memory-mapped later during MVPA without loading a large HDF5 object at once.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from mne.time_frequency import tfr_array_morlet


mne.set_log_level("ERROR")

BASE_PATH = Path(os.getenv("basepath", Path(__file__).parent.parent.parent))

# TFR parameters. Keep baseline synchronized with 04a_eeg_tfr_timing_audit.py.
FREQS = np.arange(4.0, 46.0, 1.0)
N_CYCLES_FACTOR = 0.5
EPOCH_TMIN = -2.0
EPOCH_TMAX = 2.0
EPOCH_SFREQ = 256.0
TFR_DECIM = 2                    # 256 -> 128 Hz after convolution; still safe for <=45 Hz
BASELINE = (-0.7, -0.5)
SAVE_TMIN = -0.8
SAVE_TMAX = 1.5
TIME_BIN_S = 0.020               # 20-ms power-envelope bins
FREQ_CHUNK_SIZE = 6
LOG_FLOOR = np.finfo(np.float64).tiny

CHANS_TO_DROP = ["HEOGL", "HEOGR", "VEOGL", "STI 014", "Status"]

# Match the passive ERP rejection convention in 04_eeg_erp_prep.py.
ERP_REJECT = dict(eeg=150e-6)
ERP_REJECT_LP_HZ = 30.0
ERP_REJECT_TMIN = -0.2
ERP_REJECT_TMAX = 1.0
ERP_REJECT_BASELINE = (-0.2, 0.0)
PASSIVE_TRIAL_TYPES = [f"rew{i}" for i in range(1, 6)] + [f"shk{i}" for i in range(1, 6)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare single-trial passive cue TFR power.")
    parser.add_argument("--subject", required=True, help="Participant id, e.g. sub-001")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing TFR outputs for this participant.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=int(os.getenv("NJOBS", os.getenv("SLURM_CPUS_PER_TASK", "1"))),
        help="Parallel jobs passed to MNE Morlet computation.",
    )
    return parser.parse_args()


def passive_beh_path(participant: str) -> Path:
    return BASE_PATH / participant / "eeg" / f"{participant}_task-passive_beh.tsv"


def passive_raw_path(participant: str) -> Path:
    return BASE_PATH / "derivatives" / participant / "eeg" / f"{participant}_passive_cleaned-raw.fif"


def passive_events_path(participant: str) -> Path:
    canon = BASE_PATH / participant / "eeg" / f"{participant}_task-passive_events.tsv"
    if canon.exists():
        return canon
    matches = list((BASE_PATH / participant / "eeg").glob("*task-passive*events.tsv"))
    if not matches:
        raise FileNotFoundError(
            f"{participant}: no passive events.tsv found in {BASE_PATH / participant / 'eeg'}"
        )
    if len(matches) > 1:
        raise RuntimeError(f"{participant}: multiple passive events.tsv files found: {matches}")
    return matches[0]


def out_dir(participant: str) -> Path:
    return BASE_PATH / "derivatives" / participant / "eeg" / "tfr"


def level_to_1_5(values: pd.Series) -> pd.Series:
    vals = pd.to_numeric(values, errors="coerce")
    if vals.isna().all():
        raise RuntimeError("Passive level column is entirely non-numeric")
    if vals.max(skipna=True) > 5:
        vals = vals / 20.0
    return vals.astype(float)


def load_passive_beh(participant: str) -> pd.DataFrame:
    path = passive_beh_path(participant)
    if not path.exists():
        raise FileNotFoundError(f"{participant}: missing passive behaviour file: {path}")

    beh = pd.read_csv(path, sep="\t")
    required = ["fixcross.started", "cue.started", "condition", "level", "blocks.thisRepN"]
    missing = [c for c in required if c not in beh.columns]
    if missing:
        raise KeyError(f"{participant}: passive behaviour missing columns: {missing}")

    beh = beh[~beh["fixcross.started"].isna()].copy().reset_index(drop=True)
    beh["trialsnum"] = np.arange(1, len(beh) + 1, dtype=int)
    beh["condition"] = beh["condition"].astype(str).str.lower().str.strip()
    beh["level_1_5"] = level_to_1_5(beh["level"])
    beh["fixation_duration_s"] = (
        pd.to_numeric(beh["cue.started"], errors="coerce")
        - pd.to_numeric(beh["fixcross.started"], errors="coerce")
    )

    # Exactly the two cue conditions analysed in the existing 07b scripts.
    beh = beh[beh["condition"].isin(["p", "m"])].copy()
    return beh[
        [
            "trialsnum",
            "condition",
            "level_1_5",
            "blocks.thisRepN",
            "fixation_duration_s",
        ]
    ]


def load_and_align_metadata(participant: str) -> pd.DataFrame:
    """Build passive trial metadata directly from events.tsv + behaviour.tsv.

    Event samples define EEG timing. Chronological trial number aligns the 200
    passive cue events with the 200 PsychoPy trial rows. Event-derived condition
    and level are cross-checked against behaviour so a silent one-row shift
    cannot pass unnoticed.
    """
    ev_path = passive_events_path(participant)
    events = pd.read_csv(ev_path, sep="\t")
    required = ["sample", "trial_type"]
    missing = [c for c in required if c not in events.columns]
    if missing:
        raise KeyError(f"{participant}: passive events.tsv missing columns: {missing}")

    md = events[events["trial_type"].isin(PASSIVE_TRIAL_TYPES)].copy()
    md = md.sort_values("sample").reset_index(drop=True)
    if md.empty:
        raise RuntimeError(f"{participant}: no rew1-5/shk1-5 cue events found in {ev_path}")

    md["sample"] = pd.to_numeric(md["sample"], errors="coerce")
    if md["sample"].isna().any():
        raise RuntimeError(f"{participant}: non-numeric passive cue sample(s) in events.tsv")
    md["sample"] = md["sample"].astype(int)
    if md["sample"].duplicated().any():
        raise RuntimeError(f"{participant}: duplicate passive cue sample indices")

    md["trialsnum"] = np.arange(1, len(md) + 1, dtype=int)
    md["event_condition"] = np.where(
        md["trial_type"].astype(str).str.startswith("shk"), "p", "m"
    )
    md["event_level_1_5"] = (
        md["trial_type"].astype(str).str.extract(r"(\d+)$", expand=False).astype(int)
    )

    beh = load_passive_beh(participant)
    if len(md) != len(beh):
        raise RuntimeError(
            f"{participant}: passive cue count mismatch: events={len(md)}, behaviour={len(beh)}"
        )

    merged = md[[
        "trialsnum", "sample", "trial_type", "event_condition", "event_level_1_5"
    ]].merge(beh, on="trialsnum", how="left", validate="1:1")

    if merged[["condition", "level_1_5", "blocks.thisRepN", "fixation_duration_s"]].isna().any().any():
        raise RuntimeError(f"{participant}: passive behaviour merge left unlabeled event trials")

    cond_bad = merged["event_condition"].astype(str) != merged["condition"].astype(str)
    level_bad = ~np.isclose(
        merged["event_level_1_5"].astype(float), merged["level_1_5"].astype(float)
    )
    if cond_bad.any() or level_bad.any():
        cols = [
            "trialsnum", "trial_type", "event_condition", "event_level_1_5",
            "condition", "level_1_5"
        ]
        bad = merged.loc[cond_bad | level_bad, cols]
        raise RuntimeError(
            f"{participant}: events.tsv and behaviour.tsv disagree on passive trial identity.\n"
            + bad.head(20).to_string(index=False)
        )

    unsafe = merged["fixation_duration_s"].astype(float) < abs(BASELINE[0])
    if unsafe.any():
        bad_rows = merged.loc[unsafe, ["trialsnum", "fixation_duration_s"]]
        raise RuntimeError(
            f"{participant}: planned TFR baseline {BASELINE} is not fully inside fixation "
            f"for {int(unsafe.sum())} trial(s):\n{bad_rows.head(20).to_string(index=False)}"
        )

    return merged


def add_erp_style_badtrial_flags(
    participant: str, raw: mne.io.BaseRaw, metadata: pd.DataFrame
) -> pd.DataFrame:
    """Recreate the passive ERP `badtrial` flag without needing ERP files.

    The proxy epochs exactly mirror the artifact-rejection part of
    04_eeg_erp_prep.py: 30-Hz LP filter, -0.2..1.0 s cue epochs, baseline
    (-0.2, 0), and 150-uV peak-to-peak rejection. These epochs are used ONLY
    to flag trials. TFR power is computed from the untouched cleaned raw.
    """
    reject_raw = raw.copy().filter(
        l_freq=None, h_freq=ERP_REJECT_LP_HZ, method="fir", verbose="ERROR"
    )
    events = np.column_stack([
        metadata["sample"].to_numpy(dtype=int),
        np.zeros(len(metadata), dtype=int),
        np.ones(len(metadata), dtype=int),
    ])
    proxy = mne.Epochs(
        reject_raw,
        events=events,
        event_id={"cue": 1},
        tmin=ERP_REJECT_TMIN,
        tmax=ERP_REJECT_TMAX,
        baseline=ERP_REJECT_BASELINE,
        metadata=metadata.copy(),
        preload=True,
        reject=None,
        reject_by_annotation=True,
        verbose="ERROR",
    )
    if len(proxy) != len(metadata):
        raise RuntimeError(
            f"{participant}: ERP-style rejection proxy retained {len(proxy)}/{len(metadata)} "
            "trials before peak-to-peak rejection (likely annotation/data-boundary drop)."
        )

    tmp = proxy.copy().drop_bad(reject=ERP_REJECT)
    bad_flags = np.asarray([1 if len(reason) > 0 else 0 for reason in tmp.drop_log], dtype=int)
    if len(bad_flags) != len(metadata):
        raise RuntimeError(f"{participant}: badtrial flag length mismatch")

    out = metadata.copy().reset_index(drop=True)
    out["badtrial"] = bad_flags
    return out


def build_long_epochs(
    participant: str, raw: mne.io.BaseRaw, metadata: pd.DataFrame
) -> mne.Epochs:
    # Use only the already-defined good trials. We retain original `trialsnum`
    # and sample identity in metadata, so later MVPA can align back exactly.
    good = metadata[metadata["badtrial"] == 0].copy().reset_index(drop=True)
    if good.empty:
        raise RuntimeError(f"{participant}: no good passive trials after ERP-style badtrial flag")

    events = np.column_stack(
        [
            good["sample"].to_numpy(dtype=int),
            np.zeros(len(good), dtype=int),
            np.ones(len(good), dtype=int),
        ]
    )

    epochs = mne.Epochs(
        raw,
        events=events,
        event_id={"cue": 1},
        tmin=EPOCH_TMIN,
        tmax=EPOCH_TMAX,
        baseline=None,
        metadata=good,
        preload=True,
        reject=None,
        reject_by_annotation=True,
        verbose="ERROR",
    )

    if len(epochs) != len(good):
        raise RuntimeError(
            f"{participant}: long TFR epoching retained {len(epochs)}/{len(good)} good trials. "
            "A trial was dropped because the long epoch crossed a data boundary/annotation; "
            "inspect before proceeding."
        )

    epochs.resample(EPOCH_SFREQ, npad="auto", verbose="ERROR")
    return epochs


def make_time_bins(times: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    edges = np.arange(SAVE_TMIN, SAVE_TMAX + TIME_BIN_S * 0.5, TIME_BIN_S)
    if edges[-1] < SAVE_TMAX - 1e-9:
        edges = np.append(edges, SAVE_TMAX)
    if abs(edges[-1] - SAVE_TMAX) > 1e-9:
        # Enforce an exact final edge so output dimensions are deterministic.
        edges = np.append(edges[edges < SAVE_TMAX], SAVE_TMAX)

    centers = []
    masks = []
    for i in range(len(edges) - 1):
        lo, hi = float(edges[i]), float(edges[i + 1])
        if i == len(edges) - 2:
            mask = np.where((times >= lo - 1e-12) & (times <= hi + 1e-12))[0]
        else:
            mask = np.where((times >= lo - 1e-12) & (times < hi - 1e-12))[0]
        if len(mask) == 0:
            raise RuntimeError(
                f"No TFR samples in requested time bin [{lo:.3f}, {hi:.3f}] s. "
                "Check resampling/decimation settings."
            )
        centers.append((lo + hi) / 2.0)
        masks.append(mask)
    return np.asarray(centers, dtype=float), masks


def write_npy_atomic(path: Path, array: np.ndarray) -> None:
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "wb") as f:
        np.save(f, array, allow_pickle=False)
    os.replace(tmp, path)


def main() -> None:
    args = parse_args()
    participant = args.subject

    participant_dir = BASE_PATH / participant
    if not participant_dir.exists():
        raise FileNotFoundError(f"Unknown participant directory: {participant_dir}")

    out = out_dir(participant)
    out.mkdir(parents=True, exist_ok=True)

    power_path = out / f"{participant}_passive_cue_power.npy"
    times_path = out / f"{participant}_passive_cue_times.npy"
    freqs_path = out / f"{participant}_passive_cue_freqs.npy"
    meta_path = out / f"{participant}_passive_cue_metadata.csv"
    channels_path = out / f"{participant}_passive_cue_channels.json"
    params_path = out / f"{participant}_passive_cue_params.json"
    summary_path = out / f"{participant}_passive_cue_prep_summary.json"

    outputs = [power_path, times_path, freqs_path, meta_path, channels_path, params_path, summary_path]
    existing = [p for p in outputs if p.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            f"{participant}: TFR outputs already exist. Use --overwrite to recompute.\n"
            + "\n".join(str(p) for p in existing)
        )

    metadata_all = load_and_align_metadata(participant)
    n_input_trials = len(metadata_all)

    raw_path = passive_raw_path(participant)
    if not raw_path.exists():
        raise FileNotFoundError(
            f"{participant}: cleaned passive raw missing: {raw_path}\n"
            "Run eeg/03_eeg_preprocess.py first."
        )
    print(f"[{participant}] loading cleaned passive raw")
    raw = mne.io.read_raw_fif(str(raw_path), preload=True, verbose="ERROR")
    to_drop = [c for c in CHANS_TO_DROP if c in raw.ch_names]
    if to_drop:
        raw.drop_channels(to_drop)
    raw.pick("eeg", exclude=[])

    print(f"[{participant}] recreating ERP-style badtrial flags")
    metadata_all = add_erp_style_badtrial_flags(participant, raw, metadata_all)
    n_badtrial = int((metadata_all["badtrial"] == 1).sum())

    print(f"[{participant}] making long cue epochs")
    epochs = build_long_epochs(participant, raw, metadata_all)
    metadata = epochs.metadata.reset_index(drop=True).copy()
    ch_names = list(epochs.ch_names)

    data = epochs.get_data(copy=False).astype(np.float64, copy=False)
    sfreq = float(epochs.info["sfreq"])
    if abs(sfreq - EPOCH_SFREQ) > 1e-6:
        raise RuntimeError(f"{participant}: expected {EPOCH_SFREQ} Hz after resampling, got {sfreq}")
    if not np.isfinite(data).all():
        raise RuntimeError(f"{participant}: non-finite EEG values before TFR")

    # tfr_array_morlet decimation is a sample stride after convolution.
    # At 256 Hz and decim=2, the resulting 128-Hz grid safely represents
    # our highest 45-Hz narrow-band analytic signal.
    tfr_times = epochs.times[::TFR_DECIM]
    baseline_mask = (tfr_times >= BASELINE[0] - 1e-12) & (tfr_times <= BASELINE[1] + 1e-12)
    if baseline_mask.sum() < 2:
        raise RuntimeError(f"{participant}: fewer than two TFR samples in baseline {BASELINE}")

    bin_centers, bin_masks = make_time_bins(tfr_times)

    n_trials, n_channels, _ = data.shape
    n_freqs = len(FREQS)
    n_bins = len(bin_centers)
    final_power = np.empty((n_trials, n_channels, n_freqs, n_bins), dtype=np.float32)

    print(
        f"[{participant}] Morlet power: trials={n_trials}, channels={n_channels}, "
        f"freqs={n_freqs}, bins={n_bins}, n_jobs={args.n_jobs}"
    )

    for start in range(0, n_freqs, FREQ_CHUNK_SIZE):
        stop = min(start + FREQ_CHUNK_SIZE, n_freqs)
        freqs_chunk = FREQS[start:stop]
        n_cycles_chunk = freqs_chunk * N_CYCLES_FACTOR
        print(f"[{participant}]   frequencies {freqs_chunk[0]:.0f}-{freqs_chunk[-1]:.0f} Hz")

        power = tfr_array_morlet(
            data,
            sfreq=sfreq,
            freqs=freqs_chunk,
            n_cycles=n_cycles_chunk,
            output="power",
            use_fft=True,
            zero_mean=True,
            decim=TFR_DECIM,
            n_jobs=max(1, int(args.n_jobs)),
            verbose=False,
        )

        if power.shape[-1] != len(tfr_times):
            raise RuntimeError(
                f"{participant}: TFR time dimension mismatch {power.shape[-1]} != {len(tfr_times)}"
            )
        if not np.isfinite(power).all():
            raise RuntimeError(f"{participant}: non-finite raw Morlet power")

        # Mean-log-ratio normalisation, separately for every trial/channel/frequency.
        # This is equivalent to log(power) - mean(log(power_baseline)).
        log_power = np.log(np.maximum(power, LOG_FLOOR))
        baseline_mean_log = log_power[..., baseline_mask].mean(axis=-1, keepdims=True)
        norm = log_power - baseline_mean_log

        for b, mask in enumerate(bin_masks):
            final_power[:, :, start:stop, b] = norm[..., mask].mean(axis=-1).astype(np.float32)

        del power, log_power, baseline_mean_log, norm

    if not np.isfinite(final_power).all():
        raise RuntimeError(f"{participant}: final binned TFR contains non-finite values")

    # Save only after the complete participant finished successfully.
    write_npy_atomic(power_path, final_power)
    write_npy_atomic(times_path, bin_centers.astype(np.float64))
    write_npy_atomic(freqs_path, FREQS.astype(np.float64))
    metadata.to_csv(meta_path, index=False)

    with open(channels_path, "w", encoding="utf-8") as f:
        json.dump(ch_names, f, indent=2)

    params = {
        "input_raw": str(passive_raw_path(participant)),
        "events_source": str(passive_events_path(participant)),
        "behaviour_source": str(passive_beh_path(participant)),
        "working_epoch_s": [EPOCH_TMIN, EPOCH_TMAX],
        "epoch_resample_hz": EPOCH_SFREQ,
        "frequencies_hz": FREQS.tolist(),
        "n_cycles_rule": "n_cycles = frequency * 0.5",
        "tfr_decim": TFR_DECIM,
        "tfr_grid_hz_after_decim": EPOCH_SFREQ / TFR_DECIM,
        "baseline_s": list(BASELINE),
        "baseline_normalisation": "meanlogratio: log(power) - mean(log(power_baseline))",
        "saved_time_range_s": [SAVE_TMIN, SAVE_TMAX],
        "time_bin_s": TIME_BIN_S,
        "frequency_chunk_size": FREQ_CHUNK_SIZE,
        "dtype": "float32",
        "data_shape_order": ["trial", "channel", "frequency", "time"],
        "badtrial_policy": (
            "recreated internally to match 04_eeg_erp_prep.py: 30-Hz LP, "
            "-0.2..1.0 s cue epoch, baseline -0.2..0, 150-uV peak-to-peak; "
            "bad trials excluded before TFR"
        ),
    }
    with open(params_path, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)

    summary = {
        "participant": participant,
        "n_input_passive_trials": int(n_input_trials),
        "n_existing_badtrial_excluded": int(n_badtrial),
        "n_tfr_trials": int(n_trials),
        "n_channels": int(n_channels),
        "n_frequencies": int(n_freqs),
        "n_time_bins": int(n_bins),
        "power_file_gib": float(power_path.stat().st_size / (1024 ** 3)),
        "condition_counts": {str(k): int(v) for k, v in metadata["condition"].value_counts().sort_index().items()},
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[{participant}] DONE")
    print(f"[{participant}] power shape = {final_power.shape}")
    print(f"[{participant}] saved = {power_path}")


if __name__ == "__main__":
    main()
