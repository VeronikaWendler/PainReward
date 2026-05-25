"""
ERP preparation — epoch, baseline-correct, and save single-trial epochs.

Runs three modes in sequence:
  1. decision_cue   — cue-locked (off+) for the decision phase
  2. passive_cue    — cue-locked (rew1-5, shk1-5) for the passive phase
  3. decision_resp  — response-locked (res+/res-/resm) for the decision phase

Outputs per-participant fif files and a pooled metadata CSV per mode.

Section 3 (RP → HDDM): extracts mean RP amplitude from the response-locked
epochs and merges it with the behavioural CSV to produce an HDDM-ready file.

Authors: Michel-Pierre Coll, Veronika Wendler
"""

import os
import mne
from mne.report import Report
from mne.viz import plot_evoked_joint as pej
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from pathlib import Path

warnings.simplefilter(action="ignore", category=FutureWarning)

# ──────────────────────────────────────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────────────────────────────────────
# Script lives in eeg/ → parent.parent is the project root
basepath = Path(os.getenv("basepath", Path(__file__).parent.parent.parent))
outpath = basepath / "derivatives"
outpath.mkdir(parents=True, exist_ok=True)

# Section 3 (RP → HDDM): behavioural CSV produced by 02_sv_modelling.py.
# Override via BEHAV_FILE env var if your layout differs.
BEHAV_FILE = Path(os.getenv(
    "BEHAV_FILE",
    outpath / "behav" / "behav_with_exclusion_sv_modeling.csv",
))

# ──────────────────────────────────────────────────────────────────────────────
# PARAMETERS (shared across modes)
# ──────────────────────────────────────────────────────────────────────────────
ERPREJECT = dict(eeg=150e-6)
LP_FILTER = 30          # Hz
HP_FILTER = None        # Hz (no high-pass for ERPs)
FILTER_METHOD = "fir"

# Channels to drop from raw before epoching
CHANS_TO_DROP = ["HEOGL", "HEOGR", "VEOGL", "STI 014", "Status"]

# RP extraction (Section 3) — response-locked window and ROI
RP_CHANNELS = ["Cz", "CPz", "CP1", "CP2", "C1", "C2", "FC1", "FC2"]
RP_TMIN = -0.5   # seconds relative to response
RP_TMAX = -0.1

# Channels to plot in the report
CHANS_REPORT = ["Fz", "POz", "Cz", "CPz", "Pz", "Oz"]

# ──────────────────────────────────────────────────────────────────────────────
# MODE CONFIGURATIONS
# Each mode dict fully specifies how to epoch and what to save.
# ──────────────────────────────────────────────────────────────────────────────
MODE_CONFIGS = {
    "decision_cue": {
        "raw_suffix":    "_decision_cleaned-raw.fif",
        "task":          "decision",
        "cue_events":    ["off+"],
        "lock":          "cue",
        "tmin":          -0.2,
        "tmax":          1.5,
        "baseline":      (-0.2, 0),
        "outdir_name":   "erps_decision",
        "epo_fname_tpl": "{p}_decision_cues_singletrials-epo.fif",
        "ave_fname_tpl": "{p}_decision_{tag}_ave.fif",
        "report_tpl":    "{p}_decision_cue_erps_report.html",
        "reject_csv":    "erps/decision_cue_erps_rejectionstats.csv",
        "meta_csv":      "erps/decision_erpsmeta_cue.csv",
        "amp_lat":       [[0.4, 0.8]],
        "report_section": "ERPs for cue off+",
    },
    "passive_cue": {
        "raw_suffix":    "_passive_cleaned-raw.fif",
        "task":          "passive",
        "cue_events":    [
            "rew1", "rew2", "rew3", "rew4", "rew5",
            "shk1", "shk2", "shk3", "shk4", "shk5",
        ],
        "lock":          "cue",
        "tmin":          -0.2,
        "tmax":          1.0,
        "baseline":      (-0.2, 0),
        "outdir_name":   "erps_passive",
        "epo_fname_tpl": "{p}_passive_cues_singletrials-epo.fif",
        "ave_fname_tpl": "{p}_passive_{tag}_ave.fif",
        "report_tpl":    "{p}_passive_erps_report.html",
        "reject_csv":    "erps/passive_erps_rejectionstats.csv",
        "meta_csv":      "erps/passive_erpsmeta.csv",
        "amp_lat":       [[0.4, 0.8]],
        "report_section": "ERPs for passive cues",
    },
    "decision_resp": {
        "raw_suffix":    "_decision_cleaned-raw.fif",
        "task":          "decision",
        "cue_events":    ["res+", "res-", "resm"],
        "lock":          "response",
        "tmin":          -0.8,
        "tmax":          0.2,
        "baseline":      (-0.8, -0.7),   # pre-movement baseline
        "outdir_name":   "erps_decisionresp",
        "epo_fname_tpl": "{p}_decision_resp_singletrials-epo.fif",
        "ave_fname_tpl": "{p}_decision_resp_{tag}_ave.fif",
        "report_tpl":    "{p}_decision_resp_erps_report.html",
        "reject_csv":    "erps/decision_resp_erps_rejectionstats.csv",
        "meta_csv":      "erps/decision_resp_erpsmeta.csv",
        "amp_lat":       [[-0.5, -0.1]],
        "report_section": "ERPs for responses",
    },
}

# Create erps folder in derivatives if it doesn't exist
erps_outdir = outpath / "erps"
erps_outdir.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def safe_tag(s):
    """Make an event name safe for use in file names (replace + with plus)."""
    return s.replace("+", "plus").replace("-", "minus").replace(" ", "_")


def find_events_tsv(basepath: Path, participant: str, task: str) -> Path:
    """Return path to a BIDS events TSV without requiring pybids.

    Looks for:  basepath / participant / eeg / {participant}_task-{task}_events.tsv
    Falls back to a glob if the canonical name isn't found.
    """
    canon = basepath / participant / "eeg" / f"{participant}_task-{task}_events.tsv"
    if canon.exists():
        return canon
    matches = list((basepath / participant / "eeg").glob(f"*task-{task}*events.tsv"))
    if not matches:
        raise FileNotFoundError(
            f"{participant}: no events.tsv found for task='{task}' "
            f"in {basepath / participant / 'eeg'}"
        )
    if len(matches) > 1:
        raise RuntimeError(
            f"{participant}: multiple events.tsv found for task='{task}': {matches}\n"
            f"  Resolve the ambiguity before running."
        )
    return matches[0]


def build_event_array(events_df: pd.DataFrame, cue_events: list, participant: str):
    """Filter events_df to cue_events and return (events_array, events_id).

    Returns
    -------
    events_epoch : np.ndarray, shape (n_events, 3)
    events_id    : dict {str: int}
    """
    present = set(events_df["trial_type"].dropna().astype(str).unique())
    found = [ev for ev in cue_events if ev in present]
    if not found:
        raise RuntimeError(
            f"{participant}: none of the expected events found.\n"
            f"  Expected any of: {cue_events}\n"
            f"  Found (first 40): {sorted(list(present))[:40]}"
        )
    sub = events_df[events_df["trial_type"].isin(found)].copy()
    sub = sub.sort_values("sample").reset_index(drop=True)
    events_id = {ev: i + 1 for i, ev in enumerate(found)}
    sub["cue_num"] = sub["trial_type"].map(events_id).astype(int)
    sub["empty"] = 0
    arr = np.asarray(sub[["sample", "empty", "cue_num"]], dtype=int)
    return arr, events_id, sub


def extract_rp_amplitude(epo_path: Path, participant: str) -> pd.DataFrame:
    """Load response-locked epochs and return a per-trial RP amplitude table.

    Mean amplitude is computed over RP_CHANNELS × [RP_TMIN, RP_TMAX].
    Bad trials are kept and flagged in `badtrial` (no row dropping).
    """
    epochs = mne.read_epochs(str(epo_path), preload=True, verbose="ERROR")

    missing = [c for c in RP_CHANNELS if c not in epochs.ch_names]
    if missing:
        raise ValueError(f"{participant}: RP channels missing: {missing}")

    rp_raw = (
        epochs.copy()
        .pick(RP_CHANNELS)
        .crop(tmin=RP_TMIN, tmax=RP_TMAX)
        .get_data()
        .mean(axis=(1, 2))
    )

    meta = (
        epochs.metadata.copy()
        if epochs.metadata is not None
        else pd.DataFrame(index=range(len(epochs)))
    )
    meta = meta.reset_index(drop=True)
    meta["rp_raw"] = rp_raw
    meta["participant"] = participant

    if "sample" in meta.columns:
        meta = meta.sort_values("sample").reset_index(drop=True)

    meta["trial_seq"] = np.arange(1, len(meta) + 1)

    if "badtrial" not in meta.columns:
        meta["badtrial"] = [1 if len(x) > 0 else 0 for x in epochs.drop_log]
    meta["badtrial"] = pd.to_numeric(meta["badtrial"], errors="coerce").fillna(0).astype(int)

    return meta


def zscore_within_subject(df: pd.DataFrame, value_col: str, subj_col: str, out_col: str) -> pd.DataFrame:
    """Add a within-subject z-score column to df."""
    def _z(x):
        sd = x.std(ddof=0)
        return pd.Series(np.nan, index=x.index) if (pd.isna(sd) or sd == 0) else (x - x.mean()) / sd

    df[out_col] = df.groupby(subj_col)[value_col].transform(_z)
    return df


def merge_rp_into_behav(behav: pd.DataFrame, rp_table: pd.DataFrame):
    """Align RP trial table with the behavioural CSV and merge rp_raw / rp_z / badtrial.

    behav is assumed to contain only decision trials (one row per trial).
    Trials are matched by participant + sequential trial number within participant,
    sorted by blocks.thisRepN then trials.thisN.

    Returns
    -------
    merged : pd.DataFrame  — behav extended with rp_raw, rp_z, badtrial
    diag   : pd.DataFrame  — per-participant merge diagnostics
    """
    sort_cols = ["participant"]
    if "blocks.thisRepN" in behav.columns:
        sort_cols.append("blocks.thisRepN")
    if "trials.thisN" in behav.columns:
        sort_cols.append("trials.thisN")

    behav = behav.sort_values(sort_cols).copy()
    behav["trial_seq"] = behav.groupby("participant").cumcount() + 1

    merged = behav.merge(
        rp_table[["participant", "trial_seq", "rp_raw", "rp_z", "badtrial"]],
        on=["participant", "trial_seq"],
        how="left",
        validate="one_to_one",
    )

    diag = (
        merged.groupby("participant")
        .agg(
            n_trials=("trial_seq", "size"),
            n_matched=("badtrial", lambda x: x.notna().sum()),
            n_badtrial=("badtrial", lambda x: (pd.to_numeric(x, errors="coerce").fillna(0) == 1).sum()),
        )
        .assign(
            n_goodtrial=lambda d: d["n_trials"] - d["n_badtrial"],
            merge_rate=lambda d: d["n_matched"] / d["n_trials"],
        )
        .reset_index()
    )

    return merged, diag


def average_time_win_strials(strials, chans_to_average, amp_lat):
    """Extract mean amplitude in fixed windows and add to epoch metadata.

    Parameters
    ----------
    strials          : mne.Epochs  — epochs with metadata
    chans_to_average : list of lists  — e.g. [['Fz'], ['Cz']]
    amp_lat          : list of [tmin, tmax] pairs

    Returns
    -------
    strials : mne.Epochs  — metadata updated in-place
    """
    for c in chans_to_average:
        for a in amp_lat:
            amp_epo = strials.copy().crop(tmin=a[0], tmax=a[1])
            amp_epo.pick_channels(c)
            all_amps = [np.mean(data) for data in amp_epo.get_data()]
            all_amps = np.array(all_amps)
            all_amps = (all_amps - all_amps.mean()) / all_amps.std()
            col = "amp_" + "_".join(c) + "_" + str(a[0]) + "-" + str(a[1])
            strials.metadata[col] = all_amps
    return strials


# ──────────────────────────────────────────────────────────────────────────────
# PARTICIPANT LIST
# ──────────────────────────────────────────────────────────────────────────────
part = sorted([p for p in os.listdir(basepath) if p.startswith("sub-")])

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 1 — Epoch, save evokeds and single-trial epochs
# ──────────────────────────────────────────────────────────────────────────────
for mode_name, cfg in MODE_CONFIGS.items():
    print(f"\n{'='*60}")
    print(f"MODE: {mode_name}")
    print(f"{'='*60}")

    reject_stats = pd.DataFrame({
        "part":               part,
        "n_target":           0,
        "n_kept":             0,
        "perc_removed_cues":  9999.0,
    })

    for p in part:
        print(f"\n  {p} [{mode_name}]")

        indir  = outpath / p / "eeg"
        outdir = outpath / p / "eeg" / cfg["outdir_name"]
        outdir.mkdir(parents=True, exist_ok=True)

        # --- Load raw ---
        raw_path = indir / (p + cfg["raw_suffix"])
        if not raw_path.exists():
            raise FileNotFoundError(
                f"{p} [{mode_name}]: cleaned raw not found: {raw_path}"
            )

        raw = mne.io.read_raw_fif(raw_path, preload=True)

        # --- Load events TSV ---
        events_fname = find_events_tsv(basepath, p, cfg["task"])
        events = pd.read_csv(events_fname, sep="\t")
        events_c = events[events["trial_type"].notna()].copy()

        # --- Drop EOG/trigger channels ---
        to_drop = [c for c in CHANS_TO_DROP if c in raw.ch_names]
        if to_drop:
            raw.drop_channels(to_drop)

        # --- Filter ---
        raw.filter(l_freq=HP_FILTER, h_freq=LP_FILTER, method=FILTER_METHOD)

        # --- Build event array ---
        events_arr, events_id, events_sub = build_event_array(
            events_c, cfg["cue_events"], p
        )

        n_target = len(events_arr)
        reject_stats.loc[reject_stats["part"] == p, "n_target"] = n_target

        # --- Epoch (averaged evokeds) ---
        erp_cues = mne.Epochs(
            raw,
            events=events_arr,
            event_id=events_id,
            tmin=cfg["tmin"],
            tmax=cfg["tmax"],
            baseline=cfg["baseline"],
            preload=True,
            verbose=False,
            reject=ERPREJECT,
        )

        reject_stats.loc[reject_stats["part"] == p, "n_kept"] = len(erp_cues)
        reject_stats.loc[reject_stats["part"] == p, "perc_removed_cues"] = (
            (n_target - len(erp_cues)) / n_target * 100.0
        )

        # --- MNE Report ---
        report = Report(verbose=False, subject=p,
                        title=f"ERP report [{mode_name}] {p}")
        fig_drop = mne.viz.plot_drop_log(erp_cues.drop_log, show=False)
        report.add_figure(fig_drop, title="Drop log", section="Drop log")

        # --- Save evokeds and add to report ---
        figs_butterfly = []
        evokeds = {}
        for cond in events_id:
            evokeds[cond] = erp_cues[cond].average()
        #     figs_butterfly.append(
        #         pej(
        #             evokeds[cond],
        #             title=cond,
        #             show=False,
        #             picks="all",
        #             ts_args={"time_unit": "ms"},
        #             topomap_args={"time_unit": "ms"},
        #         )
        #     )
        #     tag = safe_tag(cond)
        #     ave_fname = cfg["ave_fname_tpl"].format(p=p, tag=tag)
        #     evokeds[cond].save(str(outdir / ave_fname), overwrite=True)

        # report.add_figure(
        #     figs_butterfly,
        #     section=cfg["report_section"],
        #     title=f"Butterfly plots ({mode_name})",
        # )

        first_cond = list(evokeds.keys())[0]
        fig_img = evokeds[first_cond].plot_image(picks="eeg", show=False)
        report.add_figure(fig_img, section=cfg["report_section"], title="plot_image")

        figs_chan = []
        for c in CHANS_REPORT:
            if c in erp_cues.ch_names:
                pick = erp_cues.ch_names.index(c)
                figs_chan.append(
                    mne.viz.plot_compare_evokeds(evokeds, picks=pick, show=False)[0]
                )
        if figs_chan:
            report.add_figure(figs_chan, section=cfg["report_section"], title="Selected channels")

        report_name = cfg["report_tpl"].format(p=p)
        report.save(str(outdir / report_name), open_browser=False, overwrite=True)

        # --- Single-trial epochs ---
        n_trials = len(events_sub)
        events_sub = events_sub.copy()
        events_sub["trialsnum"] = np.arange(1, n_trials + 1)
        events_sub["trials_name"] = [f"trial_{i:03d}" for i in range(1, n_trials + 1)]
        events_sub["participant_id"] = p

        trials_dict = {
            row["trials_name"]: row["trialsnum"]
            for _, row in events_sub.iterrows()
        }
        events_strials = np.asarray(
            events_sub[["sample", "empty", "trialsnum"]], dtype=int
        )

        erp_single = mne.Epochs(
            raw,
            events=events_strials,
            event_id=trials_dict,
            tmin=cfg["tmin"],
            tmax=cfg["tmax"],
            baseline=cfg["baseline"],
            metadata=events_sub,
            preload=True,
            verbose=False,
        )

        # Mark bad trials without dropping them
        tmp = erp_single.copy().drop_bad(reject=ERPREJECT)
        bad_flags = [1 if len(li) > 0 else 0 for li in tmp.drop_log]
        erp_single.metadata["badtrial"] = bad_flags

        epo_fname = cfg["epo_fname_tpl"].format(p=p)
        erp_single.save(str(outdir / epo_fname), overwrite=True)
        print(f"    Saved {epo_fname}")

        plt.close("all")

    # --- Save rejection stats ---
    reject_stats["perc_removed_all"] = np.where(
        reject_stats["n_target"] > 0,
        (1 - reject_stats["n_kept"] / reject_stats["n_target"]) * 100,
        np.nan,
    )
    rej_path = outpath / cfg["reject_csv"]
    reject_stats.to_csv(rej_path, index=False)
    reject_stats.describe().to_csv(
        str(rej_path).replace(".csv", "_desc.csv")
    )
    print(f"\n  Rejection stats saved to {rej_path}")

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 2 — Collect amplitude metadata across participants
# ──────────────────────────────────────────────────────────────────────────────
CHANS_TO_AVERAGE = [["Fz"], ["POz"], ["Cz"], ["CPz"], ["Pz"], ["Oz"]]

for mode_name, cfg in MODE_CONFIGS.items():
    print(f"\n{'='*60}")
    print(f"Collecting metadata: {mode_name}")
    print(f"{'='*60}")

    all_meta = []
    for p in part:
        outdir = outpath / p / "eeg" / cfg["outdir_name"]
        epo_path = outdir / cfg["epo_fname_tpl"].format(p=p)
        if not epo_path.exists():
            raise FileNotFoundError(
                f"{p} [{mode_name}]: epoch file not found: {epo_path}\n"
                f"  Run Section 1 first."
            )

        epo = mne.read_epochs(str(epo_path))
        epo = average_time_win_strials(epo, CHANS_TO_AVERAGE, cfg["amp_lat"])
        epo.metadata["participant_id"] = p
        all_meta.append(epo.metadata)

    if not all_meta:
        raise RuntimeError(
            f"[{mode_name}]: no epoch files found for any participant — "
            f"nothing to collect for metadata CSV."
        )

    meta_df = pd.concat(all_meta, ignore_index=True)
    meta_path = outpath / cfg["meta_csv"]
    meta_df.to_csv(meta_path, index=False)
    print(f"  Metadata saved to {meta_path}")

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 3 — Extract RP amplitude and merge into behavioural CSV for HDDM
# Uses the response-locked epochs saved in Section 1 (decision_resp mode).
# Skipped gracefully if the behavioural file doesn't exist yet.
# ──────────────────────────────────────────────────────────────────────────────
if not BEHAV_FILE.exists():
    print(f"\nSkipping Section 3: behavioural file not found ({BEHAV_FILE})")
    print("  Run behav/02_sv_modelling.py first to generate it.")
    raise FileNotFoundError(f"Behavioural file not found: {BEHAV_FILE}")
else:
    print(f"\n{'='*60}")
    print("SECTION 3 — RP amplitude → HDDM merge")
    print(f"{'='*60}")

    resp_cfg = MODE_CONFIGS["decision_resp"]
    rp_tables = []
    for p in part:
        epo_path = (
            outpath / p / "eeg"
            / resp_cfg["outdir_name"]
            / resp_cfg["epo_fname_tpl"].format(p=p)
        )
        if not epo_path.exists():
            print(f"  {p}: epoch file missing — skipping")
            continue
        try:
            rp_tables.append(extract_rp_amplitude(epo_path, p))
            print(f"  {p}: loaded")
        except Exception as exc:
            print(f"  {p}: ERROR — {exc}")

    if not rp_tables:
        print("  No RP data found — skipping HDDM merge.")
    else:
        rp_table = pd.concat(rp_tables, ignore_index=True)
        rp_table = zscore_within_subject(rp_table, "rp_raw", "participant", "rp_z")

        behav = pd.read_csv(BEHAV_FILE)
        final_df, diag = merge_rp_into_behav(behav, rp_table)

        out_behav_dir = BEHAV_FILE.parent
        rp_table.to_csv(out_behav_dir / "rp_trial_table.csv", index=False)
        final_df.to_csv(out_behav_dir / "behav_with_exclusion_sv_modeling_with_rp.csv", index=False)
        diag.to_csv(out_behav_dir / "rp_merge_diagnostics.csv", index=False)

        print(f"\n  Saved to {out_behav_dir}")
        print(diag.to_string(index=False))
