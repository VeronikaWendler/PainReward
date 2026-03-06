#!/usr/bin/env python3
"""
build a trial-level hddm input dataframe by:
1) reading response-locked single-trial eeg epochs
2) extracting rp mean amplitude from a 9-electrode roi and time window
3) aligning erp trials with behavioural decision trials
4) merging erp values into the behavioural dataframe
5) saving a final csv ready for hddm
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

import mne
import numpy as np
import pandas as pd

# =========================
# user settings
# =========================
project_dir = Path("/rds/projects/z/zhanglp-vwendler-core/PainReward_ULaval").resolve()

deriv_dir = Path(
    "/rds/projects/z/zhanglp-vwendler-core/PainReward_ULaval/EEG/PainReward_sub-001-050/painrewardeegdata/derivatives"
).resolve()

behav_file = Path(os.getenv(
    "behav_file",
    (project_dir / "Hddm_Docker_August_24" / "data_sets" / "behavioural_sv_cleaned_final_3.csv").as_posix(),
)).resolve()

out_dir = Path(os.getenv(
    "out_dir",
    (project_dir / "Hddm_Docker_August_24" / "data_sets").as_posix(),
)).resolve()
out_dir.mkdir(parents=True, exist_ok=True)

# response-locked rp epochs directory
rp_subdir = "erps_resp_rp"

# rp roi and time window
rp_channels: List[str] = ["Cz", "CPz", "CP1", "CP2", "C1", "C2", "FC1", "FC2", "FCz"]

rp_tmin = -0.5
rp_tmax = -0.1

keep_badtrial_rows = True


# =========================
# helper functions
# =========================

def zscore_within_subject(df: pd.DataFrame, value_col: str, subj_col: str, out_col: str):

    def z(x):
        sd = x.std(ddof=0)
        if pd.isna(sd) or sd == 0:
            return pd.Series(np.nan, index=x.index)
        return (x - x.mean()) / sd

    df[out_col] = df.groupby(subj_col)[value_col].transform(z)

    return df


def sort_behaviour_trials(df: pd.DataFrame):

    sort_cols = ["participant"]

    if "blocks.thisRepN" in df.columns:
        sort_cols.append("blocks.thisRepN")

    if "trials.thisN" in df.columns:
        sort_cols.append("trials.thisN")

    return df.sort_values(sort_cols).copy()


def make_behaviour_trial_table(behav: pd.DataFrame):

    dec = behav[behav["TaskName"] == "decision"].copy()
    dec = sort_behaviour_trials(dec)
    dec["trial_seq"] = dec.groupby("participant").cumcount() + 1
    return dec


def load_single_participant_rp(participant: str):

    epo_path = deriv_dir / participant / "eeg" / rp_subdir / f"{participant}_decision_resp_rp_singletrials-epo.fif"
    print("checking", epo_path)

    if not epo_path.exists():
        raise FileNotFoundError(f"erp file not found for {participant}: {epo_path}")

    epochs = mne.read_epochs(epo_path.as_posix(), preload=True, verbose="ERROR")

    print(participant, "metadata columns:", list(epochs.metadata.columns))

    missing_chans = [c for c in rp_channels if c not in epochs.ch_names]
    if missing_chans:
        raise ValueError(f"{participant}: missing rp channels {missing_chans}")

    rp_epochs = epochs.copy().pick(rp_channels).crop(tmin=rp_tmin, tmax=rp_tmax)
    data = rp_epochs.get_data()
    rp_raw = data.mean(axis=(1, 2))

    meta = epochs.metadata.copy() if epochs.metadata is not None else pd.DataFrame(index=np.arange(len(epochs)))
    meta = meta.reset_index(drop=True)

    meta["rp_raw"] = rp_raw
    if "sample" in meta.columns:
        meta = meta.sort_values("sample").reset_index(drop=True)

    meta["participant"] = participant
    meta["trial_seq"] = np.arange(1, len(meta) + 1)

    if "badtrial" not in meta.columns:
        meta["badtrial"] = [1 if len(x) > 0 else 0 for x in epochs.drop_log]

    if keep_badtrial_rows:
        meta.loc[meta["badtrial"] == 1, "rp_raw"] = np.nan
    else:
        meta = meta[meta["badtrial"] == 0].copy()

    return meta

def build_rp_trial_table(participants: List[str]):

    tables = []
    for p in participants:
        try:
            tables.append(load_single_participant_rp(p))
            print("loaded", p)
        except FileNotFoundError:
            print("missing erp", p)

    rp = pd.concat(tables, ignore_index=True)
    rp = zscore_within_subject(rp, "rp_raw", "participant", "rp_z")
    return rp


def merge_behaviour_and_rp(behav_full: pd.DataFrame, rp_table: pd.DataFrame):

    behav_full = behav_full.copy()
    behav_full["row_id"] = np.arange(len(behav_full))

    behav_dec = behav_full[behav_full["TaskName"] == "decision"].copy()
    behav_dec = sort_behaviour_trials(behav_dec)
    behav_dec["trial_seq"] = behav_dec.groupby("participant").cumcount() + 1

    merged_dec = behav_dec.merge(
        rp_table,
        on=["participant", "trial_seq"],
        how="left",
        validate="one_to_one",
    )

    merged_subset = merged_dec[["row_id", "trial_seq", "rp_raw", "rp_z", "badtrial"]].copy()

    final = behav_full.merge(merged_subset, on="row_id", how="left")
    final = final.sort_values("row_id").drop(columns=["row_id"])

    diag = merged_dec.groupby("participant").agg(
        n_trials=("trial_seq", "size"),
        n_rp=("rp_raw", lambda x: x.notna().sum()),
    ).reset_index()

    diag["merge_rate"] = diag["n_rp"] / diag["n_trials"]

    return final, diag

# =========================
# main
# =========================

def main():

    print("reading behavioural file")
    behav = pd.read_csv(behav_file)
    participants = sorted(
        behav.loc[behav["TaskName"] == "decision", "participant"].dropna().unique()
    )
    print("participants", participants)
    rp_table = build_rp_trial_table(participants)
    rp_table_path = out_dir / "rp_trial_table.csv"
    rp_table.to_csv(rp_table_path, index=False)

    print("saved rp table", rp_table_path)
    final_df, diag = merge_behaviour_and_rp(behav, rp_table)
    final_path = out_dir / "behavioural_sv_cleaned_final_3_with_rp.csv"
    diag_path = out_dir / "merge_diagnostics.csv"
    final_df.to_csv(final_path, index=False)
    diag.to_csv(diag_path, index=False)
    print("saved final dataframe", final_path)
    print("merge diagnostics")
    print(diag)


if __name__ == "__main__":
    main()
