
'''
 # @ : -*- coding: utf-8 -*-
 # @ Author: Michel-Pierre Coll (michel-pierre.coll@psy.ulaval.ca) & Veronika Wendler (2025)
 # @ Date: 2024
 # @ Description:
 
 1.set versions
 2.cleaning and z scoring
 3. group-level tests using ddm params (v,a) as predictor for rp in 2 time windows

'''

# import libraries
import mne
from os.path import join as opj
import pandas as pd
import numpy as np
import os
from mne.decoding import Scaler
import scipy
from bids import BIDSLayout
from mne.stats import spatio_temporal_cluster_1samp_test as st_clust_1s_ttest
from scipy import stats
import os
import re
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from statsmodels.distributions.empirical_distribution import ECDF
from pathlib import Path
from mne.stats import fdr_correction
from mne.time_frequency import tfr_morlet
from mne.stats import permutation_cluster_1samp_test, combine_adjacency



# Set directory
PROJECT_DIR = Path(os.getenv("PROJECT_DIR", "/workspace"))
basepath = Path(os.getenv("DATA_DIR", PROJECT_DIR / "EEG" / "PainReward_sub-001-050" / "painrewardeegdata"))
HDDM_DIR = Path(os.getenv("HDDM_DIR"))


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)
import re
from pathlib import Path
import os

layout = BIDSLayout(basepath)
# for cluster
# disable Numba JIT caching & compilation
#os.environ["NUMBA_DISABLE_JIT"] = "1"
import numba
numba.config.CACHE_ENABLE = False

# Outpath for analysis
outpath = Path(os.getenv("OUT_DIR", basepath / 'statistics'))       
if not os.path.exists(outpath):
    os.mkdir(outpath)

#----
    
# here for decision its just erps_massuni_drift_mod_9 and for passive it is: erps_massuni_drift_mod_9_2_passive
version = 2
v1_mode = "joint" 
v2_mode = "joint" 



if version == 1:
    base_v1 = opj(outpath, "erps_massuni_sv_cuelong")
    os.makedirs(base_v1, exist_ok=True)
    if v1_mode == "joint":
        outpath = opj(base_v1, "v1_rp_drift_joint")
    else:
        raise ValueError("v1_mode must be 'joint' ")
    os.makedirs(outpath, exist_ok=True)  
elif version == 2:
    base_v2 = opj(outpath, "erps_massuni_sv_cuelong")
    os.makedirs(base_v2, exist_ok=True)
    if v2_mode == "joint":
        outpath = opj(base_v2, "v2_rp_boundary_joint")
    else:
        raise ValueError("v2_mode must be 'joint' ")
    os.makedirs(outpath, exist_ok=True)  
else:
    print("no version")


# participants
part_csv = basepath / "participants.tsv"
part = pd.read_csv(part_csv, sep="\t")["participant_id"].unique().tolist()
part.sort()

# Silence pandas warning
pd.options.mode.chained_assignment = None  # default='warn'

# Parameters # similar to MP's painlearning (2024)
param = {
    # Njobs for permutations
    'njobs': 20,                   
    # Number of permutations
    'nperms': 5000,
    # Random state to get same permutations each time
    'random_state': 23,
    'testresampfreq': 1024,
    # clustering threshold
    'cluster_threshold': 0.01}

# this is the data frame I computed in the DDM_EEG_load.py file for the best fitting DDM by adding trial-by-trial drift-scaled pain as a column & other important parameters from the DDM
# if you are testing the influence of decision threshold on neural measures mod_10 can be used
mod_data_path = HDDM_DIR / "figures" / "painreward_behavioural_data_mod_9" / "diagnostics" / "v_pain_money.csv"
mod_data = pd.read_csv(mod_data_path, sep=None, engine="python")
mod_data["rt"] = mod_data["choice_resp.rt"]
mod_data["interaction"] = mod_data["moneylevel"]*mod_data["painlevel"]
mod_data["trialsnum"] = (
    mod_data["blocks.thisRepN"].astype(int) * 25
    + mod_data["trials.thisN"].astype(int)
    + 1
)
# same file but for threshold (a) parameters
mod_data_a_path = HDDM_DIR / "figures" / "painreward_behavioural_data_mod_10" / "diagnostics" / "a_pain_money.csv"
mod_data_a = pd.read_csv(mod_data_a_path, sep=None, engine="python")

for _df in [mod_data, mod_data_a]:
    _df["rt"] = _df["choice_resp.rt"]
    _df["interaction"] = _df["moneylevel"] * _df["painlevel"]
    _df["trialsnum"] = (
        _df["blocks.thisRepN"].astype(int) * 25
        + _df["trials.thisN"].astype(int)
        + 1
    )


# -----------------------
# predictor columns based on version
if version == 1:
    beh_df = mod_data
    pred1_col = "v_painlevel_subj"
    pred2_col = "v_moneylevel_subj"
    out_prefix = "v1"
elif version == 2:
    beh_df = mod_data_a
    pred1_col = "a_painlevel_subj"
    pred2_col = "a_moneylevel_subj"
    out_prefix = "v2"
else:
    raise ValueError("version must be 1 or 2")


# Subjects in EEG 
# EEG participants from participants.tsv
eeg_participants = set(part)
beh_participants = set(beh_df["participant"].unique())
# intersection
common_participants = sorted(list(eeg_participants & beh_participants))
print("\nCommon participants:", common_participants)
print("N common:", len(common_participants))
part = common_participants
part_1_dat = beh_df[beh_df["participant"].isin(part)].copy()
part_1 = part


def _z(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)] if x.ndim == 0 else x
    return stats.zscore(x, nan_policy="omit")

def fit_ols_beta(X, y):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta

####
def extract_bin_amplitudes_channels(epo, chs=("Cz",), bins=((-0.4,-0.2), (-0.2,-0.1)), min_chs=1):
    """
    rp_amp_by_bin: dict[(tmin,tmax)] -> (n_trials,) average across channels and time in bin
    used_chs: list of channels
    times: epo.times
    """
    present = [ch for ch in chs if ch in epo.ch_names]
    if len(present) < min_chs:
        raise ValueError(f"Channels not found. Wanted {chs}, found {present}")

    e = epo.copy().pick_channels(present)
    data = e.get_data()  # (n_trials, n_ch, n_times)
    times = e.times

    rp_amp_by_bin = {}
    for (tmin, tmax) in bins:
        tidx = np.where((times >= tmin) & (times <= tmax))[0]
        if len(tidx) < 3:
            raise ValueError(f"Too few samples in bin {(tmin, tmax)}")
        # mean over time, then mean over channels -> (n_trials,)
        rp_amp_by_bin[(tmin, tmax)] = data[:, :, tidx].mean(axis=2).mean(axis=1)

    return rp_amp_by_bin, present, times



# -----------------------
# helpers
# -----------------------
def z(x):
    return stats.zscore(np.asarray(x, float), nan_policy="omit")

def ols_with_t(X, y):
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    keep = np.isfinite(y) & np.isfinite(X).all(axis=1)
    X = X[keep]
    y = y[keep]
    n, p = X.shape
    df = n - p
    if df <= 0 or n <= p + 1:
        return (np.full(p, np.nan), np.full(p, np.nan), np.full(p, np.nan), df)

    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    s2 = (resid @ resid) / df
    XtX_inv = np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(XtX_inv) * s2)
    tvals = beta / se
    pvals = 2 * stats.t.sf(np.abs(tvals), df)
    return beta, tvals, pvals, df

def group_regress_joint2_with_cov(y, x1, x2, cov):
    """
    JOINT regression across subjects:
        y ~ z(x1) + z(x2) + z(cov) + intercept
    returns unique effects for x1 and x2:
        t1,p1,b1, t2,p2,b2, n
    """
    y = np.asarray(y, float)
    x1 = np.asarray(x1, float)
    x2 = np.asarray(x2, float)
    cov = np.asarray(cov, float)

    keep = np.isfinite(y) & np.isfinite(x1) & np.isfinite(x2) & np.isfinite(cov)
    y = y[keep]; x1 = x1[keep]; x2 = x2[keep]; cov = cov[keep]
    n = len(y)
    if n < 10:
        return (np.nan, np.nan, np.nan,
                np.nan, np.nan, np.nan, n)

    X = np.column_stack([np.ones(n), z(x1), z(x2), z(cov)])
    beta, tvals, pvals, _ = ols_with_t(X, y)
    # indices: 1=x1, 2=x2, 3=cov
    return (float(tvals[1]), float(pvals[1]), float(beta[1]),
            float(tvals[2]), float(pvals[2]), float(beta[2]),
            int(n))

def bh_fdr(pvals):
    pvals = np.asarray(pvals, float)
    out = np.full_like(pvals, np.nan)
    keep = np.isfinite(pvals)
    if keep.sum() == 0:
        return out
    _, p_adj = fdr_correction(pvals[keep], alpha=0.05, method="indep")
    out[keep] = p_adj
    return out

#------------------------------------------------------------------------------------------------------------------------------------------------
# Massunivariate Regression with 3 GLMs
#
# raw_regcols = ['painlevel', 'moneylevel', 'interaction']
# v_regcols   = ['v_pain_contrib', 'v_money_contrib', 'v_interaction_contrib']

# # full list of regressors to run GLMs on (each gets its own EEG GLM)
# regvars = raw_regcols + v_regcols
# regvarsnames = [
#     'pain_raw', 'money_raw', 'interaction_raw',
#     'V_pain_contrib', 'V_money_contrib', 'V_interaction_contrib'
# ]
#all_epos = [[] for i in range(len(regvars))]
#allbetasnp = []
#betas = [[] for i in range(len(regvars))]
part.sort()

#------------------------------------------------------------------------------------------------------------------------------------------------
# Creating the dataframes
#------------------------------------------------------------------------------------------------------------------------------------------------
# Creating the dataframes (only needed for versions 1–4)

if version in [1,2]:
    filtered_data = []
    for p in part:
        df = beh_df[beh_df["participant"] == p]
        
        if version in [1,2]:
            epo = mne.read_epochs(
                opj(basepath, "derivatives", p, "eeg", "erps_resp_rp",
                    f"{p}_decision_resp_rp_singletrials-epo.fif"),
                    preload=True)
            epo_1 = epo.copy()
            # epo = mne.read_epochs(
            #     opj(basepath, p, "eeg", "erps_resp", f"{p}_decision_resp_singletrials-epo.fif"),    %no Gluth preprocessing
            #     preload=True)
            # epo_1 = epo.copy()
        else:
            print("read_epochs issue")


        participants = epo_1.metadata['participant_id'].unique()
        trialblocks = []
        blocks_idx = []

        # create blocks for metadata
        for participant in participants:
            p_df = epo_1.metadata[epo_1.metadata['participant_id'] == participant]
            blocks = list(range(25)) * 5
            blocks_idx_participant = [i for i in range(5) for _ in range(25)]
            trialblocks.extend(blocks)
            blocks_idx.extend(blocks_idx_participant)
                
        epo_1.metadata['trialblocks'] = trialblocks
        epo_1.metadata['blocks_idx'] = blocks_idx

        epo_1_filtered = pd.DataFrame()

        # filter for unique participants in the behavioral frame
        for participant in df['participant'].unique():
            erps_p_df = epo_1.metadata[epo_1.metadata['participant_id'] == participant]
            df_unique = df[df['participant'] == participant]   

            for block_x in df_unique['blocks.thisRepN'].unique():
                erps_block_df = erps_p_df[erps_p_df['blocks_idx'] == block_x]
                df_block_df = df_unique[df_unique['blocks.thisRepN'] == block_x]
                    
                filtered_block_df = erps_block_df[erps_block_df['trialblocks'].isin(df_block_df['trials.thisN'])]            
                epo_1_filtered = pd.concat([epo_1_filtered, filtered_block_df], ignore_index=True)
        
        filtered_data.append(epo_1_filtered)

    epo_1_filtered_combined = pd.concat(filtered_data, ignore_index=True)

#epo_2_filtered_combined.to_csv('D:/Aberdeen_Uni_June24/MPColl_Lab/All_Files_Relevant_For_Git/Hddm_Docker_August_24/data_sets/epo_2_filtered_combined')

    merge_left = ['participant_id', 'blocks_idx', 'trialblocks']
    merge_right = ['participant', 'blocks.thisRepN', 'trials.thisN']


    trial_map = epo_1_filtered_combined.merge(
        beh_df,
        left_on=merge_left,
        right_on=merge_right,
        how='inner'
    )

    print("trial_map shape:", trial_map.shape)
    print("trial_map columns:", trial_map.columns.tolist())

    if "trialsnum_x" in trial_map.columns:
        trial_map = trial_map.rename(columns={"trialsnum_x": "trialsnum"})
    
    if "trialsnum_y" in trial_map.columns:
        trial_map = trial_map.drop(columns=["trialsnum_y"])
    

#----------------------------------------------------------------------------------------

if version in [1,2]:

    # -----------------------
    # Only JOINT model + RT covariate
    # Only BH-FDR correction
    # Two RP bins/windows
    # -----------------------

    bins = [(-0.4, -0.2), (-0.2, -0.1)]

    # False: FDR across both bins together (within each predictor)
    # True: FDR separately within each bin (across electrode sets)

    FDR_WITHIN_EACH_WINDOW = False

    # ROI electrodes + single channel versions (Hagaard paper and Wittmann paper)
    roi_chs = ("Cz", "CPz", "CP2", "CP1", "C2", "C1", "FC1", "FC2", "FCz")

    electrode_sets = {
        "roi_CzCPzCP2CP1C2C1FC1FC2FCz": roi_chs,
        "ch_Cz":  ("Cz",),
        "ch_CPz": ("CPz",),
        "ch_CP2": ("CP2",),
        "ch_CP1": ("CP1",),
        "ch_C2":  ("C2",),
        "ch_C1":  ("C1",),
        "ch_FC1": ("FC1",),
        "ch_FC2": ("FC2",),
        "ch_FCz": ("FCz",),
    }

    # subject-level predictors (constant within subject)
    predictors = [pred1_col, pred2_col]

    # RT summary covariate (subject-level)
    rt_col = "rt"
    rt_summary = "median"  # or mean

    # epochs
    rp_epo_dirname = "erps_resp_rp"
    rp_epo_suffix  = "_decision_resp_rp_singletrials-epo.fif"

    # output
    rp_outdir = Path(outpath) / "Zscoring"
    rp_outdir.mkdir(parents=True, exist_ok=True)

    # subject-level 
    subj_rows = []
    wf_store = {}
    times_store = {}
    included, skipped = [], []

    for pa in part:
        print(f"\n[{out_prefix}] rp ~ {pred1_col}+{pred2_col} joint+RT] {pa}")
        epo_path = opj(basepath, "derivatives", pa, "eeg", "erps_resp_rp", f"{pa}_decision_resp_rp_singletrials-epo.fif")
        if not os.path.exists(epo_path):
            print("  missing epochs:", epo_path)
            skipped.append(pa)
            continue

        epo = mne.read_epochs(epo_path, preload=True)

        mod2 = trial_map[trial_map["participant_id"] == pa].copy()

        # align by trialsnum if possible
        if epo.metadata is not None and ("trialsnum" in epo.metadata.columns) and ("trialsnum" in mod2.columns):
            keep = epo.metadata["trialsnum"].isin(mod2["trialsnum"])
            epo = epo[keep]
            mod2 = mod2.set_index("trialsnum").loc[epo.metadata["trialsnum"].values].reset_index()
        else:
            mod2 = mod2.reset_index(drop=True).iloc[:len(epo)].copy()

        # drop bad trials
        if epo.metadata is not None and "badtrial" in epo.metadata.columns:
            good = np.where(epo.metadata["badtrial"].to_numpy() == 0)[0]
            epo = epo[good]
            mod2 = mod2.iloc[good].reset_index(drop=True)

        if len(epo) < 8:
            print(" too few trials:", len(epo))
            skipped.append(pa)
            continue

        # require predictors exist
        missing_cols = [c for c in predictors if c not in mod2.columns]
        if len(missing_cols):
            print("  missing cols:", missing_cols)
            skipped.append(pa)
            continue

        # subject-level predictors 
        x1 = float(mod2[pred1_col].iloc[0])
        x2 = float(mod2[pred2_col].iloc[0])

        # subject-level RT
        rt_vals = mod2[rt_col].to_numpy(dtype=float) if rt_col in mod2.columns else np.array([])
        rt_vals = rt_vals[np.isfinite(rt_vals)]
        rt_subj = (float(np.median(rt_vals)) if rt_summary == "median" else float(np.mean(rt_vals))) if len(rt_vals) else np.nan

        any_set_used = False

        for set_name, chs in electrode_sets.items():
            try:
                rp_amp_by_bin, used_chs, _ = extract_bin_amplitudes_channels(
                    epo, chs=chs, bins=bins, min_chs=1
                )
            except Exception as e:
                print(f"  [{set_name}] skip: {e}")
                continue

            any_set_used = True

            # store waveforms
            ev = epo.copy().pick_channels(list(used_chs)).average()
            wf_store.setdefault(set_name, []).append(ev.data.mean(axis=0))
            times_store[set_name] = ev.times

            for (tmin, tmax), rp_amp_trials in rp_amp_by_bin.items():
                rp_mean_uV = float(np.mean(rp_amp_trials) * 1e6)

                subj_rows.append({
                    "participant": pa,
                    "set": set_name,
                    "bin_tmin": float(tmin),
                    "bin_tmax": float(tmax),
                    "rp_mean_uV": rp_mean_uV,
                    "n_trials": int(len(rp_amp_trials)),
                    pred1_col: x1,
                    pred2_col: x2,
                    "rt_subj": rt_subj,
                    "chs_used": "+".join(used_chs),
                })

        if any_set_used:
            included.append(pa)
        else:
            skipped.append(pa)

    subj_df = pd.DataFrame(subj_rows)
    subj_df.to_csv(rp_outdir / f"{out_prefix}_subject_level_rp_means_by_bin.csv", index=False)

    np.save(rp_outdir / f"{out_prefix}_included_subjects.npy", np.array(included, dtype=object))
    np.save(rp_outdir / f"{out_prefix}_skipped_subjects.npy", np.array(skipped, dtype=object))

    # save waveforms
    for set_name, wfs in wf_store.items():
        wfs = np.vstack(wfs)
        np.save(rp_outdir / f"{out_prefix}_{set_name}__rp_subject_waveforms.npy", wfs)
        np.save(rp_outdir / f"{out_prefix}_{set_name}__rp_times.npy", times_store[set_name])

    # -----------------------
    # group-level regression
    group_rows = []

    if len(subj_df) > 0:
        for (set_name, tmin, tmax), sdf in subj_df.groupby(["set", "bin_tmin", "bin_tmax"]):
            y = sdf["rp_mean_uV"].to_numpy(dtype=float)
            rt_cov = sdf["rt_subj"].to_numpy(dtype=float)

            jt1, jp1, jb1, jt2, jp2, jb2, nj = group_regress_joint2_with_cov(
                y=y,
                x1=sdf[pred1_col].to_numpy(dtype=float),
                x2=sdf[pred2_col].to_numpy(dtype=float),
                cov=rt_cov
            )

            group_rows.append({
                "set": set_name, "bin_tmin": tmin, "bin_tmax": tmax,
                "model": "joint_plus_rt",
                "dv": "rp_mean_uV", "predictor": pred1_col,
                "n_subj": nj, "beta_z": jb1, "t": jt1, "p": jp1
            })
            group_rows.append({
                "set": set_name, "bin_tmin": tmin, "bin_tmax": tmax,
                "model": "joint_plus_rt",
                "dv": "rp_mean_uV", "predictor": pred2_col,
                "n_subj": nj, "beta_z": jb2, "t": jt2, "p": jp2
            })

    group_df = pd.DataFrame(group_rows)

    # -----------------------
    # BH-FDR correction 
    if len(group_df) > 0:
        group_df["p_fdr_bh"] = np.nan

        if FDR_WITHIN_EACH_WINDOW:
            for pred in group_df["predictor"].unique():
                for (tmin, tmax) in group_df[["bin_tmin", "bin_tmax"]].drop_duplicates().itertuples(index=False):
                    idx = (
                        (group_df["model"] == "joint_plus_rt") &
                        (group_df["predictor"] == pred) &
                        (group_df["bin_tmin"] == tmin) &
                        (group_df["bin_tmax"] == tmax)
                    )
                    group_df.loc[idx, "p_fdr_bh"] = bh_fdr(group_df.loc[idx, "p"].to_numpy(dtype=float))
        else:
            for pred in group_df["predictor"].unique():
                idx = (group_df["model"] == "joint_plus_rt") & (group_df["predictor"] == pred)
                group_df.loc[idx, "p_fdr_bh"] = bh_fdr(group_df.loc[idx, "p"].to_numpy(dtype=float))

    group_df.to_csv(rp_outdir / f"{out_prefix}_group_regress_rp_on_ddmparam_by_bin.csv", index=False)


    print(" ", rp_outdir / f"{out_prefix}_subject_level_rp_means_by_bin.csv")
    print(" ", rp_outdir / f"{out_prefix}_group_regress_rp_on_ddmparam_by_bin.csv")



