'''
 # @ : -*- coding: utf-8 -*-
 # @ Author: Michel-Pierre Coll (michel-pierre.coll@psy.ulaval.ca) & Veronika Wendler (2025)
 # @ Date: 2024
 # @ Description:
 
 1.set versions
 2.cleaning and z scoring
 3.Grand average & second-level cluster test (versions 1–3)
 
 '''

# Massunivariate Analysis and Second level test on betas
#----------------------------------------------------------------------------
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

# directory
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


# here for decision its just erps_massuni_drift_mod_9 and for passive it is: erps_massuni_drift_mod_9_2_passive
version = 2
v32_mode = "joint"   # "joint" or "separate"

if version == 1:
    outpath = opj(outpath, 'erps_massuni_drift_mod_9_passive')
    if not os.path.exists(outpath):
        os.mkdir(outpath)
if version == 2: # with RT as covariate
    outpath = opj(outpath, 'erps_massuni_regression')
    if not os.path.exists(outpath):
        os.mkdir(outpath)             
else:
    print("no version")


# participants
# participants
part_csv = basepath / "participants.tsv"
part = pd.read_csv(part_csv, sep="\t")["participant_id"].unique().tolist()
part.sort()

# Silence pandas warning
pd.options.mode.chained_assignment = None  

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


# Subjects in EEG 
eeg_participants = set(part)
# Subjects in HDDM CSV (should be 38 in total)
beh_participants = set(mod_data["participant"].unique())
# Subjects present in both datasets
common_participants = sorted(list(eeg_participants & beh_participants))

print("\n Subjects:", common_participants)                                   # should be 38
print(len(common_participants))

part = common_participants
part_1_dat = mod_data[mod_data["participant"].isin(part)]
part_1 = part

####

raw_regcols = ['painlevel', 'moneylevel']
regvars = raw_regcols  

#all_epos = [[] for i in range(len(regvars))]
#allbetasnp = []
#betas = [[] for i in range(len(regvars))]
part.sort()

#------------------------------------------------------------------------------------------------------------------------------------------------
# Creating the dataframes

#------------------------------------------------------------------------------------------------------------------------------------------------
# Creating the dataframes (only needed for versions 1–4)

if version in [1, 2]:
    filtered_data = []
    for p in part:
        # data for this participant
        df = mod_data[mod_data['participant'] == p]
        
        # Load single epochs file
        if version == 1:
            epo = mne.read_epochs(opj(basepath,  p, 'eeg', 'erps_passive',                   
                                  p + '_passive_cues_singletrials-epo.fif'))
            epo_1 = epo.copy()

        elif version in [2]:
            epo = mne.read_epochs(
                opj(basepath, "derivatives", p, "eeg", "erps",
                    f"{p}_decision_cues_singletrials-epo.fif"),
                    preload=True)
            epo_1 = epo.copy()

        elif version in [3]:
            epo = mne.read_epochs(
                opj(basepath, p, "eeg", "erps_resp_rp", f"{p}_decision_resp_rp_singletrials-epo.fif"),
                preload=True)
            epo_1 = epo.copy()
            # epo = mne.read_epochs(
            #     opj(basepath, p, "eeg", "erps_resp", f"{p}_decision_resp_singletrials-epo.fif"),
            #     preload=True)
            # epo_1 = epo.copy()
        elif version in [4]:
            epo = mne.read_epochs(
                opj(basepath, p, "eeg", "erps_long", f"{p}_decision_cues_long_singletrials-epo.fif"),
                preload=True
            )
            epo_1 = epo.copy()

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


    merge_left = ['participant_id', 'blocks_idx', 'trialblocks']
    merge_right = ['participant', 'blocks.thisRepN', 'trials.thisN']

    trial_map = epo_1_filtered_combined.merge(
        mod_data,
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
    

#------------------------------------------------------------------------------------------------------------------------------------------------
# Massunivariate 

if version in [1, 2, 3]:

    # storage
    all_epos = [[] for _ in range(len(regvars))]   # keep per-regvar epoch saves
    allbetasnp = []                                # per subject: (2, n_chan, n_time)
    betas = [[] for _ in range(len(regvars))]      # per regvar: list of Evoked beta

    included_subjects = []
    skipped_subjects = []

    z_dir = Path(outpath) / "Zscoring"
    ensure_dir(z_dir)

    for pa in part_1:
        print(f"\n--- pain = money model (v{version}): Processing {pa} ---")

        # Behavioural tables for this participant
        df2 = epo_1_filtered_combined[epo_1_filtered_combined['participant_id'] == pa]
        mod2 = part_1_dat[part_1_dat['participant'] == pa].copy()

        # Load epochs
        if version == 1:
            epo = mne.read_epochs(
                opj(basepath, pa, 'eeg', 'erps_passive', pa + '_passive_cues_singletrials-epo.fif')
            )
        else:  # version 2 or 3
            epo = mne.read_epochs(
                opj(basepath, "derivatives", p, "eeg", "erps",
                    f"{p}_decision_cues_singletrials-epo.fif"),
                    preload=True)
            epo_1 = epo.copy()


        epo_cop = epo.copy()

        # Match trials using 'trialsnum'
        matching = epo_cop.metadata['trialsnum'].isin(df2['trialsnum'])
        epo_filt = epo_cop[matching]

        # Downsample if needed
        if epo_filt.info['sfreq'] != param['testresampfreq']:
            epo_filt = epo_filt.resample(param['testresampfreq'])

        # Drop bad trials
        goodtrials = np.where(epo_filt.metadata['badtrial'] == 0)[0]
        df2 = df2.iloc[goodtrials].reset_index(drop=True)
        mod2 = mod2.iloc[goodtrials].reset_index(drop=True)
        epo_filt = epo_filt[goodtrials]

        if len(df2) < 5:
            print(f"Skipping {pa} (only {len(df2)} trials after cleaning)")
            skipped_subjects.append(pa)
            continue

        # RT column
        if "rt" in mod2.columns:
            rt_col = "rt"
        else:
            raise ValueError(f"No RT column in mod_data. Columns: {mod2.columns.tolist()}")

        # ------------------------------------------------------------
        # model trial mask 
        
        vals_pain = mod2["painlevel"].to_numpy(dtype=float)
        vals_money = mod2["moneylevel"].to_numpy(dtype=float)
        vals_rt = mod2[rt_col].to_numpy(dtype=float)

        keep = np.where(
            np.isfinite(vals_pain) &
            np.isfinite(vals_money) &
            np.isfinite(vals_rt)
        )[0]

        if len(keep) < 5:
            print(f"Skipping {pa}: only {len(keep)} valid trials for joint model")
            skipped_subjects.append(pa)
            continue

        mod2k = mod2.iloc[keep].reset_index(drop=True)
        epo_keep = epo_filt.copy()[keep]

        # Variance checks
        if np.nanstd(mod2k["painlevel"]) == 0:
            print(f"Skipping {pa}: painlevel has zero variance")
            skipped_subjects.append(pa)
            continue
        if np.nanstd(mod2k["moneylevel"]) == 0:
            print(f"Skipping {pa}: moneylevel has zero variance")
            skipped_subjects.append(pa)
            continue
        if np.nanstd(mod2k[rt_col]) == 0:
            print(f"Skipping {pa}: RT has zero variance")
            skipped_subjects.append(pa)
            continue

        # ------------------------------------------------------------
        # Z-score EEG across trials
        
        scale = Scaler(scalings='mean')
        epo_z = mne.EpochsArray(scale.fit_transform(epo_keep.get_data()),
                                epo_keep.info)

        # ------------------------------------------------------------
        # design matrix 
        # EEG ~ 1 + pain_z + money_z + RT_z

        df_reg = mod2k.copy()
        df_reg["Intercept"] = 1.0
        df_reg["pain_z"] = stats.zscore(df_reg["painlevel"].to_numpy(dtype=float))
        df_reg["money_z"] = stats.zscore(df_reg["moneylevel"].to_numpy(dtype=float))
        df_reg["RT_z"] = stats.zscore(df_reg[rt_col].to_numpy(dtype=float))

        design = df_reg[["Intercept", "pain_z", "money_z", "RT_z"]]
        names = ["Intercept", "pain_z", "money_z", "RT_z"]

        if not np.all(np.isfinite(design.to_numpy())):
            print(f"Skipping {pa}: design matrix has NaN/Inf")
            skipped_subjects.append(pa)
            continue

        # Update metadata for plotting
        df_meta = epo_keep.metadata.reset_index(drop=True).copy()
        df_meta["painlevel"] = df_reg["painlevel"].values
        df_meta["moneylevel"] = df_reg["moneylevel"].values
        df_meta[rt_col] = df_reg[rt_col].values
        epo_keep.metadata = df_meta

        # Store epochs per regressor
        # Same epochs go into both lists
        all_epos[0].append(epo_keep)
        all_epos[1].append(epo_keep)

        # ------------------------------------------------------------
        # Run regression 

        res = mne.stats.linear_regression(epo_z, design, names=names)

        beta_pain = res["pain_z"].beta   # evoked
        beta_money = res["money_z"].beta # evoked

        # regvars order
        betas[0].append(beta_pain)
        betas[1].append(beta_money)

        allbetasnp.append(np.stack([beta_pain.data, beta_money.data]))

        included_subjects.append(pa)
        print(f"Included {pa}")

    # ---------------------------------------------------------------------
    # stack betas across subjects (n_subj, 2, n_chan, n_time)

    if len(allbetasnp) == 0:
        raise RuntimeError("No subjects included")

    allbetas = np.stack(allbetasnp)
    print(f"\nTotal subjects considered: {len(part_1)}")
    print(f"Included ({len(included_subjects)}): {included_subjects}")
    print(f"Skipped  ({len(skipped_subjects)}): {skipped_subjects}")

    # grand average maps
    beta_gavg = []
    for idx, regvar in enumerate(regvars):
        beta_gavg.append(mne.grand_average(betas[idx]))

    # connectivity for cluster test
    connect, names_ch = mne.channels.find_ch_adjacency(epo_keep.info, ch_type='eeg')

    # cluster threshold
    if not isinstance(param['cluster_threshold'], dict):
        p_thresh = param['cluster_threshold'] / 2
        n_samples = allbetas.shape[0]
        cluster_threshold = -stats.t.ppf(p_thresh, n_samples - 1)
    else:
        cluster_threshold = param['cluster_threshold']

    # ---------------------------------------------------------------------
    # second-level cluster tests with pain and money betas 

    tvals, pvalues = [], []

    for idx, regvar in enumerate(regvars):
        print(f"\nSecond-level cluster test for regressor (JOINT beta): {regvar}")

        data_reg = allbetas[:, idx, :, :]     # n_subj, n_time, n_chan
        testdata = np.swapaxes(data_reg, 2, 1) # n_subj, n_time, n_chan

        tval, clusters, cluster_p_values, _ = st_clust_1s_ttest(
            testdata,
            n_jobs=param["njobs"],
            threshold=cluster_threshold,
            adjacency=connect,
            n_permutations=param['nperms'],
            buffer_size=None
        )

        pmap = np.ones_like(tval)
        for c, p_val in zip(clusters, cluster_p_values):
            pmap[c] = p_val

        tvals.append(tval)
        pvalues.append(pmap)

        np.save(z_dir / f'ols_2ndlevel_tval_{regvar}.npy', tval)
        np.save(z_dir / f'ols_2ndlevel_pval_{regvar}.npy', pmap)

    # save group-level results
    tvals = np.stack(tvals)  # (2, n_times, n_ch)
    pvals = np.stack(pvalues)

    # FDR across regressors using min cluster p per regressor
    min_cluster_ps = []
    for pmap in pvalues:
        mask = pmap < 1.0
        min_cluster_ps.append(pmap[mask].min() if np.any(mask) else 1.0)

    min_cluster_ps = np.asarray(min_cluster_ps)
    rej_fdr, p_fdr = fdr_correction(min_cluster_ps, alpha=0.05, method='indep')

    fdr_df = pd.DataFrame({
        "regressor": regvars,
        "min_cluster_p": min_cluster_ps,
        "min_cluster_p_FDR": p_fdr,
        "sig_FDR": rej_fdr
    })
    fdr_df.to_csv(z_dir / "cluster_FDR_across_regressors.csv", index=False)
    print(f"FDR summary across regressors in {z_dir}")

    np.save(z_dir / 'ols_2ndlevel_tvals.npy', tvals)
    np.save(z_dir / 'ols_2ndlevel_pvals.npy', pvals)
    np.save(z_dir / 'ols_2ndlevel_betas.npy', allbetas)
    np.save(z_dir / 'included_subjects.npy', np.array(included_subjects, dtype=object))
    np.save(z_dir / 'ols_2ndlevel_betasavg.npy', beta_gavg)

    # Save epochs per regressor 
    for idx, regvar in enumerate(regvars):
        if len(all_epos[idx]) == 0:
            continue
        epo_save = mne.concatenate_epochs(all_epos[idx])
        epo_save.save(z_dir / f'ols_2ndlevel_allepochs-epo_{regvar}.fif', overwrite=True)

    # ---------------------------------------------------------------------
    # Beta-difference cluster test for pain - money

    print("\nComputing pain - money beta-difference cluster test ...")

    pain_idx = regvars.index("painlevel")
    money_idx = regvars.index("moneylevel")

    data_pain = allbetas[:, pain_idx, :, :]      # n_subj, n_chan, n_time
    data_money = allbetas[:, money_idx, :, :]    # n_subj, n_chan, n_time

    beta_diff = data_pain - data_money           # β_pain - β_money
    testdata_diff = np.swapaxes(beta_diff, 2, 1) # n_subj, n_time, n_chan

    tval_diff, clusters_diff, cluster_p_values_diff, _ = st_clust_1s_ttest(
        testdata_diff,
        n_jobs=param["njobs"],
        threshold=cluster_threshold,
        adjacency=connect,
        n_permutations=param['nperms'],
        buffer_size=None
    )

    pvals_diff = np.ones_like(tval_diff)
    for c, p_val in zip(clusters_diff, cluster_p_values_diff):
        pvals_diff[c] = p_val

    np.save(z_dir / 'ols_2ndlevel_tval_diff_pain_minus_money.npy', tval_diff)
    np.save(z_dir / 'ols_2ndlevel_pval_diff_pain_minus_money.npy', pvals_diff)

    print("saved pain-money beta-difference maps in", z_dir)