'''
 # @ : -*- coding: utf-8 -*-
 # @ Author: Michel-Pierre Coll (michel-pierre.coll@psy.ulaval.ca), edited by Veronika Wendler (2025)
 # @ Date: 2024
 # @ Description:
 
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

# Set bids directory
PROJECT_DIR = Path(os.getenv("PROJECT_DIR", "/workspace"))
basepath = PROJECT_DIR / "EEG" / "PainReward_sub-001-050" / "painrewardeegdata" / "derivatives"

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
outpath = opj(basepath, 'statistics')       
if not os.path.exists(outpath):
    os.mkdir(outpath)
    
# here for decision its just erps_massuni_drift_mod_9_2 and for passive it is: erps_massuni_drift_mod_9_2_passive
version = 3    # version 1 is for decision and version 2 is for passive phase 

if version == 1:
    outpath = opj(outpath, 'erps_massuni_drift_mod_9_2')
    if not os.path.exists(outpath):
        os.mkdir(outpath)
elif version == 2:
    outpath = opj(outpath, 'erps_massuni_drift_mod_9_2_passive')
    if not os.path.exists(outpath):
        os.mkdir(outpath)
elif version == 3: #with RT covariate: wihtout 2 GLMs it's just: erps_massuni_drift_mod_9_2_RT
    outpath = opj(outpath, 'erps_massuni_drift_mod_9_2_RT_2GLMs')
    if not os.path.exists(outpath):
        os.mkdir(outpath)
elif version == 4: #with RT covariate
    outpath = opj(outpath, 'erps_massuni_drift_mod_9_2_RTbin')
    if not os.path.exists(outpath):
        os.mkdir(outpath)
else:
    print("no version")

# IMPORTANT
# version 1 & 2 still use the old participant loop, that is, there is no RT covariate in 1 & 2

# participants
part_csv = PROJECT_DIR / "EEG" / "PainReward_sub-001-050" / "painrewardeegdata" / "participants.tsv"
part = pd.read_csv(part_csv, sep=None, engine="python")["participant_id"].unique().tolist()
part.sort()

# Silence pandas warning
pd.options.mode.chained_assignment = None  # default='warn'

# Parameters
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

#mod_data = pd.read_csv('D:/Aberdeen_Uni_June24/MPColl_Lab/All_Files_Relevant_For_Git/Hddm_Docker_August_24/data_sets/data_with_v_sv_pain_para_contrib.csv')
#mod_data = pd.read_csv('D:/Aberdeen_Uni_June24/MPColl_Lab/All_Files_Relevant_For_Git/Hddm_Docker_August_24/data_sets/data_with_v_sv_money_contrib.csv')
#mod_data = pd.read_csv('D:/Aberdeen_Uni_June24/MPColl_Lab/All_Files_Relevant_For_Git/Hddm_Docker_August_24/data_sets/data_with_sv_pain_para_Abs_contrib.csv')
#mod_data = pd.read_csv('D:/Aberdeen_Uni_June24/MPColl_Lab/All_Files_Relevant_For_Git/Hddm_Docker_August_24/data_sets/data_with_sv_pain_para_OV_contrib.csv')
#mod_data = pd.read_csv('D:/Aberdeen_Uni_June24/MPColl_Lab/All_Files_Relevant_For_Git/Hddm_Docker_August_24/data_sets/data_with_full_sv_pain_para_contrib.csv')
#mod_data = pd.read_csv('D:/Aberdeen_Uni_June24/MPColl_Lab/All_Files_Relevant_For_Git/Hddm_Docker_August_24/data_sets/data_with_complex_pain_money_contrib.csv')
#mod_data = pd.read_csv('D:/Aberdeen_Uni_June24/MPColl_Lab/All_Files_Relevant_For_Git/Hddm_Docker_August_24/data_sets/data_with_v_sv_pain_para_contrib.csv')
#mod_data = pd.read_csv('D:/Aberdeen_Uni_June24/MPColl_Lab/All_Files_Relevant_For_Git/Hddm_Docker_August_24/data_sets/data_with_sv_pain_para_Quest.csv)

mod_data_path = PROJECT_DIR / "Hddm_Docker_August_24" / "figures_dir" / "painreward_behavioural_data_LPP_9" / "diagnostics" / "v_pain_money_interaction.csv"
mod_data = pd.read_csv(mod_data_path, sep=None, engine="python")

# Subjects in EEG
eeg_participants = set(part)
# Subjects in HDDM CSV
beh_participants = set(mod_data["participant"].unique())
# Subjects present in both datasets
common_participants = sorted(list(eeg_participants & beh_participants))

print("\n Subjects:", common_participants)


part = common_participants

part_1_dat = mod_data[mod_data["participant"].isin(part)]
part_1 = part
#mod_data = mod_data[mod_data['OV_value'] == 'high_OV']


#------------------------------------------------------------------------------------------------------------------------------------------------
# Massunivariate Regression with 2 GLMs
#
raw_regcols = ['painlevel', 'moneylevel', 'interaction']
v_regcols   = ['v_pain_contrib', 'v_money_contrib', 'v_interaction_contrib']

# full list of regressors to run GLMs on (each gets its own EEG GLM)
regvars = raw_regcols + v_regcols
regvarsnames = [
    'pain_raw', 'money_raw', 'interaction_raw',
    'V_pain_contrib', 'V_money_contrib', 'V_interaction_contrib'
]


all_epos = [[] for i in range(len(regvars))]
allbetasnp = []
betas = [[] for i in range(len(regvars))]
part.sort()

#------------------------------------------------------------------------------------------------------------------------------------------------
# Creating the dataframes

filtered_data = []
for p in part:
    # data for this part
    df = mod_data[mod_data['participant'] == p]
    
    # Load single epochs file (cotains one epoch/trial)
    if version == 1:
        epo = mne.read_epochs(opj(basepath,  p, 'eeg', 'erps',                   
                              p + '_decision_cues_singletrials-epo.fif'))
        epo_1 = epo.copy()
    elif version == 2:
        epo = mne.read_epochs(opj(basepath,  p, 'eeg', 'erps_passive',                   
                              p + '_passive_cues_singletrials-epo.fif'))
        epo_1 = epo.copy()
    elif version == 3:
        epo = mne.read_epochs(opj(basepath,  p, 'eeg', 'erps',                   
                              p + '_decision_cues_singletrials-epo.fif'))
        epo_1 = epo.copy()
    elif version == 4:
        epo = mne.read_epochs(opj(basepath,  p, 'eeg', 'erps',                   
                              p + '_decision_cues_singletrials-epo.fif'))
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

        # Loop through each block
        for block_x in df_unique['blocks.thisRepN'].unique():
            erps_block_df = erps_p_df[erps_p_df['blocks_idx'] == block_x]
            df_block_df = df_unique[df_unique['blocks.thisRepN'] == block_x]
                
            filtered_block_df = erps_block_df[erps_block_df['trialblocks'].isin(df_block_df['trials.thisN'])]            
            epo_1_filtered = pd.concat([epo_1_filtered, filtered_block_df], ignore_index=True)
    
    filtered_data.append(epo_1_filtered)

epo_1_filtered_combined = pd.concat(filtered_data, ignore_index=True)
#epo_2_filtered_combined.to_csv('D:/Aberdeen_Uni_June24/MPColl_Lab/All_Files_Relevant_For_Git/Hddm_Docker_August_24/data_sets/epo_2_filtered_combined')



#------------------------------------------------------------------------------------------------------------------------------------------------
# Massunivariate 
# For versions  1 (decision), 2 (passive phase), 3 (RT as covariate)

if version in [1, 2, 3]:

    included_subjects = []
    skipped_subjects = []

    for pa in part_1:
        print(f"\n--- Processing {pa} ---")
        df2 = epo_1_filtered_combined[epo_1_filtered_combined['participant_id'] == pa]
        mod2 = part_1_dat[part_1_dat['participant'] == pa]

        # Load epochs
        if version == 1:
            epo = mne.read_epochs(
                opj(basepath, pa, 'eeg', 'erps',
                    pa + '_decision_cues_singletrials-epo.fif')
            )
        elif version == 2:
            epo = mne.read_epochs(
                opj(basepath, pa, 'eeg', 'erps_passive',
                    pa + '_passive_cues_singletrials-epo.fif')
            )
        elif version == 3:
            epo = mne.read_epochs(
                opj(basepath, pa, 'eeg', 'erps',
                    pa + '_decision_cues_singletrials-epo.fif')
            )
        epo_cop = epo.copy()

        # Check that trials match
        matching = epo_cop.metadata['trialsnum'].isin(df2['trialsnum'])
        epo_filt = epo_cop[matching]

        # Downsample
        if epo_filt.info['sfreq'] != param['testresampfreq']:
            epo_filt = epo_filt.resample(param['testresampfreq'])

        # Drop bad trials
        goodtrials = np.where(epo_filt.metadata['badtrial'] == 0)[0]
        df2 = df2.iloc[goodtrials]
        mod2 = mod2.iloc[goodtrials]
        epo_filt = epo_filt[goodtrials]

        # Z-score EEG across trials
        scale = Scaler(scalings='mean')
        epo_z = mne.EpochsArray(scale.fit_transform(epo_filt.get_data()),
                                epo_filt.info)

        # If there are too few trials after matching, skip subject (but this here can be adjusted obviously; as long as the desgin matrix holds it should be fine)
        if len(df2) < 5:
            print(f"Skipping {pa} as only {len(df2)} working trials after cleaning")
            skipped_subjects.append(pa)
            continue

        # RT column - checking what's in there again
        if "rt" in mod2.columns:
            rt_col = "rt"
        else:
            raise ValueError(f"No RT column in mod_data. Columns: {mod2.columns}")

        betasnp = []
        subject_has_regressors = False

        for idx, regvar in enumerate(regvars):

            #get rid of NANs and inf
            vals_reg = mod2[regvar].to_numpy(dtype=float)
            vals_rt = mod2[rt_col].to_numpy(dtype=float)
            keep = np.where(np.isfinite(vals_reg) & np.isfinite(vals_rt))[0]

            if len(keep) < 5:
                print(f"Skipping {regvar} as only {len(keep)} valid trials")
                continue

            df_reg = mod2.iloc[keep].copy()
            epo_reg = epo_z.copy()[keep]
            epo_keep = epo_filt.copy()[keep]

            # check variance
            if np.nanstd(df_reg[regvar]) == 0:
                print(f"Skipping {regvar} due to zero variance")
                continue
            if np.nanstd(df_reg[rt_col]) == 0:
                print(f"Skipping {regvar} as RT has zero variance (subject {pa})")
                continue

            # # Z-score predictors
            # df_reg[regvar + "_z"] = stats.zscore(df_reg[regvar].to_numpy(dtype=float))
            # df_reg["RT_z"] = stats.zscore(df_reg[rt_col].to_numpy(dtype=float))
            # df_reg["Intercept"] = 1.0

            # design = df_reg[["Intercept", regvar + "_z", "RT_z"]]

            # # safety check
            # if not np.all(np.isfinite(design.to_numpy())):
            #     print(f"Skipping {regvar}: design matrix has NaN/Inf")
            #     continue

            # # update metadata of kept epochs
            # df_meta = epo_keep.metadata.reset_index(drop=True).copy()
            # df_meta[regvar] = df_reg[regvar].values
            # df_meta[rt_col] = df_reg[rt_col].values
            # epo_keep.metadata = df_meta

            # # Store epochs for second-level visualization
            # all_epos[idx].append(epo_keep)

            # # regression: EEG ~ Intercept + regvar + RT
            # res = mne.stats.linear_regression(
            #     epo_reg, design,
            #     names=["Intercept", regvar + "_z", "RT_z"]
            # )

            # # beta for regressor 
            # beta_reg = res[regvar + "_z"].beta
            # betas[idx].append(beta_reg)
            # betasnp.append(beta_reg.data)
            
            # ---- NO Z-SCORING OF PREDICTORS ----
            df_reg["Intercept"] = 1.0
            
            # Use the raw regressor and raw RT
            design = df_reg[["Intercept", regvar, rt_col]]
            
            # safety check
            if not np.all(np.isfinite(design.to_numpy())):
                print(f"Skipping {regvar}: design matrix has NaN/Inf")
                continue
            
            # update metadata of kept epochs
            df_meta = epo_keep.metadata.reset_index(drop=True).copy()
            df_meta[regvar] = df_reg[regvar].values
            df_meta[rt_col] = df_reg[rt_col].values
            epo_keep.metadata = df_meta
            
            # Store epochs for second-level visualization
            all_epos[idx].append(epo_keep)
            
            # regression: EEG ~ Intercept + regvar + RT
            res = mne.stats.linear_regression(
                epo_reg, design,
                names=["Intercept", regvar, rt_col]
            )
            
            # beta for (unscaled) regressor
            beta_reg = res[regvar].beta
            betas[idx].append(beta_reg)
            betasnp.append(beta_reg.data)


            subject_has_regressors = True
            print(f"Metadata columns for {pa}, regvar '{regvar}':")
            print(epo_keep.metadata.columns.tolist())

        if not subject_has_regressors:
            print(f"Skipping {pa} as no valid regressors with RT for this subject")
            skipped_subjects.append(pa)
            continue

        included_subjects.append(pa)
        allbetasnp.append(np.stack(betasnp))
        print(f"Included {pa}")

    # Stack all , shape (n_subj, n_reg, n_chan, n_time)
    allbetas = np.stack(allbetasnp)

    print(f"Total subjects: {len(part_1)}")
    print(f"Included ({len(included_subjects)}): {included_subjects}")
    print(f"Skipped  ({len(skipped_subjects)}): {skipped_subjects}")

    # ---------------------------------------------------------------------
    # Grand average & second-level cluster test (versions 1–3)
    # ---------------------------------------------------------------------
    beta_gavg = []
    for idx, regvar in enumerate(regvars):
        beta_gavg.append(mne.grand_average(betas[idx]))

    # connectivity (from last epo_filt)
    connect, names = mne.channels.find_ch_adjacency(epo_filt.info, ch_type='eeg')

    # Get cluster entering threshold
    if not isinstance(param['cluster_threshold'], dict):
        p_thresh = param['cluster_threshold'] / 2
        n_samples = allbetas.shape[0]
        cluster_threshold = -stats.t.ppf(p_thresh, n_samples - 1)
    else:
        cluster_threshold = param['cluster_threshold']

    # Perform test for each regressor
    tvals, pvalues = [], []
    for idx, regvar in enumerate(regvars):
        # allbetas: (n_subj, n_reg, n_chan, n_time)
        data_reg = allbetas[:, idx, :, :]           # (n_subj, n_chan, n_time)
        testdata = np.swapaxes(data_reg, 2, 1)      # (n_subj, n_time, n_chan)

        tval, clusters, cluster_p_values, _ = st_clust_1s_ttest(
            testdata,
            n_jobs=param["njobs"],
            threshold=cluster_threshold,
            adjacency=connect,
            n_permutations=param['nperms'],
            buffer_size=None
        )

        pvals = np.ones_like(tval)
        for c, p_val in zip(clusters, cluster_p_values):
            pvals[c] = p_val

        tvals.append(tval)
        pvalues.append(pvals)

        np.save(opj(outpath, 'ols_2ndlevel_tval_noz' + regvar + '.npy'), tvals[-1])
        np.save(opj(outpath, 'ols_2ndlevel_pval_noz' + regvar + '.npy'), pvalues[-1])

    # Stack and save group-level results
    tvals = np.stack(tvals)
    pvals = np.stack(pvalues)

    np.save(opj(outpath, 'ols_2ndlevel_tvals_noz.npy'), tvals)
    np.save(opj(outpath, 'ols_2ndlevel_pvals_noz.npy'), pvals)
    np.save(opj(outpath, 'ols_2ndlevel_betas_noz.npy'), allbetas)

    for idx, regvar in enumerate(regvars):
        epo_save = mne.concatenate_epochs(all_epos[idx])
        epo_save.save(opj(outpath, 'ols_2ndlevel_allepochs-epo_noz' + regvar + '.fif'),
                      overwrite=True)

    np.save(opj(outpath, 'ols_2ndlevel_betasavg_noz.npy'), beta_gavg)
    
    
    #---------------------------------------------------------------------------------------------------
    # ---------------------------------------------------------------------
    # Option 2: Cluster test on beta differences (drift vs raw)
    # ---------------------------------------------------------------------
    # indices: [0,1,2] = raw, [3,4,5] = drift
    diff_pairs = [
        (0, 3, 'pain'),        # pain_raw vs V_pain_contrib
        (1, 4, 'money'),       # money_raw vs V_money_contrib
        (2, 5, 'interaction')  # interaction_raw vs V_interaction_contrib
    ]
    
    for raw_idx, v_idx, label in diff_pairs:
        data_raw = allbetas[:, raw_idx, :, :]   # (n_subj, n_chan, n_time)
        data_v   = allbetas[:, v_idx, :, :]
        beta_diff = data_v - data_raw          # v - raw
    
        # shape for st_clust: (n_subj, n_times, n_channels)
        testdata = np.swapaxes(beta_diff, 2, 1)
    
        tval_diff, clusters, cluster_p_values, _ = st_clust_1s_ttest(
            testdata,
            n_jobs=param["njobs"],
            threshold=cluster_threshold,
            adjacency=connect,
            n_permutations=param['nperms'],
            buffer_size=None
        )
    
        pvals_diff = np.ones_like(tval_diff)
        for c, p_val in zip(clusters, cluster_p_values):
            pvals_diff[c] = p_val
    
        np.save(opj(outpath, f'ols_2ndlevel_tval_diff_{label}_noz.npy'), tval_diff)
        np.save(opj(outpath, f'ols_2ndlevel_pval_diff_{label}_noz.npy'), pvals_diff)


    #------------------------------------------------------------------------------------------------------
    np.save(opj(outpath, 'included_subjects.npy'),
            np.array(included_subjects, dtype=object))


    # ---------------------------------------------------------------------
    # Option 1: ROI-level R² comparison (raw vs drift)
    # ---------------------------------------------------------------------
    print("\nComputing ROI-level R² comparisons (raw vs drift)...")
    
    roi_chs = ['Fz','FCz','POz','Cz','CPz','Pz', 'Oz']   # LPP-ish ROI; adjust if needed
    tmin, tmax = 0.4, 0.8           # seconds
    
    R2_rows = []
    
    for pa in part_1:
        print(f"R² ROI: processing {pa}")
        # --- Recreate cleaned epochs & behavioural table for this subject ---
        df2 = epo_1_filtered_combined[epo_1_filtered_combined['participant_id'] == pa]
        mod2 = part_1_dat[part_1_dat['participant'] == pa]
    
        # Load epochs (decision phase here)
        epo = mne.read_epochs(
            opj(basepath, pa, 'eeg', 'erps',
                pa + '_decision_cues_singletrials-epo.fif')
        )
        epo_cop = epo.copy()
    
        # Match trials
        matching = epo_cop.metadata['trialsnum'].isin(df2['trialsnum'])
        epo_filt = epo_cop[matching]
    
        # Downsample
        if epo_filt.info['sfreq'] != param['testresampfreq']:
            epo_filt = epo_filt.resample(param['testresampfreq'])
    
        # Drop bad trials
        goodtrials = np.where(epo_filt.metadata['badtrial'] == 0)[0]
        df2_sub = df2.iloc[goodtrials].reset_index(drop=True)
        mod2_sub = mod2.iloc[goodtrials].reset_index(drop=True)
        epo_filt = epo_filt[goodtrials]
    
        # Z-score EEG across trials
        scale = Scaler(scalings='mean')
        epo_z = mne.EpochsArray(scale.fit_transform(epo_filt.get_data()),
                                epo_filt.info)
    
        if len(df2_sub) < 5:
            print(f"Skipping {pa} for R² (too few trials: {len(df2_sub)})")
            continue
    
        # RT column
        if "rt" in mod2_sub.columns:
            rt_col = "rt"
        else:
            raise ValueError(f"No RT column in mod_data for {pa}. Columns: {mod2_sub.columns}")
    
        # --- Build y: mean EEG in ROI and time window ---
        picks = mne.pick_channels(epo_z.info['ch_names'], roi_chs)
        tmask = (epo_z.times >= tmin) & (epo_z.times <= tmax)
    
        data_roi = epo_z.get_data()[:, picks][:, :, tmask]  # trials x ch x time
        y = data_roi.mean(axis=(1, 2))                      # (n_trials,)
    
        # --- Compare raw vs drift for each attribute ---
        label_list = ['pain', 'money', 'interaction']
        for raw_name, v_name, attr_label in zip(raw_regcols, v_regcols, label_list):
    
            if raw_name not in mod2_sub.columns or v_name not in mod2_sub.columns:
                print(f"Skipping {attr_label} for {pa}: {raw_name} or {v_name} not in dataframe")
                continue
    
            vals_raw = mod2_sub[raw_name].to_numpy(dtype=float)
            vals_v   = mod2_sub[v_name].to_numpy(dtype=float)
            vals_rt  = mod2_sub[rt_col].to_numpy(dtype=float)
    
            keep = (
                np.isfinite(vals_raw) &
                np.isfinite(vals_v) &
                np.isfinite(vals_rt) &
                np.isfinite(y)
            )
            if keep.sum() < 5:
                print(f"Skipping {attr_label} for {pa}: only {keep.sum()} valid trials")
                continue
    
            yk = y[keep]
            X_raw = np.column_stack([np.ones(keep.sum()), vals_raw[keep], vals_rt[keep]])
            X_v   = np.column_stack([np.ones(keep.sum()), vals_v[keep],   vals_rt[keep]])
    
            # Fit linear regression via least-squares
            beta_raw, _, _, _ = np.linalg.lstsq(X_raw, yk, rcond=None)
            pred_raw = X_raw @ beta_raw
    
            beta_v, _, _, _ = np.linalg.lstsq(X_v, yk, rcond=None)
            pred_v = X_v @ beta_v
    
            ss_tot = np.sum((yk - yk.mean())**2)
            ss_res_raw = np.sum((yk - pred_raw)**2)
            ss_res_v   = np.sum((yk - pred_v)**2)
    
            R2_raw = 1.0 - ss_res_raw / ss_tot if ss_tot > 0 else np.nan
            R2_v   = 1.0 - ss_res_v   / ss_tot if ss_tot > 0 else np.nan
    
            R2_rows.append(dict(
                participant=pa,
                attribute=attr_label,   # 'pain','money','interaction'
                R2_raw=R2_raw,
                R2_v=R2_v,
                delta_R2=R2_v - R2_raw
            ))
    
    # Save table for group-level stats (paired t-tests per attribute, etc.)
    if len(R2_rows) > 0:
        R2_df = pd.DataFrame(R2_rows)
        R2_df.to_csv(opj(outpath, 'ROI_R2_raw_vs_v.csv'), index=False)
        print("Saved ROI_R2_raw_vs_v.csv in", outpath)
    
    
# --------------------------------------------------------------------------
# -------------------------- VERSION 4 (RT BINS) ---------------------------

elif version == 4:
    print("\n RT-stratified massunivariate code (Version 4)")

    included_subjects = []
    skipped_subjects = []

    # bins: slow  medium  fast
    bin_labels = ["fast", "medium", "slow"]  # note ordering here

    # storing betas + epochs per bin
    betas_bins = {
        "fast":   [[] for _ in regvars],
        "medium": [[] for _ in regvars],
        "slow":   [[] for _ in regvars]
    }
    all_epos_bins = {
        "fast":   [[] for _ in regvars],
        "medium": [[] for _ in regvars],
        "slow":   [[] for _ in regvars]
    }

    for pa in part_1:
        print(f"\n--- Processing subject {pa} (Version 4) ---")

        df2 = epo_1_filtered_combined[epo_1_filtered_combined['participant_id'] == pa]
        mod2 = part_1_dat[part_1_dat['participant'] == pa]

        # load epochs (decision-phase ERPs)
        epo = mne.read_epochs(
            opj(basepath, pa, 'eeg', 'erps',
                pa + '_decision_cues_singletrials-epo.fif')
        )
        epo_cop = epo.copy()

        # match trials
        matching = epo_cop.metadata['trialsnum'].isin(df2['trialsnum'])
        epo_filt = epo_cop[matching]

        # downsample
        if epo_filt.info['sfreq'] != param['testresampfreq']:
            epo_filt = epo_filt.resample(param['testresampfreq'])

        # remove bad trials
        goodtrials = np.where(epo_filt.metadata['badtrial'] == 0)[0]
        df2 = df2.iloc[goodtrials]
        mod2 = mod2.iloc[goodtrials]
        epo_filt = epo_filt[goodtrials]

        # if too few trials skip subject
        if len(df2) < 10:
            print(f"Skipping {pa}: too few trials ({len(df2)})")
            skipped_subjects.append(pa)
            continue

        # choose RT column
        if "choice_rt" in mod2.columns:
            rt_vals = mod2["choice_rt"].to_numpy()
        elif "rt" in mod2.columns:
            rt_vals = mod2["rt"].to_numpy()
        else:
            raise ValueError(f"No RT column found in mod_data! Columns: {mod2.columns}")

        # compute tertile cutoffs
        q33, q66 = np.quantile(rt_vals, [0.33, 0.66])

        # assign bins
        rt_bin = np.full(len(rt_vals), "medium", dtype=object)
        rt_bin[rt_vals <= q33] = "fast"
        rt_bin[rt_vals >= q66] = "slow"

        # z-score EEG
        scale = Scaler(scalings='mean')
        epo_z = mne.EpochsArray(
            scale.fit_transform(epo_filt.get_data()),
            epo_filt.info
        )

        subject_used = False

        # run analysis separately for fast / medium / slow bins
        for bin_name in bin_labels:
            idx_bin = np.where(rt_bin == bin_name)[0]

            if len(idx_bin) < 5:
                print(f"  Bin '{bin_name}' skipped (<5 trials)")
                continue

            df_bin = mod2.iloc[idx_bin].copy()
            epo_bin = epo_z.copy()[idx_bin]
            epo_bin_keep = epo_filt.copy()[idx_bin]

            for r_idx, regvar in enumerate(regvars):

                vals = df_bin[regvar].to_numpy(float)
                if np.std(vals) == 0:
                    print(f"  {regvar} in bin {bin_name}: zero variance → skip")
                    continue

                # z-score regressor
                df_bin[regvar + "_z"] = stats.zscore(vals)
                df_bin["Intercept"] = 1.0

                design = df_bin[["Intercept", regvar + "_z"]]

                res = mne.stats.linear_regression(
                    epo_bin, design,
                    names=["Intercept", regvar + "_z"]
                )

                beta_reg = res[regvar + "_z"].beta

                betas_bins[bin_name][r_idx].append(beta_reg)
                all_epos_bins[bin_name][r_idx].append(epo_bin_keep)

                subject_used = True

        if not subject_used:
            skipped_subjects.append(pa)
        else:
            included_subjects.append(pa)

    print("\nIncluded subjects:", included_subjects)
    print("Skipped subjects:", skipped_subjects)

    # ---------------------------------------------------------------------
    # second-level cluster tests
    # ---------------------------------------------------------------------

    for bin_name in bin_labels:
        print(f"\nCluster tests for RT bin: {bin_name}")

        # for adjacency we need an example info object
        example_info = None
        for r_idx in range(len(regvars)):
            if len(betas_bins[bin_name][r_idx]) > 0:
                example_info = betas_bins[bin_name][r_idx][0].info
                break

        if example_info is None:
            print(f"No data at all in bin {bin_name}, skipping cluster tests")
            continue
        
        connect, names = mne.channels.find_ch_adjacency(example_info, ch_type='eeg')

        # For each regressor inside this bin
        for r_idx, regvar in enumerate(regvars):
            subj_betas = betas_bins[bin_name][r_idx]

            if len(subj_betas) < 2:
                print(f" Bin {bin_name}, regvar {regvar} <2 subjects. No cluster test")
                continue

            print(f"Bin {bin_name}, regvar {regvar} n_subj = {len(subj_betas)}")

            # Stack data: (n_subj, n_chan, n_time)
            data = np.stack([b.data for b in subj_betas])
            # Rearrange for cluster test: (n_subj, n_time, n_chan)
            testdata = np.swapaxes(data, 2, 1)

            # compute bin-specific t-threshold
            if not isinstance(param['cluster_threshold'], dict):
                p_thresh = param['cluster_threshold'] / 2
                n_samples = testdata.shape[0]
                cluster_threshold = -stats.t.ppf(p_thresh, n_samples - 1)
            else:
                cluster_threshold = param['cluster_threshold']

            tval, clusters, cluster_p_values, _ = st_clust_1s_ttest(
                testdata,
                n_jobs=param["njobs"],
                threshold=cluster_threshold,
                adjacency=connect,
                n_permutations=param['nperms'],
                buffer_size=None
            )

            pvals = np.ones_like(tval)
            for c, p_val in zip(clusters, cluster_p_values):
                pvals[c] = p_val
            # saving
            np.save(opj(outpath, f'{bin_name}_tvals_{regvar}.npy'), tval)
            np.save(opj(outpath, f'{bin_name}_pvals_{regvar}.npy'), pvals)

            # save epochs for this bin + regvar
            epo_list = all_epos_bins[bin_name][r_idx]
            if len(epo_list) > 0:
                epo_save = mne.concatenate_epochs(epo_list)
                epo_save.save(
                    opj(outpath, f'{bin_name}_allepochs-epo_{regvar}.fif'),
                    overwrite=True
                )

    print("\n Version 4 RT-stratified cluster tests done ;)")



### old
#-----------------------------------------------------------------------------------------------------------------------------
# for pa in part_1:
#     print(f"\n--- Processing {pa} ---")
#     df2 = epo_1_filtered_combined[epo_1_filtered_combined['participant_id'] == pa]
#     mod2 = part_1_dat[part_1_dat['participant'] == pa]
    
#     if version == 1:
#         epo = mne.read_epochs(opj(basepath,  pa, 'eeg', 'erps',                        # for averaging over more electrodes: 'eeg', 'erps_2'
#                               pa + '_decision_cues_singletrials-epo.fif'))
#         epo_cop = epo.copy()
#     elif version == 2:
#         epo = mne.read_epochs(opj(basepath,  pa, 'eeg', 'erps_passive',                        # for averaging over more electrodes: 'eeg', 'erps_2'
#                               pa + '_passive_cues_singletrials-epo.fif'))
#         epo_cop = epo.copy()
    
#     matching= epo_cop.metadata['trialsnum'].isin(df2['trialsnum'])
    
#     # Filter the Epochs object and metadata to keep matching trials
#     epo_filt = epo_cop[matching]
    
#     # Update metadata in filtered Epochs object
#     #epo_filt.metadata = epo_filt.metadata[matching]
    
#     # downsample
#     if epo_filt.info['sfreq'] != param['testresampfreq']:
#         epo_filt = epo_filt.resample(param['testresampfreq'])

#     # Drop bad trials
#     goodtrials = np.where(epo_filt.metadata['badtrial'] == 0)[0]
#     df2 = df2.iloc[goodtrials]
#     mod2 = mod2.iloc[goodtrials]
#     epo_filt = epo_filt[goodtrials]
    
#     scale = Scaler(scalings='mean')
#     epo_z = mne.EpochsArray(scale.fit_transform(epo_filt.get_data()),
#                             epo_filt.info)
     
#     matching = epo_cop.metadata['trialsnum'].isin(df2['trialsnum'])
#     print(matching)

#     if matching.sum() < 5:
#         print(f"Skipping {pa}: only {matching.sum()} matching trials")
#         skipped_subjects.append(pa)
#         continue
#     # small_value_threshold = 1e-3

#     # # exclude trials with very small data
#     # def filter_small_trials(epochs, threshold): 
#     #     keep_mask = np.max(np.abs(epochs.get_data()), axis=(1, 2)) > threshold
#     #     # mask to keep valid epochs
#     #     return epochs[keep_mask]
    
#     # betasnp = []
    
#     # for idx, regvar in enumerate(regvars):
#     #     keep = np.where(~np.isnan(mod2[regvar]))[0]
#     #     df_reg = mod2.iloc[keep]
#     #     epo_reg = epo_z.copy()[keep]
#     #     epo_keep = epo_filt.copy()[keep]

#     #     epo_reg = filter_small_trials(epo_reg, small_value_threshold)

#     betasnp = []
# #    for idx, regvar in enumerate(regvars):
# #        keep = np.where(~np.isnan(mod2[regvar]))[0]
# #        df_reg = mod2.iloc[keep]
# #        epo_reg = epo_z.copy()[keep]
# #        epo_keep = epo_filt.copy()[keep]
# #        
# #        df_reg[regvar + '_z'] = stats.zscore(df_reg[regvar])
# #
# #        # Add an intercept to the matrix
# #        epo_keep.metadata = df_reg.assign(Intercept=1)
# #        epo_reg.metadata = df_reg.assign(Intercept=1)
# #
# #        # Perform regression
# #        names = ["Intercept"] + [regvar + '_z']
# #        res = mne.stats.linear_regression(epo_reg, epo_reg.metadata[names],
# #                                          names=names)

#     betasnp = []
#     subject_has_regressors = False
    
#     for idx, regvar in enumerate(regvars):
    
#         vals = mod2[regvar].to_numpy(dtype=float)
#         keep = np.where(np.isfinite(vals))[0]

#         if len(keep) < 2:
#             print(f"  Skipping {regvar}: only {len(keep)} finite trials")
#             continue
        
#         df_reg = mod2.iloc[keep].copy()
#         epo_reg = epo_z.copy()[keep]
#         epo_keep = epo_filt.copy()[keep]

#         if np.nanstd(df_reg[regvar]) == 0:
#             print(f"  Skipping {regvar}: zero variance")
#             continue
#         df_reg[regvar + "_z"] = stats.zscore(df_reg[regvar])
#         design = df_reg.assign(Intercept=1)[["Intercept", regvar + "_z"]]

#         if not np.all(np.isfinite(design.to_numpy())):
#             print(f"  Skipping {regvar}: contains NaN")
#             continue

#         df_new = epo_keep.metadata.copy()
#         df_new[regvar] = df_reg[regvar].values
#         epo_keep.metadata = df_new.reset_index(drop=True)

#         all_epos[idx].append(epo_keep)

#         res = mne.stats.linear_regression(
#             epo_reg, design, names=["Intercept", regvar + "_z"]
#         )

#         betas[idx].append(res[regvar + "_z"].beta)
#         betasnp.append(res[regvar + "_z"].beta.data)

#         subject_has_regressors = True
#         print(f"Metadata columns for {pa}, regvar '{regvar}':")
#         print(epo_keep.metadata.columns.tolist())


#     if not subject_has_regressors:
#         print(f"Skipping {pa}: no valid regressors")
#         skipped_subjects.append(pa)
#         continue

#     included_subjects.append(pa)
#     allbetasnp.append(np.stack(betasnp))
#     print(f"Included {pa}")

