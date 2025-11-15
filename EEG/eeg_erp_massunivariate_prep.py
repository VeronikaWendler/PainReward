'''
 # @ : -*- coding: utf-8 -*-
 # @ Author: Michel-Pierre Coll (michel-pierre.coll@psy.ulaval.ca), edited by Veronika Wendler
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
version = 2    # version 1 is for decision and version 2 is for passive phase 

if version == 1:
    outpath = opj(outpath, 'erps_massuni_drift_mod_9_2')
    if not os.path.exists(outpath):
        os.mkdir(outpath)
elif version == 2:
    outpath = opj(outpath, 'erps_massuni_drift_mod_9_2_passive')
    if not os.path.exists(outpath):
        os.mkdir(outpath)
else:
    print("no version")
    
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

print("\n Subjects used:", common_participants)
print("EEG only:", eeg_participants - beh_participants)
print("Behavior only:", beh_participants - eeg_participants)

part = common_participants

part_1_dat = mod_data[mod_data["participant"].isin(part)]
part_1 = part
#mod_data = mod_data[mod_data['OV_value'] == 'high_OV']


#------------------------------------------------------------------------------------------------------------------------------------------------
# Massunivariate Regression from MP Code (for single regressors)

# Regressors
# regvars = ['full_v_sv_pain_para_contrib', 'full_a_sv_pain_para_contrib', 'full_t_sv_pain_para_contrib', 'full_z_sv_pain_para_contrib'] 
# regvarsnames = ['Full_DDM_v', 'Full_DDM_a', 'Full_DDM_t', 'Full_DDM_z'] 

# regvars = ['v_complex_v_sv_pain_para_contrib','v_complex_v_sv_money_contrib','v_complex_v_sv_pain_money_contrib']
# regvarsnames = ['v_sv_pain_para', 'v_sv_money', 'v_sv_pain_money']

# regvars = ['painlevel']
# regvarsnames = ['painlevel']
#
# regvars = ['v_sv_pain_para_Abslow_contrib', 'v_sv_pain_para_Absmid_contrib', 'v_sv_pain_para_Abshigh_contrib']
# regvarsnames = ['v_pain_Abslow', 'v_pain_Absmid', 'v_pain_Abshigh']

# regvars = ['sv_pain_para_SAI','sv_pain_para_TAI', 'sv_pain_para_PCS']         #'sv_pain_para_TAI', 
# regvarsnames = ['sv_pain_para_SAI', 'sv_pain_para_TAI' , 'sv_pain_para_PCS']    #  'sv_pain_para_TAI' 
#
regvars = ['v_pain_contrib','v_money_contrib','v_interaction_contrib']
regvarsnames = ['V_pain_contrib','V_money_contrib','V_interaction_contrib']

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

included_subjects = []
skipped_subjects = []


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


# --------------------------------------------------------------------
# Massunivariate with RT as covariate
# --------------------------------------------------------------------

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
    epo_cop = epo.copy()

    # Match trials
    matching = epo_cop.metadata['trialsnum'].isin(df2['trialsnum'])
    epo_filt = epo_cop[matching]

    # Downsample for stats
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

    # If there are too few trials after matching, skip subject
    if len(df2) < 5:
        print(f"Skipping {pa}: only {len(df2)} matching trials after cleaning")
        skipped_subjects.append(pa)
        continue


    if "choice_rt" in mod2.columns:
        rt_col = "choice_rt"
    elif "rt" in mod2.columns:
        rt_col = "rt"
    else:
        raise ValueError(f"No RT column found in mod_data! Columns: {mod2.columns}")

    betasnp = []
    subject_has_regressors = False

    for idx, regvar in enumerate(regvars):

        # keep trials where BOTH regressor and RT are finite
        vals_reg = mod2[regvar].to_numpy(dtype=float)
        vals_rt  = mod2[rt_col].to_numpy(dtype=float)
        keep = np.where(np.isfinite(vals_reg) & np.isfinite(vals_rt))[0]

        if len(keep) < 5:
            print(f"  Skipping {regvar}: only {len(keep)} valid trials (regvar+RT)")
            continue

        df_reg  = mod2.iloc[keep].copy()
        epo_reg = epo_z.copy()[keep]
        epo_keep = epo_filt.copy()[keep]

        # check variance
        if np.nanstd(df_reg[regvar]) == 0:
            print(f"  Skipping {regvar}: zero variance")
            continue
        if np.nanstd(df_reg[rt_col]) == 0:
            print(f"  Skipping {regvar}: RT has zero variance (subject {pa})")
            continue

        # Z-score predictors within subject
        df_reg[regvar + "_z"] = stats.zscore(df_reg[regvar].to_numpy(dtype=float))
        df_reg["RT_z"]        = stats.zscore(df_reg[rt_col].to_numpy(dtype=float))
        df_reg["Intercept"]   = 1.0

        design = df_reg[["Intercept", regvar + "_z", "RT_z"]]

        # safety check
        if not np.all(np.isfinite(design.to_numpy())):
            print(f"  Skipping {regvar}: design matrix has NaN/Inf")
            continue

        # update metadata of kept epochs (optional, just for reference)
        df_meta = epo_keep.metadata.reset_index(drop=True).copy()
        df_meta[regvar] = df_reg[regvar].values
        df_meta[rt_col] = df_reg[rt_col].values
        epo_keep.metadata = df_meta

        # Store epochs for second-level visualization
        all_epos[idx].append(epo_keep)

        # ------------------------
        # Run regression: EEG ~ Intercept + regvar + RT
        # ------------------------
        res = mne.stats.linear_regression(
            epo_reg, design,
            names=["Intercept", regvar + "_z", "RT_z"]
        )

        # beta for regressor of interest *controlling for RT*
        beta_reg = res[regvar + "_z"].beta
        betas[idx].append(beta_reg)
        betasnp.append(beta_reg.data)

        # (Optional) you could also inspect RT effects:
        # beta_rt = res["RT_z"].beta

        subject_has_regressors = True
        print(f"Metadata columns for {pa}, regvar '{regvar}':")
        print(epo_keep.metadata.columns.tolist())

    if not subject_has_regressors:
        print(f"Skipping {pa}: no valid regressors with RT for this subject")
        skipped_subjects.append(pa)
        continue

    included_subjects.append(pa)
    allbetasnp.append(np.stack(betasnp))
    print(f"Included {pa}")

# After this, the rest of your script (stacking allbetas, cluster test, saving)
# can remain exactly as you already have it.

# Stack all data
allbetas = np.stack(allbetasnp)

print(f"Total subjects: {len(part_1)}")
print(f"Included ({len(included_subjects)}): {included_subjects}")
print(f"Skipped  ({len(skipped_subjects)}): {skipped_subjects}")

# Grand average
beta_gavg = []
for idx, regvar in enumerate(regvars):
    beta_gavg.append(mne.grand_average(betas[idx]))

# Second level test on betas
# connectivity
connect, names = mne.channels.find_ch_adjacency(epo_filt.info, ch_type='eeg')

# Get cluster entering threshold
if type(param['cluster_threshold']) is not dict:
    p_thresh = param['cluster_threshold'] / 2  
    n_samples = allbetas.shape[0]
    param['cluster_threshold'] = -stats.t.ppf(p_thresh, n_samples - 1)


# Perform test for each regressor
tvals, pvalues = [], []
for idx, regvar in enumerate(regvars):
    # Reshape sub x time x vertices
    testdata = np.swapaxes(allbetas[:, idx, :, :], 2, 1)
    # data is (n_observations, n_times, n_vertices)
    tval, clusters, cluster_p_values, _ = st_clust_1s_ttest(testdata,
                                                            n_jobs=param["njobs"],
                                                            threshold=param['cluster_threshold'],
                                                            adjacency=connect,
                                                            n_permutations=param['nperms'],
                                                            buffer_size=None)

    # Reshape p-values
    pvals = np.ones_like(tval)
    for c, p_val in zip(clusters, cluster_p_values):
        pvals[c] = p_val

    # list for each regressor
    tvals.append(tval)
    pvalues.append(pvals)
    np.save(opj(outpath, 'ols_2ndlevel_tval_' + regvar + '.npy'), tvals[-1])
    np.save(opj(outpath, 'ols_2ndlevel_pval_' + regvar + '.npy'), pvalues[-1])

# Stack and save
tvals = np.stack(tvals)
pvals = np.stack(pvalues)

np.save(opj(outpath, 'ols_2ndlevel_tvals.npy'), tvals)
np.save(opj(outpath, 'ols_2ndlevel_pvals.npy'), pvals)
np.save(opj(outpath, 'ols_2ndlevel_betas.npy'), allbetas)

for idx, regvar in enumerate(regvars):
    epo_save = mne.concatenate_epochs(all_epos[idx])
    epo_save.save(opj(outpath, 'ols_2ndlevel_allepochs-epo_' + regvar + '.fif'),
                  overwrite=True)
np.save(opj(outpath, 'ols_2ndlevel_betasavg.npy'), beta_gavg)

