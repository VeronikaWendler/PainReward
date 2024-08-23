# Massunivariate Analysis and Second level test on betas


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

# Set bids directory
basepath = "D:/Aberdeen_Uni_June24/MPColl_Lab/All_Files_Relevant_For_Git/EEG/PainReward_sub-001-050/painrewardeegdata/derivatives"

# Outpath for analysis
outpath = opj(basepath, 'statistics')              # for averaging over more electrodes: 'statistics_2' 
if not os.path.exists(outpath):
    os.mkdir(outpath)
    

outpath = opj(outpath, 'erps_modelbased_v_sv_pain_para_Quest')
if not os.path.exists(outpath):
    os.mkdir(outpath)

# participants
part_csv = pd.read_csv('D:/Aberdeen_Uni_June24/MPColl_Lab/All_Files_Relevant_For_Git/EEG/PainReward_sub-001-050/painrewardeegdata/participants.tsv', sep='\t')
part = part_csv['participant_id'].unique().tolist()
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
    # Downsample to this frequency prior to analysis
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

mod_data = pd.read_csv('D:/Aberdeen_Uni_June24/MPColl_Lab/All_Files_Relevant_For_Git/Hddm_Docker_August_24/data_sets/data_with_sv_pain_para_Quest.csv')

#------------------------------------------------------------------------------------------------------------------------------------------------
# Massunivariate Regression from MP Code (for single regressors)

# Regressors
# regvars = ['full_v_sv_pain_para_contrib', 'full_a_sv_pain_para_contrib', 'full_t_sv_pain_para_contrib', 'full_z_sv_pain_para_contrib'] 
# regvarsnames = ['Full_DDM_v', 'Full_DDM_a', 'Full_DDM_t', 'Full_DDM_z'] 

# regvars = ['v_complex_v_sv_pain_para_contrib','v_complex_v_sv_money_contrib','v_complex_v_sv_pain_money_contrib']
# regvarsnames = ['v_sv_pain_para', 'v_sv_money', 'v_sv_pain_money']

# regvars = ['painlevel']
# regvarsnames = ['painlevel']

# regvars = ['v_sv_pain_para_Abslow_contrib', 'v_sv_pain_para_Absmid_contrib', 'v_sv_pain_para_Abshigh_contrib']
# regvarsnames = ['v_pain_Abslow', 'v_pain_Absmid', 'v_pain_Abshigh']

# regvars = ['v_sv_pain_para_contrib']
# regvarsnames = ['v_sv_pain_para_contrib']

regvars = ['sv_pain_para_SAI','sv_pain_para_TAI', 'sv_pain_para_PCS']         #'sv_pain_para_TAI', 
regvarsnames = ['sv_pain_para_SAI', 'sv_pain_para_TAI' , 'sv_pain_para_PCS']    #  'sv_pain_para_TAI' 
betas, betasnp = [], []

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
    epo = mne.read_epochs(opj(basepath,  p, 'eeg', 'erps',                   # for averaging over more electrodes: 'eeg', 'erps_2'
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

    # Initialize dataFrame for filtered block data for part
    epo_2_filtered = pd.DataFrame()

    # filter for unique participants in the behavioral frame
    for participant in df['participant'].unique():
        erps_p_df = epo_1.metadata[epo_1.metadata['participant_id'] == participant]
        df_unique = df[df['participant'] == participant]   

        # Loop through each block
        for block_x in df_unique['blocks.thisRepN'].unique():
            erps_block_df = erps_p_df[erps_p_df['blocks_idx'] == block_x]
            df_block_df = df_unique[df_unique['blocks.thisRepN'] == block_x]
                
            # keep only the rows where trialblocks col match trials.thisN col
            filtered_block_df = erps_block_df[erps_block_df['trialblocks'].isin(df_block_df['trials.thisN'])]            
            epo_2_filtered = pd.concat([epo_2_filtered, filtered_block_df], ignore_index=True)
    
    filtered_data.append(epo_2_filtered)

epo_2_filtered_combined = pd.concat(filtered_data, ignore_index=True)
#epo_2_filtered_combined.to_csv('D:/Aberdeen_Uni_June24/MPColl_Lab/All_Files_Relevant_For_Git/Hddm_Docker_August_24/data_sets/epo_2_filtered_combined')

part_2_dat = mod_data
part_2 = part_2_dat['participant'].unique().tolist()

part_2.sort()

#------------------------------------------------------------------------------------------------------------------------------------------------
# Massunivariate 

for pa in part_2:
    df2 = epo_2_filtered_combined[epo_2_filtered_combined['participant_id'] == pa]
    mod2 = part_2_dat[part_2_dat['participant'] == pa]
    
    epo = mne.read_epochs(opj(basepath,  pa, 'eeg', 'erps',                        # for averaging over more electrodes: 'eeg', 'erps_2'
                              pa + '_decision_cues_singletrials-epo.fif'))
    epo_cop = epo.copy()
    
    # 1: Matching Trials (trialsnum col)
    matching= epo_cop.metadata['trialsnum'].isin(df2['trialsnum'])
    
    # 2: Filter the Epochs object and metadata to keep matching trials
    epo_filt = epo_cop[matching]
    
    # Step 3: Update metadata in filtered Epochs object
    epo_filt.metadata = epo_filt.metadata[matching]
    
    # downsample if necessary
    if epo_filt.info['sfreq'] != param['testresampfreq']:
        epo_filt = epo_filt.resample(param['testresampfreq'])

    # Drop bad trials
    goodtrials = np.where(epo_filt.metadata['badtrial'] == 0)[0]
    df2 = df2.iloc[goodtrials]
    mod2 = mod2.iloc[goodtrials]
    epo_filt = epo_filt[goodtrials]
    
    # Robust standardize data before regression
    scale = Scaler(scalings='mean')
    epo_z = mne.EpochsArray(scale.fit_transform(epo_filt.get_data()),
                            epo_filt.info)
    
    # small_value_threshold = 1e-3

    # # exclude trials with very small data
    # def filter_small_trials(epochs, threshold): 
    #     keep_mask = np.max(np.abs(epochs.get_data()), axis=(1, 2)) > threshold
    #     # mask to keep valid epochs
    #     return epochs[keep_mask]
    
    # betasnp = []
    
    # for idx, regvar in enumerate(regvars):
    #     keep = np.where(~np.isnan(mod2[regvar]))[0]
    #     df_reg = mod2.iloc[keep]
    #     epo_reg = epo_z.copy()[keep]
    #     epo_keep = epo_filt.copy()[keep]

    #     epo_reg = filter_small_trials(epo_reg, small_value_threshold)

    betasnp = []
    for idx, regvar in enumerate(regvars):
        # Keep only rows with values on regressor
        keep = np.where(~np.isnan(mod2[regvar]))[0]
        df_reg = mod2.iloc[keep]
        epo_reg = epo_z.copy()[keep]
        epo_keep = epo_filt.copy()[keep]
        
        df_reg[regvar + '_z'] = stats.zscore(df_reg[regvar])

        # Add an intercept to the matrix
        epo_keep.metadata = df_reg.assign(Intercept=1)
        epo_reg.metadata = df_reg.assign(Intercept=1)

        # Perform regression
        names = ["Intercept"] + [regvar + '_z']
        res = mne.stats.linear_regression(epo_reg, epo_reg.metadata[names],
                                          names=names)

        # Collect betas
        betas[idx].append(res[regvar + '_z'].beta)
        betasnp.append(res[regvar + '_z'].beta.data)
        all_epos[idx].append(epo_keep)
    allbetasnp.append(np.stack(betasnp))

# Stack all data
allbetas = np.stack(allbetasnp)


# Grand average
beta_gavg = []
for idx, regvar in enumerate(regvars):
    beta_gavg.append(mne.grand_average(betas[idx]))

    

# # _________________________________________________________________
# # Second level test on betas

# _________________________________________________________________
# Second level test on betas

# Get channels connectivity
connect, names = mne.channels.find_ch_adjacency(epo_filt.info, ch_type='eeg')


# Get cluster entering threshold
if type(param['cluster_threshold']) is not dict:
    # Get cluster entering treshold
    p_thresh = param['cluster_threshold'] / 2  # two sided
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

    # Reshape p-values to match data
    pvals = np.ones_like(tval)
    for c, p_val in zip(clusters, cluster_p_values):
        pvals[c] = p_val

    # In a list for each regressor
    tvals.append(tval)
    pvalues.append(pvals)
    # Save for each regressor in case crash/stop
    np.save(opj(outpath, 'ols_2ndlevel_tval_' + regvar + '.npy'), tvals[-1])
    np.save(opj(outpath, 'ols_2ndlevel_pval_' + regvar + '.npy'), pvalues[-1])

# Stack and save
tvals = np.stack(tvals)
pvals = np.stack(pvalues)

np.save(opj(outpath, 'ols_2ndlevel_tvals.npy'), tvals)
np.save(opj(outpath, 'ols_2ndlevel_pvals.npy'), pvalues)
np.save(opj(outpath, 'ols_2ndlevel_betas.npy'), allbetas)

for idx, regvar in enumerate(regvars):
    epo_save = mne.concatenate_epochs(all_epos[idx])
    epo_save.save(opj(outpath, 'ols_2ndlevel_allepochs-epo_' + regvar + '.fif'),
                  overwrite=True)
np.save(opj(outpath, 'ols_2ndlevel_betasavg.npy'), beta_gavg)

