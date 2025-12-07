'''
 # @ : -*- coding: utf-8 -*-
 # @ Author: Michel-Pierre Coll (michel-pierre.coll@psy.ulaval.ca) & Veronika Wendler (2025)
 # @ Date: 2024
 # @ Description:
 
 1.set versions
 2.cleaning and z scoring
 3.Grand average & second-level cluster test (versions 1–3)
 4.Cluster test on beta differences (drift vs raw) (this is stupid, I should have taken more math courses)
 5.ROI-level R scquared comparison (raw vs drift)  (this is stupid, needs change, ignore)
 
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
outpath = opj(basepath, 'statistics_new')       
if not os.path.exists(outpath):
    os.mkdir(outpath)
    
# here for decision its just erps_massuni_drift_mod_9 and for passive it is: erps_massuni_drift_mod_9_2_passive
version = 5    # version 1 is for decision and version 2 is for passive phase 


if version == 1:
    outpath = opj(outpath, 'erps_massuni_drift_mod_9_passive')
    if not os.path.exists(outpath):
        os.mkdir(outpath)
if version == 2: # with RT as covariate
    outpath = opj(outpath, 'erps_massuni_drift_mod_9_RT')
    if not os.path.exists(outpath):
        os.mkdir(outpath)         
elif version == 3: #with RT covariate and doing the 3 GLMS to test the difference between raw pain and the drift of it
    outpath = opj(outpath, 'erps_massuni_drift_mod_9_RT_3GLMs')
    if not os.path.exists(outpath):
        os.mkdir(outpath)
elif version == 4: #with RT covariate
    outpath = opj(outpath, 'erps_massuni_drift_mod_9_RTbin')
    if not os.path.exists(outpath):
        os.mkdir(outpath)
elif version == 5:  #between subs
    outpath = opj(outpath, 'erps_massuni_drift_mod_9_subjectGLM')
    if not os.path.exists(outpath):
        os.mkdir(outpath)
elif version == 6:
    outpath = opj(outpath, 'erps_massuni_drift_mod_9_v6_beta_vs_drift')
    if not os.path.exists(outpath):
        os.mkdir(outpath)
elif version == 7: # this is the directory, I'll use for testing  pure sv_pain again (now that I made some changes to keep more participants)
    outpath = opj(outpath, 'erps_massuni_drift_mod_9_sv_pain_para')
    if not os.path.exists(outpath):
        os.mkdir(outpath)
elif version == 8:  # NEW: TFR ROI vs drift (between-subject)
    outpath = opj(outpath, 'tfr_mod_9_v8_drift_ROI')
    if not os.path.exists(outpath):
        os.mkdir(outpath)
elif version == 9:  # NEW: TFR trial-wise sv_pain_para betas
    outpath = opj(outpath, 'tfr_mod_9_v9_sv_pain_para_RT_control')
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
mod_data_path = PROJECT_DIR / "Hddm_Docker_August_24" / "figures_dir" / "painreward_behavioural_data_mod_9" / "diagnostics" / "v_pain_money_interaction.csv"
mod_data = pd.read_csv(mod_data_path, sep=None, engine="python")
mod_data["rt"] = mod_data["choice_resp.rt"]
mod_data["interaction"] = mod_data["moneylevel"]*mod_data["painlevel"]
mod_data["trialsnum"] = (
    mod_data["blocks.thisRepN"].astype(int) * 25
    + mod_data["trials.thisN"].astype(int)
    + 1
)
# same file but for threshold (a) parameters
mod_data_a_path = PROJECT_DIR / "Hddm_Docker_August_24" / "figures_dir" / "painreward_behavioural_data_mod_10" / "diagnostics" / "a_pain_money_interaction.csv"
mod_data_a = pd.read_csv(mod_data_a_path, sep=None, engine="python")


# Subjects in EEG
eeg_participants = set(part) # should be 1 - 50
# Subjects in HDDM CSV (should be 38 in total)
beh_participants = set(mod_data["participant"].unique())
# Subjects present in both datasets
common_participants = sorted(list(eeg_participants & beh_participants))

print("\n Subjects:", common_participants) # should be 38
print(len(common_participants))

part = common_participants

part_1_dat = mod_data[mod_data["participant"].isin(part)]
part_1 = part
#mod_data = mod_data[mod_data['OV_value'] == 'high_OV']


#------------------------------------------------------------------------------------------------------------------------------------------------
# Massunivariate Regression with 3 GLMs
#
raw_regcols = ['painlevel', 'moneylevel', 'interaction']
v_regcols   = ['v_pain_contrib', 'v_money_contrib', 'v_interaction_contrib']

# full list of regressors to run GLMs on (each gets its own EEG GLM)
regvars = raw_regcols + v_regcols
regvarsnames = [
    'pain_raw', 'money_raw', 'interaction_raw',
    'V_pain_contrib', 'V_money_contrib', 'V_interaction_contrib'
]

if version == 7:
    regvars = ['sv_pain_para']
    regvarsnames = ['SV_pain_para']
    
#all_epos = [[] for i in range(len(regvars))]
#allbetasnp = []
#betas = [[] for i in range(len(regvars))]
part.sort()

#------------------------------------------------------------------------------------------------------------------------------------------------
# Creating the dataframes

#------------------------------------------------------------------------------------------------------------------------------------------------
# Creating the dataframes (only needed for versions 1–4)

if version in [1, 2, 3, 4, 7]:
    filtered_data = []
    for p in part:
        # data for this participant
        df = mod_data[mod_data['participant'] == p]
        
        # Load single epochs file
        if version == 1:
            epo = mne.read_epochs(opj(basepath,  p, 'eeg', 'erps_passive',                   
                                  p + '_passive_cues_singletrials-epo.fif'))
            epo_1 = epo.copy()
        elif version in [2, 3, 4, 7]:
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
# For versions  1 (decision), 2 (passive phase), 3 (RT as covariate)

if version in [1, 2, 3]:
    
    all_epos = [[] for i in range(len(regvars))]
    allbetasnp = []
    betas = [[] for i in range(len(regvars))]

    included_subjects = []
    skipped_subjects = []
    
    noz_dir = Path(outpath) / "NO_Zscoring"
    noz_dir.mkdir(parents=True, exist_ok=True)

    for pa in part_1:
        print(f"\n--- NOT Z-Scored Version: Processing {pa} ---")
        df2 = epo_1_filtered_combined[epo_1_filtered_combined['participant_id'] == pa]
        mod2 = part_1_dat[part_1_dat['participant'] == pa]

        # Load epochs
        if version == 1:
            epo = mne.read_epochs(
                opj(basepath, pa, 'eeg', 'erps_passive',
                    pa + '_passive_cues_singletrials-epo.fif')
            )
        elif version == 2:
            epo = mne.read_epochs(
                opj(basepath, pa, 'eeg', 'erps',
                    pa + '_decision_cues_singletrials-epo.fif')
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
            
            # NO Z-scoring for painlevel and drift pain regressors (reason is, I fear that due to drift only being scaled with pain I'm losing its influence as z-scroing could remove that constant) (we could z score RT)
            
            # df_reg["RT_z"] = stats.zscore(df_reg[rt_col].to_numpy(dtype=float))
            # df_reg[regvar + "_z"] = stats.zscore(df_reg[regvar].to_numpy(dtype=float))

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
            
            # regression: EEG ~ Intercept + regvar + RT (or RT_Z)
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
        
        
        np.save(noz_dir / f'ols_2ndlevel_tval_noz_{regvar}.npy', tvals[-1])
        np.save(noz_dir / f'ols_2ndlevel_pval_noz_{regvar}.npy', pvalues[-1])

    # Stack and save group-level results
    tvals = np.stack(tvals)
    pvals = np.stack(pvalues)

    np.save(noz_dir / f'ols_2ndlevel_tvals_noz.npy', tvals)
    np.save(noz_dir / f'ols_2ndlevel_pvals_noz.npy', pvals)
    np.save(noz_dir / f'ols_2ndlevel_betas_noz.npy', allbetas)

    for idx, regvar in enumerate(regvars):
        epo_save = mne.concatenate_epochs(all_epos[idx])
        epo_save.save(noz_dir / f'ols_2ndlevel_allepochs-epo_noz_{regvar}.fif', overwrite=True)

    np.save(noz_dir / f'ols_2ndlevel_betasavg_noz.npy', beta_gavg)
    
    
    #---------------------------------------------------------------------------------------------------
    # ---------------------------------------------------------------------
    # Cluster test on beta differences (drift vs raw)
    # The idea is to test what topographical sig. effects can be explained by drift alone
    # Therefore, an idea is to get the difference between teh scaled drift rate*painlevel and the pure painlevel
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
    
        # shape for st_clust (n_subj, n_times, n_channels)
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
    
        np.save(noz_dir / f'ols_2ndlevel_tval_diff_{label}_noz.npy', tval_diff)
        np.save(noz_dir / f'ols_2ndlevel_pval_diff_{label}_noz.npy', pvals_diff)


    #------------------------------------------------------------------------------------------------------
    np.save(noz_dir / f'included_subjects.npy', np.array(included_subjects, dtype=object))


    # ---------------------------------------------------------------------
    # ROI-level R scquared comparison (raw vs drift)
    # ---------------------------------------------------------------------
    print("\n NOT Z-Scored version: Computing ROI-level R² comparisons (raw vs drift)...")
    
    roi_chs = ['Fz','FCz','POz','Cz','CPz','Pz', 'Oz']   # LPP 
    tmin, tmax = 0.4, 0.8           
    
    R2_rows = []
    
    for pa in part_1:
        print(f"R² ROI: processing {pa}")
        # --- Recreate cleaned epochs & behavioural table for this subject ---
        df2 = epo_1_filtered_combined[epo_1_filtered_combined['participant_id'] == pa]
        mod2 = part_1_dat[part_1_dat['participant'] == pa]
    
        #decision phase epochs
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
    
        # mean EEG in ROI and time window
        picks = mne.pick_channels(epo_z.info['ch_names'], roi_chs)
        tmask = (epo_z.times >= tmin) & (epo_z.times <= tmax)
    
        data_roi = epo_z.get_data()[:, picks][:, :, tmask]  # trials x ch x time
        y = data_roi.mean(axis=(1, 2))                      # (n_trials,)
    
        # Compare raw vs drift for each attribute 
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
        R2_df.to_csv(noz_dir / f'ROI_R2_raw_vs_v.csv', index=False)
        print("Saved ROI_R2_raw_vs_v.csv in", noz_dir)
    
    ##########################################################################################
    ##########################################################################################

    # Z scored version
    z_dir = Path(outpath) / "Zscoring"
    z_dir.mkdir(parents=True, exist_ok=True)

    all_epos = [[] for _ in range(len(regvars))]
    allbetasnp = []
    betas = [[] for _ in range(len(regvars))]
    included_subjects = []
    skipped_subjects = []
    
    for pa in part_1:
        print(f"\n--- YES: Z-Scored Version: Processing {pa} ---")
        df2 = epo_1_filtered_combined[epo_1_filtered_combined['participant_id'] == pa]
        mod2 = part_1_dat[part_1_dat['participant'] == pa]

        # Load epochs
        if version == 1:
            epo = mne.read_epochs(
                opj(basepath, pa, 'eeg', 'erps_passive',
                    pa + '_passive_cues_singletrials-epo.fif')
            )
        elif version == 2:
            epo = mne.read_epochs(
                opj(basepath, pa, 'eeg', 'erps',
                    pa + '_decision_cues_singletrials-epo.fif')
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

        # RT column - checking what's in there again (choice.rt or so)
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

            # Z-score predictors
            df_reg[regvar + "_z"] = stats.zscore(df_reg[regvar].to_numpy(dtype=float))
            df_reg["RT_z"] = stats.zscore(df_reg[rt_col].to_numpy(dtype=float))
            df_reg["Intercept"] = 1.0

            design = df_reg[["Intercept", regvar + "_z", "RT_z"]]

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
                names=["Intercept", regvar + "_z", "RT_z"]
            )

            # beta for regressor 
            beta_reg = res[regvar + "_z"].beta
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
        
        z_dir = Path(outpath) / "Zscoring"
        z_dir.mkdir(parents=True, exist_ok=True)
        
        np.save(z_dir / f'ols_2ndlevel_tval_{regvar}.npy', tvals[-1])
        np.save(z_dir / f'ols_2ndlevel_pval_{regvar}.npy', pvalues[-1])

    # Stack and save group-level results
    tvals = np.stack(tvals)
    pvals = np.stack(pvalues)

    np.save(z_dir / f'ols_2ndlevel_tvals.npy', tvals)
    np.save(z_dir / f'ols_2ndlevel_pvals.npy', pvals)
    np.save(z_dir / f'ols_2ndlevel_betas.npy', allbetas)

    for idx, regvar in enumerate(regvars):
        epo_save = mne.concatenate_epochs(all_epos[idx])
        epo_save.save(z_dir / f'ols_2ndlevel_allepochs-epo_{regvar}.fif', overwrite=True)

    np.save(z_dir / f'ols_2ndlevel_betasavg.npy', beta_gavg)
    
    
    #---------------------------------------------------------------------------------------------------
    # ---------------------------------------------------------------------
    # Cluster test on beta differences (drift vs raw)
    # The idea is to test what topographical sig. effects canbe explained by drift alone
    # Therefore, an idea is to get the difference between teh scaled drift rate*painlevel and the pure painlevel
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
    
        np.save(z_dir / f'ols_2ndlevel_tval_diff_{label}.npy', tval_diff)
        np.save(z_dir / f'ols_2ndlevel_pval_diff_{label}.npy', pvals_diff)


    #------------------------------------------------------------------------------------------------------
    np.save(z_dir / f'included_subjects.npy', np.array(included_subjects, dtype=object))


    # ---------------------------------------------------------------------
    # ROI-level R scquared comparison (raw vs drift)
    # ---------------------------------------------------------------------
    print("\n Z-Scored version: Computing ROI-level R² comparisons (raw vs drift)...")
    
    roi_chs = ['Fz','FCz','POz','Cz','CPz','Pz', 'Oz']   # LPP 
    tmin, tmax = 0.4, 0.8           
    
    R2_rows = []
    
    for pa in part_1:
        
        print(f"R² ROI: processing {pa}")
        # --- Recreate cleaned epochs & behavioural table for this subject ---
        df2 = epo_1_filtered_combined[epo_1_filtered_combined['participant_id'] == pa]
        mod2 = part_1_dat[part_1_dat['participant'] == pa]
    
        #decision phase epochs
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
    
        # mean EEG in ROI and time window
        picks = mne.pick_channels(epo_z.info['ch_names'], roi_chs)
        tmask = (epo_z.times >= tmin) & (epo_z.times <= tmax)
    
        data_roi = epo_z.get_data()[:, picks][:, :, tmask]  # trials x ch x time
        y = data_roi.mean(axis=(1, 2))                      # (n_trials,)
    
        # Compare raw vs drift for each attribute 
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
        R2_df.to_csv(z_dir / f'ROI_R2_raw_vs_v.csv', index=False)
        print("Saved ROI_R2_raw_vs_v.csv in", z_dir)
        
        # ---------------------------------------------------------------------
    # Between-subject correlation: LPP β(pain) vs HDDM v_pain
    # ---------------------------------------------------------------------
    print("\n Between-subject LPP beta painlevel vs v_pain ")


    pain_reg_name = 'painlevel'      
    if pain_reg_name not in regvars:
        raise ValueError(f"{pain_reg_name} not found in regvars: {regvars}")
    pain_idx = regvars.index(pain_reg_name)

    betas_pain = allbetas[:, pain_idx, :, :]   # (n_subj, n_chan, n_time)

    roi_chs = ['Fz', 'FCz', 'Cz', 'CPz', 'Pz', 'POz', 'Oz']
    tmin, tmax = 0.4, 0.8

    ch_names = epo_filt.info['ch_names']   
    times = epo_filt.times

    picks = mne.pick_channels(ch_names, roi_chs)
    tmask = (times >= tmin) & (times <= tmax)

   
    beta_LPP_pain = betas_pain[:, picks][:, :, tmask].mean(axis=(1, 2))

   
    v_pain_df = (
        mod_data[mod_data["participant"].isin(included_subjects)]
        .groupby("participant")["v_painlevel_subj"]
        .mean()
        .reindex(included_subjects)   
    )
    v_pain = v_pain_df.to_numpy(dtype=float)

    from scipy.stats import pearsonr
    r, p = pearsonr(beta_LPP_pain, v_pain)
    print(f"LPP β(pain, 400–800 ms, LPP ROI) vs v_pain:")
    print(f"  r = {r:.3f}, p = {p:.3g}, n = {len(included_subjects)}")

    between_df = pd.DataFrame({
        "participant": included_subjects,
        "beta_LPP_pain": beta_LPP_pain,
        "v_pain": v_pain
    })
    between_df.to_csv(z_dir / "between_subj_LPPpain_vs_vpain.csv", index=False)
    print("Saved between-subject data to", z_dir / "between_subj_LPPpain_vs_vpain.csv")
    
    
    #-------------------PARTIAL Z-SCORED VERSION--------------------------------
    ##########################################################################################
    ##########################################################################################

    # Partial Z scored version
    partz_dir = Path(outpath) / "PartZscoring"
    partz_dir.mkdir(parents=True, exist_ok=True)
    all_epos = [[] for _ in range(len(regvars))]
    allbetasnp = []
    betas = [[] for _ in range(len(regvars))]
    included_subjects = []
    skipped_subjects = []

    for pa in part_1:
        print(f"\n--- YES Partially: Z-Scored Version: Processing {pa} ---")
        df2 = epo_1_filtered_combined[epo_1_filtered_combined['participant_id'] == pa]
        mod2 = part_1_dat[part_1_dat['participant'] == pa]

        # Load epochs
        if version == 1:
            epo = mne.read_epochs(
                opj(basepath, pa, 'eeg', 'erps_passive',
                    pa + '_passive_cues_singletrials-epo.fif')
            )
        elif version == 2:
            epo = mne.read_epochs(
                opj(basepath, pa, 'eeg', 'erps',
                    pa + '_decision_cues_singletrials-epo.fif')
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

        # z -score only EEG and RT col
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

            df_reg["RT_z"] = stats.zscore(df_reg[rt_col].to_numpy(dtype=float))
            df_reg["Intercept"] = 1.0
            
            # Use the raw regressor and z- RT
            design = df_reg[["Intercept", regvar, "RT_z"]]
            
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
            
            # regression: EEG ~ Intercept + regvar + RT (or RT_Z)
            res = mne.stats.linear_regression(
                epo_reg, design,
                names=["Intercept", regvar, "RT_z"]
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

    print(f"Part Z:Total subjects: {len(part_1)}")
    print(f"Part Z:Included ({len(included_subjects)}): {included_subjects}")
    print(f"Part Z: Skipped  ({len(skipped_subjects)}): {skipped_subjects}")

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
        
        partz_dir = Path(outpath) / "PartZscoring"
        partz_dir.mkdir(parents=True, exist_ok=True)
        
        np.save(partz_dir / f'ols_2ndlevel_tval_{regvar}.npy', tvals[-1])
        np.save(partz_dir / f'ols_2ndlevel_pval_{regvar}.npy', pvalues[-1])

    # Stack and save group-level results
    tvals = np.stack(tvals)
    pvals = np.stack(pvalues)

    np.save(partz_dir / f'ols_2ndlevel_tvals.npy', tvals)
    np.save(partz_dir / f'ols_2ndlevel_pvals.npy', pvals)
    np.save(partz_dir / f'ols_2ndlevel_betas.npy', allbetas)

    for idx, regvar in enumerate(regvars):
        epo_save = mne.concatenate_epochs(all_epos[idx])
        epo_save.save(partz_dir / f'ols_2ndlevel_allepochs-epo_{regvar}.fif', overwrite=True)

    np.save(partz_dir / f'ols_2ndlevel_betasavg.npy', beta_gavg)
    
    
    #---------------------------------------------------------------------------------------------------
    # ---------------------------------------------------------------------
    # Cluster test on beta differences (drift vs raw)
    # The idea is to test what topographical sig. effects canbe explained by drift alone
    # Therefore, an idea is to get the difference between teh scaled drift rate*painlevel and the pure painlevel
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
    
        np.save(partz_dir / f'ols_2ndlevel_tval_diff_{label}.npy', tval_diff)
        np.save(partz_dir / f'ols_2ndlevel_pval_diff_{label}.npy', pvals_diff)


    #------------------------------------------------------------------------------------------------------
    np.save(partz_dir / f'included_subjects.npy', np.array(included_subjects, dtype=object))


    # ---------------------------------------------------------------------
    # ROI-level R scquared comparison (raw vs drift)
    # ---------------------------------------------------------------------
    print("\n Partially Z-Scored version: Computing ROI-level R² comparisons (raw vs drift)...")
    
    roi_chs = ['Fz','FCz','POz','Cz','CPz','Pz', 'Oz']   # LPP 
    tmin, tmax = 0.4, 0.8           
    
    R2_rows = []
    
    for pa in part_1:
        print(f"R² ROI: processing {pa}")
        # --- Recreate cleaned epochs & behavioural table for this subject ---
        df2 = epo_1_filtered_combined[epo_1_filtered_combined['participant_id'] == pa]
        mod2 = part_1_dat[part_1_dat['participant'] == pa]
    
        #decision phase epochs
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
    
        # mean EEG in ROI and time window
        picks = mne.pick_channels(epo_z.info['ch_names'], roi_chs)
        tmask = (epo_z.times >= tmin) & (epo_z.times <= tmax)
    
        data_roi = epo_z.get_data()[:, picks][:, :, tmask]  # trials x ch x time
        y = data_roi.mean(axis=(1, 2))                      # (n_trials,)
    
        # Compare raw vs drift for each attribute 
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
        R2_df.to_csv(partz_dir / f'ROI_R2_raw_vs_v.csv', index=False)
        print("Saved ROI_R2_raw_vs_v.csv in", partz_dir)    
    

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

#---------------------------------------------------------------------------------------------------------------------------- 
# Between-subjects mass-univariate GLM on ERPs averaged per subject

if version == 5:
    print("\n Between-subjects subj-level GLM ---")
    lpp_roi_chs = ['Fz', 'FCz', 'POz', 'Cz', 'CPz', 'Pz', 'Oz' ]
    lpp_tmin, lpp_tmax = 0.4, 0.8   
    
    group_dir = PROJECT_DIR / "EEG" / "PainReward_sub-001-050" / "painrewardeegdata" / "derivatives" / "group_level"

    name = "decision"   # can be changed to passive for comparison purposes later on
    group_epochs_fname = group_dir / f"{name}_off+_subaveraged-epo.fif"

    if not group_epochs_fname.exists():
        raise FileNotFoundError(f"Group-level epochs file not found: {group_epochs_fname}")

    group_epochs = mne.read_epochs(group_epochs_fname)
    data = group_epochs.get_data()
    n_subj, n_chan, n_time = data.shape
    print(f"group_epochs shape = {data.shape} (subjects, channels, times)")
    
    # adjacency for cluster tests (same logic as versions 1–3)
    connect, ch_names = mne.channels.find_ch_adjacency(group_epochs.info, ch_type='eeg')
    
    if not isinstance(param['cluster_threshold'], dict):
        p_thresh = param['cluster_threshold'] / 2
        cluster_threshold = -stats.t.ppf(p_thresh, n_subj - 1)
    else:
        cluster_threshold = param['cluster_threshold']


    # subject IDs as in the ERP metadata
    subj_ids_epochs = group_epochs.metadata["participant_id"].tolist()

    # ------------------------------------------------------------------
    # subject-level regressors from HDDM outputs
    # rt for mod 9 and 10 (hddm model) is the same 

    v_subj_cols = ['v_painlevel_subj', 'v_moneylevel_subj', 'v_interaction_subj']
    a_subj_cols = ['a_painlevel_subj', 'a_moneylevel_subj', 'a_interaction_subj']

    # subject-level v-betas + mean RT
    subj_reg_v = (
        mod_data[mod_data["participant"].isin(subj_ids_epochs)]
        .groupby("participant")[v_subj_cols + ["rt"]]
        .mean()
        .reset_index()
    )

    # subject-level a-betas
    subj_reg_a = (
        mod_data_a[mod_data_a["participant"].isin(subj_ids_epochs)]
        .groupby("participant")[a_subj_cols]
        .mean()
        .reset_index()
    )

    # merge v + a on participant
    subj_reg = subj_reg_v.merge(subj_reg_a, on="participant", how="inner")


    subj_ids_reg   = subj_reg["participant"].tolist()
    common_subj_ids = [s for s in subj_ids_epochs if s in subj_ids_reg]

    print(f"using {len(common_subj_ids)} subjects with both ERPs and HDDM regressors.")
    print("Subjects used:", common_subj_ids)

    # subset group_epochs to those subjects
    keep_idx = [i for i, s in enumerate(subj_ids_epochs) if s in common_subj_ids]
    group_epochs = group_epochs[keep_idx]
    data = group_epochs.get_data()
    subj_reg = subj_reg.set_index("participant").loc[common_subj_ids].reset_index()
    subj_ids = common_subj_ids
    n_subj = len(subj_ids)

    assert data.shape[0] == n_subj == subj_reg.shape[0], "Subject mismatch after filtering!"


    # final list of regressors, this contains both, the v ~ painlevel + moneylevel + interaction and the a ~ painlevel + moneylevel + interaction models betas
    regvars_v5 = v_subj_cols + a_subj_cols

    # ------------------------------------------------------------------
    #helper for cluster-based between-subject GLM (parallel to v1–3)
    def run_group_cluster_variant(subdir_name, zscore_reg=False, zscore_rt=False):
        print(f"\n --- Running cluster-based group GLM: {subdir_name} ---")
        variant_dir = Path(outpath) / (subdir_name + "_cluster")
        variant_dir.mkdir(parents=True, exist_ok=True)

        rt_vals = subj_reg["rt"].to_numpy(dtype=float)

        for regvar in regvars_v5:
            print(f"Regressor: {regvar}")

            x_reg = subj_reg[regvar].to_numpy(dtype=float)
            x_rt  = rt_vals.copy()

            if zscore_reg:
                x_reg = stats.zscore(x_reg)
            if zscore_rt:
                x_rt = stats.zscore(x_rt)

            # ----------------------------------------------------------
            # Orthogonalise regressor with respect to RT
            # (equivalent to including RT in the design and taking
            #  the effect of regvar while controlling for RT)

            X_cov = np.column_stack([np.ones(n_subj), x_rt])
            beta_cov, _, _, _ = np.linalg.lstsq(X_cov, x_reg, rcond=None)
            x_res = x_reg - X_cov @ beta_cov    # shape (n_subj,)

            keep = np.isfinite(x_res) & np.all(np.isfinite(data.reshape(n_subj, -1)), axis=1)
            if keep.sum() < 5:
                print(f"Skipping {regvar} in {subdir_name} as only {keep.sum()} valid subjects")
                continue

            x_res_k = x_res[keep]
            data_k  = data[keep, :, :]          # (n_subj_kept, n_chan, n_time)
            n_kept  = data_k.shape[0]

            # ----------------------------------------------------------
            # subject-level effect maps:
            #   effect_s(chan, time) = x_res(s) * EEG_s(chan, time)

            effect_data = np.empty_like(data_k)
            for i_sub in range(n_kept):
                effect_data[i_sub] = x_res_k[i_sub] * data_k[i_sub]

            testdata = np.swapaxes(effect_data, 2, 1)

            if not isinstance(param['cluster_threshold'], dict):
                p_thresh = param['cluster_threshold'] / 2
                thr = -stats.t.ppf(p_thresh, n_kept - 1)
            else:
                thr = param['cluster_threshold']

            tval, clusters, cluster_p_values, _ = st_clust_1s_ttest(
                testdata,
                n_jobs=param["njobs"],
                threshold=thr,
                adjacency=connect,
                n_permutations=param['nperms'],
                buffer_size=None
            )
            pvals = np.ones_like(tval)
            for c, p_val in zip(clusters, cluster_p_values):
                pvals[c] = p_val

            np.save(variant_dir / f'groupglm_cluster_tval_{regvar}.npy', tval)
            np.save(variant_dir / f'groupglm_cluster_pval_{regvar}.npy', pvals)

        return variant_dir

    # helper to run one design variant (NO_Z, Z, PartZ) and save maps

    def run_group_glm_variant(subdir_name, zscore_reg=False, zscore_rt=False):
        print(f"\n--- Running group GLM variant: {subdir_name} ---")

        variant_dir = Path(outpath) / subdir_name
        variant_dir.mkdir(parents=True, exist_ok=True)

        betas_variant = {}
        tvals_variant = {}
        pvals_variant = {}

        rt_vals = subj_reg["rt"].to_numpy(dtype=float)

        for regvar in regvars_v5:
            print(f"  Regressor: {regvar}")

            x_reg = subj_reg[regvar].to_numpy(dtype=float)
            x_rt = rt_vals.copy()

            if zscore_reg:
                x_reg = stats.zscore(x_reg)
            if zscore_rt:
                x_rt = stats.zscore(x_rt)

            design = pd.DataFrame({
                "Intercept": np.ones(n_subj),
                regvar: x_reg,
                "rt": x_rt,
            })

            if not np.all(np.isfinite(design.to_numpy())):
                print(f"Design has NaN/Inf for {regvar} in {subdir_name}, skipping.")
                continue

            res = mne.stats.linear_regression(
                group_epochs,
                design,
                names=["Intercept", regvar, "rt"]
            )

            beta_ev = res[regvar].beta
            t_ev    = res[regvar].t_val
            p_ev    = res[regvar].p_val

            betas_variant[regvar] = beta_ev
            tvals_variant[regvar] = t_ev
            pvals_variant[regvar] = p_ev

            # save chan * time maps
            np.save(variant_dir / f'groupglm_beta_{regvar}.npy', beta_ev.data)
            np.save(variant_dir / f'groupglm_tval_{regvar}.npy', t_ev.data)
            np.save(variant_dir / f'groupglm_pval_{regvar}.npy', p_ev.data)

        return betas_variant, tvals_variant, pvals_variant, variant_dir
        # ------------------------------------------------------------------
        
        
        
 

    # ------------------------------------------------------------------
    # three variants: NO_Zscoring, Zscoring, PartZscoring
    
    # NO_Zscoring, using raw v/a betas, raw RT
    betas_noz, tvals_noz, pvals_noz, noz_dir_v5 = run_group_glm_variant(
        subdir_name="NO_Zscoring",
        zscore_reg=False,
        zscore_rt=False
    )

    # Zscoring all predictors
    betas_z, tvals_z, pvals_z, z_dir_v5 = run_group_glm_variant(
        subdir_name="Zscoring",
        zscore_reg=True,
        zscore_rt=True
    )

    # PartZscoring, z-scored RT
    betas_partz, tvals_partz, pvals_partz, partz_dir_v5 = run_group_glm_variant(
        subdir_name="PartZscoring",
        zscore_reg=False,
        zscore_rt=True
    )
    #----------------------------------------------------------------------------------

    noz_cluster_dir_v5 = run_group_cluster_variant(
        subdir_name="NO_Zscoring",
        zscore_reg=False,
        zscore_rt=False
    )

    z_cluster_dir_v5 = run_group_cluster_variant(
        subdir_name="Zscoring",
        zscore_reg=True,
        zscore_rt=True
    )

    partz_cluster_dir_v5 = run_group_cluster_variant(
        subdir_name="PartZscoring",
        zscore_reg=False,
        zscore_rt=True
    )
    
    
    #----------------------------------------------------------------------------------
    print("\n ROI-level between-subject correlations: each regressor vs LPP (cue-locked, version 5)")
    
    from scipy.stats import pearsonr
    
    # 1) Extract subject-level LPP amplitude from cue-locked group_epochs
    roi_picks = mne.pick_channels(group_epochs.info['ch_names'], lpp_roi_chs)
    if len(roi_picks) == 0:
        raise RuntimeError(f"None of the LPP ROI channels found in data: {lpp_roi_chs}")
    
    tmask = (group_epochs.times >= lpp_tmin) & (group_epochs.times <= lpp_tmax)
    if not np.any(tmask):
        raise RuntimeError(f"No time points in LPP window {lpp_tmin}–{lpp_tmax} s for cue-locked epochs.")
    
    # data: (subjects, channels, times)
    data_roi = data[:, roi_picks][:, :, tmask]   # subj × ROI-ch × time
    y_LPP = data_roi.mean(axis=(1, 2))           # subj-level LPP amplitude
    
    # 2) For each regressor, correlate with LPP (raw and RT-controlled)
    corr_rows = []
    
    rt_vals = subj_reg["rt"].to_numpy(dtype=float)
    
    for regvar in regvars_v5:
        x = subj_reg[regvar].to_numpy(dtype=float)
    
        # valid subjects for this regressor
        keep = np.isfinite(x) & np.isfinite(y_LPP) & np.isfinite(rt_vals)
        n = keep.sum()
        if n < 5:
            print(f"Skipping ROI correlation for {regvar}: only {n} valid subjects")
            continue
    
        x_k = x[keep]
        y_k = y_LPP[keep]
        rt_k = rt_vals[keep]
    
        # --- simple Pearson correlation (no RT control) ---
        r_raw, p_raw = pearsonr(x_k, y_k)
    
        # --- partial correlation controlling for RT ---
        # regress y on RT -> residuals
        X_rt = np.column_stack([np.ones(n), rt_k])
        beta_y, _, _, _ = np.linalg.lstsq(X_rt, y_k, rcond=None)
        y_res = y_k - X_rt @ beta_y
    
        # regress x on RT -> residuals
        beta_x, _, _, _ = np.linalg.lstsq(X_rt, x_k, rcond=None)
        x_res = x_k - X_rt @ beta_x
    
        r_par, p_par = pearsonr(x_res, y_res)
    
        print(f"{regvar}: r_raw = {r_raw:.3f} (p={p_raw:.3g}), r_partial_RT = {r_par:.3f} (p={p_par:.3g}), n={n}")
    
        corr_rows.append(dict(
            regressor=regvar,
            n=n,
            r_raw=r_raw,
            p_raw=p_raw,
            r_partial_RT=r_par,
            p_partial_RT=p_par
        ))
    
    # save summary table
    if len(corr_rows) > 0:
        corr_df = pd.DataFrame(corr_rows)
        corr_df.to_csv(noz_dir_v5 / "ROI_LPP_vs_each_regressor.csv", index=False)
        print("Saved ROI_LPP_vs_each_regressor.csv in", noz_dir_v5)
        
    #---------------------------------------------------------------------------------------------------------------------
    from mne.stats import permutation_cluster_1samp_test
    
    print("\n LPP ROI time-resolved cluster regression (RT-controlled)")
    
    # ------------------------------------------------------------------
    # LPP ROI + time window (cue-locked)
    lpp_roi_chs = ['Fz', 'FCz', 'POz', 'Cz', 'CPz', 'Pz', 'Oz' ]
    lpp_tmin, lpp_tmax = 0.4, 0.8
    
    # channel & time selection
    roi_picks = mne.pick_channels(group_epochs.info['ch_names'], lpp_roi_chs)
    if len(roi_picks) == 0:
        raise RuntimeError(f"LPP ROI channels not found: {lpp_roi_chs}")
    
    times = group_epochs.times
    tmask = (times >= lpp_tmin) & (times <= lpp_tmax)
    if not np.any(tmask):
        raise RuntimeError(f"No time points in {lpp_tmin}–{lpp_tmax}s window")
    
    # subj × ROIchan × time
    data_roi = data[:, roi_picks][:, :, tmask]
    
    # mean across ROI channels → subj × time
    data_roi_mean = data_roi.mean(axis=1)
    
    rt_vals = subj_reg["rt"].to_numpy(dtype=float)
    
    roi_cluster_dir = Path(outpath) / "LPP_ROI_cluster"
    roi_cluster_dir.mkdir(parents=True, exist_ok=True)
    
    # ------------------------------------------------------------------
    # Run one regressor at a time
    for regvar in regvars_v5:
        print(f"\nLPP ROI cluster test for regressor: {regvar}")
    
        x = subj_reg[regvar].to_numpy(dtype=float)
    
        keep = (
            np.isfinite(x) &
            np.isfinite(rt_vals) &
            np.all(np.isfinite(data_roi_mean), axis=1)
        )
    
        if keep.sum() < 5:
            print(f"  Skipping {regvar}: only {keep.sum()} valid subjects")
            continue
    
        x_k  = x[keep]
        rt_k = rt_vals[keep]
        y_k  = data_roi_mean[keep]    # subj × time
        n_k  = y_k.shape[0]
    
        # --------------------------------------------------------------
        # RT-controlled regressor (residualisation)
        X_rt = np.column_stack([np.ones(n_k), rt_k])
        beta_cov, _, _, _ = np.linalg.lstsq(X_rt, x_k, rcond=None)
        x_res = x_k - X_rt @ beta_cov
    
        # --------------------------------------------------------------
        # Subject-level effect maps: subj × time
        effect = np.zeros_like(y_k)
        for s in range(n_k):
            effect[s] = x_res[s] * y_k[s]
    
        # --------------------------------------------------------------
        # Cluster test over TIME ONLY
        t_obs, clusters, cluster_p, _ = permutation_cluster_1samp_test(
            effect,
            n_permutations=5000,
            threshold=None,
            tail=0,
            out_type='mask',
            verbose=False
        )
    
        # build time-resolved p-value vector
        pvals = np.ones(effect.shape[1])
        for clu, p in zip(clusters, cluster_p):
            pvals[clu] = p
    
        # save
        np.save(roi_cluster_dir / f"LPPROI_tval_{regvar}.npy", t_obs)
        np.save(roi_cluster_dir / f"LPPROI_pval_{regvar}.npy", pvals)
    
        print(f"  Saved LPP ROI cluster results for {regvar}")

    
    print(f"\nVersion 5 finished. Subject-level GLM + ROI correlations saved in:\n  {noz_dir_v5}\n  {z_dir_v5}\n  {partz_dir_v5}")
    
    
#----------------------------------------------------------------------------------------------------------------------------------------------
# ======================================================================
# Version 6 – second-level GLM: subject β maps (from v=3) ~ HDDM drift
# ======================================================================

elif version == 6:
    print("\n Second-level β ~ drift(GLM) on v3 betas\n")

    v3_dir = basepath / "statistics_new" / "erps_massuni_drift_mod_9_RT_3GLMs"
    v3_z_dir = v3_dir / "Zscoring"

    # load subject-level beta maps & list of subjects that entered v3
    allbetas = np.load(v3_z_dir / "ols_2ndlevel_betas.npy")          # (subj, reg, chan, time)
    beta_gavg = np.load(v3_z_dir / "ols_2ndlevel_betasavg.npy",
                        allow_pickle=True)
    included_subjects = np.load(v3_z_dir / "included_subjects.npy",
                                allow_pickle=True)

    # regressor bookkeeping must match v3
    regvars = [
        "painlevel", "moneylevel", "interaction",
        "v_pain_contrib", "v_money_contrib", "v_interaction_contrib"
    ]
    reg_labels = ["pain", "money", "interaction"]   # just the raw ones

    # EEG info / adjacency
    info = beta_gavg[0].info
    times = beta_gavg[0].times
    connect, _ = mne.channels.find_ch_adjacency(info, ch_type="eeg")

    # Where to save the v6 results
    v6_dir = basepath / "statistics_new" / "erps_massuni_drift_mod_9_v6_beta_vs_drift"
    v6_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Build subject-level behavioural regressors (drift + mean RT)
    # ------------------------------------------------------------------
    # we already loaded mod_data above in the script
    mod_subj = (mod_data[mod_data["participant"].isin(included_subjects)]
                .groupby("participant")
                .agg(
                    v_pain=("v_painlevel_subj", "mean"),
                    v_money=("v_moneylevel_subj", "mean"),
                    v_inter=("v_interaction_subj", "mean"),
                    mean_rt=("rt", "mean"),
                )
                .reindex(included_subjects)   # ensure same order as allbetas
               )

    # sanity check
    assert all(mod_subj.index.to_list()[i] == included_subjects[i]
               for i in range(len(included_subjects))), "Subject order mismatch!"

    v_pain = mod_subj["v_pain"].to_numpy(dtype=float)
    v_money = mod_subj["v_money"].to_numpy(dtype=float)
    v_inter = mod_subj["v_inter"].to_numpy(dtype=float)
    mean_rt = mod_subj["mean_rt"].to_numpy(dtype=float)

    # cluster threshold params (same logic as v1–3)
    if not isinstance(param['cluster_threshold'], dict):
        p_thresh = param['cluster_threshold'] / 2
        base_thr = -stats.t.ppf(p_thresh, len(included_subjects) - 1)
    else:
        base_thr = param['cluster_threshold']

    chankeep = np.array([c not in ["M1", "M2"] for c in info["ch_names"]])

    # ------------------------------------------------------------------
    # helper: compute slope map γ1(c,t) and cluster-test it
    # ------------------------------------------------------------------
    def run_beta_vs_drift(label, reg_name, v_vec):
        """
        label    = 'pain', 'money', or 'interaction'
        reg_name = 'painlevel', 'moneylevel', 'interaction'
        v_vec    = per-subject drift array (len = n_subj)
        """
        print(f"\n  ==> β_{label} ~ v_{label} second-level GLM")

        # index of raw regressor in allbetas
        ridx = regvars.index(reg_name)

        # betas for this regressor: (subj, chan, time)
        betas_reg = allbetas[:, ridx, :, :]

        # orthogonalise drift wrt mean RT: v_res = v - (Intercept+RT)*beta
        X_cov = np.column_stack([np.ones(len(v_vec)), mean_rt])
        beta_cov, _, _, _ = np.linalg.lstsq(X_cov, v_vec, rcond=None)
        v_res = v_vec - X_cov @ beta_cov

        # keep only finite data
        eeg_finite = np.all(np.isfinite(betas_reg.reshape(len(v_res), -1)), axis=1)
        keep = np.isfinite(v_res) & eeg_finite
        if keep.sum() < 5:
            print(f"    Skipping {label}: only {keep.sum()} valid subjects")
            return

        v_res_k = v_res[keep]
        betas_k = betas_reg[keep, :, :]
        n_kept = betas_k.shape[0]

        # --- 1) slope map γ1(c,t) ------------------------------------
        denom = np.sum(v_res_k ** 2)
        gamma1 = np.tensordot(v_res_k, betas_k, axes=(0, 0)) / denom   # (chan, time)
        np.save(v6_dir / f"v6_gamma1_beta_{label}_vs_v.npy", gamma1)

        # --- 2) effect_data for cluster test -------------------------
        # effect_s(c,t) = v_res_s * β_s(c,t)
        effect_data = betas_k * v_res_k[:, None, None]   # (subj, chan, time)
        testdata = np.swapaxes(effect_data, 2, 1)        # (subj, time, chan)
        
        
        from scipy.stats import ttest_1samp
        from statsmodels.stats.multitest import fdrcorrection
        
        # effect_data has shape (n_kept, n_chan, n_time)
        n_sub, n_chan, n_time = effect_data.shape
        
        # 1) pointwise t-tests: t and uncorrected p
        tvals_pt = np.zeros((n_time, n_chan))
        pvals_pt = np.ones((n_time, n_chan))
        
        for ti in range(n_time):
            # test across subjects at each channel
            t_, p_ = ttest_1samp(effect_data[:, :, ti], popmean=0.0, axis=0, nan_policy='omit')
            tvals_pt[ti, :] = t_
            pvals_pt[ti, :] = p_
        
        # 2) restrict to channels we care about (e.g. drop M1/M2)
        chankeep = np.array([c not in ["M1", "M2"] for c in info["ch_names"]])
        pvals_flat = pvals_pt[:, chankeep].ravel()
        
        # 3) FDR correction
        alpha_fdr = 0.05  # or 0.05/3 if you Bonferroni across pain/money/interaction
        rej, pvals_fdr = fdrcorrection(pvals_flat, alpha=alpha_fdr)
        
        # 4) reshape back to (time, chan)
        sig_fdr = np.zeros_like(pvals_pt, dtype=bool)
        sig_fdr[:, chankeep] = rej.reshape(pvals_pt[:, chankeep].shape)
        
        # save for plotting
        np.save(v6_dir / f"v6_tvals_pointwise_{label}.npy", tvals_pt)
        np.save(v6_dir / f"v6_pvals_pointwise_{label}.npy", pvals_pt)
        np.save(v6_dir / f"v6_sigmask_fdr_{label}.npy", sig_fdr)


        if not isinstance(param['cluster_threshold'], dict):
            p_thresh = param['cluster_threshold'] / 2
            thr = -stats.t.ppf(p_thresh, n_kept - 1)
        else:
            thr = base_thr

        tvals, clusters, cluster_p_values, _ = st_clust_1s_ttest(
            testdata,
            n_jobs=param["njobs"],
            threshold=thr,
            adjacency=connect,
            n_permutations=param['nperms'],
            buffer_size=None
        )

        pvals = np.ones_like(tvals)
        for c, p_val in zip(clusters, cluster_p_values):
            pvals[c] = p_val

        np.save(v6_dir / f"v6_tvals_beta_{label}_vs_v.npy", tvals)
        np.save(v6_dir / f"v6_pvals_beta_{label}_vs_v.npy", pvals)

        roi_chs = ['Fz', 'FCz', 'POz', 'Cz', 'CPz', 'Pz', 'Oz']  # ['Fz'], ['FCz'], ['POz'], ['Cz'], ['CPz'], ['Pz'], ['Oz']
        tmin, tmax = 0.4, 0.8

        picks = mne.pick_channels(info['ch_names'], roi_chs)
        tmask = (times >= tmin) & (times <= tmax)

        beta_roi = betas_k[:, picks][:, :, tmask].mean(axis=(1, 2))  # (n_kept,)

        r, p = stats.pearsonr(v_res_k, beta_roi)
        print(f"    ROI β_{label}(0.4–0.8s, LPP spec. electrodes) vs v_res: r={r:.3f}, p={p:.3g}")

        return dict(
            label=label,
            n=n_kept,
            r=r,
            p=p
        )

    # run for pain / money / interaction
    roi_rows = []
    roi_rows.append(run_beta_vs_drift("pain", "painlevel", v_pain))
    roi_rows.append(run_beta_vs_drift("money", "moneylevel", v_money))
    roi_rows.append(run_beta_vs_drift("interaction", "interaction", v_inter))
    roi_rows = [row for row in roi_rows if row is not None]

    if len(roi_rows) > 0:
        R2_df = pd.DataFrame(roi_rows)
        R2_df.to_csv(v6_dir / "v6_ROI_corr_beta_vs_v.csv", index=False)
        print("\nSaved ROI summary correlations in v6_ROI_corr_beta_vs_v.csv")

    print(f"\nVersion 6 done. Results in:\n  {v6_dir}\n")
    

#-------------------------------------------------------------------------------------------------------------    

elif version == 7:
    # Z scored version
    z_dir = Path(outpath) / "Zscoring"
    z_dir.mkdir(parents=True, exist_ok=True)

    all_epos = [[] for _ in range(len(regvars))]
    allbetasnp = []
    betas = [[] for _ in range(len(regvars))]
    included_subjects = []
    skipped_subjects = []
    
    for pa in part_1:
        print(f"\n--- YES: Z-Scored Version: Processing {pa} ---")
        df2 = epo_1_filtered_combined[epo_1_filtered_combined['participant_id'] == pa]
        mod2 = part_1_dat[part_1_dat['participant'] == pa]

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

        rt_col = "rt"
        
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

            # Z-score predictors
            df_reg[regvar + "_z"] = stats.zscore(df_reg[regvar].to_numpy(dtype=float))
            df_reg["RT_z"] = stats.zscore(df_reg[rt_col].to_numpy(dtype=float))
            df_reg["Intercept"] = 1.0

            design = df_reg[["Intercept", regvar + "_z", "RT_z"]]

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
                names=["Intercept", regvar + "_z", "RT_z"]
            )

            # beta for regressor 
            beta_reg = res[regvar + "_z"].beta
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
        
        z_dir = Path(outpath) / "Zscoring"
        z_dir.mkdir(parents=True, exist_ok=True)
        
        np.save(z_dir / f'ols_2ndlevel_tval_{regvar}.npy', tvals[-1])
        np.save(z_dir / f'ols_2ndlevel_pval_{regvar}.npy', pvalues[-1])

    # Stack and save group-level results
    tvals = np.stack(tvals)
    pvals = np.stack(pvalues)

    np.save(z_dir / f'ols_2ndlevel_tvals.npy', tvals)
    np.save(z_dir / f'ols_2ndlevel_pvals.npy', pvals)
    np.save(z_dir / f'ols_2ndlevel_betas.npy', allbetas)

    for idx, regvar in enumerate(regvars):
        epo_save = mne.concatenate_epochs(all_epos[idx])
        epo_save.save(z_dir / f'ols_2ndlevel_allepochs-epo_{regvar}.fif', overwrite=True)

    np.save(z_dir / f'ols_2ndlevel_betasavg.npy', beta_gavg)
    
    
    
    
    
#------------------------------------------------------------------------------------------------------------------------------------
if version == 8:
    from mne.time_frequency import read_tfrs

    print("\n--- Version 8: TFR ROI vs drift (between-subject) ---")

    group_dir = Path(outpath)
    group_dir.mkdir(parents=True, exist_ok=True)

    # ---- define ROIs / bands / time-window ----
    roi_theta = ['Fz', 'FCz', 'Cz']          # frontal / fronto-central theta
    roi_alpha = ['Cz', 'CPz', 'Pz']          # centro-parietal alpha
    theta_band = (4., 7.)
    alpha_band = (8., 13.)
    time_win = (0.0, 1.0)                    # anticipation window after cue (off+)

    rows = []

    for pa in part:
        # TFR file (decision phase)
        tfr_fname = opj(basepath, pa, 'eeg', 'tfr',
                        f"{pa}_decision_cues_epochs-tfr.h5")
        if not os.path.exists(tfr_fname):
            print(f"Skipping {pa}, no TFR file {tfr_fname}")
            continue

        # Load single-trial TFR
        tfr_epochs = read_tfrs(tfr_fname)[0]   # EpochsTFR
        data = tfr_epochs.data                 # (n_trials, n_chan, n_freq, n_time)
        freqs = tfr_epochs.freqs
        times = tfr_epochs.times
        ch_names = np.array(tfr_epochs.ch_names)

        # Average across trials -> (n_chan, n_freq, n_time)
        subj_power = data.mean(axis=0)

        # Masks
        theta_mask = (freqs >= theta_band[0]) & (freqs <= theta_band[1])
        alpha_mask = (freqs >= alpha_band[0]) & (freqs <= alpha_band[1])
        time_mask = (times >= time_win[0]) & (times <= time_win[1])

        # Channel indices
        try:
            theta_ch_idx = [np.where(ch_names == c)[0][0] for c in roi_theta]
            alpha_ch_idx = [np.where(ch_names == c)[0][0] for c in roi_alpha]
        except IndexError as e:
            print(f"Channel missing for {pa}: {e}")
            continue

        # ROI-averaged power
        theta_power = subj_power[theta_ch_idx][:, theta_mask][:, :, time_mask].mean()
        alpha_power = subj_power[alpha_ch_idx][:, alpha_mask][:, :, time_mask].mean()

        # ---- subject-level drift summary ----
        # Here I use the mean of v_pain_contrib across trials as a "pain-drift" proxy.
        # If you have a separate subject-level drift CSV, you can replace this with that.
        sub_mod = mod_data[mod_data["participant"] == pa]
        if "v_painlevel_subj" in sub_mod.columns:
            drift_val = sub_mod["v_painlevel_subj"].mean()
        else:
            print(f"No v_pain_contrib/v_pain column for {pa}, skipping.")
            continue

        rows.append({
            "participant_id": pa,
            "theta_power": theta_power,
            "alpha_power": alpha_power,
            "drift_pain": drift_val
        })

    roi_tfr_df = pd.DataFrame(rows)
    roi_tfr_csv = group_dir / "tfr_roi_theta_alpha_vs_drift.csv"
    roi_tfr_df.to_csv(roi_tfr_csv, index=False)
    print("Saved ROI TFR vs drift summary to:", roi_tfr_csv)


# #----------------------------------------------------------------------------------------------------------------------------------------------
# # trial-wise TFR betas for sv_pain_para

# if version == 9:
#     from mne.time_frequency import tfr_morlet

#     print("\n--- Version 9: TFR trial-wise betas for sv_pain_para (recompute from ERPs) ---")

#     # sanity checks on mod_data
#     if "sv_pain_para" not in mod_data.columns:
#         raise ValueError("sv_pain_para not found in mod_data columns.")
#     if "trialsnum" not in mod_data.columns:
#         raise ValueError("mod_data is missing 'trialsnum' column.")

#     group_dir = Path(outpath)
#     group_dir.mkdir(parents=True, exist_ok=True)

#     # TFR parameters (same as in the ERP+TFR script)
#     freqs = np.arange(4., 101., 1.)
#     n_cycles = 0.5 * freqs
#     target_sfreq = 256.0      # to match your previous TFRs
#     tmin_crop = -0.50
#     tmax_crop = 1.00

#     all_betas = []     # list of (n_chan, n_freq, n_time)
#     used_subs = []
#     freqs_out = None
#     times_out = None
#     ch_names_out = None

#     for pa in part:
#         print(f"\nSubject {pa}...")

#         # -------------------------
#         # 1) behavioural side
#         # -------------------------
#         beh_sub = mod_data[mod_data["participant"] == pa].copy()
#         if beh_sub.empty:
#             print(f"  No behavioural rows in mod_data for {pa}, skipping.")
#             continue

#         beh_sub = beh_sub[["trialsnum", "sv_pain_para"]].copy()
#         beh_sub["trialsnum"] = beh_sub["trialsnum"].astype(int)

#         epo_fname = opj(
#             basepath,
#             pa, "eeg", "erps",
#             f"{pa}_decision_cue_singletrials-epo.fif"
#         )
#         if not os.path.exists(epo_fname):
#             print(f"No ERP single-trials file for {pa}, skipping.")
#             continue

#         print("Reading ERP epochs:", epo_fname)
#         epo = mne.read_epochs(epo_fname, preload=True)
#         meta = epo.metadata.copy()

#         required_cols = {"trialsnum", "badtrial"}
#         missing_meta = required_cols.difference(meta.columns)
#         if missing_meta:
#             raise ValueError(
#                 f"{pa}: ERP metadata missing {missing_meta}. "
#                 f"Columns are: {meta.columns.tolist()}"
#             )

#         n_trials_erp = len(epo)
#         print(f"  n_trials ERP: {n_trials_erp}")

#         if n_trials_erp < 5:
#             print(f"  {pa}: <5 ERP trials, skipping.")
#             continue

#         sfreq = epo.info["sfreq"]
#         if sfreq != target_sfreq:
#             decim = int(round(sfreq / target_sfreq))
#             if not np.isclose(sfreq / decim, target_sfreq):
#                 print(f"  Warning: {pa}: sfreq={sfreq}, cannot cleanly decimate to {target_sfreq} Hz, using decim={decim}")
#             print(f"  Resampling epochs from {sfreq} Hz to ~{target_sfreq} Hz (decim={decim})")
#             epo_resamp = epo.copy().resample(target_sfreq)
#         else:
#             epo_resamp = epo

#         # keep metadata aligned
#         meta = epo_resamp.metadata.copy()
#         meta["trialsnum"] = meta["trialsnum"].astype(int)

#         # -------------------------
#         # 4) merge ERP metadata with behaviour by trialsnum
#         # -------------------------
#         meta = meta.reset_index().rename(columns={"index": "row_id"})
#         merged = meta.merge(
#             beh_sub,
#             on="trialsnum",
#             how="inner"
#         )

#         print(f"  ERP trials: {n_trials_erp}, beh rows: {len(beh_sub)}, after merge: {len(merged)}")

#         if merged.empty:
#             print(f"  {pa}: no overlapping trials by trialsnum, skipping.")
#             continue

#         idx_keep = merged["row_id"].to_numpy(dtype=int)
#         if len(idx_keep) < 5:
#             print(f"  {pa}: only {len(idx_keep)} trials after matching ERP+beh, skipping.")
#             continue

#         epo_match = epo_resamp[idx_keep]
#         merged = merged.reset_index(drop=True)

#         # -------------------------
#         # 6) filter for good EEG + finite sv_pain_para
#         # -------------------------
#         bad = merged["badtrial"].to_numpy(dtype=float)
#         sv = merged["sv_pain_para"].to_numpy(dtype=float)

#         keep = (bad == 0) & np.isfinite(sv)
#         if keep.sum() < 5:
#             print(f"  {pa}: <5 good trials after badtrial+sv filtering, skipping.")
#             continue

#         epo_good = epo_match[keep]
#         sv_good = sv[keep]

#         # -------------------------
#         # 7) compute TFR on the fly from these epochs
#         # -------------------------
#         print(f"  Computing TFR for {pa}: n_good trials = {len(epo_good)}")

#         tfr = tfr_morlet(
#             epo_good,
#             freqs=freqs,
#             n_cycles=n_cycles,
#             return_itc=False,
#             use_fft=True,
#             decim=1,                
#             n_jobs=param.get("njobs", 8),
#             average=False            
#         )

#         # crop to desired window
#         tfr.crop(tmin=tmin_crop, tmax=tmax_crop)

#         data = tfr.data    # (n_trials, n_chan, n_freq, n_time)
#         n_trials, n_chan, n_freq, n_time = data.shape
#         print(f"  TFR shape after crop: {data.shape}")

#         # store freq/time/ch_names from first subject
#         if freqs_out is None:
#             freqs_out = tfr.freqs.copy()
#             times_out = tfr.times.copy()
#             ch_names_out = np.array(tfr.ch_names, dtype=object)
#         else:
#             # sanity: ensure same grid
#             if not np.array_equal(freqs_out, tfr.freqs):
#                 raise ValueError(f"{pa}: frequency grid mismatch")
#             if not np.array_equal(times_out, tfr.times):
#                 raise ValueError(f"{pa}: time grid mismatch")
#             if list(ch_names_out) != tfr.ch_names:
#                 raise ValueError(f"{pa}: channel list mismatch")

#         # -------------------------
#         # 8) z-score sv within subject & regression
#         # -------------------------
#         sv_z = (sv_good - sv_good.mean()) / sv_good.std()
#         var_sv = sv_z.var()
#         if var_sv == 0:
#             print(f"  {pa}: sv_pain_para variance = 0 after filtering, skipping.")
#             continue

#         betas_sub = np.zeros((n_chan, n_freq, n_time), dtype=float)

#         # regression at each (chan, freq, time): power ~ sv_z
#         for ci in range(n_chan):
#             Pw_ci = data[:, ci, :, :]      # (n_trials, n_freq, n_time)
#             Pw_flat = Pw_ci.reshape(n_trials, -1)
#             cov_flat = (Pw_flat * sv_z[:, None]).mean(axis=0) \
#                        - Pw_flat.mean(axis=0) * sv_z.mean()
#             beta_flat = cov_flat / var_sv
#             betas_sub[ci] = beta_flat.reshape(n_freq, n_time)

#         all_betas.append(betas_sub)
#         used_subs.append(pa)
#         print(f"  {pa}: beta map computed, n_good trials = {n_trials}")

#     # -------------------------
#     # 9) save group-level arrays
#     # -------------------------
#     if len(all_betas) == 0:
#         print("No subjects with valid beta maps; nothing saved.")
#     else:
#         all_betas = np.stack(all_betas)  # (n_subj, n_chan, n_freq, n_time)

#         np.save(group_dir / "tfr_beta_sv_pain_para_subxchxfxt.npy", all_betas)
#         np.save(group_dir / "tfr_beta_sv_pain_para_subjects.npy",
#                 np.array(used_subs, dtype=object))
#         np.save(group_dir / "tfr_beta_sv_pain_para_freqs.npy", freqs_out)
#         np.save(group_dir / "tfr_beta_sv_pain_para_times.npy", times_out)
#         np.save(group_dir / "tfr_beta_sv_pain_para_ch_names.npy", ch_names_out)

#         print("Saved beta maps for sv_pain_para to:", group_dir)
#         print("Shapes: all_betas:", all_betas.shape)
#         print("Subjects:", used_subs)
#-----------------------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------------------------------------
# trial-wise TFR betas for sv_pain_para  (cue-locked), controlling for RT
#----------------------------------------------------------------------------------------------------------------------------------------------

if version == 9:
    from mne.time_frequency import tfr_morlet

    print("\n--- Version 9: TFR trial-wise betas for sv_pain_para (cue-locked, controlling for RT) ---")

    # sanity checks on mod_data
    if "sv_pain_para" not in mod_data.columns:
        raise ValueError("sv_pain_para not found in mod_data columns.")
    if "trialsnum" not in mod_data.columns:
        raise ValueError("mod_data is missing 'trialsnum' column.")
    if "rt" not in mod_data.columns:     
        raise ValueError("RT column ('rt') not found in mod_data. "
                         "Replace with your actual RT column name.")

    group_dir = Path(outpath)
    group_dir.mkdir(parents=True, exist_ok=True)

    # TFR parameters
    freqs = np.arange(4., 101., 1.)
    n_cycles = 0.5 * freqs
    target_sfreq = 256.0
    tmin_crop = -0.50
    tmax_crop = 1.00

    all_betas_sv = []   # list of (n_chan, n_freq, n_time) – sv betas (controlling RT)
    # optionally, you can also store RT betas:
    # all_betas_rt = []
    used_subs = []
    freqs_out = None
    times_out = None
    ch_names_out = None

    for pa in part:
        print(f"\nSubject {pa}...")

        # -------------------------
        # 1) behavioural side
        # -------------------------
        beh_sub = mod_data[mod_data["participant"] == pa].copy()
        if beh_sub.empty:
            print(f"  No behavioural rows in mod_data for {pa}, skipping.")
            continue

        beh_sub = beh_sub[["trialsnum", "sv_pain_para", "rt"]].copy()
        beh_sub["trialsnum"] = beh_sub["trialsnum"].astype(int)

        epo_fname = opj(
            basepath,
            pa, "eeg", "erps",
            f"{pa}_decision_cues_singletrials-epo.fif"
        )
 
        if not os.path.exists(epo_fname):
            print(f"No ERP single-trials file for {pa}, skipping.")
            continue

        print("Reading ERP epochs:", epo_fname)
        epo = mne.read_epochs(epo_fname, preload=True)
        meta = epo.metadata.copy()

        required_cols = {"trialsnum", "badtrial"}
        missing_meta = required_cols.difference(meta.columns)
        if missing_meta:
            raise ValueError(
                f"{pa}: ERP metadata missing {missing_meta}. "
                f"Columns are: {meta.columns.tolist()}"
            )

        n_trials_erp = len(epo)
        print(f"  n_trials ERP: {n_trials_erp}")

        if n_trials_erp < 5:
            print(f"  {pa}: <5 ERP trials, skipping.")
            continue

        sfreq = epo.info["sfreq"]
        if sfreq != target_sfreq:
            decim = int(round(sfreq / target_sfreq))
            if not np.isclose(sfreq / decim, target_sfreq):
                print(f"  Warning: {pa}: sfreq={sfreq}, cannot cleanly decimate to {target_sfreq} Hz, using decim={decim}")
            print(f"  Resampling epochs from {sfreq} Hz to ~{target_sfreq} Hz (decim={decim})")
            epo_resamp = epo.copy().resample(target_sfreq)
        else:
            epo_resamp = epo

        # keep metadata aligned
        meta = epo_resamp.metadata.copy()
        meta["trialsnum"] = meta["trialsnum"].astype(int)

        # -------------------------
        # 4) merge ERP metadata with behaviour by trialsnum
        # -------------------------
        meta = meta.reset_index().rename(columns={"index": "row_id"})
        merged = meta.merge(
            beh_sub,
            on="trialsnum",
            how="inner"
        )

        print(f"  ERP trials: {n_trials_erp}, beh rows: {len(beh_sub)}, after merge: {len(merged)}")

        if merged.empty:
            print(f"  {pa}: no overlapping trials by trialsnum, skipping.")
            continue

        idx_keep = merged["row_id"].to_numpy(dtype=int)
        if len(idx_keep) < 5:
            print(f"  {pa}: only {len(idx_keep)} trials after matching ERP+beh, skipping.")
            continue

        epo_match = epo_resamp[idx_keep]
        merged = merged.reset_index(drop=True)

        # -------------------------
        # 6) filter for good EEG + finite sv_pain_para + finite RT
        # -------------------------
        bad = merged["badtrial"].to_numpy(dtype=float)
        sv = merged["sv_pain_para"].to_numpy(dtype=float)
        rt = merged["rt"].to_numpy(dtype=float)   

        keep = (bad == 0) & np.isfinite(sv) & np.isfinite(rt)
        if keep.sum() < 5:
            print(f"  {pa}: <5 good trials after badtrial+sv+rt filtering, skipping.")
            continue

        epo_good = epo_match[keep]
        sv_good = sv[keep]
        rt_good = rt[keep]

        # -------------------------
        # 7) compute TFR on the fly from these epochs
        # -------------------------
        print(f"  Computing TFR for {pa}: n_good trials = {len(epo_good)}")

        tfr = tfr_morlet(
            epo_good,
            freqs=freqs,
            n_cycles=n_cycles,
            return_itc=False,
            use_fft=True,
            decim=1,
            n_jobs=param.get("njobs", 8),
            average=False
        )

        # crop to desired window
        tfr.crop(tmin=tmin_crop, tmax=tmax_crop)

        data = tfr.data    # (n_trials, n_chan, n_freq, n_time)
        n_trials, n_chan, n_freq, n_time = data.shape
        print(f"  TFR shape after crop: {data.shape}")

        # store freq/time/ch_names from first subject
        if freqs_out is None:
            freqs_out = tfr.freqs.copy()
            times_out = tfr.times.copy()
            ch_names_out = np.array(tfr.ch_names, dtype=object)
        else:
            if not np.array_equal(freqs_out, tfr.freqs):
                raise ValueError(f"{pa}: frequency grid mismatch")
            if not np.array_equal(times_out, tfr.times):
                raise ValueError(f"{pa}: time grid mismatch")
            if list(ch_names_out) != tfr.ch_names:
                raise ValueError(f"{pa}: channel list mismatch")

        # -------------------------
        # 8) z-score sv and RT within subject & multiple regression
        # -------------------------
        # z-scoring is convenient but not strictly required
        sv_z = (sv_good - sv_good.mean()) / sv_good.std()
        rt_z = (rt_good - rt_good.mean()) / rt_good.std()

        # design matrix: [sv, RT, intercept]
        X = np.column_stack([sv_z, rt_z, np.ones_like(sv_z)])  # (n_trials, 3)

        # check rank / variance
        if np.linalg.matrix_rank(X) < 2:
            print(f"  {pa}: design matrix nearly singular (sv & RT collinear?), skipping.")
            continue

        betas_sv_sub = np.zeros((n_chan, n_freq, n_time), dtype=float)
        # betas_rt_sub = np.zeros_like(betas_sv_sub)  # if you want RT betas too

        # regression at each (chan, freq, time): power ~ sv_z + rt_z + intercept
        for ci in range(n_chan):
            Pw_ci = data[:, ci, :, :]                 # (n_trials, n_freq, n_time)
            Pw_flat = Pw_ci.reshape(n_trials, -1)     # (n_trials, n_freq*n_time)

            # Solve X * B = Y  ->  B: (3, n_freq*n_time)
            B, _, _, _ = np.linalg.lstsq(X, Pw_flat, rcond=None)

            beta_sv_flat = B[0, :]   # sv_pain_para slope controlling for RT
            # beta_rt_flat = B[1, :] # RT slope, if you want it

            betas_sv_sub[ci] = beta_sv_flat.reshape(n_freq, n_time)
            # betas_rt_sub[ci] = beta_rt_flat.reshape(n_freq, n_time)

        all_betas_sv.append(betas_sv_sub)
        # all_betas_rt.append(betas_rt_sub)
        used_subs.append(pa)
        print(f"  {pa}: beta map (sv|RT) computed, n_good trials = {n_trials}")

    # -------------------------
    # 9) save group-level arrays
    # -------------------------
    if len(all_betas_sv) == 0:
        print("No subjects with valid beta maps; nothing saved.")
    else:
        all_betas_sv = np.stack(all_betas_sv)  # (n_subj, n_chan, n_freq, n_time)

        np.save(group_dir / "tfr_beta_sv_pain_para_subxchxfxt.npy", all_betas_sv)
        # If you also want RT betas:
        # all_betas_rt = np.stack(all_betas_rt)
        # np.save(group_dir / "tfr_beta_rt_subxchxfxt.npy", all_betas_rt)

        np.save(group_dir / "tfr_beta_sv_pain_para_subjects.npy",
                np.array(used_subs, dtype=object))
        np.save(group_dir / "tfr_beta_sv_pain_para_freqs.npy", freqs_out)
        np.save(group_dir / "tfr_beta_sv_pain_para_times.npy", times_out)
        np.save(group_dir / "tfr_beta_sv_pain_para_ch_names.npy", ch_names_out)

        print("Saved beta maps for sv_pain_para | RT to:", group_dir)
        print("Shapes: all_betas_sv:", all_betas_sv.shape)
        print("Subjects:", used_subs)









### old

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

