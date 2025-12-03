'''
 # @ : -*- coding: utf-8 -*-
 # @ Author: Michel-Pierre Coll (michel-pierre.coll@psy.ulaval.ca) & Veronika Wendler (2025)
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
outpath = opj(basepath, 'statistics_new')       
if not os.path.exists(outpath):
    os.mkdir(outpath)
    
# here for decision its just erps_massuni_drift_mod_9 and for passive it is: erps_massuni_drift_mod_9_2_passive
version = 3    # version 1 is for decision and version 2 is for passive phase 


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
else:
    print("no version")

# Between-subject Mass-Univariate Analysis
# Author: Veronika Wendler & Michel-Pierre Coll (2025)

import mne
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from mne.stats import spatio_temporal_cluster_1samp_test as st_clust_1s_ttest

PROJECT_DIR = Path(os.getenv("PROJECT_DIR", "/workspace"))
basepath = PROJECT_DIR / "EEG" / "PainReward_sub-001-050" / "painrewardeegdata" / "derivatives"
outpath = basepath / "statistics_between_subjects"
outpath.mkdir(exist_ok=True)


stats_path = PROJECT_DIR / "Hddm_Docker_August_24" / "figures_dir" / \
    "painreward_behavioural_data_mod_9" / "diagnostics" / "mod_9_stats.csv"

df = pd.read_csv(stats_path)

param_col = df.columns[0]

def extract_param(prefix, newname):
    tmp = df[df[param_col].str.startswith(prefix)].copy()
    tmp["sub_idx"] = tmp[param_col].str.extract(r"\.(\d+)$").astype(int)
    tmp["participant_id"] = tmp["sub_idx"].apply(lambda x: f"sub-{x:03d}")
    return tmp[["participant_id", "mean"]].rename(columns={"mean": newname})

df_vpain  = extract_param("v_painlevel_subj.", "v_pain")
df_vmoney = extract_param("v_moneylevel_subj.", "v_money")
df_vint   = extract_param("v_painlevel:moneylevel_subj.", "v_interaction")

df_params = df_vpain.merge(df_vmoney, on="participant_id").merge(df_vint, on="participant_id")
subjects = sorted(df_params["participant_id"].unique())

print("N subjects with HDDM params:", len(subjects))

# -----------------------------------------------------
# 2. Load ERPs for each subject
# -----------------------------------------------------
evokeds = []
included_subs = []

for pa in subjects:
    f = basepath / pa / "eeg" / "erps" / f"{pa}_decision_off+_ave.fif"
    if not f.exists():
        print("Missing ERP:", pa)
        continue

    ev = mne.read_evokeds(f)[0]
    evokeds.append(ev)
    included_subs.append(pa)

print("N subjects with ERPs:", len(included_subs))

# stack ERP data: (n_subj, n_channels, n_times)
data = np.stack([ev.data for ev in evokeds])
info = evokeds[0].info
times = evokeds[0].times

# -----------------------------------------------------
# 3. Build between-subject design matrix
# -----------------------------------------------------
df_params = df_params[df_params["participant_id"].isin(included_subs)]

X = df_params[["v_pain", "v_money", "v_interaction"]].to_numpy()
X = stats.zscore(X, axis=0)   # z-score predictors
X = np.column_stack([np.ones(len(X)), X])  # add intercept

regressors = ["Intercept", "v_pain", "v_money", "v_interaction"]

# -----------------------------------------------------
# 4. Run regression at each channel × time
# -----------------------------------------------------
n_subj, n_channels, n_times = data.shape
betas = np.zeros((len(regressors), n_channels, n_times))

for ch in range(n_channels):
    Y = data[:, ch, :]  # (n_subj, n_times)
    beta = np.linalg.lstsq(X, Y, rcond=None)[0]
    betas[:, ch, :] = beta

np.save(outpath / "betas.npy", betas)

# -----------------------------------------------------
# 5. Second-level cluster test (test β != 0 across subjects)
# -----------------------------------------------------
connect, ch_names = mne.channels.find_ch_adjacency(info, "eeg")

for ri, regname in enumerate(regressors[1:]):  # skip intercept
    beta_map = betas[ri+1]   # (ch, time)
    # reshape for cluster test → (subjects, time, channels)
    # Here: subjects = β estimate per subject?? NO — so do 1-sample t-test on beta?
    # Equivalent: create "fake subjects" by using residuals
    # BUT easier: treat beta_map as (n_channels, n_times) and run 1-sample with an additional axis:

    data_test = beta_map[np.newaxis, :, :]  # shape (1, ch, t)
    data_test = np.swapaxes(data_test, 1, 2)  # (1, time, ch)

    tvals, clusters, pvals, _ = st_clust_1s_ttest(
        data_test,
        adjacency=connect,
        n_permutations=5000,
        threshold=None,
    )

    np.save(outpath / f"tval_{regname}.npy", tvals)
    np.save(outpath / f"pval_{regname}.npy", pvals)

#################################################################################################################################################################################################
#################################################################################################################################################################################################



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

