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
from mne.stats import fdr_correction
from mne.time_frequency import tfr_morlet
from mne.stats import permutation_cluster_1samp_test, combine_adjacency

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
version = 33
v32_mode = "joint"   # "joint" or "separate"


v13_mode = "joint"   # or "joint"s
v17_mode="joint"
v15_mode = "joint"

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
elif version == 8:  #  TFR ROI vs drift (between-subject)
    outpath = opj(outpath, 'tfr_mod_9_v8_drift_ROI')
    if not os.path.exists(outpath):
        os.mkdir(outpath)
elif version == 9:  # TFR trial-wise sv_pain_para betas
    outpath = opj(outpath, 'tfr_mod_9_v9_sv_pain_para_RT_control')
    if not os.path.exists(outpath):
        os.mkdir(outpath)
elif version == 10:
    outpath = opj(outpath, 'tfr_mod_9_v10_sv_vs_pain_RT')
    if not os.path.exists(outpath):
        os.mkdir(outpath)
elif version == 11:
    outpath = opj(outpath, 'erps_massuni_drift_sv_subsetOV')
    os.makedirs(outpath, exist_ok=True)
    if v11_mode == "separate":
        outpath = opj(outpath, "v11_separateGLMs")
    elif v11_mode == "joint":
        outpath = opj(outpath, "v11_jointGLM_pain_money_RT")
    else:
        raise ValueError("v11_mode must be 'separate' or 'joint'")
    os.makedirs(outpath, exist_ok=True)
    
elif version == 12:
    outpath = opj(outpath, 'erps_massuni_drift_sv_subsetVD')
    os.makedirs(outpath, exist_ok=True)
    if v11_mode == "separate":
        outpath = opj(outpath, "v12_separateGLMs")
    elif v11_mode == "joint":
        outpath = opj(outpath, "v12_jointGLM_pain_money_RT")
    else:
        raise ValueError("v12_mode must be 'separate' or 'joint'")
    os.makedirs(outpath, exist_ok=True)

elif version == 13:
    outpath = opj(outpath, 'erps_massuni_drift_sv_subsetVD')
    os.makedirs(outpath, exist_ok=True)
    if v13_mode == "separate":
        outpath = opj(outpath, "v13_separateGLMs")
    elif v13_mode == "joint":
        outpath = opj(outpath, "v13_jointGLM_pain_money_RT")
    else:
        raise ValueError("v13_mode must be 'separate' or 'joint'")
    os.makedirs(outpath, exist_ok=True)

elif version == 14:
    outpath = opj(outpath, 'erps_massuni_drift_sv_subsetAcceptPairM')
    os.makedirs(outpath, exist_ok=True)
    if v14_mode == "separate":
        outpath = opj(outpath, "v18_separateGLMs_raw")
    elif v14_mode == "joint":
        outpath = opj(outpath, "v18_jointGLM_pain_money_RT_raw")
    else:
        raise ValueError("v14_mode must be 'separate' or 'joint'")
    os.makedirs(outpath, exist_ok=True)

elif version == 15:
    outpath = opj(outpath, 'erps_massuni_drift_sv_subsetAcceptPairM')
    os.makedirs(outpath, exist_ok=True)
    if v15_mode == "separate":
        outpath = opj(outpath, "v15_separateGLMs")
    elif v15_mode == "joint":
        outpath = opj(outpath, "v15_jointGLM_pain_money_RT")
    else:
        raise ValueError("v15_mode must be 'separate' or 'joint'")
    os.makedirs(outpath, exist_ok=True)
    
elif version == 16:
    outpath = opj(outpath, 'erps_massuni_drift_sv_subsetVD')
    os.makedirs(outpath, exist_ok=True)
    if v16_mode == "separate":
        outpath = opj(outpath, "v19_separateGLMs")
    elif v16_mode == "joint":
        outpath = opj(outpath, "v19_jointGLM_pain_money_RT")
    else:
        raise ValueError("v15_mode must be 'separate' or 'joint'")
    os.makedirs(outpath, exist_ok=True)
    
elif version == 17:
    outpath = opj(outpath, "erps_massuni_drift_sv_response_classic_rp")
    os.makedirs(outpath, exist_ok=True)
    if v17_mode == "resid_joint":
        outpath = opj(outpath, "v17_resid_joint_SVmoney_RT")
    elif v17_mode == "separate":
        outpath = opj(outpath, "v17_separateGLMs_SVmoney_RT")
    elif v17_mode == "joint":
        outpath = opj(outpath, "v17_jointGLM_SVmoney_RT")
    else:
        raise ValueError("v17_mode must be 'separate' or 'joint' or 'resid_joint")
    os.makedirs(outpath, exist_ok=True)
    
elif version == 18:
    base_v18 = opj(outpath, "erps_massuni_sv_cuelong")
    os.makedirs(base_v18, exist_ok=True)
    if v18_mode == "joint":
        outpath = opj(base_v18, "v18_long_SVmoneypain_RT")
    elif v18_mode == "separate":
        outpath = opj(base_v18, "v18_long_SVmoneypain_RT_sep")
    else:
        raise ValueError("v18_mode must be 'joint' or 'separate'")
    os.makedirs(outpath, exist_ok=True)

elif version == 19:
    base_v19 = opj(outpath, "erps_massuni_sv_cuelong")
    os.makedirs(base_v19, exist_ok=True)
    if v19_mode == "joint":
        outpath = opj(base_v19, "v19_long_SVmoneypain_RT")
    elif v19_mode == "separate":
        outpath = opj(base_v19, "v19_long_SVmoneypain_RT_sep")
    else:
        raise ValueError("v19_mode must be 'joint' or 'separate'")
    os.makedirs(outpath, exist_ok=True)
      
elif version == 19:
    base_v19 = opj(outpath, "erps_massuni_sv_cuelong")
    os.makedirs(base_v19, exist_ok=True)
    if v19_mode == "joint":
        outpath = opj(base_v19, "v19_long_SVmoneypain_RT")
    elif v19_mode == "separate":
        outpath = opj(base_v19, "v19_long_SVmoneypain_RT_sep")
    else:
        raise ValueError("v19_mode must be 'joint' or 'separate'")
    os.makedirs(outpath, exist_ok=True)
       
       
elif version == 20:
    base_v20 = opj(outpath, "erps_massuni_sv_cuelong")
    os.makedirs(base_v20, exist_ok=True)
    if v20_mode == "joint":
        outpath = opj(base_v20, "v20_long_SVmoneypain_RT")
    elif v20_mode == "separate":
        outpath = opj(base_v20, "v20_long_SVmoneypain_RT_sep")
    else:
        raise ValueError("v20_mode must be 'joint' or 'separate'")
    os.makedirs(outpath, exist_ok=True)

elif version == 21:
    base_v21 = opj(outpath, "erps_massuni_sv_cuelong")
    os.makedirs(base_v21, exist_ok=True)
    if v21_mode == "joint":
        outpath = opj(base_v21, "v21_long_SVmoneypain_RT")
    elif v21_mode == "separate":
        outpath = opj(base_v21, "v21_long_SVmoneypain_RT_sep")
    else:
        raise ValueError("v21_mode must be 'joint' or 'separate'")
    os.makedirs(outpath, exist_ok=True)

elif version == 22:
    base_v22 = opj(outpath, "erps_massuni_sv_cuelong")
    os.makedirs(base_v22, exist_ok=True)
    if v22_mode == "joint":
        outpath = opj(base_v22, "v22_high_accept_joint")
    elif v22_mode == "separate":
        outpath = opj(base_v22, "v22_high_accept_sep")
    else:
        raise ValueError("v22_mode must be 'joint' or 'separate'")
    os.makedirs(outpath, exist_ok=True)
    
    

elif version == 23:
    base_v23 = opj(outpath, "erps_massuni_sv_cuelong")
    os.makedirs(base_v23, exist_ok=True)
    if v23_mode == "joint":
        outpath = opj(base_v23, "v23_low_accept_joint")
    elif v23_mode == "separate":
        outpath = opj(base_v23, "v23_low_accept_sep")
    else:
        raise ValueError("v23_mode must be 'joint' or 'separate'")
    os.makedirs(outpath, exist_ok=True)


elif version == 24:
    base_v24 = opj(outpath, "erps_massuni_sv_cuelong")
    os.makedirs(base_v24, exist_ok=True)
    if v24_mode == "joint":
        outpath = opj(base_v24, "v25_low_accept_joint")
    elif v24_mode == "separate":
        outpath = opj(base_v24, "v25_low_accept_sep")
    else:
        raise ValueError("v24_mode must be 'joint' or 'separate'")
    os.makedirs(outpath, exist_ok=True)
    
    
elif version == 25:
    base_v25 = opj(outpath, "erps_massuni_sv_cuelong")
    os.makedirs(base_v25, exist_ok=True)
    if v25_mode == "joint":
        outpath = opj(base_v25, "v25_high_STA_TAI_joint")
    elif v25_mode == "separate":
        outpath = opj(base_v25, "v25_high_STA_TAI_sep")
    else:
        raise ValueError("v25_mode must be 'joint' or 'separate'")
    os.makedirs(outpath, exist_ok=True)
    
elif version == 26:
    base_v26 = opj(outpath, "erps_massuni_sv_cuelong")
    os.makedirs(base_v26, exist_ok=True)
    if v26_mode == "joint":
        outpath = opj(base_v26, "v26_high_STA_TAI_SV_joint")
    elif v26_mode == "separate":
        outpath = opj(base_v26, "v26_high_STA_TAI_SV_sep")
    else:
        raise ValueError("v25_mode must be 'joint' or 'separate'")
    os.makedirs(outpath, exist_ok=True)
    
elif version == 27:
    base_v27 = opj(outpath, "erps_massuni_sv_cuelong")
    os.makedirs(base_v27, exist_ok=True)
    if v27_mode == "joint":
        outpath = opj(base_v27, "v26_high_STA_SAI_joint")
    elif v27_mode == "separate":
        outpath = opj(base_v27, "v26_high_STA_SAI_sep")
    else:
        raise ValueError("v27_mode must be 'joint' or 'separate'")
    os.makedirs(outpath, exist_ok=True)
    
elif version == 28:
    base_v28 = opj(outpath, "erps_massuni_sv_cuelong")
    os.makedirs(base_v28, exist_ok=True)
    if v28_mode == "joint":
        outpath = opj(base_v28, "v26_high_STA_SAI_SV_joint")
    elif v28_mode == "separate":
        outpath = opj(base_v28, "v26_high_STA_SAI_SV_sep")
    else:
        raise ValueError("v28_mode must be 'joint' or 'separate'")
    os.makedirs(outpath, exist_ok=True)
    
elif version == 29:
    base_v29 = opj(outpath, "erps_massuni_sv_cuelong")
    os.makedirs(base_v29, exist_ok=True)
    if v29_mode == "joint":
        outpath = opj(base_v29, "v29_high_PCS_joint")
    elif v29_mode == "separate":
        outpath = opj(base_v29, "v29_high_PCS_sep")
    else:
        raise ValueError("v29_mode must be 'joint' or 'separate'")
    os.makedirs(outpath, exist_ok=True)
    
elif version == 30:
    base_v30 = opj(outpath, "erps_massuni_sv_cuelong")
    os.makedirs(base_v30, exist_ok=True)
    if v30_mode == "joint":
        outpath = opj(base_v30, "v30_male_joint")
    elif v30_mode == "separate":
        outpath = opj(base_v30, "v30_male_sep")
    else:
        raise ValueError("v29_mode must be 'joint' or 'separate'")
    os.makedirs(outpath, exist_ok=True)
    
elif version == 31:
    base_v31 = opj(outpath, "erps_massuni_sv_cuelong")
    os.makedirs(base_v31, exist_ok=True)
    if v31_mode == "joint":
        outpath = opj(base_v31, "v31_first_10_joint")
    elif v31_mode == "separate":
        outpath = opj(base_v31, "v31_first_10_sep")
    else:
        raise ValueError("v29_mode must be 'joint' or 'separate'")
    os.makedirs(outpath, exist_ok=True)
    
elif version == 32:
    base_v32 = opj(outpath, "erps_massuni_sv_cuelong")
    os.makedirs(base_v32, exist_ok=True)
    if v32_mode == "joint":
        outpath = opj(base_v32, "v32_fast_RT_joint")
    elif v32_mode == "separate":
        outpath = opj(base_v32, "v32_fast_RT_sep")
    else:
        raise ValueError("v32_mode must be 'joint' or 'separate'")
    os.makedirs(outpath, exist_ok=True)

elif version == 33:
    base_v33 = opj(outpath, "erps_massuni_sv_cuelong")
    os.makedirs(base_v33, exist_ok=True)
    outpath = opj(base_v33, "v33_money_is5")
    os.makedirs(outpath, exist_ok=True)

elif version == 34:
    base_v34 = opj(outpath, "erps_massuni_sv_cuelong")
    os.makedirs(base_v34, exist_ok=True)
    outpath = opj(base_v34, "v34_RTresid_then_pain_money")
    os.makedirs(outpath, exist_ok=True)

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

if version == 11:
    mod_data = mod_data[mod_data['OV_value'] == 'high_OV'].copy()
    print("After high_OV filter, mod_data rows:", len(mod_data))
    print("After high_OV filter, unique participants:", mod_data['participant'].nunique())

if version == 12:
    mod_data = mod_data[mod_data['Abs_value'] == 'high_abs'].copy()
    print("After high_VD filter, mod_data rows:", len(mod_data))
    print("After high_VD filter, unique participants:", mod_data['participant'].nunique())

if version == 13:
    mod_data = mod_data[mod_data['Abs_value'] == 'low_abs'].copy()
    print("After low_VD filter, mod_data rows:", len(mod_data))
    print("After low_VD filter, unique participants:", mod_data['participant'].nunique())

if version == 14:
    mod_data = mod_data[mod_data['acceptance_pair'] == 'M'].copy()
    print("After acceptance_pair filter, mod_data rows:", len(mod_data))
    print("After acceptance_pair filter, unique participants:", mod_data['participant'].nunique())
    
if version == 15:
    mod_data = mod_data[mod_data['acceptance_pair'] == 'M'].copy()
    print("After acceptance_pair filter, mod_data rows:", len(mod_data))
    print("After acceptance_pair filter, unique participants:", mod_data['participant'].nunique())

if version == 16:
    mod_data = mod_data[mod_data['Abs_Money_Pain'] == 'high_abs_h_money'].copy()
    print("After Abs_Money_Pain filter, mod_data rows:", len(mod_data))
    print("After Abs_Money_Pain filter, unique participants:", mod_data['participant'].nunique())

if version == 22:
    mod_data["accept_mediansplit"] = pd.to_numeric(mod_data["accept_mediansplit"], errors="coerce")
    mod_data = mod_data.loc[mod_data["accept_mediansplit"] == 1].copy()
    print("After high-accept (median split) filter, mod_data rows:", len(mod_data))
    print("After high-accept (median split) filter, unique participants:", mod_data["participant"].nunique())
    
if version == 23:
    mod_data["accept_mediansplit"] = pd.to_numeric(mod_data["accept_mediansplit"], errors="coerce")
    mod_data = mod_data.loc[mod_data["accept_mediansplit"] == 0].copy()
    print("After low-accept (median split) filter, mod_data rows:", len(mod_data))
    print("After low-accept (median split) filter, unique participants:", mod_data["participant"].nunique())
    
if version == 24:
    mod_data["accept_mediansplit"] = pd.to_numeric(mod_data["accept_mediansplit"], errors="coerce")
    mod_data = mod_data.loc[mod_data["accept_mediansplit"] == 0].copy()
    print("After high-accept (median split) filter, mod_data rows:", len(mod_data))
    print("After high-accept (median split) filter, unique participants:", mod_data["participant"].nunique())
    

if version == 25:
    mod_data["sta_tai_mediansplit"] = pd.to_numeric(mod_data["sta_tai_mediansplit"], errors="coerce")
    mod_data = mod_data.loc[mod_data["sta_tai_mediansplit"] == 1].copy()
    print("After high sta_tai_mediansplit (median split) filter, mod_data rows:", len(mod_data))
    print("After high sta_tai_mediansplit (median split) filter, unique participants:", mod_data["participant"].nunique())
    
if version == 26:
    mod_data["sta_tai_mediansplit"] = pd.to_numeric(mod_data["sta_tai_mediansplit"], errors="coerce")
    mod_data = mod_data.loc[mod_data["sta_tai_mediansplit"] == 1].copy()
    print("After high sta_tai_mediansplit (median split) filter, mod_data rows:", len(mod_data))
    print("After high sta_tai_mediansplit (median split) filter, unique participants:", mod_data["participant"].nunique())
    

if version == 27:
    mod_data["sta_sai_mediansplit"] = pd.to_numeric(mod_data["sta_sai_mediansplit"], errors="coerce")
    mod_data = mod_data.loc[mod_data["sta_sai_mediansplit"] == 1].copy()
    print("After high sta_sai_mediansplit (median split) filter, mod_data rows:", len(mod_data))
    print("After high sta_sai_mediansplit (median split) filter, unique participants:", mod_data["participant"].nunique())
    
if version == 28:
    mod_data["sta_sai_mediansplit"] = pd.to_numeric(mod_data["sta_sai_mediansplit"], errors="coerce")
    mod_data = mod_data.loc[mod_data["sta_sai_mediansplit"] == 1].copy()
    print("After high sta_sai_mediansplit (median split) filter, mod_data rows:", len(mod_data))
    print("After high sta_sai_mediansplit (median split) filter, unique participants:", mod_data["participant"].nunique())
    
if version == 29:
    mod_data["pcs_mediansplit"] = pd.to_numeric(mod_data["pcs_mediansplit"], errors="coerce")
    mod_data = mod_data.loc[mod_data["pcs_mediansplit"] == 1].copy()
    print("After high pcs_mediansplit (median split) filter, mod_data rows:", len(mod_data))
    print("After high pcs_mediansplit (median split) filter, unique participants:", mod_data["participant"].nunique())
    
if version == 30:
    mod_data["sex_bin"] = pd.to_numeric(mod_data["sex_bin"], errors="coerce")
    mod_data = mod_data.loc[mod_data["sex_bin"] == 1].copy()
    print("After sex_bin m filter, mod_data rows:", len(mod_data))
    print("After sex_bin m filter, unique participants:", mod_data["participant"].nunique())

if version == 31:
    mod_data["first10_flag"] = pd.to_numeric(mod_data["first10_flag"], errors="coerce")
    mod_data = mod_data.loc[mod_data["first10_flag"] == 1].copy()
    print("After first25_flag filter, mod_data rows:", len(mod_data))
    print("After first25_flag filter, unique participants:", mod_data["participant"].nunique())
    
if version == 32:
    mod_data["rt_fast_slow"] = pd.to_numeric(mod_data["rt_fast_slow"], errors="coerce")
    mod_data = mod_data.loc[mod_data["rt_fast_slow"] == 1].copy()
    print("After rt_fast_slow filter, mod_data rows:", len(mod_data))
    print("After rt_fast_slow filter, unique participants:", mod_data["participant"].nunique())
    

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



def residualize(y, X):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta

def residualize_epochs_data(data, Z):
    """
    Residualize trial-wise EEG data on nuisance regressors.

    data: (n_trials, n_channels, n_times)
    Z:    (n_trials, n_nuisance) e.g. [Intercept, RT_z]

    returns: residualized data with same shape as data
    """
    n_trials, n_ch, n_t = data.shape
    Y = data.reshape(n_trials, n_ch * n_t)             
    beta, *_ = np.linalg.lstsq(Z, Y, rcond=None)       
    Y_res = Y - Z @ beta
    return Y_res.reshape(n_trials, n_ch, n_t)

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

raw_regcols = ['painlevel', 'moneylevel', 'interaction']
regvars = raw_regcols  


if version == 7:
    regvars = ['sv_pain_para']
    regvarsnames = ['SV_pain_para']
if version == 11:
    regvars = ['sv_pain_para', 'sv_money']
if version == 12:
    regvars = ['sv_pain_para', 'sv_money']
if version == 18:
    regvars = ['sv_pain_para', 'sv_money']
if version == 14:
    regvars = ['painlevel', 'moneylevel']
if version == 32:
    regvars = ['painlevel', 'moneylevel']
if version == 24:
    regvars = ['sv_pain_para']
    regvarsnames = ['SV_pain_para']
if version == 26:
    regvars = ['sv_pain_para', 'sv_money']
if version == 28:
    regvars = ['sv_pain_para', 'sv_money']
#all_epos = [[] for i in range(len(regvars))]
#allbetasnp = []
#betas = [[] for i in range(len(regvars))]
part.sort()

#------------------------------------------------------------------------------------------------------------------------------------------------
# Creating the dataframes

#------------------------------------------------------------------------------------------------------------------------------------------------
# Creating the dataframes (only needed for versions 1–4)

if version in [1, 2, 3, 4, 7, 11, 12, 13, 14,15,16,17, 18,19, 20,21, 22,23,24,25,26,27,28,29,30,31,32,33]:
    filtered_data = []
    for p in part:
        # data for this participant
        df = mod_data[mod_data['participant'] == p]
        
        # Load single epochs file
        if version == 1:
            epo = mne.read_epochs(opj(basepath,  p, 'eeg', 'erps_passive',                   
                                  p + '_passive_cues_singletrials-epo.fif'))
            epo_1 = epo.copy()
        elif version in [2, 3, 4, 7, 11, 12, 13, 14,15, 16]:
            epo = mne.read_epochs(opj(basepath,  p, 'eeg', 'erps',                   
                                  p + '_decision_cues_singletrials-epo.fif'))
            epo_1 = epo.copy()
        elif version == 17:
            # epo = mne.read_epochs(
            #     opj(basepath, p, "eeg", "erps_resp_rp", f"{p}_decision_resp_rp_singletrials-epo.fif"),
            #     preload=True)
            # epo_1 = epo.copy()
            epo = mne.read_epochs(
                opj(basepath, p, "eeg", "erps_resp", f"{p}_decision_resp_singletrials-epo.fif"),
                preload=True)
            epo_1 = epo.copy()
        elif version in [18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33]:
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

if version in [1, 2, 3]:

    all_epos = [[] for _ in range(len(regvars))]  
    allbetasnp = []                               
    betas = [[] for _ in range(len(regvars))]     

    included_subjects = []
    skipped_subjects = []

    z_dir = Path(outpath) / "Zscoring"
    z_dir.mkdir(parents=True, exist_ok=True)

    for pa in part_1:
        print(f"\n--- Z-Scored Version (v{version}): Processing {pa} ---")

        # Behavioural tables for this participant
        df2 = epo_1_filtered_combined[epo_1_filtered_combined['participant_id'] == pa]
        mod2 = part_1_dat[part_1_dat['participant'] == pa]

        # Load epochs
        if version == 1:
            # passive
            epo = mne.read_epochs(
                opj(basepath, pa, 'eeg', 'erps_passive',
                    pa + '_passive_cues_singletrials-epo.fif')
            )
        elif version in [2, 3]:
            # decision
            epo = mne.read_epochs(
                opj(basepath, pa, 'eeg', 'erps',
                    pa + '_decision_cues_singletrials-epo.fif')
            )

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

        # Z-score EEG across trials
        scale = Scaler(scalings='mean')
        epo_z = mne.EpochsArray(scale.fit_transform(epo_filt.get_data()),
                                epo_filt.info)

        if len(df2) < 5:
            print(f"Skipping {pa} as only {len(df2)} working trials after cleaning")
            skipped_subjects.append(pa)
            continue

        # RT column
        if "rt" in mod2.columns:
            rt_col = "rt"
        else:
            raise ValueError(f"No RT column in mod_data. Columns: {mod2.columns}")

        betasnp = []
        subject_has_regressors = False

        for idx, regvar in enumerate(regvars):

            print(f"  Subject {pa} – regressor: {regvar}")

            # Basic finite-mask on regvar and RT
            vals_reg = mod2[regvar].to_numpy(dtype=float)
            vals_rt = mod2[rt_col].to_numpy(dtype=float)

            # For interaction GLM we also need painlevel as covariate
            if regvar == "interaction":
                vals_pain = mod2["painlevel"].to_numpy(dtype=float)
                keep = np.where(
                    np.isfinite(vals_reg) &
                    np.isfinite(vals_rt) &
                    np.isfinite(vals_pain)
                )[0]
            else:
                keep = np.where(
                    np.isfinite(vals_reg) &
                    np.isfinite(vals_rt)
                )[0]

            if len(keep) < 5:
                print(f"    Skipping {regvar} as only {len(keep)} valid trials")
                continue

            df_reg = mod2.iloc[keep].copy()
            epo_reg = epo_z.copy()[keep]
            epo_keep = epo_filt.copy()[keep]

            # Variance checks
            if np.nanstd(df_reg[regvar]) == 0:
                print(f"    Skipping {regvar} due to zero variance")
                continue
            if np.nanstd(df_reg[rt_col]) == 0:
                print(f"    Skipping {regvar} as RT has zero variance (subject {pa})")
                continue

            # -------------------------------
            # Build design matrix (all Z-scored predictors)
            # -------------------------------
            df_reg["Intercept"] = 1.0

            # Z-score RT
            df_reg["RT_z"] = stats.zscore(df_reg[rt_col].to_numpy(dtype=float))

            if regvar in ["painlevel", "moneylevel"]:
                # Simple GLM: EEG ~ Intercept + regvar_z + RT_z
                reg_z_name = regvar + "_z"
                df_reg[reg_z_name] = stats.zscore(df_reg[regvar].to_numpy(dtype=float))

                design = df_reg[["Intercept", reg_z_name, "RT_z"]]
                names = ["Intercept", reg_z_name, "RT_z"]
                beta_key = reg_z_name

            elif regvar == "interaction":
                # Interaction GLM:
                #   EEG ~ Intercept + interaction_z + pain_z + RT_z
                df_reg["interaction_z"] = stats.zscore(
                    df_reg["interaction"].to_numpy(dtype=float)
                )
                df_reg["pain_z"] = stats.zscore(
                    df_reg["painlevel"].to_numpy(dtype=float)
                )

                design = df_reg[["Intercept", "interaction_z", "pain_z", "RT_z"]]
                names = ["Intercept", "interaction_z", "pain_z", "RT_z"]
                beta_key = "interaction_z"

            # safety check
            if not np.all(np.isfinite(design.to_numpy())):
                print(f"    Skipping {regvar}: design matrix has NaN/Inf")
                continue

            # update metadata (mainly for later plotting)
            df_meta = epo_keep.metadata.reset_index(drop=True).copy()
            df_meta[regvar] = df_reg[regvar].values
            df_meta[rt_col] = df_reg[rt_col].values
            epo_keep.metadata = df_meta

            # store epochs per regressor
            all_epos[idx].append(epo_keep)

            # regression: EEG ~ design
            res = mne.stats.linear_regression(
                epo_reg,
                design,
                names=names
            )

            beta_reg = res[beta_key].beta  # Evoked object
            betas[idx].append(beta_reg)
            betasnp.append(beta_reg.data)

            subject_has_regressors = True

        if not subject_has_regressors:
            print(f"  Skipping {pa} as no valid regressors")
            skipped_subjects.append(pa)
            continue

        included_subjects.append(pa)
        allbetasnp.append(np.stack(betasnp))  # shape (n_reg_valid, n_chan, n_time) for this subject
        print(f"Included {pa}")

    # ---------------------------------------------------------------------
    # Stack all betas across subjects → (n_subj, n_reg, n_chan, n_time)
    # ---------------------------------------------------------------------
    allbetas = np.stack(allbetasnp)
    print(f"\nTotal subjects: {len(part_1)}")
    print(f"Included ({len(included_subjects)}): {included_subjects}")
    print(f"Skipped  ({len(skipped_subjects)}): {skipped_subjects}")

    # Grand average maps for each regressor
    beta_gavg = []
    for idx, regvar in enumerate(regvars):
        beta_gavg.append(mne.grand_average(betas[idx]))

    # connectivity for cluster test (from last epo_filt)
    connect, names_ch = mne.channels.find_ch_adjacency(epo_filt.info, ch_type='eeg')

    # cluster threshold
    if not isinstance(param['cluster_threshold'], dict):
        p_thresh = param['cluster_threshold'] / 2
        n_samples = allbetas.shape[0]
        cluster_threshold = -stats.t.ppf(p_thresh, n_samples - 1)
    else:
        cluster_threshold = param['cluster_threshold']

    # ---------------------------------------------------------------------
    # Second-level cluster tests for each regressor
    # ---------------------------------------------------------------------
    tvals, pvalues = [], []
    for idx, regvar in enumerate(regvars):
        print(f"\nSecond-level cluster test for regressor: {regvar}")

        data_reg = allbetas[:, idx, :, :]      # (n_subj, n_chan, n_time)
        testdata = np.swapaxes(data_reg, 2, 1) # (n_subj, n_time, n_chan)

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

        np.save(z_dir / f'ols_2ndlevel_tval_{regvar}.npy', tvals[-1])
        np.save(z_dir / f'ols_2ndlevel_pval_{regvar}.npy', pvalues[-1])

    # Stack and save group-level results
    tvals = np.stack(tvals)
    pvals = np.stack(pvalues)
    
    min_cluster_ps = []
    for pmap in pvalues:
        # consider only points that came from clusters (p < 1)
        mask = pmap < 1.0
        if np.any(mask):
            min_cluster_ps.append(pmap[mask].min())
        else:
            min_cluster_ps.append(1.0)
    
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

    # Save concatenated epochs per regressor
    for idx, regvar in enumerate(regvars):
        if len(all_epos[idx]) == 0:
            continue
        epo_save = mne.concatenate_epochs(all_epos[idx])
        epo_save.save(z_dir / f'ols_2ndlevel_allepochs-epo_{regvar}.fif', overwrite=True)

    np.save(z_dir / 'ols_2ndlevel_betasavg.npy', beta_gavg)

    # ---------------------------------------------------------------------
    # Pain – interaction beta-difference map: β_pain - β_interaction
    # ---------------------------------------------------------------------
    print("\nComputing pain – interaction beta-difference cluster test...")

    if "painlevel" not in regvars or "interaction" not in regvars:
        raise ValueError(f"painlevel or interaction not in regvars: {regvars}")

    pain_idx = regvars.index("painlevel")
    inter_idx = regvars.index("interaction")

    data_pain = allbetas[:, pain_idx, :, :]       # (n_subj, n_chan, n_time)
    data_inter = allbetas[:, inter_idx, :, :]     # (n_subj, n_chan, n_time)

    beta_diff = data_pain - data_inter            # β_pain - β_interaction

    testdata_diff = np.swapaxes(beta_diff, 2, 1)  # (n_subj, n_time, n_chan)

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

    np.save(z_dir / 'ols_2ndlevel_tval_diff_pain_minus_interaction.npy', tval_diff)
    np.save(z_dir / 'ols_2ndlevel_pval_diff_pain_minus_interaction.npy', pvals_diff)

    print("Saved pain–interaction beta-difference maps in", z_dir)

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
    
    # frontal N2 ROI block
    n2_roi_chs = ['Fz', 'FCz', 'Cz']
    n2_tmin, n2_tmax = 0.20, 0.40

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

    # only boundary-separation (a) parameters
    a_subj_cols = [
        'a_painlevel_subj',
        'a_moneylevel_subj',
        'a_interaction_subj',
    ]

    # subject-level RT (from mod_data, model 9)
    rt_df = (
        mod_data[mod_data["participant"].isin(subj_ids_epochs)]
        .groupby("participant")[["rt"]]
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

    # merge RT + a on participant
    subj_reg = rt_df.merge(subj_reg_a, on="participant", how="inner")

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

    # final list of regressors: ONLY boundary separation terms
    regvars_v5 = a_subj_cols

    # ------------------------------------------------------------------
    # helper for cluster-based between-subject GLM (parallel to v1–3)
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

            # Orthogonalise regressor with respect to RT
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

            # subject-level effect maps: effect_s(chan, time) = x_res(s) * EEG_s(chan, time)
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

    # three variants: NO_Zscoring, Zscoring, PartZscoring
    
    # NO_Zscoring, using raw a betas, raw RT
    betas_noz, tvals_noz, pvals_noz, noz_dir_v5 = run_group_glm_variant(
        subdir_name="NO_Zscoring",
        zscore_reg=False,
        zscore_rt=False
    )

    # Zscoring all predictors (a, RT)
    betas_z, tvals_z, pvals_z, z_dir_v5 = run_group_glm_variant(
        subdir_name="Zscoring",
        zscore_reg=True,
        zscore_rt=True
    )

    # PartZscoring, z-scored RT only
    betas_partz, tvals_partz, pvals_partz, partz_dir_v5 = run_group_glm_variant(
        subdir_name="PartZscoring",
        zscore_reg=False,
        zscore_rt=True
    )

    # cluster-based versions (optional – still loop over the 3 a-regressors)
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
    
    roi_picks = mne.pick_channels(group_epochs.info['ch_names'], lpp_roi_chs)
    if len(roi_picks) == 0:
        raise RuntimeError(f"None of the LPP ROI channels found in data: {lpp_roi_chs}")
    
    tmask = (group_epochs.times >= lpp_tmin) & (group_epochs.times <= lpp_tmax)
    if not np.any(tmask):
        raise RuntimeError(f"No time points in LPP window {lpp_tmin}–{lpp_tmax} s for cue-locked epochs.")
    
    data_roi = data[:, roi_picks][:, :, tmask]   # subj × ROI-ch × time
    y_LPP = data_roi.mean(axis=(1, 2))           # subj-level LPP amplitude
    
    corr_rows = []
    rt_vals = subj_reg["rt"].to_numpy(dtype=float)
    
    for regvar in regvars_v5:
        x = subj_reg[regvar].to_numpy(dtype=float)
    
        keep = np.isfinite(x) & np.isfinite(y_LPP) & np.isfinite(rt_vals)
        n = keep.sum()
        if n < 5:
            print(f"Skipping ROI correlation for {regvar}: only {n} valid subjects")
            continue
    
        x_k = x[keep]
        y_k = y_LPP[keep]
        rt_k = rt_vals[keep]
    
        # simple Pearson correlation (no RT control)
        r_raw, p_raw = pearsonr(x_k, y_k)
    
        # partial correlation controlling for RT
        X_rt = np.column_stack([np.ones(n), rt_k])
        beta_y, _, _, _ = np.linalg.lstsq(X_rt, y_k, rcond=None)
        y_res = y_k - X_rt @ beta_y
    
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
    
    # save summary table + FDR across the 3 a-hypotheses (LPP)
    if len(corr_rows) > 0:
        corr_df = pd.DataFrame(corr_rows)
        pvals = corr_df["p_partial_RT"].to_numpy(dtype=float)
        rej, pvals_fdr = fdr_correction(pvals, alpha=0.05, method='indep')

        corr_df["p_partial_RT_FDR"] = pvals_fdr
        corr_df["sig_partial_RT_FDR"] = rej  # True = survives FDR
        corr_df.to_csv(noz_dir_v5 / "ROI_LPP_vs_a_regressors.csv", index=False)
        print("Saved ROI_LPP_vs_a_regressors.csv in", noz_dir_v5)

    #---------------------------------------------------------------------------------------------------------------------
    print("\n ROI-level between-subject correlations: each regressor vs N2 (cue-locked, version 5)")

    n2_picks = mne.pick_channels(group_epochs.info['ch_names'], n2_roi_chs)
    if len(n2_picks) == 0:
        raise RuntimeError(f"None of the N2 ROI channels found in data: {n2_roi_chs}")
    
    n2_tmask = (group_epochs.times >= n2_tmin) & (group_epochs.times <= n2_tmax)
    if not np.any(n2_tmask):
        raise RuntimeError(f"No time points in N2 window {n2_tmin}–{n2_tmax} s for cue-locked epochs.")
    
    data_n2 = data[:, n2_picks][:, :, n2_tmask]   # subj × ch × time
    y_N2 = data_n2.mean(axis=(1, 2))              # subj-level N2 amplitude
    
    corr_rows_N2 = []
    rt_vals = subj_reg["rt"].to_numpy(dtype=float)
    
    for regvar in regvars_v5:
        x = subj_reg[regvar].to_numpy(dtype=float)
    
        keep = np.isfinite(x) & np.isfinite(y_N2) & np.isfinite(rt_vals)
        n = keep.sum()
        if n < 5:
            print(f"Skipping N2 ROI correlation for {regvar}: only {n} valid subjects")
            continue
    
        x_k = x[keep]
        y_k = y_N2[keep]
        rt_k = rt_vals[keep]
    
        # raw correlation
        r_raw, p_raw = pearsonr(x_k, y_k)
    
        # partial (RT-controlled)
        X_rt = np.column_stack([np.ones(n), rt_k])
        beta_y, _, _, _ = np.linalg.lstsq(X_rt, y_k, rcond=None)
        y_res = y_k - X_rt @ beta_y
    
        beta_x, _, _, _ = np.linalg.lstsq(X_rt, x_k, rcond=None)
        x_res = x_k - X_rt @ beta_x
    
        r_par, p_par = pearsonr(x_res, y_res)
    
        print(f"N2 {regvar}: r_raw = {r_raw:.3f} (p={p_raw:.3g}), "
              f"r_partial_RT = {r_par:.3f} (p={p_par:.3g}), n={n}")
    
        corr_rows_N2.append(dict(
            regressor=regvar,
            n=n,
            r_raw=r_raw,
            p_raw=p_raw,
            r_partial_RT=r_par,
            p_partial_RT=p_par
        ))
    
    # FDR across the 3 a-hypotheses (N2)
    if len(corr_rows_N2) > 0:
        corr_df_N2 = pd.DataFrame(corr_rows_N2)
        pvals_N2 = corr_df_N2["p_partial_RT"].to_numpy(dtype=float)
        rej_N2, pvals_N2_fdr = fdr_correction(pvals_N2, alpha=0.05, method='indep')

        corr_df_N2["p_partial_RT_FDR"] = pvals_N2_fdr
        corr_df_N2["sig_partial_RT_FDR"] = rej_N2
        corr_df_N2.to_csv(noz_dir_v5 / "ROI_N2_vs_a_regressors.csv", index=False)
        print("Saved ROI_N2_vs_a_regressors.csv in", noz_dir_v5)
    
    #---------------------------------------------------------------------------------------------------------------------
    print("\n ROI-level between-subject correlations: each regressor vs P3b (cue-locked, version 5)")
    
    # P3b ROI definition (cue-locked)
    p3b_roi_chs = ['Pz', 'P3', 'P4']
    p3b_tmin, p3b_tmax = 0.25, 0.55
    
    p3b_picks = mne.pick_channels(group_epochs.info['ch_names'], p3b_roi_chs)
    if len(p3b_picks) == 0:
        raise RuntimeError(f"None of the P3b ROI channels found in data: {p3b_roi_chs}")
    
    p3b_tmask = (group_epochs.times >= p3b_tmin) & (group_epochs.times <= p3b_tmax)
    if not np.any(p3b_tmask):
        raise RuntimeError(f"No time points in P3b window {p3b_tmin}–{p3b_tmax} s")
    
    # subj × ch × time → subj-level mean P3b
    data_p3b = data[:, p3b_picks][:, :, p3b_tmask]
    y_P3b = data_p3b.mean(axis=(1, 2))
    
    corr_rows_P3b = []
    rt_vals = subj_reg["rt"].to_numpy(dtype=float)
    
    for regvar in regvars_v5:   # your 3 a-parameters
        x = subj_reg[regvar].to_numpy(dtype=float)
    
        keep = np.isfinite(x) & np.isfinite(y_P3b) & np.isfinite(rt_vals)
        n = keep.sum()
        if n < 5:
            print(f"Skipping P3b ROI correlation for {regvar}: only {n} valid subjects")
            continue
    
        x_k = x[keep]
        y_k = y_P3b[keep]
        rt_k = rt_vals[keep]
    
        # raw correlation
        r_raw, p_raw = pearsonr(x_k, y_k)
    
        # partial correlation controlling for RT
        X_rt = np.column_stack([np.ones(n), rt_k])
        beta_y, _, _, _ = np.linalg.lstsq(X_rt, y_k, rcond=None)
        y_res = y_k - X_rt @ beta_y
    
        beta_x, _, _, _ = np.linalg.lstsq(X_rt, x_k, rcond=None)
        x_res = x_k - X_rt @ beta_x
    
        r_par, p_par = pearsonr(x_res, y_res)
    
        print(
            f"P3b {regvar}: r_raw = {r_raw:.3f} (p={p_raw:.3g}), "
            f"r_partial_RT = {r_par:.3f} (p={p_par:.3g}), n={n}"
        )
    
        corr_rows_P3b.append(dict(
            regressor=regvar,
            n=n,
            r_raw=r_raw,
            p_raw=p_raw,
            r_partial_RT=r_par,
            p_partial_RT=p_par
        ))
    
    # FDR across the 3 a-hypotheses for P3b
    if len(corr_rows_P3b) > 0:
        corr_df_P3b = pd.DataFrame(corr_rows_P3b)
        pvals_P3b = corr_df_P3b["p_partial_RT"].to_numpy(dtype=float)
        rej_P3b, pvals_P3b_fdr = fdr_correction(pvals_P3b, alpha=0.05, method='indep')
    
        corr_df_P3b["p_partial_RT_FDR"] = pvals_P3b_fdr
        corr_df_P3b["sig_partial_RT_FDR"] = rej_P3b
    
        corr_df_P3b.to_csv(noz_dir_v5 / "ROI_P3b_vs_a_regressors.csv", index=False)
        print("Saved ROI_P3b_vs_a_regressors.csv in", noz_dir_v5)
    
    #---------------------------------------------------------------------------------------------------------------------
    from mne.stats import permutation_cluster_1samp_test
    
    print("\n LPP ROI time-resolved cluster regression (RT-controlled)")

    # channel & time selection for LPP
    roi_picks = mne.pick_channels(group_epochs.info['ch_names'], lpp_roi_chs)
    if len(roi_picks) == 0:
        raise RuntimeError(f"LPP ROI channels not found: {lpp_roi_chs}")
    
    times = group_epochs.times
    tmask = (times >= lpp_tmin) & (times <= lpp_tmax)
    if not np.any(tmask):
        raise RuntimeError(f"No time points in {lpp_tmin}–{lpp_tmax}s window")
    
    data_roi = data[:, roi_picks][:, :, tmask]      # subj × ROIchan × time
    data_roi_mean = data_roi.mean(axis=1)           # subj × time
    
    rt_vals = subj_reg["rt"].to_numpy(dtype=float)
    
    roi_cluster_dir = Path(outpath) / "LPP_ROI_cluster"
    roi_cluster_dir.mkdir(parents=True, exist_ok=True)
    
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
    
        # RT-controlled regressor (residualisation)
        X_rt = np.column_stack([np.ones(n_k), rt_k])
        beta_cov, _, _, _ = np.linalg.lstsq(X_rt, x_k, rcond=None)
        x_res = x_k - X_rt @ beta_cov
    
        # Subject-level effect maps: subj × time
        effect = np.zeros_like(y_k)
        for s in range(n_k):
            effect[s] = x_res[s] * y_k[s]
    
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

    #---------------------------------------------------------------------
    print("\n N2 ROI time-resolved cluster regression (RT-controlled)")

    n2_picks = mne.pick_channels(group_epochs.info['ch_names'], n2_roi_chs)
    if len(n2_picks) == 0:
        raise RuntimeError(f"N2 ROI channels not found: {n2_roi_chs}")
    
    times = group_epochs.times
    n2_tmask = (times >= n2_tmin) & (times <= n2_tmax)
    if not np.any(n2_tmask):
        raise RuntimeError(f"No time points in {n2_tmin}–{n2_tmax}s window")
    
    data_n2_roi = data[:, n2_picks][:, :, n2_tmask]
    data_n2_mean = data_n2_roi.mean(axis=1)   # subj × time
    
    rt_vals = subj_reg["rt"].to_numpy(dtype=float)
    
    n2_cluster_dir = Path(outpath) / "N2_ROI_cluster"
    n2_cluster_dir.mkdir(parents=True, exist_ok=True)
    
    for regvar in regvars_v5:
        print(f"\nN2 ROI cluster test for regressor: {regvar}")
    
        x = subj_reg[regvar].to_numpy(dtype=float)
    
        keep = (
            np.isfinite(x) &
            np.isfinite(rt_vals) &
            np.all(np.isfinite(data_n2_mean), axis=1)
        )
    
        if keep.sum() < 5:
            print(f"  Skipping {regvar}: only {keep.sum()} valid subjects")
            continue
    
        x_k  = x[keep]
        rt_k = rt_vals[keep]
        y_k  = data_n2_mean[keep]   # subj × time
        n_k  = y_k.shape[0]
    
        # residualise regressor wrt RT
        X_rt = np.column_stack([np.ones(n_k), rt_k])
        beta_cov, _, _, _ = np.linalg.lstsq(X_rt, x_k, rcond=None)
        x_res = x_k - X_rt @ beta_cov
    
        # subject-level effect maps
        effect = np.zeros_like(y_k)
        for s in range(n_k):
            effect[s] = x_res[s] * y_k[s]
    
        t_obs, clusters, cluster_p, _ = permutation_cluster_1samp_test(
            effect,
            n_permutations=5000,
            threshold=None,
            tail=0,
            out_type='mask',
            verbose=False
        )
    
        pvals = np.ones(effect.shape[1])
        for clu, p in zip(clusters, cluster_p):
            pvals[clu] = p
    
        np.save(n2_cluster_dir / f"N2ROI_tval_{regvar}.npy", t_obs)
        np.save(n2_cluster_dir / f"N2ROI_pval_{regvar}.npy", pvals)
    
        print(f"  Saved N2 ROI cluster results for {regvar}")
        
    #---------------------------------------------------------------------
    print("\n P3b ROI time-resolved cluster regression (RT-controlled)")
    
    # P3b ROI definition (cue-locked)
    p3b_roi_chs = ['Pz', 'P3', 'P4']
    p3b_tmin, p3b_tmax = 0.25, 0.55
    
    # channel & time selection
    p3b_picks = mne.pick_channels(group_epochs.info['ch_names'], p3b_roi_chs)
    if len(p3b_picks) == 0:
        raise RuntimeError(f"P3b ROI channels not found: {p3b_roi_chs}")
    
    times = group_epochs.times
    p3b_tmask = (times >= p3b_tmin) & (times <= p3b_tmax)
    if not np.any(p3b_tmask):
        raise RuntimeError(f"No time points in {p3b_tmin}–{p3b_tmax}s window")
    
    # subj × ROIchan × time
    data_p3b_roi = data[:, p3b_picks][:, :, p3b_tmask]
    # mean across ROI channels → subj × time
    data_p3b_mean = data_p3b_roi.mean(axis=1)
    
    rt_vals = subj_reg["rt"].to_numpy(dtype=float)
    
    p3b_cluster_dir = Path(outpath) / "P3b_ROI_cluster"
    p3b_cluster_dir.mkdir(parents=True, exist_ok=True)
    
    for regvar in regvars_v5:   # here regvars_v5 should be your 3 a_* columns
        print(f"\nP3b ROI cluster test for regressor: {regvar}")
    
        x = subj_reg[regvar].to_numpy(dtype=float)
    
        keep = (
            np.isfinite(x) &
            np.isfinite(rt_vals) &
            np.all(np.isfinite(data_p3b_mean), axis=1)
        )
    
        if keep.sum() < 5:
            print(f"  Skipping {regvar}: only {keep.sum()} valid subjects")
            continue
    
        x_k  = x[keep]
        rt_k = rt_vals[keep]
        y_k  = data_p3b_mean[keep]   # subj × time
        n_k  = y_k.shape[0]
    
        # residualise regressor wrt RT
        X_rt = np.column_stack([np.ones(n_k), rt_k])
        beta_cov, _, _, _ = np.linalg.lstsq(X_rt, x_k, rcond=None)
        x_res = x_k - X_rt @ beta_cov
    
        # subject-level effect maps: subj × time
        effect = np.zeros_like(y_k)
        for s in range(n_k):
            effect[s] = x_res[s] * y_k[s]
    
        # cluster test over TIME ONLY
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
        np.save(p3b_cluster_dir / f"P3bROI_tval_{regvar}.npy", t_obs)
        np.save(p3b_cluster_dir / f"P3bROI_pval_{regvar}.npy", pvals)
    
        print(f"  Saved P3b ROI cluster results for {regvar}")

    print(f"\nVersion 5 finished. Subject-level GLM + ROI correlations saved in:\n  {noz_dir_v5}\n  {z_dir_v5}\n  {partz_dir_v5}")

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
#         # 7) compute TFR from these epochs
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
        # betas_rt_sub = np.zeros_like(betas_sv_sub)  

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


elif version == 10:
    print("\n Version 10: TFR trial-wise betas for sv_pain_para AND painlevel "
          "(cue-locked; sv|pain+RT and pain|sv+RT) ")

    # ---------------------------------------------------------------------
    # 0) sanity checks
    # ---------------------------------------------------------------------
    required_cols = ["sv_pain_para", "painlevel", "trialsnum", "rt", "participant"]
    missing = [c for c in required_cols if c not in mod_data.columns]
    if missing:
        raise ValueError(f"mod_data missing columns: {missing}")

    group_dir = Path(outpath)
    group_dir.mkdir(parents=True, exist_ok=True)

    # TFR parameters
    freqs = np.arange(4., 101., 1.)
    n_cycles = 0.5 * freqs
    target_sfreq = 256.0
    tmin_crop = -0.50
    tmax_crop = 1.00

    # outputs (subject-level beta maps)
    all_betas_sv = []       # (subj, ch, f, t)  sv slope controlling pain+rt
    all_betas_pain = []     # (subj, ch, f, t)  pain slope controlling sv+rt
    used_subs = []

    freqs_out = None
    times_out = None
    ch_names_out = None
    info_out = None


    for pa in part:
        print(f"\nSubject {pa}...")

        # behavioural side
        beh_sub = mod_data[mod_data["participant"] == pa].copy()
        if beh_sub.empty:
            print(f"  No behavioural rows in mod_data for {pa}, skipping.")
            continue

        beh_sub = beh_sub[["trialsnum", "painlevel", "sv_pain_para", "rt"]].copy()
        beh_sub["trialsnum"] = beh_sub["trialsnum"].astype(int)

        # ERP epochs (trial alignment + badtrial)
        epo_fname = opj(basepath, pa, "eeg", "erps", f"{pa}_decision_cues_singletrials-epo.fif")
        if not os.path.exists(epo_fname):
            print(f"  No ERP single-trials file for {pa}, skipping.")
            continue

        epo = mne.read_epochs(epo_fname, preload=True)
        meta = epo.metadata.copy()

        required_meta = {"trialsnum", "badtrial"}
        if not required_meta.issubset(meta.columns):
            raise ValueError(f"{pa}: ERP metadata missing {required_meta - set(meta.columns)}")

        if len(epo) < 5:
            print(f"  {pa}: <5 ERP trials, skipping.")
            continue

        # resample to target sfreq
        if epo.info["sfreq"] != target_sfreq:
            epo = epo.copy().resample(target_sfreq)

        meta = epo.metadata.copy()
        meta["trialsnum"] = meta["trialsnum"].astype(int)

        # merge ERP meta with behaviour by trialsnum
        meta = meta.reset_index().rename(columns={"index": "row_id"})
        merged = meta.merge(beh_sub, on="trialsnum", how="inner")

        if merged.empty:
            print(f"  {pa}: no overlapping trials by trialsnum, skipping.")
            continue

        idx_keep = merged["row_id"].to_numpy(dtype=int)
        if len(idx_keep) < 5:
            print(f"  {pa}: only {len(idx_keep)} trials after matching, skipping.")
            continue

        epo_match = epo[idx_keep]
        merged = merged.reset_index(drop=True)

        # filter good trials
        bad  = merged["badtrial"].to_numpy(dtype=float)
        sv   = merged["sv_pain_para"].to_numpy(dtype=float)
        pain = merged["painlevel"].to_numpy(dtype=float)
        rt   = merged["rt"].to_numpy(dtype=float)

        keep = (bad == 0) & np.isfinite(sv) & np.isfinite(pain) & np.isfinite(rt)
        if keep.sum() < 5:
            print(f"  {pa}: <5 good trials after filtering, skipping.")
            continue

        epo_good  = epo_match[keep]
        sv_good   = sv[keep]
        pain_good = pain[keep]
        rt_good   = rt[keep]

        # variance checks
        if np.nanstd(sv_good) == 0 or np.nanstd(pain_good) == 0 or np.nanstd(rt_good) == 0:
            print(f"  {pa}: zero variance in sv/pain/rt after filtering, skipping.")
            continue

        # compute single-trial TFR
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
        tfr.crop(tmin=tmin_crop, tmax=tmax_crop)

        data = tfr.data  # (n_trials, n_chan, n_freq, n_time)
        n_trials, n_chan, n_freq, n_time = data.shape
        print(f"  TFR shape: {data.shape}")

        # store grids & info from first valid subject
        if freqs_out is None:
            freqs_out = tfr.freqs.copy()
            times_out = tfr.times.copy()
            ch_names_out = np.array(tfr.ch_names, dtype=object)
            info_out = tfr.info.copy()
        else:
            if not np.array_equal(freqs_out, tfr.freqs):
                raise ValueError(f"{pa}: freq grid mismatch")
            if not np.array_equal(times_out, tfr.times):
                raise ValueError(f"{pa}: time grid mismatch")
            if list(ch_names_out) != tfr.ch_names:
                raise ValueError(f"{pa}: channel list mismatch")

        # z-score predictors within subject
        sv_z   = stats.zscore(sv_good.astype(float))
        pain_z = stats.zscore(pain_good.astype(float))
        rt_z   = stats.zscore(rt_good.astype(float))

        # design matrix: [sv, pain, rt, intercept]
        X = np.column_stack([sv_z, pain_z, rt_z, np.ones_like(sv_z)])
        if np.linalg.matrix_rank(X) < X.shape[1]:
            print(f"  {pa}: design matrix not full rank (collinearity), skipping.")
            continue

        # fit regression at each (chan, freq, time)
        betas_sv_sub   = np.zeros((n_chan, n_freq, n_time), dtype=float)
        betas_pain_sub = np.zeros((n_chan, n_freq, n_time), dtype=float)

        for ci in range(n_chan):
            Pw_flat = data[:, ci, :, :].reshape(n_trials, -1)   # (trials, f*t)
            B, _, _, _ = np.linalg.lstsq(X, Pw_flat, rcond=None)  # (4, f*t)
            betas_sv_sub[ci]   = B[0, :].reshape(n_freq, n_time)  # sv | pain+rt
            betas_pain_sub[ci] = B[1, :].reshape(n_freq, n_time)  # pain | sv+rt

        all_betas_sv.append(betas_sv_sub)
        all_betas_pain.append(betas_pain_sub)
        used_subs.append(pa)
        print(f"  {pa}: saved betas (sv|pain+rt) and (pain|sv+rt), n_good={n_trials}")

    # ---------------------------------------------------------------------
    # 2) save subject-level beta arrays (like you do in other versions)
    # ---------------------------------------------------------------------
    if len(all_betas_sv) == 0:
        print("No subjects with valid beta maps; nothing saved; no stats run.")
    else:
        all_betas_sv   = np.stack(all_betas_sv)     # (subj, ch, f, t)
        all_betas_pain = np.stack(all_betas_pain)

        np.save(group_dir / "tfr_beta_sv_pain_para_subxchxfxt.npy", all_betas_sv)
        np.save(group_dir / "tfr_beta_painlevel_subxchxfxt.npy", all_betas_pain)

        np.save(group_dir / "tfr_beta_subjects.npy", np.array(used_subs, dtype=object))
        np.save(group_dir / "tfr_beta_freqs.npy", freqs_out)
        np.save(group_dir / "tfr_beta_times.npy", times_out)
        np.save(group_dir / "tfr_beta_ch_names.npy", ch_names_out)

        print("\nSaved Version 10 beta maps to:", group_dir)
        print("  sv betas shape:", all_betas_sv.shape)
        print("  pain betas shape:", all_betas_pain.shape)
        print("  subjects:", used_subs)

        # -----------------------------------------------------------------
        # 3) v1–3 style GROUP STATS (cluster test) on the beta maps
        # -----------------------------------------------------------------
        stats_dir = group_dir / "Zscoring"
        stats_dir.mkdir(parents=True, exist_ok=True)

        n_subj, n_chan, n_freq, n_time = all_betas_sv.shape

        # channel adjacency (same idea as v1–3)
        connect_ch, _ = mne.channels.find_ch_adjacency(info_out, ch_type="eeg")

        adjacency = combine_adjacency(connect_ch, n_freq, n_time)

        # cluster-forming threshold (same logic as v1–3)
        if not isinstance(param['cluster_threshold'], dict):
            p_thresh = param['cluster_threshold'] / 2
            cluster_threshold = -stats.t.ppf(p_thresh, n_subj - 1)
        else:
            cluster_threshold = param['cluster_threshold']

        def run_tfr_cluster(beta_maps, label):
            """
            beta_maps: (n_subj, n_chan, n_freq, n_time)
            Saves:
              - v10_tval_{label}.npy (chan, freq, time)
              - v10_pval_{label}.npy (chan, freq, time) cluster p-values painted in
            Returns min cluster p (for across-map FDR).
            """
            print(f"\nSecond-level cluster test for map: {label}")

            X = beta_maps.reshape(n_subj, -1)  # (subj, tests)

            t_obs, clusters, cluster_p_values, _ = permutation_cluster_1samp_test(
                X,
                n_permutations=param['nperms'],
                threshold=cluster_threshold,
                adjacency=adjacency,
                tail=0,
                out_type="mask",
                n_jobs=param["njobs"],
                verbose=True
            )

            # paint cluster p-values into full p-map (exactly your v1–3 pattern)
            pvals = np.ones_like(t_obs, dtype=float)
            for clu_mask, p_val in zip(clusters, cluster_p_values):
                pvals[clu_mask] = p_val

            t_map = t_obs.reshape(n_chan, n_freq, n_time)
            p_map = pvals.reshape(n_chan, n_freq, n_time)

            np.save(stats_dir / f"v10_tval_{label}.npy", t_map)
            np.save(stats_dir / f"v10_pval_{label}.npy", p_map)

            # for “across regressor/map FDR” (same as your min_cluster_ps logic)
            min_p = cluster_p_values.min() if len(cluster_p_values) > 0 else 1.0
            return min_p

        # run tests for the two maps + optional difference
        labels = []
        min_cluster_ps = []

        min_cluster_ps.append(run_tfr_cluster(all_betas_sv, "sv_cov_pain_rt"))
        labels.append("sv_cov_pain_rt")

        min_cluster_ps.append(run_tfr_cluster(all_betas_pain, "pain_cov_sv_rt"))
        labels.append("pain_cov_sv_rt")

        # optional: difference map (like your pain–interaction difference idea)
        beta_diff = all_betas_sv - all_betas_pain
        min_cluster_ps.append(run_tfr_cluster(beta_diff, "sv_minus_pain"))
        labels.append("sv_minus_pain")

        # FDR across maps (same “across regressors” idea as v1–3)
        min_cluster_ps = np.asarray(min_cluster_ps, dtype=float)
        rej_fdr, p_fdr = fdr_correction(min_cluster_ps, alpha=0.05, method='indep')

        fdr_df = pd.DataFrame({
            "map": labels,
            "min_cluster_p": min_cluster_ps,
            "min_cluster_p_FDR": p_fdr,
            "sig_FDR": rej_fdr
        })
        fdr_df.to_csv(stats_dir / "v10_cluster_FDR_across_maps.csv", index=False)
        print(f"\nFDR summary across maps saved in {stats_dir}")

        print("\nVersion 10 finished: subject-level beta maps + v1–3-style cluster stats saved.")
        
        
elif version == 11:

    z_dir = Path(outpath) / "Zscoring"
    z_dir.mkdir(parents=True, exist_ok=True)

    pvar = "sv_pain_para"
    mvar = "sv_money"
    rt_col = "rt"

    regvars = [pvar, mvar]                       # filenames for epochs
    regnames = ["SV_pain_para", "SV_money"]      

    all_epos = [[] for _ in range(2)]            # [pain_epochs, money_epochs]

    betas = [[] for _ in range(2)]               
    allbetasnp = []                              

    included_subjects = []
    skipped_subjects = []

    for pa in part_1:
        print(f"\n--- v11 ({v11_mode}) Processing {pa} ---")

        df2  = epo_1_filtered_combined[epo_1_filtered_combined["participant_id"] == pa]
        mod2 = part_1_dat[part_1_dat["participant"] == pa].reset_index(drop=True)

        epo = mne.read_epochs(
            opj(basepath, pa, "eeg", "erps", f"{pa}_decision_cues_singletrials-epo.fif"),
            preload=True
        )

        # match trials
        matching = epo.metadata["trialsnum"].isin(df2["trialsnum"])
        epo_filt = epo[matching]

        if epo_filt.info["sfreq"] != param["testresampfreq"]:
            epo_filt = epo_filt.resample(param["testresampfreq"])

        # drop bad trials
        goodtrials = np.where(epo_filt.metadata["badtrial"] == 0)[0]
        epo_filt = epo_filt[goodtrials]


        if len(mod2) < len(goodtrials):
            print(f"Skipping {pa}: mod2 shorter than epochs after cleaning (mod2={len(mod2)}, epo={len(goodtrials)})")
            skipped_subjects.append(pa)
            continue
        mod2 = mod2.iloc[goodtrials].reset_index(drop=True)

        if len(mod2) < 5:
            print(f"Skipping {pa}: too few trials after cleaning")
            skipped_subjects.append(pa)
            continue


        scale = Scaler(scalings="mean")
        epo_z = mne.EpochsArray(scale.fit_transform(epo_filt.get_data()), epo_filt.info)

        vals = mod2[[pvar, mvar, rt_col]].to_numpy(dtype=float)
        keep = np.all(np.isfinite(vals), axis=1)

        if keep.sum() < 5:
            print(f"Skipping {pa}: too few finite trials ({keep.sum()})")
            skipped_subjects.append(pa)
            continue

        mod2k  = mod2.iloc[keep].copy().reset_index(drop=True)
        epo_zk = epo_z[keep]
        epo_keep_for_meta = epo_filt.copy()[keep]   

        # variance checks
        if np.nanstd(mod2k[pvar]) == 0 or np.nanstd(mod2k[mvar]) == 0:
            print(f"Skipping {pa}: zero variance in SV predictors")
            skipped_subjects.append(pa)
            continue
        if np.nanstd(mod2k[rt_col]) == 0:
            print(f"Skipping {pa}: zero variance in RT")
            skipped_subjects.append(pa)
            continue

        mod2k["Intercept"] = 1.0
        mod2k["pain_z"]  = stats.zscore(mod2k[pvar].to_numpy(dtype=float))
        mod2k["money_z"] = stats.zscore(mod2k[mvar].to_numpy(dtype=float))
        mod2k["RT_z"]    = stats.zscore(mod2k[rt_col].to_numpy(dtype=float))

        md = epo_keep_for_meta.metadata.reset_index(drop=True).copy()
        md[pvar] = mod2k[pvar].values
        md[mvar] = mod2k[mvar].values
        md[rt_col] = mod2k[rt_col].values
        epo_keep_for_meta.metadata = md

        all_epos[0].append(epo_keep_for_meta.copy())  # pain epochs
        all_epos[1].append(epo_keep_for_meta.copy())  # money epochs


        if v11_mode == "separate":
            design_p = mod2k[["Intercept", "pain_z", "RT_z"]]
            res_p = mne.stats.linear_regression(
                epo_zk, design_p, names=["Intercept", "pain_z", "RT_z"]
            )
            beta_pain = res_p["pain_z"].beta
            design_m = mod2k[["Intercept", "money_z", "RT_z"]]
            res_m = mne.stats.linear_regression(
                epo_zk, design_m, names=["Intercept", "money_z", "RT_z"]
            )
            beta_money = res_m["money_z"].beta

        elif v11_mode == "joint":
            design = mod2k[["Intercept", "pain_z", "money_z", "RT_z"]]
            res = mne.stats.linear_regression(
                epo_zk, design, names=["Intercept", "pain_z", "money_z", "RT_z"]
            )
            beta_pain  = res["pain_z"].beta
            beta_money = res["money_z"].beta

        else:
            raise ValueError("v11_mode must be 'separate' or 'joint'")

        betas[0].append(beta_pain)
        betas[1].append(beta_money)
        allbetasnp.append(np.stack([beta_pain.data, beta_money.data]))  # (2, chan, time)
        included_subjects.append(pa)
        print(f"Included {pa}")


    if len(allbetasnp) == 0:
        raise RuntimeError("v11: no subjects included — check filtering/matching.")

    allbetas = np.stack(allbetasnp)
    np.save(z_dir / "ols_2ndlevel_betas.npy", allbetas)
    np.save(z_dir / "included_subjects.npy", np.array(included_subjects, dtype=object))

    print(f"\nIncluded ({len(included_subjects)}): {included_subjects}")
    print(f"Skipped  ({len(skipped_subjects)}): {skipped_subjects}")

    beta_gavg = [
        mne.grand_average(betas[0]),
        mne.grand_average(betas[1]),
    ]
    np.save(z_dir / "ols_2ndlevel_betasavg.npy", np.array(beta_gavg, dtype=object), allow_pickle=True)

    for ridx, regvar in enumerate(regvars):
        epo_save = mne.concatenate_epochs(all_epos[ridx])
        epo_save.save(z_dir / f"ols_2ndlevel_allepochs-epo_{regvar}.fif", overwrite=True)

    connect, _ = mne.channels.find_ch_adjacency(beta_gavg[0].info, ch_type="eeg")

    p_thresh = param["cluster_threshold"] / 2
    cluster_threshold = -stats.t.ppf(p_thresh, allbetas.shape[0] - 1)

    tvals_list = []
    pvals_list = []

    for idx, name in enumerate(regnames):
        data_reg = allbetas[:, idx, :, :]          # subj x chan x time
        testdata = np.swapaxes(data_reg, 2, 1)     # subj x time x chan

        tval, clusters, cluster_p_values, _ = st_clust_1s_ttest(
            testdata,
            threshold=cluster_threshold,
            adjacency=connect,
            n_permutations=param["nperms"],
            n_jobs=param["njobs"],
            buffer_size=None
        )

        pmap = np.ones_like(tval)
        for c, p_val in zip(clusters, cluster_p_values):
            pmap[c] = p_val

        np.save(z_dir / f"ols_2ndlevel_tval_{name}.npy", tval)
        np.save(z_dir / f"ols_2ndlevel_pval_{name}.npy", pmap)

        tvals_list.append(tval)
        pvals_list.append(pmap)

    tvals = np.stack(tvals_list)   # (2, n_times, n_chans)
    pvals = np.stack(pvals_list)   # (2, n_times, n_chans)
    np.save(z_dir / "ols_2ndlevel_tvals.npy", tvals)
    np.save(z_dir / "ols_2ndlevel_pvals.npy", pvals)

    beta_gavg[0].save(z_dir / "beta_gavg_SV_pain_para-ave.fif", overwrite=True)
    beta_gavg[1].save(z_dir / "beta_gavg_SV_money-ave.fif", overwrite=True)
    
    
elif version == 12:

    z_dir = Path(outpath) / "Zscoring"
    z_dir.mkdir(parents=True, exist_ok=True)

    pvar = "sv_pain_para"
    mvar = "sv_money"
    rt_col = "rt"

    regvars = [pvar, mvar]                       # filenames for epochs
    regnames = ["SV_pain_para", "SV_money"]      

    all_epos = [[] for _ in range(2)]            # [pain_epochs, money_epochs]

    betas = [[] for _ in range(2)]               
    allbetasnp = []                              

    included_subjects = []
    skipped_subjects = []

    for pa in part_1:
        print(f"\n--- v11 ({v11_mode}) Processing {pa} ---")

        df2  = epo_1_filtered_combined[epo_1_filtered_combined["participant_id"] == pa]
        mod2 = part_1_dat[part_1_dat["participant"] == pa].reset_index(drop=True)

        epo = mne.read_epochs(
            opj(basepath, pa, "eeg", "erps", f"{pa}_decision_cues_singletrials-epo.fif"),
            preload=True
        )

        # match trials
        matching = epo.metadata["trialsnum"].isin(df2["trialsnum"])
        epo_filt = epo[matching]

        if epo_filt.info["sfreq"] != param["testresampfreq"]:
            epo_filt = epo_filt.resample(param["testresampfreq"])

        # drop bad trials
        goodtrials = np.where(epo_filt.metadata["badtrial"] == 0)[0]
        epo_filt = epo_filt[goodtrials]


        if len(mod2) < len(goodtrials):
            print(f"Skipping {pa}: mod2 shorter than epochs after cleaning (mod2={len(mod2)}, epo={len(goodtrials)})")
            skipped_subjects.append(pa)
            continue
        mod2 = mod2.iloc[goodtrials].reset_index(drop=True)

        if len(mod2) < 5:
            print(f"Skipping {pa}: too few trials after cleaning")
            skipped_subjects.append(pa)
            continue


        scale = Scaler(scalings="mean")
        epo_z = mne.EpochsArray(scale.fit_transform(epo_filt.get_data()), epo_filt.info)

        vals = mod2[[pvar, mvar, rt_col]].to_numpy(dtype=float)
        keep = np.all(np.isfinite(vals), axis=1)

        if keep.sum() < 5:
            print(f"Skipping {pa}: too few finite trials ({keep.sum()})")
            skipped_subjects.append(pa)
            continue

        mod2k  = mod2.iloc[keep].copy().reset_index(drop=True)
        epo_zk = epo_z[keep]
        epo_keep_for_meta = epo_filt.copy()[keep]   

        # variance checks
        if np.nanstd(mod2k[pvar]) == 0 or np.nanstd(mod2k[mvar]) == 0:
            print(f"Skipping {pa}: zero variance in SV predictors")
            skipped_subjects.append(pa)
            continue
        if np.nanstd(mod2k[rt_col]) == 0:
            print(f"Skipping {pa}: zero variance in RT")
            skipped_subjects.append(pa)
            continue

        mod2k["Intercept"] = 1.0
        mod2k["pain_z"]  = stats.zscore(mod2k[pvar].to_numpy(dtype=float))
        mod2k["money_z"] = stats.zscore(mod2k[mvar].to_numpy(dtype=float))
        mod2k["RT_z"]    = stats.zscore(mod2k[rt_col].to_numpy(dtype=float))

        md = epo_keep_for_meta.metadata.reset_index(drop=True).copy()
        md[pvar] = mod2k[pvar].values
        md[mvar] = mod2k[mvar].values
        md[rt_col] = mod2k[rt_col].values
        epo_keep_for_meta.metadata = md

        all_epos[0].append(epo_keep_for_meta.copy())  # pain epochs
        all_epos[1].append(epo_keep_for_meta.copy())  # money epochs


        if v11_mode == "separate":
            design_p = mod2k[["Intercept", "pain_z", "RT_z"]]
            res_p = mne.stats.linear_regression(
                epo_zk, design_p, names=["Intercept", "pain_z", "RT_z"]
            )
            beta_pain = res_p["pain_z"].beta
            design_m = mod2k[["Intercept", "money_z", "RT_z"]]
            res_m = mne.stats.linear_regression(
                epo_zk, design_m, names=["Intercept", "money_z", "RT_z"]
            )
            beta_money = res_m["money_z"].beta

        elif v11_mode == "joint":
            design = mod2k[["Intercept", "pain_z", "money_z", "RT_z"]]
            res = mne.stats.linear_regression(
                epo_zk, design, names=["Intercept", "pain_z", "money_z", "RT_z"]
            )
            beta_pain  = res["pain_z"].beta
            beta_money = res["money_z"].beta

        else:
            raise ValueError("v11_mode must be 'separate' or 'joint'")

        betas[0].append(beta_pain)
        betas[1].append(beta_money)
        allbetasnp.append(np.stack([beta_pain.data, beta_money.data]))  # (2, chan, time)
        included_subjects.append(pa)
        print(f"Included {pa}")


    if len(allbetasnp) == 0:
        raise RuntimeError("v12: no subjects included — check filtering/matching.")

    allbetas = np.stack(allbetasnp)
    np.save(z_dir / "ols_2ndlevel_betas.npy", allbetas)
    np.save(z_dir / "included_subjects.npy", np.array(included_subjects, dtype=object))

    print(f"\nIncluded ({len(included_subjects)}): {included_subjects}")
    print(f"Skipped  ({len(skipped_subjects)}): {skipped_subjects}")

    beta_gavg = [
        mne.grand_average(betas[0]),
        mne.grand_average(betas[1]),
    ]
    np.save(z_dir / "ols_2ndlevel_betasavg.npy", np.array(beta_gavg, dtype=object), allow_pickle=True)

    for ridx, regvar in enumerate(regvars):
        epo_save = mne.concatenate_epochs(all_epos[ridx])
        epo_save.save(z_dir / f"ols_2ndlevel_allepochs-epo_{regvar}.fif", overwrite=True)

    connect, _ = mne.channels.find_ch_adjacency(beta_gavg[0].info, ch_type="eeg")

    p_thresh = param["cluster_threshold"] / 2
    cluster_threshold = -stats.t.ppf(p_thresh, allbetas.shape[0] - 1)

    tvals_list = []
    pvals_list = []

    for idx, name in enumerate(regnames):
        data_reg = allbetas[:, idx, :, :]          # subj x chan x time
        testdata = np.swapaxes(data_reg, 2, 1)     # subj x time x chan

        tval, clusters, cluster_p_values, _ = st_clust_1s_ttest(
            testdata,
            threshold=cluster_threshold,
            adjacency=connect,
            n_permutations=param["nperms"],
            n_jobs=param["njobs"],
            buffer_size=None
        )

        pmap = np.ones_like(tval)
        for c, p_val in zip(clusters, cluster_p_values):
            pmap[c] = p_val

        np.save(z_dir / f"ols_2ndlevel_tval_{name}.npy", tval)
        np.save(z_dir / f"ols_2ndlevel_pval_{name}.npy", pmap)

        tvals_list.append(tval)
        pvals_list.append(pmap)

    tvals = np.stack(tvals_list)   # (2, n_times, n_chans)
    pvals = np.stack(pvals_list)   # (2, n_times, n_chans)
    np.save(z_dir / "ols_2ndlevel_tvals.npy", tvals)
    np.save(z_dir / "ols_2ndlevel_pvals.npy", pvals)

    beta_gavg[0].save(z_dir / "beta_gavg_SV_pain_para-ave.fif", overwrite=True)
    beta_gavg[1].save(z_dir / "beta_gavg_SV_money-ave.fif", overwrite=True)
    

elif version == 13:

    z_dir = Path(outpath) / "Zscoring"
    z_dir.mkdir(parents=True, exist_ok=True)

    pvar = "sv_pain_para"
    mvar = "sv_money"
    rt_col = "rt"

    regvars = [pvar, mvar]                       # filenames for epochs
    regnames = ["SV_pain_para", "SV_money"]      

    all_epos = [[] for _ in range(2)]            # [pain_epochs, money_epochs]

    betas = [[] for _ in range(2)]               
    allbetasnp = []                              

    included_subjects = []
    skipped_subjects = []

    for pa in part_1:
        print(f"\n--- v11 ({v11_mode}) Processing {pa} ---")

        df2  = epo_1_filtered_combined[epo_1_filtered_combined["participant_id"] == pa]
        mod2 = part_1_dat[part_1_dat["participant"] == pa].reset_index(drop=True)

        epo = mne.read_epochs(
            opj(basepath, pa, "eeg", "erps", f"{pa}_decision_cues_singletrials-epo.fif"),
            preload=True
        )

        # match trials
        matching = epo.metadata["trialsnum"].isin(df2["trialsnum"])
        epo_filt = epo[matching]

        if epo_filt.info["sfreq"] != param["testresampfreq"]:
            epo_filt = epo_filt.resample(param["testresampfreq"])

        # drop bad trials
        goodtrials = np.where(epo_filt.metadata["badtrial"] == 0)[0]
        epo_filt = epo_filt[goodtrials]


        if len(mod2) < len(goodtrials):
            print(f"Skipping {pa}: mod2 shorter than epochs after cleaning (mod2={len(mod2)}, epo={len(goodtrials)})")
            skipped_subjects.append(pa)
            continue
        mod2 = mod2.iloc[goodtrials].reset_index(drop=True)

        if len(mod2) < 5:
            print(f"Skipping {pa}: too few trials after cleaning")
            skipped_subjects.append(pa)
            continue


        scale = Scaler(scalings="mean")
        epo_z = mne.EpochsArray(scale.fit_transform(epo_filt.get_data()), epo_filt.info)

        vals = mod2[[pvar, mvar, rt_col]].to_numpy(dtype=float)
        keep = np.all(np.isfinite(vals), axis=1)

        if keep.sum() < 5:
            print(f"Skipping {pa}: too few finite trials ({keep.sum()})")
            skipped_subjects.append(pa)
            continue

        mod2k  = mod2.iloc[keep].copy().reset_index(drop=True)
        epo_zk = epo_z[keep]
        epo_keep_for_meta = epo_filt.copy()[keep]   

        # variance checks
        if np.nanstd(mod2k[pvar]) == 0 or np.nanstd(mod2k[mvar]) == 0:
            print(f"Skipping {pa}: zero variance in SV predictors")
            skipped_subjects.append(pa)
            continue
        if np.nanstd(mod2k[rt_col]) == 0:
            print(f"Skipping {pa}: zero variance in RT")
            skipped_subjects.append(pa)
            continue

        mod2k["Intercept"] = 1.0
        mod2k["pain_z"]  = stats.zscore(mod2k[pvar].to_numpy(dtype=float))
        mod2k["money_z"] = stats.zscore(mod2k[mvar].to_numpy(dtype=float))
        mod2k["RT_z"]    = stats.zscore(mod2k[rt_col].to_numpy(dtype=float))

        md = epo_keep_for_meta.metadata.reset_index(drop=True).copy()
        md[pvar] = mod2k[pvar].values
        md[mvar] = mod2k[mvar].values
        md[rt_col] = mod2k[rt_col].values
        epo_keep_for_meta.metadata = md

        all_epos[0].append(epo_keep_for_meta.copy())  # pain epochs
        all_epos[1].append(epo_keep_for_meta.copy())  # money epochs


        if v13_mode == "separate":
            design_p = mod2k[["Intercept", "pain_z", "RT_z"]]
            res_p = mne.stats.linear_regression(
                epo_zk, design_p, names=["Intercept", "pain_z", "RT_z"]
            )
            beta_pain = res_p["pain_z"].beta
            design_m = mod2k[["Intercept", "money_z", "RT_z"]]
            res_m = mne.stats.linear_regression(
                epo_zk, design_m, names=["Intercept", "money_z", "RT_z"]
            )
            beta_money = res_m["money_z"].beta

        elif v13_mode == "joint":
            design = mod2k[["Intercept", "pain_z", "money_z", "RT_z"]]
            res = mne.stats.linear_regression(
                epo_zk, design, names=["Intercept", "pain_z", "money_z", "RT_z"]
            )
            beta_pain  = res["pain_z"].beta
            beta_money = res["money_z"].beta

        else:
            raise ValueError("v13_mode must be 'separate' or 'joint'")

        betas[0].append(beta_pain)
        betas[1].append(beta_money)
        allbetasnp.append(np.stack([beta_pain.data, beta_money.data]))  # (2, chan, time)
        included_subjects.append(pa)
        print(f"Included {pa}")


    if len(allbetasnp) == 0:
        raise RuntimeError("v13: no subjects included — check filtering/matching.")

    allbetas = np.stack(allbetasnp)
    np.save(z_dir / "ols_2ndlevel_betas.npy", allbetas)
    np.save(z_dir / "included_subjects.npy", np.array(included_subjects, dtype=object))

    print(f"\nIncluded ({len(included_subjects)}): {included_subjects}")
    print(f"Skipped  ({len(skipped_subjects)}): {skipped_subjects}")

    beta_gavg = [
        mne.grand_average(betas[0]),
        mne.grand_average(betas[1]),
    ]
    np.save(z_dir / "ols_2ndlevel_betasavg.npy", np.array(beta_gavg, dtype=object), allow_pickle=True)

    for ridx, regvar in enumerate(regvars):
        epo_save = mne.concatenate_epochs(all_epos[ridx])
        epo_save.save(z_dir / f"ols_2ndlevel_allepochs-epo_{regvar}.fif", overwrite=True)

    connect, _ = mne.channels.find_ch_adjacency(beta_gavg[0].info, ch_type="eeg")

    p_thresh = param["cluster_threshold"] / 2
    cluster_threshold = -stats.t.ppf(p_thresh, allbetas.shape[0] - 1)

    tvals_list = []
    pvals_list = []

    for idx, name in enumerate(regnames):
        data_reg = allbetas[:, idx, :, :]          # subj x chan x time
        testdata = np.swapaxes(data_reg, 2, 1)     # subj x time x chan

        tval, clusters, cluster_p_values, _ = st_clust_1s_ttest(
            testdata,
            threshold=cluster_threshold,
            adjacency=connect,
            n_permutations=param["nperms"],
            n_jobs=param["njobs"],
            buffer_size=None
        )

        pmap = np.ones_like(tval)
        for c, p_val in zip(clusters, cluster_p_values):
            pmap[c] = p_val

        np.save(z_dir / f"ols_2ndlevel_tval_{name}.npy", tval)
        np.save(z_dir / f"ols_2ndlevel_pval_{name}.npy", pmap)

        tvals_list.append(tval)
        pvals_list.append(pmap)

    tvals = np.stack(tvals_list)   # (2, n_times, n_chans)
    pvals = np.stack(pvals_list)   # (2, n_times, n_chans)
    np.save(z_dir / "ols_2ndlevel_tvals.npy", tvals)
    np.save(z_dir / "ols_2ndlevel_pvals.npy", pvals)

    beta_gavg[0].save(z_dir / "beta_gavg_SV_pain_para-ave.fif", overwrite=True)
    beta_gavg[1].save(z_dir / "beta_gavg_SV_money-ave.fif", overwrite=True)
    

elif version == 14:

    z_dir = Path(outpath) / "Zscoring"
    z_dir.mkdir(parents=True, exist_ok=True)

    pvar = "painlevel"
    mvar = "moneylevel"
    rt_col = "rt"

    regvars = [pvar, mvar]                       # filenames for epochs
    regnames = ["Painlevel", "Moneylevel"]      

    all_epos = [[] for _ in range(2)]            # [pain_epochs, money_epochs]

    betas = [[] for _ in range(2)]               
    allbetasnp = []                              

    included_subjects = []
    skipped_subjects = []

    for pa in part_1:
        print(f"\n--- v14 ({v11_mode}) Processing {pa} ---")

        df2  = epo_1_filtered_combined[epo_1_filtered_combined["participant_id"] == pa]
        mod2 = part_1_dat[part_1_dat["participant"] == pa].reset_index(drop=True)

        epo = mne.read_epochs(
            opj(basepath, pa, "eeg", "erps", f"{pa}_decision_cues_singletrials-epo.fif"),
            preload=True
        )

        # match trials
        matching = epo.metadata["trialsnum"].isin(df2["trialsnum"])
        epo_filt = epo[matching]

        if epo_filt.info["sfreq"] != param["testresampfreq"]:
            epo_filt = epo_filt.resample(param["testresampfreq"])

        # drop bad trials
        goodtrials = np.where(epo_filt.metadata["badtrial"] == 0)[0]
        epo_filt = epo_filt[goodtrials]


        if len(mod2) < len(goodtrials):
            print(f"Skipping {pa}: mod2 shorter than epochs after cleaning (mod2={len(mod2)}, epo={len(goodtrials)})")
            skipped_subjects.append(pa)
            continue
        mod2 = mod2.iloc[goodtrials].reset_index(drop=True)

        if len(mod2) < 5:
            print(f"Skipping {pa}: too few trials after cleaning")
            skipped_subjects.append(pa)
            continue


        scale = Scaler(scalings="mean")
        epo_z = mne.EpochsArray(scale.fit_transform(epo_filt.get_data()), epo_filt.info)

        vals = mod2[[pvar, mvar, rt_col]].to_numpy(dtype=float)
        keep = np.all(np.isfinite(vals), axis=1)

        if keep.sum() < 5:
            print(f"Skipping {pa}: too few finite trials ({keep.sum()})")
            skipped_subjects.append(pa)
            continue

        mod2k  = mod2.iloc[keep].copy().reset_index(drop=True)
        epo_zk = epo_z[keep]
        epo_keep_for_meta = epo_filt.copy()[keep]   

        # variance checks
        if np.nanstd(mod2k[pvar]) == 0 or np.nanstd(mod2k[mvar]) == 0:
            print(f"Skipping {pa}: zero variance in SV predictors")
            skipped_subjects.append(pa)
            continue
        if np.nanstd(mod2k[rt_col]) == 0:
            print(f"Skipping {pa}: zero variance in RT")
            skipped_subjects.append(pa)
            continue

        mod2k["Intercept"] = 1.0
        mod2k["pain_z"]  = stats.zscore(mod2k[pvar].to_numpy(dtype=float))
        mod2k["money_z"] = stats.zscore(mod2k[mvar].to_numpy(dtype=float))
        mod2k["RT_z"]    = stats.zscore(mod2k[rt_col].to_numpy(dtype=float))

        md = epo_keep_for_meta.metadata.reset_index(drop=True).copy()
        md[pvar] = mod2k[pvar].values
        md[mvar] = mod2k[mvar].values
        md[rt_col] = mod2k[rt_col].values
        epo_keep_for_meta.metadata = md

        all_epos[0].append(epo_keep_for_meta.copy())  # pain epochs
        all_epos[1].append(epo_keep_for_meta.copy())  # money epochs


        if v13_mode == "separate":
            design_p = mod2k[["Intercept", "pain_z", "RT_z"]]
            res_p = mne.stats.linear_regression(
                epo_zk, design_p, names=["Intercept", "pain_z", "RT_z"]
            )
            beta_pain = res_p["pain_z"].beta
            design_m = mod2k[["Intercept", "money_z", "RT_z"]]
            res_m = mne.stats.linear_regression(
                epo_zk, design_m, names=["Intercept", "money_z", "RT_z"]
            )
            beta_money = res_m["money_z"].beta

        elif v13_mode == "joint":
            design = mod2k[["Intercept", "pain_z", "money_z", "RT_z"]]
            res = mne.stats.linear_regression(
                epo_zk, design, names=["Intercept", "pain_z", "money_z", "RT_z"]
            )
            beta_pain  = res["pain_z"].beta
            beta_money = res["money_z"].beta

        else:
            raise ValueError("v14_mode must be 'separate' or 'joint'")

        betas[0].append(beta_pain)
        betas[1].append(beta_money)
        allbetasnp.append(np.stack([beta_pain.data, beta_money.data]))  # (2, chan, time)
        included_subjects.append(pa)
        print(f"Included {pa}")


    if len(allbetasnp) == 0:
        raise RuntimeError("v14: no subjects included — check filtering/matching.")

    allbetas = np.stack(allbetasnp)
    np.save(z_dir / "ols_2ndlevel_betas.npy", allbetas)
    np.save(z_dir / "included_subjects.npy", np.array(included_subjects, dtype=object))

    print(f"\nIncluded ({len(included_subjects)}): {included_subjects}")
    print(f"Skipped  ({len(skipped_subjects)}): {skipped_subjects}")

    beta_gavg = [
        mne.grand_average(betas[0]),
        mne.grand_average(betas[1]),
    ]
    np.save(z_dir / "ols_2ndlevel_betasavg.npy", np.array(beta_gavg, dtype=object), allow_pickle=True)

    for ridx, regvar in enumerate(regvars):
        epo_save = mne.concatenate_epochs(all_epos[ridx])
        epo_save.save(z_dir / f"ols_2ndlevel_allepochs-epo_{regvar}.fif", overwrite=True)

    connect, _ = mne.channels.find_ch_adjacency(beta_gavg[0].info, ch_type="eeg")

    p_thresh = param["cluster_threshold"] / 2
    cluster_threshold = -stats.t.ppf(p_thresh, allbetas.shape[0] - 1)

    tvals_list = []
    pvals_list = []

    for idx, name in enumerate(regnames):
        data_reg = allbetas[:, idx, :, :]          # subj x chan x time
        testdata = np.swapaxes(data_reg, 2, 1)     # subj x time x chan

        tval, clusters, cluster_p_values, _ = st_clust_1s_ttest(
            testdata,
            threshold=cluster_threshold,
            adjacency=connect,
            n_permutations=param["nperms"],
            n_jobs=param["njobs"],
            buffer_size=None
        )

        pmap = np.ones_like(tval)
        for c, p_val in zip(clusters, cluster_p_values):
            pmap[c] = p_val

        np.save(z_dir / f"ols_2ndlevel_tval_{name}.npy", tval)
        np.save(z_dir / f"ols_2ndlevel_pval_{name}.npy", pmap)

        tvals_list.append(tval)
        pvals_list.append(pmap)

    tvals = np.stack(tvals_list)   # (2, n_times, n_chans)
    pvals = np.stack(pvals_list)   # (2, n_times, n_chans)
    np.save(z_dir / "ols_2ndlevel_tvals.npy", tvals)
    np.save(z_dir / "ols_2ndlevel_pvals.npy", pvals)

    beta_gavg[0].save(z_dir / "beta_gavg_SV_pain_para-ave.fif", overwrite=True)
    beta_gavg[1].save(z_dir / "beta_gavg_SV_money-ave.fif", overwrite=True)
    
    
    
    
elif version == 15:

    z_dir = Path(outpath) / "Zscoring"
    z_dir.mkdir(parents=True, exist_ok=True)

    pvar = "sv_pain_para"
    mvar = "sv_money"
    rt_col = "rt"

    regvars = [pvar, mvar]                       # filenames for epochs
    regnames = ["SV_pain_para", "SV_money"]      

    all_epos = [[] for _ in range(2)]            # [pain_epochs, money_epochs]

    betas = [[] for _ in range(2)]               
    allbetasnp = []                              

    included_subjects = []
    skipped_subjects = []

    for pa in part_1:
        print(f"\n--- v15 ({v15_mode}) Processing {pa} ---")

        df2  = epo_1_filtered_combined[epo_1_filtered_combined["participant_id"] == pa]
        mod2 = part_1_dat[part_1_dat["participant"] == pa].reset_index(drop=True)

        epo = mne.read_epochs(
            opj(basepath, pa, "eeg", "erps", f"{pa}_decision_cues_singletrials-epo.fif"),
            preload=True
        )

        # match trials
        matching = epo.metadata["trialsnum"].isin(df2["trialsnum"])
        epo_filt = epo[matching]

        if epo_filt.info["sfreq"] != param["testresampfreq"]:
            epo_filt = epo_filt.resample(param["testresampfreq"])

        # drop bad trials
        goodtrials = np.where(epo_filt.metadata["badtrial"] == 0)[0]
        epo_filt = epo_filt[goodtrials]


        if len(mod2) < len(goodtrials):
            print(f"Skipping {pa}: mod2 shorter than epochs after cleaning (mod2={len(mod2)}, epo={len(goodtrials)})")
            skipped_subjects.append(pa)
            continue
        mod2 = mod2.iloc[goodtrials].reset_index(drop=True)

        if len(mod2) < 5:
            print(f"Skipping {pa}: too few trials after cleaning")
            skipped_subjects.append(pa)
            continue


        scale = Scaler(scalings="mean")
        epo_z = mne.EpochsArray(scale.fit_transform(epo_filt.get_data()), epo_filt.info)

        vals = mod2[[pvar, mvar, rt_col]].to_numpy(dtype=float)
        keep = np.all(np.isfinite(vals), axis=1)

        if keep.sum() < 5:
            print(f"Skipping {pa}: too few finite trials ({keep.sum()})")
            skipped_subjects.append(pa)
            continue

        mod2k  = mod2.iloc[keep].copy().reset_index(drop=True)
        epo_zk = epo_z[keep]
        epo_keep_for_meta = epo_filt.copy()[keep]   

        # variance checks
        if np.nanstd(mod2k[pvar]) == 0 or np.nanstd(mod2k[mvar]) == 0:
            print(f"Skipping {pa}: zero variance in SV predictors")
            skipped_subjects.append(pa)
            continue
        if np.nanstd(mod2k[rt_col]) == 0:
            print(f"Skipping {pa}: zero variance in RT")
            skipped_subjects.append(pa)
            continue

        mod2k["Intercept"] = 1.0
        mod2k["pain_z"]  = stats.zscore(mod2k[pvar].to_numpy(dtype=float))
        mod2k["money_z"] = stats.zscore(mod2k[mvar].to_numpy(dtype=float))
        mod2k["RT_z"]    = stats.zscore(mod2k[rt_col].to_numpy(dtype=float))

        md = epo_keep_for_meta.metadata.reset_index(drop=True).copy()
        md[pvar] = mod2k[pvar].values
        md[mvar] = mod2k[mvar].values
        md[rt_col] = mod2k[rt_col].values
        epo_keep_for_meta.metadata = md

        all_epos[0].append(epo_keep_for_meta.copy())  # pain epochs
        all_epos[1].append(epo_keep_for_meta.copy())  # money epochs


        if v13_mode == "separate":
            design_p = mod2k[["Intercept", "pain_z", "RT_z"]]
            res_p = mne.stats.linear_regression(
                epo_zk, design_p, names=["Intercept", "pain_z", "RT_z"]
            )
            beta_pain = res_p["pain_z"].beta
            design_m = mod2k[["Intercept", "money_z", "RT_z"]]
            res_m = mne.stats.linear_regression(
                epo_zk, design_m, names=["Intercept", "money_z", "RT_z"]
            )
            beta_money = res_m["money_z"].beta

        elif v13_mode == "joint":
            design = mod2k[["Intercept", "pain_z", "money_z", "RT_z"]]
            res = mne.stats.linear_regression(
                epo_zk, design, names=["Intercept", "pain_z", "money_z", "RT_z"]
            )
            beta_pain  = res["pain_z"].beta
            beta_money = res["money_z"].beta

        else:
            raise ValueError("v15_mode must be 'separate' or 'joint'")

        betas[0].append(beta_pain)
        betas[1].append(beta_money)
        allbetasnp.append(np.stack([beta_pain.data, beta_money.data]))  # (2, chan, time)
        included_subjects.append(pa)
        print(f"Included {pa}")


    if len(allbetasnp) == 0:
        raise RuntimeError("v15: no subjects included — check filtering/matching.")

    allbetas = np.stack(allbetasnp)
    np.save(z_dir / "ols_2ndlevel_betas.npy", allbetas)
    np.save(z_dir / "included_subjects.npy", np.array(included_subjects, dtype=object))

    print(f"\nIncluded ({len(included_subjects)}): {included_subjects}")
    print(f"Skipped  ({len(skipped_subjects)}): {skipped_subjects}")

    beta_gavg = [
        mne.grand_average(betas[0]),
        mne.grand_average(betas[1]),
    ]
    np.save(z_dir / "ols_2ndlevel_betasavg.npy", np.array(beta_gavg, dtype=object), allow_pickle=True)

    for ridx, regvar in enumerate(regvars):
        epo_save = mne.concatenate_epochs(all_epos[ridx])
        epo_save.save(z_dir / f"ols_2ndlevel_allepochs-epo_{regvar}.fif", overwrite=True)

    connect, _ = mne.channels.find_ch_adjacency(beta_gavg[0].info, ch_type="eeg")

    p_thresh = param["cluster_threshold"] / 2
    cluster_threshold = -stats.t.ppf(p_thresh, allbetas.shape[0] - 1)

    tvals_list = []
    pvals_list = []

    for idx, name in enumerate(regnames):
        data_reg = allbetas[:, idx, :, :]          # subj x chan x time
        testdata = np.swapaxes(data_reg, 2, 1)     # subj x time x chan

        tval, clusters, cluster_p_values, _ = st_clust_1s_ttest(
            testdata,
            threshold=cluster_threshold,
            adjacency=connect,
            n_permutations=param["nperms"],
            n_jobs=param["njobs"],
            buffer_size=None
        )

        pmap = np.ones_like(tval)
        for c, p_val in zip(clusters, cluster_p_values):
            pmap[c] = p_val

        np.save(z_dir / f"ols_2ndlevel_tval_{name}.npy", tval)
        np.save(z_dir / f"ols_2ndlevel_pval_{name}.npy", pmap)

        tvals_list.append(tval)
        pvals_list.append(pmap)

    tvals = np.stack(tvals_list)   # (2, n_times, n_chans)
    pvals = np.stack(pvals_list)   # (2, n_times, n_chans)
    np.save(z_dir / "ols_2ndlevel_tvals.npy", tvals)
    np.save(z_dir / "ols_2ndlevel_pvals.npy", pvals)

    beta_gavg[0].save(z_dir / "beta_gavg_SV_pain_para-ave.fif", overwrite=True)
    beta_gavg[1].save(z_dir / "beta_gavg_SV_money-ave.fif", overwrite=True)
    
    
    
elif version == 16:

    z_dir = Path(outpath) / "Zscoring"
    z_dir.mkdir(parents=True, exist_ok=True)

    pvar = "painlevel"
    mvar = "moneylevel"
    rt_col = "rt"

    regvars = [pvar, mvar]                       # filenames for epochs
    regnames = ["Painlevel", "Moneylevel"]      

    all_epos = [[] for _ in range(2)]            # [pain_epochs, money_epochs]

    betas = [[] for _ in range(2)]               
    allbetasnp = []                              

    included_subjects = []
    skipped_subjects = []

    for pa in part_1:
        print(f"\n--- v16 ({v11_mode}) Processing {pa} ---")

        df2  = epo_1_filtered_combined[epo_1_filtered_combined["participant_id"] == pa]
        mod2 = part_1_dat[part_1_dat["participant"] == pa].reset_index(drop=True)

        epo = mne.read_epochs(
            opj(basepath, pa, "eeg", "erps", f"{pa}_decision_cues_singletrials-epo.fif"),
            preload=True
        )

        # match trials
        matching = epo.metadata["trialsnum"].isin(df2["trialsnum"])
        epo_filt = epo[matching]

        if epo_filt.info["sfreq"] != param["testresampfreq"]:
            epo_filt = epo_filt.resample(param["testresampfreq"])

        # drop bad trials
        goodtrials = np.where(epo_filt.metadata["badtrial"] == 0)[0]
        epo_filt = epo_filt[goodtrials]


        if len(mod2) < len(goodtrials):
            print(f"Skipping {pa}: mod2 shorter than epochs after cleaning (mod2={len(mod2)}, epo={len(goodtrials)})")
            skipped_subjects.append(pa)
            continue
        mod2 = mod2.iloc[goodtrials].reset_index(drop=True)

        if len(mod2) < 5:
            print(f"Skipping {pa}: too few trials after cleaning")
            skipped_subjects.append(pa)
            continue


        scale = Scaler(scalings="mean")
        epo_z = mne.EpochsArray(scale.fit_transform(epo_filt.get_data()), epo_filt.info)

        vals = mod2[[pvar, mvar, rt_col]].to_numpy(dtype=float)
        keep = np.all(np.isfinite(vals), axis=1)

        if keep.sum() < 5:
            print(f"Skipping {pa}: too few finite trials ({keep.sum()})")
            skipped_subjects.append(pa)
            continue

        mod2k  = mod2.iloc[keep].copy().reset_index(drop=True)
        epo_zk = epo_z[keep]
        epo_keep_for_meta = epo_filt.copy()[keep]   

        # variance checks
        if np.nanstd(mod2k[pvar]) == 0 or np.nanstd(mod2k[mvar]) == 0:
            print(f"Skipping {pa}: zero variance in SV predictors")
            skipped_subjects.append(pa)
            continue
        if np.nanstd(mod2k[rt_col]) == 0:
            print(f"Skipping {pa}: zero variance in RT")
            skipped_subjects.append(pa)
            continue

        mod2k["Intercept"] = 1.0
        mod2k["pain_z"]  = stats.zscore(mod2k[pvar].to_numpy(dtype=float))
        mod2k["money_z"] = stats.zscore(mod2k[mvar].to_numpy(dtype=float))
        mod2k["RT_z"]    = stats.zscore(mod2k[rt_col].to_numpy(dtype=float))

        md = epo_keep_for_meta.metadata.reset_index(drop=True).copy()
        md[pvar] = mod2k[pvar].values
        md[mvar] = mod2k[mvar].values
        md[rt_col] = mod2k[rt_col].values
        epo_keep_for_meta.metadata = md

        all_epos[0].append(epo_keep_for_meta.copy())  # pain epochs
        all_epos[1].append(epo_keep_for_meta.copy())  # money epochs


        if v13_mode == "separate":
            design_p = mod2k[["Intercept", "pain_z", "RT_z"]]
            res_p = mne.stats.linear_regression(
                epo_zk, design_p, names=["Intercept", "pain_z", "RT_z"]
            )
            beta_pain = res_p["pain_z"].beta
            design_m = mod2k[["Intercept", "money_z", "RT_z"]]
            res_m = mne.stats.linear_regression(
                epo_zk, design_m, names=["Intercept", "money_z", "RT_z"]
            )
            beta_money = res_m["money_z"].beta

        elif v13_mode == "joint":
            design = mod2k[["Intercept", "pain_z", "money_z", "RT_z"]]
            res = mne.stats.linear_regression(
                epo_zk, design, names=["Intercept", "pain_z", "money_z", "RT_z"]
            )
            beta_pain  = res["pain_z"].beta
            beta_money = res["money_z"].beta

        else:
            raise ValueError("v14_mode must be 'separate' or 'joint'")

        betas[0].append(beta_pain)
        betas[1].append(beta_money)
        allbetasnp.append(np.stack([beta_pain.data, beta_money.data]))  # (2, chan, time)
        included_subjects.append(pa)
        print(f"Included {pa}")


    if len(allbetasnp) == 0:
        raise RuntimeError("v16: no subjects included — check filtering/matching.")

    allbetas = np.stack(allbetasnp)
    np.save(z_dir / "ols_2ndlevel_betas.npy", allbetas)
    np.save(z_dir / "included_subjects.npy", np.array(included_subjects, dtype=object))

    print(f"\nIncluded ({len(included_subjects)}): {included_subjects}")
    print(f"Skipped  ({len(skipped_subjects)}): {skipped_subjects}")

    beta_gavg = [
        mne.grand_average(betas[0]),
        mne.grand_average(betas[1]),
    ]
    np.save(z_dir / "ols_2ndlevel_betasavg.npy", np.array(beta_gavg, dtype=object), allow_pickle=True)

    for ridx, regvar in enumerate(regvars):
        epo_save = mne.concatenate_epochs(all_epos[ridx])
        epo_save.save(z_dir / f"ols_2ndlevel_allepochs-epo_{regvar}.fif", overwrite=True)

    connect, _ = mne.channels.find_ch_adjacency(beta_gavg[0].info, ch_type="eeg")

    p_thresh = param["cluster_threshold"] / 2
    cluster_threshold = -stats.t.ppf(p_thresh, allbetas.shape[0] - 1)

    tvals_list = []
    pvals_list = []

    for idx, name in enumerate(regnames):
        data_reg = allbetas[:, idx, :, :]          # subj x chan x time
        testdata = np.swapaxes(data_reg, 2, 1)     # subj x time x chan

        tval, clusters, cluster_p_values, _ = st_clust_1s_ttest(
            testdata,
            threshold=cluster_threshold,
            adjacency=connect,
            n_permutations=param["nperms"],
            n_jobs=param["njobs"],
            buffer_size=None
        )

        pmap = np.ones_like(tval)
        for c, p_val in zip(clusters, cluster_p_values):
            pmap[c] = p_val

        np.save(z_dir / f"ols_2ndlevel_tval_{name}.npy", tval)
        np.save(z_dir / f"ols_2ndlevel_pval_{name}.npy", pmap)

        tvals_list.append(tval)
        pvals_list.append(pmap)

    tvals = np.stack(tvals_list)   # (2, n_times, n_chans)
    pvals = np.stack(pvals_list)   # (2, n_times, n_chans)
    np.save(z_dir / "ols_2ndlevel_tvals.npy", tvals)
    np.save(z_dir / "ols_2ndlevel_pvals.npy", pvals)

    beta_gavg[0].save(z_dir / "beta_gavg_SV_pain_para-ave.fif", overwrite=True)
    beta_gavg[1].save(z_dir / "beta_gavg_SV_money-ave.fif", overwrite=True)
    
    
if version == 17:
    z_dir = Path(outpath) / "Zscoring"
    z_dir.mkdir(parents=True, exist_ok=True)

    # regressors you want to analyse for v17
    pvar, mvar, rt_col = "painlevel", "moneylevel", "rt"

    # what you will SAVE second-level stats for (must match betas order)
    if v17_mode == "resid_joint":
        regnames = ["pain_u", "money_u"]
    elif v17_mode == "joint":
        regnames = ["pain_z", "money_z"]
    elif v17_mode == "separate":
        regnames = ["pain_z", "money_z"]
    else:
        raise ValueError("v17_mode must be 'resid_joint', 'joint', or 'separate'")

    all_epos = [[] for _ in range(2)]          # epochs saved for binning plots
    betas = [[] for _ in range(2)]             # Evoked betas per subject (for grand-average)
    allbetasnp = []                            # numpy betas for cluster test
    included_subjects, skipped_subjects = [], []

    def residualize(y, X):
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        return y - X @ beta

    for pa in part_1:
        mod2 = part_1_dat[part_1_dat["participant"] == pa].reset_index(drop=True)

        epo = mne.read_epochs(
            opj(basepath, pa, "eeg", "erps_resp_rp", f"{pa}_decision_resp_rp_singletrials-epo.fif"),
            preload=True
        )

        # align by trialsnum
        keep_epo = epo.metadata["trialsnum"].isin(mod2["trialsnum"])
        epo_filt = epo[keep_epo]
        mod2 = mod2[mod2["trialsnum"].isin(epo_filt.metadata["trialsnum"])].copy()

        # sort both by trialsnum to guarantee row alignment
        epo_order = np.argsort(epo_filt.metadata["trialsnum"].to_numpy())
        epo_filt = epo_filt[epo_order]
        mod2 = mod2.sort_values("trialsnum").reset_index(drop=True)

        # resample
        if epo_filt.info["sfreq"] != param["testresampfreq"]:
            epo_filt = epo_filt.resample(param["testresampfreq"])

        # drop badtrials
        good = np.where(epo_filt.metadata["badtrial"] == 0)[0]
        epo_filt = epo_filt[good]
        mod2 = mod2.iloc[good].reset_index(drop=True)

        # finite mask
        vals = mod2[[pvar, mvar, rt_col]].to_numpy(float)
        ok = np.all(np.isfinite(vals), axis=1)
        if ok.sum() < 5:
            skipped_subjects.append(pa)
            continue

        epo_filt = epo_filt[ok]
        mod2k = mod2.iloc[ok].reset_index(drop=True)

        # keep a copy with metadata for later binning plots
        epo_keep_for_meta = epo_filt.copy()

        # z-score predictors
        pain_z  = stats.zscore(mod2k[pvar].to_numpy(float))
        money_z = stats.zscore(mod2k[mvar].to_numpy(float))
        rt_z    = stats.zscore(mod2k[rt_col].to_numpy(float))

        # build design(s) depending on mode
        if v17_mode == "resid_joint":
            # residualize unique variance controlling RT + the other regressor
            X_money = np.column_stack([np.ones(len(rt_z)), pain_z, rt_z])
            X_pain  = np.column_stack([np.ones(len(rt_z)), money_z, rt_z])
            money_u = stats.zscore(residualize(money_z, X_money))
            pain_u  = stats.zscore(residualize(pain_z,  X_pain))

            design = pd.DataFrame({
                "Intercept": 1.0,
                "pain_u": pain_u,
                "money_u": money_u,
                "RT_z": rt_z,
            })

        # z-score EEG across trials (IMPORTANT: preserve tmin + events)
        scale = Scaler(scalings="mean")
        data_z = scale.fit_transform(epo_filt.get_data())

        epo_z = mne.EpochsArray(
            data_z,
            info=epo_filt.info,
            events=epo_filt.events,
            tmin=epo_filt.tmin,
            event_id=epo_filt.event_id,
            metadata=epo_filt.metadata
        )

        if v17_mode == "resid_joint":
            res = mne.stats.linear_regression(epo_z, design, names=list(design.columns))
            beta_pain  = res["pain_u"].beta
            beta_money = res["money_u"].beta

        elif v17_mode == "joint":
            design = pd.DataFrame({
                "Intercept": 1.0,
                "pain_z": pain_z,
                "money_z": money_z,
                "RT_z": rt_z,
            })
            res = mne.stats.linear_regression(epo_z, design, names=list(design.columns))
            beta_pain  = res["pain_z"].beta
            beta_money = res["money_z"].beta

        elif v17_mode == "separate":
            design_p = pd.DataFrame({"Intercept": 1.0, "pain_z": pain_z, "RT_z": rt_z})
            res_p = mne.stats.linear_regression(epo_z, design_p, names=list(design_p.columns))
            beta_pain = res_p["pain_z"].beta

            design_m = pd.DataFrame({"Intercept": 1.0, "money_z": money_z, "RT_z": rt_z})
            res_m = mne.stats.linear_regression(epo_z, design_m, names=list(design_m.columns))
            beta_money = res_m["money_z"].beta

        # store metadata for binning plots (always bin by original pvar/mvar)
        md = epo_keep_for_meta.metadata.reset_index(drop=True).copy()
        md[pvar] = mod2k[pvar].values
        md[mvar] = mod2k[mvar].values
        md[rt_col] = mod2k[rt_col].values
        epo_keep_for_meta.metadata = md

        all_epos[0].append(epo_keep_for_meta.copy())
        all_epos[1].append(epo_keep_for_meta.copy())

        betas[0].append(beta_pain)
        betas[1].append(beta_money)
        allbetasnp.append(np.stack([beta_pain.data, beta_money.data]))  # (2, n_chan, n_time)
        included_subjects.append(pa)

    if len(allbetasnp) == 0:
        raise RuntimeError("v17: no subjects included after filtering/alignment.")

    allbetas = np.stack(allbetasnp)  # (n_subj, 2, n_chan, n_time)
    np.save(z_dir / "ols_2ndlevel_betas.npy", allbetas)
    np.save(z_dir / "included_subjects.npy", np.array(included_subjects, dtype=object))

    beta_gavg = [mne.grand_average(betas[0]), mne.grand_average(betas[1])]
    np.save(z_dir / "ols_2ndlevel_betasavg.npy", np.array(beta_gavg, dtype=object), allow_pickle=True)

    # save epochs for plotting bins (filenames are pvar/mvar)
    for ridx, regvar in enumerate([pvar, mvar]):
        epo_save = mne.concatenate_epochs(all_epos[ridx])
        epo_save.save(z_dir / f"ols_2ndlevel_allepochs-epo_{regvar}.fif", overwrite=True)

    # cluster test
    connect, _ = mne.channels.find_ch_adjacency(beta_gavg[0].info, ch_type="eeg")
    p_thresh = param["cluster_threshold"] / 2
    cluster_threshold = -stats.t.ppf(p_thresh, allbetas.shape[0] - 1)

    tvals_list, pvals_list = [], []
    for idx, name in enumerate(regnames):
        data_reg = allbetas[:, idx, :, :]      # (n_subj, n_chan, n_time)
        testdata = np.swapaxes(data_reg, 2, 1) # (n_subj, n_time, n_chan)

        tval, clusters, cluster_p_values, _ = st_clust_1s_ttest(
            testdata,
            threshold=cluster_threshold,
            adjacency=connect,
            n_permutations=param["nperms"],
            n_jobs=param["njobs"],
            buffer_size=None
        )

        pmap = np.ones_like(tval)
        for c, p_val in zip(clusters, cluster_p_values):
            pmap[c] = p_val

        tvals_list.append(tval)
        pvals_list.append(pmap)

    np.save(z_dir / "ols_2ndlevel_tvals.npy", np.stack(tvals_list))
    np.save(z_dir / "ols_2ndlevel_pvals.npy", np.stack(pvals_list))
    
    
    
elif version == 18:

    z_dir = Path(outpath) / "Zscoring"
    z_dir.mkdir(parents=True, exist_ok=True)

    # columns in mod_data
    pvar, mvar, rt_col = "sv_pain_para", "sv_money", "rt"

    # sanity checks (fail fast if missing)
    for col in [pvar, mvar, rt_col, "participant", "trialsnum"]:
        if col not in mod_data.columns:
            raise RuntimeError(f"v18: mod_data is missing required column: '{col}'")

    included_subjects, skipped_subjects = [], []
    betas = [[] for _ in range(2)]      # per-subject Evoked betas
    all_epos = [[] for _ in range(2)]   # epochs to save for binning plots later
    allbetasnp = []                     # (n_subj, 2, n_chan, n_time)

    # adjacency + cluster threshold (computed later after we know n_subj)
    # We'll compute adjacency from the first subject's info.

    for pa in part:
        # subject behavioural rows
        mod2 = mod_data[mod_data["participant"] == pa].copy()

        # load LONG cue-locked epochs
        epo = mne.read_epochs(
            opj(basepath, pa, "eeg", "erps_long", f"{pa}_decision_cues_long_singletrials-epo.fif"),
            preload=True
        )

        # --- align by trialsnum (most robust for your setup)
        if "trialsnum" not in epo.metadata.columns:
            raise RuntimeError("v18: epochs metadata missing 'trialsnum'. Did you save it in ERP creation?")

        # keep only trials that exist in mod2
        keep_epo = epo.metadata["trialsnum"].isin(mod2["trialsnum"])
        epo_filt = epo[keep_epo]

        mod2 = mod2[mod2["trialsnum"].isin(epo_filt.metadata["trialsnum"])].copy()

        # sort both by trialsnum to guarantee row alignment
        epo_order = np.argsort(epo_filt.metadata["trialsnum"].to_numpy())
        epo_filt = epo_filt[epo_order]
        mod2 = mod2.sort_values("trialsnum").reset_index(drop=True)

        # resample to test frequency
        if epo_filt.info["sfreq"] != param["testresampfreq"]:
            epo_filt = epo_filt.resample(param["testresampfreq"])

        # drop bad trials (from your ERP script)
        if "badtrial" in epo_filt.metadata.columns:
            good = np.where(epo_filt.metadata["badtrial"].to_numpy(int) == 0)[0]
            epo_filt = epo_filt[good]
            mod2 = mod2.iloc[good].reset_index(drop=True)

        # finite check
        vals = mod2[[pvar, mvar, rt_col]].to_numpy(float)
        ok = np.all(np.isfinite(vals), axis=1)

        if ok.sum() < 10:  # be a bit stricter for cue-long
            skipped_subjects.append(pa)
            continue

        epo_filt = epo_filt[ok]
        mod2k = mod2.iloc[ok].reset_index(drop=True)

        # keep for binning plots later (keep original scale sv_pain_para / sv_money)
        epo_keep_for_meta = epo_filt.copy()
        md = epo_keep_for_meta.metadata.reset_index(drop=True).copy()
        md[pvar] = mod2k[pvar].values
        md[mvar] = mod2k[mvar].values
        md[rt_col] = mod2k[rt_col].values
        epo_keep_for_meta.metadata = md

        # --- z-score predictors
        pain_z  = stats.zscore(mod2k[pvar].to_numpy(float))
        money_z = stats.zscore(mod2k[mvar].to_numpy(float))
        rt_z    = stats.zscore(mod2k[rt_col].to_numpy(float))

        # --- z-score EEG across trials (preserve events + tmin!)
        scale = Scaler(scalings="mean")
        data_z = scale.fit_transform(epo_filt.get_data())

        epo_z = mne.EpochsArray(
            data_z,
            info=epo_filt.info,
            events=epo_filt.events,
            tmin=epo_filt.tmin,
            event_id=epo_filt.event_id,
            metadata=epo_filt.metadata
        )

        # --- run GLM
        if v18_mode == "joint":
            design = pd.DataFrame({
                "Intercept": 1.0,
                "sv_pain_para_z": pain_z,
                "sv_money_z": money_z,
                "RT_z": rt_z,
            })
            res = mne.stats.linear_regression(epo_z, design, names=list(design.columns))
            beta_pain  = res["sv_pain_para_z"].beta
            beta_money = res["sv_money_z"].beta

        elif v18_mode == "separate":
            design_p = pd.DataFrame({"Intercept": 1.0, "sv_pain_para_z": pain_z, "RT_z": rt_z})
            res_p = mne.stats.linear_regression(epo_z, design_p, names=list(design_p.columns))
            beta_pain = res_p["sv_pain_para_z"].beta

            design_m = pd.DataFrame({"Intercept": 1.0, "sv_money_z": money_z, "RT_z": rt_z})
            res_m = mne.stats.linear_regression(epo_z, design_m, names=list(design_m.columns))
            beta_money = res_m["sv_money_z"].beta

        else:
            raise ValueError("v18_mode must be 'joint' or 'separate'")

        # store for second-level
        betas[0].append(beta_pain)
        betas[1].append(beta_money)
        allbetasnp.append(np.stack([beta_pain.data, beta_money.data]))  # (2, n_chan, n_time)

        # store epochs for binning plots later (one copy per regressor)
        all_epos[0].append(epo_keep_for_meta.copy())
        all_epos[1].append(epo_keep_for_meta.copy())

        included_subjects.append(pa)

    if len(allbetasnp) == 0:
        raise RuntimeError("v18: no subjects included after filtering/alignment.")

    allbetas = np.stack(allbetasnp)  # (n_subj, 2, n_chan, n_time)

    # save betas
    np.save(z_dir / "ols_2ndlevel_betas.npy", allbetas)
    np.save(z_dir / "included_subjects.npy", np.array(included_subjects, dtype=object))
    np.save(z_dir / "skipped_subjects.npy", np.array(skipped_subjects, dtype=object))

    # grand-average betas
    beta_gavg = [mne.grand_average(betas[0]), mne.grand_average(betas[1])]
    np.save(z_dir / "ols_2ndlevel_betasavg.npy", np.array(beta_gavg, dtype=object), allow_pickle=True)

    # save epochs for plotting / binning
    epo_save_p = mne.concatenate_epochs(all_epos[0])
    epo_save_m = mne.concatenate_epochs(all_epos[1])
    epo_save_p.save(z_dir / "ols_2ndlevel_allepochs-epo_sv_pain_para.fif", overwrite=True)
    epo_save_m.save(z_dir / "ols_2ndlevel_allepochs-epo_sv_money.fif", overwrite=True)

    # ------------------------------------------------------------
    # second-level spatio-temporal cluster test
    # ------------------------------------------------------------
    connect, _ = mne.channels.find_ch_adjacency(beta_gavg[0].info, ch_type="eeg")

    # two-sided threshold
    p_thresh = param["cluster_threshold"] / 2.0
    cluster_threshold = -stats.t.ppf(p_thresh, df=allbetas.shape[0] - 1)

    regnames = ["sv_pain_para", "sv_money"]
    tvals_list, pvals_list = [], []

    for ridx, rname in enumerate(regnames):
        # allbetas: (n_subj, 2, n_chan, n_time) -> cluster expects (n_subj, n_time, n_chan)
        data_reg = allbetas[:, ridx, :, :]          # (n_subj, n_chan, n_time)
        testdata = np.swapaxes(data_reg, 2, 1)      # (n_subj, n_time, n_chan)

        tval, clusters, cluster_p_values, _ = st_clust_1s_ttest(
            testdata,
            threshold=cluster_threshold,
            adjacency=connect,
            n_permutations=param["nperms"],
            n_jobs=param["njobs"],
            buffer_size=None,
        )

        # build p-map
        pmap = np.ones_like(tval)
        for c, p_val in zip(clusters, cluster_p_values):
            pmap[c] = p_val

        tvals_list.append(tval)
        pvals_list.append(pmap)

    np.save(z_dir / "ols_2ndlevel_tvals.npy", np.stack(tvals_list))
    np.save(z_dir / "ols_2ndlevel_pvals.npy", np.stack(pvals_list))

    print(f"v18 done. Included n={len(included_subjects)}, skipped n={len(skipped_subjects)}.")


elif version == 19:

    z_dir = Path(outpath) / "Zscoring"
    z_dir.mkdir(parents=True, exist_ok=True)

    # columns in mod_data
    pvar, mvar, rt_col = "painlevel", "moneylevel", "rt"

    # sanity checks (fail fast if missing)
    for col in [pvar, mvar, rt_col, "participant", "trialsnum"]:
        if col not in mod_data.columns:
            raise RuntimeError(f"v19: mod_data is missing required column: '{col}'")

    included_subjects, skipped_subjects = [], []
    betas = [[] for _ in range(2)]      # per-subject Evoked betas
    all_epos = [[] for _ in range(2)]   # epochs to save for binning plots later
    allbetasnp = []                     # (n_subj, 2, n_chan, n_time)

    # adjacency + cluster threshold (computed later after we know n_subj)
    # We'll compute adjacency from the first subject's info.

    for pa in part:
        # subject behavioural rows
        mod2 = mod_data[mod_data["participant"] == pa].copy()

        # load LONG cue-locked epochs
        epo = mne.read_epochs(
            opj(basepath, pa, "eeg", "erps_long", f"{pa}_decision_cues_long_singletrials-epo.fif"),
            preload=True
        )

        # --- align by trialsnum (most robust for your setup)
        if "trialsnum" not in epo.metadata.columns:
            raise RuntimeError("v19: epochs metadata missing 'trialsnum'. Did you save it in ERP creation?")

        # keep only trials that exist in mod2
        keep_epo = epo.metadata["trialsnum"].isin(mod2["trialsnum"])
        epo_filt = epo[keep_epo]

        mod2 = mod2[mod2["trialsnum"].isin(epo_filt.metadata["trialsnum"])].copy()

        # sort both by trialsnum to guarantee row alignment
        epo_order = np.argsort(epo_filt.metadata["trialsnum"].to_numpy())
        epo_filt = epo_filt[epo_order]
        mod2 = mod2.sort_values("trialsnum").reset_index(drop=True)

        # resample to test frequency
        if epo_filt.info["sfreq"] != param["testresampfreq"]:
            epo_filt = epo_filt.resample(param["testresampfreq"])

        # drop bad trials (from your ERP script)
        if "badtrial" in epo_filt.metadata.columns:
            good = np.where(epo_filt.metadata["badtrial"].to_numpy(int) == 0)[0]
            epo_filt = epo_filt[good]
            mod2 = mod2.iloc[good].reset_index(drop=True)

        # finite check
        vals = mod2[[pvar, mvar, rt_col]].to_numpy(float)
        ok = np.all(np.isfinite(vals), axis=1)

        if ok.sum() < 10:  # be a bit stricter for cue-long
            skipped_subjects.append(pa)
            continue

        epo_filt = epo_filt[ok]
        mod2k = mod2.iloc[ok].reset_index(drop=True)

        # keep for binning plots later (keep original scale sv_pain_para / sv_money)
        epo_keep_for_meta = epo_filt.copy()
        md = epo_keep_for_meta.metadata.reset_index(drop=True).copy()
        md[pvar] = mod2k[pvar].values
        md[mvar] = mod2k[mvar].values
        md[rt_col] = mod2k[rt_col].values
        epo_keep_for_meta.metadata = md

        # --- z-score predictors
        pain_z  = stats.zscore(mod2k[pvar].to_numpy(float))
        money_z = stats.zscore(mod2k[mvar].to_numpy(float))
        rt_z    = stats.zscore(mod2k[rt_col].to_numpy(float))

        # --- z-score EEG across trials (preserve events + tmin!)
        scale = Scaler(scalings="mean")
        data_z = scale.fit_transform(epo_filt.get_data())

        epo_z = mne.EpochsArray(
            data_z,
            info=epo_filt.info,
            events=epo_filt.events,
            tmin=epo_filt.tmin,
            event_id=epo_filt.event_id,
            metadata=epo_filt.metadata
        )

        # --- run GLM
        if v19_mode == "joint":
            design = pd.DataFrame({
                "Intercept": 1.0,
                "sv_painlevel_z": pain_z,
                "sv_moneylevel_z": money_z,
                "RT_z": rt_z,
            })
            res = mne.stats.linear_regression(epo_z, design, names=list(design.columns))
            beta_pain  = res["sv_painlevel_z"].beta
            beta_money = res["sv_moneylevel_z"].beta

        elif v19_mode == "separate":
            design_p = pd.DataFrame({"Intercept": 1.0, "sv_painlevel_z": pain_z, "RT_z": rt_z})
            res_p = mne.stats.linear_regression(epo_z, design_p, names=list(design_p.columns))
            beta_pain = res_p["sv_painlevel_z"].beta

            design_m = pd.DataFrame({"Intercept": 1.0, "sv_moneylevel_z": money_z, "RT_z": rt_z})
            res_m = mne.stats.linear_regression(epo_z, design_m, names=list(design_m.columns))
            beta_money = res_m["sv_moneylevel_z"].beta

        else:
            raise ValueError("v19_mode must be 'joint' or 'separate'")

        # store for second-level
        betas[0].append(beta_pain)
        betas[1].append(beta_money)
        allbetasnp.append(np.stack([beta_pain.data, beta_money.data]))  # (2, n_chan, n_time)

        # store epochs for binning plots later (one copy per regressor)
        all_epos[0].append(epo_keep_for_meta.copy())
        all_epos[1].append(epo_keep_for_meta.copy())

        included_subjects.append(pa)

    if len(allbetasnp) == 0:
        raise RuntimeError("v19: no subjects included after filtering/alignment.")

    allbetas = np.stack(allbetasnp)  # (n_subj, 2, n_chan, n_time)

    # save betas
    np.save(z_dir / "ols_2ndlevel_betas.npy", allbetas)
    np.save(z_dir / "included_subjects.npy", np.array(included_subjects, dtype=object))
    np.save(z_dir / "skipped_subjects.npy", np.array(skipped_subjects, dtype=object))

    # grand-average betas
    beta_gavg = [mne.grand_average(betas[0]), mne.grand_average(betas[1])]
    np.save(z_dir / "ols_2ndlevel_betasavg.npy", np.array(beta_gavg, dtype=object), allow_pickle=True)

    # save epochs for plotting / binning
    epo_save_p = mne.concatenate_epochs(all_epos[0])
    epo_save_m = mne.concatenate_epochs(all_epos[1])
    epo_save_p.save(z_dir / "ols_2ndlevel_allepochs-epo_sv_painlevel.fif", overwrite=True)
    epo_save_m.save(z_dir / "ols_2ndlevel_allepochs-epo_sv_moneylevel.fif", overwrite=True)

    # ------------------------------------------------------------
    # second-level spatio-temporal cluster test
    # ------------------------------------------------------------
    connect, _ = mne.channels.find_ch_adjacency(beta_gavg[0].info, ch_type="eeg")

    # two-sided threshold
    p_thresh = param["cluster_threshold"] / 2.0
    cluster_threshold = -stats.t.ppf(p_thresh, df=allbetas.shape[0] - 1)

    regnames = ["sv_painlevel", "sv_moneylevel"]
    tvals_list, pvals_list = [], []

    for ridx, rname in enumerate(regnames):
        # allbetas: (n_subj, 2, n_chan, n_time) -> cluster expects (n_subj, n_time, n_chan)
        data_reg = allbetas[:, ridx, :, :]          # (n_subj, n_chan, n_time)
        testdata = np.swapaxes(data_reg, 2, 1)      # (n_subj, n_time, n_chan)

        tval, clusters, cluster_p_values, _ = st_clust_1s_ttest(
            testdata,
            threshold=cluster_threshold,
            adjacency=connect,
            n_permutations=param["nperms"],
            n_jobs=param["njobs"],
            buffer_size=None,
        )

        # build p-map
        pmap = np.ones_like(tval)
        for c, p_val in zip(clusters, cluster_p_values):
            pmap[c] = p_val

        tvals_list.append(tval)
        pvals_list.append(pmap)

    np.save(z_dir / "ols_2ndlevel_tvals.npy", np.stack(tvals_list))
    np.save(z_dir / "ols_2ndlevel_pvals.npy", np.stack(pvals_list))

    print(f"v19 done. Included n={len(included_subjects)}, skipped n={len(skipped_subjects)}.")
    
    
# -------------------------
# v20 ANALYSIS (put this where your v19 block is)
# -------------------------
elif version == 20:

    z_dir = Path(outpath) / "Zscoring"
    z_dir.mkdir(parents=True, exist_ok=True)

    # columns in mod_data (pain/money ON SCREEN)
    pvar, mvar, rt_col = "painlevel", "moneylevel", "rt"

    # sanity checks (fail fast)
    for col in [pvar, mvar, rt_col, "participant", "trialsnum"]:
        if col not in mod_data.columns:
            raise RuntimeError(f"v20: mod_data is missing required column: '{col}'")

    # helpers
    def residualize_vec(y, X):
        """Return residuals of y after least-squares fit on X."""
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        return y - X @ beta

    included_subjects, skipped_subjects = [], []
    # betas per regressor: pain_L, money_L, pain_Q, money_Q
    betas = [[] for _ in range(4)]
    allbetasnp = []

    # store epochs for plotting / binning (pain + money original scales)
    all_epos_pain = []
    all_epos_money = []

    for pa in part:

        # subject behavioural rows
        mod2 = mod_data[mod_data["participant"] == pa].copy()

        # load LONG cue-locked epochs
        epo = mne.read_epochs(
            opj(basepath, pa, "eeg", "erps_long", f"{pa}_decision_cues_long_singletrials-epo.fif"),
            preload=True
        )

        if "trialsnum" not in epo.metadata.columns:
            raise RuntimeError("v20: epochs metadata missing 'trialsnum'. Did you save it in ERP creation?")

        # align by trialsnum
        keep_epo = epo.metadata["trialsnum"].isin(mod2["trialsnum"])
        epo_filt = epo[keep_epo]
        mod2 = mod2[mod2["trialsnum"].isin(epo_filt.metadata["trialsnum"])].copy()

        # sort both by trialsnum
        epo_order = np.argsort(epo_filt.metadata["trialsnum"].to_numpy())
        epo_filt = epo_filt[epo_order]
        mod2 = mod2.sort_values("trialsnum").reset_index(drop=True)

        # resample
        if epo_filt.info["sfreq"] != param["testresampfreq"]:
            epo_filt = epo_filt.resample(param["testresampfreq"])

        # drop bad trials if present
        if "badtrial" in epo_filt.metadata.columns:
            good = np.where(epo_filt.metadata["badtrial"].to_numpy(int) == 0)[0]
            epo_filt = epo_filt[good]
            mod2 = mod2.iloc[good].reset_index(drop=True)

        # finite check
        vals = mod2[[pvar, mvar, rt_col]].to_numpy(float)
        ok = np.all(np.isfinite(vals), axis=1)
        if ok.sum() < 10:
            skipped_subjects.append(pa)
            continue

        epo_filt = epo_filt[ok]
        mod2k = mod2.iloc[ok].reset_index(drop=True)

        # -------------------------
        # build regressors (within-subject)
        # -------------------------
        pain = mod2k[pvar].to_numpy(float)
        money = mod2k[mvar].to_numpy(float)
        rt = mod2k[rt_col].to_numpy(float)

        # center within subject
        pain_c = pain - pain.mean()
        money_c = money - money.mean()

        # linear (directional)
        pain_L = stats.zscore(pain_c)
        money_L = stats.zscore(money_c)

        # quadratic raw (extremeness)
        pain_Q_raw = pain_c ** 2
        money_Q_raw = money_c ** 2

        # orthogonalize quadratic vs intercept+linear (=> true "U-shape independent of linear")
        Xp = np.column_stack([np.ones(len(pain_c)), pain_c])
        Xm = np.column_stack([np.ones(len(money_c)), money_c])
        pain_Q_orth = residualize_vec(pain_Q_raw, Xp)
        money_Q_orth = residualize_vec(money_Q_raw, Xm)

        pain_Q = stats.zscore(pain_Q_orth)
        money_Q = stats.zscore(money_Q_orth)

        rt_z = stats.zscore(rt)

        # -------------------------
        # z-score EEG across trials (preserve events+tmin)
        # -------------------------
        scale = Scaler(scalings="mean")
        data_z = scale.fit_transform(epo_filt.get_data())

        epo_z = mne.EpochsArray(
            data_z,
            info=epo_filt.info,
            events=epo_filt.events,
            tmin=epo_filt.tmin,
            event_id=epo_filt.event_id,
            metadata=epo_filt.metadata
        )

        # -------------------------
        # GLM
        # -------------------------
        design = pd.DataFrame({
            "Intercept": 1.0,
            "pain_L": pain_L,
            "money_L": money_L,
            "pain_Q": pain_Q,
            "money_Q": money_Q,
            "RT_z": rt_z,
        })

        res = mne.stats.linear_regression(epo_z, design, names=list(design.columns))

        beta_pain_L  = res["pain_L"].beta
        beta_money_L = res["money_L"].beta
        beta_pain_Q  = res["pain_Q"].beta
        beta_money_Q = res["money_Q"].beta

        # store per-subject betas
        betas[0].append(beta_pain_L)
        betas[1].append(beta_money_L)
        betas[2].append(beta_pain_Q)
        betas[3].append(beta_money_Q)

        allbetasnp.append(
            np.stack([beta_pain_L.data, beta_money_L.data, beta_pain_Q.data, beta_money_Q.data])
        )  # (4, n_chan, n_time)

        # -------------------------
        # store epochs for binning plots (original scales)
        # (same epochs, but saved twice with different reg column used for qcut in plotting)
        # -------------------------
        epo_keep = epo_filt.copy()
        md = epo_keep.metadata.reset_index(drop=True).copy()
        md["painlevel"] = pain
        md["moneylevel"] = money
        md["rt"] = rt
        epo_keep.metadata = md

        all_epos_pain.append(epo_keep.copy())
        all_epos_money.append(epo_keep.copy())

        included_subjects.append(pa)

    if len(allbetasnp) == 0:
        raise RuntimeError("v20: no subjects included after filtering/alignment.")

    allbetas = np.stack(allbetasnp)  # (n_subj, 4, n_chan, n_time)

    # save betas
    np.save(z_dir / "ols_2ndlevel_betas.npy", allbetas)
    np.save(z_dir / "included_subjects.npy", np.array(included_subjects, dtype=object))
    np.save(z_dir / "skipped_subjects.npy", np.array(skipped_subjects, dtype=object))

    # grand-average betas
    beta_gavg = [
        mne.grand_average(betas[0]),
        mne.grand_average(betas[1]),
        mne.grand_average(betas[2]),
        mne.grand_average(betas[3]),
    ]
    np.save(z_dir / "ols_2ndlevel_betasavg.npy", np.array(beta_gavg, dtype=object), allow_pickle=True)

    # save epochs for plotting / binning
    epo_save_p = mne.concatenate_epochs(all_epos_pain)
    epo_save_m = mne.concatenate_epochs(all_epos_money)
    epo_save_p.save(z_dir / "ols_2ndlevel_allepochs-epo_painlevel.fif", overwrite=True)
    epo_save_m.save(z_dir / "ols_2ndlevel_allepochs-epo_moneylevel.fif", overwrite=True)

    # ------------------------------------------------------------
    # second-level spatio-temporal cluster test (for each beta map)
    # ------------------------------------------------------------
    connect, _ = mne.channels.find_ch_adjacency(beta_gavg[0].info, ch_type="eeg")

    # two-sided threshold
    p_thresh = param["cluster_threshold"] / 2.0
    cluster_threshold = -stats.t.ppf(p_thresh, df=allbetas.shape[0] - 1)

    regnames = ["pain_L", "money_L", "pain_Q", "money_Q"]
    tvals_list, pvals_list = [], []

    for ridx, rname in enumerate(regnames):
        # allbetas: (n_subj, 4, n_chan, n_time) -> cluster expects (n_subj, n_time, n_chan)
        data_reg = allbetas[:, ridx, :, :]     # (n_subj, n_chan, n_time)
        testdata = np.swapaxes(data_reg, 2, 1) # (n_subj, n_time, n_chan)

        tval, clusters, cluster_p_values, _ = st_clust_1s_ttest(
            testdata,
            threshold=cluster_threshold,
            adjacency=connect,
            n_permutations=param["nperms"],
            n_jobs=param["njobs"],
            buffer_size=None,
        )

        # p-map from clusters
        pmap = np.ones_like(tval)
        for c, p_val in zip(clusters, cluster_p_values):
            pmap[c] = p_val

        tvals_list.append(tval)
        pvals_list.append(pmap)

    np.save(z_dir / "ols_2ndlevel_tvals.npy", np.stack(tvals_list))  # (4, n_times, n_chans)
    np.save(z_dir / "ols_2ndlevel_pvals.npy", np.stack(pvals_list))  # (4, n_times, n_chans)

    print(f"v20 done. Included n={len(included_subjects)}, skipped n={len(skipped_subjects)}.")


# =========================

elif version == 21:

    z_dir = Path(outpath) / "Zscoring"
    z_dir.mkdir(parents=True, exist_ok=True)

    pvar, mvar, rt_col = "painlevel", "moneylevel", "rt"

    for col in [pvar, mvar, rt_col, "participant", "trialsnum"]:
        if col not in mod_data.columns:
            raise RuntimeError(f"v21: mod_data missing required column: '{col}'")

    def residualize_vec(y, X):
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        return y - X @ beta

    included_subjects, skipped_subjects = [], []
    # betas: pain, money, salience_unique
    betas = [[] for _ in range(3)]
    allbetasnp = []

    # epochs to save for plotting/binning (pain & money in original units)
    all_epos_pain = []
    all_epos_money = []

    for pa in part:

        mod2 = mod_data[mod_data["participant"] == pa].copy()

        epo = mne.read_epochs(
            opj(basepath, pa, "eeg", "erps_long", f"{pa}_decision_cues_long_singletrials-epo.fif"),
            preload=True
        )

        if "trialsnum" not in epo.metadata.columns:
            raise RuntimeError("v21: epochs metadata missing 'trialsnum'.")

        # align by trialsnum
        keep_epo = epo.metadata["trialsnum"].isin(mod2["trialsnum"])
        epo_filt = epo[keep_epo]
        mod2 = mod2[mod2["trialsnum"].isin(epo_filt.metadata["trialsnum"])].copy()

        # sort both by trialsnum
        epo_order = np.argsort(epo_filt.metadata["trialsnum"].to_numpy())
        epo_filt = epo_filt[epo_order]
        mod2 = mod2.sort_values("trialsnum").reset_index(drop=True)

        # resample
        if epo_filt.info["sfreq"] != param["testresampfreq"]:
            epo_filt = epo_filt.resample(param["testresampfreq"])

        # drop bad trials if present
        if "badtrial" in epo_filt.metadata.columns:
            good = np.where(epo_filt.metadata["badtrial"].to_numpy(int) == 0)[0]
            epo_filt = epo_filt[good]
            mod2 = mod2.iloc[good].reset_index(drop=True)

        # finite check
        vals = mod2[[pvar, mvar, rt_col]].to_numpy(float)
        ok = np.all(np.isfinite(vals), axis=1)

        if ok.sum() < 10:
            skipped_subjects.append(pa)
            continue

        epo_filt = epo_filt[ok]
        mod2k = mod2.iloc[ok].reset_index(drop=True)

        # ---------
        # regressors
        # ---------
        pain = mod2k[pvar].to_numpy(float)
        money = mod2k[mvar].to_numpy(float)
        rt = mod2k[rt_col].to_numpy(float)

        # z-scored pain/money (as in your old approach)
        pain_z = stats.zscore(pain)
        money_z = stats.zscore(money)
        rt_z = stats.zscore(rt)

        # "salience" raw = pain + money
        sal_raw = pain + money

        # make salience identifiable:
        # remove anything linearly explained by pain & money (and intercept)
        X = np.column_stack([np.ones(len(pain)), pain, money])
        sal_u = residualize_vec(sal_raw, X)
        sal_u_z = stats.zscore(sal_u)

        # keep epochs for binning plots (original units)
        epo_keep = epo_filt.copy()
        md = epo_keep.metadata.reset_index(drop=True).copy()
        md["painlevel"] = pain
        md["moneylevel"] = money
        md["salience_raw"] = sal_raw
        md["rt"] = rt
        epo_keep.metadata = md

        all_epos_pain.append(epo_keep.copy())
        all_epos_money.append(epo_keep.copy())

        # z-score EEG across trials
        scale = Scaler(scalings="mean")
        data_z = scale.fit_transform(epo_filt.get_data())

        epo_z = mne.EpochsArray(
            data_z,
            info=epo_filt.info,
            events=epo_filt.events,
            tmin=epo_filt.tmin,
            event_id=epo_filt.event_id,
            metadata=epo_filt.metadata
        )

        # ---------
        # GLM
        # ---------
        design = pd.DataFrame({
            "Intercept": 1.0,
            "pain_z": pain_z,
            "money_z": money_z,
            "salience_u_z": sal_u_z,
            "RT_z": rt_z,
        })

        res = mne.stats.linear_regression(epo_z, design, names=list(design.columns))

        beta_pain = res["pain_z"].beta
        beta_money = res["money_z"].beta
        beta_sal = res["salience_u_z"].beta

        betas[0].append(beta_pain)
        betas[1].append(beta_money)
        betas[2].append(beta_sal)

        allbetasnp.append(np.stack([beta_pain.data, beta_money.data, beta_sal.data]))  # (3, ch, t)
        included_subjects.append(pa)

    if len(allbetasnp) == 0:
        raise RuntimeError("v21: no subjects included after filtering/alignment.")

    allbetas = np.stack(allbetasnp)  # (n_subj, 3, n_chan, n_time)

    # save
    np.save(z_dir / "ols_2ndlevel_betas.npy", allbetas)
    np.save(z_dir / "included_subjects.npy", np.array(included_subjects, dtype=object))
    np.save(z_dir / "skipped_subjects.npy", np.array(skipped_subjects, dtype=object))

    beta_gavg = [
        mne.grand_average(betas[0]),
        mne.grand_average(betas[1]),
        mne.grand_average(betas[2]),
    ]
    np.save(z_dir / "ols_2ndlevel_betasavg.npy", np.array(beta_gavg, dtype=object), allow_pickle=True)

    # epochs for plotting/binning
    epo_save_p = mne.concatenate_epochs(all_epos_pain)
    epo_save_m = mne.concatenate_epochs(all_epos_money)
    epo_save_p.save(z_dir / "ols_2ndlevel_allepochs-epo_painlevel.fif", overwrite=True)
    epo_save_m.save(z_dir / "ols_2ndlevel_allepochs-epo_moneylevel.fif", overwrite=True)

    # ------------------------------------------------------------
    # second-level cluster test (pain, money, salience_u)
    # ------------------------------------------------------------
    connect, _ = mne.channels.find_ch_adjacency(beta_gavg[0].info, ch_type="eeg")

    p_thresh = param["cluster_threshold"] / 2.0
    cluster_threshold = -stats.t.ppf(p_thresh, df=allbetas.shape[0] - 1)

    regnames = ["pain_z", "money_z", "salience_u_z"]
    tvals_list, pvals_list = [], []

    for ridx, rname in enumerate(regnames):
        data_reg = allbetas[:, ridx, :, :]          # (n_subj, n_chan, n_time)
        testdata = np.swapaxes(data_reg, 2, 1)      # (n_subj, n_time, n_chan)

        tval, clusters, cluster_p_values, _ = st_clust_1s_ttest(
            testdata,
            threshold=cluster_threshold,
            adjacency=connect,
            n_permutations=param["nperms"],
            n_jobs=param["njobs"],
            buffer_size=None,
        )

        pmap = np.ones_like(tval)
        for c, p_val in zip(clusters, cluster_p_values):
            pmap[c] = p_val

        tvals_list.append(tval)
        pvals_list.append(pmap)

    np.save(z_dir / "ols_2ndlevel_tvals.npy", np.stack(tvals_list))  # (3, n_times, n_chans)
    np.save(z_dir / "ols_2ndlevel_pvals.npy", np.stack(pvals_list))  # (3, n_times, n_chans)

    print(f"v21 done. Included n={len(included_subjects)}, skipped n={len(skipped_subjects)}.")



elif version == 22:

    z_dir = Path(outpath) / "Zscoring"
    z_dir.mkdir(parents=True, exist_ok=True)

    # columns in mod_data
    pvar, mvar, rt_col = "painlevel", "moneylevel", "rt"

    # filenames for saving epochs (like v13 regvars)
    regvars  = [pvar, mvar]
    regnames = ["Painlevel", "Moneylevel"]

    # sanity checks
    for col in [pvar, mvar, rt_col, "participant", "trialsnum"]:
        if col not in mod_data.columns:
            raise RuntimeError(f"v22: mod_data missing required column: '{col}'")

    included_subjects, skipped_subjects = [], []

    all_epos = [[] for _ in range(2)]   # [pain_epochs, money_epochs]
    betas    = [[] for _ in range(2)]   # [pain_beta_evokeds, money_beta_evokeds]
    allbetasnp = []                    # list of (2, n_chan, n_time)

    for pa in part:
        print(f"\n--- v22 ({v22_mode}) Processing {pa} ---")

        # behavioural rows for subject
        mod2 = mod_data.loc[mod_data["participant"] == pa].copy()
        if len(mod2) == 0:
            print(f"Skipping {pa}: no behavioural rows")
            skipped_subjects.append(pa)
            continue

        # load LONG cue-locked epochs
        epo = mne.read_epochs(
            opj(basepath, pa, "eeg", "erps_long", f"{pa}_decision_cues_long_singletrials-epo.fif"),
            preload=True
        )

        if "trialsnum" not in epo.metadata.columns:
            raise RuntimeError(
                "v22: epochs metadata missing 'trialsnum'. "
                "Did you save it in ERP creation?"
            )

        # ----------------------------
        # robust alignment by trialsnum
        # ----------------------------
        keep_epo = epo.metadata["trialsnum"].isin(mod2["trialsnum"])
        epo_filt = epo[keep_epo]

        mod2 = mod2.loc[mod2["trialsnum"].isin(epo_filt.metadata["trialsnum"])].copy()

        if len(epo_filt) == 0 or len(mod2) == 0:
            print(f"Skipping {pa}: no overlapping trialsnum")
            skipped_subjects.append(pa)
            continue

        # sort both by trialsnum to guarantee alignment
        epo_order = np.argsort(epo_filt.metadata["trialsnum"].to_numpy())
        epo_filt = epo_filt[epo_order]
        mod2 = mod2.sort_values("trialsnum").reset_index(drop=True)

        # resample
        if epo_filt.info["sfreq"] != param["testresampfreq"]:
            epo_filt = epo_filt.resample(param["testresampfreq"])

        # drop bad trials if available
        if "badtrial" in epo_filt.metadata.columns:
            good = np.where(epo_filt.metadata["badtrial"].to_numpy(int) == 0)[0]
            epo_filt = epo_filt[good]
            mod2 = mod2.iloc[good].reset_index(drop=True)

        # finite check
        vals = mod2[[pvar, mvar, rt_col]].to_numpy(float)
        keep = np.all(np.isfinite(vals), axis=1)

        if keep.sum() < 10:
            print(f"Skipping {pa}: too few finite trials ({keep.sum()})")
            skipped_subjects.append(pa)
            continue

        epo_filt = epo_filt[keep]
        mod2k = mod2.iloc[keep].reset_index(drop=True)

        # variance checks (like v13)
        if np.nanstd(mod2k[pvar]) == 0 or np.nanstd(mod2k[mvar]) == 0:
            print(f"Skipping {pa}: zero variance in pain/money")
            skipped_subjects.append(pa)
            continue
        if np.nanstd(mod2k[rt_col]) == 0:
            print(f"Skipping {pa}: zero variance in RT")
            skipped_subjects.append(pa)
            continue

        # ----------------------------
        # keep epochs with metadata for binning plots later
        # ----------------------------
        epo_keep_for_meta = epo_filt.copy()
        md = epo_keep_for_meta.metadata.reset_index(drop=True).copy()
        md[pvar] = mod2k[pvar].values
        md[mvar] = mod2k[mvar].values
        md[rt_col] = mod2k[rt_col].values
        epo_keep_for_meta.metadata = md

        all_epos[0].append(epo_keep_for_meta.copy())  # pain epochs
        all_epos[1].append(epo_keep_for_meta.copy())  # money epochs

        # ----------------------------
        # z-score predictors (like v13)
        # ----------------------------
        mod2k["Intercept"] = 1.0
        mod2k["pain_z"]  = stats.zscore(mod2k[pvar].to_numpy(float))
        mod2k["money_z"] = stats.zscore(mod2k[mvar].to_numpy(float))
        mod2k["RT_z"]    = stats.zscore(mod2k[rt_col].to_numpy(float))

        # ----------------------------
        # z-score EEG across trials (Scaler + EpochsArray preserving tmin/events)
        # ----------------------------
        scale = Scaler(scalings="mean")
        data_z = scale.fit_transform(epo_filt.get_data())

        epo_z = mne.EpochsArray(
            data_z,
            info=epo_filt.info,
            events=epo_filt.events,
            tmin=epo_filt.tmin,
            event_id=epo_filt.event_id,
            metadata=epo_filt.metadata
        )

        # ----------------------------
        # GLM: joint vs separate (v13 style)
        # ----------------------------
        if v22_mode == "separate":
            # pain GLM
            design_p = mod2k[["Intercept", "pain_z", "RT_z"]]
            res_p = mne.stats.linear_regression(
                epo_z, design_p, names=["Intercept", "pain_z", "RT_z"]
            )
            beta_pain = res_p["pain_z"].beta

            # money GLM
            design_m = mod2k[["Intercept", "money_z", "RT_z"]]
            res_m = mne.stats.linear_regression(
                epo_z, design_m, names=["Intercept", "money_z", "RT_z"]
            )
            beta_money = res_m["money_z"].beta

        elif v22_mode == "joint":
            # one joint GLM
            design = mod2k[["Intercept", "pain_z", "money_z", "RT_z"]]
            res = mne.stats.linear_regression(
                epo_z, design, names=["Intercept", "pain_z", "money_z", "RT_z"]
            )
            beta_pain  = res["pain_z"].beta
            beta_money = res["money_z"].beta

        else:
            raise ValueError("v22_mode must be 'separate' or 'joint'")

        # ----------------------------
        # store for second-level
        # ----------------------------
        betas[0].append(beta_pain)
        betas[1].append(beta_money)

        allbetasnp.append(np.stack([beta_pain.data, beta_money.data]))  # (2, n_chan, n_time)

        included_subjects.append(pa)
        print(f"Included {pa} (n_trials={len(epo_filt)})")

    if len(allbetasnp) == 0:
        raise RuntimeError("v22: no subjects included after filtering/alignment.")

    # (n_subj, 2, n_chan, n_time)
    allbetas = np.stack(allbetasnp)

    # save betas + subject lists
    np.save(z_dir / "ols_2ndlevel_betas.npy", allbetas)
    np.save(z_dir / "included_subjects.npy", np.array(included_subjects, dtype=object))
    np.save(z_dir / "skipped_subjects.npy",  np.array(skipped_subjects, dtype=object))

    # grand-average betas (Evoked objects)
    beta_gavg = [mne.grand_average(betas[0]), mne.grand_average(betas[1])]
    np.save(z_dir / "ols_2ndlevel_betasavg.npy", np.array(beta_gavg, dtype=object), allow_pickle=True)

    # save epochs for plotting / binning (one file per regressor like v13)
    epo_save_p = mne.concatenate_epochs(all_epos[0])
    epo_save_m = mne.concatenate_epochs(all_epos[1])
    epo_save_p.save(z_dir / f"ols_2ndlevel_allepochs-epo_{pvar}.fif", overwrite=True)
    epo_save_m.save(z_dir / f"ols_2ndlevel_allepochs-epo_{mvar}.fif", overwrite=True)

    # ------------------------------------------------------------
    # second-level spatio-temporal cluster test (same output as before)
    # ------------------------------------------------------------
    connect, _ = mne.channels.find_ch_adjacency(beta_gavg[0].info, ch_type="eeg")

    # two-sided threshold
    p_thresh = param["cluster_threshold"] / 2.0
    cluster_threshold = -stats.t.ppf(p_thresh, df=allbetas.shape[0] - 1)

    tvals_list, pvals_list = [], []

    # cluster test expects (n_subj, n_time, n_chan)
    for ridx, name in enumerate(regnames):
        data_reg = allbetas[:, ridx, :, :]      # (n_subj, n_chan, n_time)
        testdata = np.swapaxes(data_reg, 2, 1)  # (n_subj, n_time, n_chan)

        tval, clusters, cluster_p_values, _ = st_clust_1s_ttest(
            testdata,
            threshold=cluster_threshold,
            adjacency=connect,
            n_permutations=param["nperms"],
            n_jobs=param["njobs"],
            buffer_size=None
        )

        # build p-map
        pmap = np.ones_like(tval)
        for c, p_val in zip(clusters, cluster_p_values):
            pmap[c] = p_val

        # (optional) per-regressor files like v13
        np.save(z_dir / f"ols_2ndlevel_tval_{name}.npy", tval)
        np.save(z_dir / f"ols_2ndlevel_pval_{name}.npy", pmap)

        tvals_list.append(tval)
        pvals_list.append(pmap)

    np.save(z_dir / "ols_2ndlevel_tvals.npy", np.stack(tvals_list))  # (2, n_time, n_chan)
    np.save(z_dir / "ols_2ndlevel_pvals.npy", np.stack(pvals_list))  # (2, n_time, n_chan)

    print(f"v22 ({v22_mode}) done. Included n={len(included_subjects)}, skipped n={len(skipped_subjects)}.")




elif version == 23:

    z_dir = Path(outpath) / "Zscoring"
    z_dir.mkdir(parents=True, exist_ok=True)

    # columns in mod_data
    pvar, mvar, rt_col = "painlevel", "moneylevel", "rt"

    # filenames for saving epochs (like v13 regvars)
    regvars  = [pvar, mvar]
    regnames = ["Painlevel", "Moneylevel"]

    # sanity checks
    for col in [pvar, mvar, rt_col, "participant", "trialsnum"]:
        if col not in mod_data.columns:
            raise RuntimeError(f"v23: mod_data missing required column: '{col}'")

    included_subjects, skipped_subjects = [], []

    all_epos = [[] for _ in range(2)]   # [pain_epochs, money_epochs]
    betas    = [[] for _ in range(2)]   # [pain_beta_evokeds, money_beta_evokeds]
    allbetasnp = []                    # list of (2, n_chan, n_time)

    for pa in part:
        print(f"\n--- v23 ({v23_mode}) Processing {pa} ---")

        # behavioural rows for subject
        mod2 = mod_data.loc[mod_data["participant"] == pa].copy()
        if len(mod2) == 0:
            print(f"Skipping {pa}: no behavioural rows")
            skipped_subjects.append(pa)
            continue

        # load LONG cue-locked epochs
        epo = mne.read_epochs(
            opj(basepath, pa, "eeg", "erps_long", f"{pa}_decision_cues_long_singletrials-epo.fif"),
            preload=True
        )

        if "trialsnum" not in epo.metadata.columns:
            raise RuntimeError(
                "v23: epochs metadata missing 'trialsnum'. "
                "Did you save it in ERP creation?"
            )

        # ----------------------------
        # robust alignment by trialsnum
        # ----------------------------
        keep_epo = epo.metadata["trialsnum"].isin(mod2["trialsnum"])
        epo_filt = epo[keep_epo]

        mod2 = mod2.loc[mod2["trialsnum"].isin(epo_filt.metadata["trialsnum"])].copy()

        if len(epo_filt) == 0 or len(mod2) == 0:
            print(f"Skipping {pa}: no overlapping trialsnum")
            skipped_subjects.append(pa)
            continue

        # sort both by trialsnum to guarantee alignment
        epo_order = np.argsort(epo_filt.metadata["trialsnum"].to_numpy())
        epo_filt = epo_filt[epo_order]
        mod2 = mod2.sort_values("trialsnum").reset_index(drop=True)

        # resample
        if epo_filt.info["sfreq"] != param["testresampfreq"]:
            epo_filt = epo_filt.resample(param["testresampfreq"])

        # drop bad trials if available
        if "badtrial" in epo_filt.metadata.columns:
            good = np.where(epo_filt.metadata["badtrial"].to_numpy(int) == 0)[0]
            epo_filt = epo_filt[good]
            mod2 = mod2.iloc[good].reset_index(drop=True)

        # finite check
        vals = mod2[[pvar, mvar, rt_col]].to_numpy(float)
        keep = np.all(np.isfinite(vals), axis=1)

        if keep.sum() < 10:
            print(f"Skipping {pa}: too few finite trials ({keep.sum()})")
            skipped_subjects.append(pa)
            continue

        epo_filt = epo_filt[keep]
        mod2k = mod2.iloc[keep].reset_index(drop=True)

        # variance checks (like v13)
        if np.nanstd(mod2k[pvar]) == 0 or np.nanstd(mod2k[mvar]) == 0:
            print(f"Skipping {pa}: zero variance in pain/money")
            skipped_subjects.append(pa)
            continue
        if np.nanstd(mod2k[rt_col]) == 0:
            print(f"Skipping {pa}: zero variance in RT")
            skipped_subjects.append(pa)
            continue

        # ----------------------------
        # keep epochs with metadata for binning plots later
        # ----------------------------
        epo_keep_for_meta = epo_filt.copy()
        md = epo_keep_for_meta.metadata.reset_index(drop=True).copy()
        md[pvar] = mod2k[pvar].values
        md[mvar] = mod2k[mvar].values
        md[rt_col] = mod2k[rt_col].values
        epo_keep_for_meta.metadata = md

        all_epos[0].append(epo_keep_for_meta.copy())  # pain epochs
        all_epos[1].append(epo_keep_for_meta.copy())  # money epochs

        # ----------------------------
        # z-score predictors (like v13)
        # ----------------------------
        mod2k["Intercept"] = 1.0
        mod2k["pain_z"]  = stats.zscore(mod2k[pvar].to_numpy(float))
        mod2k["money_z"] = stats.zscore(mod2k[mvar].to_numpy(float))
        mod2k["RT_z"]    = stats.zscore(mod2k[rt_col].to_numpy(float))

        # ----------------------------
        # z-score EEG across trials (Scaler + EpochsArray preserving tmin/events)
        # ----------------------------
        scale = Scaler(scalings="mean")
        data_z = scale.fit_transform(epo_filt.get_data())

        epo_z = mne.EpochsArray(
            data_z,
            info=epo_filt.info,
            events=epo_filt.events,
            tmin=epo_filt.tmin,
            event_id=epo_filt.event_id,
            metadata=epo_filt.metadata
        )

        # ----------------------------
        # GLM: joint vs separate (v13 style)
        # ----------------------------
        if v23_mode == "separate":
            # pain GLM
            design_p = mod2k[["Intercept", "pain_z", "RT_z"]]
            res_p = mne.stats.linear_regression(
                epo_z, design_p, names=["Intercept", "pain_z", "RT_z"]
            )
            beta_pain = res_p["pain_z"].beta

            # money GLM
            design_m = mod2k[["Intercept", "money_z", "RT_z"]]
            res_m = mne.stats.linear_regression(
                epo_z, design_m, names=["Intercept", "money_z", "RT_z"]
            )
            beta_money = res_m["money_z"].beta

        elif v23_mode == "joint":
            # one joint GLM
            design = mod2k[["Intercept", "pain_z", "money_z", "RT_z"]]
            res = mne.stats.linear_regression(
                epo_z, design, names=["Intercept", "pain_z", "money_z", "RT_z"]
            )
            beta_pain  = res["pain_z"].beta
            beta_money = res["money_z"].beta

        else:
            raise ValueError("v23_mode must be 'separate' or 'joint'")

        # ----------------------------
        # store for second-level
        # ----------------------------
        betas[0].append(beta_pain)
        betas[1].append(beta_money)

        allbetasnp.append(np.stack([beta_pain.data, beta_money.data]))  # (2, n_chan, n_time)

        included_subjects.append(pa)
        print(f"Included {pa} (n_trials={len(epo_filt)})")

    if len(allbetasnp) == 0:
        raise RuntimeError("v23: no subjects included after filtering/alignment.")

    # (n_subj, 2, n_chan, n_time)
    allbetas = np.stack(allbetasnp)

    # save betas + subject lists
    np.save(z_dir / "ols_2ndlevel_betas.npy", allbetas)
    np.save(z_dir / "included_subjects.npy", np.array(included_subjects, dtype=object))
    np.save(z_dir / "skipped_subjects.npy",  np.array(skipped_subjects, dtype=object))

    # grand-average betas (Evoked objects)
    beta_gavg = [mne.grand_average(betas[0]), mne.grand_average(betas[1])]
    np.save(z_dir / "ols_2ndlevel_betasavg.npy", np.array(beta_gavg, dtype=object), allow_pickle=True)

    # save epochs for plotting / binning (one file per regressor like v13)
    epo_save_p = mne.concatenate_epochs(all_epos[0])
    epo_save_m = mne.concatenate_epochs(all_epos[1])
    epo_save_p.save(z_dir / f"ols_2ndlevel_allepochs-epo_{pvar}.fif", overwrite=True)
    epo_save_m.save(z_dir / f"ols_2ndlevel_allepochs-epo_{mvar}.fif", overwrite=True)

    # ------------------------------------------------------------
    # second-level spatio-temporal cluster test (same output as before)
    # ------------------------------------------------------------
    connect, _ = mne.channels.find_ch_adjacency(beta_gavg[0].info, ch_type="eeg")

    # two-sided threshold
    p_thresh = param["cluster_threshold"] / 2.0
    cluster_threshold = -stats.t.ppf(p_thresh, df=allbetas.shape[0] - 1)

    tvals_list, pvals_list = [], []

    # cluster test expects (n_subj, n_time, n_chan)
    for ridx, name in enumerate(regnames):
        data_reg = allbetas[:, ridx, :, :]      # (n_subj, n_chan, n_time)
        testdata = np.swapaxes(data_reg, 2, 1)  # (n_subj, n_time, n_chan)

        tval, clusters, cluster_p_values, _ = st_clust_1s_ttest(
            testdata,
            threshold=cluster_threshold,
            adjacency=connect,
            n_permutations=param["nperms"],
            n_jobs=param["njobs"],
            buffer_size=None
        )

        # build p-map
        pmap = np.ones_like(tval)
        for c, p_val in zip(clusters, cluster_p_values):
            pmap[c] = p_val

        # (optional) per-regressor files like v13
        np.save(z_dir / f"ols_2ndlevel_tval_{name}.npy", tval)
        np.save(z_dir / f"ols_2ndlevel_pval_{name}.npy", pmap)

        tvals_list.append(tval)
        pvals_list.append(pmap)

    np.save(z_dir / "ols_2ndlevel_tvals.npy", np.stack(tvals_list))  # (2, n_time, n_chan)
    np.save(z_dir / "ols_2ndlevel_pvals.npy", np.stack(pvals_list))  # (2, n_time, n_chan)

    print(f"v23 ({v23_mode}) done. Included n={len(included_subjects)}, skipped n={len(skipped_subjects)}.")




elif version == 24:

    z_dir = Path(outpath) / "Zscoring"
    z_dir.mkdir(parents=True, exist_ok=True)

    # behavioural columns
    pvar, rt_col = "sv_pain_para", "rt"

    # filenames / labels
    regvars  = [pvar]
    regnames = ["SV_pain_para"]

    # sanity checks
    for col in [pvar, rt_col, "participant", "trialsnum"]:
        if col not in mod_data.columns:
            raise RuntimeError(f"v24: mod_data missing required column: '{col}'")

    included_subjects, skipped_subjects = [], []
    all_epos = [[] for _ in range(1)]   # [sv_pain_para epochs]
    betas    = [[] for _ in range(1)]   # [sv_pain_para beta evokeds]
    allbetasnp = []                    # list of (1, n_chan, n_time)

    for pa in part:
        print(f"\n--- v24 (single regressor) Processing {pa} ---")

        mod2 = mod_data.loc[mod_data["participant"] == pa].copy()
        if len(mod2) == 0:
            print(f"Skipping {pa}: no behavioural rows")
            skipped_subjects.append(pa)
            continue

        # load LONG cue-locked epochs (same as v23)
        epo = mne.read_epochs(
            opj(basepath, pa, "eeg", "erps_long", f"{pa}_decision_cues_long_singletrials-epo.fif"),
            preload=True
        )

        if "trialsnum" not in epo.metadata.columns:
            raise RuntimeError(
                "v24: epochs metadata missing 'trialsnum'. Did you save it in ERP creation?"
            )

        # ----------------------------
        # align by trialsnum
        # ----------------------------
        keep_epo = epo.metadata["trialsnum"].isin(mod2["trialsnum"])
        epo_filt = epo[keep_epo]
        mod2 = mod2.loc[mod2["trialsnum"].isin(epo_filt.metadata["trialsnum"])].copy()

        if len(epo_filt) == 0 or len(mod2) == 0:
            print(f"Skipping {pa}: no overlapping trialsnum")
            skipped_subjects.append(pa)
            continue

        # sort both by trialsnum
        epo_order = np.argsort(epo_filt.metadata["trialsnum"].to_numpy())
        epo_filt = epo_filt[epo_order]
        mod2 = mod2.sort_values("trialsnum").reset_index(drop=True)

        # resample
        if epo_filt.info["sfreq"] != param["testresampfreq"]:
            epo_filt = epo_filt.resample(param["testresampfreq"])

        # drop bad trials if present
        if "badtrial" in epo_filt.metadata.columns:
            good = np.where(epo_filt.metadata["badtrial"].to_numpy(int) == 0)[0]
            epo_filt = epo_filt[good]
            mod2 = mod2.iloc[good].reset_index(drop=True)

        # finite check
        vals = mod2[[pvar, rt_col]].to_numpy(float)
        keep = np.all(np.isfinite(vals), axis=1)

        if keep.sum() < 10:
            print(f"Skipping {pa}: too few finite trials ({keep.sum()})")
            skipped_subjects.append(pa)
            continue

        epo_filt = epo_filt[keep]
        mod2k = mod2.iloc[keep].reset_index(drop=True)

        # variance checks
        if np.nanstd(mod2k[pvar]) == 0:
            print(f"Skipping {pa}: zero variance in {pvar}")
            skipped_subjects.append(pa)
            continue
        if np.nanstd(mod2k[rt_col]) == 0:
            print(f"Skipping {pa}: zero variance in RT")
            skipped_subjects.append(pa)
            continue

        # ----------------------------
        # keep epochs with metadata for binning plots later
        # ----------------------------
        epo_keep_for_meta = epo_filt.copy()
        md = epo_keep_for_meta.metadata.reset_index(drop=True).copy()
        md[pvar] = mod2k[pvar].values
        md[rt_col] = mod2k[rt_col].values
        epo_keep_for_meta.metadata = md

        all_epos[0].append(epo_keep_for_meta.copy())

        # ----------------------------
        # z-score predictors (like v23)
        # ----------------------------
        mod2k["Intercept"] = 1.0
        mod2k["svpain_z"] = stats.zscore(mod2k[pvar].to_numpy(float))
        mod2k["RT_z"]     = stats.zscore(mod2k[rt_col].to_numpy(float))

        # ----------------------------
        # z-score EEG across trials (Scaler + EpochsArray preserving tmin/events)
        # ----------------------------
        scale = Scaler(scalings="mean")
        data_z = scale.fit_transform(epo_filt.get_data())

        epo_z = mne.EpochsArray(
            data_z,
            info=epo_filt.info,
            events=epo_filt.events,
            tmin=epo_filt.tmin,
            event_id=epo_filt.event_id,
            metadata=epo_filt.metadata
        )

        # ----------------------------
        # single GLM: sv_pain_para + RT
        # ----------------------------
        design = mod2k[["Intercept", "svpain_z", "RT_z"]]
        res = mne.stats.linear_regression(
            epo_z, design, names=["Intercept", "svpain_z", "RT_z"]
        )
        beta_svpain = res["svpain_z"].beta

        # store
        betas[0].append(beta_svpain)
        allbetasnp.append(np.expand_dims(beta_svpain.data, axis=0))  # (1, n_chan, n_time)

        included_subjects.append(pa)
        print(f"Included {pa} (n_trials={len(epo_filt)})")

    if len(allbetasnp) == 0:
        raise RuntimeError("v24: no subjects included after filtering/alignment.")

    # (n_subj, 1, n_chan, n_time)
    allbetas = np.stack(allbetasnp)

    # save betas + subject lists
    np.save(z_dir / "ols_2ndlevel_betas.npy", allbetas)
    np.save(z_dir / "included_subjects.npy", np.array(included_subjects, dtype=object))
    np.save(z_dir / "skipped_subjects.npy",  np.array(skipped_subjects, dtype=object))

    # grand-average beta (Evoked)
    beta_gavg = [mne.grand_average(betas[0])]
    np.save(z_dir / "ols_2ndlevel_betasavg.npy", np.array(beta_gavg, dtype=object), allow_pickle=True)

    # save epochs for plotting / binning
    epo_save = mne.concatenate_epochs(all_epos[0])
    epo_save.save(z_dir / f"ols_2ndlevel_allepochs-epo_{pvar}.fif", overwrite=True)

    # ------------------------------------------------------------
    # second-level spatio-temporal cluster test (single regressor)
    # ------------------------------------------------------------
    connect, _ = mne.channels.find_ch_adjacency(beta_gavg[0].info, ch_type="eeg")

    p_thresh = param["cluster_threshold"] / 2.0
    cluster_threshold = -stats.t.ppf(p_thresh, df=allbetas.shape[0] - 1)

    # cluster test expects (n_subj, n_time, n_chan)
    data_reg = allbetas[:, 0, :, :]         # (n_subj, n_chan, n_time)
    testdata = np.swapaxes(data_reg, 2, 1)  # (n_subj, n_time, n_chan)

    tval, clusters, cluster_p_values, _ = st_clust_1s_ttest(
        testdata,
        threshold=cluster_threshold,
        adjacency=connect,
        n_permutations=param["nperms"],
        n_jobs=param["njobs"],
        buffer_size=None
    )

    pmap = np.ones_like(tval)
    for c, p_val in zip(clusters, cluster_p_values):
        pmap[c] = p_val

    # save per-regressor (like v23)
    np.save(z_dir / "ols_2ndlevel_tval_SV_pain_para.npy", tval)
    np.save(z_dir / "ols_2ndlevel_pval_SV_pain_para.npy", pmap)

    # save stacked (keeps plotting code pattern)
    np.save(z_dir / "ols_2ndlevel_tvals.npy", np.expand_dims(tval, axis=0))  # (1, n_time, n_chan)
    np.save(z_dir / "ols_2ndlevel_pvals.npy", np.expand_dims(pmap, axis=0))  # (1, n_time, n_chan)

    print(f"v24 done. Included n={len(included_subjects)}, skipped n={len(skipped_subjects)}.")



elif version == 25:

    z_dir = Path(outpath) / "Zscoring"
    z_dir.mkdir(parents=True, exist_ok=True)

    # columns in mod_data
    pvar, mvar, rt_col = "painlevel", "moneylevel", "rt"

    # filenames for saving epochs (like v13 regvars)
    regvars  = [pvar, mvar]
    regnames = ["Painlevel", "Moneylevel"]

    # sanity checks
    for col in [pvar, mvar, rt_col, "participant", "trialsnum"]:
        if col not in mod_data.columns:
            raise RuntimeError(f"v25: mod_data missing required column: '{col}'")

    included_subjects, skipped_subjects = [], []

    all_epos = [[] for _ in range(2)]   # [pain_epochs, money_epochs]
    betas    = [[] for _ in range(2)]   # [pain_beta_evokeds, money_beta_evokeds]
    allbetasnp = []                    # list of (2, n_chan, n_time)

    for pa in part:
        print(f"\n--- v25 ({v25_mode}) Processing {pa} ---")

        # behavioural rows for subject
        mod2 = mod_data.loc[mod_data["participant"] == pa].copy()
        if len(mod2) == 0:
            print(f"Skipping {pa}: no behavioural rows")
            skipped_subjects.append(pa)
            continue

        # load LONG cue-locked epochs
        epo = mne.read_epochs(
            opj(basepath, pa, "eeg", "erps_long", f"{pa}_decision_cues_long_singletrials-epo.fif"),
            preload=True
        )

        if "trialsnum" not in epo.metadata.columns:
            raise RuntimeError(
                "v25: epochs metadata missing 'trialsnum'. "
                "Did you save it in ERP creation?"
            )

        # ----------------------------
        # robust alignment by trialsnum
        # ----------------------------
        keep_epo = epo.metadata["trialsnum"].isin(mod2["trialsnum"])
        epo_filt = epo[keep_epo]

        mod2 = mod2.loc[mod2["trialsnum"].isin(epo_filt.metadata["trialsnum"])].copy()

        if len(epo_filt) == 0 or len(mod2) == 0:
            print(f"Skipping {pa}: no overlapping trialsnum")
            skipped_subjects.append(pa)
            continue

        # sort both by trialsnum to guarantee alignment
        epo_order = np.argsort(epo_filt.metadata["trialsnum"].to_numpy())
        epo_filt = epo_filt[epo_order]
        mod2 = mod2.sort_values("trialsnum").reset_index(drop=True)

        # resample
        if epo_filt.info["sfreq"] != param["testresampfreq"]:
            epo_filt = epo_filt.resample(param["testresampfreq"])

        # drop bad trials if available
        if "badtrial" in epo_filt.metadata.columns:
            good = np.where(epo_filt.metadata["badtrial"].to_numpy(int) == 0)[0]
            epo_filt = epo_filt[good]
            mod2 = mod2.iloc[good].reset_index(drop=True)

        # finite check
        vals = mod2[[pvar, mvar, rt_col]].to_numpy(float)
        keep = np.all(np.isfinite(vals), axis=1)

        if keep.sum() < 10:
            print(f"Skipping {pa}: too few finite trials ({keep.sum()})")
            skipped_subjects.append(pa)
            continue

        epo_filt = epo_filt[keep]
        mod2k = mod2.iloc[keep].reset_index(drop=True)

        # variance checks (like v13)
        if np.nanstd(mod2k[pvar]) == 0 or np.nanstd(mod2k[mvar]) == 0:
            print(f"Skipping {pa}: zero variance in pain/money")
            skipped_subjects.append(pa)
            continue
        if np.nanstd(mod2k[rt_col]) == 0:
            print(f"Skipping {pa}: zero variance in RT")
            skipped_subjects.append(pa)
            continue

        # ----------------------------
        # keep epochs with metadata for binning plots later
        # ----------------------------
        epo_keep_for_meta = epo_filt.copy()
        md = epo_keep_for_meta.metadata.reset_index(drop=True).copy()
        md[pvar] = mod2k[pvar].values
        md[mvar] = mod2k[mvar].values
        md[rt_col] = mod2k[rt_col].values
        epo_keep_for_meta.metadata = md

        all_epos[0].append(epo_keep_for_meta.copy())  # pain epochs
        all_epos[1].append(epo_keep_for_meta.copy())  # money epochs

        # ----------------------------
        # z-score predictors (like v13)
        # ----------------------------
        mod2k["Intercept"] = 1.0
        mod2k["pain_z"]  = stats.zscore(mod2k[pvar].to_numpy(float))
        mod2k["money_z"] = stats.zscore(mod2k[mvar].to_numpy(float))
        mod2k["RT_z"]    = stats.zscore(mod2k[rt_col].to_numpy(float))

        # ----------------------------
        # z-score EEG across trials (Scaler + EpochsArray preserving tmin/events)
        # ----------------------------
        scale = Scaler(scalings="mean")
        data_z = scale.fit_transform(epo_filt.get_data())

        epo_z = mne.EpochsArray(
            data_z,
            info=epo_filt.info,
            events=epo_filt.events,
            tmin=epo_filt.tmin,
            event_id=epo_filt.event_id,
            metadata=epo_filt.metadata
        )

        # ----------------------------
        # GLM: joint vs separate (v13 style)
        # ----------------------------
        if v25_mode == "separate":
            # pain GLM
            design_p = mod2k[["Intercept", "pain_z", "RT_z"]]
            res_p = mne.stats.linear_regression(
                epo_z, design_p, names=["Intercept", "pain_z", "RT_z"]
            )
            beta_pain = res_p["pain_z"].beta

            # money GLM
            design_m = mod2k[["Intercept", "money_z", "RT_z"]]
            res_m = mne.stats.linear_regression(
                epo_z, design_m, names=["Intercept", "money_z", "RT_z"]
            )
            beta_money = res_m["money_z"].beta

        elif v25_mode == "joint":
            # one joint GLM
            design = mod2k[["Intercept", "pain_z", "money_z", "RT_z"]]
            res = mne.stats.linear_regression(
                epo_z, design, names=["Intercept", "pain_z", "money_z", "RT_z"]
            )
            beta_pain  = res["pain_z"].beta
            beta_money = res["money_z"].beta

        else:
            raise ValueError("v25_mode must be 'separate' or 'joint'")

        # ----------------------------
        # store for second-level
        # ----------------------------
        betas[0].append(beta_pain)
        betas[1].append(beta_money)

        allbetasnp.append(np.stack([beta_pain.data, beta_money.data]))  # (2, n_chan, n_time)

        included_subjects.append(pa)
        print(f"Included {pa} (n_trials={len(epo_filt)})")

    if len(allbetasnp) == 0:
        raise RuntimeError("v25: no subjects included after filtering/alignment.")

    # (n_subj, 2, n_chan, n_time)
    allbetas = np.stack(allbetasnp)

    # save betas + subject lists
    np.save(z_dir / "ols_2ndlevel_betas.npy", allbetas)
    np.save(z_dir / "included_subjects.npy", np.array(included_subjects, dtype=object))
    np.save(z_dir / "skipped_subjects.npy",  np.array(skipped_subjects, dtype=object))

    # grand-average betas (Evoked objects)
    beta_gavg = [mne.grand_average(betas[0]), mne.grand_average(betas[1])]
    np.save(z_dir / "ols_2ndlevel_betasavg.npy", np.array(beta_gavg, dtype=object), allow_pickle=True)

    # save epochs for plotting / binning (one file per regressor like v13)
    epo_save_p = mne.concatenate_epochs(all_epos[0])
    epo_save_m = mne.concatenate_epochs(all_epos[1])
    epo_save_p.save(z_dir / f"ols_2ndlevel_allepochs-epo_{pvar}.fif", overwrite=True)
    epo_save_m.save(z_dir / f"ols_2ndlevel_allepochs-epo_{mvar}.fif", overwrite=True)

    # ------------------------------------------------------------
    # second-level spatio-temporal cluster test (same output as before)
    # ------------------------------------------------------------
    connect, _ = mne.channels.find_ch_adjacency(beta_gavg[0].info, ch_type="eeg")

    # two-sided threshold
    p_thresh = param["cluster_threshold"] / 2.0
    cluster_threshold = -stats.t.ppf(p_thresh, df=allbetas.shape[0] - 1)

    tvals_list, pvals_list = [], []

    # cluster test expects (n_subj, n_time, n_chan)
    for ridx, name in enumerate(regnames):
        data_reg = allbetas[:, ridx, :, :]      # (n_subj, n_chan, n_time)
        testdata = np.swapaxes(data_reg, 2, 1)  # (n_subj, n_time, n_chan)

        tval, clusters, cluster_p_values, _ = st_clust_1s_ttest(
            testdata,
            threshold=cluster_threshold,
            adjacency=connect,
            n_permutations=param["nperms"],
            n_jobs=param["njobs"],
            buffer_size=None
        )

        # build p-map
        pmap = np.ones_like(tval)
        for c, p_val in zip(clusters, cluster_p_values):
            pmap[c] = p_val

        # (optional) per-regressor files like v13
        np.save(z_dir / f"ols_2ndlevel_tval_{name}.npy", tval)
        np.save(z_dir / f"ols_2ndlevel_pval_{name}.npy", pmap)

        tvals_list.append(tval)
        pvals_list.append(pmap)

    np.save(z_dir / "ols_2ndlevel_tvals.npy", np.stack(tvals_list))  # (2, n_time, n_chan)
    np.save(z_dir / "ols_2ndlevel_pvals.npy", np.stack(pvals_list))  # (2, n_time, n_chan)

    print(f"v25 ({v25_mode}) done. Included n={len(included_subjects)}, skipped n={len(skipped_subjects)}.")



elif version == 26:

    z_dir = Path(outpath) / "Zscoring"
    z_dir.mkdir(parents=True, exist_ok=True)

    # columns in mod_data
    pvar, mvar, rt_col = "sv_pain_para", "sv_money", "rt"

    # filenames for saving epochs (like v13 regvars)
    regvars  = [pvar, mvar]
    regnames = ["SV_pain_para", "SV_money"]

    # sanity checks
    for col in [pvar, mvar, rt_col, "participant", "trialsnum"]:
        if col not in mod_data.columns:
            raise RuntimeError(f"v26: mod_data missing required column: '{col}'")

    included_subjects, skipped_subjects = [], []

    all_epos = [[] for _ in range(2)]   # [pain_epochs, money_epochs]
    betas    = [[] for _ in range(2)]   # [pain_beta_evokeds, money_beta_evokeds]
    allbetasnp = []                    # list of (2, n_chan, n_time)

    for pa in part:
        print(f"\n--- v26 ({v26_mode}) Processing {pa} ---")

        # behavioural rows for subject
        mod2 = mod_data.loc[mod_data["participant"] == pa].copy()
        if len(mod2) == 0:
            print(f"Skipping {pa}: no behavioural rows")
            skipped_subjects.append(pa)
            continue

        # load LONG cue-locked epochs
        epo = mne.read_epochs(
            opj(basepath, pa, "eeg", "erps_long", f"{pa}_decision_cues_long_singletrials-epo.fif"),
            preload=True
        )

        if "trialsnum" not in epo.metadata.columns:
            raise RuntimeError(
                "v26: epochs metadata missing 'trialsnum'. "
                "Did you save it in ERP creation?"
            )

        # ----------------------------
        # robust alignment by trialsnum
        # ----------------------------
        keep_epo = epo.metadata["trialsnum"].isin(mod2["trialsnum"])
        epo_filt = epo[keep_epo]

        mod2 = mod2.loc[mod2["trialsnum"].isin(epo_filt.metadata["trialsnum"])].copy()

        if len(epo_filt) == 0 or len(mod2) == 0:
            print(f"Skipping {pa}: no overlapping trialsnum")
            skipped_subjects.append(pa)
            continue

        # sort both by trialsnum to guarantee alignment
        epo_order = np.argsort(epo_filt.metadata["trialsnum"].to_numpy())
        epo_filt = epo_filt[epo_order]
        mod2 = mod2.sort_values("trialsnum").reset_index(drop=True)

        # resample
        if epo_filt.info["sfreq"] != param["testresampfreq"]:
            epo_filt = epo_filt.resample(param["testresampfreq"])

        # drop bad trials if available
        if "badtrial" in epo_filt.metadata.columns:
            good = np.where(epo_filt.metadata["badtrial"].to_numpy(int) == 0)[0]
            epo_filt = epo_filt[good]
            mod2 = mod2.iloc[good].reset_index(drop=True)

        # finite check
        vals = mod2[[pvar, mvar, rt_col]].to_numpy(float)
        keep = np.all(np.isfinite(vals), axis=1)

        if keep.sum() < 10:
            print(f"Skipping {pa}: too few finite trials ({keep.sum()})")
            skipped_subjects.append(pa)
            continue

        epo_filt = epo_filt[keep]
        mod2k = mod2.iloc[keep].reset_index(drop=True)

        # variance checks (like v13)
        if np.nanstd(mod2k[pvar]) == 0 or np.nanstd(mod2k[mvar]) == 0:
            print(f"Skipping {pa}: zero variance in pain/money")
            skipped_subjects.append(pa)
            continue
        if np.nanstd(mod2k[rt_col]) == 0:
            print(f"Skipping {pa}: zero variance in RT")
            skipped_subjects.append(pa)
            continue

        # ----------------------------
        # keep epochs with metadata for binning plots later
        # ----------------------------
        epo_keep_for_meta = epo_filt.copy()
        md = epo_keep_for_meta.metadata.reset_index(drop=True).copy()
        md[pvar] = mod2k[pvar].values
        md[mvar] = mod2k[mvar].values
        md[rt_col] = mod2k[rt_col].values
        epo_keep_for_meta.metadata = md

        all_epos[0].append(epo_keep_for_meta.copy())  # pain epochs
        all_epos[1].append(epo_keep_for_meta.copy())  # money epochs

        # ----------------------------
        # z-score predictors (like v13)
        # ----------------------------
        mod2k["Intercept"] = 1.0
        mod2k["sv_pain_para_z"]  = stats.zscore(mod2k[pvar].to_numpy(float))
        mod2k["sv_money_z"] = stats.zscore(mod2k[mvar].to_numpy(float))
        mod2k["RT_z"]    = stats.zscore(mod2k[rt_col].to_numpy(float))

        # ----------------------------
        # z-score EEG across trials (Scaler + EpochsArray preserving tmin/events)
        # ----------------------------
        scale = Scaler(scalings="mean")
        data_z = scale.fit_transform(epo_filt.get_data())

        epo_z = mne.EpochsArray(
            data_z,
            info=epo_filt.info,
            events=epo_filt.events,
            tmin=epo_filt.tmin,
            event_id=epo_filt.event_id,
            metadata=epo_filt.metadata
        )

        # ----------------------------
        # GLM: joint vs separate (v13 style)
        # ----------------------------
        if v26_mode == "separate":
            # pain GLM
            design_p = mod2k[["Intercept", "sv_pain_para_z", "RT_z"]]
            res_p = mne.stats.linear_regression(
                epo_z, design_p, names=["Intercept", "sv_pain_para_z", "RT_z"]
            )
            beta_pain = res_p["sv_pain_para_z"].beta

            # money GLM
            design_m = mod2k[["Intercept", "sv_money_z", "RT_z"]]
            res_m = mne.stats.linear_regression(
                epo_z, design_m, names=["Intercept", "sv_money_z", "RT_z"]
            )
            beta_money = res_m["sv_money_z"].beta

        elif v26_mode == "joint":
            # one joint GLM
            design = mod2k[["Intercept", "sv_pain_para_z", "sv_money_z", "RT_z"]]
            res = mne.stats.linear_regression(
                epo_z, design, names=["Intercept", "sv_pain_para_z", "sv_money_z", "RT_z"]
            )
            beta_pain  = res["sv_pain_para_z"].beta
            beta_money = res["sv_money_z"].beta

        else:
            raise ValueError("v26_mode must be 'separate' or 'joint'")

        # ----------------------------
        # store for second-level
        # ----------------------------
        betas[0].append(beta_pain)
        betas[1].append(beta_money)

        allbetasnp.append(np.stack([beta_pain.data, beta_money.data]))  # (2, n_chan, n_time)

        included_subjects.append(pa)
        print(f"Included {pa} (n_trials={len(epo_filt)})")

    if len(allbetasnp) == 0:
        raise RuntimeError("v26: no subjects included after filtering/alignment.")

    # (n_subj, 2, n_chan, n_time)
    allbetas = np.stack(allbetasnp)

    # save betas + subject lists
    np.save(z_dir / "ols_2ndlevel_betas.npy", allbetas)
    np.save(z_dir / "included_subjects.npy", np.array(included_subjects, dtype=object))
    np.save(z_dir / "skipped_subjects.npy",  np.array(skipped_subjects, dtype=object))

    # grand-average betas (Evoked objects)
    beta_gavg = [mne.grand_average(betas[0]), mne.grand_average(betas[1])]
    np.save(z_dir / "ols_2ndlevel_betasavg.npy", np.array(beta_gavg, dtype=object), allow_pickle=True)

    # save epochs for plotting / binning (one file per regressor like v13)
    epo_save_p = mne.concatenate_epochs(all_epos[0])
    epo_save_m = mne.concatenate_epochs(all_epos[1])
    epo_save_p.save(z_dir / f"ols_2ndlevel_allepochs-epo_{pvar}.fif", overwrite=True)
    epo_save_m.save(z_dir / f"ols_2ndlevel_allepochs-epo_{mvar}.fif", overwrite=True)

    # ------------------------------------------------------------
    # second-level spatio-temporal cluster test (same output as before)
    # ------------------------------------------------------------
    connect, _ = mne.channels.find_ch_adjacency(beta_gavg[0].info, ch_type="eeg")

    # two-sided threshold
    p_thresh = param["cluster_threshold"] / 2.0
    cluster_threshold = -stats.t.ppf(p_thresh, df=allbetas.shape[0] - 1)

    tvals_list, pvals_list = [], []

    # cluster test expects (n_subj, n_time, n_chan)
    for ridx, name in enumerate(regnames):
        data_reg = allbetas[:, ridx, :, :]      # (n_subj, n_chan, n_time)
        testdata = np.swapaxes(data_reg, 2, 1)  # (n_subj, n_time, n_chan)

        tval, clusters, cluster_p_values, _ = st_clust_1s_ttest(
            testdata,
            threshold=cluster_threshold,
            adjacency=connect,
            n_permutations=param["nperms"],
            n_jobs=param["njobs"],
            buffer_size=None
        )

        # build p-map
        pmap = np.ones_like(tval)
        for c, p_val in zip(clusters, cluster_p_values):
            pmap[c] = p_val

        # (optional) per-regressor files like v13
        np.save(z_dir / f"ols_2ndlevel_tval_{name}.npy", tval)
        np.save(z_dir / f"ols_2ndlevel_pval_{name}.npy", pmap)

        tvals_list.append(tval)
        pvals_list.append(pmap)

    np.save(z_dir / "ols_2ndlevel_tvals.npy", np.stack(tvals_list))  # (2, n_time, n_chan)
    np.save(z_dir / "ols_2ndlevel_pvals.npy", np.stack(pvals_list))  # (2, n_time, n_chan)

    print(f"v26 ({v26_mode}) done. Included n={len(included_subjects)}, skipped n={len(skipped_subjects)}.")



elif version == 27:

    z_dir = Path(outpath) / "Zscoring"
    z_dir.mkdir(parents=True, exist_ok=True)

    # columns in mod_data
    pvar, mvar, rt_col = "painlevel", "moneylevel", "rt"

    # filenames for saving epochs (like v13 regvars)
    regvars  = [pvar, mvar]
    regnames = ["Painlevel", "Moneylevel"]

    # sanity checks
    for col in [pvar, mvar, rt_col, "participant", "trialsnum"]:
        if col not in mod_data.columns:
            raise RuntimeError(f"v27: mod_data missing required column: '{col}'")

    included_subjects, skipped_subjects = [], []

    all_epos = [[] for _ in range(2)]   # [pain_epochs, money_epochs]
    betas    = [[] for _ in range(2)]   # [pain_beta_evokeds, money_beta_evokeds]
    allbetasnp = []                    # list of (2, n_chan, n_time)

    for pa in part:
        print(f"\n--- v27 ({v27_mode}) Processing {pa} ---")

        # behavioural rows for subject
        mod2 = mod_data.loc[mod_data["participant"] == pa].copy()
        if len(mod2) == 0:
            print(f"Skipping {pa}: no behavioural rows")
            skipped_subjects.append(pa)
            continue

        # load LONG cue-locked epochs
        epo = mne.read_epochs(
            opj(basepath, pa, "eeg", "erps_long", f"{pa}_decision_cues_long_singletrials-epo.fif"),
            preload=True
        )

        if "trialsnum" not in epo.metadata.columns:
            raise RuntimeError(
                "v27: epochs metadata missing 'trialsnum'. "
                "Did you save it in ERP creation?"
            )

        # ----------------------------
        # robust alignment by trialsnum
        # ----------------------------
        keep_epo = epo.metadata["trialsnum"].isin(mod2["trialsnum"])
        epo_filt = epo[keep_epo]

        mod2 = mod2.loc[mod2["trialsnum"].isin(epo_filt.metadata["trialsnum"])].copy()

        if len(epo_filt) == 0 or len(mod2) == 0:
            print(f"Skipping {pa}: no overlapping trialsnum")
            skipped_subjects.append(pa)
            continue

        # sort both by trialsnum to guarantee alignment
        epo_order = np.argsort(epo_filt.metadata["trialsnum"].to_numpy())
        epo_filt = epo_filt[epo_order]
        mod2 = mod2.sort_values("trialsnum").reset_index(drop=True)

        # resample
        if epo_filt.info["sfreq"] != param["testresampfreq"]:
            epo_filt = epo_filt.resample(param["testresampfreq"])

        # drop bad trials if available
        if "badtrial" in epo_filt.metadata.columns:
            good = np.where(epo_filt.metadata["badtrial"].to_numpy(int) == 0)[0]
            epo_filt = epo_filt[good]
            mod2 = mod2.iloc[good].reset_index(drop=True)

        # finite check
        vals = mod2[[pvar, mvar, rt_col]].to_numpy(float)
        keep = np.all(np.isfinite(vals), axis=1)

        if keep.sum() < 10:
            print(f"Skipping {pa}: too few finite trials ({keep.sum()})")
            skipped_subjects.append(pa)
            continue

        epo_filt = epo_filt[keep]
        mod2k = mod2.iloc[keep].reset_index(drop=True)

        # variance checks (like v13)
        if np.nanstd(mod2k[pvar]) == 0 or np.nanstd(mod2k[mvar]) == 0:
            print(f"Skipping {pa}: zero variance in pain/money")
            skipped_subjects.append(pa)
            continue
        if np.nanstd(mod2k[rt_col]) == 0:
            print(f"Skipping {pa}: zero variance in RT")
            skipped_subjects.append(pa)
            continue

        # ----------------------------
        # keep epochs with metadata for binning plots later
        # ----------------------------
        epo_keep_for_meta = epo_filt.copy()
        md = epo_keep_for_meta.metadata.reset_index(drop=True).copy()
        md[pvar] = mod2k[pvar].values
        md[mvar] = mod2k[mvar].values
        md[rt_col] = mod2k[rt_col].values
        epo_keep_for_meta.metadata = md

        all_epos[0].append(epo_keep_for_meta.copy())  # pain epochs
        all_epos[1].append(epo_keep_for_meta.copy())  # money epochs

        # ----------------------------
        # z-score predictors (like v13)
        # ----------------------------
        mod2k["Intercept"] = 1.0
        mod2k["pain_z"]  = stats.zscore(mod2k[pvar].to_numpy(float))
        mod2k["money_z"] = stats.zscore(mod2k[mvar].to_numpy(float))
        mod2k["RT_z"]    = stats.zscore(mod2k[rt_col].to_numpy(float))

        # ----------------------------
        # z-score EEG across trials (Scaler + EpochsArray preserving tmin/events)
        # ----------------------------
        scale = Scaler(scalings="mean")
        data_z = scale.fit_transform(epo_filt.get_data())

        epo_z = mne.EpochsArray(
            data_z,
            info=epo_filt.info,
            events=epo_filt.events,
            tmin=epo_filt.tmin,
            event_id=epo_filt.event_id,
            metadata=epo_filt.metadata
        )

        # ----------------------------
        # GLM: joint vs separate (v13 style)
        # ----------------------------
        if v27_mode == "separate":
            # pain GLM
            design_p = mod2k[["Intercept", "pain_z", "RT_z"]]
            res_p = mne.stats.linear_regression(
                epo_z, design_p, names=["Intercept", "pain_z", "RT_z"]
            )
            beta_pain = res_p["pain_z"].beta

            # money GLM
            design_m = mod2k[["Intercept", "money_z", "RT_z"]]
            res_m = mne.stats.linear_regression(
                epo_z, design_m, names=["Intercept", "money_z", "RT_z"]
            )
            beta_money = res_m["money_z"].beta

        elif v27_mode == "joint":
            # one joint GLM
            design = mod2k[["Intercept", "pain_z", "money_z", "RT_z"]]
            res = mne.stats.linear_regression(
                epo_z, design, names=["Intercept", "pain_z", "money_z", "RT_z"]
            )
            beta_pain  = res["pain_z"].beta
            beta_money = res["money_z"].beta

        else:
            raise ValueError("v27_mode must be 'separate' or 'joint'")

        # ----------------------------
        # store for second-level
        # ----------------------------
        betas[0].append(beta_pain)
        betas[1].append(beta_money)

        allbetasnp.append(np.stack([beta_pain.data, beta_money.data]))  # (2, n_chan, n_time)

        included_subjects.append(pa)
        print(f"Included {pa} (n_trials={len(epo_filt)})")

    if len(allbetasnp) == 0:
        raise RuntimeError("v27: no subjects included after filtering/alignment.")

    # (n_subj, 2, n_chan, n_time)
    allbetas = np.stack(allbetasnp)

    # save betas + subject lists
    np.save(z_dir / "ols_2ndlevel_betas.npy", allbetas)
    np.save(z_dir / "included_subjects.npy", np.array(included_subjects, dtype=object))
    np.save(z_dir / "skipped_subjects.npy",  np.array(skipped_subjects, dtype=object))

    # grand-average betas (Evoked objects)
    beta_gavg = [mne.grand_average(betas[0]), mne.grand_average(betas[1])]
    np.save(z_dir / "ols_2ndlevel_betasavg.npy", np.array(beta_gavg, dtype=object), allow_pickle=True)

    # save epochs for plotting / binning (one file per regressor like v13)
    epo_save_p = mne.concatenate_epochs(all_epos[0])
    epo_save_m = mne.concatenate_epochs(all_epos[1])
    epo_save_p.save(z_dir / f"ols_2ndlevel_allepochs-epo_{pvar}.fif", overwrite=True)
    epo_save_m.save(z_dir / f"ols_2ndlevel_allepochs-epo_{mvar}.fif", overwrite=True)

    # ------------------------------------------------------------
    # second-level spatio-temporal cluster test (same output as before)
    # ------------------------------------------------------------
    connect, _ = mne.channels.find_ch_adjacency(beta_gavg[0].info, ch_type="eeg")

    # two-sided threshold
    p_thresh = param["cluster_threshold"] / 2.0
    cluster_threshold = -stats.t.ppf(p_thresh, df=allbetas.shape[0] - 1)

    tvals_list, pvals_list = [], []

    # cluster test expects (n_subj, n_time, n_chan)
    for ridx, name in enumerate(regnames):
        data_reg = allbetas[:, ridx, :, :]      # (n_subj, n_chan, n_time)
        testdata = np.swapaxes(data_reg, 2, 1)  # (n_subj, n_time, n_chan)

        tval, clusters, cluster_p_values, _ = st_clust_1s_ttest(
            testdata,
            threshold=cluster_threshold,
            adjacency=connect,
            n_permutations=param["nperms"],
            n_jobs=param["njobs"],
            buffer_size=None
        )

        # build p-map
        pmap = np.ones_like(tval)
        for c, p_val in zip(clusters, cluster_p_values):
            pmap[c] = p_val

        # (optional) per-regressor files like v13
        np.save(z_dir / f"ols_2ndlevel_tval_{name}.npy", tval)
        np.save(z_dir / f"ols_2ndlevel_pval_{name}.npy", pmap)

        tvals_list.append(tval)
        pvals_list.append(pmap)

    np.save(z_dir / "ols_2ndlevel_tvals.npy", np.stack(tvals_list))  # (2, n_time, n_chan)
    np.save(z_dir / "ols_2ndlevel_pvals.npy", np.stack(pvals_list))  # (2, n_time, n_chan)

    print(f"v27 ({v27_mode}) done. Included n={len(included_subjects)}, skipped n={len(skipped_subjects)}.")


elif version == 28:

    z_dir = Path(outpath) / "Zscoring"
    z_dir.mkdir(parents=True, exist_ok=True)

    # columns in mod_data
    pvar, mvar, rt_col = "sv_pain_para", "sv_money", "rt"

    # filenames for saving epochs (like v13 regvars)
    regvars  = [pvar, mvar]
    regnames = ["SV_pain_para", "SV_money"]

    # sanity checks
    for col in [pvar, mvar, rt_col, "participant", "trialsnum"]:
        if col not in mod_data.columns:
            raise RuntimeError(f"v28: mod_data missing required column: '{col}'")

    included_subjects, skipped_subjects = [], []

    all_epos = [[] for _ in range(2)]   # [pain_epochs, money_epochs]
    betas    = [[] for _ in range(2)]   # [pain_beta_evokeds, money_beta_evokeds]
    allbetasnp = []                    # list of (2, n_chan, n_time)

    for pa in part:
        print(f"\n--- v28 ({v28_mode}) Processing {pa} ---")

        # behavioural rows for subject
        mod2 = mod_data.loc[mod_data["participant"] == pa].copy()
        if len(mod2) == 0:
            print(f"Skipping {pa}: no behavioural rows")
            skipped_subjects.append(pa)
            continue

        # load LONG cue-locked epochs
        epo = mne.read_epochs(
            opj(basepath, pa, "eeg", "erps_long", f"{pa}_decision_cues_long_singletrials-epo.fif"),
            preload=True
        )

        if "trialsnum" not in epo.metadata.columns:
            raise RuntimeError(
                "v28: epochs metadata missing 'trialsnum'. "
                "Did you save it in ERP creation?"
            )

        # ----------------------------
        # robust alignment by trialsnum
        # ----------------------------
        keep_epo = epo.metadata["trialsnum"].isin(mod2["trialsnum"])
        epo_filt = epo[keep_epo]

        mod2 = mod2.loc[mod2["trialsnum"].isin(epo_filt.metadata["trialsnum"])].copy()

        if len(epo_filt) == 0 or len(mod2) == 0:
            print(f"Skipping {pa}: no overlapping trialsnum")
            skipped_subjects.append(pa)
            continue

        # sort both by trialsnum to guarantee alignment
        epo_order = np.argsort(epo_filt.metadata["trialsnum"].to_numpy())
        epo_filt = epo_filt[epo_order]
        mod2 = mod2.sort_values("trialsnum").reset_index(drop=True)

        # resample
        if epo_filt.info["sfreq"] != param["testresampfreq"]:
            epo_filt = epo_filt.resample(param["testresampfreq"])

        # drop bad trials if available
        if "badtrial" in epo_filt.metadata.columns:
            good = np.where(epo_filt.metadata["badtrial"].to_numpy(int) == 0)[0]
            epo_filt = epo_filt[good]
            mod2 = mod2.iloc[good].reset_index(drop=True)

        # finite check
        vals = mod2[[pvar, mvar, rt_col]].to_numpy(float)
        keep = np.all(np.isfinite(vals), axis=1)

        if keep.sum() < 10:
            print(f"Skipping {pa}: too few finite trials ({keep.sum()})")
            skipped_subjects.append(pa)
            continue

        epo_filt = epo_filt[keep]
        mod2k = mod2.iloc[keep].reset_index(drop=True)

        # variance checks (like v13)
        if np.nanstd(mod2k[pvar]) == 0 or np.nanstd(mod2k[mvar]) == 0:
            print(f"Skipping {pa}: zero variance in pain/money")
            skipped_subjects.append(pa)
            continue
        if np.nanstd(mod2k[rt_col]) == 0:
            print(f"Skipping {pa}: zero variance in RT")
            skipped_subjects.append(pa)
            continue

        # ----------------------------
        # keep epochs with metadata for binning plots later
        # ----------------------------
        epo_keep_for_meta = epo_filt.copy()
        md = epo_keep_for_meta.metadata.reset_index(drop=True).copy()
        md[pvar] = mod2k[pvar].values
        md[mvar] = mod2k[mvar].values
        md[rt_col] = mod2k[rt_col].values
        epo_keep_for_meta.metadata = md

        all_epos[0].append(epo_keep_for_meta.copy())  # pain epochs
        all_epos[1].append(epo_keep_for_meta.copy())  # money epochs

        # ----------------------------
        # z-score predictors (like v13)
        # ----------------------------
        mod2k["Intercept"] = 1.0
        mod2k["sv_pain_para_z"]  = stats.zscore(mod2k[pvar].to_numpy(float))
        mod2k["sv_money_z"] = stats.zscore(mod2k[mvar].to_numpy(float))
        mod2k["RT_z"]    = stats.zscore(mod2k[rt_col].to_numpy(float))

        # ----------------------------
        # z-score EEG across trials (Scaler + EpochsArray preserving tmin/events)
        # ----------------------------
        scale = Scaler(scalings="mean")
        data_z = scale.fit_transform(epo_filt.get_data())

        epo_z = mne.EpochsArray(
            data_z,
            info=epo_filt.info,
            events=epo_filt.events,
            tmin=epo_filt.tmin,
            event_id=epo_filt.event_id,
            metadata=epo_filt.metadata
        )

        # ----------------------------
        # GLM: joint vs separate (v13 style)
        # ----------------------------
        if v28_mode == "separate":
            # pain GLM
            design_p = mod2k[["Intercept", "sv_pain_para_z", "RT_z"]]
            res_p = mne.stats.linear_regression(
                epo_z, design_p, names=["Intercept", "sv_pain_para_z", "RT_z"]
            )
            beta_pain = res_p["sv_pain_para_z"].beta

            # money GLM
            design_m = mod2k[["Intercept", "sv_money_z", "RT_z"]]
            res_m = mne.stats.linear_regression(
                epo_z, design_m, names=["Intercept", "sv_money_z", "RT_z"]
            )
            beta_money = res_m["sv_money_z"].beta

        elif v28_mode == "joint":
            # one joint GLM
            design = mod2k[["Intercept", "sv_pain_para_z", "sv_money_z", "RT_z"]]
            res = mne.stats.linear_regression(
                epo_z, design, names=["Intercept", "sv_pain_para_z", "sv_money_z", "RT_z"]
            )
            beta_pain  = res["sv_pain_para_z"].beta
            beta_money = res["sv_money_z"].beta

        else:
            raise ValueError("v26_mode must be 'separate' or 'joint'")

        # ----------------------------
        # store for second-level
        # ----------------------------
        betas[0].append(beta_pain)
        betas[1].append(beta_money)

        allbetasnp.append(np.stack([beta_pain.data, beta_money.data]))  # (2, n_chan, n_time)

        included_subjects.append(pa)
        print(f"Included {pa} (n_trials={len(epo_filt)})")

    if len(allbetasnp) == 0:
        raise RuntimeError("v28: no subjects included after filtering/alignment.")

    # (n_subj, 2, n_chan, n_time)
    allbetas = np.stack(allbetasnp)

    # save betas + subject lists
    np.save(z_dir / "ols_2ndlevel_betas.npy", allbetas)
    np.save(z_dir / "included_subjects.npy", np.array(included_subjects, dtype=object))
    np.save(z_dir / "skipped_subjects.npy",  np.array(skipped_subjects, dtype=object))

    # grand-average betas (Evoked objects)
    beta_gavg = [mne.grand_average(betas[0]), mne.grand_average(betas[1])]
    np.save(z_dir / "ols_2ndlevel_betasavg.npy", np.array(beta_gavg, dtype=object), allow_pickle=True)

    # save epochs for plotting / binning (one file per regressor like v13)
    epo_save_p = mne.concatenate_epochs(all_epos[0])
    epo_save_m = mne.concatenate_epochs(all_epos[1])
    epo_save_p.save(z_dir / f"ols_2ndlevel_allepochs-epo_{pvar}.fif", overwrite=True)
    epo_save_m.save(z_dir / f"ols_2ndlevel_allepochs-epo_{mvar}.fif", overwrite=True)

    # ------------------------------------------------------------
    # second-level spatio-temporal cluster test (same output as before)
    # ------------------------------------------------------------
    connect, _ = mne.channels.find_ch_adjacency(beta_gavg[0].info, ch_type="eeg")

    # two-sided threshold
    p_thresh = param["cluster_threshold"] / 2.0
    cluster_threshold = -stats.t.ppf(p_thresh, df=allbetas.shape[0] - 1)

    tvals_list, pvals_list = [], []

    # cluster test expects (n_subj, n_time, n_chan)
    for ridx, name in enumerate(regnames):
        data_reg = allbetas[:, ridx, :, :]      # (n_subj, n_chan, n_time)
        testdata = np.swapaxes(data_reg, 2, 1)  # (n_subj, n_time, n_chan)

        tval, clusters, cluster_p_values, _ = st_clust_1s_ttest(
            testdata,
            threshold=cluster_threshold,
            adjacency=connect,
            n_permutations=param["nperms"],
            n_jobs=param["njobs"],
            buffer_size=None
        )

        # build p-map
        pmap = np.ones_like(tval)
        for c, p_val in zip(clusters, cluster_p_values):
            pmap[c] = p_val

        # (optional) per-regressor files like v13
        np.save(z_dir / f"ols_2ndlevel_tval_{name}.npy", tval)
        np.save(z_dir / f"ols_2ndlevel_pval_{name}.npy", pmap)

        tvals_list.append(tval)
        pvals_list.append(pmap)

    np.save(z_dir / "ols_2ndlevel_tvals.npy", np.stack(tvals_list))  # (2, n_time, n_chan)
    np.save(z_dir / "ols_2ndlevel_pvals.npy", np.stack(pvals_list))  # (2, n_time, n_chan)

    print(f"v28 ({v28_mode}) done. Included n={len(included_subjects)}, skipped n={len(skipped_subjects)}.")


elif version == 29:

    z_dir = Path(outpath) / "Zscoring"
    z_dir.mkdir(parents=True, exist_ok=True)

    # columns in mod_data
    pvar, mvar, rt_col = "painlevel", "moneylevel", "rt"

    # filenames for saving epochs (like v13 regvars)
    regvars  = [pvar, mvar]
    regnames = ["Painlevel", "Moneylevel"]

    # sanity checks
    for col in [pvar, mvar, rt_col, "participant", "trialsnum"]:
        if col not in mod_data.columns:
            raise RuntimeError(f"v29: mod_data missing required column: '{col}'")

    included_subjects, skipped_subjects = [], []

    all_epos = [[] for _ in range(2)]   # [pain_epochs, money_epochs]
    betas    = [[] for _ in range(2)]   # [pain_beta_evokeds, money_beta_evokeds]
    allbetasnp = []                    # list of (2, n_chan, n_time)

    for pa in part:
        print(f"\n--- v29 ({v29_mode}) Processing {pa} ---")

        # behavioural rows for subject
        mod2 = mod_data.loc[mod_data["participant"] == pa].copy()
        if len(mod2) == 0:
            print(f"Skipping {pa}: no behavioural rows")
            skipped_subjects.append(pa)
            continue

        # load LONG cue-locked epochs
        epo = mne.read_epochs(
            opj(basepath, pa, "eeg", "erps_long", f"{pa}_decision_cues_long_singletrials-epo.fif"),
            preload=True
        )

        if "trialsnum" not in epo.metadata.columns:
            raise RuntimeError(
                "v29: epochs metadata missing 'trialsnum'. "
                "Did you save it in ERP creation?"
            )

        # ----------------------------
        # robust alignment by trialsnum
        # ----------------------------
        keep_epo = epo.metadata["trialsnum"].isin(mod2["trialsnum"])
        epo_filt = epo[keep_epo]

        mod2 = mod2.loc[mod2["trialsnum"].isin(epo_filt.metadata["trialsnum"])].copy()

        if len(epo_filt) == 0 or len(mod2) == 0:
            print(f"Skipping {pa}: no overlapping trialsnum")
            skipped_subjects.append(pa)
            continue

        # sort both by trialsnum to guarantee alignment
        epo_order = np.argsort(epo_filt.metadata["trialsnum"].to_numpy())
        epo_filt = epo_filt[epo_order]
        mod2 = mod2.sort_values("trialsnum").reset_index(drop=True)

        # resample
        if epo_filt.info["sfreq"] != param["testresampfreq"]:
            epo_filt = epo_filt.resample(param["testresampfreq"])

        # drop bad trials if available
        if "badtrial" in epo_filt.metadata.columns:
            good = np.where(epo_filt.metadata["badtrial"].to_numpy(int) == 0)[0]
            epo_filt = epo_filt[good]
            mod2 = mod2.iloc[good].reset_index(drop=True)

        # finite check
        vals = mod2[[pvar, mvar, rt_col]].to_numpy(float)
        keep = np.all(np.isfinite(vals), axis=1)

        if keep.sum() < 10:
            print(f"Skipping {pa}: too few finite trials ({keep.sum()})")
            skipped_subjects.append(pa)
            continue

        epo_filt = epo_filt[keep]
        mod2k = mod2.iloc[keep].reset_index(drop=True)

        # variance checks (like v13)
        if np.nanstd(mod2k[pvar]) == 0 or np.nanstd(mod2k[mvar]) == 0:
            print(f"Skipping {pa}: zero variance in pain/money")
            skipped_subjects.append(pa)
            continue
        if np.nanstd(mod2k[rt_col]) == 0:
            print(f"Skipping {pa}: zero variance in RT")
            skipped_subjects.append(pa)
            continue

        # ----------------------------
        # keep epochs with metadata for binning plots later
        # ----------------------------
        epo_keep_for_meta = epo_filt.copy()
        md = epo_keep_for_meta.metadata.reset_index(drop=True).copy()
        md[pvar] = mod2k[pvar].values
        md[mvar] = mod2k[mvar].values
        md[rt_col] = mod2k[rt_col].values
        epo_keep_for_meta.metadata = md

        all_epos[0].append(epo_keep_for_meta.copy())  # pain epochs
        all_epos[1].append(epo_keep_for_meta.copy())  # money epochs

        # ----------------------------
        # z-score predictors (like v13)
        # ----------------------------
        mod2k["Intercept"] = 1.0
        mod2k["pain_z"]  = stats.zscore(mod2k[pvar].to_numpy(float))
        mod2k["money_z"] = stats.zscore(mod2k[mvar].to_numpy(float))
        mod2k["RT_z"]    = stats.zscore(mod2k[rt_col].to_numpy(float))

        # ----------------------------
        # z-score EEG across trials (Scaler + EpochsArray preserving tmin/events)
        # ----------------------------
        scale = Scaler(scalings="mean")
        data_z = scale.fit_transform(epo_filt.get_data())

        epo_z = mne.EpochsArray(
            data_z,
            info=epo_filt.info,
            events=epo_filt.events,
            tmin=epo_filt.tmin,
            event_id=epo_filt.event_id,
            metadata=epo_filt.metadata
        )

        # ----------------------------
        # GLM: joint vs separate (v13 style)
        # ----------------------------
        if v29_mode == "separate":
            # pain GLM
            design_p = mod2k[["Intercept", "pain_z", "RT_z"]]
            res_p = mne.stats.linear_regression(
                epo_z, design_p, names=["Intercept", "pain_z", "RT_z"]
            )
            beta_pain = res_p["pain_z"].beta

            # money GLM
            design_m = mod2k[["Intercept", "money_z", "RT_z"]]
            res_m = mne.stats.linear_regression(
                epo_z, design_m, names=["Intercept", "money_z", "RT_z"]
            )
            beta_money = res_m["money_z"].beta

        elif v29_mode == "joint":
            # one joint GLM
            design = mod2k[["Intercept", "pain_z", "money_z", "RT_z"]]
            res = mne.stats.linear_regression(
                epo_z, design, names=["Intercept", "pain_z", "money_z", "RT_z"]
            )
            beta_pain  = res["pain_z"].beta
            beta_money = res["money_z"].beta

        else:
            raise ValueError("v29_mode must be 'separate' or 'joint'")

        # ----------------------------
        # store for second-level
        # ----------------------------
        betas[0].append(beta_pain)
        betas[1].append(beta_money)

        allbetasnp.append(np.stack([beta_pain.data, beta_money.data]))  # (2, n_chan, n_time)

        included_subjects.append(pa)
        print(f"Included {pa} (n_trials={len(epo_filt)})")

    if len(allbetasnp) == 0:
        raise RuntimeError("v29: no subjects included after filtering/alignment.")

    # (n_subj, 2, n_chan, n_time)
    allbetas = np.stack(allbetasnp)

    # save betas + subject lists
    np.save(z_dir / "ols_2ndlevel_betas.npy", allbetas)
    np.save(z_dir / "included_subjects.npy", np.array(included_subjects, dtype=object))
    np.save(z_dir / "skipped_subjects.npy",  np.array(skipped_subjects, dtype=object))

    # grand-average betas (Evoked objects)
    beta_gavg = [mne.grand_average(betas[0]), mne.grand_average(betas[1])]
    np.save(z_dir / "ols_2ndlevel_betasavg.npy", np.array(beta_gavg, dtype=object), allow_pickle=True)

    # save epochs for plotting / binning (one file per regressor like v13)
    epo_save_p = mne.concatenate_epochs(all_epos[0])
    epo_save_m = mne.concatenate_epochs(all_epos[1])
    epo_save_p.save(z_dir / f"ols_2ndlevel_allepochs-epo_{pvar}.fif", overwrite=True)
    epo_save_m.save(z_dir / f"ols_2ndlevel_allepochs-epo_{mvar}.fif", overwrite=True)

    # ------------------------------------------------------------
    # second-level spatio-temporal cluster test (same output as before)
    # ------------------------------------------------------------
    connect, _ = mne.channels.find_ch_adjacency(beta_gavg[0].info, ch_type="eeg")

    # two-sided threshold
    p_thresh = param["cluster_threshold"] / 2.0
    cluster_threshold = -stats.t.ppf(p_thresh, df=allbetas.shape[0] - 1)

    tvals_list, pvals_list = [], []

    # cluster test expects (n_subj, n_time, n_chan)
    for ridx, name in enumerate(regnames):
        data_reg = allbetas[:, ridx, :, :]      # (n_subj, n_chan, n_time)
        testdata = np.swapaxes(data_reg, 2, 1)  # (n_subj, n_time, n_chan)

        tval, clusters, cluster_p_values, _ = st_clust_1s_ttest(
            testdata,
            threshold=cluster_threshold,
            adjacency=connect,
            n_permutations=param["nperms"],
            n_jobs=param["njobs"],
            buffer_size=None
        )

        # build p-map
        pmap = np.ones_like(tval)
        for c, p_val in zip(clusters, cluster_p_values):
            pmap[c] = p_val

        # (optional) per-regressor files like v13
        np.save(z_dir / f"ols_2ndlevel_tval_{name}.npy", tval)
        np.save(z_dir / f"ols_2ndlevel_pval_{name}.npy", pmap)

        tvals_list.append(tval)
        pvals_list.append(pmap)

    np.save(z_dir / "ols_2ndlevel_tvals.npy", np.stack(tvals_list))  # (2, n_time, n_chan)
    np.save(z_dir / "ols_2ndlevel_pvals.npy", np.stack(pvals_list))  # (2, n_time, n_chan)

    print(f"v29 ({v29_mode}) done. Included n={len(included_subjects)}, skipped n={len(skipped_subjects)}.")



elif version == 30:

    z_dir = Path(outpath) / "Zscoring"
    z_dir.mkdir(parents=True, exist_ok=True)

    # columns in mod_data
    pvar, mvar, rt_col = "painlevel", "moneylevel", "rt"

    # filenames for saving epochs (like v13 regvars)
    regvars  = [pvar, mvar]
    regnames = ["Painlevel", "Moneylevel"]

    # sanity checks
    for col in [pvar, mvar, rt_col, "participant", "trialsnum"]:
        if col not in mod_data.columns:
            raise RuntimeError(f"v30: mod_data missing required column: '{col}'")

    included_subjects, skipped_subjects = [], []

    all_epos = [[] for _ in range(2)]   # [pain_epochs, money_epochs]
    betas    = [[] for _ in range(2)]   # [pain_beta_evokeds, money_beta_evokeds]
    allbetasnp = []                    # list of (2, n_chan, n_time)

    for pa in part:
        print(f"\n--- v30 ({v30_mode}) Processing {pa} ---")

        # behavioural rows for subject
        mod2 = mod_data.loc[mod_data["participant"] == pa].copy()
        if len(mod2) == 0:
            print(f"Skipping {pa}: no behavioural rows")
            skipped_subjects.append(pa)
            continue

        # load LONG cue-locked epochs
        epo = mne.read_epochs(
            opj(basepath, pa, "eeg", "erps_long", f"{pa}_decision_cues_long_singletrials-epo.fif"),
            preload=True
        )

        if "trialsnum" not in epo.metadata.columns:
            raise RuntimeError(
                "v29: epochs metadata missing 'trialsnum'. "
                "Did you save it in ERP creation?"
            )

        # ----------------------------
        # robust alignment by trialsnum
        # ----------------------------
        keep_epo = epo.metadata["trialsnum"].isin(mod2["trialsnum"])
        epo_filt = epo[keep_epo]

        mod2 = mod2.loc[mod2["trialsnum"].isin(epo_filt.metadata["trialsnum"])].copy()

        if len(epo_filt) == 0 or len(mod2) == 0:
            print(f"Skipping {pa}: no overlapping trialsnum")
            skipped_subjects.append(pa)
            continue

        # sort both by trialsnum to guarantee alignment
        epo_order = np.argsort(epo_filt.metadata["trialsnum"].to_numpy())
        epo_filt = epo_filt[epo_order]
        mod2 = mod2.sort_values("trialsnum").reset_index(drop=True)

        # resample
        if epo_filt.info["sfreq"] != param["testresampfreq"]:
            epo_filt = epo_filt.resample(param["testresampfreq"])

        # drop bad trials if available
        if "badtrial" in epo_filt.metadata.columns:
            good = np.where(epo_filt.metadata["badtrial"].to_numpy(int) == 0)[0]
            epo_filt = epo_filt[good]
            mod2 = mod2.iloc[good].reset_index(drop=True)

        # finite check
        vals = mod2[[pvar, mvar, rt_col]].to_numpy(float)
        keep = np.all(np.isfinite(vals), axis=1)

        if keep.sum() < 10:
            print(f"Skipping {pa}: too few finite trials ({keep.sum()})")
            skipped_subjects.append(pa)
            continue

        epo_filt = epo_filt[keep]
        mod2k = mod2.iloc[keep].reset_index(drop=True)

        # variance checks (like v13)
        if np.nanstd(mod2k[pvar]) == 0 or np.nanstd(mod2k[mvar]) == 0:
            print(f"Skipping {pa}: zero variance in pain/money")
            skipped_subjects.append(pa)
            continue
        if np.nanstd(mod2k[rt_col]) == 0:
            print(f"Skipping {pa}: zero variance in RT")
            skipped_subjects.append(pa)
            continue

        # ----------------------------
        # keep epochs with metadata for binning plots later
        # ----------------------------
        epo_keep_for_meta = epo_filt.copy()
        md = epo_keep_for_meta.metadata.reset_index(drop=True).copy()
        md[pvar] = mod2k[pvar].values
        md[mvar] = mod2k[mvar].values
        md[rt_col] = mod2k[rt_col].values
        epo_keep_for_meta.metadata = md

        all_epos[0].append(epo_keep_for_meta.copy())  # pain epochs
        all_epos[1].append(epo_keep_for_meta.copy())  # money epochs

        # ----------------------------
        # z-score predictors (like v13)
        # ----------------------------
        mod2k["Intercept"] = 1.0
        mod2k["pain_z"]  = stats.zscore(mod2k[pvar].to_numpy(float))
        mod2k["money_z"] = stats.zscore(mod2k[mvar].to_numpy(float))
        mod2k["RT_z"]    = stats.zscore(mod2k[rt_col].to_numpy(float))

        # ----------------------------
        # z-score EEG across trials (Scaler + EpochsArray preserving tmin/events)
        # ----------------------------
        scale = Scaler(scalings="mean")
        data_z = scale.fit_transform(epo_filt.get_data())

        epo_z = mne.EpochsArray(
            data_z,
            info=epo_filt.info,
            events=epo_filt.events,
            tmin=epo_filt.tmin,
            event_id=epo_filt.event_id,
            metadata=epo_filt.metadata
        )

        # ----------------------------
        # GLM: joint vs separate (v13 style)
        # ----------------------------
        if v30_mode == "separate":
            # pain GLM
            design_p = mod2k[["Intercept", "pain_z", "RT_z"]]
            res_p = mne.stats.linear_regression(
                epo_z, design_p, names=["Intercept", "pain_z", "RT_z"]
            )
            beta_pain = res_p["pain_z"].beta

            # money GLM
            design_m = mod2k[["Intercept", "money_z", "RT_z"]]
            res_m = mne.stats.linear_regression(
                epo_z, design_m, names=["Intercept", "money_z", "RT_z"]
            )
            beta_money = res_m["money_z"].beta

        elif v30_mode == "joint":
            # one joint GLM
            design = mod2k[["Intercept", "pain_z", "money_z", "RT_z"]]
            res = mne.stats.linear_regression(
                epo_z, design, names=["Intercept", "pain_z", "money_z", "RT_z"]
            )
            beta_pain  = res["pain_z"].beta
            beta_money = res["money_z"].beta

        else:
            raise ValueError("v30_mode must be 'separate' or 'joint'")

        # ----------------------------
        # store for second-level
        # ----------------------------
        betas[0].append(beta_pain)
        betas[1].append(beta_money)

        allbetasnp.append(np.stack([beta_pain.data, beta_money.data]))  # (2, n_chan, n_time)

        included_subjects.append(pa)
        print(f"Included {pa} (n_trials={len(epo_filt)})")

    if len(allbetasnp) == 0:
        raise RuntimeError("v30: no subjects included after filtering/alignment.")

    # (n_subj, 2, n_chan, n_time)
    allbetas = np.stack(allbetasnp)

    # save betas + subject lists
    np.save(z_dir / "ols_2ndlevel_betas.npy", allbetas)
    np.save(z_dir / "included_subjects.npy", np.array(included_subjects, dtype=object))
    np.save(z_dir / "skipped_subjects.npy",  np.array(skipped_subjects, dtype=object))

    # grand-average betas (Evoked objects)
    beta_gavg = [mne.grand_average(betas[0]), mne.grand_average(betas[1])]
    np.save(z_dir / "ols_2ndlevel_betasavg.npy", np.array(beta_gavg, dtype=object), allow_pickle=True)

    # save epochs for plotting / binning (one file per regressor like v13)
    epo_save_p = mne.concatenate_epochs(all_epos[0])
    epo_save_m = mne.concatenate_epochs(all_epos[1])
    epo_save_p.save(z_dir / f"ols_2ndlevel_allepochs-epo_{pvar}.fif", overwrite=True)
    epo_save_m.save(z_dir / f"ols_2ndlevel_allepochs-epo_{mvar}.fif", overwrite=True)

    # ------------------------------------------------------------
    # second-level spatio-temporal cluster test (same output as before)
    # ------------------------------------------------------------
    connect, _ = mne.channels.find_ch_adjacency(beta_gavg[0].info, ch_type="eeg")

    # two-sided threshold
    p_thresh = param["cluster_threshold"] / 2.0
    cluster_threshold = -stats.t.ppf(p_thresh, df=allbetas.shape[0] - 1)

    tvals_list, pvals_list = [], []

    # cluster test expects (n_subj, n_time, n_chan)
    for ridx, name in enumerate(regnames):
        data_reg = allbetas[:, ridx, :, :]      # (n_subj, n_chan, n_time)
        testdata = np.swapaxes(data_reg, 2, 1)  # (n_subj, n_time, n_chan)

        tval, clusters, cluster_p_values, _ = st_clust_1s_ttest(
            testdata,
            threshold=cluster_threshold,
            adjacency=connect,
            n_permutations=param["nperms"],
            n_jobs=param["njobs"],
            buffer_size=None
        )

        # build p-map
        pmap = np.ones_like(tval)
        for c, p_val in zip(clusters, cluster_p_values):
            pmap[c] = p_val

        # (optional) per-regressor files like v13
        np.save(z_dir / f"ols_2ndlevel_tval_{name}.npy", tval)
        np.save(z_dir / f"ols_2ndlevel_pval_{name}.npy", pmap)

        tvals_list.append(tval)
        pvals_list.append(pmap)

    np.save(z_dir / "ols_2ndlevel_tvals.npy", np.stack(tvals_list))  # (2, n_time, n_chan)
    np.save(z_dir / "ols_2ndlevel_pvals.npy", np.stack(pvals_list))  # (2, n_time, n_chan)

    print(f"v30 ({v30_mode}) done. Included n={len(included_subjects)}, skipped n={len(skipped_subjects)}.")


elif version == 31:

    z_dir = Path(outpath) / "Zscoring"
    z_dir.mkdir(parents=True, exist_ok=True)

    # columns in mod_data
    pvar, mvar, rt_col = "painlevel", "moneylevel", "rt"

    # filenames for saving epochs (like v13 regvars)
    regvars  = [pvar, mvar]
    regnames = ["Painlevel", "Moneylevel"]

    # sanity checks
    for col in [pvar, mvar, rt_col, "participant", "trialsnum"]:
        if col not in mod_data.columns:
            raise RuntimeError(f"v31: mod_data missing required column: '{col}'")

    included_subjects, skipped_subjects = [], []

    all_epos = [[] for _ in range(2)]   # [pain_epochs, money_epochs]
    betas    = [[] for _ in range(2)]   # [pain_beta_evokeds, money_beta_evokeds]
    allbetasnp = []                    # list of (2, n_chan, n_time)

    for pa in part:
        print(f"\n--- v31 ({v31_mode}) Processing {pa} ---")

        # behavioural rows for subject
        mod2 = mod_data.loc[mod_data["participant"] == pa].copy()
        if len(mod2) == 0:
            print(f"Skipping {pa}: no behavioural rows")
            skipped_subjects.append(pa)
            continue

        # load LONG cue-locked epochs
        epo = mne.read_epochs(
            opj(basepath, pa, "eeg", "erps_long", f"{pa}_decision_cues_long_singletrials-epo.fif"),
            preload=True
        )

        if "trialsnum" not in epo.metadata.columns:
            raise RuntimeError(
                "v31: epochs metadata missing 'trialsnum'. "
                "Did you save it in ERP creation?"
            )

        # ----------------------------
        # robust alignment by trialsnum
        # ----------------------------
        keep_epo = epo.metadata["trialsnum"].isin(mod2["trialsnum"])
        epo_filt = epo[keep_epo]

        mod2 = mod2.loc[mod2["trialsnum"].isin(epo_filt.metadata["trialsnum"])].copy()

        if len(epo_filt) == 0 or len(mod2) == 0:
            print(f"Skipping {pa}: no overlapping trialsnum")
            skipped_subjects.append(pa)
            continue

        # sort both by trialsnum to guarantee alignment
        epo_order = np.argsort(epo_filt.metadata["trialsnum"].to_numpy())
        epo_filt = epo_filt[epo_order]
        mod2 = mod2.sort_values("trialsnum").reset_index(drop=True)

        # resample
        if epo_filt.info["sfreq"] != param["testresampfreq"]:
            epo_filt = epo_filt.resample(param["testresampfreq"])

        # drop bad trials if available
        if "badtrial" in epo_filt.metadata.columns:
            good = np.where(epo_filt.metadata["badtrial"].to_numpy(int) == 0)[0]
            epo_filt = epo_filt[good]
            mod2 = mod2.iloc[good].reset_index(drop=True)

        # finite check
        vals = mod2[[pvar, mvar, rt_col]].to_numpy(float)
        keep = np.all(np.isfinite(vals), axis=1)

        if keep.sum() < 10:
            print(f"Skipping {pa}: too few finite trials ({keep.sum()})")
            skipped_subjects.append(pa)
            continue

        epo_filt = epo_filt[keep]
        mod2k = mod2.iloc[keep].reset_index(drop=True)

        # variance checks (like v13)
        if np.nanstd(mod2k[pvar]) == 0 or np.nanstd(mod2k[mvar]) == 0:
            print(f"Skipping {pa}: zero variance in pain/money")
            skipped_subjects.append(pa)
            continue
        if np.nanstd(mod2k[rt_col]) == 0:
            print(f"Skipping {pa}: zero variance in RT")
            skipped_subjects.append(pa)
            continue

        # ----------------------------
        # keep epochs with metadata for binning plots later
        # ----------------------------
        epo_keep_for_meta = epo_filt.copy()
        md = epo_keep_for_meta.metadata.reset_index(drop=True).copy()
        md[pvar] = mod2k[pvar].values
        md[mvar] = mod2k[mvar].values
        md[rt_col] = mod2k[rt_col].values
        epo_keep_for_meta.metadata = md

        all_epos[0].append(epo_keep_for_meta.copy())  # pain epochs
        all_epos[1].append(epo_keep_for_meta.copy())  # money epochs

        # ----------------------------
        # z-score predictors (like v13)
        # ----------------------------
        mod2k["Intercept"] = 1.0
        mod2k["pain_z"]  = stats.zscore(mod2k[pvar].to_numpy(float))
        mod2k["money_z"] = stats.zscore(mod2k[mvar].to_numpy(float))
        mod2k["RT_z"]    = stats.zscore(mod2k[rt_col].to_numpy(float))

        # ----------------------------
        # z-score EEG across trials (Scaler + EpochsArray preserving tmin/events)
        # ----------------------------
        scale = Scaler(scalings="mean")
        data_z = scale.fit_transform(epo_filt.get_data())

        epo_z = mne.EpochsArray(
            data_z,
            info=epo_filt.info,
            events=epo_filt.events,
            tmin=epo_filt.tmin,
            event_id=epo_filt.event_id,
            metadata=epo_filt.metadata
        )

        # ----------------------------
        # GLM: joint vs separate (v13 style)
        # ----------------------------
        if v31_mode == "separate":
            # pain GLM
            design_p = mod2k[["Intercept", "pain_z", "RT_z"]]
            res_p = mne.stats.linear_regression(
                epo_z, design_p, names=["Intercept", "pain_z", "RT_z"]
            )
            beta_pain = res_p["pain_z"].beta

            # money GLM
            design_m = mod2k[["Intercept", "money_z", "RT_z"]]
            res_m = mne.stats.linear_regression(
                epo_z, design_m, names=["Intercept", "money_z", "RT_z"]
            )
            beta_money = res_m["money_z"].beta

        elif v31_mode == "joint":
            # one joint GLM
            design = mod2k[["Intercept", "pain_z", "money_z", "RT_z"]]
            res = mne.stats.linear_regression(
                epo_z, design, names=["Intercept", "pain_z", "money_z", "RT_z"]
            )
            beta_pain  = res["pain_z"].beta
            beta_money = res["money_z"].beta

        else:
            raise ValueError("v31_mode must be 'separate' or 'joint'")

        # ----------------------------
        # store for second-level
        # ----------------------------
        betas[0].append(beta_pain)
        betas[1].append(beta_money)

        allbetasnp.append(np.stack([beta_pain.data, beta_money.data]))  # (2, n_chan, n_time)

        included_subjects.append(pa)
        print(f"Included {pa} (n_trials={len(epo_filt)})")

    if len(allbetasnp) == 0:
        raise RuntimeError("v31: no subjects included after filtering/alignment.")

    # (n_subj, 2, n_chan, n_time)
    allbetas = np.stack(allbetasnp)

    # save betas + subject lists
    np.save(z_dir / "ols_2ndlevel_betas.npy", allbetas)
    np.save(z_dir / "included_subjects.npy", np.array(included_subjects, dtype=object))
    np.save(z_dir / "skipped_subjects.npy",  np.array(skipped_subjects, dtype=object))

    # grand-average betas (Evoked objects)
    beta_gavg = [mne.grand_average(betas[0]), mne.grand_average(betas[1])]
    np.save(z_dir / "ols_2ndlevel_betasavg.npy", np.array(beta_gavg, dtype=object), allow_pickle=True)

    # save epochs for plotting / binning (one file per regressor like v13)
    epo_save_p = mne.concatenate_epochs(all_epos[0])
    epo_save_m = mne.concatenate_epochs(all_epos[1])
    epo_save_p.save(z_dir / f"ols_2ndlevel_allepochs-epo_{pvar}.fif", overwrite=True)
    epo_save_m.save(z_dir / f"ols_2ndlevel_allepochs-epo_{mvar}.fif", overwrite=True)

    # ------------------------------------------------------------
    # second-level spatio-temporal cluster test (same output as before)
    # ------------------------------------------------------------
    connect, _ = mne.channels.find_ch_adjacency(beta_gavg[0].info, ch_type="eeg")

    # two-sided threshold
    p_thresh = param["cluster_threshold"] / 2.0
    cluster_threshold = -stats.t.ppf(p_thresh, df=allbetas.shape[0] - 1)

    tvals_list, pvals_list = [], []

    # cluster test expects (n_subj, n_time, n_chan)
    for ridx, name in enumerate(regnames):
        data_reg = allbetas[:, ridx, :, :]      # (n_subj, n_chan, n_time)
        testdata = np.swapaxes(data_reg, 2, 1)  # (n_subj, n_time, n_chan)

        tval, clusters, cluster_p_values, _ = st_clust_1s_ttest(
            testdata,
            threshold=cluster_threshold,
            adjacency=connect,
            n_permutations=param["nperms"],
            n_jobs=param["njobs"],
            buffer_size=None
        )

        # build p-map
        pmap = np.ones_like(tval)
        for c, p_val in zip(clusters, cluster_p_values):
            pmap[c] = p_val

        # (optional) per-regressor files like v13
        np.save(z_dir / f"ols_2ndlevel_tval_{name}.npy", tval)
        np.save(z_dir / f"ols_2ndlevel_pval_{name}.npy", pmap)

        tvals_list.append(tval)
        pvals_list.append(pmap)

    np.save(z_dir / "ols_2ndlevel_tvals.npy", np.stack(tvals_list))  # (2, n_time, n_chan)
    np.save(z_dir / "ols_2ndlevel_pvals.npy", np.stack(pvals_list))  # (2, n_time, n_chan)

    print(f"v31 ({v31_mode}) done. Included n={len(included_subjects)}, skipped n={len(skipped_subjects)}.")




elif version == 32:

    z_dir = Path(outpath) / "Zscoring"
    z_dir.mkdir(parents=True, exist_ok=True)

    # columns in mod_data
    pvar, mvar, rt_col = "painlevel", "moneylevel", "rt"

    # filenames for saving epochs (like v13 regvars)
    regvars  = [pvar, mvar]
    regnames = ["Painlevel", "Moneylevel"]

    # sanity checks
    for col in [pvar, mvar, rt_col, "participant", "trialsnum"]:
        if col not in mod_data.columns:
            raise RuntimeError(f"v32: mod_data missing required column: '{col}'")

    included_subjects, skipped_subjects = [], []

    all_epos = [[] for _ in range(2)]   # [pain_epochs, money_epochs]
    betas    = [[] for _ in range(2)]   # [pain_beta_evokeds, money_beta_evokeds]
    allbetasnp = []                    # list of (2, n_chan, n_time)

    for pa in part:
        print(f"\n--- v32 ({v32_mode}) Processing {pa} ---")

        # behavioural rows for subject
        mod2 = mod_data.loc[mod_data["participant"] == pa].copy()
        if len(mod2) == 0:
            print(f"Skipping {pa}: no behavioural rows")
            skipped_subjects.append(pa)
            continue

        # load LONG cue-locked epochs
        epo = mne.read_epochs(
            opj(basepath, pa, "eeg", "erps_long", f"{pa}_decision_cues_long_singletrials-epo.fif"),
            preload=True
        )

        if "trialsnum" not in epo.metadata.columns:
            raise RuntimeError(
                "v32: epochs metadata missing 'trialsnum'. "
                "Did you save it in ERP creation?"
            )

        # ----------------------------
        # robust alignment by trialsnum
        # ----------------------------
        keep_epo = epo.metadata["trialsnum"].isin(mod2["trialsnum"])
        epo_filt = epo[keep_epo]

        mod2 = mod2.loc[mod2["trialsnum"].isin(epo_filt.metadata["trialsnum"])].copy()

        if len(epo_filt) == 0 or len(mod2) == 0:
            print(f"Skipping {pa}: no overlapping trialsnum")
            skipped_subjects.append(pa)
            continue

        # sort both by trialsnum to guarantee alignment
        epo_order = np.argsort(epo_filt.metadata["trialsnum"].to_numpy())
        epo_filt = epo_filt[epo_order]
        mod2 = mod2.sort_values("trialsnum").reset_index(drop=True)

        # resample
        if epo_filt.info["sfreq"] != param["testresampfreq"]:
            epo_filt = epo_filt.resample(param["testresampfreq"])

        # drop bad trials if available
        if "badtrial" in epo_filt.metadata.columns:
            good = np.where(epo_filt.metadata["badtrial"].to_numpy(int) == 0)[0]
            epo_filt = epo_filt[good]
            mod2 = mod2.iloc[good].reset_index(drop=True)

        # finite check
        vals = mod2[[pvar, mvar, rt_col]].to_numpy(float)
        keep = np.all(np.isfinite(vals), axis=1)

        if keep.sum() < 10:
            print(f"Skipping {pa}: too few finite trials ({keep.sum()})")
            skipped_subjects.append(pa)
            continue

        epo_filt = epo_filt[keep]
        mod2k = mod2.iloc[keep].reset_index(drop=True)

        # variance checks (like v13)
        if np.nanstd(mod2k[pvar]) == 0 or np.nanstd(mod2k[mvar]) == 0:
            print(f"Skipping {pa}: zero variance in pain/money")
            skipped_subjects.append(pa)
            continue
        if np.nanstd(mod2k[rt_col]) == 0:
            print(f"Skipping {pa}: zero variance in RT")
            skipped_subjects.append(pa)
            continue

        # ----------------------------
        # keep epochs with metadata for binning plots later
        # ----------------------------
        epo_keep_for_meta = epo_filt.copy()
        md = epo_keep_for_meta.metadata.reset_index(drop=True).copy()
        md[pvar] = mod2k[pvar].values
        md[mvar] = mod2k[mvar].values
        md[rt_col] = mod2k[rt_col].values
        epo_keep_for_meta.metadata = md

        all_epos[0].append(epo_keep_for_meta.copy())  # pain epochs
        all_epos[1].append(epo_keep_for_meta.copy())  # money epochs

        # ----------------------------
        # z-score predictors (like v13)
        # ----------------------------
        mod2k["Intercept"] = 1.0
        mod2k["pain_z"]  = stats.zscore(mod2k[pvar].to_numpy(float))
        mod2k["money_z"] = stats.zscore(mod2k[mvar].to_numpy(float))
        mod2k["RT_z"]    = stats.zscore(mod2k[rt_col].to_numpy(float))

        # ----------------------------
        # z-score EEG across trials (Scaler + EpochsArray preserving tmin/events)
        # ----------------------------
        scale = Scaler(scalings="mean")
        data_z = scale.fit_transform(epo_filt.get_data())

        epo_z = mne.EpochsArray(
            data_z,
            info=epo_filt.info,
            events=epo_filt.events,
            tmin=epo_filt.tmin,
            event_id=epo_filt.event_id,
            metadata=epo_filt.metadata
        )

        # ----------------------------
        # GLM: joint vs separate (v13 style)
        # ----------------------------
        if v32_mode == "separate":
            # pain GLM
            design_p = mod2k[["Intercept", "pain_z", "RT_z"]]
            res_p = mne.stats.linear_regression(
                epo_z, design_p, names=["Intercept", "pain_z", "RT_z"]
            )
            beta_pain = res_p["pain_z"].beta

            # money GLM
            design_m = mod2k[["Intercept", "money_z", "RT_z"]]
            res_m = mne.stats.linear_regression(
                epo_z, design_m, names=["Intercept", "money_z", "RT_z"]
            )
            beta_money = res_m["money_z"].beta

        elif v32_mode == "joint":
            # one joint GLM
            design = mod2k[["Intercept", "pain_z", "money_z", "RT_z"]]
            res = mne.stats.linear_regression(
                epo_z, design, names=["Intercept", "pain_z", "money_z", "RT_z"]
            )
            beta_pain  = res["pain_z"].beta
            beta_money = res["money_z"].beta

        else:
            raise ValueError("v32_mode must be 'separate' or 'joint'")

        # ----------------------------
        # store for second-level
        # ----------------------------
        betas[0].append(beta_pain)
        betas[1].append(beta_money)

        allbetasnp.append(np.stack([beta_pain.data, beta_money.data]))  # (2, n_chan, n_time)

        included_subjects.append(pa)
        print(f"Included {pa} (n_trials={len(epo_filt)})")

    if len(allbetasnp) == 0:
        raise RuntimeError("v32: no subjects included after filtering/alignment.")

    # (n_subj, 2, n_chan, n_time)
    allbetas = np.stack(allbetasnp)

    # save betas + subject lists
    np.save(z_dir / "ols_2ndlevel_betas.npy", allbetas)
    np.save(z_dir / "included_subjects.npy", np.array(included_subjects, dtype=object))
    np.save(z_dir / "skipped_subjects.npy",  np.array(skipped_subjects, dtype=object))

    # grand-average betas (Evoked objects)
    beta_gavg = [mne.grand_average(betas[0]), mne.grand_average(betas[1])]
    np.save(z_dir / "ols_2ndlevel_betasavg.npy", np.array(beta_gavg, dtype=object), allow_pickle=True)

    # save epochs for plotting / binning (one file per regressor like v13)
    epo_save_p = mne.concatenate_epochs(all_epos[0])
    epo_save_m = mne.concatenate_epochs(all_epos[1])
    epo_save_p.save(z_dir / f"ols_2ndlevel_allepochs-epo_{pvar}.fif", overwrite=True)
    epo_save_m.save(z_dir / f"ols_2ndlevel_allepochs-epo_{mvar}.fif", overwrite=True)

    # ------------------------------------------------------------
    # second-level spatio-temporal cluster test (same output as before)
    # ------------------------------------------------------------
    connect, _ = mne.channels.find_ch_adjacency(beta_gavg[0].info, ch_type="eeg")

    # two-sided threshold
    p_thresh = param["cluster_threshold"] / 2.0
    cluster_threshold = -stats.t.ppf(p_thresh, df=allbetas.shape[0] - 1)

    tvals_list, pvals_list = [], []

    # cluster test expects (n_subj, n_time, n_chan)
    for ridx, name in enumerate(regnames):
        data_reg = allbetas[:, ridx, :, :]      # (n_subj, n_chan, n_time)
        testdata = np.swapaxes(data_reg, 2, 1)  # (n_subj, n_time, n_chan)

        tval, clusters, cluster_p_values, _ = st_clust_1s_ttest(
            testdata,
            threshold=cluster_threshold,
            adjacency=connect,
            n_permutations=param["nperms"],
            n_jobs=param["njobs"],
            buffer_size=None
        )

        # build p-map
        pmap = np.ones_like(tval)
        for c, p_val in zip(clusters, cluster_p_values):
            pmap[c] = p_val

        # (optional) per-regressor files like v13
        np.save(z_dir / f"ols_2ndlevel_tval_{name}.npy", tval)
        np.save(z_dir / f"ols_2ndlevel_pval_{name}.npy", pmap)

        tvals_list.append(tval)
        pvals_list.append(pmap)

    np.save(z_dir / "ols_2ndlevel_tvals.npy", np.stack(tvals_list))  # (2, n_time, n_chan)
    np.save(z_dir / "ols_2ndlevel_pvals.npy", np.stack(pvals_list))  # (2, n_time, n_chan)

    print(f"v32 ({v32_mode}) done. Included n={len(included_subjects)}, skipped n={len(skipped_subjects)}.")


elif version == 33:

    z_dir = Path(outpath) / "Zscoring"
    z_dir.mkdir(parents=True, exist_ok=True)

    # columns in mod_data
    pvar, mvar, rt_col = "painlevel", "moneylevel", "rt"

    regnames = ["Pain_z", "Money_is5_c"]   # names for 2nd-level loop

    # sanity checks
    for col in [pvar, mvar, rt_col, "participant", "trialsnum"]:
        if col not in mod_data.columns:
            raise RuntimeError(f"v32: mod_data missing required column: '{col}'")

    included_subjects, skipped_subjects = [], []

    all_epos = [[] for _ in range(2)]   # for saving concatenated epochs (pain + money_is5 use same epochs)
    betas    = [[] for _ in range(2)]   # beta evokeds for pain and money_is5
    allbetasnp = []                    # list of (2, n_chan, n_time)

    for pa in part:
        print(f"\n--- v32 (pain + money_is5 + RT) Processing {pa} ---")

        # behavioural rows for subject
        mod2 = mod_data.loc[mod_data["participant"] == pa].copy()
        if len(mod2) == 0:
            print(f"Skipping {pa}: no behavioural rows")
            skipped_subjects.append(pa)
            continue

        # load LONG cue-locked epochs
        epo = mne.read_epochs(
            opj(basepath, pa, "eeg", "erps_long", f"{pa}_decision_cues_long_singletrials-epo.fif"),
            preload=True
        )

        if "trialsnum" not in epo.metadata.columns:
            raise RuntimeError(
                "v32: epochs metadata missing 'trialsnum'. Did you save it in ERP creation?"
            )

        # ----------------------------
        # robust alignment by trialsnum
        # ----------------------------
        keep_epo = epo.metadata["trialsnum"].isin(mod2["trialsnum"])
        epo_filt = epo[keep_epo]
        mod2 = mod2.loc[mod2["trialsnum"].isin(epo_filt.metadata["trialsnum"])].copy()

        if len(epo_filt) == 0 or len(mod2) == 0:
            print(f"Skipping {pa}: no overlapping trialsnum")
            skipped_subjects.append(pa)
            continue

        # sort both by trialsnum to guarantee alignment
        epo_order = np.argsort(epo_filt.metadata["trialsnum"].to_numpy())
        epo_filt = epo_filt[epo_order]
        mod2 = mod2.sort_values("trialsnum").reset_index(drop=True)

        # resample
        if epo_filt.info["sfreq"] != param["testresampfreq"]:
            epo_filt = epo_filt.resample(param["testresampfreq"])

        # drop bad trials if available
        if "badtrial" in epo_filt.metadata.columns:
            good = np.where(epo_filt.metadata["badtrial"].to_numpy(int) == 0)[0]
            epo_filt = epo_filt[good]
            mod2 = mod2.iloc[good].reset_index(drop=True)

        # finite check
        vals = mod2[[pvar, mvar, rt_col]].to_numpy(float)
        keep = np.all(np.isfinite(vals), axis=1)

        if keep.sum() < 10:
            print(f"Skipping {pa}: too few finite trials ({keep.sum()})")
            skipped_subjects.append(pa)
            continue

        epo_filt = epo_filt[keep]
        mod2k = mod2.iloc[keep].reset_index(drop=True)

        # ----------------------------
        # build money_is5 and checks
        # ----------------------------
        money_is5 = (mod2k[mvar].to_numpy(float) == 5).astype(float)
        if np.std(money_is5) == 0:
            print(f"Skipping {pa}: money_is5 has zero variance (all trials same)")
            skipped_subjects.append(pa)
            continue

        if np.nanstd(mod2k[pvar]) == 0:
            print(f"Skipping {pa}: zero variance in pain")
            skipped_subjects.append(pa)
            continue

        if np.nanstd(mod2k[rt_col]) == 0:
            print(f"Skipping {pa}: zero variance in RT")
            skipped_subjects.append(pa)
            continue

        # ----------------------------
        # keep epochs with metadata for plotting / binning later
        # ----------------------------
        epo_keep_for_meta = epo_filt.copy()
        md = epo_keep_for_meta.metadata.reset_index(drop=True).copy()
        md[pvar] = mod2k[pvar].values
        md[mvar] = mod2k[mvar].values
        md[rt_col] = mod2k[rt_col].values
        md["money_is5"] = money_is5
        epo_keep_for_meta.metadata = md

        all_epos[0].append(epo_keep_for_meta.copy())
        all_epos[1].append(epo_keep_for_meta.copy())

        # ----------------------------
        # z-score predictors
        # ----------------------------
        mod2k["Intercept"] = 1.0
        mod2k["pain_z"] = stats.zscore(mod2k[pvar].to_numpy(float))
        mod2k["RT_z"]   = stats.zscore(mod2k[rt_col].to_numpy(float))

        mod2k["money_is5"] = money_is5
        mod2k["money_is5_c"] = mod2k["money_is5"] - mod2k["money_is5"].mean()

        # ----------------------------
        # z-score EEG across trials
        # ----------------------------
        scale = Scaler(scalings="mean")
        data_z = scale.fit_transform(epo_filt.get_data())

        epo_z = mne.EpochsArray(
            data_z,
            info=epo_filt.info,
            events=epo_filt.events,
            tmin=epo_filt.tmin,
            event_id=epo_filt.event_id,
            metadata=epo_filt.metadata
        )

        # ----------------------------
        # GLM: EEG ~ pain_z + money_is5_c + RT_z
        # ----------------------------
        design = mod2k[["Intercept", "pain_z", "money_is5_c", "RT_z"]]
        res = mne.stats.linear_regression(
            epo_z, design, names=["Intercept", "pain_z", "money_is5_c", "RT_z"]
        )

        beta_pain      = res["pain_z"].beta
        beta_money_is5 = res["money_is5_c"].beta

        # store for second-level
        betas[0].append(beta_pain)
        betas[1].append(beta_money_is5)
        allbetasnp.append(np.stack([beta_pain.data, beta_money_is5.data]))  # (2, n_chan, n_time)

        included_subjects.append(pa)
        print(f"Included {pa} (n_trials={len(epo_filt)})")

    if len(allbetasnp) == 0:
        raise RuntimeError("v32: no subjects included after filtering/alignment.")

    # (n_subj, 2, n_chan, n_time)
    allbetas = np.stack(allbetasnp)

    # save betas + subject lists
    np.save(z_dir / "ols_2ndlevel_betas.npy", allbetas)
    np.save(z_dir / "included_subjects.npy", np.array(included_subjects, dtype=object))
    np.save(z_dir / "skipped_subjects.npy",  np.array(skipped_subjects, dtype=object))

    # grand-average betas (Evoked objects)
    beta_gavg = [mne.grand_average(betas[0]), mne.grand_average(betas[1])]
    np.save(z_dir / "ols_2ndlevel_betasavg.npy", np.array(beta_gavg, dtype=object), allow_pickle=True)

    # save concatenated epochs for plotting / binning
    epo_save = mne.concatenate_epochs(all_epos[0])
    epo_save.save(z_dir / "ols_2ndlevel_allepochs-epo_v32.fif", overwrite=True)

    # ------------------------------------------------------------
    # second-level spatio-temporal cluster test
    # ------------------------------------------------------------
    connect, _ = mne.channels.find_ch_adjacency(beta_gavg[0].info, ch_type="eeg")

    p_thresh = param["cluster_threshold"] / 2.0
    cluster_threshold = -stats.t.ppf(p_thresh, df=allbetas.shape[0] - 1)

    tvals_list, pvals_list = [], []

    for ridx, name in enumerate(regnames):
        data_reg = allbetas[:, ridx, :, :]      # (n_subj, n_chan, n_time)
        testdata = np.swapaxes(data_reg, 2, 1)  # (n_subj, n_time, n_chan)

        tval, clusters, cluster_p_values, _ = st_clust_1s_ttest(
            testdata,
            threshold=cluster_threshold,
            adjacency=connect,
            n_permutations=param["nperms"],
            n_jobs=param["njobs"],
            buffer_size=None
        )

        pmap = np.ones_like(tval)
        for c, p_val in zip(clusters, cluster_p_values):
            pmap[c] = p_val

        np.save(z_dir / f"ols_2ndlevel_tval_{name}.npy", tval)
        np.save(z_dir / f"ols_2ndlevel_pval_{name}.npy", pmap)

        tvals_list.append(tval)
        pvals_list.append(pmap)

    np.save(z_dir / "ols_2ndlevel_tvals.npy", np.stack(tvals_list))  # (2, n_time, n_chan)
    np.save(z_dir / "ols_2ndlevel_pvals.npy", np.stack(pvals_list))  # (2, n_time, n_chan)

    print(f"v33 done. Included n={len(included_subjects)}, skipped n={len(skipped_subjects)}.")


elif version == 34:

    z_dir = Path(outpath) / "Zscoring"
    z_dir.mkdir(parents=True, exist_ok=True)

    pvar, mvar, rt_col = "painlevel", "moneylevel", "rt"
    regnames = ["Pain_z", "Money_z"]  # after RT residualization

    for col in [pvar, mvar, rt_col, "participant", "trialsnum"]:
        if col not in mod_data.columns:
            raise RuntimeError(f"v33: mod_data missing required column: '{col}'")

    included_subjects, skipped_subjects = [], []

    all_epos = [[] for _ in range(2)]
    betas    = [[] for _ in range(2)]
    allbetasnp = []   # list of (2, n_chan, n_time)

    for pa in part:
        print(f"\n--- v33 (RT-resid EEG -> pain+money) Processing {pa} ---")

        mod2 = mod_data.loc[mod_data["participant"] == pa].copy()
        if len(mod2) == 0:
            print(f"Skipping {pa}: no behavioural rows")
            skipped_subjects.append(pa)
            continue

        epo = mne.read_epochs(
            opj(basepath, pa, "eeg", "erps_long", f"{pa}_decision_cues_long_singletrials-epo.fif"),
            preload=True
        )

        if "trialsnum" not in epo.metadata.columns:
            raise RuntimeError("v33: epochs metadata missing 'trialsnum'.")

        # align by trialsnum
        keep_epo = epo.metadata["trialsnum"].isin(mod2["trialsnum"])
        epo_filt = epo[keep_epo]
        mod2 = mod2.loc[mod2["trialsnum"].isin(epo_filt.metadata["trialsnum"])].copy()

        if len(epo_filt) == 0 or len(mod2) == 0:
            print(f"Skipping {pa}: no overlapping trialsnum")
            skipped_subjects.append(pa)
            continue

        # sort both
        epo_order = np.argsort(epo_filt.metadata["trialsnum"].to_numpy())
        epo_filt = epo_filt[epo_order]
        mod2 = mod2.sort_values("trialsnum").reset_index(drop=True)

        # resample
        if epo_filt.info["sfreq"] != param["testresampfreq"]:
            epo_filt = epo_filt.resample(param["testresampfreq"])

        # drop bad trials if available
        if "badtrial" in epo_filt.metadata.columns:
            good = np.where(epo_filt.metadata["badtrial"].to_numpy(int) == 0)[0]
            epo_filt = epo_filt[good]
            mod2 = mod2.iloc[good].reset_index(drop=True)

        # finite check
        vals = mod2[[pvar, mvar, rt_col]].to_numpy(float)
        keep = np.all(np.isfinite(vals), axis=1)

        if keep.sum() < 10:
            print(f"Skipping {pa}: too few finite trials ({keep.sum()})")
            skipped_subjects.append(pa)
            continue

        epo_filt = epo_filt[keep]
        mod2k = mod2.iloc[keep].reset_index(drop=True)

        # variance checks
        if np.nanstd(mod2k[pvar]) == 0 or np.nanstd(mod2k[mvar]) == 0:
            print(f"Skipping {pa}: zero variance in pain or money")
            skipped_subjects.append(pa)
            continue
        if np.nanstd(mod2k[rt_col]) == 0:
            print(f"Skipping {pa}: zero variance in RT")
            skipped_subjects.append(pa)
            continue

        # keep metadata epochs
        epo_keep_for_meta = epo_filt.copy()
        md = epo_keep_for_meta.metadata.reset_index(drop=True).copy()
        md[pvar] = mod2k[pvar].values
        md[mvar] = mod2k[mvar].values
        md[rt_col] = mod2k[rt_col].values
        epo_keep_for_meta.metadata = md
        all_epos[0].append(epo_keep_for_meta.copy())
        all_epos[1].append(epo_keep_for_meta.copy())

        # predictors
        mod2k["Intercept"] = 1.0
        mod2k["pain_z"]  = stats.zscore(mod2k[pvar].to_numpy(float))
        mod2k["money_z"] = stats.zscore(mod2k[mvar].to_numpy(float))
        mod2k["RT_z"]    = stats.zscore(mod2k[rt_col].to_numpy(float))

        # z-score EEG across trials first (same as v32)
        scale = Scaler(scalings="mean")
        data_z = scale.fit_transform(epo_filt.get_data())

        epo_z = mne.EpochsArray(
            data_z,
            info=epo_filt.info,
            events=epo_filt.events,
            tmin=epo_filt.tmin,
            event_id=epo_filt.event_id,
            metadata=epo_filt.metadata
        )

        # ------------------------------------------------------------
        # RT residualization of EEG (trial-wise; intercept + RT_z)
        # ------------------------------------------------------------
        Z = np.c_[np.ones(len(mod2k)), mod2k["RT_z"].to_numpy(float)]  # (n_trials, 2)
        data_rtresid = residualize_epochs_data(epo_z.get_data(), Z)

        epo_rtresid = mne.EpochsArray(
            data_rtresid,
            info=epo_z.info,
            events=epo_z.events,
            tmin=epo_z.tmin,
            event_id=epo_z.event_id,
            metadata=epo_z.metadata
        )

        # now GLM WITHOUT RT: EEG_rtresid ~ pain_z + money_z
        design = mod2k[["Intercept", "pain_z", "money_z"]]
        res = mne.stats.linear_regression(
            epo_rtresid, design, names=["Intercept", "pain_z", "money_z"]
        )

        beta_pain  = res["pain_z"].beta
        beta_money = res["money_z"].beta

        betas[0].append(beta_pain)
        betas[1].append(beta_money)
        allbetasnp.append(np.stack([beta_pain.data, beta_money.data]))

        included_subjects.append(pa)
        print(f"Included {pa} (n_trials={len(epo_filt)})")

    if len(allbetasnp) == 0:
        raise RuntimeError("v33: no subjects included after filtering/alignment.")

    allbetas = np.stack(allbetasnp)  # (n_subj, 2, n_chan, n_time)

    np.save(z_dir / "ols_2ndlevel_betas.npy", allbetas)
    np.save(z_dir / "included_subjects.npy", np.array(included_subjects, dtype=object))
    np.save(z_dir / "skipped_subjects.npy",  np.array(skipped_subjects, dtype=object))

    beta_gavg = [mne.grand_average(betas[0]), mne.grand_average(betas[1])]
    np.save(z_dir / "ols_2ndlevel_betasavg.npy", np.array(beta_gavg, dtype=object), allow_pickle=True)

    epo_save = mne.concatenate_epochs(all_epos[0])
    epo_save.save(z_dir / "ols_2ndlevel_allepochs-epo_v33.fif", overwrite=True)

    # second-level cluster test
    connect, _ = mne.channels.find_ch_adjacency(beta_gavg[0].info, ch_type="eeg")

    p_thresh = param["cluster_threshold"] / 2.0
    cluster_threshold = -stats.t.ppf(p_thresh, df=allbetas.shape[0] - 1)

    tvals_list, pvals_list = [], []

    for ridx, name in enumerate(regnames):
        data_reg = allbetas[:, ridx, :, :]
        testdata = np.swapaxes(data_reg, 2, 1)

        tval, clusters, cluster_p_values, _ = st_clust_1s_ttest(
            testdata,
            threshold=cluster_threshold,
            adjacency=connect,
            n_permutations=param["nperms"],
            n_jobs=param["njobs"],
            buffer_size=None
        )

        pmap = np.ones_like(tval)
        for c, p_val in zip(clusters, cluster_p_values):
            pmap[c] = p_val

        np.save(z_dir / f"ols_2ndlevel_tval_{name}.npy", tval)
        np.save(z_dir / f"ols_2ndlevel_pval_{name}.npy", pmap)

        tvals_list.append(tval)
        pvals_list.append(pmap)

    np.save(z_dir / "ols_2ndlevel_tvals.npy", np.stack(tvals_list))
    np.save(z_dir / "ols_2ndlevel_pvals.npy", np.stack(pvals_list))

    print(f"v33 done. Included n={len(included_subjects)}, skipped n={len(skipped_subjects)}.")

#elif version == 12:
    # 9 
# TFR beta maps for sv_pain_para (cue-locked), ERP-like per band
# using FDR correction over chan x time within each band
# ----------------------------------------------------------------------


    
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

