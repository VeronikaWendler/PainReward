# Veronika Wendler
# 22.01.25
# code for the attentional drift diffusion model
# originally, I used this in summer 2024 in Quebec and was inspired by Jan WIllem De Gee's framework somewhere on his GitHub; but this version is pretty much my creation

# import libraries  
import pandas as pd
import numpy as np
import hddm
import os, sys, pickle, time
import datetime
import math
import scipy as sp
import matplotlib
matplotlib.use("Agg")                   # for backend (does not require GUI)
import os, pathlib
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import itertools
#import pp
import joblib
from IPython import embed as shell
import hddm
import kabuki
import statsmodels.formula.api as sm
from patsy import dmatrix
from joblib import Parallel, delayed
import time
import arviz as az
from joblib import Parallel, delayed
import cloudpickle, dill
cloudpickle.dump = dill.dump

# for running on the cluster
#dummy _gdbm module so “import _gdbm” never fails
import types, sys
sys.modules.setdefault('winreg', types.ModuleType('winreg'))
sys.modules.setdefault('_gdbm', types.ModuleType('_gdbm'))
# -------------------------------------------------------------------------

import dill as pickle
from copy import deepcopy   # for modfiying z to be 0.55 (like in Sebastian's Matlab)
import argparse

# warning settings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
# Plotting
# Stats 
from statsmodels.distributions.empirical_distribution import ECDF
# HDDM
from hddm.simulators.hddm_dataset_generators import simulator_h_c

from pathlib import Path

# Import my own libraries - I don't really use it anymore 
#current_directory = os.getcwd()    # we don't use this on the cluster

PROJECT_DIR = pathlib.Path(os.getenv("PROJECT_DIR", "/workspace"))

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

#from helper_functions_2 import prepare_data
#import compact_models

# for Z bias coding
from scipy.special import expit   # for inverse‑logit 


# This was an attempt to code a z-link function
def make_z_link(full_stimulus_vector):
    stim = np.asarray(full_stimulus_vector, dtype=int)

    def _link(x):
        if stim.size < len(x):
            reps = (len(x) // stim.size) + 1
            stim_aligned = np.tile(stim, reps)[:len(x)]
        else:
            stim_aligned = stim[:len(x)]

        z = np.where(stim_aligned == 0,
                     1.0 - expit(x),  
                     expit(x)) 

        if hasattr(x, "index"):            
            return pd.Series(z, index=x.index, name="z")
        return z                       

    return _link


#------------------------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------------------------

# addm regression formula
# v = β0 + β1 ⋅ (PropDwell_opt​ ⋅ V_opt​ − PropDwell_sub ⋅ V_sub) + β2 ⋅ (PropDwell_sub ⋅ V_opt​ − PropDwell_opt​ ⋅ V_sub)+ϵ
# where ß0 = intercept,
# ß1 = AttentionW,
# ß2 = InattentionW,
# ϵ = noise
# PropDwell_opt = proportion of dwell time on the option with higher expected value
# PropDwell_sub = proportion of dwell time on the option with lower expected value
# V_opt​ = value if the better option
# V_sub = value of the worse option

#data['ES_AttentionW'] = (data['PropDwell_Right'] * data['p2']) - (data['PropDwell_Left'] * data['p1'])
#data['ES_InattentionW'] = (data['PropDwell_Left'] * data['p2']) - (data['PropDwell_Right'] * data['p1'])


nr_models       = 3         # number of MCMC chains
nr_samples      = 6000      # samples per chain - do 6000 (+1000 for burn-in) but for now for a quick one we do 600
parallel        = True      # parallel
model_base_name = "garcia_replication_"
model_versions  = {
    "LE":      ["LE_1","LE_2","LE_3","LE_4"],     #"LE_5","LE_6","LE_7"
    "ES":      ["ES_1","ES_2","ES_3","ES_4","ES_5","ES_6","ES_7","ES_8","ES_9","ES_10",
                "ES_11", "ES_12", "ES_13", "ES_14", "ES_15", "ES_16", "ES_17", "ES_18", 
                "ES_19", "ES_20", "ES_21", "ES_22", "ES_23", "ES_24", "ES_25", "ES_26", 
                "ES_27", "ES_28", "ES_29", "ES_30", "ES_31", "ES_32",
                "ES_33","ES_34","ES_35","ES_36","ES_37","ES_38","ES_39","ES_40","ES_41", "ES_42", 'ES_43', 'ES_44', "ES_45", 'ES_46', 'ES_47',"ES_48",
                "ES_49", "ES_50", "ES_51", "ES_52", "ES_53", "ES_54", "ES_55", "ES_56", "ES_57", "ES_58", "ES_59", "ES_60", "ES_61", "ES_62", "ES_63", "ES_64"],
    
    "EE":      ["EE_1","EE_2","EE_3","EE_4","EE_5"],
    "ESEE":    ["ESEE_1","ESEE_2","ESEE_3","ESEE_4","ESEE_5"],
    "LEESEE":  ["LEESEE_1","LEESEE_2","LEESEE_3","LEESEE_4","LEESEE_5"],
    "ES_ZBIAS":["ES_ZBIAS_1", "ES_ZBIAS_2", "ES_ZBIAS_3", "ES_ZBIAS_4", "ES_ZBIAS_5", "ES_ZBIAS_6","ES_ZBIAS_7","ES_ZBIAS_8", "ES_ZBIAS_9","ES_ZBIAS_10", "ES_ZBIAS_11","ES_ZBIAS_12", "ES_ZBIAS_13",
                "ES_ZBIAS_14", "ES_ZBIAS_15", "ES_ZBIAS_16", "ES_ZBIAS_17", "ES_ZBIAS_18", "ES_ZBIAS_19", "ES_ZBIAS_20", "ES_ZBIAS_21", "ES_ZBIAS_22", "ES_ZBIAS_23", "ES_ZBIAS_24", "ES_ZBIAS_25", "ES_ZBIAS_26",
                "ES_ZBIAS_27"],
    "ES_quad": ["ES_quad_1","ES_quad_2"],
    "LE_RL": ["LE_RL_1","LE_RL_2"],
    "ES_VAL": ["ES_VAL_1", "ES_VAL_2", "ES_VAL_3", "ES_VAL_4", "ES_VAL_5", "ES_VAL_6", "ES_VAL_7", "ES_VAL_8","ES_VAL_9", "ES_VAL_10", "ES_VAL_11", "ES_VAL_12",  "ES_VAL_13", "ES_VAL_14", "ES_VAL_15","ES_VAL_16", "ES_VAL_17",
               "ES_VAL_18", "ES_VAL_19", "ES_VAL_20", "ES_VAL_21", "ES_VAL_22", "ES_VAL_23", "ES_VAL_24", "ES_VAL_25", "ES_VAL_26", "ES_VAL_27", "ES_VAL_28",
               "ES_VAL_29", "ES_VAL_30", "ES_VAL_31", "ES_VAL_32", "ES_VAL_33", "ES_VAL_34", "ES_VAL_35", "ES_VAL_36", "ES_VAL_37", "ES_VAL_38"],
    "For_paper": ["For_paper_1","For_paper_2","For_paper_3","For_paper_4","For_paper_5","For_paper_6","For_paper_7","For_paper_8","For_paper_9","For_paper_10","For_paper_11", "For_paper_12", "For_paper_13", "For_paper_14", "For_paper_15", "For_paper_16", "For_paper_17", "For_paper_18", "For_paper_19"],
}


PHASE_TO_SOURCE = {
    "ES_ZBIAS": "ES", 
    "ES_quad": "ES",    
    "LE_RL": "LE",
    "ES_VAL": "ES",
    "For_paper": "ES",
}

# BATCH-RUN CONTROL
PHASE_RUN_ORDER = ["For_paper"]                                         # order
SKIP_PHASES     = {"LE","ES_ZBIAS","ES","ES_VAL","EE","ES_quad", "ESEE", "LEESEE", "LE_RL"}  # ignored this phase
RUN_ALL_MODELS  = True                                           # False = just load existing fits (but loading is done in the aDDM_Garcia_LE_ES_EE.py file)

# selectivity
start_phase = "For_paper"
start_version = 16
started = False

# dir
PROJECT_DIR   = pathlib.Path(os.getenv("PROJECT_DIR", "/workspace")).resolve()
BASE_MODEL_DIR = PROJECT_DIR / "models_dir_garcia"
FIG_DIR_ROOT   = PROJECT_DIR / "figures_dir_garcia"


# reporting function
# can be seen in the cluster output
def quick_report(data, phase, version, model_name, phase_key):
    print(f"\n Phase = {phase}   Version = {version}")
    print(f"Model name          : {model_name}")
    print(f"Selected phase_key  : {phase_key}")
    print(f"N trials            : {len(data):,}")
    print(f"Participants        : {sorted(data['subj_idx'].unique())}")
    print("OVcate counts:\n", data['OVcate'].value_counts(dropna=False))

    fig, ax = plt.subplots(figsize=(6,4))
    for _, d in data.groupby('subj_idx'):
        d['rt'].hist(bins=20, histtype='step', ax=ax, alpha=.4)
    ax.set(
        title=f"RT distribution – {phase} v{version}",
        xlabel="RT (s)",
        ylabel="count"
    )
    plt.show()


# ensure directory exists
#def ensure_dir(directory):
#    if not os.path.exists(directory):
#        os.makedirs(directory)


# function to clean bits of the data that have not been cleaned yet, for instance remaining NAN's and so on
def sanitize_infdata(infdata):
    for group in infdata._groups_all:
        if hasattr(infdata, group):
            dataset = getattr(infdata, group)
            for var in dataset.data_vars:
                values = dataset[var].values
                if isinstance(values, np.ndarray) and values.dtype == "object":
                    mask = pd.isna(values)
                    if mask.any():
                        print(f"Sanitizing variable '{var}' in group '{group}' (contains pd.NA)")
                        values[mask] = np.nan
                        dataset[var].values = values
    return infdata


#####################################################################################################################################################################
# drift diffusion models
#------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------
# function that runs/defines the different versions/models of DDM regressions for the selected phase or phases

def run_model(trace_id, data, model_dir, model_name, version, phase, samples=6000, accuracy_coding=True): 
    import os
    import numpy as np
    import hddm
    from patsy import dmatrix  

    # ensure_dir(model_dir)   
    
    depends_on = {}
    
    if phase == 'LE':
        if version == 0:     # r1 # this is the 0 model with fully fixed parameters across OV levels
            v_reg = {'model': 'v ~ 1 + AttentionW + InattentionW', 'link_func': lambda x: x}
            reg_descr = [v_reg]
        elif version == 1:  # r2 fixated option weights varies by OV level 
            v_reg = {'model': 'v ~ 1 + AttentionW:C(cond) + InattentionW', 'link_func': lambda x: x}
            reg_descr = [v_reg]
        elif version == 2:  #r3  non-fixated option weights varies by OV level
            v_reg = {'model': 'v ~ 1 + AttentionW + InattentionW:C(cond)', 'link_func': lambda x: x}
            reg_descr = [v_reg]       
        elif version == 3: # r4 non-fixated options weights varies by OV level and boundary separation
            v_reg = {'model': 'v ~ 1 + AttentionW + InattentionW:C(cond)', 'link_func': lambda x: x}
            reg_descr = [v_reg]
            depends_on = {'a': 'cond'}      
        elif version == 4: # r5 r4 non-fixated options weights varies by OV level and ndt
            v_reg = {'model': 'v ~ 1 + AttentionW + InattentionW:C(cond)', 'link_func': lambda x: x}
            reg_descr = [v_reg]
            depends_on={'t': 'cond'} 
        elif version == 5: # r5 r4 non-fixated options weights varies by OV level and ndt
            v_reg = {'model': 'v ~ 1 + AttentionW:C(cond) + InattentionW', 'link_func': lambda x: x}
            reg_descr = [v_reg]
            depends_on={'a': 'cond'}  
        elif version == 6: # r5 r4 non-fixated options weights varies by OV level and ndt
            v_reg = {'model': 'v ~ 1 + AttentionW:C(cond) + InattentionW', 'link_func': lambda x: x}
            reg_descr = [v_reg]
            depends_on={'t': 'cond'}  
        else:
            raise ValueError(f"check version {version} ??")

        m = hddm.models.HDDMRegressor(data, 
                                    reg_descr,
                                    depends_on=depends_on, 
                                    p_outlier=.05, 
                                    include=['a', 't', 'v'],
                                    group_only_regressors=False,
                                    keep_regressor_trace=True)
        m.find_starting_values()
        infdata = m.sample(samples,
                   burn=1000,      #is variable
                   dbname=os.path.join(model_dir, model_name + f'_db{trace_id}'), 
                   db='pickle',
                   return_infdata=True, loglike=True, ppc=True)

        return m, infdata
    
    elif phase == 'ES':
        accuracy_coding = True
        if version == 0:    # m1 # this is the 0 model with fully fixed parameters across OV levels
            v_reg = {'model': 'v ~ 1 + AttentionW + InattentionW', 'link_func': lambda x: x}
            reg_descr = [v_reg]
        elif version == 1:  # m2 fixated option weights varies by OV level 
            v_reg = {'model': 'v ~ 1 + AttentionW:C(OVcate) + InattentionW', 'link_func': lambda x: x}
            reg_descr = [v_reg]
        elif version == 2:  #m3  non-fixated option weights varies by OV level
            v_reg = {'model': 'v ~ 1 + AttentionW + InattentionW:C(OVcate)', 'link_func': lambda x: x}
            reg_descr = [v_reg]       
        elif version == 3: # m4 non-fixated options weights varies by OV level and boundary separation
            v_reg = {'model': 'v ~ 1 + AttentionW + InattentionW:C(OVcate)', 'link_func': lambda x: x}
            reg_descr = [v_reg]
            depends_on = {'a': 'OVcate'}      
        elif version == 4:  # m5 non-fixated options weights varies by OV level and non-dec. time
            v_reg = {'model': 'v ~ 1 + AttentionW + InattentionW:C(OVcate)', 'link_func': lambda x: x}
            reg_descr = [v_reg]
            depends_on={'t': 'OVcate'} 
        elif version == 5:
            v_reg = {'model': 'v ~ 1 + AttentionW + InattentionW:C(OVcate) + gazeCI', 'link_func': lambda x: x}
            reg_descr = [v_reg]
        elif version == 6:
            v_reg = {'model': 'v ~ 1 + AttentionW + InattentionW + gazeCI:C(OVcate)', 'link_func': lambda x: x}
            reg_descr = [v_reg]
            depends_on={'a': 'OVcate'}
        elif version == 7:
            v_reg = {'model': 'v ~ 1 + AttentionW + InattentionW + gazeCI:C(OVcate)', 'link_func': lambda x: x}
            reg_descr = [v_reg]
            depends_on={'t': 'OVcate'}
        elif version == 8:
            v_reg = {'model': 'v ~ 1 + AttentionW + InattentionW + gazeSE', 'link_func': lambda x: x}
            reg_descr = [v_reg]
        elif version == 9:
            v_reg = {'model': 'v ~ 1 + AttentionW + InattentionW + gazeSE:C(OVcate)', 'link_func': lambda x: x}
            reg_descr = [v_reg]
            depends_on={'a': 'OVcate'}
        elif version == 10:
            v_reg = {'model': 'v ~ 1 + AttentionW + InattentionW + gazeSE:C(OVcate)', 'link_func': lambda x: x}
            reg_descr = [v_reg]
            depends_on={'t': 'OVcate'}
        elif version == 11:
            v_reg = {'model': 'v ~ 1 + AttentionW + InattentionW:C(OVcate)', 'link_func': lambda x: x}
            a_reg = {'model': 'a ~ 1 + C(OVcate)', 'link_func': lambda x: x}
            reg_descr = [v_reg, a_reg]
        elif version == 12:
            v_reg = {'model': 'v ~ 1 + AttentionW + InattentionW:C(OVcate)', 'link_func': lambda x: x}
            t_reg = {'model': 't ~ 1 + C(OVcate)', 'link_func': lambda x: x}
            reg_descr = [v_reg, t_reg]
        elif version == 13:  # m5 non-fixated options weights varies by OV level and non-dec. time
            v_reg = {'model': 'v ~ 1 + AttentionW + InattentionW:C(OVcate)', 'link_func': lambda x: x}
            reg_descr = [v_reg]
            depends_on={'z': 'stimulus'} 
        elif version == 14:  # m5 non-fixated options weights varies by OV level and non-dec. time
            v_reg = {'model': 'v ~ 1 + AttentionW + InattentionW:C(OVcate)', 'link_func': lambda x: x}
            reg_descr = [v_reg]
            depends_on={'z': 'chose_left'} 
        elif version == 15:  # m5 non-fixated options weights varies by OV level and non-dec. time
            v_reg = {'model': 'v ~ 1 + AttentionW + InattentionW:C(OVcate)', 'link_func': lambda x: x}
            reg_descr = [v_reg]
            depends_on={'t': 'stimulus'} 
        elif version == 16:  # m5 non-fixated options weights varies by OV level and non-dec. time
            v_reg = {'model': 'v ~ 1 + AttentionW + InattentionW:C(OVcate)', 'link_func': lambda x: x}
            reg_descr = [v_reg]
            depends_on={'t': 'chose_left'}
        elif version == 17:
            stim_vec = data["stimulus"].values         
            z_link   = make_z_link(stim_vec)            
            v_reg = {'model': 'v ~ 1 + AttentionW + InattentionW', 'link_func': lambda x: x}
            z_reg = {'model': 'z ~ 1', 'link_func': z_link}
            reg_descr = [v_reg, z_reg]
        elif version == 18:
            stim_vec = data["stimulus"].values        
            z_link   = make_z_link(stim_vec) 
            v_reg = {'model': 'v ~ 1 + AttentionW + InattentionW:C(OVcate)', 'link_func': lambda x: x}
            z_reg = {'model': 'z ~ 1', 'link_func': z_link}
            reg_descr = [v_reg, z_reg]
        elif version == 19:
            stim_vec = data["stimulus"].values         
            z_link   = make_z_link(stim_vec) 
            v_reg = {'model': 'v ~ 1 + AttentionW + InattentionW:C(OVcate)', 'link_func': lambda x: x}
            z_reg = {'model': 'z ~ 1 + C(OVcate)', 'link_func': z_link}
            reg_descr = [v_reg, z_reg]
        elif version == 20:
            stim_vec = data["stimulus"].values       
            z_link   = make_z_link(stim_vec) 
            v_reg = {'model': 'v ~ 1 + AttentionW + InattentionW:C(OVcate)', 'link_func': lambda x: x}
            z_reg = {'model': 'z ~ 1 + C(OVcate)', 'link_func': z_link}
            t_reg = {'model': 't ~ 1 + C(OVcate)', 'link_func': lambda x: x}
            reg_descr = [v_reg, z_reg, t_reg]
        elif version == 21:  # m5 non-fixated options weights varies by OV level and non-dec. time
            v_reg = {'model': 'v ~ 1 + AttentionW + InattentionW:C(OVcate)', 'link_func': lambda x: x}
            z_reg = {'model': 'z ~ FirstFix_Left', 'link_func': lambda x: x}
            reg_descr = [v_reg, z_reg]
            
        elif version == 22:  # m5 non-fixated options weights varies by OV level and non-dec. time
            v_reg = {'model': 'v ~ 1 + AttentionW + InattentionW:C(OVcate)', 'link_func': lambda x: x}
            a_reg = {'model': 'a ~ 1 + C(OVcate)', 'link_func': lambda x: x}
            t_reg = {'model': 't ~ 1 + C(OVcate)', 'link_func': lambda x: x}
            reg_descr = [v_reg,a_reg,t_reg]
            depends_on={'z': 'stimulus'} 
            
        elif version == 23:  # m5 non-fixated options weights varies by OV level and non-dec. time
            v_reg = {'model': 'v ~ 1 + AttentionW + InattentionW:C(OVcate)', 'link_func': lambda x: x}
            a_reg = {'model': 'a ~ 1 + C(OVcate)', 'link_func': lambda x: x}
            t_reg = {'model': 't ~ 1 + C(OVcate)', 'link_func': lambda x: x}
            reg_descr = [v_reg,a_reg,t_reg]
            depends_on={'z': 'stimulus'} 
        elif version == 24:  # m5 non-fixated options weights varies by OV level and non-dec. time
            v_reg = {'model': 'v ~ 1 + AttentionW + InattentionW:C(OVcate)', 'link_func': lambda x: x}
            t_reg = {'model': 't ~ 1 + AttentionW + InattentionW:C(OVcate)', 'link_func': lambda x: x}
            reg_descr = [v_reg, t_reg]
        elif version == 25:  # m5 non-fixated options weights varies by OV level and non-dec. time
            v_reg = {'model': 'v ~ 1 + AttentionW + InattentionW:C(OVcate)', 'link_func': lambda x: x}
            reg_descr = [v_reg]
            depends_on={'t': 'OVcate',
                        'a': 'OVcate'} 
        elif version == 26:
            v_reg = {'model': 'v ~ 1 + val_diff + DwellPropAdvantage + gaze_quad', 'link_func': lambda x: x}
            a_reg = {'model': 'a ~ 1 + abs_DwellPropAdv:C(OVcate)', 'link_func': lambda x: x }
            reg_descr = [v_reg, a_reg]    
        elif version == 27:
            v_reg = {'model': 'v ~ 1 + gaze_quad', 'link_func': lambda x: x}
            reg_descr = [v_reg]
        elif version == 28:
            v_reg = {'model': 'v ~ 1 + val_diff + val_bal_int', 'link_func': lambda x: x}
            reg_descr = [v_reg]
        elif version == 29:
            v_reg = {'model': 'v ~ 1 + z_AttentionW:C(OVcate) + z_IAW_chart + z_IAW_image', 'link_func': lambda x: x}
            reg_descr = [v_reg]
            depends_on={'a': 'OVcate'} 
        elif version == 30:
            v_reg = {'model': 'v ~ 1 + z_AttentionW + z_IAW_chart:C(OVcate) + z_IAW_image:C(OVcate)', 'link_func': lambda x: x}
            reg_descr = [v_reg]
            depends_on={'a': 'OVcate'} 
        elif version == 31:
            v_reg = {'model': 'v ~ 1 + z_val_diff + z_DwellPropAdvantage:C(OVcate)', 'link_func': lambda x: x}
            reg_descr = [v_reg]
            depends_on={'a': 'OVcate'}  
        elif version == 32:
            v_reg = {'model': 'v ~ 1 + z_val_diff + z_DwellPropAdvantage + z_gaze_quad:C(OVcate)', 'link_func': lambda x: x}
            reg_descr = [v_reg]
            depends_on={'a': 'OVcate'}
            
            
        elif version == 33:
            v_reg = {'model': 'v ~ 1 + z_AttentionW + z_IAW_chart:C(OVcate) + z_IAW_image:C(OVcate)', 'link_func': lambda x: x}
            a_reg = {'model': 'a ~ 1 + OVcate', 'link_func': lambda x: x}
            reg_descr = [v_reg, a_reg]
        elif version == 34:
            v_reg = {'model': 'v ~ 1 + z_AttentionW:C(OVcate) + z_IAW_chart + z_IAW_image', 'link_func': lambda x: x}
            a_reg = {'model': 'a ~ 1 + OVcate', 'link_func': lambda x: x}
            reg_descr = [v_reg, a_reg]
        elif version == 35:
            v_reg = {'model': 'v ~ 1 + z_AttentionW:C(OVcate) + z_IAW_chart + z_IAW_image', 'link_func': lambda x: x}
            t_reg = {'model': 't ~ 1 + OVcate', 'link_func': lambda x: x}
            reg_descr = [v_reg, t_reg]
        elif version == 36:
            v_reg = {'model': 'v ~ 1 + z_val_diff_corr + z_w_dv + FirstFix_Left', 'link_func': lambda x: x}
            a_reg = {'model': 'a ~ 1 + z_absDPAC','link_func': lambda x: x}
            reg_descr = [v_reg, a_reg]
        elif version == 37:
            v_reg = {'model': 'v ~ 1 + z_val_diff_corr + z_w_dv:C(OVcate) + FirstFix_Left', 'link_func': lambda x: x}
            a_reg = {'model': 'a ~ 1 + z_absDPAC','link_func': lambda x: x}
            reg_descr = [v_reg, a_reg]
        elif version == 38:
            v_reg = {'model': 'v ~ 1 + z_val_diff_corr + z_w_dv:C(OVcate) + FirstFix_Left', 'link_func': lambda x: x}
            t_reg = {'model': 't ~ 1 + z_absDPAC','link_func': lambda x: x}
            reg_descr = [v_reg, t_reg]
        elif version == 39:
            v_reg = {'model': 'v ~ 1 + z_val_diff_corr + z_w_dv', 'link_func': lambda x: x}
            a_reg = {'model': 'a ~ 1 + z_absDPAC','link_func': lambda x: x}
            reg_descr = [v_reg, a_reg]
        elif version == 40:
            v_reg = {'model': 'v ~ 1 + z_val_diff + z_val_bal_int:C(OVcate)', 'link_func': lambda x: x}
            reg_descr = [v_reg]
        elif version == 41:
            v_reg = {'model': 'v ~ 1 + z_w + z_w:C(OVcate)','link_func': lambda x: x }
            a_reg = {'model': 'a ~ 1 + OVcate', 'link_func': lambda x: x}
            reg_descr = [v_reg, a_reg]
        elif version == 42:
            v_reg = {'model': 'v ~ 1 + z_DwellPropAdvantageCorrect + z_balance:C(OVcate)','link_func': lambda x: x }  
            a_reg = {'model': 'a ~ 1 + z_balance:C(OVcate)', 'link_func': lambda x: x}
            reg_descr = [v_reg, a_reg]
        elif version == 43:
            v_reg = {'model': 'v ~ 1 + z_AW_bal:C(OVcate) + z_IAW_bal', 'link_func': lambda x: x}
            a_reg = {'model': 'a ~ 1 + z_balance:C(OVcate)', 'link_func': lambda x: x}
            reg_descr = [v_reg, a_reg]
        elif version == 44:
            v_reg = {'model': 'v ~ 1 + AttentionW_E + AttentionW_S + InattentionW_E + InattentionW_S', 'link_func': lambda x: x}
            reg_descr = [v_reg]
            depends_on = {'a': 'OVcate'}  
        elif version == 45:
            v_reg = {'model': 'v ~ 1 + AttentionW_E:C(OVcate) + AttentionW_S:C(OVcate) + InattentionW_E + InattentionW_S', 'link_func': lambda x: x}
            a_reg = {'model': 'a ~ 1 + OVcate', 'link_func': lambda x: x}
            reg_descr = [v_reg, a_reg]
        elif version == 46:
            v_reg = {'model': 'v ~ 1 + AttentionW_E + AttentionW_S + InattentionW_E:C(OVcate) + InattentionW_S:C(OVcate)', 'link_func': lambda x: x}
            a_reg = {'model': 'a ~ 1 + OVcate', 'link_func': lambda x: x}
            reg_descr = [v_reg, a_reg]
        elif version == 47:
            v_reg = {'model': 'v ~ 1 + z_AttentionW_E + z_AttentionW_S + z_InattentionW_E + z_InattentionW_S', 'link_func': lambda x: x}
            reg_descr = [v_reg]
            depends_on = {'a': 'OVcate'}  
        elif version == 48:
            v_reg = {'model': 'v ~ 1 + z_AttentionW_E + z_AttentionW_S + z_InattentionW_E + z_InattentionW_S', 'link_func': lambda x: x}
            reg_descr = [v_reg]
            depends_on = {'t': 'OVcate'}  
        elif version == 49:
            v_reg = {'model': 'v ~ 1 + AttentionW_E + AttentionW_S + InattentionW_E + InattentionW_S', 'link_func': lambda x: x}
            reg_descr = [v_reg]
            depends_on = {'a': 'OVcate', 't': 'OVcate'}
        # modle with the attnetional gaze penalty as in Cavanagh et al. (2011)
        # gazeCI:C(OVcate)  gazeCI
        elif version == 50:
            v_reg = {'model': 'v ~ 1 + AttentionW_E + AttentionW_S + InattentionW_E + InattentionW_S + gazeCI', 'link_func': lambda x: x}
            reg_descr = [v_reg]
        elif version == 51:
            v_reg = {'model': 'v ~ 1 + AttentionW_E + AttentionW_S + InattentionW_E + InattentionW_S + gazeCI:C(OVcate)', 'link_func': lambda x: x}
            reg_descr = [v_reg]
        # model with the direct gaze penalty by me: balance
        elif version == 52:
            v_reg = {'model': 'v ~ 1 + AttentionW_E + AttentionW_S + InattentionW_E + InattentionW_S + balance', 'link_func': lambda x: x}
            reg_descr = [v_reg]
            
        elif version == 53:
            v_reg = {'model': 'v ~ 1 + AttentionW_E + AttentionW_S + InattentionW_E + InattentionW_S', 'link_func': lambda x: x}
            reg_descr = [v_reg]
            
        elif version == 54:
            v_reg = {'model': 'v ~ 1 + AttentionW_E + AttentionW_S + InattentionW_E:C(OVcate) + InattentionW_S:C(OVcate)', 'link_func': lambda x: x}
            reg_descr = [v_reg]
            depends_on = {'t':'OVcate'}
        elif version == 55:
            print("Check in R if also pie>0.5 has any effect when looking at ES_AttentionW etc...")
            v_reg = {'model': 'v ~ 1 + ES_AttentionW + ES_InattentionW', 'link_func': lambda x: x}
            reg_descr = [v_reg]
        elif version == 56:
            print("Check in R if also pie>0.5 has any effect when looking at ES_AttentionW etc...")
            v_reg = {'model': 'v ~ 1 + ES_AttentionW + ES_InattentionW:C(OVcate)', 'link_func': lambda x: x}
            reg_descr = [v_reg]
        
        elif version == 57:
            v_reg = {'model': 'v ~ 1 + AttentionW_E + AttentionW_S:C(OVcate) + InattentionW_E:C(OVcate) + InattentionW_S', 'link_func': lambda x: x}
            reg_descr = [v_reg]    
            
        elif version == 58:
            v_reg = {'model': 'v ~ 1 + AttentionW_early + InattentionW_early + AttentionW_late + InattentionW_late', 'link_func': lambda x: x}
            reg_descr = [v_reg]
        elif version == 59:
            v_reg = {'model': 'v ~ 1 + AttentionW_early + InattentionW_early + AttentionW_late:C(OVcate) + InattentionW_late:C(OVcate)', 'link_func': lambda x: x}
            reg_descr = [v_reg]
        elif version == 60:
            v_reg = {'model': 'v ~ 1 + AttentionW:C(OVcate) + IAW_chart + IAW_image', 'link_func': lambda x: x}
            reg_descr = [v_reg]
            depends_on = {'a': 'OVcate'}
        elif version == 61:
            v_reg = {'model': 'v ~ 1 + AttentionW_E + AttentionW_S + InattentionW_E + InattentionW_S', 'link_func': lambda x: x}
            reg_descr = [v_reg]
            depends_on = {'a': 'OVcate'} 
        elif version == 62:
            v_reg = {'model': 'v ~ 1 + AttentionW_early + InattentionW_early + AttentionW_late + InattentionW_late', 'link_func': lambda x: x}
            reg_descr = [v_reg] 
            depends_on = {'a': 'trial_type'}
        
        else:
            raise ValueError(f"Is this version illegal ?? It feels illegal...")   
        
        
        include_list = ['a', 't', 'v']
        has_z_reg = any(
            reg['model'].strip().split('~',1)[0].strip() == 'z'
            for reg in reg_descr
        )

        # …or if z is in the depends_on dict #
        if has_z_reg or 'z' in depends_on:
            include_list.append('z')
        print(f"[run_model] version={version}  include={include_list}")

        m = hddm.models.HDDMRegressor(
            data,
            reg_descr,
            p_outlier=.05,
            include=include_list,  
            group_only_regressors=False,
            keep_regressor_trace=True
        )
        
        m.find_starting_values()
        infdata = m.sample(samples,
                   burn=200,
                   dbname=os.path.join(model_dir, model_name + f'_db{trace_id}'), 
                   db='pickle',
                   return_infdata=True, loglike=True, ppc=True)

        return m, infdata
    
    
    
        # #  Fix z at 0.55 #
        # cfg = deepcopy(hddm.model_config.model_config['ddm_hddm_base'])
        # idx_z = cfg['params'].index('z')        # position 2 in ['v','a','z','t'] according to hddm source code but not sure if this works
        # cfg['params_default'][idx_z] = 0.55     # slight bias towards E
        
        # # SANITY‐CHECK 
        # assert cfg['params'][idx_z] == 'z'
        # assert cfg['params_default'][idx_z] == 0.55, \
        #     f"z default not 0.55 but {cfg['params_default'][idx_z]}"

        # # build the model
        # m = hddm.models.HDDMRegressor(
        #     data,
        #     reg_descr,
        #     depends_on=depends_on,
        #     p_outlier=.05,
        #     include=['a', 't', 'v'],     #  z is not in include becuase not a free param
        #     group_only_regressors=False,
        #     keep_regressor_trace=True,
        #     model_config=cfg
        # )

        # print("\n[ DEBUG] model_config['params']       =", m.model_config['params'])
        # print("[ DEBUG] model_config['params_default'] =", m.model_config['params_default'])
        # zi = m.model_config['params'].index('z')
        # print(f"[ DEBUG] default for 'z' = {m.model_config['params_default'][zi]}\n")  
        
        # print("[DEBUG] sampling nodes in m.nodes_db:\n",
        #       [n for n in m.nodes_db.index if n.split('_')[0] in ['a','t','v','z']])


        # m.find_starting_values()
        # infdata = m.sample(
        #     samples,
        #     burn=200,
        #     dbname=os.path.join(model_dir, model_name + f'_db{trace_id}'),
        #     db='pickle',
        #     return_infdata=True,
        #     loglike=True,
        #     ppc=True
        # )

        # # final check that z never got sampled
        # assert "z" not in infdata.posterior.data_vars, \
        #     "ERROR: 'z' appeared in the posterior!"
        # print("[DEBUG] z absent from posterior - confirmed fixed.")

        # return m, infdata
    

     
        # #  Fix z at 0.55 #
        # from copy import deepcopy
        # cfg = deepcopy(hddm.model_config.model_config['ddm_hddm_base'])
        # idx_z = cfg['params'].index('z')        # position 2 in ['v','a','z','t'] according to hddm source code but not sure if this works
        # cfg['params_default'][idx_z] = 0.55     # slight bias towards E
        
        # # SANITY‐CHECK 
        # assert cfg['params'][idx_z] == 'z'
        # assert cfg['params_default'][idx_z] == 0.55, \
        #     f"z default not 0.55 but {cfg['params_default'][idx_z]}"

        # # build the model
        # m = hddm.models.HDDMRegressor(
        #     data,
        #     reg_descr,
        #     depends_on=depends_on,
        #     p_outlier=.05,
        #     include=['a', 't', 'v'],     #  z is not in include becuase not a free param
        #     group_only_regressors=False,
        #     keep_regressor_trace=True,
        #     model_config=cfg
        # )

        # print("\n[ZBIAS DEBUG] model_config['params']       =", m.model_config['params'])
        # print("[ZBIAS DEBUG] model_config['params_default'] =", m.model_config['params_default'])
        # zi = m.model_config['params'].index('z')
        # print(f"[ZBIAS DEBUG] default for 'z' = {m.model_config['params_default'][zi]}\n")  
        
        # print("[ZBIAS DEBUG] sampling nodes in m.nodes_db:\n",
        #       [n for n in m.nodes_db.index if n.split('_')[0] in ['a','t','v','z']])


        # m.find_starting_values()
        # infdata = m.sample(
        #     samples,
        #     burn=100,
        #     dbname=os.path.join(model_dir, model_name + f'_db{trace_id}'),
        #     db='pickle',
        #     return_infdata=True,
        #     loglike=True,
        #     ppc=True
        # )

        # # final check that z never got sampled
        # assert "z" not in infdata.posterior.data_vars, \
        #     "ERROR: 'z' appeared in the posterior!"
        # print("[DEBUG] z absent from posterior - confirmed fixed.")

        # return m, infdata
    
    elif phase == 'EE':
        accuracy_coding = True
        if version == 0:     # m1 # this is the 0 model with fully fixed parameters across OV levels
            v_reg = {'model': 'v ~ 1 + AttentionW + InattentionW', 'link_func': lambda x: x}
            reg_descr = [v_reg]
        elif version == 1:  # m2 attentional weight parameter (fixated) option weights varies by OV level 
            v_reg = {'model': 'v ~ 1 + AttentionW:C(OVcate) + InattentionW', 'link_func': lambda x: x}
            reg_descr = [v_reg]
        elif version == 2:  #m3  non-fixated option weights varies by OV level
            v_reg = {'model': 'v ~ 1 + AttentionW + InattentionW:C(OVcate)', 'link_func': lambda x: x}
            reg_descr = [v_reg]       
        elif version == 3: # m4 non-fixated options weights varies by OV level and boundary separation
            v_reg = {'model': 'v ~ 1 + AttentionW + InattentionW:C(OVcate)', 'link_func': lambda x: x}
            reg_descr = [v_reg]
            depends_on = {'a': 'OVcate'}      
        elif version == 4: # m5 non-fixated options weights varies by OV level and non-dec. time
            v_reg = {'model': 'v ~ 1 + AttentionW + InattentionW:C(OVcate)', 'link_func': lambda x: x}
            reg_descr = [v_reg]
            depends_on={'t': 'OVcate'}      
        
        m = hddm.models.HDDMRegressor(data, 
                                    reg_descr,
                                    depends_on=depends_on, 
                                    p_outlier=.05, 
                                    include=['a', 't', 'v'],
                                    group_only_regressors=False,
                                    keep_regressor_trace=True
                                    )
        m.find_starting_values()
        infdata = m.sample(samples,
                   burn=1000,
                   dbname=os.path.join(model_dir, model_name + f'_db{trace_id}'), 
                   db='pickle',
                   return_infdata=True, loglike=True, ppc=True)

        return m, infdata

    elif phase == 'ESEE':  # combined model for ES + EE (furhter confimation that theta varies by phase (not just OV))
        accuracy_coding = True
        if version == 0:  # baseline model with fixed parameters across phases (ES, EE)
            v_reg = {'model': 'v ~ 1 + AttentionW + InattentionW', 'link_func': lambda x: x}
            reg_descr = [v_reg]
        elif version == 1:  # drift rate varies by phase (ES vs. EE)
            v_reg = {'model': 'v ~ 1 + AttentionW:C(phase) + InattentionW', 'link_func': lambda x: x}
            reg_descr = [v_reg]
        elif version == 2:  # non-fix option weights vary by phase
            v_reg = {'model': 'v ~ 1 + AttentionW + InattentionW:C(phase)', 'link_func': lambda x: x}
            reg_descr = [v_reg]
        elif version == 3:  # decision threshold (a) varies by phase + InattentionW:C(phase)
            v_reg = {'model': 'v ~ 1 + AttentionW + InattentionW:C(phase)', 'link_func': lambda x: x}
            reg_descr = [v_reg]
            depends_on = {'a': 'phase'}
        elif version == 4:  # non-decision time varies by phase + InattentionW:C(phase)
            v_reg = {'model': 'v ~ 1 + AttentionW + InattentionW:C(phase)', 'link_func': lambda x: x}
            reg_descr = [v_reg]
            depends_on = {'t': 'phase'}
        else:
            raise ValueError(f"check version {version} ??")
     
        
        m = hddm.models.HDDMRegressor(data, 
                                    reg_descr,
                                    depends_on=depends_on, 
                                    p_outlier=.05, 
                                    include=['a', 't', 'v'],
                                    group_only_regressors=False,
                                    keep_regressor_trace=True
                                    )
        m.find_starting_values()
        infdata = m.sample(samples,
                   burn=1000,
                   dbname=os.path.join(model_dir, model_name + f'_db{trace_id}'), 
                   db='pickle',
                   return_infdata=True, loglike=True, ppc=True)

        return m, infdata
    
    elif phase == 'LEESEE':  # Combined model for LE + ES + EE
        accuracy_coding = True
        if version == 0:  # Baseline model with fixed parameters across phases
            v_reg = {'model': 'v ~ 1 + AttentionW + InattentionW', 'link_func': lambda x: x}
            reg_descr = [v_reg]
        elif version == 1:  # Drift rate varies by phase (LE vs ES vs. EE)
            v_reg = {'model': 'v ~ 1 + AttentionW:C(phase) + InattentionW', 'link_func': lambda x: x}
            reg_descr = [v_reg]
        elif version == 2:  # Non-fixated option weights vary by phase
            v_reg = {'model': 'v ~ 1 + AttentionW + InattentionW:C(phase)', 'link_func': lambda x: x}
            reg_descr = [v_reg]
        elif version == 3:  # Boundary separation varies by phase
            v_reg = {'model': 'v ~ 1 + AttentionW + InattentionW:C(phase)', 'link_func': lambda x: x}
            reg_descr = [v_reg]
            depends_on = {'a': 'phase'}
        elif version == 4:  # Non-decision time varies by phase
            v_reg = {'model': 'v ~ 1 + AttentionW + InattentionW:C(phase)', 'link_func': lambda x: x}
            reg_descr = [v_reg]
            depends_on = {'t': 'phase'}
        else:
            raise ValueError(f"Invalid version {version}")
     
        
        m = hddm.models.HDDMRegressor(data, 
                                    reg_descr,
                                    depends_on=depends_on, 
                                    p_outlier=.05, 
                                    include=['a', 't', 'v'],
                                    group_only_regressors=False,
                                    keep_regressor_trace=True
                                    )
        m.find_starting_values()
        infdata = m.sample(samples,
                   burn=1000,
                   dbname=os.path.join(model_dir, model_name + f'_db{trace_id}'), 
                   db='pickle',
                   return_infdata=True, loglike=True, ppc=True)

        return m, infdata
    
    elif phase == 'ES_ZBIAS':
        if version == 0:   
            v_reg = {'model': 'v ~ 1 + ES_AttentionW + ES_InattentionW', 'link_func': lambda x: x}
            reg_descr = [v_reg]
        elif version == 1:  
            v_reg = {'model': 'v ~ 1 + ES_AttentionW:C(OVcate) + ES_InattentionW', 'link_func': lambda x: x}
            reg_descr = [v_reg]
        elif version == 2: 
            v_reg = {'model': 'v ~ 1 + ES_AttentionW + ES_InattentionW:C(OVcate)', 'link_func': lambda x: x}
            reg_descr = [v_reg]       
        elif version == 3: 
            v_reg = {'model': 'v ~ 1 + ES_AttentionW + ES_InattentionW:C(OVcate)', 'link_func': lambda x: x}
            reg_descr = [v_reg]
            depends_on = {'a': 'OVcate'}      
        elif version == 4: 
            v_reg = {'model': 'v ~ 1 + ES_AttentionW + ES_InattentionW:C(OVcate)', 'link_func': lambda x: x}
            reg_descr = [v_reg]
            depends_on={'t': 'OVcate'} 
        elif version == 5:
            v_reg = {'model': 'v ~ 1 + AttentionW_E + AttentionW_S + InattentionW_E + InattentionW_S', 'link_func': lambda x: x}
            reg_descr = [v_reg]
        elif version == 6:
            v_reg = {'model': 'v ~ 1 + AttentionW_E + AttentionW_S + InattentionW_E:C(OVcate) + InattentionW_S:C(OVcate)', 'link_func': lambda x: x}
            reg_descr = [v_reg]
        elif version == 7:
            v_reg = {'model': 'v ~ 1 + z_AttentionW_E + z_AttentionW_S + z_InattentionW_E + z_InattentionW_S', 'link_func': lambda x: x}
            reg_descr = [v_reg]
        elif version == 8:
            v_reg = {'model': 'v ~ 1 + AttentionW_E + AttentionW_S + InattentionW_E + InattentionW_S', 'link_func': lambda x: x}
            reg_descr = [v_reg]
        elif version == 9:
            v_reg = {'model': 'v ~ 1 + AttentionW_E + AttentionW_S + InattentionW_E + InattentionW_S + gazeSE', 'link_func': lambda x: x}
            reg_descr = [v_reg]
        elif version == 10:
            v_reg = {'model': 'v ~ 1 + AttentionW_E + AttentionW_S + InattentionW_E + InattentionW_S + FirstFix_Left', 'link_func': lambda x: x}
            reg_descr = [v_reg]
        elif version == 11:
            v_reg = {'model': 'v ~ 1 + ES_AttentionW_early + ES_InattentionW_early + ES_AttentionW_late + ES_InattentionW_late', 'link_func': lambda x: x}
            reg_descr = [v_reg]
        elif version == 12:
            v_reg = {'model': 'v ~ 1 + ES_AttentionW_early + ES_InattentionW_early + ES_AttentionW_late:C(OVcate) + ES_InattentionW_late:C(OVcate)', 'link_func': lambda x: x}
            reg_descr = [v_reg]
        elif version == 13:
            v_reg = {'model': 'v ~ 1 + ES_AttentionW_early + ES_InattentionW_early + ES_AttentionW_late + ES_InattentionW_late', 'link_func': lambda x: x}
            reg_descr = [v_reg]
            depends_on = {'a': 'trial_type'}
        elif version == 14:
            v_reg = {'model': 'v ~ 1 + ES_AttentionW_early + ES_InattentionW_early + ES_AttentionW_late + ES_InattentionW_late', 'link_func': lambda x: x}
            reg_descr = [v_reg]
            depends_on = {'a': 'OVcate'}
        elif version == 15:
            v_reg = {'model': 'v ~ 1 + ES_AttentionW_early + ES_InattentionW_early + ES_AttentionW_late + ES_InattentionW_late', 'link_func': lambda x: x}
            reg_descr = [v_reg]

        elif version == 16:
            v_reg = {'model': 'v ~ 1 + ES_AttentionW_early + ES_InattentionW_early + ES_AttentionW_late + ES_InattentionW_late', 'link_func': lambda x: x}
            reg_descr = [v_reg]
            depends_on = {'a': 'trial_type', 't': 'trial_type'}
        elif version == 17:
            v_reg = {'model': 'v ~ 1 + ES_AttentionW_early + ES_InattentionW_early + ES_AttentionW_late + ES_InattentionW_late', 'link_func': lambda x: x}
            reg_descr = [v_reg]
            depends_on = {'t': 'trial_type'}
        elif version == 18:
            v_reg = {'model': 'v ~ 1 + ES_AttentionW_early:C(OVcate) + ES_InattentionW_early:C(OVcate) + ES_AttentionW_late + ES_InattentionW_late', 'link_func': lambda x: x}
            reg_descr = [v_reg]
            depends_on = {'a': 'trial_type'}
        elif version == 19:
            v_reg = {'model': 'v ~ 1 + ES_AttentionW_early:C(OVcate) + ES_InattentionW_early + ES_AttentionW_late:C(OVcate) + ES_InattentionW_late', 'link_func': lambda x: x}
            reg_descr = [v_reg]
            depends_on = {'a': 'trial_type'}
        elif version == 20:
            v_reg = {'model': 'v ~ 1 + ES_AttentionW_early + ES_InattentionW_early:C(OVcate) + ES_AttentionW_late + ES_InattentionW_late:C(OVcate)', 'link_func': lambda x: x}
            reg_descr = [v_reg]
            depends_on = {'a': 'trial_type'}
        elif version == 21:
            v_reg = {'model': 'v ~ 1 + ES_AttentionW_early:C(OVcate) + ES_InattentionW_early:C(OVcate) + ES_AttentionW_late + ES_InattentionW_late', 'link_func': lambda x: x}
            reg_descr = [v_reg]
            depends_on = {'a': 'trial_type', 't': 'trial_type'}
        elif version == 22:
            v_reg = {'model': 'v ~ 1 + ES_AttentionW_early:C(OVcate) + ES_InattentionW_early + ES_AttentionW_late:C(OVcate) + ES_InattentionW_late', 'link_func': lambda x: x}
            reg_descr = [v_reg]
            depends_on = {'a': 'trial_type', 't': 'trial_type'}
        elif version == 23:
            v_reg = {'model': 'v ~ 1 + ES_AttentionW_early + ES_InattentionW_early:C(OVcate) + ES_AttentionW_late + ES_InattentionW_late:C(OVcate)', 'link_func': lambda x: x}
            reg_descr = [v_reg]
            depends_on = {'a': 'trial_type', 't': 'trial_type'}
        elif version == 24: ## z = 0.55
            v_reg = {'model': 'v ~ 1 + ES_AttentionW + ES_InattentionW', 'link_func': lambda x: x}
            reg_descr = [v_reg]
        elif version == 25: # here we include z in include list
            v_reg = {'model': 'v ~ 1 + ES_AttentionW + ES_InattentionW', 'link_func': lambda x: x}
            reg_descr = [v_reg]
        elif version == 26: # z is 0.5 and not a free parameter, not included in the include_list
            v_reg = {'model': 'v ~ 1 + ES_AttentionW + ES_InattentionW', 'link_func': lambda x: x}
            reg_descr = [v_reg]
            
        include_list = ['a', 't', 'v']
        has_z_reg = any(
            reg['model'].strip().split('~',1)[0].strip() == 'z'
            for reg in reg_descr
        )

        # …or if z is in the depends_on dict #
        if has_z_reg or 'z' in depends_on:
            include_list.append('z')
        print(f"[run_model] version={version}  include={include_list}")

        m = hddm.models.HDDMRegressor(
            data,
            reg_descr,
            p_outlier=.05,
            include=include_list,  
            group_only_regressors=False,
            keep_regressor_trace=True
        )
        
        m.find_starting_values()
        infdata = m.sample(samples,
                   burn=200,
                   dbname=os.path.join(model_dir, model_name + f'_db{trace_id}'), 
                   db='pickle',
                   return_infdata=True, loglike=True, ppc=True)

        return m, infdata
    
               
            
        # #  Fix z
        # from copy import deepcopy
        # cfg = deepcopy(hddm.model_config.model_config['ddm_hddm_base'])
        # idx_z = cfg['params'].index('z')        # position 2 in ['v','a','z','t'] according to hddm source code but not sure if this works
        # cfg['params_default'][idx_z] = 0.55     #  - changing it again to E being upper (chose_left) - if S shoudl be upper, then set (chose_right)
        
        # # SANITY‐CHECK 
        # assert cfg['params'][idx_z] == 'z'
        # assert cfg['params_default'][idx_z] == 0.55, \
        #     f"z default not 0.55 but {cfg['params_default'][idx_z]}"

        # # build the model
        # m = hddm.models.HDDMRegressor(
        #     data,
        #     reg_descr,
        #     depends_on=depends_on,
        #     p_outlier=.05,
        #     include=['a', 't', 'v'],     #  z is not in include as not a free param
        #     group_only_regressors=False,
        #     keep_regressor_trace=True,
        #     model_config=cfg
        # )

        # print("\n[ZBIAS DEBUG] model_config['params']       =", m.model_config['params'])
        # print("[ZBIAS DEBUG] model_config['params_default'] =", m.model_config['params_default'])
        # zi = m.model_config['params'].index('z')
        # print(f"[ZBIAS DEBUG] default for 'z' = {m.model_config['params_default'][zi]}\n")  
        
        # print("[ZBIAS DEBUG] sampling nodes in m.nodes_db:\n",
        #       [n for n in m.nodes_db.index if n.split('_')[0] in ['a','t','v','z']])


        # m.find_starting_values()
        # infdata = m.sample(
        #     samples,
        #     burn=200,
        #     dbname=os.path.join(model_dir, model_name + f'_db{trace_id}'),
        #     db='pickle',
        #     return_infdata=True,
        #     loglike=True,
        #     ppc=True
        # )

        # # final check that z never got sampled
        # assert "z" not in infdata.posterior.data_vars, \
        #     "ERROR: 'z' appeared in the posterior!"
        # print("[DEBUG] z absent from posterior - confirmed fixed.")

        # return m, infdata
    
    
    elif phase == 'ES_quad':
        if version == 0:
            v_reg = {'model':'v ~ 1 + DTA + DTA2', 'link_func': lambda x:x}
            reg_descr = [v_reg]
        elif version == 1:  # m5 non-fixated options weights varies by OV level and non-dec. time
            v_reg = {'model': 'v ~ 1 + ES_AttentionW + ES_InattentionW:C(OVcate)', 'link_func': lambda x: x}
            reg_descr = [v_reg]
            depends_on={'z': 'stimulus'} 

        m = hddm.models.HDDMRegressor(data, 
                                    reg_descr,
                                    depends_on=depends_on, 
                                    p_outlier=.05, 
                                    include=['a', 't', 'v', 'z'],
                                    group_only_regressors=False,
                                    keep_regressor_trace=True
                                    )
        m.find_starting_values()
        infdata = m.sample(samples,
                   burn=200,
                   dbname=os.path.join(model_dir, model_name + f'_db{trace_id}'), 
                   db='pickle',
                   return_infdata=True, loglike=True, ppc=True)

        return m, infdata
    
    
    elif phase == "ES_VAL":
        depends_on = {}
        if version == 0:   
            v_reg = {'model': 'v ~ Value_diff', 'link_func': lambda x: x}
            reg_descr = [v_reg]
        elif version == 1:  
            v_reg = {'model': 'v ~ 1 + V_E + V_S', 'link_func': lambda x: x}
            reg_descr = [v_reg]
        elif version == 2:
            v_reg = {'model': 'v ~ 1 + V_E + V_S + PropDwell_Left + PropDwell_Right', 'link_func': lambda x: x}
        elif version == 3:
            v_reg = {'model': 'v ~ 1 + Value_diff + DTA', 'link_func': lambda x: x}
            reg_descr = [v_reg]
        elif version == 4:
            v_reg = {'model': 'v ~ 1 + PropDwell_Left + PropDwell_Right', 'link_func': lambda x: x}
            reg_descr = [v_reg]
        elif version == 5:
            v_reg = {'model': 'v ~ 0 + Value_diff', 'link_func': lambda x: x}
            reg_descr = [v_reg]
        
        elif version == 6:
            v_reg = {'model': 'v ~ 0 + ES_AttentionW + ES_InattentionW', 'link_func': lambda x: x}
            reg_descr = [v_reg]
        # ATTENTION    
        # changing the boundary here - s is upper bound
        elif version == 7:
            v_reg = {'model': 'v ~ 0 + ES_AttentionW + ES_InattentionW', 'link_func': lambda x: x}
            reg_descr = [v_reg]
        elif version == 8:
            v_reg = {'model': 'v ~ 0 + V_E + V_S + PropDwell_Left + PropDwell_Right', 'link_func': lambda x: x}
            reg_descr = [v_reg]
        elif version == 9:
            v_reg = {'model': 'v ~ 0 + Value_diff + DTA', 'link_func': lambda x: x}
            reg_descr = [v_reg]
        elif version == 10:
            v_reg = {'model': 'v ~ 0 + PropDwell_Left + PropDwell_Right', 'link_func': lambda x: x}
            reg_descr = [v_reg]
        elif version == 11:
            v_reg = {'model': 'v ~ 0 + Value_diff', 'link_func': lambda x: x}
            reg_descr = [v_reg]
        elif version == 12:
            v_reg = {'model': 'v ~ 0 + AttentionW_E + AttentionW_S + InattentionW_E + InattentionW_S', 'link_func': lambda x: x}
            reg_descr = [v_reg]
        
        # E = upper boundary 
        elif version == 13:   
            v_reg = {'model': 'v ~ Value_diff', 'link_func': lambda x: x}
            reg_descr = [v_reg]
        elif version == 14:
            v_reg = {'model': 'v ~ 1 + ES_AttentionW + ES_InattentionW', 'link_func': lambda x: x}
            reg_descr = [v_reg]
            
        elif version == 15:
            v_reg = {'model': 'v ~ 0 + Value_diff', 'link_func': lambda x: x}
            reg_descr = [v_reg]
        elif version == 16:
            v_reg = {'model': 'v ~ 0 + ES_AttentionW + ES_InattentionW', 'link_func': lambda x: x}
            reg_descr = [v_reg]
        
        # E is upper boudnary, z is included
        elif version == 17:
            v_reg = {'model': 'v ~ 1 + AttentionW_E + AttentionW_S + InattentionW_E + InattentionW_S', 'link_func': lambda x: x}
            reg_descr = [v_reg]
        elif version == 18:
            v_reg = {'model': 'v ~ 1 + ES_AttentionW_E + ES_AttentionW_S + ES_InattentionW_E + ES_InattentionW_S', 'link_func': lambda x: x}
            reg_descr = [v_reg]
        elif version == 19:
            v_reg = {'model': 'v ~ 1 + ES_AttentionW + ES_InattentionW:C(OVcate)', 'link_func': lambda x: x}
            reg_descr = [v_reg]
            depends_on = {'a': 'OVcate'} 
        elif version == 20:
            v_reg = {'model': 'v ~ 1 + ES_AttentionW + ES_InattentionW:C(OVcate)', 'link_func': lambda x: x}
            reg_descr = [v_reg]
            depends_on = {'t': 'OVcate'}    
        elif version == 21:
            v_reg = {'model': 'v ~ 0 + AttentionW_E + AttentionW_S + InattentionW_E + InattentionW_S', 'link_func': lambda x: x}
            reg_descr = [v_reg]
        elif version == 22:
            v_reg = {'model': 'v ~ 0 + ES_AttentionW_E + ES_AttentionW_S + ES_InattentionW_E + ES_InattentionW_S', 'link_func': lambda x: x}
            reg_descr = [v_reg]
        elif version == 23:
            v_reg = {'model': 'v ~ 0 + ES_AttentionW + ES_InattentionW:C(OVcate)', 'link_func': lambda x: x}
            reg_descr = [v_reg]
            depends_on = {'a': 'OVcate'} 
        elif version == 24:
            v_reg = {'model': 'v ~ 0 + ES_AttentionW + ES_InattentionW:C(OVcate)', 'link_func': lambda x: x}
            reg_descr = [v_reg]
            depends_on = {'t': 'OVcate'} 
        # S is upper boundary
        elif version == 25:
            v_reg = {'model': 'v ~ 0 + ES_AttentionW + ES_InattentionW', 'link_func': lambda x: x}
            reg_descr = [v_reg]
            depends_on = {'a': 'OVcate'} 
        elif version == 26:
            v_reg = {'model': 'v ~ 1 + ES_AttentionW + ES_InattentionW', 'link_func': lambda x: x}
            reg_descr = [v_reg]
        # E is upper bound and z fixed at 0.52 (throw out of include list)
        elif version == 27:
            v_reg = {'model': 'v ~ 0 + ES_AttentionW + ES_InattentionW', 'link_func': lambda x: x}
            reg_descr = [v_reg]
        # S is upper bound and z fixed at 0.48 (throw out of include list)
        elif version == 28:
            v_reg = {'model': 'v ~ 0 + ES_AttentionW + ES_InattentionW', 'link_func': lambda x: x}
            reg_descr = [v_reg]
            
        # S is upper boundary and we're calculating seperate inattent weights
        elif version == 29:
            v_reg = {'model': 'v ~ 1 + ES_AttentionW + ES_InattentionW_E - ES_InattentionW_S', 'link_func': lambda x: x}
            reg_descr = [v_reg]
            
        elif version == 30:
            v_reg = {'model': 'v ~ 0 + ES_AttentionW + ES_InattentionW_E - ES_InattentionW_S', 'link_func': lambda x: x}
            reg_descr = [v_reg]
            
        # E was upper boundary
        elif version == 31:
            v_reg = {'model': 'v ~ 1 + ES_AttentionW + ES_InattentionW_E - ES_InattentionW_S', 'link_func': lambda x: x}
            reg_descr = [v_reg]
            
        elif version == 32:
            v_reg = {'model': 'v ~ 0 + ES_AttentionW + ES_InattentionW_E - ES_InattentionW_S', 'link_func': lambda x: x}
            reg_descr = [v_reg]
        # S is upper boundary
        elif version == 33:
            v_reg = {'model': 'v ~ 1 + ES_AttentionW + ES_InattentionW_E + ES_InattentionW_S', 'link_func': lambda x: x}
            reg_descr = [v_reg]
        elif version == 34:
            v_reg = {'model': 'v ~ 0 + ES_AttentionW + ES_InattentionW_E + ES_InattentionW_S', 'link_func': lambda x: x}
            reg_descr = [v_reg]
        elif version == 35:
            v_reg = {'model': 'v ~ 0 + ES_AttentionW + ES_InattentionW_E + ES_InattentionW_S', 'link_func': lambda x: x}
            reg_descr = [v_reg]
            depends_on = {'a': 'OVcate'} 
        elif version == 36:
            v_reg = {'model': 'v ~ 0 + ES_AttentionW + ES_InattentionW_E:C(OVcate) + ES_InattentionW_S:C(OVcate)', 'link_func': lambda x: x}
            reg_descr = [v_reg]
            depends_on = {'a': 'OVcate'} 
        
        
        #re-runnign the most successful models with a sampling nr that makes reviewers happy
    elif phase ==  "For_paper":
        depends_on = {}
        # 0 model:
        if version == 0:
            # jsut start sampling

            m = hddm.models.HDDM(data, 
                                    p_outlier=.05, 
                                    include=['a', 't', 'v', 'z'],   #'z'
                                    depends_on=depends_on,
                                    )
            m.find_starting_values()
            infdata = m.sample(samples,
                               burn=1000,
                               dbname=os.path.join(model_dir, model_name + f'_db{trace_id}'), 
                               db='pickle',
                               return_infdata=True, loglike=True, ppc=True)
            return m, infdata
        # aDDM + SP
        elif version == 1:
            v_reg = {'model': 'v ~ 0 + ES_AttentionW + ES_InattentionW', 'link_func': lambda x: x}
            reg_descr = [v_reg]
        elif version == 2:
            v_reg = {'model': 'v ~ 0 + ES_AttentionW + ES_InattentionW', 'link_func': lambda x: x}
            reg_descr = [v_reg]
            depends_on = {'a': 'OVcate'} 
        elif version == 3:
            v_reg = {'model': 'v ~ 1 + ES_AttentionW + ES_InattentionW', 'link_func': lambda x: x}
            reg_descr = [v_reg]
        elif version == 4:
            v_reg = {'model': 'v ~ 1 + ES_AttentionW + ES_InattentionW', 'link_func': lambda x: x}
            reg_descr = [v_reg]
            depends_on = {'a': 'OVcate'} 
        # ES - dual inattention models - do what Sebastian said regarding the recoding/cahnging the sgin of the ES_InattnetionW_S param

        elif version == 5:
            v_reg = {'model': 'v ~ 0 + ES_AttentionW + ES_InattentionW_E + ES_InattentionW_S', 'link_func': lambda x: x}
            reg_descr = [v_reg]
        elif version == 6:
            v_reg = {'model': 'v ~ 0 + ES_AttentionW + ES_InattentionW_E + ES_InattentionW_S', 'link_func': lambda x: x}
            reg_descr = [v_reg]
            depends_on = {'a': 'OVcate'} 
        elif version == 7:
            v_reg = {'model': 'v ~ 1 + ES_AttentionW + ES_InattentionW_E + ES_InattentionW_S', 'link_func': lambda x: x}
            reg_descr = [v_reg]
        elif version == 8:
            v_reg = {'model': 'v ~ 1 + ES_AttentionW + ES_InattentionW_E + ES_InattentionW_S', 'link_func': lambda x: x}
            reg_descr = [v_reg]
            depends_on = {'a': 'OVcate'} 
        elif version == 9:
            v_reg = {'model': 'v ~ 1 + ES_AttentionW_E + ES_AttentionW_S + ES_InattentionW', 'link_func': lambda x: x}
            reg_descr = [v_reg]
            
        # again with more samples (10 thousand each)
        elif version == 10:
            v_reg = {'model': 'v ~ 0 + ES_AttentionW_E + ES_AttentionW_S + ES_InattentionW', 'link_func': lambda x: x}
            reg_descr = [v_reg]
        
        
        elif version == 11:
            v_reg = {'model': 'v ~ 0 + V_E + V_S + PropDwell_Left + PropDwell_Right', 'link_func': lambda x: x}
            reg_descr = [v_reg]
            
        elif version == 12:
            v_reg = {'model': 'v ~ 0 + ES_AttentionW_dwell + ES_InattentionW_E_dwell + ES_InattentionW_S_dwell', 'link_func': lambda x: x}
            reg_descr = [v_reg]
        # z is inlcuded - DDM + SP    - 6000 samples
        elif version == 13:
            v_reg = {'model': 'v ~ 0 + Value_diff', 'link_func': lambda x: x}
            reg_descr = [v_reg]
        # z not included
        elif version == 14:
            v_reg = {'model': 'v ~ 0 + Value_diff', 'link_func': lambda x: x}
            reg_descr = [v_reg]
        # z not included
        elif version == 15:
            v_reg = {'model': 'v ~ 0 + ES_AttentionW + ES_InattentionW', 'link_func': lambda x: x}
            reg_descr = [v_reg]
            
        
        # z included
        elif version == 16:
            v_reg = {'model': 'v ~ 0 + ES_AttentionW + ES_IAW_chart + ES_IAW_image', 'link_func': lambda x: x}
            reg_descr = [v_reg]
        elif version == 17:
            v_reg = {'model': 'v ~ 0 + ES_AttentionW + ES_IAW_chart + ES_IAW_image', 'link_func': lambda x: x}
            reg_descr = [v_reg]
            depends_on={'a':'OVcate'}
        else:
            raise ValueError(f"Invalid version {version}")
        
        m = hddm.models.HDDMRegressor(data, 
                                    reg_descr,
                                    p_outlier=.05, 
                                    include=['a', 't', 'v', 'z'],   #'z'
                                    depends_on=depends_on,
                                    group_only_regressors=False,
                                    keep_regressor_trace=True
                                    )
        m.find_starting_values()
        infdata = m.sample(samples,
                   burn=1000,
                   dbname=os.path.join(model_dir, model_name + f'_db{trace_id}'), 
                   db='pickle',
                   return_infdata=True, loglike=True, ppc=True)

        return m, infdata
        
        
        
        # #  Fix z
        # from copy import deepcopy
        # cfg = deepcopy(hddm.model_config.model_config['ddm_hddm_base'])
        # idx_z = cfg['params'].index('z')        # position 2 in ['v','a','z','t'] according to hddm source code but not sure if this works
        # cfg['params_default'][idx_z] = 0.52     #  - changing it again to E being upper (chose_left) - if S shoudl be upper, then set (chose_right)
        
        # # SANITY‐CHECK 
        # assert cfg['params'][idx_z] == 'z'
        # assert cfg['params_default'][idx_z] == 0.52, \
        #     f"z default not 0.55 but {cfg['params_default'][idx_z]}"

        # # build the model
        # m = hddm.models.HDDMRegressor(
        #     data,
        #     reg_descr,
        #     depends_on=depends_on,
        #     p_outlier=.05,
        #     include=['a', 't', 'v'],     #  z is not in include as not a free param
        #     group_only_regressors=False,
        #     keep_regressor_trace=True,
        #     model_config=cfg
        # )

        # print("\n[ZBIAS DEBUG] model_config['params']       =", m.model_config['params'])
        # print("[ZBIAS DEBUG] model_config['params_default'] =", m.model_config['params_default'])
        # zi = m.model_config['params'].index('z')
        # print(f"[ZBIAS DEBUG] default for 'z' = {m.model_config['params_default'][zi]}\n")  
        
        # print("[ZBIAS DEBUG] sampling nodes in m.nodes_db:\n",
        #       [n for n in m.nodes_db.index if n.split('_')[0] in ['a','t','v','z']])


        # m.find_starting_values()
        # infdata = m.sample(
        #     samples,
        #     burn=200,
        #     dbname=os.path.join(model_dir, model_name + f'_db{trace_id}'),
        #     db='pickle',
        #     return_infdata=True,
        #     loglike=True,
        #     ppc=True
        # )

        # # final check that z never got sampled
        # assert "z" not in infdata.posterior.data_vars, \
        #     "ERROR: 'z' appeared in the posterior!"
        # print("[DEBUG] z absent from posterior - confirmed fixed.")

        # return m, infdata
        
    
    
    
    elif phase == 'LE_RL':
        if version == 0:
            m = hddm.models.HDDMrl(data, include=['a', 't', 'v', 'alpha'])
            m.find_starting_values()
            infdata = m.sample(samples,
                               burn=300,
                               dbname=os.path.join(model_dir, model_name + f'_db{trace_id}'), 
                               db='pickle',
                               return_infdata=True, loglike=True, ppc=False)

            return m, infdata
        else:
            raise ValueError(f"Invalid version {version}")
    
     
###############################################################################################################    
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Main function for running/loading models

#
### ONLY FOR RUNIING RL MODELS ##-------------------------------------------------------------------------------------------
# def run_and_save(trace_id, data, model_dir, model_name, version, phase, samples):
#     # 1) instantiate + sample exactly as you do in run_model()
#     model, infdata = run_model(
#         trace_id=trace_id,
#         data=data,
#         model_dir=model_dir,
#         model_name=model_name,
#         version=version,
#         phase=phase,
#         samples=samples,
#     )
#     # 2) save *inside* the worker
#     fname = f"{model_name}_{trace_id}"
#     model.save(os.path.join(model_dir, fname + ".hddm"))
#     with open(os.path.join(model_dir, fname + ".pkl"), "wb") as f:
#         pickle.dump(model, f)
#     infdata = sanitize_infdata(infdata)
#     az.to_netcdf(infdata, os.path.join(model_dir, fname + ".nc"))

#     # 3) return something small
#     return fname
#----------------------------------------------------------------------------------------------------------------

import dill as pickle  # to create the pkl object

def drift_diffusion_hddm(data, 
                         samples=6000,
                         n_jobs=5,
                         run=True,
                         parallel=True,
                         model_name='model',
                         model_dir='.', 
                         accuracy_coding=True,
                         version=None,
                         phase=None):

    if run:
        if parallel:
            start_time = time.time()
            results = Parallel(n_jobs=n_jobs)(
                delayed(run_model)(
                    trace_id=trace_id,
                    data=data,
                    model_dir=model_dir,
                    model_name=model_name,
                    version=version,
                    phase=phase,
                    samples=samples,
                    accuracy_coding=accuracy_coding
                    )
                for trace_id in range(n_jobs)
            )
            print("Time elapsed:", time.time() - start_time, "s")
            
           
            for i in range(n_jobs):
                model, infdata = results[i]
                model.save(os.path.join(model_dir, f"{model_name}_{i}.hddm"))
                
                with open(os.path.join(model_dir, f"{model_name}_{i}.pkl"), "wb") as f:
                    pickle.dump(model, f)
                infdata = sanitize_infdata(infdata)  # clean before saving
                az.to_netcdf(infdata, os.path.join(model_dir, f"{model_name}_{i}.nc"))


        else: 
            model, infdata = run_model(1,
                                       data,
                                       model_dir,
                                       model_name,
                                       version, 
                                       samples,
                                       accuracy_coding 
                                       )
            model.save(os.path.join(model_dir, model_name + ".hddm"))

            with open(os.path.join(model_dir, f"{model_name}.pkl"), "wb") as f:
                pickle.dump(model, f)
            infdata = sanitize_infdata(infdata)
            az.to_netcdf(infdata, os.path.join(model_dir, f"{model_name}.nc"))

    else:
        print('Loading existing models')
        models = [hddm.load(os.path.join(model_dir, f"{model_name}_{i}.hddm")) for i in range(n_jobs)]
        return models
#---------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------
# for the RL models (if used)
import dill as pickle

from joblib import Parallel, delayed

def drift_diffusion_hddmRL(
    data,
    samples=2000,
    n_jobs=3,
    run=True,
    parallel=True,
    model_name='model',
    model_dir='.',
    version=None,
    phase=None,
):
    if not run:
        print('Loading existing RL models')
        return [
            hddm.load(os.path.join(model_dir, f"{model_name}_{i}.hddm"))
            for i in range(n_jobs)
        ]

    start_time = time.time()
    if parallel:
        results = Parallel(n_jobs=n_jobs, backend="threading")(
            delayed(run_model)(
                trace_id=i,
                data=data,
                model_dir=model_dir,
                model_name=model_name,
                version=version,
                phase=phase,
                samples=samples,
            )
            for i in range(n_jobs)
        )
        for i, (model, infdata) in enumerate(results):
            fname = f"{model_name}_{i}"
            # model.save(os.path.join(model_dir, fname + ".hddm"))
            # with open(os.path.join(model_dir, fname + ".pkl"), "wb") as f:
            #     pickle.dump(model, f)
            infdata = sanitize_infdata(infdata)
            az.to_netcdf(infdata, os.path.join(model_dir, fname + ".nc"))

        print(f"RL chains finished and saved: {[f'{model_name}_{i}' for i in range(n_jobs)]}")
    else:
        # single‐threaded sampling + save
        model, infdata = run_model(
            trace_id=0,
            data=data,
            model_dir=model_dir,
            model_name=model_name,
            version=version,
            phase=phase,
            samples=samples,
        )
        fname = f"{model_name}_0"
        model.save(os.path.join(model_dir, fname + ".hddm"))
        with open(os.path.join(model_dir, fname + ".pkl"), "wb") as f:
            pickle.dump(model, f)
        infdata = sanitize_infdata(infdata)
        az.to_netcdf(infdata, os.path.join(model_dir, fname + ".nc"))
        print(f"RL chain finished and saved: {fname}")

    print("Time elapsed:", time.time() - start_time, "s")
    
#########################################################################################################################################################
#---------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------
# Analyzing the models

def analyze_model(models, fig_dir, nr_models, version, phase):
    # 'sns.set_theme(style='darkgrid', font='sans-serif', font_scale=0.5)
    # # combine the 3 modles with kabuki utils
    # combined_model = kabuki.utils.concat_models(models)'
    
    print(f"Analyzing {len(models)} models for {phase}, version {version}")
    print(f"Saving figures to: {fig_dir}")

    sns.set_theme(style='darkgrid', font='sans-serif', font_scale=0.5)

    if not models or models[0] is None:
        print("ERROR: Models are empty or invalid.")
        return

    # Try combining models
    try:
        combined_model = kabuki.utils.concat_models(models)
        print("Models combined successfully.")
    except Exception as e:
        print(f"Error combining models: {e}")
        return
    
    # names parameters 
    
    if phase == 'LE':
        if version == 0:
            params_of_interest = [
                'a',
                't',
                'alpha',
                'v_Intercept',
                'v_AttentionW',
                'v_InattentionW',
                ]
            params_of_interest_s = [
                'a_subj', 
                't_subj', 
                'alpha_subj',
                'v_Intercept_subj',
                'v_AttentionW_subj',
                'v_InattentionW_subj', 
                ]
            titles = [
                'Boundary sep.',
                'Non-dec. time',
                'Learning rate'
                'Intercept drift rate',
                'Drift AttentionW',
                'Drift InattentionW',
                'starting point'
            ]
        elif version == 1:
            params_of_interest = [
            'a',
            't', 
            'v_Intercept',
            'v_AttentionW:C(OVcate)[low]',
            'v_AttentionW:C(OVcate)[medium]',
            'v_AttentionW:C(OVcate)[high]',
            'v_InattentionW'
            ]
            params_of_interest_s = [
            'a_subj',
            't_subj', 
            'v_Intercept_subj',
            'v_AttentionW:C(OVcate)[low]_subj', 
            'v_AttentionW:C(OVcate)[medium]_subj',
            'v_AttentionW:C(OVcate)[high]_subj',
            'v_InattentionW_subj'
            
            ]
            titles = [
            'Boundary sep.',
            'Non-dec. time',
            'Intercept drift rate',
            'Drift AttentionW:C(OVcate)[low]',
            'Drift AttentionW:C(OVcate)[medium]', 
            'Drift AttentionW:C(OVcate)[high]',
            'Drift InattentionW', 
            ]
        elif version == 2:
            params_of_interest = [
            'a',
            't',
            'v_Intercept',
            'v_AttentionW',
            'v_InattentionW:C(OVcate)[low]',
            'v_InattentionW:C(OVcate)[medium]', 
            'v_InattentionW:C(OVcate)[high]',
            ]
            params_of_interest_s = [
            'a_subj', 
            't_subj', 
            'v_Intercept_subj',
            'v_AttentionW_subj',
            'v_InattentionW:C(OVcate)[low]_subj',
            'v_InattentionW:C(OVcate)[medium]_subj', 
            'v_InattentionW:C(OVcate)[high]_subj',
            ]
            titles = [
            'Boundary sep.',
            'Non-dec. time',
            'Intercept drift rate',
            'Drift AttentionW',
            'Drift InattentionW:C(OVcate)[low]',
            'Drift InattentionW:C(OVcate)[medium]',
            'Drift InattentionW:C(OVcate)[high]'
            
            ]
        elif version == 3:
            params_of_interest = [
            'a(low)',
            'a(medium)',
            'a(high)',
            't',
            'v_Intercept',
            'v_AttentionW',
            'v_InattentionW:C(OVcate)[low]',
            'v_InattentionW:C(OVcate)[medium]',
            'v_InattentionW:C(OVcate)[high]',
            ]
            params_of_interest_s = [
            'a(low)_subj',
            'a(medium)_subj',
            'a(high)_subj',
            't_subj',
            'v_Intercept_subj',
            'v_AttentionW_subj',
            'v_InattentionW:C(OVcate)[low]_subj',
            'v_InattentionW:C(OVcate)[medium]_subj',
            'v_InattentionW:C(OVcate)[high]_subj',
            ]
            titles = [
            'Boundary sep. (low OVcate)',
            'Boundary sep. (medium OVcate)',
            'Boundary sep. (high OVcate)',
            'Non-dec. time',
            'Intercept drift rate',
            'Drift AttentionW',
            'Drift InattentionW:C(OVcate)[low]',
            'Drift InattentionW:C(OVcate)[medium]',
            'Drift InattentionW:C(OVcate)[high]',
            ]
        elif version == 4:
            params_of_interest = [
            'a',
            't(low)',
            't(medium)',
            't(high)',
            'v_Intercept',
            'v_AttentionW',
            'v_InattentionW:C(OVcate)[low]',
            'v_InattentionW:C(OVcate)[medium]',
            'v_InattentionW:C(OVcate)[high]',
            ]
            params_of_interest_s = [
            'a_subj',
            't(low)_subj',
            't(medium)_subj',
            't(high)_subj',
            'v_Intercept_subj',
            'v_AttentionW_subj',
            'v_InattentionW:C(OVcate)[low]_subj',
            'v_InattentionW:C(OVcate)[medium]_subj',
            'v_InattentionW:C(OVcate)[high]_subj',
            ]
            titles = [
            'Boundary sep.',
            'Non-dec. time (low OVcate)',
            'Non-dec. time (medium OVcate)',
            'Non-dec. time (high OVcate)',
            'Intercept drift rate',
            'Drift AttentionW',
            'Drift InattentionW:C(OVcate)[low]',
            'Drift InattentionW:C(OVcate)[medium]',
            'Drift InattentionW:C(OVcate)[high]',
            ]
            
        elif version == 5:
            params_of_interest = [
            'a',
            't(low)',
            't(medium)',
            't(high)',
            'v_Intercept',
            'v_AttentionW',
            'v_InattentionW:C(OVcate)[low]',
            'v_InattentionW:C(OVcate)[medium]',
            'v_InattentionW:C(OVcate)[high]',
            ]
            params_of_interest_s = [
            'a_subj',
            't(low)_subj',
            't(medium)_subj',
            't(high)_subj',
            'v_Intercept_subj',
            'v_AttentionW_subj',
            'v_InattentionW:C(OVcate)[low]_subj',
            'v_InattentionW:C(OVcate)[medium]_subj',
            'v_InattentionW:C(OVcate)[high]_subj',
            ]
            titles = [
            'Boundary sep.',
            'Non-dec. time (low OVcate)',
            'Non-dec. time (medium OVcate)',
            'Non-dec. time (high OVcate)',
            'Intercept drift rate',
            'Drift AttentionW',
            'Drift InattentionW:C(OVcate)[low]',
            'Drift InattentionW:C(OVcate)[medium]',
            'Drift InattentionW:C(OVcate)[high]',
            ]
            
    if phase == 'ES':
        if version == 0:
            params_of_interest = [
                'a',
                't',
                'v_Intercept',
                'v_AttentionW',
                'v_InattentionW',
                ]
            params_of_interest_s = [
                'a_subj', 
                't_subj', 
                'v_Intercept_subj',
                'v_AttentionW_subj',
                'v_InattentionW_subj', 
                ]
            titles = [
            'Boundary sep.',
            'Non-dec. time',
            'Intercept drift rate',
            'Drift AttentionW',
            'Drift InattentionW',
            ]
        elif version == 1:
            params_of_interest = [
            'a',
            't', 
            'v_Intercept',
            'v_AttentionW:C(OVcate)[low]',
            'v_AttentionW:C(OVcate)[medium]',
            'v_AttentionW:C(OVcate)[high]',
            'v_InattentionW',
            ]
            params_of_interest_s = [
            'a_subj',
            't_subj', 
            'v_Intercept_subj',
            'v_AttentionW:C(OVcate)[low]_subj', 
            'v_AttentionW:C(OVcate)[medium]_subj',
            'v_AttentionW:C(OVcate)[high]_subj',
            'v_InattentionW_subj', 
            ]
            titles = [
            'Boundary sep.',
            'Non-dec. time',
            'Intercept drift rate',
            'Drift AttentionW:C(OVcate)[low]',
            'Drift AttentionW:C(OVcate)[medium]', 
            'Drift AttentionW:C(OVcate)[high]',
            'Drift InattentionW', 
            ]
        elif version == 2:
            params_of_interest = [
            'a',
            't',
            'v_Intercept',
            'v_AttentionW',
            'v_InattentionW:C(OVcate)[low]',
            'v_InattentionW:C(OVcate)[medium]', 
            'v_InattentionW:C(OVcate)[high]',
            ]
            params_of_interest_s = [
            'a_subj', 
            't_subj', 
            'v_Intercept_subj',
            'v_AttentionW_subj',
            'v_InattentionW:C(OVcate)[low]_subj',
            'v_InattentionW:C(OVcate)[medium]_subj', 
            'v_InattentionW:C(OVcate)[high]_subj',
            ]
            titles = [
            'Boundary sep.',
            'Non-dec. time',
            'Intercept drift rate',
            'Drift AttentionW',
            'Drift InattentionW:C(OVcate)[low]',
            'Drift InattentionW:C(OVcate)[medium]',
            'Drift InattentionW:C(OVcate)[high]',
            ]
        elif version == 3:
            params_of_interest = [
            'a(low)',
            'a(medium)',
            'a(high)',
            't',
            'v_Intercept',
            'v_AttentionW',
            'v_InattentionW:C(OVcate)[low]',
            'v_InattentionW:C(OVcate)[medium]',
            'v_InattentionW:C(OVcate)[high]',
            ]
            params_of_interest_s = [
            'a(low)_subj',
            'a(medium)_subj',
            'a(high)_subj',
            't_subj',
            'v_Intercept_subj',
            'v_AttentionW_subj',
            'v_InattentionW:C(OVcate)[low]_subj',
            'v_InattentionW:C(OVcate)[medium]_subj',
            'v_InattentionW:C(OVcate)[high]_subj',
            ]
            titles = [
            'Boundary sep. (low OVcate)',
            'Boundary sep. (medium OVcate)',
            'Boundary sep. (high OVcate)',
            'Non-dec. time',
            'Intercept drift rate',
            'Drift AttentionW',
            'Drift InattentionW:C(OVcate)[low]',
            'Drift InattentionW:C(OVcate)[medium]',
            'Drift InattentionW:C(OVcate)[high]',
            ]
        elif version == 4:
            params_of_interest = [
            'a',
            't(low)',
            't(medium)',
            't(high)',
            'v_Intercept',
            'v_AttentionW',
            'v_InattentionW:C(OVcate)[low]',
            'v_InattentionW:C(OVcate)[medium]',
            'v_InattentionW:C(OVcate)[high]',
            ]
            params_of_interest_s = [
            'a_subj',
            't(low)_subj',
            't(medium)_subj',
            't(high)_subj',
            'v_Intercept_subj',
            'v_AttentionW_subj',
            'v_InattentionW:C(OVcate)[low]_subj',
            'v_InattentionW:C(OVcate)[medium]_subj',
            'v_InattentionW:C(OVcate)[high]_subj',
            ]
            titles = [
            'Boundary sep.',
            'Non-dec. time (low OVcate)',
            'Non-dec. time (medium OVcate)',
            'Non-dec. time (high OVcate)',
            'Intercept drift rate',
            'Drift AttentionW',
            'Drift InattentionW:C(OVcate)[low]',
            'Drift InattentionW:C(OVcate)[medium]',
            'Drift InattentionW:C(OVcate)[high]',
            ]
    if phase == 'EE':
        if version == 0:
            params_of_interest = [
                'a',
                't',
                'v_Intercept',
                'v_AttentionW',
                'v_InattentionW',
                ]
            params_of_interest_s = [
                'a_subj', 
                't_subj', 
                'v_Intercept_subj',
                'v_AttentionW_subj',
                'v_InattentionW_subj', 
                ]
            titles = [
            'Boundary sep.',
            'Non-dec. time',
            'Intercept drift rate',
            'Drift AttentionW',
            'Drift InattentionW',
            ]
        elif version == 1:
            params_of_interest = [
            'a',
            't', 
            'v_Intercept',
            'v_AttentionW:C(OVcate)[low]',
            'v_AttentionW:C(OVcate)[medium]',
            'v_AttentionW:C(OVcate)[high]',
            'v_InattentionW',
            ]
            params_of_interest_s = [
            'a_subj',
            't_subj', 
            'v_Intercept_subj',
            'v_AttentionW:C(OVcate)[low]_subj', 
            'v_AttentionW:C(OVcate)[medium]_subj',
            'v_AttentionW:C(OVcate)[high]_subj',
            'v_InattentionW_subj'
            ]
            titles = [
            'Boundary sep.',
            'Non-dec. time',
            'Intercept drift rate',
            'Drift AttentionW:C(OVcate)[low]',
            'Drift AttentionW:C(OVcate)[medium]', 
            'Drift AttentionW:C(OVcate)[high]',
            'Drift InattentionW', 
            ]
        elif version == 2:
            params_of_interest = [
            'a',
            't',
            'v_Intercept',
            'v_AttentionW',
            'v_InattentionW:C(OVcate)[low]',
            'v_InattentionW:C(OVcate)[medium]', 
            'v_InattentionW:C(OVcate)[high]',
            ]
            params_of_interest_s = [
            'a_subj', 
            't_subj', 
            'v_Intercept_subj',
            'v_AttentionW_subj',
            'v_InattentionW:C(OVcate)[low]_subj',
            'v_InattentionW:C(OVcate)[medium]_subj', 
            'v_InattentionW:C(OVcate)[high]_subj',
            ]
            titles = [
            'Boundary sep.',
            'Non-dec. time',
            'Intercept drift rate',
            'Drift AttentionW',
            'Drift InattentionW:C(OVcate)[low]',
            'Drift InattentionW:C(OVcate)[medium]',
            'Drift InattentionW:C(OVcate)[high]',
            ]
        elif version == 3:
            params_of_interest = [
            'a(low)',
            'a(medium)',
            'a(high)',
            't',
            'v_Intercept',
            'v_AttentionW',
            'v_InattentionW:C(OVcate)[low]',
            'v_InattentionW:C(OVcate)[medium]',
            'v_InattentionW:C(OVcate)[high]',
            ]
            params_of_interest_s = [
            'a(low)_subj',
            'a(medium)_subj',
            'a(high)_subj',
            't_subj',
            'v_Intercept_subj',
            'v_AttentionW_subj',
            'v_InattentionW:C(OVcate)[low]_subj',
            'v_InattentionW:C(OVcate)[medium]_subj',
            'v_InattentionW:C(OVcate)[high]_subj'
            
            ]
            titles = [
            'Boundary sep. (low OVcate)',
            'Boundary sep. (medium OVcate)',
            'Boundary sep. (high OVcate)',
            'Non-dec. time',
            'Intercept drift rate',
            'Drift AttentionW',
            'Drift InattentionW:C(OVcate)[low]',
            'Drift InattentionW:C(OVcate)[medium]',
            'Drift InattentionW:C(OVcate)[high]',
            ]
        elif version == 4:
            params_of_interest = [
            'a',
            't(low)',
            't(medium)',
            't(high)',
            'v_Intercept',
            'v_AttentionW',
            'v_InattentionW:C(OVcate)[low]',
            'v_InattentionW:C(OVcate)[medium]',
            'v_InattentionW:C(OVcate)[high]',
            ]
            params_of_interest_s = [
            'a_subj',
            't(low)_subj',
            't(medium)_subj',
            't(high)_subj',
            'v_Intercept_subj',
            'v_AttentionW_subj',
            'v_InattentionW:C(OVcate)[low]_subj',
            'v_InattentionW:C(OVcate)[medium]_subj',
            'v_InattentionW:C(OVcate)[high]_subj',
            ]
            titles = [
            'Boundary sep.',
            'Non-dec. time (low OVcate)',
            'Non-dec. time (medium OVcate)',
            'Non-dec. time (high OVcate)',
            'Intercept drift rate',
            'Drift AttentionW',
            'Drift InattentionW:C(OVcate)[low]',
            'Drift InattentionW:C(OVcate)[medium]',
            'Drift InattentionW:C(OVcate)[high]',
            ]
            
    if phase == 'ESEE':
        if version == 0:
            params_of_interest = [
                'a',
                't',
                'v_Intercept',
                'v_AttentionW',
                'v_InattentionW',
                ]
            params_of_interest_s = [
                'a_subj', 
                't_subj', 
                'v_Intercept_subj',
                'v_AttentionW_subj',
                'v_InattentionW_subj', 
                ]
            titles = [
                'Boundary sep.',
                'Non-dec. time',
            'Intercept drift rate',
            'Drift AttentionW',
            'Drift InattentionW',
            ]
        elif version == 1:
            params_of_interest = [
            'a',
            't', 
            'v_Intercept',
            'v_AttentionW:C(phase)[ES]',
            'v_AttentionW:C(phase)[EE]',
            'v_InattentionW',
            ]
            params_of_interest_s = [
            'a_subj',
            't_subj', 
            'v_Intercept_subj',
            'v_AttentionW:C(phase)[ES]_subj', 
            'v_AttentionW:C(phase)[EE]_subj',
            'v_InattentionW_subj'
            ]
            titles = [
            'Boundary sep.',
            'Non-dec. time',
            'Intercept drift rate',
            'Drift AttentionW:C(phase)[ES]',
            'Drift AttentionW:C(phase)[EE]', 
            'Drift InattentionW', 
            ]
        elif version == 2:
            params_of_interest = [
            'a',
            't',
            'v_Intercept',
            'v_AttentionW',
            'v_InattentionW:C(phase)[ES]',
            'v_InattentionW:C(phase)[EE]', 
            ]
            params_of_interest_s = [
            'a_subj', 
            't_subj', 
            'v_Intercept_subj',
            'v_AttentionW_subj',
            'v_InattentionW:C(phase)[ES]_subj',
            'v_InattentionW:C(phase)[EE]_subj', 
            ]
            titles = [
            'Boundary sep.',
            'Non-dec. time',
            'Intercept drift rate',
            'Drift AttentionW',
            'Drift InattentionW:C(phase)[ES]',
            'Drift InattentionW:C(phase)[EE]',
            ]
        elif version == 3:
            params_of_interest = [
            'a(ES)',
            'a(EE)',
            't',
            'v_Intercept',
            'v_AttentionW',
            'v_InattentionW:C(phase)[ES]',
            'v_InattentionW:C(phase)[EE]'
            ]
            params_of_interest_s = [
            'a(ES)_subj',
            'a(EE)_subj',
            't_subj',
            'v_Intercept_subj',
            'v_AttentionW_subj',
            'v_InattentionW:C(phase)[ES]_subj',
            'v_InattentionW:C(phase)[EE]_subj'            
            ]
            titles = [
            'Boundary sep. (ES)',
            'Boundary sep. (EE)',
            'Non-dec. time',
            'Intercept drift rate',
            'Drift AttentionW',
            'Drift InattentionW:C(phase)[ES]',
            'Drift InattentionW:C(phase)[EE]',
            ]
        elif version == 4:
            params_of_interest = [
            'a',
            't(ES)',
            't(EE)',
            'v_Intercept',
            'v_AttentionW',
            'v_InattentionW:C(phase)[ES]',
            'v_InattentionW:C(phase)[EE]',
            ]
            params_of_interest_s = [
            'a_subj',
            't(ES)_subj',
            't(EE)_subj',
            'v_Intercept_subj',
            'v_AttentionW_subj',
            'v_InattentionW:C(phase)[ES]_subj',
            'v_InattentionW:C(phase)[EE]_subj',
            ]
            titles = [
            'Boundary sep.',
            'Non-dec. time (ES)',
            'Non-dec. time (EE)',
            'Intercept drift rate',
            'Drift AttentionW',
            'Drift InattentionW:C(phase)[ES]',
            'Drift InattentionW:C(phase)[EE]',
            ]
            
            
    if phase == 'LEESEE':
        if version == 0:
            params_of_interest = [
                'a',
                't',
                'v_Intercept',
                'v_AttentionW',
                'v_InattentionW',
                ]
            params_of_interest_s = [
                'a_subj', 
                't_subj', 
                'v_Intercept_subj',
                'v_AttentionW_subj',
                'v_InattentionW_subj', 
                ]
            titles = [
                'Boundary sep.',
                'Non-dec. time',
                'Intercept drift rate',
                'Drift AttentionW',
                'Drift InattentionW',
                ]
        elif version == 1:
            params_of_interest = [
            'a',
            't', 
            'v_Intercept',
            'v_AttentionW:C(phase)[LE]',
            'v_AttentionW:C(phase)[ES]',
            'v_AttentionW:C(phase)[EE]',
            'v_InattentionW',
            ]
            params_of_interest_s = [
            'a_subj',
            't_subj', 
            'v_Intercept_subj',
            'v_AttentionW:C(phase)[LE]_subj', 
            'v_AttentionW:C(phase)[ES]_subj', 
            'v_AttentionW:C(phase)[EE]_subj',
            'v_InattentionW_subj'
            ]
            titles = [
            'Boundary sep.',
            'Non-dec. time',
            'Intercept drift rate',
            'Drift AttentionW:C(phase)[LE]',
            'Drift AttentionW:C(phase)[ES]',
            'Drift AttentionW:C(phase)[EE]', 
            'Drift InattentionW', 
            ]
        elif version == 2:
            params_of_interest = [
            'a',
            't',
            'v_Intercept',
            'v_AttentionW',
            'v_InattentionW:C(phase)[LE]',
            'v_InattentionW:C(phase)[ES]',
            'v_InattentionW:C(phase)[EE]', 
            ]
            params_of_interest_s = [
            'a_subj', 
            't_subj', 
            'v_Intercept_subj',
            'v_AttentionW_subj',
            'v_InattentionW:C(phase)[LE]_subj',
            'v_InattentionW:C(phase)[ES]_subj',
            'v_InattentionW:C(phase)[EE]_subj', 
            ]
            titles = [
            'Boundary sep.',
            'Non-dec. time',
            'Intercept drift rate',
            'Drift AttentionW',
            'Drift InattentionW:C(phase)[LE]',
            'Drift InattentionW:C(phase)[ES]',
            'Drift InattentionW:C(phase)[EE]',
            ]
        elif version == 3:
            params_of_interest = [
            'a(ES)',
            'a(ES)',
            'a(EE)',
            't',
            'v_Intercept',
            'v_AttentionW',
            'v_InattentionW:C(phase)[LE]',
            'v_InattentionW:C(phase)[ES]',
            'v_InattentionW:C(phase)[EE]'
            ]
            params_of_interest_s = [
            'a(LE)_subj',
            'a(ES)_subj',
            'a(EE)_subj',
            't_subj',
            'v_Intercept_subj',
            'v_AttentionW_subj',
            'v_InattentionW:C(phase)[LE]_subj',
            'v_InattentionW:C(phase)[ES]_subj',
            'v_InattentionW:C(phase)[EE]_subj'            
            ]
            titles = [
            'Boundary sep. (LE)',
            'Boundary sep. (ES)',
            'Boundary sep. (EE)',
            'Non-dec. time',
            'Intercept drift rate',
            'Drift AttentionW',
            'Drift InattentionW:C(phase)[LE]',
            'Drift InattentionW:C(phase)[ES]',
            'Drift InattentionW:C(phase)[EE]',
            ]
        elif version == 4:
            params_of_interest = [
            'a',
            't(LE)',
            't(ES)',
            't(EE)',
            'v_Intercept',
            'v_AttentionW',
            'v_InattentionW:C(phase)[LE]',
            'v_InattentionW:C(phase)[ES]',
            'v_InattentionW:C(phase)[EE]',
            ]
            params_of_interest_s = [
            'a_subj',
            't(LE)_subj',
            't(ES)_subj',
            't(EE)_subj',
            'v_Intercept_subj',
            'v_AttentionW_subj',
            'v_InattentionW:C(phase)[LE]_subj',
            'v_InattentionW:C(phase)[ES]_subj',
            'v_InattentionW:C(phase)[EE]_subj',
            ]
            titles = [
            'Boundary sep.',
            'Non-dec. time (LE)',
            'Non-dec. time (ES)',
            'Non-dec. time (EE)',
            'Intercept drift rate',
            'Drift AttentionW',
            'Drift InattentionW:C(phase)[LE]',
            'Drift InattentionW:C(phase)[ES]',
            'Drift InattentionW:C(phase)[EE]',
            ]
            
    # diagnostics
    diag_dir = Path(fig_dir) / "diagnostics"
    ensure_dir(diag_dir)
    
    # Gelman-Rubin
    gr = hddm.analyze.gelman_rubin(models)
    with open(diag_dir / "gelman_rubin.txt", "w") as f:
        for param, val in gr.items():
            f.write(f"{param}: {val}\n")
    # DIC
    dic = combined_model.dic
    (diag_dir / "DIC.txt").write_text(f"DIC: {dic}\n")
    size_plot = len(combined_model.data.subj_idx.unique()) / 3.0 * 1.5
    combined_model.plot_posterior_predictive(samples=10, bins=100, figsize=(6, size_plot), save=True, path=str(diag_dir), format="pdf")
    
    # shrink font 
    matplotlib.rcParams.update({"font.size": 6})
    combined_model.plot_posteriors(save=True,
                                   path=str(diag_dir),
                                   format="pdf")
    matplotlib.rcParams.update({"font.size": 12})

    # stats table
    results = combined_model.gen_stats()
    results.to_csv(diag_dir / "results.csv")
    
    # Posterior‐trace KDEs
    traces = [combined_model.nodes_db.node[p].trace() for p in params_of_interest]
    # optional alpha‐transform if RL is used for instance
    if "alpha" in params_of_interest:
        idx = params_of_interest.index("alpha")
        traces[idx] = np.exp(traces[idx]) / (1 + np.exp(traces[idx]))
    
    stats = [min(np.mean(t>0), np.mean(t<0)) for t in traces]
    n_cols = 5
    n_rows = int(np.ceil(len(traces) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*3, n_rows*4))
    axes = axes.flatten()
    
    for i, (trace, title) in enumerate(zip(traces, titles)):
        sns.kdeplot(trace, vertical=True, shade=True, color='purple', ax=axes[i])
        axes[i].set_title(f"{title}\np={stats[i]:.3f}", fontsize=6)
        axes[i].set_xlim(left=0)
        if i % n_cols == 0:
            axes[i].set_ylabel("Parameter estimate (a.u.)")
        if i >= len(traces) - n_cols:
            axes[i].set_xlabel("Posterior probability")
        for side in ["top","bottom","left","right"]:
            axes[i].spines[side].set_linewidth(0.5)
            axes[i].tick_params(width=0.5, labelsize=6)   
            
    for ax in axes[len(traces):]:
        fig.delaxes(ax)
    sns.despine(offset=10, trim=True)
    plt.tight_layout()
    fig.savefig(diag_dir / "posteriors.pdf", bbox_inches="tight")
    plt.close(fig) 
    
    
    # save inidviudal parameters
    parameters = []
    for p in params_of_interest_s:
        param_values = []
        for s in np.unique(combined_model.data.subj_idx):
            param_name = f"{p}.{s}"
            try:
                val = results.loc[results.index == param_name, 'mean'].values
                if len(val):
                    v = val[0]
                    if 'alpha' in p:
                        # inverse‐logit transform for alpha‐params
                        v = np.exp(v) / (1 + np.exp(v))
                    param_values.append(v)
            except KeyError:
                print(f"Param {param_name} missing. Skipping…")
        parameters.append(param_values)

    # turn into DataFrame, transpose so each subj is a row
    param_df = pd.DataFrame(parameters).T
    param_df.columns = params_of_interest_s
    param_df.to_csv(diag_dir / "params_of_interest_s.csv", index=False)
    

# directories
#model_dir = 'models_dir_garcia/'
#ensure_dir(model_dir)

model_dir = BASE_MODEL_DIR


# ==================================================================
# BATCH DRIVER – runs every (phase, version) pairing
# ==================================================================

if __name__ == "__main__":

    data_full = pd.read_csv((PROJECT_DIR / "data_sets" / "data_sets_Garcia" / "GarciaParticipants_Eye_Response_Feed_Allfix_addm_OV_Abs_CCT.csv").as_posix(), sep=",")
    # loop over phases and versions
    for phase in PHASE_RUN_ORDER:
        if phase in SKIP_PHASES:
            continue                    
        
        phase_key = phase


        for version, model_name in enumerate(model_versions[phase]):
            if not started:
                if phase == start_phase and version >= start_version:
                    started = True
                elif PHASE_RUN_ORDER.index(phase) > PHASE_RUN_ORDER.index(start_phase):
                    started = True
                else:
                    continue 
            
            full_model_name = model_base_name + model_name
            print(f"\n===  PHASE {phase} : {model_name}  ===")

            # filter data for this phase
            source_phase = PHASE_TO_SOURCE.get(phase, phase)   #assignes ES_ZBIAS

            if phase == "ESEE":
                data = data_full[data_full["phase"].isin(["ES", "EE"])].copy()
            elif phase == "LEESEE":
                data = data_full[data_full["phase"].isin(["LE", "ES", "EE"])].copy()
            else:
                data = data_full[data_full["phase"] == source_phase].copy() 
            
            if data.empty:
                raise ValueError(f"No rows left after filtering for phase '{phase}' "
                                 f"(source = '{source_phase}')")

            # preprocessing 
            data["gazeCI"]  = pd.to_numeric(data["gazeCI"],  errors="coerce")
            data["gazeSE"]= pd.to_numeric(data["gazeSE"],errors="coerce")
            data["phase"]       = data["phase"].astype("category")
            data["rt"]          = pd.to_numeric(data["rtime"], errors="coerce")
            data["chose_right"] = pd.to_numeric(data["chose_right"], errors="coerce")
            data["chose_left"] = pd.to_numeric(data["chose_left"], errors="coerce")

            data                = data[data["rt"] > 0.250]
            # data["response"]    = pd.to_numeric(data["corr"], errors="coerce")

            # here, it's important to be selctive depending on whether chose_right or chose_left is the upper bound
            if phase in ("ES_ZBIAS", "ES_quad", "ES_VAL", "For_paper"):

                data["response"] = pd.to_numeric(data["chose_right"], errors="coerce")
                print("[ZBIAS DEBUG] head of response mapping:")
                print(data[["chose_right","corr","response"]].head(10).to_string(index=False))
                print("counts:", data["response"].value_counts(dropna=False).to_dict())
                # sanity checks ...are we actually filtering the right things
                mismatches = (data["response"] != data["chose_right"]).sum()
                assert mismatches == 0, f"{mismatches} rows where response ≠ chose_right!"
            else:
                data["response"] = pd.to_numeric(data["corr"], errors="coerce")
                print(data[["chose_right","corr","response"]].head(5).to_string(index=False))

            
            print(f"[DEBUG] phase={phase}  response counts:\n",
            data["response"].value_counts(dropna=False).head())
            data["OVcate"]      = data["OVcate_2"].astype("category")
            data["Abscate"]     = data["Abscate_2"].astype("category")
            data["cond"]     = data["cond"].fillna(-1)
            data["cond"]     = data["cond"].astype("int")
            data["AttentionW"]  = pd.to_numeric(data["AttentionW"],  errors="coerce")
            data["InattentionW"]= pd.to_numeric(data["InattentionW"],errors="coerce")
            data["ES_AttentionW"]  = pd.to_numeric(data["ES_AttentionW"],  errors="coerce")
            data["ES_InattentionW"]= pd.to_numeric(data["ES_InattentionW"],errors="coerce")
            data["subj_idx"]    = data["sub_id"]
            #data["stimulus"] = pd.to_numeric(data["stimulus"], errors="coerce")
            data["DTA"] = pd.to_numeric(data["DTA"],errors="coerce")
            data["DwellPropAdvantage"] = pd.to_numeric(data["DwellPropAdvantage"],errors="coerce")
            data["DwellLeft"]  = pd.to_numeric(data["DwellLeft"],  errors="coerce")
            data["DwellRight"] = pd.to_numeric(data["DwellRight"], errors="coerce")
            data["AttentionW_E"]  = pd.to_numeric(data["AttentionW_E"],  errors="coerce")
            data["AttentionW_S"] = pd.to_numeric(data["AttentionW_S"], errors="coerce")
            data["InattentionW_E"] = pd.to_numeric(data["InattentionW_E"], errors="coerce")
            data["InattentionW_S"] = pd.to_numeric(data["InattentionW_S"], errors="coerce")
            data["ES_AttentionW_E"]  = pd.to_numeric(data["ES_AttentionW_E"],  errors="coerce")
            data["ES_AttentionW_S"] = pd.to_numeric(data["ES_AttentionW_S"], errors="coerce")
            data["ES_InattentionW_E"] = pd.to_numeric(data["ES_InattentionW_E"], errors="coerce")
            data["ES_InattentionW_S"] = pd.to_numeric(data["ES_InattentionW_S"], errors="coerce")
            data["ES_AttentionW"]  = pd.to_numeric(data["ES_AttentionW"],  errors="coerce")
            data["ES_InattentionW"]  = pd.to_numeric(data["ES_InattentionW"],  errors="coerce")
            data["V_E"] = pd.to_numeric(data["V_E"], errors="coerce")
            data["V_S"] = pd.to_numeric(data["V_S"], errors="coerce")
            data["Value_diff"] = pd.to_numeric(data["Value_diff"], errors="coerce")
            data["ES_AttentionW_S_dwell"] = pd.to_numeric(data["ES_AttentionW_S_dwell"], errors="coerce")
            data["ES_InattentionW_E_dwell"] = pd.to_numeric(data["ES_InattentionW_E_dwell"], errors="coerce")
            data["ES_InattentionW_S_dwell"] = pd.to_numeric(data["ES_InattentionW_S_dwell"], errors="coerce")
            
            # keep only trials with strictly positive dwell time on both sides, this can be changed; depends on the goal
            #data = data[(data["DwellLeft"] > -1) & (data["DwellRight"] > -1)]
            data = data[~data["subj_idx"].isin({1,4,5,6,14,99})]
            data = data.dropna(subset=["rt",
                                       "response",
                                       "OVcate",
                                       "Abscate",
                                       "subj_idx",
                                       "AttentionW",
                                       "InattentionW",
                                       "cond",
                                       "gazeSE",
                                       "gazeCI",
                                       "ES_AttentionW",
                                       "ES_InattentionW",
                                       "chose_left",
                                       "DTA",
                                       "AttentionW_E",
                                       "AttentionW_S",
                                       "InattentionW_E",
                                       "InattentionW_S",
                                       "z_AttentionW_E",
                                       "z_AttentionW_S",
                                       "z_InattentionW_E",
                                       "z_InattentionW_S",
                                       "V_E", "V_S", "Value_diff",
                                       "ES_AttentionW_E",
                                       "ES_AttentionW_S",
                                       "ES_InattentionW_E",
                                       "ES_InattentionW_S",
                                       "ES_AttentionW",
                                       "ES_InattentionW",
                                       "ES_AttentionW_S_dwell",
                                       "ES_InattentionW_E_dwell",
                                       "ES_InattentionW_S_dwell",
                                       ])   
            
            # quick report at the start
            quick_report(data, phase, version, model_name, phase_key)

            # fig_dir = os.path.join("figures_dir_garcia", full_model_name)
            # ensure_dir(os.path.join(fig_dir, "diagnostics"))
            
            fig_dir = FIG_DIR_ROOT / full_model_name
            ensure_dir(fig_dir / "diagnostics")

            # # run hddm function 
            drift_diffusion_hddm(
                data=data,
                samples=nr_samples,
                n_jobs=nr_models,
                run=RUN_ALL_MODELS,
                parallel=parallel,
                model_name=full_model_name,
                model_dir=BASE_MODEL_DIR,        
                version=version,
                phase=phase,
                accuracy_coding=True
            )
            #only when running RL in the LE phase
            # drift_diffusion_hddmRL(
            #     data=data,
            #     samples=nr_samples,
            #     n_jobs=nr_models,
            #     run=RUN_ALL_MODELS,
            #     parallel=parallel,
            #     model_name=full_model_name,
            #     model_dir=BASE_MODEL_DIR,        
            #     version=version,
            #     phase=phase,
            # )