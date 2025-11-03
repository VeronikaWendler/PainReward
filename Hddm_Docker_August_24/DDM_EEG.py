## Running the hddm models
#
# Pipeline for running hddm regression models for the PainReward task
# Veronika Wendler
# some inspiration comes from Python2 code from Jan Willem de Gee that I translated into Python3.
#
# TO DO: More models

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




#------------------------------------------------------------------------------------------------------------------
# very important! If you try to plot parameters such as 'v_C(Abs_value)_subj' without specifying the exact levels like [low_abs],
# [mid_abs], and [high_abs], the code will raise an error. 
#------------------------------------------------------------------------------------------------------------------
# params:

nr_models       = 4         # number of MCMC chains
nr_samples      = 12000      # samples per chain - do 6000 (+1000 for burn-in) but for now for a quick one we do 600
parallel        = True      # parallel
model_base_name = "painreward_behavioural_data_"
model_versions  = {
    "dec":      ["LPP_0","LPP_1","LPP_2","LPP_3","LPP_4","LPP_5","LPP_6","LPP_7","LPP_8"]     
}

PHASE_TO_SOURCE = {
    "dec": "decision", 
}

# BATCH-RUN CONTROL
PHASE_RUN_ORDER = ["dec"]                                      # order
SKIP_PHASES     = {}                                             # ignored this phase
RUN_ALL_MODELS  = True                                           # False = just load existing fits (but loading is done in the aDDM_Garcia_LE_ES_EE.py file)

# selectivity
start_phase = "dec"
start_version = 5
started = False

# dir
PROJECT_DIR   = pathlib.Path(os.getenv("PROJECT_DIR", "/workspace")).resolve()
BASE_MODEL_DIR = PROJECT_DIR / "Hddm_Docker_August_24/models_dir"
FIG_DIR_ROOT   = PROJECT_DIR / "Hddm_Docker_August_24/figures_dir"


# reporting function
# can be seen in the cluster output
def quick_report(data, phase, version, model_name, phase_key):
    print(f"\n Phase = {phase}   Version = {version}")
    print(f"Model name          : {model_name}")
    print(f"Selected phase_key  : {phase_key}")
    print(f"N trials            : {len(data):,}")
    print(f"Participants        : {sorted(data['subj_idx'].unique())}")

    fig, ax = plt.subplots(figsize=(6,4))
    for _, d in data.groupby('subj_idx'):
        d['rt'].hist(bins=20, histtype='step', ax=ax, alpha=.4)
    ax.set(
        title=f"RT distribution – {phase} v{version}",
        xlabel="RT (s)",
        ylabel="count"
    )
    plt.show()

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




# drop entire participants for quest data only, NO FOR ENTIRE DATA, otherwise the operating system kills the worker
# quest_vers = [x, z, u, i]  # questionnaire versions 
# if version in quest_vers:
#     data.dropna(subset=["STA_SAI_Score","STA_TAI_Score","PCS_Score"], inplace=True)


## this is optional and depends on your data and requirements:
# def standardize_data(data):
#     for s in np.unique(data['subj_idx']):
#         data.loc[data['subj_idx'] == s, 'painlevel'] = (data.loc[data['subj_idx'] == s, 'painlevel'] - np.mean(data.loc[data['subj_idx'] == s, 'painlevel'])) / np.std(data.loc[data['subj_idx'] == s, 'painlevel'])
#         data.loc[data['subj_idx'] == s, 'moneylevel'] = (data.loc[data['subj_idx'] == s, 'moneylevel'] - np.mean(data.loc[data['subj_idx'] == s, 'moneylevel'])) / np.std(data.loc[data['subj_idx'] == s, 'moneylevel'])
#         data.loc[data['subj_idx'] == s, 'fixduration'] = (data.loc[data['subj_idx'] == s, 'fixduration'] - np.mean(data.loc[data['subj_idx'] == s, 'fixduration'])) / np.std(data.loc[data['subj_idx'] == s, 'fixduration'])
#         data.loc[data['subj_idx'] == s, 'sv_money'] = (data.loc[data['subj_idx'] == s, 'sv_money'] - np.mean(data.loc[data['subj_idx'] == s, 'sv_money'])) / np.std(data.loc[data['subj_idx'] == s, 'sv_money'])
#         data.loc[data['subj_idx'] == s, 'sv_pain'] = (data.loc[data['subj_idx'] == s, 'sv_pain'] - np.mean(data.loc[data['subj_idx'] == s, 'sv_pain'])) / np.std(data.loc[data['subj_idx'] == s, 'sv_pain'])
#         data.loc[data['subj_idx'] == s, 'sv_both'] = (data.loc[data['subj_idx'] == s, 'sv_both'] - np.mean(data.loc[data['subj_idx'] == s, 'sv_both'])) / np.std(data.loc[data['subj_idx'] == s, 'sv_both'])
#         #data.loc[data['subj_idx'] == s, 'p_pain_all'] = (data.loc[data['subj_idx'] == s, 'p_pain_all'] - np.mean(data.loc[data['subj_idx'] == s, 'p_pain_all'])) / np.std(data.loc[data['subj_idx'] == s, 'p_pain_all'])
#     return data

# data = standardize_data(data)
# data
#

#%%
# drift diffusion models
#------------------------------------------------------------------------------------------------------------------
# function that runs the different versions of DDM regressions
def run_model(trace_id, data, model_dir, model_name, version, phase, samples=12000, accuracy_coding=True): 
    import os
    import numpy as np
    import hddm
    from patsy import dmatrix  

    # ensure_dir(model_dir)   
    
    depends_on = {}
    
    if phase == 'dec':
        depends_on = {}
    
        if version == 0:
            # jsut start sampling

            m = hddm.models.HDDM(data, 
                                    p_outlier=.05, 
                                    include=['a', 't', 'v', 'z'],   #'z'
                                    depends_on=depends_on,
                                    )
            m.find_starting_values()
            infdata = m.sample(samples,
                               burn=2000,
                               dbname=os.path.join(model_dir, model_name + f'_db{trace_id}'), 
                               db='pickle',
                               return_infdata=True, loglike=True, ppc=True)
            return m, infdata
        
        elif version == 1:  # drift rate is dependent on the the sv_pain_para
            v_reg = {'model': 'v ~ 1 + sv_pain_para', 'link_func': lambda x: x}
            reg_descr = [v_reg]
        elif version == 2:  # drift rate is dependent on the the sv_pain_para
            v_reg = {'model': 'v ~ 0 + sv_pain_para', 'link_func': lambda x: x}
            reg_descr = [v_reg]
        elif version == 3:  # drift rate is dependent on the the sv_pain_para
            a_reg = {'model': 'a ~ 1 + sv_pain_para', 'link_func': lambda x: x}
            reg_descr = [a_reg] 
            #this model (NR 4) doesn't work and fails to find starting values   
        elif version == 4:  # drift rate is dependent on the the sv_pain_para
            a_reg = {'model': 'a ~ 0 + sv_pain_para', 'link_func': lambda x: x}
            reg_descr = [a_reg]
        elif version == 5:  # drift rate is dependent on the the sv_pain_para
            z_reg = {'model': 'z ~ 1 + sv_pain_para', 'link_func': lambda x: x}
            reg_descr = [z_reg]
        elif version == 6:  # drift rate is dependent on the the sv_pain_para
            z_reg = {'model': 'z ~ 0 + sv_pain_para', 'link_func': lambda x: x}
            reg_descr = [z_reg]
        elif version == 7:  # drift rate is dependent on the the sv_pain_para
            t_reg = {'model': 't ~ 1 + sv_pain_para', 'link_func': lambda x: x}
            reg_descr = [t_reg]    
        elif version == 8:  # drift rate is dependent on the the sv_pain_para
            t_reg = {'model': 't ~ 0 + sv_pain_para', 'link_func': lambda x: x}
            reg_descr = [t_reg]       
        else:
            raise ValueError(f"Is this version correct ? ")   
        

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
                   burn=2000,
                   dbname=os.path.join(model_dir, model_name + f'_db{trace_id}'), 
                   db='pickle',
                   return_infdata=True, loglike=True, ppc=True)
        return m, infdata
        
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Main function for running/loading models
import dill as pickle  # to create the pkl object

def drift_diffusion_hddm(data, 
                         samples=12000,
                         n_jobs=4,
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
    


model_dir = BASE_MODEL_DIR


#_________________________________________________________________________________________________________________________________________________________________________
# Main running function

if __name__ == "__main__":
    
    data_full = pd.read_csv((PROJECT_DIR / "Hddm_Docker_August_24" / "data_sets" / "behavioural_sv_cleaned_final_3.csv").as_posix(), sep=",")
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

            if phase == "dec":
                data = data_full[data_full["TaskName"].isin(["decision"])].copy()
            elif phase == "pas":
                data = data_full[data_full["TaskName"].isin(["passive"])].copy()
            else:
                data = data_full[data_full["TaskName"] == source_phase].copy() 
            
            if data.empty:
                raise ValueError(f"No rows left after filtering for phase '{phase}' "
                                 f"(source = '{source_phase}')")


            data['Abs_Money_Pain'] = data['Abs_Money_Pain'].astype("category")
            data['OV_Money_Pain'] = data['OV_Money_Pain'].astype("category")
            data['Abs_value'] = data['Abs_value'].astype("category")
            data['OV_value'] = data['OV_value'].astype("category")
            data['acceptance_pair'] = data['acceptance_pair'].astype("category")

            data                = data[data["rt"] > 0.250]
            data["response"]    = pd.to_numeric(data["response"], errors="coerce")

            data["subj_idx"]    = data["subj_idx"]
            subjects = np.unique(data.subj_idx)
            nr_subjects = subjects.shape[0]
            print(nr_subjects)
            
                       
            #data = data[~data["subj_idx"].isin({})]
            data.dropna(subset=['rt', 
                                "painlevel",
                                "moneylevel",
                                "accepted",
                                'acceptance_pair',
                                'sv_money',
                                'sv_pain',
                                'sv_both',
                                'p_pain_all',
                                'Abs_Money_Pain',
                                'OV_Money_Pain',
                                'sv_pain_para',
                                'sv_both_para',
                                'k_pain_para',
                                'beta_para',
                                'bias_para',
                                'STA_SAI_Score',
                                'STA_TAI_Score',
                                'PCS_Score'], inplace = True)    #'STA_SAI_Score','STA_TAI_Score','PCS_Score'

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


#_________________________________________________________________________________________________________________________________________________________________________
# Getting the tiral-by trial param betas (influenced by sv_pain_para) for the EEG regression analysis



