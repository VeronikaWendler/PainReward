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
nr_samples      = 6000      # samples per chain - do 6000 (+1000 for burn-in) but for now for a quick one we do 600
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
start_version = 1
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


#%%
# drift diffusion models
#------------------------------------------------------------------------------------------------------------------
# function that runs the different versions of DDM regressions
def run_model(trace_id, data, model_dir, model_name, version, phase, samples=6000, accuracy_coding=True): 
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
                               burn=1000,
                               dbname=os.path.join(model_dir, model_name + f'_db{trace_id}'), 
                               db='pickle',
                               return_infdata=True, loglike=True, ppc=True)
            return m, infdata
        
        elif version == 1:  # drift rate is dependent on the the sv_pain_para
            v_reg = {'model': 'v ~ 1 + sv_pain_para', 'link_func': lambda x: x}
            reg_descr = [v_reg]
        elif version == 2:  # drift rate is dependent on the the sv_pain_para
            v_reg = {'model': 'v ~ sv_pain_para', 'link_func': lambda x: x}
            reg_descr = [v_reg]
            
        elif version == 3:  # drift rate is dependent on the the sv_pain_para
            a_reg = {'model': 'a ~ 1 + sv_pain_para', 'link_func': lambda x: x}
            reg_descr = [a_reg]    
        elif version == 4:  # drift rate is dependent on the the sv_pain_para
            a_reg = {'model': 'a ~ sv_pain_para', 'link_func': lambda x: x}
            reg_descr = [a_reg]
            
        elif version == 5:  # drift rate is dependent on the the sv_pain_para
            z_reg = {'model': 'z ~ 1 + sv_pain_para', 'link_func': lambda x: x}
            reg_descr = [z_reg]
        elif version == 6:  # drift rate is dependent on the the sv_pain_para
            z_reg = {'model': 'z ~ sv_pain_para', 'link_func': lambda x: x}
            reg_descr = [z_reg]
            
        elif version == 7:  # drift rate is dependent on the the sv_pain_para
            t_reg = {'model': 't ~ 1 + sv_pain_para', 'link_func': lambda x: x}
            reg_descr = [t_reg]    
        elif version == 8:  # drift rate is dependent on the the sv_pain_para
            t_reg = {'model': 't ~ sv_pain_para', 'link_func': lambda x: x}
            reg_descr = [t_reg]       
        else:
            raise ValueError(f"Is this version illegal ?? It feels illegal...")   
        

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
                   burn=500,
                   dbname=os.path.join(model_dir, model_name + f'_db{trace_id}'), 
                   db='pickle',
                   return_infdata=True, loglike=True, ppc=True)
        return m, infdata
        
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Main function for running/loading models
import dill as pickle  # to create the pkl object

def drift_diffusion_hddm(data, 
                         samples=6000,
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
    
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Function to plot the parameters
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
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
    
    if version == 0:
        params_of_interest = ['z', 'a', 't', 'v']
        params_of_interest_s = ['z_subj', 'a_subj', 't_subj', 'v_Intercept_subj']
        titles = ['Starting point', 'Boundary sep.', 'Non-dec. time', 'Drift rate']
    if version == 1:
        params_of_interest = ['z', 'a', 't', 'sv', 'sz', 'st', 'v_Intercept', 'v_sv_pain_para']
        params_of_interest_s = ['z_subj', 'a_subj', 't_subj', 'v_Intercept_subj', 'v_sv_pain_para_subj']
        titles = [
            'Starting point', 'Boundary sep.', 'Non-dec. time',
            'Inter-trial variability in drift rate','Inter-trial variability in starting point', 'Inter-trial variability in non-decision time',
            'v_Intercept rate', 'Drift v_sv_pain_para']
    elif version == 2:
        params_of_interest = [
            'sv', 'sz', 'st',
            'v_Intercept', 'v_sv_pain_para',
            't_Intercept', 't_sv_pain_para',
            'a_Intercept', 'a_sv_pain_para', 
            'z_Intercept', 'z_sv_pain_para',
            ]
        params_of_interest_s = [
            'sv_subj', 'sz_subj', 'st_subj',
            'v_Intercept_subj', 'v_sv_pain_para_subj', 
            't_Intercept_subj', 't_sv_pain_para_subj',
            'a_Intercept_subj', 'a_sv_pain_para_subj',
            'z_Intercept_subj', 'z_sv_pain_para_subj',
            ]
        titles = [
            'Inter-trial variability in drift rate', 'Inter-trial variability in starting point', 'Inter-trial variability in non-decision time',
            'Intercept drift rate', 'Drift rate sv_pain_para',
            'Intercept non-decision time', 'Non-decision time sv_pain_para',
            'Intercept boundary separation', 'Boundary separation sv_pain_para',
            'Intercept Strarting Point', 'Starting point sv_pain_para',
            ]
    elif version == 3:
        params_of_interest = ['z', 'a', 't', 'sv', 'sz', 'st', 'v_Intercept', 'v_sv_money']
        params_of_interest_s = ['z_subj', 'a_subj', 't_subj', 'v_Intercept_subj', 'v_sv_money_subj']
        titles = ['Starting point', 'Boundary sep.', 'Non-dec. time',
            'Inter-trial variability in drift rate', 'Inter-trial variability in starting point', 'Inter-trial variability in non-decision time',
            'v_Intercept rate', 'Drift v_sv_money']
    elif version == 4:
        params_of_interest = [
            'sv', 'sz', 'st',
            'v_Intercept', 'v_sv_money',
            't_Intercept', 't_sv_money',
            'a_Intercept', 'a_sv_money',
            'z_Intercept', 'z_sv_money']
        params_of_interest_s = [
            'sv_subj', 'sz_subj', 'st_subj',
            'v_Intercept_subj', 'v_sv_money_subj', 
            't_Intercept_subj', 't_sv_money_subj',
            'a_Intercept_subj', 'a_sv_money_subj',
            'z_Intercept_subj', 'z_sv_money_subj']
        titles = [
            'Starting point', 'Inter-trial variability in drift rate', 'Inter-trial variability in starting point', 'Inter-trial variability in non-decision time',
            'Intercept drift rate', 'Drift rate sv_money',
            'Intercept non-decision time', 'Non-decision time sv_money',
            'Intercept boundary separation', 'Boundary separation sv_money',
            'Intercept Starting Point', 'Starting point sv_money']
#     elif version == 5:
#         params_of_interest = ['z', 'sv', 'sz', 'st', 't_Intercept', 't_sv_pain_para']
#         params_of_interest_s = ['z_subj', 'sv_subj', 'sz_subj', 'st_subj','t_Intercept_subj', 't_sv_pain_para_subj']
#         titles = ['Starting point', 'Inter-trial variability in drift rate', 'Inter-trial variability in starting point', 'Inter-trial variability in non-decision time',
#                 'Intercept non-decision time', 'Non-decision time sv_pain_para']
#     elif version == 6:
#         params_of_interest = ['z', 'sv', 'sz', 'st', 'a_Intercept', 'a_sv_pain_para']
#         params_of_interest_s = ['z_subj', 'sv_subj', 'sz_subj', 'st_subj', 'a_Intercept_subj', 'a_sv_pain_para_subj']
#         titles = ['Starting point', 'Inter-trial variability in drift rate', 'Inter-trial variability in starting point', 'Inter-trial variability in non-decision time','Intercept boundary separation', 'Boundary separation sv_pain_para']   
#     elif version == 7:
#         params_of_interest = ['z', 'sv', 'sz', 'st', 't_Intercept', 't_sv_money']
#         params_of_interest_s = ['z_subj', 'sv_subj', 'sz_subj', 'st_subj', 't_Intercept_subj', 't_sv_money_subj']
#         titles = ['Starting point', 'Inter-trial variability in drift rate', 'Inter-trial variability in starting point', 'Inter-trial variability in non-decision time','Intercept non-decision time', 'Non-decision time sv_money']
#     elif version == 8:
#         params_of_interest = ['z', 'sv', 'sz', 'st', 'a_Intercept', 'a_sv_money']
#         params_of_interest_s = ['z_subj', 'sv_subj', 'sz_subj', 'st_subj', 'a_Intercept_subj', 'a_sv_money_subj']
#         titles = ['Starting point', 'Inter-trial variability in drift rate', 'Inter-trial variability in starting point', 'Inter-trial variability in non-decision time','Intercept boundary separation', 'Boundary separation sv_money']
#     elif version == 9:
#         params_of_interest = ['z', 'a', 't', 'sv', 'sz', 'st', 'v_Intercept', 'v_sv_pain_para', 'v_sv_money', 'v_sv_pain_para:sv_money']
#         params_of_interest_s = ['z_subj', 'a_subj', 't_subj', 'v_Intercept_subj', 'v_sv_pain_para_subj', 'v_sv_money_subj', 'v_sv_pain_para:sv_money_subj']
#         titles = ['Starting point', 'Boundary sep.', 'Non-dec. time', 
#             'Inter-trial variability in drift rate', 'Inter-trial variability in starting point', 'Inter-trial variability in non-decision time',
#             'v_Intercept rate', 'Drift v_sv_pain_para', 'Drift v_sv_money', 'Drift rate interaction']
#     elif version == 10:
#         params_of_interest = [
#             'sv', 'sz', 'st',
#             'v_Intercept', 'v_sv_pain_para', 'v_sv_money', 'v_sv_pain_para:sv_money',
#             't_Intercept', 't_sv_pain_para', 't_sv_money', 't_sv_pain_para:sv_money',
#             'a_Intercept', 'a_sv_pain_para', 'a_sv_money', 'a_sv_pain_para:sv_money',
#             'z_Intercept', 'z_sv_pain_para', 'z_sv_money', 'z_sv_pain_para:sv_money']
#         params_of_interest_s = [
#             'sv_subj', 'sz_subj', 'st_subj',
#             'v_Intercept_subj', 'v_sv_pain_para_subj', 'v_sv_money_subj', 'v_sv_pain_para:sv_money_subj',
#             't_Intercept_subj', 't_sv_pain_para_subj', 't_sv_money_subj', 't_sv_pain_para:sv_money_subj',
#             'a_Intercept_subj', 'a_sv_pain_para_subj', 'a_sv_money_subj', 'a_sv_pain_para:sv_money_subj',
#             'z_Intercept_subj', 'z_sv_pain_para_subj', 'z_sv_money_subj', 'z_sv_pain_para:sv_money_subj']
#         titles = [
#             'Starting point', 'Inter-trial variability in drift rate', 'Inter-trial variability in starting point', 'Inter-trial variability in non-decision time',
#             'Intercept drift rate', 'Drift rate sv_pain_para', 'Drift rate sv_money', 'Interaction v_sv_pain_para:sv_money_subj',
#             'Intercept non-decision time', 'Non-decision time sv_pain_para', 'Non-decision time sv_money', 'Interaction t_sv_pain_para:sv_money_subj',
#             'Intercept boundary separation', 'Boundary separation sv_pain_para', 'Boundary separation sv_money', 'Interaction a_sv_pain_para:sv_money_subj',
#             'Intercept Starting Point', 'Starting Point sv_pain_para', 'Starting Point sv_money', 'Starting Point a_sv_pain_para:sv_money_subj']
#     elif version == 11:
#         params_of_interest = ['z', 'a', 't', 'sv', 'sz', 'st', 'v_Intercept', 'v_sv_pain_para', 'v_sv_money']
#         params_of_interest_s = ['z_subj', 'a_subj', 't_subj', 'v_Intercept_subj', 'v_sv_pain_para_subj', 'v_sv_money_subj']
#         titles = ['Starting point', 'Boundary sep.', 'Non-dec. time', 
#             'Inter-trial variability in drift rate', 'Inter-trial variability in starting point', 'Inter-trial variability in non-decision time',
#             'v_Intercept rate', 'Drift v_sv_pain_para',  'Drift v_sv_money']
#     elif version == 12:
#         params_of_interest = [
#             'sv', 'sz', 'st', 
#             'v_Intercept', 'v_sv_pain_para', 'v_sv_money', 
#             't_Intercept', 't_sv_pain_para', 't_sv_money', 
#             'a_Intercept', 'a_sv_pain_para', 'a_sv_money',
#             ]
#         params_of_interest_s = [
#             'sv_subj', 'sz_subj', 'st_subj',
#             'v_Intercept_subj', 'v_sv_pain_para_subj', 'v_sv_money_subj',
#             't_Intercept_subj', 't_sv_pain_para_subj', 't_sv_money_subj',
#             ]
#         titles = [
#             'Starting point', 'Inter-trial variability in drift rate', 'Inter-trial variability in starting point', 'Inter-trial variability in non-decision time',
#             'Intercept drift rate', 'Drift rate sv_pain_para', 'Drift rate sv_money'
#             'Intercept non-decision time', 'Non-decision time sv_pain_para', 'Non-decision time sv_money',
#             'Intercept boundary separation', 'Boundary separation sv_pain_para', 'Boundary separation sv_money',
#             ] 
#     elif version == 13:
#         params_of_interest = ['z', 'a', 't', 'sv', 'sz', 'st', 'v_Intercept', 'v_sv_both']
#         params_of_interest_s = ['z_subj', 'a_subj', 't_subj', 'v_Intercept_subj', 'v_sv_both_subj']
#         titles = ['Starting point', 'Boundary sep.', 'Non-dec. time', 
#             'Inter-trial variability in drift rate', 'Inter-trial variability in starting point', 'Inter-trial variability in non-decision time', 
#             'v_Intercept rate', 'Drift v_sv_money'] 
#     elif version == 14:
#         params_of_interest = ['z', 'sv', 'sz', 'st','v_Intercept', 'v_sv_both', 'v_sv_pain_para', 'v_sv_money', 'v_sv_both:sv_pain_para:sv_money']
#         params_of_interest_s = ['z_subj', 'sv_subj', 'sz_subj', 'st_subj',
#                             'v_Intercept_subj', 'v_sv_both_subj', 'v_sv_pain_para_subj', 'v_sv_money_subj', 'v_sv_both:sv_pain_para:sv_money_subj']
#         titles = ['Starting point', 'Inter-trial variability in drift rate', 'Inter-trial variability in starting point', 'Inter-trial variability in non-decision time',
#                 'Intercept drift rate', 'Drift v_sv_both', 'Drift v_sv_pain_para', 'Drift v_sv_money', 'Drift rate (sv_both * sv_pain_para * sv_money)']
#     elif version == 15:  
#         params_of_interest = [
#             'sv', 'sz', 'st',
#             'v_Intercept', 'v_sv_both', 'v_sv_pain_para', 'v_sv_money', 'v_sv_both:sv_pain_para', 'v_sv_both:sv_money', 'v_sv_pain_para:sv_money', 'v_sv_both:sv_pain_para:sv_money',
#             't_Intercept', 't_sv_both', 't_sv_pain_para', 't_sv_money', 't_sv_both:sv_pain_para', 't_sv_both:sv_money', 't_sv_pain_para:sv_money', 't_sv_both:sv_pain_para:sv_money',
#             'a_Intercept', 'a_sv_both', 'a_sv_pain_para', 'a_sv_money', 'a_sv_both:sv_pain_para', 'a_sv_both:sv_money', 'a_sv_pain_para:sv_money', 'a_sv_both:sv_pain_para:sv_money',
#             'z_Intercept', 'z_sv_both', 'z_sv_pain_para', 'z_sv_money', 'z_sv_both:sv_pain_para', 'z_sv_both:sv_money', 'z_sv_pain_para:sv_money', 'z_sv_both:sv_pain_para:sv_money']
#         params_of_interest_s = [
#             'sv_subj', 'sz_subj', 'st_subj',
#             'v_Intercept_subj', 'v_sv_both_subj', 'v_sv_pain_para_subj', 'v_sv_money_subj', 'v_sv_both:sv_pain_para_subj', 'v_sv_both:sv_money_subj', 'v_sv_pain_para:sv_money_subj', 'v_sv_both:sv_pain_para:sv_money_subj',
#             't_Intercept_subj', 't_sv_both_subj', 't_sv_pain_para_subj', 't_sv_money_subj', 't_sv_both:sv_pain_para_subj', 't_sv_both:sv_money_subj', 't_sv_pain_para:sv_money_subj', 't_sv_both:sv_pain_para:sv_money_subj',
#             'a_Intercept_subj', 'a_sv_both_subj', 'a_sv_pain_para_subj', 'a_sv_money_subj', 'a_sv_both:sv_pain_para_subj', 'a_sv_both:sv_money_subj', 'a_sv_pain_para:sv_money_subj', 'a_sv_both:sv_pain_para:sv_money_subj',
#             'z_Intercept_subj', 'z_sv_both_subj', 'z_sv_pain_para_subj', 'z_sv_money_subj', 'z_sv_both:sv_pain_para_subj', 'z_sv_both:sv_money_subj', 'z_sv_pain_para:sv_money_subj', 'z_sv_both:sv_pain_para:sv_money_subj']
#         titles = [
#             'Starting point', 'Boundary separation', 'Non-decision time', 'Inter-trial variability in drift rate', 'Inter-trial variability in starting point', 'Inter-trial variability in non-decision time',
#             'Intercept drift rate', 'Drift v_sv_both', 'Drift v_sv_pain_para', 'Drift v_sv_money', 'Drift rate interaction (sv_both * sv_pain_para)', 'Drift rate interaction (sv_both * sv_money)', 'Drift rate interaction (sv_pain_para * sv_money)', 'Drift rate interaction (sv_both * sv_pain_para * sv_money)',
#             'Intercept non-decision time', 'Non-decision time sv_both', 'Non-decision time sv_pain_para', 'Non-decision time sv_money', 'Non-decision time interaction (sv_both * sv_pain_para)', 'Non-decision time interaction (sv_both * sv_money)', 'Non-decision time interaction (sv_pain_para * sv_money)', 'Non-decision time interaction (sv_both * sv_pain_para * sv_money)',
#             'Intercept boundary separation', 'Boundary separation sv_both', 'Boundary separation sv_pain_para', 'Boundary separation sv_money', 'Boundary separation interaction (sv_both * sv_pain_para)', 'Boundary separation interaction (sv_both * sv_money)', 'Boundary separation interaction (sv_pain_para * sv_money)', 'Boundary separation interaction (sv_both * sv_pain_para * sv_money)',
#             'Intercept starting point bias', 'Starting point bias sv_both', 'Starting point bias sv_pain_para', 'Starting point bias sv_money', 'Starting point bias interaction (sv_both * sv_pain_para)', 'Starting point bias interaction (sv_both * sv_money)', 'Starting point bias interaction (sv_pain_para * sv_money)', 'Starting point bias interaction (sv_both * sv_pain_para * sv_money)']
    elif version == 16:
        params_of_interest = [
            'z', 'a', 't', 'sv', 'sz', 'st',
            'v_Intercept',
            'v_sv_pain_para:C(Abs_value)[low_abs]', 'v_sv_pain_para:C(Abs_value)[high_abs]', 'v_sv_pain_para:C(Abs_value)[mid_abs]']
        params_of_interest_s = [
            'z_subj', 'a_subj', 't_subj', 'sv_subj', 'sz_subj', 'st_subj',
            'v_Intercept_subj', 
            'v_sv_pain_para:C(Abs_value)[low_abs]_subj', 'v_sv_pain_para:C(Abs_value)[high_abs]_subj', 'v_sv_pain_para:C(Abs_value)[mid_abs]_subj']
        titles = [
            'Starting point', 'Boundary separation', 'Non-decision time', 'Inter-trial variability in drift rate',
            'Inter-trial variability in starting point', 'Inter-trial variability in non-decision time',
            'Intercept drift rate',
            'Interaction: sv_pain_para * Abs_value[low_abs]', 'Interaction: sv_pain_para * Abs_value[high_abs]', 'Interaction: sv_pain_para * Abs_value[mid_abs]']
    elif version == 17:
        params_of_interest = [
            'z', 'a', 't', 'sv', 'sz', 'st',
            'v_Intercept',
            'v_sv_pain_para:C(OV_value)[low_OV]', 'v_sv_pain_para:C(OV_value)[high_OV]']
        params_of_interest_s = [
            'z_subj', 'a_subj', 't_subj', 'sv_subj', 'sz_subj', 'st_subj',
            'v_Intercept_subj', 
            'v_sv_pain_para:C(OV_value)[low_OV]_subj', 'v_sv_pain_para:C(OV_value)[high_OV]_subj']
        titles = [
            'Starting point', 'Boundary separation', 'Non-decision time', 'Inter-trial variability in drift rate',
            'Inter-trial variability in starting point', 'Inter-trial variability in non-decision time',
            'Intercept drift rate',
            'Interaction: sv_pain_para * OV_value[low_OV]', 'Interaction: sv_pain_para * OV_value[high_OV]']
        
        
#     elif version == 18:
#         params_of_interest = [
#             'z', 'a', 't', 'sv', 'sz', 'st',
#             'v_Intercept',
#             'v_sv_pain_para:C(Abs_Money_Pain)[low_abs_h_money]', 'v_sv_pain_para:C(Abs_Money_Pain)[low_abs_h_pain]',
#             'v_sv_pain_para:C(Abs_Money_Pain)[high_abs_h_money]', 'v_sv_pain_para:C(Abs_Money_Pain)[high_abs_h_pain]', 'v_sv_pain_para:C(Abs_Money_Pain)[mid_abs]']
#         params_of_interest_s = [
#             'z_subj', 'a_subj', 't_subj', 'sv_subj', 'sz_subj', 'st_subj',
#             'v_sv_pain_para:C(Abs_Money_Pain)[low_abs_h_money]_subj', 'v_sv_pain_para:C(Abs_Money_Pain)[low_abs_h_pain]_subj',
#             'v_sv_pain_para:C(Abs_Money_Pain)[high_abs_h_money]_subj', 'v_sv_pain_para:C(Abs_Money_Pain)[high_abs_h_pain]_subj', 'v_sv_pain_para:C(Abs_Money_Pain)[mid_abs]_subj']
#         titles = [
#             'Starting point', 'Boundary separation', 'Non-decision time', 'Inter-trial variability in drift rate',
#             'Inter-trial variability in starting point', 'Inter-trial variability in non-decision time',
#             'Intercept drift rate',
#             'Interaction: sv_pain_para * Abs_Money_Pain[low_abs_h_money]', 'Interaction: sv_pain_para * Abs_Money_Pain[low_abs_h_pain]',
#             'Interaction: sv_pain_para * Abs_Money_Pain[high_abs_h_money]', 'Interaction: sv_pain_para * Abs_Money_Pain[high_abs_h_pain]', 'Interaction: sv_pain_para * Abs_Money_Pain[mid_abs]']
#     elif version == 19:
#         params_of_interest = [
#             'z', 'a', 't', 'sv', 'sz', 'st',
#             'v_Intercept',
#             'v_sv_pain_para:C(OV_Money_Pain)[h_OV_h_money]', 'v_sv_pain_para:C(OV_Money_Pain)[h_OV_h_pain]',
#             'v_sv_pain_para:C(OV_Money_Pain)[low_OV_h_money]', 'v_sv_pain_para:C(OV_Money_Pain)[low_OV_h_pain]']
#         params_of_interest_s = [
#             'z_subj', 'a_subj', 't_subj', 'sv_subj', 'sz_subj', 'st_subj',
#             'v_Intercept_subj',
#             'v_sv_pain_para:C(OV_Money_Pain)[h_OV_h_money]_subj', 'v_sv_pain_para:C(OV_Money_Pain)[h_OV_h_pain]_subj',
#             'v_sv_pain_para:C(OV_Money_Pain)[low_OV_h_money]_subj', 'v_sv_pain_para:C(OV_Money_Pain)[low_OV_h_pain]_subj'
#             ]
#         titles = [
#         'Starting point', 'Boundary separation', 'Non-decision time', 'Inter-trial variability in drift rate',
#         'Inter-trial variability in starting point', 'Inter-trial variability in non-decision time',
#         'Intercept drift rate',
#         'Interaction: sv_pain_para * OV_Money_Pain[h_OV_h_money]', 'Interaction: sv_pain_para * OV_Money_Pain[h_OV_h_pain]',
#         'Interaction: sv_pain_para * OV_Money_Pain[low_OV_h_money]', 'Interaction: sv_pain_para * OV_Money_Pain[low_OV_h_pain]'
#     ]
#     elif version == 20:
#         params_of_interest = [
#             'z', 'a', 't', 'sv', 'sz', 'st',
#             'v_sv_pain_para:C(OV_Money_Pain)[h_OV_h_money]:C(Abs_Money_Pain)[low_abs_h_money]',
#             'v_sv_pain_para:C(OV_Money_Pain)[h_OV_h_money]:C(Abs_Money_Pain)[low_abs_h_pain]',
#             'v_sv_pain_para:C(OV_Money_Pain)[h_OV_h_money]:C(Abs_Money_Pain)[high_abs_h_money]',
#             'v_sv_pain_para:C(OV_Money_Pain)[h_OV_h_money]:C(Abs_Money_Pain)[high_abs_h_pain]',
#             'v_sv_pain_para:C(OV_Money_Pain)[h_OV_h_money]:C(Abs_Money_Pain)[mid_abs]',
#             'v_sv_pain_para:C(OV_Money_Pain)[h_OV_h_pain]:C(Abs_Money_Pain)[low_abs_h_money]',
#             'v_sv_pain_para:C(OV_Money_Pain)[h_OV_h_pain]:C(Abs_Money_Pain)[low_abs_h_pain]',
#             'v_sv_pain_para:C(OV_Money_Pain)[h_OV_h_pain]:C(Abs_Money_Pain)[high_abs_h_money]',
#             'v_sv_pain_para:C(OV_Money_Pain)[h_OV_h_pain]:C(Abs_Money_Pain)[high_abs_h_pain]',
#             'v_sv_pain_para:C(OV_Money_Pain)[h_OV_h_pain]:C(Abs_Money_Pain)[mid_abs]',
#             'v_sv_pain_para:C(OV_Money_Pain)[low_OV_h_money]:C(Abs_Money_Pain)[low_abs_h_money]',
#             'v_sv_pain_para:C(OV_Money_Pain)[low_OV_h_money]:C(Abs_Money_Pain)[low_abs_h_pain]',
#             'v_sv_pain_para:C(OV_Money_Pain)[low_OV_h_money]:C(Abs_Money_Pain)[high_abs_h_money]',
#             'v_sv_pain_para:C(OV_Money_Pain)[low_OV_h_money]:C(Abs_Money_Pain)[high_abs_h_pain]',
#             'v_sv_pain_para:C(OV_Money_Pain)[low_OV_h_money]:C(Abs_Money_Pain)[mid_abs]',
#             'v_sv_pain_para:C(OV_Money_Pain)[low_OV_h_pain]:C(Abs_Money_Pain)[low_abs_h_money]',
#             'v_sv_pain_para:C(OV_Money_Pain)[low_OV_h_pain]:C(Abs_Money_Pain)[low_abs_h_pain]',
#             'v_sv_pain_para:C(OV_Money_Pain)[low_OV_h_pain]:C(Abs_Money_Pain)[high_abs_h_money]',
#             'v_sv_pain_para:C(OV_Money_Pain)[low_OV_h_pain]:C(Abs_Money_Pain)[high_abs_h_pain]',
#             'v_sv_pain_para:C(OV_Money_Pain)[low_OV_h_pain]:C(Abs_Money_Pain)[mid_abs]'
#         ]
#         params_of_interest_s = [
#             'z_subj', 'a_subj', 't_subj', 'sv_subj', 'sz_subj', 'st_subj',
#             'v_sv_pain_para:C(OV_Money_Pain)[h_OV_h_money]:C(Abs_Money_Pain)[low_abs_h_money]_subj',
#             'v_sv_pain_para:C(OV_Money_Pain)[h_OV_h_money]:C(Abs_Money_Pain)[low_abs_h_pain]_subj',
#             'v_sv_pain_para:C(OV_Money_Pain)[h_OV_h_money]:C(Abs_Money_Pain)[high_abs_h_money]_subj',
#             'v_sv_pain_para:C(OV_Money_Pain)[h_OV_h_money]:C(Abs_Money_Pain)[high_abs_h_pain]_subj',
#             'v_sv_pain_para:C(OV_Money_Pain)[h_OV_h_money]:C(Abs_Money_Pain)[mid_abs]_subj',
#             'v_sv_pain_para:C(OV_Money_Pain)[h_OV_h_pain]:C(Abs_Money_Pain)[low_abs_h_money]_subj',
#             'v_sv_pain_para:C(OV_Money_Pain)[h_OV_h_pain]:C(Abs_Money_Pain)[low_abs_h_pain]_subj',
#             'v_sv_pain_para:C(OV_Money_Pain)[h_OV_h_pain]:C(Abs_Money_Pain)[high_abs_h_money]_subj',
#             'v_sv_pain_para:C(OV_Money_Pain)[h_OV_h_pain]:C(Abs_Money_Pain)[high_abs_h_pain]_subj',
#             'v_sv_pain_para:C(OV_Money_Pain)[h_OV_h_pain]:C(Abs_Money_Pain)[mid_abs]_subj',
#             'v_sv_pain_para:C(OV_Money_Pain)[low_OV_h_money]:C(Abs_Money_Pain)[low_abs_h_money]_subj',
#             'v_sv_pain_para:C(OV_Money_Pain)[low_OV_h_money]:C(Abs_Money_Pain)[low_abs_h_pain]_subj',
#             'v_sv_pain_para:C(OV_Money_Pain)[low_OV_h_money]:C(Abs_Money_Pain)[high_abs_h_money]_subj',
#             'v_sv_pain_para:C(OV_Money_Pain)[low_OV_h_money]:C(Abs_Money_Pain)[high_abs_h_pain]_subj',
#             'v_sv_pain_para:C(OV_Money_Pain)[low_OV_h_money]:C(Abs_Money_Pain)[mid_abs]_subj',
#             'v_sv_pain_para:C(OV_Money_Pain)[low_OV_h_pain]:C(Abs_Money_Pain)[low_abs_h_money]_subj',
#             'v_sv_pain_para:C(OV_Money_Pain)[low_OV_h_pain]:C(Abs_Money_Pain)[low_abs_h_pain]_subj',
#             'v_sv_pain_para:C(OV_Money_Pain)[low_OV_h_pain]:C(Abs_Money_Pain)[high_abs_h_money]_subj',
#             'v_sv_pain_para:C(OV_Money_Pain)[low_OV_h_pain]:C(Abs_Money_Pain)[high_abs_h_pain]_subj',
#             'v_sv_pain_para:C(OV_Money_Pain)[low_OV_h_pain]:C(Abs_Money_Pain)[mid_abs]_subj'
#         ]
#         titles = [
#         'Starting point', 'Boundary separation', 'Non-decision time', 'Inter-trial variability in drift rate',
#         'Inter-trial variability in starting point', 'Inter-trial variability in non-decision time',
#         'v_sv_pain_para * OV[h_OV_h_money] * Abs[low_abs_h_money]',
#         'v_sv_pain_para * OV[h_OV_h_money] * Abs[low_abs_h_pain]',
#         'v_sv_pain_para * OV[h_OV_h_money] * Abs[high_abs_h_money]',
#         'v_sv_pain_para * OV[h_OV_h_money] * Abs[high_abs_h_pain]',
#         'v_sv_pain_para * OV[h_OV_h_money] * Abs[mid_abs]',
#         'v_sv_pain_para * OV[h_OV_h_pain] * Abs[low_abs_h_money]',
#         'v_sv_pain_para * OV[h_OV_h_pain] * Abs[low_abs_h_pain]',
#         'v_sv_pain_para * OV[h_OV_h_pain] * Abs[high_abs_h_money]',
#         'v_sv_pain_para * OV[h_OV_h_pain] * Abs[high_abs_h_pain]',
#         'v_sv_pain_para * OV[h_OV_h_pain] * Abs[mid_abs]',
#         'v_sv_pain_para * OV[low_OV_h_money] * Abs[low_abs_h_money]',
#         'v_sv_pain_para * OV[low_OV_h_money] * Abs[low_abs_h_pain]',
#         'v_sv_pain_para * OV[low_OV_h_money] * Abs[high_abs_h_money]',
#         'v_sv_pain_para * OV[low_OV_h_money] * Abs[high_abs_h_pain]',
#         'v_sv_pain_para * OV[low_OV_h_money] * Abs[mid_abs]',
#         'v_sv_pain_para * OV[low_OV_h_pain] * Abs[low_abs_h_money]',
#         'v_sv_pain_para * OV[low_OV_h_pain] * Abs[low_abs_h_pain]',
#         'v_sv_pain_para * OV[low_OV_h_pain] * Abs[high_abs_h_money]',
#         'v_sv_pain_para * OV[low_OV_h_pain] * Abs[high_abs_h_pain]',
#         'v_sv_pain_para * OV[low_OV_h_pain] * Abs[mid_abs]'
#         ]
#     elif version == 21:
#         params_of_interest = [
#             'z', 'a', 't', 'sv', 'sz', 'st',
#             'v_sv_pain_para:C(Abs_value)[low_abs]:C(acceptance_pair)[P]',
#             'v_sv_pain_para:C(Abs_value)[low_abs]:C(acceptance_pair)[M]',
#             'v_sv_pain_para:C(Abs_value)[low_abs]:C(acceptance_pair)[I]',
#             'v_sv_pain_para:C(Abs_value)[mid_abs]:C(acceptance_pair)[P]',
#             'v_sv_pain_para:C(Abs_value)[mid_abs]:C(acceptance_pair)[M]',
#             'v_sv_pain_para:C(Abs_value)[mid_abs]:C(acceptance_pair)[I]',
#             'v_sv_pain_para:C(Abs_value)[high_abs]:C(acceptance_pair)[P]',
#             'v_sv_pain_para:C(Abs_value)[high_abs]:C(acceptance_pair)[M]',
#             'v_sv_pain_para:C(Abs_value)[high_abs]:C(acceptance_pair)[I]']
#         params_of_interest_s = [
#             'z_subj', 'a_subj', 't_subj', 'sv_subj', 'sz_subj', 'st_subj',
#             'v_sv_pain_para:C(Abs_value)[low_abs]:C(acceptance_pair)[P]_subj',
#             'v_sv_pain_para:C(Abs_value)[low_abs]:C(acceptance_pair)[M]_subj',
#             'v_sv_pain_para:C(Abs_value)[low_abs]:C(acceptance_pair)[I]_subj',
#             'v_sv_pain_para:C(Abs_value)[mid_abs]:C(acceptance_pair)[P]_subj',
#             'v_sv_pain_para:C(Abs_value)[mid_abs]:C(acceptance_pair)[M]_subj',
#             'v_sv_pain_para:C(Abs_value)[mid_abs]:C(acceptance_pair)[I]_subj',
#             'v_sv_pain_para:C(Abs_value)[high_abs]:C(acceptance_pair)[P]_subj',
#             'v_sv_pain_para:C(Abs_value)[high_abs]:C(acceptance_pair)[M]_subj',
#             'v_sv_pain_para:C(Abs_value)[high_abs]:C(acceptance_pair)[I]_subj']
#         titles = [
#             'Starting point', 'Boundary separation', 'Non-decision time', 'Inter-trial variability in drift rate',
#             'Inter-trial variability in starting point', 'Inter-trial variability in non-decision time',
#             'v_sv_pain_para * Abs_value[low_abs] * acceptance_pair[P]',
#             'v_sv_pain_para * Abs_value[low_abs] * acceptance_pair[M]',
#             'v_sv_pain_para * Abs_value[low_abs] * acceptance_pair[I]',
#             'v_sv_pain_para * Abs_value[mid_abs] * acceptance_pair[P]',
#             'v_sv_pain_para * Abs_value[mid_abs] * acceptance_pair[M]',
#             'v_sv_pain_para * Abs_value[mid_abs] * acceptance_pair[I]',
#             'v_sv_pain_para * Abs_value[high_abs] * acceptance_pair[P]',
#             'v_sv_pain_para * Abs_value[high_abs] * acceptance_pair[M]',
#             'v_sv_pain_para * Abs_value[high_abs] * acceptance_pair[I]']
#     elif version == 22:
#         params_of_interest = [
#         'z', 'a', 't', 'sv', 'sz', 'st',
#         'v_sv_pain_para:C(OV_value)[low_OV]:C(acceptance_pair)[P]',
#         'v_sv_pain_para:C(OV_value)[low_OV]:C(acceptance_pair)[M]',
#         'v_sv_pain_para:C(OV_value)[low_OV]:C(acceptance_pair)[I]',
#         'v_sv_pain_para:C(OV_value)[high_OV]:C(acceptance_pair)[P]',
#         'v_sv_pain_para:C(OV_value)[high_OV]:C(acceptance_pair)[M]',
#         'v_sv_pain_para:C(OV_value)[high_OV]:C(acceptance_pair)[I]'
#         ]
#         params_of_interest_s = [
#         'z_subj', 'a_subj', 't_subj', 'sv_subj', 'sz_subj', 'st_subj',
#         'v_sv_pain_para:C(OV_value)[low_OV]:C(acceptance_pair)[P]_subj',
#         'v_sv_pain_para:C(OV_value)[low_OV]:C(acceptance_pair)[M]_subj',
#         'v_sv_pain_para:C(OV_value)[low_OV]:C(acceptance_pair)[I]_subj',
#         'v_sv_pain_para:C(OV_value)[high_OV]:C(acceptance_pair)[P]_subj',
#         'v_sv_pain_para:C(OV_value)[high_OV]:C(acceptance_pair)[M]_subj',
#         'v_sv_pain_para:C(OV_value)[high_OV]:C(acceptance_pair)[I]_subj'
#         ]
#         titles = [
#         'Starting point', 'Boundary separation', 'Non-decision time', 'Inter-trial variability in drift rate',
#         'Inter-trial variability in starting point', 'Inter-trial variability in non-decision time',
#         'v_sv_pain_para * OV_value[low_OV] * acceptance_pair[P]',
#         'v_sv_pain_para * OV_value[low_OV] * acceptance_pair[M]',
#         'v_sv_pain_para * OV_value[low_OV] * acceptance_pair[I]',
#         'v_sv_pain_para * OV_value[high_OV] * acceptance_pair[P]',
#         'v_sv_pain_para * OV_value[high_OV] * acceptance_pair[M]',
#         'v_sv_pain_para * OV_value[high_OV] * acceptance_pair[I]'
#         ]
#     elif version == 23:
#         params_of_interest = [
#         'z', 'a', 't', 'sv', 'sz', 'st',
#         'v_sv_pain_para:C(OV_value)[low_OV]:C(Abs_value)[low_abs]',
#         'v_sv_pain_para:C(OV_value)[low_OV]:C(Abs_value)[mid_abs]',
#         'v_sv_pain_para:C(OV_value)[low_OV]:C(Abs_value)[high_abs]',
#         'v_sv_pain_para:C(OV_value)[high_OV]:C(Abs_value)[low_abs]',
#         'v_sv_pain_para:C(OV_value)[high_OV]:C(Abs_value)[mid_abs]',
#         'v_sv_pain_para:C(OV_value)[high_OV]:C(Abs_value)[high_abs]'
#         ]
#         params_of_interest_s = [
#         'z_subj', 'a_subj', 't_subj', 'sv_subj', 'sz_subj', 'st_subj',
#         'v_sv_pain_para:C(OV_value)[low_OV]:C(Abs_value)[low_abs]_subj',
#         'v_sv_pain_para:C(OV_value)[low_OV]:C(Abs_value)[mid_abs]_subj',
#         'v_sv_pain_para:C(OV_value)[low_OV]:C(Abs_value)[high_abs]_subj',
#         'v_sv_pain_para:C(OV_value)[high_OV]:C(Abs_value)[low_abs]_subj',
#         'v_sv_pain_para:C(OV_value)[high_OV]:C(Abs_value)[mid_abs]_subj',
#         'v_sv_pain_para:C(OV_value)[high_OV]:C(Abs_value)[high_abs]_subj'
#         ]
#         titles = [
#         'Starting point', 'Boundary separation', 'Non-decision time', 'Inter-trial variability in drift rate',
#         'Inter-trial variability in starting point', 'Inter-trial variability in non-decision time',
#         'v_sv_pain_para * OV_value[low_OV] * Abs_value[low_abs]',
#         'v_sv_pain_para * OV_value[low_OV] * Abs_value[mid_abs]',
#         'v_sv_pain_para * OV_value[low_OV] * Abs_value[high_abs]',
#         'v_sv_pain_para * OV_value[high_OV] * Abs_value[low_abs]',
#         'v_sv_pain_para * OV_value[high_OV] * Abs_value[mid_abs]',
#         'v_sv_pain_para * OV_value[high_OV] * Abs_value[high_abs]'
#         ]
#     elif version == 24:
#         params_of_interest = [
#         'z', 'a', 't', 'sv', 'sz', 'st',
#         'v_sv_pain_para:C(OV_value)[low_OV]:C(Abs_value)[low_abs]:C(acceptance_pair)[P]',
#         'v_sv_pain_para:C(OV_value)[low_OV]:C(Abs_value)[low_abs]:C(acceptance_pair)[M]',
#         'v_sv_pain_para:C(OV_value)[low_OV]:C(Abs_value)[low_abs]:C(acceptance_pair)[I]',
#         'v_sv_pain_para:C(OV_value)[low_OV]:C(Abs_value)[mid_abs]:C(acceptance_pair)[P]',
#         'v_sv_pain_para:C(OV_value)[low_OV]:C(Abs_value)[mid_abs]:C(acceptance_pair)[M]',
#         'v_sv_pain_para:C(OV_value)[low_OV]:C(Abs_value)[mid_abs]:C(acceptance_pair)[I]',
#         'v_sv_pain_para:C(OV_value)[low_OV]:C(Abs_value)[high_abs]:C(acceptance_pair)[P]',
#         'v_sv_pain_para:C(OV_value)[low_OV]:C(Abs_value)[high_abs]:C(acceptance_pair)[M]',
#         'v_sv_pain_para:C(OV_value)[low_OV]:C(Abs_value)[high_abs]:C(acceptance_pair)[I]',
#         'v_sv_pain_para:C(OV_value)[high_OV]:C(Abs_value)[low_abs]:C(acceptance_pair)[P]',
#         'v_sv_pain_para:C(OV_value)[high_OV]:C(Abs_value)[low_abs]:C(acceptance_pair)[M]',
#         'v_sv_pain_para:C(OV_value)[high_OV]:C(Abs_value)[low_abs]:C(acceptance_pair)[I]',
#         'v_sv_pain_para:C(OV_value)[high_OV]:C(Abs_value)[mid_abs]:C(acceptance_pair)[P]',
#         'v_sv_pain_para:C(OV_value)[high_OV]:C(Abs_value)[mid_abs]:C(acceptance_pair)[M]',
#         'v_sv_pain_para:C(OV_value)[high_OV]:C(Abs_value)[mid_abs]:C(acceptance_pair)[I]',
#         'v_sv_pain_para:C(OV_value)[high_OV]:C(Abs_value)[high_abs]:C(acceptance_pair)[P]',
#         'v_sv_pain_para:C(OV_value)[high_OV]:C(Abs_value)[high_abs]:C(acceptance_pair)[M]',
#         'v_sv_pain_para:C(OV_value)[high_OV]:C(Abs_value)[high_abs]:C(acceptance_pair)[I]'
#         ]
#         params_of_interest_s = [
#         'z_subj', 'a_subj', 't_subj', 'sv_subj', 'sz_subj', 'st_subj',
#         'v_sv_pain_para:C(OV_value)[low_OV]:C(Abs_value)[low_abs]:C(acceptance_pair)[P]_subj',
#         'v_sv_pain_para:C(OV_value)[low_OV]:C(Abs_value)[low_abs]:C(acceptance_pair)[M]_subj',
#         'v_sv_pain_para:C(OV_value)[low_OV]:C(Abs_value)[low_abs]:C(acceptance_pair)[I]_subj',
#         'v_sv_pain_para:C(OV_value)[low_OV]:C(Abs_value)[mid_abs]:C(acceptance_pair)[P]_subj',
#         'v_sv_pain_para:C(OV_value)[low_OV]:C(Abs_value)[mid_abs]:C(acceptance_pair)[M]_subj',
#         'v_sv_pain_para:C(OV_value)[low_OV]:C(Abs_value)[mid_abs]:C(acceptance_pair)[I]_subj',
#         'v_sv_pain_para:C(OV_value)[low_OV]:C(Abs_value)[high_abs]:C(acceptance_pair)[P]_subj',
#         'v_sv_pain_para:C(OV_value)[low_OV]:C(Abs_value)[high_abs]:C(acceptance_pair)[M]_subj',
#         'v_sv_pain_para:C(OV_value)[low_OV]:C(Abs_value)[high_abs]:C(acceptance_pair)[I]_subj',
#         'v_sv_pain_para:C(OV_value)[high_OV]:C(Abs_value)[low_abs]:C(acceptance_pair)[P]_subj',
#         'v_sv_pain_para:C(OV_value)[high_OV]:C(Abs_value)[low_abs]:C(acceptance_pair)[M]_subj',
#         'v_sv_pain_para:C(OV_value)[high_OV]:C(Abs_value)[low_abs]:C(acceptance_pair)[I]_subj',
#         'v_sv_pain_para:C(OV_value)[high_OV]:C(Abs_value)[mid_abs]:C(acceptance_pair)[P]_subj',
#         'v_sv_pain_para:C(OV_value)[high_OV]:C(Abs_value)[mid_abs]:C(acceptance_pair)[M]_subj',
#         'v_sv_pain_para:C(OV_value)[high_OV]:C(Abs_value)[mid_abs]:C(acceptance_pair)[I]_subj',
#         'v_sv_pain_para:C(OV_value)[high_OV]:C(Abs_value)[high_abs]:C(acceptance_pair)[P]_subj',
#         'v_sv_pain_para:C(OV_value)[high_OV]:C(Abs_value)[high_abs]:C(acceptance_pair)[M]_subj',
#         'v_sv_pain_para:C(OV_value)[high_OV]:C(Abs_value)[high_abs]:C(acceptance_pair)[I]_subj'
#         ]
#         titles = [
#         'Starting point', 'Boundary separation', 'Non-decision time', 'Inter-trial variability in drift rate',
#         'Inter-trial variability in starting point', 'Inter-trial variability in non-decision time',
#         'v_sv_pain_para*OV_value[low_OV]*Abs_value[low_abs]*acceptance_pair[P]',
#         'v_sv_pain_para*OV_value[low_OV]*Abs_value[low_abs]*acceptance_pair[M]',
#         'v_sv_pain_para*OV_value[low_OV]*Abs_value[low_abs]*acceptance_pair[I]',
#         'v_sv_pain_para*OV_value[low_OV]*Abs_value[mid_abs]*acceptance_pair[P]',
#         'v_sv_pain_para*OV_value[low_OV]*Abs_value[mid_abs]*acceptance_pair[M]',
#         'v_sv_pain_para*OV_value[low_OV]*Abs_value[mid_abs]*acceptance_pair[I]',
#         'v_sv_pain_para*OV_value[low_OV]*Abs_value[high_abs]*acceptance_pair[P]',
#         'v_sv_pain_para*OV_value[low_OV]*Abs_value[high_abs]*acceptance_pair[M]',
#         'v_sv_pain_para*OV_value[low_OV]*Abs_value[high_abs]*acceptance_pair[I]',
#         'v_sv_pain_para*OV_value[high_OV]*Abs_value[low_abs]*acceptance_pair[P]',
#         'v_sv_pain_para*OV_value[high_OV]*Abs_value[low_abs]*acceptance_pair[M]',
#         'v_sv_pain_para*OV_value[high_OV]*Abs_value[low_abs]*acceptance_pair[I]',
#         'v_sv_pain_para*OV_value[high_OV]*Abs_value[mid_abs]*acceptance_pair[P]',
#         'v_sv_pain_para*OV_value[high_OV]*Abs_value[mid_abs]*acceptance_pair[M]',
#         'v_sv_pain_para*OV_value[high_OV]*Abs_value[mid_abs]*acceptance_pair[I]',
#         'v_sv_pain_para*OV_value[high_OV]*Abs_value[high_abs]*acceptance_pair[P]',
#         'v_sv_pain_para*OV_value[high_OV]*Abs_value[high_abs]*acceptance_pair[M]',
#         'v_sv_pain_para*OV_value[high_OV]*Abs_value[high_abs]*acceptance_pair[I]'
#         ]
    elif version == 25:
        params_of_interest = [
        'z', 'a', 't', 'sv', 'sz', 'st',
        'v_sv_pain_para:C(OV_value)[low_OV]', 'v_sv_pain_para:C(OV_value)[high_OV]',
        'v_sv_pain_para:C(acceptance_pair)[P]', 'v_sv_pain_para:C(acceptance_pair)[M]', 'v_sv_pain_para:C(acceptance_pair)[I]',
        'v_C(OV_value)[low_OV]:C(acceptance_pair)[P]', 'v_C(OV_value)[low_OV]:C(acceptance_pair)[M]', 'v_C(OV_value)[low_OV]:C(acceptance_pair)[I]',
        'v_C(OV_value)[high_OV]:C(acceptance_pair)[P]', 'v_C(OV_value)[high_OV]:C(acceptance_pair)[M]', 'v_C(OV_value)[high_OV]:C(acceptance_pair)[I]',
        'v_sv_pain_para:C(OV_value)[low_OV]:C(acceptance_pair)[P]', 'v_sv_pain_para:C(OV_value)[low_OV]:C(acceptance_pair)[M]', 'v_sv_pain_para:C(OV_value)[low_OV]:C(acceptance_pair)[I]',
        'v_sv_pain_para:C(OV_value)[high_OV]:C(acceptance_pair)[P]', 'v_sv_pain_para:C(OV_value)[high_OV]:C(acceptance_pair)[M]', 'v_sv_pain_para:C(OV_value)[high_OV]:C(acceptance_pair)[I]'
        ]
        params_of_interest_s = [
        'z_subj', 'a_subj', 't_subj', 'sv_subj', 'sz_subj', 'st_subj',
        'v_sv_pain_para:C(OV_value)[low_OV]_subj', 'v_sv_pain_para:C(OV_value)[high_OV]_subj',
        'v_sv_pain_para:C(acceptance_pair)[P]_subj', 'v_sv_pain_para:C(acceptance_pair)[M]_subj', 'v_sv_pain_para:C(acceptance_pair)[I]_subj',
        'v_C(OV_value)[low_OV]:C(acceptance_pair)[P]_subj', 'v_C(OV_value)[low_OV]:C(acceptance_pair)[M]_subj', 'v_C(OV_value)[low_OV]:C(acceptance_pair)[I]_subj',
        'v_C(OV_value)[high_OV]:C(acceptance_pair)[P]_subj', 'v_C(OV_value)[high_OV]:C(acceptance_pair)[M]_subj', 'v_C(OV_value)[high_OV]:C(acceptance_pair)[I]_subj',
        'v_sv_pain_para:C(OV_value)[low_OV]:C(acceptance_pair)[P]_subj', 'v_sv_pain_para:C(OV_value)[low_OV]:C(acceptance_pair)[M]_subj', 'v_sv_pain_para:C(OV_value)[low_OV]:C(acceptance_pair)[I]_subj',
        'v_sv_pain_para:C(OV_value)[high_OV]:C(acceptance_pair)[P]_subj', 'v_sv_pain_para:C(OV_value)[high_OV]:C(acceptance_pair)[M]_subj', 'v_sv_pain_para:C(OV_value)[high_OV]:C(acceptance_pair)[I]_subj'
        ]
        titles = [
        'Starting Point', 'Boundary separation', 'Non-decision time', 'Inter-trial variability in drift rate',
        'Inter-trial variability in starting point', 'Inter-trial variability in non-decision time',
        'v_sv_pain_para * OV_value[low_OV]', 'v_sv_pain_para * OV_value[high_OV]',
        'v_sv_pain_para * acceptance_pair[P]', 'v_sv_pain_para * acceptance_pair[M]', 'v_sv_pain_para * acceptance_pair[I]',
        'OV_value[low_OV] * acceptance_pair[P]', 'OV_value[low_OV] * acceptance_pair[M]', 'OV_value[low_OV] * acceptance_pair[I]',
        'OV_value[high_OV] * acceptance_pair[P]', 'OV_value[high_OV] * acceptance_pair[M]', 'OV_value[high_OV] * acceptance_pair[I]',
        'v_sv_pain_para * OV_value[low_OV] * acceptance_pair[P]', 'v_sv_pain_para * OV_value[low_OV] * acceptance_pair[M]',
        'v_sv_pain_para * OV_value[low_OV] * acceptance_pair[I]', 'v_sv_pain_para * OV_value[high_OV] * acceptance_pair[P]',
        'v_sv_pain_para * OV_value[high_OV] * acceptance_pair[M]', 'v_sv_pain_para * OV_value[high_OV] * acceptance_pair[I]'
        ]
#     elif version == 26:
#         params_of_interest = [
#         'v_sv_pain_para:C(Abs_value)[low_abs]', 'v_sv_pain_para:C(Abs_value)[mid_abs]', 'v_sv_pain_para:C(Abs_value)[high_abs]',
#         't_sv_pain_para:C(Abs_value)[low_abs]', 't_sv_pain_para:C(Abs_value)[mid_abs]', 't_sv_pain_para:C(Abs_value)[high_abs]',
#         'a_sv_pain_para:C(Abs_value)[low_abs]', 'a_sv_pain_para:C(Abs_value)[mid_abs]', 'a_sv_pain_para:C(Abs_value)[high_abs]',
#         'z_sv_pain_para:C(Abs_value)[low_abs]', 'z_sv_pain_para:C(Abs_value)[mid_abs]', 'z_sv_pain_para:C(Abs_value)[high_abs]'
#         ]
#         params_of_interest_s = [
#         'v_sv_pain_para:C(Abs_value)[low_abs]_subj', 'v_sv_pain_para:C(Abs_value)[mid_abs]_subj', 'v_sv_pain_para:C(Abs_value)[high_abs]_subj',
#         't_sv_pain_para:C(Abs_value)[low_abs]_subj', 't_sv_pain_para:C(Abs_value)[mid_abs]_subj', 't_sv_pain_para:C(Abs_value)[high_abs]_subj',
#         'a_sv_pain_para:C(Abs_value)[low_abs]_subj', 'a_sv_pain_para:C(Abs_value)[mid_abs]_subj', 'a_sv_pain_para:C(Abs_value)[high_abs]_subj',
#         'z_sv_pain_para:C(Abs_value)[low_abs]_subj', 'z_sv_pain_para:C(Abs_value)[mid_abs]_subj', 'z_sv_pain_para:C(Abs_value)[high_abs]_subj'
#         ]
#         titles = [
#         'Drift rate interaction: sv_pain_para * Abs_value[low_abs]', 'Drift rate interaction: sv_pain_para * Abs_value[mid_abs]', 'Drift rate interaction: sv_pain_para * Abs_value[high_abs]',
#         'Non-decision time interaction: sv_pain_para * Abs_value[low_abs]', 'Non-decision time interaction: sv_pain_para * Abs_value[mid_abs]', 'Non-decision time interaction: sv_pain_para * Abs_value[high_abs]',
#         'Boundary separation interaction: sv_pain_para * Abs_value[low_abs]', 'Boundary separation interaction: sv_pain_para * Abs_value[mid_abs]', 'Boundary separation interaction: sv_pain_para * Abs_value[high_abs]',
#         'Starting point interaction: sv_pain_para * Abs_value[low_abs]', 'Starting point interaction: sv_pain_para * Abs_value[mid_abs]', 'Starting point interaction: sv_pain_para * Abs_value[high_abs]'
#         ]
#     elif version == 27:
#         params_of_interest = [
#         'z', 'a', 't', 'sv', 'sz', 'st',
#         'v_sv_pain_para:C(OV_value)[low_OV]', 'v_sv_pain_para:C(OV_value)[high_OV]',
#         't_sv_pain_para:C(OV_value)[low_OV]', 't_sv_pain_para:C(OV_value)[high_OV]',
#         'a_sv_pain_para:C(OV_value)[low_OV]', 'a_sv_pain_para:C(OV_value)[high_OV]',
#         'z_sv_pain_para:C(OV_value)[low_OV]', 'z_sv_pain_para:C(OV_value)[high_OV]'
#         ]
#         params_of_interest_s = [
#         'z_subj', 'a_subj', 't_subj', 'sv_subj', 'sz_subj', 'st_subj',
#         'v_sv_pain_para:C(OV_value)[low_OV]_subj', 'v_sv_pain_para:C(OV_value)[high_OV]_subj',
#         't_sv_pain_para:C(OV_value)[low_OV]_subj', 't_sv_pain_para:C(OV_value)[high_OV]_subj',
#         'a_sv_pain_para:C(OV_value)[low_OV]_subj', 'a_sv_pain_para:C(OV_value)[high_OV]_subj',
#         'z_sv_pain_para:C(OV_value)[low_OV]_subj', 'z_sv_pain_para:C(OV_value)[high_OV]_subj'
#         ]
#         titles = [
#         'Starting point', 'Boundary separation', 'Non-decision time', 'Inter-trial variability in drift rate',
#         'Inter-trial variability in starting point', 'Inter-trial variability in non-decision time',
#         'Drift rate interaction: sv_pain_para * OV_value[low_OV]', 'Drift rate interaction: sv_pain_para * OV_value[high_OV]',
#         'Non-decision time interaction: sv_pain_para * OV_value[low_OV]', 'Non-decision time interaction: sv_pain_para * OV_value[high_OV]',
#         'Boundary separation interaction: sv_pain_para * OV_value[low_OV]', 'Boundary separation interaction: sv_pain_para * OV_value[high_OV]',
#         'Starting point interaction: sv_pain_para * OV_value[low_OV]', 'Starting point interaction: sv_pain_para * OV_value[high_OV]'
#         ]
#     elif version == 28:
#         params_of_interest = [
#         'z', 'a', 't', 'sv', 'sz', 'st',
#         'v_sv_pain_para:C(OV_value)[low_OV]:C(acceptance_pair)[P]', 'v_sv_pain_para:C(OV_value)[low_OV]:C(acceptance_pair)[M]', 'v_sv_pain_para:C(OV_value)[low_OV]:C(acceptance_pair)[I]',
#         'v_sv_pain_para:C(OV_value)[high_OV]:C(acceptance_pair)[P]', 'v_sv_pain_para:C(OV_value)[high_OV]:C(acceptance_pair)[M]', 'v_sv_pain_para:C(OV_value)[high_OV]:C(acceptance_pair)[I]',
#         't_sv_pain_para:C(OV_value)[low_OV]:C(acceptance_pair)[P]', 't_sv_pain_para:C(OV_value)[low_OV]:C(acceptance_pair)[M]', 't_sv_pain_para:C(OV_value)[low_OV]:C(acceptance_pair)[I]',
#         't_sv_pain_para:C(OV_value)[high_OV]:C(acceptance_pair)[P]', 't_sv_pain_para:C(OV_value)[high_OV]:C(acceptance_pair)[M]', 't_sv_pain_para:C(OV_value)[high_OV]:C(acceptance_pair)[I]',
#         'a_sv_pain_para:C(OV_value)[low_OV]:C(acceptance_pair)[P]', 'a_sv_pain_para:C(OV_value)[low_OV]:C(acceptance_pair)[M]', 'a_sv_pain_para:C(OV_value)[low_OV]:C(acceptance_pair)[I]',
#         'a_sv_pain_para:C(OV_value)[high_OV]:C(acceptance_pair)[P]', 'a_sv_pain_para:C(OV_value)[high_OV]:C(acceptance_pair)[M]', 'a_sv_pain_para:C(OV_value)[high_OV]:C(acceptance_pair)[I]',
#         'z_sv_pain_para:C(OV_value)[low_OV]:C(acceptance_pair)[P]', 'z_sv_pain_para:C(OV_value)[low_OV]:C(acceptance_pair)[M]', 'z_sv_pain_para:C(OV_value)[low_OV]:C(acceptance_pair)[I]',
#         'z_sv_pain_para:C(OV_value)[high_OV]:C(acceptance_pair)[P]', 'z_sv_pain_para:C(OV_value)[high_OV]:C(acceptance_pair)[M]', 'z_sv_pain_para:C(OV_value)[high_OV]:C(acceptance_pair)[I]'
#         ]
#         params_of_interest_s = [
#         'z_subj', 'a_subj', 't_subj', 'sv_subj', 'sz_subj', 'st_subj',
#         'v_sv_pain_para:C(OV_value)[low_OV]:C(acceptance_pair)[P]_subj', 'v_sv_pain_para:C(OV_value)[low_OV]:C(acceptance_pair)[M]_subj', 'v_sv_pain_para:C(OV_value)[low_OV]:C(acceptance_pair)[I]_subj',
#         'v_sv_pain_para:C(OV_value)[high_OV]:C(acceptance_pair)[P]_subj', 'v_sv_pain_para:C(OV_value)[high_OV]:C(acceptance_pair)[M]_subj', 'v_sv_pain_para:C(OV_value)[high_OV]:C(acceptance_pair)[I]_subj',
#         't_sv_pain_para:C(OV_value)[low_OV]:C(acceptance_pair)[P]_subj', 't_sv_pain_para:C(OV_value)[low_OV]:C(acceptance_pair)[M]_subj', 't_sv_pain_para:C(OV_value)[low_OV]:C(acceptance_pair)[I]_subj',
#         't_sv_pain_para:C(OV_value)[high_OV]:C(acceptance_pair)[P]_subj', 't_sv_pain_para:C(OV_value)[high_OV]:C(acceptance_pair)[M]_subj', 't_sv_pain_para:C(OV_value)[high_OV]:C(acceptance_pair)[I]_subj',
#         'a_sv_pain_para:C(OV_value)[low_OV]:C(acceptance_pair)[P]_subj', 'a_sv_pain_para:C(OV_value)[low_OV]:C(acceptance_pair)[M]_subj', 'a_sv_pain_para:C(OV_value)[low_OV]:C(acceptance_pair)[I]_subj',
#         'a_sv_pain_para:C(OV_value)[high_OV]:C(acceptance_pair)[P]_subj', 'a_sv_pain_para:C(OV_value)[high_OV]:C(acceptance_pair)[M]_subj', 'a_sv_pain_para:C(OV_value)[high_OV]:C(acceptance_pair)[I]_subj',
#         'z_sv_pain_para:C(OV_value)[low_OV]:C(acceptance_pair)[P]_subj', 'z_sv_pain_para:C(OV_value)[low_OV]:C(acceptance_pair)[M]_subj', 'z_sv_pain_para:C(OV_value)[low_OV]:C(acceptance_pair)[I]_subj',
#         'z_sv_pain_para:C(OV_value)[high_OV]:C(acceptance_pair)[P]_subj', 'z_sv_pain_para:C(OV_value)[high_OV]:C(acceptance_pair)[M]_subj', 'z_sv_pain_para:C(OV_value)[high_OV]:C(acceptance_pair)[I]_subj'
#         ]
#         titles = [
#         'Starting point', 'Boundary separation', 'Non-decision time', 'Inter-trial variability in drift rate',
#         'Inter-trial variability in starting point', 'Inter-trial variability in non-decision time',
#         'Drift rate sv_pain_para * OV_value[low_OV] * acceptance_pair[P]', 'Drift rate sv_pain_para * OV_value[low_OV] * acceptance_pair[M]', 'Drift rate sv_pain_para * OV_value[low_OV] * acceptance_pair[I]',
#         'Drift rate sv_pain_para * OV_value[high_OV] * acceptance_pair[P]', 'Drift rate sv_pain_para * OV_value[high_OV] * acceptance_pair[M]', 'Drift rate sv_pain_para * OV_value[high_OV] * acceptance_pair[I]',
#         'Non-decision time sv_pain_para * OV_value[low_OV] * acceptance_pair[P]', 'Non-decision time sv_pain_para * OV_value[low_OV] * acceptance_pair[M]', 'Non-decision time sv_pain_para * OV_value[low_OV] * acceptance_pair[I]',
#         'Non-decision time sv_pain_para * OV_value[high_OV] * acceptance_pair[P]', 'Non-decision time sv_pain_para * OV_value[high_OV] * acceptance_pair[M]', 'Non-decision time sv_pain_para * OV_value[high_OV] * acceptance_pair[I]',
#         'Boundary separation sv_pain_para * OV_value[low_OV] * acceptance_pair[P]', 'Boundary separation sv_pain_para * OV_value[low_OV] * acceptance_pair[M]', 'Boundary separation sv_pain_para * OV_value[low_OV] * acceptance_pair[I]',
#         'Boundary separation sv_pain_para * OV_value[high_OV] * acceptance_pair[P]', 'Boundary separation sv_pain_para * OV_value[high_OV] * acceptance_pair[M]', 'Boundary separation sv_pain_para * OV_value[high_OV] * acceptance_pair[I]',
#         'Starting point sv_pain_para * OV_value[low_OV] * acceptance_pair[P]', 'Starting point sv_pain_para * OV_value[low_OV] * acceptance_pair[M]', 'Starting point sv_pain_para * OV_value[low_OV] * acceptance_pair[I]',
#         'Starting point sv_pain_para * OV_value[high_OV] * acceptance_pair[P]', 'Starting point sv_pain_para * OV_value[high_OV] * acceptance_pair[M]', 'Starting point sv_pain_para * OV_value[high_OV] * acceptance_pair[I]'
#         ]
#     elif version == 29:
#         params_of_interest = [
#         'z', 'a', 't', 'sv', 'sz', 'st',
#         'v_sv_pain_para:C(Abs_value)[low_abs]:C(acceptance_pair)[P]', 'v_sv_pain_para:C(Abs_value)[low_abs]:C(acceptance_pair)[M]', 'v_sv_pain_para:C(Abs_value)[low_abs]:C(acceptance_pair)[I]',
#         'v_sv_pain_para:C(Abs_value)[mid_abs]:C(acceptance_pair)[P]', 'v_sv_pain_para:C(Abs_value)[mid_abs]:C(acceptance_pair)[M]', 'v_sv_pain_para:C(Abs_value)[mid_abs]:C(acceptance_pair)[I]',
#         'v_sv_pain_para:C(Abs_value)[high_abs]:C(acceptance_pair)[P]', 'v_sv_pain_para:C(Abs_value)[high_abs]:C(acceptance_pair)[M]', 'v_sv_pain_para:C(Abs_value)[high_abs]:C(acceptance_pair)[I]',
#         't_sv_pain_para:C(Abs_value)[low_abs]:C(acceptance_pair)[P]', 't_sv_pain_para:C(Abs_value)[low_abs]:C(acceptance_pair)[M]', 't_sv_pain_para:C(Abs_value)[low_abs]:C(acceptance_pair)[I]',
#         't_sv_pain_para:C(Abs_value)[mid_abs]:C(acceptance_pair)[P]', 't_sv_pain_para:C(Abs_value)[mid_abs]:C(acceptance_pair)[M]', 't_sv_pain_para:C(Abs_value)[mid_abs]:C(acceptance_pair)[I]',
#         't_sv_pain_para:C(Abs_value)[high_abs]:C(acceptance_pair)[P]', 't_sv_pain_para:C(Abs_value)[high_abs]:C(acceptance_pair)[M]', 't_sv_pain_para:C(Abs_value)[high_abs]:C(acceptance_pair)[I]',
#         'a_sv_pain_para:C(Abs_value)[low_abs]:C(acceptance_pair)[P]', 'a_sv_pain_para:C(Abs_value)[low_abs]:C(acceptance_pair)[M]', 'a_sv_pain_para:C(Abs_value)[low_abs]:C(acceptance_pair)[I]',
#         'a_sv_pain_para:C(Abs_value)[mid_abs]:C(acceptance_pair)[P]', 'a_sv_pain_para:C(Abs_value)[mid_abs]:C(acceptance_pair)[M]', 'a_sv_pain_para:C(Abs_value)[mid_abs]:C(acceptance_pair)[I]',
#         'a_sv_pain_para:C(Abs_value)[high_abs]:C(acceptance_pair)[P]', 'a_sv_pain_para:C(Abs_value)[high_abs]:C(acceptance_pair)[M]', 'a_sv_pain_para:C(Abs_value)[high_abs]:C(acceptance_pair)[I]',
#         'z_sv_pain_para:C(Abs_value)[low_abs]:C(acceptance_pair)[P]', 'z_sv_pain_para:C(Abs_value)[low_abs]:C(acceptance_pair)[M]', 'z_sv_pain_para:C(Abs_value)[low_abs]:C(acceptance_pair)[I]',
#         'z_sv_pain_para:C(Abs_value)[mid_abs]:C(acceptance_pair)[P]', 'z_sv_pain_para:C(Abs_value)[mid_abs]:C(acceptance_pair)[M]', 'z_sv_pain_para:C(Abs_value)[mid_abs]:C(acceptance_pair)[I]',
#         'z_sv_pain_para:C(Abs_value)[high_abs]:C(acceptance_pair)[P]', 'z_sv_pain_para:C(Abs_value)[high_abs]:C(acceptance_pair)[M]', 'z_sv_pain_para:C(Abs_value)[high_abs]:C(acceptance_pair)[I]'
#         ]
#         params_of_interest_s = [
#         'z_subj', 'a_subj', 't_subj', 'sv_subj', 'sz_subj', 'st_subj',
#         'v_sv_pain_para:C(Abs_value)[low_abs]:C(acceptance_pair)[P]_subj', 'v_sv_pain_para:C(Abs_value)[low_abs]:C(acceptance_pair)[M]_subj', 'v_sv_pain_para:C(Abs_value)[low_abs]:C(acceptance_pair)[I]_subj',
#         'v_sv_pain_para:C(Abs_value)[mid_abs]:C(acceptance_pair)[P]_subj', 'v_sv_pain_para:C(Abs_value)[mid_abs]:C(acceptance_pair)[M]_subj', 'v_sv_pain_para:C(Abs_value)[mid_abs]:C(acceptance_pair)[I]_subj',
#         'v_sv_pain_para:C(Abs_value)[high_abs]:C(acceptance_pair)[P]_subj', 'v_sv_pain_para:C(Abs_value)[high_abs]:C(acceptance_pair)[M]_subj', 'v_sv_pain_para:C(Abs_value)[high_abs]:C(acceptance_pair)[I]_subj',
#         't_sv_pain_para:C(Abs_value)[low_abs]:C(acceptance_pair)[P]_subj', 't_sv_pain_para:C(Abs_value)[low_abs]:C(acceptance_pair)[M]_subj', 't_sv_pain_para:C(Abs_value)[low_abs]:C(acceptance_pair)[I]_subj',
#         't_sv_pain_para:C(Abs_value)[mid_abs]:C(acceptance_pair)[P]_subj', 't_sv_pain_para:C(Abs_value)[mid_abs]:C(acceptance_pair)[M]_subj', 't_sv_pain_para:C(Abs_value)[mid_abs]:C(acceptance_pair)[I]_subj',
#         't_sv_pain_para:C(Abs_value)[high_abs]:C(acceptance_pair)[P]_subj', 't_sv_pain_para:C(Abs_value)[high_abs]:C(acceptance_pair)[M]_subj', 't_sv_pain_para:C(Abs_value)[high_abs]:C(acceptance_pair)[I]_subj',
#         'a_sv_pain_para:C(Abs_value)[low_abs]:C(acceptance_pair)[P]_subj', 'a_sv_pain_para:C(Abs_value)[low_abs]:C(acceptance_pair)[M]_subj', 'a_sv_pain_para:C(Abs_value)[low_abs]:C(acceptance_pair)[I]_subj',
#         'a_sv_pain_para:C(Abs_value)[mid_abs]:C(acceptance_pair)[P]_subj', 'a_sv_pain_para:C(Abs_value)[mid_abs]:C(acceptance_pair)[M]_subj', 'a_sv_pain_para:C(Abs_value)[mid_abs]:C(acceptance_pair)[I]_subj',
#         'a_sv_pain_para:C(Abs_value)[high_abs]:C(acceptance_pair)[P]_subj', 'a_sv_pain_para:C(Abs_value)[high_abs]:C(acceptance_pair)[M]_subj', 'a_sv_pain_para:C(Abs_value)[high_abs]:C(acceptance_pair)[I]_subj',
#         'z_sv_pain_para:C(Abs_value)[low_abs]:C(acceptance_pair)[P]_subj', 'z_sv_pain_para:C(Abs_value)[low_abs]:C(acceptance_pair)[M]_subj', 'z_sv_pain_para:C(Abs_value)[low_abs]:C(acceptance_pair)[I]_subj',
#         'z_sv_pain_para:C(Abs_value)[mid_abs]:C(acceptance_pair)[P]_subj', 'z_sv_pain_para:C(Abs_value)[mid_abs]:C(acceptance_pair)[M]_subj', 'z_sv_pain_para:C(Abs_value)[mid_abs]:C(acceptance_pair)[I]_subj',
#         'z_sv_pain_para:C(Abs_value)[high_abs]:C(acceptance_pair)[P]_subj', 'z_sv_pain_para:C(Abs_value)[high_abs]:C(acceptance_pair)[M]_subj', 'z_sv_pain_para:C(Abs_value)[high_abs]:C(acceptance_pair)[I]_subj'
#         ]
#         titles = [
#         'Starting point', 'Boundary separation', 'Non-decision time', 'Inter-trial variability in drift rate',
#         'Inter-trial variability in starting point', 'Inter-trial variability in non-decision time',
#         'Drift rate sv_pain_para * Abs_value[low_abs] * acceptance_pair[P]', 'Drift rate sv_pain_para * Abs_value[low_abs] * acceptance_pair[M]', 'Drift rate sv_pain_para * Abs_value[low_abs] * acceptance_pair[I]',
#         'Drift rate sv_pain_para * Abs_value[mid_abs] * acceptance_pair[P]', 'Drift rate sv_pain_para * Abs_value[mid_abs] * acceptance_pair[M]', 'Drift rate sv_pain_para * Abs_value[mid_abs] * acceptance_pair[I]',
#         'Drift rate sv_pain_para * Abs_value[high_abs] * acceptance_pair[P]', 'Drift rate sv_pain_para * Abs_value[high_abs] * acceptance_pair[M]', 'Drift rate sv_pain_para * Abs_value[high_abs] * acceptance_pair[I]',
#         'Non-decision time sv_pain_para * Abs_value[low_abs] * acceptance_pair[P]', 'Non-decision time sv_pain_para * Abs_value[low_abs] * acceptance_pair[M]', 'Non-decision time sv_pain_para * Abs_value[low_abs] * acceptance_pair[I]',
#         'Non-decision time sv_pain_para * Abs_value[mid_abs] * acceptance_pair[P]', 'Non-decision time sv_pain_para * Abs_value[mid_abs] * acceptance_pair[M]', 'Non-decision time sv_pain_para * Abs_value[mid_abs] * acceptance_pair[I]',
#         'Non-decision time sv_pain_para * Abs_value[high_abs] * acceptance_pair[P]', 'Non-decision time sv_pain_para * Abs_value[high_abs] * acceptance_pair[M]', 'Non-decision time sv_pain_para * Abs_value[high_abs] * acceptance_pair[I]',
#         'Boundary separation sv_pain_para * Abs_value[low_abs] * acceptance_pair[P]', 'Boundary separation sv_pain_para * Abs_value[low_abs] * acceptance_pair[M]', 'Boundary separation sv_pain_para * Abs_value[low_abs] * acceptance_pair[I]',
#         'Boundary separation sv_pain_para * Abs_value[mid_abs] * acceptance_pair[P]', 'Boundary separation sv_pain_para * Abs_value[mid_abs] * acceptance_pair[M]', 'Boundary separation sv_pain_para * Abs_value[mid_abs] * acceptance_pair[I]',
#         'Boundary separation sv_pain_para * Abs_value[high_abs] * acceptance_pair[P]', 'Boundary separation sv_pain_para * Abs_value[high_abs] * acceptance_pair[M]', 'Boundary separation sv_pain_para * Abs_value[high_abs] * acceptance_pair[I]',
#         'Starting point sv_pain_para * Abs_value[low_abs] * acceptance_pair[P]', 'Starting point sv_pain_para * Abs_value[low_abs] * acceptance_pair[M]', 'Starting point sv_pain_para * Abs_value[low_abs] * acceptance_pair[I]',
#         'Starting point sv_pain_para * Abs_value[mid_abs] * acceptance_pair[P]', 'Starting point sv_pain_para * Abs_value[mid_abs] * acceptance_pair[M]', 'Starting point sv_pain_para * Abs_value[mid_abs] * acceptance_pair[I]',
#         'Starting point sv_pain_para * Abs_value[high_abs] * acceptance_pair[P]', 'Starting point sv_pain_para * Abs_value[high_abs] * acceptance_pair[M]', 'Starting point sv_pain_para * Abs_value[high_abs] * acceptance_pair[I]'
#         ]
#     elif version == 33:
#         params_of_interest = [
#         'z', 'a', 't', 'sv', 'sz', 'st',
#         'v_sv_pain_para:C(OV_value)[low_OV]:C(acceptance_pair)[P]', 'v_sv_pain_para:C(OV_value)[low_OV]:C(acceptance_pair)[M]', 'v_sv_pain_para:C(OV_value)[low_OV]:C(acceptance_pair)[I]',
#         'v_sv_pain_para:C(OV_value)[high_OV]:C(acceptance_pair)[P]', 'v_sv_pain_para:C(OV_value)[high_OV]:C(acceptance_pair)[M]', 'v_sv_pain_para:C(OV_value)[high_OV]:C(acceptance_pair)[I]',
#         't_sv_pain_para',
#         'a_sv_pain_para:C(OV_value)[low_OV]:C(acceptance_pair)[P]', 'a_sv_pain_para:C(OV_value)[low_OV]:C(acceptance_pair)[M]', 'a_sv_pain_para:C(OV_value)[low_OV]:C(acceptance_pair)[I]',
#         'a_sv_pain_para:C(OV_value)[high_OV]:C(acceptance_pair)[P]', 'a_sv_pain_para:C(OV_value)[high_OV]:C(acceptance_pair)[M]', 'a_sv_pain_para:C(OV_value)[high_OV]:C(acceptance_pair)[I]',
#         'z_sv_pain_para:C(acceptance_pair)[P]', 'z_sv_pain_para:C(acceptance_pair)[M]', 'z_sv_pain_para:C(acceptance_pair)[I]'
#         ]
#         params_of_interest_s = [
#         'z_subj', 'a_subj', 't_subj', 'sv_subj', 'sz_subj', 'st_subj',
#         'v_sv_pain_para:C(OV_value)[low_OV]:C(acceptance_pair)[P]_subj', 'v_sv_pain_para:C(OV_value)[low_OV]:C(acceptance_pair)[M]_subj', 'v_sv_pain_para:C(OV_value)[low_OV]:C(acceptance_pair)[I]_subj',
#         'v_sv_pain_para:C(OV_value)[high_OV]:C(acceptance_pair)[P]_subj', 'v_sv_pain_para:C(OV_value)[high_OV]:C(acceptance_pair)[M]_subj', 'v_sv_pain_para:C(OV_value)[high_OV]:C(acceptance_pair)[I]_subj',
#         't_sv_pain_para_subj',
#         'a_sv_pain_para:C(OV_value)[low_OV]:C(acceptance_pair)[P]_subj', 'a_sv_pain_para:C(OV_value)[low_OV]:C(acceptance_pair)[M]_subj', 'a_sv_pain_para:C(OV_value)[low_OV]:C(acceptance_pair)[I]_subj',
#         'a_sv_pain_para:C(OV_value)[high_OV]:C(acceptance_pair)[P]_subj', 'a_sv_pain_para:C(OV_value)[high_OV]:C(acceptance_pair)[M]_subj', 'a_sv_pain_para:C(OV_value)[high_OV]:C(acceptance_pair)[I]_subj',
#         'z_sv_pain_para:C(acceptance_pair)[P]_subj', 'z_sv_pain_para:C(acceptance_pair)[M]_subj', 'z_sv_pain_para:C(acceptance_pair)[I]_subj'
#         ]
#         titles = [
#         'Starting point', 'Boundary separation', 'Non-decision time', 'Inter-trial variability in drift rate',
#         'Inter-trial variability in starting point', 'Inter-trial variability in non-decision time',
#         'Drift rate sv_pain_para * OV_value[low_OV] * acceptance_pair[P]', 'Drift rate sv_pain_para * OV_value[low_OV] * acceptance_pair[M]', 'Drift rate sv_pain_para * OV_value[low_OV] * acceptance_pair[I]',
#         'Drift rate sv_pain_para * OV_value[high_OV] * acceptance_pair[P]', 'Drift rate sv_pain_para * OV_value[high_OV] * acceptance_pair[M]', 'Drift rate sv_pain_para * OV_value[high_OV] * acceptance_pair[I]',
#         'Non-decision time sv_pain_para',
#         'Boundary separation sv_pain_para * OV_value[low_OV] * acceptance_pair[P]', 'Boundary separation sv_pain_para * OV_value[low_OV] * acceptance_pair[M]', 'Boundary separation sv_pain_para * OV_value[low_OV] * acceptance_pair[I]',
#         'Boundary separation sv_pain_para * OV_value[high_OV] * acceptance_pair[P]', 'Boundary separation sv_pain_para * OV_value[high_OV] * acceptance_pair[M]', 'Boundary separation sv_pain_para * OV_value[high_OV] * acceptance_pair[I]',
#         'Starting point sv_pain_para * acceptance_pair[P]', 'Starting point sv_pain_para * acceptance_pair[M]', 'Starting point sv_pain_para * acceptance_pair[I]'
#         ]
# #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#     # Quest data        
#     elif version == 34:  # v depends on STA_SAI_Score
#         params_of_interest = ['z', 'a', 't', 'sv', 'sz', 'st', 'v_Intercept', 'v_STA_SAI_Score']
#         params_of_interest_s = ['z_subj', 'a_subj', 't_subj', 'v_Intercept_subj', 'v_STA_SAI_Score_subj']
#         titles = [
#             'Starting point', 'Boundary sep.', 'Non-dec. time',
#             'Inter-trial variability in drift rate', 'Inter-trial variability in starting point', 'Inter-trial variability in non-decision time',
#             'Intercept drift rate', 'Drift rate STA_SAI_Score']

#     elif version == 35:  # a depends on STA_SAI_Score
#         params_of_interest = ['z', 't', 'sv', 'sz', 'st', 'a_Intercept', 'a_STA_SAI_Score']
#         params_of_interest_s = ['z_subj', 't_subj', 'sv_subj', 'sz_subj', 'st_subj', 'a_Intercept_subj', 'a_STA_SAI_Score_subj']
#         titles = [
#             'Starting point', 'Non-dec. time', 
#             'Inter-trial variability in drift rate', 'Inter-trial variability in starting point', 'Inter-trial variability in non-decision time',
#             'Intercept boundary separation', 'Boundary separation STA_SAI_Score']

#     elif version == 36:  # v depends on STA_TAI_Score
#         params_of_interest = ['z', 'a', 't', 'sv', 'sz', 'st', 'v_Intercept', 'v_STA_TAI_Score']
#         params_of_interest_s = ['z_subj', 'a_subj', 't_subj', 'v_Intercept_subj', 'v_STA_TAI_Score_subj']
#         titles = [
#             'Starting point', 'Boundary sep.', 'Non-dec. time',
#             'Inter-trial variability in drift rate', 'Inter-trial variability in starting point', 'Inter-trial variability in non-decision time',
#             'Intercept drift rate', 'Drift rate STA_TAI_Score']

#     elif version == 37:  # a depends on STA_TAI_Score
#         params_of_interest = ['z', 't', 'sv', 'sz', 'st', 'a_Intercept', 'a_STA_TAI_Score']
#         params_of_interest_s = ['z_subj', 't_subj', 'sv_subj', 'sz_subj', 'st_subj', 'a_Intercept_subj', 'a_STA_TAI_Score_subj']
#         titles = [
#             'Starting point', 'Non-dec. time', 
#             'Inter-trial variability in drift rate', 'Inter-trial variability in starting point', 'Inter-trial variability in non-decision time',
#             'Intercept boundary separation', 'Boundary separation STA_TAI_Score']

#     elif version == 38:  # v depends on PCS_Score
#         params_of_interest = ['z', 'a', 't', 'sv', 'sz', 'st', 'v_Intercept', 'v_PCS_Score']
#         params_of_interest_s = ['z_subj', 'a_subj', 't_subj', 'v_Intercept_subj', 'v_PCS_Score_subj']
#         titles = [
#             'Starting point', 'Boundary sep.', 'Non-dec. time',
#             'Inter-trial variability in drift rate', 'Inter-trial variability in starting point', 'Inter-trial variability in non-decision time',
#             'Intercept drift rate', 'Drift rate PCS_Score']

#     elif version == 39:  # a depends on PCS_Score
#         params_of_interest = ['z', 't', 'sv', 'sz', 'st', 'a_Intercept', 'a_PCS_Score']
#         params_of_interest_s = ['z_subj', 't_subj', 'sv_subj', 'sz_subj', 'st_subj', 'a_Intercept_subj', 'a_PCS_Score_subj']
#         titles = [
#             'Starting point', 'Non-dec. time', 
#             'Inter-trial variability in drift rate', 'Inter-trial variability in starting point', 'Inter-trial variability in non-decision time',
#             'Intercept boundary separation', 'Boundary separation PCS_Score']

#     elif version == 40:  
#         params_of_interest = [
#             'sv', 'sz', 'st',
#             'v_Intercept', 'v_STA_SAI_Score',
#             't_Intercept', 't_STA_SAI_Score',
#             'a_Intercept', 'a_STA_SAI_Score',
#             'z_Intercept', 'z_STA_SAI_Score']
#         params_of_interest_s = [
#             'sv_subj', 'sz_subj', 'st_subj',
#             'v_Intercept_subj', 'v_STA_SAI_Score_subj',
#             't_Intercept_subj', 't_STA_SAI_Score_subj',
#             'a_Intercept_subj', 'a_STA_SAI_Score_subj',
#             'z_Intercept_subj', 'z_STA_SAI_Score_subj']
#         titles = [
#             'Inter-trial variability in drift rate', 'Inter-trial variability in starting point', 'Inter-trial variability in non-decision time',
#             'Intercept drift rate', 'Drift rate STA_SAI_Score',
#             'Intercept non-decision time', 'Non-decision time STA_SAI_Score',
#             'Intercept boundary separation', 'Boundary separation STA_SAI_Score',
#             'Intercept starting point', 'Starting point STA_SAI_Score']

#     elif version == 41:  
#         params_of_interest = [
#             'sv', 'sz', 'st',
#             'v_Intercept', 'v_STA_TAI_Score',
#             't_Intercept', 't_STA_TAI_Score',
#             'a_Intercept', 'a_STA_TAI_Score',
#             'z_Intercept', 'z_STA_TAI_Score']
#         params_of_interest_s = [
#             'sv_subj', 'sz_subj', 'st_subj',
#             'v_Intercept_subj', 'v_STA_TAI_Score_subj',
#             't_Intercept_subj', 't_STA_TAI_Score_subj',
#             'a_Intercept_subj', 'a_STA_TAI_Score_subj',
#             'z_Intercept_subj', 'z_STA_TAI_Score_subj']
#         titles = [
#             'Inter-trial variability in drift rate', 'Inter-trial variability in starting point', 'Inter-trial variability in non-decision time',
#             'Intercept drift rate', 'Drift rate STA_TAI_Score',
#             'Intercept non-decision time', 'Non-decision time STA_TAI_Score',
#             'Intercept boundary separation', 'Boundary separation STA_TAI_Score',
#             'Intercept starting point', 'Starting point STA_TAI_Score']

#     elif version == 42:  
#         params_of_interest = [
#             'sv', 'sz', 'st',
#             'v_Intercept', 'v_PCS_Score',
#             't_Intercept', 't_PCS_Score',
#             'a_Intercept', 'a_PCS_Score',
#             'z_Intercept', 'z_PCS_Score']
#         params_of_interest_s = [
#             'sv_subj', 'sz_subj', 'st_subj',
#             'v_Intercept_subj', 'v_PCS_Score_subj',
#             't_Intercept_subj', 't_PCS_Score_subj',
#             'a_Intercept_subj', 'a_PCS_Score_subj',
#             'z_Intercept_subj', 'z_PCS_Score_subj']
#         titles = [
#             'Inter-trial variability in drift rate', 'Inter-trial variability in starting point', 'Inter-trial variability in non-decision time',
#             'Intercept drift rate', 'Drift rate PCS_Score',
#             'Intercept non-decision time', 'Non-decision time PCS_Score',
#             'Intercept boundary separation', 'Boundary separation PCS_Score',
#             'Intercept starting point', 'Starting point PCS_Score']    
        
#     elif version == 43:  
#         params_of_interest = [
#             'sv', 'sz', 'st',
#             'v_Intercept', 'v_STA_SAI_Score:sv_pain_para:C(OV_value)',
#             't_Intercept', 't_STA_SAI_Score:sv_pain_para:C(OV_value)',
#             'a_Intercept', 'a_STA_SAI_Score:sv_pain_para:C(OV_value)',
#             'z_Intercept', 'z_STA_SAI_Score:sv_pain_para:C(OV_value)']
#         params_of_interest_s = [
#             'sv_subj', 'sz_subj', 'st_subj',
#             'v_Intercept_subj', 'v_STA_SAI_Score:sv_pain_para:C(OV_value)_subj',
#             't_Intercept_subj', 't_STA_SAI_Score:sv_pain_para:C(OV_value)_subj',
#             'a_Intercept_subj', 'a_STA_SAI_Score:sv_pain_para:C(OV_value)_subj',
#             'z_Intercept_subj', 'z_STA_SAI_Score:sv_pain_para:C(OV_value)_subj']
#         titles = [
#             'Inter-trial variability in drift rate', 'Inter-trial variability in starting point', 'Inter-trial variability in non-decision time',
#             'Intercept drift rate', 'Drift rate interaction (STA_SAI_Score:sv_pain_para:C(OV_value))',
#             'Intercept non-decision time', 'Non-decision time interaction (STA_SAI_Score:sv_pain_para:C(OV_value))',
#             'Intercept boundary separation', 'Boundary separation interaction (STA_SAI_Score:sv_pain_para:C(OV_value))',
#             'Intercept starting point', 'Starting point interaction (STA_SAI_Score:sv_pain_para:C(OV_value))']

#     elif version == 44:  
#         params_of_interest = [
#             'sv', 'sz', 'st',
#             'v_Intercept', 'v_STA_TAI_Score:sv_pain_para:C(OV_value)',
#             't_Intercept', 't_STA_TAI_Score:sv_pain_para:C(OV_value)',
#             'a_Intercept', 'a_STA_TAI_Score:sv_pain_para:C(OV_value)',
#             'z_Intercept', 'z_STA_TAI_Score:sv_pain_para:C(OV_value)']
#         params_of_interest_s = [
#             'sv_subj', 'sz_subj', 'st_subj',
#             'v_Intercept_subj', 'v_STA_TAI_Score:sv_pain_para:C(OV_value)_subj',
#             't_Intercept_subj', 't_STA_TAI_Score:sv_pain_para:C(OV_value)_subj',
#             'a_Intercept_subj', 'a_STA_TAI_Score:sv_pain_para:C(OV_value)_subj',
#             'z_Intercept_subj', 'z_STA_TAI_Score:sv_pain_para:C(OV_value)_subj']
#         titles = [
#             'Inter-trial variability in drift rate', 'Inter-trial variability in starting point', 'Inter-trial variability in non-decision time',
#             'Intercept drift rate', 'Drift rate interaction (STA_TAI_Score:sv_pain_para:C(OV_value))',
#             'Intercept non-decision time', 'Non-decision time interaction (STA_TAI_Score:sv_pain_para:C(OV_value))',
#             'Intercept boundary separation', 'Boundary separation interaction (STA_TAI_Score:sv_pain_para:C(OV_value))',
#             'Intercept starting point', 'Starting point interaction (STA_TAI_Score:sv_pain_para:C(OV_value))']

#     elif version == 45:  
#         params_of_interest = [
#             'sv', 'sz', 'st',
#             'v_Intercept', 'v_PCS_Score:sv_pain_para:C(OV_value)',
#             't_Intercept', 't_PCS_Score:sv_pain_para:C(OV_value)',
#             'a_Intercept', 'a_PCS_Score:sv_pain_para:C(OV_value)',
#             'z_Intercept', 'z_PCS_Score:sv_pain_para:C(OV_value)']
#         params_of_interest_s = [
#             'sv_subj', 'sz_subj', 'st_subj',
#             'v_Intercept_subj', 'v_PCS_Score:sv_pain_para:C(OV_value)_subj',
#             't_Intercept_subj', 't_PCS_Score:sv_pain_para:C(OV_value)_subj',
#             'a_Intercept_subj', 'a_PCS_Score:sv_pain_para:C(OV_value)_subj',
#             'z_Intercept_subj', 'z_PCS_Score:sv_pain_para:C(OV_value)_subj']
#         titles = [
#             'Inter-trial variability in drift rate', 'Inter-trial variability in starting point', 'Inter-trial variability in non-decision time',
#             'Intercept drift rate', 'Drift rate interaction (PCS_Score:sv_pain_para:C(OV_value))',
#             'Intercept non-decision time', 'Non-decision time interaction (PCS_Score:sv_pain_para:C(OV_value))',
#             'Intercept boundary separation', 'Boundary separation interaction (PCS_Score:sv_pain_para:C(OV_value))',
#             'Intercept starting point', 'Starting point interaction (PCS_Score:sv_pain_para:C(OV_value))']

#     elif version == 46:  
#         params_of_interest = [
#             'sv', 'sz', 'st',
#             'v_Intercept', 'v_STA_SAI_Score:sv_pain_para:C(Abs_value)',
#             't_Intercept', 't_STA_SAI_Score:sv_pain_para:C(Abs_value)',
#             'a_Intercept', 'a_STA_SAI_Score:sv_pain_para:C(Abs_value)',
#             'z_Intercept', 'z_STA_SAI_Score:sv_pain_para:C(Abs_value)']
#         params_of_interest_s = [
#             'sv_subj', 'sz_subj', 'st_subj',
#             'v_Intercept_subj', 'v_STA_SAI_Score:sv_pain_para:C(Abs_value)_subj',
#             't_Intercept_subj', 't_STA_SAI_Score:sv_pain_para:C(Abs_value)_subj',
#             'a_Intercept_subj', 'a_STA_SAI_Score:sv_pain_para:C(Abs_value)_subj',
#             'z_Intercept_subj', 'z_STA_SAI_Score:sv_pain_para:C(Abs_value)_subj']
#         titles = [
#             'Inter-trial variability in drift rate', 'Inter-trial variability in starting point', 'Inter-trial variability in non-decision time',
#             'Intercept drift rate', 'Drift rate interaction (STA_SAI_Score:sv_pain_para:C(Abs_value))',
#             'Intercept non-decision time', 'Non-decision time interaction (STA_SAI_Score:sv_pain_para:C(Abs_value))',
#             'Intercept boundary separation', 'Boundary separation interaction (STA_SAI_Score:sv_pain_para:C(Abs_value))',
#             'Intercept starting point', 'Starting point interaction (STA_SAI_Score:sv_pain_para:C(Abs_value))']

#     elif version == 47:  
#         params_of_interest = [
#             'sv', 'sz', 'st',
#             'v_Intercept', 'v_STA_TAI_Score:sv_pain_para:C(Abs_value)',
#             't_Intercept', 't_STA_TAI_Score:sv_pain_para:C(Abs_value)',
#             'a_Intercept', 'a_STA_TAI_Score:sv_pain_para:C(Abs_value)',
#             'z_Intercept', 'z_STA_TAI_Score:sv_pain_para:C(Abs_value)']
#         params_of_interest_s = [
#             'sv_subj', 'sz_subj', 'st_subj',
#             'v_Intercept_subj', 'v_STA_TAI_Score:sv_pain_para:C(Abs_value)_subj',
#             't_Intercept_subj', 't_STA_TAI_Score:sv_pain_para:C(Abs_value)_subj',
#             'a_Intercept_subj', 'a_STA_TAI_Score:sv_pain_para:C(Abs_value)_subj',
#             'z_Intercept_subj', 'z_STA_TAI_Score:sv_pain_para:C(Abs_value)_subj']
#         titles = [
#             'Inter-trial variability in drift rate', 'Inter-trial variability in starting point', 'Inter-trial variability in non-decision time',
#             'Intercept drift rate', 'Drift rate interaction (STA_TAI_Score:sv_pain_para:C(Abs_value))',
#             'Intercept non-decision time', 'Non-decision time interaction (STA_TAI_Score:sv_pain_para:C(Abs_value))',
#             'Intercept boundary separation', 'Boundary separation interaction (STA_TAI_Score:sv_pain_para:C(Abs_value))',
#             'Intercept starting point', 'Starting point interaction (STA_TAI_Score:sv_pain_para:C(Abs_value))']

#     elif version == 48:  
#         params_of_interest = [
#             'sv', 'sz', 'st',
#             'v_Intercept', 'v_PCS_Score:sv_pain_para:C(Abs_value)',
#             't_Intercept', 't_PCS_Score:sv_pain_para:C(Abs_value)',
#             'a_Intercept', 'a_PCS_Score:sv_pain_para:C(Abs_value)',
#             'z_Intercept', 'z_PCS_Score:sv_pain_para:C(Abs_value)']
#         params_of_interest_s = [
#             'sv_subj', 'sz_subj', 'st_subj',
#             'v_Intercept_subj', 'v_PCS_Score:sv_pain_para:C(Abs_value)_subj',
#             't_Intercept_subj', 't_PCS_Score:sv_pain_para:C(Abs_value)_subj',
#             'a_Intercept_subj', 'a_PCS_Score:sv_pain_para:C(Abs_value)_subj',
#             'z_Intercept_subj', 'z_PCS_Score:sv_pain_para:C(Abs_value)_subj']
#         titles = [
#             'Inter-trial variability in drift rate', 'Inter-trial variability in starting point', 'Inter-trial variability in non-decision time',
#             'Intercept drift rate', 'Drift rate interaction (PCS_Score:sv_pain_para:C(Abs_value))',
#             'Intercept non-decision time', 'Non-decision time interaction (PCS_Score:sv_pain_para:C(Abs_value))',
#             'Intercept boundary separation', 'Boundary separation interaction (PCS_Score:sv_pain_para:C(Abs_value))',
#             'Intercept starting point', 'Starting point interaction (PCS_Score:sv_pain_para:C(Abs_value))']
#     elif version == 49:  
#         params_of_interest = [
#             'sv', 'sz', 'st',
#             'v_Intercept', 'v_STA_SAI_Score:sv_pain_para:C(acceptance_pair)',
#             't_Intercept', 't_STA_SAI_Score:sv_pain_para:C(acceptance_pair)',
#             'a_Intercept', 'a_STA_SAI_Score:sv_pain_para:C(acceptance_pair)',
#             'z_Intercept', 'z_STA_SAI_Score:sv_pain_para:C(acceptance_pair)']
#         params_of_interest_s = [
#             'sv_subj', 'sz_subj', 'st_subj',
#             'v_Intercept_subj', 'v_STA_SAI_Score:sv_pain_para:C(acceptance_pair)_subj',
#             't_Intercept_subj', 't_STA_SAI_Score:sv_pain_para:C(acceptance_pair)_subj',
#             'a_Intercept_subj', 'a_STA_SAI_Score:sv_pain_para:C(acceptance_pair)_subj',
#             'z_Intercept_subj', 'z_STA_SAI_Score:sv_pain_para:C(acceptance_pair)_subj']
#         titles = [
#             'Inter-trial variability in drift rate', 'Inter-trial variability in starting point', 'Inter-trial variability in non-decision time',
#             'Intercept drift rate', 'Drift rate interaction (STA_SAI_Score:sv_pain_para:C(acceptance_pair))',
#             'Intercept non-decision time', 'Non-decision time interaction (STA_SAI_Score:sv_pain_para:C(acceptance_pair))',
#             'Intercept boundary separation', 'Boundary separation interaction (STA_SAI_Score:sv_pain_para:C(acceptance_pair))',
#             'Intercept starting point', 'Starting point interaction (STA_SAI_Score:sv_pain_para:C(acceptance_pair))']

#     elif version == 50:  
#         params_of_interest = [
#             'sv', 'sz', 'st',
#             'v_Intercept', 'v_STA_TAI_Score:sv_pain_para:C(acceptance_pair)',
#             't_Intercept', 't_STA_TAI_Score:sv_pain_para:C(acceptance_pair)',
#             'a_Intercept', 'a_STA_TAI_Score:sv_pain_para:C(acceptance_pair)',
#             'z_Intercept', 'z_STA_TAI_Score:sv_pain_para:C(acceptance_pair)']
#         params_of_interest_s = [
#             'sv_subj', 'sz_subj', 'st_subj',
#             'v_Intercept_subj', 'v_STA_TAI_Score:sv_pain_para:C(acceptance_pair)_subj',
#             't_Intercept_subj', 't_STA_TAI_Score:sv_pain_para:C(acceptance_pair)_subj',
#             'a_Intercept_subj', 'a_STA_TAI_Score:sv_pain_para:C(acceptance_pair)_subj',
#             'z_Intercept_subj', 'z_STA_TAI_Score:sv_pain_para:C(acceptance_pair)_subj']
#         titles = [
#             'Inter-trial variability in drift rate', 'Inter-trial variability in starting point', 'Inter-trial variability in non-decision time',
#             'Intercept drift rate', 'Drift rate interaction (STA_TAI_Score:sv_pain_para:C(acceptance_pair))',
#             'Intercept non-decision time', 'Non-decision time interaction (STA_TAI_Score:sv_pain_para:C(acceptance_pair))',
#             'Intercept boundary separation', 'Boundary separation interaction (STA_TAI_Score:sv_pain_para:C(acceptance_pair))',
#             'Intercept starting point', 'Starting point interaction (STA_TAI_Score:sv_pain_para:C(acceptance_pair))']

#     elif version == 51:  
#         params_of_interest = [
#             'sv', 'sz', 'st',
#             'v_Intercept', 'v_PCS_Score:sv_pain_para:C(acceptance_pair)',
#             't_Intercept', 't_PCS_Score:sv_pain_para:C(acceptance_pair)',
#             'a_Intercept', 'a_PCS_Score:sv_pain_para:C(acceptance_pair)',
#             'z_Intercept', 'z_PCS_Score:sv_pain_para:C(acceptance_pair)']
#         params_of_interest_s = [
#             'sv_subj', 'sz_subj', 'st_subj',
#             'v_Intercept_subj', 'v_PCS_Score:sv_pain_para:C(acceptance_pair)_subj',
#             't_Intercept_subj', 't_PCS_Score:sv_pain_para:C(acceptance_pair)_subj',
#             'a_Intercept_subj', 'a_PCS_Score:sv_pain_para:C(acceptance_pair)_subj',
#             'z_Intercept_subj', 'z_PCS_Score:sv_pain_para:C(acceptance_pair)_subj']
#         titles = [
#             'Inter-trial variability in drift rate', 'Inter-trial variability in starting point', 'Inter-trial variability in non-decision time',
#             'Intercept drift rate', 'Drift rate interaction (PCS_Score:sv_pain_para:C(acceptance_pair))',
#             'Intercept non-decision time', 'Non-decision time interaction (PCS_Score:sv_pain_para:C(acceptance_pair))',
#             'Intercept boundary separation', 'Boundary separation interaction (PCS_Score:sv_pain_para:C(acceptance_pair))',
#             'Intercept starting point', 'Starting point interaction (PCS_Score:sv_pain_para:C(acceptance_pair))']

#     elif version == 52:  
#         params_of_interest = [
#             'sv', 'sz', 'st',
#             'v_Intercept', 'v_sv_pain_para', 
#             'v_STA_SAI_Score', 'v_STA_TAI_Score', 'v_PCS_Score', 
#             'v_sv_pain_para:STA_SAI_Score', 'v_sv_pain_para:STA_TAI_Score', 'v_sv_pain_para:PCS_Score', 
#             'v_sv_pain_para:STA_SAI_Score:STA_TAI_Score', 'v_sv_pain_para:STA_SAI_Score:PCS_Score', 'v_sv_pain_para:STA_TAI_Score:PCS_Score', 
#             'v_sv_pain_para:STA_SAI_Score:STA_TAI_Score:PCS_Score']
#         params_of_interest_s = [
#             'sv_subj', 'sz_subj', 'st_subj',
#             'v_Intercept_subj', 'v_sv_pain_para_subj', 
#             'v_STA_SAI_Score_subj', 'v_STA_TAI_Score_subj', 'v_PCS_Score_subj', 
#             'v_sv_pain_para:STA_SAI_Score_subj', 'v_sv_pain_para:STA_TAI_Score_subj', 'v_sv_pain_para:PCS_Score_subj', 
#             'v_sv_pain_para:STA_SAI_Score:STA_TAI_Score_subj', 'v_sv_pain_para:STA_SAI_Score:PCS_Score_subj', 'v_sv_pain_para:STA_TAI_Score:PCS_Score_subj', 
#             'v_sv_pain_para:STA_SAI_Score:STA_TAI_Score:PCS_Score_subj']
#         titles = [
#             'Inter-trial variability in drift rate', 'Inter-trial variability in starting point', 'Inter-trial variability in non-decision time',
#             'Intercept drift rate', 'Drift rate sv_pain_para',
#             'Effect of STA_SAI_Score', 'Effect of STA_TAI_Score', 'Effect of PCS_Score', 
#             'Interaction: sv_pain_para * STA_SAI_Score', 'Interaction: sv_pain_para * STA_TAI_Score', 'Interaction: sv_pain_para * PCS_Score', 
#             'Interaction: sv_pain_para * STA_SAI_Score * STA_TAI_Score', 'Interaction: sv_pain_para * STA_SAI_Score * PCS_Score', 'Interaction: sv_pain_para * STA_TAI_Score * PCS_Score', 
#             'Interaction: sv_pain_para * STA_SAI_Score * STA_TAI_Score * PCS_Score']

#     elif version == 53:  
#         params_of_interest = [
#             'sv', 'sz', 'st',
#             'v_Intercept', 'v_sv_pain_para', 
#             'v_STA_SAI_Score', 'v_STA_TAI_Score', 'v_PCS_Score', 
#             'v_sv_pain_para:STA_SAI_Score', 'v_sv_pain_para:STA_TAI_Score', 'v_sv_pain_para:PCS_Score', 
#             'v_sv_pain_para:STA_SAI_Score:STA_TAI_Score', 'v_sv_pain_para:STA_SAI_Score:PCS_Score', 'v_sv_pain_para:STA_TAI_Score:PCS_Score', 
#             'v_sv_pain_para:STA_SAI_Score:STA_TAI_Score:PCS_Score',
#             't_Intercept', 't_sv_pain_para', 
#             't_STA_SAI_Score', 't_STA_TAI_Score', 't_PCS_Score', 
#             't_sv_pain_para:STA_SAI_Score', 't_sv_pain_para:STA_TAI_Score', 't_sv_pain_para:PCS_Score', 
#             't_sv_pain_para:STA_SAI_Score:STA_TAI_Score', 't_sv_pain_para:STA_SAI_Score:PCS_Score', 't_sv_pain_para:STA_TAI_Score:PCS_Score', 
#             't_sv_pain_para:STA_SAI_Score:STA_TAI_Score:PCS_Score',
#             'a_Intercept', 'a_sv_pain_para', 
#             'a_STA_SAI_Score', 'a_STA_TAI_Score', 'a_PCS_Score', 
#             'a_sv_pain_para:STA_SAI_Score', 'a_sv_pain_para:STA_TAI_Score', 'a_sv_pain_para:PCS_Score', 
#             'a_sv_pain_para:STA_SAI_Score:STA_TAI_Score', 'a_sv_pain_para:STA_SAI_Score:PCS_Score', 'a_sv_pain_para:STA_TAI_Score:PCS_Score', 
#             'a_sv_pain_para:STA_SAI_Score:STA_TAI_Score:PCS_Score',
#             'z_Intercept', 'z_sv_pain_para', 
#             'z_STA_SAI_Score', 'z_STA_TAI_Score', 'z_PCS_Score', 
#             'z_sv_pain_para:STA_SAI_Score', 'z_sv_pain_para:STA_TAI_Score', 'z_sv_pain_para:PCS_Score', 
#             'z_sv_pain_para:STA_SAI_Score:STA_TAI_Score', 'z_sv_pain_para:STA_SAI_Score:PCS_Score', 'z_sv_pain_para:STA_TAI_Score:PCS_Score', 
#             'z_sv_pain_para:STA_SAI_Score:STA_TAI_Score:PCS_Score']
#         params_of_interest_s = [
#             'sv_subj', 'sz_subj', 'st_subj',
#             'v_Intercept_subj', 'v_sv_pain_para_subj', 
#             'v_STA_SAI_Score_subj', 'v_STA_TAI_Score_subj', 'v_PCS_Score_subj', 
#             'v_sv_pain_para:STA_SAI_Score_subj', 'v_sv_pain_para:STA_TAI_Score_subj', 'v_sv_pain_para:PCS_Score_subj', 
#             'v_sv_pain_para:STA_SAI_Score:STA_TAI_Score_subj', 'v_sv_pain_para:STA_SAI_Score:PCS_Score_subj', 'v_sv_pain_para:STA_TAI_Score:PCS_Score_subj', 
#             'v_sv_pain_para:STA_SAI_Score:STA_TAI_Score:PCS_Score_subj',
#             't_Intercept_subj', 't_sv_pain_para_subj', 
#             't_STA_SAI_Score_subj', 't_STA_TAI_Score_subj', 't_PCS_Score_subj', 
#             't_sv_pain_para:STA_SAI_Score_subj', 't_sv_pain_para:STA_TAI_Score_subj', 't_sv_pain_para:PCS_Score_subj', 
#             't_sv_pain_para:STA_SAI_Score:STA_TAI_Score_subj', 't_sv_pain_para:STA_SAI_Score:PCS_Score_subj', 't_sv_pain_para:STA_TAI_Score:PCS_Score_subj', 
#             't_sv_pain_para:STA_SAI_Score:STA_TAI_Score:PCS_Score_subj',
#             'a_Intercept_subj', 'a_sv_pain_para_subj', 
#             'a_STA_SAI_Score_subj', 'a_STA_TAI_Score_subj', 'a_PCS_Score_subj', 
#             'a_sv_pain_para:STA_SAI_Score_subj', 'a_sv_pain_para:STA_TAI_Score_subj', 'a_sv_pain_para:PCS_Score_subj', 
#             'a_sv_pain_para:STA_SAI_Score:STA_TAI_Score_subj', 'a_sv_pain_para:STA_SAI_Score:PCS_Score_subj', 'a_sv_pain_para:STA_TAI_Score:PCS_Score_subj', 
#             'a_sv_pain_para:STA_SAI_Score:STA_TAI_Score:PCS_Score_subj',
#             'z_Intercept_subj', 'z_sv_pain_para_subj', 
#             'z_STA_SAI_Score_subj', 'z_STA_TAI_Score_subj', 'z_PCS_Score_subj', 
#             'z_sv_pain_para:STA_SAI_Score_subj', 'z_sv_pain_para:STA_TAI_Score_subj', 'z_sv_pain_para:PCS_Score_subj', 
#             'z_sv_pain_para:STA_SAI_Score:STA_TAI_Score_subj', 'z_sv_pain_para:STA_SAI_Score:PCS_Score_subj', 'z_sv_pain_para:STA_TAI_Score:PCS_Score_subj', 
#             'z_sv_pain_para:STA_SAI_Score:STA_TAI_Score:PCS_Score_subj']
#         titles = [
#             'Inter-trial variability in drift rate', 'Inter-trial variability in starting point', 'Inter-trial variability in non-decision time',
#             'Intercept drift rate', 'Drift rate sv_pain_para',
#             'Effect of STA_SAI_Score', 'Effect of STA_TAI_Score', 'Effect of PCS_Score', 
#             'Interaction: sv_pain_para * STA_SAI_Score', 'Interaction: sv_pain_para * STA_TAI_Score', 'Interaction: sv_pain_para * PCS_Score', 
#             'Interaction: sv_pain_para * STA_SAI_Score * STA_TAI_Score', 'Interaction: sv_pain_para * STA_SAI_Score * PCS_Score', 'Interaction: sv_pain_para * STA_TAI_Score * PCS_Score', 
#             'Interaction: sv_pain_para * STA_SAI_Score * STA_TAI_Score * PCS_Score',
#             'Intercept non-decision time', 'Non-decision time sv_pain_para',
#             'Effect of STA_SAI_Score', 'Effect of STA_TAI_Score', 'Effect of PCS_Score', 
#             'Interaction: sv_pain_para * STA_SAI_Score', 'Interaction: sv_pain_para * STA_TAI_Score', 'Interaction: sv_pain_para * PCS_Score', 
#             'Interaction: sv_pain_para * STA_SAI_Score * STA_TAI_Score', 'Interaction: sv_pain_para * STA_SAI_Score * PCS_Score', 'Interaction: sv_pain_para * STA_TAI_Score * PCS_Score', 
#             'Interaction: sv_pain_para * STA_SAI_Score * STA_TAI_Score * PCS_Score',
#             'Intercept boundary separation', 'Boundary separation sv_pain_para',
#             'Effect of STA_SAI_Score', 'Effect of STA_TAI_Score', 'Effect of PCS_Score', 
#             'Interaction: sv_pain_para * STA_SAI_Score', 'Interaction: sv_pain_para * STA_TAI_Score', 'Interaction: sv_pain_para * PCS_Score', 
#             'Interaction: sv_pain_para * STA_SAI_Score * STA_TAI_Score', 'Interaction: sv_pain_para * STA_SAI_Score * PCS_Score', 'Interaction: sv_pain_para * STA_TAI_Score * PCS_Score', 
#             'Interaction: sv_pain_para * STA_SAI_Score * STA_TAI_Score * PCS_Score',
#             'Intercept starting point', 'Starting point sv_pain_para',
#             'Effect of STA_SAI_Score', 'Effect of STA_TAI_Score', 'Effect of PCS_Score', 
#             'Interaction: sv_pain_para * STA_SAI_Score', 'Interaction: sv_pain_para * STA_TAI_Score', 'Interaction: sv_pain_para * PCS_Score', 
#             'Interaction: sv_pain_para * STA_SAI_Score * STA_TAI_Score', 'Interaction: sv_pain_para * STA_SAI_Score * PCS_Score', 'Interaction: sv_pain_para * STA_TAI_Score * PCS_Score', 
#             'Interaction: sv_pain_para * STA_SAI_Score * STA_TAI_Score * PCS_Score']
    
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


if __name__ == "__main__":
    
    
    #data:
    # hddm_models_path = os.path.join(current_directory,'Hddm_models')
    # sys.path.append(hddm_models_path)
    # data_path1 = os.path.join(current_directory, 'data_sets', 'behavioural_sv_cleaned_final_3.csv')
    # data = pd.read_csv(data_path1, sep = ',')
    # data.dropna(subset=['rt', "painlevel", "moneylevel", "accepted",'acceptance_pair','sv_money', 'sv_pain', 'sv_both', 'p_pain_all', 'Abs_Money_Pain','OV_Money_Pain', 'sv_pain_para','sv_both_para','k_pain_para','beta_para','bias_para','STA_SAI_Score','STA_TAI_Score','PCS_Score'], inplace = True)    #'STA_SAI_Score','STA_TAI_Score','PCS_Score'
    
    # drop entire participants for quest data only, NO FOR ENTIRE DATA, otherwise the operating system kills the worker
    # quest_vers = [x, z, u, i]  # questionnaire versions 
    # if version in quest_vers:
    #     data.dropna(subset=["STA_SAI_Score","STA_TAI_Score","PCS_Score"], inplace=True)
    


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
            
                       
            # keep only trials with strictly positive dwell time on both sides, this can be changed; depends on the goal
            #data = data[(data["DwellLeft"] > -1) & (data["DwellRight"] > -1)]
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


#_________________________________________________________________________________________________________________________________________________________________________________________________
# Getting the tiral-by trial param betas (influenced by sv_pain_para) for the EEG regression analysis

# for model NR1
def v_sv_pain_para_contributions(models, data):
    
    data_with_v_sv_pain_para = data.copy()
    data_with_v_sv_pain_para['v_sv_pain_para_contrib'] = np.nan
   
   # looping through subjects and concatenate all thhe models, get the sub-specific paramter from the posterior nodes
    for subj_id in data['subj_idx'].unique():
        model = kabuki.utils.concat_models(models)  
        subj_data = data[data['subj_idx'] == subj_id]
        v_sv_pain_para = model.nodes_db.loc[f'v_sv_pain_para_subj.{subj_id}', 'node'].trace()
        v_sv_pain_para_contrib_list = []
        
        # sv_pain_para weight on dirft rate for every trial and participant
        for idx, trial in subj_data.iterrows():
            trial_sv_pain_para = trial['sv_pain_para'] 
            # weight of sv_pain_para on the drift rate from model 1
            v_sv_pain_para_contrib_samples = v_sv_pain_para * trial_sv_pain_para
            # simple trace mean just for v_sv_pain_para
            v_sv_pain_para_contrib_mean = v_sv_pain_para_contrib_samples.mean()    
            v_sv_pain_para_contrib_list.append(v_sv_pain_para_contrib_mean)
            data_with_v_sv_pain_para.loc[idx, 'v_sv_pain_para_contrib'] = v_sv_pain_para_contrib_mean  

    return data_with_v_sv_pain_para


def a_sv_pain_para_contributions(models, data):
    
    data_with_a_sv_pain_para = data.copy()
    data_with_a_sv_pain_para['a_sv_pain_para_contrib'] = np.nan
   
   # looping through subjects and concatenate all thhe models, get the sub-specific paramter from the posterior nodes
    for subj_id in data['subj_idx'].unique():
        model = kabuki.utils.concat_models(models)  
        subj_data = data[data['subj_idx'] == subj_id]
        a_sv_pain_para = model.nodes_db.loc[f'a_sv_pain_para_subj.{subj_id}', 'node'].trace()
        a_sv_pain_para_contrib_list = []
        
        # sv_pain_para weight on dirft rate for every trial and participant
        for idx, trial in subj_data.iterrows():
            trial_sv_pain_para = trial['sv_pain_para'] 
            # weight of sv_pain_para on the drift rate from model 1
            a_sv_pain_para_contrib_samples = a_sv_pain_para * trial_sv_pain_para
            # simple trace mean just for v_sv_pain_para
            a_sv_pain_para_contrib_mean = a_sv_pain_para_contrib_samples.mean()    
            a_sv_pain_para_contrib_list.append(a_sv_pain_para_contrib_mean)
            data_with_a_sv_pain_para.loc[idx, 'a_sv_pain_para_contrib'] = a_sv_pain_para_contrib_mean  

    return data_with_a_sv_pain_para


def t_sv_pain_para_contributions(models, data):
    
    data_with_t_sv_pain_para = data.copy()
    data_with_t_sv_pain_para['t_sv_pain_para_contrib'] = np.nan
   
   # looping through subjects and concatenate all thhe models, get the sub-specific paramter from the posterior nodes
    for subj_id in data['subj_idx'].unique():
        model = kabuki.utils.concat_models(models)  
        subj_data = data[data['subj_idx'] == subj_id]
        t_sv_pain_para = model.nodes_db.loc[f't_sv_pain_para_subj.{subj_id}', 'node'].trace()
        t_sv_pain_para_contrib_list = []
        
        # sv_pain_para weight on dirft rate for every trial and participant
        for idx, trial in subj_data.iterrows():
            trial_sv_pain_para = trial['sv_pain_para'] 
            # weight of sv_pain_para on the drift rate from model 1
            t_sv_pain_para_contrib_samples = t_sv_pain_para * trial_sv_pain_para
            # simple trace mean just for v_sv_pain_para
            t_sv_pain_para_contrib_mean = t_sv_pain_para_contrib_samples.mean()    
            t_sv_pain_para_contrib_list.append(t_sv_pain_para_contrib_mean)
            data_with_t_sv_pain_para.loc[idx, 't_sv_pain_para_contrib'] = t_sv_pain_para_contrib_mean  

    return data_with_t_sv_pain_para



def z_sv_pain_para_contributions(models, data):
    
    data_with_z_sv_pain_para = data.copy()
    data_with_z_sv_pain_para['z_sv_pain_para_contrib'] = np.nan
   
   # looping through subjects and concatenate all thhe models, get the sub-specific paramter from the posterior nodes
    for subj_id in data['subj_idx'].unique():
        model = kabuki.utils.concat_models(models)  
        subj_data = data[data['subj_idx'] == subj_id]
        z_sv_pain_para = model.nodes_db.loc[f'z_sv_pain_para_subj.{subj_id}', 'node'].trace()
        z_sv_pain_para_contrib_list = []
        
        # sv_pain_para weight on dirft rate for every trial and participant
        for idx, trial in subj_data.iterrows():
            trial_sv_pain_para = trial['sv_pain_para'] 
            # weight of sv_pain_para on the drift rate from model 1
            z_sv_pain_para_contrib_samples = z_sv_pain_para * trial_sv_pain_para
            # simple trace mean just for v_sv_pain_para
            z_sv_pain_para_contrib_mean = z_sv_pain_para_contrib_samples.mean()    
            z_sv_pain_para_contrib_list.append(z_sv_pain_para_contrib_mean)
            data_with_z_sv_pain_para.loc[idx, 'z_sv_pain_para_contrib'] = z_sv_pain_para_contrib_mean  

    return data_with_z_sv_pain_para








# for model NR2
def full_sv_pain_para_contributions(models, data):
    data_full_sv_pain_para = data.copy()

    data_full_sv_pain_para['full_v_sv_pain_para_contrib'] = np.nan
    data_full_sv_pain_para['full_a_sv_pain_para_contrib'] = np.nan
    data_full_sv_pain_para['full_t_sv_pain_para_contrib'] = np.nan
    data_full_sv_pain_para['full_z_sv_pain_para_contrib'] = np.nan

    for subj_id in data['subj_idx'].unique():
        model = kabuki.utils.concat_models(models)  
        subj_data = data[data['subj_idx'] == subj_id]

        #Get subject-specific paramter from the posteriors
        v_sv_pain_para = model.nodes_db.loc[f'v_sv_pain_para_subj.{subj_id}', 'node'].trace()
        a_sv_pain_para = model.nodes_db.loc[f'a_sv_pain_para_subj.{subj_id}', 'node'].trace()
        t_sv_pain_para = model.nodes_db.loc[f't_sv_pain_para_subj.{subj_id}', 'node'].trace()
        z_sv_pain_para = model.nodes_db.loc[f'z_sv_pain_para_subj.{subj_id}', 'node'].trace()

        v_sv_pain_para_contrib_list = []
        a_sv_pain_para_contrib_list = []
        t_sv_pain_para_contrib_list = []
        z_sv_pain_para_contrib_list = []
        
        # sv_pain_para weight on dirft rate for every trial and participant
        for idx, trial in subj_data.iterrows():
            trial_sv_pain_para = trial['sv_pain_para']
            
            # weight of sv_pain_para on the params from model 2
            v_sv_pain_para_contrib_samples = v_sv_pain_para * trial_sv_pain_para
            a_sv_pain_para_contrib_samples = a_sv_pain_para * trial_sv_pain_para
            t_sv_pain_para_contrib_samples = t_sv_pain_para * trial_sv_pain_para
            z_sv_pain_para_contrib_samples = z_sv_pain_para * trial_sv_pain_para
            
            # simple trace mean just for v_sv_pain_para
            v_sv_pain_para_trace_mean = v_sv_pain_para.mean()  
            v_sv_pain_para_contrib_mean = v_sv_pain_para_contrib_samples.mean()
            a_sv_pain_para_trace_mean = a_sv_pain_para.mean()  
            a_sv_pain_para_contrib_mean = a_sv_pain_para_contrib_samples.mean()
            t_sv_pain_para_trace_mean = t_sv_pain_para.mean()  
            t_sv_pain_para_contrib_mean = t_sv_pain_para_contrib_samples.mean()
            z_sv_pain_para_trace_mean = z_sv_pain_para.mean()  
            z_sv_pain_para_contrib_mean = z_sv_pain_para_contrib_samples.mean()
            
            v_sv_pain_para_contrib_list.append(v_sv_pain_para_contrib_mean)
            a_sv_pain_para_contrib_list.append(a_sv_pain_para_contrib_mean)
            t_sv_pain_para_contrib_list.append(t_sv_pain_para_contrib_mean)
            z_sv_pain_para_contrib_list.append(z_sv_pain_para_contrib_mean)

            data_full_sv_pain_para.loc[idx, 'full_v_sv_pain_para_contrib'] = v_sv_pain_para_contrib_mean 
            data_full_sv_pain_para.loc[idx, 'full_a_sv_pain_para_contrib'] = a_sv_pain_para_contrib_mean  
            data_full_sv_pain_para.loc[idx, 'full_t_sv_pain_para_contrib'] = t_sv_pain_para_contrib_mean 
            data_full_sv_pain_para.loc[idx, 'full_z_sv_pain_para_contrib'] = z_sv_pain_para_contrib_mean  

    return data_full_sv_pain_para
    

# for model NR3
def v_sv_money_contributions(models, data):
    data_sv_money = data.copy()
    data_sv_money['v_sv_money_contrib'] = np.nan
    
    for subj_id in data['subj_idx'].unique():
        model = kabuki.utils.concat_models(models)  
        subj_data = data[data['subj_idx'] == subj_id]
        v_sv_money_contrib_list = []
        v_sv_money = model.nodes_db.loc[f'v_sv_money_subj.{subj_id}', 'node'].trace()
        v_sv_money_contrib_list = []
        
        for idx, trial in subj_data.iterrows():
            trial_sv_money = trial['sv_money']
            v_sv_money_contrib_samples = v_sv_money * trial_sv_money
            v_sv_money_contrib_mean = v_sv_money_contrib_samples.mean()  
            v_sv_money_contrib_list.append(v_sv_money_contrib_mean)
            data_sv_money.loc[idx, 'v_sv_money_contrib'] = v_sv_money_contrib_mean  

    return data_sv_money


# model NR.16
def v_sv_pain_para_Abs_contributions(models, data):
    data_abs_sv_pain_para = data.copy()
    data_abs_sv_pain_para['v_sv_pain_para_Abslow_contrib'] = np.nan
    data_abs_sv_pain_para['v_sv_pain_para_Absmid_contrib'] = np.nan
    data_abs_sv_pain_para['v_sv_pain_para_Abshigh_contrib'] = np.nan

    for subj_id in data['subj_idx'].unique():
        model = kabuki.utils.concat_models(models)  
        subj_data = data[data['subj_idx'] == subj_id]
        
        v_sv_pain_para_Abslow = model.nodes_db.loc[f'v_sv_pain_para:C(Abs_value)[low_abs]_subj.{subj_id}', 'node'].trace()
        v_sv_pain_para_Absmid = model.nodes_db.loc[f'v_sv_pain_para:C(Abs_value)[mid_abs]_subj.{subj_id}', 'node'].trace()
        v_sv_pain_para_Abshigh = model.nodes_db.loc[f'v_sv_pain_para:C(Abs_value)[high_abs]_subj.{subj_id}', 'node'].trace()

        v_sv_pain_para_Abslow_contrib_list = []
        v_sv_pain_para_Absmid_contrib_list = []
        v_sv_pain_para_Abshigh_contrib_list = []

        for idx, trial in subj_data.iterrows():
            trial_sv_pain_para = trial['sv_pain_para'] 
            v_sv_pain_para_Abslow_contrib_samples = v_sv_pain_para_Abslow * trial_sv_pain_para
            v_sv_pain_para_Absmid_contrib_samples = v_sv_pain_para_Absmid * trial_sv_pain_para
            v_sv_pain_para_Abshigh_contrib_samples = v_sv_pain_para_Abshigh * trial_sv_pain_para

            v_sv_pain_para_Abslow_contrib_mean = v_sv_pain_para_Abslow_contrib_samples.mean()    
            v_sv_pain_para_Absmid_contrib_mean = v_sv_pain_para_Absmid_contrib_samples.mean()    
            v_sv_pain_para_Abshigh_contrib_mean = v_sv_pain_para_Abshigh_contrib_samples.mean()    
            
            v_sv_pain_para_Abslow_contrib_list.append(v_sv_pain_para_Abslow_contrib_mean)
            data_abs_sv_pain_para.loc[idx, 'v_sv_pain_para_Abslow_contrib'] = v_sv_pain_para_Abslow_contrib_mean  
            v_sv_pain_para_Absmid_contrib_list.append(v_sv_pain_para_Absmid_contrib_mean)
            data_abs_sv_pain_para.loc[idx, 'v_sv_pain_para_Absmid_contrib'] = v_sv_pain_para_Absmid_contrib_mean  
            v_sv_pain_para_Abshigh_contrib_list.append(v_sv_pain_para_Abshigh_contrib_mean)
            data_abs_sv_pain_para.loc[idx, 'v_sv_pain_para_Abshigh_contrib'] = v_sv_pain_para_Abshigh_contrib_mean  

    return data_abs_sv_pain_para

# model NR.17
def v_sv_pain_para_OV_contributions(models, data):
    data_ov_sv_pain_para = data.copy()
    data_ov_sv_pain_para['v_sv_pain_para_OVlow_contrib'] = np.nan
    data_ov_sv_pain_para['v_sv_pain_para_OVhigh_contrib'] = np.nan

    for subj_id in data['subj_idx'].unique():
        model = kabuki.utils.concat_models(models)  
        subj_data = data[data['subj_idx'] == subj_id]
        
        v_sv_pain_para_ovlow = model.nodes_db.loc[f'v_sv_pain_para:C(OV_value)[low_OV]_subj.{subj_id}', 'node'].trace()
        v_sv_pain_para_ovhigh = model.nodes_db.loc[f'v_sv_pain_para:C(OV_value)[high_OV]_subj.{subj_id}', 'node'].trace()
        v_sv_pain_para_ovlow_contrib_list = []
        v_sv_pain_para_ovhigh_contrib_list = []

        for idx, trial in subj_data.iterrows():
            trial_sv_pain_para = trial['sv_pain_para'] 
            v_sv_pain_para_ovlow_contrib_samples = v_sv_pain_para_ovlow * trial_sv_pain_para
            v_sv_pain_para_ovhigh_contrib_samples = v_sv_pain_para_ovhigh * trial_sv_pain_para

            v_sv_pain_para_ovlow_contrib_mean = v_sv_pain_para_ovlow_contrib_samples.mean()    
            v_sv_pain_para_ovshigh_contrib_mean = v_sv_pain_para_ovhigh_contrib_samples.mean()    
            
            v_sv_pain_para_ovlow_contrib_list.append(v_sv_pain_para_ovlow_contrib_mean)
            data_ov_sv_pain_para.loc[idx, 'v_sv_pain_para_OVlow_contrib'] = v_sv_pain_para_ovlow_contrib_mean  
            v_sv_pain_para_ovhigh_contrib_list.append(v_sv_pain_para_ovshigh_contrib_mean)
            data_ov_sv_pain_para.loc[idx, 'v_sv_pain_para_OVhigh_contrib'] = v_sv_pain_para_ovshigh_contrib_mean  
            
    return data_ov_sv_pain_para
    
## Function to run the models or load and analyse the models
# if run:
#     print('Running {}'.format(model_base_name + model_name))
#     models = drift_diffusion_hddm(data=data,
#                                   samples=nr_samples,
#                                   n_jobs=nr_models,
#                                   run=run,
#                                   parallel=parallel,
#                                   model_name=model_base_name + model_name,
#                                   model_dir=model_dir, 
#                                   version=version,
#                                   accuracy_coding=False)
# else:
#     models = drift_diffusion_hddm(data=data,
#                                   samples=nr_samples,
#                                   n_jobs=nr_models,
#                                   run=run, 
#                                   parallel=parallel, 
#                                   model_name=model_base_name + model_name, 
#                                   model_dir=model_dir, 
#                                   version=version, 
#                                   accuracy_coding=False)
#     analyze_model(models, fig_dir, nr_models, version)
#     if version == 1:
#         sv_contribute = v_sv_pain_para_contributions(models, data)
#         sv_contribute.to_csv(os.path.join(fig_dir, 'diagnostics', 'data_with_v_sv_pain_para_contrib.csv'), index=False)
#     elif version == 2:
#         sv_contribute = full_sv_pain_para_contributions(models, data)
#         sv_contribute.to_csv(os.path.join(fig_dir, 'diagnostics', 'data_with_full_sv_pain_para_contrib.csv'), index=False)
#     elif version == 3:
#         sv_contribute = v_sv_money_contributions(models, data)
#         sv_contribute.to_csv(os.path.join(fig_dir, 'diagnostics', 'data_with_v_sv_money_contrib.csv' ))
#     elif version == 16:
#         sv_contribute = v_sv_pain_para_Abs_contributions(models, data)
#         sv_contribute.to_csv(os.path.join(fig_dir, 'diagnostics', 'data_with_sv_pain_para_Abs_contrib.csv' ))
#     elif version == 17:
#         sv_contribute = v_sv_pain_para_OV_contributions(models, data)
#         sv_contribute.to_csv(os.path.join(fig_dir, 'diagnostics', 'data_with_sv_pain_para_OV_contrib.csv' ))


