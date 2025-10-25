# Veronika Wendler
# 22.01.25
# code for the attentional drift diffusion model
# - originally, I used a very basic version of this in summer 2024 in Quebec and was inspired by Jan Willem De Gee's Python2 code found somewhere on his GitHub - but this version is pretty much mine
 
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
from IPython import embed as shells
import hddm
import kabuki
import statsmodels.formula.api as sm
from patsy import dmatrix
from joblib import Parallel, delayed
import time
import arviz as az
import dill as pickle
import re
# warning settings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
# Plotting
# Stats 
from statsmodels.distributions.empirical_distribution import ECDF
# HDDM
from hddm.simulators.hddm_dataset_generators import simulator_h_c

from pathlib import Path

PROJECT_DIR = pathlib.Path(os.getenv("PROJECT_DIR", "/workspace"))

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)
import re
from pathlib import Path

import os
# disable _all_ Numba JIT caching & compilation
os.environ["NUMBA_DISABLE_JIT"] = "1"

import numba
numba.config.CACHE_ENABLE = False



#------------------------------------------------------------------------------------------------------------------
# Structure of saving:

#------------------------------------------------------------------------------------------------------------------

# addm regression formula
# v = β0 + β1 ⋅ (PropDwell_opt​ ⋅ V_opt​ − PropDwell_sub ⋅ V_sub) + β2 ⋅ (PropDwell_sub ⋅ V_opt​ − PropDwell_opt​ ⋅ V_sub)+ϵ
# where ß0 = intercept,
# ß1 = AttentionW,
# ß2 = InattentionW,

##
# ϵ = noise
# PropDwell_opt = proportion of dwell time on the option with higher expected value
# PropDwell_sub = proportion of dwell time on the option with lower expected value
# V_opt​ = value if the better option
# V_sub = value of the worse option

# params:
version = 1    # defining version #
run = False        # if True, the the models run, if False the models load

phase = ['dec']  #['ES', 'EE']  # Defines which phase you want ('ES', 'EE', 'LE', or the combinations)

if len(phase) == 1:
    phase_key = phase[0]  # single phase model (LE, ES, or EE)
else:
    raise ValueError(f"Invalid phase: {phase}")

phase = phase_key    

PROJECT_DIR   = pathlib.Path(os.getenv("PROJECT_DIR", "/workspace")).resolve()

BASE_MODEL_DIR = PROJECT_DIR / "Hddm_Docker_August_24/models_dir"
FIG_DIR_ROOT   = PROJECT_DIR / "Hddm_Docker_August_24/figures_dir"

model_base_name = "painreward_behavioural_data_"

nr_models       = 4         # number of MCMC chains
nr_samples      = 6000      # samples per chain - do 6000 (+1000 for burn-in) but for now for a quick one we do 600
parallel        = True      # parallel
model_base_name = "painreward_behavioural_data_"
model_versions  = {
    "dec":      ["LPP_0","LPP_1","LPP_2","LPP_3","LPP_4","LPP_5","LPP_6","LPP_7","LPP_8","LPP_9"]     
}

# debugging, tip, python starts at 0, unlike Matlab
# honestly, for whoever wants to run this I am really sorry because it's still quite messy, essentially, if you want to run a model e.g. model 1, you load the data from 0 (because of the indexing mismatch)
if phase not in model_versions:
    raise ValueError(f"Invalid phase '{phase}'. Choose from: {list(model_versions.keys())}")

PHASE_TO_SOURCE = {
    "dec": "decision", 
}

model_name = model_versions[phase][version]

# set the data path
#data_path1 = os.path.join(current_directory, 'data_sets/data_sets_Garcia', 'GarciaParticipants_Eye_Response_Feed_Allfix_addm_OV_Abs_CCT.csv')
#data = pd.read_csv(data_path1, sep=',')

data = pd.read_csv((PROJECT_DIR / "Hddm_Docker_August_24" / "data_sets" / "behavioural_sv_cleaned_final_3.csv").as_posix(), sep=",")
source_phase = PHASE_TO_SOURCE.get(phase, phase)  

if phase == "dec":
    data = data[data["TaskName"].isin(["decision"])].copy()
elif phase == "pas":
    data = data[data["TaskName"].isin(["passive"])].copy()
else:
    data = data[data["TaskName"] == source_phase].copy() 

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
            
exclude_part = {}   # there's a nr of reasons as to why to exclude these ones (e.g. missing edf files, not enough fixations etc..)

#data = data[data['phase'] == phase]

data = data[~data['subj_idx'].isin(exclude_part)]    
data.dropna(subset=['rt', "painlevel","moneylevel",
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


# debugging information
print(f"\nFiltering data for phase: {phase}")
print("Unique phases in filtered data:", data['TaskName'].unique())
print(f"Data shape after filtering: {data.shape}")
print(f"Unique participants in filtered data: {data['subj_idx'].unique()}")
print(f"Selected phase_key: {phase_key}")
print(f"Model to run: {model_base_name + model_name}")
print(f"Filtered Data Unique Phases: {data['TaskName'].unique()}")
print(f"Data Shape After Filtering: {data.shape}")   


# this is the response histogram for correct and incorrect responses
# data.loc[data['response'] == 0, 'rt'] = -data.loc[data['response'] == 0, 'rt']
# fig = plt.figure()
# ax = fig.add_subplot(111, xlabel='RT', ylabel='count', title='RT distributions')
# ax.set_xlim(-10, 10)
# for i, subj_data in data.groupby('subj_idx'):
#     subj_data['rt'].hist(bins=20, histtype='step', ax=ax)
# plt.show()
# #data
#------------------------------------------------------------------------------------------------------------------
#Flipping Errors only for EE and ES phases the RL model does not work on this
# data = hddm.utils.flip_errors(data)
    
# Plotting RT distributions
fig = plt.figure(figsize=(12, 8))
ax  = fig.add_subplot(111, xlabel='RT', ylabel='count', title='RT distributions')
for _, subj_data in data.groupby('subj_idx'):
    subj_data.rt.hist(bins=20, histtype='step', ax=ax)
# instead of plt.show():
fig.savefig((FIG_DIR_ROOT / f"{model_base_name}{model_name}" / "diagnostics" / "rt_distributions.pdf").as_posix(),
            bbox_inches="tight")
plt.close(fig)


# Functions 
#-------------------------------------------------------------------------------------------------------------------

# ensure directory exists
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# model dir:
model_dir = BASE_MODEL_DIR
ensure_dir(model_dir)

def sanitize_infdata(infdata):
    """Convert pd.NA values to np.nan in all groups of the InferenceData object (important for if you have columns which you don't use, for example)."""
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


def _sanitize_filename(fname):
    # replace any of : ( ) [ ] , with underscore
    safe = re.sub(r'[:\(\)\[\],]', '_', fname)
    # collapse runs of underscores to a single underscore
    safe = re.sub(r'_+', '_', safe)
    return safe

def _inv_logit(x):
    return 1.0 / (1.0 + np.exp(-x))

def _summ_from_samples(arr_1d):
    arr = np.asarray(arr_1d).ravel()
    qs = np.percentile(arr, [2.5, 25, 50, 75, 97.5])
    return {
        "mean": float(np.mean(arr)),
        "std":  float(np.std(arr, ddof=1)),
        "2.5q": qs[0],
        "25q":  qs[1],
        "50q":  qs[2],
        "75q":  qs[3],
        "97.5q":qs[4],
    }


# here, I am drawing 1000 posterior samples for PPC instead of mean, SD (for For_model 7) 
def export_posterior_draws(model_name, model_dir, n_jobs=3, S=1000):
    # load and combine chains
    idatas = [az.from_netcdf(Path(model_dir) / f"{model_name}_{i}.nc") for i in range(n_jobs)]
    idata  = az.concat(idatas, dim="chain")
    post = idata.posterior.stack(sample=("chain","draw"))

    n_samps = post.sizes["sample"]
    idx = np.random.choice(n_samps, size=min(S, n_samps), replace=False)
    post_s = post.isel(sample=idx)
    
    # data frame
    all_params = list(post_s.data_vars)
    df_all = post_s[all_params].to_dataframe().reset_index(drop=True)
    out_csv = Path(model_dir) / f"{model_name}_posterior_draws.csv"
    df_all.to_csv(out_csv, index=False)
    print(f"Saved {df_all.shape[0]} draws × {df_all.shape[1]} columns to {out_csv}")

    return out_csv


fig_dir = FIG_DIR_ROOT / f"{model_base_name}{model_name}"
ensure_dir(fig_dir / "diagnostics")

# try:
#     os.system('mkdir {}'.format(fig_dir))
#     os.system('mkdir {}'.format(os.path.join(fig_dir, 'diagnostics')))
# except:
#     pass

# fig_dir = FIG_DIR_ROOT / full_model_name
# ensure_dir(fig_dir / "diagnostics")

## subjects
subjects = np.unique(data.subj_idx)
nr_subjects = subjects.shape[0]
print(nr_subjects)
print(subjects)


###################################################################################################################
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
            a_reg = {'model': 'a ~ 1 + sv_pain_para', 'link_func': lambda x: x}
            reg_descr = [a_reg]
        elif version == 3:  # drift rate is dependent on the the sv_pain_para
            t_reg = {'model': 't ~ 1 + sv_pain_para', 'link_func': lambda x: x}
            reg_descr = [t_reg]    
        elif version == 4:  # drift rate is dependent on the the sv_pain_para
            z_reg = {'model': 'z ~ 1 + sv_pain_para', 'link_func': lambda x: x}
            reg_descr = [z_reg]
        elif version == 5:  # drift rate is dependent on the the sv_pain_para
            v_reg = {'model': 'v ~ sv_pain_para', 'link_func': lambda x: x}
            reg_descr = [v_reg]
        elif version == 6:  # drift rate is dependent on the the sv_pain_para
            a_reg = {'model': 'a ~ sv_pain_para', 'link_func': lambda x: x}
            reg_descr = [a_reg]
        elif version == 7:  # drift rate is dependent on the the sv_pain_para
            t_reg = {'model': 't ~ sv_pain_para', 'link_func': lambda x: x}
            reg_descr = [t_reg]    
        elif version == 8:  # drift rate is dependent on the the sv_pain_para
            z_reg = {'model': 'z ~ sv_pain_para', 'link_func': lambda x: x}
            reg_descr = [z_reg]       
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

###############################################################################################################    
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
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
                         version=version,
                         phase=phase,
                         accuracy_coding=True):

    if run:
        if parallel:
            start_time = time.time()
            results = Parallel(n_jobs=n_jobs)(
                delayed(run_model)(trace_id,
                                   data,
                                   model_dir,
                                   model_name,
                                   version, 
                                   samples,
                                   accuracy_coding
                                   ) 
                for trace_id in range(n_jobs)
            )
            print("Time elapsed:", time.time() - start_time, "s")
            
            # for i in range(n_jobs):
            #     model = results[i]
                
            #     #HDDM format
            #     model.save(os.path.join(model_dir, f"{model_name}_{i}.hddm"))

            #     with open(os.path.join(model_dir, f"{model_name}_{i}.pkl"), "wb") as f:
            #         pickle.dump(model, f)  
                    
            for i in range(n_jobs):
                model, infdata = results[i]
                model.save(os.path.join(model_dir, f"{model_name}_{i}.hddm"))
                with open(os.path.join(model_dir, f"{model_name}_{i}.pkl"), "wb") as f:
                    pickle.dump(model, f)
                infdata = sanitize_infdata(infdata)  #clean
                az.to_netcdf(infdata, os.path.join(model_dir, f"{model_name}_{i}.nc"))


        else:
            # model = run_model(1,
            #                   data,
            #                   model_dir,
            #                   model_name,
            #                   version, 
            #                   samples,
            #                   accuracy_coding 
            #                   )
            
            # model.save(os.path.join(model_dir, model_name + ".hddm"))

            # with open(os.path.join(model_dir, f"{model_name}_{i}.pkl"), "wb") as f:
            #     pickle.dump(model, f)  
            
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

def drift_diffusion_hddmRL(data, 
                         samples=11000, #6000
                         n_jobs=5,
                         run=True,
                         parallel=True,
                         model_name='model',
                         model_dir='.', 
                         version=version,
                         phase=phase,
                         accuracy_coding=True):

    if run:
        if parallel:
            start_time = time.time()
            results = Parallel(n_jobs=n_jobs)(
                delayed(run_model)(trace_id,
                                   data,
                                   model_dir,
                                   model_name,
                                   version, 
                                   samples,
                                   accuracy_coding
                                   ) 
                for trace_id in range(n_jobs)
            )
            print("Time elapsed:", time.time() - start_time, "s")
            
            for i in range(n_jobs):
                model = results[i]
                
                # Save in HDDM format
                model.save(os.path.join(model_dir, f"{model_name}_{i}.hddm"))

                with open(os.path.join(model_dir, f"{model_name}_{i}.pkl"), "wb") as f:
                    model = pickle.load(f)

        else:
            model = run_model(1,
                              data,
                              model_dir,
                              model_name,
                              version, 
                              samples,
                              accuracy_coding 
                              )
            
            model.save(os.path.join(model_dir, model_name + ".hddm"))

            with open(os.path.join(model_dir, model_name + ".pkl"), 'wb') as f:
                pickle.dump(model, f)

    else:
        print('Loading existing models')
        # models = [hddm.load(os.path.join(model_dir, f"{model_name}_{i}.hddm")) for i in range(n_jobs)]
        # return models
      
        infdatas = []
        for i in range(n_jobs):
            nc_path = os.path.join(model_dir, f"{model_name}_{i}.nc")
            infdatas.append(az.from_netcdf(nc_path))
        return infdatas

    
#########################################################################################################################################################
#---------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------
# Analyzing the models

full_model_name = model_base_name + model_name
fig_dir        = FIG_DIR_ROOT / full_model_name
ensure_dir(fig_dir)
ensure_dir(fig_dir/"diagnostics")


def analyze_model(models, fig_dir, nr_models, version, phase):
    # 'sns.set_theme(style='darkgrid', font='sans-serif', font_scale=0.5)
    # # combine the models with kabuki utils
    # combined_model = kabuki.utils.concat_models(models)'
    
    print(f"Analyzing {len(models)} models for {phase}, version {version}")
    print(f"Saving figures to: {fig_dir}")

    sns.set_theme(style='darkgrid', font='sans-serif', font_scale=0.5)

    if not models or models[0] is None:
        print("ERROR: Models are empty or invalid.")
        return
    try:
        combined_model = kabuki.utils.concat_models(models)
        print("Models combined successfully.")
    except Exception as e:
        print(f"Error combining models: {e}")
        return
    
    # names parameters 
    
    if phase == 'dec':
        if version == 0:
            params_of_interest = ['z', 'a', 't', 'v']
            params_of_interest_s = [f'{p}_subj' for p in params_of_interest]
            titles = ['Starting point', 'Boundary sep.', 'Non-dec. time', 'Drift rate']
        elif version == 1:
            params_of_interest = ['z',
                                  'a',
                                  't', 
                                  'v_Intercept', 
                                  'v_sv_pain_para']
            params_of_interest_s = [f'{p}_subj' for p in params_of_interest]
            titles = [
                'Starting point',
                'Boundary sep.', 
                'Non-dec. time',
                'Drift Intercept',
                'Drift sv_pain_para']
        elif version == 2:
            params_of_interest = ['z',
                                  'a',
                                  't',  
                                  'v_sv_pain_para']
            params_of_interest_s = [f'{p}_subj' for p in params_of_interest]
            titles = [
                'Starting point',
                'Boundary sep.', 
                'Non-dec. time',
                'Drift sv_pain_para']
        else:
            raise ValueError(f"Invalid version {version}")
        
            
    elif phase == "LE_RL":
        if version == 0:
            params_of_interest = [
                "a",
                "t",
                "v",
                "alpha"
            ]
            params_of_interest_s = [p + "_subj" for p in params_of_interest]
            titles = [
                "a",
                "t",
                "v",
                "alpha",]
    
    
    
    # diagnistics
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
    
    # shrink font for the next set of plots
    matplotlib.rcParams.update({"font.size": 6})
    combined_model.plot_posteriors(save=True,
                                   path=str(diag_dir),
                                   format="pdf")
    matplotlib.rcParams.update({"font.size": 12})

    # stats table
    results = combined_model.gen_stats()
    results.to_csv(diag_dir / "results.csv")
    
    
    #  helper to get the trace 
    def _get_trace(model, name):
        try:
            return model.nodes_db.loc[name, "node"].trace()
        except Exception:
            return None
    
    # HORIZONTAL KDE PANEL FOR ATTENTION/INATTENTION WEIGHTS
    panel_params = [
        ("z",   "Starting point"),
        ("a",   "Boundary sep."),
        ("t",   "Non-dec. time"),
        ("v_Intercept", "Drift Intercept"),
        ("v_sv_pain_para", "Drift sv_pain_para"),   
    ]
    panel_traces = []
    panel_labels = []
    for p, label in panel_params:
        tr = _get_trace(combined_model, p)
        if tr is not None:
            panel_traces.append(np.asarray(tr))
            panel_labels.append(label)
    
    if panel_traces:
        # big fonts
        big_title_size = 27
        big_label_size = 26
        big_tick_size  = 24
    
        n = len(panel_traces)
        fig, axes = plt.subplots(
            1, n, figsize=(6.0 * n, 5), constrained_layout=True
        )
        if n == 1:
            axes = [axes]
    
        for ax, tr, label in zip(axes, panel_traces, panel_labels):
            # horizontal KDE
            sns.kdeplot(x=tr, fill=True, ax=ax)
            ax.axvline(0.0, ls="--", lw=1, color="grey")
            ax.set_title(label, fontsize=big_title_size, pad=12)
            ax.set_xlabel("Parameter value", fontsize=big_label_size, labelpad=6)
            ax.set_ylabel("Density", fontsize=big_label_size)
            ax.tick_params(axis="both", labelsize=big_tick_size, width=1.2)
            for side in ["top","right"]:
                ax.spines[side].set_visible(False)
            for side in ["left","bottom"]:
                ax.spines[side].set_linewidth(1.2)
    
        fig.suptitle("Posterior densities (horizontal)", fontsize=big_title_size+2)
        fig.savefig(diag_dir / "kde_horizontal.pdf", bbox_inches="tight")
        plt.close(fig)
    else:
        print("No traces found for attention/inattention weights; skipping panel.")
    
    
    group_params_to_plot = [
        'z',
        'a',
        't',
        'v_Intercept',
        'v_sv_pain_para'
        ]
    
    group_vplot_dir = diag_dir / "group_param_vertical_kdes"
    group_vplot_dir.mkdir(parents=True, exist_ok=True)
    
    # bigger, readable fonts
    vz_title = 27
    vz_label = 26
    vz_tick  = 24
    
    for param in group_params_to_plot:
        tr = _get_trace(combined_model, param)
        if tr is None:
            print(f"Skipping missing parameter: {param}")
            continue
    
        fig, ax = plt.subplots(figsize=(5, 8))
        sns.kdeplot(y=tr, fill=True, ax=ax)
        ax.set_facecolor("white")
    
        if param == "z":
            ax.axhline(0.5, color="red", linestyle="--", linewidth=5)
    
            # Two-sided posterior probability that z != 0.5
            tr_arr = np.asarray(tr)
            p_gt = np.mean(tr_arr > 0.5)
            p_lt = np.mean(tr_arr < 0.5)
            p_two_sided = 2 * min(p_gt, p_lt)
    
            # HDI for delta = z - 0.5 ( to check whether it's sig. differnet from 50%)
            delta = tr_arr - 0.5
            hdi_lo, hdi_hi = az.hdi(delta, hdi_prob=0.95).ravel()
            hdi_text = f"95% HDI(z-0.5)=[{hdi_lo:.3f}, {hdi_hi:.3f}]"
    
            # ROPE around 0.5 (0.02 by default similar to the tutorials by Pan et al., 2025)
            rope = 0.02
            p_in_rope = np.mean((np.abs(delta) <= rope))
    
            ax.set_title(
                f"{param}  |P(z!=0.5)={1-p_two_sided:.3f}\n{hdi_text} | P(|z-0.5|<={rope:.2f})={p_in_rope:.3f}",
                fontsize=vz_title, pad=12
            )
        else:
            ax.set_title(param, fontsize=vz_title, pad=12)
    
        ax.set_xlabel("Density", fontsize=vz_label, labelpad=10)
        ax.set_ylabel("Value", fontsize=vz_label)
        ax.tick_params(axis="both", labelsize=vz_tick, width=1.2)
        for side in ["top","right"]:
            ax.spines[side].set_visible(False)
        for side in ["left","bottom"]:
            ax.spines[side].set_linewidth(1.2)
    
        plt.tight_layout()
        fig.savefig(group_vplot_dir / f"{param}_vertical_kde_big.pdf", bbox_inches="tight")
        plt.close(fig)
    
    
    #  z-diagnostics text file
    z_trace = _get_trace(combined_model, "z")
    if z_trace is not None:
        z_arr = np.asarray(z_trace)
        delta = z_arr - 0.5
        p_gt = np.mean(z_arr > 0.5)
        p_lt = np.mean(z_arr < 0.5)
        p_two_sided = 2 * min(p_gt, p_lt)
        hdi_lo, hdi_hi = az.hdi(delta, hdi_prob=0.95).ravel()
        rope = 0.02
        p_in_rope = np.mean((np.abs(delta) <= rope))
    
        with open(diag_dir / "z_diagnostics.txt", "w") as f:
            f.write("z diagnostics (group-level)\n")
            f.write("---------------------------\n")
            f.write(f"mean(z)        = {z_arr.mean():.4f}\n")
            f.write(f"sd(z)          = {z_arr.std(ddof=1):.4f}\n")
            f.write(f"P(z > 0.5)     = {p_gt:.4f}\n")
            f.write(f"P(z < 0.5)     = {p_lt:.4f}\n")
            f.write(f"Two-sided P(z != 0.5) = {1 - p_two_sided:.4f}\n")
            f.write(f"95% HDI(z-0.5) = [{hdi_lo:.4f}, {hdi_hi:.4f}]  (excludes 0? {'YES' if (hdi_lo>0 or hdi_hi<0) else 'NO'})\n")
            f.write(f"ROPE +- {rope:.2f}: P(|z-0.5| <= ROPE) = {p_in_rope:.4f}\n")
    else:
        print("No group-level z trace found; skipping z_diagnostics.")
    
    
    for f in os.listdir(diag_dir):
        if not f.endswith('.pdf') and not f.endswith('.csv'):
            continue
        safe = _sanitize_filename(f)
        if safe != f:
            os.rename(diag_dir / f, diag_dir / safe)

# you can use this function in case you are interestd inseeign whether, at the individual level, parameter differences include 0 in HDI. THis is important because the group level estimate might hide lots of individual varibaility

# def plot_inatt_forest(
#     fig_dir,
#     model_dir,
#     model_base,
#     hdi_prob=0.95,
#     param_E="v_ES_InattentionW_E_subj",
#     param_S="v_ES_InattentionW_S_subj",
#     n_chains=3
#     ):
#     """
#     HDI forest plot from .nc posterior samples.
#     Also computes Bayes factor (Savage-Dickey) for group-level Δ = |S| - |E|.
#     """

#     from scipy.stats import gaussian_kde, norm

#     out_dir = Path(fig_dir) / "diagnostics"
#     out_dir.mkdir(parents=True, exist_ok=True)

#     # load nc files
#     nc_files = []
#     for c in range(n_chains):
#         candidate = Path(model_dir) / f"{model_base}_{c}.nc"
#         if candidate.exists():
#             nc_files.append(candidate)
#     if not nc_files:
#         print(f"[HDI] No .nc files found under {model_dir} for base '{model_base}_<chain>.nc'")
#         return

#     idatas = [az.from_netcdf(str(f)) for f in nc_files]
#     idata  = az.concat(idatas, dim="chain")
#     post   = idata.posterior.stack(sample=("chain", "draw"))

#     # find all subject-level vars
#     subj_E = [v for v in post.data_vars if v.startswith(param_E)]
#     subj_S = [v for v in post.data_vars if v.startswith(param_S)]

#     ids_E = {int(v.split(".")[-1]) for v in subj_E}
#     ids_S = {int(v.split(".")[-1]) for v in subj_S}
#     subj_ids = sorted(ids_E & ids_S)

#     if not subj_ids:
#         print("No overlapping subjects in .nc posterior")
#         return

#     rows = []
#     all_deltas = []
#     for subj in subj_ids:
#         keyE = f"{param_E}.{subj}"
#         keyS = f"{param_S}.{subj}"
#         E = np.abs(np.asarray(post[keyE]))
#         S = np.abs(np.asarray(post[keyS]))
#         delta = S - E
#         all_deltas.append(delta)

#         hdi_bounds = np.asarray(az.hdi(delta, hdi_prob=hdi_prob)).ravel()
#         hdi_low, hdi_high = float(hdi_bounds[0]), float(hdi_bounds[-1])

#         rows.append({
#             "subj": subj,
#             "delta_mean": float(delta.mean()),
#             "hdi_low": hdi_low,
#             "hdi_high": hdi_high,
#             "credible": int((hdi_low > 0) or (hdi_high < 0))
#         })

#     hdi_df = pd.DataFrame(rows).sort_values("subj")
#     hdi_csv = out_dir / "inatt_asymmetry_HDI.csv"
#     hdi_df.to_csv(hdi_csv, index=False)
#     print(f"[HDI] Saved: {hdi_csv}")

#     # group-level Bayes factor
#     group_delta = np.concatenate(all_deltas)
#     kde = gaussian_kde(group_delta)
#     post_at_0 = kde.evaluate([0])[0]

#     # prior density at 0
#     prior_at_0 = norm.pdf(0, loc=0, scale=1)

#     BF_01 = post_at_0 / prior_at_0
#     BF_10 = 1 / BF_01

#     bf_file = out_dir / "inatt_asymmetry_BayesFactor.txt"
#     with open(bf_file, "w") as f:
#         f.write(f"BF_01 (H0/H1): {BF_01:.3f}\n")
#         f.write(f"BF_10 (H1/H0): {BF_10:.3f}\n")

#     print(f"Saved Bayes factor results to {bf_file}")
#     print(f"  BF_01 = {BF_01:.3f}, BF_10 = {BF_10:.3f}")

#     # forest plot
#     fig, ax = plt.subplots(figsize=(6, 0.35 * len(hdi_df)))
#     ax.set_facecolor("white")
#     ax.grid(False)

#     ypos = np.arange(len(hdi_df))
#     for i, row in enumerate(hdi_df.itertuples(index=False)):
#         ax.plot([row.hdi_low, row.hdi_high], [ypos[i], ypos[i]], "k-", lw=1)
#         ax.plot(row.delta_mean, ypos[i], "o", color="purple")

#     ax.axvline(0, color="red", ls="--", lw=1)
#     ax.set_yticks(ypos)
#     ax.set_yticklabels(hdi_df["subj"])
#     ax.invert_yaxis()
#     ax.set_xlabel(f"Δ inattentional weight (|S| − |E|), {int(hdi_prob*100)}% HDI")
#     ax.set_title(f"Subject-level inattentional asymmetry (HDI)\nGroup BF_10={BF_10:.2f}")
#     fig.tight_layout()
#     fig.savefig(out_dir / "forest_inatt_asymmetry_HDI.pdf", bbox_inches="tight")
#     plt.close(fig)

    

def analyze_rl(infdatas, fig_dir, version):
    fig_dir = Path(fig_dir)
    diag_dir = fig_dir / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)
    # infdata is an arviz specific object
    # concatenate chains
    idata = az.concat(infdatas, dim="chain")
    print(idata)
    print(idata.posterior)
    print("Data variables in posterior:", list(idata.posterior.data_vars))

            
    rhat = az.rhat(idata)
    with open(diag_dir / "gelman_rubin.txt", "w") as f:
        for var in rhat.data_vars:
            val = float(rhat[var].values)  # extract scalar
            f.write(f"{var}: {val:.3f}\n")

    # optional other scores
    # waic_res = az.waic(idata)
    # dic_df = pd.DataFrame({
    #     "metric": ["DIC", "DIC_se"],
    #     "value":  [waic_res.waic, waic_res.waic_se]
    # })
    # dic_df.to_csv(diag_dir/"dic.csv", index=False)

    # Posterior predictive check
    # az.plot_ppc(idata)  
    # plt.savefig(diag_dir / "posterior_predictive.pdf")
    # plt.close()

    # stats table
    summary = az.summary(idata)
    summary.to_csv(diag_dir / "results.csv")

    # Collect posterior arrays
    if "alpha" not in idata.posterior.data_vars:
        print("[alpha-transform] No 'alpha' in posterior; skipping transformed CSV.")
        return

    # Transforming group-level alpha
    alpha_draws = idata.posterior["alpha"].values.reshape(-1)  # (chains*draws,)
    alpha_prob  = _inv_logit(alpha_draws)
    alpha_summ  = _summ_from_samples(alpha_prob)

    # Transform subject-level alphas (if present)
    subj_vars = [v for v in idata.posterior.data_vars if v.startswith("alpha_subj.")]
    subj_summ_rows = {}
    subj_prob_matrix = []  # will become shape (n_draws, n_subj) for SD on prob-scale

    if subj_vars:
        # build matrix: columns = subjects, rows = draws (all chains collapsed)
        for v in sorted(subj_vars, key=lambda x: int(x.split("alpha_subj.")[-1])):
            arr = idata.posterior[v].values.reshape(-1)
            arr_prob = _inv_logit(arr)
            subj_prob_matrix.append(arr_prob)
            subj_summ_rows[v] = _summ_from_samples(arr_prob)

        subj_prob_matrix = np.vstack(subj_prob_matrix).T  # (draws, subj)
        # group SD on probability scale, computed correctly across subjects per draw
        sd_draws = np.std(subj_prob_matrix, axis=1, ddof=1)
        alpha_std_summ = _summ_from_samples(sd_draws)
    else:
        alpha_std_summ = None

    # transformed copy of the ArviZ summary --> replace alpha rows 
    summary_t = summary.copy()

    if "alpha" in summary_t.index:
        for k, v in alpha_summ.items():
            summary_t.loc["alpha", k] = v

    # Replace alpha_std row (if present & we could compute it) #
    if ("alpha_std" in summary_t.index) and (alpha_std_summ is not None):
        for k, v in alpha_std_summ.items():
            summary_t.loc["alpha_std", k] = v

    # Replace subject rows
    for v, stats in subj_summ_rows.items():
        if v in summary_t.index:
            for k, val in stats.items():
                summary_t.loc[v, k] = val

    out_csv = diag_dir / "results_alpha_transformed.csv"
    summary_t.to_csv(out_csv)

    # write per-subject means (prob. scale) for convenience
    if subj_vars:
        means = []
        for v in sorted(subj_vars, key=lambda x: int(x.split("alpha_subj.")[-1])):
            arr = idata.posterior[v].values.reshape(-1)
            arr_prob = _inv_logit(arr)
            means.append({"param": v, "mean_prob": float(np.mean(arr_prob))})
        pd.DataFrame(means).to_csv(diag_dir / "params_of_interest_s_alpha_transformed.csv", index=False)

    print(f"[alpha-transform] Wrote:\n  - {out_csv}")
    if subj_vars:
        print(f"  - {diag_dir / 'params_of_interest_s_alpha_transformed.csv'}")


    #Posterior‐trace + KDE plots (one PDF each)
    var_names = ["alpha"]
    titles = ["alpha"]
    # Trace
    az.plot_trace(idata, var_names=var_names)
    plt.tight_layout()
    plt.savefig(diag_dir / "trace_plots.pdf")
    plt.close()
    # Posterior KDEs
    matplotlib.rcParams.update({"font.size": 6})
    fig, axes = plt.subplots(1, len(var_names), figsize=(len(var_names) * 2, 4))
    axes_flat = np.atleast_1d(axes).flatten()
    
    for i, p in enumerate(var_names):
        ax = axes_flat[i]
        arr = idata.posterior[p].values.reshape(-1)
        if p == "alpha":
            arr = np.exp(arr) / (1 + np.exp(arr))
        sns.kdeplot(y=arr, fill=True, ax=ax)

        ax.set_title(p)
        ax.set_xlim(0, 15)      # x-axis from 0 to 15
        ax.set_ylim(0, 0.5)     # y-axis (density) from 0 to 0.5
        ax.set_ylabel("Density")
        ax.set_xlabel("Value")

    
    plt.tight_layout()
    plt.savefig(diag_dir / "posteriors.pdf", bbox_inches="tight")
    plt.close(fig)
        # Per-subject parameter CSV
        # Per-subject parameter CSV
    subj_vars = [v for v in idata.posterior.data_vars if v.startswith("alpha_subj.")]
    if subj_vars:
        subj_means = {}
        for var in subj_vars:
            subj = int(var.split("alpha_subj.")[-1])
            arr  = idata.posterior[var].values  
            subj_means[subj] = arr.reshape(-1).mean()
        df = pd.DataFrame.from_dict(
            subj_means, orient="index", columns=["alpha_subj"]
        )
        df.index.name = "subj_idx"
        df.reset_index(inplace=True)
        df.to_csv(diag_dir/"params_of_interest_s.csv", index=False)
    else:
        print("ERROR")
    

model_dir = BASE_MODEL_DIR
ensure_dir(model_dir)


#------------------------------------------------------------------------------------------------------------
# functions for trial by trial drift 

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









#ingle model running version - use for manual
# this calls our ddm functions depending on whether we run or load models
if run:
    if phase == 'decision' or phase == 'dec':
        print(f'Running DDM... {model_base_name + model_name}')
        models = drift_diffusion_hddm(
            data=data,
            samples=nr_samples,
            n_jobs=nr_models,
            run=run,
            parallel=parallel,
            model_name=model_base_name + model_name,
            model_dir=model_dir,
            version=version,
            phase=phase, 
            accuracy_coding=True
        )
else:
    if phase == 'decision' or phase == 'dec':
        print(f'loading DDM... {model_base_name + model_name}')
        models = drift_diffusion_hddm(
            data=data,
            samples=nr_samples,
            n_jobs=nr_models,
            run=run,
            parallel=parallel,
            model_name=model_base_name + model_name,
            model_dir=model_dir,
            version=version,
            phase=phase,  
            accuracy_coding=True
        )
        analyze_model(models, fig_dir, nr_models, version, phase)
        
        if version == 1:
            sv_contribute = v_sv_pain_para_contributions(models, data)
            sv_contribute.to_csv(os.path.join(fig_dir, 'diagnostics', 'data_with_v_sv_pain_para_contrib.csv'), index=False)
        elif version == 2:
            sv_contribute = a_sv_pain_para_contributions(models, data)
            sv_contribute.to_csv(os.path.join(fig_dir, 'diagnostics', 'data_with_full_sv_pain_para_contrib.csv'), index=False)
        elif version == 3:
            sv_contribute = t_sv_pain_para_contributions(models, data)
            sv_contribute.to_csv(os.path.join(fig_dir, 'diagnostics', 'data_with_v_sv_money_contrib.csv' ))
        elif version == 4:
            sv_contribute = z_sv_pain_para_contributions(models, data)
            sv_contribute.to_csv(os.path.join(fig_dir, 'diagnostics', 'data_with_sv_pain_para_Abs_contrib.csv' ))

        # diag_dir = Path(fig_dir) / "diagnostics"
        # plot_inatt_forest(
        #     fig_dir=fig_dir,
        #     model_dir=model_dir,
        #     model_base=model_base_name + model_name,
        #     param_E="v_ES_InattentionW_E_subj",
        #     param_S="v_ES_InattentionW_S_subj"
        # )
        
    else:
        print(f'Running HDDMRL... {model_base_name + model_name}')
        models = drift_diffusion_hddmRL(
            data=data,
            samples=nr_samples,
            n_jobs=nr_models,
            run=run,
            parallel=parallel,
            model_name=model_base_name + model_name,
            model_dir=model_dir,
            version=version,
            phase=phase, 
        )
        analyze_model(models, fig_dir, nr_models, version, phase)
    





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


