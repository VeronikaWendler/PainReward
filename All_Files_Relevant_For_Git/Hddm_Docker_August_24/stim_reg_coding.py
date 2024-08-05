## Running the hddm models
# Pipeline for running hddm
# stim reg coding data
#
# orientated to code structure from Jan Willem de Gee on 2011-02-16. Copyright (c) 2011 __MyCompanyName__. All rights reserved.
#
# TO DO: More models
#------------------------------------------------------------------------------------------------------------------
# Importing the libraries

import pandas as pd
import numpy as np
import hddm
import os, sys, pickle, time
import datetime
import math
import scipy as sp
import matplotlib
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
# warning settings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
# Plotting
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
# Stats functionality
from statsmodels.distributions.empirical_distribution import ECDF
# HDDM
from hddm.simulators.hddm_dataset_generators import simulator_h_c

# import own libraries
current_directory = os.getcwd()
hddm_models_path = os.path.join(current_directory, 'DockerData', 'Hddm_models')
sys.path.append(hddm_models_path)
from helper_functions import prepare_data
#import compact_models

# in this code we have 2 approaches:
# 1. The first approach will include defining custom link functions for specific hddm parameters in the hddm regression model
# 2. The second approach will use stimulus coding where link functions are used in regression models

#------------------------------------------------------------------------------------------------------------------
# Structure of saving:

# /D:/Aberdeen_Uni_June24/MPColl_Lab/DockerData
#     ├── data_sets/
#     │   └── behavioural_sv_cleaned.csv
#     ├── model_dir_link_reg/
#     │   ├── l
#     │   ├── l1_1
#     │   ├── l1_2
#     ├── figures_link_reg/
#     │   └── painreward_behavioural_data_combined_new_s1b/
#     │       ├── diagnostics/
#     │       │   ├── gelman_rubic.txt
#     │       │   ├── DIC.txt
#     │       │   ├── results.csv
#     │       │   └── posteriors.pdf
#     ├── other_script.py
#------------------------------------------------------------------------------------------------------------------

# params:
version = 13 # 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15, 
run = True

# standard params:
model_base_name = 'painreward_behavioural_data_combined_stimreg_'
model_names = ['l0', 'l1', 'l2', 'l3', 'l4', 'l5', 'l6', 'l7', 'l8', 'l9', 'l10', 'l11', 'l12', 'l13', 'l14', 'l15', 'l16', 'l17', 'l18']
nr_samples = 500
nr_models = 3
parallel = True
accuracy_coding = False

# settings:
# model_name
model_name = model_names[version]

# data:
hddm_models_path = os.path.join(current_directory,'Hddm_models')
sys.path.append(hddm_models_path)
data_path1 = os.path.join(current_directory, 'data_sets', 'behavioural_sv_cleaned.csv')
data = pd.read_csv(data_path1)
data.dropna(subset=["painlevel", "moneylevel", "accepted", 'fixduration','sv_money', 'sv_pain', 'sv_both', 'p_pain_all', 'acceptance_pair', 'OV_Money_Pain', 'Abs_Money_Pain', 'OV_value', 'Abs_value'], inplace = True)

# drop all rows where money == pain for analysis
data = data[data['acceptance_pair'] != 'I']

# making sure the directory exists
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# model dir:
model_dir = 'model_dir_link_reg/'
ensure_dir(model_dir)

# figures dir:
fig_dir = os.path.join('figures_link_reg', model_base_name + model_name)
try:
    os.system('mkdir {}'.format(fig_dir))
    os.system('mkdir {}'.format(os.path.join(fig_dir, 'diagnostics')))
except:
    pass

# subjects:
subjects = np.unique(data.subj_idx)
nr_subjects = subjects.shape[0]

# turing below into categorical types for regression
data['Abs_Money_Pain'] = data['Abs_Money_Pain'].astype("category")
data['OV_Money_Pain'] = data['OV_Money_Pain'].astype("category")
data['Abs_value'] = data['Abs_value'].astype("category")
data['OV_value'] = data['OV_value'].astype("category")
data['acceptance_pair'] = data['acceptance_pair'].astype("category")
    
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

#------------------------------------------------------------------------------------------------------------------
# drift diffusion models
# sv_money: Subjective value of money.
# sv_pain: Subjective value of pain.
# sv_both: Combined subjective value of both pain and money. Calculated as sv_pain - sv_money.
# p_pain_all: Represents the probability of choosing pain. Calculated using the softmax function.
#------------------------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------------------------
# transforming link functions overall value

# function that applies stimulus coding for z
# what do we want stimulus to be?
# I want to figure out if during the decision process we can distinguish between different levels of pain and money,
# e.g. pain higher than money, or money higher than pain which also vary in difficulty when it comes to
# overall value, that is, when the overall value is rather high or rather low 
# in order to model this I will stimulus code z (bias) so that it differentiates between money>pain - and pain>money conditions irrespective of overall value level,
# I am hypothesizing a split direction in bias and more response caution regarding this differentiation as people will be more biased towards choosing trials where the money level is higher than the pain level
# therefore 'stimulus', z ~ (money>pain), -z (pain>money)  
# the 'condition' will be: high vs low overall value


#------------------------------------------------------------------------------------------------------------------
# response coding of z and v (this means z and v depend on accept/reject in a very constrained way)
#------------------------------------------------------------------------------------------------------------------
def z_link_func_r(x, data=data):
    stim = (np.asarray(dmatrix('0 + C(s, [[0], [1]])',
                              {'s': data.accepted.loc[x.index]})))
    # Apply z = (1 - x) to flip them along 0.5
    z_flip = stim - x
    # The above inverts those values we do not want to flip,
    # so invert them back
    z_flip[stim == 0] *= -1
    return z_flip

# some piecewise linear relationship: response coding of z, 'accept', z  (accept), z (reject)
def z_link_func_piece_r(x, data=data):
    accepted = data.accepted.loc[x.index]
    z_new = x.copy()
    z_new[accepted==0] = z_new[accepted==0]*0.5  # moving towards the lower boundary (multiplied by z/half which is the baseline condition, meaning 'no bias')
    z_new[accepted==1] = z_new[accepted==1]*0.5 + 0.5 # moving towards the upper decision boundary (1) 

# Quadratic relationship: response coding of v.
# If the response is 'accepted', v ~ (accept)^2, if the response is 'rejected', v ~ (reject)^2 - bounded (0,1) as z cannot logically extend beyond these bounds
def z_link_func_quadratic_r(x, data=data):
    # Create design matrix for 'accepted' condition
    stim = np.asarray(dmatrix('0 + C(s)', {'s': data.accepted.iloc[x.index]}))
    return np.clip((x * stim) ** 2, 0, 1)

def v_link_func_r( x, data = data ):
        stim = (dmatrix( "0 + C(s, [[1], [-1]])",{ 's': data.accepted.loc[x.index] },return_type = 'dataframe' ))
        return np.multiply(x.to_frame(), stim)

def v_link_func_quadratic_r(x, data=data):
    stim = np.asarray(dmatrix('0 + C(s)', {'s': data.accepted.iloc[x.index]}))
    return (x * stim) ** 2

def v_link_func_exponential_r(x, data=data):
    stim = np.asarray(dmatrix('0 + C(s)', {'s': data.accepted.iloc[x.index]}))
    return np.exp(x * stim)

#------------------------------------------------------------------------------------------------------------------
# custom-made link functions for regression models to transform z and v, depending on acceptance_pair
#------------------------------------------------------------------------------------------------------------------

# linear transformation that flips the sign
def z_link_func_OV(x, data=data):
    stim = (np.asarray(dmatrix('0 + C(s, [[0], [1]])',
                              {'s': data.OV_value.loc[x.index]})))
    # Apply z = (1 - x) to flip them along 0.5
    z_flip = stim - x
    # The above inverts those values we do not want to flip,
    # so invert them back
    z_flip[stim == 0] *= -1
    return z_flip

# some piecewise linear relationship: response coding of z, 'accept', z  (accept), z (reject)
def z_link_func_piece_OV(x, data=data):
    accepted = data.OV_value.loc[x.index]
    z_new = x.copy()
    z_new[accepted==0] = z_new[accepted==0]*0.5  # moving towards the lower boundary (multiplied by z/half which is the baseline condition, meaning 'no bias')
    z_new[accepted==1] = z_new[accepted==1]*0.5 + 0.5 # moving towards the upper decision boundary (1) 

# Quadratic relationship: response coding of v.
# If the response is 'accepted', v ~ (accept)^2, if the response is 'rejected', v ~ (reject)^2 - bounded (0,1) as z cannot logically extend beyond these bounds
def z_link_func_quadratic_OV(x, data=data):
    # Create design matrix for 'accepted' condition
    stim = np.asarray(dmatrix('0 + C(s)', {'s': data.OV_value.iloc[x.index]}))
    return np.clip((x * stim) ** 2, 0, 1)

def v_link_func_OV( x, data = data ):
        stim = ( dmatrix( "0 + C(s, [[1], [-1]])",{ 's': data.OV_value.loc[x.index] },return_type = 'dataframe' ))
        return np.multiply(x.to_frame(), stim)

def v_link_func_quadratic_OV(x, data=data):
    stim = np.asarray(dmatrix('0 + C(s)', {'s': data.OV_value.iloc[x.index]}))
    return (x * stim) ** 2

def v_link_func_exponential_OV(x, data=data):
    stim = np.asarray(dmatrix('0 + C(s)', {'s': data.OV_value.iloc[x.index]}))
    return np.exp(x * stim)

#------------------------------------------------------------------------------------------------------------------
# running the regression models with link functions
#------------------------------------------------------------------------------------------------------------------
def run_model_reg_link(trace_id, selected_data, model_dir, model_name, version, samples=1000, accuracy_coding=False):
    import os
    import numpy as np
    import hddm
    from patsy import dmatrix  

    ensure_dir(model_dir)
#------------------------------------------------------------------------------------------------------------------
# 1. RESPONSE CODED: acceptance pair variations (money > pain , pain > money) BUT we drop identical, we are only interested in low and high
#------------------------------------------------------------------------------------------------------------------
    if version == 0:      
        v_reg = {'model': 'v ~ 0 + sv_pain:C(acceptance_pair)','link_func': lambda x: x}
        z_reg = {'model': 'z ~ 0 + sv_pain:C(acceptance_pair)','link_func': lambda x: x}
        reg_descr = [v_reg, z_reg]
    elif version == 1:    
        v_reg = {'model': 'v ~ 0 + sv_pain:C(acceptance_pair)','link_func': lambda x: v_link_func_r}
        z_reg = {'model': 'z ~ 0 + sv_pain:C(acceptance_pair)','link_func': lambda x: z_link_func_r}
        reg_descr = [v_reg, z_reg]
    elif version == 2:      
        v_reg = {'model': 'v ~ 0 + sv_pain:C(acceptance_pair)','link_func': lambda x: v_link_func_r}
        z_reg = {'model': 'z ~ 0 + sv_pain:C(acceptance_pair)','link_func': lambda x: z_link_func_piece_r}
        reg_descr = [v_reg, z_reg]
    elif version == 3:    
        v_reg = {'model': 'v ~ 0 + sv_pain:C(acceptance_pair)','link_func': lambda x: v_link_func_quadratic_r}
        z_reg = {'model': 'z ~ 0 + sv_pain:C(acceptance_pair)','link_func': lambda x: z_link_func_r}
        reg_descr = [v_reg, z_reg]
    elif version == 4:    
        v_reg = {'model': 'v ~ 0 + sv_pain:C(acceptance_pair)','link_func': lambda x: v_link_func_quadratic_r}
        z_reg = {'model': 'z ~ 0 + sv_pain:C(acceptance_pair)','link_func': lambda x: z_link_func_quadratic_r}
        reg_descr = [v_reg, z_reg]
    elif version == 5:    
        v_reg = {'model': 'v ~ 0 + sv_pain:C(acceptance_pair)','link_func': lambda x: v_link_func_exponential_r}
        z_reg = {'model': 'z ~ 0 + sv_pain:C(acceptance_pair)','link_func': lambda x: z_link_func_r}
        reg_descr = [v_reg, z_reg]
    elif version == 6:    
        v_reg = {'model': 'v ~ 0 + sv_pain:C(acceptance_pair)','link_func': lambda x: v_link_func_exponential_r}
        z_reg = {'model': 'z ~ 0 + sv_pain:C(acceptance_pair)','link_func': lambda x: z_link_func_r}
        a_reg = {'model': 'a ~ 0 + sv_pain:C(acceptance_pair)','link_func': lambda x: x}
        t_reg = {'model': 't ~ 0 + sv_pain:C(acceptance_pair)','link_func': lambda x: x}
        reg_descr = [v_reg, z_reg, a_reg, t_reg]
    elif version == 7:    
        v_reg = {'model': 'v ~ 0 + sv_pain:C(acceptance_pair):C(OV_value)', 'link_func': lambda x: v_link_func_exponential_r}
        z_reg = {'model': 'z ~ 0 + sv_pain:C(acceptance_pair):C(OV_value)', 'link_func': lambda x: z_link_func_r}
        a_reg = {'model': 'a ~ 0 + sv_pain:C(acceptance_pair):C(OV_value)', 'link_func':lambda x: x}
        t_reg = {'model': 't ~ 0 + sv_pain:C(acceptance_pair):C(OV_value)', 'link_func': lambda x: x}
        reg_descr = [v_reg, z_reg, a_reg, t_reg]
        
#------------------------------------------------------------------------------------------------------------------
# 1. Stimulus coding: 
#------------------------------------------------------------------------------------------------------------------    
        
    elif  version == 8:      
        v_reg = {'model': 'v ~ 0 + sv_pain:C(acceptance_pair)','link_func': lambda x: x}
        z_reg = {'model': 'z ~ 0 + sv_pain:C(acceptance_pair)','link_func': lambda x: x}
        reg_descr = [v_reg, z_reg]
    elif version == 9:    
        v_reg = {'model': 'v ~ 0 + sv_pain:C(acceptance_pair)','link_func': lambda x: v_link_func_OV}
        z_reg = {'model': 'z ~ 0 + sv_pain:C(acceptance_pair)','link_func': lambda x: z_link_func_OV}
        reg_descr = [v_reg, z_reg]
    elif version == 10:      
        v_reg = {'model': 'v ~ 0 + sv_pain:C(acceptance_pair)','link_func': lambda x: v_link_func_OV}
        z_reg = {'model': 'z ~ 0 + sv_pain:C(acceptance_pair)','link_func': lambda x: z_link_func_piece_OV}
        reg_descr = [v_reg, z_reg]
    elif version == 11:    
        v_reg = {'model': 'v ~ 0 + sv_pain:C(acceptance_pair)','link_func': lambda x: v_link_func_quadratic_OV}
        z_reg = {'model': 'z ~ 0 + sv_pain:C(acceptance_pair)','link_func': lambda x: z_link_func_OV}
        reg_descr = [v_reg, z_reg]
    elif version == 12:    
        v_reg = {'model': 'v ~ 0 + sv_pain:C(acceptance_pair)','link_func': lambda x: v_link_func_quadratic_OV}
        z_reg = {'model': 'z ~ 0 + sv_pain:C(acceptance_pair)','link_func': lambda x: z_link_func_quadratic_OV}
        reg_descr = [v_reg, z_reg]
    elif version == 13:    
        v_reg = {'model': 'v ~ 0 + sv_pain:C(acceptance_pair)','link_func': lambda x: v_link_func_exponential_OV}
        z_reg = {'model': 'z ~ 0 + sv_pain:C(acceptance_pair)','link_func': lambda x: z_link_func_OV}
        reg_descr = [v_reg, z_reg]
    elif version == 14:    
        v_reg = {'model': 'v ~ 0 + sv_pain:C(acceptance_pair)','link_func': lambda x: v_link_func_exponential_OV}
        z_reg = {'model': 'z ~ 0 + sv_pain:C(acceptance_pair)','link_func': lambda x: z_link_func_OV}
        a_reg = {'model': 'a ~ 0 + sv_pain:C(acceptance_pair)','link_func': lambda x: x}
        t_reg = {'model': 't ~ 0 + sv_pain:C(acceptance_pair)','link_func': lambda x: x}
        reg_descr = [v_reg, z_reg, a_reg, t_reg]
    elif version == 15:    
        v_reg = {'model': 'v ~ 0 + sv_pain:C(acceptance_pair):C(Abs_value)', 'link_func': lambda x: v_link_func_exponential_OV}
        z_reg = {'model': 'z ~ 0 + sv_pain:C(acceptance_pair):C(Abs_value)', 'link_func': lambda x: z_link_func_OV}
        a_reg = {'model': 'a ~ 0 + sv_pain:C(acceptance_pair):C(Abs_value)', 'link_func':lambda x: x}
        t_reg = {'model': 't ~ 0 + sv_pain:C(acceptance_pair):C(Abs_value)', 'link_func': lambda x: x}
        reg_descr = [v_reg, z_reg, a_reg, t_reg]   
#----------------------------------------------------------------------------------------------------------------------------------------------        
#------------------------------------------------------------------------------------------------------------------
    m = hddm.models.HDDMRegressor(data, 
                                  reg_descr, 
                                  p_outlier=.05, 
                                  bias=True, 
                                  include=['a','z','v','t','st', 'sz', 'sv'],
                                  group_only_regressors=False,
                                  keep_regressor_trace=True)
    m.find_starting_values()
    m.sample(samples, burn=samples/2, dbname=os.path.join(model_dir, model_name + '_db{}'.format(trace_id)), db='pickle')
    return m



# Main function for running/loading models
def drift_diffusion_hddm(data, samples=1000, n_jobs=3, run=True, parallel=True, model_name='model', model_dir='.', version=13, accuracy_coding=False):
    if run:
        if parallel:
            start_time = time.time()
            
            results = Parallel(n_jobs=n_jobs)(
                delayed(run_model)(trace_id, data, model_dir, model_name, version, samples, accuracy_coding) 
                for trace_id in range(n_jobs)
            )
            
            print("Time elapsed:", time.time() - start_time, "s")
            
            for i in range(n_jobs):
                model = results[i]
                model.save(os.path.join(model_dir, f"{model_name}_{i}"))
        else:
            model = run_model(1, data, model_dir, model_name, version, samples, accuracy_coding)
            model.save(os.path.join(model_dir, model_name))
    else:
        print('Loading existing model(s)')
        models = [hddm.load(os.path.join(model_dir, f"{model_name}_{i}")) for i in range(n_jobs)]

    return models

def analyze_model(models, fig_dir, nr_models, version):
    sns.set_theme(style='darkgrid', font='sans-serif', font_scale=1)

    if version in [0, 1, 2]:
        params_of_interest = ['v_Intercept', 'v_C(OV_value)[T.low_OV]', 'v_C(OV_value)[T.high_OV]', 'v_sv_pain', 'v_sv_pain:C(OV_value)[T.low_OV]', 'v_sv_pain:C(OV_value)[T.high_OV]']
        params_of_interest_s = ['v_Intercept_subj', 'v_C(OV_value)[T.low_OV]_subj', 'v_C(OV_value)[T.high_OV]_subj', 'v_sv_pain_subj', 'v_sv_pain:C(OV_value)[T.low_OV]_subj', 'v_sv_pain:C(OV_value)[T.high_OV]_subj']
        titles = ['Intercept Drift Rate', 'Drift Rate (Low OV)', 'Drift Rate (High OV)', 'Drift Rate sv_pain', 'Drift Rate Interaction (sv_pain * Low OV)', 'Drift Rate Interaction (sv_pain * High OV)']
    elif version in [3, 4, 5, 6, 7, 8, 9]:
        params_of_interest = [
            'v_Intercept', 'v_C(OV_value)[T.low_OV]', 'v_C(OV_value)[T.high_OV]', 'v_sv_pain', 'v_sv_pain:C(OV_value)[T.low_OV]', 'v_sv_pain:C(OV_value)[T.high_OV]',
            'z_Intercept', 'z_C(OV_value)[T.low_OV]', 'z_C(OV_value)[T.high_OV]', 'z_sv_pain', 'z_sv_pain:C(OV_value)[T.low_OV]', 'z_sv_pain:C(OV_value)[T.high_OV]',
            'a_Intercept', 'a_C(OV_value)[T.low_OV]', 'a_C(OV_value)[T.high_OV]', 'a_sv_pain', 'a_sv_pain:C(OV_value)[T.low_OV]', 'a_sv_pain:C(OV_value)[T.high_OV]',
            't_Intercept', 't_C(OV_value)[T.low_OV]', 't_C(OV_value)[T.high_OV]', 't_sv_pain', 't_sv_pain:C(OV_value)[T.low_OV]', 't_sv_pain:C(OV_value)[T.high_OV]'
        ]
        params_of_interest_s = [
            'v_Intercept_subj', 'v_C(OV_value)[T.low_OV]_subj', 'v_C(OV_value)[T.high_OV]_subj', 'v_sv_pain_subj', 'v_sv_pain:C(OV_value)[T.low_OV]_subj', 'v_sv_pain:C(OV_value)[T.high_OV]_subj',
            'z_Intercept_subj', 'z_C(OV_value)[T.low_OV]_subj', 'z_C(OV_value)[T.high_OV]_subj', 'z_sv_pain_subj', 'z_sv_pain:C(OV_value)[T.low_OV]_subj', 'z_sv_pain:C(OV_value)[T.high_OV]_subj',
            'a_Intercept_subj', 'a_C(OV_value)[T.low_OV]_subj', 'a_C(OV_value)[T.high_OV]_subj', 'a_sv_pain_subj', 'a_sv_pain:C(OV_value)[T.low_OV]_subj', 'a_sv_pain:C(OV_value)[T.high_OV]_subj',
            't_Intercept_subj', 't_C(OV_value)[T.low_OV]_subj', 't_C(OV_value)[T.high_OV]_subj', 't_sv_pain_subj', 't_sv_pain:C(OV_value)[T.low_OV]_subj', 't_sv_pain:C(OV_value)[T.high_OV]_subj'
        ]
        titles = [
            'Intercept Drift Rate', 'Drift Rate (Low OV)', 'Drift Rate (High OV)', 'Drift Rate sv_pain', 'Drift Rate Interaction (sv_pain * Low OV)', 'Drift Rate Interaction (sv_pain * High OV)',
            'Intercept Starting Point', 'Starting Point (Low OV)', 'Starting Point (High OV)', 'Starting Point sv_pain', 'Starting Point Interaction (sv_pain * Low OV)', 'Starting Point Interaction (sv_pain * High OV)',
            'Intercept Boundary Separation', 'Boundary Separation (Low OV)', 'Boundary Separation (High OV)', 'Boundary Separation sv_pain', 'Boundary Separation Interaction (sv_pain * Low OV)', 'Boundary Separation Interaction (sv_pain * High OV)',
            'Intercept Non-decision Time', 'Non-decision Time (Low OV)', 'Non-decision Time (High OV)', 'Non-decision Time sv_pain', 'Non-decision Time Interaction (sv_pain * Low OV)', 'Non-decision Time Interaction (sv_pain * High OV)'
        ]
    elif version in [10, 11, 12, 13, 14, 15, 16]:
        params_of_interest = [
            'v_Intercept', 'v_C(Abs_value)[T.low_abs]', 'v_C(Abs_value)[T.mid_abs]', 'v_C(Abs_value)[T.high_abs]', 'v_sv_pain', 'v_sv_pain:C(Abs_value)[T.low_abs]', 'v_sv_pain:C(Abs_value)[T.mid_abs]', 'v_sv_pain:C(Abs_value)[T.high_abs]',
            'z_Intercept', 'z_C(Abs_value)[T.low_abs]', 'z_C(Abs_value)[T.mid_abs]', 'z_C(Abs_value)[T.high_abs]', 'z_sv_pain', 'z_sv_pain:C(Abs_value)[T.low_abs]', 'z_sv_pain:C(Abs_value)[T.mid_abs]', 'z_sv_pain:C(Abs_value)[T.high_abs]',
            'a_Intercept', 'a_C(Abs_value)[T.low_abs]', 'a_C(Abs_value)[T.mid_abs]', 'a_C(Abs_value)[T.high_abs]', 'a_sv_pain', 'a_sv_pain:C(Abs_value)[T.low_abs]', 'a_sv_pain:C(Abs_value)[T.mid_abs]', 'a_sv_pain:C(Abs_value)[T.high_abs]',
            't_Intercept', 't_C(Abs_value)[T.low_abs]', 't_C(Abs_value)[T.mid_abs]', 't_C(Abs_value)[T.high_abs]', 't_sv_pain', 't_sv_pain:C(Abs_value)[T.low_abs]', 't_sv_pain:C(Abs_value)[T.mid_abs]', 't_sv_pain:C(Abs_value)[T.high_abs]'
        ]
        params_of_interest_s = [
            'v_Intercept_subj', 'v_C(Abs_value)[T.low_abs]_subj', 'v_C(Abs_value)[T.mid_abs]_subj', 'v_C(Abs_value)[T.high_abs]_subj', 'v_sv_pain_subj', 'v_sv_pain:C(Abs_value)[T.low_abs]_subj', 'v_sv_pain:C(Abs_value)[T.mid_abs]_subj', 'v_sv_pain:C(Abs_value)[T.high_abs]_subj',
            'z_Intercept_subj', 'z_C(Abs_value)[T.low_abs]_subj', 'z_C(Abs_value)[T.mid_abs]_subj', 'z_C(Abs_value)[T.high_abs]_subj', 'z_sv_pain_subj', 'z_sv_pain:C(Abs_value)[T.low_abs]_subj', 'z_sv_pain:C(Abs_value)[T.mid_abs]_subj', 'z_sv_pain:C(Abs_value)[T.high_abs]_subj',
            'a_Intercept_subj', 'a_C(Abs_value)[T.low_abs]_subj', 'a_C(Abs_value)[T.mid_abs]_subj', 'a_C(Abs_value)[T.high_abs]_subj', 'a_sv_pain_subj', 'a_sv_pain:C(Abs_value)[T.low_abs]_subj', 'a_sv_pain:C(Abs_value)[T.mid_abs]_subj', 'a_sv_pain:C(Abs_value)[T.high_abs]_subj',
            't_Intercept_subj', 't_C(Abs_value)[T.low_abs]_subj', 't_C(Abs_value)[T.mid_abs]_subj', 't_C(Abs_value)[T.high_abs]_subj', 't_sv_pain_subj', 't_sv_pain:C(Abs_value)[T.low_abs]_subj', 't_sv_pain:C(Abs_value)[T.mid_abs]_subj', 't_sv_pain:C(Abs_value)[T.high_abs]_subj'
        ]
        titles = [
            'Intercept Drift Rate', 'Drift Rate (Low Abs)', 'Drift Rate (Mid Abs)', 'Drift Rate (High Abs)', 'Drift Rate sv_pain', 'Drift Rate Interaction (sv_pain * Low Abs)', 'Drift Rate Interaction (sv_pain * Mid Abs)', 'Drift Rate Interaction (sv_pain * High Abs)',
            'Intercept Starting Point', 'Starting Point (Low Abs)', 'Starting Point (Mid Abs)', 'Starting Point (High Abs)', 'Starting Point sv_pain', 'Starting Point Interaction (sv_pain * Low Abs)', 'Starting Point Interaction (sv_pain * Mid Abs)', 'Starting Point Interaction (sv_pain * High Abs)',
            'Intercept Boundary Separation', 'Boundary Separation (Low Abs)', 'Boundary Separation (Mid Abs)', 'Boundary Separation (High Abs)', 'Boundary Separation sv_pain', 'Boundary Separation Interaction (sv_pain * Low Abs)', 'Boundary Separation Interaction (sv_pain * Mid Abs)', 'Boundary Separation Interaction (sv_pain * High Abs)',
            'Intercept Non-decision Time', 'Non-decision Time (Low Abs)', 'Non-decision Time (Mid Abs)', 'Non-decision Time (High Abs)', 'Non-decision Time sv_pain', 'Non-decision Time Interaction (sv_pain * Low Abs)', 'Non-decision Time Interaction (sv_pain * Mid Abs)', 'Non-decision Time Interaction (sv_pain * High Abs)'
        ]
        elif version in [17, 18, 19, 20, 21, 22, 23]:
        params_of_interest = [
            'v_Intercept', 'v_C(Abs_value)[T.low_abs]', 'v_C(Abs_value)[T.mid_abs]', 'v_C(Abs_value)[T.high_abs]', 'v_C(OV_value)[T.low_OV]', 'v_C(OV_value)[T.high_OV]', 'v_sv_pain', 
            'v_sv_pain:C(Abs_value)[T.low_abs]', 'v_sv_pain:C(Abs_value)[T.mid_abs]', 'v_sv_pain:C(Abs_value)[T.high_abs]', 'v_sv_pain:C(OV_value)[T.low_OV]', 'v_sv_pain:C(OV_value)[T.high_OV]', 
            'v_sv_pain:C(Abs_value)[T.low_abs]:C(OV_value)[T.low_OV]', 'v_sv_pain:C(Abs_value)[T.low_abs]:C(OV_value)[T.high_OV]', 'v_sv_pain:C(Abs_value)[T.mid_abs]:C(OV_value)[T.low_OV]', 
            'v_sv_pain:C(Abs_value)[T.mid_abs]:C(OV_value)[T.high_OV]', 'v_sv_pain:C(Abs_value)[T.high_abs]:C(OV_value)[T.low_OV]', 'v_sv_pain:C(Abs_value)[T.high_abs]:C(OV_value)[T.high_OV]',
            'z_Intercept', 'z_C(Abs_value)[T.low_abs]', 'z_C(Abs_value)[T.mid_abs]', 'z_C(Abs_value)[T.high_abs]', 'z_C(OV_value)[T.low_OV]', 'z_C(OV_value)[T.high_OV]', 'z_sv_pain',
            'z_sv_pain:C(Abs_value)[T.low_abs]', 'z_sv_pain:C(Abs_value)[T.mid_abs]', 'z_sv_pain:C(Abs_value)[T.high_abs]', 'z_sv_pain:C(OV_value)[T.low_OV]', 'z_sv_pain:C(OV_value)[T.high_OV]',
            'z_sv_pain:C(Abs_value)[T.low_abs]:C(OV_value)[T.low_OV]', 'z_sv_pain:C(Abs_value)[T.low_abs]:C(OV_value)[T.high_OV]', 'z_sv_pain:C(Abs_value)[T.mid_abs]:C(OV_value)[T.low_OV]',
            'z_sv_pain:C(Abs_value)[T.mid_abs]:C(OV_value)[T.high_OV]', 'z_sv_pain:C(Abs_value)[T.high_abs]:C(OV_value)[T.low_OV]', 'z_sv_pain:C(Abs_value)[T.high_abs]:C(OV_value)[T.high_OV]',
            'a_Intercept', 'a_C(Abs_value)[T.low_abs]', 'a_C(Abs_value)[T.mid_abs]', 'a_C(Abs_value)[T.high_abs]', 'a_C(OV_value)[T.low_OV]', 'a_C(OV_value)[T.high_OV]', 'a_sv_pain',
            'a_sv_pain:C(Abs_value)[T.low_abs]', 'a_sv_pain:C(Abs_value)[T.mid_abs]', 'a_sv_pain:C(Abs_value)[T.high_abs]', 'a_sv_pain:C(OV_value)[T.low_OV]', 'a_sv_pain:C(OV_value)[T.high_OV]',
            'a_sv_pain:C(Abs_value)[T.low_abs]:C(OV_value)[T.low_OV]', 'a_sv_pain:C(Abs_value)[T.low_abs]:C(OV_value)[T.high_OV]', 'a_sv_pain:C(Abs_value)[T.mid_abs]:C(OV_value)[T.low_OV]',
            'a_sv_pain:C(Abs_value)[T.mid_abs]:C(OV_value)[T.high_OV]', 'a_sv_pain:C(Abs_value)[T.high_abs]:C(OV_value)[T.low_OV]', 'a_sv_pain:C(Abs_value)[T.high_abs]:C(OV_value)[T.high_OV]',
            't_Intercept', 't_C(Abs_value)[T.low_abs]', 't_C(Abs_value)[T.mid_abs]', 't_C(Abs_value)[T.high_abs]', 't_C(OV_value)[T.low_OV]', 't_C(OV_value)[T.high_OV]', 't_sv_pain',
            't_sv_pain:C(Abs_value)[T.low_abs]', 't_sv_pain:C(Abs_value)[T.mid_abs]', 't_sv_pain:C(Abs_value)[T.high_abs]', 't_sv_pain:C(OV_value)[T.low_OV]', 't_sv_pain:C(OV_value)[T.high_OV]',
            't_sv_pain:C(Abs_value)[T.low_abs]:C(OV_value)[T.low_OV]', 't_sv_pain:C(Abs_value)[T.low_abs]:C(OV_value)[T.high_OV]', 't_sv_pain:C(Abs_value)[T.mid_abs]:C(OV_value)[T.low_OV]',
            't_sv_pain:C(Abs_value)[T.mid_abs]:C(OV_value)[T.high_OV]', 't_sv_pain:C(Abs_value)[T.high_abs]:C(OV_value)[T.low_OV]', 't_sv_pain:C(Abs_value)[T.high_abs]:C(OV_value)[T.high_OV]'
        ]
        params_of_interest_s = [
            'v_Intercept_subj', 'v_C(Abs_value)[T.low_abs]_subj', 'v_C(Abs_value)[T.mid_abs]_subj', 'v_C(Abs_value)[T.high_abs]_subj', 'v_C(OV_value)[T.low_OV]_subj', 'v_C(OV_value)[T.high_OV]_subj', 'v_sv_pain_subj',
            'v_sv_pain:C(Abs_value)[T.low_abs]_subj', 'v_sv_pain:C(Abs_value)[T.mid_abs]_subj', 'v_sv_pain:C(Abs_value)[T.high_abs]_subj', 'v_sv_pain:C(OV_value)[T.low_OV]_subj', 'v_sv_pain:C(OV_value)[T.high_OV]_subj',
            'v_sv_pain:C(Abs_value)[T.low_abs]:C(OV_value)[T.low_OV]_subj', 'v_sv_pain:C(Abs_value)[T.low_abs]:C(OV_value)[T.high_OV]_subj', 'v_sv_pain:C(Abs_value)[T.mid_abs]:C(OV_value)[T.low_OV]_subj',
            'v_sv_pain:C(Abs_value)[T.mid_abs]:C(OV_value)[T.high_OV]_subj', 'v_sv_pain:C(Abs_value)[T.high_abs]:C(OV_value)[T.low_OV]_subj', 'v_sv_pain:C(Abs_value)[T.high_abs]:C(OV_value)[T.high_OV]_subj',
            'z_Intercept_subj', 'z_C(Abs_value)[T.low_abs]_subj', 'z_C(Abs_value)[T.mid_abs]_subj', 'z_C(Abs_value)[T.high_abs]_subj', 'z_C(OV_value)[T.low_OV]_subj', 'z_C(OV_value)[T.high_OV]_subj', 'z_sv_pain_subj',
            'z_sv_pain:C(Abs_value)[T.low_abs]_subj', 'z_sv_pain:C(Abs_value)[T.mid_abs]_subj', 'z_sv_pain:C(Abs_value)[T.high_abs]_subj', 'z_sv_pain:C(OV_value)[T.low_OV]_subj', 'z_sv_pain:C(OV_value)[T.high_OV]_subj',
            'z_sv_pain:C(Abs_value)[T.low_abs]:C(OV_value)[T.low_OV]_subj', 'z_sv_pain:C(Abs_value)[T.low_abs]:C(OV_value)[T.high_OV]_subj', 'z_sv_pain:C(Abs_value)[T.mid_abs]:C(OV_value)[T.low_OV]_subj',
            'z_sv_pain:C(Abs_value)[T.mid_abs]:C(OV_value)[T.high_OV]_subj', 'z_sv_pain:C(Abs_value)[T.high_abs]:C(OV_value)[T.low_OV]_subj', 'z_sv_pain:C(Abs_value)[T.high_abs]:C(OV_value)[T.high_OV]_subj',
            'a_Intercept_subj', 'a_C(Abs_value)[T.low_abs]_subj', 'a_C(Abs_value)[T.mid_abs]_subj', 'a_C(Abs_value)[T.high_abs]_subj', 'a_C(OV_value)[T.low_OV]_subj', 'a_C(OV_value)[T.high_OV]_subj', 'a_sv_pain_subj',
            'a_sv_pain:C(Abs_value)[T.low_abs]_subj', 'a_sv_pain:C(Abs_value)[T.mid_abs]_subj', 'a_sv_pain:C(Abs_value)[T.high_abs]_subj', 'a_sv_pain:C(OV_value)[T.low_OV]_subj', 'a_sv_pain:C(OV_value)[T.high_OV]_subj',
            'a_sv_pain:C(Abs_value)[T.low_abs]:C(OV_value)[T.low_OV]_subj', 'a_sv_pain:C(Abs_value)[T.low_abs]:C(OV_value)[T.high_OV]_subj', 'a_sv_pain:C(Abs_value)[T.mid_abs]:C(OV_value)[T.low_OV]_subj',
            'a_sv_pain:C(Abs_value)[T.mid_abs]:C(OV_value)[T.high_OV]_subj', 'a_sv_pain:C(Abs_value)[T.high_abs]:C(OV_value)[T.low_OV]_subj', 'a_sv_pain:C(Abs_value)[T.high_abs]:C(OV_value)[T.high_OV]_subj',
            't_Intercept_subj', 't_C(Abs_value)[T.low_abs]_subj', 't_C(Abs_value)[T.mid_abs]_subj', 't_C(Abs_value)[T.high_abs]_subj', 't_C(OV_value)[T.low_OV]_subj', 't_C(OV_value)[T.high_OV]_subj', 't_sv_pain_subj',
            't_sv_pain:C(Abs_value)[T.low_abs]_subj', 't_sv_pain:C(Abs_value)[T.mid_abs]_subj', 't_sv_pain:C(Abs_value)[T.high_abs]_subj', 't_sv_pain:C(OV_value)[T.low_OV]_subj', 't_sv_pain:C(OV_value)[T.high_OV]_subj',
            't_sv_pain:C(Abs_value)[T.low_abs]:C(OV_value)[T.low_OV]_subj', 't_sv_pain:C(Abs_value)[T.low_abs]:C(OV_value)[T.high_OV]_subj', 't_sv_pain:C(Abs_value)[T.mid_abs]:C(OV_value)[T.low_OV]_subj',
            't_sv_pain:C(Abs_value)[T.mid_abs]:C(OV_value)[T.high_OV]_subj', 't_sv_pain:C(Abs_value)[T.high_abs]:C(OV_value)[T.low_OV]_subj', 't_sv_pain:C(Abs_value)[T.high_abs]:C(OV_value)[T.high_OV]_subj'
        ]
        titles = [
            'Intercept Drift Rate', 'Drift Rate (Low Abs)', 'Drift Rate (Mid Abs)', 'Drift Rate (High Abs)', 'Drift Rate (Low OV)', 'Drift Rate (High OV)', 'Drift Rate sv_pain',
            'Drift Rate Interaction (sv_pain * Low Abs)', 'Drift Rate Interaction (sv_pain * Mid Abs)', 'Drift Rate Interaction (sv_pain * High Abs)', 'Drift Rate Interaction (sv_pain * Low OV)', 'Drift Rate Interaction (sv_pain * High OV)',
            'Drift Rate Interaction (sv_pain * Low Abs * Low OV)', 'Drift Rate Interaction (sv_pain * Low Abs * High OV)', 'Drift Rate Interaction (sv_pain * Mid Abs * Low OV)',
            'Drift Rate Interaction (sv_pain * Mid Abs * High OV)', 'Drift Rate Interaction (sv_pain * High Abs * Low OV)', 'Drift Rate Interaction (sv_pain * High Abs * High OV)',
            'Intercept Starting Point', 'Starting Point (Low Abs)', 'Starting Point (Mid Abs)', 'Starting Point (High Abs)', 'Starting Point (Low OV)', 'Starting Point (High OV)', 'Starting Point sv_pain',
            'Starting Point Interaction (sv_pain * Low Abs)', 'Starting Point Interaction (sv_pain * Mid Abs)', 'Starting Point Interaction (sv_pain * High Abs)', 'Starting Point Interaction (sv_pain * Low OV)', 'Starting Point Interaction (sv_pain * High OV)',
            'Starting Point Interaction (sv_pain * Low Abs * Low OV)', 'Starting Point Interaction (sv_pain * Low Abs * High OV)', 'Starting Point Interaction (sv_pain * Mid Abs * Low OV)',
            'Starting Point Interaction (sv_pain * Mid Abs * High OV)', 'Starting Point Interaction (sv_pain * High Abs * Low OV)', 'Starting Point Interaction (sv_pain * High Abs * High OV)',
            'Intercept Boundary Separation', 'Boundary Separation (Low Abs)', 'Boundary Separation (Mid Abs)', 'Boundary Separation (High Abs)', 'Boundary Separation (Low OV)', 'Boundary Separation (High OV)', 'Boundary Separation sv_pain',
            'Boundary Separation Interaction (sv_pain * Low Abs)', 'Boundary Separation Interaction (sv_pain * Mid Abs)', 'Boundary Separation Interaction (sv_pain * High Abs)', 'Boundary Separation Interaction (sv_pain * Low OV)', 'Boundary Separation Interaction (sv_pain * High OV)',
            'Boundary Separation Interaction (sv_pain * Low Abs * Low OV)', 'Boundary Separation Interaction (sv_pain * Low Abs * High OV)', 'Boundary Separation Interaction (sv_pain * Mid Abs * Low OV)',
            'Boundary Separation Interaction (sv_pain * Mid Abs * High OV)', 'Boundary Separation Interaction (sv_pain * High Abs * Low OV)', 'Boundary Separation Interaction (sv_pain * High Abs * High OV)',
            'Intercept Non-decision Time', 'Non-decision Time (Low Abs)', 'Non-decision Time (Mid Abs)', 'Non-decision Time (High Abs)', 'Non-decision Time (Low OV)', 'Non-decision Time (High OV)', 'Non-decision Time sv_pain',
            'Non-decision Time Interaction (sv_pain * Low Abs)', 'Non-decision Time Interaction (sv_pain * Mid Abs)', 'Non-decision Time Interaction (sv_pain * High Abs)', 'Non-decision Time Interaction (sv_pain * Low OV)', 'Non-decision Time Interaction (sv_pain * High OV)',
            'Non-decision Time Interaction (sv_pain * Low Abs * Low OV)', 'Non-decision Time Interaction (sv_pain * Low Abs * High OV)', 'Non-decision Time Interaction (sv_pain * Mid Abs * Low OV)',
            'Non-decision Time Interaction (sv_pain * Mid Abs * High OV)', 'Non-decision Time Interaction (sv_pain * High Abs * Low OV)', 'Non-decision Time Interaction (sv_pain * High Abs * High OV)'
        ]

    combined_model = kabuki.utils.concat_models(models)

    # Gelman-Rubin diagnostic
    gr = hddm.analyze.gelman_rubin(models)
    ensure_dir(os.path.join(fig_dir, 'diagnostics'))
    with open(os.path.join(fig_dir, 'diagnostics', 'gelman_rubin.txt'), 'w') as text_file:
        for p in gr.items():
            text_file.write(f"{p[0]}: {p[1]}\n")

    # DIC
    dic = combined_model.dic
    with open(os.path.join(fig_dir, 'diagnostics', 'DIC.txt'), 'w') as text_file:
        text_file.write(f"DIC: {dic}\n")
        
    # Plots
    model_nr = 0
    #combined_model = kabuki.utils.concat_models(models)
    size_plot = len(models[model_nr].data.subj_idx.unique()) / 3.0 * 1.5
    models[model_nr].plot_posterior_predictive(samples=10, bins=100, figsize=(6, size_plot), save=True, path=os.path.join(fig_dir, 'diagnostics'), format='pdf')
    models[model_nr].plot_posteriors(save=True, path=os.path.join(fig_dir, 'diagnostics'), format='pdf')
    
    # Point estimates and results
    results = models[model_nr].gen_stats()
    results.to_csv(os.path.join(fig_dir, 'diagnostics', 'results.csv'))
    
    # Posterior analysis and fixed starting point as in J.W. de Gee code
    traces = [models[model_nr].nodes_db.node[p].trace() for p in params_of_interest]
    traces[0] = 1 / (1 + np.exp(-(traces[0])))

    stats = []
    for trace in traces:
        stat = min(np.mean(trace > 0), np.mean(trace < 0))
        stats.append(min(stat, 1 - stat))
    stats = np.array(stats)

    n_cols = 5
    n_rows = int(np.ceil(len(params_of_interest) / n_cols))

    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(n_cols * 3, n_rows * 4))
    axes = axes.flatten()

    for ax_nr, (trace, title) in enumerate(zip(traces, titles)):
        sns.kdeplot(trace, vertical=True, shade=True, color='black', ax=axes[ax_nr])
        if ax_nr % n_cols == 0:
            axes[ax_nr].set_ylabel('Parameter estimate (a.u.)')
        if ax_nr >= len(params_of_interest) - n_cols:
            axes[ax_nr].set_xlabel('Posterior probability')
        axes[ax_nr].set_title(f'{title}\np={round(stats[ax_nr], 3)}')
        axes[ax_nr].set_xlim(xmin=0)
        for axis in ['top', 'bottom', 'left', 'right']:
            axes[ax_nr].spines[axis].set_linewidth(0.5)
            axes[ax_nr].tick_params(width=0.5)

    for ax in axes[len(params_of_interest):]:
        fig.delaxes(ax)

    sns.despine(offset=10, trim=True)
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'posteriors.pdf'))

    # Parameter extraction
    parameters = []
    for p in params_of_interest_s:
        param_values = []
        for s in np.unique(models[model_nr].data.subj_idx):
            param_name = f"{p}.{s}"
            param_value = results.loc[results.index == param_name, 'mean'].values
            if len(param_value) > 0:
                param_values.append(param_value[0])
        parameters.append(param_values)

    parameters = pd.DataFrame(parameters).T
    parameters.columns = params_of_interest_s
    parameters.to_csv(os.path.join(fig_dir, 'diagnostics', 'params_of_interest_s.csv'))

# directories
model_dir = 'model_dir/'
ensure_dir(model_dir)
fig_dir = os.path.join('figures', model_base_name + model_name)
ensure_dir(fig_dir)
ensure_dir(os.path.join(fig_dir, 'diagnostics'))

# Load models or run
if run:
    print('Running {}'.format(model_base_name + model_name))
    models = drift_diffusion_hddm(data=data, samples=nr_samples, n_jobs=nr_models, run=run, parallel=parallel, model_name=model_base_name + model_name, model_dir=model_dir, version=version, accuracy_coding=False)
else:
    models = drift_diffusion_hddm(data=data, samples=nr_samples, n_jobs=nr_models, run=run, parallel=parallel, model_name=model_base_name + model_name, model_dir=model_dir, version=version, accuracy_coding=False)
analyze_model(models, fig_dir, nr_models, version)
# %%























# #####
# # m = hddm.HDDMStimCoding(data, 
#                             stim_col='OV_Money_Pain', 
#                             split_param='v', 
#                             drift_criterion=True, 
#                             bias=True, 
#                             include=('sv',), 
#                             depends_on={'v': 'OV_Money_Pain', 't': 'sv_pain', 'a': 'sv_pain', 'z': 'OV_Money_Pain'}, 
#                             p_outlier=.05)

#     m.find_starting_values()
#     m.sample(samples, burn=samples//10, thin=3, dbname=os.path.join(model_dir, model_name + '_db{}'.format(trace_id)), db='pickle')

#     return m

# # Run the model
# trace_id = 1
# model_dir = './models'
# model_name = 'hddm_stimcoding_example'
# samples = 1000

# # Assuming 'data' is your preprocessed dataframe
# model_v23 = run_model_stim(trace_id, data, model_dir, model_name, version=23, samples=samples)

# # Extract nodes for different levels of 'OV_Money_Pain'
# v_levels = model_v23.nodes_db.node[[f'v({level})' for level in data['OV_Money_Pain'].unique()]]
# z_levels = model_v23.nodes_db.node[[f'z({level})' for level in data['OV_Money_Pain'].unique()]]
# a_levels = model_v23.nodes_db.node[[f'a({level})' for level in data['OV_Money_Pain'].unique()]]
# t_levels = model_v23.nodes_db.node[[f't({level})' for level in data['OV_Money_Pain'].unique()]]

# # Plot posterior nodes for drift rate 'v'
# hddm.analyze.plot_posterior_nodes(v_levels)
# plt.xlabel('drift-rate')
# plt.ylabel('Posterior probability')
# plt.title('Posterior of drift-rate for different OV_Money_Pain levels')
# plt.savefig('drift_rate_OV_Money_Pain_levels.pdf')
# plt.show()

# # Plot posterior nodes for starting point bias 'z'
# hddm.analyze.plot_posterior_nodes(z_levels)
# plt.xlabel('starting point bias')
# plt.ylabel('Posterior probability')
# plt.title('Posterior of starting point bias for different OV_Money_Pain levels')
# plt.savefig('starting_point_bias_OV_Money_Pain_levels.pdf')
# plt.show()
