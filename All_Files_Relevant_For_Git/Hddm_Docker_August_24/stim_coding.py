## Running the hddm models
# Pipeline for running hddm
# stim coding data
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

# in this code we will use the stimulus coding approach:

#------------------------------------------------------------------------------------------------------------------
# Structure of saving:

# /D:/Aberdeen_Uni_June24/MPColl_Lab/DockerData
#     ├── data_sets/
#     │   └── behavioural_sv_cleaned.csv
#     ├── model_dir_stim/
#     │   ├── s1_0
#     │   ├── s1_1
#     │   ├── s1_2
#     ├── figures_stim/
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
model_base_name = 'painreward_behavioural_data_combined_new_'
model_names = ['s0', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18']
nr_samples = 1000
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
data_accept = data[data['acceptance_pair'] != 'I']

# making sure the directory exists
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# model dir:
model_dir = 'model_dir_stim/'
ensure_dir(model_dir)

# figures dir:
fig_dir = os.path.join('figures_stim', model_base_name + model_name)
try:
    os.system('mkdir {}'.format(fig_dir))
    os.system('mkdir {}'.format(os.path.join(fig_dir, 'diagnostics')))
except:
    pass

# subjects:
subjects = np.unique(data.subj_idx)
nr_subjects = subjects.shape[0]


def run_model_stim(trace_id, data, model_dir, model_name, version, samples=1000, accuracy_coding=False):
    import os
    import numpy as np
    import hddm
    from patsy import dmatrix  
    ensure_dir(model_dir)
    
    if version == 0:
        stim_col = 'stimulus_mapped_a',     
        split_param='v',                    
        drift_criterion=True,               
        bias=True,                         
        include=['sv']                      
    elif version == 1:
        stim_col = 'stimulus_mapped_a',     
        split_param='z',                    
        #drift_criterion=True,               
        bias=True,                          
        include=['sz']                      
    if version == 2:
        stim_col = 'stimulus_mapped_b',     
        split_param='v',                    
        drift_criterion=True,               
        bias=True,                          
        include=['sv']                      
    elif version == 3:
        stim_col = 'stimulus_mapped_b',    
        split_param='z',                    
        #drift_criterion=True,               
        bias=True,                          
        include=['sz']                      
    if version == 4:
        stim_col = 'stimulus_mapped_a',    
        split_param='v',                
        drift_criterion=True,               
        bias=True,                          
        include=['sv'],
        depends_on_dict = {
                'v': ['stimulus_mapped_a', 'sv_pain'], 
                'z': ['stimulus_mapped_a', 'sv_pain'], 
                                        }
    if version == 5:
        stim_col = 'stimulus_mapped_a',     
        split_param='v',                    
        drift_criterion=True,               
        bias=True,                          
        include=['sv'],
        depends_on_dict = {
                'v': ['stimulus_mapped_b', 'sv_pain'], 
                'z': ['stimulus_mapped_b', 'sv_pain'], 
                                        }
    if version == 6:
        stim_col = 'stimulus_mapped_a',     
        split_param='z',                    
        drift_criterion=True,               
        bias=True,                          
        include=['sv'],
        depends_on_dict = {
                'v': ['stimulus_mapped_a', 'sv_pain'], 
                'z': ['stimulus_mapped_a', 'sv_pain'], 
                                        }
    if version == 7:
        stim_col = 'stimulus_mapped_a',     
        split_param='z',                    
        drift_criterion=True,               
        bias=True,                          
        include=['sv'],
        depends_on_dict = {
                'v': ['stimulus_mapped_b', 'sv_pain'], 
                'z': ['stimulus_mapped_b', 'sv_pain'], 
                                        }
    #------------------------------------------------------------------------------------------------------------------
    
    # Define the parameters based on version
    if version == 1:
        stim_col = 'stimulus_mapped_a',     # these are the created 'stimulus-coded' data frames
        split_param='v',                    # this model tests how v varies between high money (1) and high pain (0) 
        drift_criterion=True,               # the model estimates an intercept (dc) for v
        bias=True,                          # the model includes a bias for z
        include=['sv'],                     # this model only includes drift rate variability
        depends_on_dict = {                 # the depends on dict specifies that t,v,a,z additionally vary with the subjective value of pain  
            't': 'sv_pain', 
            'v': 'sv_pain', 
            'a': 'sv_pain', 
            'z': 'sv_pain'
        }
    elif version == 2:
        stim_col = 'stimulus_mapped_a',     # these are the created 'stimulus-coded' data frames
        split_param='v',                    # this model tests how v varies between high money (1) and high pain (0) 
        drift_criterion=True,               # the model estimates an intercept (dc) for v
        bias=True,                          # the model includes a bias for z
        include=['sv'],                     # this model only includes drift rate variability
        depends_on_dict = {                 # the depends on dict specifies that t,v,a,z additionally vary with the subjective value of pain  
            't': ['stimulus_mapped_a', 'sv_pain'], 
            'v': ['stimulus_mapped_b', 'sv_pain'], 
            'a': ['stimulus_mapped_a', 'sv_pain'], 
            'z': ['stimulus_mapped_a', 'sv_pain'], 
        }
    # Define the parameters based on version
    elif version == 3:
        stim_col = 'stimulus_mapped_a',     # these are the created 'stimulus-coded' data frames
        split_param='v',                    # this model tests how v varies between high money (1) and high pain (0) 
        drift_criterion=True,               # the model estimates an intercept (dc) for v
        bias=True,                          # the model includes a bias for z
        include=['z', 'a', 't', 'v', 'sv', 'st', 'sz'],     # this model includes variability in z,t,a,v,sv,st,sz
        depends_on_dict = {                 # the depends on dict specifies that t,v,a,z additionally vary with the subjective value of pain  
            't': 'sv_pain', 
            'v': 'sv_pain', 
            'a': 'sv_pain', 
            'z': 'sv_pain'
        }
    elif version == 4:
        stim_col = 'stimulus_mapped_a',     # these are the created 'stimulus-coded' data frames
        split_param='v',                    # this model tests how v varies between high money (1) and high pain (0) 
        drift_criterion=True,               # the model estimates an intercept (dc) for v
        bias=True,                          # the model includes a bias for z
        include=['z', 'a', 't', 'v', 'sv', 'st', 'sz'],                     # this model only includes drift rate variability
        depends_on_dict = {                 # the depends on dict specifies that t,v,a,z additionally vary with the subjective value of pain  
            't': ['stimulus_mapped_a', 'sv_pain'], 
            'v': ['stimulus_mapped_b', 'sv_pain'], 
            'a': ['stimulus_mapped_a', 'sv_pain'], 
            'z': ['stimulus_mapped_a', 'sv_pain'], 
        }
        # Define the parameters based on version
    elif version == 5:
        stim_col = 'stimulus_mapped_b',     # these are the created 'stimulus-coded' data frames
        split_param='v',                    # this model tests how v varies between high money (1) and high pain (0) 
        drift_criterion=True,               # the model estimates an intercept (dc) for v
        bias=True,                          # the model includes a bias for z
        include=['z', 'a', 't', 'v', 'sv', 'st', 'sz'],     # this model includes variability in z,t,a,v,sv,st,sz
        depends_on_dict = {                 # the depends on dict specifies that t,v,a,z additionally vary with the subjective value of pain  
            't': 'sv_pain', 
            'v': 'sv_pain', 
            'a': 'sv_pain', 
            'z': 'sv_pain'
        }
    elif version == 6:
        stim_col = 'stimulus_mapped_b',     # these are the created 'stimulus-coded' data frames
        split_param='v',                    # this model tests how v varies between high money (1) and high pain (0) 
        drift_criterion=True,               # the model estimates an intercept (dc) for v
        bias=True,                          # the model includes a bias for z
        include=['z', 'a', 't', 'v', 'sv', 'st', 'sz'],                     # this model only includes drift rate variability
        depends_on_dict = {                 # the depends on dict specifies that t,v,a,z additionally vary with the subjective value of pain  
            't': ['stimulus_mapped_a', 'sv_pain'], 
            'v': ['stimulus_mapped_b', 'sv_pain'], 
            'a': ['stimulus_mapped_a', 'sv_pain'], 
            'z': ['stimulus_mapped_a', 'sv_pain'], 
        }
    elif version == 7:
        stim_col = 'stimulus_mapped_a',     # these are the created 'stimulus-coded' data frames
        split_param='z',                    # this model tests how v varies between high money (1) and high pain (0) 
        drift_criterion=True,               # the model estimates an intercept (dc) for v
        bias=True,                          # the model includes a bias for z
        include=['z', 'a', 't', 'v', 'sv', 'st', 'sz'],                     # this model only includes drift rate variability
        depends_on_dict = {                 # the depends on dict specifies that t,v,a,z additionally vary with the subjective value of pain  
            't': ['stimulus_mapped_a', 'sv_pain'], 
            'v': ['stimulus_mapped_b', 'sv_pain'], 
            'a': ['stimulus_mapped_a', 'sv_pain'], 
            'z': ['stimulus_mapped_a', 'sv_pain'], 
        }
    elif version == 8:
        stim_col = 'stimulus_mapped_b',     # these are the created 'stimulus-coded' data frames
        split_param='z',                    # this model tests how v varies between high money (1) and high pain (0) 
        drift_criterion=True,               # the model estimates an intercept (dc) for v
        bias=True,                          # the model includes a bias for z
        include=['z', 'a', 't', 'v', 'sv', 'st', 'sz'],                     # this model only includes drift rate variability
        depends_on_dict = {                 # the depends on dict specifies that t,v,a,z additionally vary with the subjective value of pain  
            't': ['stimulus_mapped_a', 'sv_pain'], 
            'v': ['stimulus_mapped_b', 'sv_pain'], 
            'a': ['stimulus_mapped_a', 'sv_pain'], 
            'z': ['stimulus_mapped_a', 'sv_pain'], 
        }
        
#------------------------------------------------------------------------------------------------------------------       
# stimulus coding absolute value        
        
        
        
        
        
        
        # Define the parameters based on version
    if version == 1:
        stim_col = 'stimulus_mapped_c',     # these are the created 'stimulus-coded' data frames
        split_param='v',                    # this model tests how v varies between high money (1) and high pain (0) 
        drift_criterion=True,               # the model estimates an intercept (dc) for v
        bias=True,                          # the model includes a bias for z
        include=['sv'],                     # this model only includes drift rate variability
        depends_on_dict = {                 # the depends on dict specifies that t,v,a,z additionally vary with the subjective value of pain  
            't': 'sv_pain', 
            'v': 'sv_pain', 
            'a': 'sv_pain', 
            'z': 'sv_pain'
        }
    if version == 1:
        stim_col = 'stimulus_mapped_c',     # these are the created 'stimulus-coded' data frames
        split_param='v',                    # this model tests how v varies between high money (1) and high pain (0) 
        drift_criterion=True,               # the model estimates an intercept (dc) for v
        bias=True,                          # the model includes a bias for z
        include=['sv'],                     # this model only includes drift rate variability
        depends_on_dict = {                 # the depends on dict specifies that t,v,a,z additionally vary with the subjective value of pain  
            't': ['stimulus_mapped_c', 'sv_pain'], 
            'v': ['stimulus_mapped_d', 'sv_pain'], 
            'a': ['stimulus_mapped_c', 'sv_pain'], 
            'z': ['stimulus_mapped_c', 'sv_pain'], 
        }
    # Define the parameters based on version
    if version == 1:
        stim_col = 'stimulus_mapped_c',     # these are the created 'stimulus-coded' data frames
        split_param='v',                    # this model tests how v varies between high money (1) and high pain (0) 
        drift_criterion=True,               # the model estimates an intercept (dc) for v
        bias=True,                          # the model includes a bias for z
        include=['z', 'a', 't', 'v', 'sv', 'st', 'sz'],     # this model includes variability in z,t,a,v,sv,st,sz
        depends_on_dict = {                 # the depends on dict specifies that t,v,a,z additionally vary with the subjective value of pain  
            't': 'sv_pain', 
            'v': 'sv_pain', 
            'a': 'sv_pain', 
            'z': 'sv_pain'
        }
    if version == 1:
        stim_col = 'stimulus_mapped_a',     # these are the created 'stimulus-coded' data frames
        split_param='v',                    # this model tests how v varies between high money (1) and high pain (0) 
        drift_criterion=True,               # the model estimates an intercept (dc) for v
        bias=True,                          # the model includes a bias for z
        include=['z', 'a', 't', 'v', 'sv', 'st', 'sz'],                     # this model only includes drift rate variability
        depends_on_dict = {                 # the depends on dict specifies that t,v,a,z additionally vary with the subjective value of pain  
            't': ['stimulus_mapped_a', 'sv_pain'], 
            'v': ['stimulus_mapped_b', 'sv_pain'], 
            'a': ['stimulus_mapped_a', 'sv_pain'], 
            'z': ['stimulus_mapped_a', 'sv_pain'], 
        }
        
        
        
        
    # Define the parameters based on version
    if version == 1:
        stim_col = 'stimulus_mapped_b',     # these are the created 'stimulus-coded' data frames
        split_param='v',                    # this model tests how v varies between high money (1) and high pain (0) 
        drift_criterion=True,               # the model estimates an intercept (dc) for v
        bias=True,                          # the model includes a bias for z
        include=['z', 'a', 't', 'v', 'sv', 'st', 'sz'],     # this model includes variability in z,t,a,v,sv,st,sz
        depends_on_dict = {                 # the depends on dict specifies that t,v,a,z additionally vary with the subjective value of pain  
            't': 'sv_pain', 
            'v': 'sv_pain', 
            'a': 'sv_pain', 
            'z': 'sv_pain'
        }
    if version == 1:
        stim_col = 'stimulus_mapped_b',     # these are the created 'stimulus-coded' data frames
        split_param='v',                    # this model tests how v varies between high money (1) and high pain (0) 
        drift_criterion=True,               # the model estimates an intercept (dc) for v
        bias=True,                          # the model includes a bias for z
        include=['z', 'a', 't', 'v', 'sv', 'st', 'sz'],                     # this model only includes drift rate variability
        depends_on_dict = {                 # the depends on dict specifies that t,v,a,z additionally vary with the subjective value of pain  
            't': ['stimulus_mapped_a', 'sv_pain'], 
            'v': ['stimulus_mapped_b', 'sv_pain'], 
            'a': ['stimulus_mapped_a', 'sv_pain'], 
            'z': ['stimulus_mapped_a', 'sv_pain'], 
        }
    if version == 3:
        stim_col = 'accepted'
        depends_on_dict = {
            't': 'split', 
            'v': 'split', 
            'a': 'split', 
            'z': 'split'
        }
    if version == 4:
        stim_col = 'accepted'
        depends_on_dict = {
            't': 'split', 
            'v': ['split', 'sv_pain'],  # Interaction with continuous variable            
            'a': 'split', 
            'z': 'split'
        }
    if version == 5:
        stim_col = 'accepted'
        depends_on_dict = {
            't': 'split', 
            'v': ['split', 'sv_pain'],  # Interaction with continuous variable            
            'a': 'split', 
            'z': ['split', 'sv_pain']
        }
        
    
    
    
    
    
    m = hddm.HDDMStimCoding(data, 
                            stim_col='Abs_Money_Pain', 
                            split_param='v', 
                            drift_criterion=True, 
                            bias=True, 
                            include=('sv',), 
                            depends_on={
                                't': 'split', 
                                'v': ['split', 'sv_pain'],  # Interaction with continuous variable
                                'a': 'split', 
                                'z': 'split'
                            }, 
                            p_outlier=.05)







