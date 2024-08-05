## Running the hddm models
# Pipeline for running hddm
#
# orientated to code structure from Jan Willem de Gee on 2011-02-16. Copyright (c) 2011 __MyCompanyName__. All rights reserved.
#
# TO DO: More models

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


#------------------------------------------------------------------------------------------------------------------
# Structure of saving:

# /D:/Aberdeen_Uni_June24/MPColl_Lab/DockerData
#     ├── data_sets/
#     │   └── behavioural_sv_cleaned.csv
#     ├── model_dir/
#     │   ├── r1_0
#     │   ├── r1_1
#     │   ├── r1_2
#     ├── figures/
#     │   └── painreward_behavioural_data_combined_new_r1b/
#     │       ├── diagnostics/
#     │       │   ├── gelman_rubic.txt
#     │       │   ├── DIC.txt
#     │       │   ├── results.csv
#     │       │   └── posteriors.pdf
#     ├── other_script.py
#------------------------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------------------------
# very important! If you try to plot parameters such as 'v_C(Abs_value)_subj' without specifying the exact levels like [low_abs],
# [mid_abs], and [high_abs], the code will raise an error. 
#------------------------------------------------------------------------------------------------------------------
# params:
version = 20 # 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15, 
run = True

# standard params:
model_base_name = 'painreward_behavioural_data_combined_new_'
model_names = [
               'r0', 'r1', 'r2', 'r3', 'r4', 'r5', 'r6', 'r7', 'r8', 'r9', 'r10',
               'r11', 'r12', 'r13', 'r14', 'r15', 'r16', 'r17', 'r18','r19', 'r20',
               'r21', 'r22', 'r23', 'r24','r25', 'r26', 'r27', 'r28', 'r29', 'r30',
               'r31','r32', 'r33','r34','r35', 'r36', 'r37', 'r38', 'r39', 'r40',
               'r41','r42', 'r43','r44','r45', 'r46', 'r47', 'r48', 'r49', 'r50', 
               'r51', 'r52', 'r53'
               ]

nr_samples = 500
nr_models = 3
parallel = True
accuracy_coding = False


# settings:
# model_name

model_name = model_names[version]

#data:
hddm_models_path = os.path.join(current_directory,'Hddm_models')
sys.path.append(hddm_models_path)
data_path1 = os.path.join(current_directory, 'data_sets', 'behavioural_sv_cleaned_final_3.csv')
data = pd.read_csv(data_path1, sep = ',')
data.dropna(subset=['rt', "painlevel", "moneylevel", "accepted",'acceptance_pair','sv_money', 'sv_pain', 'sv_both', 'p_pain_all', 'Abs_Money_Pain','OV_Money_Pain', 'sv_pain_para','sv_both_para','k_pain_para','beta_para','bias_para','STA_SAI_Score','STA_TAI_Score','PCS_Score'], inplace = True)    #'STA_SAI_Score','STA_TAI_Score','PCS_Score'

# drop entire participants for quest data only, NO FOR ENTIRE DATA, otherwise the operating system kills the worker
# quest_vers = [x, z, u, i]  # questionnaire versions 
# if version in quest_vers:
#     data.dropna(subset=["STA_SAI_Score","STA_TAI_Score","PCS_Score"], inplace=True)

# converting from object to category for the DDM
data['Abs_Money_Pain'] = data['Abs_Money_Pain'].astype("category")
data['OV_Money_Pain'] = data['OV_Money_Pain'].astype("category")
data['Abs_value'] = data['Abs_value'].astype("category")
data['OV_value'] = data['OV_value'].astype("category")
data['acceptance_pair'] = data['acceptance_pair'].astype("category")


def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# model dir:
model_dir = 'model_dir/'
ensure_dir(model_dir)


# figures dir:
fig_dir = os.path.join('figures', model_base_name + model_name)
try:
    os.system('mkdir {}'.format(fig_dir))
    os.system('mkdir {}'.format(os.path.join(fig_dir, 'diagnostics')))
except:
    pass

# subjects:
subjects = np.unique(data.subj_idx)
nr_subjects = subjects.shape[0]


#
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
# drift diffusion models.
#------------------------------------------------------------------------------------------------------------------
def run_model(trace_id, data, model_dir, model_name, version, samples=500, accuracy_coding=False):
    import os
    import numpy as np
    import hddm
    from patsy import dmatrix  

    ensure_dir(model_dir)   
    
    if version == 0:  # this is the 0 model
        m = hddm.HDDM(data, 
                      include=['a', 'z', 'v', 't', 'st', 'sz', 'sv'],
                      p_outlier=.05,
                      trace_subjs=True,
                      is_group_model=True)
        m.find_starting_values()
        m.sample(samples, burn=samples/2, dbname=os.path.join(model_dir, model_name + '_db{}'.format(trace_id)), db='pickle')
        return m
    elif version == 1:  # drift rate is dependent on the the sv_pain_para
        v_reg = {'model': 'v ~ 1 + sv_pain_para', 'link_func': lambda x: x}
        reg_descr = [v_reg]
    elif version == 2:  # all parameters vary with the sv_pain_para
        v_reg = {'model': 'v ~ 1 + sv_pain_para', 'link_func': lambda x: x}
        t_reg = {'model': 't ~ 1 + sv_pain_para', 'link_func': lambda x: x}
        a_reg = {'model': 'a ~ 1 + sv_pain_para', 'link_func': lambda x: x}
        z_reg = {'model': 'z ~ 1 + sv_pain_para', 'link_func': lambda x: x}
        reg_descr = [v_reg, t_reg, a_reg, z_reg]
    elif version == 3:
        v_reg = {'model': 'v ~ 1 + sv_money', 'link_func': lambda x: x}  # drift rate varies with the sv_money
        reg_descr = [v_reg]
    elif version == 4:  # all parameters vary with the sv_money
        v_reg = {'model': 'v ~ 1 + sv_money', 'link_func': lambda x: x}
        t_reg = {'model': 't ~ 1 + sv_money', 'link_func': lambda x: x}
        a_reg = {'model': 'a ~ 1 + sv_money', 'link_func': lambda x: x}
        z_reg = {'model': 'z ~ 1 + sv_money', 'link_func': lambda x: x}
        reg_descr = [v_reg, t_reg, a_reg, z_reg]
    elif version == 5:  # only the non-decision time varies with the sv_pain_para, all other parameters are fixed
        t_reg = {'model': 't ~ 1 + sv_pain_para', 'link_func': lambda x: x}
        reg_descr = [t_reg]
    elif version == 6:  # only the threshold varies with the sv_pain_para, all other parameters are fixed
        a_reg = {'model': 'a ~ 1 + sv_pain_para', 'link_func': lambda x: x}
        reg_descr = [a_reg]
    elif version == 7:  # only the non-decision time varies with sv_money, all other parameters are fixed
        t_reg = {'model': 't ~ 1 + sv_money', 'link_func': lambda x: x}
        reg_descr = [t_reg]
    elif version == 8:  # only the threshold varies with sv_money, all other parameters are fixed
        a_reg = {'model': 'a ~ 1 + sv_money', 'link_func': lambda x: x}
        reg_descr = [a_reg]
    elif version == 9:  # the drift rate depends on the interaction between sv_money and sv_pain_para
        v_reg = {'model': 'v ~ 1 + sv_pain_para + sv_money + sv_pain_para * sv_money', 'link_func': lambda x: x}
        reg_descr = [v_reg]
    elif version == 10:  # all parameters depend on the interaction between sv_pain_para and sv_money
        v_reg = {'model': 'v ~ 1 + sv_pain_para + sv_money + sv_pain_para * sv_money', 'link_func': lambda x: x}
        t_reg = {'model': 't ~ 1 + sv_pain_para + sv_money + sv_pain_para * sv_money', 'link_func': lambda x: x}
        a_reg = {'model': 'a ~ 1 + sv_pain_para + sv_money + sv_pain_para * sv_money', 'link_func': lambda x: x}
        z_reg = {'model': 'z ~ 1 + sv_pain_para + sv_money + sv_pain_para * sv_money', 'link_func': lambda x: x}
        reg_descr = [v_reg, t_reg, a_reg, z_reg]
    elif version == 11:  # the drift rate depends on the main effects of sv_pain_para and sv_money
        v_reg = {'model': 'v ~ 1 + sv_pain_para + sv_money', 'link_func': lambda x: x}
        reg_descr = [v_reg]
    elif version == 12:  # all parameters vary with the main effects of sv_pain_para and sv_money
        v_reg = {'model': 'v ~ 1 + sv_pain_para + sv_money', 'link_func': lambda x: x}
        t_reg = {'model': 't ~ 1 + sv_pain_para + sv_money', 'link_func': lambda x: x}
        a_reg = {'model': 'a ~ 1 + sv_pain_para + sv_money', 'link_func': lambda x: x}
        z_reg = {'model': 'a ~ 1 + sv_pain_para + sv_money', 'link_func': lambda x: x}
        reg_descr = [v_reg, t_reg, a_reg, z_reg]
    elif version == 13:  # the drift rate depends on sv_both
        v_reg = {'model': 'v ~ 1 + sv_both', 'link_func': lambda x: x}
        reg_descr = [v_reg]
    elif version == 14:  # the drift rate depends on the interaction between sv_both * sv_pain_para * sv_money
        v_reg = {'model': 'v ~ 1 + sv_both * sv_pain_para * sv_money', 'link_func': lambda x: x}
        reg_descr = [v_reg]
    elif version == 15:  # all parameters depend on the interaction between sv_both, sv_pain_para, and sv_money
        v_reg = {'model': 'v ~ 1 + sv_both * sv_pain_para * sv_money', 'link_func': lambda x: x}
        t_reg = {'model': 't ~ 1 + sv_both * sv_pain_para * sv_money', 'link_func': lambda x: x}
        a_reg = {'model': 'a ~ 1 + sv_both * sv_pain_para * sv_money', 'link_func': lambda x: x}
        z_reg = {'model': 'z ~ 1 + sv_both * sv_pain_para * sv_money', 'link_func': lambda x: x}
        reg_descr = [v_reg, t_reg, a_reg, z_reg]
    elif version == 16:  # the drift rate depends on the interaction between sv_pain_para and the conditions of Abs_value
        v_reg = {'model': 'v ~ sv_pain_para:C(Abs_value)', 'link_func': lambda x: x}
        reg_descr = [v_reg]
    elif version == 17:  # the drift rate depends on the interaction between sv_pain_para and the OV_value conditions
        v_reg = {'model': 'v ~ sv_pain_para:C(OV_value)', 'link_func': lambda x: x}
        reg_descr = [v_reg]
    elif version == 18:  # the drift rate depends on the interaction between sv_pain_para and the Abs_Money_Pain conditions
        v_reg = {'model': 'v ~ sv_pain_para:C(Abs_Money_Pain)', 'link_func': lambda x: x}
        reg_descr = [v_reg]
    elif version == 19:  # the drift rate depends on the interaction between sv_pain_para and the OV_Money_Pain conditions
        v_reg = {'model': 'v ~ sv_pain_para:C(OV_Money_Pain)', 'link_func': lambda x: x}
        reg_descr = [v_reg]
#------------------------------------------------------------------------------------------------------------------         
    elif version == 20:  # the drift rate depends on the interaction between sv_pain_para and the OV_Money_Pain conditions
        v_reg = {'model': 'v ~ 0 + sv_pain_para:C(OV_Money_Pain):C(Abs_Money_Pain)', 'link_func': lambda x: x}
        reg_descr = [v_reg]
    elif version == 21:  # the drift rate depends on the interaction between sv_pain_para, the conditions of OV_value, and the conditions of acceptance pair
        v_reg = {'model': 'v ~ 0 + sv_pain_para:C(Abs_value):C(acceptance_pair)', 'link_func': lambda x: x}
        reg_descr = [v_reg]
    elif version == 22:  # the drift rate depends on the interaction between sv_pain_para, the conditions of OV_value, and the conditions of acceptance pair
        v_reg = {'model': 'v ~ 0 + sv_pain_para:C(OV_value):C(acceptance_pair)', 'link_func': lambda x: x}
        reg_descr = [v_reg]
    elif version == 23:  # the drift rate depends on the interaction between sv_pain_para, the conditions of OV_value, and the conditions of acceptance pair
        v_reg = {'model': 'v ~ 0 + sv_pain_para:C(OV_value):C(Abs_value)', 'link_func': lambda x: x}
        reg_descr = [v_reg]
    elif version == 24:  # the drift rate depends on the interaction between sv_pain_para, the conditions of OV_value, and the conditions of acceptance pair
        v_reg = {'model': 'v ~ 0 + sv_pain_para:C(OV_value):C(Abs_value):C(acceptance_pair)', 'link_func': lambda x: x}
        reg_descr = [v_reg]
    elif version == 25:  # the drift rate depends on the interaction between sv_pain_para, the conditions of OV_value, and the conditions of acceptance pair
        v_reg = {'model': 'v ~ 0 + sv_pain_para + C(OV_value) + C(acceptance_pair) + sv_pain_para:C(OV_value) + sv_pain_para:C(acceptance_pair) + C(OV_value):C(acceptance_pair) + sv_pain_para:C(OV_value):C(acceptance_pair)', 'link_func': lambda x: x}
        reg_descr = [v_reg]
    elif version == 26:  # all parameters depend on the interaction between sv_pain_para and the conditions of Abs_value
        v_reg = {'model': 'v ~ 0 + sv_pain_para:C(Abs_value)', 'link_func': lambda x: x}
        t_reg = {'model': 't ~ 0 + sv_pain_para:C(Abs_value)', 'link_func': lambda x: x}
        a_reg = {'model': 'a ~ 0 + sv_pain_para:C(Abs_value)', 'link_func': lambda x: x}
        z_reg = {'model': 'z ~ 0 + sv_pain_para:C(Abs_value)', 'link_func': lambda x: x}
        reg_descr = [v_reg, t_reg, a_reg, z_reg]
    elif version == 27:  # all parameters depend on the interaction between sv_pain_para and the OV_value conditions
        v_reg = {'model': 'v ~ 0 + sv_pain_para:C(OV_value)', 'link_func': lambda x: x}
        t_reg = {'model': 't ~ 0 + sv_pain_para:C(OV_value)', 'link_func': lambda x: x}
        a_reg = {'model': 'a ~ 0 + sv_pain_para:C(OV_value)', 'link_func': lambda x: x}
        z_reg = {'model': 'z ~ 0 + sv_pain_para:C(OV_value)', 'link_func': lambda x: x}
        reg_descr = [v_reg, t_reg, a_reg, z_reg]
    elif version == 28:  # all parameters depend on the interaction between sv_pain_para, the conditions of OV_value, and the conditions of acceptance pair
        v_reg = {'model': 'v ~ 0 + sv_pain_para:C(OV_value):C(acceptance_pair)', 'link_func': lambda x: x}
        t_reg = {'model': 't ~ 0 + sv_pain_para:C(OV_value):C(acceptance_pair)', 'link_func': lambda x: x}
        a_reg = {'model': 'a ~ 0 + sv_pain_para:C(OV_value):C(acceptance_pair)', 'link_func': lambda x: x}
        z_reg = {'model': 'z ~ 0 + sv_pain_para:C(OV_value):C(acceptance_pair)', 'link_func': lambda x: x}
        reg_descr = [v_reg, t_reg, a_reg, z_reg]
    elif version == 29:  # all parameters depend on the interaction between sv_pain_para, the conditions of Abs_value, and the conditions of acceptance pair
        v_reg = {'model': 'v ~ 0 + sv_pain_para:C(Abs_value):C(acceptance_pair)', 'link_func': lambda x: x}
        t_reg = {'model': 't ~ 0 + sv_pain_para:C(Abs_value):C(acceptance_pair)', 'link_func': lambda x: x}
        a_reg = {'model': 'a ~ 0 + sv_pain_para:C(Abs_value):C(acceptance_pair)', 'link_func': lambda x: x}
        z_reg = {'model': 'z ~ 0 + sv_pain_para:C(Abs_value):C(acceptance_pair)', 'link_func': lambda x: x}
        reg_descr = [v_reg, t_reg, a_reg, z_reg]
    # elif version == 30:                                                 # all parameters depend on the interaction between sv_pain_para, the conditions of OV_value and the conditions of acceptance pair 
    #     v_reg = {'model': 'v ~ 0 + sv_pain_para + C(OV_value) + C(acceptance_pair) + sv_pain_para:C(OV_value) + sv_pain_para:C(acceptance_pair) + C(OV_value):C(acceptance_pair) + sv_pain_para:C(OV_value):C(acceptance_pair)', 'link_func': lambda x: x}
    #     t_reg = {'model': 't ~ 0 + sv_pain_para + C(OV_value) + C(acceptance_pair) + sv_pain_para:C(OV_value) + sv_pain_para:C(acceptance_pair) + C(OV_value):C(acceptance_pair) + sv_pain_para:C(OV_value):C(acceptance_pair)', 'link_func': lambda x: x}
    #     a_reg = {'model': 'a ~ 0 + sv_pain_para + C(OV_value) + C(acceptance_pair) + sv_pain_para:C(OV_value) + sv_pain_para:C(acceptance_pair) + C(OV_value):C(acceptance_pair) + sv_pain_para:C(OV_value):C(acceptance_pair)', 'link_func': lambda x: x}
    #     z_reg = {'model': 'z ~ 0 + sv_pain_para + C(OV_value) + C(acceptance_pair) + sv_pain_para:C(OV_value) + sv_pain_para:C(acceptance_pair) + C(OV_value):C(acceptance_pair) + sv_pain_para:C(OV_value):C(acceptance_pair)', 'link_func': lambda x: x}
    #     reg_descr = [v_reg, t_reg, a_reg, z_reg]
    # elif version == 31:                                                 # all parameters depend on the interaction between sv_pain_para, the conditions of Abs_value and the conditions of acceptance pair 
    #     v_reg = {'model': 'v ~ 0 + sv_pain_para + C(Abs_value) + C(acceptance_pair) + sv_pain_para:C(Abs_value) + sv_pain_para:C(acceptance_pair) + C(Abs_value):C(acceptance_pair) + sv_pain_para:C(Abs_value):C(acceptance_pair)', 'link_func': lambda x: x}
    #     t_reg = {'model': 't ~ 0 + sv_pain_para + C(Abs_value) + C(acceptance_pair) + sv_pain_para:C(Abs_value) + sv_pain_para:C(acceptance_pair) + C(Abs_value):C(acceptance_pair) + sv_pain_para:C(Abs_value):C(acceptance_pair)', 'link_func': lambda x: x}
    #     a_reg = {'model': 'a ~ 0 + sv_pain_para + C(Abs_value) + C(acceptance_pair) + sv_pain_para:C(Abs_value) + sv_pain_para:C(acceptance_pair) + C(Abs_value):C(acceptance_pair) + sv_pain_para:C(Abs_value):C(acceptance_pair)', 'link_func': lambda x: x}
    #     z_reg = {'model': 'z ~ 0 + sv_pain_para + C(Abs_value) + C(acceptance_pair) + sv_pain_para:C(Abs_value) + sv_pain_para:C(acceptance_pair) + C(Abs_value):C(acceptance_pair) + sv_pain_para:C(Abs_value):C(acceptance_pair)', 'link_func': lambda x: x}
    #     reg_descr = [v_reg, t_reg, a_reg, z_reg]
    # elif version == 32:  # all parameters depend on the interaction between sv_pain_para, the conditions of OV_value, Abs_value, and the conditions of acceptance pair
    #     v_reg = {'model': 'v ~ 0 + sv_pain_para:C(OV_value):C(Abs_value):C(acceptance_pair)','link_func': lambda x: x}
    #     v_reg = {'model': 'v ~ 0 + sv_pain_para:C(OV_value):C(Abs_value):C(acceptance_pair)','link_func': lambda x: x}
    #     v_reg = {'model': 'v ~ 0 + sv_pain_para:C(OV_value):C(Abs_value):C(acceptance_pair)','link_func': lambda x: x}
    #     v_reg = {'model': 'v ~ 0 + sv_pain_para:C(OV_value):C(Abs_value):C(acceptance_pair)','link_func': lambda x: x}
    #     reg_descr = [v_reg, t_reg, a_reg, z_reg]
    elif version == 33:  # v and a parameters depend on the interaction between sv_pain_para, OV_value, and acceptance pair; z depends on the interaction between sv_pain_para and acceptance pair; t only depends on sv_pain_para
        v_reg = {'model': 'v ~ 0 + sv_pain_para:C(OV_value):C(acceptance_pair)', 'link_func': lambda x: x}
        t_reg = {'model': 't ~ 0 + sv_pain_para', 'link_func': lambda x: x}
        a_reg = {'model': 'a ~ 0 + sv_pain_para:C(OV_value):C(acceptance_pair)', 'link_func': lambda x: x}
        z_reg = {'model': 'z ~ 0 + sv_pain_para:C(acceptance_pair)', 'link_func': lambda x: x}
        reg_descr = [v_reg, t_reg, a_reg, z_reg]
    # elif version == 34:  # all parameters depend on the interaction between sv_pain_para, the conditions of Abs_value and the conditions of acceptance pair
    #     v_reg = {'model': 'v ~ 0 + sv_pain_para:C(Abs_value):C(acceptance_pair)', 'link_func': lambda x: x}
    #     t_reg = {'model': 't ~ 0 + sv_pain_para', 'link_func': lambda x: x}
    #     a_reg = {'model': 'a ~ 0 + sv_pain_para:C(Abs_value):C(acceptance_pair)', 'link_func': lambda x: x}
    #     z_reg = {'model': 'z ~ 0 + sv_pain_para:C(acceptance_pair)', 'link_func': lambda x: x}
    #     reg_descr = [v_reg, t_reg, a_reg, z_reg]
    # elif version == 35:  # all parameters depend on the interaction between sv_pain_para and the conditions of Abs_value
    #     v_reg = {'model': 'v ~ 0 + sv_pain_para:C(Abs_value)', 'link_func': lambda x: x}
    #     t_reg = {'model': 't ~ 0 + C(Abs_value) ', 'link_func': lambda x: x}
    #     a_reg = {'model': 'a ~ 0 + sv_pain_para:C(Abs_value)', 'link_func': lambda x: x}
    #     z_reg = {'model': 'z ~ 0 + sv_pain_para', 'link_func': lambda x: x}
    #     reg_descr = [v_reg, t_reg, a_reg, z_reg]
    # elif version == 36:  # all parameters depend on the interaction between sv_pain_para and the OV_value conditions
    #     v_reg = {'model': 'v ~ 0 + sv_pain_para:C(OV_value)', 'link_func': lambda x: x}
    #     t_reg = {'model': 't ~ 0 + sv_pain_para', 'link_func': lambda x: x}
    #     a_reg = {'model': 'a ~ 0 + sv_pain_para:C(OV_value)', 'link_func': lambda x: x}
    #     z_reg = {'model': 'z ~ 0 + sv_pain_para', 'link_func': lambda x: x}
    #     reg_descr = [v_reg, t_reg, a_reg, z_reg]
    # elif version == 37:  # all parameters depend on the interaction between sv_pain_para and the Abs_Money_Pain conditions
    #     v_reg = {'model': 'v ~ 0 + sv_pain_para:C(Abs_Money_Pain)', 'link_func': lambda x: x}
    #     t_reg = {'model': 't ~ 0 + sv_pain_para', 'link_func': lambda x: x}
    #     a_reg = {'model': 'a ~ 0 + sv_pain_para:C(Abs_Money_Pain)', 'link_func': lambda x: x}
    #     z_reg = {'model': 'z ~ 0 + sv_pain_para', 'link_func': lambda x: x}
    #     reg_descr = [v_reg, t_reg, a_reg, z_reg]
    # elif version == 38:  # all parameters depend on the interaction between sv_pain_para and the OV_Money_Pain conditions
    #     v_reg = {'model': 'v ~ 0 + sv_pain_para:C(OV_Money_Pain)', 'link_func': lambda x: x}
    #     t_reg = {'model': 't ~ 0 + sv_pain_para', 'link_func': lambda x: x}
    #     a_reg = {'model': 'a ~ 0 + sv_pain_para:C(OV_Money_Pain)', 'link_func': lambda x: x}
    #     z_reg = {'model': 'z ~ 0 + sv_pain_para', 'link_func': lambda x: x}
    #     reg_descr = [v_reg, t_reg, a_reg, z_reg] 
    
#------------------------------------------------------------------------------------------------------------------         
    # Questionnaire data    
    elif version == 34:  # v depends on STA_SAI_Score
        v_reg = {'model': 'v ~ STA_SAI_Score', 'link_func': lambda x: x}
        reg_descr = [v_reg]
    elif version == 35:  # a depends on STA_SAI_Score
        a_reg = {'model': 'a ~ STA_SAI_Score', 'link_func': lambda x: x}
        reg_descr = [a_reg]
    elif version == 36:  # v depends on STA_TAI_Score
        v_reg = {'model': 'v ~ STA_TAI_Score', 'link_func': lambda x: x}
        reg_descr = [v_reg]
    elif version == 37:  # a depends on STA_TAI_Score
        a_reg = {'model': 'a ~ STA_TAI_Score', 'link_func': lambda x: x}
        reg_descr = [a_reg]
    elif version == 38:  # v depends on PCS_Score
        v_reg = {'model': 'v ~ PCS_Score', 'link_func': lambda x: x}
        reg_descr = [v_reg]
    elif version == 39:  # a depends on PCS_Score
        a_reg = {'model': 'a ~ PCS_Score', 'link_func': lambda x: x}
        reg_descr = [a_reg]

    elif version == 40: 
        v_reg = {'model': 'v ~ STA_SAI_Score', 'link_func': lambda x: x}
        z_reg = {'model': 'z ~ STA_SAI_Score', 'link_func': lambda x: x}
        t_reg = {'model': 't ~ STA_SAI_Score', 'link_func': lambda x: x}
        a_reg = {'model': 'a ~ STA_SAI_Score', 'link_func': lambda x: x}
        reg_descr = [v_reg, z_reg, t_reg, a_reg]

    # STA-TAI 
    elif version == 41:
        v_reg = {'model': 'v ~ STA_TAI_Score', 'link_func': lambda x: x}
        z_reg = {'model': 'z ~ STA_TAI_Score', 'link_func': lambda x: x}
        t_reg = {'model': 't ~ STA_TAI_Score', 'link_func': lambda x: x}
        a_reg = {'model': 'a ~ STA_TAI_Score', 'link_func': lambda x: x}
        reg_descr = [v_reg, z_reg, t_reg, a_reg]

    # PCS 
    elif version == 42:
        v_reg = {'model': 'v ~ PCS_Score', 'link_func': lambda x: x}
        z_reg = {'model': 'z ~ PCS_Score', 'link_func': lambda x: x}
        t_reg = {'model': 't ~ PCS_Score', 'link_func': lambda x: x}
        a_reg = {'model': 'a ~ PCS_Score', 'link_func': lambda x: x}
        reg_descr = [v_reg, z_reg, t_reg, a_reg]
    
    
    elif version == 43:  # all parameters depend on STA_SAI_Score and sv_pain_para with OV_value interaction
        v_reg = {'model': 'v ~ 0 + STA_SAI_Score:sv_pain_para:C(OV_value)', 'link_func': lambda x: x}
        z_reg = {'model': 'z ~ 0 + STA_SAI_Score:sv_pain_para:C(OV_value)', 'link_func': lambda x: x}
        t_reg = {'model': 't ~ 0 + STA_SAI_Score:sv_pain_para:C(OV_value)', 'link_func': lambda x: x}
        a_reg = {'model': 'a ~ 0 + STA_SAI_Score:sv_pain_para:C(OV_value)', 'link_func': lambda x: x}
        reg_descr = [v_reg, z_reg, t_reg, a_reg]
    elif version == 44:  # all parameters depend on STA_TAI_Score and sv_pain_para with OV_value interaction
        v_reg = {'model': 'v ~ 0 + STA_TAI_Score:sv_pain_para:C(OV_value)', 'link_func': lambda x: x}
        z_reg = {'model': 'z ~ 0 + STA_TAI_Score:sv_pain_para:C(OV_value)', 'link_func': lambda x: x}
        t_reg = {'model': 't ~ 0 + STA_TAI_Score:sv_pain_para:C(OV_value)', 'link_func': lambda x: x}
        a_reg = {'model': 'a ~ 0 + STA_TAI_Score:sv_pain_para:C(OV_value)', 'link_func': lambda x: x}
        reg_descr = [v_reg, z_reg, t_reg, a_reg]
    elif version == 45:  # all parameters depend on PCS_Score and sv_pain_para with OV_value interaction
        v_reg = {'model': 'v ~ 0 + PCS_Score:sv_pain_para:C(OV_value)', 'link_func': lambda x: x}
        z_reg = {'model': 'z ~ 0 + PCS_Score:sv_pain_para:C(OV_value)', 'link_func': lambda x: x}
        t_reg = {'model': 't ~ 0 + PCS_Score:sv_pain_para:C(OV_value)', 'link_func': lambda x: x}
        a_reg = {'model': 'a ~ 0 + PCS_Score:sv_pain_para:C(OV_value)', 'link_func': lambda x: x}
        reg_descr = [v_reg, z_reg, t_reg, a_reg]
    elif version == 46:  # all parameters depend on STA_SAI_Score and sv_pain_para with Abs_value interaction
        v_reg = {'model': 'v ~ 0 + STA_SAI_Score:sv_pain_para:C(Abs_value)', 'link_func': lambda x: x}
        z_reg = {'model': 'z ~ 0 + STA_SAI_Score:sv_pain_para:C(Abs_value)', 'link_func': lambda x: x}
        t_reg = {'model': 't ~ 0 + STA_SAI_Score:sv_pain_para:C(Abs_value)', 'link_func': lambda x: x}
        a_reg = {'model': 'a ~ 0 + STA_SAI_Score:sv_pain_para:C(Abs_value)', 'link_func': lambda x: x}
        reg_descr = [v_reg, z_reg, t_reg, a_reg]
    elif version == 47:  # all parameters depend on STA_TAI_Score and sv_pain_para with Abs_value interaction
        v_reg = {'model': 'v ~ 0 + STA_TAI_Score:sv_pain_para:C(Abs_value)', 'link_func': lambda x: x}
        z_reg = {'model': 'z ~ 0 + STA_TAI_Score:sv_pain_para:C(Abs_value)', 'link_func': lambda x: x}
        t_reg = {'model': 't ~ 0 + STA_TAI_Score:sv_pain_para:C(Abs_value)', 'link_func': lambda x: x}
        a_reg = {'model': 'a ~ 0 + STA_TAI_Score:sv_pain_para:C(Abs_value)', 'link_func': lambda x: x}
        reg_descr = [v_reg, z_reg, t_reg, a_reg]
    elif version == 48:  # all parameters depend on PCS_Score and sv_pain_para with Abs_value interaction
        v_reg = {'model': 'v ~ 0 + PCS_Score:sv_pain_para:C(Abs_value)', 'link_func': lambda x: x}
        z_reg = {'model': 'z ~ 0 + PCS_Score:sv_pain_para:C(Abs_value)', 'link_func': lambda x: x}
        t_reg = {'model': 't ~ 0 + PCS_Score:sv_pain_para:C(Abs_value)', 'link_func': lambda x: x}
        a_reg = {'model': 'a ~ 0 + PCS_Score:sv_pain_para:C(Abs_value)', 'link_func': lambda x: x}
        reg_descr = [v_reg, z_reg, t_reg, a_reg]
    elif version == 49:  # all parameters depend on STA_SAI_Score and sv_pain_para with acceptance_pair interaction
        v_reg = {'model': 'v ~ 0 + STA_SAI_Score:sv_pain_para:C(acceptance_pair)', 'link_func': lambda x: x}
        z_reg = {'model': 'z ~ 0 + STA_SAI_Score:sv_pain_para:C(acceptance_pair)', 'link_func': lambda x: x}
        t_reg = {'model': 't ~ 0 + STA_SAI_Score:sv_pain_para:C(acceptance_pair)', 'link_func': lambda x: x}
        a_reg = {'model': 'a ~ 0 + STA_SAI_Score:sv_pain_para:C(acceptance_pair)', 'link_func': lambda x: x}
        reg_descr = [v_reg, z_reg, t_reg, a_reg]
    elif version == 50:  # all parameters depend on STA_TAI_Score and sv_pain_para with acceptance_pair interaction
        v_reg = {'model': 'v ~ 0 + STA_TAI_Score:sv_pain_para:C(acceptance_pair)', 'link_func': lambda x: x}
        z_reg = {'model': 'z ~ 0 + STA_TAI_Score:sv_pain_para:C(acceptance_pair)', 'link_func': lambda x: x}
        t_reg = {'model': 't ~ 0 + STA_TAI_Score:sv_pain_para:C(acceptance_pair)', 'link_func': lambda x: x}
        a_reg = {'model': 'a ~ 0 + STA_TAI_Score:sv_pain_para:C(acceptance_pair)', 'link_func': lambda x: x}
        reg_descr = [v_reg, z_reg, t_reg, a_reg]
    elif version == 51:  # all parameters depend on PCS_Score and sv_pain_para with acceptance_pair interaction
        v_reg = {'model': 'v ~ 0 + PCS_Score:sv_pain_para:C(acceptance_pair)', 'link_func': lambda x: x}
        z_reg = {'model': 'z ~ 0 + PCS_Score:sv_pain_para:C(acceptance_pair)', 'link_func': lambda x: x}
        t_reg = {'model': 't ~ 0 + PCS_Score:sv_pain_para:C(acceptance_pair)', 'link_func': lambda x: x}
        a_reg = {'model': 'a ~ 0 + PCS_Score:sv_pain_para:C(acceptance_pair)', 'link_func': lambda x: x}
        reg_descr = [v_reg, z_reg, t_reg, a_reg]
        
    elif version == 52:  # v depends on the interaction between sv_pain_para and STA_SAI_Score, STA_TAI_Score, and PCS_Score
        v_reg = {'model': 'v ~ 1 + sv_pain_para * STA_SAI_Score + sv_pain_para * STA_TAI_Score + sv_pain_para * PCS_Score + sv_pain_para * STA_SAI_Score * STA_TAI_Score * PCS_Score', 'link_func': lambda x: x}
        reg_descr = [v_reg]
    elif version == 53:  # all parameters depend on the interaction between sv_pain_para, STA_SAI_Score, STA_TAI_Score, and PCS_Score
        v_reg = {'model': 'v ~ 1 + sv_pain_para * STA_SAI_Score + sv_pain_para * STA_TAI_Score + sv_pain_para * PCS_Score + sv_pain_para * STA_SAI_Score * STA_TAI_Score * PCS_Score', 'link_func': lambda x: x}
        t_reg = {'model': 't ~ 1 + sv_pain_para * STA_SAI_Score + sv_pain_para * STA_TAI_Score + sv_pain_para * PCS_Score + sv_pain_para * STA_SAI_Score * STA_TAI_Score * PCS_Score', 'link_func': lambda x: x}
        a_reg = {'model': 'a ~ 1 + sv_pain_para * STA_SAI_Score + sv_pain_para * STA_TAI_Score + sv_pain_para * PCS_Score + sv_pain_para * STA_SAI_Score * STA_TAI_Score * PCS_Score', 'link_func': lambda x: x}
        z_reg = {'model': 'z ~ 1 + sv_pain_para * STA_SAI_Score + sv_pain_para * STA_TAI_Score + sv_pain_para * PCS_Score + sv_pain_para * STA_SAI_Score * STA_TAI_Score * PCS_Score', 'link_func': lambda x: x}
        reg_descr = [v_reg, t_reg, a_reg, z_reg]
    
    # Questionnaire data    
    m = hddm.models.HDDMRegressor(data, 
                                  reg_descr, 
                                  p_outlier=.05, 
                                  include=['a','z','v','t','st', 'sz', 'sv'],
                                  group_only_regressors=False,
                                  keep_regressor_trace=True)
    m.find_starting_values()
    m.sample(samples, burn=samples/2, dbname=os.path.join(model_dir, model_name + '_db{}'.format(trace_id)), db='pickle')

    return m
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Main function for running/loading models
def drift_diffusion_hddm(data, samples=500, n_jobs=3, run=True, parallel=True, model_name='model', model_dir='.', version=version, accuracy_coding=False):
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
    
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Function to plot the parameters
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def analyze_model(models, fig_dir, nr_models, version):
    sns.set_theme(style='darkgrid', font='sans-serif', font_scale=0.5)
    
    combined_model = kabuki.utils.concat_models(models)
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
    elif version == 5:
        params_of_interest = ['z', 'sv', 'sz', 'st', 't_Intercept', 't_sv_pain_para']
        params_of_interest_s = ['z_subj', 'sv_subj', 'sz_subj', 'st_subj','t_Intercept_subj', 't_sv_pain_para_subj']
        titles = ['Starting point', 'Inter-trial variability in drift rate', 'Inter-trial variability in starting point', 'Inter-trial variability in non-decision time',
                'Intercept non-decision time', 'Non-decision time sv_pain_para']
    elif version == 6:
        params_of_interest = ['z', 'sv', 'sz', 'st', 'a_Intercept', 'a_sv_pain_para']
        params_of_interest_s = ['z_subj', 'sv_subj', 'sz_subj', 'st_subj', 'a_Intercept_subj', 'a_sv_pain_para_subj']
        titles = ['Starting point', 'Inter-trial variability in drift rate', 'Inter-trial variability in starting point', 'Inter-trial variability in non-decision time','Intercept boundary separation', 'Boundary separation sv_pain_para']   
    elif version == 7:
        params_of_interest = ['z', 'sv', 'sz', 'st', 't_Intercept', 't_sv_money']
        params_of_interest_s = ['z_subj', 'sv_subj', 'sz_subj', 'st_subj', 't_Intercept_subj', 't_sv_money_subj']
        titles = ['Starting point', 'Inter-trial variability in drift rate', 'Inter-trial variability in starting point', 'Inter-trial variability in non-decision time','Intercept non-decision time', 'Non-decision time sv_money']
    elif version == 8:
        params_of_interest = ['z', 'sv', 'sz', 'st', 'a_Intercept', 'a_sv_money']
        params_of_interest_s = ['z_subj', 'sv_subj', 'sz_subj', 'st_subj', 'a_Intercept_subj', 'a_sv_money_subj']
        titles = ['Starting point', 'Inter-trial variability in drift rate', 'Inter-trial variability in starting point', 'Inter-trial variability in non-decision time','Intercept boundary separation', 'Boundary separation sv_money']
    elif version == 9:
        params_of_interest = ['z', 'a', 't', 'sv', 'sz', 'st', 'v_Intercept', 'v_sv_pain_para', 'v_sv_money', 'v_sv_pain_para:sv_money']
        params_of_interest_s = ['z_subj', 'a_subj', 't_subj', 'v_Intercept_subj', 'v_sv_pain_para_subj', 'v_sv_money_subj', 'v_sv_pain_para:sv_money_subj']
        titles = ['Starting point', 'Boundary sep.', 'Non-dec. time', 
            'Inter-trial variability in drift rate', 'Inter-trial variability in starting point', 'Inter-trial variability in non-decision time',
            'v_Intercept rate', 'Drift v_sv_pain_para', 'Drift v_sv_money', 'Drift rate interaction']
    elif version == 10:
        params_of_interest = [
            'sv', 'sz', 'st',
            'v_Intercept', 'v_sv_pain_para', 'v_sv_money', 'v_sv_pain_para:sv_money',
            't_Intercept', 't_sv_pain_para', 't_sv_money', 't_sv_pain_para:sv_money',
            'a_Intercept', 'a_sv_pain_para', 'a_sv_money', 'a_sv_pain_para:sv_money',
            'z_Intercept', 'z_sv_pain_para', 'z_sv_money', 'z_sv_pain_para:sv_money']
        params_of_interest_s = [
            'sv_subj', 'sz_subj', 'st_subj',
            'v_Intercept_subj', 'v_sv_pain_para_subj', 'v_sv_money_subj', 'v_sv_pain_para:sv_money_subj',
            't_Intercept_subj', 't_sv_pain_para_subj', 't_sv_money_subj', 't_sv_pain_para:sv_money_subj',
            'a_Intercept_subj', 'a_sv_pain_para_subj', 'a_sv_money_subj', 'a_sv_pain_para:sv_money_subj',
            'z_Intercept_subj', 'z_sv_pain_para_subj', 'z_sv_money_subj', 'z_sv_pain_para:sv_money_subj']
        titles = [
            'Starting point', 'Inter-trial variability in drift rate', 'Inter-trial variability in starting point', 'Inter-trial variability in non-decision time',
            'Intercept drift rate', 'Drift rate sv_pain_para', 'Drift rate sv_money', 'Interaction v_sv_pain_para:sv_money_subj',
            'Intercept non-decision time', 'Non-decision time sv_pain_para', 'Non-decision time sv_money', 'Interaction t_sv_pain_para:sv_money_subj',
            'Intercept boundary separation', 'Boundary separation sv_pain_para', 'Boundary separation sv_money', 'Interaction a_sv_pain_para:sv_money_subj',
            'Intercept Starting Point', 'Starting Point sv_pain_para', 'Starting Point sv_money', 'Starting Point a_sv_pain_para:sv_money_subj']
    elif version == 11:
        params_of_interest = ['z', 'a', 't', 'sv', 'sz', 'st', 'v_Intercept', 'v_sv_pain_para', 'v_sv_money']
        params_of_interest_s = ['z_subj', 'a_subj', 't_subj', 'v_Intercept_subj', 'v_sv_pain_para_subj', 'v_sv_money_subj']
        titles = ['Starting point', 'Boundary sep.', 'Non-dec. time', 
            'Inter-trial variability in drift rate', 'Inter-trial variability in starting point', 'Inter-trial variability in non-decision time',
            'v_Intercept rate', 'Drift v_sv_pain_para',  'Drift v_sv_money']
    elif version == 12:
        params_of_interest = [
            'sv', 'sz', 'st', 
            'v_Intercept', 'v_sv_pain_para', 'v_sv_money', 
            't_Intercept', 't_sv_pain_para', 't_sv_money', 
            'a_Intercept', 'a_sv_pain_para', 'a_sv_money',
            'z_Intercept', 'z_sv_pain_para', 'z_sv_money']
        params_of_interest_s = [
            'sv_subj', 'sz_subj', 'st_subj',
            'v_Intercept_subj', 'v_sv_pain_para_subj', 'v_sv_money_subj',
            't_Intercept_subj', 't_sv_pain_para_subj', 't_sv_money_subj',
            'z_Intercept_subj', 'z_sv_pain_para_subj', 'z_sv_money_subj']
        titles = [
            'Starting point', 'Inter-trial variability in drift rate', 'Inter-trial variability in starting point', 'Inter-trial variability in non-decision time',
            'Intercept drift rate', 'Drift rate sv_pain_para', 'Drift rate sv_money'
            'Intercept non-decision time', 'Non-decision time sv_pain_para', 'Non-decision time sv_money',
            'Intercept boundary separation', 'Boundary separation sv_pain_para', 'Boundary separation sv_money',
            'Intercept Starting Point', 'Intercept Starting Point sv_pain_para', 'Intercept Starting Point sv_money'] 
    elif version == 13:
        params_of_interest = ['z', 'a', 't', 'sv', 'sz', 'st', 'v_Intercept', 'v_sv_both']
        params_of_interest_s = ['z_subj', 'a_subj', 't_subj', 'v_Intercept_subj', 'v_sv_both_subj']
        titles = ['Starting point', 'Boundary sep.', 'Non-dec. time', 
            'Inter-trial variability in drift rate', 'Inter-trial variability in starting point', 'Inter-trial variability in non-decision time', 
            'v_Intercept rate', 'Drift v_sv_money'] 
    elif version == 14:
        params_of_interest = ['z', 'sv', 'sz', 'st','v_Intercept', 'v_sv_both', 'v_sv_pain_para', 'v_sv_money', 'v_sv_both:sv_pain_para:sv_money']
        params_of_interest_s = ['z_subj', 'sv_subj', 'sz_subj', 'st_subj',
                            'v_Intercept_subj', 'v_sv_both_subj', 'v_sv_pain_para_subj', 'v_sv_money_subj', 'v_sv_both:sv_pain_para:sv_money_subj']
        titles = ['Starting point', 'Inter-trial variability in drift rate', 'Inter-trial variability in starting point', 'Inter-trial variability in non-decision time',
                'Intercept drift rate', 'Drift v_sv_both', 'Drift v_sv_pain_para', 'Drift v_sv_money', 'Drift rate (sv_both * sv_pain_para * sv_money)']
    elif version == 15:  
        params_of_interest = [
            'sv', 'sz', 'st',
            'v_Intercept', 'v_sv_both', 'v_sv_pain_para', 'v_sv_money', 'v_sv_both:sv_pain_para', 'v_sv_both:sv_money', 'v_sv_pain_para:sv_money', 'v_sv_both:sv_pain_para:sv_money',
            't_Intercept', 't_sv_both', 't_sv_pain_para', 't_sv_money', 't_sv_both:sv_pain_para', 't_sv_both:sv_money', 't_sv_pain_para:sv_money', 't_sv_both:sv_pain_para:sv_money',
            'a_Intercept', 'a_sv_both', 'a_sv_pain_para', 'a_sv_money', 'a_sv_both:sv_pain_para', 'a_sv_both:sv_money', 'a_sv_pain_para:sv_money', 'a_sv_both:sv_pain_para:sv_money',
            'z_Intercept', 'z_sv_both', 'z_sv_pain_para', 'z_sv_money', 'z_sv_both:sv_pain_para', 'z_sv_both:sv_money', 'z_sv_pain_para:sv_money', 'z_sv_both:sv_pain_para:sv_money']
        params_of_interest_s = [
            'sv_subj', 'sz_subj', 'st_subj',
            'v_Intercept_subj', 'v_sv_both_subj', 'v_sv_pain_para_subj', 'v_sv_money_subj', 'v_sv_both:sv_pain_para_subj', 'v_sv_both:sv_money_subj', 'v_sv_pain_para:sv_money_subj', 'v_sv_both:sv_pain_para:sv_money_subj',
            't_Intercept_subj', 't_sv_both_subj', 't_sv_pain_para_subj', 't_sv_money_subj', 't_sv_both:sv_pain_para_subj', 't_sv_both:sv_money_subj', 't_sv_pain_para:sv_money_subj', 't_sv_both:sv_pain_para:sv_money_subj',
            'a_Intercept_subj', 'a_sv_both_subj', 'a_sv_pain_para_subj', 'a_sv_money_subj', 'a_sv_both:sv_pain_para_subj', 'a_sv_both:sv_money_subj', 'a_sv_pain_para:sv_money_subj', 'a_sv_both:sv_pain_para:sv_money_subj',
            'z_Intercept_subj', 'z_sv_both_subj', 'z_sv_pain_para_subj', 'z_sv_money_subj', 'z_sv_both:sv_pain_para_subj', 'z_sv_both:sv_money_subj', 'z_sv_pain_para:sv_money_subj', 'z_sv_both:sv_pain_para:sv_money_subj']
        titles = [
            'Starting point', 'Boundary separation', 'Non-decision time', 'Inter-trial variability in drift rate', 'Inter-trial variability in starting point', 'Inter-trial variability in non-decision time',
            'Intercept drift rate', 'Drift v_sv_both', 'Drift v_sv_pain_para', 'Drift v_sv_money', 'Drift rate interaction (sv_both * sv_pain_para)', 'Drift rate interaction (sv_both * sv_money)', 'Drift rate interaction (sv_pain_para * sv_money)', 'Drift rate interaction (sv_both * sv_pain_para * sv_money)',
            'Intercept non-decision time', 'Non-decision time sv_both', 'Non-decision time sv_pain_para', 'Non-decision time sv_money', 'Non-decision time interaction (sv_both * sv_pain_para)', 'Non-decision time interaction (sv_both * sv_money)', 'Non-decision time interaction (sv_pain_para * sv_money)', 'Non-decision time interaction (sv_both * sv_pain_para * sv_money)',
            'Intercept boundary separation', 'Boundary separation sv_both', 'Boundary separation sv_pain_para', 'Boundary separation sv_money', 'Boundary separation interaction (sv_both * sv_pain_para)', 'Boundary separation interaction (sv_both * sv_money)', 'Boundary separation interaction (sv_pain_para * sv_money)', 'Boundary separation interaction (sv_both * sv_pain_para * sv_money)',
            'Intercept starting point bias', 'Starting point bias sv_both', 'Starting point bias sv_pain_para', 'Starting point bias sv_money', 'Starting point bias interaction (sv_both * sv_pain_para)', 'Starting point bias interaction (sv_both * sv_money)', 'Starting point bias interaction (sv_pain_para * sv_money)', 'Starting point bias interaction (sv_both * sv_pain_para * sv_money)']
    elif version == 16:
        params_of_interest = [
            'z', 'a', 't', 'sv', 'sz', 'st',
            'v_Intercept', 'v_sv_pain_para', 'v_C(Abs_value)[low_abs]', 'v_C(Abs_value)[high_abs]', 'v_C(Abs_value)[mid_abs]',
            'v_sv_pain_para:C(Abs_value)[low_abs]', 'v_sv_pain_para:C(Abs_value)[high_abs]', 'v_sv_pain_para:C(Abs_value)[mid_abs]']
        params_of_interest_s = [
            'z_subj', 'a_subj', 't_subj', 'sv_subj', 'sz_subj', 'st_subj',
            'v_Intercept_subj', 'v_sv_pain_para_subj', 'v_C(Abs_value)[low_abs]_subj', 'v_C(Abs_value)[high_abs]_subj', 'v_C(Abs_value)[mid_abs]_subj',
            'v_sv_pain_para:C(Abs_value)[low_abs]_subj', 'v_sv_pain_para:C(Abs_value)[high_abs]_subj', 'v_sv_pain_para:C(Abs_value)[mid_abs]_subj']
        titles = [
            'Starting point', 'Boundary separation', 'Non-decision time', 'Inter-trial variability in drift rate',
            'Inter-trial variability in starting point', 'Inter-trial variability in non-decision time',
            'Intercept drift rate', 'Drift rate sv_pain_para',
            'Effect of Abs_value[low_abs]', 'Effect of Abs_value[high_abs]', 'Effect of Abs_value[mid_abs]',
            'Interaction: sv_pain_para * Abs_value[low_abs]', 'Interaction: sv_pain_para * Abs_value[high_abs]', 'Interaction: sv_pain_para * Abs_value[mid_abs]']
    elif version == 17:
        params_of_interest = [
            'z', 'a', 't', 'sv', 'sz', 'st',
            'v_Intercept', 'v_sv_pain_para', 'v_C(OV_value)[low_OV]', 'v_C(OV_value)[high_OV]',
            'v_sv_pain_para:C(OV_value)[low_OV]', 'v_sv_pain_para:C(OV_value)[high_OV]']
        params_of_interest_s = [
            'z_subj', 'a_subj', 't_subj', 'sv_subj', 'sz_subj', 'st_subj',
            'v_Intercept_subj', 'v_sv_pain_para_subj', 'v_C(OV_value)[low_OV]_subj', 'v_C(OV_value)[high_OV]_subj',
            'v_sv_pain_para:C(OV_value)[low_OV]_subj', 'v_sv_pain_para:C(OV_value)[high_OV]_subj']
        titles = [
            'Starting point', 'Boundary separation', 'Non-decision time', 'Inter-trial variability in drift rate',
            'Inter-trial variability in starting point', 'Inter-trial variability in non-decision time',
            'Intercept drift rate', 'Drift rate sv_pain_para',
            'Effect of OV_value[low_OV]', 'Effect of OV_value[high_OV]',
            'Interaction: sv_pain_para * OV_value[low_OV]', 'Interaction: sv_pain_para * OV_value[high_OV]']
    elif version == 18:
        params_of_interest = [
            'z', 'a', 't', 'sv', 'sz', 'st',
            'v_Intercept', 'v_sv_pain_para', 'v_C(Abs_Money_Pain)[low_abs_h_money]', 'v_C(Abs_Money_Pain)[low_abs_h_pain]',
            'v_C(Abs_Money_Pain)[high_abs_h_money]', 'v_C(Abs_Money_Pain)[high_abs_h_pain]', 'v_C(Abs_Money_Pain)[mid_abs]',
            'v_sv_pain_para:C(Abs_Money_Pain)[low_abs_h_money]', 'v_sv_pain_para:C(Abs_Money_Pain)[low_abs_h_pain]',
            'v_sv_pain_para:C(Abs_Money_Pain)[high_abs_h_money]', 'v_sv_pain_para:C(Abs_Money_Pain)[high_abs_h_pain]', 'v_sv_pain_para:C(Abs_Money_Pain)[mid_abs]']
        params_of_interest_s = [
            'z_subj', 'a_subj', 't_subj', 'sv_subj', 'sz_subj', 'st_subj',
            'v_Intercept_subj', 'v_sv_pain_para_subj', 'v_C(Abs_Money_Pain)[low_abs_h_money]_subj', 'v_C(Abs_Money_Pain)[low_abs_h_pain]_subj',
            'v_C(Abs_Money_Pain)[high_abs_h_money]_subj', 'v_C(Abs_Money_Pain)[high_abs_h_pain]_subj', 'v_C(Abs_Money_Pain)[mid_abs]_subj',
            'v_sv_pain_para:C(Abs_Money_Pain)[low_abs_h_money]_subj', 'v_sv_pain_para:C(Abs_Money_Pain)[low_abs_h_pain]_subj',
            'v_sv_pain_para:C(Abs_Money_Pain)[high_abs_h_money]_subj', 'v_sv_pain_para:C(Abs_Money_Pain)[high_abs_h_pain]_subj', 'v_sv_pain_para:C(Abs_Money_Pain)[mid_abs]_subj']
        titles = [
            'Starting point', 'Boundary separation', 'Non-decision time', 'Inter-trial variability in drift rate',
            'Inter-trial variability in starting point', 'Inter-trial variability in non-decision time',
            'Intercept drift rate', 'Drift rate sv_pain_para',
            'Effect of Abs_Money_Pain[low_abs_h_money]', 'Effect of Abs_Money_Pain[low_abs_h_pain]',
            'Effect of Abs_Money_Pain[high_abs_h_money]', 'Effect of Abs_Money_Pain[high_abs_h_pain]', 'Effect of Abs_Money_Pain[mid_abs]',
            'Interaction: sv_pain_para * Abs_Money_Pain[low_abs_h_money]', 'Interaction: sv_pain_para * Abs_Money_Pain[low_abs_h_pain]',
            'Interaction: sv_pain_para * Abs_Money_Pain[high_abs_h_money]', 'Interaction: sv_pain_para * Abs_Money_Pain[high_abs_h_pain]', 'Interaction: sv_pain_para * Abs_Money_Pain[mid_abs]']
    elif version == 19:
        params_of_interest = [
            'z', 'a', 't', 'sv', 'sz', 'st',
            'v_Intercept', 'v_sv_pain_para', 'v_C(OV_Money_Pain)[h_OV_h_money]', 'v_C(OV_Money_Pain)[h_OV_h_pain]',
            'v_C(OV_Money_Pain)[low_OV_h_money]', 'v_C(OV_Money_Pain)[low_OV_h_pain]',
            'v_sv_pain_para:C(OV_Money_Pain)[h_OV_h_money]', 'v_sv_pain_para:C(OV_Money_Pain)[h_OV_h_pain]',
            'v_sv_pain_para:C(OV_Money_Pain)[low_OV_h_money]', 'v_sv_pain_para:C(OV_Money_Pain)[low_OV_h_pain]']
        params_of_interest_s = [
            'z_subj', 'a_subj', 't_subj', 'sv_subj', 'sz_subj', 'st_subj',
            'v_Intercept_subj', 'v_sv_pain_para_subj', 'v_C(OV_Money_Pain)[h_OV_h_money]_subj', 'v_C(OV_Money_Pain)[h_OV_h_pain]_subj',
            'v_C(OV_Money_Pain)[low_OV_h_money]_subj', 'v_C(OV_Money_Pain)[low_OV_h_pain]_subj',
            'v_sv_pain_para:C(OV_Money_Pain)[h_OV_h_money]_subj', 'v_sv_pain_para:C(OV_Money_Pain)[h_OV_h_pain]_subj',
            'v_sv_pain_para:C(OV_Money_Pain)[low_OV_h_money]_subj', 'v_sv_pain_para:C(OV_Money_Pain)[low_OV_h_pain]_subj'
            ]
        titles = [
        'Starting point', 'Boundary separation', 'Non-decision time', 'Inter-trial variability in drift rate',
        'Inter-trial variability in starting point', 'Inter-trial variability in non-decision time',
        'Intercept drift rate', 'Drift rate sv_pain_para',
        'Effect of OV_Money_Pain[h_OV_h_money]', 'Effect of OV_Money_Pain[h_OV_h_pain]',
        'Effect of OV_Money_Pain[low_OV_h_money]', 'Effect of OV_Money_Pain[low_OV_h_pain]',
        'Interaction: sv_pain_para * OV_Money_Pain[h_OV_h_money]', 'Interaction: sv_pain_para * OV_Money_Pain[h_OV_h_pain]',
        'Interaction: sv_pain_para * OV_Money_Pain[low_OV_h_money]', 'Interaction: sv_pain_para * OV_Money_Pain[low_OV_h_pain]'
    ]
    elif version == 20:
        params_of_interest = [
            'z', 'a', 't', 'sv', 'sz', 'st',
            'v_C(OV_Money_Pain)[h_OV_h_money]', 'v_C(OV_Money_Pain)[h_OV_h_pain]',
            'v_C(OV_Money_Pain)[low_OV_h_money]', 'v_C(OV_Money_Pain)[low_OV_h_pain]',
            'v_C(Abs_Money_Pain)[low_abs_h_money]', 'v_C(Abs_Money_Pain)[low_abs_h_pain]',
            'v_C(Abs_Money_Pain)[high_abs_h_money]', 'v_C(Abs_Money_Pain)[high_abs_h_pain]', 'v_C(Abs_Money_Pain)[mid_abs]',
            'v_sv_pain_para:C(OV_Money_Pain)[h_OV_h_money]:C(Abs_Money_Pain)[low_abs_h_money]',
            'v_sv_pain_para:C(OV_Money_Pain)[h_OV_h_money]:C(Abs_Money_Pain)[low_abs_h_pain]',
            'v_sv_pain_para:C(OV_Money_Pain)[h_OV_h_money]:C(Abs_Money_Pain)[high_abs_h_money]',
            'v_sv_pain_para:C(OV_Money_Pain)[h_OV_h_money]:C(Abs_Money_Pain)[high_abs_h_pain]',
            'v_sv_pain_para:C(OV_Money_Pain)[h_OV_h_money]:C(Abs_Money_Pain)[mid_abs]',
            'v_sv_pain_para:C(OV_Money_Pain)[h_OV_h_pain]:C(Abs_Money_Pain)[low_abs_h_money]',
            'v_sv_pain_para:C(OV_Money_Pain)[h_OV_h_pain]:C(Abs_Money_Pain)[low_abs_h_pain]',
            'v_sv_pain_para:C(OV_Money_Pain)[h_OV_h_pain]:C(Abs_Money_Pain)[high_abs_h_money]',
            'v_sv_pain_para:C(OV_Money_Pain)[h_OV_h_pain]:C(Abs_Money_Pain)[high_abs_h_pain]',
            'v_sv_pain_para:C(OV_Money_Pain)[h_OV_h_pain]:C(Abs_Money_Pain)[mid_abs]',
            'v_sv_pain_para:C(OV_Money_Pain)[low_OV_h_money]:C(Abs_Money_Pain)[low_abs_h_money]',
            'v_sv_pain_para:C(OV_Money_Pain)[low_OV_h_money]:C(Abs_Money_Pain)[low_abs_h_pain]',
            'v_sv_pain_para:C(OV_Money_Pain)[low_OV_h_money]:C(Abs_Money_Pain)[high_abs_h_money]',
            'v_sv_pain_para:C(OV_Money_Pain)[low_OV_h_money]:C(Abs_Money_Pain)[high_abs_h_pain]',
            'v_sv_pain_para:C(OV_Money_Pain)[low_OV_h_money]:C(Abs_Money_Pain)[mid_abs]',
            'v_sv_pain_para:C(OV_Money_Pain)[low_OV_h_pain]:C(Abs_Money_Pain)[low_abs_h_money]',
            'v_sv_pain_para:C(OV_Money_Pain)[low_OV_h_pain]:C(Abs_Money_Pain)[low_abs_h_pain]',
            'v_sv_pain_para:C(OV_Money_Pain)[low_OV_h_pain]:C(Abs_Money_Pain)[high_abs_h_money]',
            'v_sv_pain_para:C(OV_Money_Pain)[low_OV_h_pain]:C(Abs_Money_Pain)[high_abs_h_pain]',
            'v_sv_pain_para:C(OV_Money_Pain)[low_OV_h_pain]:C(Abs_Money_Pain)[mid_abs]'
        ]
        params_of_interest_s = [
            'z_subj', 'a_subj', 't_subj', 'sv_subj', 'sz_subj', 'st_subj',
            'v_C(OV_Money_Pain)[h_OV_h_money]_subj', 'v_C(OV_Money_Pain)[h_OV_h_pain]_subj',
            'v_C(OV_Money_Pain)[low_OV_h_money]_subj', 'v_C(OV_Money_Pain)[low_OV_h_pain]_subj',
            'v_C(Abs_Money_Pain)[low_abs_h_money]_subj', 'v_C(Abs_Money_Pain)[low_abs_h_pain]_subj',
            'v_C(Abs_Money_Pain)[high_abs_h_money]_subj', 'v_C(Abs_Money_Pain)[high_abs_h_pain]_subj', 'v_C(Abs_Money_Pain)[mid_abs]_subj',
            'v_sv_pain_para:C(OV_Money_Pain)[h_OV_h_money]:C(Abs_Money_Pain)[low_abs_h_money]_subj',
            'v_sv_pain_para:C(OV_Money_Pain)[h_OV_h_money]:C(Abs_Money_Pain)[low_abs_h_pain]_subj',
            'v_sv_pain_para:C(OV_Money_Pain)[h_OV_h_money]:C(Abs_Money_Pain)[high_abs_h_money]_subj',
            'v_sv_pain_para:C(OV_Money_Pain)[h_OV_h_money]:C(Abs_Money_Pain)[high_abs_h_pain]_subj',
            'v_sv_pain_para:C(OV_Money_Pain)[h_OV_h_money]:C(Abs_Money_Pain)[mid_abs]_subj',
            'v_sv_pain_para:C(OV_Money_Pain)[h_OV_h_pain]:C(Abs_Money_Pain)[low_abs_h_money]_subj',
            'v_sv_pain_para:C(OV_Money_Pain)[h_OV_h_pain]:C(Abs_Money_Pain)[low_abs_h_pain]_subj',
            'v_sv_pain_para:C(OV_Money_Pain)[h_OV_h_pain]:C(Abs_Money_Pain)[high_abs_h_money]_subj',
            'v_sv_pain_para:C(OV_Money_Pain)[h_OV_h_pain]:C(Abs_Money_Pain)[high_abs_h_pain]_subj',
            'v_sv_pain_para:C(OV_Money_Pain)[h_OV_h_pain]:C(Abs_Money_Pain)[mid_abs]_subj',
            'v_sv_pain_para:C(OV_Money_Pain)[low_OV_h_money]:C(Abs_Money_Pain)[low_abs_h_money]_subj',
            'v_sv_pain_para:C(OV_Money_Pain)[low_OV_h_money]:C(Abs_Money_Pain)[low_abs_h_pain]_subj',
            'v_sv_pain_para:C(OV_Money_Pain)[low_OV_h_money]:C(Abs_Money_Pain)[high_abs_h_money]_subj',
            'v_sv_pain_para:C(OV_Money_Pain)[low_OV_h_money]:C(Abs_Money_Pain)[high_abs_h_pain]_subj',
            'v_sv_pain_para:C(OV_Money_Pain)[low_OV_h_money]:C(Abs_Money_Pain)[mid_abs]_subj',
            'v_sv_pain_para:C(OV_Money_Pain)[low_OV_h_pain]:C(Abs_Money_Pain)[low_abs_h_money]_subj',
            'v_sv_pain_para:C(OV_Money_Pain)[low_OV_h_pain]:C(Abs_Money_Pain)[low_abs_h_pain]_subj',
            'v_sv_pain_para:C(OV_Money_Pain)[low_OV_h_pain]:C(Abs_Money_Pain)[high_abs_h_money]_subj',
            'v_sv_pain_para:C(OV_Money_Pain)[low_OV_h_pain]:C(Abs_Money_Pain)[high_abs_h_pain]_subj',
            'v_sv_pain_para:C(OV_Money_Pain)[low_OV_h_pain]:C(Abs_Money_Pain)[mid_abs]_subj'
        ]
        titles = [
        'Starting point', 'Boundary separation', 'Non-decision time', 'Inter-trial variability in drift rate',
        'Inter-trial variability in starting point', 'Inter-trial variability in non-decision time',
        'OV_Money_Pain[h_OV_h_money]', 'OV_Money_Pain[h_OV_h_pain]',
        'OV_Money_Pain[low_OV_h_money]', 'OV_Money_Pain[low_OV_h_pain]',
        'Abs_Money_Pain[low_abs_h_money]', 'Abs_Money_Pain[low_abs_h_pain]',
        'Abs_Money_Pain[high_abs_h_money]', 'Abs_Money_Pain[high_abs_h_pain]', 'Abs_Money_Pain[mid_abs]',
        'Interaction: sv_pain_para * OV_Money_Pain[h_OV_h_money] * Abs_Money_Pain[low_abs_h_money]',
        'Interaction: sv_pain_para * OV_Money_Pain[h_OV_h_money] * Abs_Money_Pain[low_abs_h_pain]',
        'Interaction: sv_pain_para * OV_Money_Pain[h_OV_h_money] * Abs_Money_Pain[high_abs_h_money]',
        'Interaction: sv_pain_para * OV_Money_Pain[h_OV_h_money] * Abs_Money_Pain[high_abs_h_pain]',
        'Interaction: sv_pain_para * OV_Money_Pain[h_OV_h_money] * Abs_Money_Pain[mid_abs]',
        'Interaction: sv_pain_para * OV_Money_Pain[h_OV_h_pain] * Abs_Money_Pain[low_abs_h_money]',
        'Interaction: sv_pain_para * OV_Money_Pain[h_OV_h_pain] * Abs_Money_Pain[low_abs_h_pain]',
        'Interaction: sv_pain_para * OV_Money_Pain[h_OV_h_pain] * Abs_Money_Pain[high_abs_h_money]',
        'Interaction: sv_pain_para * OV_Money_Pain[h_OV_h_pain] * Abs_Money_Pain[high_abs_h_pain]',
        'Interaction: sv_pain_para * OV_Money_Pain[h_OV_h_pain] * Abs_Money_Pain[mid_abs]',
        'Interaction: sv_pain_para * OV_Money_Pain[low_OV_h_money] * Abs_Money_Pain[low_abs_h_money]',
        'Interaction: sv_pain_para * OV_Money_Pain[low_OV_h_money] * Abs_Money_Pain[low_abs_h_pain]',
        'Interaction: sv_pain_para * OV_Money_Pain[low_OV_h_money] * Abs_Money_Pain[high_abs_h_money]',
        'Interaction: sv_pain_para * OV_Money_Pain[low_OV_h_money] * Abs_Money_Pain[high_abs_h_pain]',
        'Interaction: sv_pain_para * OV_Money_Pain[low_OV_h_money] * Abs_Money_Pain[mid_abs]',
        'Interaction: sv_pain_para * OV_Money_Pain[low_OV_h_pain] * Abs_Money_Pain[low_abs_h_money]',
        'Interaction: sv_pain_para * OV_Money_Pain[low_OV_h_pain] * Abs_Money_Pain[low_abs_h_pain]',
        'Interaction: sv_pain_para * OV_Money_Pain[low_OV_h_pain] * Abs_Money_Pain[high_abs_h_money]',
        'Interaction: sv_pain_para * OV_Money_Pain[low_OV_h_pain] * Abs_Money_Pain[high_abs_h_pain]',
        'Interaction: sv_pain_para * OV_Money_Pain[low_OV_h_pain] * Abs_Money_Pain[mid_abs]'
        ]
    elif version == 21:
        params_of_interest = [
            'z', 'a', 't', 'sv', 'sz', 'st',
            'v_C(Abs_value)[low_abs]', 'v_C(Abs_value)[mid_abs]', 'v_C(Abs_value)[high_abs]',
            'v_C(acceptance_pair)[P]', 'v_C(acceptance_pair)[M]', 'v_C(acceptance_pair)[I]',
            'v_sv_pain_para:C(Abs_value)[low_abs]:C(acceptance_pair)[P]',
            'v_sv_pain_para:C(Abs_value)[low_abs]:C(acceptance_pair)[M]',
            'v_sv_pain_para:C(Abs_value)[low_abs]:C(acceptance_pair)[I]',
            'v_sv_pain_para:C(Abs_value)[mid_abs]:C(acceptance_pair)[P]',
            'v_sv_pain_para:C(Abs_value)[mid_abs]:C(acceptance_pair)[M]',
            'v_sv_pain_para:C(Abs_value)[mid_abs]:C(acceptance_pair)[I]',
            'v_sv_pain_para:C(Abs_value)[high_abs]:C(acceptance_pair)[P]',
            'v_sv_pain_para:C(Abs_value)[high_abs]:C(acceptance_pair)[M]',
            'v_sv_pain_para:C(Abs_value)[high_abs]:C(acceptance_pair)[I]']
        params_of_interest_s = [
            'z_subj', 'a_subj', 't_subj', 'sv_subj', 'sz_subj', 'st_subj',
            'v_C(Abs_value)[low_abs]_subj', 'v_C(Abs_value)[mid_abs]_subj', 'v_C(Abs_value)[high_abs]_subj',
            'v_C(acceptance_pair)[P]_subj', 'v_C(acceptance_pair)[M]_subj', 'v_C(acceptance_pair)[I]_subj',
            'v_sv_pain_para:C(Abs_value)[low_abs]:C(acceptance_pair)[P]_subj',
            'v_sv_pain_para:C(Abs_value)[low_abs]:C(acceptance_pair)[M]_subj',
            'v_sv_pain_para:C(Abs_value)[low_abs]:C(acceptance_pair)[I]_subj',
            'v_sv_pain_para:C(Abs_value)[mid_abs]:C(acceptance_pair)[P]_subj',
            'v_sv_pain_para:C(Abs_value)[mid_abs]:C(acceptance_pair)[M]_subj',
            'v_sv_pain_para:C(Abs_value)[mid_abs]:C(acceptance_pair)[I]_subj',
            'v_sv_pain_para:C(Abs_value)[high_abs]:C(acceptance_pair)[P]_subj',
            'v_sv_pain_para:C(Abs_value)[high_abs]:C(acceptance_pair)[M]_subj',
            'v_sv_pain_para:C(Abs_value)[high_abs]:C(acceptance_pair)[I]_subj']
        titles = [
            'Starting point', 'Boundary separation', 'Non-decision time', 'Inter-trial variability in drift rate',
            'Inter-trial variability in starting point', 'Inter-trial variability in non-decision time',
            'Effect of Abs_value[low_abs]', 'Effect of Abs_value[mid_abs]', 'Effect of Abs_value[high_abs]',
            'Effect of acceptance_pair[P]', 'Effect of acceptance_pair[M]', 'Effect of acceptance_pair[I]',
            'Interaction: sv_pain_para * Abs_value[low_abs] * acceptance_pair[P]',
            'Interaction: sv_pain_para * Abs_value[low_abs] * acceptance_pair[M]',
            'Interaction: sv_pain_para * Abs_value[low_abs] * acceptance_pair[I]',
            'Interaction: sv_pain_para * Abs_value[mid_abs] * acceptance_pair[P]',
            'Interaction: sv_pain_para * Abs_value[mid_abs] * acceptance_pair[M]',
            'Interaction: sv_pain_para * Abs_value[mid_abs] * acceptance_pair[I]',
            'Interaction: sv_pain_para * Abs_value[high_abs] * acceptance_pair[P]',
            'Interaction: sv_pain_para * Abs_value[high_abs] * acceptance_pair[M]',
            'Interaction: sv_pain_para * Abs_value[high_abs] * acceptance_pair[I]']
    elif version == 22:
        params_of_interest = [
        'z', 'a', 't', 'sv', 'sz', 'st',
        'v_C(OV_value)[low_OV]', 'v_C(OV_value)[high_OV]',
        'v_C(acceptance_pair)[P]', 'v_C(acceptance_pair)[M]', 'v_C(acceptance_pair)[I]',
        'v_sv_pain_para:C(OV_value)[low_OV]:C(acceptance_pair)[P]',
        'v_sv_pain_para:C(OV_value)[low_OV]:C(acceptance_pair)[M]',
        'v_sv_pain_para:C(OV_value)[low_OV]:C(acceptance_pair)[I]',
        'v_sv_pain_para:C(OV_value)[high_OV]:C(acceptance_pair)[P]',
        'v_sv_pain_para:C(OV_value)[high_OV]:C(acceptance_pair)[M]',
        'v_sv_pain_para:C(OV_value)[high_OV]:C(acceptance_pair)[I]'
        ]
        params_of_interest_s = [
        'z_subj', 'a_subj', 't_subj', 'sv_subj', 'sz_subj', 'st_subj',
        'v_C(OV_value)[low_OV]_subj', 'v_C(OV_value)[high_OV]_subj',
        'v_C(acceptance_pair)[P]_subj', 'v_C(acceptance_pair)[M]_subj', 'v_C(acceptance_pair)[I]_subj',
        'v_sv_pain_para:C(OV_value)[low_OV]:C(acceptance_pair)[P]_subj',
        'v_sv_pain_para:C(OV_value)[low_OV]:C(acceptance_pair)[M]_subj',
        'v_sv_pain_para:C(OV_value)[low_OV]:C(acceptance_pair)[I]_subj',
        'v_sv_pain_para:C(OV_value)[high_OV]:C(acceptance_pair)[P]_subj',
        'v_sv_pain_para:C(OV_value)[high_OV]:C(acceptance_pair)[M]_subj',
        'v_sv_pain_para:C(OV_value)[high_OV]:C(acceptance_pair)[I]_subj'
        ]
        titles = [
        'Starting point', 'Boundary separation', 'Non-decision time', 'Inter-trial variability in drift rate',
        'Inter-trial variability in starting point', 'Inter-trial variability in non-decision time',
        'Effect of OV_value[low_OV]', 'Effect of OV_value[high_OV]',
        'Effect of acceptance_pair[P]', 'Effect of acceptance_pair[M]', 'Effect of acceptance_pair[I]',
        'Interaction: sv_pain_para * OV_value[low_OV] * acceptance_pair[P]',
        'Interaction: sv_pain_para * OV_value[low_OV] * acceptance_pair[M]',
        'Interaction: sv_pain_para * OV_value[low_OV] * acceptance_pair[I]',
        'Interaction: sv_pain_para * OV_value[high_OV] * acceptance_pair[P]',
        'Interaction: sv_pain_para * OV_value[high_OV] * acceptance_pair[M]',
        'Interaction: sv_pain_para * OV_value[high_OV] * acceptance_pair[I]'
        ]
    elif version == 23:
        params_of_interest = [
        'z', 'a', 't', 'sv', 'sz', 'st',
        'v_C(OV_value)[low_OV]', 'v_C(OV_value)[high_OV]',
        'v_C(Abs_value)[low_abs]', 'v_C(Abs_value)[mid_abs]', 'v_C(Abs_value)[high_abs]',
        'v_sv_pain_para:C(OV_value)[low_OV]:C(Abs_value)[low_abs]',
        'v_sv_pain_para:C(OV_value)[low_OV]:C(Abs_value)[mid_abs]',
        'v_sv_pain_para:C(OV_value)[low_OV]:C(Abs_value)[high_abs]',
        'v_sv_pain_para:C(OV_value)[high_OV]:C(Abs_value)[low_abs]',
        'v_sv_pain_para:C(OV_value)[high_OV]:C(Abs_value)[mid_abs]',
        'v_sv_pain_para:C(OV_value)[high_OV]:C(Abs_value)[high_abs]'
        ]
        params_of_interest_s = [
        'z_subj', 'a_subj', 't_subj', 'sv_subj', 'sz_subj', 'st_subj',
        'v_C(OV_value)[low_OV]_subj', 'v_C(OV_value)[high_OV]_subj',
        'v_C(Abs_value)[low_abs]_subj', 'v_C(Abs_value)[mid_abs]_subj', 'v_C(Abs_value)[high_abs]_subj',
        'v_sv_pain_para:C(OV_value)[low_OV]:C(Abs_value)[low_abs]_subj',
        'v_sv_pain_para:C(OV_value)[low_OV]:C(Abs_value)[mid_abs]_subj',
        'v_sv_pain_para:C(OV_value)[low_OV]:C(Abs_value)[high_abs]_subj',
        'v_sv_pain_para:C(OV_value)[high_OV]:C(Abs_value)[low_abs]_subj',
        'v_sv_pain_para:C(OV_value)[high_OV]:C(Abs_value)[mid_abs]_subj',
        'v_sv_pain_para:C(OV_value)[high_OV]:C(Abs_value)[high_abs]_subj'
        ]
        titles = [
        'Starting point', 'Boundary separation', 'Non-decision time', 'Inter-trial variability in drift rate',
        'Inter-trial variability in starting point', 'Inter-trial variability in non-decision time',
        'Effect of OV_value[low_OV]', 'Effect of OV_value[high_OV]',
        'Effect of Abs_value[low_abs]', 'Effect of Abs_value[mid_abs]', 'Effect of Abs_value[high_abs]',
        'Interaction: sv_pain_para * OV_value[low_OV] * Abs_value[low_abs]',
        'Interaction: sv_pain_para * OV_value[low_OV] * Abs_value[mid_abs]',
        'Interaction: sv_pain_para * OV_value[low_OV] * Abs_value[high_abs]',
        'Interaction: sv_pain_para * OV_value[high_OV] * Abs_value[low_abs]',
        'Interaction: sv_pain_para * OV_value[high_OV] * Abs_value[mid_abs]',
        'Interaction: sv_pain_para * OV_value[high_OV] * Abs_value[high_abs]'
        ]
    elif version == 24:
        params_of_interest = [
        'z', 'a', 't', 'sv', 'sz', 'st',
        'v_C(OV_value)[low_OV]', 'v_C(OV_value)[high_OV]',
        'v_C(Abs_value)[low_abs]', 'v_C(Abs_value)[mid_abs]', 'v_C(Abs_value)[high_abs]',
        'v_C(acceptance_pair)[P]', 'v_C(acceptance_pair)[M]', 'v_C(acceptance_pair)[I]',
        'v_sv_pain_para:C(OV_value)[low_OV]:C(Abs_value)[low_abs]:C(acceptance_pair)[P]',
        'v_sv_pain_para:C(OV_value)[low_OV]:C(Abs_value)[low_abs]:C(acceptance_pair)[M]',
        'v_sv_pain_para:C(OV_value)[low_OV]:C(Abs_value)[low_abs]:C(acceptance_pair)[I]',
        'v_sv_pain_para:C(OV_value)[low_OV]:C(Abs_value)[mid_abs]:C(acceptance_pair)[P]',
        'v_sv_pain_para:C(OV_value)[low_OV]:C(Abs_value)[mid_abs]:C(acceptance_pair)[M]',
        'v_sv_pain_para:C(OV_value)[low_OV]:C(Abs_value)[mid_abs]:C(acceptance_pair)[I]',
        'v_sv_pain_para:C(OV_value)[low_OV]:C(Abs_value)[high_abs]:C(acceptance_pair)[P]',
        'v_sv_pain_para:C(OV_value)[low_OV]:C(Abs_value)[high_abs]:C(acceptance_pair)[M]',
        'v_sv_pain_para:C(OV_value)[low_OV]:C(Abs_value)[high_abs]:C(acceptance_pair)[I]',
        'v_sv_pain_para:C(OV_value)[high_OV]:C(Abs_value)[low_abs]:C(acceptance_pair)[P]',
        'v_sv_pain_para:C(OV_value)[high_OV]:C(Abs_value)[low_abs]:C(acceptance_pair)[M]',
        'v_sv_pain_para:C(OV_value)[high_OV]:C(Abs_value)[low_abs]:C(acceptance_pair)[I]',
        'v_sv_pain_para:C(OV_value)[high_OV]:C(Abs_value)[mid_abs]:C(acceptance_pair)[P]',
        'v_sv_pain_para:C(OV_value)[high_OV]:C(Abs_value)[mid_abs]:C(acceptance_pair)[M]',
        'v_sv_pain_para:C(OV_value)[high_OV]:C(Abs_value)[mid_abs]:C(acceptance_pair)[I]',
        'v_sv_pain_para:C(OV_value)[high_OV]:C(Abs_value)[high_abs]:C(acceptance_pair)[P]',
        'v_sv_pain_para:C(OV_value)[high_OV]:C(Abs_value)[high_abs]:C(acceptance_pair)[M]',
        'v_sv_pain_para:C(OV_value)[high_OV]:C(Abs_value)[high_abs]:C(acceptance_pair)[I]'
        ]
        params_of_interest_s = [
        'z_subj', 'a_subj', 't_subj', 'sv_subj', 'sz_subj', 'st_subj',
        'v_C(OV_value)[low_OV]_subj', 'v_C(OV_value)[high_OV]_subj',
        'v_C(Abs_value)[low_abs]_subj', 'v_C(Abs_value)[mid_abs]_subj', 'v_C(Abs_value)[high_abs]_subj',
        'v_C(acceptance_pair)[P]_subj', 'v_C(acceptance_pair)[M]_subj', 'v_C(acceptance_pair)[I]_subj',
        'v_sv_pain_para:C(OV_value)[low_OV]:C(Abs_value)[low_abs]:C(acceptance_pair)[P]_subj',
        'v_sv_pain_para:C(OV_value)[low_OV]:C(Abs_value)[low_abs]:C(acceptance_pair)[M]_subj',
        'v_sv_pain_para:C(OV_value)[low_OV]:C(Abs_value)[low_abs]:C(acceptance_pair)[I]_subj',
        'v_sv_pain_para:C(OV_value)[low_OV]:C(Abs_value)[mid_abs]:C(acceptance_pair)[P]_subj',
        'v_sv_pain_para:C(OV_value)[low_OV]:C(Abs_value)[mid_abs]:C(acceptance_pair)[M]_subj',
        'v_sv_pain_para:C(OV_value)[low_OV]:C(Abs_value)[mid_abs]:C(acceptance_pair)[I]_subj',
        'v_sv_pain_para:C(OV_value)[low_OV]:C(Abs_value)[high_abs]:C(acceptance_pair)[P]_subj',
        'v_sv_pain_para:C(OV_value)[low_OV]:C(Abs_value)[high_abs]:C(acceptance_pair)[M]_subj',
        'v_sv_pain_para:C(OV_value)[low_OV]:C(Abs_value)[high_abs]:C(acceptance_pair)[I]_subj',
        'v_sv_pain_para:C(OV_value)[high_OV]:C(Abs_value)[low_abs]:C(acceptance_pair)[P]_subj',
        'v_sv_pain_para:C(OV_value)[high_OV]:C(Abs_value)[low_abs]:C(acceptance_pair)[M]_subj',
        'v_sv_pain_para:C(OV_value)[high_OV]:C(Abs_value)[low_abs]:C(acceptance_pair)[I]_subj',
        'v_sv_pain_para:C(OV_value)[high_OV]:C(Abs_value)[mid_abs]:C(acceptance_pair)[P]_subj',
        'v_sv_pain_para:C(OV_value)[high_OV]:C(Abs_value)[mid_abs]:C(acceptance_pair)[M]_subj',
        'v_sv_pain_para:C(OV_value)[high_OV]:C(Abs_value)[mid_abs]:C(acceptance_pair)[I]_subj',
        'v_sv_pain_para:C(OV_value)[high_OV]:C(Abs_value)[high_abs]:C(acceptance_pair)[P]_subj',
        'v_sv_pain_para:C(OV_value)[high_OV]:C(Abs_value)[high_abs]:C(acceptance_pair)[M]_subj',
        'v_sv_pain_para:C(OV_value)[high_OV]:C(Abs_value)[high_abs]:C(acceptance_pair)[I]_subj'
        ]
        titles = [
        'Starting point', 'Boundary separation', 'Non-decision time', 'Inter-trial variability in drift rate',
        'Inter-trial variability in starting point', 'Inter-trial variability in non-decision time',
        'Effect of OV_value[low_OV]', 'Effect of OV_value[high_OV]',
        'Effect of Abs_value[low_abs]', 'Effect of Abs_value[mid_abs]', 'Effect of Abs_value[high_abs]',
        'Effect of acceptance_pair[P]', 'Effect of acceptance_pair[M]', 'Effect of acceptance_pair[I]',
        'sv_pain_para * OV_value[low_OV] * Abs_value[low_abs] * acceptance_pair[P]',
        'sv_pain_para * OV_value[low_OV] * Abs_value[low_abs] * acceptance_pair[M]',
        'sv_pain_para * OV_value[low_OV] * Abs_value[low_abs] * acceptance_pair[I]',
        'sv_pain_para * OV_value[low_OV] * Abs_value[mid_abs] * acceptance_pair[P]',
        'sv_pain_para * OV_value[low_OV] * Abs_value[mid_abs] * acceptance_pair[M]',
        'sv_pain_para * OV_value[low_OV] * Abs_value[mid_abs] * acceptance_pair[I]',
        'sv_pain_para * OV_value[low_OV] * Abs_value[high_abs] * acceptance_pair[P]',
        'sv_pain_para * OV_value[low_OV] * Abs_value[high_abs] * acceptance_pair[M]',
        'sv_pain_para * OV_value[low_OV] * Abs_value[high_abs] * acceptance_pair[I]',
        'sv_pain_para * OV_value[high_OV] * Abs_value[low_abs] * acceptance_pair[P]',
        'sv_pain_para * OV_value[high_OV] * Abs_value[low_abs] * acceptance_pair[M]',
        'sv_pain_para * OV_value[high_OV] * Abs_value[low_abs] * acceptance_pair[I]',
        'sv_pain_para * OV_value[high_OV] * Abs_value[mid_abs] * acceptance_pair[P]',
        'sv_pain_para * OV_value[high_OV] * Abs_value[mid_abs] * acceptance_pair[M]',
        'sv_pain_para * OV_value[high_OV] * Abs_value[mid_abs] * acceptance_pair[I]',
        'sv_pain_para * OV_value[high_OV] * Abs_value[high_abs] * acceptance_pair[P]',
        'sv_pain_para * OV_value[high_OV] * Abs_value[high_abs] * acceptance_pair[M]',
        'sv_pain_para * OV_value[high_OV] * Abs_value[high_abs] * acceptance_pair[I]'
        ]
    elif version == 25:
        params_of_interest = [
        'z', 'a', 't', 'sv', 'sz', 'st',
        'v_C(OV_value)[low_OV]', 'v_C(OV_value)[high_OV]',
        'v_C(acceptance_pair)[P]', 'v_C(acceptance_pair)[M]', 'v_C(acceptance_pair)[I]',
        'v_sv_pain_para:C(OV_value)[low_OV]', 'v_sv_pain_para:C(OV_value)[high_OV]',
        'v_sv_pain_para:C(acceptance_pair)[P]', 'v_sv_pain_para:C(acceptance_pair)[M]', 'v_sv_pain_para:C(acceptance_pair)[I]',
        'v_C(OV_value)[low_OV]:C(acceptance_pair)[P]', 'v_C(OV_value)[low_OV]:C(acceptance_pair)[M]', 'v_C(OV_value)[low_OV]:C(acceptance_pair)[I]',
        'v_C(OV_value)[high_OV]:C(acceptance_pair)[P]', 'v_C(OV_value)[high_OV]:C(acceptance_pair)[M]', 'v_C(OV_value)[high_OV]:C(acceptance_pair)[I]',
        'v_sv_pain_para:C(OV_value)[low_OV]:C(acceptance_pair)[P]', 'v_sv_pain_para:C(OV_value)[low_OV]:C(acceptance_pair)[M]', 'v_sv_pain_para:C(OV_value)[low_OV]:C(acceptance_pair)[I]',
        'v_sv_pain_para:C(OV_value)[high_OV]:C(acceptance_pair)[P]', 'v_sv_pain_para:C(OV_value)[high_OV]:C(acceptance_pair)[M]', 'v_sv_pain_para:C(OV_value)[high_OV]:C(acceptance_pair)[I]'
        ]
        params_of_interest_s = [
        'z_subj', 'a_subj', 't_subj', 'sv_subj', 'sz_subj', 'st_subj',
        'v_C(OV_value)[low_OV]_subj', 'v_C(OV_value)[high_OV]_subj',
        'v_C(acceptance_pair)[P]_subj', 'v_C(acceptance_pair)[M]_subj', 'v_C(acceptance_pair)[I]_subj',
        'v_sv_pain_para:C(OV_value)[low_OV]_subj', 'v_sv_pain_para:C(OV_value)[high_OV]_subj',
        'v_sv_pain_para:C(acceptance_pair)[P]_subj', 'v_sv_pain_para:C(acceptance_pair)[M]_subj', 'v_sv_pain_para:C(acceptance_pair)[I]_subj',
        'v_C(OV_value)[low_OV]:C(acceptance_pair)[P]_subj', 'v_C(OV_value)[low_OV]:C(acceptance_pair)[M]_subj', 'v_C(OV_value)[low_OV]:C(acceptance_pair)[I]_subj',
        'v_C(OV_value)[high_OV]:C(acceptance_pair)[P]_subj', 'v_C(OV_value)[high_OV]:C(acceptance_pair)[M]_subj', 'v_C(OV_value)[high_OV]:C(acceptance_pair)[I]_subj',
        'v_sv_pain_para:C(OV_value)[low_OV]:C(acceptance_pair)[P]_subj', 'v_sv_pain_para:C(OV_value)[low_OV]:C(acceptance_pair)[M]_subj', 'v_sv_pain_para:C(OV_value)[low_OV]:C(acceptance_pair)[I]_subj',
        'v_sv_pain_para:C(OV_value)[high_OV]:C(acceptance_pair)[P]_subj', 'v_sv_pain_para:C(OV_value)[high_OV]:C(acceptance_pair)[M]_subj', 'v_sv_pain_para:C(OV_value)[high_OV]:C(acceptance_pair)[I]_subj'
        ]
        titles = [
        'Starting point', 'Boundary separation', 'Non-decision time', 'Inter-trial variability in drift rate',
        'Inter-trial variability in starting point', 'Inter-trial variability in non-decision time',
        'Effect of OV_value[low_OV]', 'Effect of OV_value[high_OV]',
        'Effect of acceptance_pair[P]', 'Effect of acceptance_pair[M]', 'Effect of acceptance_pair[I]',
        'Interaction: sv_pain_para * OV_value[low_OV]', 'Interaction: sv_pain_para * OV_value[high_OV]',
        'Interaction: sv_pain_para * acceptance_pair[P]', 'Interaction: sv_pain_para * acceptance_pair[M]', 'Interaction: sv_pain_para * acceptance_pair[I]',
        'Interaction: OV_value[low_OV] * acceptance_pair[P]', 'Interaction: OV_value[low_OV] * acceptance_pair[M]', 'Interaction: OV_value[low_OV] * acceptance_pair[I]',
        'Interaction: OV_value[high_OV] * acceptance_pair[P]', 'Interaction: OV_value[high_OV] * acceptance_pair[M]', 'Interaction: OV_value[high_OV] * acceptance_pair[I]',
        'Three-way interaction: sv_pain_para * OV_value[low_OV] * acceptance_pair[P]', 'Three-way interaction: sv_pain_para * OV_value[low_OV] * acceptance_pair[M]',
        'Three-way interaction: sv_pain_para * OV_value[low_OV] * acceptance_pair[I]', 'Three-way interaction: sv_pain_para * OV_value[high_OV] * acceptance_pair[P]',
        'Three-way interaction: sv_pain_para * OV_value[high_OV] * acceptance_pair[M]', 'Three-way interaction: sv_pain_para * OV_value[high_OV] * acceptance_pair[I]'
        ]
    elif version == 26:
        params_of_interest = [
        'z', 'a', 't', 'sv', 'sz', 'st',
        'v_sv_pain_para:C(Abs_value)[low_abs]', 'v_sv_pain_para:C(Abs_value)[mid_abs]', 'v_sv_pain_para:C(Abs_value)[high_abs]',
        't_sv_pain_para:C(Abs_value)[low_abs]', 't_sv_pain_para:C(Abs_value)[mid_abs]', 't_sv_pain_para:C(Abs_value)[high_abs]',
        'a_sv_pain_para:C(Abs_value)[low_abs]', 'a_sv_pain_para:C(Abs_value)[mid_abs]', 'a_sv_pain_para:C(Abs_value)[high_abs]',
        'z_sv_pain_para:C(Abs_value)[low_abs]', 'z_sv_pain_para:C(Abs_value)[mid_abs]', 'z_sv_pain_para:C(Abs_value)[high_abs]'
        ]
        params_of_interest_s = [
        'z_subj', 'a_subj', 't_subj', 'sv_subj', 'sz_subj', 'st_subj',
        'v_sv_pain_para:C(Abs_value)[low_abs]_subj', 'v_sv_pain_para:C(Abs_value)[mid_abs]_subj', 'v_sv_pain_para:C(Abs_value)[high_abs]_subj',
        't_sv_pain_para:C(Abs_value)[low_abs]_subj', 't_sv_pain_para:C(Abs_value)[mid_abs]_subj', 't_sv_pain_para:C(Abs_value)[high_abs]_subj',
        'a_sv_pain_para:C(Abs_value)[low_abs]_subj', 'a_sv_pain_para:C(Abs_value)[mid_abs]_subj', 'a_sv_pain_para:C(Abs_value)[high_abs]_subj',
        'z_sv_pain_para:C(Abs_value)[low_abs]_subj', 'z_sv_pain_para:C(Abs_value)[mid_abs]_subj', 'z_sv_pain_para:C(Abs_value)[high_abs]_subj'
        ]
        titles = [
        'Starting point', 'Boundary separation', 'Non-decision time', 'Inter-trial variability in drift rate',
        'Inter-trial variability in starting point', 'Inter-trial variability in non-decision time',
        'Drift rate interaction: sv_pain_para * Abs_value[low_abs]', 'Drift rate interaction: sv_pain_para * Abs_value[mid_abs]', 'Drift rate interaction: sv_pain_para * Abs_value[high_abs]',
        'Non-decision time interaction: sv_pain_para * Abs_value[low_abs]', 'Non-decision time interaction: sv_pain_para * Abs_value[mid_abs]', 'Non-decision time interaction: sv_pain_para * Abs_value[high_abs]',
        'Boundary separation interaction: sv_pain_para * Abs_value[low_abs]', 'Boundary separation interaction: sv_pain_para * Abs_value[mid_abs]', 'Boundary separation interaction: sv_pain_para * Abs_value[high_abs]',
        'Starting point interaction: sv_pain_para * Abs_value[low_abs]', 'Starting point interaction: sv_pain_para * Abs_value[mid_abs]', 'Starting point interaction: sv_pain_para * Abs_value[high_abs]'
        ]
    elif version == 27:
        params_of_interest = [
        'z', 'a', 't', 'sv', 'sz', 'st',
        'v_sv_pain_para:C(OV_value)[low_OV]', 'v_sv_pain_para:C(OV_value)[high_OV]',
        't_sv_pain_para:C(OV_value)[low_OV]', 't_sv_pain_para:C(OV_value)[high_OV]',
        'a_sv_pain_para:C(OV_value)[low_OV]', 'a_sv_pain_para:C(OV_value)[high_OV]',
        'z_sv_pain_para:C(OV_value)[low_OV]', 'z_sv_pain_para:C(OV_value)[high_OV]'
        ]
        params_of_interest_s = [
        'z_subj', 'a_subj', 't_subj', 'sv_subj', 'sz_subj', 'st_subj',
        'v_sv_pain_para:C(OV_value)[low_OV]_subj', 'v_sv_pain_para:C(OV_value)[high_OV]_subj',
        't_sv_pain_para:C(OV_value)[low_OV]_subj', 't_sv_pain_para:C(OV_value)[high_OV]_subj',
        'a_sv_pain_para:C(OV_value)[low_OV]_subj', 'a_sv_pain_para:C(OV_value)[high_OV]_subj',
        'z_sv_pain_para:C(OV_value)[low_OV]_subj', 'z_sv_pain_para:C(OV_value)[high_OV]_subj'
        ]
        titles = [
        'Starting point', 'Boundary separation', 'Non-decision time', 'Inter-trial variability in drift rate',
        'Inter-trial variability in starting point', 'Inter-trial variability in non-decision time',
        'Drift rate interaction: sv_pain_para * OV_value[low_OV]', 'Drift rate interaction: sv_pain_para * OV_value[high_OV]',
        'Non-decision time interaction: sv_pain_para * OV_value[low_OV]', 'Non-decision time interaction: sv_pain_para * OV_value[high_OV]',
        'Boundary separation interaction: sv_pain_para * OV_value[low_OV]', 'Boundary separation interaction: sv_pain_para * OV_value[high_OV]',
        'Starting point interaction: sv_pain_para * OV_value[low_OV]', 'Starting point interaction: sv_pain_para * OV_value[high_OV]'
        ]
    elif version == 28:
        params_of_interest = [
        'z', 'a', 't', 'sv', 'sz', 'st',
        'v_sv_pain_para:C(OV_value)[low_OV]:C(acceptance_pair)[P]', 'v_sv_pain_para:C(OV_value)[low_OV]:C(acceptance_pair)[M]', 'v_sv_pain_para:C(OV_value)[low_OV]:C(acceptance_pair)[I]',
        'v_sv_pain_para:C(OV_value)[high_OV]:C(acceptance_pair)[P]', 'v_sv_pain_para:C(OV_value)[high_OV]:C(acceptance_pair)[M]', 'v_sv_pain_para:C(OV_value)[high_OV]:C(acceptance_pair)[I]',
        't_sv_pain_para:C(OV_value)[low_OV]:C(acceptance_pair)[P]', 't_sv_pain_para:C(OV_value)[low_OV]:C(acceptance_pair)[M]', 't_sv_pain_para:C(OV_value)[low_OV]:C(acceptance_pair)[I]',
        't_sv_pain_para:C(OV_value)[high_OV]:C(acceptance_pair)[P]', 't_sv_pain_para:C(OV_value)[high_OV]:C(acceptance_pair)[M]', 't_sv_pain_para:C(OV_value)[high_OV]:C(acceptance_pair)[I]',
        'a_sv_pain_para:C(OV_value)[low_OV]:C(acceptance_pair)[P]', 'a_sv_pain_para:C(OV_value)[low_OV]:C(acceptance_pair)[M]', 'a_sv_pain_para:C(OV_value)[low_OV]:C(acceptance_pair)[I]',
        'a_sv_pain_para:C(OV_value)[high_OV]:C(acceptance_pair)[P]', 'a_sv_pain_para:C(OV_value)[high_OV]:C(acceptance_pair)[M]', 'a_sv_pain_para:C(OV_value)[high_OV]:C(acceptance_pair)[I]',
        'z_sv_pain_para:C(OV_value)[low_OV]:C(acceptance_pair)[P]', 'z_sv_pain_para:C(OV_value)[low_OV]:C(acceptance_pair)[M]', 'z_sv_pain_para:C(OV_value)[low_OV]:C(acceptance_pair)[I]',
        'z_sv_pain_para:C(OV_value)[high_OV]:C(acceptance_pair)[P]', 'z_sv_pain_para:C(OV_value)[high_OV]:C(acceptance_pair)[M]', 'z_sv_pain_para:C(OV_value)[high_OV]:C(acceptance_pair)[I]'
        ]
        params_of_interest_s = [
        'z_subj', 'a_subj', 't_subj', 'sv_subj', 'sz_subj', 'st_subj',
        'v_sv_pain_para:C(OV_value)[low_OV]:C(acceptance_pair)[P]_subj', 'v_sv_pain_para:C(OV_value)[low_OV]:C(acceptance_pair)[M]_subj', 'v_sv_pain_para:C(OV_value)[low_OV]:C(acceptance_pair)[I]_subj',
        'v_sv_pain_para:C(OV_value)[high_OV]:C(acceptance_pair)[P]_subj', 'v_sv_pain_para:C(OV_value)[high_OV]:C(acceptance_pair)[M]_subj', 'v_sv_pain_para:C(OV_value)[high_OV]:C(acceptance_pair)[I]_subj',
        't_sv_pain_para:C(OV_value)[low_OV]:C(acceptance_pair)[P]_subj', 't_sv_pain_para:C(OV_value)[low_OV]:C(acceptance_pair)[M]_subj', 't_sv_pain_para:C(OV_value)[low_OV]:C(acceptance_pair)[I]_subj',
        't_sv_pain_para:C(OV_value)[high_OV]:C(acceptance_pair)[P]_subj', 't_sv_pain_para:C(OV_value)[high_OV]:C(acceptance_pair)[M]_subj', 't_sv_pain_para:C(OV_value)[high_OV]:C(acceptance_pair)[I]_subj',
        'a_sv_pain_para:C(OV_value)[low_OV]:C(acceptance_pair)[P]_subj', 'a_sv_pain_para:C(OV_value)[low_OV]:C(acceptance_pair)[M]_subj', 'a_sv_pain_para:C(OV_value)[low_OV]:C(acceptance_pair)[I]_subj',
        'a_sv_pain_para:C(OV_value)[high_OV]:C(acceptance_pair)[P]_subj', 'a_sv_pain_para:C(OV_value)[high_OV]:C(acceptance_pair)[M]_subj', 'a_sv_pain_para:C(OV_value)[high_OV]:C(acceptance_pair)[I]_subj',
        'z_sv_pain_para:C(OV_value)[low_OV]:C(acceptance_pair)[P]_subj', 'z_sv_pain_para:C(OV_value)[low_OV]:C(acceptance_pair)[M]_subj', 'z_sv_pain_para:C(OV_value)[low_OV]:C(acceptance_pair)[I]_subj',
        'z_sv_pain_para:C(OV_value)[high_OV]:C(acceptance_pair)[P]_subj', 'z_sv_pain_para:C(OV_value)[high_OV]:C(acceptance_pair)[M]_subj', 'z_sv_pain_para:C(OV_value)[high_OV]:C(acceptance_pair)[I]_subj'
        ]
        titles = [
        'Starting point', 'Boundary separation', 'Non-decision time', 'Inter-trial variability in drift rate',
        'Inter-trial variability in starting point', 'Inter-trial variability in non-decision time',
        'Drift rate sv_pain_para * OV_value[low_OV] * acceptance_pair[P]', 'Drift rate sv_pain_para * OV_value[low_OV] * acceptance_pair[M]', 'Drift rate sv_pain_para * OV_value[low_OV] * acceptance_pair[I]',
        'Drift rate sv_pain_para * OV_value[high_OV] * acceptance_pair[P]', 'Drift rate sv_pain_para * OV_value[high_OV] * acceptance_pair[M]', 'Drift rate sv_pain_para * OV_value[high_OV] * acceptance_pair[I]',
        'Non-decision time sv_pain_para * OV_value[low_OV] * acceptance_pair[P]', 'Non-decision time sv_pain_para * OV_value[low_OV] * acceptance_pair[M]', 'Non-decision time sv_pain_para * OV_value[low_OV] * acceptance_pair[I]',
        'Non-decision time sv_pain_para * OV_value[high_OV] * acceptance_pair[P]', 'Non-decision time sv_pain_para * OV_value[high_OV] * acceptance_pair[M]', 'Non-decision time sv_pain_para * OV_value[high_OV] * acceptance_pair[I]',
        'Boundary separation sv_pain_para * OV_value[low_OV] * acceptance_pair[P]', 'Boundary separation sv_pain_para * OV_value[low_OV] * acceptance_pair[M]', 'Boundary separation sv_pain_para * OV_value[low_OV] * acceptance_pair[I]',
        'Boundary separation sv_pain_para * OV_value[high_OV] * acceptance_pair[P]', 'Boundary separation sv_pain_para * OV_value[high_OV] * acceptance_pair[M]', 'Boundary separation sv_pain_para * OV_value[high_OV] * acceptance_pair[I]',
        'Starting point sv_pain_para * OV_value[low_OV] * acceptance_pair[P]', 'Starting point sv_pain_para * OV_value[low_OV] * acceptance_pair[M]', 'Starting point sv_pain_para * OV_value[low_OV] * acceptance_pair[I]',
        'Starting point sv_pain_para * OV_value[high_OV] * acceptance_pair[P]', 'Starting point sv_pain_para * OV_value[high_OV] * acceptance_pair[M]', 'Starting point sv_pain_para * OV_value[high_OV] * acceptance_pair[I]'
        ]
    elif version == 29:
        params_of_interest = [
        'z', 'a', 't', 'sv', 'sz', 'st',
        'v_sv_pain_para:C(Abs_value)[low_abs]:C(acceptance_pair)[P]', 'v_sv_pain_para:C(Abs_value)[low_abs]:C(acceptance_pair)[M]', 'v_sv_pain_para:C(Abs_value)[low_abs]:C(acceptance_pair)[I]',
        'v_sv_pain_para:C(Abs_value)[mid_abs]:C(acceptance_pair)[P]', 'v_sv_pain_para:C(Abs_value)[mid_abs]:C(acceptance_pair)[M]', 'v_sv_pain_para:C(Abs_value)[mid_abs]:C(acceptance_pair)[I]',
        'v_sv_pain_para:C(Abs_value)[high_abs]:C(acceptance_pair)[P]', 'v_sv_pain_para:C(Abs_value)[high_abs]:C(acceptance_pair)[M]', 'v_sv_pain_para:C(Abs_value)[high_abs]:C(acceptance_pair)[I]',
        't_sv_pain_para:C(Abs_value)[low_abs]:C(acceptance_pair)[P]', 't_sv_pain_para:C(Abs_value)[low_abs]:C(acceptance_pair)[M]', 't_sv_pain_para:C(Abs_value)[low_abs]:C(acceptance_pair)[I]',
        't_sv_pain_para:C(Abs_value)[mid_abs]:C(acceptance_pair)[P]', 't_sv_pain_para:C(Abs_value)[mid_abs]:C(acceptance_pair)[M]', 't_sv_pain_para:C(Abs_value)[mid_abs]:C(acceptance_pair)[I]',
        't_sv_pain_para:C(Abs_value)[high_abs]:C(acceptance_pair)[P]', 't_sv_pain_para:C(Abs_value)[high_abs]:C(acceptance_pair)[M]', 't_sv_pain_para:C(Abs_value)[high_abs]:C(acceptance_pair)[I]',
        'a_sv_pain_para:C(Abs_value)[low_abs]:C(acceptance_pair)[P]', 'a_sv_pain_para:C(Abs_value)[low_abs]:C(acceptance_pair)[M]', 'a_sv_pain_para:C(Abs_value)[low_abs]:C(acceptance_pair)[I]',
        'a_sv_pain_para:C(Abs_value)[mid_abs]:C(acceptance_pair)[P]', 'a_sv_pain_para:C(Abs_value)[mid_abs]:C(acceptance_pair)[M]', 'a_sv_pain_para:C(Abs_value)[mid_abs]:C(acceptance_pair)[I]',
        'a_sv_pain_para:C(Abs_value)[high_abs]:C(acceptance_pair)[P]', 'a_sv_pain_para:C(Abs_value)[high_abs]:C(acceptance_pair)[M]', 'a_sv_pain_para:C(Abs_value)[high_abs]:C(acceptance_pair)[I]',
        'z_sv_pain_para:C(Abs_value)[low_abs]:C(acceptance_pair)[P]', 'z_sv_pain_para:C(Abs_value)[low_abs]:C(acceptance_pair)[M]', 'z_sv_pain_para:C(Abs_value)[low_abs]:C(acceptance_pair)[I]',
        'z_sv_pain_para:C(Abs_value)[mid_abs]:C(acceptance_pair)[P]', 'z_sv_pain_para:C(Abs_value)[mid_abs]:C(acceptance_pair)[M]', 'z_sv_pain_para:C(Abs_value)[mid_abs]:C(acceptance_pair)[I]',
        'z_sv_pain_para:C(Abs_value)[high_abs]:C(acceptance_pair)[P]', 'z_sv_pain_para:C(Abs_value)[high_abs]:C(acceptance_pair)[M]', 'z_sv_pain_para:C(Abs_value)[high_abs]:C(acceptance_pair)[I]'
        ]
        params_of_interest_s = [
        'z_subj', 'a_subj', 't_subj', 'sv_subj', 'sz_subj', 'st_subj',
        'v_sv_pain_para:C(Abs_value)[low_abs]:C(acceptance_pair)[P]_subj', 'v_sv_pain_para:C(Abs_value)[low_abs]:C(acceptance_pair)[M]_subj', 'v_sv_pain_para:C(Abs_value)[low_abs]:C(acceptance_pair)[I]_subj',
        'v_sv_pain_para:C(Abs_value)[mid_abs]:C(acceptance_pair)[P]_subj', 'v_sv_pain_para:C(Abs_value)[mid_abs]:C(acceptance_pair)[M]_subj', 'v_sv_pain_para:C(Abs_value)[mid_abs]:C(acceptance_pair)[I]_subj',
        'v_sv_pain_para:C(Abs_value)[high_abs]:C(acceptance_pair)[P]_subj', 'v_sv_pain_para:C(Abs_value)[high_abs]:C(acceptance_pair)[M]_subj', 'v_sv_pain_para:C(Abs_value)[high_abs]:C(acceptance_pair)[I]_subj',
        't_sv_pain_para:C(Abs_value)[low_abs]:C(acceptance_pair)[P]_subj', 't_sv_pain_para:C(Abs_value)[low_abs]:C(acceptance_pair)[M]_subj', 't_sv_pain_para:C(Abs_value)[low_abs]:C(acceptance_pair)[I]_subj',
        't_sv_pain_para:C(Abs_value)[mid_abs]:C(acceptance_pair)[P]_subj', 't_sv_pain_para:C(Abs_value)[mid_abs]:C(acceptance_pair)[M]_subj', 't_sv_pain_para:C(Abs_value)[mid_abs]:C(acceptance_pair)[I]_subj',
        't_sv_pain_para:C(Abs_value)[high_abs]:C(acceptance_pair)[P]_subj', 't_sv_pain_para:C(Abs_value)[high_abs]:C(acceptance_pair)[M]_subj', 't_sv_pain_para:C(Abs_value)[high_abs]:C(acceptance_pair)[I]_subj',
        'a_sv_pain_para:C(Abs_value)[low_abs]:C(acceptance_pair)[P]_subj', 'a_sv_pain_para:C(Abs_value)[low_abs]:C(acceptance_pair)[M]_subj', 'a_sv_pain_para:C(Abs_value)[low_abs]:C(acceptance_pair)[I]_subj',
        'a_sv_pain_para:C(Abs_value)[mid_abs]:C(acceptance_pair)[P]_subj', 'a_sv_pain_para:C(Abs_value)[mid_abs]:C(acceptance_pair)[M]_subj', 'a_sv_pain_para:C(Abs_value)[mid_abs]:C(acceptance_pair)[I]_subj',
        'a_sv_pain_para:C(Abs_value)[high_abs]:C(acceptance_pair)[P]_subj', 'a_sv_pain_para:C(Abs_value)[high_abs]:C(acceptance_pair)[M]_subj', 'a_sv_pain_para:C(Abs_value)[high_abs]:C(acceptance_pair)[I]_subj',
        'z_sv_pain_para:C(Abs_value)[low_abs]:C(acceptance_pair)[P]_subj', 'z_sv_pain_para:C(Abs_value)[low_abs]:C(acceptance_pair)[M]_subj', 'z_sv_pain_para:C(Abs_value)[low_abs]:C(acceptance_pair)[I]_subj',
        'z_sv_pain_para:C(Abs_value)[mid_abs]:C(acceptance_pair)[P]_subj', 'z_sv_pain_para:C(Abs_value)[mid_abs]:C(acceptance_pair)[M]_subj', 'z_sv_pain_para:C(Abs_value)[mid_abs]:C(acceptance_pair)[I]_subj',
        'z_sv_pain_para:C(Abs_value)[high_abs]:C(acceptance_pair)[P]_subj', 'z_sv_pain_para:C(Abs_value)[high_abs]:C(acceptance_pair)[M]_subj', 'z_sv_pain_para:C(Abs_value)[high_abs]:C(acceptance_pair)[I]_subj'
        ]
        titles = [
        'Starting point', 'Boundary separation', 'Non-decision time', 'Inter-trial variability in drift rate',
        'Inter-trial variability in starting point', 'Inter-trial variability in non-decision time',
        'Drift rate sv_pain_para * Abs_value[low_abs] * acceptance_pair[P]', 'Drift rate sv_pain_para * Abs_value[low_abs] * acceptance_pair[M]', 'Drift rate sv_pain_para * Abs_value[low_abs] * acceptance_pair[I]',
        'Drift rate sv_pain_para * Abs_value[mid_abs] * acceptance_pair[P]', 'Drift rate sv_pain_para * Abs_value[mid_abs] * acceptance_pair[M]', 'Drift rate sv_pain_para * Abs_value[mid_abs] * acceptance_pair[I]',
        'Drift rate sv_pain_para * Abs_value[high_abs] * acceptance_pair[P]', 'Drift rate sv_pain_para * Abs_value[high_abs] * acceptance_pair[M]', 'Drift rate sv_pain_para * Abs_value[high_abs] * acceptance_pair[I]',
        'Non-decision time sv_pain_para * Abs_value[low_abs] * acceptance_pair[P]', 'Non-decision time sv_pain_para * Abs_value[low_abs] * acceptance_pair[M]', 'Non-decision time sv_pain_para * Abs_value[low_abs] * acceptance_pair[I]',
        'Non-decision time sv_pain_para * Abs_value[mid_abs] * acceptance_pair[P]', 'Non-decision time sv_pain_para * Abs_value[mid_abs] * acceptance_pair[M]', 'Non-decision time sv_pain_para * Abs_value[mid_abs] * acceptance_pair[I]',
        'Non-decision time sv_pain_para * Abs_value[high_abs] * acceptance_pair[P]', 'Non-decision time sv_pain_para * Abs_value[high_abs] * acceptance_pair[M]', 'Non-decision time sv_pain_para * Abs_value[high_abs] * acceptance_pair[I]',
        'Boundary separation sv_pain_para * Abs_value[low_abs] * acceptance_pair[P]', 'Boundary separation sv_pain_para * Abs_value[low_abs] * acceptance_pair[M]', 'Boundary separation sv_pain_para * Abs_value[low_abs] * acceptance_pair[I]',
        'Boundary separation sv_pain_para * Abs_value[mid_abs] * acceptance_pair[P]', 'Boundary separation sv_pain_para * Abs_value[mid_abs] * acceptance_pair[M]', 'Boundary separation sv_pain_para * Abs_value[mid_abs] * acceptance_pair[I]',
        'Boundary separation sv_pain_para * Abs_value[high_abs] * acceptance_pair[P]', 'Boundary separation sv_pain_para * Abs_value[high_abs] * acceptance_pair[M]', 'Boundary separation sv_pain_para * Abs_value[high_abs] * acceptance_pair[I]',
        'Starting point sv_pain_para * Abs_value[low_abs] * acceptance_pair[P]', 'Starting point sv_pain_para * Abs_value[low_abs] * acceptance_pair[M]', 'Starting point sv_pain_para * Abs_value[low_abs] * acceptance_pair[I]',
        'Starting point sv_pain_para * Abs_value[mid_abs] * acceptance_pair[P]', 'Starting point sv_pain_para * Abs_value[mid_abs] * acceptance_pair[M]', 'Starting point sv_pain_para * Abs_value[mid_abs] * acceptance_pair[I]',
        'Starting point sv_pain_para * Abs_value[high_abs] * acceptance_pair[P]', 'Starting point sv_pain_para * Abs_value[high_abs] * acceptance_pair[M]', 'Starting point sv_pain_para * Abs_value[high_abs] * acceptance_pair[I]'
        ]
    elif version == 33:
        params_of_interest = [
        'z', 'a', 't', 'sv', 'sz', 'st',
        'v_sv_pain_para:C(OV_value)[low_OV]:C(acceptance_pair)[P]', 'v_sv_pain_para:C(OV_value)[low_OV]:C(acceptance_pair)[M]', 'v_sv_pain_para:C(OV_value)[low_OV]:C(acceptance_pair)[I]',
        'v_sv_pain_para:C(OV_value)[high_OV]:C(acceptance_pair)[P]', 'v_sv_pain_para:C(OV_value)[high_OV]:C(acceptance_pair)[M]', 'v_sv_pain_para:C(OV_value)[high_OV]:C(acceptance_pair)[I]',
        't_sv_pain_para',
        'a_sv_pain_para:C(OV_value)[low_OV]:C(acceptance_pair)[P]', 'a_sv_pain_para:C(OV_value)[low_OV]:C(acceptance_pair)[M]', 'a_sv_pain_para:C(OV_value)[low_OV]:C(acceptance_pair)[I]',
        'a_sv_pain_para:C(OV_value)[high_OV]:C(acceptance_pair)[P]', 'a_sv_pain_para:C(OV_value)[high_OV]:C(acceptance_pair)[M]', 'a_sv_pain_para:C(OV_value)[high_OV]:C(acceptance_pair)[I]',
        'z_sv_pain_para:C(acceptance_pair)[P]', 'z_sv_pain_para:C(acceptance_pair)[M]', 'z_sv_pain_para:C(acceptance_pair)[I]'
        ]
        params_of_interest_s = [
        'z_subj', 'a_subj', 't_subj', 'sv_subj', 'sz_subj', 'st_subj',
        'v_sv_pain_para:C(OV_value)[low_OV]:C(acceptance_pair)[P]_subj', 'v_sv_pain_para:C(OV_value)[low_OV]:C(acceptance_pair)[M]_subj', 'v_sv_pain_para:C(OV_value)[low_OV]:C(acceptance_pair)[I]_subj',
        'v_sv_pain_para:C(OV_value)[high_OV]:C(acceptance_pair)[P]_subj', 'v_sv_pain_para:C(OV_value)[high_OV]:C(acceptance_pair)[M]_subj', 'v_sv_pain_para:C(OV_value)[high_OV]:C(acceptance_pair)[I]_subj',
        't_sv_pain_para_subj',
        'a_sv_pain_para:C(OV_value)[low_OV]:C(acceptance_pair)[P]_subj', 'a_sv_pain_para:C(OV_value)[low_OV]:C(acceptance_pair)[M]_subj', 'a_sv_pain_para:C(OV_value)[low_OV]:C(acceptance_pair)[I]_subj',
        'a_sv_pain_para:C(OV_value)[high_OV]:C(acceptance_pair)[P]_subj', 'a_sv_pain_para:C(OV_value)[high_OV]:C(acceptance_pair)[M]_subj', 'a_sv_pain_para:C(OV_value)[high_OV]:C(acceptance_pair)[I]_subj',
        'z_sv_pain_para:C(acceptance_pair)[P]_subj', 'z_sv_pain_para:C(acceptance_pair)[M]_subj', 'z_sv_pain_para:C(acceptance_pair)[I]_subj'
        ]
        titles = [
        'Starting point', 'Boundary separation', 'Non-decision time', 'Inter-trial variability in drift rate',
        'Inter-trial variability in starting point', 'Inter-trial variability in non-decision time',
        'Drift rate sv_pain_para * OV_value[low_OV] * acceptance_pair[P]', 'Drift rate sv_pain_para * OV_value[low_OV] * acceptance_pair[M]', 'Drift rate sv_pain_para * OV_value[low_OV] * acceptance_pair[I]',
        'Drift rate sv_pain_para * OV_value[high_OV] * acceptance_pair[P]', 'Drift rate sv_pain_para * OV_value[high_OV] * acceptance_pair[M]', 'Drift rate sv_pain_para * OV_value[high_OV] * acceptance_pair[I]',
        'Non-decision time sv_pain_para',
        'Boundary separation sv_pain_para * OV_value[low_OV] * acceptance_pair[P]', 'Boundary separation sv_pain_para * OV_value[low_OV] * acceptance_pair[M]', 'Boundary separation sv_pain_para * OV_value[low_OV] * acceptance_pair[I]',
        'Boundary separation sv_pain_para * OV_value[high_OV] * acceptance_pair[P]', 'Boundary separation sv_pain_para * OV_value[high_OV] * acceptance_pair[M]', 'Boundary separation sv_pain_para * OV_value[high_OV] * acceptance_pair[I]',
        'Starting point sv_pain_para * acceptance_pair[P]', 'Starting point sv_pain_para * acceptance_pair[M]', 'Starting point sv_pain_para * acceptance_pair[I]'
        ]
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # Quest data        
    elif version == 34:  # v depends on STA_SAI_Score
        params_of_interest = ['z', 'a', 't', 'sv', 'sz', 'st', 'v_Intercept', 'v_STA_SAI_Score']
        params_of_interest_s = ['z_subj', 'a_subj', 't_subj', 'v_Intercept_subj', 'v_STA_SAI_Score_subj']
        titles = [
            'Starting point', 'Boundary sep.', 'Non-dec. time',
            'Inter-trial variability in drift rate', 'Inter-trial variability in starting point', 'Inter-trial variability in non-decision time',
            'Intercept drift rate', 'Drift rate STA_SAI_Score']

    elif version == 35:  # a depends on STA_SAI_Score
        params_of_interest = ['z', 't', 'sv', 'sz', 'st', 'a_Intercept', 'a_STA_SAI_Score']
        params_of_interest_s = ['z_subj', 't_subj', 'sv_subj', 'sz_subj', 'st_subj', 'a_Intercept_subj', 'a_STA_SAI_Score_subj']
        titles = [
            'Starting point', 'Non-dec. time', 
            'Inter-trial variability in drift rate', 'Inter-trial variability in starting point', 'Inter-trial variability in non-decision time',
            'Intercept boundary separation', 'Boundary separation STA_SAI_Score']

    elif version == 36:  # v depends on STA_TAI_Score
        params_of_interest = ['z', 'a', 't', 'sv', 'sz', 'st', 'v_Intercept', 'v_STA_TAI_Score']
        params_of_interest_s = ['z_subj', 'a_subj', 't_subj', 'v_Intercept_subj', 'v_STA_TAI_Score_subj']
        titles = [
            'Starting point', 'Boundary sep.', 'Non-dec. time',
            'Inter-trial variability in drift rate', 'Inter-trial variability in starting point', 'Inter-trial variability in non-decision time',
            'Intercept drift rate', 'Drift rate STA_TAI_Score']

    elif version == 37:  # a depends on STA_TAI_Score
        params_of_interest = ['z', 't', 'sv', 'sz', 'st', 'a_Intercept', 'a_STA_TAI_Score']
        params_of_interest_s = ['z_subj', 't_subj', 'sv_subj', 'sz_subj', 'st_subj', 'a_Intercept_subj', 'a_STA_TAI_Score_subj']
        titles = [
            'Starting point', 'Non-dec. time', 
            'Inter-trial variability in drift rate', 'Inter-trial variability in starting point', 'Inter-trial variability in non-decision time',
            'Intercept boundary separation', 'Boundary separation STA_TAI_Score']

    elif version == 38:  # v depends on PCS_Score
        params_of_interest = ['z', 'a', 't', 'sv', 'sz', 'st', 'v_Intercept', 'v_PCS_Score']
        params_of_interest_s = ['z_subj', 'a_subj', 't_subj', 'v_Intercept_subj', 'v_PCS_Score_subj']
        titles = [
            'Starting point', 'Boundary sep.', 'Non-dec. time',
            'Inter-trial variability in drift rate', 'Inter-trial variability in starting point', 'Inter-trial variability in non-decision time',
            'Intercept drift rate', 'Drift rate PCS_Score']

    elif version == 39:  # a depends on PCS_Score
        params_of_interest = ['z', 't', 'sv', 'sz', 'st', 'a_Intercept', 'a_PCS_Score']
        params_of_interest_s = ['z_subj', 't_subj', 'sv_subj', 'sz_subj', 'st_subj', 'a_Intercept_subj', 'a_PCS_Score_subj']
        titles = [
            'Starting point', 'Non-dec. time', 
            'Inter-trial variability in drift rate', 'Inter-trial variability in starting point', 'Inter-trial variability in non-decision time',
            'Intercept boundary separation', 'Boundary separation PCS_Score']

    elif version == 40:  
        params_of_interest = [
            'sv', 'sz', 'st',
            'v_Intercept', 'v_STA_SAI_Score',
            't_Intercept', 't_STA_SAI_Score',
            'a_Intercept', 'a_STA_SAI_Score',
            'z_Intercept', 'z_STA_SAI_Score']
        params_of_interest_s = [
            'sv_subj', 'sz_subj', 'st_subj',
            'v_Intercept_subj', 'v_STA_SAI_Score_subj',
            't_Intercept_subj', 't_STA_SAI_Score_subj',
            'a_Intercept_subj', 'a_STA_SAI_Score_subj',
            'z_Intercept_subj', 'z_STA_SAI_Score_subj']
        titles = [
            'Inter-trial variability in drift rate', 'Inter-trial variability in starting point', 'Inter-trial variability in non-decision time',
            'Intercept drift rate', 'Drift rate STA_SAI_Score',
            'Intercept non-decision time', 'Non-decision time STA_SAI_Score',
            'Intercept boundary separation', 'Boundary separation STA_SAI_Score',
            'Intercept starting point', 'Starting point STA_SAI_Score']

    elif version == 41:  
        params_of_interest = [
            'sv', 'sz', 'st',
            'v_Intercept', 'v_STA_TAI_Score',
            't_Intercept', 't_STA_TAI_Score',
            'a_Intercept', 'a_STA_TAI_Score',
            'z_Intercept', 'z_STA_TAI_Score']
        params_of_interest_s = [
            'sv_subj', 'sz_subj', 'st_subj',
            'v_Intercept_subj', 'v_STA_TAI_Score_subj',
            't_Intercept_subj', 't_STA_TAI_Score_subj',
            'a_Intercept_subj', 'a_STA_TAI_Score_subj',
            'z_Intercept_subj', 'z_STA_TAI_Score_subj']
        titles = [
            'Inter-trial variability in drift rate', 'Inter-trial variability in starting point', 'Inter-trial variability in non-decision time',
            'Intercept drift rate', 'Drift rate STA_TAI_Score',
            'Intercept non-decision time', 'Non-decision time STA_TAI_Score',
            'Intercept boundary separation', 'Boundary separation STA_TAI_Score',
            'Intercept starting point', 'Starting point STA_TAI_Score']

    elif version == 42:  
        params_of_interest = [
            'sv', 'sz', 'st',
            'v_Intercept', 'v_PCS_Score',
            't_Intercept', 't_PCS_Score',
            'a_Intercept', 'a_PCS_Score',
            'z_Intercept', 'z_PCS_Score']
        params_of_interest_s = [
            'sv_subj', 'sz_subj', 'st_subj',
            'v_Intercept_subj', 'v_PCS_Score_subj',
            't_Intercept_subj', 't_PCS_Score_subj',
            'a_Intercept_subj', 'a_PCS_Score_subj',
            'z_Intercept_subj', 'z_PCS_Score_subj']
        titles = [
            'Inter-trial variability in drift rate', 'Inter-trial variability in starting point', 'Inter-trial variability in non-decision time',
            'Intercept drift rate', 'Drift rate PCS_Score',
            'Intercept non-decision time', 'Non-decision time PCS_Score',
            'Intercept boundary separation', 'Boundary separation PCS_Score',
            'Intercept starting point', 'Starting point PCS_Score']    
        
    elif version == 43:  
        params_of_interest = [
            'sv', 'sz', 'st',
            'v_Intercept', 'v_STA_SAI_Score:sv_pain_para:C(OV_value)',
            't_Intercept', 't_STA_SAI_Score:sv_pain_para:C(OV_value)',
            'a_Intercept', 'a_STA_SAI_Score:sv_pain_para:C(OV_value)',
            'z_Intercept', 'z_STA_SAI_Score:sv_pain_para:C(OV_value)']
        params_of_interest_s = [
            'sv_subj', 'sz_subj', 'st_subj',
            'v_Intercept_subj', 'v_STA_SAI_Score:sv_pain_para:C(OV_value)_subj',
            't_Intercept_subj', 't_STA_SAI_Score:sv_pain_para:C(OV_value)_subj',
            'a_Intercept_subj', 'a_STA_SAI_Score:sv_pain_para:C(OV_value)_subj',
            'z_Intercept_subj', 'z_STA_SAI_Score:sv_pain_para:C(OV_value)_subj']
        titles = [
            'Inter-trial variability in drift rate', 'Inter-trial variability in starting point', 'Inter-trial variability in non-decision time',
            'Intercept drift rate', 'Drift rate interaction (STA_SAI_Score:sv_pain_para:C(OV_value))',
            'Intercept non-decision time', 'Non-decision time interaction (STA_SAI_Score:sv_pain_para:C(OV_value))',
            'Intercept boundary separation', 'Boundary separation interaction (STA_SAI_Score:sv_pain_para:C(OV_value))',
            'Intercept starting point', 'Starting point interaction (STA_SAI_Score:sv_pain_para:C(OV_value))']

    elif version == 44:  
        params_of_interest = [
            'sv', 'sz', 'st',
            'v_Intercept', 'v_STA_TAI_Score:sv_pain_para:C(OV_value)',
            't_Intercept', 't_STA_TAI_Score:sv_pain_para:C(OV_value)',
            'a_Intercept', 'a_STA_TAI_Score:sv_pain_para:C(OV_value)',
            'z_Intercept', 'z_STA_TAI_Score:sv_pain_para:C(OV_value)']
        params_of_interest_s = [
            'sv_subj', 'sz_subj', 'st_subj',
            'v_Intercept_subj', 'v_STA_TAI_Score:sv_pain_para:C(OV_value)_subj',
            't_Intercept_subj', 't_STA_TAI_Score:sv_pain_para:C(OV_value)_subj',
            'a_Intercept_subj', 'a_STA_TAI_Score:sv_pain_para:C(OV_value)_subj',
            'z_Intercept_subj', 'z_STA_TAI_Score:sv_pain_para:C(OV_value)_subj']
        titles = [
            'Inter-trial variability in drift rate', 'Inter-trial variability in starting point', 'Inter-trial variability in non-decision time',
            'Intercept drift rate', 'Drift rate interaction (STA_TAI_Score:sv_pain_para:C(OV_value))',
            'Intercept non-decision time', 'Non-decision time interaction (STA_TAI_Score:sv_pain_para:C(OV_value))',
            'Intercept boundary separation', 'Boundary separation interaction (STA_TAI_Score:sv_pain_para:C(OV_value))',
            'Intercept starting point', 'Starting point interaction (STA_TAI_Score:sv_pain_para:C(OV_value))']

    elif version == 45:  
        params_of_interest = [
            'sv', 'sz', 'st',
            'v_Intercept', 'v_PCS_Score:sv_pain_para:C(OV_value)',
            't_Intercept', 't_PCS_Score:sv_pain_para:C(OV_value)',
            'a_Intercept', 'a_PCS_Score:sv_pain_para:C(OV_value)',
            'z_Intercept', 'z_PCS_Score:sv_pain_para:C(OV_value)']
        params_of_interest_s = [
            'sv_subj', 'sz_subj', 'st_subj',
            'v_Intercept_subj', 'v_PCS_Score:sv_pain_para:C(OV_value)_subj',
            't_Intercept_subj', 't_PCS_Score:sv_pain_para:C(OV_value)_subj',
            'a_Intercept_subj', 'a_PCS_Score:sv_pain_para:C(OV_value)_subj',
            'z_Intercept_subj', 'z_PCS_Score:sv_pain_para:C(OV_value)_subj']
        titles = [
            'Inter-trial variability in drift rate', 'Inter-trial variability in starting point', 'Inter-trial variability in non-decision time',
            'Intercept drift rate', 'Drift rate interaction (PCS_Score:sv_pain_para:C(OV_value))',
            'Intercept non-decision time', 'Non-decision time interaction (PCS_Score:sv_pain_para:C(OV_value))',
            'Intercept boundary separation', 'Boundary separation interaction (PCS_Score:sv_pain_para:C(OV_value))',
            'Intercept starting point', 'Starting point interaction (PCS_Score:sv_pain_para:C(OV_value))']

    elif version == 46:  
        params_of_interest = [
            'sv', 'sz', 'st',
            'v_Intercept', 'v_STA_SAI_Score:sv_pain_para:C(Abs_value)',
            't_Intercept', 't_STA_SAI_Score:sv_pain_para:C(Abs_value)',
            'a_Intercept', 'a_STA_SAI_Score:sv_pain_para:C(Abs_value)',
            'z_Intercept', 'z_STA_SAI_Score:sv_pain_para:C(Abs_value)']
        params_of_interest_s = [
            'sv_subj', 'sz_subj', 'st_subj',
            'v_Intercept_subj', 'v_STA_SAI_Score:sv_pain_para:C(Abs_value)_subj',
            't_Intercept_subj', 't_STA_SAI_Score:sv_pain_para:C(Abs_value)_subj',
            'a_Intercept_subj', 'a_STA_SAI_Score:sv_pain_para:C(Abs_value)_subj',
            'z_Intercept_subj', 'z_STA_SAI_Score:sv_pain_para:C(Abs_value)_subj']
        titles = [
            'Inter-trial variability in drift rate', 'Inter-trial variability in starting point', 'Inter-trial variability in non-decision time',
            'Intercept drift rate', 'Drift rate interaction (STA_SAI_Score:sv_pain_para:C(Abs_value))',
            'Intercept non-decision time', 'Non-decision time interaction (STA_SAI_Score:sv_pain_para:C(Abs_value))',
            'Intercept boundary separation', 'Boundary separation interaction (STA_SAI_Score:sv_pain_para:C(Abs_value))',
            'Intercept starting point', 'Starting point interaction (STA_SAI_Score:sv_pain_para:C(Abs_value))']

    elif version == 47:  
        params_of_interest = [
            'sv', 'sz', 'st',
            'v_Intercept', 'v_STA_TAI_Score:sv_pain_para:C(Abs_value)',
            't_Intercept', 't_STA_TAI_Score:sv_pain_para:C(Abs_value)',
            'a_Intercept', 'a_STA_TAI_Score:sv_pain_para:C(Abs_value)',
            'z_Intercept', 'z_STA_TAI_Score:sv_pain_para:C(Abs_value)']
        params_of_interest_s = [
            'sv_subj', 'sz_subj', 'st_subj',
            'v_Intercept_subj', 'v_STA_TAI_Score:sv_pain_para:C(Abs_value)_subj',
            't_Intercept_subj', 't_STA_TAI_Score:sv_pain_para:C(Abs_value)_subj',
            'a_Intercept_subj', 'a_STA_TAI_Score:sv_pain_para:C(Abs_value)_subj',
            'z_Intercept_subj', 'z_STA_TAI_Score:sv_pain_para:C(Abs_value)_subj']
        titles = [
            'Inter-trial variability in drift rate', 'Inter-trial variability in starting point', 'Inter-trial variability in non-decision time',
            'Intercept drift rate', 'Drift rate interaction (STA_TAI_Score:sv_pain_para:C(Abs_value))',
            'Intercept non-decision time', 'Non-decision time interaction (STA_TAI_Score:sv_pain_para:C(Abs_value))',
            'Intercept boundary separation', 'Boundary separation interaction (STA_TAI_Score:sv_pain_para:C(Abs_value))',
            'Intercept starting point', 'Starting point interaction (STA_TAI_Score:sv_pain_para:C(Abs_value))']

    elif version == 48:  
        params_of_interest = [
            'sv', 'sz', 'st',
            'v_Intercept', 'v_PCS_Score:sv_pain_para:C(Abs_value)',
            't_Intercept', 't_PCS_Score:sv_pain_para:C(Abs_value)',
            'a_Intercept', 'a_PCS_Score:sv_pain_para:C(Abs_value)',
            'z_Intercept', 'z_PCS_Score:sv_pain_para:C(Abs_value)']
        params_of_interest_s = [
            'sv_subj', 'sz_subj', 'st_subj',
            'v_Intercept_subj', 'v_PCS_Score:sv_pain_para:C(Abs_value)_subj',
            't_Intercept_subj', 't_PCS_Score:sv_pain_para:C(Abs_value)_subj',
            'a_Intercept_subj', 'a_PCS_Score:sv_pain_para:C(Abs_value)_subj',
            'z_Intercept_subj', 'z_PCS_Score:sv_pain_para:C(Abs_value)_subj']
        titles = [
            'Inter-trial variability in drift rate', 'Inter-trial variability in starting point', 'Inter-trial variability in non-decision time',
            'Intercept drift rate', 'Drift rate interaction (PCS_Score:sv_pain_para:C(Abs_value))',
            'Intercept non-decision time', 'Non-decision time interaction (PCS_Score:sv_pain_para:C(Abs_value))',
            'Intercept boundary separation', 'Boundary separation interaction (PCS_Score:sv_pain_para:C(Abs_value))',
            'Intercept starting point', 'Starting point interaction (PCS_Score:sv_pain_para:C(Abs_value))']
    elif version == 49:  
        params_of_interest = [
            'sv', 'sz', 'st',
            'v_Intercept', 'v_STA_SAI_Score:sv_pain_para:C(acceptance_pair)',
            't_Intercept', 't_STA_SAI_Score:sv_pain_para:C(acceptance_pair)',
            'a_Intercept', 'a_STA_SAI_Score:sv_pain_para:C(acceptance_pair)',
            'z_Intercept', 'z_STA_SAI_Score:sv_pain_para:C(acceptance_pair)']
        params_of_interest_s = [
            'sv_subj', 'sz_subj', 'st_subj',
            'v_Intercept_subj', 'v_STA_SAI_Score:sv_pain_para:C(acceptance_pair)_subj',
            't_Intercept_subj', 't_STA_SAI_Score:sv_pain_para:C(acceptance_pair)_subj',
            'a_Intercept_subj', 'a_STA_SAI_Score:sv_pain_para:C(acceptance_pair)_subj',
            'z_Intercept_subj', 'z_STA_SAI_Score:sv_pain_para:C(acceptance_pair)_subj']
        titles = [
            'Inter-trial variability in drift rate', 'Inter-trial variability in starting point', 'Inter-trial variability in non-decision time',
            'Intercept drift rate', 'Drift rate interaction (STA_SAI_Score:sv_pain_para:C(acceptance_pair))',
            'Intercept non-decision time', 'Non-decision time interaction (STA_SAI_Score:sv_pain_para:C(acceptance_pair))',
            'Intercept boundary separation', 'Boundary separation interaction (STA_SAI_Score:sv_pain_para:C(acceptance_pair))',
            'Intercept starting point', 'Starting point interaction (STA_SAI_Score:sv_pain_para:C(acceptance_pair))']

    elif version == 50:  
        params_of_interest = [
            'sv', 'sz', 'st',
            'v_Intercept', 'v_STA_TAI_Score:sv_pain_para:C(acceptance_pair)',
            't_Intercept', 't_STA_TAI_Score:sv_pain_para:C(acceptance_pair)',
            'a_Intercept', 'a_STA_TAI_Score:sv_pain_para:C(acceptance_pair)',
            'z_Intercept', 'z_STA_TAI_Score:sv_pain_para:C(acceptance_pair)']
        params_of_interest_s = [
            'sv_subj', 'sz_subj', 'st_subj',
            'v_Intercept_subj', 'v_STA_TAI_Score:sv_pain_para:C(acceptance_pair)_subj',
            't_Intercept_subj', 't_STA_TAI_Score:sv_pain_para:C(acceptance_pair)_subj',
            'a_Intercept_subj', 'a_STA_TAI_Score:sv_pain_para:C(acceptance_pair)_subj',
            'z_Intercept_subj', 'z_STA_TAI_Score:sv_pain_para:C(acceptance_pair)_subj']
        titles = [
            'Inter-trial variability in drift rate', 'Inter-trial variability in starting point', 'Inter-trial variability in non-decision time',
            'Intercept drift rate', 'Drift rate interaction (STA_TAI_Score:sv_pain_para:C(acceptance_pair))',
            'Intercept non-decision time', 'Non-decision time interaction (STA_TAI_Score:sv_pain_para:C(acceptance_pair))',
            'Intercept boundary separation', 'Boundary separation interaction (STA_TAI_Score:sv_pain_para:C(acceptance_pair))',
            'Intercept starting point', 'Starting point interaction (STA_TAI_Score:sv_pain_para:C(acceptance_pair))']

    elif version == 51:  
        params_of_interest = [
            'sv', 'sz', 'st',
            'v_Intercept', 'v_PCS_Score:sv_pain_para:C(acceptance_pair)',
            't_Intercept', 't_PCS_Score:sv_pain_para:C(acceptance_pair)',
            'a_Intercept', 'a_PCS_Score:sv_pain_para:C(acceptance_pair)',
            'z_Intercept', 'z_PCS_Score:sv_pain_para:C(acceptance_pair)']
        params_of_interest_s = [
            'sv_subj', 'sz_subj', 'st_subj',
            'v_Intercept_subj', 'v_PCS_Score:sv_pain_para:C(acceptance_pair)_subj',
            't_Intercept_subj', 't_PCS_Score:sv_pain_para:C(acceptance_pair)_subj',
            'a_Intercept_subj', 'a_PCS_Score:sv_pain_para:C(acceptance_pair)_subj',
            'z_Intercept_subj', 'z_PCS_Score:sv_pain_para:C(acceptance_pair)_subj']
        titles = [
            'Inter-trial variability in drift rate', 'Inter-trial variability in starting point', 'Inter-trial variability in non-decision time',
            'Intercept drift rate', 'Drift rate interaction (PCS_Score:sv_pain_para:C(acceptance_pair))',
            'Intercept non-decision time', 'Non-decision time interaction (PCS_Score:sv_pain_para:C(acceptance_pair))',
            'Intercept boundary separation', 'Boundary separation interaction (PCS_Score:sv_pain_para:C(acceptance_pair))',
            'Intercept starting point', 'Starting point interaction (PCS_Score:sv_pain_para:C(acceptance_pair))']

    elif version == 52:  
        params_of_interest = [
            'sv', 'sz', 'st',
            'v_Intercept', 'v_sv_pain_para', 
            'v_STA_SAI_Score', 'v_STA_TAI_Score', 'v_PCS_Score', 
            'v_sv_pain_para:STA_SAI_Score', 'v_sv_pain_para:STA_TAI_Score', 'v_sv_pain_para:PCS_Score', 
            'v_sv_pain_para:STA_SAI_Score:STA_TAI_Score', 'v_sv_pain_para:STA_SAI_Score:PCS_Score', 'v_sv_pain_para:STA_TAI_Score:PCS_Score', 
            'v_sv_pain_para:STA_SAI_Score:STA_TAI_Score:PCS_Score']
        params_of_interest_s = [
            'sv_subj', 'sz_subj', 'st_subj',
            'v_Intercept_subj', 'v_sv_pain_para_subj', 
            'v_STA_SAI_Score_subj', 'v_STA_TAI_Score_subj', 'v_PCS_Score_subj', 
            'v_sv_pain_para:STA_SAI_Score_subj', 'v_sv_pain_para:STA_TAI_Score_subj', 'v_sv_pain_para:PCS_Score_subj', 
            'v_sv_pain_para:STA_SAI_Score:STA_TAI_Score_subj', 'v_sv_pain_para:STA_SAI_Score:PCS_Score_subj', 'v_sv_pain_para:STA_TAI_Score:PCS_Score_subj', 
            'v_sv_pain_para:STA_SAI_Score:STA_TAI_Score:PCS_Score_subj']
        titles = [
            'Inter-trial variability in drift rate', 'Inter-trial variability in starting point', 'Inter-trial variability in non-decision time',
            'Intercept drift rate', 'Drift rate sv_pain_para',
            'Effect of STA_SAI_Score', 'Effect of STA_TAI_Score', 'Effect of PCS_Score', 
            'Interaction: sv_pain_para * STA_SAI_Score', 'Interaction: sv_pain_para * STA_TAI_Score', 'Interaction: sv_pain_para * PCS_Score', 
            'Interaction: sv_pain_para * STA_SAI_Score * STA_TAI_Score', 'Interaction: sv_pain_para * STA_SAI_Score * PCS_Score', 'Interaction: sv_pain_para * STA_TAI_Score * PCS_Score', 
            'Interaction: sv_pain_para * STA_SAI_Score * STA_TAI_Score * PCS_Score']

    elif version == 53:  
        params_of_interest = [
            'sv', 'sz', 'st',
            'v_Intercept', 'v_sv_pain_para', 
            'v_STA_SAI_Score', 'v_STA_TAI_Score', 'v_PCS_Score', 
            'v_sv_pain_para:STA_SAI_Score', 'v_sv_pain_para:STA_TAI_Score', 'v_sv_pain_para:PCS_Score', 
            'v_sv_pain_para:STA_SAI_Score:STA_TAI_Score', 'v_sv_pain_para:STA_SAI_Score:PCS_Score', 'v_sv_pain_para:STA_TAI_Score:PCS_Score', 
            'v_sv_pain_para:STA_SAI_Score:STA_TAI_Score:PCS_Score',
            't_Intercept', 't_sv_pain_para', 
            't_STA_SAI_Score', 't_STA_TAI_Score', 't_PCS_Score', 
            't_sv_pain_para:STA_SAI_Score', 't_sv_pain_para:STA_TAI_Score', 't_sv_pain_para:PCS_Score', 
            't_sv_pain_para:STA_SAI_Score:STA_TAI_Score', 't_sv_pain_para:STA_SAI_Score:PCS_Score', 't_sv_pain_para:STA_TAI_Score:PCS_Score', 
            't_sv_pain_para:STA_SAI_Score:STA_TAI_Score:PCS_Score',
            'a_Intercept', 'a_sv_pain_para', 
            'a_STA_SAI_Score', 'a_STA_TAI_Score', 'a_PCS_Score', 
            'a_sv_pain_para:STA_SAI_Score', 'a_sv_pain_para:STA_TAI_Score', 'a_sv_pain_para:PCS_Score', 
            'a_sv_pain_para:STA_SAI_Score:STA_TAI_Score', 'a_sv_pain_para:STA_SAI_Score:PCS_Score', 'a_sv_pain_para:STA_TAI_Score:PCS_Score', 
            'a_sv_pain_para:STA_SAI_Score:STA_TAI_Score:PCS_Score',
            'z_Intercept', 'z_sv_pain_para', 
            'z_STA_SAI_Score', 'z_STA_TAI_Score', 'z_PCS_Score', 
            'z_sv_pain_para:STA_SAI_Score', 'z_sv_pain_para:STA_TAI_Score', 'z_sv_pain_para:PCS_Score', 
            'z_sv_pain_para:STA_SAI_Score:STA_TAI_Score', 'z_sv_pain_para:STA_SAI_Score:PCS_Score', 'z_sv_pain_para:STA_TAI_Score:PCS_Score', 
            'z_sv_pain_para:STA_SAI_Score:STA_TAI_Score:PCS_Score']
        params_of_interest_s = [
            'sv_subj', 'sz_subj', 'st_subj',
            'v_Intercept_subj', 'v_sv_pain_para_subj', 
            'v_STA_SAI_Score_subj', 'v_STA_TAI_Score_subj', 'v_PCS_Score_subj', 
            'v_sv_pain_para:STA_SAI_Score_subj', 'v_sv_pain_para:STA_TAI_Score_subj', 'v_sv_pain_para:PCS_Score_subj', 
            'v_sv_pain_para:STA_SAI_Score:STA_TAI_Score_subj', 'v_sv_pain_para:STA_SAI_Score:PCS_Score_subj', 'v_sv_pain_para:STA_TAI_Score:PCS_Score_subj', 
            'v_sv_pain_para:STA_SAI_Score:STA_TAI_Score:PCS_Score_subj',
            't_Intercept_subj', 't_sv_pain_para_subj', 
            't_STA_SAI_Score_subj', 't_STA_TAI_Score_subj', 't_PCS_Score_subj', 
            't_sv_pain_para:STA_SAI_Score_subj', 't_sv_pain_para:STA_TAI_Score_subj', 't_sv_pain_para:PCS_Score_subj', 
            't_sv_pain_para:STA_SAI_Score:STA_TAI_Score_subj', 't_sv_pain_para:STA_SAI_Score:PCS_Score_subj', 't_sv_pain_para:STA_TAI_Score:PCS_Score_subj', 
            't_sv_pain_para:STA_SAI_Score:STA_TAI_Score:PCS_Score_subj',
            'a_Intercept_subj', 'a_sv_pain_para_subj', 
            'a_STA_SAI_Score_subj', 'a_STA_TAI_Score_subj', 'a_PCS_Score_subj', 
            'a_sv_pain_para:STA_SAI_Score_subj', 'a_sv_pain_para:STA_TAI_Score_subj', 'a_sv_pain_para:PCS_Score_subj', 
            'a_sv_pain_para:STA_SAI_Score:STA_TAI_Score_subj', 'a_sv_pain_para:STA_SAI_Score:PCS_Score_subj', 'a_sv_pain_para:STA_TAI_Score:PCS_Score_subj', 
            'a_sv_pain_para:STA_SAI_Score:STA_TAI_Score:PCS_Score_subj',
            'z_Intercept_subj', 'z_sv_pain_para_subj', 
            'z_STA_SAI_Score_subj', 'z_STA_TAI_Score_subj', 'z_PCS_Score_subj', 
            'z_sv_pain_para:STA_SAI_Score_subj', 'z_sv_pain_para:STA_TAI_Score_subj', 'z_sv_pain_para:PCS_Score_subj', 
            'z_sv_pain_para:STA_SAI_Score:STA_TAI_Score_subj', 'z_sv_pain_para:STA_SAI_Score:PCS_Score_subj', 'z_sv_pain_para:STA_TAI_Score:PCS_Score_subj', 
            'z_sv_pain_para:STA_SAI_Score:STA_TAI_Score:PCS_Score_subj']
        titles = [
            'Inter-trial variability in drift rate', 'Inter-trial variability in starting point', 'Inter-trial variability in non-decision time',
            'Intercept drift rate', 'Drift rate sv_pain_para',
            'Effect of STA_SAI_Score', 'Effect of STA_TAI_Score', 'Effect of PCS_Score', 
            'Interaction: sv_pain_para * STA_SAI_Score', 'Interaction: sv_pain_para * STA_TAI_Score', 'Interaction: sv_pain_para * PCS_Score', 
            'Interaction: sv_pain_para * STA_SAI_Score * STA_TAI_Score', 'Interaction: sv_pain_para * STA_SAI_Score * PCS_Score', 'Interaction: sv_pain_para * STA_TAI_Score * PCS_Score', 
            'Interaction: sv_pain_para * STA_SAI_Score * STA_TAI_Score * PCS_Score',
            'Intercept non-decision time', 'Non-decision time sv_pain_para',
            'Effect of STA_SAI_Score', 'Effect of STA_TAI_Score', 'Effect of PCS_Score', 
            'Interaction: sv_pain_para * STA_SAI_Score', 'Interaction: sv_pain_para * STA_TAI_Score', 'Interaction: sv_pain_para * PCS_Score', 
            'Interaction: sv_pain_para * STA_SAI_Score * STA_TAI_Score', 'Interaction: sv_pain_para * STA_SAI_Score * PCS_Score', 'Interaction: sv_pain_para * STA_TAI_Score * PCS_Score', 
            'Interaction: sv_pain_para * STA_SAI_Score * STA_TAI_Score * PCS_Score',
            'Intercept boundary separation', 'Boundary separation sv_pain_para',
            'Effect of STA_SAI_Score', 'Effect of STA_TAI_Score', 'Effect of PCS_Score', 
            'Interaction: sv_pain_para * STA_SAI_Score', 'Interaction: sv_pain_para * STA_TAI_Score', 'Interaction: sv_pain_para * PCS_Score', 
            'Interaction: sv_pain_para * STA_SAI_Score * STA_TAI_Score', 'Interaction: sv_pain_para * STA_SAI_Score * PCS_Score', 'Interaction: sv_pain_para * STA_TAI_Score * PCS_Score', 
            'Interaction: sv_pain_para * STA_SAI_Score * STA_TAI_Score * PCS_Score',
            'Intercept starting point', 'Starting point sv_pain_para',
            'Effect of STA_SAI_Score', 'Effect of STA_TAI_Score', 'Effect of PCS_Score', 
            'Interaction: sv_pain_para * STA_SAI_Score', 'Interaction: sv_pain_para * STA_TAI_Score', 'Interaction: sv_pain_para * PCS_Score', 
            'Interaction: sv_pain_para * STA_SAI_Score * STA_TAI_Score', 'Interaction: sv_pain_para * STA_SAI_Score * PCS_Score', 'Interaction: sv_pain_para * STA_TAI_Score * PCS_Score', 
            'Interaction: sv_pain_para * STA_SAI_Score * STA_TAI_Score * PCS_Score']
    
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
    size_plot = len(combined_model.data.subj_idx.unique()) / 3.0 * 1.5
    combined_model.plot_posterior_predictive(samples=10, bins=100, figsize=(6, size_plot), save=True, path=os.path.join(fig_dir, 'diagnostics'), format='pdf')
    
    # Adjust text size
    matplotlib.rcParams.update({'font.size': 6}) 
    combined_model.plot_posteriors(save=True, path=os.path.join(fig_dir, 'diagnostics'), format='pdf')
    
    # Reset to default
    matplotlib.rcParams.update({'font.size': 12})

    # Point estimates and results
    results = combined_model.gen_stats()
    results.to_csv(os.path.join(fig_dir, 'diagnostics', 'results.csv'))
    
    # Posterior analysis and fixed starting point as in J.W. de Gee code
    traces = [combined_model.nodes_db.node[p].trace() for p in params_of_interest]
    traces[0] = 1 / (1 + np.exp(-(traces[0])))

    #Posterior Statistics for parameter traces, significance testing
    stats = []
    for trace in traces:
        stat = min(np.mean(trace > 0), np.mean(trace < 0))
        stats.append(min(stat, 1 - stat))
    stats = np.array(stats)
    
    # 5 columns for plots
    n_cols = 5
    n_rows = int(np.ceil(len(params_of_interest) / n_cols))

    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(n_cols * 3, n_rows * 4))
    axes = axes.flatten()

    for ax_nr, (trace, title) in enumerate(zip(traces, titles)):
        sns.kdeplot(trace, vertical=True, shade=True, color='purple', ax=axes[ax_nr])
        if ax_nr % n_cols == 0:
            axes[ax_nr].set_ylabel('Parameter estimate (a.u.)')
        if ax_nr >= len(params_of_interest) - n_cols:
            axes[ax_nr].set_xlabel('Posterior probability')
        axes[ax_nr].set_title(f'{title}\np={round(stats[ax_nr], 3)}', fontsize=6)  
        axes[ax_nr].set_xlim(xmin=0)
        for axis in ['top', 'bottom', 'left', 'right']:
            axes[ax_nr].spines[axis].set_linewidth(0.5)
            axes[ax_nr].tick_params(width=0.5, labelsize=6)  

    for ax in axes[len(params_of_interest):]:
        fig.delaxes(ax)

    sns.despine(offset=10, trim=True)
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'posteriors.pdf'), bbox_inches='tight') 

        
    parameters = []
    for p in params_of_interest_s:
        param_values = []
        for s in np.unique(combined_model.data.subj_idx):
            param_name = f"{p}.{s}"
            try:
                param_value = results.loc[results.index == param_name, 'mean'].values
                if len(param_value) > 0:
                    param_values.append(param_value[0])
            except KeyError:
                print(f"Param {param_name} missing. Skipping...")
                continue
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