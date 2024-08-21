# preparing and concatenating the dataframes to one dataframe that will be used in the modelling analysis

import pandas as pd
import numpy as np
import hddm
import os, sys, pickle, time
# import own libraries
current_directory = os.getcwd()
hddm_models_path = os.path.join(current_directory, 'DockerData', 'Hddm_models')
sys.path.append(hddm_models_path)

# Import the prepare_data function from helper_functions
from helper_functions import prepare_data
current_directory = os.getcwd()

# load the data
data_subs_path = os.path.join(current_directory, 'data_sets', 'subs_concatenated_001_050_3.csv')
data_fit_path = os.path.join(current_directory, 'data_sets', 'fit_models.xlsx')
data_subs = prepare_data(data_subs_path)
data_fit = pd.read_excel(data_fit_path, engine='openpyxl')

# remove missing data to get both frames to the same size
data_subs['moneylevel'] = pd.to_numeric(data_subs['moneylevel'], errors='coerce')
data_subs['painlevel'] = pd.to_numeric(data_subs['painlevel'], errors='coerce')
data_subs['accepted'] = pd.to_numeric(data_subs['accepted'], errors='coerce')
data_subs['choice_resp.rt'] = pd.to_numeric(data_subs['choice_resp.rt'], errors='coerce')
data_subs["painlevel"] = data_subs["painlevel"].astype(int)
data_subs["moneylevel"] = data_subs["moneylevel"].astype(int)
data_subs['subj_idx'] = data_subs['subj_idx'].astype('category')
data_subs.dropna(subset=["painlevel", "moneylevel", "accepted", "choice_resp.rt", 'choice_resp.keys'], inplace = True)

# select columns of interest
columns_app_data_fit = ['fixduration', 'moneylevel', 'sv_money', 'sv_pain', 'sv_both', 'p_pain_all']
data_fit_2 = data_fit[columns_app_data_fit]

for column in columns_app_data_fit:
    data_fit_2[column] = pd.to_numeric(data_fit_2[column], errors='coerce')


# reset index before concatenating data frames
data_subs.reset_index(drop=True, inplace=True)
data_fit_2.reset_index(drop=True, inplace=True)

data = pd.concat([data_subs, data_fit_2], axis=1)


output_file_path = os.path.join(current_directory, 'data_sets', 'behavioural_sv_cleaned.csv')
data.to_csv(output_file_path, index=False)