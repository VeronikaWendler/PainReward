# multiplying the questionnaire data scores with the sv_pain to get the interaction term which can be used in the EEG regression model

import pandas as pd
import numpy as np 
import csv 


data_path = 'D:/Aberdeen_Uni_June24/MPColl_Lab/All_Files_Relevant_For_Git/Hddm_Docker_August_24/data_sets/data_with_full_sv_pain_para_contrib.csv'
data = pd.read_csv(data_path, sep=',')


def sv_pain_para_Quest_contributions(data):
    data_sv_pain_quest = data.copy()
    data_sv_pain_quest['sv_pain_para_SAI'] = np.nan
    data_sv_pain_quest['sv_pain_para_TAI'] = np.nan
    data_sv_pain_quest['sv_pain_para_PCS'] = np.nan

    for part in data['participant'].unique():
        subj_data = data[data['participant'] == part]
        
        sv_pain_para_SAI_contrib_list = []
        sv_pain_para_TAI_contrib_list = []
        sv_pain_para_PCS_contrib_list = []

        for idx, trial in subj_data.iterrows():
            trial_sv_pain_para = trial['sv_pain_para'] 
            trial_SAI = trial['STA_SAI_Score']
            trial_TAI = trial['STA_TAI_Score']
            trial_PCS = trial['PCS_Score']

            trial_SAI_contrib = trial_SAI * trial_sv_pain_para
            trial_TAI_contrib = trial_TAI * trial_sv_pain_para
            trial_PCS_contrib = trial_PCS * trial_sv_pain_para
            
            sv_pain_para_SAI_contrib_list.append(trial_SAI_contrib)
            data_sv_pain_quest.loc[idx, 'sv_pain_para_SAI'] = trial_SAI_contrib  
            sv_pain_para_TAI_contrib_list.append(trial_TAI_contrib)
            data_sv_pain_quest.loc[idx, 'sv_pain_para_TAI'] = trial_TAI_contrib  
            sv_pain_para_PCS_contrib_list.append(trial_PCS_contrib)
            data_sv_pain_quest.loc[idx, 'sv_pain_para_PCS'] = trial_PCS_contrib  

    return data_sv_pain_quest

sv_contribute = sv_pain_para_Quest_contributions(data)
sv_contribute.to_csv('D:/Aberdeen_Uni_June24/MPColl_Lab/All_Files_Relevant_For_Git/Hddm_Docker_August_24/data_sets/data_with_sv_pain_para_Quest.csv')

