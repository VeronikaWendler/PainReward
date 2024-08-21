# Preparing DataFrame so it can be directly loaded into hddm environments

import pandas as pd
import matplotlib.pyplot as plt
import hddm

def prepare_data(file_path):
    data = pd.read_csv(file_path, sep=";")
    
    # Renaming column headers so they fit the model
    data["rt"] = data["choice_resp.rt"]
    data["response"] = data["accepted"]
    data['moneylevel'] = pd.to_numeric(data['moneylevel'], errors='coerce')
    data['painlevel'] = pd.to_numeric(data['painlevel'], errors='coerce')
    data['response'] = pd.to_numeric(data['response'], errors='coerce')
    data['rt'] = pd.to_numeric(data['rt'], errors='coerce')
    data["subj_idx"] = data['participant']
    # drop nanas
    data = data.dropna(subset=['moneylevel', 'painlevel', 'response', 'rt', 'choice_resp.keys'])
    
    # Flipping Errors
    data = hddm.utils.flip_errors(data)
    
    # Plotting RT distributions
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, xlabel='RT', ylabel='count', title='RT distributions')
    for i, subj_data in data.groupby('subj_idx'):
        subj_data.rt.hist(bins=20, histtype='step', ax=ax)
    plt.show()
    
    # Creating stimulus categories to be used in the DDM
    stim_categories = []
    data['acceptance_pair'] = ''
    for index, row in data.iterrows():
        money = row['moneylevel']
        pain = row['painlevel']
        if money == pain:
            category = 'I'
        elif money > pain:
            category = 'M'
        else:
            category = 'P'
        stim_categories.append(category)
    data['acceptance_pair'] = stim_categories
    
    # Defining Overall Value (OV) and Absolute Value (abs)
    OV_cat = []
    Abs_cat = []
    data['OV_value'] = ''
    data['Abs_value'] = ''
    for index, row in data.iterrows():
        money = row['moneylevel']
        pain = row['painlevel']
        if money + pain <= 5:
            OV = 'low_OV'
        else:
            OV = 'high_OV'
        OV_cat.append(OV)
    data['OV_value'] = OV_cat
    
    for index, row in data.iterrows():
        money = row['moneylevel']
        pain = row['painlevel']
        if abs(money - pain) < 2:
            Ab = 'low_abs'
        elif abs(money - pain) > 2:
            Ab = 'high_abs'
        else:
            Ab = 'mid_abs'
        Abs_cat.append(Ab)
    data['Abs_value'] = Abs_cat
    
    # High and Low Pain and Money in OV and Abs
    OV_cat = []
    Abs_cat = []
    data['OV_Money_Pain'] = ''
    data['Abs_Money_Pain'] = ''
    for index, row in data.iterrows():
        money = row['moneylevel']
        pain = row['painlevel']
        if money + pain > 5 and money > pain:
            OV = 'h_OV_h_money'
        elif money + pain > 5 and pain >= money:
            OV = 'h_OV_h_pain'
        elif money + pain <= 5 and money > pain:
            OV = 'low_OV_h_money'
        else:
            OV = 'low_OV_h_pain'
        OV_cat.append(OV)
    data['OV_Money_Pain'] = OV_cat
    
    for index, row in data.iterrows():
        money = row['moneylevel']
        pain = row['painlevel']
        if abs(money - pain) < 2 and money > pain:
            Ab = 'low_abs_h_money'
        elif abs(money - pain) < 2 and pain >= money:
            Ab = 'low_abs_h_pain'
        elif abs(money - pain) > 2 and money > pain:
            Ab = 'high_abs_h_money'
        elif abs(money - pain) > 2 and pain >= money:
            Ab = 'high_abs_h_pain'
        else:
            Ab = 'mid_abs'
        Abs_cat.append(Ab)
    data['Abs_Money_Pain'] = Abs_cat
    
    return data