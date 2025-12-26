'''
 # @ : -*- coding: utf-8 -*-
 # @ Author: Michel-Pierre Coll (michel-pierre.coll@psy.ulaval.ca), edited by Veronika Wendler (wendler.vroni@gmail.com)
 # @ Date: 2024
 # @ Description: Using the cleaned data from 03-painreward_eeg_preprocess to create ERPs and TFR
 '''

# importing libraries
from mne.report import Report
import pprint
import mne
import os
from os.path import join as opj
import pandas as pd
import numpy as np
from mne.viz import plot_evoked_joint as pej
from bids import BIDSLayout
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
import os
from scipy.stats import pearsonr
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from statsmodels.distributions.empirical_distribution import ECDF
from pathlib import Path


# set the version to run (either decision or passive phase)
version = 1    # 1 = decision, 2 = passive

# what to lock to: 'cue' (off+) or 'response'
lock_type = 'response'       # or 'response'

erp_mode = "classic_rp"        # Gluth 2013

# Set bids directory
PROJECT_DIR = Path(os.getenv("PROJECT_DIR", "/workspace"))
basepath = PROJECT_DIR / "EEG" / "PainReward_sub-001-050" / "painrewardeegdata"
def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)
import re
from pathlib import Path
import os
layout = BIDSLayout(basepath)
# disable Numba JIT caching & compilation
#os.environ["NUMBA_DISABLE_JIT"] = "1"
import numba
numba.config.CACHE_ENABLE = False
outpath = opj(basepath, "derivatives")
os.makedirs(outpath, exist_ok=True)
# List participants
part = [p for p in os.listdir(opj(basepath)) if "sub" in p]
part.sort()

# defining parameters for ERPs
# param = {
#     # Additional LP filter fora ERPs
#     'erplpfilter': 30,
#     # Filter to use
#     'filtertype': 'fir',
#     # Length of baseline
#     'erpbaseline': -0.2,
#     'erpepochend': 1,
#     # Threshold to reject trials
#     'erpreject': dict(eeg=150e-6),
#     # Threshold to reject shock trials
#     'erprejectshock': dict(eeg=150e-6),
  
# }

# this is for the classic_rp script, otherwise use the above
param = {
    "filtertype": "fir",
    "erpreject": dict(eeg=150e-6),
    'erprejectshock': dict(eeg=150e-6),
    "hp": None,
    "lp": 30,
    "erpbaseline": -0.2,
    "erpepochend": 1.0,
}
if erp_mode == "classic_rp":
    param["hp"] = 0.1
    param["lp"] = 10
    print('lp = 10')
    lock_type = "response"          
    param["erpbaseline"] = -0.8   
    param["erpepochend"] = 0.2




if version == 1 and lock_type == 'response':
    param['erpbaseline'] = -0.8
    param['erpepochend'] = 0.2

#-----------------------------------------------------------------------------------------------------------------------
# epoching erps
# reject_stats = pd.DataFrame(data={'part': part, 'perc_removed_cues': 9999,
#                                   'perc_removed_shocks': 9999,
#                                    "Off+": 0,
#                                 #    "dIN8": 0,
#                                 #    "Res+": 0,
#                                 #    "Fix+": 0,
#                                 #    "Fee+": 0,
#                                 #    "Fee-": 0,
#                                 #    "Fix+": 0,
#                                 #    "Cdow": 0,
#                                 #    "Shk-": 0,
#                                   })

# col_name = 'Off+'
# if lock_type == 'response':
#     col_name = 'Resp_any'

# elif lock_type == 'cue':
#     count_col = 'Off+'      # number of off+ epochs kept
# else:
#     count_col = 'Resp_any'  # number of response-locked epochs kept

if lock_type == 'cue':
    count_col = 'Off+'
else:
    count_col = 'Resp_any'


reject_stats = pd.DataFrame(data={
    'part': part,
    'perc_removed_cues': 9999,
    'perc_removed_shocks': 9999,
    count_col: 0,
})



for p in part:
    # 
    # Make out dir
    indir = opj(outpath,  p, 'eeg')
    # erp dircetory
    

    if version == 1:      
        if lock_type == 'cue':
            outdir = opj(outpath,  p, 'eeg', 'erps')  
            if not os.path.exists(outdir):
                os.mkdir(outdir)
            print("version 1: outdir = opj(outpath,  p, 'eeg', 'erps')")
        elif lock_type == 'response':
            if erp_mode == "classic_rp":
                outdir = opj(outpath, p, "eeg", "erps_resp_rp")
                if not os.path.exists(outdir):
                    os.mkdir(outdir)
            else:
                outdir = opj(outpath, p, "eeg", "erps_resp")
                if not os.path.exists(outdir):
                    os.mkdir(outdir)
        else:
            raise ValueError("lock_type must be 'cue' or 'response'")

    elif version == 2:
        outdir = opj(outpath,  p, 'eeg', 'erps_passive')              
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        print("version 2: outdir = opj(outpath,  p, 'eeg', 'erps_passive')")
    else:
        print("No version")        
        


    # _______________________________________________________
    # Initialise MNE report
    report = Report(verbose=False, subject=p,
                    title='ERP report for part ' + p)
    report.add_html(pprint.pformat(param),
                    title='Parameters',
                    section='Parameters')
  
    # ______________________________________________________
    # Load cleaned raw file (for decision phase) or for the passive phase
    if version == 1:
        raw = mne.io.read_raw_fif(opj(indir,
                                  p + '_decision_cleaned-raw.fif'),
                              preload=True)
        print("Version 1: _decision_cleaned-raw.fif")
    elif version == 2:
        raw = mne.io.read_raw_fif(opj(indir,
                                  p + '_passive_cleaned-raw.fif'),
                              preload=True)
        print("Version 2: passive_cleaned-raw.fif")
    else:
        print("No version")
        
    subject_i = p.split('-')[-1]
    # Load trial info in scr data
    events = pd.read_csv(layout.get(subject=subject_i, extension='tsv',
                                    suffix='events',
                                    return_type='filename')[0], sep='\t')
  
    # Drop unused channels
    chans_to_drop = [c for c in ['HEOGL', 'HEOGR', 'VEOGL',
                                 'STI 014', 'Status'] if c in raw.ch_names]
    raw.drop_channels(chans_to_drop)

    # # Filter for erpss
    # raw = raw.filter(
    #     None,
    #     param['erplpfilter'],
    #     method=param['filtertype'])
    raw = raw.filter(
    l_freq=param["hp"],
    h_freq=param["lp"],
    method=param["filtertype"]
    )

    # Add empty column to make it easier to create the event array
    events['empty'] = 0
    events_c = events[events['trial_type'].notna()]

    
    if lock_type == 'cue':
        # CUE-LOCKED (existing behaviour)
        events_id = {"off+": 2}
        events_c = events_c[events_c['trial_type'] == 'off+']
        events_c['cue_num'] = events_c['trial_type'].map(events_id)
        events_epoch = np.asarray(events_c[['sample', 'empty', 'cue_num']])
    
    elif lock_type == 'response':
        # RESPONSE-LOCKED: any response event (accept, reject, miss)
        events_id = {"resp_any": 3}   # arbitrary code, just one condition
        events_c = events_c[events_c['trial_type'].isin(['res+', 'res-', 'resm'])]
        events_c['cue_num'] = events_id['resp_any']
        events_epoch = np.asarray(events_c[['sample', 'empty', 'cue_num']])
    else:
        raise ValueError("lock_type must be 'cue' or 'response'")

    
    
    # events_c = events_c[events_c['trial_type'] != 'DIN7']
    # events_c = events_c[events_c['trial_type'] != 'RSTR']
    # valid_trial_types = ["off+", "DIN8", "res+", "fix+", "fee+", "fee-", "fix+", "cdow", "shk-"]
    # events_c = events_c[events_c['trial_type'].isin(valid_trial_types)]
  
    # #------------------------------------------------------------------------------------------------------
    # # # Epoch around  off+
    # events_id = {
    #     "off+": 2,
    #     # "DIN8": 1,
    #     # "res+": 3,
    #     # "fix+": 6,
    #     # "fee+": 7,
    #     # "fee-": 8,
    #     # "fix+": 9,
    #     # "cdow": 10,
    #     # "shk-": 11,
    # }
    # #events_c['cue_num'] = [events_id[s] for s in events_c.trial_type]
    # events_c = events_c[events_c['trial_type'] == 'off+']
    # events_c['cue_num'] = events_c['trial_type'].map(events_id)
    # events_cues = np.asarray(events_c[['sample', 'empty', 'cue_num']])
  
    # events_c['cue_num'] = [events_id[s] for s in events_c.trial_type]
    # events_cues = np.asarray(events_c[['sample', 'empty', 'cue_num']])
    # erp_cues = mne.Epochs(
    #     raw,
    #     events=events_cues,
    #     event_id=events_id,
    #     tmin=param['erpbaseline'],
    #     baseline=(param['erpbaseline'], 0),
    #     tmax=param['erpepochend'],
    #     preload=True,
    #     verbose=False,
    #     reject=param['erpreject']
    # )
    
    if lock_type == "response":
        baseline = (-0.8, -0.7)
    else:
        baseline = (param["erpbaseline"], 0)
    
    # erp_cues = mne.Epochs(
    #     raw,
    #     events=events_epoch,
    #     event_id=events_id,
    #     tmin=param['erpbaseline'],
    #     baseline=(param['erpbaseline'], 0),
    #     tmax=param['erpepochend'],
    #     preload=True,
    #     verbose=False,
    #     reject=param['erpreject']
    # )
    
    erp_cues = mne.Epochs(
        raw,
        events=events_epoch,
        event_id=events_id,
        tmin=param["erpbaseline"],
        tmax=param["erpepochend"],
        baseline=baseline,
        preload=True,
        verbose=False,
        reject=param["erpreject"],
        )


    fig = mne.viz.plot_drop_log(erp_cues.drop_log, show=False)
    report.add_figure(fig, title='Drop log', section='Drop log')
    reject_stats.loc[reject_stats.part == p,
                     reject_stats.columns == 'perc_removed_cues'] = ((125 - len(erp_cues)) / 125 * 100)
        
    if lock_type == 'cue':
        # number of off+ trials kept
        reject_stats.loc[reject_stats.part == p, count_col] = len(erp_cues['off+'])
    else:
        # number of response-locked epochs kept
        reject_stats.loc[reject_stats.part == p, count_col] = len(erp_cues)

    # reject_stats.loc[reject_stats.part == p, reject_stats.columns == 'dIN8'] = len(erp_cues['DIN8'])
    # reject_stats.loc[reject_stats.part == p, reject_stats.columns == 'Res+'] = len(erp_cues['res+'])
    # reject_stats.loc[reject_stats.part == p, reject_stats.columns == 'Fix+'] = len(erp_cues['fix+'])
    # reject_stats.loc[reject_stats.part == p, reject_stats.columns == 'Fee+'] = len(erp_cues['fee+'])
    # reject_stats.loc[reject_stats.part == p, reject_stats.columns == 'Fee-'] = len(erp_cues['fee-'])
    # reject_stats.loc[reject_stats.part == p, reject_stats.columns == 'Fix+'] = len(erp_cues['fix+'])
    # reject_stats.loc[reject_stats.part == p, reject_stats.columns == 'Cdow'] = len(erp_cues['cdow'])
    # reject_stats.loc[reject_stats.part == p, reject_stats.columns == 'Shk-'] = len(erp_cues['shk-'])

    # We create evokeds 
    # Average across trials and plot
    if version == 1:
        figs_butter = []
        evokeds = dict()
        for cond in events_id.keys():
            evokeds[cond] = erp_cues[cond].average()
            figs_butter.append(
                pej(evokeds[cond],
                    title=cond,
                    show=False,
                    picks='eeg',
                    exclude=['HEOGL', 'HEOGR', 'VEOGL'],
                    ts_args={'time_unit': 'ms'},
                    topomap_args={'time_unit': 'ms'})
            )
    
            if lock_type == "cue":
                fname = f"{p}_decision_{cond}_ave.fif"   # off+
            
            elif lock_type == "response" and erp_mode != "classic_rp":
                fname = f"{p}_decision_resp_{cond}_ave.fif"   # standard response-locked
            
            elif lock_type == "response" and erp_mode == "classic_rp":
                fname = f"{p}_decision_resp_rp_{cond}_ave.fif"  # classic RP
            
            else:
                raise ValueError("Invalid lock_type / erp_mode combination")
            evokeds[cond].save(opj(outdir, fname), overwrite=True)
    
        # choose which condition to show in images
        first_cond = list(events_id.keys())[0]  # 'off+' or 'resp_any'
    
        if lock_type == "cue":
            section = "ERPs for cue off+"
            title = "Butterfly plots for off+"
            report_name = f"{p}_decision_cue_erps_report.html"
        
        elif lock_type == "response" and erp_mode != "classic_rp":
            section = "ERPs for responses"
            title = "Butterfly plots for responses"
            report_name = f"{p}_decision_resp_erps_report.html"
        
        elif lock_type == "response" and erp_mode == "classic_rp":
            section = "ERPs for classic response potential (RP)"
            title = "Butterfly plots for classic RP (response-locked)"
            report_name = f"{p}_decision_resp_rp_erps_report.html"
        
        else:
            raise ValueError("Invalid lock_type / erp_mode combination")
    
        report.add_figure(figs_butter, section=section, title=title)
    
        # image for main condition
        fig_img = evokeds[first_cond].plot_image(picks="eeg")
        report.add_figure(fig_img, section=section, title='plot_image')
    
        # Plot some channels and add to report
        chans_to_plot = ['Fz', 'FCz', 'POz', 'Cz', 'CPz', 'Pz', 'Oz']
        figs_chan = []
        for c in chans_to_plot:
            pick = erp_cues.ch_names.index(c)
            figs_chan.append(
                mne.viz.plot_compare_evokeds(evokeds, picks=pick, show=False)[0]
            )
        report.add_figure(figs_chan, section=section, title='Cues/chans')
    
        report.save(opj(outdir, report_name), open_browser=False, overwrite=True)

        
    elif version == 2:
        figs_butter = []
        evokeds = dict()
        for cond in events_id.keys():
            evokeds[cond] = erp_cues[cond].average()                   
            figs_butter.append(pej(evokeds[cond],
                                   title=cond,
                                   show=False,
                                   picks='eeg',
                                   exclude=['HEOGL', 'HEOGR', 'VEOGL'],
                                   ts_args={'time_unit': 'ms'},
                                   topomap_args={'time_unit': 'ms'}))
            evokeds[cond].save(opj(outdir, p + '_passive_' + cond
                                   + '_ave.fif'), overwrite=True)
        report.add_figure(figs_butter,
                          section='ERPs for cues off+',
                          title='Butterfly plots for cues off+')
        
        print("ERPs for cues off+ for Version 2")
        # Adding plot_image
        #off_fig = mne.viz.plot_image(evokeds['off+'], show=False)
        off_fig = evokeds['off+'].plot_image(picks="eeg")
        report.add_figure(off_fig, section='ERPs for cues off+', title='plot_image for off+')
        # Plot some channels and add to report
        chans_to_plot = ['Fz', 'FCz', 'POz', 'Cz', 'CPz', 'Pz', 'Oz' ]
        
        figs_chan = []                                                          
        for c in chans_to_plot:                                               
            pick = erp_cues.ch_names.index(c)
            figs_chan.append(mne.viz.plot_compare_evokeds(evokeds, picks=pick,
                                                          show=False)[0])
        report.add_figure(figs_chan,
                          section='ERPs for cues off+', title='Cues/chans')
        
        report.save(opj(outdir,  p + '_passive_erps_report.html'),
                    open_browser=False, overwrite=True)
        print("report.save(opj(outdir,  p + '_passive_erps_report.html')")
        
    #
    # Single trials for cues
    events_c['trialsnum'] = range(1, 126)
    events_c['trials_name'] = ['trial_' + str(s).zfill(3)
                               for s in range(1, 126)]
    events_c['participant_id'] = p
    events_cues = np.asarray(events_c[['sample', 'empty', 'trialsnum']])
    trials_dict = dict()
    for idx, rows in events_c.iterrows():
        trials_dict[rows['trials_name']] = rows['trialsnum']
    erp_cues_single = mne.Epochs(
        raw,
        events=events_cues,
        event_id=trials_dict,
        tmin=param['erpbaseline'],
        baseline=baseline,
        tmax=param['erpepochend'],
        metadata=events_c,
        preload=True,
        verbose=True)

    # Add bad trials to metadata
    strials_drop = erp_cues_single.copy()
    strials_drop.drop_bad(reject=param['erpreject'])
    badtrials = [1 if len(li) > 0 else 0 for li in strials_drop.drop_log]
    erp_cues_single.metadata['badtrial'] = badtrials
    
    if version == 1:
        if lock_type == "cue":
            fname = f"{p}_decision_cues_singletrials-epo.fif"
            print("Saving cue-locked single trials:", fname)
    
        elif lock_type == "response":
            suffix = "_rp" if erp_mode == "classic_rp" else ""
            fname = f"{p}_decision_resp{suffix}_singletrials-epo.fif"
            print("Saving response-locked single trials:", fname)
    
        else:
            raise ValueError("lock_type must be 'cue' or 'response'")
    
        erp_cues_single.save(opj(outdir, fname), overwrite=True)
    
    elif version == 2:
        fname = f"{p}_passive_cues_singletrials-epo.fif"
        erp_cues_single.save(opj(outdir, fname), overwrite=True)
        print("Saving passive single trials:", fname)
    
    else:
        print("no version")
    plt.close('all')
    #-------------------------------------------------------------------------------------------

# Save rejection stats
if version == 1 and lock_type == "cue":
    reject_stats["perc_removed_all"] = (1 - reject_stats[count_col] / 125) * 100
    reject_stats.to_csv(opj(outpath, "decision_cue_erps_rejectionstats.csv"), index=False)
    reject_stats.describe().to_csv(opj(outpath, "decision_cue_erps_rejectionstats_desc.csv"))
    print("Saved decision cue ERP rejection stats.")

elif version == 1 and lock_type == "response":
    reject_stats["perc_removed_all"] = (1 - reject_stats[count_col] / 125) * 100

    suffix = "_rp" if erp_mode == "classic_rp" else ""
    reject_stats.to_csv(opj(outpath, f"decision_resp{suffix}_erps_rejectionstats.csv"), index=False)
    reject_stats.describe().to_csv(opj(outpath, f"decision_resp{suffix}_erps_rejectionstats_desc.csv"))
    print(f"Saved decision response ERP rejection stats{suffix}.")

elif version == 2:
    reject_stats["perc_removed_all"] = (1 - reject_stats[["Off+"]].sum(axis=1) / 125) * 100
    reject_stats.to_csv(opj(outpath, "passive_erps_rejectionstats.csv"), index=False)
    reject_stats.describe().to_csv(opj(outpath, "passive_erps_rejectionstats_desc.csv"))
    print("Saved passive ERP rejection stats.")

else:
    print("No version")
    
    
    
    
def average_time_win_strials(strials, chans_to_average, amp_lat):
    """Extract mean amplitude between fixed latencies at specified channels
    Parameters
    ----------
    strials : mne Epochs
        MNE epochs data with metadata
    chans_to_average : list
        Channels to include in the average
    amp_lat : list of lists
        Latencies of the segment to average
    Returns
    ---------
    mne Epochs
        Epochs with metadata updated with amplitude columns
    """
    for c in chans_to_average:
        for a in amp_lat:
            ampsepoch = strials.copy()
            # Crop epochs around latencies and drop unused channels
            ampsepoch.crop(tmin=a[0], tmax=a[1])
            ampsepoch.pick_channels(c)
            all_amps = []
            for idx, data in enumerate(ampsepoch.get_data()):
                amp = np.average(data)
                all_amps.append(amp)
            # Normalize across trials (optional, can be removed if not needed)
            all_amps = (all_amps - np.mean(all_amps)) / np.std(all_amps)
            # Update metadata with the calculated amplitudes
            strials.metadata['amp_' + '_'.join(c) + '_' + str(a[0]) + '-'
                             + str(a[1])] = all_amps
    return strials


# Parameters to define
chans_to_average = [['Fz'], ['FCz'], ['POz'], ['Cz'], ['CPz'], ['Pz'], ['Oz']]    #for averaging over more channels: [['F3'], ['F4'], ['Fz'], ['FC5'], ['FC6'], ['FC1'], ['FC2'], ['FCz'], ['C3'], ['C4'], ['CP1'], ['CP2'], ['CP5'], ['CP6'], ['P3'], ['Pz'], ['P4'], ['P7'], ['P8'], ['PO3'], ['PO7'], ['PO4'], ['O1'], ['Oz'], ['O2']]                                                                          

if version == 1 and lock_type == "response":
    amp_lat = [[-0.5, 0.0]] if erp_mode == "classic_rp" else [[-0.5, 0.1]]
    print(f"amp_lat = {amp_lat[0][0]} -> {amp_lat[0][1]} (response-locked, {erp_mode})")
else:
    amp_lat = [[0.4, 0.8]]
    print("amp_lat = 0.4 -> 0.8 (cue-locked / LPP)")


#LPP is centro-parietal  (currently our focus for pain anticipation or (also) value-based choice)
# N2 is fronto-central   (not looking at this currently)
# P3 is centro-parietal 
# Pain-related N1 is occipito-parietal (not looking at this atm)
if version == 1:
    all_meta = []

    for p in part:
        if lock_type == "cue":
            outdir = opj(outpath, p, "eeg", "erps")
            epo_fname = f"{p}_decision_cues_singletrials-epo.fif"
            meta_outname = "decision_erpsmeta_cue.csv"

        elif lock_type == "response":
            if erp_mode == "classic_rp":
                outdir = opj(outpath, p, "eeg", "erps_resp_rp")
                epo_fname = f"{p}_decision_resp_rp_singletrials-epo.fif"
                meta_outname = "decision_erpsmeta_response_classic_rp.csv"
            else:
                outdir = opj(outpath, p, "eeg", "erps_resp")
                epo_fname = f"{p}_decision_resp_singletrials-epo.fif"
                meta_outname = "decision_erpsmeta_response.csv"
        else:
            raise ValueError("lock_type must be 'cue' or 'response'")

        epo = mne.read_epochs(opj(outdir, epo_fname))

        # add amplitudes
        epo = average_time_win_strials(epo, chans_to_average, amp_lat)

        # add participant ID
        epo.metadata["participant_id"] = p
        all_meta.append(epo.metadata)

    all_meta = pd.concat(all_meta, ignore_index=True)
    all_meta.to_csv(opj(outpath, meta_outname), index=False)

elif version == 2:
    all_meta = []
    for p in part:
        outdir = opj(outpath, p, "eeg", "erps_passive")
        epo = mne.read_epochs(opj(outdir, f"{p}_passive_cues_singletrials-epo.fif"))

        epo = average_time_win_strials(epo, chans_to_average, amp_lat)
        epo.metadata["participant_id"] = p
        all_meta.append(epo.metadata)

    all_meta = pd.concat(all_meta, ignore_index=True)
    all_meta.to_csv(opj(outpath, "passive_erpsmeta.csv"), index=False)

else:
    raise ValueError("version must be 1 or 2")
    
# ## #------------------------------------------------------------------------------------------------------------------
# ## # TFR 
# #---------------------------------------------------------------------------------------------------------------------
# # TFR specific libs

# from mne.report import Report
# import pprint
# import mne
# import os
# from os.path import join as opj
# import pandas as pd
# import numpy as np
# from mne.viz import plot_evoked_joint as pej
# from bids import BIDSLayout
# import matplotlib.pyplot as plt
# from tqdm import tqdm
# import seaborn as sns
# import os
# from scipy.stats import pearsonr
# import warnings
# warnings.simplefilter(action='ignore', category=FutureWarning)
# from statsmodels.distributions.empirical_distribution import ECDF
# from pathlib import Path
# from mne.time_frequency import tfr_morlet

# # Set bids directory
# PROJECT_DIR = Path(os.getenv("PROJECT_DIR", "/workspace"))
# basepath = PROJECT_DIR / "EEG" / "PainReward_sub-001-050" / "painrewardeegdata"

# def ensure_dir(path):
#     Path(path).mkdir(parents=True, exist_ok=True)
# import re
# from pathlib import Path
# import os

# layout = BIDSLayout(basepath)

# # disable Numba JIT caching & compilation
# #os.environ["NUMBA_DISABLE_JIT"] = "1"
# import numba
# numba.config.CACHE_ENABLE = False

# outpath = opj(basepath, "derivatives")
# os.makedirs(outpath, exist_ok=True)

# # List participants
# part = [p for p in os.listdir(opj(basepath)) if "sub" in p]
# part.sort()


# tfr_param = {
#     'tfrbaseline': -0.50,      # crop start (what you keep in the final TFR)
#     'tfrcropend': 1.0,         # crop end
#     'tfrepochstart': -2.0,     # epoch start around event (for TF transform)
#     'tfrepochend': 2.0,        # epoch end
#     'ttfreqs': np.arange(4, 101, 1),
#     'n_cycles': 0.5 * np.arange(4, 101, 1),
#     'testresampfreq': 256,
#     'njobs': 8,
# }

# removed_frame = pd.DataFrame(index=part)
# removed_frame['percleft_cue'] = 999
# percleft_cue = []
# percremoved_cue_comperp = []

# if version == 1:
#     for p in tqdm(part):
#         print(f"\n--- TFR for {p} (version=1, lock_type={lock_type})")

#         indir = opj(outpath, p, 'eeg')

#         if lock_type == 'cue':
#             outdir_erp = opj(outpath, p, 'eeg', 'erps')
#             erp_fname = f"{p}_decision_cues_singletrials-epo.fif"
#         else:
#             outdir_erp = opj(outpath, p, 'eeg', 'erps_resp')
#             erp_fname = f"{p}_decision_resp_singletrials-epo.fif"

#         if not os.path.exists(outdir_erp):
#             raise RuntimeError(f"ERP dir not found for {p}: {outdir_erp}")

#         # TFR directory
#         outdir_tfr = opj(outpath, p, 'eeg', 'tfr')
#         os.makedirs(outdir_tfr, exist_ok=True)
#         # raw events
#         raw = mne.io.read_raw_fif(
#             opj(indir, f"{p}_decision_cleaned-raw.fif"),
#             preload=True,
#         )

#         subject_i = p.split('-')[-1]
#         events = pd.read_csv(
#             layout.get(
#                 subject=subject_i,
#                 extension='tsv',
#                 suffix='events',
#                 return_type='filename'
#             )[0],
#             sep='\t'
#         )

#         # erp single-trial metadata
#         erps = mne.read_epochs(opj(outdir_erp, erp_fname))
#         meta = erps.metadata.copy()
#         allbad = int(np.sum(meta.badtrial))

#         print(f"{p}: ERP single-trials loaded: {len(erps)} epochs")
#         print(f"{p}: metadata length: {len(meta)}, bad trials: {allbad}")

#         # Prepare events for TFR
#         # Drop unused channels
#         chans_to_drop = [c for c in ['HEOGL', 'HEOGR', 'VEOGL',
#                                      'STI 014', 'Status'] if c in raw.ch_names]
#         raw.drop_channels(chans_to_drop)

#         events['empty'] = 0
#         events_c = events[events['trial_type'].notna()].copy()

#         if lock_type == 'cue':
#             events_id = {"off+": 2}
#             events_c = events_c[events_c['trial_type'] == 'off+'].copy()
#             events_c['cue_num'] = events_c['trial_type'].map(events_id)
#         elif lock_type == 'response':
#             events_id = {"resp_any": 3}
#             events_c = events_c[events_c['trial_type'].isin(['res+', 'res-', 'resm'])].copy()
#             events_c['cue_num'] = events_id['resp_any']
#         else:
#             raise ValueError("lock_type must be 'cue' or 'response'")

#         events_c = events_c.sort_values('sample').reset_index(drop=True)
        
#         if 'trialsnum' in meta.columns:
#             meta = meta.sort_values('trialsnum').reset_index(drop=True)
#         else:
#             print(f"'trialsnum' not in metadata for {p}; relying on row order only.")

#         n_ev = len(events_c)
#         n_meta = len(meta)
#         print(f"{p}: events_c rows: {n_ev}, metadata rows: {n_meta}")

#         if n_ev != n_meta:
#             raise RuntimeError(
#                 f"Metadata / events length mismatch for {p}: "
#                 f"{n_meta} metadata rows vs {n_ev} events_c rows."
#             )

#         events_cues = np.asarray(events_c[['sample', 'empty', 'cue_num']])

#         # Epoch for TFR 
#         tf_cues_strials = mne.Epochs(
#             raw,
#             events=events_cues,
#             event_id=events_id,
#             tmin=tfr_param['tfrepochstart'],
#             tmax=tfr_param['tfrepochend'],
#             baseline=None,
#             metadata=meta,
#             preload=True,
#             verbose=False
#         )

#         print(f"{p}: tf_cues_strials n_epochs = {len(tf_cues_strials)}")

#         # Morlet TFR, single-trial
#         strials = tfr_morlet(
#             tf_cues_strials,
#             freqs=tfr_param['ttfreqs'],
#             n_cycles=tfr_param['n_cycles'],
#             return_itc=False,
#             use_fft=True,
#             decim=int(1024 / tfr_param["testresampfreq"]),
#             n_jobs=tfr_param['njobs'],
#             average=False  
#         )

#         tf_cues_strials = None  # free memory

#         print(f"{p}: TFR data shape BEFORE crop: {strials.data.shape}")

#         strials.crop(
#             tmin=tfr_param['tfrbaseline'],
#             tmax=tfr_param['tfrcropend']
#         )

#         print(f"{p}: TFR data shape AFTER crop:  {strials.data.shape}")
#         print(f"{p}: metadata length in TFR: {len(strials.metadata)}")


#         percleft_cue.append(
#             (len(strials) - np.sum(meta.badtrial)) / len(strials) * 100
#         )
#         percremoved_cue_comperp.append(
#             100 - ((125 - allbad) / 125 * 100)
#         )

#         # Save TFR
#         if lock_type == 'cue':
#             fname = f"{p}_decision_cues_epochs-tfr.h5"
#             tfr_prefix = "decision_cue"
#         else:
#             fname = f"{p}_decision_resp_epochs-tfr.h5"
#             tfr_prefix = "decision_resp"

#         out_fname = opj(outdir_tfr, fname)
#         print(f"{p}: saving TFR to {out_fname}")
#         strials.save(out_fname, overwrite=True)

#         # free memory
#         strials = None

#     # Save rejection stats for TFR
#     removed_frame['percleft_cue'] = percleft_cue
#     removed_frame['percremoved_cue_comperp'] = percremoved_cue_comperp
#     removed_frame.to_csv(opj(outpath, f'{tfr_prefix}_tfr_rejectionstats.csv'))
    
# elif version == 2:
#     for p in tqdm(part):
        
#         #--------------------------------------------------------------------------------
#         # directories
#         indir = opj(outpath,  p, 'eeg')
        
#         # erp dircetory 
#         outdir_erp = opj(outpath,  p, 'eeg', 'erps_passive')
#         if not os.path.exists(outdir_erp):
#             os.mkdir(outdir_erp)
#         # tfr directory
#         outdir_tfr = opj(outpath, p, 'eeg', 'tfr_passive')
#         if not os.path.exists(outdir_tfr):
#             os.mkdir(outdir_tfr)
#         else:
#             print("No Version")
#         #--------------------------------------------------------------------------------
#         # Load cleaned raw file and events
#         raw = mne.io.read_raw_fif(opj(indir,
#                                     p + '_passive_cleaned-raw.fif'),
#                                 preload=True)
#         # get participants events
#         subject_i = p.split('-')[-1]
#         # Load trial info in scr data
#         events = pd.read_csv(layout.get(subject=subject_i, extension='tsv',
#                                        suffix='events',
#                                        return_type='filename')[0], sep='\t')
#         # Get erps metadata
#         erps = mne.read_epochs(
#             opj(outdir_erp, p + '_passive_cues_singletrials-epo.fif'))
#         meta = erps.metadata
#         allbad = np.sum(meta.badtrial)
        
#         #---------------------------------------------------------------------------------
#         # Epoch according to condition
#         # Drop unused channels
#         chans_to_drop = [c for c in ['HEOGL', 'HEOGR', 'VEOGL',
#                                     'STI 014', 'Status'] if c in raw.ch_names]
#         raw.drop_channels(chans_to_drop)
    
#         events['empty'] = 0
#         events_c = events[events['trial_type'].notna()]
#         # # Epoch around  off+
#         events_id = {
#             "off+":2
#         }
    
#         events_c = events_c[events_c['trial_type'] == 'off+']
#         events_c['cue_num'] = [events_id[s] for s in events_c.trial_type]
#         events_cues = np.asarray(events_c[['sample', 'empty', 'cue_num']])
    
#         #----------------------------------------------------------------------------------
#         # Epoch for TFR
#         tf_cues_strials = mne.Epochs(
#             raw,
#             events=events_cues,
#             event_id=events_id,
#             tmin=param['tfrepochstart'],
#             baseline=None,
#             metadata=meta,
#             tmax=param['tfrepochend'],
#             preload=True,
#             verbose=False)
    
#         # # TFR single trials
#         strials = tfr_morlet(
#             tf_cues_strials,
#             freqs=param['ttfreqs'],
#             n_cycles=param['n_cycles'],
#             return_itc=False,
#             use_fft=True,
#             decim=int(1024/param["testresampfreq"]),
#             n_jobs=param['njobs'],
#             average=False)
    
#         # Clear for memory
#         tf_cues_strials = None
        
#         # Remove unused part
#         strials.crop(tmin=param['tfrbaseline'],
#                      tmax=param['tfrcropend'])
    
#         # Check drop statistics
#         percleft_cue.append(
#             (len(strials) - np.sum(meta.badtrial))/len(strials)*100)
#         percremoved_cue_comperp.append(100-((125 - allbad)/125*100))
#         #----------------------------------------------------------------------------------
#         # save tfr 
#         strials.save(opj(outdir_tfr,  p + '_passive_cues_'
#                          + 'epochs-tfr.h5'), overwrite=True)
#         # clear for memory
#         strials = None  
        
#     removed_frame['percleft_cue'] = percleft_cue
#     removed_frame['percremoved_cue_comperp'] = percremoved_cue_comperp
#     removed_frame.to_csv(opj(outpath, 'passive_tfr_rejectionstats.csv'))
    
# else:
#     print("No version for tfr")        



# # -------------------------------------------------------------------
# # Group-level subject-averaged ERPs - chan * time per subject
# # this is for the massunivariate between-subjects file

# print("\n--- group-level subject-averaged ERP matrices ---")

# group_dir = opj(outpath, "group_level")
# os.makedirs(group_dir, exist_ok=True)

# evoked_data = []
# sub_ids = []

# for p in part:
#     if version == 1:
#         if lock_type == 'cue':
#             evoked_fname = opj(outpath, p, "eeg", "erps",
#                                f"{p}_decision_off+_ave.fif")
#             prefix = "decision_cue"
#         else:
#             evoked_fname = opj(outpath, p, "eeg", "erps_resp",
#                                f"{p}_decision_resp_resp_any_ave.fif")
#             prefix = "decision_resp"
#     elif version == 2:
#         evoked_fname = opj(outpath, p, "eeg", "erps_passive",
#                            f"{p}_passive_off+_ave.fif")
#         prefix = "passive"
#     else:
#         raise RuntimeError("Group-level ERPs only implemented for version 1 or 2")

#     # Load subject-level ERP (off+)
#     ev = mne.read_evokeds(evoked_fname)[0]  # Evoked object
#     evoked_data.append(ev.data)             # n_channels * n_times
#     sub_ids.append(p)

# if len(evoked_data) > 0:
#     # Shape: (n_subjects, n_channels, n_times)
#     data_3d = np.stack(evoked_data, axis=0)
#     ch_names = ev.ch_names
#     times = ev.times

#     # Save as numpy arrays for flexible use
#     np.save(opj(group_dir, f"{prefix}_subxchxtime.npy"), data_3d)
#     np.save(opj(group_dir, f"{prefix}_times.npy"), times)
#     np.save(opj(group_dir, f"{prefix}_ch_names.npy"),
#             np.array(ch_names, dtype=object))
#     np.save(opj(group_dir, f"{prefix}_subjects.npy"),
#             np.array(sub_ids, dtype=object))

#     print(f"Saved {prefix}_off+_subxchxtime.npy with shape "
#           f"{data_3d.shape} = (n_subj, n_channels, n_times)")

#     info = ev.info  # reuse montage, sfreq, etc.
#     meta_df = pd.DataFrame({"participant_id": sub_ids})
#     group_epochs = mne.EpochsArray(
#         data_3d,
#         info,
#         tmin=times[0],
#         metadata=meta_df
#     )

#     group_epochs_fname = opj(group_dir,
#                          f"{prefix}_subaveraged-epo.fif")
#     group_epochs.save(group_epochs_fname, overwrite=True)
#     print(f"Saved group-level epochs as {group_epochs_fname}")
# else:
#     print("No evoked files found for group-level averaging")
