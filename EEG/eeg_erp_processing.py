# Using the cleaned data from 03-painreward_eeg_preprocess to create ERPs and TFR

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


# Set bids directory
basepath = "D:/Aberdeen_Uni_June24/MPColl_Lab/All_Files_Relevant_For_Git/EEG/PainReward_sub-001-050/painrewardeegdata"

# Choose output directory
outpath = opj(basepath, "derivatives")
layout = BIDSLayout(basepath)

# List participants
part = [p for p in os.listdir(opj(basepath)) if "sub" in p]
part.sort()

# defining parameters for ERPs
param = {
    # Additional LP filter fora ERPs
    'erplpfilter': 30,
    # Filter to use
    'filtertype': 'fir',
    # Length of baseline
    'erpbaseline': -0.2,
    'erpepochend': 1.5,
    # Threshold to reject trials
    'erpreject': dict(eeg=150e-6),
    # Threshold to reject shock trials
    'erprejectshock': dict(eeg=150e-6),
    
}

#-----------------------------------------------------------------------------------------------------------------------
# epoching erps

# Initialise array to collect rejection stats
reject_stats = pd.DataFrame(data={'part': part, 'perc_removed_cues': 9999,
                                  'perc_removed_shocks': 9999,
                                   "Off+": 0,
                                #    "dIN8": 0,
                                #    "Res+": 0,
                                #    "Fix+": 0,
                                #    "Fee+": 0,
                                #    "Fee-": 0,
                                #    "Fix+": 0,
                                #    "Cdow": 0,
                                #    "Shk-": 0,
                                  })


for p in part:
    # ______________________________________________________
    # Make out dir
    indir = opj(outpath,  p, 'eeg')

    # erp dircetory
    outdir = opj(outpath,  p, 'eeg', 'erps')
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    
    
    # _______________________________________________________
    # Initialise MNE report
    report = Report(verbose=False, subject=p,
                    title='ERP report for part ' + p)

    report.add_html(pprint.pformat(param),
                    title='Parameters',
                    section='Parameters')
    
    # ______________________________________________________
    # Load cleaned raw file
    raw = mne.io.read_raw_fif(opj(indir,
                                  p + '_decision_cleaned-raw.fif'),
                              preload=True)
    
    subject_i = p.split('-')[-1]
    # Load trial info in scr data
    events = pd.read_csv(layout.get(subject=subject_i, extension='tsv',
                                    suffix='events',
                                    return_type='filename')[0], sep='\t')

    
    # Drop unused channels
    chans_to_drop = [c for c in ['HEOGL', 'HEOGR', 'VEOGL',
                                 'STI 014', 'Status'] if c in raw.ch_names]
    raw.drop_channels(chans_to_drop)
    
    # Filter for erpss
    raw = raw.filter(
        None,
        param['erplpfilter'],
        method=param['filtertype'])

    # Add empty column to make it easier to create the event array
    events['empty'] = 0
    events_c = events[events['trial_type'].notna()]
    # events_c = events_c[events_c['trial_type'] != 'DIN7']
    # events_c = events_c[events_c['trial_type'] != 'RSTR']
    # valid_trial_types = ["off+", "DIN8", "res+", "fix+", "fee+", "fee-", "fix+", "cdow", "shk-"]
    # events_c = events_c[events_c['trial_type'].isin(valid_trial_types)]

    
    # # ______________________________________________________________
    # # Epoch around  off+
    events_id = {
        "off+": 2,
        # "DIN8": 1,
        # "res+": 3,
        # "fix+": 6,
        # "fee+": 7,
        # "fee-": 8,
        # "fix+": 9,
        # "cdow": 10,
        # "shk-": 11,
    }

    #events_c['cue_num'] = [events_id[s] for s in events_c.trial_type]

    events_c = events_c[events_c['trial_type'] == 'off+']
    events_c['cue_num'] = events_c['trial_type'].map(events_id)
    events_cues = np.asarray(events_c[['sample', 'empty', 'cue_num']])
    
    # events_c['cue_num'] = [events_id[s] for s in events_c.trial_type]
    # events_cues = np.asarray(events_c[['sample', 'empty', 'cue_num']])

    erp_cues = mne.Epochs(
        raw,
        events=events_cues,
        event_id=events_id,
        tmin=param['erpbaseline'],
        baseline=(param['erpbaseline'], 0),
        tmax=param['erpepochend'],
        preload=True,
        verbose=False,
        reject=param['erpreject']
    )

    fig = mne.viz.plot_drop_log(erp_cues.drop_log, show=False)
    report.add_figure(fig, title='Drop log', section='Drop log')

    reject_stats.loc[reject_stats.part == p, reject_stats.columns == 'perc_removed_cues'] = ((125-len(erp_cues))/125*100)
    reject_stats.loc[reject_stats.part == p, reject_stats.columns == 'Off+'] = len(erp_cues['off+'])
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
    figs_butter = []
    evokeds = dict()
    for cond in events_id.keys():
        evokeds[cond] = erp_cues[cond].average()                   # gives evoked for off+
        figs_butter.append(pej(evokeds[cond],
                               title=cond,
                               show=False,
                               picks='eeg',
                               exclude=['HEOGL', 'HEOGR', 'VEOGL'],
                               ts_args={'time_unit': 'ms'},
                               topomap_args={'time_unit': 'ms'}))

        evokeds[cond].save(opj(outdir, p + '_decision_' + cond
                               + '_ave.fif'), overwrite=True)

    report.add_figure(figs_butter,
                      section='ERPs for cues off+',
                      title='Butterfly plots for cues off+')
    
    # Adding plot_image
    #off_fig = mne.viz.plot_image(evokeds['off+'], show=False)
    off_fig = evokeds['off+'].plot_image(picks="eeg")
    report.add_figure(off_fig, section='ERPs for cues off+', title='plot_image for off+')


    # Plot some channels and add to report
    chans_to_plot = ['Cz', 'Pz', 'CPz', 'Oz', 'Fz', 'FCz', 'POz', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8',
                     'F7', 'F5', 'F3','F1', 'Fz', 'F2', 'F4', 'F6', 'F8',
                     'C5', 'C3', 'C1', 'C2', 'C4', 'C6',
                     'PO7', 'PO3', 'PO4', 'PO8' ]
    figs_chan = []                                                          
    for c in chans_to_plot:                                               
        pick = erp_cues.ch_names.index(c)
        figs_chan.append(mne.viz.plot_compare_evokeds(evokeds, picks=pick,
                                                      show=False)[0])

    report.add_figure(figs_chan,
                      section='ERPs for cues off+', title='Cues/chans')
    

    report.save(opj(outdir,  p + '_decision_erps_report.html'),
                open_browser=False, overwrite=True)
    
    #________________________________________________________
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
        baseline=(param['erpbaseline'], 0),
        tmax=param['erpepochend'],
        metadata=events_c,
        preload=True,
        verbose=True)
    
    # Add bad trials to metadata
    strials_drop = erp_cues_single.copy()
    strials_drop.drop_bad(reject=param['erpreject'])
    badtrials = [1 if len(li) > 0 else 0 for li in strials_drop.drop_log]

    erp_cues_single.metadata['badtrial'] = badtrials

    # Save
    erp_cues_single.save(opj(outdir, p
                             + '_decision_cues_singletrials-epo.fif'),
                         overwrite=True)
    
    plt.close('all')

    #-------------------------------------------------------------------------------------------
    
# Save rejection stats
reject_stats['perc_removed_all'] = (
    1-reject_stats[['Off+']].sum(axis=1)/(125))*100
reject_stats.to_csv(opj(outpath,
                        'decision_erps_rejectionstats.csv'))


reject_stats.describe().to_csv(opj(outpath,
                                   'decision_'
                                   + 'erps_rejectionstats_desc.csv'))


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
    -------
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
chans_to_average = [['Fz'], ['FCz'], ['POz'], ['Cz'], ['CPz'], ['Pz'], ['Oz']]  
amp_lat = [[0.4, 0.8]] 

all_meta = []
for p in part:
    outdir = opj(outpath,  p, 'eeg', 'erps')
    epo = mne.read_epochs(opj(outdir, p + '_decision_cues_singletrials-epo.fif'))

    # amplitude extraction to the epochs
    epo = average_time_win_strials(epo, chans_to_average, amp_lat)

    # participant ID added to the metadata
    epo.metadata['participant_id'] = p
    
    all_meta.append(epo.metadata)

all_meta = pd.concat(all_meta)
all_meta.to_csv(opj(outpath, 'decision_erpsmeta.csv'), index=False)



# #_____________________________________________________________________________________________________________________________________________________________
# # TFR 
# #_____________________________________________________________________________________________________________________________________________________________

# # importing libraries
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

# # library for tfr
# from mne.time_frequency import tfr_morlet

# # bids directory
# basepath = "D:/Aberdeen_Uni_June24/MPColl_Lab/All_Files_Relevant_For_Git/EEG/PainReward_sub-001-050/painrewardeegdata"

# #output directory
# outpath = opj(basepath, "derivatives")
# layout = BIDSLayout(basepath)

# # List participants
# part = [p for p in os.listdir(opj(basepath)) if "sub" in p]
# part.sort()

# # set params
# param = {
#     # Length of epochs
#     'erpbaseline': -0.20,  # Used for trial rejection
#     'erpepochend': 1.5,
#     'tfrbaseline': -0.50,  # Used for trial rejection
#     'tfrcropend': 1.5,
#     'tfrepochstart': -2,  # Used for TFR transform
#     'tfrepochend': 2,
#     'ttfreqs': np.arange(4, 101, 1),  # Frequencies
#     'n_cycles': 0.5*np.arange(4, 101, 1),  # Wavelet cycles
#     'testresampfreq': 256,  # Sfreq to downsample to
#     'njobs': 3,  # N cpus to run TFR
#     # Removed shocked trails
#     #'ignoreshocks': False,
# }

# removed_frame = pd.DataFrame(index=part)
# removed_frame['percleft_cue'] = 999
# percleft_cue = []
# percremoved_cue_comperp = []

# for p in tqdm(part):
#     #--------------------------------------------------------------------------------
#     # directories
#     indir = opj(outpath,  p, 'eeg')
    
#     # erp dircetory
#     outdir_erp = opj(outpath,  p, 'eeg', 'erps')
#     if not os.path.exists(outdir_erp):
#         os.mkdir(outdir_erp)
        
#     # tfr directory
#     outdir_tfr = opj(outpath, p, 'eeg', 'tfr')
#     if not os.path.exists(outdir_tfr):
#         os.mkdir(outdir_tfr)
        
#     #--------------------------------------------------------------------------------
#     # Load cleaned raw file and events
#     raw = mne.io.read_raw_fif(opj(indir,
#                                   p + '_decision_cleaned-raw.fif'),
#                               preload=True)
#     # get participants events
#     subject_i = p.split('-')[-1]
#     # Load trial info in scr data
#     events = pd.read_csv(layout.get(subject=subject_i, extension='tsv',
#                                     suffix='events',
#                                     return_type='filename')[0], sep='\t')
#     # Get erps metadata
#     erps = mne.read_epochs(
#         opj(outdir_erp, p + '_decision_cues_singletrials-epo.fif'))
#     meta = erps.metadata
#     allbad = np.sum(meta.badtrial)
    
#     #---------------------------------------------------------------------------------
#     # Epoch according to condition
#     # Drop unused channels
#     chans_to_drop = [c for c in ['HEOGL', 'HEOGR', 'VEOGL',
#                                  'STI 014', 'Status'] if c in raw.ch_names]
#     raw.drop_channels(chans_to_drop)

#     events['empty'] = 0
#     events_c = events[events['trial_type'].notna()]
#     # # Epoch around  off+
#     events_id = {
#         "off+":2
#     }
    
#     events_c = events_c[events_c['trial_type'] == 'off+']
#     events_c['cue_num'] = [events_id[s] for s in events_c.trial_type]
#     events_cues = np.asarray(events_c[['sample', 'empty', 'cue_num']])
    
#     #----------------------------------------------------------------------------------
#     # Epoch for TFR
#     tf_cues_strials = mne.Epochs(
#         raw,
#         events=events_cues,
#         event_id=events_id,
#         tmin=param['tfrepochstart'],
#         baseline=None,
#         metadata=meta,
#         tmax=param['tfrepochend'],
#         preload=True,
#         verbose=False)

#     # # TFR single trials
#     strials = tfr_morlet(
#         tf_cues_strials,
#         freqs=param['ttfreqs'],
#         n_cycles=param['n_cycles'],
#         return_itc=False,
#         use_fft=True,
#         decim=int(1024/param["testresampfreq"]),
#         n_jobs=param['njobs'],
#         average=False)

#     # Clear for memory
#     tf_cues_strials = None
    
#     # Remove unused part
#     strials.crop(tmin=param['tfrbaseline'],
#                  tmax=param['tfrcropend'])

#     # Check drop statistics
#     percleft_cue.append(
#         (len(strials) - np.sum(meta.badtrial))/len(strials)*100)
#     percremoved_cue_comperp.append(100-((125 - allbad)/125*100))
    
    
#     #----------------------------------------------------------------------------------
#     # save tfr 
#     strials.save(opj(outdir_tfr,  p + '_decision_cues_'
#                      + 'epochs-tfr.h5'), overwrite=True)
#     # clear for memory
#     strials = None  # Clear for memory
    
# removed_frame['percleft_cue'] = percleft_cue
# removed_frame['percremoved_cue_comperp'] = percremoved_cue_comperp
# removed_frame.to_csv(opj(outpath, 'decision_tfr_rejectionstats.csv'))

