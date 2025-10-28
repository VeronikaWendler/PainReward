import mne
import os
from os.path import join as opj
from bids import BIDSLayout
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from mne_connectivity import spectral_connectivity_epochs
import gc  # Garbage collection
import time


def plot_granger_causality(freqs, gc_ab_data, gc_ba_data, trgc_data, outdir, fname):
    """Plot Granger causality results and save the plot"""
    net_gc = gc_ab_data - gc_ba_data  # Compute net Granger causality
    
    # Plot individual Granger causality (A => B, B => A)
    fig, ax = plt.subplots()
    ax.plot(freqs, gc_ab_data[0], label='Parietal to Frontal (A => B)')
    ax.plot(freqs, gc_ba_data[0], label='Frontal to Parietal (B => A)')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Granger Causality (A.U.)')
    ax.legend()
    plt.title(f'Granger Causality - {fname}')
    plt.savefig(opj(outdir, f'{fname}_granger_causality_F_P_low.png'))
    plt.show()
    time.sleep(1)

    # Plot net Granger causality
    fig, ax = plt.subplots()
    ax.plot(freqs, net_gc[0], label='Net GC (A => B - B => A)')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Net Connectivity (A.U.)')
    ax.legend()
    plt.title(f'Net Granger Causality - {fname}')
    plt.savefig(opj(outdir, f'{fname}_net_granger_causality_F_P_low.png'))
    plt.show()
    time.sleep(1)

    # Plot TRGC (time-reversed GC)
    fig, ax = plt.subplots()
    ax.plot((freqs[0], freqs[-1]), (0, 0), linewidth=2, linestyle="--", color="k")  # Reference line at 0
    ax.plot(freqs, trgc_data[0], linewidth=2, label="TRGC (A => B - TR[A => B])")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Connectivity (A.U.)")
    ax.legend()
    plt.title(f'TRGC - {fname}')
    plt.savefig(opj(outdir, f'{fname}_trgc_O_P_low.png'))
    plt.show()

# BIDS directory
basepath = "D:/Aberdeen_Uni_June24/MPColl_Lab/All_Files_Relevant_For_Git/EEG/PainReward_sub-001-050/painrewardeegdata"
outpath = opj(basepath, "derivatives")
layout = BIDSLayout(basepath)

# List of participants
part = [p for p in os.listdir(opj(basepath)) if "sub" in p]
part.sort()

# Set analysis parameters
param = {
    'tfrepochstart': -0.05,  # Start of epoch for Granger causality
    'tfrepochend': 1.2,      # End of epoch for Granger causality
    'fmin': 5,              # Minimum frequency for Granger causality
    'fmax': 30,              # Maximum frequency for Granger causality 
    'gc_n_lags': 20          # Number of lags for Granger causality
}

# Group-level accumulation of Granger causality and TRGC
group_gc_ab_sum = None
group_gc_ba_sum = None
group_gc_tr_ab_sum = None
group_gc_tr_ba_sum = None
n_participants = 0

# Loop over participants
for p in tqdm(part):
    try:
        # Directories for output
        indir = opj(outpath, p, 'eeg')
        outdir_gc = opj(outpath, p, 'eeg', 'granger_causality_fro_to_par_low')
        if not os.path.exists(outdir_gc):
            os.mkdir(outdir_gc)

        # erp dircetory
        outdir_erp = opj(outpath,  p, 'eeg', 'erps')
        if not os.path.exists(outdir_erp):
            os.mkdir(outdir_erp)

        # Load cleaned raw data and events
        raw = mne.io.read_raw_fif(opj(indir, p + '_decision_cleaned-raw.fif'), preload=True)
        
        # Get participant's events
        subject_i = p.split('-')[-1]
        
        # Filter events for analysis (you may need to adapt this for your specific task)
        events = pd.read_csv(layout.get(subject=subject_i, extension='tsv',
                                    suffix='events',
                                    return_type='filename')[0], sep='\t')
        # Get erps metadata
        erps = mne.read_epochs(
            opj(outdir_erp, p + '_decision_cues_singletrials-epo.fif'))
        meta = erps.metadata
        allbad = np.sum(meta.badtrial)
        
        chans_to_drop = [c for c in ['HEOGL', 'HEOGR', 'VEOGL',
                                 'STI 014', 'Status'] if c in raw.ch_names]
        raw.drop_channels(chans_to_drop)

        events['empty'] = 0
        events_c = events[events['trial_type'].notna()]
        # # Epoch around  off+
        events_id = {
            "off+": 2
        }
    
        events_c = events_c[events_c['trial_type'] == 'off+']
        events_c['cue_num'] = [events_id[s] for s in events_c.trial_type]
        events_cues = np.asarray(events_c[['sample', 'empty', 'cue_num']])
        # Epoch data for Granger causality analysis
        epochs = mne.Epochs(raw,
                            events=events_cues, 
                            event_id=events_id,
                            tmin=param['tfrepochstart'],
                            tmax=param['tfrepochend'], 
                            preload=True,
                            verbose=False)

        # Define ROIs: occipital and parietal sensors
        frontal_sensors = [idx for idx, ch_info in enumerate(raw.info["chs"]) if ch_info["ch_name"].startswith("F")][:8]
        parietal_sensor_names = ['PO3', 'PO4', 'P3', 'P4', 'P1', 'P2', 'Pz', 'P5', 'P6']
        parietal_sensors = [raw.info['ch_names'].index(ch_name) for ch_name in parietal_sensor_names if ch_name in raw.info['ch_names']]

        indices_ab = ([frontal_sensors], [parietal_sensors])
        indices_ba = ([parietal_sensors], [frontal_sensors])

        # Compute Granger causality for original signals
        gc_ab = spectral_connectivity_epochs(epochs, method=["gc"], indices=indices_ab, fmin=param['fmin'], fmax=param['fmax'], 
                                             gc_n_lags=param['gc_n_lags'], sfreq=raw.info['sfreq'], tmin=0.0, n_jobs=1)
        gc_ba = spectral_connectivity_epochs(epochs, method=["gc"], indices=indices_ba, fmin=param['fmin'], fmax=param['fmax'], 
                                             gc_n_lags=param['gc_n_lags'], sfreq=raw.info['sfreq'], tmin=0.0, n_jobs=1)

        # Extract data for plotting
        gc_ab_data = gc_ab.get_data()
        gc_ba_data = gc_ba.get_data()

        # Compute Granger causality for time-reversed signals
        gc_tr_ab = spectral_connectivity_epochs(epochs, method=["gc_tr"], indices=indices_ab, fmin=param['fmin'], fmax=param['fmax'], 
                                                gc_n_lags=param['gc_n_lags'], sfreq=raw.info['sfreq'], tmin=0.0, n_jobs=1)
        gc_tr_ba = spectral_connectivity_epochs(epochs, method=["gc_tr"], indices=indices_ba, fmin=param['fmin'], fmax=param['fmax'], 
                                                gc_n_lags=param['gc_n_lags'], sfreq=raw.info['sfreq'], tmin=0.0, n_jobs=1)

        # Extract time-reversed data
        gc_tr_ab_data = gc_tr_ab.get_data()
        gc_tr_ba_data = gc_tr_ba.get_data()

        # Compute net GC for time-reversed signals
        net_gc_tr = gc_tr_ab_data - gc_tr_ba_data

        # Plot and save participant-level Granger causality and TRGC
        plot_granger_causality(gc_ab.freqs, gc_ab_data, gc_ba_data, net_gc_tr, outdir_gc, p)

        # Accumulate Granger causality results for group-level analysis
        if group_gc_ab_sum is None:
            group_gc_ab_sum = np.zeros_like(gc_ab_data)
            group_gc_ba_sum = np.zeros_like(gc_ba_data)
            group_gc_tr_ab_sum = np.zeros_like(gc_tr_ab_data)
            group_gc_tr_ba_sum = np.zeros_like(gc_tr_ba_data)

        group_gc_ab_sum += gc_ab_data
        group_gc_ba_sum += gc_ba_data
        group_gc_tr_ab_sum += gc_tr_ab_data
        group_gc_tr_ba_sum += gc_tr_ba_data
        n_participants += 1

    except Exception as e:
        print(f"Error with participant {p}: {e}")
        continue

#----------------------------------------------------------------------------------
# Group-level averaging of Granger causality
average_gc_ab = group_gc_ab_sum / n_participants
average_gc_ba = group_gc_ba_sum / n_participants
average_gc_tr_ab = group_gc_tr_ab_sum / n_participants
average_gc_tr_ba = group_gc_tr_ba_sum / n_participants

# Compute net GC and net GC for time-reversed signals at group level
net_gc = average_gc_ab - average_gc_ba
net_gc_tr = average_gc_tr_ab - average_gc_tr_ba

# Compute TRGC at the group level
trgc = net_gc - net_gc_tr

# Plot and save group-level Granger causality and TRGC
outdir_group = opj(outpath, 'group_results_granger_F_to_P_low')
if not os.path.exists(outdir_group):
    os.mkdir(outdir_group)

plot_granger_causality(gc_ab.freqs, average_gc_ab, average_gc_ba, trgc, outdir_group, 'group_average_granger_F_to_P_low')

# Ensure memory cleanup at the end
gc.collect()




# ###########################################################################################################################

# # Occipital to Parietal - lower frequency


# import mne
# import os
# from os.path import join as opj
# from bids import BIDSLayout
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from tqdm import tqdm
# from mne_connectivity import spectral_connectivity_epochs
# import gc  # Garbage collection
# import time

# # Plotting function for Granger causality results
# def plot_granger_causality(freqs, gc_ab_data, gc_ba_data, trgc_data, outdir, fname):
#     """Plot Granger causality results and save the plot."""
#     net_gc = gc_ab_data - gc_ba_data  # Compute net Granger causality
    
#     # Plot individual Granger causality (A => B, B => A)
#     fig, ax = plt.subplots()
#     ax.plot(freqs, gc_ab_data[0], label='Occipital to Parietal (A => B)')
#     ax.plot(freqs, gc_ba_data[0], label='Parietal to Occipital (B => A)')
#     ax.set_xlabel('Frequency (Hz)')
#     ax.set_ylabel('Granger Causality (A.U.)')
#     ax.legend()
#     plt.title(f'Granger Causality - {fname}')
#     plt.savefig(opj(outdir, f'{fname}_granger_causality_O_P_low.png'))
#     plt.show()
#     time.sleep(1)

#     # Plot net Granger causality
#     fig, ax = plt.subplots()
#     ax.plot(freqs, net_gc[0], label='Net GC (A => B - B => A)')
#     ax.set_xlabel('Frequency (Hz)')
#     ax.set_ylabel('Net Connectivity (A.U.)')
#     ax.legend()
#     plt.title(f'Net Granger Causality - {fname}')
#     plt.savefig(opj(outdir, f'{fname}_net_granger_causality_O_P_low.png'))
#     plt.show()
#     time.sleep(1)

#     # Plot TRGC (time-reversed GC)
#     fig, ax = plt.subplots()
#     ax.plot((freqs[0], freqs[-1]), (0, 0), linewidth=2, linestyle="--", color="k")  # Reference line at 0
#     ax.plot(freqs, trgc_data[0], linewidth=2, label="TRGC (A => B - TR[A => B])")
#     ax.set_xlabel("Frequency (Hz)")
#     ax.set_ylabel("Connectivity (A.U.)")
#     ax.legend()
#     plt.title(f'TRGC - {fname}')
#     plt.savefig(opj(outdir, f'{fname}_trgc_O_P_low.png'))
#     plt.show()

# # BIDS directory
# basepath = "D:/Aberdeen_Uni_June24/MPColl_Lab/All_Files_Relevant_For_Git/EEG/PainReward_sub-001-050/painrewardeegdata"
# outpath = opj(basepath, "derivatives")
# layout = BIDSLayout(basepath)

# # List of participants
# part = [p for p in os.listdir(opj(basepath)) if "sub" in p]
# part.sort()

# # Set analysis parameters
# param = {
#     'tfrepochstart': -0.05,  # Start of epoch for Granger causality
#     'tfrepochend': 1.2,      # End of epoch for Granger causality
#     'fmin': 5,              # Minimum frequency for Granger causality
#     'fmax': 30,              # Maximum frequency for Granger causality 
#     'gc_n_lags': 20          # Number of lags for Granger causality
# }

# # Group-level accumulation of Granger causality and TRGC
# group_gc_ab_sum = None
# group_gc_ba_sum = None
# group_gc_tr_ab_sum = None
# group_gc_tr_ba_sum = None
# n_participants = 0

# # Loop over participants
# for p in tqdm(part):
#     try:
#         # Directories for output
#         indir = opj(outpath, p, 'eeg')
#         outdir_gc = opj(outpath, p, 'eeg', 'granger_causality_oc_to_par_low')
#         if not os.path.exists(outdir_gc):
#             os.mkdir(outdir_gc)

#         # erp dircetory
#         outdir_erp = opj(outpath,  p, 'eeg', 'erps')
#         if not os.path.exists(outdir_erp):
#             os.mkdir(outdir_erp)

#         # Load cleaned raw data and events
#         raw = mne.io.read_raw_fif(opj(indir, p + '_decision_cleaned-raw.fif'), preload=True)
        
#         # Get participant's events
#         subject_i = p.split('-')[-1]
        
#         # Filter events for analysis (you may need to adapt this for your specific task)
#         events = pd.read_csv(layout.get(subject=subject_i, extension='tsv',
#                                     suffix='events',
#                                     return_type='filename')[0], sep='\t')
#         # Get erps metadata
#         erps = mne.read_epochs(
#             opj(outdir_erp, p + '_decision_cues_singletrials-epo.fif'))
#         meta = erps.metadata
#         allbad = np.sum(meta.badtrial)
        
#         chans_to_drop = [c for c in ['HEOGL', 'HEOGR', 'VEOGL',
#                                  'STI 014', 'Status'] if c in raw.ch_names]
#         raw.drop_channels(chans_to_drop)

#         events['empty'] = 0
#         events_c = events[events['trial_type'].notna()]
#         # # Epoch around  off+
#         events_id = {
#             "off+": 2
#         }
    
#         events_c = events_c[events_c['trial_type'] == 'off+']
#         events_c['cue_num'] = [events_id[s] for s in events_c.trial_type]
#         events_cues = np.asarray(events_c[['sample', 'empty', 'cue_num']])
#         # Epoch data for Granger causality analysis
#         epochs = mne.Epochs(raw,
#                             events=events_cues, 
#                             event_id=events_id,
#                             tmin=param['tfrepochstart'],
#                             tmax=param['tfrepochend'], 
#                             preload=True,
#                             verbose=False)

#         # Define ROIs: occipital and parietal sensors
#         occipital_sensors = [idx for idx, ch_info in enumerate(raw.info["chs"]) if ch_info["ch_name"].startswith("O")][:8]
#         parietal_sensor_names = ['PO3', 'PO4', 'P3', 'P4', 'P1', 'P2', 'Pz', 'P5', 'P6']
#         parietal_sensors = [raw.info['ch_names'].index(ch_name) for ch_name in parietal_sensor_names if ch_name in raw.info['ch_names']]

#         indices_ab = ([occipital_sensors], [parietal_sensors])
#         indices_ba = ([parietal_sensors], [occipital_sensors])

#         # Compute Granger causality for original signals
#         gc_ab = spectral_connectivity_epochs(epochs, method=["gc"], indices=indices_ab, fmin=param['fmin'], fmax=param['fmax'], 
#                                              gc_n_lags=param['gc_n_lags'], sfreq=raw.info['sfreq'], tmin=0.0, n_jobs=1)
#         gc_ba = spectral_connectivity_epochs(epochs, method=["gc"], indices=indices_ba, fmin=param['fmin'], fmax=param['fmax'], 
#                                              gc_n_lags=param['gc_n_lags'], sfreq=raw.info['sfreq'], tmin=0.0, n_jobs=1)

#         # Extract data for plotting
#         gc_ab_data = gc_ab.get_data()
#         gc_ba_data = gc_ba.get_data()

#         # Compute Granger causality for time-reversed signals
#         gc_tr_ab = spectral_connectivity_epochs(epochs, method=["gc_tr"], indices=indices_ab, fmin=param['fmin'], fmax=param['fmax'], 
#                                                 gc_n_lags=param['gc_n_lags'], sfreq=raw.info['sfreq'], tmin=0.0, n_jobs=1)
#         gc_tr_ba = spectral_connectivity_epochs(epochs, method=["gc_tr"], indices=indices_ba, fmin=param['fmin'], fmax=param['fmax'], 
#                                                 gc_n_lags=param['gc_n_lags'], sfreq=raw.info['sfreq'], tmin=0.0, n_jobs=1)

#         # Extract time-reversed data
#         gc_tr_ab_data = gc_tr_ab.get_data()
#         gc_tr_ba_data = gc_tr_ba.get_data()

#         # Compute net GC for time-reversed signals
#         net_gc_tr = gc_tr_ab_data - gc_tr_ba_data

#         # Plot and save participant-level Granger causality and TRGC
#         plot_granger_causality(gc_ab.freqs, gc_ab_data, gc_ba_data, net_gc_tr, outdir_gc, p)

#         # Accumulate Granger causality results for group-level analysis
#         if group_gc_ab_sum is None:
#             group_gc_ab_sum = np.zeros_like(gc_ab_data)
#             group_gc_ba_sum = np.zeros_like(gc_ba_data)
#             group_gc_tr_ab_sum = np.zeros_like(gc_tr_ab_data)
#             group_gc_tr_ba_sum = np.zeros_like(gc_tr_ba_data)

#         group_gc_ab_sum += gc_ab_data
#         group_gc_ba_sum += gc_ba_data
#         group_gc_tr_ab_sum += gc_tr_ab_data
#         group_gc_tr_ba_sum += gc_tr_ba_data
#         n_participants += 1

#     except Exception as e:
#         print(f"Error with participant {p}: {e}")
#         continue

# #----------------------------------------------------------------------------------
# # Group-level averaging of Granger causality
# average_gc_ab = group_gc_ab_sum / n_participants
# average_gc_ba = group_gc_ba_sum / n_participants
# average_gc_tr_ab = group_gc_tr_ab_sum / n_participants
# average_gc_tr_ba = group_gc_tr_ba_sum / n_participants

# # Compute net GC and net GC for time-reversed signals at group level
# net_gc = average_gc_ab - average_gc_ba
# net_gc_tr = average_gc_tr_ab - average_gc_tr_ba

# # Compute TRGC at the group level
# trgc = net_gc - net_gc_tr

# # Plot and save group-level Granger causality and TRGC
# outdir_group = opj(outpath, 'group_results_granger_O_to_P_low')
# if not os.path.exists(outdir_group):
#     os.mkdir(outdir_group)

# plot_granger_causality(gc_ab.freqs, average_gc_ab, average_gc_ba, trgc, outdir_group, 'group_average_granger_O_to_P_low')

# # Ensure memory cleanup at the end
# gc.collect()






###########################################################################################################################

# Occipital to Parietal - high frequency


# import mne
# import os
# from os.path import join as opj
# from bids import BIDSLayout
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from tqdm import tqdm
# from mne_connectivity import spectral_connectivity_epochs
# import gc  # Garbage collection
# import time

# # Plotting function for Granger causality results
# def plot_granger_causality(freqs, gc_ab_data, gc_ba_data, trgc_data, outdir, fname):
#     """Plot Granger causality results and save the plot."""
#     net_gc = gc_ab_data - gc_ba_data  # Compute net Granger causality
    
#     # Plot individual Granger causality (A => B, B => A)
#     fig, ax = plt.subplots()
#     ax.plot(freqs, gc_ab_data[0], label='Occipital to Parietal (A => B)')
#     ax.plot(freqs, gc_ba_data[0], label='Parietal to Occipital (B => A)')
#     ax.set_xlabel('Frequency (Hz)')
#     ax.set_ylabel('Granger Causality (A.U.)')
#     ax.legend()
#     plt.title(f'Granger Causality - {fname}')
#     plt.savefig(opj(outdir, f'{fname}_granger_causality_O_P_high.png'))
#     plt.show()
#     time.sleep(1)

#     # Plot net Granger causality
#     fig, ax = plt.subplots()
#     ax.plot(freqs, net_gc[0], label='Net GC (A => B - B => A)')
#     ax.set_xlabel('Frequency (Hz)')
#     ax.set_ylabel('Net Connectivity (A.U.)')
#     ax.legend()
#     plt.title(f'Net Granger Causality - {fname}')
#     plt.savefig(opj(outdir, f'{fname}_net_granger_causality_O_P_high.png'))
#     plt.show()
#     time.sleep(1)

#     # Plot TRGC (time-reversed GC)
#     fig, ax = plt.subplots()
#     ax.plot((freqs[0], freqs[-1]), (0, 0), linewidth=2, linestyle="--", color="k")  # Reference line at 0
#     ax.plot(freqs, trgc_data[0], linewidth=2, label="TRGC (A => B - TR[A => B])")
#     ax.set_xlabel("Frequency (Hz)")
#     ax.set_ylabel("Connectivity (A.U.)")
#     ax.legend()
#     plt.title(f'TRGC - {fname}')
#     plt.savefig(opj(outdir, f'{fname}_trgc_O_P_high.png'))
#     plt.show()

# # BIDS directory
# basepath = "D:/Aberdeen_Uni_June24/MPColl_Lab/All_Files_Relevant_For_Git/EEG/PainReward_sub-001-050/painrewardeegdata"
# outpath = opj(basepath, "derivatives")
# layout = BIDSLayout(basepath)

# # List of participants
# part = [p for p in os.listdir(opj(basepath)) if "sub" in p]
# part.sort()

# # Set analysis parameters
# param = {
#     'tfrepochstart': -0.05,  # Start of epoch for Granger causality
#     'tfrepochend': 1.2,      # End of epoch for Granger causality
#     'fmin': 15,              # Minimum frequency for Granger causality
#     'fmax': 80,              # Maximum frequency for Granger causality 
#     'gc_n_lags': 20          # Number of lags for Granger causality
# }

# # Group-level accumulation of Granger causality and TRGC
# group_gc_ab_sum = None
# group_gc_ba_sum = None
# group_gc_tr_ab_sum = None
# group_gc_tr_ba_sum = None
# n_participants = 0

# # Loop over participants
# for p in tqdm(part):
#     try:
#         # Directories for output
#         indir = opj(outpath, p, 'eeg')
#         outdir_gc = opj(outpath, p, 'eeg', 'granger_causality_oc_to_par_high')
#         if not os.path.exists(outdir_gc):
#             os.mkdir(outdir_gc)

#         # erp dircetory
#         outdir_erp = opj(outpath,  p, 'eeg', 'erps')
#         if not os.path.exists(outdir_erp):
#             os.mkdir(outdir_erp)

#         # Load cleaned raw data and events
#         raw = mne.io.read_raw_fif(opj(indir, p + '_decision_cleaned-raw.fif'), preload=True)
        
#         # Get participant's events
#         subject_i = p.split('-')[-1]
        
#         # Filter events for analysis (you may need to adapt this for your specific task)
#         events = pd.read_csv(layout.get(subject=subject_i, extension='tsv',
#                                     suffix='events',
#                                     return_type='filename')[0], sep='\t')
#         # Get erps metadata
#         erps = mne.read_epochs(
#             opj(outdir_erp, p + '_decision_cues_singletrials-epo.fif'))
#         meta = erps.metadata
#         allbad = np.sum(meta.badtrial)
        
#         chans_to_drop = [c for c in ['HEOGL', 'HEOGR', 'VEOGL',
#                                  'STI 014', 'Status'] if c in raw.ch_names]
#         raw.drop_channels(chans_to_drop)

#         events['empty'] = 0
#         events_c = events[events['trial_type'].notna()]
#         # # Epoch around  off+
#         events_id = {
#             "off+": 2
#         }
    
#         events_c = events_c[events_c['trial_type'] == 'off+']
#         events_c['cue_num'] = [events_id[s] for s in events_c.trial_type]
#         events_cues = np.asarray(events_c[['sample', 'empty', 'cue_num']])
#         # Epoch data for Granger causality analysis
#         epochs = mne.Epochs(raw,
#                             events=events_cues, 
#                             event_id=events_id,
#                             tmin=param['tfrepochstart'],
#                             tmax=param['tfrepochend'], 
#                             preload=True,
#                             verbose=False)

#         # Define ROIs: occipital and parietal sensors
#         occipital_sensors = [idx for idx, ch_info in enumerate(raw.info["chs"]) if ch_info["ch_name"].startswith("O")][:8]
#         parietal_sensor_names = ['PO3', 'PO4', 'P3', 'P4', 'P1', 'P2', 'Pz', 'P5', 'P6']
#         parietal_sensors = [raw.info['ch_names'].index(ch_name) for ch_name in parietal_sensor_names if ch_name in raw.info['ch_names']]

#         indices_ab = ([occipital_sensors], [parietal_sensors])
#         indices_ba = ([parietal_sensors], [occipital_sensors])

#         # Compute Granger causality for original signals
#         gc_ab = spectral_connectivity_epochs(epochs, method=["gc"], indices=indices_ab, fmin=param['fmin'], fmax=param['fmax'], 
#                                              gc_n_lags=param['gc_n_lags'], sfreq=raw.info['sfreq'], tmin=0.0, n_jobs=1)
#         gc_ba = spectral_connectivity_epochs(epochs, method=["gc"], indices=indices_ba, fmin=param['fmin'], fmax=param['fmax'], 
#                                              gc_n_lags=param['gc_n_lags'], sfreq=raw.info['sfreq'], tmin=0.0, n_jobs=1)

#         # Extract data for plotting
#         gc_ab_data = gc_ab.get_data()
#         gc_ba_data = gc_ba.get_data()

#         # Compute Granger causality for time-reversed signals
#         gc_tr_ab = spectral_connectivity_epochs(epochs, method=["gc_tr"], indices=indices_ab, fmin=param['fmin'], fmax=param['fmax'], 
#                                                 gc_n_lags=param['gc_n_lags'], sfreq=raw.info['sfreq'], tmin=0.0, n_jobs=1)
#         gc_tr_ba = spectral_connectivity_epochs(epochs, method=["gc_tr"], indices=indices_ba, fmin=param['fmin'], fmax=param['fmax'], 
#                                                 gc_n_lags=param['gc_n_lags'], sfreq=raw.info['sfreq'], tmin=0.0, n_jobs=1)

#         # Extract time-reversed data
#         gc_tr_ab_data = gc_tr_ab.get_data()
#         gc_tr_ba_data = gc_tr_ba.get_data()

#         # Compute net GC for time-reversed signals
#         net_gc_tr = gc_tr_ab_data - gc_tr_ba_data

#         # Plot and save participant-level Granger causality and TRGC
#         plot_granger_causality(gc_ab.freqs, gc_ab_data, gc_ba_data, net_gc_tr, outdir_gc, p)

#         # Accumulate Granger causality results for group-level analysis
#         if group_gc_ab_sum is None:
#             group_gc_ab_sum = np.zeros_like(gc_ab_data)
#             group_gc_ba_sum = np.zeros_like(gc_ba_data)
#             group_gc_tr_ab_sum = np.zeros_like(gc_tr_ab_data)
#             group_gc_tr_ba_sum = np.zeros_like(gc_tr_ba_data)

#         group_gc_ab_sum += gc_ab_data
#         group_gc_ba_sum += gc_ba_data
#         group_gc_tr_ab_sum += gc_tr_ab_data
#         group_gc_tr_ba_sum += gc_tr_ba_data
#         n_participants += 1

#     except Exception as e:
#         print(f"Error with participant {p}: {e}")
#         continue

# #----------------------------------------------------------------------------------
# # Group-level averaging of Granger causality
# average_gc_ab = group_gc_ab_sum / n_participants
# average_gc_ba = group_gc_ba_sum / n_participants
# average_gc_tr_ab = group_gc_tr_ab_sum / n_participants
# average_gc_tr_ba = group_gc_tr_ba_sum / n_participants

# # Compute net GC and net GC for time-reversed signals at group level
# net_gc = average_gc_ab - average_gc_ba
# net_gc_tr = average_gc_tr_ab - average_gc_tr_ba

# # Compute TRGC at the group level
# trgc = net_gc - net_gc_tr

# # Plot and save group-level Granger causality and TRGC
# outdir_group = opj(outpath, 'group_results_granger_O_to_P_high')
# if not os.path.exists(outdir_group):
#     os.mkdir(outdir_group)

# plot_granger_causality(gc_ab.freqs, average_gc_ab, average_gc_ba, trgc, outdir_group, 'group_average_granger_O_to_P_high')

# # Ensure memory cleanup at the end
# gc.collect()
