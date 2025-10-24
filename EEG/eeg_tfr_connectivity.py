#_____________________________________________________________________________________________________________________________________________________________
# TFR: Functional Connectivity but the lame version (sensor space, did not have sufficient memory for a free ____________________________________________________________________________________________________________________________________________

# importing libraries
# importing libraries
import mne
import os
from os.path import join as opj
from bids import BIDSLayout
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from mne_connectivity import spectral_connectivity_epochs
from mne_connectivity.viz import plot_sensors_connectivity
from mne_connectivity.viz import plot_connectivity_circle
import gc  # Garbage collection

# Function to plot the connectivity matrix
def plot_connectivity_circle_plot(con_matrix, labels, outdir, fname):
    """Plot the connectivity circle and save it to file."""
        
    # Create the figure with polar axes
    fig, ax = plt.subplots(figsize=(8, 8), facecolor="black", subplot_kw=dict(polar=True))
    
    # Define vmin and vmax based on the connectivity matrix values
    vmin = np.min(con_matrix)
    vmax = np.max(con_matrix)
    
    # Plot the connectivity circle
    plot_connectivity_circle(con_matrix, labels, n_lines=300, title=f'Connectivity Circle - {fname}', 
                             ax=ax, vmin=vmin, vmax=vmax, colormap='hot')
    
    # Create the color bar with properly linked data
    sm = plt.cm.ScalarMappable(cmap='hot', norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array(con_matrix)  # Link the actual data to the color bar
    cbar = fig.colorbar(sm, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
    
    # Set title and labels for the color bar
    cbar.ax.set_title('PLI', fontsize=12)  # Set title size
    cbar.set_ticks([vmin, (vmin + vmax) / 2, vmax])  # Set ticks at min, mid, and max
    
    # Set tick labels and their colors
    cbar.set_ticklabels([f'{vmin:.2f}', f'{(vmin + vmax) / 2:.2f}', f'{vmax:.2f}'], fontsize=10)
    cbar.ax.yaxis.set_tick_params(color='blue')  # Set the tick color to blue
    cbar.ax.yaxis.set_tick_params(labelcolor='blue')  # Set the label color to blue
    
    # Save the plot
    plt.savefig(opj(outdir, f'{fname}_connectivity_circle.png'))
    plt.close()
    
    
# connectivity analyisi as heatmap
def plot_matrix(con_matrix, method, outdir, fname):
    """Plot the connectivity matrix as a heatmap and save it."""
    plt.imshow(con_matrix, cmap='viridis')
    clb = plt.colorbar()
    clb.ax.set_title(method)
    plt.title(f'Connectivity Matrix - {fname}')
    plt.savefig(opj(outdir, f'{fname}_connectivity_matrix.png'))
    plt.close()
    
    
# bids directory
basepath = "D:/Aberdeen_Uni_June24/MPColl_Lab/All_Files_Relevant_For_Git/EEG/PainReward_sub-001-050/painrewardeegdata"

# output directory
outpath = opj(basepath, "derivatives")
layout = BIDSLayout(basepath)

# List participants
part = [p for p in os.listdir(opj(basepath)) if "sub" in p]
part.sort()

# set params
param = {
    'tfrepochstart': -0.05,  # Start of epoch for connectivity
    'tfrepochend': 1.2,  # End of epoch for connectivity
}

# Data frame to collect rejection stats
removed_frame = pd.DataFrame(index=part)
removed_frame['percleft_cue'] = 999
percleft_cue = []
percremoved_cue_comperp = []

# To store sum of connectivity matrices for average computation
total_connectivity = None
n_participants = 0

for p in tqdm(part):
    try:
        #--------------------------------------------------------------------------------
        # directories
        indir = opj(outpath,  p, 'eeg')

        # erp dircetory
        outdir_erp = opj(outpath,  p, 'eeg', 'erps')
        if not os.path.exists(outdir_erp):
            os.mkdir(outdir_erp)

        # connectivity directory
        outdir_con = opj(outpath, p, 'eeg', 'connect_delta_late')
        if not os.path.exists(outdir_con):
            os.mkdir(outdir_con)

        #--------------------------------------------------------------------------------
        # Load cleaned raw file and events
        raw = mne.io.read_raw_fif(opj(indir, p + '_decision_cleaned-raw.fif'), preload=True)

        # Get participant's events
        subject_i = p.split('-')[-1]
        events = pd.read_csv(layout.get(subject=subject_i, extension='tsv', suffix='events', return_type='filename')[0], sep='\t')

        # Get erps metadata
        erps = mne.read_epochs(opj(outdir_erp, p + '_decision_cues_singletrials-epo.fif'))
        meta = erps.metadata
        allbad = np.sum(meta.badtrial)

        #---------------------------------------------------------------------------------
        # Epoch according to condition

        # Drop unused channels
        chans_to_drop = [c for c in ['HEOGL', 'HEOGR', 'VEOGL', 'STI 014', 'Status'] if c in raw.ch_names]
        raw.drop_channels(chans_to_drop)

        events['empty'] = 0
        events_c = events[events['trial_type'].notna()]

        # Epoch around  off+
        events_id = {"off+": 2}
        events_c = events_c[events_c['trial_type'] == 'off+']
        events_c['cue_num'] = [events_id[s] for s in events_c.trial_type]
        events_cues = np.asarray(events_c[['sample', 'empty', 'cue_num']])

        #----------------------------------------------------------------------------------
        # Epoch for Connectivity at the sensor space level

        connectivity_cues_strials = mne.Epochs(
            raw,
            events=events_cues,
            event_id=events_id,
            tmin=param['tfrepochstart'],
            tmax=param['tfrepochend'],
            preload=True,
            verbose=False)

        # Set frequency band of interest (e.g., beta)
        fmin, fmax = 0.5, 4.0  # Beta band
        sfreq = raw.info["sfreq"]  # Sampling frequency

        # Compute connectivity using PLI
        con = spectral_connectivity_epochs(
            connectivity_cues_strials,
            method="pli",
            mode="multitaper",
            sfreq=sfreq,
            fmin=fmin,
            fmax=fmax,
            faverage=True,
            tmin=0.0,  # Exclude baseline
            mt_adaptive=False,
            n_jobs=1,  # Use one job to prevent overload
        )
 
        labels = connectivity_cues_strials.ch_names  # EEG sensor labels

        con_matrix = con.get_data(output='dense')[:, :, 0]  # Get connectivity matrix

        # Save individual participant's connectivity plot
        plot_connectivity_circle_plot(con_matrix, labels, outdir_con, p)
        
        # Save the connectivity matrix plot
        plot_matrix(con_matrix, 'pli', outdir_con, p)

        # Accumulate for average connectivity
        if total_connectivity is None:
            total_connectivity = np.zeros_like(con_matrix)

        total_connectivity += con_matrix
        n_participants += 1

        # Clear for memory
        del connectivity_cues_strials, con_matrix, con
        gc.collect()  # Trigger garbage collection

        #----------------------------------------------------------------------------------
        # Check drop statistics and append
        percleft_cue.append((len(erps) - np.sum(meta.badtrial)) / len(erps) * 100)
        percremoved_cue_comperp.append(100 - ((125 - allbad) / 125 * 100))

    except Exception as e:
        print(f"Error with participant {p}: {e}")
        continue

#----------------------------------------------------------------------------------
# Save rejection stats for all participants
removed_frame['percleft_cue'] = percleft_cue
removed_frame['percremoved_cue_comperp'] = percremoved_cue_comperp
removed_frame.to_csv(opj(outpath, 'connectivity_rejectionstats.csv'))

#----------------------------------------------------------------------------------
# Plot and save average connectivity across participants

average_connectivity = total_connectivity / n_participants

# Save the average connectivity plot
plot_connectivity_circle_plot(average_connectivity, labels, outpath, 'average_connectivity_circle')


# Save the average connectivity matrix plot
plot_matrix(average_connectivity, 'pli', outpath, 'average')

gc.collect()  # Ensure memory cleanup at the end
