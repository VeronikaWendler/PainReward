# @ : -*- coding: utf-8 -*-
# @ Author: Michel-Pierre Coll (michel-pierre.coll@psy.ulaval.ca), edited by Veronika Wendler (2025)
# @ Date: 2024
# @ Description: plotting the eeg regression models from the eeg_erp_massunivariate_prep.py file

#---------------------------------------------------------------------------------------------------
# importing libraries

import mne
import pandas as pd
import numpy as np
from os.path import join as opj
import matplotlib.pyplot as plt
from bids import BIDSLayout
from mne.viz import plot_topomap
import seaborn as sns
import os
import scipy.stats
import re
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from statsmodels.distributions.empirical_distribution import ECDF
from pathlib import Path
from mne.stats import spatio_temporal_cluster_1samp_test, combine_adjacency
from mne.channels import find_ch_adjacency
import scipy.stats as stats

#-----------------------------------------------------------------------------------------------------
#
# Set bids directory
PROJECT_DIR = Path(os.getenv("PROJECT_DIR", "/workspace"))
inpath = PROJECT_DIR / "EEG" / "PainReward_sub-001-050" / "painrewardeegdata"
outpathall = PROJECT_DIR / "EEG" / "PainReward_sub-001-050" / "painrewardeegdata" / "derivatives"

layout = BIDSLayout(inpath)
part = pd.read_csv(opj(inpath, 'participants.tsv'), sep='\t')
layout = BIDSLayout(outpathall)

version = 11 # 1 for decision phase, 2 for passive phase, 3 = decision RT + 3 GLMs

# noz     - NO_Zscoring      (raw regressors + raw RT)
# z       - Zscoring         (z-scored regressors + z-scored RT)
# partz   - PartZscoring     (raw regressors + z-scored RT)

glm_version = 'z'   

if version == 1:
    outpath = opj(outpathall, 'statistics_new/erps_massuni_drift_mod_9_passive')
    outfigpath = opj(outpathall, 'figures/erps_massuni_drift_mod_9_passive')
    if not os.path.exists(outfigpath):
        os.mkdir(outfigpath)
elif version == 2:
    outpath = opj(outpathall, 'statistics_new/erps_massuni_drift_mod_9_RT')
    outfigpath = opj(outpathall, 'figures/erps_massuni_drift_mod_9_RT')
    if not os.path.exists(outfigpath):
        os.mkdir(outfigpath)
elif version == 3:
    outpath = opj(outpathall, 'statistics_new/erps_massuni_drift_mod_9_RT_3GLMs')
    outfigpath = opj(outpathall, 'figures/erps_massuni_drift_mod_9_RT_3GLMs')
    if not os.path.exists(outfigpath):
        os.mkdir(outfigpath)
elif version == 4:
    outpath = opj(outpathall, 'statistics_new/erps_massuni_drift_mod_9_RTbin')
    outfigpath = opj(outpathall, 'figures/erps_massuni_drift_mod_9_RTbin')
    if not os.path.exists(outfigpath):
        os.mkdir(outfigpath)
elif version == 5:
    outpath = opj(outpathall, 'statistics_new/erps_massuni_drift_mod_9_subjectGLM')
    outfigpath = opj(outpathall, 'figures/erps_massuni_drift_mod_9_subjectGLM')
    if not os.path.exists(outfigpath):
        os.mkdir(outfigpath)
elif version == 6:        
    outpath = opj(outpathall, 'statistics_new/erps_massuni_drift_mod_9_v6_beta_vs_drift')
    outfigpath = opj(outpathall, 'figures/erps_massuni_drift_mod_9_v6_beta_vs_drift')
    if not os.path.exists(outfigpath):
        os.mkdir(outfigpath)
elif version == 7:        
    outpath = opj(outpathall, 'statistics_new/erps_massuni_drift_mod_9_sv_pain_para')
    outfigpath = opj(outpathall, 'figures/erps_massuni_drift_mod_9_sv_pain_para')
    if not os.path.exists(outfigpath):
        os.mkdir(outfigpath)
elif version == 8:
    outpath = opj(outpathall, 'statistics_new/tfr_mod_9_v8_drift_ROI')
    outfigpath = opj(outpathall, 'figures/tfr_mod_9_v8_drift_ROI')
    if not os.path.exists(outfigpath):
        os.mkdir(outfigpath)
elif version == 9:
    outpath = opj(outpathall, 'statistics_new/tfr_mod_9_v9_sv_pain_para_RT_control')
    outfigpath = opj(outpathall, 'figures/tfr_mod_9_v9_sv_pain_para_RT_control')
    if not os.path.exists(outfigpath):
        os.mkdir(outfigpath)
elif version == 10:
    outpath = opj(outpathall, 'statistics_new/tfr_mod_9_v10_sv_vs_pain_RT')
    outfigpath = opj(outpathall, 'figures/tfr_mod_9_v10_sv_vs_pain_RT')
    if not os.path.exists(outfigpath):
        os.mkdir(outfigpath)
elif version == 11:
    outpath = opj(outpathall, 'statistics_new/tfr_mod_9_v9_sv_pain_para')
    outfigpath = opj(outpathall, 'figures/tfr_mod_9_v9_sv_pain_para_wholebrain')
    if not os.path.exists(outfigpath):
        os.mkdir(outfigpath)
else:
    print("No Version")

# map glm_version subfolder + file + figure name
if glm_version == 'noz':
    stats_subdir = 'NO_Zscoring'
    suffix = '_noz'       
    fig_prefix = 'noz_'   
elif glm_version == 'z':
    stats_subdir = 'Zscoring'
    suffix = ''           
    fig_prefix = 'z_'
elif glm_version == 'partz':
    stats_subdir = 'PartZscoring'
    suffix = ''          
    fig_prefix = 'partz_'
else:
    raise ValueError(f"Check glm_version: {glm_version}")

outpath_glm = opj(outpath, stats_subdir)

# 6 regressors total
param = {
    'alpha': 0.05,     
    'titlefontsize': 12,
    'labelfontsize': 12,
    'ticksfontsize': 11,
    'legendfontsize': 10,
    'testresampfreq': 1024,
}

plt.rc("axes.spines", top=False, right=False)
plt.rcParams['font.family'] = 'DejaVu Sans'

# -----------------------------------------------------------------------------------------------------------------
# Regressor bookkeeping – must match massunivariate script
# -----------------------------------------------------------------------------------------------------------------

regvars = ['painlevel', 'moneylevel', 'interaction']
regvarsnames = ['Pain', 'Money', 'Interaction']

# full_regvars = [
#     'painlevel', 'moneylevel', 'interaction',
#     'v_pain_contrib', 'v_money_contrib', 'v_interaction_contrib'
# ]

# regvarsnames = [
#     'pain_raw', 'money_raw', 'interaction_raw',
#     'V_pain_contrib', 'V_money_contrib', 'V_interaction_contrib'
# ]

# regvars_v5 = [
#     'v_painlevel_subj', 'v_moneylevel_subj', 'v_interaction_subj',
#     'a_painlevel_subj', 'a_moneylevel_subj', 'a_interaction_subj'
# ]

# regvarsnames_v5 = [
#     'V_pain_subj', 'V_money_subj', 'V_interaction_subj',
#     'A_pain_subj', 'A_money_subj', 'A_interaction_subj'
# ]

if version == 7:
    regvars = ['sv_pain_para']
    regvarsnames = ['SV_pain_para']
    
    
plot_times = [0.2, 0.4, 0.6, 0.8, 1.0]
chan_to_plot = ['Fz', 'FCz', 'POz', 'Cz', 'CPz', 'Pz', 'Oz']


# Version 1, 2, 3 ---------------------------------------------------------------------------------------------------


if version in [1, 2, 3]:

    # Load second-level stats and betas from the new (z-scored) mass-univariate GLM
    tvals = np.load(opj(outpath_glm, 'ols_2ndlevel_tvals.npy'))     # shape: (n_reg, n_times, n_chans)
    pvals = np.load(opj(outpath_glm, 'ols_2ndlevel_pvals.npy'))     # shape: (n_reg, n_times, n_chans)

    beta_gavg = np.load(opj(outpath_glm, 'ols_2ndlevel_betasavg.npy'),
                        allow_pickle=True)                          # list-like of Evoked
    allbetas = np.load(opj(outpath_glm, 'ols_2ndlevel_betas.npy'),
                       allow_pickle=True)                           # shape: (n_subj, n_reg, n_chans, n_times)

    # Time points for topomaps (in seconds → indices)
    times_pos = [np.abs(beta_gavg[0].times - t).argmin() for t in plot_times]

    # Loop over the 3 regressors: pain, money, interaction
    for ridx, regvar in enumerate(regvars):

        regvarname = regvarsnames[ridx]

        #simple colour choice per regressor
        if regvar == 'painlevel':
            vminmax = 6
            cmap = 'Blues'
        elif regvar == 'moneylevel':
            vminmax = 6
            cmap = 'Greens'
        elif regvar == 'interaction':
            vminmax = 6
            cmap = 'Purples'
        else:
            vminmax = 6
            cmap = 'viridis'

        # Epochs that were used for this regressor GLM
        all_epos = mne.read_epochs(
            opj(outpath_glm, f'ols_2ndlevel_allepochs-epo_{regvar}.fif')
        )

        beta_gavg_nomast = beta_gavg[ridx].copy()
        # boolean mask for "keep" channels (exclude mastoids)
        chankeep = np.array([c not in ['M1', 'M2']
                             for c in beta_gavg[ridx].ch_names])

        # -----------------------------------------------------------------
        # Topo of beta – per time window
        # -----------------------------------------------------------------
        for tidx, timepos in enumerate(times_pos):
            fig, topo_axis = plt.subplots(figsize=(1, 1))

            # p-values at this time for all channels
            p_row = pvals[ridx][timepos, :]  # shape (n_channels,)

            # full-length mask: only non-mastoid sig channels are True
            mask = np.zeros_like(p_row, dtype=bool)
            sig_non_mastoid = (p_row < param['alpha']) & chankeep
            mask[sig_non_mastoid] = True

            im, _ = plot_topomap(
                beta_gavg_nomast.data[:, timepos],
                pos=beta_gavg_nomast.info,
                mask=mask,
                mask_params=dict(marker='o',
                                 markerfacecolor='w',
                                 markeredgecolor='k',
                                 linewidth=0,
                                 markersize=2),
                cmap=cmap,
                show=False,
                ch_type='eeg',
                outlines='head',
                extrapolate='head',
                vlim=(-0.15, 0.15),
                axes=topo_axis,
                sensors=False,
                contours=0,
            )
            topo_axis.set_title(f"{int(plot_times[tidx] * 1000)} ms",
                                fontdict={'size': param['labelfontsize']-1},
                                pad=0.1)

            if tidx + 1 == len(plot_times):
                fig2, ax = plt.subplots(figsize=(0.2, 1))
                cbar1 = fig2.colorbar(im, cax=ax,
                                      orientation='vertical', aspect=1)
                cbar1.set_label('Beta (z)', rotation=270,
                                labelpad=12,
                                fontdict={'fontsize': param["labelfontsize"]-1})
                cbar1.ax.tick_params(labelsize=param['ticksfontsize']-2)
                fig2.savefig(opj(outfigpath,
                                 f'{fig_prefix}fig_topo_beta_cbar_{regvar}.svg'),
                             dpi=600, bbox_inches='tight')

            fig.savefig(
                opj(outfigpath,
                    f'{fig_prefix}fig_ols_erps_betas_topo_{regvar}_{tidx}.svg'),
                dpi=600,
                bbox_inches='tight'
            )

        # -----------------------------------------------------------------
        # Binned-by-regressor line plots and topomaps 
        # -----------------------------------------------------------------
        for c in chan_to_plot:
            fig, line_axis = plt.subplots(1, 1, figsize=(4, 2.5))
            all_epos.metadata.reset_index()

            # Binning on regressor (trial-wise Z-scored regressor still has continuous spread)
            nbins = 5
            all_epos.metadata['bin'] = 0
            unique_vals = all_epos.metadata[regvar].nunique()
            nbins_eff = min(nbins, unique_vals)

            all_epos.metadata['bin'], bins = pd.qcut(
                all_epos.metadata[regvar],
                q=nbins_eff,
                labels=False,
                retbins=True,
                duplicates='drop'
            )
            all_epos.metadata['bin' + '_' + regvar] = all_epos.metadata['bin']

            # Bin labels (for sanity / debugging, not heavily used)
            bin_labels = []
            for bidx, b in enumerate(bins):
                if bidx < len(bins) - 1:
                    lab = f"{round(b, 3)}–{round(bins[bidx+1], 3)}"
                    bin_labels.append(lab)

            # Average within participants
            sub_evokeds = []
            for p_id in all_epos.metadata['participant_id'].unique():
                sub_dat = all_epos[all_epos.metadata['participant_id'] == p_id]
                sub_evoked = {}
                for val in range(nbins_eff):
                    if np.sum(sub_dat.metadata['bin'] == val) != 0:
                        sub_evoked[val] = sub_dat[sub_dat.metadata['bin'] == val].average()
                    else:
                        sub_evoked[val] = 0
                sub_evokeds.append(sub_evoked)

            # Grand average over subjects
            evokeds = dict()
            for i in range(nbins_eff):
                evoked_list = [sub_evoked[i] for sub_evoked in sub_evokeds
                               if sub_evoked[i] != 0]
                if len(evoked_list) == 0:
                    print(f"Skipping bin {i+1}: no valid epochs in this bin for any sub")
                    continue
                evokeds[str(i+1)] = mne.grand_average(evoked_list)

            pick = beta_gavg[ridx].ch_names.index(c)

            line_axis.set_ylabel(f'ERP (binned by {regvarname})',
                                 fontdict={'size': param['labelfontsize']})

            # Colourbar (separate figure, using mne convenience)
            if len(evokeds) > 0:
                _, axis = plt.subplots(figsize=(4, 2.5))
                cbarout = mne.viz.plot_compare_evokeds(
                    evokeds,
                    picks=pick,
                    cmap=(regvarname + "\n(Bin)", cmap),
                    show_sensors=False,
                    show=False,
                    axes=axis
                )
                cbarout[0].axes[-1].yaxis.label.set_size(param['labelfontsize'])
                cbarout[0].axes[-1].tick_params(labelsize=param['ticksfontsize'])
                cbarout[0].axes[0].remove()
                cbarout[0].savefig(
                    opj(outfigpath,
                        f'{fig_prefix}fig_ols_erps_betas_line_cbar_{regvar}_{c}.svg'),
                    dpi=800,
                    bbox_inches='tight'
                )

            bin_ids = sorted(evokeds.keys(), key=lambda x: int(x))

            for idx2, bin_id in enumerate(bin_ids):
                line_axis.plot(
                    all_epos[0].times * 1000,
                    evokeds[bin_id].data[pick, :] * 1e6,
                    label=str(idx2 + 1),
                    linewidth=2,
                    color=plt.get_cmap(cmap)(idx2 / max(1, len(bin_ids)-1))
                )

            line_axis.tick_params(labelsize=12)
            line_axis.set_xlabel('Time (ms)',
                                 fontdict={'size': param['labelfontsize']})
            line_axis.set_ylabel('Amplitude (µV)',
                                 fontdict={'size': param['labelfontsize']})
            line_axis.axhline(0, linestyle='--', color='gray')
            line_axis.axvline(0, ymin=-0.2, ymax=0.2,
                              linestyle='--', color='gray')
            line_axis.get_xaxis().tick_bottom()
            line_axis.get_yaxis().tick_left()
            line_axis.set_xticks(ticks=np.arange(-200, 1200, 200))
            line_axis.set_xticklabels(
                labels=[str(i) for i in np.arange(-200, 1200, 200)]
            )
            line_axis.tick_params(labelsize=param['ticksfontsize'])
            fig.tight_layout()
            fig.savefig(
                opj(outfigpath,
                    f'{fig_prefix}fig_ols_erps_amp_bins_{regvar}_{c}.svg'),
                dpi=600,
                bbox_inches='tight'
            )

        # Topo of binned amplitude at 0.6 s
        bin_ids = sorted(evokeds.keys(), key=lambda x: int(x))

        for idx2, binnum in enumerate(bin_ids):
            fig, topo_axis = plt.subplots(figsize=(1, 1))

            tidx = np.argmin(np.abs(evokeds[binnum].times - 0.6))
            dat = evokeds[binnum].data[:, tidx] * 1e6

            im, _ = plot_topomap(
                dat,
                pos=evokeds[binnum].info,
                cmap=cmap,
                show=False,
                ch_type='eeg',
                outlines='head',
                vlim=(-vminmax, vminmax),
                extrapolate='head',
                axes=topo_axis,
                sensors=False,
                contours=0,
            )
            topo_axis.set_title('Bin ' + binnum,
                                fontdict={'size': param['labelfontsize']-1},
                                pad=0.1)

            fig.savefig(
                opj(outfigpath,
                    f'{fig_prefix}fig_binsamp_topo_{regvar}_bin{binnum}.svg'),
                dpi=600, bbox_inches='tight'
            )

            if idx2 + 1 == len(bin_ids):
                fig2, ax = plt.subplots(figsize=(0.2, 1))
                cbar1 = fig2.colorbar(im, cax=ax,
                                      orientation='vertical', aspect=1)
                cbar1.set_label(
                    'Amplitude (µV)',
                    rotation=270,
                    labelpad=12,
                    fontdict={'fontsize': param["labelfontsize"]-1}
                )
                cbar1.ax.tick_params(labelsize=param['ticksfontsize']-2)
                fig2.savefig(
                    opj(outfigpath,
                        f'{fig_prefix}fig_topo_bins_cbar_{regvar}.svg'),
                    dpi=600, bbox_inches='tight'
                )

        # -----------------------------------------------------------------
        # Mean beta and SEM over participants
        # -----------------------------------------------------------------
        for c in chan_to_plot:
            fig, line_axis = plt.subplots(1, 1, figsize=(4, 2.5))

            all_epos.metadata.reset_index()
            pick = beta_gavg[ridx].ch_names.index(c)

            sub_avg = []
            for s in range(allbetas.shape[0]):
                sub_avg.append(allbetas[s, ridx, pick, :])
            sub_avg = np.stack(sub_avg)

            sem = scipy.stats.sem(sub_avg, axis=0)
            mean = beta_gavg[ridx].data[pick, :]

            line_axis.set_ylabel(f'β ({regvarname}, z)',
                                 fontdict={'size': param['labelfontsize']})
            line_axis.set_xlabel('Time (ms)',
                                 fontdict={'size': param['labelfontsize']})

            line_axis.plot(all_epos[0].times * 1000,
                           mean,
                           linewidth=3)
            line_axis.fill_between(all_epos[0].times * 1000,
                                   mean - sem,
                                   mean + sem,
                                   alpha=0.3)

            line_axis.set_ylim((-0.25, 0.25))
            line_axis.axhline(0, linestyle='--', color='gray')
            line_axis.axvline(0, ymin=0, ymax=0.2,
                              linestyle='--', color='gray')
            line_axis.get_xaxis().tick_bottom()
            line_axis.get_yaxis().tick_left()
            line_axis.tick_params(axis='both',
                                  labelsize=param['ticksfontsize'])

            timestep = 1000.0 / param['testresampfreq']  # ms step
            for tidx2, t2 in enumerate(all_epos[0].times * 1000):
                if pvals[ridx][tidx2, pick] < param['alpha']:
                    line_axis.fill_between(
                        [t2, t2 + timestep],
                        -0.02, -0.005,
                        alpha=0.3,
                        facecolor='red'
                    )

            line_axis.set_xticks(ticks=np.arange(-200, 1200, 200))
            line_axis.set_xticklabels(
                labels=[str(i) for i in np.arange(-200, 1200, 200)]
            )
            fig.tight_layout()
            fig.savefig(
                opj(outfigpath,
                    f'{fig_prefix}fig_ols_erps_betas_{regvar}_{c}.svg'),
                dpi=600,
                bbox_inches='tight')


# ---------------------------------------------------------------------------------------------------
# beta-difference cluster test: pain vs interaction
# ---------------------------------------------------------------------------------------------------

if version in [1, 2, 3]:
    diff_label = 'pain_minus_interaction'  

    times = beta_gavg[0].times
    info = beta_gavg[0].info
    diff_times_pos = [np.abs(times - t).argmin() for t in plot_times]

    # Exclude mastoids in masks
    chankeep = np.array([c not in ['M1', 'M2'] for c in info['ch_names']])

    tdiff = np.load(opj(outpath_glm, f'ols_2ndlevel_tval_diff_{diff_label}.npy'))
    pdiff = np.load(opj(outpath_glm, f'ols_2ndlevel_pval_diff_{diff_label}.npy'))

    for tidx, time_idx in enumerate(diff_times_pos):
        t_time = plot_times[tidx]
        t_ms = int(t_time * 1000)

        # p-values at this time
        p_row = pdiff[time_idx, :]

        alpha_diff = param['alpha']   # = 0.05/3
        mask = (p_row < alpha_diff) & chankeep
        
        fig, ax = plt.subplots(figsize=(2, 2))
        im, _ = plot_topomap(
            tdiff[time_idx, :],
            pos=info,
            mask=mask,
            mask_params=dict(marker='o',
                             markerfacecolor='w',
                             markeredgecolor='k',
                             linewidth=0,
                             markersize=3),
            cmap='RdBu_r',
            show=False,
            ch_type='eeg',
            outlines='head',
            extrapolate='head',
            axes=ax,
            sensors=False,
            contours=0,
        )
        ax.set_title(f'pain − interaction, {t_ms} ms',
                     fontdict={'size': param['labelfontsize']-1},
                     pad=0.1)

        # Colourbar
        fig2, cax = plt.subplots(figsize=(0.2, 1))
        cbar = fig2.colorbar(im, cax=cax, orientation='vertical', aspect=1)
        cbar.set_label('t (pain − interaction)', rotation=270, labelpad=12,
                       fontdict={'fontsize': param["labelfontsize"]-1})
        cbar.ax.tick_params(labelsize=param['ticksfontsize']-2)

        fig.savefig(
            opj(outfigpath,
                f'{fig_prefix}fig_topo_diff_{diff_label}_{t_ms}ms.svg'),
            dpi=600, bbox_inches='tight'
        )
        fig2.savefig(
            opj(outfigpath,
                f'{fig_prefix}fig_topo_diff_{diff_label}_{t_ms}ms_cbar.svg'),
            dpi=600, bbox_inches='tight'
        )



# Version 4 --------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------

elif version == 4:
    print("\nPlotting RT-binned results (Version 4)\n")

    # RT bins used in the massunivariate prep
    bin_labels = ["fast", "medium", "slow"]

    # per-bin plotting loop
    for bin_name in bin_labels:
        for ridx, regvar in enumerate(regvars):
            # colour map for regressor
            if ridx == 0:
                cmap = 'viridis'
            elif ridx == 1:
                cmap = 'cividis'
            elif ridx == 2:
                cmap = 'plasma'

            tvals_path = opj(outpath, f"{bin_name}_tvals_{regvar}.npy")
            pvals_path = opj(outpath, f"{bin_name}_pvals_{regvar}.npy")
            epo_path = opj(outpath, f"{bin_name}_allepochs-epo_{regvar}.fif")

            if not (os.path.exists(tvals_path)
                    and os.path.exists(pvals_path)
                    and os.path.exists(epo_path)):
                print(f"Skipping bin '{bin_name}', regvar '{regvar}' - missing files.")
                continue

            # load stats + epochs
            tvals = np.load(tvals_path)   # shape (n_times, n_channels)
            pvals = np.load(pvals_path)   # shape (n_times, n_channels)
            all_epos = mne.read_epochs(epo_path, preload=False)

            times = all_epos.times
            ch_names = all_epos.ch_names
            chankeep = [True if c not in ['M1', 'M2'] else False for c in ch_names]

            # -----------------------------------------------------------------
            # Topomaps of t-values (cluster-corrected)
            # -----------------------------------------------------------------
            times_pos = [np.abs(times - t).argmin() for t in plot_times]

            for tidx, timepos in enumerate(times_pos):
                fig, topo_axis = plt.subplots(figsize=(1, 1))

                dat = tvals[timepos, :]  # (n_channels,)
                # mask only for non-mastoid channels
                mask_vals = pvals[timepos, chankeep] < param['alpha']

                im, _ = plot_topomap(
                    dat[chankeep],
                    pos=mne.pick_info(all_epos.info,
                                      [i for i, k in enumerate(chankeep) if k]),
                    mask=mask_vals,
                    mask_params=dict(marker='o',
                                     markerfacecolor='w',
                                     markeredgecolor='k',
                                     linewidth=0,
                                     markersize=2),
                    cmap=cmap,
                    show=False,
                    ch_type='eeg',
                    outlines='head',
                    extrapolate='head',
                    vlim=(None, None),
                    axes=topo_axis,
                    sensors=False,
                    contours=0,
                )
                topo_axis.set_title(
                    f"{bin_name} – {int(plot_times[tidx]*1000)} ms",
                    fontdict={'size': param['labelfontsize']-1},
                    pad=0.1
                )

                if tidx + 1 == len(plot_times):
                    fig2, ax = plt.subplots(figsize=(0.2, 1))
                    cbar1 = fig2.colorbar(im, cax=ax,
                                          orientation='vertical', aspect=1)
                    cbar1.set_label(
                        't-value',
                        rotation=270,
                        labelpad=12,
                        fontdict={'fontsize': param["labelfontsize"]-1}
                    )
                    cbar1.ax.tick_params(labelsize=param['ticksfontsize']-2)
                    fig2.savefig(
                        opj(outfigpath,
                            f'fig_topo_tvals_cbar_{bin_name}_{regvar}.svg'),
                        dpi=600,
                        bbox_inches='tight'
                    )

                fig.savefig(
                    opj(outfigpath,
                        f'fig_ols_erps_tvals_topo_{bin_name}_{regvar}_{tidx}.svg'),
                    dpi=600,
                    bbox_inches='tight'
                )

            # -----------------------------------------------------------------
            # t-value timecourses at centro-parietal channels (per bin)
            # -----------------------------------------------------------------
            for c in chan_to_plot:
                if c not in ch_names:
                    continue

                pick = ch_names.index(c)
                fig, ax = plt.subplots(1, 1, figsize=(4, 2.5))

                t_series = tvals[:, pick]
                p_series = pvals[:, pick]

                ax.plot(times * 1000, t_series, linewidth=2)
                ax.axhline(0, linestyle='--', color='gray')
                ax.axvline(0, ymin=0, ymax=0.2,
                           linestyle='--', color='gray')

                ax.set_xlabel('Time (ms)',
                              fontdict={'size': param['labelfontsize']})
                ax.set_ylabel('t-value (' + regvarsnames[ridx] + ')',
                              fontdict={'size': param['labelfontsize']})
                ax.tick_params(labelsize=param['ticksfontsize'])

                # significance shading
                timestep = 1000.0 * (times[1] - times[0])  # in ms
                for tidx2, t2 in enumerate(times * 1000):
                    if p_series[tidx2] < param['alpha']:
                        ax.fill_between(
                            [t2, t2 + timestep],
                            ax.get_ylim()[0],
                            ax.get_ylim()[0] + 0.2 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
                            alpha=0.3,
                            facecolor='red'
                        )

                ax.set_xticks(ticks=np.arange(-200, 1200, 200))
                ax.set_xticklabels(
                    labels=[str(i) for i in np.arange(-200, 1200, 200)]
                )

                fig.tight_layout()
                fig.savefig(
                    opj(outfigpath,
                        f'fig_ols_erps_tvals_{bin_name}_{regvar}_{c}.svg'),
                    dpi=600,
                    bbox_inches='tight'
                )

    print("\nVersion 4 plotting done ;))))\n")

# -------------------------------------------------------------------------------------
# Between-subject subject-level GLM 


if version == 5:
    from pathlib import Path

    # Use the same glm_version / stats_subdir / fig_prefix as defined above
    stats_dir   = Path(outpath) / stats_subdir
    cluster_dir = Path(outpath) / f"{stats_subdir}_cluster"

    # Load group-level epochs for info / time axis
    group_dir = PROJECT_DIR / "EEG" / "PainReward_sub-001-050" / "painrewardeegdata" / "derivatives" / "group_level"
    name = "decision"
    group_epochs_fname = group_dir / f"{name}_off+_subaveraged-epo.fif"
    group_epochs = mne.read_epochs(group_epochs_fname)

    info  = group_epochs.info
    times = group_epochs.times

    # Regressors: MUST match version-5 mass-univariate (only a_* terms)
    regvars_v5 = [
        'a_painlevel_subj',
        'a_moneylevel_subj',
        'a_interaction_subj'
    ]
    regvarsnames_v5 = [
        'A_pain_subj',
        'A_money_subj',
        'A_interaction_subj'
    ]

    # Time indices for topomaps (same windows as other versions)
    plot_times = [0.2, 0.4, 0.6, 0.8, 1.0]
    times_pos = [np.abs(times - t).argmin() for t in plot_times]

    # Exclude mastoids
    chankeep = np.array([c not in ['M1', 'M2'] for c in info['ch_names']])

    # For version 5 we rely ONLY on the cluster correction (space × time) per regressor.
    # No extra Bonferroni across regressors here.
    alpha_clust = 0.05

    # ------------------------------------------------------------------
    # Topomaps of subject-level betas with cluster-corrected mask
    # ------------------------------------------------------------------
    for ridx, regvar in enumerate(regvars_v5):
        regvarname = regvarsnames_v5[ridx]

        beta_file   = stats_dir   / f'groupglm_beta_{regvar}.npy'
        pfile_clust = cluster_dir / f'groupglm_cluster_pval_{regvar}.npy'

        if not (beta_file.exists() and pfile_clust.exists()):
            print(f"Skipping {regvar}: files not found in {stats_dir} / {cluster_dir}")
            continue

        # load beta and cluster-corrected p-values
        beta_data   = np.load(beta_file)        # (n_chan, n_time)
        pvals_clust = np.load(pfile_clust)      # (n_time, n_chan)

        # Wrap beta into an Evoked for convenience
        beta_ev = mne.EvokedArray(beta_data, info, tmin=times[0])

        # --------- topomap over time windows ----------
        for tidx, tpos in enumerate(times_pos):
            fig, topo_axis = plt.subplots(figsize=(1.5, 1.5))

            p_row = pvals_clust[tpos, :]   # (n_chan,)
            mask  = np.zeros_like(p_row, dtype=bool)
            sig_non_mastoid = (p_row < alpha_clust) & chankeep
            mask[sig_non_mastoid] = True

            im, _ = plot_topomap(
                beta_ev.data[:, tpos],
                pos=beta_ev.info,
                mask=mask,
                mask_params=dict(marker='o',
                                 markerfacecolor='w',
                                 markeredgecolor='k',
                                 linewidth=0,
                                 markersize=2),
                cmap='viridis',
                show=False,
                ch_type='eeg',
                outlines='head',
                extrapolate='head',
                vlim=(-0.15, 0.15),
                axes=topo_axis,
                sensors=False,
                contours=0,
            )
            topo_axis.set_title(
                f"{regvarname}\n{int(plot_times[tidx]*1000)} ms",
                fontdict={'size': param['labelfontsize']-1},
                pad=0.1
            )

            fig.savefig(
                opj(outfigpath,
                    f'{fig_prefix}v5_topo_beta_{regvar}_{tidx}.svg'),
                dpi=600,
                bbox_inches='tight'
            )

            # save colourbar on last time point
            if tidx + 1 == len(times_pos):
                fig2, ax = plt.subplots(figsize=(0.3, 1.2))
                cbar = fig2.colorbar(im, cax=ax, orientation='vertical', aspect=1)
                cbar.set_label('Beta', rotation=270, labelpad=12,
                               fontdict={'fontsize': param['labelfontsize']-1})
                cbar.ax.tick_params(labelsize=param['ticksfontsize']-2)
                fig2.savefig(
                    opj(outfigpath,
                        f'{fig_prefix}v5_topo_beta_cbar_{regvar}.svg'),
                    dpi=600,
                    bbox_inches='tight'
                )

        # ------------------------------------------------------------------
        # Time-course at ROI channels with significance bar
        # ------------------------------------------------------------------
        for c in chan_to_plot:
            if c not in beta_ev.ch_names:
                continue

            pick = beta_ev.ch_names.index(c)
            fig, ax = plt.subplots(1, 1, figsize=(4, 2.5))

            y = beta_ev.data[pick, :]      # beta over time
            ax.plot(times * 1000, y, linewidth=2)

            ax.set_xlabel('Time (ms)', fontdict={'size': param['labelfontsize']})
            ax.set_ylabel(f'Beta ({regvarname})', fontdict={'size': param['labelfontsize']})
            ax.axhline(0, linestyle='--', color='gray')
            ax.axvline(0, linestyle='--', color='gray')

            # mark cluster-corrected significant samples (per regressor, α = 0.05)
            timestep = 1000.0 / param['testresampfreq']   # ms
            ymin = y.min()
            for tidx2, t in enumerate(times * 1000):
                if pvals_clust[tidx2, pick] < alpha_clust:
                    ax.fill_between(
                        [t, t + timestep],
                        ymin - 0.02,
                        ymin - 0.005,
                        alpha=0.4
                    )

            ax.set_xticks(np.arange(-200, 1200, 200))
            ax.set_xticklabels([str(i) for i in np.arange(-200, 1200, 200)])
            ax.tick_params(labelsize=param['ticksfontsize'])
            fig.tight_layout()
            fig.savefig(
                opj(outfigpath,
                    f'{fig_prefix}v5_timecourse_{regvar}_{c}.svg'),
                dpi=600,
                bbox_inches='tight'
            )

    # ------------------------------------------------------------
    # LPP ROI time-cluster plot
    # -----------------------------------------------------------
    roi_cluster_dir = Path(outpath) / "LPP_ROI_cluster"
    
    if roi_cluster_dir.exists():
        print("\nPlotting LPP ROI time-cluster results (Version 5)")

        lpp_tmin, lpp_tmax = 0.4, 0.8
        tmask = (times >= lpp_tmin) & (times <= lpp_tmax)
        times_roi = times[tmask] * 1000  # ms

        for ridx, regvar in enumerate(regvars_v5):
            regvarname = regvarsnames_v5[ridx]

            tfile = roi_cluster_dir / f"LPPROI_tval_{regvar}.npy"
            pfile = roi_cluster_dir / f"LPPROI_pval_{regvar}.npy"

            if not (tfile.exists() and pfile.exists()):
                continue

            tvals = np.load(tfile)   # (time,)
            pvals = np.load(pfile)

            fig, ax = plt.subplots(figsize=(4, 2.5))

            ax.plot(times_roi, tvals, lw=2)
            ax.axhline(0, linestyle='--', color='gray')

            # cluster significance bar over time (cluster-corrected p < 0.05)
            dt = times_roi[1] - times_roi[0]
            for i, t in enumerate(times_roi):
                if pvals[i] < 0.05:
                    ax.fill_between(
                        [t, t + dt],
                        tvals.min() - 0.1,
                        tvals.min() - 0.05,
                        color='red',
                        alpha=0.4
                    )

            ax.set_xlabel("Time (ms)")
            ax.set_ylabel("Cluster t-value")
            ax.set_title(f"LPP ROI – {regvarname}")
            ax.tick_params(labelsize=param['ticksfontsize'])

            fig.tight_layout()
            fig.savefig(
                opj(outfigpath, f"{fig_prefix}v5_LPPROI_timecluster_{regvar}.svg"),
                dpi=600,
                bbox_inches="tight"
            )
    else:
        print("No LPP_ROI_cluster directory found — skipping LPP ROI cluster plots.")

        # ------------------------------------------------------------
    # N2 ROI time-cluster plots
    # ------------------------------------------------------------
    n2_cluster_dir = Path(outpath) / "N2_ROI_cluster"
    
    if n2_cluster_dir.exists():
        print("\nPlotting N2 ROI time-cluster results (Version 5)")

        # must match the analysis window used in version-5 massuni
        n2_tmin, n2_tmax = 0.20, 0.40
        n2_tmask = (times >= n2_tmin) & (times <= n2_tmax)
        times_n2 = times[n2_tmask] * 1000  # ms

        for ridx, regvar in enumerate(regvars_v5):
            regvarname = regvarsnames_v5[ridx]

            tfile = n2_cluster_dir / f"N2ROI_tval_{regvar}.npy"
            pfile = n2_cluster_dir / f"N2ROI_pval_{regvar}.npy"

            if not (tfile.exists() and pfile.exists()):
                continue

            tvals = np.load(tfile)   # (time,)
            pvals = np.load(pfile)   # (time,)

            fig, ax = plt.subplots(figsize=(4, 2.5))

            ax.plot(times_n2, tvals, lw=2)
            ax.axhline(0, linestyle='--', color='gray')

            # cluster significance bar (cluster-corrected p < 0.05)
            if len(times_n2) > 1:
                dt = times_n2[1] - times_n2[0]
            else:
                dt = 1.0  # fallback, shouldn't really happen

            for i, t in enumerate(times_n2):
                if pvals[i] < 0.05:
                    ax.fill_between(
                        [t, t + dt],
                        tvals.min() - 0.1,
                        tvals.min() - 0.05,
                        color='red',
                        alpha=0.4
                    )

            ax.set_xlabel("Time (ms)")
            ax.set_ylabel("Cluster t-value")
            ax.set_title(f"N2 ROI – {regvarname}")
            ax.tick_params(labelsize=param['ticksfontsize'])

            fig.tight_layout()
            fig.savefig(
                opj(outfigpath, f"{fig_prefix}v5_N2ROI_timecluster_{regvar}.svg"),
                dpi=600,
                bbox_inches="tight"
            )
    else:
        print("No N2_ROI_cluster directory found — skipping N2 ROI cluster plots.")


    # ------------------------------------------------------------
    # P3b ROI time-cluster plots
    # ------------------------------------------------------------
    p3b_cluster_dir = Path(outpath) / "P3b_ROI_cluster"
    
    if p3b_cluster_dir.exists():
        print("\nPlotting P3b ROI time-cluster results (Version 5)")

        # must match the analysis window used in version-5 massuni
        p3b_tmin, p3b_tmax = 0.25, 0.55
        p3b_tmask = (times >= p3b_tmin) & (times <= p3b_tmax)
        times_p3b = times[p3b_tmask] * 1000  # ms

        for ridx, regvar in enumerate(regvars_v5):
            regvarname = regvarsnames_v5[ridx]

            tfile = p3b_cluster_dir / f"P3bROI_tval_{regvar}.npy"
            pfile = p3b_cluster_dir / f"P3bROI_pval_{regvar}.npy"

            if not (tfile.exists() and pfile.exists()):
                continue

            tvals = np.load(tfile)   # (time,)
            pvals = np.load(pfile)   # (time,)

            fig, ax = plt.subplots(figsize=(4, 2.5))

            ax.plot(times_p3b, tvals, lw=2)
            ax.axhline(0, linestyle='--', color='gray')

            # cluster significance bar (cluster-corrected p < 0.05)
            if len(times_p3b) > 1:
                dt = times_p3b[1] - times_p3b[0]
            else:
                dt = 1.0

            for i, t in enumerate(times_p3b):
                if pvals[i] < 0.05:
                    ax.fill_between(
                        [t, t + dt],
                        tvals.min() - 0.1,
                        tvals.min() - 0.05,
                        color='red',
                        alpha=0.4
                    )

            ax.set_xlabel("Time (ms)")
            ax.set_ylabel("Cluster t-value")
            ax.set_title(f"P3b ROI – {regvarname}")
            ax.tick_params(labelsize=param['ticksfontsize'])

            fig.tight_layout()
            fig.savefig(
                opj(outfigpath, f"{fig_prefix}v5_P3bROI_timecluster_{regvar}.svg"),
                dpi=600,
                bbox_inches="tight"
            )
    else:
        print("No P3b_ROI_cluster directory found — skipping P3b ROI cluster plots.")



# ----------------------------------------------------------------------------------------------------------------------
#
# comparision between beta pain and drift at second-level 

elif version == 6:
    print("\nPlotting Version 6 (beta_pain ~ v) results\n")

    v6_dir = PROJECT_DIR / "EEG" / "PainReward_sub-001-050" / "painrewardeegdata" \
        / "derivatives" / "statistics_new" / "erps_massuni_drift_mod_9_v6_beta_vs_drift"

    # We need info / times from the v3 beta grand-average
    v3_z_dir = PROJECT_DIR / "EEG" / "PainReward_sub-001-050" / "painrewardeegdata" \
        / "derivatives" / "statistics_new" / "erps_massuni_drift_mod_9_RT_3GLMs" / "Zscoring"

    beta_gavg = np.load(v3_z_dir / "ols_2ndlevel_betasavg.npy", allow_pickle=True)
    info = beta_gavg[0].info
    times = beta_gavg[0].times

    reg_labels = ["pain", "money", "interaction"]
    names = ["pain", "money", "interaction"]

    plot_times = [0.2, 0.4, 0.6, 0.8, 1.0]
    times_pos = [np.abs(times - t).argmin() for t in plot_times]
    chankeep = np.array([c not in ['M1', 'M2'] for c in info['ch_names']])

    for label in reg_labels:
        gamma_file = v6_dir / f"v6_gamma1_beta_{label}_vs_v.npy"
        tval_file = v6_dir / f"v6_tvals_beta_{label}_vs_v.npy"
        pval_file = v6_dir / f"v6_pvals_beta_{label}_vs_v.npy"

        if not (gamma_file.exists() and tval_file.exists() and pval_file.exists()):
            print(f"  Missing files for {label}, skipping.")
            continue

        gamma1 = np.load(gamma_file)   # (chan, time)
        tvals = np.load(tval_file)     # (time, chan)
        pvals = np.load(pval_file)     # (time, chan)

        gamma_ev = mne.EvokedArray(gamma1, info, tmin=times[0])
        sig_fdr = np.load(v6_dir / f"v6_sigmask_fdr_{label}.npy") 
        # --------------------------------------------------------------
        # Topomaps of γ1 at selected times (mask = cluster p<α)
        # --------------------------------------------------------------
        for tidx, tpos in enumerate(times_pos):
            fig, ax = plt.subplots(figsize=(1.5, 1.5))
            sig_fdr = np.load(v6_dir / f"v6_sigmask_fdr_{label}.npy")  # (time, chan)
            p_row_mask = sig_fdr[tpos, :]
            mask = p_row_mask & chankeep
            

            vmax = np.max(np.abs(gamma_ev.data))
            im, _ = plot_topomap(
                gamma_ev.data[:, tpos],
                pos=gamma_ev.info,
                mask=mask,
                mask_params=dict(marker='o',
                                 markerfacecolor='w',
                                 markeredgecolor='k',
                                 linewidth=0,
                                 markersize=2),
                cmap='RdBu_r',
                show=False,
                ch_type='eeg',
                outlines='head',
                extrapolate='head',
                vlim=(-vmax, vmax),
                axes=ax,
                sensors=False,
                contours=0,
            )
            ax.set_title(f"{names[label]}\n{int(plot_times[tidx]*1000)} ms",
                         fontdict={'size': param['labelfontsize']-1},
                         pad=0.1)

            fig.savefig(
                opj(outfigpath,
                    f'{fig_prefix}v6_topo_gamma1_{label}_{tidx}.svg'),
                dpi=600,
                bbox_inches='tight'
            )

            if tidx + 1 == len(times_pos):
                fig2, cax = plt.subplots(figsize=(0.3, 1.2))
                cbar = fig2.colorbar(im, cax=cax,
                                     orientation='vertical', aspect=1)
                cbar.set_label('Slope γ₁ (µV / v-unit)',
                               rotation=270, labelpad=12,
                               fontdict={'fontsize': param['labelfontsize']-1})
                cbar.ax.tick_params(labelsize=param['ticksfontsize']-2)
                fig2.savefig(
                    opj(outfigpath,
                        f'{fig_prefix}v6_topo_gamma1_cbar_{label}.svg'),
                    dpi=600,
                    bbox_inches='tight'
                )

        # --------------------------------------------------------------
        # Time-course of γ1 at LPP channels with significance bar
        # --------------------------------------------------------------
        for c in chan_to_plot:
            if c not in gamma_ev.ch_names:
                continue

            pick = gamma_ev.ch_names.index(c)
            fig, ax = plt.subplots(1, 1, figsize=(4, 2.5))

            y = gamma_ev.data[pick, :]
            ax.plot(times * 1000, y, linewidth=2)

            ax.set_xlabel('Time (ms)',
                          fontdict={'size': param['labelfontsize']})
            ax.set_ylabel(f'Slope γ₁ ({names[label]})',
                          fontdict={'size': param['labelfontsize']})
            ax.axhline(0, linestyle='--', color='gray')
            ax.axvline(0, linestyle='--', color='gray')

            timestep = 1000.0 * (times[1] - times[0])
            for ti, tt in enumerate(times * 1000):
                if sig_fdr[ti, pick]:   
                    ax.fill_between(
                        [tt, tt + timestep],
                        ax.get_ylim()[0],
                        ax.get_ylim()[0] + 0.15*(ax.get_ylim()[1]-ax.get_ylim()[0]),
                        alpha=0.3
                        )
            
            
            ax.set_xticks(np.arange(-200, 1200, 200))
            ax.set_xticklabels([str(i) for i in np.arange(-200, 1200, 200)])
            ax.tick_params(labelsize=param['ticksfontsize'])
            fig.tight_layout()
            fig.savefig(
                opj(outfigpath,
                    f'{fig_prefix}v6_timecourse_gamma1_{label}_{c}.svg'),
                dpi=600,
                bbox_inches='tight'
            )


elif version == 7:

    tvals = np.load(opj(outpath_glm, 'ols_2ndlevel_tvals.npy'))
    pvals = np.load(opj(outpath_glm, 'ols_2ndlevel_pvals.npy'))
    beta_gavg = np.load(opj(outpath_glm, 'ols_2ndlevel_betasavg.npy'),
                        allow_pickle=True)
    allbetas = np.load(opj(outpath_glm, 'ols_2ndlevel_betas.npy'),
                       allow_pickle=True)

    times_pos = [np.abs(beta_gavg[0].times - 0.2 - t).argmin() for t in plot_times]

    # only one regressor: sv_pain_para
    for ridx, regvar in enumerate(regvars):
        regvarname = regvarsnames[ridx]

        # pick a colourmap / vminmax for this one
        vminmax = 6
        cmap = 'viridis'

        all_epos = mne.read_epochs(
            opj(outpath_glm, f'ols_2ndlevel_allepochs-epo_{regvar}.fif')
        )

        beta_gavg_nomast = beta_gavg[ridx].copy()
        chankeep = np.array([c not in ['M1', 'M2']
                             for c in beta_gavg[ridx].ch_names])

        # Topo of beta – per time window
        # -----------------------------------------------------------------
        for tidx, timepos in enumerate(times_pos):
            fig, topo_axis = plt.subplots(figsize=(1, 1))

            # p-values at this time for all channels
            p_row = pvals[ridx][timepos, :]  # shape (n_channels,)

            # full-length mask: only non-mastoid sig channels are True
            mask = np.zeros_like(p_row, dtype=bool)
            sig_non_mastoid = (p_row < param['alpha']) & chankeep
            mask[sig_non_mastoid] = True

            im, _ = plot_topomap(
                beta_gavg_nomast.data[:, timepos],
                pos=beta_gavg_nomast.info,
                mask=mask,
                mask_params=dict(marker='o',
                                 markerfacecolor='w',
                                 markeredgecolor='k',
                                 linewidth=0,
                                 markersize=2),
                cmap=cmap,
                show=False,
                ch_type='eeg',
                outlines='head',
                extrapolate='head',
                vlim=(-0.15, 0.15),
                axes=topo_axis,
                sensors=False,
                contours=0,
            )
            topo_axis.set_title(str(int(plot_times[tidx] * 1000)) + ' ms',
                                fontdict={'size': param['labelfontsize']-1},
                                pad=0.1)

            if tidx + 1 == len(plot_times):
                fig2, ax = plt.subplots(figsize=(0.2, 1))
                cbar1 = fig2.colorbar(im, cax=ax,
                                      orientation='vertical', aspect=1)
                cbar1.set_label('Beta', rotation=270,
                                labelpad=12,
                                fontdict={'fontsize': param["labelfontsize"]-1})
                cbar1.ax.tick_params(labelsize=param['ticksfontsize']-2)
                fig2.savefig(opj(outfigpath,
                                 f'{fig_prefix}fig_topo_beta_cbar_{regvar}.svg'),
                             dpi=600, bbox_inches='tight')

            fig.savefig(
                opj(outfigpath,
                    f'{fig_prefix}fig_ols_erps_betas_topo_{regvar}_{tidx}.svg'),
                dpi=600,
                bbox_inches='tight'
            )
            

        # -----------------------------------------------------------------
        # Binned-by-regressor line plots and topomaps 
        # -----------------------------------------------------------------
        for c in chan_to_plot:
            fig, line_axis = plt.subplots(1, 1, figsize=(4, 2.5))
            all_epos.metadata.reset_index()

            # Binning on regressor
            nbins = 5
            all_epos.metadata['bin'] = 0
            unique_vals = all_epos.metadata[regvar].nunique()
            nbins_eff = min(nbins, unique_vals)

            all_epos.metadata['bin'], bins = pd.qcut(
                all_epos.metadata[regvar],
                q=nbins_eff,
                labels=False,
                retbins=True,
                duplicates='drop'
            )
            all_epos.metadata['bin' + '_' + regvar] = all_epos.metadata['bin']

            # Bin labels
            bin_labels = []
            for bidx, b in enumerate(bins):
                if b < 0:
                    b = 0
                if bidx < len(bins)-1:
                    lab = str(round(b, 10)) + '-' + str(round(bins[bidx+1], 10))
                    bin_labels.append(lab)

            # Average within participants
            sub_evokeds = []
            for p_id in all_epos.metadata['participant_id'].unique():
                sub_dat = all_epos[all_epos.metadata['participant_id'] == p_id]
                sub_evoked = {}
                for val in range(nbins):
                    if np.sum(sub_dat.metadata['bin'] == val) != 0:
                        sub_evoked[val] = sub_dat[sub_dat.metadata['bin']
                                                  == val].average()
                    else:
                        sub_evoked[val] = 0
                sub_evokeds.append(sub_evoked)

            # Grand average over subjects
            # evokeds = dict()
            # for i in range(len(bin_labels)):
            #     evoked = [sub_evoked[i] for sub_evoked in sub_evokeds
            #               if sub_evoked[i] != 0]
            #     evokeds[str(i+1)] = mne.grand_average(evoked)
            
            
            evokeds = dict()
            for i in range(len(bin_labels)):
                evoked_list = [sub_evoked[i] for sub_evoked in sub_evokeds
                               if sub_evoked[i] != 0]
            
                if len(evoked_list) == 0:
                    print(f"Skipping bin {i+1}: no valid epochs in this bin for any sub")
                    continue
            
                evokeds[str(i+1)] = mne.grand_average(evoked_list)


            pick = beta_gavg[ridx].ch_names.index(c)

            line_axis.set_ylabel('Beta (' + regvarname + ')',
                                 fontdict={'size': param['labelfontsize']})

            # Colourbar (separate figure)
            _, axis = plt.subplots(figsize=(4, 2.5))
            cbarout = mne.viz.plot_compare_evokeds(
                evokeds,
                picks=pick,
                cmap=(regvarname + "\n(Decile)", cmap),
                show_sensors=False,
                show=False,
                axes=axis
            )
            cbarout[0].axes[-1].yaxis.label.set_size(param['labelfontsize'])
            cbarout[0].axes[-1].tick_params(labelsize=param['ticksfontsize'])
            cbarout[0].axes[0].remove()
            cbarout[0].savefig(opj(outfigpath,
                                   f'{fig_prefix}fig_ols_erps_betas_line_cbar_{regvar}_{c}.svg'),dpi=800,
                               bbox_inches='tight')


            bin_ids = sorted(evokeds.keys(), key=lambda x: int(x))

            for idx2, bin_id in enumerate(bin_ids):
                line_axis.plot(
                    all_epos[0].times * 1000,
                    evokeds[bin_id].data[pick, :] * 1000000,
                    label=str(idx2 + 1),
                    linewidth=2,
                    color=plt.get_cmap(cmap)(idx2 / len(bin_ids))
                )

            line_axis.tick_params(labelsize=12)
            line_axis.set_xlabel('Time (ms)',
                                 fontdict={'size': param['labelfontsize']})
            line_axis.set_ylabel('Amplitude (uV)',
                                 fontdict={'size': param['labelfontsize']})
            line_axis.axhline(0, linestyle='--', color='gray')
            line_axis.axvline(0, ymin=-0.2, ymax=0.2,
                              linestyle='--', color='gray')
            line_axis.get_xaxis().tick_bottom()
            line_axis.get_yaxis().tick_left()
            line_axis.set_xticks(ticks=np.arange(-200, 1200, 200))
            line_axis.set_xticklabels(
                labels=[str(i) for i in np.arange(-200, 1200, 200)]
            )
            line_axis.tick_params(labelsize=param['ticksfontsize'])
            fig.tight_layout()
            fig.savefig(
                opj(outfigpath,
                    f'{fig_prefix}fig_ols_erps_amp_bins_{regvar}_{c}.svg'),
                dpi=600,
                bbox_inches='tight'
            )


        # Topo of binned amplitude at 0.6 s
        bin_ids = sorted(evokeds.keys(), key=lambda x: int(x))

        for idx2, binnum in enumerate(bin_ids):
            fig, topo_axis = plt.subplots(figsize=(1, 1))

            tidx = np.argmin(np.abs(evokeds[binnum].times - 0.6))
            dat = evokeds[binnum].data[:, tidx] * 1000000

            im, _ = plot_topomap(
                dat,
                pos=evokeds[binnum].info,
                cmap=cmap,
                show=False,
                ch_type='eeg',
                outlines='head',
                vlim=(-vminmax, vminmax),
                extrapolate='head',
                axes=topo_axis,
                sensors=False,
                contours=0,
            )
            topo_axis.set_title('Ventile ' + binnum,
                                fontdict={'size': param['labelfontsize']-1},
                                pad=0.1)

            fig.savefig(
                opj(outfigpath,
                    f'{fig_prefix}fig_binsamp_topo_{regvar}_bin{binnum}.svg'),
                dpi=600, bbox_inches='tight'
            )
            

            if idx2 + 1 == len(bin_ids):
                fig2, ax = plt.subplots(figsize=(0.2, 1))
                cbar1 = fig2.colorbar(im, cax=ax,
                                      orientation='vertical', aspect=1)
                cbar1.set_label(
                    'Amplitude (uV)',
                    rotation=270,
                    labelpad=12,
                    fontdict={'fontsize': param["labelfontsize"]-1}
                )
                cbar1.ax.tick_params(labelsize=param['ticksfontsize']-2)
                fig2.savefig(
                    opj(outfigpath,
                        f'{fig_prefix}fig_topo_bins_cbar_{regvar}.svg'),
                    dpi=600, bbox_inches='tight'
                )


        # -----------------------------------------------------------------
        # Mean beta and SEM over participants
        # -----------------------------------------------------------------
        for c in chan_to_plot:
            fig, line_axis = plt.subplots(1, 1, figsize=(4, 2.5))

            all_epos.metadata.reset_index()
            pick = beta_gavg[ridx].ch_names.index(c)

            sub_avg = []
            for s in range(allbetas.shape[0]):
                sub_avg.append(allbetas[s, ridx, pick, :])
            sub_avg = np.stack(sub_avg)

            sem = scipy.stats.sem(sub_avg, axis=0)
            mean = beta_gavg[ridx].data[pick, :]

            clrs = sns.color_palette("deep", 5)

            line_axis.set_ylabel('Beta (' + regvarname + ')',
                                 fontdict={'size': param['labelfontsize']})
            line_axis.set_xlabel('Time (ms)',
                                 fontdict={'size': param['labelfontsize']})

            line_axis.plot(all_epos[0].times * 1000,
                           mean,
                           linewidth=3)
            line_axis.fill_between(all_epos[0].times * 1000,
                                   mean - sem,
                                   mean + sem,
                                   alpha=0.3,
                                   facecolor=clrs[0])

            line_axis.set_ylim((-0.25, 0.25))
            line_axis.axhline(0, linestyle='--', color='gray')
            line_axis.axvline(0, ymin=0, ymax=0.2,
                              linestyle='--', color='gray')
            line_axis.get_xaxis().tick_bottom()
            line_axis.get_yaxis().tick_left()
            line_axis.tick_params(axis='both',
                                  labelsize=param['ticksfontsize'])

            timestep = 1024 / param['testresampfreq']
            for tidx2, t2 in enumerate(all_epos[0].times * 1000):
                if pvals[ridx][tidx2, pick] < param['alpha']:
                    line_axis.fill_between(
                        [t2, t2 + timestep],
                        -0.02, -0.005,
                        alpha=0.3,
                        facecolor='red'
                    )

            line_axis.set_xticks(ticks=np.arange(-200, 1200, 200))
            line_axis.set_xticklabels(
                labels=[str(i) for i in np.arange(-200, 1200, 200)]
            )
            fig.tight_layout()
            fig.savefig(
                opj(outfigpath,
                    f'{fig_prefix}fig_ols_erps_betas_{regvar}_{c}.svg'),
                dpi=600,
                bbox_inches='tight')

# 8
if version == 8:
    print("\n--- Plotting Version 8: TFR ROI vs drift ---")

    # Load ROI summary
    csv_path = opj(outpath, "tfr_roi_theta_alpha_vs_drift.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"ROI TFR CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    print(df.head())

    # Basic correlations
    r_theta, p_theta = scipy.stats.pearsonr(df["theta_power"], df["drift_pain"])
    r_alpha, p_alpha = scipy.stats.pearsonr(df["alpha_power"], df["drift_pain"])
    print(f"Theta vs drift: r={r_theta:.3f}, p={p_theta:.3g}")
    print(f"Alpha vs drift: r={r_alpha:.3f}, p={p_alpha:.3g}")

    # ---------- Scatter plot: theta vs drift ----------
    fig, ax = plt.subplots(figsize=(4, 4))
    sns.regplot(
        x="drift_pain",
        y="theta_power",
        data=df,
        ax=ax
    )
    ax.set_xlabel("Pain drift (v_pain_subj)", fontsize=param["labelfontsize"])
    ax.set_ylabel("Frontal theta power (ROI)", fontsize=param["labelfontsize"])
    ax.tick_params(labelsize=param["ticksfontsize"])
    ax.set_title(f"Theta vs drift\nr={r_theta:.2f}, p={p_theta:.3g}",
                 fontsize=param["titlefontsize"])
    fig.tight_layout()
    fig.savefig(opj(outfigpath, "v8_theta_vs_drift_scatter.svg"),
                dpi=600, bbox_inches="tight")

    # ---------- Scatter plot: alpha vs drift ----------
    fig, ax = plt.subplots(figsize=(4, 4))
    sns.regplot(
        x="drift_pain",
        y="alpha_power",
        data=df,
        ax=ax
    )
    ax.set_xlabel("Pain drift (v_pain_subj)", fontsize=param["labelfontsize"])
    ax.set_ylabel("Parietal alpha power (ROI)", fontsize=param["labelfontsize"])
    ax.tick_params(labelsize=param["ticksfontsize"])
    ax.set_title(f"Alpha vs drift\nr={r_alpha:.2f}, p={p_alpha:.3g}",
                 fontsize=param["titlefontsize"])
    fig.tight_layout()
    fig.savefig(opj(outfigpath, "v8_alpha_vs_drift_scatter.svg"),
                dpi=600, bbox_inches="tight")



# # 9 
# if version == 9: 
#     from scipy.stats import ttest_1samp
#     from statsmodels.stats.multitest import fdrcorrection
#     from mne.channels import make_standard_montage

#     print("\n--- Version 9: TFR betas for sv_pain_para (ROI-based bands, RESPONSE-LOCKED) ---")

#     group_dir = Path(outpath)  # already tfr_mod_9_v9_sv_pain_para
#     betas_file = group_dir / "tfr_beta_sv_pain_para_subxchxfxt.npy"
#     freqs_file = group_dir / "tfr_beta_sv_pain_para_freqs.npy"
#     times_file = group_dir / "tfr_beta_sv_pain_para_times.npy"
#     ch_file    = group_dir / "tfr_beta_sv_pain_para_ch_names.npy"

#     if not (betas_file.exists() and freqs_file.exists()
#             and times_file.exists() and ch_file.exists()):
#         raise FileNotFoundError("One or more TFR beta files are missing in "
#                                 f"{group_dir}")

#     # shape: (n_subj, n_chan, n_freq, n_time)
#     all_betas = np.load(betas_file)
#     freqs     = np.load(freqs_file)                    # (n_freq,)
#     times     = np.load(times_file)                    # (n_time,)
#     ch_names  = np.load(ch_file, allow_pickle=True).tolist()

#     n_subj, n_chan, n_freq, n_time = all_betas.shape

#     print("all_betas shape:", all_betas.shape)
#     print("n_freq:", len(freqs), "n_time:", len(times), "n_chan:", len(ch_names))

#     # ------------------------------------------------------------------
#     # Build MNE Info (needed for topomaps & timecourses)
#     # ------------------------------------------------------------------
#     dt = float(times[1] - times[0])   # seconds
#     sfreq = 1.0 / dt                  # e.g. ~256 Hz

#     info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
#     montage = make_standard_montage('standard_1020')
#     info.set_montage(montage)

#     # exclude mastoids (same as ERP code)
#     chankeep = np.array([c not in ['M1', 'M2'] for c in ch_names])

#     # ----- RESPONSE-LOCKED plotting times and xticks -----
#     plot_times_resp = [-0.5, -0.3, -0.1, -0.05]     # in seconds
#     xticks_resp_ms  = np.arange(-800, 300, 200)   # for timecourses (in ms)

#     # ------------------------------------------------------------------
#     # Define *only* the hypotheses we care about
#     # NOTE: time_window is now response-locked: -0.5 to 0.1 s
#     # ------------------------------------------------------------------
#     HYPOTHESES = {
#         "beta_motor": {
#             "band_name": "beta",
#             "freq_range": (13., 30.),
#             "roi": ["C3", "CP3", "C4", "CP4", "Cz"],
#             "time_window": (-0.4, 0.0),
#         },
#         "theta_frontal": {
#             "band_name": "theta",
#             "freq_range": (4., 7.),
#             "roi": ["Fz", "FCz"],
#             "time_window": (-0.4, 0.0),
#         },
#     }
    

#     # bins relative to response for reporting (also negative)
#     BIN_DEF = [
#         ("early", -0.4, -0.25),
#         ("mid",   -0.25, -0.1),
#         ("late",  -0.1,  0.0),
#     ]


#     # ------------------------------------------------------------------
#     # Loop over hypotheses: average over freq, then ERP-style stats & plots
#     # ------------------------------------------------------------------
#     for hyp_key, cfg in HYPOTHESES.items():
#         band_name = cfg["band_name"]
#         f_lo, f_hi = cfg["freq_range"]
#         roi = cfg["roi"]
#         t_lo_roi, t_hi_roi = cfg["time_window"]

#         print(f"\n--- Hypothesis: {hyp_key} | band {band_name} {f_lo}-{f_hi} Hz, "
#               f"ROI={roi}, t={t_lo_roi}-{t_hi_roi} s ---")

#         # Frequency mask for this band
#         f_mask = (freqs >= f_lo) & (freqs <= f_hi)
#         if not np.any(f_mask):
#             print("  -> No frequencies in this range, skipping.")
#             continue

#         # 1) Collapse freq dimension within band
#         all_betas_band = all_betas[:, :, f_mask, :].mean(axis=2)

#         # 2) Grand-average betas across subjects: (n_chan, n_time)
#         beta_mean_band = all_betas_band.mean(axis=0)

#         # ROI indices
#         roi_idx = [ch_names.index(c) for c in roi if c in ch_names]
#         if len(roi_idx) == 0:
#             print("  -> ROI channels not found in ch_names, skipping tests.")
#             sig_mask_band = np.zeros((n_time, n_chan), dtype=bool)
#             pvals_fdr_band = np.full((n_time, n_chan), np.nan)
#             beta_ev_band = mne.EvokedArray(beta_mean_band, info, tmin=times[0])
#             continue

#         # ---- ROI-mean tests in main window (-0.5 to 0.1 s) ----
#         time_mask_roi = (times >= t_lo_roi) & (times <= t_hi_roi)
#         if np.any(time_mask_roi):
#             beta_roi = all_betas[:, roi_idx][:, :, f_mask][:, :, :, time_mask_roi].mean(axis=(1, 2, 3))
#             t_full, p_full = ttest_1samp(beta_roi, popmean=0.0)
#             print(f"  ROI-mean beta {band_name} {t_lo_roi*1000:.0f}-{t_hi_roi*1000:.0f} ms: "
#                   f"t({len(beta_roi)-1}) = {t_full:.3f}, p = {p_full:.3g}")
#         else:
#             print("  -> No time points in main ROI window for ROI-mean test.")

#         # ---- binned ROI tests (early/mid/late) ----
#         for bin_name, tb_lo, tb_hi in BIN_DEF:
#             tb_mask = (times >= tb_lo) & (times <= tb_hi)
#             if not np.any(tb_mask):
#                 continue
#             beta_bin = all_betas[:, roi_idx][:, :, f_mask][:, :, :, tb_mask].mean(axis=(1, 2, 3))
#             t_bin, p_bin = ttest_1samp(beta_bin, popmean=0.0)
#             print(f"    Bin {bin_name} {tb_lo*1000:.0f}-{tb_hi*1000:.0f} ms: "
#                   f"t({len(beta_bin)-1}) = {t_bin:.3f}, p = {p_bin:.3g}")

#         # 3) Second-level t-test vs 0 at each (chan, time) for plotting
#         tvals_band = np.zeros((n_time, n_chan))
#         pvals_band = np.zeros((n_time, n_chan))
#         for ti in range(n_time):
#             b_t = all_betas_band[:, :, ti]
#             t_t, p_t = ttest_1samp(
#                 b_t,
#                 popmean=0.0,
#                 axis=0,
#                 nan_policy="omit"
#             )
#             tvals_band[ti, :] = t_t
#             pvals_band[ti, :] = p_t

#         # 4) FDR correction ONLY in ROI × main time window
#         sig_mask_band = np.zeros_like(pvals_band, dtype=bool)
#         pvals_fdr_band = np.full_like(pvals_band, np.nan)
#         if np.any(time_mask_roi):
#             p_roi = pvals_band[time_mask_roi][:, roi_idx]
#             p_flat = p_roi.reshape(-1)
#             rej_flat, p_fdr_flat = fdrcorrection(p_flat, alpha=0.05)
#             sig_roi = rej_flat.reshape(p_roi.shape)
#             p_fdr_roi = p_fdr_flat.reshape(p_roi.shape)
#             sig_mask_band[np.ix_(time_mask_roi, roi_idx)] = sig_roi
#             pvals_fdr_band[np.ix_(time_mask_roi, roi_idx)] = p_fdr_roi
#             print(f"  -> Significant samples in ROI window (FDR, p<0.05): "
#                   f"{sig_mask_band.sum()}")

#         # 5) Wrap mean beta into Evoked for plotting
#         beta_ev_band = mne.EvokedArray(beta_mean_band, info, tmin=times[0])

#         # ------------------------------------------------------------------
#         # 5A. Topomaps at selected times (response-locked)
#         # ------------------------------------------------------------------
#         times_pos = [np.abs(beta_ev_band.times - t).argmin() for t in plot_times_resp]

#         for tidx, tpos in enumerate(times_pos):
#             fig, topo_axis = plt.subplots(figsize=(1.5, 1.5))

#             p_row = pvals_fdr_band[tpos, :]
#             valid = np.isfinite(p_row)
#             sig_non_mastoid = valid & (p_row < 0.05) & chankeep
#             mask = sig_non_mastoid

#             vmax = np.max(np.abs(beta_ev_band.data))
#             im, _ = plot_topomap(
#                 beta_ev_band.data[:, tpos],
#                 pos=beta_ev_band.info,
#                 mask=mask,
#                 mask_params=dict(marker='o',
#                                  markerfacecolor='w',
#                                  markeredgecolor='k',
#                                  linewidth=0,
#                                  markersize=3),
#                 cmap='RdBu_r',
#                 show=False,
#                 ch_type='eeg',
#                 outlines='head',
#                 extrapolate='head',
#                 vlim=(-vmax, vmax),
#                 axes=topo_axis,
#                 sensors=False,
#                 contours=0,
#             )
#             topo_axis.set_title(
#                 f"{band_name} {int(plot_times_resp[tidx]*1000)} ms",
#                 fontdict={'size': param['labelfontsize']-1},
#                 pad=0.1
#             )

#             fig.savefig(
#                 opj(outfigpath,
#                     f'{fig_prefix}v9_{hyp_key}_{band_name}_topo_beta_t{int(plot_times_resp[tidx]*1000)}.svg'),
#                 dpi=600,
#                 bbox_inches='tight'
#             )

#             if tidx + 1 == len(times_pos):
#                 fig2, ax = plt.subplots(figsize=(0.3, 1.2))
#                 cbar = fig2.colorbar(im, cax=ax, orientation='vertical', aspect=1)
#                 cbar.set_label(
#                     f'Beta (power ~ sv_pain_para)\n{band_name}',
#                     rotation=270, labelpad=12,
#                     fontdict={'fontsize': param['labelfontsize']-1}
#                 )
#                 cbar.ax.tick_params(labelsize=param['ticksfontsize']-2)
#                 fig2.savefig(
#                     opj(outfigpath,
#                         f'{fig_prefix}v9_{hyp_key}_{band_name}_topo_beta_cbar.svg'),
#                     dpi=600,
#                     bbox_inches='tight'
#                 )

#         # ------------------------------------------------------------------
#         # 5B. Timecourses at ROI channels with sig bar (response-locked)
#         # ------------------------------------------------------------------
#         for c in roi:
#             if c not in beta_ev_band.ch_names:
#                 continue

#             pick = beta_ev_band.ch_names.index(c)
#             fig, ax = plt.subplots(1, 1, figsize=(4, 2.5))

#             y = beta_ev_band.data[pick, :]
#             ax.plot(times * 1000, y, linewidth=2)

#             ax.set_xlabel('Time (ms)',
#                           fontdict={'size': param['labelfontsize']})
#             ax.set_ylabel(f'Beta ({band_name}, power ~ sv_pain_para) – {c}',
#                           fontdict={'size': param['labelfontsize']})
#             ax.axhline(0, linestyle='--', color='gray')
#             ax.axvline(0, linestyle='--', color='gray')

#             timestep = 1000.0 / param['testresampfreq']
#             for tidx2, t_ms in enumerate(times * 1000):
#                 if sig_mask_band[tidx2, pick]:
#                     ax.fill_between(
#                         [t_ms, t_ms + timestep],
#                         y.min() - 0.02,
#                         y.min() - 0.005,
#                         alpha=0.4,
#                         facecolor='red'
#                     )

#             ax.set_xticks(xticks_resp_ms)
#             ax.set_xticklabels([str(i) for i in xticks_resp_ms])
#             ax.tick_params(labelsize=param['ticksfontsize'])
#             fig.tight_layout()
#             fig.savefig(
#                 opj(outfigpath,
#                     f'{fig_prefix}v9_{hyp_key}_{band_name}_timecourse_{c}.svg'),
#                 dpi=600,
#                 bbox_inches='tight'
#             )

#     print("\nVersion 9 ROI-based TFR plotting done (response-locked).\n")


if version == 9:
    from scipy.stats import ttest_1samp
    from statsmodels.stats.multitest import fdrcorrection
    from mne.channels import make_standard_montage

    print("\n--- Version 9: TFR betas for sv_pain_para (ROI-based bands) ---")

    group_dir = Path(outpath)  # already tfr_mod_9_v9_sv_pain_para
    betas_file = group_dir / "tfr_beta_sv_pain_para_subxchxfxt.npy"
    freqs_file = group_dir / "tfr_beta_sv_pain_para_freqs.npy"
    times_file = group_dir / "tfr_beta_sv_pain_para_times.npy"
    ch_file    = group_dir / "tfr_beta_sv_pain_para_ch_names.npy"

    if not (betas_file.exists() and freqs_file.exists()
            and times_file.exists() and ch_file.exists()):
        raise FileNotFoundError("One or more TFR beta files are missing in "
                                f"{group_dir}")

    # shape: (n_subj, n_chan, n_freq, n_time)
    all_betas = np.load(betas_file)
    freqs     = np.load(freqs_file)                    # (n_freq,)
    times     = np.load(times_file)                    # (n_time,)
    ch_names  = np.load(ch_file, allow_pickle=True).tolist()

    n_subj, n_chan, n_freq, n_time = all_betas.shape

    print("all_betas shape:", all_betas.shape)
    print("n_freq:", len(freqs), "n_time:", len(times), "n_chan:", len(ch_names))

    # ------------------------------------------------------------------
    # Build MNE Info (needed for topomaps & timecourses)
    # ------------------------------------------------------------------
    dt = float(times[1] - times[0])   # seconds
    sfreq = 1.0 / dt                  # e.g. ~256 Hz

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    montage = make_standard_montage('standard_1020')
    info.set_montage(montage)

    # exclude mastoids (same as ERP code)
    chankeep = np.array([c not in ['M1', 'M2'] for c in ch_names])

    # ------------------------------------------------------------------
    # Define *only* the hypotheses we care about
    # ------------------------------------------------------------------
    HYPOTHESES = {
        "theta_CP": {
            "band_name": "theta",
            "freq_range": (4., 7.),
            "roi": ["Cz", "CPz", "Pz", "POz", "P1", "P2"],
            "time_window": (0.3, 0.8),
        },
        "alpha_frontal": {
            "band_name": "alpha",
            "freq_range": (8., 12.),
            "roi": ["Fz", "FCz", "F1", "F2", "F4"],
            "time_window": (0.3, 0.8),
        },
    }

    # ------------------------------------------------------------------
    # Loop over hypotheses: average over freq, then ERP-style stats & plots
    # ------------------------------------------------------------------
    for hyp_key, cfg in HYPOTHESES.items():
        band_name = cfg["band_name"]
        f_lo, f_hi = cfg["freq_range"]
        roi = cfg["roi"]
        t_lo_roi, t_hi_roi = cfg["time_window"]

        print(f"\n--- Hypothesis: {hyp_key} | band {band_name} {f_lo}-{f_hi} Hz, "
              f"ROI={roi}, t={t_lo_roi}-{t_hi_roi} s ---")

        # Frequency mask for this band
        f_mask = (freqs >= f_lo) & (freqs <= f_hi)
        if not np.any(f_mask):
            print("  -> No frequencies in this range, skipping.")
            continue

        # 1) Collapse freq dimension within band
        #    all_betas_band: (n_subj, n_chan, n_time)
        all_betas_band = all_betas[:, :, f_mask, :].mean(axis=2)

        # 2) Grand-average betas across subjects: (n_chan, n_time)
        beta_mean_band = all_betas_band.mean(axis=0)

        # ROI indices
        roi_idx = [ch_names.index(c) for c in roi if c in ch_names]
        if len(roi_idx) == 0:
            print("  -> ROI channels not found in ch_names, skipping tests.")
            sig_mask_band = np.zeros((n_time, n_chan), dtype=bool)
            pvals_fdr_band = np.full((n_time, n_chan), np.nan)
            beta_ev_band = mne.EvokedArray(beta_mean_band, info, tmin=times[0])
            continue

        time_mask_roi = (times >= t_lo_roi) & (times <= t_hi_roi)
        if np.any(time_mask_roi):
            # shape: (subj, ROI, time_in_window) -> mean -> (subj,)
            beta_roi = all_betas[:, roi_idx][:, :, f_mask][:, :, :, time_mask_roi].mean(axis=(1, 2, 3))
            t_full, p_full = ttest_1samp(beta_roi, popmean=0.0)
            print(f"  ROI-mean beta {band_name} {t_lo_roi*1000:.0f}-{t_hi_roi*1000:.0f} ms: "
                  f"t({len(beta_roi)-1}) = {t_full:.3f}, p = {p_full:.3g}")
        else:
            print("  -> No time points in main ROI window for ROI-mean test.")

        BIN_DEF = [
            ("early", 0.3, 0.5),
            ("mid",   0.5, 0.7),
            ("late",  0.7, 0.9),
        ]
        for bin_name, tb_lo, tb_hi in BIN_DEF:
            tb_mask = (times >= tb_lo) & (times <= tb_hi)
            if not np.any(tb_mask):
                continue
            beta_bin = all_betas[:, roi_idx][:, :, f_mask][:, :, :, tb_mask].mean(axis=(1, 2, 3))
            t_bin, p_bin = ttest_1samp(beta_bin, popmean=0.0)
            print(f"    Bin {bin_name} {tb_lo*1000:.0f}-{tb_hi*1000:.0f} ms: "
                  f"t({len(beta_bin)-1}) = {t_bin:.3f}, p = {p_bin:.3g}")

        # 3) Second-level t-test vs 0 at each (chan, time) for plotting
        tvals_band = np.zeros((n_time, n_chan))
        pvals_band = np.zeros((n_time, n_chan))
        for ti in range(n_time):
            b_t = all_betas_band[:, :, ti]   # (subj, chan)
            t_t, p_t = ttest_1samp(
                b_t,
                popmean=0.0,
                axis=0,
                nan_policy="omit"
            )
            tvals_band[ti, :] = t_t
            pvals_band[ti, :] = p_t

        # 4) FDR correction ONLY in ROI × main time window (for sig bars/dots)
        sig_mask_band = np.zeros_like(pvals_band, dtype=bool)
        pvals_fdr_band = np.full_like(pvals_band, np.nan)
        if np.any(time_mask_roi):
            p_roi = pvals_band[time_mask_roi][:, roi_idx]
            p_flat = p_roi.reshape(-1)
            rej_flat, p_fdr_flat = fdrcorrection(p_flat, alpha=0.05)
            sig_roi = rej_flat.reshape(p_roi.shape)
            p_fdr_roi = p_fdr_flat.reshape(p_roi.shape)
            sig_mask_band[np.ix_(time_mask_roi, roi_idx)] = sig_roi
            pvals_fdr_band[np.ix_(time_mask_roi, roi_idx)] = p_fdr_roi
            print(f"  -> Significant samples in ROI window (FDR, p<0.05): "
                  f"{sig_mask_band.sum()}")

        # 5) Wrap mean beta into Evoked for plotting (like beta_gavg in ERP)
        beta_ev_band = mne.EvokedArray(beta_mean_band, info, tmin=times[0])

        # ------------------------------------------------------------------
        # 5A. Topomaps at selected times (like ERP v7)
        # ------------------------------------------------------------------
        times_pos = [np.abs(beta_ev_band.times - t).argmin() for t in plot_times]

        for tidx, tpos in enumerate(times_pos):
            fig, topo_axis = plt.subplots(figsize=(1.5, 1.5))

            p_row = pvals_fdr_band[tpos, :]     # (n_chan,)
            valid = np.isfinite(p_row)
            sig_non_mastoid = valid & (p_row < 0.05) & chankeep
            mask = sig_non_mastoid

            vmax = np.max(np.abs(beta_ev_band.data))
            im, _ = plot_topomap(
                beta_ev_band.data[:, tpos],
                pos=beta_ev_band.info,
                mask=mask,
                mask_params=dict(marker='o',
                                 markerfacecolor='w',
                                 markeredgecolor='k',
                                 linewidth=0,
                                 markersize=3),
                cmap='RdBu_r',
                show=False,
                ch_type='eeg',
                outlines='head',
                extrapolate='head',
                vlim=(-vmax, vmax),
                axes=topo_axis,
                sensors=False,
                contours=0,
            )
            topo_axis.set_title(
                f"{band_name} {int(plot_times[tidx]*1000)} ms",
                fontdict={'size': param['labelfontsize']-1},
                pad=0.1
            )

            fig.savefig(
                opj(outfigpath,
                    f'{fig_prefix}v9_{hyp_key}_{band_name}_topo_beta_t{int(plot_times[tidx]*1000)}.svg'),
                dpi=600,
                bbox_inches='tight'
            )

            if tidx + 1 == len(times_pos):
                fig2, ax = plt.subplots(figsize=(0.3, 1.2))
                cbar = fig2.colorbar(im, cax=ax, orientation='vertical', aspect=1)
                cbar.set_label(
                    f'Beta (power ~ sv_pain_para)\n{band_name}',
                    rotation=270, labelpad=12,
                    fontdict={'fontsize': param['labelfontsize']-1}
                )
                cbar.ax.tick_params(labelsize=param['ticksfontsize']-2)
                fig2.savefig(
                    opj(outfigpath,
                        f'{fig_prefix}v9_{hyp_key}_{band_name}_topo_beta_cbar.svg'),
                    dpi=600,
                    bbox_inches='tight'
                )

        # ------------------------------------------------------------------
        # 5B. Timecourses at ROI channels with sig bar (like ERP v7)
        # ------------------------------------------------------------------
        for c in roi:
            if c not in beta_ev_band.ch_names:
                continue

            pick = beta_ev_band.ch_names.index(c)
            fig, ax = plt.subplots(1, 1, figsize=(4, 2.5))

            y = beta_ev_band.data[pick, :]      # beta over time
            ax.plot(times * 1000, y, linewidth=2)

            ax.set_xlabel('Time (ms)',
                          fontdict={'size': param['labelfontsize']})
            ax.set_ylabel(f'Beta ({band_name}, power ~ sv_pain_para) – {c}',
                          fontdict={'size': param['labelfontsize']})
            ax.axhline(0, linestyle='--', color='gray')
            ax.axvline(0, linestyle='--', color='gray')

            timestep = 1000.0 / param['testresampfreq']   # ms
            for tidx2, t_ms in enumerate(times * 1000):
                if sig_mask_band[tidx2, pick]:
                    ax.fill_between(
                        [t_ms, t_ms + timestep],
                        y.min() - 0.02,
                        y.min() - 0.005,
                        alpha=0.4,
                        facecolor='red'
                    )

            ax.set_xticks(np.arange(-200, 1200, 200))
            ax.set_xticklabels([str(i) for i in np.arange(-200, 1200, 200)])
            ax.tick_params(labelsize=param['ticksfontsize'])
            fig.tight_layout()
            fig.savefig(
                opj(outfigpath,
                    f'{fig_prefix}v9_{hyp_key}_{band_name}_timecourse_{c}.svg'),
                dpi=600,
                bbox_inches='tight'
            )

    print("\nVersion 9 ROI-based TFR plotting done.\n")
    
#-------------------------------------------------------------------------------------------------------------------------------------------    

elif version == 10:
    """
    Plotting for Version 10:
    - loads v10_tval_*.npy and v10_pval_*.npy (cluster p-values painted in)
    - makes:
        1) time×freq heatmap (ROI-averaged t-values; sig overlay)
        2) topomaps for selected (freq band, time window) with sig mask
        3) ROI band-limited timecourse with sig bar
    """

    from pathlib import Path
    from mne.channels import make_standard_montage

    print("\n--- Plotting Version 10: full TFR mass-univariate cluster results ---")

    group_dir = Path(outpath)  # .../tfr_mod_9_v10_sv_vs_pain_RT
    stats_dir = group_dir / "Zscoring"
    if not stats_dir.exists():
        raise FileNotFoundError(f"Stats dir not found: {stats_dir}")

    # ----------------------------
    # Load grids saved by v10
    # ----------------------------
    freqs = np.load(group_dir / "tfr_beta_freqs.npy")              # (n_freq,)
    times = np.load(group_dir / "tfr_beta_times.npy")              # (n_time,)
    ch_names = np.load(group_dir / "tfr_beta_ch_names.npy", allow_pickle=True).tolist()

    n_chan = len(ch_names)
    n_freq = len(freqs)
    n_time = len(times)

    # Build Info for topomaps
    dt = float(times[1] - times[0])
    sfreq = 1.0 / dt
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    info.set_montage(make_standard_montage("standard_1020"))

    # exclude mastoids like elsewhere
    chankeep = np.array([c not in ["M1", "M2"] for c in ch_names], dtype=bool)

    # ----------------------------
    # What to plot (edit here)
    # ----------------------------
    MAPS = [
        ("sv_cov_pain_rt",  "SV | pain + RT"),
        ("pain_cov_sv_rt",  "Pain | SV + RT"),
        ("sv_minus_pain",   "SV − Pain"),
    ]

    # Choose ROIs and bands (edit as you like)
    ROI_CP = ["Cz", "CPz", "Pz", "POz", "P1", "P2"]
    ROI_F  = ["Fz", "FCz", "F1", "F2", "F4"]

    BANDS = {
        "theta": (4., 7.),
        "alpha": (8., 12.),
        "beta":  (13., 30.),
    }

    # plotting times for topomaps (seconds)
    plot_times = [0.2, 0.4, 0.6, 0.8, 1.0]

    # utility
    def _idx_from_list(chlist):
        return [ch_names.index(c) for c in chlist if c in ch_names]

    def _nearest_time_idx(t):
        return int(np.argmin(np.abs(times - t)))

    def _mask_band(band):
        f_lo, f_hi = BANDS[band]
        return (freqs >= f_lo) & (freqs <= f_hi)

    def _savefig(fig, fname):
        fig.tight_layout()
        fig.savefig(opj(outfigpath, fname), dpi=600, bbox_inches="tight")
        plt.close(fig)


    for map_key, map_title in MAPS:
        tfile = stats_dir / f"v10_tval_{map_key}.npy"
        pfile = stats_dir / f"v10_pval_{map_key}.npy"
        if not (tfile.exists() and pfile.exists()):
            print(f"  Missing files for {map_key}, skipping.")
            continue

        t_map = np.load(tfile)   # (chan, freq, time)
        p_map = np.load(pfile)   # (chan, freq, time) painted cluster p-values

        assert t_map.shape == (n_chan, n_freq, n_time), f"t_map shape mismatch: {t_map.shape}"
        assert p_map.shape == (n_chan, n_freq, n_time), f"p_map shape mismatch: {p_map.shape}"


        for roi_name, roi_list in [("CP", ROI_CP), ("F", ROI_F)]:
            roi_idx = _idx_from_list(roi_list)
            if len(roi_idx) == 0:
                continue

            t_roi = t_map[roi_idx].mean(axis=0)          # (freq, time)
            p_roi = p_map[roi_idx].min(axis=0)           # (freq, time) conservative "any channel sig"

            fig, ax = plt.subplots(figsize=(6.0, 3.2))
            im = ax.imshow(
                t_roi,
                aspect="auto",
                origin="lower",
                extent=[times[0]*1000, times[-1]*1000, freqs[0], freqs[-1]],
            )
            ax.set_title(f"{map_title} – ROI {roi_name} (mean t)")
            ax.set_xlabel("Time (ms)")
            ax.set_ylabel("Frequency (Hz)")
            fig.colorbar(im, ax=ax, shrink=0.9, label="t-value")

            # overlay significance contour (cluster p<0.05)
            sig = (p_roi < 0.05)
            if np.any(sig):
                # contour needs x/y grid
                xx = np.linspace(times[0]*1000, times[-1]*1000, n_time)
                yy = np.linspace(freqs[0], freqs[-1], n_freq)
                ax.contour(xx, yy, sig.astype(int), levels=[0.5], linewidths=1)

            _savefig(fig, f"{fig_prefix}v10_{map_key}_heatmap_t_ROI{roi_name}.svg")


        # ----------------------------------------
        for band_name in ["theta", "alpha", "beta"]:
            fmask = _mask_band(band_name)
            if not np.any(fmask):
                continue

            t_band = t_map[:, fmask, :].mean(axis=1)  # (chan, time)

            sig_band = (p_map[:, fmask, :] < 0.05).any(axis=1)  # (chan, time)

            times_pos = [_nearest_time_idx(t) for t in plot_times]

            # vmax symmetric
            vmax = np.nanmax(np.abs(t_band[chankeep, :]))
            if not np.isfinite(vmax) or vmax == 0:
                vmax = 1.0

            for tidx, tpos in enumerate(times_pos):
                fig, ax = plt.subplots(figsize=(1.8, 1.8))

                mask = np.zeros(n_chan, dtype=bool)
                mask[chankeep] = sig_band[chankeep, tpos]

                im, _ = plot_topomap(
                    t_band[:, tpos],
                    pos=info,
                    mask=mask,
                    mask_params=dict(marker='o',
                                     markerfacecolor='w',
                                     markeredgecolor='k',
                                     linewidth=0,
                                     markersize=2),
                    cmap="RdBu_r",
                    show=False,
                    outlines="head",
                    extrapolate="head",
                    vlim=(-vmax, vmax),
                    axes=ax,
                    sensors=False,
                    contours=0,
                )
                ax.set_title(f"{map_title}\n{band_name} @ {int(plot_times[tidx]*1000)} ms",
                             fontsize=param["labelfontsize"]-1)

                _savefig(fig, f"{fig_prefix}v10_{map_key}_topo_t_{band_name}_{int(plot_times[tidx]*1000)}ms.svg")

                if tidx + 1 == len(times_pos):
                    fig2, cax = plt.subplots(figsize=(0.25, 1.6))
                    cb = fig2.colorbar(im, cax=cax, orientation="vertical")
                    cb.set_label("t-value", rotation=270, labelpad=12,
                                 fontdict={"fontsize": param["labelfontsize"]-1})
                    cb.ax.tick_params(labelsize=param["ticksfontsize"]-2)
                    _savefig(fig2, f"{fig_prefix}v10_{map_key}_topo_t_{band_name}_cbar.svg")

        # ----------------------------------------
        for roi_name, roi_list in [("CP", ROI_CP), ("F", ROI_F)]:
            roi_idx = _idx_from_list(roi_list)
            if len(roi_idx) == 0:
                continue

            for band_name in ["theta", "alpha", "beta"]:
                fmask = _mask_band(band_name)
                if not np.any(fmask):
                    continue

                # ROI mean t(time) after averaging over chan and freq
                t_tc = t_map[roi_idx][:, fmask, :].mean(axis=(0, 1))  # (time,)

                # significance at time if ANY point in ROI×band is in a significant cluster
                sig_tc = (p_map[roi_idx][:, fmask, :] < 0.05).any(axis=(0, 1))  # (time,)

                fig, ax = plt.subplots(figsize=(4.8, 2.8))
                ax.plot(times * 1000, t_tc, lw=2)
                ax.axhline(0, linestyle="--", color="gray")
                ax.axvline(0, linestyle="--", color="gray")
                ax.set_xlabel("Time (ms)")
                ax.set_ylabel("Mean t-value")
                ax.set_title(f"{map_title} – ROI {roi_name} – {band_name}")

                # sig bar
                if len(times) > 1:
                    dt_ms = (times[1] - times[0]) * 1000
                else:
                    dt_ms = 1.0

                y0 = np.nanmin(t_tc) - 0.2
                y1 = y0 + 0.15
                for i, tt in enumerate(times * 1000):
                    if sig_tc[i]:
                        ax.fill_between([tt, tt + dt_ms], y0, y1, alpha=0.3)

                ax.tick_params(labelsize=param["ticksfontsize"])
                _savefig(fig, f"{fig_prefix}v10_{map_key}_timecourse_t_ROI{roi_name}_{band_name}.svg")

    print("\nVersion 10 plotting done.\n")
    
    
    
if version == 11:
    from scipy.stats import ttest_1samp
    from statsmodels.stats.multitest import fdrcorrection

    print("\n--- Version 9: TFR betas for sv_pain_para (band-limited, ERP-style) ---")

    group_dir = Path(outpath)  # already tfr_mod_9_v9_sv_pain_para
    betas_file = group_dir / "tfr_beta_sv_pain_para_subxchxfxt.npy"
    freqs_file = group_dir / "tfr_beta_sv_pain_para_freqs.npy"
    times_file = group_dir / "tfr_beta_sv_pain_para_times.npy"
    ch_file    = group_dir / "tfr_beta_sv_pain_para_ch_names.npy"

    if not (betas_file.exists() and freqs_file.exists()
            and times_file.exists() and ch_file.exists()):
        raise FileNotFoundError("One or more TFR beta files are missing in "
                                f"{group_dir}")

    # shape: (n_subj, n_chan, n_freq, n_time)
    all_betas = np.load(betas_file)
    freqs     = np.load(freqs_file)                    # (n_freq,)
    times     = np.load(times_file)                    # (n_time,)
    ch_names  = np.load(ch_file, allow_pickle=True).tolist()

    n_subj, n_chan, n_freq, n_time = all_betas.shape

    print("all_betas shape:", all_betas.shape)
    print("n_freq:", len(freqs), "n_time:", len(times), "n_chan:", len(ch_names))

    # ------------------------------------------------------------------
    # Build MNE Info (needed for topomaps & timecourses)
    # ------------------------------------------------------------------
    dt = float(times[1] - times[0])   # seconds
    sfreq = 1.0 / dt                  # e.g. ~256 Hz
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    montage = make_standard_montage('easycap-M1')  
    info.set_montage(montage)

    # exclude mastoids (same as ERP code)
    chankeep = np.array([c not in ['M1', 'M2'] for c in ch_names])

    # ------------------------------------------------------------------
    # Define frequency bands (you can tweak these)
    # ------------------------------------------------------------------
    BANDS = {
        "delta": (0.5, 4.),
        "theta": (4., 7.),
        "alpha": (8., 12.),
        "beta":  (13., 30.),
        "low_gamma": (31., 45.)
    }

    # ------------------------------------------------------------------
    # Loop over bands: average over freq, then do ERP-style stats & plots
    # ------------------------------------------------------------------
    for band_name, (f_lo, f_hi) in BANDS.items():
        print(f"\n--- Band: {band_name} ({f_lo}-{f_hi} Hz) ---")

        # Frequency mask for this band
        f_mask = (freqs >= f_lo) & (freqs <= f_hi)
        if not np.any(f_mask):
            print(f"  -> No frequencies in this range, skipping.")
            continue

        # 1) Collapse freq dimension within band
        #    all_betas_band: (n_subj, n_chan, n_time)
        all_betas_band = all_betas[:, :, f_mask, :].mean(axis=2)

        # 2) Grand-average betas across subjects: (n_chan, n_time)
        beta_mean_band = all_betas_band.mean(axis=0)

        # 3) Second-level t-test vs 0 at each (chan, time)
        #    We'll store as (n_time, n_chan) to match your ERP pvals shape.
        tvals_band = np.zeros((n_time, n_chan))
        pvals_band = np.zeros((n_time, n_chan))

        for ti in range(n_time):
            # betas at this time across subjects: shape (n_subj, n_chan)
            b_t = all_betas_band[:, :, ti]
            t_t, p_t = ttest_1samp(
                b_t,
                popmean=0.0,
                axis=0,
                nan_policy="omit"
            )
            tvals_band[ti, :] = t_t
            pvals_band[ti, :] = p_t

        # 4) FDR correction over chan x time (within this band)
        p_flat = pvals_band.reshape(-1)
        rej_flat, p_fdr_flat = fdrcorrection(p_flat, alpha=0.05)
        sig_mask_band = rej_flat.reshape(pvals_band.shape)      # (time, chan)
        pvals_fdr_band = p_fdr_flat.reshape(pvals_band.shape)   # same shape

        print(f"  -> Significant samples (FDR, p<0.05): {sig_mask_band.sum()}")

        # 5) Wrap mean beta into Evoked for plotting (like beta_gavg in ERP)
        beta_ev_band = mne.EvokedArray(beta_mean_band, info, tmin=times[0])

        # ------------------------------------------------------------------
        # 5A. Topomaps at selected times (like ERP v7)
        # ------------------------------------------------------------------
        # here we use the same plot_times defined earlier in your script
        times_pos = [np.abs(beta_ev_band.times - t).argmin() for t in plot_times]

        for tidx, tpos in enumerate(times_pos):
            fig, topo_axis = plt.subplots(figsize=(1.5, 1.5))

            # FDR-corrected p-values at this time point
            p_row = pvals_fdr_band[tpos, :]     # (n_chan,)

            # Sig mask for non-mastoid channels
            sig_non_mastoid = (p_row < 0.05) & chankeep
            mask = sig_non_mastoid

            vmax = np.max(np.abs(beta_ev_band.data))
            im, _ = plot_topomap(
                beta_ev_band.data[:, tpos],
                pos=beta_ev_band.info,
                mask=mask,
                mask_params=dict(marker='o',
                                 markerfacecolor='w',
                                 markeredgecolor='k',
                                 linewidth=0,
                                 markersize=3),
                cmap='RdBu_r',
                show=False,
                ch_type='eeg',
                outlines='head',
                extrapolate='head',
                vlim=(-vmax, vmax),
                axes=topo_axis,
                sensors=False,
                contours=0,
            )
            topo_axis.set_title(
                f"{band_name} {int(plot_times[tidx]*1000)} ms",
                fontdict={'size': param['labelfontsize']-1},
                pad=0.1
            )

            fig.savefig(
                opj(outfigpath,
                    f'{fig_prefix}v9_{band_name}_topo_beta_t{int(plot_times[tidx]*1000)}.svg'),
                dpi=600,
                bbox_inches='tight'
            )

            # save colourbar on last time point
            if tidx + 1 == len(times_pos):
                fig2, ax = plt.subplots(figsize=(0.3, 1.2))
                cbar = fig2.colorbar(im, cax=ax, orientation='vertical', aspect=1)
                cbar.set_label(
                    f'Beta (power ~ sv_pain_para)\n{band_name}',
                    rotation=270, labelpad=12,
                    fontdict={'fontsize': param['labelfontsize']-1}
                )
                cbar.ax.tick_params(labelsize=param['ticksfontsize']-2)
                fig2.savefig(
                    opj(outfigpath,
                        f'{fig_prefix}v9_{band_name}_topo_beta_cbar.svg'),
                    dpi=600,
                    bbox_inches='tight'
                )

        # ------------------------------------------------------------------
        # 5B. Timecourses at ROI channels with sig bar (like ERP v7)
        # ------------------------------------------------------------------
        for c in chan_to_plot:
            if c not in beta_ev_band.ch_names:
                continue

            pick = beta_ev_band.ch_names.index(c)
            fig, ax = plt.subplots(1, 1, figsize=(4, 2.5))

            y = beta_ev_band.data[pick, :]      # beta over time
            ax.plot(times * 1000, y, linewidth=2)

            ax.set_xlabel('Time (ms)',
                          fontdict={'size': param['labelfontsize']})
            ax.set_ylabel(f'Beta ({band_name}, power ~ sv_pain_para)',
                          fontdict={'size': param['labelfontsize']})
            ax.axhline(0, linestyle='--', color='gray')
            ax.axvline(0, linestyle='--', color='gray')

            # significance shading at the bottom (FDR-corrected)
            timestep = 1000.0 / param['testresampfreq']   # ms
            for tidx2, t_ms in enumerate(times * 1000):
                if sig_mask_band[tidx2, pick]:
                    ax.fill_between(
                        [t_ms, t_ms + timestep],
                        y.min() - 0.02,
                        y.min() - 0.005,
                        alpha=0.4,
                        facecolor='red'
                    )

            ax.set_xticks(np.arange(-200, 1200, 200))
            ax.set_xticklabels([str(i) for i in np.arange(-200, 1200, 200)])
            ax.tick_params(labelsize=param['ticksfontsize'])
            fig.tight_layout()
            fig.savefig(
                opj(outfigpath,
                    f'{fig_prefix}v9_{band_name}_timecourse_{c}.svg'),
                dpi=600,
                bbox_inches='tight'
            )

    print("\nVersion 9 band-wise TFR plotting done (ERP-style).\n")


# old ------------------------------------------------------------------------------------------------------------------------
##############################################################################################################################

# param = {
#     'alpha': 0.05/3,     
#     'titlefontsize': 12,
#     'labelfontsize': 12,
#     'ticksfontsize': 11,
#     'legendfontsize': 10,
#     'testresampfreq': 1024,
# }

# plt.rc("axes.spines", top=False, right=False)
# plt.rcParams['font.family'] = 'Liberation Sans'


# ##-----------------------------------------------------------------------------------------------------
# ## Multivariate Regression Plots from MP for only 1 regressor
# #
# #tvals = np.load(opj(outpath, 'ols_2ndlevel_tvals.npy'))
# #pvals = np.load(opj(outpath, 'ols_2ndlevel_pvals.npy'))
# #
# #beta_gavg = np.load(opj(outpath, 'ols_2ndlevel_betasavg.npy'),
# #                    allow_pickle=True)
# #allbetas = np.load(opj(outpath, 'ols_2ndlevel_betas.npy'),
# #                   allow_pickle=True)
# #
# ## regression variable v_sv_pain_para_contrib
# #regvar = ['painlevel','moneylevel', 'v_pain_contrib','v_money_contrib','v_interaction_contrib']
# #regvarname = ['painlevel','moneylevel', 'v_pain_contrib','v_money_contrib','v_interaction_contrib']
# #
# ## colormap and vminmax for this single regressor
# #vminmax = 10
# #cmap = 'viridis'
# #
# ## Plot descriptive topo data
# #plot_times = [0.2, 0.4, 0.6, 0.8]
# #times_pos = [np.abs(beta_gavg[0].times - 0.2 - t).argmin() for t in plot_times]
# #
# #chan_to_plot = ['Fz', 'FCz', 'POz', 'Cz', 'CPz', 'Pz', 'Oz' ]      
# #
# #all_epos = mne.read_epochs(
# #    opj(outpath, 'ols_2ndlevel_allepochs-epo_' + regvar + '.fif'), preload=False)
# #
# #beta_gavg_nomast = beta_gavg[0].copy()
# #chankeep = [True if c not in ['M1', 'M2'] else False for c in
# #            beta_gavg_nomast.ch_names]
# #
# #for tidx, timepos in enumerate(times_pos):
# #    fig, topo_axis = plt.subplots(figsize=(1, 1))
# #
# #    im, _ = plot_topomap(beta_gavg_nomast.data[:, timepos],
# #                         pos=beta_gavg_nomast.info,
# #                         mask=pvals[0][timepos,
# #                                       chankeep] < param['alpha'],
# #                         mask_params=dict(marker='o',
# #                                          markerfacecolor='w',
# #                                          markeredgecolor='k',
# #                                          linewidth=0,
# #                                          markersize=2),
# #                         cmap=cmap,
# #                         show=False,
# #                         ch_type='eeg',
# #                         outlines='head',
# #                         extrapolate='head',
# #                         vlim=(None, None),
# #                         axes=topo_axis,
# #                         sensors=False,
# #                         contours=0,)
# #    topo_axis.set_title(str(int(plot_times[tidx] * 1000)) + ' ms', fontdict={'size': param['labelfontsize']-1}, pad=0.1)
# #
# #    if tidx+1 == len(plot_times):
# #        fig, ax = plt.subplots(figsize=(0.2, 1))
# #        cbar1 = fig.colorbar(im, cax=ax,
# #                             orientation='vertical', aspect=1)
# #        cbar1.set_label('Beta', rotation=270, labelpad=12, fontdict={'fontsize': param["labelfontsize"]-1})
# #        cbar1.ax.tick_params(labelsize=param['ticksfontsize']-2)
# #        fig.savefig(opj(outfigpath, 'fig_topo_beta_cbar_' + regvar + '.svg'), dpi=600, bbox_inches='tight')
# #    
# #    fig.savefig(opj(outfigpath, 'fig_ols_erps_betas_topo_'
# #                    + regvar + '_' + str(tidx) + '.svg'), dpi=600, bbox_inches='tight')
# #    
# #    
# ##loop over channels instead of regression variables
# #for c in chan_to_plot:
# #    fig, line_axis = plt.subplots(1, 1, figsize=(4, 2.5))
# #    
# #    
# #    # Initialize variables for binning
# #    all_epos.metadata.reset_index()
# #    bina = 'Quartile'
# #    nbins = 2
# #    all_epos.metadata['bin'] = 0
# #    all_epos.metadata['bin'], bins = pd.qcut(all_epos.metadata[regvar], 
# #                                             nbins, 
# #                                             labels=False, retbins=True)
# #    all_epos.metadata['bin_' + regvar] = all_epos.metadata['bin']
# #    
# #    # Bin labels
# #    bin_labels = []
# #    for bidx, b in enumerate(bins):
# #        if b < 0:
# #            b = 0
# #        if bidx < len(bins)-1:
# #            lab = f"{round(b, 10)}-{round(bins[bidx + 1], 10)}"
# #            count = np.where(all_epos.metadata['bin'] == bidx)[0].shape[0]
# #            bin_labels.append(lab)
# #    
# #    # Bin colors
# #    colors = {str(val): val for val in all_epos.metadata['bin'].unique()}
# #    
# #    # Average within participants
# #    sub_evokeds = []
# #    for p in all_epos.metadata['participant'].unique():
# #        sub_dat = all_epos[all_epos.metadata['participant'] == p]
# #        sub_evoked = {}
# #        for val in range(nbins):
# #            if np.sum(sub_dat.metadata['bin'] == val) != 0:
# #                sub_evoked[val] = sub_dat[sub_dat.metadata['bin'] == val].average()
# #            else:
# #                sub_evoked[val] = 0
# #        sub_evokeds.append(sub_evoked)
# #
# #    # Grand average
# #    evokeds = dict()
# #    for i in range(len(bin_labels)):
# #        evoked = [sub_evoked[i] for sub_evoked in sub_evokeds if sub_evoked[i] != 0]
# #        evokeds[str(i + 1)] = mne.grand_average(evoked)
# #        
# #    pick = beta_gavg_nomast.ch_names.index(c)
# #    
# #    line_axis.set_ylabel(f'Beta ({regvarname})', fontdict={'size': param['labelfontsize']})
# #    _, axis = plt.subplots(figsize=(4, 2.5))
# #    
# #    cbarout = mne.viz.plot_compare_evokeds(evokeds, picks=pick, cmap=(regvarname + "\n(Quartile)", cmap), show_sensors=False, show=False, axes=axis)
# #    cbarout[0].axes[-1].yaxis.label.set_size(param['labelfontsize'])
# #    cbarout[0].axes[-1].tick_params(labelsize=param['ticksfontsize'])
# #    cbarout[0].axes[0].remove()
# #    cbarout[0].savefig(opj(outfigpath, 'fig_ols_erps_betas_line_cbar_' + regvar + '_' + c + '.svg'), dpi=800, bbox_inches='tight')
# #    
# #    for idx, bin in enumerate([str(i + 1) for i in range(nbins)]):
# #        line_axis.plot(all_epos[0].times * 1000, evokeds[bin].data[pick, :] * 1000000, label=str(idx + 1), linewidth=2, color=plt.cm.get_cmap(cmap, nbins)(idx / nbins))
# #    
# #    line_axis.tick_params(labelsize=12)
# #    line_axis.set_xlabel('Time (ms)', fontdict={'size': param['labelfontsize']})
# #    line_axis.set_ylabel('Amplitude (uV)', fontdict={'size': param['labelfontsize']})
# #    line_axis.axhline(0, linestyle='--', color='gray')
# #    line_axis.axvline(0, ymin=0, ymax=0.2, linestyle='--', color='gray')
# #    line_axis.get_xaxis().tick_bottom()
# #    line_axis.get_yaxis().tick_left()
# #    line_axis.set_xticks(ticks=np.arange(-200, 1200, 200))
# #    line_axis.set_xticklabels(labels=[str(i) for i in np.arange(-200, 1200, 200)])
# #    line_axis.tick_params(labelsize=param['ticksfontsize'])
# #    
# #    fig.tight_layout()
# #    fig.savefig(opj(outfigpath, 'fig_ols_erps_amp_bins_' + regvar + '_' + c + '.svg'), dpi=600, bbox_inches='tight')
# #
# #
# #bins_topo = list(range(nbins))
# #
# #for idx, binnum in enumerate([str(i + 1) for i in bins_topo]):
# #    fig, topo_axis = plt.subplots(figsize=(1, 1))
# #    
# #    tidx = np.argmin(np.abs(evokeds[binnum].times - 0.6))
# #    dat = evokeds[binnum].data[:, tidx] * 1000000    
# #    im, _ = plot_topomap(dat, 
# #                         pos=evokeds[binnum].info,
# #                         cmap=cmap,
# #                         show=False,
# #                         ch_type='eeg', 
# #                         outlines='head',
# #                         vlim=(None, None),
# #                         extrapolate='head',
# #                         axes=topo_axis, 
# #                         sensors=False,
# #                         contours=0)
# #    topo_axis.set_title(bina + ' ' + binnum, fontdict={'size': param['labelfontsize'] - 1}, pad=0.1)
# #    
# #    fig.savefig(opj(outfigpath, 'fig_binsamp_topo_' 
# #                    + regvar + '_bin' + binnum + '.svg'),dpi=600, bbox_inches='tight')
# #    
# #    if idx+1 == len(bins_topo):
# #        fig, ax = plt.subplots(figsize=(0.2, 1))
# #        cbar1 = fig.colorbar(im, 
# #                             cax=ax, 
# #                             orientation='vertical', 
# #                             aspect=1)
# #        cbar1.set_label('Amplitude (uV)',rotation=270, labelpad=12,fontdict={'fontsize': param["labelfontsize"] - 1})
# #        cbar1.ax.tick_params(labelsize=param['ticksfontsize'] - 2)
# #        fig.savefig(opj(outfigpath, 'fig_topo_bins_cbar_' + regvar + '.svg'), dpi=600, bbox_inches='tight')
# #
# #
# #for c in chan_to_plot:
# #    fig, line_axis = plt.subplots(1, 1, figsize=(4, 2.5))
# #    
# #    pick = beta_gavg_nomast.ch_names.index(c)
# #    
# #    # mean and SEM for the channel 
# #    sub_avg = []
# #    for s in range(allbetas.shape[0]):
# #        sub_avg.append(allbetas[s, 0, pick, :])
# #    
# #    sub_avg = np.stack(sub_avg)
# #    sem = scipy.stats.sem(sub_avg, axis=0)
# #    mean = beta_gavg_nomast.data[pick, :]
# #
# #    clrs = sns.color_palette("deep", 5)
# #    
# #    line_axis.set_ylabel('Beta (' + regvarname + ')', 
# #                         fontdict={'size': param['labelfontsize']})
# #    line_axis.set_xlabel('Time (ms)', 
# #                         fontdict={'size': param['labelfontsize']})
# #    line_axis.plot(all_epos[0].times * 1000,mean, label=str(idx+1),linewidth=3)
# #    line_axis.fill_between(all_epos[0].times * 1000, mean - sem, 
# #                           mean + sem, alpha=0.3, 
# #                           facecolor=clrs[0])
# #    line_axis.set_ylim((-0.2, 0.25))
# #    line_axis.axhline(0, linestyle='--', color='gray')
# #    line_axis.axvline(0, ymin=0, ymax=0.2, linestyle='--', color='gray')
# #    line_axis.get_xaxis().tick_bottom()
# #    line_axis.get_yaxis().tick_left()
# #    line_axis.tick_params(axis='both', labelsize=param['ticksfontsize'])
# #    
# #    # significant time points
# #    timestep = 1024 / param['testresampfreq']
# #    for tidx2, t2 in enumerate(all_epos[0].times * 1000):
# #        if pvals[0][tidx2, pick] < param['alpha']:
# #            line_axis.fill_between([t2, 
# #                                    t2 + timestep], 
# #                                   -0.02, -0.005, 
# #                                   alpha=0.3, 
# #                                   facecolor='red')
# #    
# #    line_axis.set_xticks(ticks=np.arange(-200, 1200, 200))
# #    
# #    line_axis.set_xticklabels(labels=[str(i) for i in 
# #                                      np.arange(-200, 1200, 200)])
# #    
# #    fig.tight_layout()
# #    fig.savefig(opj(outfigpath, 
# #                    'fig_ols_erps_betas_' + regvar + '_' 
# #                    + c + '.svg'), dpi=600, bbox_inches='tight')
# #
# #

# #-----------------------------------------------------------------------------------------------------------------
# #-----------------------------------------------------------------------------------------------------------------
# # for multiple regvars
# #-----------------------------------------------------------------------------------------------------------------
# #-----------------------------------------------------------------------------------------------------------------

# tvals = np.load(opj(outpath, 'ols_2ndlevel_tvals.npy'))
# pvals = np.load(opj(outpath, 'ols_2ndlevel_pvals.npy'))

# beta_gavg = np.load(opj(outpath, 'ols_2ndlevel_betasavg.npy'),
#                     allow_pickle=True)
# allbetas = np.load(opj(outpath, 'ols_2ndlevel_betas.npy'),
#                    allow_pickle=True)

# # Must be in the same order as in the stats code
# regvars = ['v_pain_contrib','v_money_contrib','v_interaction_contrib']
# regvarsnames = ['v_pain_contrib','v_money_contrib','v_interaction_contrib']

# # ## Plot
# # Plot descritive topo data
# plot_times = [0.2, 0.4, 0.6, 0.8, 1.0]
# times_pos = [np.abs(beta_gavg[0].times-0.2 - t).argmin() for t in plot_times]

# chan_to_plot = ['Fz', 'FCz', 'POz', 'Cz', 'CPz', 'Pz', 'Oz'] 
  
# for ridx, regvar in enumerate(regvars):

#     if ridx == 0:
#         vminmax = 6
#         cmap = 'viridis'
#     elif ridx == 1:
#         vminmax = 6
#         cmap = 'cividis'
#     elif ridx == 2:
#         vminmax = 6
#         cmap = 'plasma'

#     all_epos = mne.read_epochs(
#         opj(outpath, 'ols_2ndlevel_allepochs-epo_' + regvar + '.fif'))

#     regvarname = regvarsnames[ridx]

#     beta_gavg_nomast = beta_gavg[ridx].copy()
#     chankeep = [True if c not in ['M1', 'M2'] else False for c in
#                 beta_gavg[ridx].ch_names]

#     for tidx, timepos in enumerate(times_pos):
#         fig, topo_axis = plt.subplots(figsize=(1, 1))

#         im, _ = plot_topomap(beta_gavg_nomast.data[:, timepos],
#                              pos=beta_gavg_nomast.info,
#                              mask=pvals[ridx][timepos,
#                                               chankeep] < param['alpha'],
#                              mask_params=dict(marker='o',
#                                               markerfacecolor='w',
#                                               markeredgecolor='k',
#                                               linewidth=0,
#                                               markersize=2),
#                              cmap=cmap,
#                              show=False,
#                              ch_type='eeg',
#                              outlines='head',
#                              extrapolate='head',
#                              vlim=(-0.15, 0.15),
#                              axes=topo_axis,
#                              sensors=False,
#                              contours=0,)
#         topo_axis.set_title(str(int(plot_times[tidx] * 1000)) + ' ms',
#                             fontdict={'size': param['labelfontsize']-1}, pad=0.1)

#         if tidx+1 == len(plot_times):
#             fig, ax = plt.subplots(figsize=(0.2, 1))
#             cbar1 = fig.colorbar(im, cax=ax,
#                                  orientation='vertical', aspect=1)
#             cbar1.set_label('Beta', rotation=270,
#                             labelpad=12, fontdict={'fontsize': param["labelfontsize"]-1})
#             cbar1.ax.tick_params(labelsize=param['ticksfontsize']-2)
#             fig.savefig(opj(outfigpath, 'fig_topo_beta_cbar' + str(ridx) + '.svg'),
#                         dpi=600, bbox_inches='tight')
#         # fig.tight_layout()
#         fig.savefig(opj(outfigpath, 'fig_ols_erps_betas_topo_'
#                         + regvar + '_' + str(tidx) + '.svg'),
#                     dpi=600, bbox_inches='tight')

#     for c in chan_to_plot:
#         fig, line_axis = plt.subplots(1, 1, figsize=(4, 2.5))
#         regvarname = regvarsnames[ridx]
#         all_epos.metadata.reset_index()
#         if regvarname == 'Expectation':
#             bina = 'Ventile'
#             nbins = 5
#         else:
#             bina = 'Ventile'
#             nbins = 5
#         all_epos.metadata['bin'] = 0
#         unique_vals = all_epos.metadata[regvar].nunique()
#         nbins_eff = min(nbins, unique_vals)

#         all_epos.metadata['bin'], bins = pd.qcut(all_epos.metadata[regvar],
#                                                 q=nbins_eff,
#                                                 labels=False,
#                                                 retbins=True,
#                                                 duplicates='drop')
#         all_epos.metadata['bin' + '_' + regvar] = all_epos.metadata['bin']
        
#         # Bin labels
#         bin_labels = []
#         for bidx, b in enumerate(bins):
#             if b < 0:
#                 b = 0
#             if bidx < len(bins)-1:
#                 lab = [str(round(b, 10)) + '-'
#                        + str(round(bins[bidx+1], 10))][0]
#                 count = np.where(all_epos.metadata['bin'] == bidx)[0].shape[0]

#                 bin_labels.append(lab)

#         colors = {str(val): val for val in all_epos.metadata['bin'].unique()}

#         # Average within participants
#         sub_evokeds = []
#         sub_evoked_plot = dict()
#         for p in all_epos.metadata['participant_id'].unique():
#             sub_dat = all_epos[all_epos.metadata['participant_id'] == p]
#             sub_evoked = {}
#             for val in range(nbins):
#                 if np.sum(sub_dat.metadata['bin'] == val) != 0:
#                     sub_evoked[val] = sub_dat[sub_dat.metadata['bin']
#                                               == val].average()
#                 else:
#                     sub_evoked[val] = 0
#             sub_evokeds.append(sub_evoked)

#         # Grand average
#         evokeds = dict()
#         for i in range(len(bin_labels)):
#             evoked = [sub_evoked[i] for sub_evoked in sub_evokeds
#                       if sub_evoked[i] != 0]
#             evokeds[str(i+1)] = mne.grand_average(evoked)

#         pick = beta_gavg[ridx].ch_names.index(c)

#         line_axis.set_ylabel('Beta (' + regvarname + ')',
#                              fontdict={'size': param['labelfontsize']})

#         _, axis = plt.subplots(figsize=(4, 2.5))
#         cbarout = mne.viz.plot_compare_evokeds(evokeds, picks=pick, cmap=(regvarname + "\n(Decile)", cmap), show_sensors=False,
#                                                show=False, axes=axis)
#         cbarout[0].axes[-1].yaxis.label.set_size(param['labelfontsize'])
#         cbarout[0].axes[-1].tick_params(labelsize=param['ticksfontsize'])
#         cbarout[0].axes[0].remove()
#         cbarout[0].savefig(opj(outfigpath, 'fig_ols_erps_betas_line_cbar' + regvar + '_' + c + '.svg'),
#                            dpi=800, bbox_inches='tight')
        
#         bin_ids = sorted(evokeds.keys(), key=lambda x: int(x))
        
#         for idx, bin_id in enumerate(bin_ids):
#             line_axis.plot(
#                 all_epos[0].times * 1000,
#                 evokeds[bin_id].data[pick, :] * 1000000,
#                 label=str(idx + 1),
#                 linewidth=2,
#                 color=plt.get_cmap(cmap)(idx / len(bin_ids))
#             )
        

#         line_axis.tick_params(labelsize=12)
#         line_axis.set_xlabel('Time (ms)',
#                              fontdict={'size': param['labelfontsize']})
#         line_axis.set_ylabel('Amplitude (uV)',
#                              fontdict={'size': param['labelfontsize']})
#         line_axis.axhline(0, linestyle='--', color='gray')
#         line_axis.axvline(0, ymin=-0.2,
#                           ymax=0.2,
#                           linestyle='--', color='gray')
#         line_axis.get_xaxis().tick_bottom()
#         line_axis.get_yaxis().tick_left()
#         line_axis.set_xticks(ticks=np.arange(-200, 1200, 200))
#         line_axis

#         line_axis.set_xticklabels(labels=[str(i) for i in
#                                           np.arange(-200, 1200, 200)])
#         line_axis.tick_params(labelsize=param['ticksfontsize'])
#         fig.tight_layout()
#         fig.savefig(opj(outfigpath,
#                         'fig_ols_erps_amp_bins_' + regvar + '_'
#                         + c + '.svg'),
#                     dpi=600, bbox_inches='tight')
        
#     # Use only actual bins that exist in evokeds
#     bin_ids = sorted(evokeds.keys(), key=lambda x: int(x))
    
#     for idx, binnum in enumerate(bin_ids):
#         fig, topo_axis = plt.subplots(figsize=(1, 1))
    
#         # safe indexing — no more KeyError
#         tidx = np.argmin(np.abs(evokeds[binnum].times - 0.6))
#         dat = evokeds[binnum].data[:, tidx] * 1000000
    
#         im, _ = plot_topomap(
#             dat,
#             pos=evokeds[binnum].info,
#             cmap=cmap,
#             show=False,
#             ch_type='eeg',
#             outlines='head',
#             vlim=(-vminmax, vminmax),
#             extrapolate='head',
#             axes=topo_axis,
#             sensors=False,
#             contours=0,
#         )
#         topo_axis.set_title(bina + ' ' + binnum,
#                             fontdict={'size': param['labelfontsize']-1},
#                             pad=0.1)
    
#         fig.savefig(opj(outfigpath,
#                         f'fig_binsamp_topo_{regvar}_bin{binnum}.svg'),
#                     dpi=600, bbox_inches='tight')
    
#         # add colorbar at last bin
#         if idx + 1 == len(bin_ids):
#             fig, ax = plt.subplots(figsize=(0.2, 1))
#             cbar1 = fig.colorbar(im, cax=ax, orientation='vertical', aspect=1)
#             cbar1.set_label(
#                 'Amplitude (uV)',
#                 rotation=270,
#                 labelpad=12,
#                 fontdict={'fontsize': param["labelfontsize"]-1}
#             )
#             cbar1.ax.tick_params(labelsize=param['ticksfontsize']-2)
#             fig.savefig(opj(outfigpath,
#                             f'fig_topo_bins_cbar{ridx}.svg'),
#                         dpi=600, bbox_inches='tight')

#     for c in chan_to_plot:
#         fig, line_axis = plt.subplots(1, 1, figsize=(4, 2.5))

#         regvarname = regvarsnames[ridx]
#         all_epos.metadata.reset_index()
#         pick = beta_gavg[ridx].ch_names.index(c)

#         sub_avg = []
#         for s in range(allbetas.shape[0]):
#             sub_avg.append(allbetas[s, ridx, pick, :])

#         sub_avg = np.stack(sub_avg)

#         sem = scipy.stats.sem(sub_avg, axis=0)
#         mean = beta_gavg[ridx].data[pick, :]

#         clrs = sns.color_palette("deep", 5)

#         line_axis.set_ylabel('Beta (' + regvarname + ')',
#                              fontdict={'size': param['labelfontsize']})
#         line_axis.set_xlabel('Time (ms)',
#                              fontdict={'size': param['labelfontsize']})

#         line_axis.plot(all_epos[0].times * 1000,
#                        beta_gavg[ridx].data[pick, :],
#                        label=str(idx + 1),
#                        linewidth=3)
#         line_axis.fill_between(all_epos[0].times * 1000,
#                                mean - sem, mean + sem, alpha=0.3,
#                                facecolor=clrs[0])
#         # Make it nice
#         line_axis.set_ylim((-0.25, 0.25))

#         line_axis.axhline(0, linestyle='--', color='gray')
#         line_axis.axvline(0, ymin=0,
#                           ymax=0.2,
#                           linestyle='--', color='gray')
#         line_axis.get_xaxis().tick_bottom()
#         line_axis.get_yaxis().tick_left()
#         line_axis.tick_params(axis='both',
#                               labelsize=param['ticksfontsize'])

#         pvals[ridx][:, pick]
#         timestep = 1024 / param['testresampfreq']
#         for tidx2, t2 in enumerate(all_epos[0].times * 1000):
#             if pvals[ridx][tidx2, pick] < param['alpha']:
#                 line_axis.fill_between([t2,
#                                         t2 + timestep],
#                                        -0.02, -0.005, alpha=0.3,
#                                        facecolor='red')

#         line_axis.set_xticks(ticks=np.arange(-200, 1200, 200))

#         line_axis.set_xticklabels(labels=[str(i) for i in
#                                           np.arange(-200, 1200, 200)])

#         fig.tight_layout()
#         fig.savefig(opj(outfigpath,
#                         'fig_ols_erps_betas_' + regvar + '_'
#                         + c + '.svg'),
#                     dpi=600, bbox_inches='tight')



# # Load 2nd-level results
# #---------------------------------------------------------------------------------------------------------------------------------------
# #---------------------------------------------------------------------------------------------------------------------------------------
# #---------------------------------------------------------------------------------------------------------------------------------------


# # tvals = np.load(opj(outpath, 'ols_2ndlevel_tvals.npy'))
# # pvals = np.load(opj(outpath, 'ols_2ndlevel_pvals.npy'))

# # beta_gavg = np.load(
# #     opj(outpath, 'ols_2ndlevel_betasavg.npy'),
# #     allow_pickle=True
# # )
# # allbetas = np.load(
# #     opj(outpath, 'ols_2ndlevel_betas.npy'),
# #     allow_pickle=True
# # )

# # # Must be in the same order as in the stats code
# # regvars = ['v_pain_contrib', 'v_money_contrib', 'v_interaction_contrib']
# # regvarsnames = ['v_pain_contrib', 'v_money_contrib', 'v_interaction_contrib']

# # # Plot descriptive topo data
# # plot_times = [0.2, 0.4, 0.6, 0.8, 1.0]
# # times_pos = [np.abs(beta_gavg[0].times - 0.2 - t).argmin() for t in plot_times]

# # chan_to_plot = ['Fz', 'FCz', 'POz', 'Cz', 'CPz', 'Pz', 'Oz']

# # # =================================================================================================
# # # MAIN LOOP OVER REGRESSORS
# # # =================================================================================================

# # for ridx, regvar in enumerate(regvars):

# #     if ridx == 0:
# #         vminmax = 6
# #         cmap = 'viridis'
# #     elif ridx == 1:
# #         vminmax = 6
# #         cmap = 'cividis'
# #     elif ridx == 2:
# #         vminmax = 6
# #         cmap = 'plasma'

# #     regvarname = regvarsnames[ridx]

# #     # -------------------------------------------------------------------------
# #     # Load epochs for this regressor
# #     # -------------------------------------------------------------------------
# #     all_epos = mne.read_epochs(
# #         opj(outpath, f'ols_2ndlevel_allepochs-epo_{regvar}.fif')
# #     )

# #     # -------------------------------------------------------------------------
# #     # RT BINNING (Fast / Medium / Slow)
# #     # -------------------------------------------------------------------------
# #     print(f"\nRT stratification for regressor {regvar}...")

# #     # Which RT column?
# #     if "choice_rt" in all_epos.metadata.columns:
# #         rt_col = "choice_rt"
# #     elif "rt" in all_epos.metadata.columns:
# #         rt_col = "rt"
# #     else:
# #         raise ValueError(
# #             f"No RT column found in metadata! Available: {all_epos.metadata.columns}"
# #         )

# #     # remove trials with missing RT
# #     all_epos = all_epos[all_epos.metadata[rt_col].notnull()]

# #     # Fast / Medium / Slow
# #     rt_nbins = 3

# #     # Create RT bins
# #     all_epos.metadata["RT_bin"], rt_bins = pd.qcut(
# #         all_epos.metadata[rt_col],
# #         q=rt_nbins,
# #         labels=False,
# #         retbins=True,
# #         duplicates="drop"
# #     )

# #     rt_labels = ["Fast", "Medium", "Slow"][:rt_nbins]
# #     print("RT bin edges:", rt_bins)
# #     print(all_epos.metadata["RT_bin"].value_counts())

# #     # Subject-level averaging for RT bins
# #     sub_rt_evokeds = []
# #     for subj in all_epos.metadata["participant"].unique():
# #         subj_dat = all_epos[all_epos.metadata["participant"] == subj]
# #         subj_evoked = {}
# #         for b in range(rt_nbins):
# #             if np.sum(subj_dat.metadata["RT_bin"] == b) > 0:
# #                 subj_evoked[b] = subj_dat[subj_dat.metadata["RT_bin"] == b].average()
# #             else:
# #                 subj_evoked[b] = None
# #         sub_rt_evokeds.append(subj_evoked)

# #     # Grand averages per RT bin
# #     rt_evokeds = {}
# #     for b in range(rt_nbins):
# #         evs = [sub[b] for sub in sub_rt_evokeds if sub[b] is not None]
# #         rt_evokeds[rt_labels[b]] = mne.grand_average(evs)

# #     print("RT stratification complete.")

# #     # -------------------------------------------------------------------------
# #     # ORIGINAL BETA TOPO PLOTS WITH SIGNIFICANCE MASK
# #     # -------------------------------------------------------------------------
# #     beta_gavg_nomast = beta_gavg[ridx].copy()
# #     chankeep = [True if c not in ['M1', 'M2'] else False
# #                 for c in beta_gavg[ridx].ch_names]

# #     for tidx, timepos in enumerate(times_pos):
# #         fig, topo_axis = plt.subplots(figsize=(1, 1))

# #         im, _ = plot_topomap(
# #             beta_gavg_nomast.data[:, timepos],
# #             pos=beta_gavg_nomast.info,
# #             mask=pvals[ridx][timepos, chankeep] < param['alpha'],
# #             mask_params=dict(
# #                 marker='o',
# #                 markerfacecolor='w',
# #                 markeredgecolor='k',
# #                 linewidth=0,
# #                 markersize=2
# #             ),
# #             cmap=cmap,
# #             show=False,
# #             ch_type='eeg',
# #             outlines='head',
# #             extrapolate='head',
# #             vlim=(-0.15, 0.15),
# #             axes=topo_axis,
# #             sensors=False,
# #             contours=0,
# #         )
# #         topo_axis.set_title(
# #             f"{int(plot_times[tidx] * 1000)} ms",
# #             fontdict={'size': param['labelfontsize'] - 1},
# #             pad=0.1
# #         )

# #         if tidx + 1 == len(plot_times):
# #             fig_cb, ax_cb = plt.subplots(figsize=(0.2, 1))
# #             cbar1 = fig_cb.colorbar(im, cax=ax_cb,
# #                                     orientation='vertical', aspect=1)
# #             cbar1.set_label(
# #                 'Beta',
# #                 rotation=270,
# #                 labelpad=12,
# #                 fontdict={'fontsize': param["labelfontsize"] - 1}
# #             )
# #             cbar1.ax.tick_params(labelsize=param['ticksfontsize'] - 2)
# #             fig_cb.savefig(
# #                 opj(outfigpath, f'fig_topo_beta_cbar{ridx}.svg'),
# #                 dpi=600,
# #                 bbox_inches='tight'
# #             )

# #         fig.savefig(
# #             opj(outfigpath,
# #                 f'fig_ols_erps_betas_topo_{regvar}_{tidx}.svg'),
# #             dpi=600,
# #             bbox_inches='tight'
# #         )

# #     # -------------------------------------------------------------------------
# #     # VALUE-BINNED ERPs & TOPO (DECILES)
# #     # -------------------------------------------------------------------------
# #     for c in chan_to_plot:
# #         fig, line_axis = plt.subplots(1, 1, figsize=(4, 2.5))

# #         all_epos.metadata.reset_index(drop=True)
# #         if regvarname == 'Expectation':
# #             bina = 'Ventile'
# #             nbins = 5
# #         else:
# #             bina = 'Ventile'
# #             nbins = 5

# #         all_epos.metadata['bin'] = 0
# #         unique_vals = all_epos.metadata[regvar].nunique()
# #         nbins_eff = min(nbins, unique_vals)

# #         all_epos.metadata['bin'], bins = pd.qcut(
# #             all_epos.metadata[regvar],
# #             q=nbins_eff,
# #             labels=False,
# #             retbins=True,
# #             duplicates='drop'
# #         )
# #         all_epos.metadata[f'bin_{regvar}'] = all_epos.metadata['bin']

# #         # Bin labels
# #         bin_labels = []
# #         for bidx, b in enumerate(bins):
# #             if b < 0:
# #                 b = 0
# #             if bidx < len(bins) - 1:
# #                 lab = f"{round(b, 10)}-{round(bins[bidx + 1], 10)}"
# #                 bin_labels.append(lab)

# #         # Average within participants
# #         sub_evokeds = []
# #         for p in all_epos.metadata['participant'].unique():
# #             sub_dat = all_epos[all_epos.metadata['participant'] == p]
# #             sub_evoked = {}
# #             for val in range(nbins):
# #                 if np.sum(sub_dat.metadata['bin'] == val) != 0:
# #                     sub_evoked[val] = sub_dat[sub_dat.metadata['bin'] == val].average()
# #                 else:
# #                     sub_evoked[val] = 0
# #             sub_evokeds.append(sub_evoked)

# #         # Grand average per value bin
# #         evokeds = dict()
# #         for i in range(len(bin_labels)):
# #             evoked = [sub_evoked[i] for sub_evoked in sub_evokeds
# #                       if sub_evoked[i] != 0]
# #             evokeds[str(i + 1)] = mne.grand_average(evoked)

# #         pick = beta_gavg[ridx].ch_names.index(c)

# #         line_axis.set_ylabel(
# #             'Beta (' + regvarname + ')',
# #             fontdict={'size': param['labelfontsize']}
# #         )

# #         # colorbar for deciles
# #         fig_cb, axis_cb = plt.subplots(figsize=(4, 2.5))
# #         cbarout = mne.viz.plot_compare_evokeds(
# #             evokeds,
# #             picks=pick,
# #             cmap=(regvarname + "\n(Decile)", cmap),
# #             show_sensors=False,
# #             show=False,
# #             axes=axis_cb
# #         )
# #         cbarout[0].axes[-1].yaxis.label.set_size(param['labelfontsize'])
# #         cbarout[0].axes[-1].tick_params(labelsize=param['ticksfontsize'])
# #         cbarout[0].axes[0].remove()
# #         cbarout[0].savefig(
# #             opj(outfigpath,
# #                 f'fig_ols_erps_betas_line_cbar{regvar}_{c}.svg'),
# #             dpi=800,
# #             bbox_inches='tight'
# #         )

# #         bin_ids = sorted(evokeds.keys(), key=lambda x: int(x))

# #         # Plot decile ERPs
# #         for idx, bin_id in enumerate(bin_ids):
# #             line_axis.plot(
# #                 all_epos[0].times * 1000,
# #                 evokeds[bin_id].data[pick, :] * 1e6,
# #                 label=str(idx + 1),
# #                 linewidth=2,
# #                 color=plt.get_cmap(cmap)(idx / len(bin_ids))
# #             )

# #         line_axis.tick_params(labelsize=12)
# #         line_axis.set_xlabel(
# #             'Time (ms)',
# #             fontdict={'size': param['labelfontsize']}
# #         )
# #         line_axis.set_ylabel(
# #             'Amplitude (uV)',
# #             fontdict={'size': param['labelfontsize']}
# #         )
# #         line_axis.axhline(0, linestyle='--', color='gray')
# #         line_axis.axvline(0, ymin=-0.2, ymax=0.2,
# #                           linestyle='--', color='gray')
# #         line_axis.get_xaxis().tick_bottom()
# #         line_axis.get_yaxis().tick_left()
# #         line_axis.set_xticks(ticks=np.arange(-200, 1200, 200))
# #         line_axis.set_xticklabels(
# #             labels=[str(i) for i in np.arange(-200, 1200, 200)]
# #         )
# #         line_axis.tick_params(labelsize=param['ticksfontsize'])
# #         fig.tight_layout()
# #         fig.savefig(
# #             opj(outfigpath,
# #                 f'fig_ols_erps_amp_bins_{regvar}_{c}.svg'),
# #             dpi=600,
# #             bbox_inches='tight'
# #         )

# #     # Topos for value bins at ~600 ms
# #     bin_ids = sorted(evokeds.keys(), key=lambda x: int(x))
# #     for idx, binnum in enumerate(bin_ids):
# #         fig, topo_axis = plt.subplots(figsize=(1, 1))

# #         tidx = np.argmin(np.abs(evokeds[binnum].times - 0.6))
# #         dat = evokeds[binnum].data[:, tidx] * 1e6

# #         im, _ = plot_topomap(
# #             dat,
# #             pos=evokeds[binnum].info,
# #             cmap=cmap,
# #             show=False,
# #             ch_type='eeg',
# #             outlines='head',
# #             vlim=(-vminmax, vminmax),
# #             extrapolate='head',
# #             axes=topo_axis,
# #             sensors=False,
# #             contours=0,
# #         )
# #         topo_axis.set_title(
# #             bina + ' ' + binnum,
# #             fontdict={'size': param['labelfontsize'] - 1},
# #             pad=0.1
# #         )

# #         fig.savefig(
# #             opj(outfigpath,
# #                 f'fig_binsamp_topo_{regvar}_bin{binnum}.svg'),
# #             dpi=600,
# #             bbox_inches='tight'
# #         )

# #         # add colorbar at last bin
# #         if idx + 1 == len(bin_ids):
# #             fig_cb, ax_cb = plt.subplots(figsize=(0.2, 1))
# #             cbar1 = fig_cb.colorbar(im, cax=ax_cb,
# #                                     orientation='vertical', aspect=1)
# #             cbar1.set_label(
# #                 'Amplitude (uV)',
# #                 rotation=270,
# #                 labelpad=12,
# #                 fontdict={'fontsize': param["labelfontsize"] - 1}
# #             )
# #             cbar1.ax.tick_params(labelsize=param['ticksfontsize'] - 2)
# #             fig_cb.savefig(
# #                 opj(outfigpath, f'fig_topo_bins_cbar{ridx}.svg'),
# #                 dpi=600,
# #                 bbox_inches='tight'
# #             )

# #     # -------------------------------------------------------------------------
# #     # BETA TIMECOURSE WITH SEM + SIGNIFICANT TIMEPOINT SHADING
# #     # -------------------------------------------------------------------------
# #     for c in chan_to_plot:
# #         fig, line_axis = plt.subplots(1, 1, figsize=(4, 2.5))

# #         all_epos.metadata.reset_index(drop=True)
# #         pick = beta_gavg[ridx].ch_names.index(c)

# #         sub_avg = []
# #         for s in range(allbetas.shape[0]):
# #             sub_avg.append(allbetas[s, ridx, pick, :])
# #         sub_avg = np.stack(sub_avg)

# #         sem = scipy.stats.sem(sub_avg, axis=0)
# #         mean = beta_gavg[ridx].data[pick, :]

# #         clrs = sns.color_palette("deep", 5)

# #         line_axis.set_ylabel(
# #             'Beta (' + regvarname + ')',
# #             fontdict={'size': param['labelfontsize']}
# #         )
# #         line_axis.set_xlabel(
# #             'Time (ms)',
# #             fontdict={'size': param['labelfontsize']}
# #         )

# #         line_axis.plot(
# #             all_epos[0].times * 1000,
# #             mean,
# #             label=str(c),
# #             linewidth=3
# #         )
# #         line_axis.fill_between(
# #             all_epos[0].times * 1000,
# #             mean - sem,
# #             mean + sem,
# #             alpha=0.3,
# #             facecolor=clrs[0]
# #         )
# #         line_axis.set_ylim((-0.25, 0.25))

# #         line_axis.axhline(0, linestyle='--', color='gray')
# #         line_axis.axvline(0, ymin=0, ymax=0.2,
# #                           linestyle='--', color='gray')
# #         line_axis.get_xaxis().tick_bottom()
# #         line_axis.get_yaxis().tick_left()
# #         line_axis.tick_params(axis='both',
# #                               labelsize=param['ticksfontsize'])

# #         timestep = 1024 / param['testresampfreq']
# #         for tidx2, t2 in enumerate(all_epos[0].times * 1000):
# #             if pvals[ridx][tidx2, pick] < param['alpha']:
# #                 line_axis.fill_between(
# #                     [t2, t2 + timestep],
# #                     -0.02,
# #                     -0.005,
# #                     alpha=0.3,
# #                     facecolor='red'
# #                 )

# #         line_axis.set_xticks(ticks=np.arange(-200, 1200, 200))
# #         line_axis.set_xticklabels(
# #             labels=[str(i) for i in np.arange(-200, 1200, 200)]
# #         )

# #         fig.tight_layout()
# #         fig.savefig(
# #             opj(outfigpath,
# #                 f'fig_ols_erps_betas_{regvar}_{c}.svg'),
# #             dpi=600,
# #             bbox_inches='tight'
# #         )

# #     # -------------------------------------------------------------------------
# #     # RT-STRATIFIED ERPs (Fast / Medium / Slow) PER CHANNEL
# #     # -------------------------------------------------------------------------
# #     for c in chan_to_plot:
# #         pick = beta_gavg[ridx].ch_names.index(c)

# #         fig, ax = plt.subplots(figsize=(4, 2.5))
# #         for label, ev in rt_evokeds.items():
# #             ax.plot(
# #                 ev.times * 1000,
# #                 ev.data[pick, :] * 1e6,
# #                 linewidth=2,
# #                 label=label
# #             )

# #         ax.set_title(f"RT-binned ERPs at {c} – {regvarname}")
# #         ax.set_xlabel("Time (ms)")
# #         ax.set_ylabel("Amplitude (µV)")
# #         ax.axvline(0, color="gray", linestyle="--")
# #         ax.axhline(0, color="gray", linestyle="--")
# #         ax.legend()

# #         fig.tight_layout()
# #         fig.savefig(
# #             opj(outfigpath,
# #                 f"RTbins_{regvarname}_{c}.svg"),
# #             dpi=600,
# #             bbox_inches='tight'
# #         )

# #     # -------------------------------------------------------------------------
# #     # RT-STRATIFIED TOPO MAPS (Fast / Medium / Slow)
# #     # -------------------------------------------------------------------------
# #     print(f"Creating RT-topomaps for regressor {regvarname}...")

# #     for label, ev in rt_evokeds.items():   # Fast / Medium / Slow
# #         for t in plot_times:               # 0.2, 0.4, 0.6, 0.8, 1.0 seconds
# #             tidx_rt = np.argmin(np.abs(ev.times - t))

# #             fig, topo_axis = plt.subplots(figsize=(1.2, 1.2))

# #             im, _ = plot_topomap(
# #                 ev.data[:, tidx_rt] * 1e6,        # convert to µv
# #                 pos=ev.info,
# #                 cmap=cmap,
# #                 show=False,
# #                 ch_type='eeg',
# #                 outlines='head',
# #                 extrapolate='head',
# #                 vlim=(-vminmax, vminmax),
# #                 axes=topo_axis,
# #                 sensors=False,
# #                 contours=0
# #             )

# #             topo_axis.set_title(
# #                 f"{label} RT – {int(t * 1000)} ms",
# #                 fontdict={'size': param['labelfontsize'] - 1},
# #                 pad=0.1
# #             )

# #             fig.savefig(
# #                 opj(outfigpath,
# #                     f"fig_RTtopo_{regvar}_{label}_{int(t * 1000)}ms.svg"),
# #                 dpi=600,
# #                 bbox_inches='tight'
# #             )

# #         # colorbar for this RT bin
# #         fig_cb, ax_cb = plt.subplots(figsize=(0.25, 1))
# #         cbar = fig_cb.colorbar(im, cax=ax_cb, orientation='vertical', aspect=1)
# #         cbar.set_label(
# #             'Amplitude (µV)',
# #             rotation=270,
# #             labelpad=12,
# #             fontdict={'fontsize': param["labelfontsize"] - 1}
# #         )
# #         cbar.ax.tick_params(labelsize=param['ticksfontsize'] - 2)

# #         fig_cb.savefig(
# #             opj(outfigpath,
# #                 f"fig_RTtopo_cbar_{regvar}_{label}.svg"),
# #             dpi=600,
# #             bbox_inches='tight'
# #         )

# #     print("RT topomaps done")

