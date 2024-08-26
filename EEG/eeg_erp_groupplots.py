'''
 # @ : -*- coding: utf-8 -*-
 # @ Author: Michel-Pierre Coll (michel-pierre.coll@psy.ulaval.ca), edited by Veronika Wendler
 # @ Date: 2024
 # @ Description: lotting the eeg regression models from the eeg_erp_massunivariate_prep.py file
 '''
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

#-----------------------------------------------------------------------------------------------------

# Set bids directory
inpath = "D:/Aberdeen_Uni_June24/MPColl_Lab/All_Files_Relevant_For_Git/EEG/PainReward_sub-001-050/painrewardeegdata"
outpathall = "D:/Aberdeen_Uni_June24/MPColl_Lab/All_Files_Relevant_For_Git/EEG/PainReward_sub-001-050/painrewardeegdata/derivatives"

layout = BIDSLayout(inpath)

part = pd.read_csv(opj(inpath, 'participants.tsv'), sep='\t')
# part = part_csv['participant_id'].unique().tolist()

#layout = BIDSLayout(basepath)

# Outpath for analysis
outpath = opj(outpathall, 'statistics_2/erps_modelbased_sv_money_OV_high')    # for averaging over more electrodes: statistics_2
# Outpath for figures
outfigpath = opj(outpathall, 'figures_2/erps_modelbased_sv_money_OV_high')    # for averaging over more electrodes: figures_2
if not os.path.exists(outfigpath):
    os.mkdir(outfigpath)
param = {
    # Alpha Threshold
    'alpha': 0.05,

    # Font sizez in plot
    'titlefontsize': 12,
    'labelfontsize': 12,
    'ticksfontsize': 11,
    'legendfontsize': 10,
    # Downsample to this frequency prior to analysis
    'testresampfreq': 1024,
}

plt.rc("axes.spines", top=False, right=False)
plt.rcParams['font.family'] = 'Verdana'

#-----------------------------------------------------------------------------------------------------
# Multivariate Regression Plots from MP for only 1 regressor

tvals = np.load(opj(outpath, 'ols_2ndlevel_tvals.npy'))
pvals = np.load(opj(outpath, 'ols_2ndlevel_pvals.npy'))

beta_gavg = np.load(opj(outpath, 'ols_2ndlevel_betasavg.npy'),
                    allow_pickle=True)
allbetas = np.load(opj(outpath, 'ols_2ndlevel_betas.npy'),
                   allow_pickle=True)

# regression variable v_sv_pain_para_contrib
regvar = 'sv_money'
regvarname = 'sv_money'

# colormap and vminmax for this single regressor
vminmax = 10
cmap = 'viridis'

# Plot descriptive topo data
plot_times = [0.2, 0.4, 0.6, 0.8]
times_pos = [np.abs(beta_gavg[0].times - 0.2 - t).argmin() for t in plot_times]

chan_to_plot = ['F3', 'F4', 'Fz', 'FC5','FC6', 'FC1', 'FC2', 'FCz', 'C3','C4','CP1', 'CP2', 'CP5', 'CP6', 'P3', 'Pz', 'P4', 'P7', 'P8', 'PO3', 'PO7', 'PO4', 'O1', 'Oz', 'O2']    #['POz', 'Cz', 'Oz', 'Pz', 'CPz']                                                                                       

all_epos = mne.read_epochs(
    opj(outpath, 'ols_2ndlevel_allepochs-epo_' + regvar + '.fif'), preload=False)

beta_gavg_nomast = beta_gavg[0].copy()
chankeep = [True if c not in ['M1', 'M2'] else False for c in
            beta_gavg_nomast.ch_names]

for tidx, timepos in enumerate(times_pos):
    fig, topo_axis = plt.subplots(figsize=(1, 1))

    im, _ = plot_topomap(beta_gavg_nomast.data[:, timepos],
                         pos=beta_gavg_nomast.info,
                         mask=pvals[0][timepos,
                                       chankeep] < param['alpha'],
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
                         contours=0,)
    topo_axis.set_title(str(int(plot_times[tidx] * 1000)) + ' ms', fontdict={'size': param['labelfontsize']-1}, pad=0.1)

    if tidx+1 == len(plot_times):
        fig, ax = plt.subplots(figsize=(0.2, 1))
        cbar1 = fig.colorbar(im, cax=ax,
                             orientation='vertical', aspect=1)
        cbar1.set_label('Beta', rotation=270, labelpad=12, fontdict={'fontsize': param["labelfontsize"]-1})
        cbar1.ax.tick_params(labelsize=param['ticksfontsize']-2)
        fig.savefig(opj(outfigpath, 'fig_topo_beta_cbar_' + regvar + '.svg'), dpi=600, bbox_inches='tight')
    
    fig.savefig(opj(outfigpath, 'fig_ols_erps_betas_topo_'
                    + regvar + '_' + str(tidx) + '.svg'), dpi=600, bbox_inches='tight')
    
    
#loop over channels instead of regression variables
for c in chan_to_plot:
    fig, line_axis = plt.subplots(1, 1, figsize=(4, 2.5))
    
    
    # Initialize variables for binning
    all_epos.metadata.reset_index()
    bina = 'Quartile'
    nbins = 2
    all_epos.metadata['bin'] = 0
    all_epos.metadata['bin'], bins = pd.qcut(all_epos.metadata[regvar], 
                                             nbins, 
                                             labels=False, retbins=True)
    all_epos.metadata['bin_' + regvar] = all_epos.metadata['bin']
    
    # Bin labels
    bin_labels = []
    for bidx, b in enumerate(bins):
        if b < 0:
            b = 0
        if bidx < len(bins)-1:
            lab = f"{round(b, 10)}-{round(bins[bidx + 1], 10)}"
            count = np.where(all_epos.metadata['bin'] == bidx)[0].shape[0]
            bin_labels.append(lab)
    
    # Bin colors
    colors = {str(val): val for val in all_epos.metadata['bin'].unique()}
    
    # Average within participants
    sub_evokeds = []
    for p in all_epos.metadata['participant'].unique():
        sub_dat = all_epos[all_epos.metadata['participant'] == p]
        sub_evoked = {}
        for val in range(nbins):
            if np.sum(sub_dat.metadata['bin'] == val) != 0:
                sub_evoked[val] = sub_dat[sub_dat.metadata['bin'] == val].average()
            else:
                sub_evoked[val] = 0
        sub_evokeds.append(sub_evoked)

    # Grand average
    evokeds = dict()
    for i in range(len(bin_labels)):
        evoked = [sub_evoked[i] for sub_evoked in sub_evokeds if sub_evoked[i] != 0]
        evokeds[str(i + 1)] = mne.grand_average(evoked)
        
    pick = beta_gavg_nomast.ch_names.index(c)
    
    line_axis.set_ylabel(f'Beta ({regvarname})', fontdict={'size': param['labelfontsize']})
    _, axis = plt.subplots(figsize=(4, 2.5))
    
    cbarout = mne.viz.plot_compare_evokeds(evokeds, picks=pick, cmap=(regvarname + "\n(Quartile)", cmap), show_sensors=False, show=False, axes=axis)
    cbarout[0].axes[-1].yaxis.label.set_size(param['labelfontsize'])
    cbarout[0].axes[-1].tick_params(labelsize=param['ticksfontsize'])
    cbarout[0].axes[0].remove()
    cbarout[0].savefig(opj(outfigpath, 'fig_ols_erps_betas_line_cbar_' + regvar + '_' + c + '.svg'), dpi=800, bbox_inches='tight')
    
    for idx, bin in enumerate([str(i + 1) for i in range(nbins)]):
        line_axis.plot(all_epos[0].times * 1000, evokeds[bin].data[pick, :] * 1000000, label=str(idx + 1), linewidth=2, color=plt.cm.get_cmap(cmap, nbins)(idx / nbins))
    
    line_axis.tick_params(labelsize=12)
    line_axis.set_xlabel('Time (ms)', fontdict={'size': param['labelfontsize']})
    line_axis.set_ylabel('Amplitude (uV)', fontdict={'size': param['labelfontsize']})
    line_axis.axhline(0, linestyle='--', color='gray')
    line_axis.axvline(0, ymin=0, ymax=0.2, linestyle='--', color='gray')
    line_axis.get_xaxis().tick_bottom()
    line_axis.get_yaxis().tick_left()
    line_axis.set_xticks(ticks=np.arange(-200, 1200, 200))
    line_axis.set_xticklabels(labels=[str(i) for i in np.arange(-200, 1200, 200)])
    line_axis.tick_params(labelsize=param['ticksfontsize'])
    
    fig.tight_layout()
    fig.savefig(opj(outfigpath, 'fig_ols_erps_amp_bins_' + regvar + '_' + c + '.svg'), dpi=600, bbox_inches='tight')


bins_topo = list(range(nbins))

for idx, binnum in enumerate([str(i + 1) for i in bins_topo]):
    fig, topo_axis = plt.subplots(figsize=(1, 1))
    
    tidx = np.argmin(np.abs(evokeds[binnum].times - 0.6))
    dat = evokeds[binnum].data[:, tidx] * 1000000    
    im, _ = plot_topomap(dat, 
                         pos=evokeds[binnum].info,
                         cmap=cmap,
                         show=False,
                         ch_type='eeg', 
                         outlines='head',
                         vlim=(None, None),
                         extrapolate='head',
                         axes=topo_axis, 
                         sensors=False,
                         contours=0)
    topo_axis.set_title(bina + ' ' + binnum, fontdict={'size': param['labelfontsize'] - 1}, pad=0.1)
    
    fig.savefig(opj(outfigpath, 'fig_binsamp_topo_' 
                    + regvar + '_bin' + binnum + '.svg'),dpi=600, bbox_inches='tight')
    
    if idx+1 == len(bins_topo):
        fig, ax = plt.subplots(figsize=(0.2, 1))
        cbar1 = fig.colorbar(im, 
                             cax=ax, 
                             orientation='vertical', 
                             aspect=1)
        cbar1.set_label('Amplitude (uV)',rotation=270, labelpad=12,fontdict={'fontsize': param["labelfontsize"] - 1})
        cbar1.ax.tick_params(labelsize=param['ticksfontsize'] - 2)
        fig.savefig(opj(outfigpath, 'fig_topo_bins_cbar_' + regvar + '.svg'), dpi=600, bbox_inches='tight')


for c in chan_to_plot:
    fig, line_axis = plt.subplots(1, 1, figsize=(4, 2.5))
    
    pick = beta_gavg_nomast.ch_names.index(c)
    
    # mean and SEM for the channel 
    sub_avg = []
    for s in range(allbetas.shape[0]):
        sub_avg.append(allbetas[s, 0, pick, :])
    
    sub_avg = np.stack(sub_avg)
    sem = scipy.stats.sem(sub_avg, axis=0)
    mean = beta_gavg_nomast.data[pick, :]

    clrs = sns.color_palette("deep", 5)
    
    line_axis.set_ylabel('Beta (' + regvarname + ')', 
                         fontdict={'size': param['labelfontsize']})
    line_axis.set_xlabel('Time (ms)', 
                         fontdict={'size': param['labelfontsize']})
    line_axis.plot(all_epos[0].times * 1000,mean, label=str(idx+1),linewidth=3)
    line_axis.fill_between(all_epos[0].times * 1000, mean - sem, 
                           mean + sem, alpha=0.3, 
                           facecolor=clrs[0])
    line_axis.set_ylim((-0.2, 0.25))
    line_axis.axhline(0, linestyle='--', color='gray')
    line_axis.axvline(0, ymin=0, ymax=0.2, linestyle='--', color='gray')
    line_axis.get_xaxis().tick_bottom()
    line_axis.get_yaxis().tick_left()
    line_axis.tick_params(axis='both', labelsize=param['ticksfontsize'])
    
    # significant time points
    timestep = 1024 / param['testresampfreq']
    for tidx2, t2 in enumerate(all_epos[0].times * 1000):
        if pvals[0][tidx2, pick] < param['alpha']:
            line_axis.fill_between([t2, 
                                    t2 + timestep], 
                                   -0.02, -0.005, 
                                   alpha=0.3, 
                                   facecolor='red')
    
    line_axis.set_xticks(ticks=np.arange(-200, 1200, 200))
    
    line_axis.set_xticklabels(labels=[str(i) for i in 
                                      np.arange(-200, 1200, 200)])
    
    fig.tight_layout()
    fig.savefig(opj(outfigpath, 
                    'fig_ols_erps_betas_' + regvar + '_' 
                    + c + '.svg'), dpi=600, bbox_inches='tight')



#-----------------------------------------------------------------------------------------------------------------
# for multiple regvars

# tvals = np.load(opj(outpath, 'ols_2ndlevel_tvals.npy'))
# pvals = np.load(opj(outpath, 'ols_2ndlevel_pvals.npy'))

# beta_gavg = np.load(opj(outpath, 'ols_2ndlevel_betasavg.npy'),
#                     allow_pickle=True)
# allbetas = np.load(opj(outpath, 'ols_2ndlevel_betas.npy'),
#                    allow_pickle=True)

# # Must be in the same order as in the stats code
# regvars = ['sv_pain_para_SAI','sv_pain_para_TAI', 'sv_pain_para_PCS']
# regvarsnames = ['sv_pain_para_SAI','sv_pain_para_TAI', 'sv_pain_para_PCS']

# # ## Plot
# # Plot descritive topo data
# plot_times = [0.2, 0.4, 0.6, 0.8]
# times_pos = [np.abs(beta_gavg[0].times-0.2 - t).argmin() for t in plot_times]

# chan_to_plot = ['POz', 'Cz', 'Oz', 'Pz', 'CPz']

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
#         all_epos.metadata['bin'], bins = pd.qcut(all_epos.metadata[regvar],
#                                                  nbins,
#                                                  labels=False, retbins=True)
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
#         for p in all_epos.metadata['participant'].unique():
#             sub_dat = all_epos[all_epos.metadata['participant'] == p]
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
#         for idx, bin in enumerate([str(i+1) for i in range(nbins)]):

#             line_axis.plot(all_epos[0].times * 1000,
#                            evokeds[bin].data[pick, :] * 1000000,
#                            label=str(idx + 1),
#                            linewidth=2,
#                            color=plt.cm.get_cmap(cmap,
#                                                  nbins)(idx / nbins))

#         line_axis.tick_params(labelsize=12)
#         line_axis.set_xlabel('Time (ms)',
#                              fontdict={'size': param['labelfontsize']})
#         line_axis.set_ylabel('Amplitude (uV)',
#                              fontdict={'size': param['labelfontsize']})
#         line_axis.axhline(0, linestyle='--', color='gray')
#         line_axis.axvline(0, ymin=0,
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

#     bins_topo = list(range(nbins))
#     for idx, binnum in enumerate([str(i+1) for i in bins_topo]):
#         fig, topo_axis = plt.subplots(figsize=(1, 1))
#         tidx = np.argmin(np.abs(evokeds[binnum].times - 0.6))
#         dat = evokeds[binnum].data[:, tidx]*1000000

#         im, _ = plot_topomap(dat,
#                              pos=evokeds[binnum].info,
#                              cmap=cmap,
#                              show=False,
#                              ch_type='eeg',
#                              outlines='head',
#                              vlim=(-vminmax, vminmax),
#                              extrapolate='head',
#                              axes=topo_axis,
#                              sensors=False,
#                              contours=0,)
#         topo_axis.set_title(bina + ' ' + binnum,
#                             fontdict={'size': param['labelfontsize']-1}, pad=0.1)

#         fig.savefig(opj(outfigpath, 'fig_binsamp_topo_'
#                         + regvar + '_bin' + binnum + '.svg'),
#                     dpi=600, bbox_inches='tight')
#         if idx+1 == len(bins_topo):
#             fig, ax = plt.subplots(figsize=(0.2, 1))
#             cbar1 = fig.colorbar(im, cax=ax,
#                                  orientation='vertical', aspect=1)
#             cbar1.set_label('Amplitude (uV)', rotation=270,
#                             labelpad=12, fontdict={'fontsize': param["labelfontsize"]-1})
#             cbar1.ax.tick_params(labelsize=param['ticksfontsize']-2)
#             fig.savefig(opj(outfigpath, 'fig_topo_bins_cbar' + str(ridx) + '.svg'),
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
#         line_axis.set_ylim((-0.02, 0.25))

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
