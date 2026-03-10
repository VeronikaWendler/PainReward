# @ : -*- coding: utf-8 -*-
# @ Author: Michel-Pierre Coll (michel-pierre.coll@psy.ulaval.ca),
#           edited by Veronika Wendler (2025), edited (joint-only) 2026
# @ Date: 2024
# @ Description: plotting the EEG regression models from the massunivariate script
#

import mne
import pandas as pd
import numpy as np
from os.path import join as opj
import matplotlib.pyplot as plt
from bids import BIDSLayout
from mne.viz import plot_topomap
import os
import scipy.stats
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
from pathlib import Path

# ---------------------------------------------------------------------------------------------------
# Directories (MATCH MASSUNIVARIATE)
PROJECT_DIR = Path(os.getenv("PROJECT_DIR", "/workspace"))
basepath    = Path(os.getenv("DATA_DIR", PROJECT_DIR / "EEG" / "PainReward_sub-001-050" / "painrewardeegdata"))

# massunivariate: outpath = OUT_DIR default basepath / "statistics"
OUT_DIR = Path(os.getenv("OUT_DIR", basepath / "statistics"))

layout = BIDSLayout(basepath)

# ---------------------------------------------------------------------------------------------------
# Version + GLM version 
# version = 2 -> decision plotting 
# version = 1 -> passive plotting 
version = 2
glm_version = "z"  # noz / z / partz

# ---------------------------------------------------------------------------------------------------
# Output path selection (MATCH MASSUNIVARIATE)
if version == 1:
    outpath = opj(OUT_DIR, "erps_massuni_passive")
elif version == 2:
    outpath = opj(OUT_DIR, "erps_massuni_decision")
else:
    raise ValueError("No Version")

# Figure output folder: keep consistent + simple
outfigpath = opj(outpath, "figures")
os.makedirs(outfigpath, exist_ok=True)

# ---------------------------------------------------------------------------------------------------
# map glm_version to stats subfolder (MATCH MASSUNIVARIATE naming)
if glm_version == "noz":
    stats_subdir = "NO_Zscoring"
    fig_prefix = "noz_"
elif glm_version == "z":
    stats_subdir = "Zscoring"
    fig_prefix = "z_"
elif glm_version == "partz":
    stats_subdir = "PartZscoring"
    fig_prefix = "partz_"
else:
    raise ValueError(f"Check glm_version: {glm_version}")

outpath_glm = opj(outpath, stats_subdir)

# ---------------------------------------------------------------------------------------------------
# plotting params

regvars = ["painlevel", "moneylevel"]
regvarsnames = ["Painlevel", "Moneylevel"]

param = {
    "alpha": 0.05,
    "titlefontsize": 12,
    "labelfontsize": 12,
    "ticksfontsize": 11,
    "legendfontsize": 10,
    "testresampfreq": 1024,
}

plt.rc("axes.spines", top=False, right=False)
plt.rcParams["font.family"] = "DejaVu Sans"

# ---------------------------------------------------------------------------------------------------
# Regressors (MATCH MASSUNIVARIATE regvars list + filenames)
# For BOTH decision and passive, group-level outputs are saved under these names in your script.
regvars = ["painlevel", "moneylevel"]
regvarsnames = ["Painlevel", "Moneylevel"]

plot_times = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.3, 1.4]
chan_to_plot = ["Fz", "FCz", "POz", "Cz", "CPz", "Pz", "Oz"]

# ---------------------------------------------------------------------------------------------------
# Load group-level stats
tvals = np.load(opj(outpath_glm, "ols_2ndlevel_tvals.npy"))         # (n_reg, n_times, n_chans)
pvals = np.load(opj(outpath_glm, "ols_2ndlevel_pvals_fdr.npy"))     # corrected p-maps

beta_gavg = np.load(opj(outpath_glm, "ols_2ndlevel_betasavg.npy"), allow_pickle=True)  # list of Evoked
allbetas = np.load(opj(outpath_glm, "ols_2ndlevel_betas.npy"), allow_pickle=True)     # (n_subj, n_reg, n_ch, n_t)

times_pos = [np.abs(beta_gavg[0].times - t).argmin() for t in plot_times]

# ---------------------------------------------------------------------------------------------------
# Main plots: per regressor
# (Decision plotting functionality is unchanged; passive works automatically by version path)
for ridx, regvar in enumerate(regvars):

    regvarname = regvarsnames[ridx]

    if regvar == "painlevel":
        cmap = "Blues"
    elif regvar == "moneylevel":
        cmap = "Greens"
    else:
        cmap = "viridis"

    # Load epochs saved by massunivariate (MATCH filenames)
    # NOTE: passive regression script saves these .fif epochs too, so this works for BOTH versions.
    epo_path = opj(outpath_glm, f"ols_2ndlevel_allepochs-epo_{regvar}.fif")
    if not os.path.exists(epo_path):
        raise FileNotFoundError(
            f"Missing epochs file for plotting: {epo_path}\n"
            f"Make sure the massunivariate script saved 'ols_2ndlevel_allepochs-epo_{regvar}.fif'."
        )
    all_epos = mne.read_epochs(epo_path, preload=True)

    beta_ev = beta_gavg[ridx].copy()
    chankeep = np.array([c not in ["M1", "M2"] for c in beta_ev.ch_names])

    # -----------------------------
    # Topo: beta maps at plot_times
    # -----------------------------
    for tidx, timepos in enumerate(times_pos):
        fig, ax = plt.subplots(figsize=(1, 1))

        p_row = pvals[ridx][timepos, :]
        mask = np.zeros_like(p_row, dtype=bool)
        mask[(p_row < param["alpha"]) & chankeep] = True

        im, _ = plot_topomap(
            beta_ev.data[:, timepos],
            pos=beta_ev.info,
            mask=mask,
            mask_params=dict(marker="o", markerfacecolor="w", markeredgecolor="k",
                             linewidth=0, markersize=2),
            cmap=cmap,
            show=False,
            ch_type="eeg",
            outlines="head",
            extrapolate="head",
            vlim=(-0.15, 0.15),
            axes=ax,
            sensors=False,
            contours=0,
        )

        ax.set_title(f"{int(plot_times[tidx] * 1000)} ms",
                     fontdict={"size": param["labelfontsize"] - 1}, pad=0.1)

        # save colorbar once per regressor
        if tidx + 1 == len(plot_times):
            fig2, cax = plt.subplots(figsize=(0.2, 1))
            cbar = fig2.colorbar(im, cax=cax, orientation="vertical", aspect=1)
            cbar.set_label("Beta (z)", rotation=270, labelpad=12,
                           fontdict={"fontsize": param["labelfontsize"] - 1})
            cbar.ax.tick_params(labelsize=param["ticksfontsize"] - 2)
            fig2.savefig(opj(outfigpath, f"{fig_prefix}fig_topo_beta_cbar_{regvar}.svg"),
                         dpi=600, bbox_inches="tight")
            plt.close(fig2)

        fig.savefig(opj(outfigpath, f"{fig_prefix}fig_ols_erps_betas_topo_{regvar}_{tidx}.svg"),
                    dpi=600, bbox_inches="tight")
        plt.close(fig)

    # -----------------------------
    # Binned ERP plots
    # -----------------------------
    for ch in chan_to_plot:
        if ch not in beta_ev.ch_names:
            continue

        fig, ax = plt.subplots(figsize=(4, 2.5))


        all_epos.metadata = all_epos.metadata.reset_index(drop=True)
        level_vals = pd.to_numeric(all_epos.metadata[regvar], errors="coerce")
        unique_levels = np.sort(level_vals.dropna().unique())
        unique_levels = unique_levels[unique_levels > 0]
        level_to_bin = {lev: i for i, lev in enumerate(unique_levels)}
        all_epos.metadata["bin"] = level_vals.map(level_to_bin)
        nbins_eff = len(unique_levels)

        # participant-wise averages then grand average
        sub_evokeds = []
        for p_id in all_epos.metadata["participant_id"].unique():
            sub_dat = all_epos[all_epos.metadata["participant_id"] == p_id]
            sub_evoked = {}
            for b in range(nbins_eff):
                if np.sum(sub_dat.metadata["bin"] == b) != 0:
                    sub_evoked[b] = sub_dat[sub_dat.metadata["bin"] == b].average()
                else:
                    sub_evoked[b] = 0
            sub_evokeds.append(sub_evoked)

        evokeds = {}
        for b in range(nbins_eff):
            evoked_list = [sd[b] for sd in sub_evokeds if sd[b] != 0]
            if len(evoked_list) == 0:
                continue
            evokeds[str(b + 1)] = mne.grand_average(evoked_list)

        pick = beta_ev.ch_names.index(ch)

        ax.set_title(f"{ch} – binned by {regvarname}", fontsize=param["titlefontsize"])
        ax.set_xlabel("Time (ms)", fontsize=param["labelfontsize"])
        ax.set_ylabel("Amplitude (µV)", fontsize=param["labelfontsize"])

        bin_ids = sorted(evokeds.keys(), key=lambda x: int(x))
        for i, bin_id in enumerate(bin_ids):
            actual_level = unique_levels[int(bin_id) - 1]
            ax.plot(
                all_epos[0].times * 1000,
                evokeds[bin_id].data[pick, :] * 1e6,
                linewidth=2,
                label=str(int(actual_level)),
                color=plt.get_cmap(cmap)(i / max(1, len(bin_ids) - 1)),
            )

        ax.axhline(0, linestyle="--", color="gray")
        ax.axvline(0, linestyle="--", color="gray")
        ax.set_xticks(np.arange(-200, 1200, 200))
        ax.set_xticklabels([str(i) for i in np.arange(-200, 1200, 200)])
        ax.tick_params(labelsize=param["ticksfontsize"])
        ax.legend(fontsize=param["legendfontsize"], frameon=False, title="Bin")

        fig.tight_layout()
        fig.savefig(opj(outfigpath, f"{fig_prefix}fig_ols_erps_amp_bins_{regvar}_{ch}.svg"),
                    dpi=600, bbox_inches="tight")
        plt.close(fig)

    # -----------------------------
    # Mean beta + SEM (sig marks)
    # -----------------------------
    for ch in chan_to_plot:
        if ch not in beta_ev.ch_names:
            continue

        fig, ax = plt.subplots(figsize=(4, 2.5))
        pick = beta_ev.ch_names.index(ch)

        sub_avg = np.stack([allbetas[s, ridx, pick, :] for s in range(allbetas.shape[0])])
        sem = scipy.stats.sem(sub_avg, axis=0)
        mean = beta_ev.data[pick, :]

        ax.set_xlabel("Time (ms)", fontsize=param["labelfontsize"])
        ax.set_ylabel(f"β ({regvarname}, z)", fontsize=param["labelfontsize"])

        ax.plot(all_epos[0].times * 1000, mean, linewidth=3)
        ax.fill_between(all_epos[0].times * 1000, mean - sem, mean + sem, alpha=0.3)

        ax.set_ylim((-0.25, 0.25))
        ax.axhline(0, linestyle="--", color="gray")
        ax.axvline(0, linestyle="--", color="gray")

        timestep = 1000.0 / param["testresampfreq"]
        for ti, t_ms in enumerate(all_epos[0].times * 1000):
            if pvals[ridx][ti, pick] < param["alpha"]:
                ax.fill_between([t_ms, t_ms + timestep], -0.02, -0.005, alpha=0.3, facecolor="red")

        ax.set_xticks(np.arange(-200, 1200, 200))
        ax.set_xticklabels([str(i) for i in np.arange(-200, 1200, 200)])
        ax.tick_params(labelsize=param["ticksfontsize"])

        fig.tight_layout()
        fig.savefig(opj(outfigpath, f"{fig_prefix}fig_ols_erps_betas_{regvar}_{ch}.svg"),
                    dpi=600, bbox_inches="tight")
        plt.close(fig)

# ---------------------------------------------------------------------------------------------------
# Difference maps: pain - money

diff_t_path = opj(outpath_glm, "ols_2ndlevel_tval_diff_pain_minus_money.npy")
diff_p_path = opj(outpath_glm, "ols_2ndlevel_pval_fdr_diff_pain_minus_money.npy")

if os.path.exists(diff_t_path) and os.path.exists(diff_p_path):
    tdiff = np.load(diff_t_path)
    pdiff = np.load(diff_p_path)

    times = beta_gavg[0].times
    info = beta_gavg[0].info
    diff_times_pos = [np.abs(times - t).argmin() for t in plot_times]
    chankeep = np.array([c not in ["M1", "M2"] for c in info["ch_names"]])

    for tidx, time_idx in enumerate(diff_times_pos):
        t_ms = int(plot_times[tidx] * 1000)
        p_row = pdiff[time_idx, :]
        mask = (p_row < param["alpha"]) & chankeep

        fig, ax = plt.subplots(figsize=(2, 2))
        im, _ = plot_topomap(
            tdiff[time_idx, :],
            pos=info,
            mask=mask,
            mask_params=dict(marker="o", markerfacecolor="w", markeredgecolor="k",
                             linewidth=0, markersize=3),
            cmap="RdBu_r",
            show=False,
            ch_type="eeg",
            outlines="head",
            extrapolate="head",
            axes=ax,
            sensors=False,
            contours=0,
        )
        ax.set_title(f"pain − money, {t_ms} ms",
                     fontdict={"size": param["labelfontsize"] - 1}, pad=0.1)

        fig2, cax = plt.subplots(figsize=(0.2, 1))
        cbar = fig2.colorbar(im, cax=cax, orientation="vertical", aspect=1)
        cbar.set_label("t (pain − money)", rotation=270, labelpad=12,
                       fontdict={"fontsize": param["labelfontsize"] - 1})
        cbar.ax.tick_params(labelsize=param["ticksfontsize"] - 2)

        fig.savefig(opj(outfigpath, f"{fig_prefix}fig_topo_diff_pain_minus_money_{t_ms}ms.svg"),
                    dpi=600, bbox_inches="tight")
        fig2.savefig(opj(outfigpath, f"{fig_prefix}fig_topo_diff_pain_minus_money_{t_ms}ms_cbar.svg"),
                     dpi=600, bbox_inches="tight")

        plt.close(fig)
        plt.close(fig2)
else:
    print("Difference-map files not found; skipping pain-minus-money topographies:")
    print("  ", diff_t_path)
    print("  ", diff_p_path)

print("Done. Figures saved to:", outfigpath)











# import mne
# import pandas as pd
# import numpy as np
# from os.path import join as opj
# import matplotlib.pyplot as plt
# from bids import BIDSLayout
# from mne.viz import plot_topomap
# import os
# import scipy.stats
# import warnings
# warnings.simplefilter(action='ignore', category=FutureWarning)
# from pathlib import Path

# # ---------------------------------------------------------------------------------------------------
# # Directories (MATCH MASSUNIVARIATE)
# PROJECT_DIR = Path(os.getenv("PROJECT_DIR", "/workspace"))
# basepath    = Path(os.getenv("DATA_DIR", PROJECT_DIR / "EEG" / "PainReward_sub-001-050" / "painrewardeegdata"))

# # massunivariate: outpath = OUT_DIR default basepath / "statistics"
# OUT_DIR = Path(os.getenv("OUT_DIR", basepath / "statistics"))

# layout = BIDSLayout(basepath)

# # ---------------------------------------------------------------------------------------------------
# # Version + GLM version (keep version-sensitive structure)
# version = 2
# glm_version = "z"  # noz / z / partz

# # ---------------------------------------------------------------------------------------------------
# # Output path selection (MATCH MASSUNIVARIATE)
# if version == 1:
#     outpath = opj(OUT_DIR, "erps_massuni_passive_regression")
# elif version == 2:
#     outpath = opj(OUT_DIR, "erps_massuni_regression")
# else:
#     raise ValueError("No Version")

# # Figure output folder: keep consistent + simple
# outfigpath = opj(outpath, "figures")
# os.makedirs(outfigpath, exist_ok=True)

# # ---------------------------------------------------------------------------------------------------
# # map glm_version to stats subfolder (MATCH MASSUNIVARIATE naming)
# if glm_version == "noz":
#     stats_subdir = "NO_Zscoring"
#     fig_prefix = "noz_"
# elif glm_version == "z":
#     stats_subdir = "Zscoring"
#     fig_prefix = "z_"
# elif glm_version == "partz":
#     stats_subdir = "PartZscoring"
#     fig_prefix = "partz_"
# else:
#     raise ValueError(f"Check glm_version: {glm_version}")

# outpath_glm = opj(outpath, stats_subdir)

# # ---------------------------------------------------------------------------------------------------
# # plotting params
# param = {
#     "alpha": 0.05,
#     "titlefontsize": 12,
#     "labelfontsize": 12,
#     "ticksfontsize": 11,
#     "legendfontsize": 10,
#     "testresampfreq": 1024,
# }

# plt.rc("axes.spines", top=False, right=False)
# plt.rcParams["font.family"] = "DejaVu Sans"

# # ---------------------------------------------------------------------------------------------------
# # Regressors (MATCH MASSUNIVARIATE regvars list + filenames)
# regvars = ["painlevel", "moneylevel"]
# regvarsnames = ["Painlevel", "Moneylevel"]

# plot_times = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.3, 1.4]
# chan_to_plot = ["Fz", "FCz", "POz", "Cz", "CPz", "Pz", "Oz"]

# # ---------------------------------------------------------------------------------------------------
# # Load group-level stats (MATCH filenames saved by massunivariate)
# tvals = np.load(opj(outpath_glm, "ols_2ndlevel_tvals.npy"))   # (n_reg, n_times, n_chans)
# pvals = np.load(opj(outpath_glm, "ols_2ndlevel_pvals.npy"))   # (n_reg, n_times, n_chans)

# beta_gavg = np.load(opj(outpath_glm, "ols_2ndlevel_betasavg.npy"), allow_pickle=True)  # list of Evoked
# allbetas = np.load(opj(outpath_glm, "ols_2ndlevel_betas.npy"), allow_pickle=True)     # (n_subj, n_reg, n_ch, n_t)

# times_pos = [np.abs(beta_gavg[0].times - t).argmin() for t in plot_times]

# # ---------------------------------------------------------------------------------------------------
# # Main plots: per regressor
# for ridx, regvar in enumerate(regvars):

#     regvarname = regvarsnames[ridx]

#     if regvar == "painlevel":
#         cmap = "Blues"
#         vminmax = 6
#     elif regvar == "moneylevel":
#         cmap = "Greens"
#         vminmax = 6
#     else:
#         cmap = "viridis"
#         vminmax = 6

#     # Load epochs saved by massunivariate (MATCH filenames)
#     epo_path = opj(outpath_glm, f"ols_2ndlevel_allepochs-epo_{regvar}.fif")
#     all_epos = mne.read_epochs(epo_path, preload=True)

#     beta_ev = beta_gavg[ridx].copy()
#     chankeep = np.array([c not in ["M1", "M2"] for c in beta_ev.ch_names])

#     # -----------------------------
#     # Topo: beta maps at plot_times
#     # -----------------------------
#     for tidx, timepos in enumerate(times_pos):
#         fig, ax = plt.subplots(figsize=(1, 1))

#         p_row = pvals[ridx][timepos, :]
#         mask = np.zeros_like(p_row, dtype=bool)
#         mask[(p_row < param["alpha"]) & chankeep] = True

#         im, _ = plot_topomap(
#             beta_ev.data[:, timepos],
#             pos=beta_ev.info,
#             mask=mask,
#             mask_params=dict(marker="o", markerfacecolor="w", markeredgecolor="k",
#                              linewidth=0, markersize=2),
#             cmap=cmap,
#             show=False,
#             ch_type="eeg",
#             outlines="head",
#             extrapolate="head",
#             vlim=(-0.15, 0.15),
#             axes=ax,
#             sensors=False,
#             contours=0,
#         )

#         ax.set_title(f"{int(plot_times[tidx] * 1000)} ms",
#                      fontdict={"size": param["labelfontsize"] - 1}, pad=0.1)

#         # save colorbar once per regressor
#         if tidx + 1 == len(plot_times):
#             fig2, cax = plt.subplots(figsize=(0.2, 1))
#             cbar = fig2.colorbar(im, cax=cax, orientation="vertical", aspect=1)
#             cbar.set_label("Beta (z)", rotation=270, labelpad=12,
#                            fontdict={"fontsize": param["labelfontsize"] - 1})
#             cbar.ax.tick_params(labelsize=param["ticksfontsize"] - 2)
#             fig2.savefig(opj(outfigpath, f"{fig_prefix}fig_topo_beta_cbar_{regvar}.svg"),
#                          dpi=600, bbox_inches="tight")
#             plt.close(fig2)

#         fig.savefig(opj(outfigpath, f"{fig_prefix}fig_ols_erps_betas_topo_{regvar}_{tidx}.svg"),
#                     dpi=600, bbox_inches="tight")
#         plt.close(fig)

#     # -----------------------------
#     # Binned ERP plots
#     # -----------------------------
#     for ch in chan_to_plot:
#         if ch not in beta_ev.ch_names:
#             continue

#         fig, ax = plt.subplots(figsize=(4, 2.5))

#         nbins = 5
#         all_epos.metadata = all_epos.metadata.reset_index(drop=True)
#         all_epos.metadata["bin"] = 0

#         unique_vals = all_epos.metadata[regvar].nunique()
#         nbins_eff = min(nbins, unique_vals)

#         all_epos.metadata["bin"], bins = pd.qcut(
#             all_epos.metadata[regvar],
#             q=nbins_eff,
#             labels=False,
#             retbins=True,
#             duplicates="drop"
#         )

#         # participant-wise averages then grand average
#         sub_evokeds = []
#         for p_id in all_epos.metadata["participant_id"].unique():
#             sub_dat = all_epos[all_epos.metadata["participant_id"] == p_id]
#             sub_evoked = {}
#             for b in range(nbins_eff):
#                 if np.sum(sub_dat.metadata["bin"] == b) != 0:
#                     sub_evoked[b] = sub_dat[sub_dat.metadata["bin"] == b].average()
#                 else:
#                     sub_evoked[b] = 0
#             sub_evokeds.append(sub_evoked)

#         evokeds = {}
#         for b in range(nbins_eff):
#             evoked_list = [sd[b] for sd in sub_evokeds if sd[b] != 0]
#             if len(evoked_list) == 0:
#                 continue
#             evokeds[str(b + 1)] = mne.grand_average(evoked_list)

#         pick = beta_ev.ch_names.index(ch)

#         ax.set_title(f"{ch} – binned by {regvarname}", fontsize=param["titlefontsize"])
#         ax.set_xlabel("Time (ms)", fontsize=param["labelfontsize"])
#         ax.set_ylabel("Amplitude (µV)", fontsize=param["labelfontsize"])

#         bin_ids = sorted(evokeds.keys(), key=lambda x: int(x))
#         for i, bin_id in enumerate(bin_ids):
#             ax.plot(
#                 all_epos[0].times * 1000,
#                 evokeds[bin_id].data[pick, :] * 1e6,
#                 linewidth=2,
#                 label=str(i + 1),
#                 color=plt.get_cmap(cmap)(i / max(1, len(bin_ids) - 1)),
#             )

#         ax.axhline(0, linestyle="--", color="gray")
#         ax.axvline(0, linestyle="--", color="gray")
#         ax.set_xticks(np.arange(-200, 1200, 200))
#         ax.set_xticklabels([str(i) for i in np.arange(-200, 1200, 200)])
#         ax.tick_params(labelsize=param["ticksfontsize"])
#         ax.legend(fontsize=param["legendfontsize"], frameon=False, title="Bin")

#         fig.tight_layout()
#         fig.savefig(opj(outfigpath, f"{fig_prefix}fig_ols_erps_amp_bins_{regvar}_{ch}.svg"),
#                     dpi=600, bbox_inches="tight")
#         plt.close(fig)

#     # -----------------------------
#     # Mean beta + SEM (sig marks)
#     # -----------------------------
#     for ch in chan_to_plot:
#         if ch not in beta_ev.ch_names:
#             continue

#         fig, ax = plt.subplots(figsize=(4, 2.5))
#         pick = beta_ev.ch_names.index(ch)

#         sub_avg = np.stack([allbetas[s, ridx, pick, :] for s in range(allbetas.shape[0])])
#         sem = scipy.stats.sem(sub_avg, axis=0)
#         mean = beta_ev.data[pick, :]

#         ax.set_xlabel("Time (ms)", fontsize=param["labelfontsize"])
#         ax.set_ylabel(f"β ({regvarname}, z)", fontsize=param["labelfontsize"])

#         ax.plot(all_epos[0].times * 1000, mean, linewidth=3)
#         ax.fill_between(all_epos[0].times * 1000, mean - sem, mean + sem, alpha=0.3)

#         ax.set_ylim((-0.25, 0.25))
#         ax.axhline(0, linestyle="--", color="gray")
#         ax.axvline(0, linestyle="--", color="gray")

#         timestep = 1000.0 / param["testresampfreq"]
#         for ti, t_ms in enumerate(all_epos[0].times * 1000):
#             if pvals[ridx][ti, pick] < param["alpha"]:
#                 ax.fill_between([t_ms, t_ms + timestep], -0.02, -0.005, alpha=0.3, facecolor="red")

#         ax.set_xticks(np.arange(-200, 1200, 200))
#         ax.set_xticklabels([str(i) for i in np.arange(-200, 1200, 200)])
#         ax.tick_params(labelsize=param["ticksfontsize"])

#         fig.tight_layout()
#         fig.savefig(opj(outfigpath, f"{fig_prefix}fig_ols_erps_betas_{regvar}_{ch}.svg"),
#                     dpi=600, bbox_inches="tight")
#         plt.close(fig)

# # ---------------------------------------------------------------------------------------------------
# # Difference maps: pain - money (MATCH exact saved filenames)
# diff_t_path = opj(outpath_glm, "ols_2ndlevel_tval_diff_pain_minus_money.npy")
# diff_p_path = opj(outpath_glm, "ols_2ndlevel_pval_diff_pain_minus_money.npy")

# tdiff = np.load(diff_t_path)
# pdiff = np.load(diff_p_path)

# times = beta_gavg[0].times
# info = beta_gavg[0].info
# diff_times_pos = [np.abs(times - t).argmin() for t in plot_times]
# chankeep = np.array([c not in ["M1", "M2"] for c in info["ch_names"]])

# for tidx, time_idx in enumerate(diff_times_pos):
#     t_ms = int(plot_times[tidx] * 1000)
#     p_row = pdiff[time_idx, :]
#     mask = (p_row < param["alpha"]) & chankeep

#     fig, ax = plt.subplots(figsize=(2, 2))
#     im, _ = plot_topomap(
#         tdiff[time_idx, :],
#         pos=info,
#         mask=mask,
#         mask_params=dict(marker="o", markerfacecolor="w", markeredgecolor="k",
#                          linewidth=0, markersize=3),
#         cmap="RdBu_r",
#         show=False,
#         ch_type="eeg",
#         outlines="head",
#         extrapolate="head",
#         axes=ax,
#         sensors=False,
#         contours=0,
#     )
#     ax.set_title(f"pain − money, {t_ms} ms",
#                  fontdict={"size": param["labelfontsize"] - 1}, pad=0.1)

#     fig2, cax = plt.subplots(figsize=(0.2, 1))
#     cbar = fig2.colorbar(im, cax=cax, orientation="vertical", aspect=1)
#     cbar.set_label("t (pain − money)", rotation=270, labelpad=12,
#                    fontdict={"fontsize": param["labelfontsize"] - 1})
#     cbar.ax.tick_params(labelsize=param["ticksfontsize"] - 2)

#     fig.savefig(opj(outfigpath, f"{fig_prefix}fig_topo_diff_pain_minus_money_{t_ms}ms.svg"),
#                 dpi=600, bbox_inches="tight")
#     fig2.savefig(opj(outfigpath, f"{fig_prefix}fig_topo_diff_pain_minus_money_{t_ms}ms_cbar.svg"),
#                  dpi=600, bbox_inches="tight")

#     plt.close(fig)
#     plt.close(fig2)

# print("Done. Figures saved to:", outfigpath)