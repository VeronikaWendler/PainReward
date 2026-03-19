# @ : -*- coding: utf-8 -*-
# @ Author: Michel-Pierre Coll (michel-pierre.coll@psy.ulaval.ca),
#           edited by Veronika Wendler (2025), edited (joint-only) 2026
# @ Date: 2024
# @ Description: plotting the EEG regression models from the massunivariate script
#

import os
import warnings
from pathlib import Path
from os.path import join as opj

import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
from bids import BIDSLayout
from mne.viz import plot_topomap

warnings.simplefilter(action="ignore", category=FutureWarning)

# ---------------------------------------------------------------------------------------------------
# Directories (MATCH MASSUNIVARIATE)
PROJECT_DIR = Path(os.getenv("PROJECT_DIR", "/workspace"))
basepath = Path(os.getenv("DATA_DIR", PROJECT_DIR / "EEG" / "PainReward_sub-001-050" / "painrewardeegdata"))

# massunivariate: outpath = OUT_DIR default basepath / "statistics"
OUT_DIR = Path(os.getenv("OUT_DIR", basepath / "statistics"))

layout = BIDSLayout(basepath)

# ---------------------------------------------------------------------------------------------------
# Version
# version = 2 -> decision plotting
# version = 1 -> passive plotting
version = 1

# Optional preference if both TFCE and cluster outputs exist
# set to "tfce", "cluster", or None
preferred_inference = "tfce"

# Kept only for figure naming compatibility
glm_version = "z"

# ---------------------------------------------------------------------------------------------------
# Output path selection (MATCH MASSUNIVARIATE)
if version == 1:
    outpath = Path(OUT_DIR) / "erps_massuni_passive"
elif version == 2:
    outpath = Path(OUT_DIR) / "erps_massuni_decision"
else:
    raise ValueError("No Version")

# Figure output folder
outfigpath = outpath / "figures"
outfigpath.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------------------------------
# Helpers

def detect_stats_dir(outpath, preferred_inference=None):
    """
    Detect the correct stats folder produced by the massunivariate script.

    New naming:
        Zscoring_tfce
        Zscoring_cluster

    Legacy fallback:
        Zscoring
    """
    candidates = {
        "tfce": Path(outpath) / "Zscoring_tfce",
        "cluster": Path(outpath) / "Zscoring_cluster",
        "legacy": Path(outpath) / "Zscoring",
    }

    if preferred_inference in ["tfce", "cluster"]:
        pref_path = candidates[preferred_inference]
        if pref_path.exists():
            return pref_path, preferred_inference

    if candidates["tfce"].exists() and not candidates["cluster"].exists():
        return candidates["tfce"], "tfce"

    if candidates["cluster"].exists() and not candidates["tfce"].exists():
        return candidates["cluster"], "cluster"

    if candidates["tfce"].exists() and candidates["cluster"].exists():
        # default preference if both exist and preferred_inference=None or invalid
        return candidates["tfce"], "tfce"

    if candidates["legacy"].exists():
        return candidates["legacy"], "legacy"

    raise FileNotFoundError(
        f"Could not find a stats folder in {outpath}.\n"
        f"Expected one of:\n"
        f"  {candidates['tfce']}\n"
        f"  {candidates['cluster']}\n"
        f"  {candidates['legacy']}"
    )


def load_first_existing(path_list, allow_pickle=False):
    """
    Load the first existing file from a list of candidate paths.
    """
    for p in path_list:
        if os.path.exists(p):
            print(f"Loading: {p}")
            return np.load(p, allow_pickle=allow_pickle)
    raise FileNotFoundError("None of these files exist:\n" + "\n".join([str(p) for p in path_list]))


def get_existing_path(path_list):
    """
    Return the first existing path from a list, otherwise None.
    """
    for p in path_list:
        if os.path.exists(p):
            return p
    return None


def get_bin_colors(cmap_name, n_bins, minval=0.25, maxval=0.95):
    cmap = plt.get_cmap(cmap_name)
    if n_bins == 1:
        return [cmap(0.7)]
    return [cmap(x) for x in np.linspace(minval, maxval, n_bins)]


def significance_label(inference_method):
    if inference_method == "tfce":
        return "TFCE-corrected p < .05"
    elif inference_method == "cluster":
        return "Cluster-corrected p < .05"
    else:
        return "Corrected p < .05"


# ---------------------------------------------------------------------------------------------------
# Detect stats directory / inference mode

outpath_glm, inference_method = detect_stats_dir(outpath, preferred_inference=preferred_inference)

if glm_version == "noz":
    fig_prefix = f"noz_{inference_method}_"
elif glm_version == "z":
    fig_prefix = f"z_{inference_method}_"
elif glm_version == "partz":
    fig_prefix = f"partz_{inference_method}_"
else:
    fig_prefix = f"{inference_method}_"

print("Using stats folder:", outpath_glm)
print("Detected inference method:", inference_method)

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

plot_times = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.3, 1.4]
chan_to_plot = ["Fz", "FCz", "POz", "Cz", "CPz", "Pz", "Oz"]

# ---------------------------------------------------------------------------------------------------
# Load group-level stats

tvals = np.load(opj(outpath_glm, "ols_2ndlevel_tvals.npy"))  # (n_reg, n_times, n_chans)

# New massunivariate script writes corrected pointwise p maps under these names
pvals = load_first_existing([
    opj(outpath_glm, "ols_2ndlevel_pvals_corr.npy"),
    opj(outpath_glm, "ols_2ndlevel_pvals.npy"),
    opj(outpath_glm, "ols_2ndlevel_pvals_fdr.npy"),  # legacy fallback
])

beta_gavg = np.load(opj(outpath_glm, "ols_2ndlevel_betasavg.npy"), allow_pickle=True)  # list of Evoked
allbetas = np.load(opj(outpath_glm, "ols_2ndlevel_betas.npy"), allow_pickle=True)       # (n_subj, n_reg, n_ch, n_t)

times_pos = [np.abs(beta_gavg[0].times - t).argmin() for t in plot_times]

# ---------------------------------------------------------------------------------------------------
# Main plots: per regressor

for ridx, regvar in enumerate(regvars):

    regvarname = regvarsnames[ridx]

    if regvar == "painlevel":
        cmap = "Reds"
    elif regvar == "moneylevel":
        cmap = "Blues"
    else:
        cmap = "viridis"

    # Load epochs saved by massunivariate
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
            mask_params=dict(
                marker="o",
                markerfacecolor="w",
                markeredgecolor="k",
                linewidth=0,
                markersize=2,
            ),
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

        ax.set_title(
            f"{int(plot_times[tidx] * 1000)} ms\n({inference_method.upper()})",
            fontdict={"size": param["labelfontsize"] - 1},
            pad=0.1,
        )

        # save colorbar once per regressor
        if tidx + 1 == len(plot_times):
            fig2, cax = plt.subplots(figsize=(0.2, 1))
            cbar = fig2.colorbar(im, cax=cax, orientation="vertical", aspect=1)
            cbar.set_label(
                "Beta (z)",
                rotation=270,
                labelpad=12,
                fontdict={"fontsize": param["labelfontsize"] - 1},
            )
            cbar.ax.tick_params(labelsize=param["ticksfontsize"] - 2)
            fig2.savefig(
                opj(outfigpath, f"{fig_prefix}fig_topo_beta_cbar_{regvar}.svg"),
                dpi=600,
                bbox_inches="tight",
            )
            plt.close(fig2)

        fig.savefig(
            opj(outfigpath, f"{fig_prefix}fig_ols_erps_betas_topo_{regvar}_{tidx}.svg"),
            dpi=600,
            bbox_inches="tight",
        )
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
        bin_colors = get_bin_colors(cmap, len(bin_ids), minval=0.25, maxval=0.95)

        for i, bin_id in enumerate(bin_ids):
            actual_level = unique_levels[int(bin_id) - 1]
            ax.plot(
                all_epos[0].times * 1000,
                evokeds[bin_id].data[pick, :] * 1e6,
                linewidth=2,
                label=str(int(actual_level)),
                color=bin_colors[i],
            )

        ax.axhline(0, linestyle="--", color="gray")
        ax.axvline(0, linestyle="--", color="gray")
        ax.set_xticks(np.arange(-200, 1200, 200))
        ax.set_xticklabels([str(i) for i in np.arange(-200, 1200, 200)])
        ax.tick_params(labelsize=param["ticksfontsize"])

        ax.legend(
            fontsize=8,
            title="Bin",
            title_fontsize=9,
            frameon=False,
            loc="upper left",
            bbox_to_anchor=(0.02, 0.98),
            borderaxespad=0.0,
            handlelength=1.6,
            labelspacing=0.3,
        )

        fig.tight_layout()
        fig.savefig(
            opj(outfigpath, f"{fig_prefix}fig_ols_erps_amp_bins_{regvar}_{ch}.svg"),
            dpi=600,
            bbox_inches="tight",
        )
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
        sig_ymin, sig_ymax = -0.02, -0.005

        for ti, t_ms in enumerate(all_epos[0].times * 1000):
            if pvals[ridx][ti, pick] < param["alpha"]:
                ax.fill_between(
                    [t_ms, t_ms + timestep],
                    sig_ymin,
                    sig_ymax,
                    alpha=0.3,
                    facecolor="red",
                )

        ax.text(
            0.99,
            0.02,
            significance_label(inference_method),
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            alpha=0.8,
        )

        ax.set_xticks(np.arange(-200, 1200, 200))
        ax.set_xticklabels([str(i) for i in np.arange(-200, 1200, 200)])
        ax.tick_params(labelsize=param["ticksfontsize"])

        fig.tight_layout()
        fig.savefig(
            opj(outfigpath, f"{fig_prefix}fig_ols_erps_betas_{regvar}_{ch}.svg"),
            dpi=600,
            bbox_inches="tight",
        )
        plt.close(fig)

# ---------------------------------------------------------------------------------------------------
# Difference maps: pain - money

diff_t_path = opj(outpath_glm, "ols_2ndlevel_tval_diff_pain_minus_money.npy")

diff_p_path = get_existing_path([
    opj(outpath_glm, "ols_2ndlevel_pval_diff_pain_minus_money.npy"),
    opj(outpath_glm, "ols_2ndlevel_pval_corr_diff_pain_minus_money.npy"),
    opj(outpath_glm, "ols_2ndlevel_pval_fdr_diff_pain_minus_money.npy"),  # legacy fallback
])

if os.path.exists(diff_t_path) and (diff_p_path is not None):
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
            mask_params=dict(
                marker="o",
                markerfacecolor="w",
                markeredgecolor="k",
                linewidth=0,
                markersize=3,
            ),
            cmap="RdBu_r",
            show=False,
            ch_type="eeg",
            outlines="head",
            extrapolate="head",
            axes=ax,
            sensors=False,
            contours=0,
        )
        ax.set_title(
            f"pain − money, {t_ms} ms\n({inference_method.upper()})",
            fontdict={"size": param["labelfontsize"] - 1},
            pad=0.1,
        )

        fig2, cax = plt.subplots(figsize=(0.2, 1))
        cbar = fig2.colorbar(im, cax=cax, orientation="vertical", aspect=1)
        cbar.set_label(
            "t (pain − money)",
            rotation=270,
            labelpad=12,
            fontdict={"fontsize": param["labelfontsize"] - 1},
        )
        cbar.ax.tick_params(labelsize=param["ticksfontsize"] - 2)

        fig.savefig(
            opj(outfigpath, f"{fig_prefix}fig_topo_diff_pain_minus_money_{t_ms}ms.svg"),
            dpi=600,
            bbox_inches="tight",
        )
        fig2.savefig(
            opj(outfigpath, f"{fig_prefix}fig_topo_diff_pain_minus_money_{t_ms}ms_cbar.svg"),
            dpi=600,
            bbox_inches="tight",
        )

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