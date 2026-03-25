# @ : -*- coding: utf-8 -*-
# plotting script for massunivariate joint model (painlevel, moneylevel) for versions 1–3.

import os
from pathlib import Path
import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt
from mne.viz import plot_topomap
import scipy.stats
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

# -----------------------------
# SETTINGS
# -----------------------------
version = 2          # 
glm_version = "z"    # "z", "noz", "partz"  

# regressors in JOINT model 
regvars = ["painlevel", "moneylevel"]
regvarsnames = ["Pain", "Money"]

plot_times = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.3, 1.4]
chan_to_plot = ["Fz", "FCz", "POz", "Cz", "CPz", "Pz", "Oz"]

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

# -----------------------------
# PATHS (MATCH your massuni)
# -----------------------------
PROJECT_DIR = Path(os.getenv("PROJECT_DIR", "/workspace"))
basepath = Path(os.getenv("DATA_DIR", PROJECT_DIR / "EEG" / "PainReward_sub-001-050" / "painrewardeegdata"))
OUT_DIR = Path(os.getenv("OUT_DIR", basepath / "statistics"))

VERSION_TO_FOLDER = {
    1: "erps_massuni_drift_mod_9_passive",  
    2: "erps_massuni_regression",           
    3: "erps_massuni_drift_mod_9_RT_3GLMs", 
}

if version not in VERSION_TO_FOLDER:
    raise ValueError(f"version must be one of {sorted(VERSION_TO_FOLDER.keys())}")

version_folder = VERSION_TO_FOLDER[version]
outpath = OUT_DIR / version_folder

# glm_version subfolder
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
    raise ValueError("glm_version must be 'z', 'noz', or 'partz'")

stats_dir = outpath / stats_subdir
if not stats_dir.exists():
    raise FileNotFoundError(f"Stats dir not found: {stats_dir}")

# Where to put figures
outfigpath = outpath / f"figures_joint_{stats_subdir}"
outfigpath.mkdir(parents=True, exist_ok=True)

print(f"\nRunning plotting for:")
print(f"  version         = {version}")
print(f"  version_folder  = {version_folder}")
print(f"  glm_version     = {glm_version}")
print(f"  stats_dir       = {stats_dir}")
print(f"  outfigpath      = {outfigpath}\n")

# -----------------------------
# Load group outputs
# -----------------------------
tvals_path = stats_dir / "ols_2ndlevel_tvals.npy"
pvals_path = stats_dir / "ols_2ndlevel_pvals.npy"
betas_path = stats_dir / "ols_2ndlevel_betas.npy"
betasavg_path = stats_dir / "ols_2ndlevel_betasavg.npy"

for fp in [tvals_path, pvals_path, betas_path, betasavg_path]:
    if not fp.exists():
        raise FileNotFoundError(f"Missing required file: {fp}")

tvals = np.load(tvals_path)  # (n_reg, n_times, n_ch)
pvals = np.load(pvals_path)  # (n_reg, n_times, n_ch)
allbetas = np.load(betas_path, allow_pickle=True)  # (n_subj, n_reg, n_ch, n_times)
beta_gavg = np.load(betasavg_path, allow_pickle=True)  # list of Evoked

# sanity check regressor count
if tvals.shape[0] != len(regvars):
    raise RuntimeError(
        f"Expected {len(regvars)} regressors in saved tvals, found {tvals.shape[0]}.\n"
        f"Your plotting regvars={regvars} must match the massuni save order."
    )

times = beta_gavg[0].times
times_pos = [np.abs(times - t).argmin() for t in plot_times]

# -----------------------------
# Per-regressor plots
# -----------------------------
for ridx, regvar in enumerate(regvars):
    regname = regvarsnames[ridx]

    if regvar == "painlevel":
        cmap = "Blues"
        vminmax_bins = 6
    elif regvar == "moneylevel":
        cmap = "Greens"
        vminmax_bins = 6
    else:
        cmap = "viridis"
        vminmax_bins = 6

    epo_path = stats_dir / f"ols_2ndlevel_allepochs-epo_{regvar}.fif"
    if not epo_path.exists():
        raise FileNotFoundError(f"Missing epochs file: {epo_path}")

    all_epos = mne.read_epochs(epo_path, preload=True)

    # exclude mastoids in significance masks
    chankeep = np.array([c not in ["M1", "M2"] for c in beta_gavg[ridx].ch_names])

    # ---- Topomaps of beta ----
    last_im = None
    for tidx, timepos in enumerate(times_pos):
        fig, ax = plt.subplots(figsize=(1.2, 1.2))

        p_row = pvals[ridx][timepos, :]
        mask = np.zeros_like(p_row, dtype=bool)
        mask[(p_row < param["alpha"]) & chankeep] = True

        im, _ = plot_topomap(
            beta_gavg[ridx].data[:, timepos],
            pos=beta_gavg[ridx].info,
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
        last_im = im

        ax.set_title(f"{regname} β, {int(plot_times[tidx]*1000)} ms",
                     fontdict={"size": param["labelfontsize"]-1}, pad=0.1)

        fig.savefig(outfigpath / f"{fig_prefix}fig_topo_beta_{regvar}_{tidx}.svg",
                    dpi=600, bbox_inches="tight")
        plt.close(fig)

    # one colorbar per regressor
    if last_im is not None:
        fig2, cax = plt.subplots(figsize=(0.25, 1.2))
        cbar = fig2.colorbar(last_im, cax=cax, orientation="vertical", aspect=1)
        cbar.set_label("Beta (z)", rotation=270, labelpad=12,
                       fontdict={"fontsize": param["labelfontsize"]-1})
        cbar.ax.tick_params(labelsize=param["ticksfontsize"]-2)
        fig2.savefig(outfigpath / f"{fig_prefix}fig_topo_beta_cbar_{regvar}.svg",
                     dpi=600, bbox_inches="tight")
        plt.close(fig2)

    # ---- ERP binned-by-regressor ----
    for ch in chan_to_plot:
        if ch not in all_epos.ch_names:
            print(f"[{regvar}] Channel {ch} not found; skipping ERP bin plot.")
            continue

        fig, ax = plt.subplots(figsize=(4, 2.5))
        all_epos.metadata = all_epos.metadata.reset_index(drop=True)

        nbins = 5
        unique_vals = all_epos.metadata[regvar].nunique()
        nbins_eff = min(nbins, unique_vals)

        all_epos.metadata["bin"], _bins = pd.qcut(
            all_epos.metadata[regvar],
            q=nbins_eff,
            labels=False,
            retbins=True,
            duplicates="drop",
        )

        # within-subject averages per bin
        sub_evokeds = []
        for p_id in all_epos.metadata["participant_id"].unique():
            sub_dat = all_epos[all_epos.metadata["participant_id"] == p_id]
            sub_dict = {}
            for b in range(nbins_eff):
                if np.sum(sub_dat.metadata["bin"] == b) > 0:
                    sub_dict[b] = sub_dat[sub_dat.metadata["bin"] == b].average()
                else:
                    sub_dict[b] = None
            sub_evokeds.append(sub_dict)

        # grand-average across participants
        evokeds = {}
        for b in range(nbins_eff):
            ev_list = [sd[b] for sd in sub_evokeds if sd[b] is not None]
            if len(ev_list) > 0:
                evokeds[str(b+1)] = mne.grand_average(ev_list)

        pick = all_epos.ch_names.index(ch)
        bin_ids = sorted(evokeds.keys(), key=lambda x: int(x))

        for i, bin_id in enumerate(bin_ids):
            ax.plot(
                all_epos.times * 1000,
                evokeds[bin_id].data[pick, :] * 1e6,
                linewidth=2,
                label=f"Bin {bin_id}",
                color=plt.get_cmap(cmap)(i / max(1, len(bin_ids)-1)),
            )

        ax.axhline(0, linestyle="--", color="gray")
        ax.axvline(0, linestyle="--", color="gray")
        ax.set_xlabel("Time (ms)", fontdict={"size": param["labelfontsize"]})
        ax.set_ylabel("Amplitude (µV)", fontdict={"size": param["labelfontsize"]})
        ax.set_title(f"{ch} – binned by {regname}", fontdict={"size": param["titlefontsize"]})
        ax.set_xticks(np.arange(-200, 1200, 200))
        ax.tick_params(labelsize=param["ticksfontsize"])
        ax.legend(fontsize=param["legendfontsize"], frameon=False)

        fig.tight_layout()
        fig.savefig(outfigpath / f"{fig_prefix}fig_erp_bins_{regvar}_{ch}.svg",
                    dpi=600, bbox_inches="tight")
        plt.close(fig)

    # ---- Mean beta ± SEM + sig markers ----
    for ch in chan_to_plot:
        if ch not in beta_gavg[ridx].ch_names:
            print(f"[{regvar}] Channel {ch} not found in beta_gavg; skipping beta timecourse.")
            continue

        fig, ax = plt.subplots(figsize=(4, 2.5))
        pick = beta_gavg[ridx].ch_names.index(ch)

        sub_avg = allbetas[:, ridx, pick, :]  # (n_subj, n_times)
        sem = scipy.stats.sem(sub_avg, axis=0)
        mean = beta_gavg[ridx].data[pick, :]

        ax.plot(times * 1000, mean, linewidth=3)
        ax.fill_between(times * 1000, mean - sem, mean + sem, alpha=0.3)

        timestep = 1000.0 / param["testresampfreq"]
        for ti, t_ms in enumerate(times * 1000):
            if pvals[ridx][ti, pick] < param["alpha"]:
                ax.fill_between([t_ms, t_ms + timestep], -0.02, -0.005, alpha=0.3, facecolor="red")

        ax.axhline(0, linestyle="--", color="gray")
        ax.axvline(0, linestyle="--", color="gray")
        ax.set_ylim((-0.25, 0.25))
        ax.set_xlabel("Time (ms)", fontdict={"size": param["labelfontsize"]})
        ax.set_ylabel(f"β ({regname}, z)", fontdict={"size": param["labelfontsize"]})
        ax.set_xticks(np.arange(-200, 1200, 200))
        ax.tick_params(labelsize=param["ticksfontsize"])

        fig.tight_layout()
        fig.savefig(outfigpath / f"{fig_prefix}fig_beta_timecourse_{regvar}_{ch}.svg",
                    dpi=600, bbox_inches="tight")
        plt.close(fig)

# -----------------------------
# Diff plots: pain - money
# -----------------------------
diff_label = "pain_minus_money"
tdiff_path = stats_dir / f"ols_2ndlevel_tval_diff_{diff_label}.npy"
pdiff_path = stats_dir / f"ols_2ndlevel_pval_diff_{diff_label}.npy"

if tdiff_path.exists() and pdiff_path.exists():
    tdiff = np.load(tdiff_path)
    pdiff = np.load(pdiff_path)

    info = beta_gavg[0].info
    chankeep = np.array([c not in ["M1", "M2"] for c in info["ch_names"]])
    diff_times_pos = [np.abs(times - t).argmin() for t in plot_times]

    last_im = None
    for tidx, time_idx in enumerate(diff_times_pos):
        t_ms = int(plot_times[tidx] * 1000)
        p_row = pdiff[time_idx, :]
        mask = (p_row < param["alpha"]) & chankeep

        fig, ax = plt.subplots(figsize=(2.2, 2.2))
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
        last_im = im
        ax.set_title(f"t (pain − money), {t_ms} ms",
                     fontdict={"size": param["labelfontsize"]-1}, pad=0.1)

        fig.savefig(outfigpath / f"{fig_prefix}fig_topo_diff_{diff_label}_{t_ms}ms.svg",
                    dpi=600, bbox_inches="tight")
        plt.close(fig)

    if last_im is not None:
        fig2, cax = plt.subplots(figsize=(0.25, 1.2))
        cbar = fig2.colorbar(last_im, cax=cax, orientation="vertical", aspect=1)
        cbar.set_label("t (pain − money)", rotation=270, labelpad=12,
                       fontdict={"fontsize": param["labelfontsize"]-1})
        cbar.ax.tick_params(labelsize=param["ticksfontsize"]-2)
        fig2.savefig(outfigpath / f"{fig_prefix}fig_topo_diff_{diff_label}_cbar.svg",
                     dpi=600, bbox_inches="tight")
        plt.close(fig2)
else:
    print(f"\n[INFO] Diff files not found (skipping pain-money diff):")
    print(f"  {tdiff_path}")
    print(f"  {pdiff_path}")

print(f"\nDone. Figures saved in:\n{outfigpath}\n")