# @ : -*- coding: utf-8 -*-
# @ Author: Michel-Pierre Coll (michel-pierre.coll@psy.ulaval.ca) & Veronika Wendler (2025)
# @ Date: 2024
# @ Description: Decision phase EEG mass-univariate analysis — end-to-end pipeline
#
# Sections:
#   1. Per-subject first-level regression (painlevel / moneylevel, RT-controlled)
#   2. Second-level permutation inference (TFCE or cluster-mass)
#   3. Stats summary  → CSV reports of significant windows & peak stats
#   4. Figures        → topomaps, binned ERPs, beta+SEM time-courses
#
# NOTE: The trial-matching logic in Section 1 has a known bug — see the
# comment at the `matching =` line.  Section 1 cannot run until that is fixed.
# Sections 2-4 can be run standalone if output files already exist in z_dir.

import os
import warnings
from pathlib import Path
from os.path import join as opj

import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
from mne.decoding import Scaler
from mne.stats import spatio_temporal_cluster_1samp_test as st_clust_1s_ttest
from mne.viz import plot_topomap
from scipy import stats

warnings.simplefilter(action="ignore", category=FutureWarning)

# ===========================================================
# Paths
# basepath = BIDS data root (contains sub-* folders, participants.tsv)
# Override with the `basepath` env var, matching 01_behav.py convention.
# HDDM_DIR must point to the HDDM output directory.
# ===========================================================
basepath = Path(os.getenv("basepath", Path(__file__).parent.parent.parent))
HDDM_DIR = Path(os.getenv("HDDM_DIR", basepath / "derivatives" / "hddm"))


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


outroot = basepath / "derivatives" / "statistics"
ensure_dir(outroot)

outpath = outroot / "erps_massuni_decision"
ensure_dir(outpath)

# ===========================================================
# Params
# ===========================================================
param = {
    # --- computation ---
    "njobs": int(os.getenv("NJOBS", max(1, (os.cpu_count() or 4) - 1))),
    "nperms": 5000,
    "random_state": 23,
    "testresampfreq": 250,

    # Inference mode: "tfce" or "cluster"
    "inference_method": "tfce",

    # Classic cluster only: cluster-forming p-threshold (two-sided → t-threshold)
    "cluster_forming_p": 0.01,

    # TFCE only
    "tfce_start": 0.0,
    "tfce_step": 0.2,

    # Across-map correction
    "map_alpha": 0.05,
    "map_correction": "holm",   # "holm", "bonferroni", or "none"
    "point_alpha": 0.05,

    # --- figures ---
    "titlefontsize": 12,
    "labelfontsize": 12,
    "ticksfontsize": 11,
    "legendfontsize": 10,
}

z_dir = outpath / f"Zscoring_{param['inference_method'].lower()}"
ensure_dir(z_dir)

outfigpath = outpath / "figures"
ensure_dir(outfigpath)

fig_prefix = f"z_{param['inference_method']}_"

regvars = ["painlevel", "moneylevel"]
regvarsnames = ["Painlevel", "Moneylevel"]

plot_times = [-0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
chan_to_plot = ["Fz", "FCz", "POz", "Cz", "CPz", "Pz", "Oz"]

plt.rc("axes.spines", top=False, right=False)
plt.rcParams["font.family"] = "DejaVu Sans"

# ===========================================================
# Participants — EEG ∩ HDDM
# ===========================================================
part_csv = basepath / "participants.tsv"
part_eeg = pd.read_csv(part_csv, sep="\t")["participant_id"].unique().tolist()
part_eeg.sort()


def get_common_subjects_eeg_hddm(participants_eeg):
    if not HDDM_DIR.exists():
        raise RuntimeError(
            f"HDDM_DIR does not exist: {HDDM_DIR}\n"
            "Set the HDDM_DIR env var to the HDDM output directory."
        )
    mod_data_path = (
        HDDM_DIR / "figures" / "painreward_behavioural_data_mod_9" /
        "diagnostics" / "v_pain_money.csv"
    )
    if not mod_data_path.exists():
        raise RuntimeError(f"Missing HDDM decision file: {mod_data_path}")
    mod_data = pd.read_csv(mod_data_path, sep=None, engine="python")
    hddm_subjects = sorted(mod_data["participant"].unique().tolist())
    common = sorted(list(set(participants_eeg) & set(hddm_subjects)))
    return common, mod_data


common_participants, mod_data_decision = get_common_subjects_eeg_hddm(part_eeg)
print("\nSubjects in EEG ∩ HDDM:", common_participants)
print("N =", len(common_participants))

# ===========================================================
# Inference helpers
# ===========================================================
def compute_threshold(n_samples, param):
    method = param["inference_method"].lower()
    if method == "tfce":
        return {"start": float(param["tfce_start"]), "step": float(param["tfce_step"])}
    if method == "cluster":
        p_thresh = float(param["cluster_forming_p"]) / 2.0
        return float(-stats.t.ppf(p_thresh, n_samples - 1))
    raise ValueError(f"Unknown inference_method: {method}")


def run_massuni_test(data_3d, connect, param):
    """data_3d: (n_subj, n_ch, n_time) → returns result dict."""
    testdata = np.swapaxes(data_3d, 2, 1)  # (n_subj, n_time, n_ch)
    threshold = compute_threshold(testdata.shape[0], param)

    stat_map, clusters, cluster_p_values, _ = st_clust_1s_ttest(
        testdata,
        n_jobs=param["njobs"],
        threshold=threshold,
        adjacency=connect,
        n_permutations=param["nperms"],
        buffer_size=None,
        seed=param["random_state"],
        tail=0,
    )

    cluster_p_values = np.asarray(cluster_p_values, dtype=float)
    pmap_corrected = np.ones(stat_map.shape, dtype=float)
    for clu, p_val in zip(clusters, cluster_p_values):
        pmap_corrected[clu] = np.minimum(pmap_corrected[clu], p_val)

    return {
        "stat_map": stat_map,
        "clusters": clusters,
        "cluster_p_values": cluster_p_values,
        "pmap_corrected": pmap_corrected,
        "sig_mask": pmap_corrected < float(param["point_alpha"]),
        "threshold_used": threshold,
        "method": param["inference_method"].lower(),
    }


def holm_correction(pvals, alpha=0.05):
    pvals = np.asarray(pvals, dtype=float)
    m = len(pvals)
    if m == 0:
        return np.array([], dtype=bool), np.array([], dtype=float)
    order = np.argsort(pvals)
    adj = np.maximum.accumulate([(m - i) * pvals[order[i]] for i in range(m)])
    adj = np.clip(adj, 0, 1)
    p_adj = np.empty(m)
    p_adj[order] = adj
    return p_adj < alpha, p_adj


def correct_across_maps(results, alpha=0.05, method="holm"):
    raw_ps = np.array([
        float(np.min(r["cluster_p_values"])) if r["cluster_p_values"].size else 1.0
        for r in results
    ])
    method = method.lower()
    if method == "holm":
        reject, p_adj = holm_correction(raw_ps, alpha=alpha)
    elif method == "bonferroni":
        p_adj = np.clip(raw_ps * len(raw_ps), 0, 1)
        reject = p_adj < alpha
    elif method == "none":
        p_adj = raw_ps.copy()
        reject = p_adj < alpha
    else:
        raise ValueError(f"Unknown map_correction: {method}")

    rows = []
    for i, r in enumerate(results):
        r["map_p_raw"] = float(raw_ps[i])
        r["map_p_adj"] = float(p_adj[i])
        r["map_sig"] = bool(reject[i])
        rows.append({
            "map": r["name"],
            "inference_method": r["method"],
            "threshold_used": str(r["threshold_used"]),
            "map_p_raw": r["map_p_raw"],
            "map_p_adj": r["map_p_adj"],
            "map_sig": r["map_sig"],
        })
    return results, pd.DataFrame(rows)


def save_massuni_outputs(results, regvars_main, z_dir):
    """Write per-map npy files; also write stacked arrays for the main regressors."""
    for res in results:
        name = res["name"]
        np.save(z_dir / f"ols_2ndlevel_tval_{name}.npy", res["stat_map"])
        np.save(z_dir / f"ols_2ndlevel_pval_corr_{name}.npy", res["pmap_corrected"])
        np.save(z_dir / f"ols_2ndlevel_sigmask_{name}.npy", res["sig_mask"])
        np.save(z_dir / f"ols_2ndlevel_cluster_pvals_{name}.npy",
                np.asarray(res["cluster_p_values"], dtype=float))

    main = [r for r in results if r["name"] in regvars_main]
    np.save(z_dir / "ols_2ndlevel_tvals.npy", np.stack([r["stat_map"] for r in main]))
    pvals_corr = np.stack([r["pmap_corrected"] for r in main])
    np.save(z_dir / "ols_2ndlevel_pvals.npy", pvals_corr)        # backward-compat alias
    np.save(z_dir / "ols_2ndlevel_pvals_corr.npy", pvals_corr)
    np.save(z_dir / "ols_2ndlevel_sigmasks.npy",
            np.stack([r["sig_mask"] for r in main]))


def run_second_level_family(allbetas, beta_gavg, regvars, z_dir, param):
    """allbetas: (n_subj, 2, n_ch, n_time)"""
    connect, _ = mne.channels.find_ch_adjacency(beta_gavg[0].info, ch_type="eeg")
    results = []

    for idx, regvar in enumerate(regvars):
        print(f"\nSecond-level {param['inference_method'].upper()} — {regvar}")
        res = run_massuni_test(allbetas[:, idx, :, :], connect, param)
        res["name"] = regvar
        results.append(res)

    print(f"\nSecond-level {param['inference_method'].upper()} — pain − money difference")
    res_diff = run_massuni_test(allbetas[:, 0, :, :] - allbetas[:, 1, :, :], connect, param)
    res_diff["name"] = "diff_pain_minus_money"
    results.append(res_diff)

    results, map_table = correct_across_maps(results, alpha=param["map_alpha"],
                                             method=param["map_correction"])
    map_table.to_csv(z_dir / "map_table_corrected.csv", index=False)
    save_massuni_outputs(results, regvars_main=regvars, z_dir=z_dir)
    return results, map_table


# ===========================================================
# Stats summary helpers
# ===========================================================
def contiguous_true_runs(x):
    runs, start = [], None
    for i, v in enumerate(x):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(x) - 1))
    return runs


def summarize_time_windows(sig_mask, times_ms):
    time_any = sig_mask.any(axis=1)
    return [
        {"start_idx": s, "end_idx": e,
         "start_ms": times_ms[s], "end_ms": times_ms[e],
         "duration_ms": times_ms[e] - times_ms[s]}
        for s, e in contiguous_true_runs(time_any)
    ]


def peak_stat_in_window(stat_map, sig_mask, s, e, ch_names, times_ms):
    masked = np.where(sig_mask[s:e+1, :], stat_map[s:e+1, :], np.nan)
    if np.all(np.isnan(masked)):
        return None
    flat_idx = np.nanargmax(np.abs(masked))
    t_rel, ch_idx = np.unravel_index(flat_idx, masked.shape)
    peak_val = masked[t_rel, ch_idx]
    return {
        "peak_stat": float(peak_val),
        "peak_time_ms": float(times_ms[s + t_rel]),
        "peak_channel": ch_names[ch_idx],
        "sign": "positive" if peak_val > 0 else "negative",
    }


def channels_in_window(sig_mask, s, e, ch_names, min_timepoints=1):
    counts = sig_mask[s:e+1, :].sum(axis=0)
    return [ch_names[i] for i in np.where(counts >= min_timepoints)[0]], counts


def summarize_beta_map(beta_data, sig_mask, times_ms, ch_names):
    masked = np.where(sig_mask, beta_data.T, np.nan)
    if np.all(np.isnan(masked)):
        return None
    vals = masked[np.isfinite(masked)]
    max_idx = np.nanargmax(masked)
    min_idx = np.nanargmin(masked)
    max_t, max_ch = np.unravel_index(max_idx, masked.shape)
    min_t, min_ch = np.unravel_index(min_idx, masked.shape)
    return {
        "mean_beta_sig": float(np.nanmean(masked)),
        "median_beta_sig": float(np.nanmedian(masked)),
        "sd_beta_sig": float(np.nanstd(masked)),
        "prop_positive_beta": float(np.mean(vals > 0)),
        "prop_negative_beta": float(np.mean(vals < 0)),
        "max_beta": float(masked[max_t, max_ch]),
        "max_beta_time_ms": float(times_ms[max_t]),
        "max_beta_channel": ch_names[max_ch],
        "min_beta": float(masked[min_t, min_ch]),
        "min_beta_time_ms": float(times_ms[min_t]),
        "min_beta_channel": ch_names[min_ch],
    }


def summarize_difference_direction(diff_tvals, diff_mask, times_ms, ch_names):
    masked = np.where(diff_mask, diff_tvals, np.nan)
    if np.all(np.isnan(masked)):
        return None
    vals = masked[np.isfinite(masked)]
    return {
        "prop_pain_gt_money": float(np.mean(vals > 0)),
        "prop_money_gt_pain": float(np.mean(vals < 0)),
        "mean_diff_stat": float(np.nanmean(masked)),
        "median_diff_stat": float(np.nanmedian(masked)),
    }


# ===========================================================
# Figure helpers
# ===========================================================
def get_bin_colors(cmap_name, n_bins, minval=0.25, maxval=0.95):
    cmap = plt.get_cmap(cmap_name)
    if n_bins == 1:
        return [cmap(0.7)]
    return [cmap(x) for x in np.linspace(minval, maxval, n_bins)]


def significance_label(method):
    return {"tfce": "TFCE-corrected p < .05",
            "cluster": "Cluster-corrected p < .05"}.get(method, "Corrected p < .05")


# ===========================================================
# SECTION 1: DECISION FIRST-LEVEL REGRESSION
#
# BUG: epo_cop.metadata["trialsnum"] is not present in freshly-loaded epochs.
# The `trialsnum` column is only added to epo_1 in-memory during the trial-map
# loop below and is never persisted to the fif file.  Additionally, df2 is
# built from epo_1.metadata (not mod_data), so df2["trialsnum"] also does not
# exist.  The trial-matching line will raise KeyError until the logic is
# rewritten to use integer positional matching or the fif files are regenerated
# with trialsnum in their metadata.
# ===========================================================
mod_data = mod_data_decision.copy()
mod_data = mod_data[mod_data["participant"].isin(common_participants)].copy()
mod_data["rt"] = mod_data["choice_resp.rt"]
mod_data["trialsnum"] = (
    mod_data["blocks.thisRepN"].astype(int) * 25
    + mod_data["trials.thisN"].astype(int)
    + 1
)

part = common_participants

# ---- Build trial map (first epoch load) ----
filtered_data = []
for p in part:
    df = mod_data[mod_data["participant"] == p]

    epo_1 = mne.read_epochs(
        opj(basepath, "derivatives", p, "eeg", "erps_decision",
            f"{p}_decision_cues_singletrials-epo.fif"),
        preload=True,
    )

    participants_loop = epo_1.metadata["participant_id"].unique()
    trialblocks = []
    blocks_idx = []
    for _participant in participants_loop:
        trialblocks.extend(list(range(25)) * 5)
        blocks_idx.extend([i for i in range(5) for _ in range(25)])
    epo_1.metadata["trialblocks"] = trialblocks
    epo_1.metadata["blocks_idx"] = blocks_idx

    epo_1_filtered = pd.DataFrame()
    for participant in df["participant"].unique():
        erps_p_df = epo_1.metadata[epo_1.metadata["participant_id"] == participant]
        df_unique = df[df["participant"] == participant]
        for block_x in df_unique["blocks.thisRepN"].unique():
            erps_block_df = erps_p_df[erps_p_df["blocks_idx"] == block_x]
            df_block_df = df_unique[df_unique["blocks.thisRepN"] == block_x]
            filtered_block_df = erps_block_df[
                erps_block_df["trialblocks"].isin(df_block_df["trials.thisN"])
            ]
            epo_1_filtered = pd.concat([epo_1_filtered, filtered_block_df],
                                       ignore_index=True)
    filtered_data.append(epo_1_filtered)

epo_1_filtered_combined = pd.concat(filtered_data, ignore_index=True)

# ---- Run decision regression ----
all_epos = [[] for _ in range(len(regvars))]
allbetasnp = []
betas = [[] for _ in range(len(regvars))]
included_subjects = []
skipped_subjects = []

for pa in part:
    print(f"\n--- Processing {pa} ---")

    df2 = epo_1_filtered_combined[epo_1_filtered_combined["participant_id"] == pa]
    mod2 = mod_data[mod_data["participant"] == pa].copy()

    epo_cop = mne.read_epochs(
        opj(basepath, "derivatives", pa, "eeg", "erps_decision",
            f"{pa}_decision_cues_singletrials-epo.fif"),
        preload=True,
    ).copy()

    # BUG: epo_cop.metadata["trialsnum"] does not exist — see module header.
    matching = epo_cop.metadata["trialsnum"].isin(df2["trialsnum"])
    epo_filt = epo_cop[matching]

    if epo_filt.info["sfreq"] != param["testresampfreq"]:
        epo_filt = epo_filt.resample(param["testresampfreq"])

    goodtrials = np.where(epo_filt.metadata["badtrial"] == 0)[0]
    mod2 = mod2.iloc[goodtrials].reset_index(drop=True)
    epo_filt = epo_filt[goodtrials]

    if len(epo_filt) < 5:
        print("  Skipping: too few trials")
        skipped_subjects.append(pa)
        continue

    vals_pain = mod2["painlevel"].to_numpy(dtype=float)
    vals_money = mod2["moneylevel"].to_numpy(dtype=float)
    vals_rt = mod2["rt"].to_numpy(dtype=float)

    keep = np.where(np.isfinite(vals_pain) & np.isfinite(vals_money) & np.isfinite(vals_rt))[0]
    if len(keep) < 5:
        print("  Skipping: too few valid trials after NaN removal")
        skipped_subjects.append(pa)
        continue

    mod2k = mod2.iloc[keep].reset_index(drop=True)
    epo_keep = epo_filt.copy()[keep]

    if epo_keep.metadata is None:
        epo_keep.metadata = pd.DataFrame(index=np.arange(len(epo_keep)))
    else:
        epo_keep.metadata = epo_keep.metadata.reset_index(drop=True).copy()

    epo_keep.metadata["painlevel"] = mod2k["painlevel"].to_numpy(dtype=float)
    epo_keep.metadata["moneylevel"] = mod2k["moneylevel"].to_numpy(dtype=float)
    epo_keep.metadata["participant_id"] = pa
    epo_keep.metadata["rt"] = mod2k["rt"].to_numpy(dtype=float)

    scale = Scaler(scalings="mean")
    epo_z = mne.EpochsArray(scale.fit_transform(epo_keep.get_data()), epo_keep.info)

    df_reg = mod2k.copy()
    df_reg["Intercept"] = 1.0
    df_reg["pain_z"] = stats.zscore(df_reg["painlevel"].to_numpy(dtype=float))
    df_reg["money_z"] = stats.zscore(df_reg["moneylevel"].to_numpy(dtype=float))
    df_reg["RT_z"] = stats.zscore(df_reg["rt"].to_numpy(dtype=float))

    names = ["Intercept", "pain_z", "money_z", "RT_z"]
    design = df_reg[names]

    if not np.all(np.isfinite(design.to_numpy())):
        print("  Skipping: NaN/Inf in design matrix")
        skipped_subjects.append(pa)
        continue

    res = mne.stats.linear_regression(epo_z, design, names=names)
    beta_pain = res["pain_z"].beta
    beta_money = res["money_z"].beta

    betas[0].append(beta_pain)
    betas[1].append(beta_money)
    allbetasnp.append(np.stack([beta_pain.data, beta_money.data]))
    all_epos[0].append(epo_keep)
    all_epos[1].append(epo_keep)
    included_subjects.append(pa)

if len(allbetasnp) == 0:
    raise RuntimeError("No subjects included in DECISION analysis.")

allbetas = np.stack(allbetasnp)                               # (n_subj, 2, n_ch, n_time)
beta_gavg = [mne.grand_average(betas[i]) for i in range(len(regvars))]
all_epos_cat = [mne.concatenate_epochs(all_epos[i]) for i in range(len(regvars))]

print(f"\nIncluded: {included_subjects}")
print(f"Skipped:  {skipped_subjects}")

# ===========================================================
# SECTION 2: SECOND-LEVEL INFERENCE
# ===========================================================
massuni_results, map_table = run_second_level_family(
    allbetas=allbetas,
    beta_gavg=beta_gavg,
    regvars=regvars,
    z_dir=z_dir,
    param=param,
)

np.save(z_dir / "ols_2ndlevel_betas.npy", allbetas)
np.save(z_dir / "included_subjects.npy", np.array(included_subjects, dtype=object))
np.save(z_dir / "ols_2ndlevel_betasavg.npy", np.array(beta_gavg, dtype=object))

for idx, regvar in enumerate(regvars):
    all_epos_cat[idx].save(
        z_dir / f"ols_2ndlevel_allepochs-epo_{regvar}.fif", overwrite=True
    )

print("\nInference method:", param["inference_method"])
print("Outputs in:", z_dir)

# ===========================================================
# SECTION 3: STATS SUMMARY
# ===========================================================
times_ms = beta_gavg[0].times * 1000
ch_names = list(beta_gavg[0].ch_names)

results_by_name = {r["name"]: r for r in massuni_results}
beta_data_lookup = {
    "painlevel": beta_gavg[0].data,
    "moneylevel": beta_gavg[1].data,
}

all_maps = regvars + ["diff_pain_minus_money"]
results_rows = []
window_rows = []

for map_name in all_maps:
    res = results_by_name[map_name]
    stat_map = res["stat_map"]
    sig_mask = res["sig_mask"]

    row = {
        "map": map_name,
        "inference_method": res["method"],
        "threshold_used": str(res["threshold_used"]),
        "map_p_raw": res["map_p_raw"],
        "map_p_adj": res["map_p_adj"],
        "map_sig": res["map_sig"],
        "n_sig_points": int(sig_mask.sum()),
        "n_sig_timepoints": int(sig_mask.any(axis=1).sum()),
        "n_sig_channels": int(sig_mask.any(axis=0).sum()),
    }

    windows = summarize_time_windows(sig_mask, times_ms)

    if len(windows) == 0:
        results_rows.append(row)
        continue

    all_peak = peak_stat_in_window(
        stat_map, sig_mask,
        windows[0]["start_idx"], windows[-1]["end_idx"],
        ch_names, times_ms,
    )
    if all_peak is not None:
        row.update({
            "peak_stat": all_peak["peak_stat"],
            "peak_time_ms": all_peak["peak_time_ms"],
            "peak_channel": all_peak["peak_channel"],
            "peak_stat_sign": all_peak["sign"],
        })

    if map_name in beta_data_lookup:
        beta_summary = summarize_beta_map(
            beta_data_lookup[map_name], sig_mask, times_ms, ch_names
        )
        if beta_summary is not None:
            row.update(beta_summary)

    if map_name == "diff_pain_minus_money":
        diff_summary = summarize_difference_direction(
            stat_map, sig_mask, times_ms, ch_names
        )
        if diff_summary is not None:
            row.update(diff_summary)

    results_rows.append(row)

    for wi, w in enumerate(windows, start=1):
        peak = peak_stat_in_window(
            stat_map, sig_mask,
            w["start_idx"], w["end_idx"],
            ch_names, times_ms,
        )
        chans, _ = channels_in_window(sig_mask, w["start_idx"], w["end_idx"], ch_names)

        wr = {
            "map": map_name,
            "window_id": wi,
            "start_ms": w["start_ms"],
            "end_ms": w["end_ms"],
            "duration_ms": w["duration_ms"],
            "n_channels_in_window": len(chans),
            "channels": ", ".join(chans[:20]) + (" ..." if len(chans) > 20 else ""),
        }

        if peak is not None:
            wr.update({
                "peak_stat": peak["peak_stat"],
                "peak_time_ms": peak["peak_time_ms"],
                "peak_channel": peak["peak_channel"],
                "peak_stat_sign": peak["sign"],
            })

        if map_name in beta_data_lookup:
            bdata_t = beta_data_lookup[map_name].T
            wm = sig_mask[w["start_idx"]:w["end_idx"]+1, :]
            masked_b = np.where(wm, bdata_t[w["start_idx"]:w["end_idx"]+1, :], np.nan)
            if np.any(np.isfinite(masked_b)):
                vals = masked_b[np.isfinite(masked_b)]
                wr.update({
                    "mean_beta_window": float(np.nanmean(masked_b)),
                    "median_beta_window": float(np.nanmedian(masked_b)),
                    "prop_positive_beta_window": float(np.mean(vals > 0)),
                    "prop_negative_beta_window": float(np.mean(vals < 0)),
                })

        if map_name == "diff_pain_minus_money":
            wm = sig_mask[w["start_idx"]:w["end_idx"]+1, :]
            ws = stat_map[w["start_idx"]:w["end_idx"]+1, :]
            masked = np.where(wm, ws, np.nan)
            if np.any(np.isfinite(masked)):
                vals = masked[np.isfinite(masked)]
                wr.update({
                    "prop_pain_gt_money_window": float(np.mean(vals > 0)),
                    "prop_money_gt_pain_window": float(np.mean(vals < 0)),
                    "mean_diff_stat_window": float(np.nanmean(masked)),
                })

        window_rows.append(wr)

results_df = pd.DataFrame(results_rows)
windows_df = pd.DataFrame(window_rows)
results_df.to_csv(z_dir / "massuni_summary_maps.csv", index=False)
windows_df.to_csv(z_dir / "massuni_summary_windows.csv", index=False)

print("\n===== MAP-LEVEL SUMMARY =====")
print(results_df.to_string())
print("\n===== WINDOW-LEVEL SUMMARY =====")
print(windows_df.to_string())

# ===========================================================
# SECTION 4: FIGURES
# ===========================================================
tvals_stack = np.stack([results_by_name[r]["stat_map"] for r in regvars])
pvals_stack = np.stack([results_by_name[r]["pmap_corrected"] for r in regvars])

times_pos = [np.abs(beta_gavg[0].times - t).argmin() for t in plot_times]
timestep_ms = 1000.0 / param["testresampfreq"]

for ridx, regvar in enumerate(regvars):
    regvarname = regvarsnames[ridx]
    cmap = "Reds" if regvar == "painlevel" else "Blues"

    beta_ev = beta_gavg[ridx].copy()
    epo_cat = all_epos_cat[ridx]
    chankeep = np.array([c not in ["M1", "M2"] for c in beta_ev.ch_names])

    # ---- Topomaps at plot_times ----
    for tidx, timepos in enumerate(times_pos):
        fig, ax = plt.subplots(figsize=(1, 1))
        p_row = pvals_stack[ridx][timepos, :]
        mask = np.zeros_like(p_row, dtype=bool)
        mask[(p_row < param["point_alpha"]) & chankeep] = True

        im, _ = plot_topomap(
            beta_ev.data[:, timepos],
            pos=beta_ev.info,
            mask=mask,
            mask_params=dict(marker="o", markerfacecolor="w",
                             markeredgecolor="k", linewidth=0, markersize=2),
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
            f"{int(plot_times[tidx] * 1000)} ms\n({param['inference_method'].upper()})",
            fontdict={"size": param["labelfontsize"] - 1},
            pad=0.1,
        )

        if tidx + 1 == len(plot_times):
            fig2, cax = plt.subplots(figsize=(0.2, 1))
            cbar = fig2.colorbar(im, cax=cax, orientation="vertical", aspect=1)
            cbar.set_label("Beta (z)", rotation=270, labelpad=12,
                           fontdict={"fontsize": param["labelfontsize"] - 1})
            cbar.ax.tick_params(labelsize=param["ticksfontsize"] - 2)
            fig2.savefig(
                outfigpath / f"{fig_prefix}fig_topo_beta_cbar_{regvar}.svg",
                dpi=600, bbox_inches="tight",
            )
            plt.close(fig2)

        fig.savefig(
            outfigpath / f"{fig_prefix}fig_ols_erps_betas_topo_{regvar}_{tidx}.svg",
            dpi=600, bbox_inches="tight",
        )
        plt.close(fig)

    # ---- Binned ERP time-courses ----
    for ch in chan_to_plot:
        if ch not in beta_ev.ch_names:
            continue

        fig, ax = plt.subplots(figsize=(4, 2.5))
        epo_cat.metadata = epo_cat.metadata.reset_index(drop=True)
        level_vals = pd.to_numeric(epo_cat.metadata[regvar], errors="coerce")
        unique_levels = np.sort(level_vals.dropna().unique())
        unique_levels = unique_levels[unique_levels > 0]
        level_to_bin = {lev: i for i, lev in enumerate(unique_levels)}
        epo_cat.metadata["_bin"] = level_vals.map(level_to_bin)
        nbins_eff = len(unique_levels)
        bin_colors = get_bin_colors(cmap, nbins_eff)

        sub_evokeds = []
        for p_id in epo_cat.metadata["participant_id"].unique():
            sub = epo_cat[epo_cat.metadata["participant_id"] == p_id]
            sub_evoked = {
                b: sub[sub.metadata["_bin"] == b].average()
                if (sub.metadata["_bin"] == b).sum() > 0 else 0
                for b in range(nbins_eff)
            }
            sub_evokeds.append(sub_evoked)

        evokeds = {}
        for b in range(nbins_eff):
            evoked_list = [sd[b] for sd in sub_evokeds if sd[b] != 0]
            if evoked_list:
                evokeds[b] = mne.grand_average(evoked_list)

        pick = beta_ev.ch_names.index(ch)
        ax.set_title(f"{ch} – binned by {regvarname}", fontsize=param["titlefontsize"])
        ax.set_xlabel("Time (ms)", fontsize=param["labelfontsize"])
        ax.set_ylabel("Amplitude (µV)", fontsize=param["labelfontsize"])

        for b in sorted(evokeds):
            ax.plot(
                epo_cat.times * 1000,
                evokeds[b].data[pick, :] * 1e6,
                linewidth=2,
                label=str(int(unique_levels[b])),
                color=bin_colors[b],
            )

        ax.axhline(0, linestyle="--", color="gray")
        ax.axvline(0, linestyle="--", color="gray")
        ax.set_xticks(np.arange(-600, 1200, 200))
        ax.set_xticklabels([str(i) for i in np.arange(-600, 1200, 200)])
        ax.tick_params(labelsize=param["ticksfontsize"])
        ax.legend(fontsize=8, title="Level", title_fontsize=9,
                  frameon=False, loc="upper left", bbox_to_anchor=(0.02, 0.98),
                  borderaxespad=0.0, handlelength=1.6, labelspacing=0.3)
        fig.tight_layout()
        fig.savefig(
            outfigpath / f"{fig_prefix}fig_ols_erps_amp_bins_{regvar}_{ch}.svg",
            dpi=600, bbox_inches="tight",
        )
        plt.close(fig)

    # ---- Beta mean ± SEM with significance shading ----
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
        ax.plot(epo_cat.times * 1000, mean, linewidth=3)
        ax.fill_between(epo_cat.times * 1000, mean - sem, mean + sem, alpha=0.3)
        ax.set_ylim((-0.25, 0.25))
        ax.axhline(0, linestyle="--", color="gray")
        ax.axvline(0, linestyle="--", color="gray")

        sig_ymin, sig_ymax = -0.02, -0.005
        for ti, t_ms in enumerate(epo_cat.times * 1000):
            if pvals_stack[ridx][ti, pick] < param["point_alpha"]:
                ax.fill_between([t_ms, t_ms + timestep_ms],
                                sig_ymin, sig_ymax, alpha=0.3, facecolor="red")

        ax.text(0.99, 0.02, significance_label(param["inference_method"]),
                transform=ax.transAxes, ha="right", va="bottom", fontsize=8, alpha=0.8)
        ax.set_xticks(np.arange(-600, 1200, 200))
        ax.set_xticklabels([str(i) for i in np.arange(-600, 1200, 200)])
        ax.tick_params(labelsize=param["ticksfontsize"])

        fig.tight_layout()
        fig.savefig(
            outfigpath / f"{fig_prefix}fig_ols_erps_betas_{regvar}_{ch}.svg",
            dpi=600, bbox_inches="tight",
        )
        plt.close(fig)

# ---- Difference maps (pain − money) ----
diff_res = results_by_name["diff_pain_minus_money"]
tdiff = diff_res["stat_map"]
pdiff = diff_res["pmap_corrected"]
times = beta_gavg[0].times
info = beta_gavg[0].info
chankeep = np.array([c not in ["M1", "M2"] for c in info["ch_names"]])
diff_times_pos = [np.abs(times - t).argmin() for t in plot_times]

for tidx, time_idx in enumerate(diff_times_pos):
    t_ms = int(plot_times[tidx] * 1000)
    mask = (pdiff[time_idx, :] < param["point_alpha"]) & chankeep

    fig, ax = plt.subplots(figsize=(2, 2))
    im, _ = plot_topomap(
        tdiff[time_idx, :],
        pos=info,
        mask=mask,
        mask_params=dict(marker="o", markerfacecolor="w",
                         markeredgecolor="k", linewidth=0, markersize=3),
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
        f"pain − money, {t_ms} ms\n({param['inference_method'].upper()})",
        fontdict={"size": param["labelfontsize"] - 1},
        pad=0.1,
    )

    fig2, cax = plt.subplots(figsize=(0.2, 1))
    cbar = fig2.colorbar(im, cax=cax, orientation="vertical", aspect=1)
    cbar.set_label("t (pain − money)", rotation=270, labelpad=12,
                   fontdict={"fontsize": param["labelfontsize"] - 1})
    cbar.ax.tick_params(labelsize=param["ticksfontsize"] - 2)

    fig.savefig(
        outfigpath / f"{fig_prefix}fig_topo_diff_pain_minus_money_{t_ms}ms.svg",
        dpi=600, bbox_inches="tight",
    )
    fig2.savefig(
        outfigpath / f"{fig_prefix}fig_topo_diff_pain_minus_money_{t_ms}ms_cbar.svg",
        dpi=600, bbox_inches="tight",
    )
    plt.close(fig)
    plt.close(fig2)

print("\nAll done.")
print("Stats outputs:", z_dir)
print("Figures:", outfigpath)
