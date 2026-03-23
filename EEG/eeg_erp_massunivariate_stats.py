import os
from pathlib import Path
import numpy as np
import pandas as pd
import mne

# =========================================================
# SETTINGS
# =========================================================
base_stats = Path(r"/rds/projects/z/zhanglp-vwendler-core/PainReward_ULaval/EEG/PainReward_sub-001-050/painrewardeegdata/derivatives/erps_massuni_passive/Zscoring_tfce")
# or decision:
# base_stats = Path(r"/path/to/your/statistics/erps_massuni_decision/Zscoring_tfce")

alpha = 0.05
maps = ["painlevel", "moneylevel", "diff_pain_minus_money"]

# =========================================================
# LOAD CORE FILES
# =========================================================
map_table = pd.read_csv(base_stats / "map_table_corrected.csv")
beta_gavg = np.load(base_stats / "ols_2ndlevel_betasavg.npy", allow_pickle=True)
allbetas = np.load(base_stats / "ols_2ndlevel_betas.npy", allow_pickle=True)

# beta_gavg[0] = pain, beta_gavg[1] = money
times_ms = beta_gavg[0].times * 1000
ch_names = beta_gavg[0].info["ch_names"]

# =========================================================
# HELPERS
# =========================================================
def contiguous_true_runs(x):
    """Return list of (start_idx, end_idx) for contiguous True runs."""
    runs = []
    start = None
    for i, val in enumerate(x):
        if val and start is None:
            start = i
        elif not val and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(x) - 1))
    return runs

def summarize_time_windows(sig_mask, times_ms):
    """
    Summarize significant time windows by collapsing over channels.
    sig_mask shape: (time, channels)
    """
    time_any = sig_mask.any(axis=1)
    runs = contiguous_true_runs(time_any)
    out = []
    for s, e in runs:
        out.append({
            "start_idx": s,
            "end_idx": e,
            "start_ms": times_ms[s],
            "end_ms": times_ms[e],
            "duration_ms": times_ms[e] - times_ms[s]
        })
    return out

def peak_stat_in_window(stat_map, sig_mask, start_idx, end_idx, ch_names, times_ms):
    """
    Find peak absolute stat in a significant window.
    stat_map shape: (time, channels)
    """
    window_stats = stat_map[start_idx:end_idx+1, :]
    window_mask = sig_mask[start_idx:end_idx+1, :]
    masked = np.where(window_mask, window_stats, np.nan)

    if np.all(np.isnan(masked)):
        return None

    flat_idx = np.nanargmax(np.abs(masked))
    t_rel, ch_idx = np.unravel_index(flat_idx, masked.shape)
    t_idx = start_idx + t_rel

    peak_val = masked[t_rel, ch_idx]
    return {
        "peak_stat": float(peak_val),
        "peak_time_ms": float(times_ms[t_idx]),
        "peak_channel": ch_names[ch_idx],
        "sign": "positive" if peak_val > 0 else "negative"
    }

def channels_in_window(sig_mask, start_idx, end_idx, ch_names, min_timepoints=1):
    """
    Return channels significant at least min_timepoints in the window.
    """
    window_mask = sig_mask[start_idx:end_idx+1, :]
    counts = window_mask.sum(axis=0)
    idx = np.where(counts >= min_timepoints)[0]
    return [ch_names[i] for i in idx], counts

def summarize_beta_map(beta_data, sig_mask, times_ms, ch_names):
    """
    beta_data shape: (channels, time)
    sig_mask shape: (time, channels)
    Return descriptive stats over significant region.
    """
    beta_time_chan = beta_data.T  # -> (time, channels)
    masked_beta = np.where(sig_mask, beta_time_chan, np.nan)

    if np.all(np.isnan(masked_beta)):
        return None

    vals = masked_beta[np.isfinite(masked_beta)]

    # Peak positive / negative beta in significant region
    max_idx = np.nanargmax(masked_beta)
    min_idx = np.nanargmin(masked_beta)

    max_t, max_ch = np.unravel_index(max_idx, masked_beta.shape)
    min_t, min_ch = np.unravel_index(min_idx, masked_beta.shape)

    return {
        "mean_beta_sig": float(np.nanmean(masked_beta)),
        "median_beta_sig": float(np.nanmedian(masked_beta)),
        "sd_beta_sig": float(np.nanstd(masked_beta)),
        "prop_positive_beta": float(np.mean(vals > 0)),
        "prop_negative_beta": float(np.mean(vals < 0)),
        "max_beta": float(masked_beta[max_t, max_ch]),
        "max_beta_time_ms": float(times_ms[max_t]),
        "max_beta_channel": ch_names[max_ch],
        "min_beta": float(masked_beta[min_t, min_ch]),
        "min_beta_time_ms": float(times_ms[min_t]),
        "min_beta_channel": ch_names[min_ch],
    }

def summarize_difference_direction(diff_tvals, diff_mask, times_ms, ch_names):
    """
    For difference map only: positive t means pain > money, negative t means money > pain.
    """
    masked = np.where(diff_mask, diff_tvals, np.nan)
    if np.all(np.isnan(masked)):
        return None

    vals = masked[np.isfinite(masked)]
    pos_prop = np.mean(vals > 0)
    neg_prop = np.mean(vals < 0)

    return {
        "prop_pain_gt_money": float(pos_prop),
        "prop_money_gt_pain": float(neg_prop),
        "mean_diff_stat": float(np.nanmean(masked)),
        "median_diff_stat": float(np.nanmedian(masked)),
    }

# =========================================================
# LOAD MAP-SPECIFIC FILES
# =========================================================
# pain and money beta grand averages
beta_map_lookup = {
    "painlevel": beta_gavg[0].data,   # (channels, time)
    "moneylevel": beta_gavg[1].data,  # (channels, time)
}

results_rows = []
window_rows = []

for map_name in maps:
    stat_map = np.load(base_stats / f"ols_2ndlevel_tval_{map_name}.npy")          # (time, channels)
    pmap = np.load(base_stats / f"ols_2ndlevel_pval_corr_{map_name}.npy") \
        if (base_stats / f"ols_2ndlevel_pval_corr_{map_name}.npy").exists() \
        else np.load(base_stats / f"ols_2ndlevel_pval_{map_name}.npy")
    sig_mask = np.load(base_stats / f"ols_2ndlevel_sigmask_{map_name}.npy")

    row = map_table.loc[map_table["map"] == map_name].iloc[0].to_dict()
    row["n_sig_points"] = int(sig_mask.sum())
    row["n_sig_timepoints"] = int(sig_mask.any(axis=1).sum())
    row["n_sig_channels"] = int(sig_mask.any(axis=0).sum())

    # Time windows
    windows = summarize_time_windows(sig_mask, times_ms)

    if len(windows) == 0:
        results_rows.append(row)
        continue

    # Peak over whole map
    all_peak = peak_stat_in_window(
        stat_map=stat_map,
        sig_mask=sig_mask,
        start_idx=windows[0]["start_idx"],
        end_idx=windows[-1]["end_idx"],
        ch_names=ch_names,
        times_ms=times_ms
    )

    if all_peak is not None:
        row.update({
            "peak_stat": all_peak["peak_stat"],
            "peak_time_ms": all_peak["peak_time_ms"],
            "peak_channel": all_peak["peak_channel"],
            "peak_stat_sign": all_peak["sign"],
        })

    # Beta summaries for pain and money maps
    if map_name in beta_map_lookup:
        beta_summary = summarize_beta_map(
            beta_data=beta_map_lookup[map_name],
            sig_mask=sig_mask,
            times_ms=times_ms,
            ch_names=ch_names
        )
        if beta_summary is not None:
            row.update(beta_summary)

    # Difference direction summary
    if map_name == "diff_pain_minus_money":
        diff_summary = summarize_difference_direction(
            diff_tvals=stat_map,
            diff_mask=sig_mask,
            times_ms=times_ms,
            ch_names=ch_names
        )
        if diff_summary is not None:
            row.update(diff_summary)

    results_rows.append(row)

    # window-level details
    for wi, w in enumerate(windows, start=1):
        peak = peak_stat_in_window(
            stat_map=stat_map,
            sig_mask=sig_mask,
            start_idx=w["start_idx"],
            end_idx=w["end_idx"],
            ch_names=ch_names,
            times_ms=times_ms
        )

        chans, counts = channels_in_window(
            sig_mask=sig_mask,
            start_idx=w["start_idx"],
            end_idx=w["end_idx"],
            ch_names=ch_names,
            min_timepoints=1
        )

        wr = {
            "map": map_name,
            "window_id": wi,
            "start_ms": w["start_ms"],
            "end_ms": w["end_ms"],
            "duration_ms": w["duration_ms"],
            "n_channels_in_window": len(chans),
            "channels": ", ".join(chans[:20]) + (" ..." if len(chans) > 20 else "")
        }

        if peak is not None:
            wr.update({
                "peak_stat": peak["peak_stat"],
                "peak_time_ms": peak["peak_time_ms"],
                "peak_channel": peak["peak_channel"],
                "peak_stat_sign": peak["sign"],
            })

        # For pain/money maps: beta sign summary within this window
        if map_name in beta_map_lookup:
            beta_data = beta_map_lookup[map_name].T  # (time, channels)
            window_mask = sig_mask[w["start_idx"]:w["end_idx"]+1, :]
            window_beta = beta_data[w["start_idx"]:w["end_idx"]+1, :]
            masked_beta = np.where(window_mask, window_beta, np.nan)

            if np.any(np.isfinite(masked_beta)):
                vals = masked_beta[np.isfinite(masked_beta)]
                wr.update({
                    "mean_beta_window": float(np.nanmean(masked_beta)),
                    "median_beta_window": float(np.nanmedian(masked_beta)),
                    "prop_positive_beta_window": float(np.mean(vals > 0)),
                    "prop_negative_beta_window": float(np.mean(vals < 0)),
                })

        # For difference map: pain > money vs money > pain in this window
        if map_name == "diff_pain_minus_money":
            window_mask = sig_mask[w["start_idx"]:w["end_idx"]+1, :]
            window_stat = stat_map[w["start_idx"]:w["end_idx"]+1, :]
            masked = np.where(window_mask, window_stat, np.nan)
            if np.any(np.isfinite(masked)):
                vals = masked[np.isfinite(masked)]
                wr.update({
                    "prop_pain_gt_money_window": float(np.mean(vals > 0)),
                    "prop_money_gt_pain_window": float(np.mean(vals < 0)),
                    "mean_diff_stat_window": float(np.nanmean(masked)),
                })

        window_rows.append(wr)

# =========================================================
# SAVE AND PRINT
# =========================================================
results_df = pd.DataFrame(results_rows)
windows_df = pd.DataFrame(window_rows)

results_df.to_csv(base_stats / "massuni_summary_maps.csv", index=False)
windows_df.to_csv(base_stats / "massuni_summary_windows.csv", index=False)

print("\n===== MAP-LEVEL SUMMARY =====")
print(results_df)

print("\n===== WINDOW-LEVEL SUMMARY =====")
print(windows_df)







# import numpy as np
# import mne

# # -----------------------------
# # LOAD YOUR DATA
# # -----------------------------
# base_path = r"/rds/projects/z/zhanglp-vwendler-core/PainReward_ULaval/EEG/PainReward_sub-001-050/painrewardeegdata/derivatives/erps_massuni_passive/Zscoring_tfce/"

# # choose map #
# maps = ["painlevel", "moneylevel", "diff_pain_minus_money"]

# beta_gavg = np.load(base_path + "ols_2ndlevel_betasavg.npy", allow_pickle=True)
# times = beta_gavg[0].times * 1000  #to ms
# ch_names = beta_gavg[0].info['ch_names']

# # -----------------------------
# # HELPER FUNCTIONS
# # -----------------------------
# def find_time_clusters(sig_mask):
#     """Find contiguous time clusters where ANY channel is significant."""
#     time_sig = sig_mask.any(axis=1)  # collapse channels

#     clusters = []
#     current = []

#     for i, val in enumerate(time_sig):
#         if val:
#             current.append(i)
#         else:
#             if len(current) > 0:
#                 clusters.append(current)
#                 current = []

#     if len(current) > 0:
#         clusters.append(current)

#     return clusters

# def summarize_cluster(cluster_idx, sig_mask, tvals, times, ch_names):
#     """Extract peak stats for one cluster."""
    
#     cluster_tvals = tvals[cluster_idx, :]
#     cluster_mask = sig_mask[cluster_idx, :]

#     # mask non-significant
#     cluster_tvals_masked = np.where(cluster_mask, cluster_tvals, np.nan)

#     # peak
#     peak_idx = np.nanargmax(np.abs(cluster_tvals_masked))
#     t_idx, ch_idx = np.unravel_index(peak_idx, cluster_tvals_masked.shape)

#     peak_time = times[cluster_idx[t_idx]]
#     peak_channel = ch_names[ch_idx]
#     peak_t = cluster_tvals_masked[t_idx, ch_idx]

#     # electrodes involved
#     active_channels = np.where(cluster_mask.sum(axis=0) > 0)[0]
#     channel_list = [ch_names[i] for i in active_channels]

#     return {
#         "time_start": times[cluster_idx[0]],
#         "time_end": times[cluster_idx[-1]],
#         "peak_time": peak_time,
#         "peak_channel": peak_channel,
#         "peak_t": peak_t,
#         "channels": channel_list
#     }

# # -----------------------------
# # MAIN LOOP
# # -----------------------------
# for map_name in maps:
#     print("\n" + "="*50)
#     print(f"MAP: {map_name}")
#     print("="*50)

#     sig_mask = np.load(base_path + f"ols_2ndlevel_sigmask_{map_name}.npy")
#     tvals = np.load(base_path + f"ols_2ndlevel_tval_{map_name}.npy")

#     clusters = find_time_clusters(sig_mask)

#     if len(clusters) == 0:
#         print("No significant clusters.")
#         continue

#     for i, cluster in enumerate(clusters):
#         summary = summarize_cluster(cluster, sig_mask, tvals, times, ch_names)

#         print(f"\nCluster {i+1}:")
#         print(f"Time window: {summary['time_start']:.0f}–{summary['time_end']:.0f} ms")
#         print(f"Peak: t = {summary['peak_t']:.2f} at {summary['peak_channel']} ({summary['peak_time']:.0f} ms)")
#         print(f"Channels involved: {', '.join(summary['channels'][:8])} ...")