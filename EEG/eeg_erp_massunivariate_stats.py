import numpy as np
import mne

# -----------------------------
# LOAD YOUR DATA
# -----------------------------
base_path = r"/rds/projects/z/zhanglp-vwendler-core/PainReward_ULaval/EEG/PainReward_sub-001-050/painrewardeegdata/derivatives/erps_massuni_passive/Zscoring_tfce/"

# choose map #
maps = ["painlevel", "moneylevel", "diff_pain_minus_money"]

beta_gavg = np.load(base_path + "ols_2ndlevel_betasavg.npy", allow_pickle=True)
times = beta_gavg[0].times * 1000  #to ms
ch_names = beta_gavg[0].info['ch_names']

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
def find_time_clusters(sig_mask):
    """Find contiguous time clusters where ANY channel is significant."""
    time_sig = sig_mask.any(axis=1)  # collapse channels

    clusters = []
    current = []

    for i, val in enumerate(time_sig):
        if val:
            current.append(i)
        else:
            if len(current) > 0:
                clusters.append(current)
                current = []

    if len(current) > 0:
        clusters.append(current)

    return clusters

def summarize_cluster(cluster_idx, sig_mask, tvals, times, ch_names):
    """Extract peak stats for one cluster."""
    
    cluster_tvals = tvals[cluster_idx, :]
    cluster_mask = sig_mask[cluster_idx, :]

    # mask non-significant
    cluster_tvals_masked = np.where(cluster_mask, cluster_tvals, np.nan)

    # peak
    peak_idx = np.nanargmax(np.abs(cluster_tvals_masked))
    t_idx, ch_idx = np.unravel_index(peak_idx, cluster_tvals_masked.shape)

    peak_time = times[cluster_idx[t_idx]]
    peak_channel = ch_names[ch_idx]
    peak_t = cluster_tvals_masked[t_idx, ch_idx]

    # electrodes involved
    active_channels = np.where(cluster_mask.sum(axis=0) > 0)[0]
    channel_list = [ch_names[i] for i in active_channels]

    return {
        "time_start": times[cluster_idx[0]],
        "time_end": times[cluster_idx[-1]],
        "peak_time": peak_time,
        "peak_channel": peak_channel,
        "peak_t": peak_t,
        "channels": channel_list
    }

# -----------------------------
# MAIN LOOP
# -----------------------------
for map_name in maps:
    print("\n" + "="*50)
    print(f"MAP: {map_name}")
    print("="*50)

    sig_mask = np.load(base_path + f"ols_2ndlevel_sigmask_{map_name}.npy")
    tvals = np.load(base_path + f"ols_2ndlevel_tval_{map_name}.npy")

    clusters = find_time_clusters(sig_mask)

    if len(clusters) == 0:
        print("No significant clusters.")
        continue

    for i, cluster in enumerate(clusters):
        summary = summarize_cluster(cluster, sig_mask, tvals, times, ch_names)

        print(f"\nCluster {i+1}:")
        print(f"Time window: {summary['time_start']:.0f}–{summary['time_end']:.0f} ms")
        print(f"Peak: t = {summary['peak_t']:.2f} at {summary['peak_channel']} ({summary['peak_time']:.0f} ms)")
        print(f"Channels involved: {', '.join(summary['channels'][:8])} ...")