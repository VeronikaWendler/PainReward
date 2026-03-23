# # code from: https://github.com/nmarinsek/data-visualization-notebooks/blob/master/drift-diffusion-plot.ipynb


# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

# #set font size of labels on matplotlib plots
# plt.rc('font', size=16)

# #define a custom palette
# customPalette = ["#7C1313", "#095C98", '#D3500C', '#FFB139']
# plt.rcParams['axes.prop_cycle'] = plt.cycler(color=customPalette)


# t = 1000   #number of timepoints
# n = 100    #number of timeseries
# bias = 0.09  #bias in random walk

# #generate "biased random walk" timeseries
# data = pd.DataFrame(np.reshape(np.cumsum(np.random.randn(t,n)+bias,axis=0),(t,n)))
# data.head()

# def drift_diffusion_plot(values, upperbound, lowerbound, 
#                          upperlabel='', lowerlabel='', 
#                          stickybounds=True, **kwargs):
#     """
#     Creates a formatted drift-diffusion plot for a given timeseries.
    
#     Inputs:
#        - values: array of values in timeseries
#        - upperbound: numeric value of upper bound
#        - lowerbound: numeric value of lower bound
#        - upperlabel: optional label for upper bound
#        - lowerlabel: optional label for lower bound
#        - stickybounds: if true, timeseries stops when bound is hit
#        - kwargs: https://matplotlib.org/api/_as_gen/matplotlib.pyplot.plot.html
    
#     Output:
#        - ax: handle to plot axis
#     """
    
#     #if bounds are sticky, hide timepoints that follow the first bound hit
#     if stickybounds:
#         #check to see if (and when) a bound was hit
#         bound_hits = np.where((values>upperbound) | (values<lowerbound))[0]
#         #if a bound was hit, replace subsequent values with NaN
#         if len(bound_hits)>0:
#             values = values.copy()
#             values[bound_hits[0]+1:] = np.nan
    
#     #plot timeseries
#     ax = plt.gca()
#     plt.plot(values, **kwargs)
    
#     #format plot
#     ax.set_ylim(lowerbound, upperbound)
#     ax.set_yticks([lowerbound,upperbound])
#     ax.set_yticklabels([lowerlabel,upperlabel])
#     ax.axhline(y=np.mean([upperbound, lowerbound]), color='lightgray', zorder=0)
#     ax.set_xlim(0,len(values))
#     ax.set_xlabel('time')
#     ax.spines['left'].set_visible(False)
#     ax.spines['right'].set_visible(False)
    
#     return ax


# n=10

# #group 1 (positive drift)
# data1 = pd.DataFrame(np.reshape(np.cumsum(np.random.randn(t,n)+bias,axis=0),(t,n)))
# data1.apply(drift_diffusion_plot, upperbound=100, lowerbound=-100, 
#             color=customPalette[1], alpha=0.3);

# #group 2 (negative drift)
# data2 = pd.DataFrame(np.reshape(np.cumsum(np.random.randn(t,n)-bias,axis=0),(t,n)))
# data2.apply(drift_diffusion_plot, upperbound=100, lowerbound=-100, 
#             color=customPalette[0], alpha=0.3);

# #overlay means
# drift_diffusion_plot(np.mean(data1, axis=1), upperbound=100, lowerbound=-100,
#                      color=customPalette[1], lw=4, alpha=1);
# drift_diffusion_plot(np.mean(data2, axis=1), upperbound=100, lowerbound=-100, 
#                      upperlabel='accept', lowerlabel='reject', 
#                      color=customPalette[0], lw=4, alpha=1);




import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Global styling
# -----------------------------
plt.rcParams.update({
    'font.size': 20,
    'axes.labelsize': 22,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20
})

customPalette = ["#7C1313", "#095C98", '#D3500C', '#FFB139']
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=customPalette)

# -----------------------------
# Timing parameters in ms
# -----------------------------
total_ms = 1500          # full x-axis extent
dt_ms = 10               # resolution of simulation
ndt_ms = 420             # non-decision time
accum_ms = total_ms - ndt_ms
bias = 0.09  #bias in random walk

n_total = total_ms // dt_ms
n_ndt = ndt_ms // dt_ms
n_accum = accum_ms // dt_ms

time_pre = np.arange(0, ndt_ms, dt_ms)
time_post = np.arange(ndt_ms, total_ms, dt_ms)

# -----------------------------
# Diffusion parameters
# -----------------------------
n = 10
upperbound = 100
lowerbound = -100

def simulate_ddm_path(n_accum, bias, noise=1.0, start_value=0.0):
    """Simulate one diffusion path only during the accumulation period."""
    increments = np.random.randn(n_accum) * noise + bias
    walk = np.cumsum(increments) + start_value
    return walk

def apply_sticky_bounds(values, upperbound, lowerbound):
    """Stop path after first boundary crossing."""
    values = values.copy()
    bound_hits = np.where((values > upperbound) | (values < lowerbound))[0]
    if len(bound_hits) > 0:
        values[bound_hits[0] + 1:] = np.nan
    return values

# -----------------------------
# Simulate two groups
# -----------------------------
np.random.seed(7)

# positive drift: reward-dominant
data1 = pd.DataFrame({
    i: simulate_ddm_path(n_accum, bias=+2.5, noise=6.0)
    for i in range(n)
})

# negative drift: pain-dominant
data2 = pd.DataFrame({
    i: simulate_ddm_path(n_accum, bias=-1.7, noise=6.0)
    for i in range(n)
})

# -----------------------------
# Plot
# -----------------------------
fig, ax = plt.subplots(figsize=(11, 6.5))

# Neutral gray pre-accumulation segment
ax.plot([0, ndt_ms], [0, 0], color='gray', lw=3, alpha=0.9, zorder=2)

# Neutral gray center line after accumulation onset
ax.hlines(0, xmin=ndt_ms, xmax=total_ms, color='lightgray', lw=3, zorder=0)

# Vertical onset marker
ax.axvline(ndt_ms, color='gray', linestyle=':', lw=2)

# Individual reward-dominant paths
for col in data1.columns:
    y = apply_sticky_bounds(data1[col].values, upperbound, lowerbound)
    ax.plot(time_post, y, color=customPalette[1], alpha=0.25, lw=1.8)

# Individual pain-dominant paths
for col in data2.columns:
    y = apply_sticky_bounds(data2[col].values, upperbound, lowerbound)
    ax.plot(time_post, y, color=customPalette[0], alpha=0.25, lw=1.8)

# Mean reward path
mean1 = apply_sticky_bounds(data1.mean(axis=1).values, upperbound, lowerbound)
ax.plot(time_post, mean1, color=customPalette[1], lw=4)

# Mean pain path
mean2 = apply_sticky_bounds(data2.mean(axis=1).values, upperbound, lowerbound)
ax.plot(time_post, mean2, color=customPalette[0], lw=4)

# Bounds
ax.axhline(upperbound, color='black', lw=1.2)
ax.axhline(lowerbound, color='black', lw=1.2)

# Formatting
ax.set_xlim(0, total_ms)
ax.set_ylim(lowerbound, upperbound)
ax.set_xlabel('time (ms)')
ax.set_yticks([lowerbound, upperbound])
ax.set_yticklabels(['reject', 'accept'])

# Clean up spines
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.tight_layout()
plt.show()