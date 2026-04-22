# #Veronika Wendler

# Posterior Predictive Checks for the aDDM (this produces only RT & Accuracy, we programmed anther version in Matlab)
# can be used for both experiments, the 'garcia' quasi-replication (Exp1) and the 'OV' experiment, in which we manipulated overall value levels during learning
# you just need to set the paths accordingly


# libraries
import os
import gc
import hddm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import arviz as az

model_paths = [
    "/home/jovyan/OfficialTutorials/For_Linux/models_dir_OV/OV_replication_EE_5_4.hddm",
    "/home/jovyan/OfficialTutorials/For_Linux/models_dir_OV/OV_replication_EE_5_3.hddm",
    "/home/jovyan/OfficialTutorials/For_Linux/models_dir_OV/OV_replication_EE_5_2.hddm",
    "/home/jovyan/OfficialTutorials/For_Linux/models_dir_OV/OV_replication_EE_5_1.hddm",
    "/home/jovyan/OfficialTutorials/For_Linux/models_dir_OV/OV_replication_EE_5_0.hddm"
]

# initialize variables for selecting the best model (lowest DIC) 
best_model = None
best_model_path = None
best_dic = float('inf')
best_model_name = None

for path in model_paths:
    print(f"Loading model from: {path}")
    m = hddm.load(path)
    current_dic = m.dic
    print(f"Model DIC: {current_dic}")
    
    if current_dic < best_dic:
        best_dic = current_dic
        best_model = m
        best_model_path = path  
        best_model_name = os.path.basename(path).replace(".hddm", "")
    else:
        del m
        gc.collect()

print("Best model selected:", best_model_name, "with DIC =", best_dic)


# Posterior Predictive Data
print("Generating posterior predictive data with (nr of samples) samples per node...")
ppc_data = hddm.utils.post_pred_gen(best_model, samples=2000, append_data=True)     #samples=500
print("Posterior predictive data (first few rows):")
print(ppc_data.head())
output_dir = "/home/jovyan/OfficialTutorials/For_Linux/figures_dir_OV/OV_replication_EE_5/diagnostics"
os.makedirs(output_dir, exist_ok=True)

# RT Distribution
bins = np.histogram_bin_edges(best_model.data['rt'], bins=50)
fig, ax = plt.subplots(figsize=(8,6))
ax.hist(best_model.data['rt'], bins=bins, alpha=0.8, color='blue', label='Real RTs',
        density=True, edgecolor='black', linewidth=0.5)
ax.hist(ppc_data['rt_sampled'], bins=bins, alpha=0.8, color='red', label='Simulated RTs',
        density=True, edgecolor='black', linewidth=0.5)
ax.set_xlim(-10, 10)
ax.set_xlabel("Reaction Time (RT)", fontsize=13)
ax.set_ylabel("Frequency", fontsize=13)
ax.set_title(f"Posterior Predictive Check - {best_model_name} - RT Distribution", fontsize=15)
ax.tick_params(axis='both', which='major', labelsize=11)
ax.legend(fontsize=11, facecolor='white', framealpha=1, edgecolor='black')
ax.set_facecolor("white")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_color('black')
ax.spines['left'].set_color('black')
rt_plot_path = os.path.join(output_dir, f"RT_Distribution_{best_model_name}.png")
plt.savefig(rt_plot_path, dpi=300, bbox_inches='tight')
plt.close(fig) 

# Response Distribution
real_response_counts = best_model.data['response'].value_counts(normalize=True).sort_index()
simulated_response_counts = ppc_data['response_sampled'].value_counts(normalize=True).sort_index()

fig, ax = plt.subplots(figsize=(8,6))
ax.bar(real_response_counts.index - 0.2, real_response_counts.values, width=0.4, color='blue', label='Real Responses')
ax.bar(simulated_response_counts.index + 0.2, simulated_response_counts.values, width=0.4, color='red', label='Simulated Responses')
ax.legend(fontsize=11, facecolor='white', framealpha=1, edgecolor='black')
ax.tick_params(axis='both', which='major', labelsize=11)
ax.set_facecolor("white")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_color('black')
ax.spines['left'].set_color('black')
ax.set_xticks([0, 1])
ax.set_xticklabels(["Response 0", "Response 1"], fontsize=13)
ax.set_ylabel("Proportion", fontsize=13)
ax.set_title(f"Posterior Predictive Check - {best_model_name} - Response Proportions", fontsize=14)
response_plot_path = os.path.join(output_dir, f"Response_Proportions_{best_model_name}.png")
plt.savefig(response_plot_path, dpi=300, bbox_inches='tight')
plt.close(fig)

# Generate and Save Summary Statistics
print("Generating summary statistics with 800 samples per node...")
ppc_data_2 = hddm.utils.post_pred_gen(best_model, samples=2000)        # , samples=500
ppc_stats = hddm.utils.post_pred_stats(best_model.data, ppc_data_2)

print("Posterior predictive summary statistics:")
print(ppc_stats)

summary_stats_path = os.path.join(output_dir, f"posterior_predictive_summary_{best_model_name}.csv")
ppc_stats.to_csv(summary_stats_path)
print(f"Summary statistics saved to {summary_stats_path}")

del best_model, ppc_data, ppc_data_2
gc.collect()



