# Behavioural data preprocessing for HDDM modelling (PainReward task)
# Loads per-subject decision TSVs, encodes moneylevel, cleans data,
# applies acceptance-rate exclusion, and saves intermediate CSVs.

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from os.path import join as opj
import scipy.stats as stats
from statsmodels.formula.api import mixedlm


# Paths
basepath = str(os.getenv("basepath", Path(__file__).parent.parent.parent))
outpath = opj(basepath, "derivatives", "behav")
os.makedirs(outpath, exist_ok=True)

# 1. Load decision TSVs and concatenate across participants

cols_new_frame = [
    'participant', 'painstim', 'moneystim', 'all_trials_rewards', 'painlevel', 'painlevel_J',
    'leftstim', 'rightstim', 'acceptside', 'acceptkey', 'accepted', 'choice_resp.keys',
    'choice_resp.rt', 'blocks.thisRepN', 'trials.thisN', 'TaskName', 'P1', 'P2', 'P3', 'P4',
    'P5', 'fail_ma', 'fixduration'
]

participant_ids = [f"sub-{i:03d}" for i in range(1, 51) if i != 3]  # Exclude sub-003, does not have data

all_data = []
for participant in participant_ids:
    file_path = opj(basepath, participant, "eeg", f"{participant}_task-decision_beh.tsv")

    print(f"Processing {participant}...")
    data = pd.read_csv(file_path, sep="\t", usecols=cols_new_frame)

    # Parse the cumulative reward string from the last row and sum to a scalar total.
    reward_str = data['all_trials_rewards'].iloc[-1]
    total_reward = sum(int(x) for x in re.findall(r'\d+', str(reward_str)))
    data['all_trials_rewards'] = total_reward

    data = data.iloc[:125]

    all_data.append(data)
    print(data.shape)


# Shape: (n_participants * 125, n_columns)
preprocessed_data = pd.concat(all_data, ignore_index=True)

# Merge questionnaire data (STAI, PCS) from participants.tsv
questionnaire_data = pd.read_csv(opj(basepath, 'participants.tsv'), sep='\t')

questionnaire_data = questionnaire_data[['participant_id', 'STA_TAI_Score', 'STA_SAI_Score', 'PCS_Score']]

questionnaire_data.rename(columns={'participant_id': 'participant'}, inplace=True)
preprocessed_data = pd.merge(preprocessed_data, questionnaire_data, on='participant', how='left')

#


 
# 2. Recode moneystim ('m3' -> 3) to integer moneylevel

def convert_moneylevel(moneystim):
    if isinstance(moneystim, str) and moneystim.startswith('m'):
        return int(moneystim[1:])
    return None

preprocessed_data['moneylevel'] = preprocessed_data['moneystim'].apply(convert_moneylevel)

# 
# 3. Drop NaNs and cast to numeric

preprocessed_data = preprocessed_data.dropna(subset=['moneylevel', 'painlevel', 'accepted'])

for col in ['moneylevel', 'painlevel', 'accepted', 'choice_resp.rt']:
    preprocessed_data[col] = pd.to_numeric(preprocessed_data[col], errors='coerce')

# 6. Exclude participants accepting > 99% of trials (sub-003 has no data)

# Threshold: 95% of 125 trials
participant_ids_clean = [f"sub-{i:03d}" for i in range(1, 51) if i != 3]

def drop_participants(data, participant_ids, threshold=120):
    return [
        p for p in participant_ids
        if data[data['participant'] == p]['accepted'].sum() < threshold and data[data['participant'] == p]['accepted'].sum() > 125-threshold
    ]

included_participants = drop_participants(preprocessed_data, participant_ids_clean)
preprocessed_data = preprocessed_data[preprocessed_data['participant'].isin(included_participants)]

preprocessed_data.to_csv(opj(outpath, 'behav_cleaned_with_exclusions.csv'), sep=',', index=False)

print(f"Included participants after exclusion: {included_participants}")
print(f"Number of participants included: {len(included_participants)}")

# 4. Heatmap: mean RT by pain and money level (from cleaned dataset)


subject_means = (
    preprocessed_data
    .groupby(['participant', 'moneylevel', 'painlevel'], as_index=False)['choice_resp.rt']
    .mean()
)
agg_rt = (
    subject_means
    .groupby(['moneylevel', 'painlevel'], as_index=False)['choice_resp.rt']
    .agg(['mean', 'std'])
    .reset_index()
    .rename(columns={'mean': 'rt_mean', 'std': 'rt_sd'})
)

rt_pivot = agg_rt.pivot(index='painlevel', columns='moneylevel', values='rt_mean')
rt_pivot.index = rt_pivot.index.astype(int)

plt.figure(figsize=(4, 3))
ax = sns.heatmap(
    rt_pivot,
    cmap='viridis', cbar_kws={'label': 'Mean RT (s)'}
)
ax.set_xlabel('Money Level', fontsize=12)
ax.set_ylabel('Pain Level', fontsize=12)
ax.tick_params(labelsize=9)
ax.collections[0].colorbar.ax.tick_params(labelsize=9, pad=8)
ax.collections[0].colorbar.set_label('Response time (s)', fontsize=12, labelpad=15)

plt.tight_layout()

plt.savefig(opj(outpath, 'behav_heatmap_rt.svg'), dpi=800, transparent=True)
plt.close('all')

subject_rt = preprocessed_data.groupby('participant')['choice_resp.rt'].mean()
print(f"Mean RT: {subject_rt.mean():.3f} s  SD: {subject_rt.std():.3f} s")

# 5. Heatmap: acceptance rate by pain and money level

agg_acc = (
    preprocessed_data
    .groupby(['moneylevel', 'painlevel'], as_index=False)['accepted']
    .mean()
)

acc_pivot = agg_acc.pivot(index='painlevel', columns='moneylevel', values='accepted')
acc_pivot.index = acc_pivot.index.astype(int)

plt.figure(figsize=(4, 3))
ax = sns.heatmap(
    acc_pivot,
    cmap='cividis', cbar_kws={'label': 'Proportion accepted'}, vmin=0, vmax=1
)
ax.set_xlabel('Money Level', fontsize=12)
ax.set_ylabel('Pain Level', fontsize=12)
# Set label size for colorbar label
ax.tick_params(labelsize=9)
ax.collections[0].colorbar.ax.tick_params(labelsize=9, pad=8)
ax.collections[0].colorbar.set_label('Acceptance rate', fontsize=12, labelpad=15)
plt.tight_layout()

plt.savefig(opj(outpath, 'behav_heatmap_acceptance.svg'), dpi=800, transparent=True)
plt.close('all')

subject_acc = preprocessed_data.groupby('participant')['accepted'].mean()
print(f"Mean acceptance: {subject_acc.mean()*100:.1f}%  SD: {subject_acc.std()*100:.1f}%")
print(f"Acceptance range across subjects: {subject_acc.min()*100:.1f}% - {subject_acc.max()*100:.1f}%")

# Combined figure: RT and acceptance heatmaps side by side
fig, axes = plt.subplots(1, 2, figsize=(8, 3))

sns.heatmap(
    rt_pivot,
    cmap='plasma', ax=axes[0], cbar_kws={'label': 'Response time (s)'},
)
axes[0].set_xlabel('Money Level', fontsize=12)
axes[0].set_ylabel('Pain Level', fontsize=12)
axes[0].tick_params(labelsize=9)
# axes[0].set_title('Response Time', fontsize=14)
axes[0].collections[0].colorbar.ax.tick_params(labelsize=9, pad=8)
axes[0].collections[0].colorbar.set_label('Response time (s)', fontsize=12, labelpad=4)

sns.heatmap(
    acc_pivot,
    cmap='cividis', ax=axes[1], cbar_kws={'label': 'Acceptance rate'}, vmin=0, vmax=1,
)
axes[1].set_xlabel('Money Level', fontsize=12)
axes[1].set_ylabel('Pain Level', fontsize=12)
axes[1].tick_params(labelsize=9)
# axes[1].set_title('Acceptance Rate', fontsize=14)
axes[1].collections[0].colorbar.ax.tick_params(labelsize=9, pad=8)
axes[1].collections[0].colorbar.set_label('Acceptance rate', fontsize=12, labelpad=4)

plt.tight_layout()
fig_outpath = opj(basepath, "derivatives", "behav", "figures")
os.makedirs(fig_outpath, exist_ok=True)
fig.savefig(opj(fig_outpath, 'behav_heatmaps.svg'), dpi=800, transparent=True)
plt.close('all')


# Stats
# Mixed models: pain * money on acceptance and RT
# Fixed effects: painlevel, moneylevel, painlevel:moneylevel (z-scored)
# Random effects: random intercept + random slopes for painlevel and moneylevel by participant

# Z-score continuous predictors for interpretable fixed effects
# (main effects become effect at mean of other predictor, not at zero)
preprocessed_data = preprocessed_data.copy()
preprocessed_data['painlevel_z']  = (preprocessed_data['painlevel']  - preprocessed_data['painlevel'].mean())  / preprocessed_data['painlevel'].std()
preprocessed_data['moneylevel_z'] = (preprocessed_data['moneylevel'] - preprocessed_data['moneylevel'].mean()) / preprocessed_data['moneylevel'].std()

rt_data = preprocessed_data.dropna(subset=['choice_resp.rt', 'painlevel_z', 'moneylevel_z'])

rt_data.rename(columns={'choice_resp.rt': 'rt'}, inplace=True)

def extract_lmm_table(result, model_name):
    """Return a tidy fixed-effects summary DataFrame."""
    fe = result.fe_params
    fe_idx = fe.index
    # bse / tvalues / pvalues / conf_int include RE terms too — slice to FE only
    bse = result.bse[fe_idx]
    tvals = result.tvalues[fe_idx]
    pvals = result.pvalues[fe_idx]
    ci = result.conf_int().loc[fe_idx]
    table = pd.DataFrame({
        'model': model_name,
        'term': fe_idx,
        'estimate': fe.values,
        'se': bse.values,
        't': tvals.values,
        'p': pvals.values,
        'ci_low': ci.iloc[:, 0].values,
        'ci_high': ci.iloc[:, 1].values,
    })
    return table

# Model 1: RT ~ painlevel_z * moneylevel_z + (1 + painlevel_z + moneylevel_z | participant)
lmm_rt = mixedlm(
    "rt ~ painlevel_z * moneylevel_z",
    data=rt_data,
    groups=rt_data["participant"],
    re_formula="~painlevel_z + moneylevel_z",
).fit(reml=True)
print(lmm_rt.summary())
with open(opj(outpath, 'lmm_rt_summary.txt'), 'w') as f:
    f.write(str(lmm_rt.summary()))

# Model 2: accepted ~ painlevel_z * moneylevel_z + (1 + painlevel_z + moneylevel_z | participant)
lmm_acc = mixedlm(
    "accepted ~ painlevel_z * moneylevel_z",
    data=preprocessed_data,
    groups=preprocessed_data["participant"],
    re_formula="~painlevel_z + moneylevel_z",
).fit(reml=True)
print(lmm_acc.summary())
with open(opj(outpath, 'lmm_acc_summary.txt'), 'w') as f:
    f.write(str(lmm_acc.summary()))

# Save fixed-effects tables
lmm_results = pd.concat([
    extract_lmm_table(lmm_rt, "RT"),
    extract_lmm_table(lmm_acc, "Acceptance"),
], ignore_index=True)
lmm_results.to_csv(opj(outpath, 'lmm_pain_money_results.csv'), index=False)
print(f"\nLMM results saved to {opj(outpath, 'lmm_pain_money_results.csv')}")

# Decompose pain*money interaction on acceptance via simple slopes
# For each level of the moderator, simple slope = b_focal + b_interaction * moderator_z
# SE = sqrt(var(b_focal) + mod_z^2 * var(b_interaction) + 2*mod_z * cov(b_focal, b_interaction))

pain_mean  = preprocessed_data['painlevel'].mean()
pain_sd    = preprocessed_data['painlevel'].std()
money_mean = preprocessed_data['moneylevel'].mean()
money_sd   = preprocessed_data['moneylevel'].std()

fe_idx = lmm_acc.fe_params.index
vcov = lmm_acc.cov_params().loc[fe_idx, fe_idx]

money_levels = sorted(preprocessed_data['moneylevel'].dropna().unique().astype(int))
pain_levels  = sorted(preprocessed_data['painlevel'].dropna().unique().astype(int))

def simple_slopes(result, focal, moderator, mod_levels, mod_mean, mod_sd, vcov, label):
    """Simple slope of `focal` at each raw level of `moderator` (moderator expressed in raw units)."""
    rows = []
    b_focal   = result.fe_params[focal]
    inter_key = (f'{focal}:{moderator}' if f'{focal}:{moderator}' in result.fe_params.index
                 else f'{moderator}:{focal}')
    b_inter   = result.fe_params[inter_key]
    var_focal = vcov.loc[focal, focal]
    var_inter = vcov.loc[inter_key, inter_key]
    cov_fi    = vcov.loc[focal, inter_key]
    for lv in mod_levels:
        # moderator is already z-scored in the model; express each raw level as z-score
        mod_z = (lv - mod_mean) / mod_sd
        slope = b_focal + b_inter * mod_z
        se    = (var_focal + mod_z**2 * var_inter + 2 * mod_z * cov_fi) ** 0.5
        t     = slope / se
        p     = 2 * stats.t.sf(abs(t), df=result.df_resid)
        rows.append({
            'decomposition': label,
            'moderator_level': lv,
            'simple_slope': slope,
            'se': se,
            't': t,
            'p': p,
            'ci_low':  slope - 1.96 * se,
            'ci_high': slope + 1.96 * se,
        })
    return pd.DataFrame(rows)

ss_pain_by_money = simple_slopes(
    lmm_acc, 'painlevel_z', 'moneylevel_z',
    money_levels, money_mean, money_sd, vcov,
    label='pain at each money level',
)
ss_money_by_pain = simple_slopes(
    lmm_acc, 'moneylevel_z', 'painlevel_z',
    pain_levels, pain_mean, pain_sd, vcov,
    label='money at each pain level',
)

simple_slopes_results = pd.concat([ss_pain_by_money, ss_money_by_pain], ignore_index=True)
simple_slopes_results.to_csv(opj(outpath, 'lmm_acc_simple_slopes.csv'), index=False)
print(simple_slopes_results.to_string(index=False))
print(f"\nSimple slopes saved to {opj(outpath, 'lmm_acc_simple_slopes.csv')}")
plt.close('all')