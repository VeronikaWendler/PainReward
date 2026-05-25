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
figpath = opj(basepath, "derivatives", "behav", "figures")
os.makedirs(figpath, exist_ok=True)

sns.set_theme(style="ticks", context="paper")

# 1. Load decision TSVs and concatenate across participants

cols_new_frame = [
    'participant', 'painstim', 'moneystim', 'all_trials_rewards', 'painlevel', 'painlevel_J',
    'leftstim', 'rightstim', 'acceptside', 'acceptkey', 'accepted', 'choice_resp.keys',
    'choice_resp.rt', 'blocks.thisRepN', 'trials.thisN', 'TaskName', 'P1', 'P2', 'P3', 'P4',
    'P5', 'fail_ma', 'fixduration'
]

# Load decision output to get exclusion list
decision_file = pd.read_csv(opj(basepath, "derivatives", "behav", "behav_cleaned_with_exclusions.csv"))

participant_ids = decision_file['participant'].unique()

all_data = []
for participant in participant_ids:
    file_path = opj(basepath, participant, "eeg", f"{participant}_task-passive_beh.tsv")

    data = pd.read_csv(file_path, sep="\t")

    all_data.append(data[:200])
    print(data.shape)

# Shape: (n_participants * 125, n_columns)
preprocessed_data = pd.concat(all_data, ignore_index=True)

# Merge questionnaire data (STAI, PCS) from participants.tsv
questionnaire_data = pd.read_csv(opj(basepath, 'participants.tsv'), sep='\t')

questionnaire_data = questionnaire_data[['participant_id', 'STA_TAI_Score', 'STA_SAI_Score', 'PCS_Score']]

questionnaire_data.rename(columns={'participant_id': 'participant'}, inplace=True)
preprocessed_data = pd.merge(preprocessed_data, questionnaire_data, on='participant', how='left')

#

# Check and plot correlation between pain level and rating
data_catch_pain = preprocessed_data[~preprocessed_data['catch_pain'].isna()]

level_rate_correlation = []
for participant in preprocessed_data['participant'].unique():
    participant_data = preprocessed_data[preprocessed_data['participant'] == participant]

    # Keep only catch pain trials
    participant_data = participant_data[~participant_data['catch_pain'].isna()]

    # Calculate rank correlation
    corr, p_value = stats.spearmanr(participant_data['catch_pain'], participant_data['level'])

    level_rate_correlation.append({
        'participant': participant,
        'correlation': corr,
        'p_value': p_value,
    })

level_rate_correlation = pd.DataFrame(level_rate_correlation)

level_rate_correlation.to_csv(opj(outpath, "catch_pain_level_rating_correlation.csv"), index=False)
level_rate_correlation.describe().to_csv(opj(outpath, "catch_pain_level_rating_correlation_summary.csv"), index=True)

# Same with catch money
data_catch_money = preprocessed_data[~preprocessed_data['catch_money'].isna()]

# Calculate correlation between money level and rating
money_rate_correlation = []
for participant in preprocessed_data['participant'].unique():
    participant_data = preprocessed_data[preprocessed_data['participant'] == participant]

    # Keep only catch money trials
    participant_data = participant_data[~participant_data['catch_money'].isna()]

    # Calculate accuracy between response (catch money) and level
    accuracy = (participant_data['catch_money'] == participant_data['level']/100*5).mean()

    money_rate_correlation.append({
        'participant': participant,
        'accuracy': accuracy,
    })

money_rate_correlation = pd.DataFrame(money_rate_correlation)

money_rate_correlation.to_csv(opj(outpath, "catch_money_accuracy.csv"), index=False)
money_rate_correlation.describe().to_csv(opj(outpath, "catch_money_accuracy_summary.csv"), index=True)

# Combined figure: passive-phase catch-trial quality checks
fig, axes = plt.subplots(1, 3, figsize=(9, 3))

# Panel A: distribution of pain level–rating Spearman correlations across participants
ax = axes[0]
sns.boxplot(
    y=level_rate_correlation['correlation'],
    ax=ax,
    showfliers=False,
    color="steelblue",
    width=0.4,
)
sns.stripplot(
    y=level_rate_correlation['correlation'],
    ax=ax,
    color="black",
    alpha=0.5,
    size=3,
    jitter=True,
)
ax.set_xlabel("")
ax.set_xticks([])
ax.set_ylabel("Spearman correlation", fontsize=12)
ax.set_title("Pain level–rating\ncorrelation", fontsize=12)
ax.tick_params(labelsize=8)
ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
sns.despine(ax=ax, bottom=True)

# Panel B: per-participant regression of pain rating on pain level
ax = axes[1]
for participant in preprocessed_data['participant'].unique():
    participant_data = preprocessed_data[preprocessed_data['participant'] == participant]
    participant_data = participant_data[~participant_data['catch_pain'].isna()]
    sns.regplot(
        x=participant_data['level'],
        y=participant_data['catch_pain'],
        ax=ax,
        scatter=False,
        line_kws={"linewidth": 0.8, "alpha": 0.5},
        ci=None,
    )
ax.set_xlabel("Pain level (stimulus intensity)", fontsize=12)
ax.set_ylabel("Pain rating", fontsize=12)
ax.set_title("Pain rating vs. level\n(per participant)", fontsize=12)
ax.tick_params(labelsize=8)
sns.despine(ax=ax)

# Panel C: distribution of catch money accuracy across participants
ax = axes[2]
sns.boxplot(
    y=money_rate_correlation['accuracy'],
    ax=ax,
    showfliers=False,
    color="goldenrod",
    width=0.4,
)
sns.stripplot(
    y=money_rate_correlation['accuracy'],
    ax=ax,
    color="black",
    alpha=0.5,
    size=3,
    jitter=True,
)
ax.set_xlabel("")
ax.set_xticks([])
ax.set_ylabel("Accuracy", fontsize=12)
ax.set_ylim(0, 1.05)
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
ax.tick_params(labelsize=8)
ax.set_title("Catch money\ntrial accuracy", fontsize=12)
sns.despine(ax=ax, bottom=True)

fig.tight_layout()
fig.savefig(opj(figpath, "behav_passive_catch_checks.svg"), format="svg", bbox_inches="tight")
plt.close(fig)