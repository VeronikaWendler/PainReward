import os
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
PROJECT_DIR = Path(os.getenv("PROJECT_DIR", "/workspace"))

basepath = (
    PROJECT_DIR
    / "EEG"
    / "PainReward_sub-001-050"
    / "painrewardeegdata"
    / "derivatives"
)

# this should be the directory where version 3 (or 2) z-scored results live
massuni_dir = (
    basepath
    / "statistics_new"
    / "erps_massuni_drift_mod_9_RT_3GLMs"
    / "Zscoring"
)

# HDDM CSV with v_pain
mod_data_path = (
    PROJECT_DIR
    / "Hddm_Docker_August_24"
    / "figures_dir"
    / "painreward_behavioural_data_mod_9"
    / "diagnostics"
    / "v_pain_money_interaction.csv"
)

# ---------------------------------------------------------------------
# 1) Load subject beta maps from mass-univariate GLM
# ---------------------------------------------------------------------
# shape: (n_subj, n_reg, n_chan, n_time)
allbetas = np.load(massuni_dir / "ols_2ndlevel_betas.npy")

# subjects that actually entered the analysis
included_subjects = np.load(
    massuni_dir / "included_subjects.npy",
    allow_pickle=True
)

# We need channel names + times: grab them from any saved epochs
# here I assume you saved an epochs file for painlevel; adjust name if needed
epo_example = mne.read_epochs(
    massuni_dir / "ols_2ndlevel_allepochs-epo_painlevel.fif",
    preload=False
)

ch_names = epo_example.ch_names
times = epo_example.times


regvars = [
    "painlevel",
    "moneylevel",
    "interaction",
    "v_pain_contrib",
    "v_money_contrib",
    "v_interaction_contrib",
]

reg_idx_pain = regvars.index("painlevel")

# beta for painlevel: (n_subj, n_chan, n_time)
beta_pain = allbetas[:, reg_idx_pain, :, :]

# LPP ROI + time window
roi_chs = ["Fz", "FCz", "POz", "Cz", "CPz", "Pz", "Oz"]
picks = mne.pick_channels(ch_names, roi_chs)

tmin, tmax = 0.4, 0.8
tmask = (times >= tmin) & (times <= tmax)
beta_LPP_pain = beta_pain[:, picks][:, :, tmask].mean(axis=(1, 2))

# ---------------------------------------------------------------------
# 3) Load HDDM drift parameters (v_pain per subject)
# ---------------------------------------------------------------------
mod_data = pd.read_csv(mod_data_path, sep=None, engine="python")

# subject mean v_pain across trials
subj_v = (
    mod_data[mod_data["participant"].isin(included_subjects)]
    .groupby("participant")["v_painlevel_subj"]
    .mean()
    .reindex(included_subjects)  # align order with allbetas
)

v_pain = subj_v.to_numpy(dtype=float)



# mean_rt = (
#     mod_data[mod_data["participant"].isin(included_subjects)]
#     .groupby("participant")["rt"]
#     .mean()
#     .reindex(included_subjects)
#     .to_numpy(dtype=float)
# )
#
# X_cov = np.column_stack([np.ones(len(v_pain)), mean_rt])
# beta_cov, _, _, _ = np.linalg.lstsq(X_cov, v_pain, rcond=None)
# v_pain_resid = v_pain - X_cov @ beta_cov
#
# x = v_pain_resid
# x_name = "v_pain_resid"
# 
# # if you *don’t* residualise, just do:
x = v_pain
x_name = "v_pain"

# ---------------------------------------------------------------------
# 5) Correlation + save table
# ---------------------------------------------------------------------
r, p = pearsonr(x, beta_LPP_pain)
print(f"Correlation LPP β_pain (0.4–0.8s, LPP ROI) vs {x_name}: r = {r:.3f}, p = {p:.3g}")

out = pd.DataFrame({
    "participant": included_subjects,
    "beta_LPP_pain": beta_LPP_pain,
    "v_pain": v_pain,
    # "v_pain_resid": v_pain_resid,   # uncomment if using residuals
})

out.to_csv(massuni_dir / "LPP_pain_vs_vpain.csv", index=False)
print("Saved per-subject values to LPP_pain_vs_vpain.csv")
