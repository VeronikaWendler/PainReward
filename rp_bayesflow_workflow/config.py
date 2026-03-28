"""Configuration for the true integrative RP–drift BayesFlow model.

Model idea
----------
Pain and money are fixed trial-wise design inputs.
They determine the mean of a latent *trial-wise drift* variable.
The latent trial-wise drift then generates both:
    1) the choice RT via a DDM simulator
    2) the observed single-trial RP value via a Gaussian measurement model

This makes the model *truly integrative* rather than directed:
RP is not a regressor on drift; instead, latent drift generates RP and behavior.
"""

from __future__ import annotations

PARAM_NAMES = [
    "v_intercept",      # baseline drift
    "v_pain",           # pain effect on mean drift
    "v_money",          # money effect on mean drift
    "boundary",         # boundary separation a
    "ndt",              # non-decision time t
    "drift_sd",         # across-trial SD of latent drift
    "rp_intercept",     # baseline RP
    "rp_loading",       # loading of latent drift onto RP
    "rp_noise",         # EEG measurement noise
]

PRIOR_LOW = (
    -3.0,   # v_intercept
    -3.0,   # v_pain
    -3.0,   # v_money
     0.4,   # boundary
     0.10,  # ndt
     0.05,  # drift_sd
    -2.0,   # rp_intercept
    -2.5,   # rp_loading
     0.05,  # rp_noise
)

PRIOR_HIGH = (
     3.0,   # v_intercept
     3.0,   # v_pain
     3.0,   # v_money
     2.5,   # boundary
     0.80,  # ndt
     1.50,  # drift_sd
     2.0,   # rp_intercept
     2.5,   # rp_loading
     1.50,  # rp_noise
)

DEFAULT_COLUMNS = {
    "subject": "subj_idx",
    "pain": "painlevel",
    "money": "moneylevel",
    "rp": "rp_z",
    "rt": "rt",
    "response": "response",
}

DEFAULT_TRAINING = {
    "epochs": 400,
    "batch_size": 32,
    "iterations_per_epoch": 500,
    "capacity": 300,
    "n_trials_min": 80,
    "n_trials_max": 220,
    "posterior_draws": 1000,
    "recovery_param_sets": 500,
}

DEFAULT_DDM = {
    "dt": 0.005,
    "noise_scale": 1.0,
}