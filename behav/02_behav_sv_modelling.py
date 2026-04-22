# -*- coding: utf-8 -*-
"""
Hierarchical Bayesian subjective-value modelling (pain utility functions).
Based on Vogel et al. — fits 8 candidate pain-scaling functions and
compares them via LOO / WAIC.

Original author: Todd A. Vogel, McGill University
Adapted for PainReward project (Veronika Wendler / Michel-Pierre Coll).
"""
import os
import numpy as np
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt
import pymc as pm
from os.path import join as opj
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths — all relative to this script's location so the script can be run
# from any working directory.
# ---------------------------------------------------------------------------
basepath = str(os.getenv("basepath", Path(__file__).parent.parent.parent))
outpath = opj(basepath, "derivatives", "behav", "sv_modeling")
os.makedirs(outpath, exist_ok=True)

# Input: cleaned behavioural dataset produced by 01_behav.py
file_path = opj(basepath, "derivatives", "behav",
                         "behav_cleaned_with_exclusions.csv")

df = pd.read_csv(file_path)

df['moneylevel'] = pd.to_numeric(df['moneylevel'], errors='coerce')
df['painlevel'] = pd.to_numeric(df['painlevel'], errors='coerce')
df['accepted'] = pd.to_numeric(df['accepted'], errors='coerce').astype('Int64')
df = df.dropna(subset=['moneylevel', 'painlevel', 'participant', 'choice_resp.rt', 'accepted'])

# Pymc hierarchical model
def run_model(data, pain_func, n_samples=2000, n_tune=2000, n_cores=4):
    part_idx = pd.Categorical(data["participant"]).codes
    unique_parts = len(np.unique(part_idx))

    with pm.Model() as model:
        # Define the dimension 'part'
        model.add_coord('part', np.arange(unique_parts))

        # Indexes for participants
        p_idx = pm.Data("p_idx", part_idx)
        # Pain and money levels (both scaled by 0.1 to match sv_pain range)
        painlevel = data["painlevel"].values
        sv_money = data["moneylevel"].values * 0.1

        # Temperature parameter
        sigmabeta = pm.HalfNormal("sigma_beta", sigma=3)
        beta_param = pm.HalfNormal("beta_param", sigma=sigmabeta, dims="part")

        # Value functions
        if pain_func != "none":  # if pain_func is not none, then we need to estimate k_pain
            # Scaling parameter
            sigmak = pm.HalfNormal("sigmak", sigma=3)
            k_pain = pm.HalfNormal("k_pain", sigma=sigmak, dims="part")

        # Scaling functions (with 0.1 scaling to avoid numerical issues)
        if pain_func == "linear":
            sv_pain = pm.Deterministic("sv_pain", (k_pain[p_idx] * painlevel * 0.1))
        elif pain_func == "para":
            sv_pain = pm.Deterministic("sv_pain", (k_pain[p_idx] * (painlevel**2) * 0.1))
        elif pain_func == "hyper":
            sv_pain = pm.Deterministic("sv_pain", (1 / ((1 + k_pain[p_idx] * painlevel))) * 0.1)
        elif pain_func == "cubic":
            sv_pain = pm.Deterministic("sv_pain", (k_pain[p_idx] * (painlevel**3)) * 0.1)
        elif pain_func == "expo":
            sv_pain = pm.Deterministic("sv_pain", pm.math.exp(k_pain[p_idx] * painlevel * 0.1))
        elif pain_func == "root":
            sv_pain = pm.Deterministic("sv_pain", pm.math.sqrt(k_pain[p_idx] * painlevel * 0.1))
        elif pain_func == "logarithmic":
            sv_pain = pm.Deterministic("sv_pain", pm.math.log(k_pain[p_idx] * painlevel * 0.1))
        else:
            sv_pain = painlevel * 0.1

        if pain_func != "none":
            sv_both = pm.Deterministic("sv_both", sv_pain - sv_money)
        else:
            sv_both = sv_pain - sv_money  # sv_pain = painlevel * 0.1 for "none"

        # Likelihood: logit_p = -beta * sv_both * 0.1
        # (sv_pain and sv_money are both on the 0.1x scale, so 0.1 rather than 0.01 keeps logit magnitudes comparable)
        _ = pm.Bernoulli("accepted", logit_p=-beta_param[p_idx] * sv_both * 0.1, observed=data["accepted"].values)

        trace = pm.sample(
            n_samples,
            tune=n_tune,
            return_inferencedata=True,
            cores=n_cores,
            progressbar=True,
            idata_kwargs={"log_likelihood": True},
            target_accept=0.95,
        )

    return model, trace


# Set the function names to be fit for pain
pain_models = [
    "none",
    "linear",
    "para",
    "expo",
    "cubic",
    "logarithmic",
    "root",
    "hyper",
]

# Create empty dictionaries to store the model stats
models_comp_dict_loo = {}
models_comp_dict_waic = {}

# Loop through all possible combinations of models
for pain_func in pain_models:

    # Run the model
    model, trace = run_model(df, pain_func, n_samples=2000, n_tune=3000)

    # Calculate the LOO and WAIC for model comparison (pointwise=True needed for diagnostics)
    loo_pw = az.loo(trace, pointwise=True)
    waic_pw = az.waic(trace, pointwise=True)
    models_comp_dict_loo[pain_func] = loo_pw
    models_comp_dict_waic[pain_func] = waic_pw

    # Save values
    # pd.Categorical sorts categories alphabetically, so participant i in the
    # model corresponds to the i-th entry in cat.categories — use that same
    # ordering when writing per-participant values back to the dataframe.
    cat_parts = pd.Categorical(df["participant"]).categories  # sorted alphabetically

    if pain_func != "none":  # if pain_func is not none, then we need to estimate k_pain and calculate sv_pain and sv_both
        part_ks = trace.posterior["k_pain"].mean(dim=["chain", "draw"]).values
        df["sv_pain_" + pain_func] = trace.posterior["sv_pain"].mean(dim=["chain", "draw"]).values
        df["sv_both_" + pain_func] = trace.posterior["sv_both"].mean(dim=["chain", "draw"]).values
    else:  # if pain_func is none, then we don't need to estimate k_pain and we can just use the painlevel/moneylevel
        df["sv_pain_" + pain_func] = df["painlevel"]
        df["sv_both_" + pain_func] = df["painlevel"] - df["moneylevel"]
        part_ks = np.ones(len(cat_parts))
    part_betas = trace.posterior["beta_param"].mean(dim=["chain", "draw"]).values

    for i, participant in enumerate(cat_parts):
        df.loc[df["participant"] == participant, "k_pain_" + pain_func] = part_ks[i]
        df.loc[df["participant"] == participant, "beta_" + pain_func] = part_betas[i]

    # Save the trace and summary
    if pain_func == "none":
        az.plot_trace(
            trace,
            var_names=[
                "beta_param",
                "sigma_beta",
            ],
            combined=True,
        )
    else:
        az.plot_trace(
            trace,
            var_names=[
                "beta_param",
                "k_pain",
                "sigma_beta",
                "sigmak",
            ],
        )

    # Make some plots
    plt.savefig(os.path.join(outpath, pain_func + "_trace_plot.png"))
    plt.close()
    
    # # Check if graphviz is installed, if not, skip the graph visualization
    # try:
    #     pm.model_to_graphviz(model).render(os.path.join(outpath, pain_func + "_model"), cleanup=True)
    # except ImportError:
    #     print(f"Graphviz not installed. Skipping graph visualization for {pain_func}.")

    summary = az.summary(trace)
    summary.to_csv(os.path.join(outpath, pain_func + "_summary.csv"))

    # --- WAIC / LOO diagnostics ---
    # Compute per-observation variance of log-predictive densities (the WAIC warning quantity)
    log_lik = trace.log_likelihood["accepted"]
    var_lpd = log_lik.stack(samples=["chain", "draw"]).var("samples").values
    n_flagged = int((var_lpd > 0.4).sum())

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"WAIC/LOO diagnostics — {pain_func}", fontsize=13)

    # Left: per-observation WAIC variance; red line = warning threshold
    axes[0].scatter(range(len(var_lpd)), var_lpd, alpha=0.3, s=8)
    axes[0].axhline(0.4, color="red", linestyle="--", label="Warning threshold (0.4)")
    axes[0].set_xlabel("Observation index")
    axes[0].set_ylabel("Var(log p(y | θ))")
    axes[0].set_title("Per-observation WAIC variance")
    axes[0].annotate(
        f"{n_flagged}/{len(var_lpd)} obs > 0.4",
        xy=(0.98, 0.95), xycoords="axes fraction", ha="right", va="top",
    )
    axes[0].legend()

    # Right: LOO Pareto-k values (k > 0.5 suspect, > 0.7 bad)
    az.plot_khat(loo_pw, ax=axes[1], show_bins=True)
    axes[1].set_title("LOO Pareto-k values")

    plt.tight_layout()
    plt.savefig(os.path.join(outpath, pain_func + "_waic_loo_diagnostics.png"), dpi=150)
    plt.close()

# Model comparison
model_comp = az.compare(models_comp_dict_loo)
model_comp.to_csv(os.path.join(outpath, "model_comparison_loo.csv"))
model_comp = az.compare(models_comp_dict_waic)
model_comp.to_csv(os.path.join(outpath, "model_comparison_waic.csv"))
df.to_csv(opj(basepath, "derivatives", "behav", "behav_with_exclusion_sv_modeling.csv"))      