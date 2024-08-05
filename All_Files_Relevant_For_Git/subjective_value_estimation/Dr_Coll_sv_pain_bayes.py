# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 12:18:54 2020
Updated on Thu Jun 11 2020

@author: Todd A. Vogel, McGill University

BAYESIAN CODE
Replication of Code from T.A.Vogel
"""
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 12:18:54 2020
Updated on Thu Jun 11 2020

@author: Todd A. Vogel, McGill University

BAYESIAN CODE
Replication of Code from T.A.Vogel

"""
import os
import numpy as np
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt
import pymc as pm

# Set up directories
current_directory = os.getcwd()
data_path = os.path.join(current_directory, 'data_sets')
if not os.path.exists(data_path):
    os.makedirs(data_path)

outpath = os.path.join(current_directory, "derivatives", "sv_modeling")
if not os.path.exists(outpath):
    os.makedirs(outpath)

file_path = os.path.join(data_path, 'subs_concatenated_001_050_5.csv')
df = pd.read_csv(file_path, sep=",")

df['moneylevel'] = pd.to_numeric(df['moneylevel'], errors='coerce')
df['painlevel'] = pd.to_numeric(df['painlevel'], errors='coerce')
df['response'] = pd.to_numeric(df['response'], errors='coerce')
df["participant"] = df['participant']
df = df.dropna(subset=['moneylevel', 'painlevel', 'response', 'participant', 'choice_resp.rt'])

# Pymc hierarchical model
def run_model(data, pain_func, n_samples=2000, n_tune=2000, n_cores=4):
    part_idx = pd.Categorical(data["participant"]).codes
    unique_parts = len(np.unique(part_idx))

    with pm.Model() as model:
        # Define the dimension 'part'
        model.add_coord('part', np.arange(unique_parts))

        # Indexes for participants
        p_idx = pm.Data("p_idx", part_idx)
        # Pain and money levels
        painlevel = data["painlevel"].values
        sv_money = data["moneylevel"].values

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
            sv_pain = pm.Deterministic("sv_pain", (1 / ((1 - k_pain[p_idx] * painlevel))) * 0.1)
        elif pain_func == "cubic":
            sv_pain = pm.Deterministic("sv_pain", (k_pain[p_idx] * (painlevel**3)) * 0.1)
        elif pain_func == "expo":
            sv_pain = pm.Deterministic("sv_pain", pm.math.exp(k_pain[p_idx] * painlevel * 0.1))
        elif pain_func == "root":
            sv_pain = pm.Deterministic("sv_pain", np.sqrt(k_pain[p_idx] * painlevel * 0.1))
        elif pain_func == "logarithmic":
            sv_pain = pm.Deterministic("sv_pain", np.log(k_pain[p_idx] * painlevel * 0.1))
        else:
            sv_pain = painlevel * 0.1

        # If no scaling, sv is not deterministic
        if pain_func != "none":
            sv_both = pm.Deterministic("sv_both", sv_pain - sv_money)
        else:
            sv_both = sv_money - painlevel

        p_pain = pm.Deterministic(
            "p_pain",
            1 / (1 + pm.math.exp(beta_param[p_idx] * (sv_both * 0.01))),
        )

        logit = pm.math.log(p_pain / (1 - p_pain))
        # Likelihood
        _ = pm.Bernoulli("accepted", logit_p=logit, observed=data["accepted"].values)

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

    # Calculate the LOO and WAIC for model comparison
    models_comp_dict_loo[pain_func] = az.loo(trace)
    models_comp_dict_waic[pain_func] = az.waic(trace)

    # Save values
    if pain_func != "none":  # if pain_func is not none, then we need to estimate k_pain and calculate sv_pain and sv_both
        part_ks = trace.posterior["k_pain"].data[0, :, :].mean(axis=0)
        df["sv_pain_" + pain_func] = trace.posterior["sv_pain"].data[0, :, :].mean(axis=0)
        df["sv_both_" + pain_func] = trace.posterior["sv_both"].data[0, :, :].mean(axis=0)
    else:  # if pain_func is none, then we don't need to estimate k_pain and we can just use the painlevel/moneylevel
        df["sv_pain_" + pain_func] = df["painlevel"]
        df["sv_both_" + pain_func] = df["painlevel"] - df["moneylevel"]
        part_ks = np.ones(len(df["participant"].unique()))
    part_betas = trace.posterior["beta_param"].data[0, :, :].mean(axis=0)

    for i, participant in enumerate(df["participant"].unique()):
        df.loc[df["participant"] == participant, "k_pain_" + pain_func] = part_ks[i]
        df.loc[df["participant"] == participant, "beta_" + pain_func] = part_betas[i]
        df.loc[df["participant"] == participant, "bias_" + pain_func] = part_betas[i]

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

# Model comparison
model_comp = az.compare(models_comp_dict_loo)
model_comp.to_csv(os.path.join(outpath, "model_comparison_loo.csv"))
model_comp = az.compare(models_comp_dict_waic)
model_comp.to_csv(os.path.join(outpath, "model_comparison_waic.csv"))
df.to_csv(os.path.join(outpath, "data_sv_modeling.csv"))      