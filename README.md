# PainReward — collaborator README

This repository contains code for the **PainReward** project at Université Laval. In practical terms, the repo is organized around three main analysis streams:

1. **EEG preprocessing and ERP / massunivariate and decoding analyses**
2. **HDDM / DDM behavioural modelling**
3. **A newer BayesFlow-based integrative modelling workflow - this is in process and just a try-out**

There are also a few older or exploratory notebooks and standalone scripts.

---

- **main HDDM modelling pipeline:** go to `Hddm_Docker_August_24/`
- **EEG preprocessing / ERP / mass-univariate / decoding analyses:** go to `EEG/`
- **EEG --> HDDM bridge dataset:** look at `EEG/rp_into_hddm.py`
- **You want older exploratory analyses / notebook history:** see `Initial_Preprocessing_and_Basic_Hddm/`, `Quest_Code/`, and `subjective_value_estimation/`

---

## repository map

### `EEG/`
Main EEG analysis folder. This is where most signal-processing and ERP analysis code lives

Key files:
- `eeg_preprocess.py` — raw EEG cleaning / preprocessing pipeline
- `eeg_erp_prep.py` — prepares ERP data for downstream analyses
- `eeg_erp_rp.py` — readiness-potential (RP) analyses and regression summaries
- `eeg_erp_massunivariate.py` — mass-univariate EEG analyses linking EEG with model-derived parameters
- `eeg_erp_massunivariate_plot.py` — plots topos and amplitudes and regression plots (e.g. differnece plots) for massunivariate
- `eeg_erp_massunivariate_stats.py` — post-processing / summarising statistically significant windows and channel effects
- `eeg_erp_groupplots_rp.py` — group-level RP figures
- **Files starting with eeg_erp_decoding are work in progress and messy claude-code trials**
- `eeg_erp_decoding_passive.py` — time-resolved decoding in passive phase
- `eeg_erp_decoding_decision.py` — time-resolved decoding in decision phase
- `eeg_erp_crossphase_decoding.py` — cross-phase time-generalisation decoding
- `mvpa_erp_decoding_searchlight.py` — spatiotemporal searchlight decoding 
- `rp_into_hddm.py` — creates an updated CSV by extracting RP means and merging them into the behavioural/HDDM dataset
- `DDM_image.py` — figure / illustration script for DDM figure in paper
- **IMPORTANT: These EEG files above are launched by the run_EEG.sh script**

Also inside `EEG/`:
- `MNE_tutorials/` — tutorial material / reference work - stuff for learning
- `Simulations/` — work in progress: This is for model ppc and parameter recovery of the neurocognitve models
- `mvpa_old/` — older MVPA code


---

### `Hddm_Docker_August_24/`
Main HDDM/DDM modelling folder. This is the large behavioural-model fitting pipeline, especially for cluster/container execution.

Key files:
- `DDM_EEG_load.py` — main pipeline to load the hddm models fit in the DDM_EEG.py files launched by the root `run_hddm.sh` script
- `DDM_EEG.py` — main pipeline to run the hddm models launched by the root `run_hddm.sh` script
- `MAP_estimates.py` — computes group MAP summaries and model-level parameter comparisons for selected model versions
- `helper_functions.py` — converts behavioural data into HDDM-compatible format
- `dataframe.py` — merges prepared behavioural data with fit/model information into a modelling dataframe
- `rename_files.py`, `rewrite_file.py` — utility scripts for file management / rewriting
- `data_sets/` — local behavioural CSV inputs used by the HDDM pipeline

Inside `data_sets/`:
- `behavioural_sv_cleaned_final_3.csv`
- `behavioural_sv_cleaned_final_3_rp.csv`
- `behavioural_sv_cleaned_try.csv`
- `old/` — older dataset versions


---

### `rp_bayesflow_workflow/`
This is a work-in-progress BayesFlow implementation of an truely integrative RP–drift model, where pain and money determine a latent drift variable, and that latent drift generates both signed RTs, choices and RP values - this has not been run yet

Key files:
- `config.py` — model parameter names, prior ranges, column defaults, training defaults
- `data_utils.py` — data loading, cleaning, signed-RT creation, subject design-bank creation, posterior summaries
- `simulator.py` — prior sampler, simple DDM simulator, latent-drift data simulator, BayesFlow batch simulator
- `model_utils.py` — amortizer and BayesFlow trainer setup
- `train_model.py` — model training
- `validate_recovery.py` — parameter-recovery workflow
- `fit_real_data.py` — posterior fitting for real subjects
- `permutation_test.py` — within-subject RP shuffling control analysis
- `posterior_predictive.py` — posterior predictive checks for RT and RP
- `plotting_utils.py` — training and recovery plots
- `requirements.txt` — local dependency list for this workflow
- `README.md` — folder-specific usage guide

Default expected columns for this workflow:
- `subj_idx`
- `pain_z`
- `money_z`
- `rp_z`
- `rt`
- `response`

The workflow creates subject design matrices from real data, trains amortized inference with BayesFlow, does recovery, posterior fitting, permutation tests, and posterior predictive checks - could be used as a thrid modle in our paper

---

### `subjective_value_estimation/`
Contains an older standalone **PyMC-based hierarchical subjective-value model**:
- `Coll_Vogel_sv_pain_bayes.py`

This script compares several candidate pain value transforms (`none`, `linear`, `para`, `expo`, `cubic`, `logarithmic`, `root`, `hyper`) and writes model-comparison outputs plus subject-level value summaries


---

### `Initial_Preprocessing_and_Basic_Hddm/`
Early notebook-based work:
- `First_DataFrame_Prep.ipynb`
- `First_hddm_painreward.ipynb`

This folder looks like the earliest data-prep + first-pass HDDM exploration

---

### `Quest_Code/`
Contains:
- `quest_analysis.ipynb`

notebook for questionnair-related analysis (not used)

---

## Important root-level files

- `run_EEG.sh` — SLURM / Apptainer launch script for EEG analysis on cluster infrastructure
- `run_hddm.sh` — SLURM / Apptainer launch script for HDDM analysis
- `run_bayesflow.sh` — SLURM launch script for BayesFlow training
- `running_images_cluster_info.txt` — notes on how container images were set up and run on the clusters
- `.gitignore`, `LICENSE` — standard repository stuf

---

## How the pieces connect

A useful mental model is:

1. **Behavioural data are cleaned/prepared**
   - older preparation helpers live in `Hddm_Docker_August_24/helper_functions.py` and related files
   - BayesFlow has its own cleaner in `rp_bayesflow_workflow/data_utils.py`

2. **EEG is preprocessed and transformed into ERP/RP features**
   - mainly in `EEG/eeg_preprocess.py`, `EEG/eeg_erp_prep.py`, and `EEG/eeg_erp_rp.py`

3. **RP features can be merged back into behavioural/HDDM inputs**
   - via `EEG/rp_into_hddm.py`

4. **Behavioural / joint modelling is then done in one of two main ways**
   - classic HDDM/DDM workflow in `Hddm_Docker_August_24/`
   - newer amortized BayesFlow workflow in `rp_bayesflow_workflow/` - not yet in use

5. **Group summaries and follow-up statistics / figures**
   - EEG statistics and figures are mainly in `EEG/`
   - HDDM summaries are in `MAP_estimates.py` and related scripts
   - BayesFlow summaries are produced by `validate_recovery.py`, `fit_real_data.py`, and `posterior_predictive.py`

---


### Most current / most structured
- `Hddm_Docker_August_24/`
- the main analysis scripts in `EEG/`

### Older / exploratory / reference
- `Initial_Preprocessing_and_Basic_Hddm/`
- `Quest_Code/`
- `subjective_value_estimation/`
- `EEG/mvpa_old/`


---

## Environment and reproducibility notes

A lot of this repo is designed for **cluster execution** 

Things to know before running anything:
- many scripts expect environment variables such as `PROJECT_DIR`, `DATA_DIR`, `OUT_DIR`, and `HDDM_DIR`
- several paths are hard-coded to RDS / cluster locations
- the root run scripts use **SLURM** and **Apptainer/Singularity**
- most EEG and HDDM workflows rely on container images (e.g. mne pythoin and hddm docker image) than a single shared local Python environment
