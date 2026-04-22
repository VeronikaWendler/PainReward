#!/usr/bin/env bash
# Run scripts in order (fail fast on any error).
set -Eeuo pipefail

trap 'echo "[run_all.sh] ERROR at line $LINENO: command failed: $BASH_COMMAND" >&2' ERR


PROJECT_DIR="/media/labmp/eSSD-004"
# # BEHAVIOUR
# python behav/01a_behav_decision.py # Decision behaviour and stats/figures
# python behav/01b_behav_passive.py # Passive behaviour and stats/figures
# python behav/02_behav_sv_modelling.py # Subjective value modelling

# # EEG
# python eeg/03_eeg_preprocess.py # EEG preprocessing
#python eeg/04_eeg_erp_prep.py # ERP analyses and group plots for RP

# HDDM MODELING
#python hddm/05_hddm_prep.py # Prepare data for HDDM modelling

# Run all models
# docker run --rm \
#   -v PROJECT_DIR:/project \
#   -e PROJECT_DIR=/project \
#   hcp4715/hddm \
#   python /project/code/hddm/06_hddm_fit.py

# PPC — posterior predictive checks for key models
# for version in 9 10 19; do
#   docker run --rm \
#     -v PROJECT_DIR:/project \
#     -e PROJECT_DIR=/project \
#     hcp4715/hddm \
#     python /project/code/hddm/Simulations/ppc.py --version $version
# done

# Parameter recovery — group and individual level
# for version in 9 10 19; do
#   docker run --rm \
#     -v /media/labmp/eSSD-004:/project \
#     -e PROJECT_DIR=/project \
#     hcp4715/hddm \
#     python /project/code/hddm/Simulations/param_recovery.py --version $version \
#       --n-reps 20 --samples 2000 --burn 200
# done

# Model recovery — can DIC distinguish the main behavioural models?
# docker run --rm \
#   -v /media/labmp/eSSD-004:/project \
#   -e PROJECT_DIR=/project \
#   hcp4715/hddm \
#   python /project/code/hddm/Simulations/model_recovery.py \
#     --versions 9 10 11 19 --n-reps 10 --samples 2000 --burn 200

# Collect HDDM results
# for version in 0 1 2 3 9 10 11 12 17 18 19 20; do
#   docker run --rm \
#     -v /media/labmp/eSSD-004:/project \
#     -e PROJECT_DIR=/project \
#     hcp4715/hddm \
#     python /project/code/hddm/07_hddm_results.py --version $version
# done

# Model comparison figure (reads DIC files written by the loop above)
# docker run --rm \
#   -v /media/labmp/eSSD-004:/project \
#   -e PROJECT_DIR=/project \
#   hcp4715/hddm \
#   python /project/code/hddm/07_hddm_results.py --compare


# EEG - 2 needs to be run after HDDM results are in, to plot ERPs by HDDM parameters
# python eeg/05a_eeg_erp_massunivariate_passive.py # Mass univariate ERP analyses
python eeg/05b_eeg_erp_massunivariate_decision.py # Mass univariate ERP analyses

