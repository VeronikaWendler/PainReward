#!/bin/bash
#SBATCH --job-name=eeg_erp_rp_groupplots
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --output=logs/eeg_%j.out
#SBATCH --error=logs/eeg_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=VAW508@student.bham.ac.uk


set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

# Modules for BEAR
module purge
module load bb-singularity-conf/live


# Environment 
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mplcache"
mkdir -p "$MPLCONFIGDIR"

# need to get the mne image still
IMAGE="$HOME/containers/mne_latest.sif"
PROJECT="$HOME/projects/PainReward"
DATA_HOST="/rds/projects/z/zhanglp-vwendler-core/PainReward_ULaval"
DATA_CONT="/pr"

export PROJECT_DIR="/workspace"
export DATA_DIR="${DATA_CONT}/EEG/PainReward_sub-001-050/painrewardeegdata"
export OUT_DIR="${DATA_CONT}/EEG/PainReward_sub-001-050/painrewardeegdata/derivatives"
export HDDM_DIR="${DATA_CONT}/derivatives/hddm"

apptainer exec --cleanenv \
  --bind "${PROJECT}:${PROJECT_DIR}" \
  --bind "${DATA_HOST}:${DATA_CONT}" \
  --bind "$HOME/pydeps_icalabel_only:/pydeps" \
  --env PYTHONPATH="/pydeps" \
  --env PROJECT_DIR="${PROJECT_DIR}" \
  --env DATA_DIR="${DATA_DIR}" \
  --env OUT_DIR="${OUT_DIR}" \
  --env HDDM_DIR="${HDDM_DIR}" \
  "${IMAGE}" \
  python "${PROJECT_DIR}/EEG/eeg_erp_crossphase_decoding.py"


