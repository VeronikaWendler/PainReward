#!/bin/bash
#SBATCH --job-name=eeg_prep
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
IMAGE="$HOME/containers/hddm_latest.sif"
PROJECT="$HOME/projects/PainReward"  

# big data lives in project RDS
DATA_ROOT="/rds/projects/z/zhanglp-vwendler-core/PainReward_ULaval/EEG/PainReward_sub-001-050/painrewardeegdata"
RDS_OUT="/rds/projects/z/zhanglp-vwendler-core/PainReward_ULaval/derivatives/hddm"
mkdir -p "${RDS_OUT}"/{models,figures,logs}
#inside-container paths
export PROJECT_DIR="/workspace"
export DATA_DIR="/data"   
export OUT_DIR="/data/derivatives"

# bind code to /workspace, bind data to /data
apptainer exec --cleanenv \
  --bind "${PROJECT}:${PROJECT_DIR}" \
  --bind "${DATA_ROOT}:${DATA_DIR}" \
  --bind "${RDS_OUT}:/rds_out" \
  --bind "$HOME/pydeps_icalabel_only:/pydeps" \
  --env PYTHONPATH="/pydeps" \
  --env PROJECT_DIR="/workspace" \
  --env DATA_DIR="/data" \
  --env MODEL_DIR="/rds_out/models" \
  --env FIG_DIR="/rds_out/figures" \
  --env LOG_DIR="/rds_out/logs" \
  "${IMAGE}" \
  python "${PROJECT_DIR}/Hddm_Docker_August_24/DDM_EEG_load.py"


