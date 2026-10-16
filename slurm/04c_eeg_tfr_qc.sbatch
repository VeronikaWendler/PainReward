#!/bin/bash
#SBATCH --job-name=pr_tfr_qc
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=logs/pr_tfr_qc_%j.out
#SBATCH --error=logs/pr_tfr_qc_%j.err

set -Eeuo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source ./cluster_env.sh

module purge
module load bb-singularity-conf/live

apptainer exec --cleanenv \
  --bind "$CODE_ROOT:/workspace" \
  --bind "$RUN_ROOT:/data" \
  --env basepath="/data" \
  --env PROJECT_DIR="/data" \
  --env HDDM_DIR="/data/derivatives/hddm" \
  --env MPLBACKEND="Agg" \
  --env PYTHONUNBUFFERED="1" \
  "$IMAGE_MNE" \
  python /workspace/eeg/04c_eeg_tfr_qc.py
