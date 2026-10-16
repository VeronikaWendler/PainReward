#!/bin/bash
#SBATCH --job-name=pr_tfr_timing
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=logs/pr_tfr_timing_%j.out
#SBATCH --error=logs/pr_tfr_timing_%j.err
#SBATCH --mail-user=VAW508@student.bham.ac.uk


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
  python /workspace/eeg/04a_eeg_tfr_timing_audit.py --require-safe
