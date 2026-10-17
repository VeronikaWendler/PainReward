#!/bin/bash
#SBATCH --job-name=pr_tfr_decpass
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=logs/tfr/pr_tfr_decpass_%j.log
#SBATCH --error=logs/tfr/pr_tfr_decpass_%j.log
#SBATCH --mail-user=VAW508@student.bham.ac.uk
#SBATCH --mail-type=END,FAIL

set -Eeuo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source ./cluster_env.sh
mkdir -p logs/tfr

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
  python /workspace/eeg/05c_eeg_tfr_decode_passive_within.py \
    --n-permutations 10000
