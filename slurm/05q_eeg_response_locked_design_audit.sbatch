#!/bin/bash
#SBATCH --job-name=pr_resp_audit
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=logs/tfr_decision/pr_resp_audit_%j.out
#SBATCH --error=logs/tfr_decision/pr_resp_audit_%j.err

# Behavior/timing-only design audit. No EEG power values are read.
set -Eeuo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
source ./cluster_env.sh
mkdir -p logs/tfr_decision

module purge
module load bb-singularity-conf/live

apptainer exec --cleanenv \
    --bind "$CODE_ROOT:/workspace" \
    --bind "$RUN_ROOT:/data" \
    --env basepath="/data" \
    --env PYTHONUNBUFFERED="1" \
    "$IMAGE_MNE" \
    python /workspace/eeg/05q_eeg_response_locked_design_audit.py
