#!/bin/bash
#SBATCH --job-name=pr_sv_audit
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=logs/sv_modeling/pr_sv_audit_%j.out
#SBATCH --error=logs/sv_modeling/pr_sv_audit_%j.err

set -Eeuo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
source ./cluster_env.sh

mkdir -p logs/sv_modeling

module purge
module load bb-singularity-conf/live

apptainer exec --cleanenv \
    --bind "$CODE_ROOT:/workspace" \
    --bind "$RUN_ROOT:/data" \
    --bind "$SOURCE_BIDS:/source" \
    --env basepath="/data" \
    --env SOURCE_BIDS="/source" \
    "$IMAGE_MNE" \
    python /workspace/behav/06a_acceptance_inclusion_audit.py
