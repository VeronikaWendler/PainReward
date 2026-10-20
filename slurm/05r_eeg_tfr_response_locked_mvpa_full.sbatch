#!/bin/bash
#SBATCH --job-name=pr_resp_mvpa2
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/tfr_decision/pr_resp_mvpa2_%j.out
#SBATCH --error=logs/tfr_decision/pr_resp_mvpa2_%j.err

# One job, participants sequentially, then group inference.
set -Eeuo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
source ./cluster_env.sh
mkdir -p logs/tfr_decision

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

module purge
module load bb-singularity-conf/live

apptainer exec --cleanenv \
    --bind "$CODE_ROOT:/workspace" \
    --bind "$RUN_ROOT:/data" \
    --env basepath="/data" \
    --env PYTHONUNBUFFERED="1" \
    --env OMP_NUM_THREADS="1" \
    --env MKL_NUM_THREADS="1" \
    --env OPENBLAS_NUM_THREADS="1" \
    --env NUMEXPR_NUM_THREADS="1" \
    "$IMAGE_MNE" \
    python /workspace/eeg/05r_eeg_tfr_response_locked_mvpa_full.py \
        --n-jobs "$SLURM_CPUS_PER_TASK" \
        --n-permutations 5000
