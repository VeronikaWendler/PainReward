#!/bin/bash
#SBATCH --job-name=pr_resp_mvpa
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=logs/tfr_decision/pr_resp_mvpa_%j.out
#SBATCH --error=logs/tfr_decision/pr_resp_mvpa_%j.err

# ==============================================================================
# Response-locked decision TFR decoding.
#
# ONE SLURM job:
#   - all participants processed sequentially
#   - group cluster inference afterwards
#   - one stdout + one stderr file
# ==============================================================================

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

echo "============================================================"
echo "RESPONSE-LOCKED DECISION TFR DECODING"
echo "SLURM job ID:       $SLURM_JOB_ID"
echo "CPUs requested:     $SLURM_CPUS_PER_TASK"
echo "RUN_ROOT:           $RUN_ROOT"
echo "MNE container:      $IMAGE_MNE"
echo "============================================================"

apptainer exec --cleanenv \
    --bind "$CODE_ROOT:/workspace" \
    --bind "$RUN_ROOT:/data" \
    --env basepath="/data" \
    --env MPLBACKEND="Agg" \
    --env PYTHONUNBUFFERED="1" \
    --env OMP_NUM_THREADS="1" \
    --env MKL_NUM_THREADS="1" \
    --env OPENBLAS_NUM_THREADS="1" \
    --env NUMEXPR_NUM_THREADS="1" \
    "$IMAGE_MNE" \
    python /workspace/eeg/05p_eeg_tfr_decode_decision_response_locked.py \
        --n-jobs "$SLURM_CPUS_PER_TASK" \
        --n-permutations 5000

echo
echo "============================================================"
echo "RESPONSE-LOCKED DECODING JOB FINISHED"
echo "============================================================"
