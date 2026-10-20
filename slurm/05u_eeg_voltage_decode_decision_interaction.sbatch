#!/bin/bash
#SBATCH --job-name=pr_dec_pxM_v
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/tfr_decision/pr_dec_interaction_voltage_%j.out
#SBATCH --error=logs/tfr_decision/pr_dec_interaction_voltage_%j.err

# ==============================================================================
# Cue-locked Pain x Money interaction: signed-voltage MVPA
#
# One SLURM job.
# Participants are processed sequentially.
# One stdout + one stderr file for the whole analysis.
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
echo "CUE-LOCKED PAIN x MONEY INTERACTION: SIGNED VOLTAGE"
echo "SLURM job ID:       $SLURM_JOB_ID"
echo "CPUs requested:     $SLURM_CPUS_PER_TASK"
echo "RUN_ROOT:           $RUN_ROOT"
echo "SOURCE_BIDS:        $SOURCE_BIDS"
echo "MNE container:      $IMAGE_MNE"
echo "============================================================"

apptainer exec --cleanenv \
    --bind "$CODE_ROOT:/workspace" \
    --bind "$RUN_ROOT:/data" \
    --bind "$SOURCE_BIDS:/source" \
    --env basepath="/data" \
    --env SOURCE_BIDS="/source" \
    --env MPLBACKEND="Agg" \
    --env PYTHONUNBUFFERED="1" \
    --env OMP_NUM_THREADS="1" \
    --env MKL_NUM_THREADS="1" \
    --env OPENBLAS_NUM_THREADS="1" \
    --env NUMEXPR_NUM_THREADS="1" \
    "$IMAGE_MNE" \
    python /workspace/eeg/05u_eeg_voltage_decode_decision_interaction.py \
        --n-jobs "$SLURM_CPUS_PER_TASK" \
        --n-permutations 5000

echo
echo "============================================================"
echo "SIGNED-VOLTAGE PAIN x MONEY INTERACTION JOB FINISHED"
echo "============================================================"
