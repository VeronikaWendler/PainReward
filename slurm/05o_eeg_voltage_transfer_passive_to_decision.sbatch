#!/bin/bash
#SBATCH --job-name=pr_p2d_voltage
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=logs/tfr_decision/pr_p2d_voltage_%j.out
#SBATCH --error=logs/tfr_decision/pr_p2d_voltage_%j.err

# ==============================================================================
# Passive -> decision signed-voltage transfer
#
# ONE SLURM job. No participant arrays.
# ONE stdout and ONE stderr file for the whole analysis.
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
echo "PASSIVE -> DECISION SIGNED-VOLTAGE TRANSFER"
echo "Job ID: $SLURM_JOB_ID"
echo "CPUs:   $SLURM_CPUS_PER_TASK"
echo "============================================================"

apptainer exec --cleanenv \
    --bind "$CODE_ROOT:/workspace" \
    --bind "$RUN_ROOT:/data" \
    --bind "$SOURCE_BIDS:/source:ro" \
    --env basepath="/data" \
    --env SOURCE_BIDS="/source" \
    --env MPLBACKEND="Agg" \
    --env PYTHONUNBUFFERED="1" \
    --env OMP_NUM_THREADS="1" \
    --env MKL_NUM_THREADS="1" \
    --env OPENBLAS_NUM_THREADS="1" \
    --env NUMEXPR_NUM_THREADS="1" \
    "$IMAGE_MNE" \
    python /workspace/eeg/05o_eeg_voltage_transfer_passive_to_decision.py \
        --n-jobs "$SLURM_CPUS_PER_TASK" \
        --n-permutations 5000

echo
echo "PASSIVE -> DECISION SIGNED-VOLTAGE TRANSFER JOB FINISHED"
