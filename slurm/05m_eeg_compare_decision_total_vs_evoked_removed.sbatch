#!/bin/bash
#SBATCH --job-name=pr_dec_total_vs_evrem
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --output=logs/tfr_decision/pr_dec_total_vs_evrem_%j.out
#SBATCH --error=logs/tfr_decision/pr_dec_total_vs_evrem_%j.err

# Direct paired total-power vs evoked-removed comparison.
# One job, one stdout file, one stderr file.

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
echo "DECISION TOTAL VS EVOKED-REMOVED COMPARISON"
echo "Job: $SLURM_JOB_ID"
echo "============================================================"

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
    python /workspace/eeg/05m_eeg_compare_decision_total_vs_evoked_removed.py \
        --n-jobs "$SLURM_CPUS_PER_TASK" \
        --n-permutations 5000

echo
echo "DECISION TOTAL VS EVOKED-REMOVED COMPARISON FINISHED"
