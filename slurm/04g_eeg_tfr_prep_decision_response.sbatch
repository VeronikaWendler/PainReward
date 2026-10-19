#!/bin/bash
#SBATCH --job-name=pr_resp_tfr
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --output=logs/tfr_decision/pr_resp_tfr_%j.out
#SBATCH --error=logs/tfr_decision/pr_resp_tfr_%j.err

# ==============================================================================
# Response-locked TFR preparation for all PainReward participants.
#
# IMPORTANT:
# - one sequential SLURM job
# - no participant job array
# - one stdout + one stderr file for the entire participant loop
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

mapfile -t PARTICIPANTS < <(
    find -L "$RUN_ROOT" \
        -maxdepth 1 \
        -mindepth 1 \
        -type d \
        -name 'sub-*' \
        -printf '%f\n' \
        | sort
)

if [[ "${#PARTICIPANTS[@]}" -eq 0 ]]; then
    echo "ERROR: no sub-* participants found under $RUN_ROOT" >&2
    exit 1
fi

echo "============================================================"
echo "RESPONSE-LOCKED TFR PREPARATION"
echo "Job ID:       $SLURM_JOB_ID"
echo "Participants: ${#PARTICIPANTS[@]}"
echo "CPUs:         $SLURM_CPUS_PER_TASK"
echo "============================================================"

for INDEX in "${!PARTICIPANTS[@]}"; do
    SUBJECT="${PARTICIPANTS[$INDEX]}"

    echo
    echo "[$((INDEX + 1))/${#PARTICIPANTS[@]}] $SUBJECT"

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
        python /workspace/eeg/04g_eeg_tfr_prep_decision_response.py \
            --subject "$SUBJECT" \
            --n-jobs "$SLURM_CPUS_PER_TASK"

    echo "[$SUBJECT] COMPLETE"
done

echo
echo "============================================================"
echo "RESPONSE-LOCKED TFR PREPARATION COMPLETE"
echo "Participants completed: ${#PARTICIPANTS[@]}"
echo "============================================================"
