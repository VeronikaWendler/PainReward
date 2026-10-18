#!/bin/bash
#SBATCH --job-name=pr_dec_tfr
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=logs/tfr_decision/pr_dec_tfr_%j.out
#SBATCH --error=logs/tfr_decision/pr_dec_tfr_%j.err

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

mapfile -t SUBJECTS < <(
    find "$RUN_ROOT" \
        -maxdepth 1 \
        -mindepth 1 \
        -type d \
        -name 'sub-*' \
        -printf '%f\n' \
    | sort
)

N=${#SUBJECTS[@]}

if (( N == 0 )); then
    echo "ERROR: no sub-* directories found under $RUN_ROOT" >&2
    exit 1
fi

echo "============================================================"
echo "DECISION TFR PREPARATION"
echo "Participants: $N"
echo "RUN_ROOT:     $RUN_ROOT"
echo "SOURCE_BIDS:  $SOURCE_BIDS"
echo "============================================================"
echo

for i in "${!SUBJECTS[@]}"; do

    SUBJECT="${SUBJECTS[$i]}"

    echo
    echo "============================================================"
    echo "[$((i+1))/$N] $SUBJECT"
    echo "============================================================"

    apptainer exec --cleanenv \
        --bind "$CODE_ROOT:/workspace" \
        --bind "$RUN_ROOT:/data" \
        --bind "$SOURCE_BIDS:/source:ro" \
        --env basepath="/data" \
        --env SOURCE_BIDS="/source" \
        --env PYTHONUNBUFFERED="1" \
        --env OMP_NUM_THREADS="1" \
        --env MKL_NUM_THREADS="1" \
        --env OPENBLAS_NUM_THREADS="1" \
        --env NUMEXPR_NUM_THREADS="1" \
        "$IMAGE_MNE" \
        python /workspace/eeg/04e_eeg_tfr_prep_decision.py \
            --subject "$SUBJECT" \
            --n-jobs "$SLURM_CPUS_PER_TASK"

    echo
    echo "[$SUBJECT] DONE"
done

echo
echo "============================================================"
echo "DECISION TFR PREPARATION COMPLETE"
echo "Processed $N participants"
echo "============================================================"