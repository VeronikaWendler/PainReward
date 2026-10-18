#!/bin/bash
#SBATCH --job-name=pr_dec_evrem_prep
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=logs/tfr_decision/pr_dec_evrem_prep_%j.out
#SBATCH --error=logs/tfr_decision/pr_dec_evrem_prep_%j.err

# One sequential SLURM job.
# No job array.
# No participant-specific SLURM logs.

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
    find -L "$RUN_ROOT" \
        -maxdepth 1 \
        -mindepth 1 \
        -type d \
        -name 'sub-*' \
        -printf '%f\n' \
    | sort
)

N_SUBJECTS=${#SUBJECTS[@]}

if (( N_SUBJECTS == 0 )); then
    echo "ERROR: no sub-* participants found under $RUN_ROOT" >&2
    exit 1
fi

echo "============================================================"
echo "DECISION EVOKED-REMOVED TFR PREPARATION"
echo "Participants: $N_SUBJECTS"
echo "Job: $SLURM_JOB_ID"
echo "============================================================"

for participant_index in "${!SUBJECTS[@]}"; do
    participant="${SUBJECTS[$participant_index]}"

    echo
    echo "============================================================"
    echo "[$((participant_index + 1))/$N_SUBJECTS] $participant"
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
        python /workspace/eeg/04f_eeg_tfr_prep_decision_evoked_removed.py \
            --subject "$participant" \
            --n-jobs "$SLURM_CPUS_PER_TASK"

done

echo
echo "============================================================"
echo "DECISION EVOKED-REMOVED TFR PREPARATION COMPLETE"
echo "Processed $N_SUBJECTS participants"
echo "============================================================"
