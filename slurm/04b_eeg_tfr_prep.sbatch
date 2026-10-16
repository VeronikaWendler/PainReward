#!/bin/bash
#SBATCH --job-name=pr_tfr_prep
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=logs/pr_tfr_prep_%A_%a.out
#SBATCH --error=logs/pr_tfr_prep_%A_%a.err
#SBATCH --mail-user=VAW508@student.bham.ac.uk


# IMPORTANT: submit this file as a job array via run_tfr_prep_cluster.sh.
set -Eeuo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source ./cluster_env.sh

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "ERROR: 04b_eeg_tfr_prep.sbatch must be submitted as a SLURM array." >&2
  exit 2
fi

mapfile -t SUBJECTS < <(
  find "$RUN_ROOT" -maxdepth 1 -mindepth 1 -type d -name 'sub-*' -printf '%f\n' | sort
)

if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= ${#SUBJECTS[@]} )); then
  echo "ERROR: array index $SLURM_ARRAY_TASK_ID outside subject list (n=${#SUBJECTS[@]})." >&2
  exit 2
fi

SUBJECT="${SUBJECTS[$SLURM_ARRAY_TASK_ID]}"
echo "Array task $SLURM_ARRAY_TASK_ID -> $SUBJECT"

module purge
module load bb-singularity-conf/live

apptainer exec --cleanenv \
  --bind "$CODE_ROOT:/workspace" \
  --bind "$RUN_ROOT:/data" \
  --env basepath="/data" \
  --env PROJECT_DIR="/data" \
  --env HDDM_DIR="/data/derivatives/hddm" \
  --env NJOBS="$SLURM_CPUS_PER_TASK" \
  --env MPLBACKEND="Agg" \
  --env PYTHONUNBUFFERED="1" \
  "$IMAGE_MNE" \
  python /workspace/eeg/04b_eeg_tfr_prep.py \
    --subject "$SUBJECT" \
    --n-jobs "$SLURM_CPUS_PER_TASK"
