#!/bin/bash
#SBATCH --job-name=pr_tfr_prep
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=logs/tfr/pr_tfr_prep_%j.log
#SBATCH --mail-user=VAW508@student.bham.ac.uk
#SBATCH --mail-type=END,FAIL

# One SLURM job. Participants are processed sequentially.
# set -e means the job stops immediately on the first participant failure;
# the downstream QC job then remains blocked by afterok.
set -Eeuo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source ./cluster_env.sh

mapfile -t SUBJECTS < <(
  find -L "$RUN_ROOT" -maxdepth 1 -mindepth 1 -type d -name 'sub-*' -printf '%f\n' | sort
)

if (( ${#SUBJECTS[@]} == 0 )); then
  echo "ERROR: no sub-* directories found under RUN_ROOT=$RUN_ROOT" >&2
  exit 2
fi

if [[ ! -d "$SOURCE_BIDS/derivatives" ]]; then
  echo "ERROR: SOURCE_BIDS derivatives not found: $SOURCE_BIDS/derivatives" >&2
  exit 2
fi

module purge
module load bb-singularity-conf/live

echo "========================================================================"
echo "PASSIVE TFR PREPARATION"
echo "Participants: ${#SUBJECTS[@]}"
echo "Input source: $SOURCE_BIDS"
echo "Output root : $RUN_ROOT"
echo "Mode        : sequential (one participant at a time)"
echo "========================================================================"

for i in "${!SUBJECTS[@]}"; do
  SUBJECT="${SUBJECTS[$i]}"
  N=$((i + 1))

  echo
  echo "========================================================================"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] START ${SUBJECT} (${N}/${#SUBJECTS[@]})"
  echo "========================================================================"

  apptainer exec --cleanenv \
    --bind "$CODE_ROOT:/workspace" \
    --bind "$RUN_ROOT:/data" \
    --bind "$SOURCE_BIDS:/source:ro" \
    --env basepath="/data" \
    --env SOURCE_BIDS="/source" \
    --env PROJECT_DIR="/data" \
    --env HDDM_DIR="/data/derivatives/hddm" \
    --env NJOBS="$SLURM_CPUS_PER_TASK" \
    --env MPLBACKEND="Agg" \
    --env PYTHONUNBUFFERED="1" \
    "$IMAGE_MNE" \
    python /workspace/eeg/04b_eeg_tfr_prep.py \
      --subject "$SUBJECT" \
      --n-jobs "$SLURM_CPUS_PER_TASK"

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] DONE  ${SUBJECT} (${N}/${#SUBJECTS[@]})"
done

echo
echo "========================================================================"
echo "ALL ${#SUBJECTS[@]} PARTICIPANTS COMPLETED"
echo "========================================================================"
