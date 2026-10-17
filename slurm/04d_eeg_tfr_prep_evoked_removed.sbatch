#!/bin/bash
#SBATCH --job-name=pr_tfr_evrem
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --output=logs/tfr/pr_tfr_evrem_%j.log
#SBATCH --error=logs/tfr/pr_tfr_evrem_%j.log
#SBATCH --mail-user=VAW508@student.bham.ac.uk
#SBATCH --mail-type=END,FAIL
set -Eeuo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source ./cluster_env.sh
mkdir -p logs/tfr
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
module purge
module load bb-singularity-conf/live
mapfile -t SUBS < <(awk 'NR>1 && $1 ~ /^sub-/ {print $1}' "$RUN_ROOT/participants.tsv")
echo "Sequential evoked-removed TFR prep for ${#SUBS[@]} participants"
for i in "${!SUBS[@]}"; do
  sub="${SUBS[$i]}"
  echo "=============================================================================="
  echo "[$((i+1))/${#SUBS[@]}] $sub"
  apptainer exec --cleanenv \
    --bind "$CODE_ROOT:/workspace" \
    --bind "$RUN_ROOT:/data" \
    --bind "$SOURCE_BIDS:/source:ro" \
    --env basepath="/data" --env SOURCE_BIDS="/source" \
    --env PYTHONUNBUFFERED="1" \
    --env OMP_NUM_THREADS="1" --env MKL_NUM_THREADS="1" \
    --env OPENBLAS_NUM_THREADS="1" --env NUMEXPR_NUM_THREADS="1" \
    "$IMAGE_MNE" \
    python /workspace/eeg/04d_eeg_tfr_prep_evoked_removed.py \
      --subject "$sub" --n-jobs "$SLURM_CPUS_PER_TASK"
done
echo "ALL EVOKED-REMOVED TFR PREP COMPLETE"
