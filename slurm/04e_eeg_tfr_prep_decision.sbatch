#!/bin/bash
#SBATCH --job-name=pr_dec_tfr
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=logs/tfr_decision/pr_dec_tfr_%A_%a.log
#SBATCH --error=logs/tfr_decision/pr_dec_tfr_%A_%a.log
set -Eeuo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source ./cluster_env.sh
mkdir -p logs/tfr_decision
mapfile -t SUBJECTS < <(find "$RUN_ROOT" -maxdepth 1 -mindepth 1 -name 'sub-*' -printf '%f\n' | sort)
SUBJECT="${SUBJECTS[$SLURM_ARRAY_TASK_ID]}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
module purge
module load bb-singularity-conf/live
apptainer exec --cleanenv \
  --bind "$CODE_ROOT:/workspace" \
  --bind "$RUN_ROOT:/data" \
  --bind "$SOURCE_BIDS:/source:ro" \
  --env basepath=/data \
  --env SOURCE_BIDS=/source \
  --env PYTHONUNBUFFERED=1 \
  --env OMP_NUM_THREADS=1 --env MKL_NUM_THREADS=1 --env OPENBLAS_NUM_THREADS=1 --env NUMEXPR_NUM_THREADS=1 \
  "$IMAGE_MNE" \
  python /workspace/eeg/04e_eeg_tfr_prep_decision.py --subject "$SUBJECT" --n-jobs "$SLURM_CPUS_PER_TASK"
