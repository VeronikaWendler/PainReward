#!/bin/bash
#SBATCH --job-name=pr_dec_mvpa
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=logs/tfr_decision/pr_dec_mvpa_%j.log
#SBATCH --error=logs/tfr_decision/pr_dec_mvpa_%j.log
set -Eeuo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source ./cluster_env.sh
mkdir -p logs/tfr_decision
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
module purge
module load bb-singularity-conf/live
apptainer exec --cleanenv \
  --bind "$CODE_ROOT:/workspace" \
  --bind "$RUN_ROOT:/data" \
  --env basepath=/data \
  --env MPLBACKEND=Agg \
  --env PYTHONUNBUFFERED=1 \
  --env OMP_NUM_THREADS=1 --env MKL_NUM_THREADS=1 --env OPENBLAS_NUM_THREADS=1 --env NUMEXPR_NUM_THREADS=1 \
  "$IMAGE_MNE" \
  python /workspace/eeg/05j_eeg_tfr_decode_decision_native.py --n-jobs "$SLURM_CPUS_PER_TASK" --n-permutations 5000
