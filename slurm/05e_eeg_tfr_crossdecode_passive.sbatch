#!/bin/bash
#SBATCH --job-name=pr_tfr_cross
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/tfr/pr_tfr_cross_%j.log
#SBATCH --error=logs/tfr/pr_tfr_cross_%j.log
#SBATCH --mail-type=END,FAIL

set -Eeuo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source ./cluster_env.sh
mkdir -p logs/tfr

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
module purge
module load bb-singularity-conf/live

apptainer exec --cleanenv \
  --bind "$CODE_ROOT:/workspace" \
  --bind "$RUN_ROOT:/data" \
  --env basepath="/data" \
  --env MPLBACKEND="Agg" \
  --env PYTHONUNBUFFERED="1" \
  --env OMP_NUM_THREADS="1" \
  --env MKL_NUM_THREADS="1" \
  --env OPENBLAS_NUM_THREADS="1" \
  --env NUMEXPR_NUM_THREADS="1" \
  "$IMAGE_MNE" \
  python /workspace/eeg/05e_eeg_tfr_crossdecode_passive.py \
    --n-jobs "$SLURM_CPUS_PER_TASK" \
    --n-permutations 5000
