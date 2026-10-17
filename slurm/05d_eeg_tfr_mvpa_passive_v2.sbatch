#!/bin/bash
#SBATCH --job-name=pr_tfr_mvpa2
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/tfr/pr_tfr_mvpa2_%j.log
#SBATCH --error=logs/tfr/pr_tfr_mvpa2_%j.log
#SBATCH --mail-user=VAW508@student.bham.ac.uk
#SBATCH --mail-type=END,FAIL

set -Eeuo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source ./cluster_env.sh
mkdir -p logs/tfr

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

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
  python /workspace/eeg/05d_eeg_tfr_mvpa_passive_v2.py \
    --n-jobs "$SLURM_CPUS_PER_TASK" \
    --n-permutations 5000
