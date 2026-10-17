#!/bin/bash
#SBATCH --job-name=pr_tfr_tot_evrem
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=logs/tfr/pr_tfr_tot_evrem_%j.log
#SBATCH --error=logs/tfr/pr_tfr_tot_evrem_%j.log
#SBATCH --mail-user=VAW508@student.bham.ac.uk
#SBATCH --mail-type=END,FAIL
set -Eeuo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source ./cluster_env.sh
mkdir -p logs/tfr
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
module purge
module load bb-singularity-conf/live
apptainer exec --cleanenv \
  --bind "$CODE_ROOT:/workspace" --bind "$RUN_ROOT:/data" \
  --env basepath="/data" --env PYTHONUNBUFFERED="1" \
  --env OMP_NUM_THREADS="1" --env MKL_NUM_THREADS="1" --env OPENBLAS_NUM_THREADS="1" --env NUMEXPR_NUM_THREADS="1" \
  "$IMAGE_MNE" \
  python /workspace/eeg/05i_eeg_compare_total_vs_evoked_removed.py \
    --n-jobs "$SLURM_CPUS_PER_TASK" --n-permutations 5000
