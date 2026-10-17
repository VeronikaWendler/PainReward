#!/bin/bash
#SBATCH --job-name=pr_erp_mvpa
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=logs/tfr/pr_erp_mvpa_%j.log
#SBATCH --error=logs/tfr/pr_erp_mvpa_%j.log
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
  --bind "$CODE_ROOT:/workspace" --bind "$RUN_ROOT:/data" --bind "$SOURCE_BIDS:/source:ro" \
  --env basepath="/data" --env SOURCE_BIDS="/source" --env MPLBACKEND="Agg" --env PYTHONUNBUFFERED="1" \
  --env OMP_NUM_THREADS="1" --env MKL_NUM_THREADS="1" --env OPENBLAS_NUM_THREADS="1" --env NUMEXPR_NUM_THREADS="1" \
  "$IMAGE_MNE" \
  python /workspace/eeg/05h_eeg_erp_voltage_decode_passive.py --n-permutations 5000
