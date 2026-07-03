#!/bin/bash
#SBATCH --job-name=pr_check_hddm
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=logs/pr_check_hddm_%j.out
#SBATCH --error=logs/pr_check_hddm_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source ./cluster_env.sh

module purge
module load bb-singularity-conf/live

apptainer exec --cleanenv \
  --bind "$CODE_ROOT:/workspace" \
  --bind "$RUN_ROOT:/data" \
  --env PROJECT_DIR="/data" \
  --env MODEL_DIR="/data/derivatives/hddm/models" \
  --env FIG_DIR="/data/derivatives/hddm/figures" \
  --env MPLBACKEND="Agg" \
  "$IMAGE_HDDM" \
  python -c "import hddm, kabuki, pandas, numpy, arviz, cloudpickle, dill, joblib; print('HDDM imports OK')"
