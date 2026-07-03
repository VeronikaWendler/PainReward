#!/bin/bash
#SBATCH --job-name=pr_check_mne
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=logs/pr_check_mne_%j.out
#SBATCH --error=logs/pr_check_mne_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source ./cluster_env.sh

module purge
module load bb-singularity-conf/live

apptainer exec --cleanenv \
  --bind "$CODE_ROOT:/workspace" \
  --bind "$RUN_ROOT:/data" \
  --bind "$HOME/pydeps_icalabel_only:/pydeps_icalabel_only" --bind "$HOME/pydeps_pyprep_only:/pydeps_pyprep_only" \
  --env PYTHONPATH="/pydeps_pyprep_only:/pydeps_icalabel_only" \
  --env basepath="/data" \
  --env PROJECT_DIR="/data" \
  --env HDDM_DIR="/data/derivatives/hddm" \
  --env MPLBACKEND="Agg" \
  "$IMAGE_MNE" \
  python -c "import numpy, pandas, scipy, matplotlib, seaborn, statsmodels, sklearn, mne, pyprep, mne_icalabel; print('MNE/basic analysis imports OK')"
