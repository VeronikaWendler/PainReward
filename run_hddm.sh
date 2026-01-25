#!/bin/bash
#SBATCH --job-name=hddm_run
#SBATCH --partition=compute
#SBATCH --cpus-per-task=6
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm.%j.out
#SBATCH --error=logs/slurm.%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=<your_bham_email_here>

set -euo pipefail
mkdir -p logs

# Modules
module purge
module load Singularity   

# Env
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mplcache"
mkdir -p "$MPLCONFIGDIR"

# Paths 
IMAGE="$HOME/containers/hddm_latest.sif"
PROJECT="$HOME/projects/PainReward"   

# container workspace mountpoint
export PROJECT_DIR=/workspace

singularity exec \
  --bind "${PROJECT}:/workspace" \
  --bind "${TMPDIR:-/tmp}:/tmp" \
  "${IMAGE}" \
  python /workspace/Hddm_Docker_August_24/DDM_EEG_load.py

