# BEAR compatible version
#!/bin/bash
#SBATCH --job-name=eeg_clean
#SBATCH --partition=compute
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=24:00:00
#SBATCH --output=logs/eeg_%j.out
#SBATCH --error=logs/eeg_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=VAW508@student.bham.ac.uk

set -euo pipefail
mkdir -p logs

# Modules for BEAR
module purge
module load Singularity   # on BEAR 

# Environment 
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mplcache"
mkdir -p "$MPLCONFIGDIR"

# need to get the mne image still
IMAGE="$HOME/containers/mne_latest.sif"
PROJECT="$HOME/projects/PainReward"   

# Inside container
export PROJECT_DIR=/workspace

# Run
singularity exec \
  --bind "${PROJECT}:/workspace" \
  "${IMAGE}" \
  python /workspace/EEG/eeg_erp_groupplots_cues.py

