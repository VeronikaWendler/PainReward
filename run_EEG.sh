#!/bin/bash
#SBATCH --job-name=eeg_clean
#SBATCH --partition=compute                # same as HDDM
#SBATCH --cpus-per-task=8                  # adjust if needed
#SBATCH --mem=64G                          # EEG usually doesn't need 180G
#SBATCH --time=48:00:00                    # adjust depending on dataset size
#SBATCH -o logs/eeg_%j.out
#SBATCH -e logs/eeg_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=u04vw21@abdn.ac.uk

# Load Singularity module
module load singularity/3.8.5

# Environment variables
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export MPLCONFIGDIR=/tmp/mplcache

# Path to your container and project directory
IMAGE=$HOME/containers/mne_latest.sif
PROJECT=$HOME/sharedscratch/PainReward_ULaval

# Bind the EEG folder too (important!)
singularity exec \
    --bind ${PROJECT}:/workspace \
    ${IMAGE} \
    python /workspace/EEG/eeg_preprocess.py


