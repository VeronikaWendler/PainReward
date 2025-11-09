#!/bin/bash
#SBATCH --job-name=eeg_clean
#SBATCH --partition=compute                
#SBATCH --cpus-per-task=8                  
#SBATCH --mem=180G                          # EEG 
#SBATCH --time=24:00:00                    
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

export PROJECT_DIR=/workspace
export PATH=$HOME/.local/bin:$PATH

# Bind the EEG folder too
singularity exec \
    --bind ${PROJECT}:/workspace \
    ${IMAGE} \
    python /workspace/EEG/eeg_erp_prep.py


