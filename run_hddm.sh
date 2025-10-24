#!/bin/bash
#SBATCH --partition=compute                 # CPU partition (on MacLeod, not sure about Maxwell)
#SBATCH --cpus-per-task=8                   # number of CPU cores for chains
#SBATCH --mem=180G                          # total memory for the job
#SBATCH -o logs/slurm.%j.out                # STDOUT goes to this file
#SBATCH -e logs/slurm.%j.err                # STDERR goes to this file
#SBATCH --mail-type=ALL                     # email when job ends or fails
#SBATCH --mail-user=u04vw21@abdn.ac.uk      # university email (still Aberdeen)

#Singularity module 
module load singularity/3.8.5

export PYTHONUNBUFFERED=1                     # prints appear immediately
export MPLCONFIGDIR=/tmp/mplcache


# path for container and workspace
IMAGE=$HOME/containers/hddm_latest.sif
PROJECT=$HOME/sharedscratch/HDDM_Vero

export PROJECT_DIR=/workspace
export MPLBACKEND=Agg

# Run inside the container
singularity exec \
    --bind ${PROJECT}:/workspace \
    ${IMAGE} \
    python /workspace/CCT_MAP_Garcia.py

