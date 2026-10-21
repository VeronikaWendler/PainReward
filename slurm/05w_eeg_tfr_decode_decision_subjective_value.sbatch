#!/bin/bash
#SBATCH --job-name=pr_dec_sv_tfr
#SBATCH --time=36:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=56G
#SBATCH --output=logs/tfr_decision/pr_dec_sv_tfr_%j.out
#SBATCH --error=logs/tfr_decision/pr_dec_sv_tfr_%j.err

set -Eeuo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source ./cluster_env.sh
mkdir -p logs/tfr_decision

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
    python /workspace/eeg/05w_eeg_tfr_decode_decision_subjective_value.py \
        --n-jobs "$SLURM_CPUS_PER_TASK" \
        --n-permutations 5000
