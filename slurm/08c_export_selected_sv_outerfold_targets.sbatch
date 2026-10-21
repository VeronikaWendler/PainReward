#!/bin/bash
#SBATCH --job-name=pr_sv_eeg_targets
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --output=logs/sv_modeling/pr_sv_eeg_targets_%j.out
#SBATCH --error=logs/sv_modeling/pr_sv_eeg_targets_%j.err

set -Eeuo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source ./cluster_env.sh
mkdir -p logs/sv_modeling

module purge
module load bb-singularity-conf/live

PYMC_IMAGE="${IMAGE_PYMC:-/rds/projects/z/zhanglp-vwendler-core/containers/pymc_5.26.sif}"
PYMC_PYTHON="/opt/conda/envs/pymc-dev/bin/python"
PYMC_BIN="/opt/conda/envs/pymc-dev/bin"
PYMC_CXX="/opt/conda/envs/pymc-dev/bin/g++"
MPL_DIR="${SLURM_TMPDIR:-/tmp}/matplotlib-${SLURM_JOB_ID}"
PYTENSOR_CACHE="${SLURM_TMPDIR:-/tmp}/pytensor-${SLURM_JOB_ID}"
mkdir -p "$MPL_DIR" "$PYTENSOR_CACHE"

apptainer exec --cleanenv \
    --bind "$CODE_ROOT:/workspace" \
    --bind "$RUN_ROOT:/data" \
    --bind "$SOURCE_BIDS:/source" \
    --env basepath="/data" \
    --env SOURCE_BIDS="/source" \
    --env PATH="$PYMC_BIN:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
    --env MPLCONFIGDIR="$MPL_DIR" \
    --env PYTENSOR_FLAGS="cxx=$PYMC_CXX,base_compiledir=$PYTENSOR_CACHE" \
    --env OMP_NUM_THREADS="1" \
    --env MKL_NUM_THREADS="1" \
    --env OPENBLAS_NUM_THREADS="1" \
    "$PYMC_IMAGE" \
    "$PYMC_PYTHON" \
    /workspace/behav/08c_export_selected_sv_outerfold_targets.py \
        --draws 1500 \
        --tune 1500 \
        --chains 4 \
        --cores 4 \
        --target-accept 0.99
