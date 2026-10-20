#!/bin/bash
#SBATCH --job-name=pr_sv_oof
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --output=logs/sv_modeling/pr_sv_oof_%j.out
#SBATCH --error=logs/sv_modeling/pr_sv_oof_%j.err

set -Eeuo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
source ./cluster_env.sh

mkdir -p logs/sv_modeling

module purge
module load bb-singularity-conf/live

PYMC_IMAGE="${IMAGE_PYMC:-/rds/projects/z/zhanglp-vwendler-core/containers/pymc_5.26.sif}"
PYMC_PYTHON="/opt/conda/envs/pymc-dev/bin/python"
PYMC_BIN="/opt/conda/envs/pymc-dev/bin"
PYMC_CXX="/opt/conda/envs/pymc-dev/bin/x86_64-conda-linux-gnu-c++"

if [[ ! -f "$PYMC_IMAGE" ]]; then
    echo "ERROR: PyMC image not found: $PYMC_IMAGE" >&2
    exit 1
fi

echo "============================================================"
echo "OUT-OF-FOLD SUBJECTIVE-VALUE TARGET GENERATION"
echo "SLURM job:    $SLURM_JOB_ID"
echo "PyMC image:   $PYMC_IMAGE"
echo "PyMC Python:  $PYMC_PYTHON"
echo "PyMC C++:     $PYMC_CXX"
echo "RUN_ROOT:     $RUN_ROOT"
echo "SOURCE_BIDS:  $SOURCE_BIDS"
echo "============================================================"

# Writable per-job caches.
MPL_DIR="${SLURM_TMPDIR:-/tmp}/matplotlib-${SLURM_JOB_ID}"
PYTENSOR_CACHE="${SLURM_TMPDIR:-/tmp}/pytensor-${SLURM_JOB_ID}"
mkdir -p "$MPL_DIR" "$PYTENSOR_CACHE"

# Fail fast: verify exact Python and compiler.
apptainer exec --cleanenv \
    --env PATH="$PYMC_BIN:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
    "$PYMC_IMAGE" \
    bash -lc '
        echo "Compiler:"
        which g++
        g++ --version | head -1

        echo "Python/PyMC:"
        /opt/conda/envs/pymc-dev/bin/python -c "
import pymc as pm
import arviz as az
import pytensor

print(\"PyMC\", pm.__version__)
print(\"ArviZ\", az.__version__)
print(\"PyTensor\", pytensor.__version__)

if int(pm.__version__.split(\".\")[0]) < 5:
    raise SystemExit(\"ERROR: this pipeline requires PyMC >= 5\")
"
    '

# Generate out-of-fold subjective-value targets with compiled PyTensor ops.
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
    /workspace/behav/06d_generate_oof_subjective_value_targets.py \
        --draws 1500 \
        --tune 1500 \
        --chains 4 \
        --cores 4 \
        --target-accept 0.95
