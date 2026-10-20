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

PYMC_IMAGE="${IMAGE_PYMC:-$HOME/containers/pymc_5.26.sif}"

if [[ ! -f "$PYMC_IMAGE" ]]; then
    echo "ERROR: Modern PyMC image not found: $PYMC_IMAGE" >&2
    echo "Create it first with:" >&2
    echo "  mkdir -p \$HOME/containers" >&2
    echo "  apptainer pull \$HOME/containers/pymc_5.26.sif docker://pymc/pymc:5.26" >&2
    exit 1
fi

echo "============================================================"
echo "OUT-OF-FOLD SUBJECTIVE-VALUE TARGET GENERATION"
echo "SLURM job:    $SLURM_JOB_ID"
echo "PyMC image:   $PYMC_IMAGE"
echo "============================================================"

apptainer exec --cleanenv \
    "$PYMC_IMAGE" \
    python -c '
import pymc as pm
import arviz as az
major = int(pm.__version__.split(".")[0])
print("PyMC", pm.__version__)
print("ArviZ", az.__version__)
if major < 5:
    raise SystemExit("ERROR: this pipeline requires PyMC >= 5")
'

MPL_DIR="${SLURM_TMPDIR:-/tmp}/matplotlib-${SLURM_JOB_ID}"
mkdir -p "$MPL_DIR"

apptainer exec --cleanenv \
    --bind "$CODE_ROOT:/workspace" \
    --bind "$RUN_ROOT:/data" \
    --bind "$SOURCE_BIDS:/source" \
    --env basepath="/data" \
    --env SOURCE_BIDS="/source" \
    --env MPLCONFIGDIR="$MPL_DIR" \
    --env OMP_NUM_THREADS="1" \
    --env MKL_NUM_THREADS="1" \
    --env OPENBLAS_NUM_THREADS="1" \
    "$PYMC_IMAGE" \
    python /workspace/behav/06d_generate_oof_subjective_value_targets.py \
        --draws 1500 \
        --tune 1500 \
        --chains 4 \
        --cores 4 \
        --target-accept 0.95
