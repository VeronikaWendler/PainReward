#!/bin/bash
#SBATCH --job-name=pr_sv_models
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --output=logs/sv_modeling/pr_sv_models_%j.out
#SBATCH --error=logs/sv_modeling/pr_sv_models_%j.err

set -Eeuo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
source ./cluster_env.sh

mkdir -p logs/sv_modeling

module purge
module load bb-singularity-conf/live

select_pymc_image() {
    for candidate in "${IMAGE_MNE:-}" "${IMAGE_HDDM:-}"; do
        if [[ -z "$candidate" ]]; then
            continue
        fi

        if apptainer exec --cleanenv "$candidate" \
            python -c "import pymc, arviz, pandas, sklearn" \
            >/dev/null 2>&1
        then
            echo "$candidate"
            return 0
        fi
    done

    echo "ERROR: Neither IMAGE_MNE nor IMAGE_HDDM contains the required PyMC stack." >&2
    return 1
}

PYMC_IMAGE="$(select_pymc_image)"

echo "============================================================"
echo "SUBJECTIVE-VALUE MODEL COMPARISON"
echo "SLURM job:    $SLURM_JOB_ID"
echo "PyMC image:   $PYMC_IMAGE"
echo "RUN_ROOT:     $RUN_ROOT"
echo "SOURCE_BIDS:  $SOURCE_BIDS"
echo "============================================================"

apptainer exec --cleanenv \
    --bind "$CODE_ROOT:/workspace" \
    --bind "$RUN_ROOT:/data" \
    --bind "$SOURCE_BIDS:/source" \
    --env basepath="/data" \
    --env SOURCE_BIDS="/source" \
    --env OMP_NUM_THREADS="1" \
    --env MKL_NUM_THREADS="1" \
    --env OPENBLAS_NUM_THREADS="1" \
    "$PYMC_IMAGE" \
    python /workspace/behav/06b_fit_subjective_value_models.py \
        --draws 2000 \
        --tune 2000 \
        --chains 4 \
        --cores 4 \
        --target-accept 0.95
