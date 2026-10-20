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
echo "OUT-OF-FOLD SUBJECTIVE-VALUE TARGET GENERATION"
echo "SLURM job:    $SLURM_JOB_ID"
echo "PyMC image:   $PYMC_IMAGE"
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
    python /workspace/behav/06d_generate_oof_subjective_value_targets.py \
        --draws 1500 \
        --tune 1500 \
        --chains 4 \
        --cores 4 \
        --target-accept 0.95
