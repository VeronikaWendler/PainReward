#!/bin/bash
#SBATCH --job-name=rp_bf_train
#SBATCH --mem=128G
#SBATCH --output=logs/eeg_%j.out
#SBATCH --error=logs/eeg_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=VAW508@student.bham.ac.uk

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

# Modules for BEAR
module purge
module load bb-singularity-conf/live

# Environment 
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mplcache"
mkdir -p "$MPLCONFIGDIR"

# Paths
PROJECT_DIR=${PROJECT_DIR:-/rds/projects/z/zhanglp-vwendler-core/PainReward_ULaval/rp_bayesflow_workflow}
DATA_CSV=${DATA_CSV:-/rds/homes/v/vaw508/projects/PainReward/Hddm_Docker_August_24/data_sets/behavioural_sv_cleaned_final_3_with_rp.csv}
OUTDIR=${OUTDIR:-${PROJECT_DIR}/runs/pilot_run_01}
PYTHON_BIN=${PYTHON_BIN:-python}

mkdir -p "${PROJECT_DIR}/logs"
mkdir -p "${OUTDIR}"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export TF_NUM_INTRAOP_THREADS=${SLURM_CPUS_PER_TASK}
export TF_NUM_INTEROP_THREADS=2

cd "${PROJECT_DIR}"

"${PYTHON_BIN}" train_model.py \
  --data "${DATA_CSV}" \
  --outdir "${OUTDIR}" \
  --epochs 200 \
  --batch-size 16 \
  --iterations-per-epoch 300 \
  --capacity 100 \
  --n-trials-min 80 \
  --n-trials-max 220
