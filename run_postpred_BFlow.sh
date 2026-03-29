#!/bin/bash
#SBATCH --job-name=rp_bf_ppc_run3
#SBATCH --time=3:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --output=logs/rp_bf_ppc_run3_%j.out
#SBATCH --error=logs/rp_bf_ppc_run3_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=VAW508@student.bham.ac.uk

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

module purge
module load bear-apps/2021a/live
module load Python/3.9.5-GCCcore-10.3.0
source /rds/projects/z/zhanglp-vwendler-core/PainReward_ULaval/bayesflow_py39/bin/activate

export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mplcache"
mkdir -p "${MPLCONFIGDIR}"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export TF_NUM_INTRAOP_THREADS=${SLURM_CPUS_PER_TASK}
export TF_NUM_INTEROP_THREADS=2

PROJECT_DIR=/rds/homes/v/vaw508/projects/PainReward/rp_bayesflow_workflow
DATA_CSV=/rds/homes/v/vaw508/projects/PainReward/Hddm_Docker_August_24/data_sets/behavioural_sv_cleaned_final_3_with_rp.csv
RUN_DIR=/rds/projects/z/zhanglp-vwendler-core/PainReward_ULaval/runs/pilot_run_03
CHECKPOINT_DIR=${RUN_DIR}/checkpoints
OUTDIR=${RUN_DIR}/posterior_predictive

mkdir -p "${OUTDIR}"

cd "${PROJECT_DIR}"

echo "PROJECT_DIR=${PROJECT_DIR}"
echo "DATA_CSV=${DATA_CSV}"
echo "RUN_DIR=${RUN_DIR}"
echo "CHECKPOINT_DIR=${CHECKPOINT_DIR}"
echo "OUTDIR=${OUTDIR}"
echo "Starting BayesFlow posterior predictive checks for pilot_run_03..."

python posterior_predictive.py \
  --data "${DATA_CSV}" \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  --outdir "${OUTDIR}" \
  --n-posterior-draws 250 \
  --n-sim-draws 100 \
  --seed 123