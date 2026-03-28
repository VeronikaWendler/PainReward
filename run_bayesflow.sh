#!/bin/bash
#SBATCH --job-name=rp_bf_recovery
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --output=logs/rp_bf_recovery_%j.out
#SBATCH --error=logs/rp_bf_recovery_%j.err
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
mkdir -p "$MPLCONFIGDIR"

PROJECT_DIR=/rds/homes/v/vaw508/projects/PainReward/rp_bayesflow_workflow
DATA_CSV=/rds/homes/v/vaw508/projects/PainReward/Hddm_Docker_August_24/data_sets/behavioural_sv_cleaned_final_3_with_rp.csv
RUN_DIR=/rds/projects/z/zhanglp-vwendler-core/PainReward_ULaval/runs/pilot_run_01
CHECKPOINT_DIR=/rds/projects/z/zhanglp-vwendler-core/PainReward_ULaval/runs/pilot_run_01/checkpoints
OUTDIR=/rds/projects/z/zhanglp-vwendler-core/PainReward_ULaval/runs/pilot_run_01/recovery

mkdir -p "${OUTDIR}"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export TF_NUM_INTRAOP_THREADS=${SLURM_CPUS_PER_TASK}
export TF_NUM_INTEROP_THREADS=2

cd "${PROJECT_DIR}"

python validate_recovery.py \
  --data "${DATA_CSV}" \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  --outdir "${OUTDIR}" \
  --n-param-sets 250 \
  --n-trials 160 \
  --n-posterior-draws 1000 \
  --seed 123


# # python train_model.py \
# #   --data "${DATA_CSV}" \
# #   --outdir "${OUTDIR}" \
# #   --epochs 200 \
# #   --batch-size 16 \
# #   --iterations-per-epoch 300 \
# #   --capacity 100 \
# #   --n-trials-min 80 \
# #   --n-trials-max 220