#!/bin/bash
#SBATCH --job-name=pr_dec_mvpa
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=logs/tfr_decision/pr_dec_mvpa_%j.out
#SBATCH --error=logs/tfr_decision/pr_dec_mvpa_%j.err


# ==============================================================================
# Native decision-phase TFR decoding
#
# This is ONE SLURM job. The Python script itself processes all participants
# sequentially and then runs the group-level cluster-permutation inference.
#
# Outputs from SLURM:
#   logs/tfr_decision/pr_dec_mvpa_<JOBID>.out
#   logs/tfr_decision/pr_dec_mvpa_<JOBID>.err
#
# Scientific outputs:
#   $RUN_ROOT/derivatives/statistics/tfr_mvpa_decision_native/
# ==============================================================================

set -Eeuo pipefail


# ------------------------------------------------------------------------------
# Move to the project directory from which the job was submitted and load the
# central project-path/environment definitions.
# ------------------------------------------------------------------------------

cd "${SLURM_SUBMIT_DIR:-$PWD}"
source ./cluster_env.sh

mkdir -p logs/tfr_decision


# ------------------------------------------------------------------------------
# Prevent NumPy / BLAS libraries from independently spawning many threads.
#
# MNE receives the requested SLURM CPU count explicitly through --n-jobs below.
# ------------------------------------------------------------------------------

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1


# ------------------------------------------------------------------------------
# Load Apptainer on BEAR.
# ------------------------------------------------------------------------------

module purge
module load bb-singularity-conf/live


echo "============================================================"
echo "NATIVE DECISION-PHASE TFR DECODING"
echo "SLURM job ID:       $SLURM_JOB_ID"
echo "CPUs requested:     $SLURM_CPUS_PER_TASK"
echo "RUN_ROOT:           $RUN_ROOT"
echo "MNE container:      $IMAGE_MNE"
echo "============================================================"


# ------------------------------------------------------------------------------
# Run the decoder inside the MNE container.
#
# Mounts:
#   $CODE_ROOT -> /workspace   analysis code
#   $RUN_ROOT  -> /data        prepared TFRs + statistical outputs
#
# The Python script:
#   1. decodes pain and money magnitude for each participant;
#   2. repeats decoding with the RT-sensitivity residualisation;
#   3. saves participant score arrays;
#   4. runs group frequency x time cluster tests;
#   5. runs canonical-band temporal cluster tests.
# ------------------------------------------------------------------------------

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
    python /workspace/eeg/05j_eeg_tfr_decode_decision_native.py \
        --n-jobs "$SLURM_CPUS_PER_TASK" \
        --n-permutations 5000


echo
echo "============================================================"
echo "DECISION-PHASE TFR DECODING JOB FINISHED"
echo "============================================================"
