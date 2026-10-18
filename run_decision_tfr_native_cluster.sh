#!/usr/bin/env bash

set -Eeuo pipefail

cd "$(dirname "$0")"
source ./cluster_env.sh

mkdir -p logs/tfr_decision

echo "Submitting sequential decision-TFR preparation..."

PREP_RAW=$(sbatch --parsable \
    slurm/04e_eeg_tfr_prep_decision.sbatch)

PREP_ID=${PREP_RAW%%;*}

echo "Decision TFR preparation job: $PREP_ID"

echo "Submitting native decision MVPA with dependency..."

DEC_RAW=$(sbatch --parsable \
    --dependency="afterok:${PREP_ID}" \
    slurm/05j_eeg_tfr_decode_decision_native.sbatch)

DEC_ID=${DEC_RAW%%;*}

echo
echo "============================================================"
echo "Submitted decision TFR pipeline"
echo
echo "TFR prep: $PREP_ID"
echo "MVPA:     $DEC_ID"
echo
echo "Prep stdout:"
echo "  logs/tfr_decision/pr_dec_tfr_${PREP_ID}.out"
echo
echo "Prep stderr:"
echo "  logs/tfr_decision/pr_dec_tfr_${PREP_ID}.err"
echo
echo "MVPA waits for successful completion of prep."
echo "============================================================"