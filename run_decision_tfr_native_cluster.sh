#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")"
source ./cluster_env.sh
mkdir -p logs/tfr_decision
mapfile -t SUBJECTS < <(find "$RUN_ROOT" -maxdepth 1 -mindepth 1 -name 'sub-*' -printf '%f\n' | sort)
N=${#SUBJECTS[@]}
if (( N == 0 )); then echo "No sub-* directories found under $RUN_ROOT" >&2; exit 1; fi
MAX_CONCURRENT=${MAX_CONCURRENT_DEC_TFR:-6}
PREP_RAW=$(sbatch --parsable --array="0-$((N-1))%${MAX_CONCURRENT}" slurm/04e_eeg_tfr_prep_decision.sbatch)
PREP_ID=${PREP_RAW%%;*}
DEC_RAW=$(sbatch --parsable --dependency="afterok:${PREP_ID}" slurm/05j_eeg_tfr_decode_decision_native.sbatch)
DEC_ID=${DEC_RAW%%;*}
echo "Submitted decision TFR array: $PREP_ID"
echo "Submitted native decision MVPA: $DEC_ID (afterok:$PREP_ID)"
echo "Decoder log: logs/tfr_decision/pr_dec_mvpa_${DEC_ID}.log"
