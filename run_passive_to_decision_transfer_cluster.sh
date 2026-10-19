#!/usr/bin/env bash
#
# run_passive_to_decision_transfer_cluster.sh
#
# Submit the next two representational-reuse tests:
#
#   1. same-time TFR passive -> decision transfer
#   2. same-time signed-voltage passive -> decision transfer
#
# The second job starts only if the first completes successfully.
# There are no participant job arrays.

set -Eeuo pipefail

cd "$(dirname "$0")"
source ./cluster_env.sh

mkdir -p logs/tfr_decision

echo "Submitting passive -> decision TFR transfer..."

TFR_RAW=$(sbatch --parsable \
    slurm/05n_eeg_tfr_transfer_passive_to_decision.sbatch)

TFR_ID=${TFR_RAW%%;*}

echo "  TFR transfer job: $TFR_ID"


echo "Submitting passive -> decision voltage transfer after TFR transfer..."

VOLTAGE_RAW=$(sbatch --parsable \
    --dependency="afterok:${TFR_ID}" \
    slurm/05o_eeg_voltage_transfer_passive_to_decision.sbatch)

VOLTAGE_ID=${VOLTAGE_RAW%%;*}

echo "  Voltage transfer job: $VOLTAGE_ID"


echo
echo "============================================================"
echo "PASSIVE -> DECISION TRANSFER JOBS SUBMITTED"
echo "============================================================"
echo "1. TFR transfer:     $TFR_ID"
echo "2. Voltage transfer: $VOLTAGE_ID"
echo
echo "Logs:"
echo "  logs/tfr_decision/pr_p2d_tfr_${TFR_ID}.out"
echo "  logs/tfr_decision/pr_p2d_tfr_${TFR_ID}.err"
echo
echo "  logs/tfr_decision/pr_p2d_voltage_${VOLTAGE_ID}.out"
echo "  logs/tfr_decision/pr_p2d_voltage_${VOLTAGE_ID}.err"
echo "============================================================"
