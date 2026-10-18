#!/usr/bin/env bash
#
# run_decision_format_tests_cluster.sh
#
# Submit the mechanistic decision-format tests as a fail-fast dependency chain:
#
#   1. matched voltage decoding
#   2. evoked-removed TFR preparation
#   3. evoked-removed TFR decoding
#   4. paired total-vs-evoked-removed comparison
#
# Every stage is ONE SLURM job. There are no participant job arrays.

set -Eeuo pipefail

cd "$(dirname "$0")"
source ./cluster_env.sh

mkdir -p logs/tfr_decision

echo "Submitting matched decision voltage decoding..."

VOLTAGE_RAW=$(sbatch --parsable \
    slurm/05k_eeg_voltage_decode_decision_matched.sbatch)

VOLTAGE_ID=${VOLTAGE_RAW%%;*}

echo "  voltage job: $VOLTAGE_ID"


echo "Submitting evoked-removed TFR preparation after voltage job..."

PREP_RAW=$(sbatch --parsable \
    --dependency="afterok:${VOLTAGE_ID}" \
    slurm/04f_eeg_tfr_prep_decision_evoked_removed.sbatch)

PREP_ID=${PREP_RAW%%;*}

echo "  evoked-removed prep job: $PREP_ID"


echo "Submitting evoked-removed TFR decoding after prep..."

DECODE_RAW=$(sbatch --parsable \
    --dependency="afterok:${PREP_ID}" \
    slurm/05l_eeg_tfr_decode_decision_evoked_removed.sbatch)

DECODE_ID=${DECODE_RAW%%;*}

echo "  evoked-removed decoding job: $DECODE_ID"


echo "Submitting paired total-vs-evoked-removed comparison after decoding..."

COMPARE_RAW=$(sbatch --parsable \
    --dependency="afterok:${DECODE_ID}" \
    slurm/05m_eeg_compare_decision_total_vs_evoked_removed.sbatch)

COMPARE_ID=${COMPARE_RAW%%;*}

echo "  paired comparison job: $COMPARE_ID"


echo
echo "============================================================"
echo "DECISION FORMAT TESTS SUBMITTED"
echo "============================================================"
echo "1. Voltage decoding:       $VOLTAGE_ID"
echo "2. Evoked-removed prep:    $PREP_ID"
echo "3. Evoked-removed decoder: $DECODE_ID"
echo "4. Paired comparison:      $COMPARE_ID"
echo
echo "Logs:"
echo "  logs/tfr_decision/pr_dec_voltage_${VOLTAGE_ID}.out"
echo "  logs/tfr_decision/pr_dec_voltage_${VOLTAGE_ID}.err"
echo
echo "  logs/tfr_decision/pr_dec_evrem_prep_${PREP_ID}.out"
echo "  logs/tfr_decision/pr_dec_evrem_prep_${PREP_ID}.err"
echo
echo "  logs/tfr_decision/pr_dec_evrem_mvpa_${DECODE_ID}.out"
echo "  logs/tfr_decision/pr_dec_evrem_mvpa_${DECODE_ID}.err"
echo
echo "  logs/tfr_decision/pr_dec_total_vs_evrem_${COMPARE_ID}.out"
echo "  logs/tfr_decision/pr_dec_total_vs_evrem_${COMPARE_ID}.err"
echo "============================================================"
