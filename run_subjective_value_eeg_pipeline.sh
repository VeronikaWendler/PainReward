#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")"

# Submit the strict fold-aligned pipeline:
#   1) export five selected-model outer-fold SV target tables
#   2) signed-voltage SV MVPA
#   3) TFR SV MVPA
# The TFR stage waits for voltage only to keep the workflow simple; scientifically
# it can be interpreted as the planned second representation analysis.

job_targets=$(sbatch --parsable slurm/08c_export_selected_sv_outerfold_targets.sbatch)
echo "SV outer-fold target export: $job_targets"

job_voltage=$(sbatch --parsable --dependency=afterok:"$job_targets" \
    slurm/05v_eeg_voltage_decode_decision_subjective_value.sbatch)
echo "Signed-voltage SV MVPA:      $job_voltage"

job_tfr=$(sbatch --parsable --dependency=afterok:"$job_voltage" \
    slurm/05w_eeg_tfr_decode_decision_subjective_value.sbatch)
echo "TFR SV MVPA:                 $job_tfr"
