#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")"
source ./cluster_env.sh
mkdir -p logs/tfr

PREP_RAW=$(sbatch --parsable slurm/04d_eeg_tfr_prep_evoked_removed.sbatch)
PREP=${PREP_RAW%%;*}

DEC_RAW=$(sbatch --parsable --dependency=afterok:${PREP} slurm/05g_eeg_tfr_decode_evoked_removed.sbatch)
DEC=${DEC_RAW%%;*}

ERP_RAW=$(sbatch --parsable slurm/05h_eeg_erp_voltage_decode_passive.sbatch)
ERP=${ERP_RAW%%;*}

CMP_RAW=$(sbatch --parsable --dependency=afterok:${DEC} slurm/05i_eeg_compare_total_vs_evoked_removed.sbatch)
CMP=${CMP_RAW%%;*}

cat <<EOF
Submitted phase-locked vs induced-enriched pipeline.

1) Cross-fitted evoked-removal TFR prep: ${PREP}
   logs/tfr/pr_tfr_evrem_${PREP}.log

2) Evoked-removed TFR decoding (after prep): ${DEC}
   logs/tfr/pr_tfr_evrem_dec_${DEC}.log

3) ERP/voltage decoding (runs independently): ${ERP}
   logs/tfr/pr_erp_mvpa_${ERP}.log

4) Paired TOTAL vs EVOKED-REMOVED comparison (after decoding): ${CMP}
   logs/tfr/pr_tfr_tot_evrem_${CMP}.log

Monitor:
  squeue -u \$USER
EOF
