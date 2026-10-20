#!/bin/bash
set -Eeuo pipefail

cd "$(dirname "$0")"
source ./cluster_env.sh
mkdir -p logs/tfr_decision

audit_job=$(sbatch --parsable slurm/05q_eeg_response_locked_design_audit.sbatch)

mvpa_job=$(sbatch --parsable \
    --dependency=afterok:"$audit_job" \
    slurm/05r_eeg_tfr_response_locked_mvpa_full.sbatch)

rsa_job=$(sbatch --parsable \
    --dependency=afterok:"$audit_job" \
    slurm/05s_eeg_tfr_response_locked_rsa.sbatch)

echo "Submitted proper response-locked representation pipeline"
echo "  05q behavior/timing audit: $audit_job"
echo "  05r full response MVPA:     $mvpa_job (afterok:$audit_job)"
echo "  05s response RSA:           $rsa_job (afterok:$audit_job)"
echo
echo "No participant arrays are used."
echo "Each stage has one stdout and one stderr file."
