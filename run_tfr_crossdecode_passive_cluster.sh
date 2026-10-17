#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")"
source ./cluster_env.sh
mkdir -p logs/tfr
JOB_RAW=$(sbatch --parsable slurm/05e_eeg_tfr_crossdecode_passive.sbatch)
JOB_ID=${JOB_RAW%%;*}
cat <<EOF
Submitted passive TFR cross-decoding.
Job: ${JOB_ID}
Log: logs/tfr/pr_tfr_cross_${JOB_ID}.log
Monitor:
  squeue -u \$USER
  tail -f logs/tfr/pr_tfr_cross_${JOB_ID}.log
EOF
