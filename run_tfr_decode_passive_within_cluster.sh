#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "$0")"
source ./cluster_env.sh
mkdir -p logs/tfr

JOB_RAW=$(sbatch --parsable slurm/05c_eeg_tfr_decode_passive_within.sbatch)
JOB_ID=${JOB_RAW%%;*}

cat <<EOF
Submitted passive within-domain TFR decoding.

Job: ${JOB_ID}
Log: logs/tfr/pr_tfr_decpass_${JOB_ID}.log

Monitor:
  squeue -u \$USER
  tail -f logs/tfr/pr_tfr_decpass_${JOB_ID}.log
EOF
