#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "$0")"
source ./cluster_env.sh
mkdir -p logs/tfr

JOB_RAW=$(sbatch --parsable slurm/05d_eeg_tfr_mvpa_passive_v2.sbatch)
JOB_ID=${JOB_RAW%%;*}

cat <<EOF
Submitted redesigned passive spectral MVPA v2.

Job: ${JOB_ID}
Log: logs/tfr/pr_tfr_mvpa2_${JOB_ID}.log

Monitor:
  squeue -u \$USER
  tail -f logs/tfr/pr_tfr_mvpa2_${JOB_ID}.log
EOF
