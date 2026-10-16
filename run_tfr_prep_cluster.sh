#!/usr/bin/env bash
# Submit the complete PRE-DECODING TFR preparation chain:
#   1) timing audit
#   2) one passive-TFR job per participant (SLURM array)
#   3) group QC after every array task succeeds
#
# This intentionally does NOT submit any decoding analyses.
set -Eeuo pipefail

cd "$(dirname "$0")"
source ./cluster_env.sh
mkdir -p logs

mapfile -t SUBJECTS < <(
  find "$RUN_ROOT" -maxdepth 1 -mindepth 1 -type d -name 'sub-*' -printf '%f\n' | sort
)
N_SUBJECTS=${#SUBJECTS[@]}
if (( N_SUBJECTS == 0 )); then
  echo "ERROR: no sub-* directories found under RUN_ROOT=$RUN_ROOT" >&2
  exit 2
fi

# Limit simultaneous heavy TFR jobs without changing the fact that all
# participants are submitted immediately. Override, e.g.:
#   MAX_CONCURRENT_TFR=10 bash run_tfr_prep_cluster.sh
MAX_CONCURRENT_TFR=${MAX_CONCURRENT_TFR:-6}

AUDIT_RAW=$(sbatch --parsable slurm/04a_eeg_tfr_timing_audit.sbatch)
AUDIT_JOB=${AUDIT_RAW%%;*}

PREP_RAW=$(sbatch --parsable \
  --dependency="afterok:${AUDIT_JOB}" \
  --array="0-$((N_SUBJECTS - 1))%${MAX_CONCURRENT_TFR}" \
  slurm/04b_eeg_tfr_prep.sbatch)
PREP_JOB=${PREP_RAW%%;*}

QC_RAW=$(sbatch --parsable \
  --dependency="afterok:${PREP_JOB}" \
  slurm/04c_eeg_tfr_qc.sbatch)
QC_JOB=${QC_RAW%%;*}

cat <<EOF
Submitted pre-decoding TFR pipeline for ${N_SUBJECTS} participants.

Timing audit job : ${AUDIT_JOB}
TFR array job    : ${PREP_JOB} (max ${MAX_CONCURRENT_TFR} simultaneous tasks)
Group QC job     : ${QC_JOB}

The TFR array starts only if the timing audit passes.
The group QC starts only if every TFR array task succeeds.

Monitor with:
  squeue -u \$USER
  tail -f logs/pr_tfr_timing_${AUDIT_JOB}.out
EOF
