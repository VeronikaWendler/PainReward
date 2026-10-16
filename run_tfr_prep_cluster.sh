#!/usr/bin/env bash
# Submit the complete PRE-DECODING TFR preparation chain:
#   1) timing audit
#   2) one TFR job that processes all participants sequentially
#   3) group QC after the TFR job succeeds
#
# This intentionally does NOT submit any decoding analyses.
set -Eeuo pipefail

cd "$(dirname "$0")"
source ./cluster_env.sh
mkdir -p logs/tfr

mapfile -t SUBJECTS < <(
  find -L "$RUN_ROOT" -maxdepth 1 -mindepth 1 -type d -name 'sub-*' -printf '%f\n' | sort
)
N_SUBJECTS=${#SUBJECTS[@]}

if (( N_SUBJECTS == 0 )); then
  echo "ERROR: no sub-* directories found under RUN_ROOT=$RUN_ROOT" >&2
  exit 2
fi

if [[ ! -d "$SOURCE_BIDS/derivatives" ]]; then
  echo "ERROR: SOURCE_BIDS derivatives not found: $SOURCE_BIDS/derivatives" >&2
  exit 2
fi

AUDIT_RAW=$(sbatch --parsable slurm/04a_eeg_tfr_timing_audit.sbatch)
AUDIT_JOB=${AUDIT_RAW%%;*}

PREP_RAW=$(sbatch --parsable \
  --dependency="afterok:${AUDIT_JOB}" \
  slurm/04b_eeg_tfr_prep.sbatch)
PREP_JOB=${PREP_RAW%%;*}

QC_RAW=$(sbatch --parsable \
  --dependency="afterok:${PREP_JOB}" \
  slurm/04c_eeg_tfr_qc.sbatch)
QC_JOB=${QC_RAW%%;*}

cat <<EOF
Submitted pre-decoding TFR pipeline for ${N_SUBJECTS} participants.

Timing audit job     : ${AUDIT_JOB}
Sequential TFR job   : ${PREP_JOB}
Group QC job         : ${QC_JOB}

The TFR job starts only if the timing audit passes.
Participants are processed one after another inside ONE SLURM job.
The group QC starts only if the complete TFR job succeeds.

Logs:
  timing : logs/tfr/pr_tfr_timing_${AUDIT_JOB}.log
  TFR    : logs/tfr/pr_tfr_prep_${PREP_JOB}.log
  QC     : logs/tfr/pr_tfr_qc_${QC_JOB}.log

Monitor with:
  squeue -u \$USER
  tail -f logs/tfr/pr_tfr_prep_${PREP_JOB}.log
EOF
