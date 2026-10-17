#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")"
source ./cluster_env.sh
mkdir -p logs/tfr
J=$(sbatch --parsable slurm/05f_eeg_tfr_haufe_topographies.sbatch); J=${J%%;*}
echo "Submitted Haufe topographies: $J"
echo "Log: logs/tfr/pr_haufe_${J}.log"
echo "Monitor: tail -f logs/tfr/pr_haufe_${J}.log"
