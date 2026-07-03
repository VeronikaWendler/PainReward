#!/bin/bash
#SBATCH --job-name=pr_check_layout
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=logs/pr_check_layout_%j.out
#SBATCH --error=logs/pr_check_layout_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source ./cluster_env.sh

echo "CODE_ROOT=$CODE_ROOT"
echo "RUN_ROOT=$RUN_ROOT"
echo "ORIGINAL_BIDS=$ORIGINAL_BIDS"

test -d "$CODE_ROOT"
test -d "$RUN_ROOT"
test -f "$RUN_ROOT/participants.tsv"

echo "Number of sub-* entries in RUN_ROOT:"
find "$RUN_ROOT" -maxdepth 1 -name 'sub-*' | wc -l

echo "Checking example subject files:"
ls "$RUN_ROOT/sub-001/eeg" | head

test -f "$RUN_ROOT/sub-001/eeg/sub-001_task-passive_beh.tsv"
test -f "$RUN_ROOT/sub-001/eeg/sub-001_task-decision_beh.tsv"

echo "Layout check OK"
