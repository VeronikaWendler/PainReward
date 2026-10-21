#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "$0")"

mode="${1:-cv}"

case "$mode" in
    cv)
        sbatch slurm/08a_compare_subjective_value_models_regularized_blockcv.sbatch
        ;;
    final)
        if [[ $# -lt 2 ]]; then
            echo "Usage: $0 final <model_name>" >&2
            exit 1
        fi
        sbatch --export=ALL,SV_MODEL="$2" \
            slurm/08b_refit_selected_subjective_value_model.sbatch
        ;;
    *)
        echo "Usage:" >&2
        echo "  $0 cv" >&2
        echo "  $0 final <model_name>" >&2
        exit 1
        ;;
esac
