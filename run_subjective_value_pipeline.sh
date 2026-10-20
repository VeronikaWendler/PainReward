#!/bin/bash
set -Eeuo pipefail

cd "$(dirname "$0")"
source ./cluster_env.sh

usage() {
    cat <<'EOF'
Usage:
    ./run_subjective_value_pipeline.sh audit
    ./run_subjective_value_pipeline.sh models
    ./run_subjective_value_pipeline.sh select
    ./run_subjective_value_pipeline.sh oof

Recommended order:
    1. audit
    2. inspect audit and freeze sample
    3. models
    4. inspect LOO + diagnostics
    5. select
    6. inspect selected model
    7. oof

The script intentionally does NOT auto-chain all stages because the behavioral
sample and model comparison should be inspected before expensive downstream
fits.
EOF
}

case "${1:-}" in
    audit)
        sbatch slurm/06a_acceptance_inclusion_audit.sbatch
        ;;
    models)
        sbatch slurm/06b_fit_subjective_value_models.sbatch
        ;;
    select)
        sbatch slurm/06c_select_subjective_value_model.sbatch
        ;;
    oof)
        sbatch slurm/06d_generate_oof_subjective_value_targets.sbatch
        ;;
    *)
        usage
        exit 1
        ;;
esac
