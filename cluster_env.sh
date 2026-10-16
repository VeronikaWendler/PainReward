export CODE_ROOT="$HOME/projects/PainReward_mpreview_cluster"
export ORIGINAL_BIDS="/rds/projects/z/zhanglp-vwendler-core/PainReward_ULaval/EEG/PainReward_sub-001-050/painrewardeegdata"
# Existing cleaned EEG + ERP derivatives are read from the original BIDS tree.
export SOURCE_BIDS="$ORIGINAL_BIDS"

# New/heavy outputs from this cluster run are written here.
export RUN_ROOT="/rds/projects/z/zhanglp-vwendler-core/PainReward_mpreview_run/data"
export QUEST_DATA="$RUN_ROOT/participants_new.csv"

export basepath="$RUN_ROOT"
export PROJECT_DIR="$RUN_ROOT"
export HDDM_DIR="$RUN_ROOT/derivatives/hddm"

export IMAGE_MNE="$HOME/containers/mne_latest.sif"
export IMAGE_HDDM="$HOME/containers/hddm_latest.sif"

export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
