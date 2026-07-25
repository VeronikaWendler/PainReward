export CODE_ROOT="$HOME/projects/PainReward_mpreview_cluster" # this folder contains only code which is also synchronized with the GitHub repository
export ORIGINAL_BIDS="/rds/projects/z/zhanglp-vwendler-core/PainReward_ULaval/EEG/PainReward_sub-001-050/painrewardeegdata"  # contains the original data
export RUN_ROOT="/rds/projects/z/zhanglp-vwendler-core/PainReward_mpreview_run" # contains heavy data on the cluster's RDS
export QUEST_DATA="/rds/projects/z/zhanglp-vwendler-core/PainReward_mpreview_run/data/participants_new.csv" # location of the participant file on the RDS

export basepath="$RUN_ROOT"
export PROJECT_DIR="$RUN_ROOT"
export HDDM_DIR="$RUN_ROOT/derivatives/hddm"

export IMAGE_MNE="$HOME/containers/mne_latest.sif"
export IMAGE_HDDM="$HOME/containers/hddm_latest.sif"

export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
