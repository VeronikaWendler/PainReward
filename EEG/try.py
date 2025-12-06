
# importing libraries
from mne.report import Report
import pprint
import mne
import os
from os.path import join as opj
import pandas as pd
import numpy as np
from mne.viz import plot_evoked_joint as pej
from bids import BIDSLayout
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
import os
from scipy.stats import pearsonr
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from statsmodels.distributions.empirical_distribution import ECDF
from pathlib import Path

# Set bids directory
PROJECT_DIR = Path(os.getenv("PROJECT_DIR", "/workspace"))
basepath = PROJECT_DIR / "EEG" / "PainReward_sub-001-050" / "painrewardeegdata"
def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)
import re
from pathlib import Path
import os
layout = BIDSLayout(basepath)
# disable Numba JIT caching & compilation
#os.environ["NUMBA_DISABLE_JIT"] = "1"
import numba
numba.config.CACHE_ENABLE = False


from mne.time_frequency import read_tfrs

fname_cue = "/workspace/EEG/PainReward_sub-001-050/painrewardeegdata/derivatives/sub-004/eeg/tfr/sub-004_decision_cues_epochs-tfr.h5"
fname_resp = "/workspace/EEG/PainReward_sub-001-050/painrewardeegdata/derivatives/sub-004/eeg/tfr/sub-004_decision_resp_epochs-tfr.h5"

for f in [fname_cue, fname_resp]:
    if os.path.exists(f):
        tfr = read_tfrs(f)[0]
        print("\nFile:", f)
        print("  data shape:", tfr.data.shape)      # (n_epochs, n_ch, n_freq, n_time)
        print("  n_epochs from metadata:", len(tfr.metadata) if tfr.metadata is not None else "no metadata")
        if tfr.metadata is not None and "trialsnum" in tfr.metadata.columns:
            print("  example trialsnum:", tfr.metadata["trialsnum"].head().tolist())
    else:
        print("\nFile does not exist:", f)
