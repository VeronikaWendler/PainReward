# Attempt at reconstructing the source of our sv_pain signal in the brain using MNE Python for EEG
# Code is mashed together from some tutorials on MNE and our own things
# Veronika Wendler, but see below for the authors
# this code is intended to run on a cluster

# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Joan Massich <mailsik@gmail.com>
#          Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.


# libs
import numpy as np
import mne
from mne.datasets import eegbci, fetch_fsaverage
from mayavi import mlab
import os
os.environ["MNE_3D_OPTION_ANTIALIAS"] = "true"
os.environ["ETS_TOOLKIT"] = "null"
os.environ["QT_QPA_PLATFORM"] = "offscreen"


# 3D head + electrodes alignment (rendered offscreen)
mne.viz.set_3d_backend('pyvistaqt')  # use the MNE-3D compatible backend
mne.viz.set_3d_options(antialias=True)


#-----------------------------------------------------------------------------------------------
# 1. EEG forward operator with a template MRI
# 1.1 

# Download fsaverage files
fs_dir = fetch_fsaverage(verbose=True)
subjects_dir = fs_dir.parent

# The files live in:
subject = "fsaverage"
trans = "fsaverage"  # MNE has a built-in fsaverage transformation
src = fs_dir / "bem" / "fsaverage-ico-5-src.fif"
bem = fs_dir / "bem" / "fsaverage-5120-5120-5120-bem-sol.fif"


#-----------------------------------------------------------------------------------------------
# 1.2. load the data 

(raw_fname,) = eegbci.load_data(subjects=1, runs=[6])
raw = mne.io.read_raw_edf(raw_fname, preload=True)

# Clean channel names to be able to use a standard 1005 montage
eegbci.standardize(raw)

# Read and set the EEG electrode locations, which are already in fsaverage's
# space (MNI space) for standard_1020:

raw.set_montage("easycap-M1", on_missing="warn")
raw.set_eeg_reference(projection=True)  # needed for inverse modeling

# Create the scene (but don't show it interactively)
fig = mne.viz.plot_alignment(
    raw.info,
    src=src,
    eeg=["original", "projected"],
    trans=trans,
    show_axes=True,
    mri_fiducials=True,
    dig="fiducials",
    surfaces=["head", "brain"],
)

# Save the figure to disk instead of showing it
mlab.savefig("/home/u04vw21/sharedscratch/PainReward_ULaval/figures/sub-001_alignment.png")
mlab.close(all=True)


#-----------------------------------------------------------------------------------------------
# 1.3 set up source space and compute the forward  

fwd = mne.make_forward_solution(
    raw.info, trans=trans, src=src, bem=bem, eeg=True, mindist=5.0, n_jobs=None
)
fwd