# -*- coding: utf-8 -*-
"""
Author: michel-pierre.coll
Date: 2023-01-24
Project: painreward
Description: Cleans eeg data

Description (for both passive and decision task):
- Loads data
- Removes bad channels (flagged in visual inspection)
- Plot events and timing
- Run ICA to remove artifacts and apply to continuous data
- Filter data
- Set average reference
- Interpolate bad channels
- Save cleaned continous data
"""
import seaborn as sns
import pandas as pd
import mne
import matplotlib.pyplot as plt
import numpy as np
from mne.preprocessing import ICA, create_eog_epochs
import os
from os.path import join as opj
from mne.report import Report
from mne_icalabel import label_components
from scipy.stats import pearsonr

# Set bids directory
basepath = "D:/Aberdeen_Uni_June24/MPColl_Lab/All_Files_Relevant_For_Git/EEG/PainReward_sub-001-050/painrewardeegdata"

# Choose output directory
outpath = opj(basepath, "derivatives")

# List participants
part = [p for p in os.listdir(opj(basepath)) if "sub" in p]
part.sort()

# ###############################################################################
# ###############################################################################
# # Passive task
# ###############################################################################
# ###############################################################################

# Create a data frame to collect stats
stats_frame = pd.DataFrame(
    index=part,
    columns=[
        "n_removed_icas_passive",
        "n_bad_chans_passive",
        "n_removed_icas_decision",
        "n_bad_chans_decision",
        "corr_pain_passive",
        "accuracy_catch_money",
        "acceptance_rate_decision",
        "l1",
        "l2",
        "l3",
        "l4",
        "l5",
    ],
)


for p in part:
    if not os.path.exists(opj(outpath, p, "eeg")):
        os.makedirs(opj(outpath, p, "eeg"))
    passive_file = opj(basepath, p, "eeg", p + "_task-passive_eeg.vhdr")
    passive_behav = pd.read_csv(
        opj(basepath, p, "eeg", p + "_task-passive_beh.tsv"), sep="\t"
    )

    # Initialize report
    report = Report(verbose=False, subject=p, title="EEG report for part " + p)

    # Remove rows with no cross onset (nans)
    passive_behav = passive_behav[~passive_behav["fixcross.started"].isna()]

    # Add levels to stats frame
    stats_frame.loc[p, "l1"] = passive_behav["P1"][0]
    stats_frame.loc[p, "l2"] = passive_behav["P2"][0]
    stats_frame.loc[p, "l3"] = passive_behav["P3"][0]
    stats_frame.loc[p, "l4"] = passive_behav["P4"][0]
    stats_frame.loc[p, "l5"] = passive_behav["P5"][0]

    # Add pain ratings to report
    catch_pain = passive_behav[~passive_behav["catch_pain"].isna()]
    fig = plt.figure()
    sns.pointplot(data=catch_pain, x="level", y="catch_pain")
    r, _ = pearsonr(catch_pain["level"], catch_pain["catch_pain"])
    report.add_figure(fig, "Passive pain ratings")
    stats_frame.loc[p, "corr_pain_passive"] = r

    # Add catch money ratings to report
    catch_money = passive_behav[~passive_behav["catch_money"].isna()]
    fig = plt.figure()
    sns.regplot(data=catch_money, x="level", y="catch_money")
    report.add_figure(fig, "Passive catch money ratings")
    acc = np.sum(
        np.where(
            catch_money["catch_money"].astype(int) * 20 == catch_money["level"], 1, 0
        )
    ) / len(catch_money)
    stats_frame.loc[p, "accuracy_catch_money"] = acc

    # Load data
    passive_raw = mne.io.read_raw_brainvision(passive_file)

    # Load channel definitions to get channels marked as bad in inspection
    channels = pd.read_csv(
        opj(basepath, p, "eeg", p + "_task-passive_channels.tsv"), sep="\t"
    )

    # Flag bad channels
    passive_raw.info["bads"] = channels[channels["status"] != "good"]["name"].tolist()

    # passive.drop_channels(['Iz'])
    stats_frame.loc[p, "n_bad_chans_passive"] = len(passive_raw.info["bads"])

    # Set montage
    passive_raw.set_montage("easycap-M1", on_missing="warn")

    # plot channels
    fig = passive_raw.plot_sensors(show_names=True, show=False)
    report.add_figure(fig, "Sensor positions (bad in red) passive")

    # Plot spectrum (first 200 seconds)
    fig = (
        passive_raw.copy()
        .drop_channels(["FCz"])
        .plot_psd(fmax=100, tmax=200, show=False)
    )
    report.add_figure(fig, "Raw passive spectrum")

    # High pass filter
    events, events_id = mne.events_from_annotations(passive_raw)

    # Remove the "New Segment" event
    events_id = {
        k.replace("Comment/", ""): v for k, v in events_id.items() if v != 99999
    }
    events = events[events[:, 2] != 99999]

    # Remove the "DIN7" event (photosensor)
    if "DIN7" in events_id.keys():
        events = events[events[:, 2] != events_id["DIN7"]]
        events_id = {k: v for k, v in events_id.items() if k != "DIN7"}

    # Plot events
    fig = mne.viz.plot_events(
        events, first_samp=passive_raw.first_samp, event_id=events_id, show=False
    )
    report.add_figure(fig, "Passive events")

    # Check events timings
    # Put all in a data frame ot see time difference between events
    events_times = [e[0] for e in events if e[2]]
    events_types = [e[2] for e in events if e[2]]
    diff = np.diff(events_times)
    diff = np.insert(diff, 0, 0)
    inv_map = {v: k for k, v in events_id.items()}
    names = [inv_map[e[2]] for e in events if e[2]]
    frame = pd.DataFrame(dict(diff=diff, type=names, times=events_times))

    # RECODE events to uniformize across participants
    id_list = {
        i: events_id[i]
        for i in [
            "rew1",
            "rew2",
            "rew3",
            "rew4",
            "rew5",
            "shk1",
            "shk2",
            "shk3",
            "shk4",
            "shk5",
            "DIN8",
        ]
    }

    inv_id_list = {v: k for k, v in id_list.items()}
    new_id_list = {
        "rew1": 81,
        "rew2": 82,
        "rew3": 83,
        "rew4": 84,
        "rew5": 85,
        "shk1": 91,
        "shk2": 92,
        "shk3": 93,
        "shk4": 94,
        "shk5": 95,
        "DIN8": 1,
    }
    val = list(id_list.values())
    for e in events:
        if e[2] in val:
            e[2] = new_id_list[inv_id_list[e[2]]]

    # Get actual difference in psychopy
    fix_dur = (passive_behav["cue.started"] - passive_behav["fixcross.started"]) * 1000
    off_dur = (
        passive_behav["blank_text.started"] - passive_behav["cue.started"]
    ) * 1000
    blank_dur = (
        passive_behav["feedback_text.started"] - passive_behav["blank_text.started"]
    ) * 1000

    # Calculate statistics of differences
    stim_frame = frame[
        frame["type"].isin(
            [
                "off",
                "shk1",
                "shk2",
                "shk3",
                "shk4",
                "shk5",
                "shk6",
                "shk8",
                "rew1",
                "rew2",
                "rew3",
                "rew4",
                "rew5",
                "rew6",
                "rew8",
            ]
        )
    ]["diff"]
    fix_dur_check = pd.DataFrame(
        np.asarray(
            frame[
                frame["type"].isin(
                    [
                        "off",
                        "shk1",
                        "shk2",
                        "shk3",
                        "shk4",
                        "shk5",
                        "shk6",
                        "shk8",
                        "rew1",
                        "rew2",
                        "rew3",
                        "rew4",
                        "rew5",
                        "rew6",
                        "rew8",
                    ]
                )
            ]["diff"]
        )
        - fix_dur
    ).describe()

    fig = plt.figure()
    sns.histplot(
        np.asarray(
            frame[
                frame["type"].isin(
                    [
                        "off",
                        "shk1",
                        "shk2",
                        "shk3",
                        "shk4",
                        "shk5",
                        "shk6",
                        "shk8",
                        "rew1",
                        "rew2",
                        "rew3",
                        "rew4",
                        "rew5",
                        "rew6",
                        "rew8",
                    ]
                )
            ]["diff"]
        )
        - fix_dur
    )
    plt.title("Distribution of fix duration error")
    report.add_figure(fig, "Distribution of fix duration error")

    off_dur_check = pd.DataFrame(
        np.asarray(frame[frame["type"] == "blan"]["diff"]) - off_dur
    ).describe()

    fig = plt.figure()
    sns.histplot(np.asarray(frame[frame["type"] == "blan"]["diff"]) - off_dur)
    plt.title("Distribution of offer duration error")
    report.add_figure(fig, "Distribution of offer duration error")

    blank_dur_check = pd.DataFrame(
        np.asarray(frame[frame["type"].isin(["DIN8", "mon+"])]["diff"]) - blank_dur
    ).describe()

    fig = plt.figure()
    sns.histplot(
        np.asarray(frame[frame["type"].isin(["DIN8", "mon+"])]["diff"]) - blank_dur
    )
    plt.title("Distribution of blank duration error")
    report.add_figure(fig, "Distribution of blank duration error")

    timing_frame = pd.concat([fix_dur_check, off_dur_check, blank_dur_check], axis=1)
    timing_frame.columns = [
        "fix_duration_error",
        "cue_duration_error",
        "blank_duration_error",
    ]

    report.add_html(timing_frame.to_html(), "Timing duration errors (ms) - passive")
    plt.close("all")

    ###############################################################################
    # Run ICA on segmented data
    ###############################################################################

    # Remove shock to avoid overlap in ICA (will be included in shkX epochs)
    id_list = new_id_list.copy()
    del id_list["DIN8"]

    # Epoch -2 to 2 seconds around the event
    epo_cues = (
        mne.Epochs(
            passive_raw.copy().load_data().filter(1, 100),
            event_id=id_list,
            baseline=None,
            tmin=-1,
            tmax=3,
            preload=True,
            events=events,
        )
        .drop_bad(reject=dict(eeg=500e-6))
        .set_eeg_reference("average", projection=False)
    )

    # Use ICA to remove eog artifacts (with decimation)
    ica = ICA(random_state=1, method="infomax", fit_params=dict(extended=True))
    ica.fit(epo_cues, decim=4)

    ica_labels = label_components(epo_cues, ica, method="iclabel")

    remove = [
        (
            1
            if ic
            in [
                "channel noise",
                "eye blink",
                "muscle artifact",
                "line noise",
                "heart beat",
                "eye movement",
                "other",
            ]
            and prob > 0.70
            else 0
        )
        for ic, prob in zip(ica_labels["labels"], ica_labels["y_pred_proba"])
    ]
    ica.exclude = list(np.argwhere(remove).flatten())

    stats_frame.loc[p, "n_removed_icas_passive"] = len(ica.exclude)

    report.add_html(pd.DataFrame(ica_labels).to_html(), "ICA labels - passive cues")

    # PLot ICA
    fig = ica.plot_components(show=False, res=25)
    report.add_ica(ica=ica, inst=epo_cues, title="ICA components - passive cues")

    # Remove ICA flagged by IClabels
    passive_raw = ica.apply(passive_raw.load_data())

    # Filter
    passive_raw.load_data().filter(0.1, 100)

    # Set average reference
    passive_raw = passive_raw.set_eeg_reference("average", projection=False)

    # Interpolate bad channels
    passive_raw = passive_raw.interpolate_bads()

    # Save cleaned data
    passive_raw.save(
        opj(outpath, p, "eeg", p + "_passive_cleaned-raw.fif"), overwrite=True
    )

    report.save(
        opj(outpath, p, "eeg", p + "_preprocess_report.html"),
        overwrite=True,
        open_browser=False,
    )

    # Clear memory
    epo_cues, passive_raw = None, None

    plt.close("all")

    ###############################################################################
    ###############################################################################
    # Decision task
    ###############################################################################
    ###############################################################################

    decision_file = opj(basepath, p, "eeg", p + "_task-decision_eeg.vhdr")
    decision_behav = pd.read_csv(
        opj(basepath, p, "eeg", p + "_task-decision_beh.tsv"), sep="\t"
    )
    decision_behav = decision_behav[~decision_behav["fixcross.started"].isna()]

    # Acceptance matrix
    decision_behav["moneylevel"] = [
        int(m.replace("m", "").replace("'", "")) for m in decision_behav["moneystim"]
    ]

    heat_acc = decision_behav.pivot_table(
        "accepted", "moneylevel", "painlevel", aggfunc=np.mean
    )

    heat_rt = decision_behav.pivot_table(
        "choice_resp.rt", "moneylevel", "painlevel", aggfunc=np.mean
    )

    heat_acc.columns = [1, 2, 3, 4, 5]

    fig, ax = plt.subplots(1, 1, figsize=(2.5, 2))
    sns.heatmap(heat_acc, cmap="cividis", annot=False, axes=ax)
    ax.figure.axes[-1].set_ylabel("Prop. accepted", fontsize=9)
    ax.figure.axes[-1].tick_params(labelsize=9)
    ax.set_xlabel("Pain rank", fontsize=9)
    ax.set_ylabel("Money ($)", fontsize=9)
    ax.set_ylim(0, 5)
    ax.tick_params(axis="both", which="major", labelsize=9)
    ax.tick_params(axis="both", which="minor", labelsize=9)
    fig.tight_layout()
    report.add_figure(fig, "Decision acceptance matrix")

    fig, ax = plt.subplots(1, 1, figsize=(2.5, 2))
    sns.heatmap(heat_rt, cmap="viridis", annot=False, axes=ax)
    ax.figure.axes[-1].set_ylabel("RT (s)", fontsize=9)
    ax.figure.axes[-1].tick_params(labelsize=9)
    ax.set_xlabel("Pain rank", fontsize=9)
    ax.set_ylabel("Money ($)", fontsize=9)
    ax.set_ylim(0, 5)
    ax.tick_params(axis="both", which="major", labelsize=9)
    ax.tick_params(axis="both", which="minor", labelsize=9)
    fig.tight_layout()
    report.add_figure(fig, "Decision rt matrix")

    # Acceptance rate
    stats_frame.loc[p, "acceptance_rate_decision"] = decision_behav["accepted"].mean(
        numeric_only=True
    )

    # Load data and include only events we care about using the "include" parameter
    decision_raw = mne.io.read_raw_brainvision(decision_file)

    # Rename channels
    decision_raw.set_montage("easycap-M1")

    # Load channel definitions to get channels marked as bad in inspection
    channels = pd.read_csv(
        opj(basepath, p, "eeg", p + "_task-decision_channels.tsv"), sep="\t"
    )
    decision_raw.info["bads"] = channels[channels["status"] != "good"]["name"].tolist()

    stats_frame.loc[p, "n_bad_chans_decision"] = len(decision_raw.info["bads"])

    # plot channels
    fig = decision_raw.plot_sensors(show_names=True)
    report.add_figure(fig, "Sensor positions (bad in red) decision")

    # Plot spectrum
    fig = (
        decision_raw.copy()
        .drop_channels(["FCz"])
        .plot_psd(fmax=100, tmax=200, show=False)
    )
    report.add_figure(fig, "Decision raw spectrum")

    # Get events
    events, events_id = mne.events_from_annotations(decision_raw)

    # Remove the "New Segment" event
    events_id = {
        k.replace("Comment/", ""): v for k, v in events_id.items() if v != 99999
    }
    events = events[events[:, 2] != 99999]

    # Remove the "DIN7" event (photosensor)
    if "DIN7" in events_id.keys():
        events = events[events[:, 2] != events_id["DIN7"]]
        events_id = {k: v for k, v in events_id.items() if k != "DIN7"}

    # Plot events
    fig = mne.viz.plot_events(
        events, first_samp=decision_raw.first_samp, event_id=events_id, show=False
    )
    report.add_figure(fig, "Decision events")

    # Check events timings
    # Put all in a data frame ot see time difference between events
    events_times = [e[0] for e in events if e[2]]
    events_types = [e[2] for e in events if e[2]]
    diff = np.diff(events_times)
    diff = np.insert(diff, 0, 0)
    inv_map = {v: k for k, v in events_id.items()}
    names = [inv_map[e[2]] for e in events if e[2]]
    frame = pd.DataFrame(dict(diff=diff, type=names, times=events_times))

    # RECODE events to uniformize across participants
    id_list = {
        i: events_id[i]
        for i in ["off+", "DIN8", "res+", "res-", "resm", "fee+", "fee-", "shk-"]
        if i in events_id.keys()
    }

    new_id_list = {
        "off+": 2,
        "DIN8": 1,
        "res+": 3,
        "res-": 4,
        "resm": 5,
        "fix+": 6,
        "fee+": 7,
        "fee-": 8,
        "fix+": 9,
        "cdow": 10,
        "shk-": 11,
    }
    inv_id_list = {v: k for k, v in id_list.items()}

    val = list(id_list.values())
    for e in events:
        if e[2] in val:
            e[2] = new_id_list[inv_id_list[e[2]]]

    # Get actual difference in psychopy
    fix_dur = (
        decision_behav["cue_right1.started"] - decision_behav["fixcross.started"]
    ) * 1000
    off_dur = (decision_behav["choice_resp.rt"]) * 1000
    feed_dur = (
        decision_behav["feedback_text.started"] - decision_behav["choice_resp.started"]
    ) * 1000
    cdowndur = np.append(np.asarray(decision_behav["fixcross.started"]), 0) - np.insert(
        np.asarray(decision_behav["coold.started"]), 0, 0
    )

    # Calculate statistics of differences
    fix_dur_check = pd.DataFrame(
        np.asarray(frame[frame["type"] == "off+"]["diff"]) - fix_dur
    ).describe()
    off_dur_check = pd.DataFrame(
        np.asarray(frame[frame["type"].isin(["res+", "res-", "resp", "resm"])]["diff"])
        - off_dur
    ).describe()

    fig = plt.figure()
    sns.histplot(np.asarray(frame[frame["type"] == "off+"]["diff"]) - fix_dur)
    plt.title("Distribution of fix duration error")
    report.add_figure(fig, "Distribution of fix duration error")

    fig = plt.figure()
    sns.histplot(
        np.asarray(frame[frame["type"].isin(["res+", "res-", "resp", "resm"])]["diff"])
        - off_dur
    )
    plt.title("Distribution of offer duration error")
    report.add_figure(fig, "Distribution of offer duration error")

    timing_frame = pd.concat([fix_dur_check, off_dur_check], axis=1)
    timing_frame.columns = ["fix_duration_error", "response_duration_error"]

    report.add_html(timing_frame.to_html(), "Timing duration errors (ms) - decision")
    plt.close("all")

    ###############################################################################
    # ICA for decision task
    ###############################################################################

    # Remove very bad epochs and filter for ICA, average reference for ICAlabel
    epo_cues = (
        mne.Epochs(
            decision_raw.copy().load_data().filter(1, 100),
            event_id={"offer": 2},
            baseline=None,
            tmin=-1,
            tmax=3,
            preload=True,
            events=events,
        )
        .drop_bad(reject=dict(eeg=500e-6))
        .set_eeg_reference("average")
    )

    # Use ICA to remove eog artifacts
    ica = ICA(random_state=1, method="infomax", fit_params=dict(extended=True))
    ica.fit(epo_cues, decim=4)
    ica_labels = label_components(epo_cues, ica, method="iclabel")

    remove = [
        (
            1
            if ic
            in [
                "channel noise",
                "eye blink",
                "muscle artifact",
                "line noise",
                "heart beat",
                "eye movement",
                "other",
            ]
            and prob > 0.70
            else 0
        )
        for ic, prob in zip(ica_labels["labels"], ica_labels["y_pred_proba"])
    ]
    ica.exclude = list(np.argwhere(remove).flatten())

    stats_frame.loc[p, "n_removed_icas_decision"] = len(ica.exclude)

    report.add_html(pd.DataFrame(ica_labels).to_html(), "ICA labels - decision cues")

    # PLot ICA
    report.add_ica(ica=ica, inst=epo_cues, title="ICA components - decision cues")

    # Remove ICA flagged by IClabels
    decision_raw = ica.apply(decision_raw.load_data())

    # Filter
    decision_raw.load_data().filter(0.1, 100)

    # Set average reference
    decision_raw = decision_raw.set_eeg_reference("average", projection=False)

    # Interpolate bad channels
    decision_raw = decision_raw.interpolate_bads()

    # Save cleaned data
    decision_raw.save(
        opj(outpath, p, "eeg", p + "_decision_cleaned-raw.fif"), overwrite=True
    )

    # Save stats frame for this part and add to report
    stats_frame[stats_frame.index == p].to_csv(
        opj(outpath, p, "eeg", p + "_preprocess_stats.csv")
    )
    report.add_html(stats_frame[stats_frame.index == p].to_html(), "Participant stats")

    # Save final report
    report.save(
        opj(outpath, p, "eeg", p + "_preprocess_report.html"),
        overwrite=True,
        open_browser=False,
    )

    # Clear memory
    epo_cues, decision_raw = None, None
    plt.close("all")

# Save stats frame for all partiicpants
stats_frame.to_csv(opj(outpath, "preprocess_stats.csv"))
