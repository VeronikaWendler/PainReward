# @ : -*- coding: utf-8 -*-
# @ Author: Michel-Pierre Coll (michel-pierre.coll@psy.ulaval.ca) & Veronika Wendler (2025)
# @ Date: 2024
# @ Description:
#
# 1. versions
# 2. cleaning and z scoring
# 3. grand average & second-level mass-univariate inference
# 4. switchable inference: classic cluster-mass OR TFCE
# 5. across-map correction 
#

import os
import warnings
from pathlib import Path
from os.path import join as opj

import mne
import numpy as np
import pandas as pd
from mne.decoding import Scaler
from mne.stats import spatio_temporal_cluster_1samp_test as st_clust_1s_ttest
from scipy import stats
from bids import BIDSLayout

warnings.simplefilter(action="ignore", category=FutureWarning)

# -----------------------
# Paths
# -----------------------
PROJECT_DIR = Path(os.getenv("PROJECT_DIR", "/workspace"))
basepath = Path(os.getenv(
    "DATA_DIR",
    PROJECT_DIR / "EEG" / "PainReward_sub-001-050" / "painrewardeegdata"
))
HDDM_DIR = Path(os.getenv("HDDM_DIR", ""))
layout = BIDSLayout(basepath)

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

outroot = Path(os.getenv("OUT_DIR", basepath / "statistics"))
ensure_dir(outroot)

# -----------------------
# Params
# -----------------------
param = {
    "njobs": 20,
    "nperms": 5000,
    "random_state": 23,
    "testresampfreq": 1024,

    # Inference mode:
    #   "tfce"    -> threshold-free cluster enhancement
    #   "cluster" -> classic cluster-mass inference
    "inference_method": "tfce",

    # For classic cluster inference only:
    # cluster-forming p-threshold (two-sided converted to t-threshold)
    "cluster_forming_p": 0.01,

    # For TFCE only:
    # MNE activates TFCE when threshold is a dict with "start" and "step"
    "tfce_start": 0.0,
    "tfce_step": 0.2,

    # Across-map correction (pain, money, diff)
    "map_alpha": 0.05,
    "map_correction": "holm",   # "holm", "bonferroni", or "none"

    # Pointwise threshold for writing binary significance masks
    "point_alpha": 0.05,
}

# -----------------------
# Choose version
# -----------------------
# version = 1 passive level from passive_beh.tsv; no RT
# version = 2 decision HDDM-based pain/money + RT
version = 1

if version == 1:
    outpath = outroot / "erps_massuni_passive"
elif version == 2:
    outpath = outroot / "erps_massuni_decision"
else:
    raise ValueError("version must be 1 (passive) or 2 (decision)")
ensure_dir(outpath)

# TFCE and cluster outputs are separate
z_dir = outpath / f"Zscoring_{param['inference_method'].lower()}"
ensure_dir(z_dir)

# -----------------------
# Participants
# -----------------------
part_csv = basepath / "participants.tsv"
part_eeg = pd.read_csv(part_csv, sep="\t")["participant_id"].unique().tolist()
part_eeg.sort()

# -----------------------
# Helper: EEG ∩ HDDM
# -----------------------
def get_common_subjects_eeg_hddm(participants_eeg):
    if HDDM_DIR == Path("") or not HDDM_DIR.exists():
        raise RuntimeError(
            "HDDM_DIR is not set or does not exist, but you asked to restrict subjects to EEG ∩ HDDM."
        )

    mod_data_path = (
        HDDM_DIR / "figures" / "painreward_behavioural_data_mod_9" /
        "diagnostics" / "v_pain_money.csv"
    )
    if not mod_data_path.exists():
        raise RuntimeError(f"Missing HDDM decision file: {mod_data_path}")

    mod_data = pd.read_csv(mod_data_path, sep=None, engine="python")
    hddm_subjects = sorted(mod_data["participant"].unique().tolist())

    common = sorted(list(set(participants_eeg) & set(hddm_subjects)))
    return common, mod_data

common_participants, mod_data_decision = get_common_subjects_eeg_hddm(part_eeg)
print("\nSubjects in EEG ∩ HDDM:", common_participants)
print("N =", len(common_participants))

# -----------------------
# Inference helpers
# -----------------------
def compute_threshold(n_samples, param):
    """
    Returns the threshold argument to pass into
    mne.stats.spatio_temporal_cluster_1samp_test.

    - For classic cluster inference: numeric t-threshold
    - For TFCE: dict(start=..., step=...)
    """
    method = param["inference_method"].lower()

    if method == "tfce":
        return {
            "start": float(param["tfce_start"]),
            "step": float(param["tfce_step"]),
        }

    if method == "cluster":
        # two-sided threshold because tail=0 below
        p_thresh = float(param["cluster_forming_p"]) / 2.0
        t_thresh = -stats.t.ppf(p_thresh, n_samples - 1)
        return float(t_thresh)

    raise ValueError(f"Unknown inference_method: {method}")


def run_massuni_test(data_3d, connect, param):
    """
    data_3d: (n_subj, n_ch, n_time)

    Returns dict with:
        stat_map:          (n_time, n_ch)
        clusters:          cluster definitions returned by MNE
        cluster_p_values:  corrected p-values returned by MNE
        pmap_corrected:    corrected pointwise p-map
        sig_mask:          corrected significance mask
        threshold_used:    numeric threshold or TFCE dict
        method:            "cluster" or "tfce"
    """
    testdata = np.swapaxes(data_3d, 2, 1)  # -> (n_subj, n_time, n_ch)
    threshold = compute_threshold(testdata.shape[0], param)

    stat_map, clusters, cluster_p_values, _ = st_clust_1s_ttest(
        testdata,
        n_jobs=param["njobs"],
        threshold=threshold,
        adjacency=connect,
        n_permutations=param["nperms"],
        buffer_size=None,
        seed=param["random_state"],
        tail=0,  # two-sided
    )

    cluster_p_values = np.asarray(cluster_p_values, dtype=float)

    # Build corrected pointwise p-map
    pmap_corrected = np.ones(stat_map.shape, dtype=float)
    for clu, p_val in zip(clusters, cluster_p_values):
        pmap_corrected[clu] = np.minimum(pmap_corrected[clu], p_val)

    sig_mask = pmap_corrected < float(param["point_alpha"])

    return {
        "stat_map": stat_map,
        "clusters": clusters,
        "cluster_p_values": cluster_p_values,
        "pmap_corrected": pmap_corrected,
        "sig_mask": sig_mask,
        "threshold_used": threshold,
        "method": param["inference_method"].lower(),
    }


def extract_map_level_p(result):
    """
    Derive one omnibus p-value per statistical map.

    We use the minimum corrected cluster/TFCE p-value returned by MNE.
    If nothing is returned, assign p=1.
    """
    pvals = np.asarray(result["cluster_p_values"], dtype=float)
    if pvals.size == 0:
        return 1.0
    return float(np.min(pvals))


def holm_correction(pvals, alpha=0.05):
    """
    Holm-Bonferroni correction.
    Returns:
        reject, p_adj
    """
    pvals = np.asarray(pvals, dtype=float)
    m = len(pvals)

    if m == 0:
        return np.array([], dtype=bool), np.array([], dtype=float)

    order = np.argsort(pvals)
    p_sorted = pvals[order]

    adj_sorted = np.empty(m, dtype=float)
    for i, p in enumerate(p_sorted):
        adj_sorted[i] = (m - i) * p

    # enforce monotonicity
    adj_sorted = np.maximum.accumulate(adj_sorted)
    adj_sorted = np.clip(adj_sorted, 0, 1)

    p_adj = np.empty(m, dtype=float)
    p_adj[order] = adj_sorted

    reject = p_adj < float(alpha)
    return reject, p_adj


def correct_across_maps(results, alpha=0.05, method="holm"):
    """
    results: list of dicts returned by run_massuni_test, each with a 'name' key.

    Adds:
        map_p_raw
        map_p_adj
        map_sig

    Returns:
        updated results, map_table
    """
    raw_ps = np.array([extract_map_level_p(r) for r in results], dtype=float)

    method = method.lower()
    if method == "holm":
        reject, p_adj = holm_correction(raw_ps, alpha=alpha)
    elif method == "bonferroni":
        p_adj = np.clip(raw_ps * len(raw_ps), 0, 1)
        reject = p_adj < alpha
    elif method == "none":
        p_adj = raw_ps.copy()
        reject = p_adj < alpha
    else:
        raise ValueError(f"Unknown map_correction method: {method}")

    rows = []
    for i, r in enumerate(results):
        r["map_p_raw"] = float(raw_ps[i])
        r["map_p_adj"] = float(p_adj[i])
        r["map_sig"] = bool(reject[i])

        rows.append({
            "map": r["name"],
            "inference_method": r["method"],
            "threshold_used": str(r["threshold_used"]),
            "map_p_raw": r["map_p_raw"],
            "map_p_adj": r["map_p_adj"],
            "map_sig": r["map_sig"],
        })

    return results, pd.DataFrame(rows)


def save_massuni_outputs(results, regvars_main, z_dir):
    """
    Saves per-map stats and corrected p-maps.
    """
    for res in results:
        name = res["name"]

        np.save(z_dir / f"ols_2ndlevel_tval_{name}.npy", res["stat_map"])
        np.save(z_dir / f"ols_2ndlevel_pval_corr_{name}.npy", res["pmap_corrected"])
        np.save(z_dir / f"ols_2ndlevel_sigmask_{name}.npy", res["sig_mask"])
        np.save(
            z_dir / f"ols_2ndlevel_cluster_pvals_{name}.npy",
            np.asarray(res["cluster_p_values"], dtype=float),
        )

    main_results = [r for r in results if r["name"] in regvars_main]

    tvals = np.stack([r["stat_map"] for r in main_results])
    pvals_corr = np.stack([r["pmap_corrected"] for r in main_results])
    sigmasks = np.stack([r["sig_mask"] for r in main_results])

    np.save(z_dir / "ols_2ndlevel_tvals.npy", tvals)
    np.save(z_dir / "ols_2ndlevel_pvals.npy", pvals_corr)  # backward-compatible final corrected maps
    np.save(z_dir / "ols_2ndlevel_pvals_corr.npy", pvals_corr)
    np.save(z_dir / "ols_2ndlevel_sigmasks.npy", sigmasks)


def run_second_level_family(allbetas, beta_gavg, regvars, z_dir, param):
    """
    Shared second-level inference for both version 1 and version 2.

    allbetas shape: (n_subj, 2, n_ch, n_time)
    """
    connect, _ = mne.channels.find_ch_adjacency(beta_gavg[0].info, ch_type="eeg")

    massuni_results = []

    for idx, regvar in enumerate(regvars):
        print(f"\nSecond-level {param['inference_method'].upper()} test for {regvar}")
        data_reg = allbetas[:, idx, :, :]
        res_reg = run_massuni_test(data_reg, connect, param)
        res_reg["name"] = regvar
        massuni_results.append(res_reg)

    print(f"\nSecond-level {param['inference_method'].upper()} test for pain - money beta difference ...")
    beta_diff = allbetas[:, 0, :, :] - allbetas[:, 1, :, :]
    res_diff = run_massuni_test(beta_diff, connect, param)
    res_diff["name"] = "diff_pain_minus_money"
    massuni_results.append(res_diff)

    massuni_results, map_table = correct_across_maps(
        massuni_results,
        alpha=param["map_alpha"],
        method=param["map_correction"],
    )
    map_table.to_csv(z_dir / "map_table_corrected.csv", index=False)

    save_massuni_outputs(massuni_results, regvars_main=regvars, z_dir=z_dir)

    diff_res = [r for r in massuni_results if r["name"] == "diff_pain_minus_money"][0]
    np.save(z_dir / "ols_2ndlevel_tval_diff_pain_minus_money.npy", diff_res["stat_map"])
    np.save(z_dir / "ols_2ndlevel_pval_diff_pain_minus_money.npy", diff_res["pmap_corrected"])
    np.save(z_dir / "ols_2ndlevel_sigmask_diff_pain_minus_money.npy", diff_res["sig_mask"])

    return massuni_results, map_table

# ============================================================
# v2 DECISION
# ============================================================
if version == 2:
    regvars = ["painlevel", "moneylevel"]

    mod_data = mod_data_decision.copy()
    mod_data = mod_data[mod_data["participant"].isin(common_participants)].copy()
    mod_data["rt"] = mod_data["choice_resp.rt"]
    mod_data["trialsnum"] = (
        mod_data["blocks.thisRepN"].astype(int) * 25
        + mod_data["trials.thisN"].astype(int)
        + 1
    )

    part = common_participants
    part_1_dat = mod_data
    part_1 = part

    # ------------ Create trial map for decision ------------
    filtered_data = []
    for p in part:
        df = mod_data[mod_data["participant"] == p]

        epo = mne.read_epochs(
            opj(basepath, "derivatives", p, "eeg", "erps",
                f"{p}_decision_cues_singletrials-epo.fif"),
            preload=True
        )
        epo_1 = epo.copy()

        participants = epo_1.metadata["participant_id"].unique()
        trialblocks = []
        blocks_idx = []

        for _participant in participants:
            blocks = list(range(25)) * 5
            blocks_idx_participant = [i for i in range(5) for _ in range(25)]
            trialblocks.extend(blocks)
            blocks_idx.extend(blocks_idx_participant)

        epo_1.metadata["trialblocks"] = trialblocks
        epo_1.metadata["blocks_idx"] = blocks_idx

        epo_1_filtered = pd.DataFrame()

        for participant in df["participant"].unique():
            erps_p_df = epo_1.metadata[epo_1.metadata["participant_id"] == participant]
            df_unique = df[df["participant"] == participant]

            for block_x in df_unique["blocks.thisRepN"].unique():
                erps_block_df = erps_p_df[erps_p_df["blocks_idx"] == block_x]
                df_block_df = df_unique[df_unique["blocks.thisRepN"] == block_x]

                filtered_block_df = erps_block_df[
                    erps_block_df["trialblocks"].isin(df_block_df["trials.thisN"])
                ]
                epo_1_filtered = pd.concat([epo_1_filtered, filtered_block_df], ignore_index=True)

        filtered_data.append(epo_1_filtered)

    epo_1_filtered_combined = pd.concat(filtered_data, ignore_index=True)

    # ------------ Run decision regression ------------
    all_epos = [[] for _ in range(len(regvars))]
    allbetasnp = []
    betas = [[] for _ in range(len(regvars))]
    included_subjects = []
    skipped_subjects = []

    for pa in part_1:
        print(f"\n--- DECISION (v2): Processing {pa} ---")

        df2 = epo_1_filtered_combined[epo_1_filtered_combined["participant_id"] == pa]
        mod2 = part_1_dat[part_1_dat["participant"] == pa].copy()

        epo = mne.read_epochs(
            opj(basepath, "derivatives", pa, "eeg", "erps",
                f"{pa}_decision_cues_singletrials-epo.fif"),
            preload=True
        )
        epo_cop = epo.copy()

        matching = epo_cop.metadata["trialsnum"].isin(df2["trialsnum"])
        epo_filt = epo_cop[matching]

        if epo_filt.info["sfreq"] != param["testresampfreq"]:
            epo_filt = epo_filt.resample(param["testresampfreq"])

        goodtrials = np.where(epo_filt.metadata["badtrial"] == 0)[0]
        mod2 = mod2.iloc[goodtrials].reset_index(drop=True)
        epo_filt = epo_filt[goodtrials]

        if len(epo_filt) < 5:
            print(f"Skipping {pa}: too few trials")
            skipped_subjects.append(pa)
            continue

        rt_col = "rt"
        vals_pain = mod2["painlevel"].to_numpy(dtype=float)
        vals_money = mod2["moneylevel"].to_numpy(dtype=float)
        vals_rt = mod2[rt_col].to_numpy(dtype=float)

        keep = np.where(np.isfinite(vals_pain) & np.isfinite(vals_money) & np.isfinite(vals_rt))[0]
        if len(keep) < 5:
            print(f"Skipping {pa}: too few valid trials")
            skipped_subjects.append(pa)
            continue

        mod2k = mod2.iloc[keep].reset_index(drop=True)
        epo_keep = epo_filt.copy()[keep]

        if epo_keep.metadata is None:
            epo_keep.metadata = pd.DataFrame(index=np.arange(len(epo_keep)))
        else:
            epo_keep.metadata = epo_keep.metadata.reset_index(drop=True).copy()

        epo_keep.metadata["painlevel"] = mod2k["painlevel"].to_numpy(dtype=float)
        epo_keep.metadata["moneylevel"] = mod2k["moneylevel"].to_numpy(dtype=float)
        epo_keep.metadata["participant_id"] = pa
        epo_keep.metadata["rt"] = mod2k[rt_col].to_numpy(dtype=float)

        scale = Scaler(scalings="mean")
        epo_z = mne.EpochsArray(scale.fit_transform(epo_keep.get_data()), epo_keep.info)

        df_reg = mod2k.copy()
        df_reg["Intercept"] = 1.0
        df_reg["pain_z"] = stats.zscore(df_reg["painlevel"].to_numpy(dtype=float))
        df_reg["money_z"] = stats.zscore(df_reg["moneylevel"].to_numpy(dtype=float))
        df_reg["RT_z"] = stats.zscore(df_reg[rt_col].to_numpy(dtype=float))

        names = ["Intercept", "pain_z", "money_z", "RT_z"]
        design = df_reg[names]

        if not np.all(np.isfinite(design.to_numpy())):
            print(f"Skipping {pa}: NaN/Inf in design")
            skipped_subjects.append(pa)
            continue

        res = mne.stats.linear_regression(epo_z, design, names=names)
        beta_pain = res["pain_z"].beta
        beta_money = res["money_z"].beta

        betas[0].append(beta_pain)
        betas[1].append(beta_money)
        allbetasnp.append(np.stack([beta_pain.data, beta_money.data]))

        all_epos[0].append(epo_keep)
        all_epos[1].append(epo_keep)

        included_subjects.append(pa)

    if len(allbetasnp) == 0:
        raise RuntimeError("No subjects included (decision).")

    allbetas = np.stack(allbetasnp)  # (n_subj, 2, n_ch, n_time)
    beta_gavg = [mne.grand_average(betas[i]) for i in range(len(regvars))]

    massuni_results, map_table = run_second_level_family(
        allbetas=allbetas,
        beta_gavg=beta_gavg,
        regvars=regvars,
        z_dir=z_dir,
        param=param,
    )

    np.save(z_dir / "ols_2ndlevel_betas.npy", allbetas)
    np.save(z_dir / "included_subjects.npy", np.array(included_subjects, dtype=object))
    np.save(z_dir / "ols_2ndlevel_betasavg.npy", np.array(beta_gavg, dtype=object))

    for idx, regvar in enumerate(regvars):
        if len(all_epos[idx]) == 0:
            continue
        epo_save = mne.concatenate_epochs(all_epos[idx])
        epo_save.save(z_dir / f"ols_2ndlevel_allepochs-epo_{regvar}.fif", overwrite=True)

    print("\nDECISION finished.")
    print("Inference method:", param["inference_method"])
    print("Saved outputs in:", z_dir)
    print("Included subjects:", included_subjects)
    print("Skipped subjects:", skipped_subjects)

# ============================================================
# v1 PASSIVE
# ============================================================
elif version == 1:
    regvars = ["painlevel", "moneylevel"]

    all_epos = [[] for _ in range(len(regvars))]
    allbetasnp = []
    betas = [[] for _ in range(len(regvars))]
    included_subjects = []
    skipped_subjects = []

    def find_passive_epochs(pa: str) -> Path:
        cand1 = basepath / "derivatives" / pa / "eeg" / "erps_passive" / f"{pa}_passive_cues_singletrials-epo.fif"
        cand2 = basepath / pa / "eeg" / "erps_passive" / f"{pa}_passive_cues_singletrials-epo.fif"
        return cand1 if cand1.exists() else cand2

    def find_passive_beh(pa: str) -> Path:
        cand1 = basepath / pa / "eeg" / f"{pa}_task-passive_beh.tsv"
        cand2 = basepath / "derivatives" / pa / "eeg" / f"{pa}_task-passive_beh.tsv"
        return cand1 if cand1.exists() else cand2

    def load_passive_beh_with_trialsnum(beh_path: Path) -> pd.DataFrame:
        beh = pd.read_csv(beh_path, sep="\t")

        if "fixcross.started" in beh.columns:
            beh = beh[~beh["fixcross.started"].isna()].copy()

        beh = beh.reset_index(drop=True)
        beh["trialsnum"] = np.arange(1, len(beh) + 1)

        if "condition" not in beh.columns or "level" not in beh.columns:
            raise ValueError(f"Passive beh file missing 'condition'/'level'. Columns: {list(beh.columns)}")

        beh["condition"] = beh["condition"].astype(str).str.lower().str.strip()
        beh["level"] = pd.to_numeric(beh["level"], errors="coerce")
        beh = beh[beh["condition"].isin(["p", "m"]) & np.isfinite(beh["level"])].copy()
        return beh

    def merge_beh_into_epochs_on_trialsnum(epo: mne.Epochs, beh: pd.DataFrame, pa: str) -> mne.Epochs:
        if epo.metadata is None:
            md = pd.DataFrame({"trialsnum": np.arange(1, len(epo) + 1)})
        else:
            md = epo.metadata.reset_index(drop=True).copy()

        if "trialsnum" not in md.columns:
            md["trialsnum"] = np.arange(1, len(epo) + 1)

        md["trialsnum"] = pd.to_numeric(md["trialsnum"], errors="coerce").astype(int)
        beh["trialsnum"] = pd.to_numeric(beh["trialsnum"], errors="coerce").astype(int)

        merged = md.merge(
            beh[["trialsnum", "condition", "level"]],
            on="trialsnum",
            how="left",
            validate="1:1",
        )

        if merged["condition"].isna().any() or merged["level"].isna().any():
            n_bad = int(merged["condition"].isna().sum())
            raise ValueError(
                f"{pa}: trialsnum merge produced {n_bad} unlabeled epochs. "
                "This suggests epochs order and beh.tsv order differ for this subject."
            )

        epo.metadata = merged
        return epo

    part_passive = []
    for p in common_participants:
        if find_passive_epochs(p).exists() and find_passive_beh(p).exists():
            part_passive.append(p)

    print(f"\nPASSIVE: subjects with epochs+beh AND in EEG∩HDDM: {len(part_passive)}")

    for pa in part_passive:
        print(f"\n--- PASSIVE (v1): Processing {pa} ---")
        try:
            epo_path = find_passive_epochs(pa)
            beh_path = find_passive_beh(pa)

            epo = mne.read_epochs(str(epo_path), preload=True)
            beh = load_passive_beh_with_trialsnum(beh_path)

            epo = merge_beh_into_epochs_on_trialsnum(epo, beh, pa)

            if epo.info["sfreq"] != param["testresampfreq"]:
                epo = epo.resample(param["testresampfreq"])

            if "badtrial" in epo.metadata.columns:
                good_mask = epo.metadata["badtrial"].fillna(0).astype(int).to_numpy() == 0
                epo = epo.copy()[good_mask]

            if len(epo) < 20:
                print(f"Skipping {pa}: too few epochs after dropping bad trials (n={len(epo)})")
                skipped_subjects.append(pa)
                continue

            cond = epo.metadata["condition"].astype(str).str.lower().str.strip().to_numpy()
            level = epo.metadata["level"].to_numpy(dtype=float)

            is_pain = cond == "p"
            is_money = cond == "m"

            if is_pain.sum() < 5 or is_money.sum() < 5:
                print(f"Skipping {pa}: too few trials per condition (p={is_pain.sum()}, m={is_money.sum()})")
                skipped_subjects.append(pa)
                continue

            if np.nanstd(level[is_pain]) == 0 or np.nanstd(level[is_money]) == 0:
                print(f"Skipping {pa}: zero variance in level within condition")
                skipped_subjects.append(pa)
                continue

            epo.metadata["painlevel"] = np.where(is_pain, level, 0.0)
            epo.metadata["moneylevel"] = np.where(is_money, level, 0.0)
            epo.metadata["participant_id"] = pa

            scale = Scaler(scalings="mean")
            epo_z = mne.EpochsArray(scale.fit_transform(epo.get_data()), epo.info)

            df_reg = epo.metadata.copy()
            df_reg["Intercept"] = 1.0
            df_reg["cue_type_pm"] = np.where(is_pain, 0.5, -0.5)

            pain_z = np.zeros(len(df_reg), dtype=float)
            money_z = np.zeros(len(df_reg), dtype=float)
            pain_z[is_pain] = stats.zscore(level[is_pain])
            money_z[is_money] = stats.zscore(level[is_money])

            df_reg["pain_z_masked"] = pain_z
            df_reg["money_z_masked"] = money_z

            names = ["Intercept", "cue_type_pm", "pain_z_masked", "money_z_masked"]
            design = df_reg[names]

            if not np.all(np.isfinite(design.to_numpy())):
                print(f"Skipping {pa}: NaN/Inf in design")
                skipped_subjects.append(pa)
                continue

            res = mne.stats.linear_regression(epo_z, design, names=names)
            beta_pain = res["pain_z_masked"].beta
            beta_money = res["money_z_masked"].beta

            betas[0].append(beta_pain)
            betas[1].append(beta_money)
            allbetasnp.append(np.stack([beta_pain.data, beta_money.data]))

            all_epos[0].append(epo)
            all_epos[1].append(epo)

            included_subjects.append(pa)
            print(f"Included {pa} (n_epochs={len(epo)})")

        except Exception as e:
            print(f"Skipping {pa}: {e}")
            skipped_subjects.append(pa)

    if len(allbetasnp) == 0:
        raise RuntimeError("No subjects included in PASSIVE analysis (after merge).")

    allbetas = np.stack(allbetasnp)
    beta_gavg = [mne.grand_average(betas[i]) for i in range(len(regvars))]

    massuni_results, map_table = run_second_level_family(
        allbetas=allbetas,
        beta_gavg=beta_gavg,
        regvars=regvars,
        z_dir=z_dir,
        param=param,
    )

    np.save(z_dir / "ols_2ndlevel_betas.npy", allbetas)
    np.save(z_dir / "included_subjects.npy", np.array(included_subjects, dtype=object))
    np.save(z_dir / "ols_2ndlevel_betasavg.npy", np.array(beta_gavg, dtype=object))

    for idx, regvar in enumerate(regvars):
        if len(all_epos[idx]) == 0:
            continue
        epo_save = mne.concatenate_epochs(all_epos[idx])
        epo_save.save(z_dir / f"ols_2ndlevel_allepochs-epo_{regvar}.fif", overwrite=True)

    print("\nPASSIVE finished.")
    print("Inference method:", param["inference_method"])
    print("Saved outputs in:", z_dir)
    print("Included subjects:", included_subjects)
    print("Skipped subjects:", skipped_subjects)

























# '''
#  # @ : -*- coding: utf-8 -*-
#  # @ Author: Michel-Pierre Coll (michel-pierre.coll@psy.ulaval.ca) & Veronika Wendler (2025)
#  # @ Date: 2024
#  # @ Description:
 
#  1.set versions
#  2.cleaning and z scoring
#  3.Grand average & second-level cluster test (versions 1–3)
 
#  '''

# import os
# import warnings
# from pathlib import Path
# from os.path import join as opj

# import mne
# import numpy as np
# import pandas as pd
# from mne.decoding import Scaler
# from mne.stats import spatio_temporal_cluster_1samp_test as st_clust_1s_ttest
# from mne.stats import fdr_correction
# from scipy import stats
# from bids import BIDSLayout

# warnings.simplefilter(action="ignore", category=FutureWarning)

# # -----------------------
# # Paths
# # -----------------------
# PROJECT_DIR = Path(os.getenv("PROJECT_DIR", "/workspace"))
# basepath = Path(os.getenv(
#     "DATA_DIR",
#     PROJECT_DIR / "EEG" / "PainReward_sub-001-050" / "painrewardeegdata"
# ))
# HDDM_DIR = Path(os.getenv("HDDM_DIR", ""))  # required for v2; also used to define subject set for v1
# layout = BIDSLayout(basepath)

# def ensure_dir(path: Path):
#     path.mkdir(parents=True, exist_ok=True)

# # Out root for stats
# outroot = Path(os.getenv("OUT_DIR", basepath / "statistics"))
# ensure_dir(outroot)

# # -----------------------
# # Choose version
# # -----------------------
# # version = 1 passive level from passive_beh.tsv; no RT
# # version = 2 decision HDDM-based pain/money + RT
# version = 1

# if version == 1:
#     outpath = outroot / "erps_massuni_passive_regression"
# elif version == 2:
#     outpath = outroot / "erps_massuni_regression"
# else:
#     raise ValueError("version must be 1 (passive) or 2 (decision)")
# ensure_dir(outpath)

# # -----------------------
# # Params
# # -----------------------
# param = {
#     "njobs": 20,
#     "nperms": 5000,
#     "random_state": 23,
#     "testresampfreq": 1024,
#     "cluster_threshold": 0.01,
# }

# # -----------------------
# # Participants from participants.tsv (EEG roster)
# # -----------------------
# part_csv = basepath / "participants.tsv"
# part_eeg = pd.read_csv(part_csv, sep="\t")["participant_id"].unique().tolist()
# part_eeg.sort()

# # -----------------------
# # Helper: subject set = EEG ∩ HDDM (same as decision selection)
# # -----------------------
# def get_common_subjects_eeg_hddm(participants_eeg):
#     if HDDM_DIR == Path("") or not HDDM_DIR.exists():
#         raise RuntimeError(
#             "HDDM_DIR is not set or does not exist, but you asked to restrict subjects to EEG ∩ HDDM."
#         )

#     mod_data_path = HDDM_DIR / "figures" / "painreward_behavioural_data_mod_9" / "diagnostics" / "v_pain_money.csv"
#     if not mod_data_path.exists():
#         raise RuntimeError(f"Missing HDDM decision file: {mod_data_path}")

#     mod_data = pd.read_csv(mod_data_path, sep=None, engine="python")
#     hddm_subjects = sorted(mod_data["participant"].unique().tolist())

#     common = sorted(list(set(participants_eeg) & set(hddm_subjects)))
#     return common, mod_data

# common_participants, mod_data_decision = get_common_subjects_eeg_hddm(part_eeg)
# print("\nSubjects in EEG ∩ HDDM (decision selection):", common_participants)
# print("N =", len(common_participants))

# # ============================================================
# # v2 DECISION (your model: EEG ~ 1 + pain_z + money_z + RT_z)
# # ============================================================
# if version == 2:
#     regvars = ["painlevel", "moneylevel"]
#     z_dir = Path(outpath) / "Zscoring"
#     ensure_dir(z_dir)

#     # limit decision mod_data to common subjects
#     mod_data = mod_data_decision.copy()
#     mod_data = mod_data[mod_data["participant"].isin(common_participants)].copy()
#     mod_data["rt"] = mod_data["choice_resp.rt"]
#     mod_data["trialsnum"] = (
#         mod_data["blocks.thisRepN"].astype(int) * 25
#         + mod_data["trials.thisN"].astype(int)
#         + 1
#     )

#     part = common_participants
#     part_1_dat = mod_data
#     part_1 = part

#     # ------------ Create trial map for decision (YOUR existing logic) ------------
#     filtered_data = []
#     for p in part:
#         df = mod_data[mod_data["participant"] == p]

#         epo = mne.read_epochs(
#             opj(basepath, "derivatives", p, "eeg", "erps",
#                 f"{p}_decision_cues_singletrials-epo.fif"),
#             preload=True
#         )
#         epo_1 = epo.copy()

#         participants = epo_1.metadata["participant_id"].unique()
#         trialblocks = []
#         blocks_idx = []

#         for _participant in participants:
#             blocks = list(range(25)) * 5
#             blocks_idx_participant = [i for i in range(5) for _ in range(25)]
#             trialblocks.extend(blocks)
#             blocks_idx.extend(blocks_idx_participant)

#         epo_1.metadata["trialblocks"] = trialblocks
#         epo_1.metadata["blocks_idx"] = blocks_idx

#         epo_1_filtered = pd.DataFrame()

#         for participant in df["participant"].unique():
#             erps_p_df = epo_1.metadata[epo_1.metadata["participant_id"] == participant]
#             df_unique = df[df["participant"] == participant]

#             for block_x in df_unique["blocks.thisRepN"].unique():
#                 erps_block_df = erps_p_df[erps_p_df["blocks_idx"] == block_x]
#                 df_block_df = df_unique[df_unique["blocks.thisRepN"] == block_x]

#                 filtered_block_df = erps_block_df[
#                     erps_block_df["trialblocks"].isin(df_block_df["trials.thisN"])
#                 ]
#                 epo_1_filtered = pd.concat([epo_1_filtered, filtered_block_df], ignore_index=True)

#         filtered_data.append(epo_1_filtered)

#     epo_1_filtered_combined = pd.concat(filtered_data, ignore_index=True)

#     # ------------ Run decision regression ------------
#     all_epos = [[] for _ in range(len(regvars))]
#     allbetasnp = []
#     betas = [[] for _ in range(len(regvars))]
#     included_subjects = []
#     skipped_subjects = []

#     for pa in part_1:
#         print(f"\n--- DECISION (v2): Processing {pa} ---")

#         df2 = epo_1_filtered_combined[epo_1_filtered_combined["participant_id"] == pa]
#         mod2 = part_1_dat[part_1_dat["participant"] == pa].copy()

#         epo = mne.read_epochs(
#             opj(basepath, "derivatives", pa, "eeg", "erps",
#                 f"{pa}_decision_cues_singletrials-epo.fif"),
#             preload=True
#         )
#         epo_cop = epo.copy()

#         # keep only mapped trials
#         matching = epo_cop.metadata["trialsnum"].isin(df2["trialsnum"])
#         epo_filt = epo_cop[matching]

#         if epo_filt.info["sfreq"] != param["testresampfreq"]:
#             epo_filt = epo_filt.resample(param["testresampfreq"])

#         # drop bad trials (metadata stays aligned)
#         goodtrials = np.where(epo_filt.metadata["badtrial"] == 0)[0]
#         mod2 = mod2.iloc[goodtrials].reset_index(drop=True)
#         epo_filt = epo_filt[goodtrials]

#         if len(epo_filt) < 5:
#             print(f"Skipping {pa}: too few trials")
#             skipped_subjects.append(pa)
#             continue

#         rt_col = "rt"
#         vals_pain = mod2["painlevel"].to_numpy(dtype=float)
#         vals_money = mod2["moneylevel"].to_numpy(dtype=float)
#         vals_rt = mod2[rt_col].to_numpy(dtype=float)

#         keep = np.where(np.isfinite(vals_pain) & np.isfinite(vals_money) & np.isfinite(vals_rt))[0]
#         if len(keep) < 5:
#             print(f"Skipping {pa}: too few valid trials")
#             skipped_subjects.append(pa)
#             continue

#         mod2k = mod2.iloc[keep].reset_index(drop=True)
#         epo_keep = epo_filt.copy()[keep]

#         scale = Scaler(scalings="mean")
#         epo_z = mne.EpochsArray(scale.fit_transform(epo_keep.get_data()), epo_keep.info)

#         df_reg = mod2k.copy()
#         df_reg["Intercept"] = 1.0
#         df_reg["pain_z"] = stats.zscore(df_reg["painlevel"].to_numpy(dtype=float))
#         df_reg["money_z"] = stats.zscore(df_reg["moneylevel"].to_numpy(dtype=float))
#         df_reg["RT_z"] = stats.zscore(df_reg[rt_col].to_numpy(dtype=float))

#         names = ["Intercept", "pain_z", "money_z", "RT_z"]
#         design = df_reg[names]

#         res = mne.stats.linear_regression(epo_z, design, names=names)
#         beta_pain = res["pain_z"].beta
#         beta_money = res["money_z"].beta

#         betas[0].append(beta_pain)
#         betas[1].append(beta_money)
#         allbetasnp.append(np.stack([beta_pain.data, beta_money.data]))

#         all_epos[0].append(epo_keep)
#         all_epos[1].append(epo_keep)

#         included_subjects.append(pa)

#     if len(allbetasnp) == 0:
#         raise RuntimeError("No subjects included (decision).")

#     allbetas = np.stack(allbetasnp)
#     beta_gavg = [mne.grand_average(betas[i]) for i in range(len(regvars))]
#     connect, _ = mne.channels.find_ch_adjacency(beta_gavg[0].info, ch_type="eeg")

#     if not isinstance(param["cluster_threshold"], dict):
#         p_thresh = param["cluster_threshold"] / 2
#         n_samples = allbetas.shape[0]
#         cluster_threshold = -stats.t.ppf(p_thresh, n_samples - 1)
#     else:
#         cluster_threshold = param["cluster_threshold"]

#     tvals_list, pvals_list = [], []
#     for idx, regvar in enumerate(regvars):
#         data_reg = allbetas[:, idx, :, :]
#         testdata = np.swapaxes(data_reg, 2, 1)

#         tval, clusters, cluster_p_values, _ = st_clust_1s_ttest(
#             testdata,
#             n_jobs=param["njobs"],
#             threshold=cluster_threshold,
#             adjacency=connect,
#             n_permutations=param["nperms"],
#             buffer_size=None
#         )

#         pmap = np.ones_like(tval)
#         for c, p_val in zip(clusters, cluster_p_values):
#             pmap[c] = p_val

#         np.save(z_dir / f"ols_2ndlevel_tval_{regvar}.npy", tval)
#         np.save(z_dir / f"ols_2ndlevel_pval_{regvar}.npy", pmap)

#         tvals_list.append(tval)
#         pvals_list.append(pmap)




#     cluster_results = []
#     for idx, regvar in enumerate(regvars):
#         data_reg = allbetas[:, idx, :, :]
#         testdata = np.swapaxes(data_reg, 2, 1)

#         tval, clusters, cluster_p_values, _ = st_clust_1s_ttest(
#             testdata,
#             n_jobs=param["njobs"],
#             threshold=cluster_threshold,
#             adjacency=connect,
#             n_permutations=param["nperms"],
#             buffer_size=None
#         )

#         cluster_results.append({
#             "name": regvar,
#             "tval": tval,
#             "clusters": clusters,
#             "cluster_p_values": np.asarray(cluster_p_values, float),
#         })

#     tvals = np.stack(tvals_list)
#     pvals = np.stack(pvals_list)

#     # FDR across regressors (min cluster p per regressor)
#     min_cluster_ps = []
#     for pmap in pvals_list:
#         mask = pmap < 1.0
#         min_cluster_ps.append(pmap[mask].min() if np.any(mask) else 1.0)
#     min_cluster_ps = np.asarray(min_cluster_ps)

#     rej_fdr, p_fdr = fdr_correction(min_cluster_ps, alpha=0.05, method="indep")
#     pd.DataFrame({
#         "regressor": regvars,
#         "min_cluster_p": min_cluster_ps,
#         "min_cluster_p_FDR": p_fdr,
#         "sig_FDR": rej_fdr
#     }).to_csv(z_dir / "cluster_FDR_across_regressors.csv", index=False)

#     np.save(z_dir / "ols_2ndlevel_tvals.npy", tvals)
#     np.save(z_dir / "ols_2ndlevel_pvals.npy", pvals)
#     np.save(z_dir / "ols_2ndlevel_betas.npy", allbetas)
#     np.save(z_dir / "included_subjects.npy", np.array(included_subjects, dtype=object))
#     np.save(z_dir / "ols_2ndlevel_betasavg.npy", np.array(beta_gavg, dtype=object))

# # ============================================================
# # v1 PASSIVE (FIXED): merge like your decoding pipeline
# # ============================================================
# elif version == 1:
#     regvars = ["painlevel", "moneylevel"]
#     z_dir = Path(outpath) / "Zscoring"
#     ensure_dir(z_dir)

#     all_epos = [[] for _ in range(len(regvars))]
#     allbetasnp = []
#     betas = [[] for _ in range(len(regvars))]
#     included_subjects = []
#     skipped_subjects = []

#     def find_passive_epochs(pa: str) -> Path:
#         cand1 = basepath / "derivatives" / pa / "eeg" / "erps_passive" / f"{pa}_passive_cues_singletrials-epo.fif"
#         cand2 = basepath / pa / "eeg" / "erps_passive" / f"{pa}_passive_cues_singletrials-epo.fif"
#         return cand1 if cand1.exists() else cand2

#     def find_passive_beh(pa: str) -> Path:
#         cand1 = basepath / pa / "eeg" / f"{pa}_task-passive_beh.tsv"
#         cand2 = basepath / "derivatives" / pa / "eeg" / f"{pa}_task-passive_beh.tsv"
#         return cand1 if cand1.exists() else cand2

#     def load_passive_beh_with_trialsnum(beh_path: Path) -> pd.DataFrame:
#         """
#         MATCHES YOUR WORKING DECODING SCRIPT:
#           - drop rows with missing fixcross.started (if present)
#           - trialsnum = 1..N (event order)
#           - normalize condition/level
#         """
#         beh = pd.read_csv(beh_path, sep="\t")

#         if "fixcross.started" in beh.columns:
#             beh = beh[~beh["fixcross.started"].isna()].copy()

#         beh = beh.reset_index(drop=True)

#         # Passive trialsnum is just order 1..N (not blocks math)
#         beh["trialsnum"] = np.arange(1, len(beh) + 1)

#         # normalize
#         if "condition" not in beh.columns or "level" not in beh.columns:
#             raise ValueError(f"Passive beh file missing 'condition'/'level'. Columns: {list(beh.columns)}")

#         beh["condition"] = beh["condition"].astype(str).str.lower().str.strip()
#         beh["level"] = pd.to_numeric(beh["level"], errors="coerce")

#         beh = beh[beh["condition"].isin(["p", "m"]) & np.isfinite(beh["level"])].copy()
#         return beh

#     def merge_beh_into_epochs_on_trialsnum(epo: mne.Epochs, beh: pd.DataFrame, pa: str) -> mne.Epochs:
#         """
#         Merge BEFORE dropping bad trials.
#         This avoids MNE error: metadata rows must equal epochs/events rows.
#         """
#         if epo.metadata is None:
#             md = pd.DataFrame({"trialsnum": np.arange(1, len(epo) + 1)})
#         else:
#             md = epo.metadata.reset_index(drop=True).copy()

#         if "trialsnum" not in md.columns:
#             md["trialsnum"] = np.arange(1, len(epo) + 1)

#         md["trialsnum"] = pd.to_numeric(md["trialsnum"], errors="coerce").astype(int)
#         beh["trialsnum"] = pd.to_numeric(beh["trialsnum"], errors="coerce").astype(int)

#         merged = md.merge(
#             beh[["trialsnum", "condition", "level"]],
#             on="trialsnum",
#             how="left",
#             validate="1:1",
#         )

#         if merged["condition"].isna().any() or merged["level"].isna().any():
#             n_bad = int(merged["condition"].isna().sum())
#             raise ValueError(
#                 f"{pa}: trialsnum merge produced {n_bad} unlabeled epochs. "
#                 "This suggests epochs order and beh.tsv order differ for this subject."
#             )

#         epo.metadata = merged
#         return epo

#     # Only subjects who are in EEG ∩ HDDM set (keeping your decision-selection restriction)
#     part_passive = []
#     for p in common_participants:
#         if find_passive_epochs(p).exists() and find_passive_beh(p).exists():
#             part_passive.append(p)

#     print(f"\nPASSIVE: subjects with epochs+beh AND in EEG∩HDDM: {len(part_passive)}")

#     for pa in part_passive:
#         print(f"\n--- PASSIVE (v1): Processing {pa} ---")
#         try:
#             epo_path = find_passive_epochs(pa)
#             beh_path = find_passive_beh(pa)

#             epo = mne.read_epochs(str(epo_path), preload=True)

#             beh = load_passive_beh_with_trialsnum(beh_path)

#             # 1) MERGE FIRST (while lengths still match)
#             epo = merge_beh_into_epochs_on_trialsnum(epo, beh, pa)

#             # 2) resample if needed
#             if epo.info["sfreq"] != param["testresampfreq"]:
#                 epo = epo.resample(param["testresampfreq"])

#             # 3) now drop bad trials (metadata stays aligned)
#             if "badtrial" in epo.metadata.columns:
#                 good_mask = epo.metadata["badtrial"].fillna(0).astype(int).to_numpy() == 0
#                 epo = epo.copy()[good_mask]

#             if len(epo) < 20:
#                 print(f"Skipping {pa}: too few epochs after dropping bad trials (n={len(epo)})")
#                 skipped_subjects.append(pa)
#                 continue

#             # Build regressors from condition/level
#             cond = epo.metadata["condition"].astype(str).str.lower().str.strip().to_numpy()
#             level = epo.metadata["level"].to_numpy(dtype=float)

#             is_pain = cond == "p"
#             is_money = cond == "m"

#             if is_pain.sum() < 5 or is_money.sum() < 5:
#                 print(f"Skipping {pa}: too few trials per condition (p={is_pain.sum()}, m={is_money.sum()})")
#                 skipped_subjects.append(pa)
#                 continue

#             if np.nanstd(level[is_pain]) == 0 or np.nanstd(level[is_money]) == 0:
#                 print(f"Skipping {pa}: zero variance in level within condition")
#                 skipped_subjects.append(pa)
#                 continue

#             # Keep plotting compatibility: create painlevel/moneylevel columns
#             epo.metadata["painlevel"] = np.where(is_pain, level, 0.0)
#             epo.metadata["moneylevel"] = np.where(is_money, level, 0.0)
#             epo.metadata["participant_id"] = pa

#             # Z-score EEG across trials
#             scale = Scaler(scalings="mean")
#             epo_z = mne.EpochsArray(scale.fit_transform(epo.get_data()), epo.info)

#             # Design matrix: Intercept + cue_type_pm + pain_z_masked + money_z_masked
#             df_reg = epo.metadata.copy()
#             df_reg["Intercept"] = 1.0
#             df_reg["cue_type_pm"] = np.where(is_pain, 0.5, -0.5)

#             pain_z = np.zeros(len(df_reg), dtype=float)
#             money_z = np.zeros(len(df_reg), dtype=float)
#             pain_z[is_pain] = stats.zscore(level[is_pain])
#             money_z[is_money] = stats.zscore(level[is_money])

#             df_reg["pain_z_masked"] = pain_z
#             df_reg["money_z_masked"] = money_z

#             names = ["Intercept", "cue_type_pm", "pain_z_masked", "money_z_masked"]
#             design = df_reg[names]

#             if not np.all(np.isfinite(design.to_numpy())):
#                 print(f"Skipping {pa}: NaN/Inf in design")
#                 skipped_subjects.append(pa)
#                 continue

#             # Regression
#             res = mne.stats.linear_regression(epo_z, design, names=names)
#             beta_pain = res["pain_z_masked"].beta
#             beta_money = res["money_z_masked"].beta

#             betas[0].append(beta_pain)
#             betas[1].append(beta_money)
#             allbetasnp.append(np.stack([beta_pain.data, beta_money.data]))

#             # Save epochs for plotting
#             all_epos[0].append(epo)
#             all_epos[1].append(epo)

#             included_subjects.append(pa)
#             print(f"Included {pa} (n_epochs={len(epo)})")

#         except Exception as e:
#             print(f"Skipping {pa}: {e}")
#             skipped_subjects.append(pa)

#     # -----------------------------
#     # Group-level
#     # -----------------------------
#     if len(allbetasnp) == 0:
#         raise RuntimeError("No subjects included in PASSIVE analysis (after merge).")

#     allbetas = np.stack(allbetasnp)  # (n_subj, 2, n_ch, n_time)
#     beta_gavg = [mne.grand_average(betas[i]) for i in range(len(regvars))]

#     connect, _ = mne.channels.find_ch_adjacency(beta_gavg[0].info, ch_type="eeg")

#     if not isinstance(param["cluster_threshold"], dict):
#         p_thresh = param["cluster_threshold"] / 2
#         n_samples = allbetas.shape[0]
#         cluster_threshold = -stats.t.ppf(p_thresh, n_samples - 1)
#     else:
#         cluster_threshold = param["cluster_threshold"]

#     tvals_list, pvals_list = [], []
#     for idx, regvar in enumerate(regvars):
#         print(f"\nPASSIVE second-level cluster test for {regvar}")

#         data_reg = allbetas[:, idx, :, :]
#         testdata = np.swapaxes(data_reg, 2, 1)

#         tval, clusters, cluster_p_values, _ = st_clust_1s_ttest(
#             testdata,
#             n_jobs=param["njobs"],
#             threshold=cluster_threshold,
#             adjacency=connect,
#             n_permutations=param["nperms"],
#             buffer_size=None,
#         )

#         pmap = np.ones_like(tval)
#         for c, p_val in zip(clusters, cluster_p_values):
#             pmap[c] = p_val

#         np.save(z_dir / f"ols_2ndlevel_tval_{regvar}.npy", tval)
#         np.save(z_dir / f"ols_2ndlevel_pval_{regvar}.npy", pmap)

#         tvals_list.append(tval)
#         pvals_list.append(pmap)

#     tvals = np.stack(tvals_list)
#     pvals = np.stack(pvals_list)

#     # FDR across regressors
#     min_cluster_ps = []
#     for pmap in pvals_list:
#         mask = pmap < 1.0
#         min_cluster_ps.append(pmap[mask].min() if np.any(mask) else 1.0)
#     min_cluster_ps = np.asarray(min_cluster_ps)

#     rej_fdr, p_fdr = fdr_correction(min_cluster_ps, alpha=0.05, method="indep")
#     pd.DataFrame({
#         "regressor": regvars,
#         "min_cluster_p": min_cluster_ps,
#         "min_cluster_p_FDR": p_fdr,
#         "sig_FDR": rej_fdr
#     }).to_csv(z_dir / "cluster_FDR_across_regressors.csv", index=False)

#     # Save group-level files expected by plotting script
#     np.save(z_dir / "ols_2ndlevel_tvals.npy", tvals)
#     np.save(z_dir / "ols_2ndlevel_pvals.npy", pvals)
#     np.save(z_dir / "ols_2ndlevel_betas.npy", allbetas)
#     np.save(z_dir / "included_subjects.npy", np.array(included_subjects, dtype=object))
#     np.save(z_dir / "ols_2ndlevel_betasavg.npy", np.array(beta_gavg, dtype=object))

#     # Save epochs per regressor for plotting
#     for idx, regvar in enumerate(regvars):
#         if len(all_epos[idx]) == 0:
#             continue
#         epo_save = mne.concatenate_epochs(all_epos[idx])
#         epo_save.save(z_dir / f"ols_2ndlevel_allepochs-epo_{regvar}.fif", overwrite=True)

#     # pain - money beta-diff cluster
#     print("\nPASSIVE: pain - money beta difference cluster test ...")
#     beta_diff = allbetas[:, 0, :, :] - allbetas[:, 1, :, :]
#     testdata_diff = np.swapaxes(beta_diff, 2, 1)

#     tval_diff, clusters_diff, cluster_p_values_diff, _ = st_clust_1s_ttest(
#         testdata_diff,
#         n_jobs=param["njobs"],
#         threshold=cluster_threshold,
#         adjacency=connect,
#         n_permutations=param["nperms"],
#         buffer_size=None,
#     )

#     pvals_diff = np.ones_like(tval_diff)
#     for c, p_val in zip(clusters_diff, cluster_p_values_diff):
#         pvals_diff[c] = p_val

#     np.save(z_dir / "ols_2ndlevel_tval_diff_pain_minus_money.npy", tval_diff)
#     np.save(z_dir / "ols_2ndlevel_pval_diff_pain_minus_money.npy", pvals_diff)

#     print("\nPASSIVE finished.")
#     print("Saved outputs in:", z_dir)
#     print("Included subjects:", included_subjects)
#     print("Skipped subjects:", skipped_subjects)














# # Massunivariate Analysis and Second level test on betas
# #----------------------------------------------------------------------------
# # import libraries
# import mne
# from os.path import join as opj
# import pandas as pd
# import numpy as np
# import os
# from mne.decoding import Scaler
# import scipy
# from bids import BIDSLayout
# from mne.stats import spatio_temporal_cluster_1samp_test as st_clust_1s_ttest
# from scipy import stats
# import os
# import re
# import warnings
# warnings.simplefilter(action='ignore', category=FutureWarning)
# from statsmodels.distributions.empirical_distribution import ECDF
# from pathlib import Path
# from mne.stats import fdr_correction
# from mne.time_frequency import tfr_morlet
# from mne.stats import permutation_cluster_1samp_test, combine_adjacency

# # directory
# PROJECT_DIR = Path(os.getenv("PROJECT_DIR", "/workspace"))
# basepath = Path(os.getenv("DATA_DIR", PROJECT_DIR / "EEG" / "PainReward_sub-001-050" / "painrewardeegdata"))
# HDDM_DIR = Path(os.getenv("HDDM_DIR"))

# def ensure_dir(path):
#     Path(path).mkdir(parents=True, exist_ok=True)
# import re
# from pathlib import Path
# import os

# layout = BIDSLayout(basepath)
# # for cluster
# # disable Numba JIT caching & compilation
# #os.environ["NUMBA_DISABLE_JIT"] = "1"
# import numba
# numba.config.CACHE_ENABLE = False

# # Outpath for analysis
# outpath = Path(os.getenv("OUT_DIR", basepath / 'statistics'))       
# if not os.path.exists(outpath):
#     os.mkdir(outpath)


# # here for decision its just erps_massuni_drift_mod_9 and for passive it is: erps_massuni_drift_mod_9_2_passive
# version = 1
# v32_mode = "joint"   # "joint" or "separate"

# if version == 1:
#     outpath = opj(outpath, 'erps_massuni_passive_regression')
#     if not os.path.exists(outpath):
#         os.mkdir(outpath)
# elif version == 2: # with RT as covariate
#     outpath = opj(outpath, 'erps_massuni_regression')
#     if not os.path.exists(outpath):
#         os.mkdir(outpath)             
# else:
#     print("no version")


# # participants
# # participants
# part_csv = basepath / "participants.tsv"
# part = pd.read_csv(part_csv, sep="\t")["participant_id"].unique().tolist()
# part.sort()

# # Silence pandas warning
# pd.options.mode.chained_assignment = None  

# # Parameters # similar to MP's painlearning (2024)
# param = {
#     # Njobs for permutations
#     'njobs': 20,                   
#     # Number of permutations
#     'nperms': 5000,
#     # Random state to get same permutations each time
#     'random_state': 23,
#     'testresampfreq': 1024,
#     # clustering threshold
#     'cluster_threshold': 0.01}

# # this is the data frame I computed in the DDM_EEG_load.py file for the best fitting DDM by adding trial-by-trial drift-scaled pain as a column & other important parameters from the DDM
# # if you are testing the influence of decision threshold on neural measures mod_10 can be used

# mod_data_path = HDDM_DIR / "figures" / "painreward_behavioural_data_mod_9" / "diagnostics" / "v_pain_money.csv"
# mod_data = pd.read_csv(mod_data_path, sep=None, engine="python")
# mod_data["rt"] = mod_data["choice_resp.rt"]
# mod_data["interaction"] = mod_data["moneylevel"]*mod_data["painlevel"]
# mod_data["trialsnum"] = (
#     mod_data["blocks.thisRepN"].astype(int) * 25
#     + mod_data["trials.thisN"].astype(int)
#     + 1
# )
# # same file but for threshold (a) parameters
# mod_data_a_path = HDDM_DIR / "figures" / "painreward_behavioural_data_mod_10" / "diagnostics" / "a_pain_money.csv"
# mod_data_a = pd.read_csv(mod_data_a_path, sep=None, engine="python")


# # Subjects in EEG 
# eeg_participants = set(part)
# # Subjects in HDDM CSV (should be 38 in total)
# beh_participants = set(mod_data["participant"].unique())
# # Subjects present in both datasets
# common_participants = sorted(list(eeg_participants & beh_participants))

# print("\n Subjects:", common_participants)                                   # should be 38
# print(len(common_participants))

# part = common_participants
# part_1_dat = mod_data[mod_data["participant"].isin(part)]
# part_1 = part

# ####

# raw_regcols = ['painlevel', 'moneylevel']
# regvars = raw_regcols  

# #all_epos = [[] for i in range(len(regvars))]
# #allbetasnp = []
# #betas = [[] for i in range(len(regvars))]
# part.sort()

# #------------------------------------------------------------------------------------------------------------------------------------------------
# # Creating the dataframes

# #------------------------------------------------------------------------------------------------------------------------------------------------
# # Creating the dataframes (only needed for versions 1–4)

# if version in [1, 2]:
#     filtered_data = []
#     for p in part:
#         # data for this participant
#         df = mod_data[mod_data['participant'] == p]
        
#         # Load single epochs file
#         if version == 1:
#             epo = mne.read_epochs(opj(basepath,  p, 'eeg', 'erps_passive',                   
#                                   p + '_passive_cues_singletrials-epo.fif'))
#             epo_1 = epo.copy()

#         elif version in [2]:
#             epo = mne.read_epochs(
#                 opj(basepath, "derivatives", p, "eeg", "erps",
#                     f"{p}_decision_cues_singletrials-epo.fif"),
#                     preload=True)
#             epo_1 = epo.copy()

#         elif version in [3]:
#             epo = mne.read_epochs(
#                 opj(basepath, p, "eeg", "erps_resp_rp", f"{p}_decision_resp_rp_singletrials-epo.fif"),
#                 preload=True)
#             epo_1 = epo.copy()
#             # epo = mne.read_epochs(
#             #     opj(basepath, p, "eeg", "erps_resp", f"{p}_decision_resp_singletrials-epo.fif"),
#             #     preload=True)
#             # epo_1 = epo.copy()
#         elif version in [4]:
#             epo = mne.read_epochs(
#                 opj(basepath, p, "eeg", "erps_long", f"{p}_decision_cues_long_singletrials-epo.fif"),
#                 preload=True
#             )
#             epo_1 = epo.copy()

#         participants = epo_1.metadata['participant_id'].unique()
#         trialblocks = []
#         blocks_idx = []

#         # create blocks for metadata
#         for participant in participants:
#             p_df = epo_1.metadata[epo_1.metadata['participant_id'] == participant]
#             blocks = list(range(25)) * 5
#             blocks_idx_participant = [i for i in range(5) for _ in range(25)]
#             trialblocks.extend(blocks)
#             blocks_idx.extend(blocks_idx_participant)
                
#         epo_1.metadata['trialblocks'] = trialblocks
#         epo_1.metadata['blocks_idx'] = blocks_idx

#         epo_1_filtered = pd.DataFrame()

#         # filter for unique participants in the behavioral frame
#         for participant in df['participant'].unique():
#             erps_p_df = epo_1.metadata[epo_1.metadata['participant_id'] == participant]
#             df_unique = df[df['participant'] == participant]   

#             for block_x in df_unique['blocks.thisRepN'].unique():
#                 erps_block_df = erps_p_df[erps_p_df['blocks_idx'] == block_x]
#                 df_block_df = df_unique[df_unique['blocks.thisRepN'] == block_x]
                    
#                 filtered_block_df = erps_block_df[erps_block_df['trialblocks'].isin(df_block_df['trials.thisN'])]            
#                 epo_1_filtered = pd.concat([epo_1_filtered, filtered_block_df], ignore_index=True)
        
#         filtered_data.append(epo_1_filtered)

#     epo_1_filtered_combined = pd.concat(filtered_data, ignore_index=True)


#     merge_left = ['participant_id', 'blocks_idx', 'trialblocks']
#     merge_right = ['participant', 'blocks.thisRepN', 'trials.thisN']

#     trial_map = epo_1_filtered_combined.merge(
#         mod_data,
#         left_on=merge_left,
#         right_on=merge_right,
#         how='inner'
#     )
#     print("trial_map shape:", trial_map.shape)
#     print("trial_map columns:", trial_map.columns.tolist())

#     if "trialsnum_x" in trial_map.columns:
#         trial_map = trial_map.rename(columns={"trialsnum_x": "trialsnum"})
    
#     if "trialsnum_y" in trial_map.columns:
#         trial_map = trial_map.drop(columns=["trialsnum_y"])
    

# #------------------------------------------------------------------------------------------------------------------------------------------------
# # Massunivariate 

# if version == 2:

#     # storage
#     all_epos = [[] for _ in range(len(regvars))]   # keep per-regvar epoch saves
#     allbetasnp = []                                # per subject: (2, n_chan, n_time)
#     betas = [[] for _ in range(len(regvars))]      # per regvar: list of Evoked beta

#     included_subjects = []
#     skipped_subjects = []

#     z_dir = Path(outpath) / "Zscoring"
#     ensure_dir(z_dir)

#     for pa in part_1:
#         print(f"\n--- pain = money model (v{version}): Processing {pa} ---")

#         # Behavioural tables for this participant
#         df2 = epo_1_filtered_combined[epo_1_filtered_combined['participant_id'] == pa]
#         mod2 = part_1_dat[part_1_dat['participant'] == pa].copy()

#         # Load epochs
#         if version == 1:
#             epo = mne.read_epochs(
#                 opj(basepath, pa, 'eeg', 'erps_passive', pa + '_passive_cues_singletrials-epo.fif')
#             )
#         else:  # version 2 or 3
#             epo = mne.read_epochs(
#                 opj(basepath, "derivatives", pa, "eeg", "erps",
#                     f"{pa}_decision_cues_singletrials-epo.fif"),
#                     preload=True)
#             epo_1 = epo.copy()


#         epo_cop = epo.copy()

#         # Match trials using 'trialsnum'
#         matching = epo_cop.metadata['trialsnum'].isin(df2['trialsnum'])
#         epo_filt = epo_cop[matching]

#         # Downsample if needed
#         if epo_filt.info['sfreq'] != param['testresampfreq']:
#             epo_filt = epo_filt.resample(param['testresampfreq'])

#         # Drop bad trials
#         goodtrials = np.where(epo_filt.metadata['badtrial'] == 0)[0]
#         df2 = df2.iloc[goodtrials].reset_index(drop=True)
#         mod2 = mod2.iloc[goodtrials].reset_index(drop=True)
#         epo_filt = epo_filt[goodtrials]

#         if len(df2) < 5:
#             print(f"Skipping {pa} (only {len(df2)} trials after cleaning)")
#             skipped_subjects.append(pa)
#             continue

#         # RT column
#         if "rt" in mod2.columns:
#             rt_col = "rt"
#         else:
#             raise ValueError(f"No RT column in mod_data. Columns: {mod2.columns.tolist()}")

#         # ------------------------------------------------------------
#         # model trial mask 
        
#         vals_pain = mod2["painlevel"].to_numpy(dtype=float)
#         vals_money = mod2["moneylevel"].to_numpy(dtype=float)
#         vals_rt = mod2[rt_col].to_numpy(dtype=float)

#         keep = np.where(
#             np.isfinite(vals_pain) &
#             np.isfinite(vals_money) &
#             np.isfinite(vals_rt)
#         )[0]

#         if len(keep) < 5:
#             print(f"Skipping {pa}: only {len(keep)} valid trials for joint model")
#             skipped_subjects.append(pa)
#             continue

#         mod2k = mod2.iloc[keep].reset_index(drop=True)
#         epo_keep = epo_filt.copy()[keep]

#         # Variance checks
#         if np.nanstd(mod2k["painlevel"]) == 0:
#             print(f"Skipping {pa}: painlevel has zero variance")
#             skipped_subjects.append(pa)
#             continue
#         if np.nanstd(mod2k["moneylevel"]) == 0:
#             print(f"Skipping {pa}: moneylevel has zero variance")
#             skipped_subjects.append(pa)
#             continue
#         if np.nanstd(mod2k[rt_col]) == 0:
#             print(f"Skipping {pa}: RT has zero variance")
#             skipped_subjects.append(pa)
#             continue

#         # ------------------------------------------------------------
#         # Z-score EEG across trials
        
#         scale = Scaler(scalings='mean')
#         epo_z = mne.EpochsArray(scale.fit_transform(epo_keep.get_data()),
#                                 epo_keep.info)

#         # ------------------------------------------------------------
#         # design matrix 
#         # EEG ~ 1 + pain_z + money_z + RT_z

#         df_reg = mod2k.copy()
#         df_reg["Intercept"] = 1.0
#         df_reg["pain_z"] = stats.zscore(df_reg["painlevel"].to_numpy(dtype=float))
#         df_reg["money_z"] = stats.zscore(df_reg["moneylevel"].to_numpy(dtype=float))
#         df_reg["RT_z"] = stats.zscore(df_reg[rt_col].to_numpy(dtype=float))

#         design = df_reg[["Intercept", "pain_z", "money_z", "RT_z"]]
#         names = ["Intercept", "pain_z", "money_z", "RT_z"]

#         if not np.all(np.isfinite(design.to_numpy())):
#             print(f"Skipping {pa}: design matrix has NaN/Inf")
#             skipped_subjects.append(pa)
#             continue

#         # Update metadata for plotting
#         df_meta = epo_keep.metadata.reset_index(drop=True).copy()
#         df_meta["painlevel"] = df_reg["painlevel"].values
#         df_meta["moneylevel"] = df_reg["moneylevel"].values
#         df_meta[rt_col] = df_reg[rt_col].values
#         epo_keep.metadata = df_meta

#         # Store epochs per regressor
#         # Same epochs go into both lists
#         all_epos[0].append(epo_keep)
#         all_epos[1].append(epo_keep)

#         # ------------------------------------------------------------
#         # Run regression 

#         res = mne.stats.linear_regression(epo_z, design, names=names)

#         beta_pain = res["pain_z"].beta   # evoked
#         beta_money = res["money_z"].beta # evoked

#         # regvars order
#         betas[0].append(beta_pain)
#         betas[1].append(beta_money)

#         allbetasnp.append(np.stack([beta_pain.data, beta_money.data]))

#         included_subjects.append(pa)
#         print(f"Included {pa}")

#     # ---------------------------------------------------------------------
#     # stack betas across subjects (n_subj, 2, n_chan, n_time)

#     if len(allbetasnp) == 0:
#         raise RuntimeError("No subjects included")

#     allbetas = np.stack(allbetasnp)
#     print(f"\nTotal subjects considered: {len(part_1)}")
#     print(f"Included ({len(included_subjects)}): {included_subjects}")
#     print(f"Skipped  ({len(skipped_subjects)}): {skipped_subjects}")

#     # grand average maps
#     beta_gavg = []
#     for idx, regvar in enumerate(regvars):
#         beta_gavg.append(mne.grand_average(betas[idx]))

#     # connectivity for cluster test
#     connect, names_ch = mne.channels.find_ch_adjacency(epo_keep.info, ch_type='eeg')

#     # cluster threshold
#     if not isinstance(param['cluster_threshold'], dict):
#         p_thresh = param['cluster_threshold'] / 2
#         n_samples = allbetas.shape[0]
#         cluster_threshold = -stats.t.ppf(p_thresh, n_samples - 1)
#     else:
#         cluster_threshold = param['cluster_threshold']

#     # ---------------------------------------------------------------------
#     # second-level cluster tests with pain and money betas 

#     tvals, pvalues = [], []

#     for idx, regvar in enumerate(regvars):
#         print(f"\nSecond-level cluster test for regressor (JOINT beta): {regvar}")

#         data_reg = allbetas[:, idx, :, :]     # n_subj, n_time, n_chan
#         testdata = np.swapaxes(data_reg, 2, 1) # n_subj, n_time, n_chan

#         tval, clusters, cluster_p_values, _ = st_clust_1s_ttest(
#             testdata,
#             n_jobs=param["njobs"],
#             threshold=cluster_threshold,
#             adjacency=connect,
#             n_permutations=param['nperms'],
#             buffer_size=None
#         )

#         pmap = np.ones_like(tval)
#         for c, p_val in zip(clusters, cluster_p_values):
#             pmap[c] = p_val

#         tvals.append(tval)
#         pvalues.append(pmap)

#         np.save(z_dir / f'ols_2ndlevel_tval_{regvar}.npy', tval)
#         np.save(z_dir / f'ols_2ndlevel_pval_{regvar}.npy', pmap)

#     # save group-level results
#     tvals = np.stack(tvals)  # (2, n_times, n_ch)
#     pvals = np.stack(pvalues)

#     # FDR across regressors using min cluster p per regressor
#     min_cluster_ps = []
#     for pmap in pvalues:
#         mask = pmap < 1.0
#         min_cluster_ps.append(pmap[mask].min() if np.any(mask) else 1.0)

#     min_cluster_ps = np.asarray(min_cluster_ps)
#     rej_fdr, p_fdr = fdr_correction(min_cluster_ps, alpha=0.05, method='indep')

#     fdr_df = pd.DataFrame({
#         "regressor": regvars,
#         "min_cluster_p": min_cluster_ps,
#         "min_cluster_p_FDR": p_fdr,
#         "sig_FDR": rej_fdr
#     })
#     fdr_df.to_csv(z_dir / "cluster_FDR_across_regressors.csv", index=False)
#     print(f"FDR summary across regressors in {z_dir}")

#     np.save(z_dir / 'ols_2ndlevel_tvals.npy', tvals)
#     np.save(z_dir / 'ols_2ndlevel_pvals.npy', pvals)
#     np.save(z_dir / 'ols_2ndlevel_betas.npy', allbetas)
#     np.save(z_dir / 'included_subjects.npy', np.array(included_subjects, dtype=object))
#     np.save(z_dir / 'ols_2ndlevel_betasavg.npy', beta_gavg)

#     # Save epochs per regressor 
#     for idx, regvar in enumerate(regvars):
#         if len(all_epos[idx]) == 0:
#             continue
#         epo_save = mne.concatenate_epochs(all_epos[idx])
#         epo_save.save(z_dir / f'ols_2ndlevel_allepochs-epo_{regvar}.fif', overwrite=True)

#     # ---------------------------------------------------------------------
#     # Beta-difference cluster test for pain - money

#     print("\nComputing pain - money beta-difference cluster test ...")

#     pain_idx = regvars.index("painlevel")
#     money_idx = regvars.index("moneylevel")

#     data_pain = allbetas[:, pain_idx, :, :]      # n_subj, n_chan, n_time
#     data_money = allbetas[:, money_idx, :, :]    # n_subj, n_chan, n_time

#     beta_diff = data_pain - data_money           # β_pain - β_money
#     testdata_diff = np.swapaxes(beta_diff, 2, 1) # n_subj, n_time, n_chan

#     tval_diff, clusters_diff, cluster_p_values_diff, _ = st_clust_1s_ttest(
#         testdata_diff,
#         n_jobs=param["njobs"],
#         threshold=cluster_threshold,
#         adjacency=connect,
#         n_permutations=param['nperms'],
#         buffer_size=None
#     )

#     pvals_diff = np.ones_like(tval_diff)
#     for c, p_val in zip(clusters_diff, cluster_p_values_diff):
#         pvals_diff[c] = p_val

#     np.save(z_dir / 'ols_2ndlevel_tval_diff_pain_minus_money.npy', tval_diff)
#     np.save(z_dir / 'ols_2ndlevel_pval_diff_pain_minus_money.npy', pvals_diff)

#     print("saved pain-money beta-difference maps in", z_dir)





# # =========================
# # PASSIVE MASSUNIVARIATE v1
# # condition/level in epochs metadata
# # =========================

# if version == 1:

#     # paths
#     z_dir = Path(outpath) / "Zscoring"
#     ensure_dir(z_dir)

#     # regressors we will SAVE (for plotting compatibility)
#     regvars = ["painlevel", "moneylevel"]

#     all_epos = [[] for _ in range(len(regvars))]
#     allbetasnp = []
#     betas = [[] for _ in range(len(regvars))]

#     included_subjects = []
#     skipped_subjects = []

#     # participants: for passive, just use those who have the epochs file
#     part_passive = []
#     for p in part:
#         epo_path = basepath / "derivatives" / p / "eeg" / "erps_passive" / f"{p}_passive_cues_singletrials-epo.fif"
#         if epo_path.exists():
#             part_passive.append(p)

#     print(f"Found passive epochs for {len(part_passive)} participants")

#     for pa in part_passive:
#         print(f"\n--- PASSIVE (v{version}): Processing {pa} ---")

#         epo_path = basepath / "derivatives" / pa / "eeg" / "erps_passive" / f"{pa}_passive_cues_singletrials-epo.fif"
#         epo = mne.read_epochs(str(epo_path), preload=True)

#         if epo.metadata is None:
#             print(f"Skipping {pa}: no metadata in epochs")
#             skipped_subjects.append(pa)
#             continue

#         meta = epo.metadata.reset_index(drop=True).copy()

#         # REQUIRE these columns
#         needed = {"condition", "level"}
#         if not needed.issubset(set(meta.columns)):
#             print(f"Skipping {pa}: missing columns {needed - set(meta.columns)} in metadata")
#             skipped_subjects.append(pa)
#             continue

#         # Downsample if needed
#         if epo.info["sfreq"] != param["testresampfreq"]:
#             epo = epo.resample(param["testresampfreq"])
#             meta = epo.metadata.reset_index(drop=True).copy()

#         # Drop bad trials if present
#         if "badtrial" in meta.columns:
#             good_idx = np.where(meta["badtrial"].to_numpy().astype(int) == 0)[0]
#             epo = epo[good_idx]
#             meta = epo.metadata.reset_index(drop=True).copy()

#         if len(epo) < 10:
#             print(f"Skipping {pa}: too few trials after cleaning ({len(epo)})")
#             skipped_subjects.append(pa)
#             continue

#         # -----------------------------
#         # Build passive regressors
#         # condition: 'p' or 'm'
#         # level: 20..100  (we can keep it as is; z-scoring removes scaling)
#         # -----------------------------
#         cond = meta["condition"].astype(str).str.lower().str.strip().to_numpy()
#         level = meta["level"].to_numpy(dtype=float)

#         # keep valid
#         keep = np.where(np.isfinite(level) & np.isin(cond, ["p", "m"]))[0]
#         if len(keep) < 10:
#             print(f"Skipping {pa}: not enough valid (condition, level) trials ({len(keep)})")
#             skipped_subjects.append(pa)
#             continue

#         epo = epo.copy()[keep]
#         meta = epo.metadata.reset_index(drop=True).copy()
#         cond = meta["condition"].astype(str).str.lower().str.strip().to_numpy()
#         level = meta["level"].to_numpy(dtype=float)

#         is_pain = (cond == "p")
#         is_money = (cond == "m")

#         if is_pain.sum() < 5 or is_money.sum() < 5:
#             print(f"Skipping {pa}: too few pain or money trials (p={is_pain.sum()}, m={is_money.sum()})")
#             skipped_subjects.append(pa)
#             continue

#         if np.nanstd(level[is_pain]) == 0 or np.nanstd(level[is_money]) == 0:
#             print(f"Skipping {pa}: zero variance in level within a condition")
#             skipped_subjects.append(pa)
#             continue

#         # Create painlevel/moneylevel columns for plotting + binning later
#         meta["painlevel"] = np.where(is_pain, level, 0.0)
#         meta["moneylevel"] = np.where(is_money, level, 0.0)
#         meta["participant_id"] = pa
#         epo.metadata = meta

#         # -----------------------------
#         # Z-score EEG across trials
#         # -----------------------------
#         scale = Scaler(scalings="mean")
#         epo_z = mne.EpochsArray(scale.fit_transform(epo.get_data()), epo.info)

#         # -----------------------------
#         # Design matrix:
#         # EEG ~ Intercept + cue_type_pm + pain_z_masked + money_z_masked
#         #
#         # cue_type_pm: +0.5 for pain, -0.5 for money (nuisance)
#         # pain_z_masked: z(level) within pain trials else 0
#         # money_z_masked: z(level) within money trials else 0
#         # -----------------------------
#         df_reg = meta.copy()
#         df_reg["Intercept"] = 1.0
#         df_reg["cue_type_pm"] = np.where(is_pain, 0.5, -0.5)

#         pain_z = np.zeros(len(df_reg), dtype=float)
#         money_z = np.zeros(len(df_reg), dtype=float)
#         pain_z[is_pain] = stats.zscore(level[is_pain])
#         money_z[is_money] = stats.zscore(level[is_money])
#         df_reg["pain_z_masked"] = pain_z
#         df_reg["money_z_masked"] = money_z

#         names = ["Intercept", "cue_type_pm", "pain_z_masked", "money_z_masked"]
#         design = df_reg[names]

#         if not np.all(np.isfinite(design.to_numpy())):
#             print(f"Skipping {pa}: NaN/Inf in design")
#             skipped_subjects.append(pa)
#             continue

#         # Store epochs for plotting (same epochs saved under both regvars)
#         all_epos[0].append(epo)
#         all_epos[1].append(epo)

#         # Run regression
#         res = mne.stats.linear_regression(epo_z, design, names=names)

#         beta_pain = res["pain_z_masked"].beta
#         beta_money = res["money_z_masked"].beta

#         betas[0].append(beta_pain)
#         betas[1].append(beta_money)
#         allbetasnp.append(np.stack([beta_pain.data, beta_money.data]))

#         included_subjects.append(pa)
#         print(f"Included {pa}")

#     # -----------------------------
#     # Group level
#     # -----------------------------
#     if len(allbetasnp) == 0:
#         raise RuntimeError("No subjects included in passive analysis.")

#     allbetas = np.stack(allbetasnp)  # (n_subj, 2, n_ch, n_time)

#     beta_gavg = [mne.grand_average(betas[i]) for i in range(len(regvars))]

#     connect, _ = mne.channels.find_ch_adjacency(beta_gavg[0].info, ch_type="eeg")

#     if not isinstance(param["cluster_threshold"], dict):
#         p_thresh = param["cluster_threshold"] / 2
#         n_samples = allbetas.shape[0]
#         cluster_threshold = -stats.t.ppf(p_thresh, n_samples - 1)
#     else:
#         cluster_threshold = param["cluster_threshold"]

#     tvals, pvalues = [], []

#     for idx, regvar in enumerate(regvars):
#         print(f"\nPASSIVE second-level cluster test for {regvar}")

#         data_reg = allbetas[:, idx, :, :]        # (n_subj, n_ch, n_time)
#         testdata = np.swapaxes(data_reg, 2, 1)   # (n_subj, n_time, n_ch)

#         tval, clusters, cluster_p_values, _ = st_clust_1s_ttest(
#             testdata,
#             n_jobs=param["njobs"],
#             threshold=cluster_threshold,
#             adjacency=connect,
#             n_permutations=param["nperms"],
#             buffer_size=None,
#         )

#         pmap = np.ones_like(tval)
#         for c, p_val in zip(clusters, cluster_p_values):
#             pmap[c] = p_val

#         tvals.append(tval)
#         pvalues.append(pmap)

#         np.save(z_dir / f"ols_2ndlevel_tval_{regvar}.npy", tval)
#         np.save(z_dir / f"ols_2ndlevel_pval_{regvar}.npy", pmap)

#     tvals = np.stack(tvals)  # (2, n_times, n_ch)
#     pvals = np.stack(pvalues)

#     # FDR across regressors (min cluster p)
#     min_cluster_ps = []
#     for pmap in pvalues:
#         mask = pmap < 1.0
#         min_cluster_ps.append(pmap[mask].min() if np.any(mask) else 1.0)
#     min_cluster_ps = np.asarray(min_cluster_ps)

#     rej_fdr, p_fdr = fdr_correction(min_cluster_ps, alpha=0.05, method="indep")
#     pd.DataFrame({
#         "regressor": regvars,
#         "min_cluster_p": min_cluster_ps,
#         "min_cluster_p_FDR": p_fdr,
#         "sig_FDR": rej_fdr
#     }).to_csv(z_dir / "cluster_FDR_across_regressors.csv", index=False)

#     np.save(z_dir / "ols_2ndlevel_tvals.npy", tvals)
#     np.save(z_dir / "ols_2ndlevel_pvals.npy", pvals)
#     np.save(z_dir / "ols_2ndlevel_betas.npy", allbetas)
#     np.save(z_dir / "included_subjects.npy", np.array(included_subjects, dtype=object))
#     np.save(z_dir / "ols_2ndlevel_betasavg.npy", np.array(beta_gavg, dtype=object))

#     # save epochs per regressor for plotting
#     for idx, regvar in enumerate(regvars):
#         if len(all_epos[idx]) == 0:
#             continue
#         epo_save = mne.concatenate_epochs(all_epos[idx])
#         epo_save.save(z_dir / f"ols_2ndlevel_allepochs-epo_{regvar}.fif", overwrite=True)

#     # beta-diff cluster: pain - money
#     print("\nPASSIVE: pain - money beta difference ...")
#     beta_diff = allbetas[:, 0, :, :] - allbetas[:, 1, :, :]
#     testdata_diff = np.swapaxes(beta_diff, 2, 1)

#     tval_diff, clusters_diff, cluster_p_values_diff, _ = st_clust_1s_ttest(
#         testdata_diff,
#         n_jobs=param["njobs"],
#         threshold=cluster_threshold,
#         adjacency=connect,
#         n_permutations=param["nperms"],
#         buffer_size=None,
#     )

#     pvals_diff = np.ones_like(tval_diff)
#     for c, p_val in zip(clusters_diff, cluster_p_values_diff):
#         pvals_diff[c] = p_val

#     np.save(z_dir / "ols_2ndlevel_tval_diff_pain_minus_money.npy", tval_diff)
#     np.save(z_dir / "ols_2ndlevel_pval_diff_pain_minus_money.npy", pvals_diff)

#     print("PASSIVE finished. Saved in:", z_dir)