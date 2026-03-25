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
