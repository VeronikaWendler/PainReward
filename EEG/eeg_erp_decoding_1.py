# -*- coding: utf-8 -*-
"""
Step 1 (Passive): Time-resolved decoding of money level (and pain as control)

- Loads passive epochs from derivatives
- Loads passive beh.tsv from raw painrewardeegdata
- Merges beh columns (condition, level, blocks.thisN, trials.thisN) into epochs.metadata
- Runs time-resolved decoding (AUC) for:
    money-only: low (20/40) vs high (80/100), drop 60
    pain-only : same (positive control)
- Group-level cluster permutation test over time on (AUC - 0.5)

Optional (recommended once):
- Decode stimulus type: money vs pain (sanity check that pipeline works)

Outputs:
  derivatives/statistics/mvpa_passive_step1/
"""

from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from mne.decoding import SlidingEstimator, cross_val_multiscore
from mne.stats import permutation_cluster_1samp_test
from scipy import stats


# -----------------------------
# Paths 
# -----------------------------
DATA_DIR = Path(os.getenv("DATA_DIR", "")).expanduser()
OUT_BASE = Path(os.getenv("OUT_DIR", "")).expanduser()

if not DATA_DIR:
    raise RuntimeError("DATA_DIR env var is not set. In SLURM you export DATA_DIR=/pr/...")

RAW_DIR = DATA_DIR
DERIV_DIR = RAW_DIR / "derivatives"

# output folder (use OUT_DIR if provided, else default under derivatives)
if OUT_BASE and OUT_BASE.exists():
    OUT_DIR = OUT_BASE / "statistics" / "mvpa_passive_step1"
else:
    OUT_DIR = DERIV_DIR / "statistics" / "mvpa_passive_step1"

OUT_DIR.mkdir(parents=True, exist_ok=True)
DEBUG_DIR = OUT_DIR / "debug"
DEBUG_DIR.mkdir(parents=True, exist_ok=True)

# epochs/beh locations relative to subject folder
EPO_DIR = Path("eeg") / "erps_passive"
EPO_SUFFIX = "_passive_cues_singletrials-epo.fif"
BEH_SUFFIX = "_task-passive_beh.tsv"

# -----------------------------
# Decoding params
# -----------------------------
RANDOM_STATE = 23
N_SPLITS = 5
N_PERM = 5000
ALPHA_CLUSTER = 0.05
CHANCE = 0.5

# downsample for speed
RESAMPLE_SFREQ = 256  # set None to keep original

# extra validation
RUN_SHUFFLE_CONTROL = True  # should return ~chance
RUN_STIMTYPE_SANITY = True  # set True once: decode money vs pain across all passive trials

# Passive beh columns
COL_COND = "condition"   # 'p' or 'm'
COL_LEVEL = "level"      # 20/40/60/80/100

KEY_BLOCK = "blocks.thisN"   # 0..4 
KEY_TRIAL = "trials.thisN"   # 0..39 within each block

COND_MONEY = "m"
COND_PAIN = "p"


def list_subjects(deriv_dir: Path) -> list[str]:
    return sorted([p.name for p in deriv_dir.iterdir()
                   if p.is_dir() and p.name.startswith("sub-")])


def load_passive_epochs(sub: str) -> mne.Epochs:
    epo_path = DERIV_DIR / sub / EPO_DIR / f"{sub}{EPO_SUFFIX}"
    if not epo_path.exists():
        raise FileNotFoundError(f"Missing epochs for {sub}: {epo_path}")
    return mne.read_epochs(epo_path, preload=True, verbose="ERROR")


def load_passive_beh(sub: str) -> pd.DataFrame:
    beh_path = RAW_DIR / sub / "eeg" / f"{sub}{BEH_SUFFIX}"
    if not beh_path.exists():
        raise FileNotFoundError(f"Missing beh.tsv for {sub}: {beh_path}")
    beh = pd.read_csv(beh_path, sep="\t")
    return beh.reset_index(drop=True)


def _coerce_int_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("Int64")


def merge_beh_into_epochs(epo: mne.Epochs, beh: pd.DataFrame, sub: str) -> mne.Epochs:
    """
    Ensure epo.metadata contains condition + level from beh.tsv.

    Strategy:
      1) If epochs metadata already has condition+level, keep them
      2) Else try merge on (blocks.thisN, trials.thisN) if present in BOTH.
      3) Else fallback: order-based assignment if lengths match exactly.

    We write debug CSVs if something fails.
    """
    if epo.metadata is None:
        raise ValueError(f"{sub}: epochs has no metadata at all; cannot merge beh.")

    md = epo.metadata.reset_index(drop=True).copy()

    if (COL_COND in md.columns) and (COL_LEVEL in md.columns):
        return epo

    for col in [COL_COND, COL_LEVEL]:
        if col not in beh.columns:
            beh.head(30).to_csv(DEBUG_DIR / f"{sub}_beh_head.csv", index=False)
            raise ValueError(f"{sub}: beh.tsv missing required column '{col}'. Found: {list(beh.columns)}")

    can_key_merge = (
        (KEY_BLOCK in md.columns) and (KEY_TRIAL in md.columns) and
        (KEY_BLOCK in beh.columns) and (KEY_TRIAL in beh.columns)
    )

    if can_key_merge:
        md2 = md.copy()
        beh2 = beh.copy()

        md2[KEY_BLOCK] = _coerce_int_series(md2[KEY_BLOCK])
        md2[KEY_TRIAL] = _coerce_int_series(md2[KEY_TRIAL])
        beh2[KEY_BLOCK] = _coerce_int_series(beh2[KEY_BLOCK])
        beh2[KEY_TRIAL] = _coerce_int_series(beh2[KEY_TRIAL])

        beh_small = beh2[[KEY_BLOCK, KEY_TRIAL, COL_COND, COL_LEVEL]].copy()

        merged = md2.merge(
            beh_small,
            on=[KEY_BLOCK, KEY_TRIAL],
            how="left",
            validate="1:1"
        )

        if merged[COL_COND].isna().any() or merged[COL_LEVEL].isna().any():
            md.head(30).to_csv(DEBUG_DIR / f"{sub}_epo_md_head.csv", index=False)
            beh.head(30).to_csv(DEBUG_DIR / f"{sub}_beh_head.csv", index=False)
            n_bad = int(merged[COL_COND].isna().sum())
            raise ValueError(
                f"{sub}: key-merge produced {n_bad} unlabeled epochs. "
                "Likely keys don't align between events-derived metadata and beh.tsv."
            )

        epo.metadata = merged
        return epo

    if len(md) == len(beh):
        merged = md.copy()
        merged[COL_COND] = beh[COL_COND].to_numpy()
        merged[COL_LEVEL] = beh[COL_LEVEL].to_numpy()
        epo.metadata = merged
        return epo

    md.head(50).to_csv(DEBUG_DIR / f"{sub}_epo_md_head.csv", index=False)
    beh.head(50).to_csv(DEBUG_DIR / f"{sub}_beh_head.csv", index=False)

    raise ValueError(
        f"{sub}: cannot merge beh into epochs.\n"
        f"- epochs n={len(md)}; beh n={len(beh)} (not equal, so order-based merge not possible)\n"
        f"- keys '{KEY_BLOCK}'/'{KEY_TRIAL}' not present in BOTH epochs.metadata and beh.tsv\n"
        "See debug CSVs in OUT_DIR/debug for columns present in each."
    )


def make_binary_labels(level: np.ndarray) -> np.ndarray:
    """
    level values expected: 20/40/60/80/100
    returns y: 0=low (20,40), 1=high (80,100)
    Assumes level 60 already removed.
    """
    level = np.asarray(level, dtype=float)
    y = np.full(len(level), -1, dtype=int)
    y[np.isin(level, [20, 40])] = 0
    y[np.isin(level, [80, 100])] = 1
    return y


def select_trials_money_or_pain(epo: mne.Epochs, which: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """
    which: "money" or "pain"
    Returns X, y, times, md_used
    """
    if epo.metadata is None:
        raise ValueError("Epochs has no metadata.")

    md = epo.metadata.reset_index(drop=True)

    for col in [COL_COND, COL_LEVEL]:
        if col not in md.columns:
            raise ValueError(f"Missing metadata column '{col}'. Have: {md.columns.tolist()}")

    cond = md[COL_COND].astype(str).str.lower().to_numpy()
    if which == "money":
        keep = (cond == COND_MONEY)
    elif which == "pain":
        keep = (cond == COND_PAIN)
    else:
        raise ValueError("which must be 'money' or 'pain'")

    epo_f = epo.copy()[keep]
    md_f = epo_f.metadata.reset_index(drop=True)

    # drop level 60
    levels = md_f[COL_LEVEL].to_numpy(dtype=float)
    keep2 = ~np.isin(levels, [60])
    epo_f = epo_f.copy()[keep2]
    md_f = epo_f.metadata.reset_index(drop=True)

    if len(epo_f) < 10:
        raise ValueError(f"Too few trials after filtering for {which}. n={len(epo_f)}")

    y = make_binary_labels(md_f[COL_LEVEL].to_numpy(dtype=float))
    if np.any(y < 0):
        raise ValueError(f"Unlabeled trials exist. Levels seen: {np.unique(md_f[COL_LEVEL])}")

    X = epo_f.get_data()  # (n_trials, n_chans, n_times)
    times = epo_f.times.copy()
    return X, y, times, md_f


def select_trials_stimtype(epo: mne.Epochs) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Sanity check: decode stimulus type across all passive trials.
    labels: 0 = pain (p), 1 = money (m)
    """
    if epo.metadata is None:
        raise ValueError("Epochs has no metadata.")

    md = epo.metadata.reset_index(drop=True)
    if COL_COND not in md.columns:
        raise ValueError(f"Missing metadata column '{COL_COND}'.")

    cond = md[COL_COND].astype(str).str.lower().to_numpy()
    keep = np.isin(cond, [COND_PAIN, COND_MONEY])
    epo_f = epo.copy()[keep]
    md_f = epo_f.metadata.reset_index(drop=True)

    cond_f = md_f[COL_COND].astype(str).str.lower().to_numpy()
    y = np.full(len(cond_f), -1, dtype=int)
    y[cond_f == COND_PAIN] = 0
    y[cond_f == COND_MONEY] = 1

    if np.any(y < 0):
        raise ValueError(f"Unexpected condition labels: {np.unique(cond_f)}")

    X = epo_f.get_data()
    times = epo_f.times.copy()
    return X, y, times, md_f


def subject_decode(X: np.ndarray, y: np.ndarray, shuffle: bool = False) -> np.ndarray:
    """
    Option 1: keep all trials; handle imbalance via class_weight='balanced'
    """
    rng = np.random.default_rng(RANDOM_STATE)
    y_use = rng.permutation(y) if shuffle else y

    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            solver="liblinear",
            max_iter=2000,
            random_state=RANDOM_STATE,
            class_weight="balanced",   
        )
    )

    time_decod = SlidingEstimator(clf, scoring="roc_auc")
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    scores = cross_val_multiscore(time_decod, X, y_use, cv=cv, n_jobs=1)
    return scores.mean(axis=0)  # (n_times,)


def group_cluster(scores_by_subj: np.ndarray, times: np.ndarray):
    """
    Cluster permutation test on (AUC - 0.5) across time.
    """
    X = scores_by_subj - CHANCE  # (n_subj, n_times)

    p_form = 0.01
    t_thresh = stats.t.ppf(1 - p_form / 2, df=X.shape[0] - 1)

    T_obs, clusters, cluster_pv, H0 = permutation_cluster_1samp_test(
        X,
        n_permutations=N_PERM,
        threshold=t_thresh,
        tail=0,
        out_type="mask",
        n_jobs=1,
        seed=RANDOM_STATE,
        buffer_size=None,
    )

    p_map = np.ones(len(times), dtype=float)
    for cl, p in zip(clusters, cluster_pv):
        p_map[cl] = np.minimum(p_map[cl], p)

    return dict(T_obs=T_obs, clusters=clusters, cluster_pv=cluster_pv, p_map=p_map, t_thresh=t_thresh)


def plot_group(scores_by_subj: np.ndarray, times: np.ndarray, p_map: np.ndarray, title: str, out_png: Path):
    mean = scores_by_subj.mean(axis=0)
    sem = scores_by_subj.std(axis=0, ddof=1) / np.sqrt(scores_by_subj.shape[0])

    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(times, mean, linewidth=2)
    ax.fill_between(times, mean - sem, mean + sem, alpha=0.25)

    ax.axhline(CHANCE, linestyle="--", linewidth=1)
    ax.axvline(0, linestyle="--", linewidth=1)

    sig = p_map < ALPHA_CLUSTER
    if np.any(sig):
        ax.fill_between(times, CHANCE - 0.02, CHANCE - 0.01, where=sig, alpha=0.9)

    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Decoding (AUC)")
    ax.set_ylim(0.35, 0.85)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def save_trial_counts(md_used: pd.DataFrame, sub: str, which: str):
    """
    is my labeling sane? artifact
    For stimtype sanity, we just count conditions.
    """
    if which == "stimtype":
        tab = (
            md_used
            .assign(condition=md_used[COL_COND].astype(str).str.lower())
            .groupby(["condition"])
            .size()
            .reset_index(name="n")
            .sort_values(["condition"])
        )
    else:
        tab = (
            md_used
            .assign(condition=md_used[COL_COND].astype(str).str.lower())
            .groupby(["condition", COL_LEVEL])
            .size()
            .reset_index(name="n")
            .sort_values(["condition", COL_LEVEL])
        )

    tab.to_csv(OUT_DIR / f"{sub}_passive_{which}_trial_counts.csv", index=False)


def run(which: str, shuffle: bool = False):
    """
    which:
      - "money"   : decode money level within money-only trials
      - "pain"    : decode pain level within pain-only trials
      - "stimtype": decode money vs pain across all passive trial - check
    """
    subs = list_subjects(DERIV_DIR)
    scores_all, included, skipped = [], [], []
    times_ref = None

    for sub in subs:
        try:
            epo = load_passive_epochs(sub)
            beh = load_passive_beh(sub)
            epo = merge_beh_into_epochs(epo, beh, sub=sub)

            if RESAMPLE_SFREQ is not None:
                epo = epo.copy().resample(RESAMPLE_SFREQ, npad="auto")

            if which in ("money", "pain"):
                X, y, times, md_used = select_trials_money_or_pain(epo, which=which)
            elif which == "stimtype":
                X, y, times, md_used = select_trials_stimtype(epo)
            else:
                raise ValueError("which must be 'money', 'pain', or 'stimtype'")

            save_trial_counts(md_used, sub=sub, which=which)

            if times_ref is None:
                times_ref = times
            else:
                if len(times) != len(times_ref) or np.max(np.abs(times - times_ref)) > 1e-9:
                    raise RuntimeError("Time axis mismatch across subjects.")

            scores = subject_decode(X, y, shuffle=shuffle)
            scores_all.append(scores)
            included.append(sub)

            print(f"Included {sub} ({which}{'_shuf' if shuffle else ''}): trials={len(y)}")

        except Exception as e:
            skipped.append((sub, str(e)))
            print(f"Skipped {sub} ({which}{'_shuf' if shuffle else ''}): {e}")

    if len(scores_all) < 8:
        raise RuntimeError(f"Too few subjects included for group stats ({which}): n={len(scores_all)}")

    scores_all = np.stack(scores_all, axis=0)
    times = times_ref

    stats_out = group_cluster(scores_all, times)

    tag = f"passive_{which}" + ("_shuffle" if shuffle else "")
    np.savez(
        OUT_DIR / f"{tag}_group_results.npz",
        scores_by_subj=scores_all,
        times=times,
        included=np.array(included, dtype=object),
        skipped=np.array(skipped, dtype=object),
        T_obs=stats_out["T_obs"],
        p_map=stats_out["p_map"],
        cluster_pv=stats_out["cluster_pv"],
        t_thresh=stats_out["t_thresh"],
        chance=CHANCE,
        resample_sfreq=RESAMPLE_SFREQ if RESAMPLE_SFREQ is not None else -1,
    )

    pd.DataFrame(scores_all, index=included, columns=np.round(times, 6)).to_csv(
        OUT_DIR / f"{tag}_scores_by_subject.csv"
    )

    plot_group(
        scores_all,
        times,
        stats_out["p_map"],
        title=f"Passive {which} ({'SHUFFLED' if shuffle else 'REAL'}): AUC",
        out_png=OUT_DIR / f"{tag}_group_plot.png",
    )

    min_p = float(np.min(stats_out["cluster_pv"])) if len(stats_out["cluster_pv"]) else 1.0
    print(f"\nDONE {which}{'_shuffle' if shuffle else ''}: included n={len(included)}, min cluster p={min_p:.6f}")


def main():
    mne.set_log_level("WARNING")

    # Step 1 main analyses
    run("money", shuffle=False)
    run("pain", shuffle=False)

    # sanity check money vs pain
    if RUN_STIMTYPE_SANITY:
        run("stimtype", shuffle=False)

    # negative control should be ~chance
    if RUN_SHUFFLE_CONTROL:
        run("money", shuffle=True)
        run("pain", shuffle=True)
        if RUN_STIMTYPE_SANITY:
            run("stimtype", shuffle=True)


if __name__ == "__main__":
    main()
