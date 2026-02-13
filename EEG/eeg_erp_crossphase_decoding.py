# -*- coding: utf-8 -*-
"""
Step 3: Cross-phase time×time generalization (train time x test time heatmaps)

This script supports BOTH:
A) Classification time×time (accuracy; 5-class; chance=0.2; tail=1)
B) Regression time×time (Ridge; Pearson r; chance=0; tail=0)

Decision-phase labels:
- moneystim = 'm1'..'m5'  -> 20/40/60/80/100
- painstim  = 'p1'..'p5'  -> 20/40/60/80/100

Controls (REGRESSION block):
1) no control
2) pain balanced across money (subsampling test set)
3) residualize pain^2 from test EEG features (optionally pain+pain^2)

Outputs per analysis:
- NPZ with subj_mats (n_subj, n_train_t, n_test_t), p_map, cluster pvals, etc.
- CSV cluster table + summary JSON
- Heatmap PNG with sig overlay

"""

from __future__ import annotations
import os
from pathlib import Path
import json
import re
import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GroupKFold, StratifiedKFold, KFold

from mne.decoding import GeneralizingEstimator
from mne.stats import permutation_cluster_1samp_test, combine_adjacency
from scipy import stats
from tqdm.auto import tqdm


# =============================================================================
# Paths
# =============================================================================
DATA_DIR_STR = os.getenv("DATA_DIR", "").strip()
OUT_DIR_STR = os.getenv("OUT_DIR", "").strip()
if DATA_DIR_STR == "":
    raise RuntimeError("DATA_DIR env var not set")

RAW_DIR = Path(DATA_DIR_STR).expanduser()
DERIV_DIR = RAW_DIR / "derivatives"

if OUT_DIR_STR != "":
    OUT_BASE = Path(OUT_DIR_STR).expanduser()
    OUT_DIR = OUT_BASE / "statistics" / "mvpa_crossphase_step3_timegen"
else:
    OUT_DIR = DERIV_DIR / "statistics" / "mvpa_crossphase_step3_timegen"

OUT_DIR.mkdir(parents=True, exist_ok=True)
DEBUG_DIR = OUT_DIR / "debug"
DEBUG_DIR.mkdir(parents=True, exist_ok=True)

# ---- PASSIVE ----
PASS_EPO_DIR = Path("eeg") / "erps_passive"
PASS_EPO_SUFFIX = "_passive_cues_singletrials-epo.fif"
PASS_BEH_SUFFIX = "_task-passive_beh.tsv"

# ---- DECISION ----
DEC_EPO_DIR = Path("eeg") / "erps"
DEC_EPO_SUFFIX = "_decision_cues_singletrials-epo.fif"
DEC_BEH_SUFFIX = "_task-decision_beh.tsv"


# =============================================================================
# Parameters
# =============================================================================
RANDOM_STATE = 23
N_SPLITS = 5
N_PERM = 5000
ALPHA_CLUSTER = 0.05

RESAMPLE_SFREQ = 256
TMIN_STAT = 0.0
TMAX_STAT = 0.8

# Metadata keys
KEY_BLOCK = "blocks.thisN"
KEY_TRIAL = "trials.thisN"
KEY_TRIALNUM = "trialsnum"

# Passive columns
COL_COND = "condition"      # 'm' or 'p'
COL_LEVEL = "level"         # 20/40/60/80/100
COND_MONEY = "m"
COND_PAIN = "p"

# Decision columns (your actual design)
DEC_MONEY_COL_CANDIDATES = ["moneystim"]
DEC_PAIN_COL_CANDIDATES  = ["painstim"]

LEVEL_CODE_TO_LEVEL = {1: 20, 2: 40, 3: 60, 4: 80, 5: 100}
LEVELS_ALL = np.array([20, 40, 60, 80, 100], dtype=int)

# classification setup
CHANCE_5CLASS = 1.0 / 5.0

# regression setup
CHANCE_R = 0.0


# =============================================================================
# Utilities
# =============================================================================
def list_subjects(deriv_dir: Path) -> list[str]:
    return sorted([p.name for p in deriv_dir.iterdir()
                   if p.is_dir() and p.name.startswith("sub-")])


def _coerce_int_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("Int64")


def load_epochs(sub: str, phase: str) -> mne.Epochs:
    if phase == "passive":
        epo_path = DERIV_DIR / sub / PASS_EPO_DIR / f"{sub}{PASS_EPO_SUFFIX}"
    elif phase == "decision":
        epo_path = DERIV_DIR / sub / DEC_EPO_DIR / f"{sub}{DEC_EPO_SUFFIX}"
    else:
        raise ValueError("phase must be 'passive' or 'decision'")
    if not epo_path.exists():
        raise FileNotFoundError(f"Missing epochs for {sub} ({phase}): {epo_path}")
    return mne.read_epochs(epo_path, preload=True, verbose="ERROR")


def load_beh(sub: str, phase: str) -> pd.DataFrame:
    if phase == "passive":
        beh_path = RAW_DIR / sub / "eeg" / f"{sub}{PASS_BEH_SUFFIX}"
    elif phase == "decision":
        beh_path = RAW_DIR / sub / "eeg" / f"{sub}{DEC_BEH_SUFFIX}"
    else:
        raise ValueError("phase must be 'passive' or 'decision'")
    if not beh_path.exists():
        raise FileNotFoundError(f"Missing beh.tsv for {sub} ({phase}): {beh_path}")

    beh = pd.read_csv(beh_path, sep="\t")
    if "fixcross.started" in beh.columns:
        beh = beh[~beh["fixcross.started"].isna()].copy()

    beh = beh.reset_index(drop=True)
    #beh[KEY_TRIALNUM] = np.arange(1, len(beh) + 1)
    return beh


def merge_beh_into_epochs(epo: mne.Epochs, beh: pd.DataFrame, sub: str, phase: str) -> mne.Epochs:
    if epo.metadata is None:
        raise ValueError(f"{sub} {phase}: epochs has no metadata; cannot merge beh.")
    md = epo.metadata.reset_index(drop=True).copy()

    # trialsnum merge
    if (KEY_TRIALNUM in md.columns) and (KEY_TRIALNUM in beh.columns):
        merged = md.merge(beh, on=KEY_TRIALNUM, how="left", validate="1:1")
        epo.metadata = merged
        return epo

    # key merge
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
        merged = md2.merge(beh2, on=[KEY_BLOCK, KEY_TRIAL], how="left", validate="1:1")
        epo.metadata = merged
        return epo

    # order merge
    if len(md) == len(beh):
        merged = md.copy()
        for c in beh.columns:
            if c not in merged.columns:
                merged[c] = beh[c].to_numpy()
        epo.metadata = merged
        return epo

    md.head(30).to_csv(DEBUG_DIR / f"{sub}_{phase}_epo_md_head.csv", index=False)
    beh.head(30).to_csv(DEBUG_DIR / f"{sub}_{phase}_beh_head.csv", index=False)
    raise ValueError(f"{sub} {phase}: cannot merge beh into epochs; see debug heads.")


def drop_badtrials(epo: mne.Epochs) -> mne.Epochs:
    if epo.metadata is None:
        return epo
    if "badtrial" in epo.metadata.columns:
        bad = epo.metadata["badtrial"].fillna(0).astype(int).to_numpy()
        if bad.sum() > 0:
            epo = epo.copy()[bad == 0]
    return epo


def ensure_resampled(epo: mne.Epochs) -> mne.Epochs:
    if RESAMPLE_SFREQ is not None:
        epo = epo.copy().resample(RESAMPLE_SFREQ, npad="auto")
    return epo


def pick_first_existing_col(df: pd.DataFrame, candidates: list[str], *, label: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"Could not find {label} column in metadata. Tried: {candidates}")


def to_class_labels_from_levels(levels: np.ndarray) -> np.ndarray:
    levels = np.asarray(levels).astype(int)
    mapping = {20: 0, 40: 1, 60: 2, 80: 3, 100: 4}
    y = np.array([mapping.get(int(v), -1) for v in levels], dtype=int)
    if np.any(y < 0):
        raise ValueError(f"Unexpected levels seen: {np.unique(levels)}")
    return y


def parse_stim_code_to_level(series: pd.Series, prefix: str) -> np.ndarray:
    s = series.astype(str).str.strip().str.lower()
    pat = rf"^{re.escape(prefix)}\s*([1-5])$"
    codes = s.str.extract(pat, expand=False)
    if codes.isna().any():
        bad = s[codes.isna()].unique()[:10]
        raise ValueError(f"Unexpected {prefix} stim codes (examples): {bad}")
    codes_int = codes.astype(int).to_numpy()
    levels = np.array([LEVEL_CODE_TO_LEVEL[int(c)] for c in codes_int], dtype=int)
    return levels


def balance_nuisance_within_target(
    df: pd.DataFrame,
    *,
    target_levels: np.ndarray,   # numeric 20..100
    nuisance_levels: np.ndarray, # numeric 20..100
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Subsample indices so nuisance distribution is matched across target bins.
    Returns: indices into df (same index space).
    """
    d = df.copy()
    d = d.assign(_target=target_levels, _nuis=nuisance_levels)[["_target", "_nuis"]].dropna()

    target_vals = sorted(d["_target"].unique())
    nuis_vals = sorted(d["_nuis"].unique())

    counts = {(t, n): int(((d["_target"] == t) & (d["_nuis"] == n)).sum())
              for t in target_vals for n in nuis_vals}

    min_per_nuis = {n: min(counts[(t, n)] for t in target_vals) for n in nuis_vals}

    keep_indices = []
    for t in target_vals:
        for n in nuis_vals:
            k = min_per_nuis[n]
            if k <= 0:
                continue
            cell_idx = d.index[(d["_target"] == t) & (d["_nuis"] == n)].to_numpy()
            if cell_idx.size < k:
                continue
            keep_indices.append(rng.choice(cell_idx, size=k, replace=False))

    if len(keep_indices) == 0:
        raise ValueError("Balancing produced empty selection (check target/nuis values).")
    return np.unique(np.concatenate(keep_indices))


def residualize_X_by_nuisance(
    X: np.ndarray,                 # (n_trials, n_ch, n_t)
    nuisance_levels: np.ndarray,    # (n_trials,) e.g. pain levels 20..100
    *,
    model: str = "pain2",          # "pain2" or "pain+pain2"
) -> np.ndarray:
    n_trials, n_ch, n_t = X.shape
    z = nuisance_levels.astype(float)

    if model == "pain2":
        D = np.c_[np.ones(n_trials), (z ** 2)]
    elif model == "pain+pain2":
        D = np.c_[np.ones(n_trials), z, (z ** 2)]
    else:
        raise ValueError("model must be 'pain2' or 'pain+pain2'")

    X_flat = X.reshape(n_trials, -1)
    beta, *_ = np.linalg.lstsq(D, X_flat, rcond=None)
    X_hat = D @ beta
    R = X_flat - X_hat
    return R.reshape(n_trials, n_ch, n_t)


# =============================================================================
# Selectors per phase
# =============================================================================
def prepare_passive(epo: mne.Epochs, label: str):
    if epo.metadata is None:
        raise ValueError("No metadata.")
    md = epo.metadata.reset_index(drop=True)

    cond = md[COL_COND].astype(str).str.lower().to_numpy()
    keep = (cond == (COND_MONEY if label == "money" else COND_PAIN))
    epo_f = epo.copy()[keep]
    md_f = epo_f.metadata.reset_index(drop=True)

    levels_f = md_f[COL_LEVEL].to_numpy(dtype=int)
    X = epo_f.get_data()

    return X, levels_f, md_f


def prepare_decision(epo: mne.Epochs, label: str):
    if epo.metadata is None:
        raise ValueError("No metadata.")
    md = epo.metadata.reset_index(drop=True)

    if label == "money":
        col = pick_first_existing_col(md, DEC_MONEY_COL_CANDIDATES, label="decision money (moneystim)")
        levels = parse_stim_code_to_level(md[col], prefix="m")
    elif label == "pain":
        col = pick_first_existing_col(md, DEC_PAIN_COL_CANDIDATES, label="decision pain (painstim)")
        levels = parse_stim_code_to_level(md[col], prefix="p")
    else:
        raise ValueError("label must be 'money' or 'pain'")

    print("Before decision filtering:", len(epo))
    print("Levels parsed unique:", np.unique(levels))

    keep = np.isin(levels, LEVELS_ALL)
    epo_f = epo.copy()[keep]
    md_f = epo_f.metadata.reset_index(drop=True)

    print("After decision filtering:", len(epo_f))

    # recompute after filtering
    if label == "money":
        levels_f = parse_stim_code_to_level(md_f[col], prefix="m")
    else:
        levels_f = parse_stim_code_to_level(md_f[col], prefix="p")

    X = epo_f.get_data()
    return X, levels_f, md_f



# =============================================================================
# Estimators: classification + regression
# =============================================================================
def make_timegen_estimator_classification():
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            solver="lbfgs",
            multi_class="auto",
            max_iter=5000,
            random_state=RANDOM_STATE,
        )
    )
    return GeneralizingEstimator(clf, scoring="accuracy", n_jobs=1)


def _corr_scorer(estimator, X, y_true) -> float:
    y_pred = estimator.predict(X)
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if y_true.size < 3:
        return 0.0
    if np.std(y_true) < 1e-12 or np.std(y_pred) < 1e-12:
        return 0.0
    r = np.corrcoef(y_true, y_pred)[0, 1]
    return 0.0 if np.isnan(r) else float(r)


def make_timegen_estimator_regression():
    reg = make_pipeline(
        StandardScaler(),
        Ridge(alpha=1.0, random_state=RANDOM_STATE),
    )
    return GeneralizingEstimator(reg, scoring=_corr_scorer, n_jobs=1)


# =============================================================================
# Within-subject time×time (CV on training set)
# =============================================================================
def subject_timegen_crossphase(
    X_train: np.ndarray,
    y_train,
    X_test: np.ndarray,
    y_test,
    *,
    groups_train: np.ndarray | None,
    shuffle_train: bool,
    shuffle_test: bool,
    mode: str,  # "class" or "reg"
):
    rng = np.random.default_rng(RANDOM_STATE)
    y_tr = rng.permutation(y_train) if shuffle_train else y_train
    y_te = rng.permutation(y_test)  if shuffle_test  else y_test

    if mode == "class":
        timegen = make_timegen_estimator_classification()
        # classification CV
        if groups_train is not None and (~pd.isna(groups_train)).all():
            n_groups = len(np.unique(groups_train))
            cv = GroupKFold(n_splits=min(N_SPLITS, n_groups)) if n_groups >= 2 else StratifiedKFold(
                n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE
            )
        else:
            cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

        split_iter = cv.split(X_train, y_tr, groups=groups_train if isinstance(cv, GroupKFold) else None)

    elif mode == "reg":
        timegen = make_timegen_estimator_regression()
        # regression CV
        if groups_train is not None and (~pd.isna(groups_train)).all():
            n_groups = len(np.unique(groups_train))
            cv = GroupKFold(n_splits=min(N_SPLITS, n_groups)) if n_groups >= 2 else KFold(
                n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE
            )
        else:
            cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

        split_iter = cv.split(X_train, y_tr, groups=groups_train if isinstance(cv, GroupKFold) else None)

    else:
        raise ValueError("mode must be 'class' or 'reg'")

    mats = []
    for tr_idx, _ in split_iter:
        Xtr = X_train[tr_idx]
        ytr = np.asarray(y_tr)[tr_idx]
        timegen.fit(Xtr, ytr)
        mat = timegen.score(X_test, y_te)  # (n_train_times, n_test_times)
        mats.append(mat)

    return np.mean(np.stack(mats, axis=0), axis=0)


# =============================================================================
# Group stats on 2D grid
# =============================================================================
def group_cluster_timegen(mats_by_subj: np.ndarray, *, chance: float, tail: int):
    X = mats_by_subj - chance
    n_subj, n_tr, n_te = X.shape

    adjacency = combine_adjacency(n_tr, n_te)
    X_flat = X.reshape(n_subj, n_tr * n_te)

    tfce_thresh = dict(start=0.0, step=0.2)
    try:
        T_obs, clusters, cluster_pv, H0 = permutation_cluster_1samp_test(
            X_flat,
            n_permutations=N_PERM,
            threshold=tfce_thresh,
            tail=tail,
            adjacency=adjacency,
            out_type="mask",
            n_jobs=1,
            seed=RANDOM_STATE,
        )
        thresh_used = "tfce"
    except Exception:
        T_obs, clusters, cluster_pv, H0 = permutation_cluster_1samp_test(
            X_flat,
            n_permutations=N_PERM,
            threshold=None,
            tail=tail,
            adjacency=adjacency,
            out_type="mask",
            n_jobs=1,
            seed=RANDOM_STATE,
        )
        thresh_used = "threshold=None"

    p_map = np.ones((n_tr * n_te,), dtype=float)
    for cl, p in zip(clusters, cluster_pv):
        if cl is None:
            continue
        m = np.asarray(cl, dtype=bool).ravel()
        if m.size == p_map.size and m.any():
            p_map[m] = np.minimum(p_map[m], float(p))

    return dict(
        T_obs=T_obs.reshape(n_tr, n_te),
        clusters=clusters,
        cluster_pv=np.asarray(cluster_pv, dtype=float),
        p_map=p_map.reshape(n_tr, n_te),
        thresh_used=thresh_used,
    )


# =============================================================================
# Plotting + saving
# =============================================================================
def plot_timegen_heatmap(
    mean_mat: np.ndarray,
    times_train: np.ndarray,
    times_test: np.ndarray,
    p_map: np.ndarray,
    *,
    chance: float,
    title: str,
    out_path: Path,
    alpha: float,
    cbar_label: str,
):
    data = mean_mat - chance

    fig, ax = plt.subplots(figsize=(6.0, 5.2))
    im = ax.imshow(
        data,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=[times_test[0], times_test[-1], times_train[0], times_train[-1]],
    )
    cbar = fig.colorbar(im, ax=ax, shrink=0.9, pad=0.02)
    cbar.set_label(cbar_label)

    sig = (p_map < alpha)
    if np.any(sig):
        ax.contour(
            sig.astype(float),
            levels=[0.5],
            linewidths=1.5,
            origin="lower",
            extent=[times_test[0], times_test[-1], times_train[0], times_train[-1]],
        )

    ax.axvline(0, linewidth=1)
    ax.axhline(0, linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Test time (s)")
    ax.set_ylabel("Train time (s)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def save_cluster_table_2d(out_csv: Path, times_train: np.ndarray, times_test: np.ndarray, stats_out: dict):
    rows = []
    clusters = stats_out["clusters"]
    cluster_pv = stats_out["cluster_pv"]

    n_tr = len(times_train)
    n_te = len(times_test)

    for i, (cl, p) in enumerate(zip(clusters, cluster_pv)):
        if cl is None:
            continue
        m = np.asarray(cl, dtype=bool).ravel()
        if m.size != n_tr * n_te or (not m.any()):
            continue
        m2 = m.reshape(n_tr, n_te)

        tr_inds = np.where(np.any(m2, axis=1))[0]
        te_inds = np.where(np.any(m2, axis=0))[0]

        rows.append(dict(
            cluster=i,
            p_value=float(p),
            train_t_start_s=float(times_train[tr_inds[0]]),
            train_t_end_s=float(times_train[tr_inds[-1]]),
            test_t_start_s=float(times_test[te_inds[0]]),
            test_t_end_s=float(times_test[te_inds[-1]]),
            n_cells=int(m.sum()),
        ))

    df = pd.DataFrame(rows).sort_values("p_value") if len(rows) else pd.DataFrame(
        columns=["cluster","p_value","train_t_start_s","train_t_end_s","test_t_start_s","test_t_end_s","n_cells"]
    )
    df.to_csv(out_csv, index=False)


# =============================================================================
# Core runner (single analysis)
# =============================================================================
def run_one_analysis(
    *,
    tag: str,
    mode: str,                     # "class" or "reg"
    train_phase: str,
    train_label: str,
    test_phase: str,
    test_label: str,
    balance_nuisance: bool,
    nuisance_label: str | None,
    residualize_test: bool,
    residualize_nuisance: str | None,   # "pain" or "money"
    residualize_model: str,             # "pain2" or "pain+pain2"
    shuffle_train: bool,
    shuffle_test: bool,
):
    out_dir = OUT_DIR / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    figs_dir = out_dir / "figs"
    figs_dir.mkdir(exist_ok=True)

    def log(msg: str):
        tqdm.write(msg)

    rng = np.random.default_rng(RANDOM_STATE)

    subs = list_subjects(DERIV_DIR)
    included, skipped = [], []
    mats = []
    times_train = None
    times_test = None

    pbar = tqdm(subs, desc=tag, unit="sub", dynamic_ncols=True)
    for sub in pbar:
        try:
            # ---- load train ----
            epo_tr = ensure_resampled(drop_badtrials(
                merge_beh_into_epochs(load_epochs(sub, train_phase), load_beh(sub, train_phase), sub=sub, phase=train_phase)
            ))

            print(f"\n{sub} TRAIN ({train_phase})")
            print("Epochs:", len(epo_tr))
            print("Metadata rows:", len(epo_tr.metadata))
            
            if "trialsnum" in epo_tr.metadata.columns:
                print("Unique trialsnum in epochs:", epo_tr.metadata["trialsnum"].nunique())
            else:
                print("trialsnum missing in epochs metadata")

            print("NaNs in level columns:",
                  epo_tr.metadata.isna().sum().sort_values(ascending=False).head(5))

            # ---- load test ----
            epo_te = ensure_resampled(drop_badtrials(
                merge_beh_into_epochs(load_epochs(sub, test_phase), load_beh(sub, test_phase), sub=sub, phase=test_phase)
            ))

            print(f"\n{sub} TEST ({test_phase})")
            print("Epochs:", len(epo_te))
            print("Metadata rows:", len(epo_te.metadata))
            print("Unique trialsnum in epochs:", len(np.unique(epo_te.metadata.get('trialsnum', []))))

            if "moneystim" in epo_te.metadata.columns:
                print("Unique moneystim values:",
                      epo_te.metadata["moneystim"].dropna().unique()[:15])
            if "painstim" in epo_te.metadata.columns:
                print("Unique painstim values:",
                      epo_te.metadata["painstim"].dropna().unique()[:15])

            print("NaNs in moneystim:",
                  epo_te.metadata["moneystim"].isna().sum() if "moneystim" in epo_te.metadata else "missing")


            # ---- select train ----
            if train_phase == "passive":
                Xtr, tr_levels, md_tr = prepare_passive(epo_tr, train_label)
            else:
                Xtr, tr_levels, md_tr = prepare_decision(epo_tr, train_label)

            # ---- select test ----
            if test_phase == "passive":
                Xte, te_levels, md_te = prepare_passive(epo_te, test_label)
            else:
                Xte, te_levels, md_te = prepare_decision(epo_te, test_label)

            # ---- nuisance levels for decision test set (needed for balancing/residualization) ----
            # Only meaningful when test_phase == "decision"
            if test_phase == "decision":
                pain_levels_all = parse_stim_code_to_level(md_te[pick_first_existing_col(md_te, DEC_PAIN_COL_CANDIDATES, label="painstim")], "p")
                money_levels_all = parse_stim_code_to_level(md_te[pick_first_existing_col(md_te, DEC_MONEY_COL_CANDIDATES, label="moneystim")], "m")
            else:
                pain_levels_all = None
                money_levels_all = None

            # ---- balancing on TEST ----
            if balance_nuisance:
                if test_phase != "decision":
                    raise ValueError("Balancing is implemented for decision test only.")
                if nuisance_label is None:
                    raise ValueError("nuisance_label must be set when balance_nuisance=True")

                if nuisance_label == "pain":
                    nuis = pain_levels_all
                elif nuisance_label == "money":
                    nuis = money_levels_all
                else:
                    raise ValueError("nuisance_label must be 'pain' or 'money'")

                keep_idx = balance_nuisance_within_target(
                    md_te,
                    target_levels=te_levels,
                    nuisance_levels=nuis,
                    rng=rng,
                )

                keep_mask = md_te.index.isin(keep_idx)
                Xte = Xte[keep_mask]
                te_levels = te_levels[keep_mask]
                md_te = md_te.loc[keep_mask].reset_index(drop=True)

                # recompute nuis arrays aligned to kept set (important!)
                if test_phase == "decision":
                    pain_levels_all = parse_stim_code_to_level(md_te[pick_first_existing_col(md_te, DEC_PAIN_COL_CANDIDATES, label="painstim")], "p")
                    money_levels_all = parse_stim_code_to_level(md_te[pick_first_existing_col(md_te, DEC_MONEY_COL_CANDIDATES, label="moneystim")], "m")

                if len(te_levels) < 20:
                    raise ValueError("Too few test trials after balancing.")

            # ---- residualize on TEST EEG ----
            if residualize_test:
                if test_phase != "decision":
                    raise ValueError("Residualization implemented for decision test only.")
                if residualize_nuisance is None:
                    raise ValueError("residualize_nuisance must be 'pain' or 'money'")

                if residualize_nuisance == "pain":
                    nuis_levels = pain_levels_all
                elif residualize_nuisance == "money":
                    nuis_levels = money_levels_all
                else:
                    raise ValueError("residualize_nuisance must be 'pain' or 'money'")

                Xte = residualize_X_by_nuisance(Xte, nuis_levels, model=residualize_model)

            # ---- build y for model ----
            if mode == "class":
                ytr = to_class_labels_from_levels(tr_levels)
                yte = to_class_labels_from_levels(te_levels)
                chance = CHANCE_5CLASS
                tail = 1
                cbar_label = "Accuracy − chance"
            else:
                # regression: use numeric levels directly
                ytr = tr_levels.astype(float)
                yte = te_levels.astype(float)
                chance = CHANCE_R
                tail = 0
                cbar_label = "Pearson r (pred − true)"

            # ---- training groups ----
            groups_tr = None
            if (md_tr is not None) and (KEY_BLOCK in md_tr.columns):
                groups_tr = md_tr[KEY_BLOCK].to_numpy()

            # ---- time axes ----
            if times_train is None:
                times_train = epo_tr.times.copy()
                times_test = epo_te.times.copy()
            else:
                if len(epo_tr.times) != len(times_train) or np.max(np.abs(epo_tr.times - times_train)) > 1e-9:
                    raise RuntimeError("Train time axis mismatch across subjects.")
                if len(epo_te.times) != len(times_test) or np.max(np.abs(epo_te.times - times_test)) > 1e-9:
                    raise RuntimeError("Test time axis mismatch across subjects.")

            # ---- compute subject matrix ----
            mat = subject_timegen_crossphase(
                Xtr, ytr, Xte, yte,
                groups_train=groups_tr,
                shuffle_train=shuffle_train,
                shuffle_test=shuffle_test,
                mode=mode,
            )
            mats.append(mat)
            included.append(sub)

            pbar.set_postfix_str(f"{sub} | tr={len(ytr)} te={len(yte)} | mean={np.mean(mat):.3f}")

        except Exception as e:
            skipped.append((sub, str(e)))
            log(f"Skipped {sub}: {e}")

    if len(mats) < 8:
        raise RuntimeError(f"{tag}: too few subjects for group stats (n={len(mats)})")

    mats = np.stack(mats, axis=0)  # (n_subj, n_tr, n_te)

    # ---- restrict to stats window ----
    tr_mask = (times_train >= TMIN_STAT) & (times_train <= TMAX_STAT)
    te_mask = (times_test >= TMIN_STAT) & (times_test <= TMAX_STAT)
    tr_times_stat = times_train[tr_mask]
    te_times_stat = times_test[te_mask]
    mats_stat = mats[:, tr_mask][:, :, te_mask]

    # ---- group stats ----
    stats_out = group_cluster_timegen(mats_stat, chance=chance, tail=tail)

    mean_mat = np.mean(mats_stat, axis=0)
    sem_mat = np.std(mats_stat, axis=0, ddof=1) / np.sqrt(mats_stat.shape[0])

    # ---- save ----
    np.savez(
        out_dir / f"{tag}_timegen_group_results.npz",
        subj_mats=mats,
        subj_mats_stat=mats_stat,
        times_train=times_train,
        times_test=times_test,
        times_train_stat=tr_times_stat,
        times_test_stat=te_times_stat,
        included=np.array(included, dtype=object),
        skipped=np.array(skipped, dtype=object),
        mode=mode,
        chance=float(chance),
        mean_mat=mean_mat,
        sem_mat=sem_mat,
        T_obs=stats_out["T_obs"],
        p_map=stats_out["p_map"],
        cluster_pv=stats_out["cluster_pv"],
        thresh_used=stats_out["thresh_used"],
        alpha_cluster=float(ALPHA_CLUSTER),
        n_perm=int(N_PERM),
    )

    save_cluster_table_2d(
        out_csv=out_dir / f"{tag}_cluster_table.csv",
        times_train=tr_times_stat,
        times_test=te_times_stat,
        stats_out=stats_out,
    )

    meta = dict(
        tag=tag,
        mode=mode,
        n_subjects=int(mats_stat.shape[0]),
        n_train_times=int(mats_stat.shape[1]),
        n_test_times=int(mats_stat.shape[2]),
        chance=float(chance),
        alpha=float(ALPHA_CLUSTER),
        tail=int(tail),
        thresh_used=stats_out["thresh_used"],
        min_cluster_p=float(np.min(stats_out["cluster_pv"])) if len(stats_out["cluster_pv"]) else 1.0,
        included_subjects=included,
        train_phase=train_phase,
        train_label=train_label,
        test_phase=test_phase,
        test_label=test_label,
        balance_nuisance=bool(balance_nuisance),
        nuisance_label=nuisance_label,
        residualize_test=bool(residualize_test),
        residualize_nuisance=residualize_nuisance,
        residualize_model=residualize_model,
        shuffle_train=bool(shuffle_train),
        shuffle_test=bool(shuffle_test),
        resample_sfreq=RESAMPLE_SFREQ if RESAMPLE_SFREQ is not None else -1,
        tmin_stat=float(TMIN_STAT),
        tmax_stat=float(TMAX_STAT),
    )
    with open(out_dir / f"{tag}_summary.json", "w") as f:
        json.dump(meta, f, indent=2)

    plot_timegen_heatmap(
        mean_mat=mean_mat,
        times_train=tr_times_stat,
        times_test=te_times_stat,
        p_map=stats_out["p_map"],
        chance=chance,
        title=f"{tag} ({mode}): time×time generalization",
        out_path=figs_dir / f"{tag}_heatmap.png",
        alpha=ALPHA_CLUSTER,
        cbar_label=cbar_label,
    )

    log(f"DONE {tag}: included={len(included)} | min cluster p="
        f"{(float(np.min(stats_out['cluster_pv'])) if len(stats_out['cluster_pv']) else 1.0):.6f}")


# =============================================================================
# Main
# =============================================================================
def main():
    mne.set_log_level("WARNING")

    analyses = []

    ###
    
    analyses.append(dict(
        tag="REG_trainPASS_money__testDEC_money__painBalanced",
        mode="reg",
        train_phase="passive", train_label="money",
        test_phase="decision", test_label="money",
        balance_nuisance=True, nuisance_label="pain",
        residualize_test=False, residualize_nuisance=None, residualize_model="pain2",
        shuffle_train=False, shuffle_test=False,
    ))

    analyses.append(dict(
        tag="REG_trainPASS_money__testDEC_money__residPain2",
        mode="reg",
        train_phase="passive", train_label="money",
        test_phase="decision", test_label="money",
        balance_nuisance=False, nuisance_label=None,
        residualize_test=True, residualize_nuisance="pain", residualize_model="pain2",
        shuffle_train=False, shuffle_test=False,
    ))

    analyses.append(dict(
        tag="REG_trainPASS_money__testDEC_money__residPainPlusPain2",
        mode="reg",
        train_phase="passive", train_label="money",
        test_phase="decision", test_label="money",
        balance_nuisance=False, nuisance_label=None,
        residualize_test=True, residualize_nuisance="pain", residualize_model="pain+pain2",
        shuffle_train=False, shuffle_test=False,
    ))

    analyses.append(dict(
        tag="REG_trainPASS_money__testDEC_money__noControl",
        mode="reg",
        train_phase="passive", train_label="money",
        test_phase="decision", test_label="money",
        balance_nuisance=False, nuisance_label=None,
        residualize_test=False, residualize_nuisance=None, residualize_model="pain2",
        shuffle_train=False, shuffle_test=False,
    ))

    # classification 
    analyses.append(dict(
        tag="CLF_trainPASS_money__testDEC_money__noControl",
        mode="class",
        train_phase="passive", train_label="money",
        test_phase="decision", test_label="money",
        balance_nuisance=False, nuisance_label=None,
        residualize_test=False, residualize_nuisance=None, residualize_model="pain2",
        shuffle_train=False, shuffle_test=False,
    ))

    for cfg in analyses:
        run_one_analysis(**cfg)


if __name__ == "__main__":
    main()
