# Parameter recovery for group and participant level

#libraries
import os, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import arviz as az
import hddm
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import trange
import re

warnings.filterwarnings("ignore", category=FutureWarning)

PROJECT_DIR    = Path(os.getenv("PROJECT_DIR", "/workspace")).resolve()
BASE_MODEL_DIR = PROJECT_DIR / "models_dir_OV"
FIG_DIR        = PROJECT_DIR / "figures_dir_OV/OV_replication_For_paper_5/recovery_For_paper_m5"
FIG_DIR.mkdir(parents=True, exist_ok=True)

EMPIRICAL_POST_PATHS = [
    BASE_MODEL_DIR / "OV_replication_For_paper_5_0.nc",
    BASE_MODEL_DIR / "OV_replication_For_paper_5_1.nc",
    BASE_MODEL_DIR / "OV_replication_For_paper_5_2.nc",
]

#try 2
N_REPS    = 2      
N_SAMPLES = 1000
BURN      = 100

# group-level parameters
PARAM_LIST = [
    'a',
    't',
    'z',
    'v_ES_AttentionW',
    'v_ES_InattentionW_E',
    'v_ES_InattentionW_S'
]
  
# group-level SD
PARAM_LIST_SD = [
    'a_std',
    't_std',
    'z_std',
    'v_ES_AttentionW_std',
    'v_ES_InattentionW_E_std',
    'v_ES_InattentionW_S_std'
]
PARAM_SD_MAP = dict(zip(PARAM_LIST, PARAM_LIST_SD))

# HDDM model
v_reg = {'model': 'v ~ 0 + ES_AttentionW + ES_InattentionW_E + ES_InattentionW_S', 'link_func': lambda x: x}
reg_descr = [v_reg]
depends_on={}  #'a':'OVcate'

# helper functions
# CSV writer
def atomic_to_csv(df: pd.DataFrame, path: Path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)

def _read_or_empty(path, cols):
    # empty or missing file ... empty dataframe with columns
    if (not path.exists()) or (path.stat().st_size == 0):
        return pd.DataFrame(columns=cols)
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=cols)
    keep = [c for c in cols if c in df.columns]
    return df[keep] if keep else pd.DataFrame(columns=cols)


def extract_group_sample(idata, *, seed=None):
    rng   = np.random.default_rng(seed)
    draw  = rng.integers(idata.posterior.dims["draw"])
    chain = rng.integers(idata.posterior.dims["chain"])
    return {p: float(idata.posterior[p].isel(chain=chain, draw=draw)) for p in PARAM_LIST}

def extract_group_sd(idata, *, seed=None):
    rng   = np.random.default_rng(seed)
    draw  = rng.integers(idata.posterior.dims["draw"])
    chain = rng.integers(idata.posterior.dims["chain"])
    out = {}
    for p in PARAM_LIST:
        sd_name = PARAM_SD_MAP.get(p)
        if (sd_name is not None) and (sd_name in idata.posterior):
            out[p] = float(idata.posterior[sd_name].isel(chain=chain, draw=draw))
        else:
            out[p] = 0.0

    if "a_std" in idata.posterior:
        out["a_std"] = float(idata.posterior["a_std"].isel(chain=chain, draw=draw))
    else:
        out["a_std"] = 0.0

    return out


# def sample_true_subjects(mu_dict, sd_dict, subjects, *, seed=None):
#     rng = np.random.default_rng(seed)
#     true_individuals = {}

#     # SD for all boundary separation params
#     a_sd = sd_dict.get("a_std", 0.0)

#     for s in subjects:
#         pars = {}

#         for a_key in ['a(high)', 'a(low)', 'a(medium)']:
#             mu = mu_dict[a_key]
#             sd_use = a_sd if a_sd > 0 else sd_dict.get(a_key, 0.0)
#             if sd_use == 0:
#                 sd_use = 0.1  
#             pars[a_key] = float(rng.normal(mu, sd_use))

#         for p in PARAM_LIST:
#             if p in {'a(high)', 'a(low)', 'a(medium)'}:
#                 continue
#             mu, sd = mu_dict[p], sd_dict.get(p, 0.0)
#             pars[p] = float(rng.normal(mu, sd)) if sd > 0 else float(mu)

#         true_individuals[s] = pars

#     print("a_std used:", a_sd)
#     print("Subject 1 a's:", {k: true_individuals[subjects[0]][k] for k in ['a(high)','a(low)','a(medium)']})

#     return true_individuals
def sample_true_subjects(mu_dict, sd_dict, subjects, *, seed=None):
    rng = np.random.default_rng(seed)
    true_individuals = {}
    a_sd = sd_dict.get("a_std", 0.0)
    if a_sd == 0:
        print("ERROR: Posterior does not contain sd")

    for s in subjects:
        pars = {}
        pars["a"] = float(rng.normal(mu_dict["a"], a_sd))
        for p in PARAM_LIST:
            if p == "a":
                continue
            mu = mu_dict[p]
            sd = sd_dict.get(p, 0.0)
            pars[p] = float(rng.normal(mu, sd)) if sd > 0 else float(mu)
        true_individuals[s] = pars
    return true_individuals


#helper to flatten draws for participants
def flatten_true_subjects(true_individuals, rep):
    rows = []
    for subj, pmap in true_individuals.items():
        for p, val in pmap.items():
            rows.append(dict(rep=rep, subj=subj, parameter=p, true=val))
    return rows


def simulate_dataset(true_individuals, raw_df):
    # simulation code
    # a varies by OV
    # v is computed from regressors
    # t & z are constants

    sim_rows = []
    tmp = raw_df.copy()

    tmp["subj_idx"] = tmp["subj_idx"].astype(int)
    tmp["trial"] = tmp.groupby("subj_idx").cumcount()


    # def _norm_ov(ov_raw):
    #     ov = str(ov_raw).strip().lower()
    #     if ov in {"low"}: return "low"
    #     if ov in {"medium"}: return "medium"
    #     if ov in {"high"}: return "high"
    #     return ov  
    for _, tr in tmp.iterrows():
        subj = int(tr["subj_idx"])
#        ov   = _norm_ov(tr["OVcate"])
        pars = true_individuals[subj]
#        a_key = f"a({ov})"
#        if a_key not in pars:
#            map_ = {"low": "a(low)", "medium": "a(medium)", "high": "a(high)"}
#            a_key = map_.get(ov, "a(low)")
#        a_val = float(pars[a_key])

        v_trial = (
            pars["v_ES_AttentionW"] * float(tr["ES_AttentionW"]) +
            pars["v_ES_InattentionW_E"] * float(tr["ES_InattentionW_E"]) +
            pars["v_ES_InattentionW_S"] * float(tr["ES_InattentionW_S"])
        )

        par_dict = {"v": v_trial, "a": float(pars["a"]), "t": float(pars["t"])}   # would b a_val for 'a' if a~OV
        if "z" in pars:
            par_dict["z"] = float(pars["z"])
        trial_df, _ = hddm.generate.gen_rand_data(par_dict, size=1, subjs=1)

        for col in ["subj_idx", "OVcate", "ES_AttentionW", "ES_InattentionW_E", "ES_InattentionW_S"]:
            trial_df[col] = tr[col]
        sim_rows.append(trial_df)

    return pd.concat(sim_rows, ignore_index=True)


def refit_and_get_means(sim_df, seed):
    np.random.seed(seed)
    mdl = hddm.HDDMRegressor(
        sim_df, reg_descr,
        include=["a","t","v","z"],
        p_outlier=0.05,
        depends_on=depends_on,
        keep_regressor_trace=True,
        group_only_regressors=False,
    )
    mdl.find_starting_values()
    mdl.sample(N_SAMPLES, burn=BURN, db='ram', dbname=f'ram_{seed}')
    means = {p: mdl.nodes_db.loc[p, 'node'].trace().mean() for p in PARAM_LIST}
    return means, mdl


def extract_individual_means(mdl):
    out = {}
    for node in mdl.nodes_db.index:
        if "_subj." not in node and "_subj(" not in node:
            continue

        for regex in [
            r"^([A-Za-z_]+)_subj\(([^)]+)\)\.(\d+)$",       # a_subj(high).1              # this is the notation if we have OV in the exp
            r"^([A-Za-z_]+)_subj\.(\d+)$"                   # z_subj.1, t_subj.1          # this is the notation for the other params
        ]:
            m = re.match(regex, node)
            if not m:
                continue

            if len(m.groups()) == 3:
                base, level, subj = m.groups()
                param = f"{base}({level})"
            elif len(m.groups()) == 2:
                base, subj = m.groups()
                param = base
            else:
                continue

            if param in PARAM_LIST:
                out[(int(subj), param)] = mdl.nodes_db.loc[node, 'node'].trace().mean()
                break  

    return out



# -----------------------------------------------------------------------

# load fit & predictors
idatas = [az.from_netcdf(p) for p in EMPIRICAL_POST_PATHS]
empirical = az.concat(idatas, dim="chain")
raw_df = idatas[0].observed_data.to_dataframe().reset_index(drop=True)
raw_df["subj_idx"] = raw_df["subj_idx"].astype(int)
subjects = sorted(raw_df["subj_idx"].unique())

group_records = []
indiv_records = []
true_draw_records = [] 

# I want to save draws in case something crashes (hence, partial)
GROUP_PARTIAL_CSV = FIG_DIR / "partial_group5.csv"
INDIV_PARTIAL_CSV = FIG_DIR / "partial_individual5.csv"
TRUE_PARTIAL_CSV  = FIG_DIR / "partial_true_subject_draws5.csv"

expected_per_rep = len(PARAM_LIST)

group_cols = ["rep", "parameter", "true", "recovered"]
indiv_cols = ["rep", "subj", "parameter", "true", "recovered"]
true_cols  = ["rep", "subj", "parameter", "true"]

group_partial = _read_or_empty(GROUP_PARTIAL_CSV, group_cols)
indiv_partial = _read_or_empty(INDIV_PARTIAL_CSV, indiv_cols)
true_partial  = _read_or_empty(TRUE_PARTIAL_CSV,  true_cols)

counts = group_partial.groupby("rep")["parameter"].count()
complete_reps = set(counts[counts >= expected_per_rep].index.tolist())
all_reps_seen = set(group_partial["rep"].unique())

incomplete_reps = all_reps_seen - complete_reps
if incomplete_reps:
    group_partial = group_partial[~group_partial["rep"].isin(incomplete_reps)]
    indiv_partial = indiv_partial[~indiv_partial["rep"].isin(incomplete_reps)]
    true_partial  = true_partial[~true_partial["rep"].isin(incomplete_reps)]

start_rep = (max(complete_reps) + 1) if len(complete_reps) else 0
print(f"Resuming from rep {start_rep} (completed reps: {sorted(complete_reps)})")

group_records = group_partial.to_dict("records")
indiv_records = indiv_partial.to_dict("records")
true_draw_records = true_partial.to_dict("records")

# ---------------------------------------------------------------


for rep in trange(start_rep, N_REPS, desc="parameter-recovery", unit="rep"):
    try:
        mu = extract_group_sample(empirical, seed=rep)
        sd = extract_group_sd(empirical, seed=rep+999)

        true_individuals = sample_true_subjects(mu, sd, subjects, seed=42+rep)
        true_draw_records.extend(flatten_true_subjects(true_individuals, rep))
        atomic_to_csv(pd.DataFrame(true_draw_records), TRUE_PARTIAL_CSV)  # save ASAP

        sim_df = simulate_dataset(true_individuals, raw_df)
        atomic_to_csv(sim_df, FIG_DIR / f"sim_df_rep{rep}.csv")

        θ_hat_group, mdl = refit_and_get_means(sim_df, seed=10000 + rep)
        for p in PARAM_LIST:
            group_records.append(dict(rep=rep, parameter=p, true=mu[p], recovered=θ_hat_group[p]))

        np.random.seed(20000 + rep)
        mdl = hddm.HDDMRegressor(sim_df, reg_descr, include=["a","t","v","z"], depends_on=depends_on,
                                 p_outlier=0.05, keep_regressor_trace=True,
                                 group_only_regressors=False)
        mdl.find_starting_values()
        mdl.sample(N_SAMPLES, burn=BURN, db='ram', dbname=f'indiv_{rep}')

        indiv_means = extract_individual_means(mdl)
        for (subj, param), rec_val in indiv_means.items():
            true_val = true_individuals[subj][param]
            indiv_records.append(dict(rep=rep, subj=subj, parameter=param, true=true_val, recovered=rec_val))
            
            
    finally:
        # write what we have so far
        atomic_to_csv(pd.DataFrame(group_records), GROUP_PARTIAL_CSV)
        atomic_to_csv(pd.DataFrame(indiv_records), INDIV_PARTIAL_CSV)


# final CSVs
pd.DataFrame(group_records).to_csv(FIG_DIR/"true_vs_recovered_group5.csv", index=False)
pd.DataFrame(indiv_records).to_csv(FIG_DIR/"true_vs_recovered_individual5.csv", index=False)
pd.DataFrame(true_draw_records).to_csv(FIG_DIR/"true_subject_draws_all5.csv", index=False)  

# some immediate plotting
sns.set_style("white")

# group plot
grp = pd.DataFrame(group_records)
g = sns.FacetGrid(grp, col="parameter", col_wrap=3, sharex=False, sharey=False, height=3.0)
g.map_dataframe(sns.scatterplot, x="true", y="recovered", s=28, alpha=.85)
for ax in g.axes.ravel():
    lo = min(ax.get_xlim()[0], ax.get_ylim()[0])
    hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([lo, hi], [lo, hi], "--k", lw=1)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
g.set_axis_labels("true value", "posterior mean (recovered)")
g.tight_layout()
g.savefig(FIG_DIR/"scatter_group6.png", dpi=300)

# individual plot
ind = pd.DataFrame(indiv_records)
h = sns.FacetGrid(ind, col="parameter", col_wrap=3, sharex=False, sharey=False, height=3.0)
h.map_dataframe(sns.scatterplot, x="true", y="recovered", s=16, alpha=.5)
for ax in h.axes.ravel():
    lo = min(ax.get_xlim()[0], ax.get_ylim()[0])
    hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([lo, hi], [lo, hi], "--k", lw=1)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
h.set_axis_labels("true value", "posterior mean (recovered)")
h.tight_layout()
h.savefig(FIG_DIR/"scatter_individual5.png", dpi=300)

print("Done.")

