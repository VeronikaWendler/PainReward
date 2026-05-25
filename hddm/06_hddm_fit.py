# hddm_fit.py — HDDM model fitting for the PainReward task
#
# Input:  derivatives/behav/hddm_ready.csv  (output of hddm_prep.py)
# Output: MODEL_DIR/<model_name>_{0..n_chains-1}.pkl  and  .nc  and  .hddm
#
# Run: python hddm_fit.py [--start-version N] [--n-chains 4] [--samples 50000] [--burn 10000]
# Env vars: PROJECT_DIR, MODEL_DIR, FIG_DIR

import os
import sys
import time
import types
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import arviz as az
import matplotlib
matplotlib.use("Agg")
import cloudpickle
import dill
if hasattr(dill, "dump"):
    cloudpickle.dump = dill.dump
import dill as pickle

import hddm
import kabuki  # required by hddm at import time

from joblib import Parallel, delayed
from model_specs import MODEL_SPECS, REQUIRED_COLS, MODEL_RUN_ORDER, MODEL_BASE_NAME

warnings.simplefilter(action="ignore", category=FutureWarning)

# Stub Windows-only module so imports never fail on Linux/Mac
sys.modules.setdefault("winreg", types.ModuleType("winreg"))
sys.modules.setdefault("_gdbm",  types.ModuleType("_gdbm"))

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_DIR    = Path(os.getenv("PROJECT_DIR", str(Path(__file__).resolve().parent.parent.parent))).resolve()
BASE_MODEL_DIR = Path(os.getenv("MODEL_DIR",
                      str(PROJECT_DIR / "derivatives" / "hddm" / "models"))).resolve()

INPUT_CSV = PROJECT_DIR / "derivatives" / "behav" / "hddm_ready.csv"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sanitize_infdata(infdata):
    """Replace pd.NA with np.nan in InferenceData (kabuki compatibility fix)."""
    for group in infdata._groups_all:
        if hasattr(infdata, group):
            dataset = getattr(infdata, group)
            for var in dataset.data_vars:
                values = dataset[var].values
                if isinstance(values, np.ndarray) and values.dtype == "object":
                    mask = pd.isna(values)
                    if mask.any():
                        values[mask] = np.nan
                        dataset[var].values = values
    return infdata


def quick_report(data: pd.DataFrame, version: int, model_name: str) -> None:
    print(f"\n--- Version {version} | {model_name} ---")
    print(f"N trials      : {len(data):,}")
    print(f"Participants  : {sorted(data['subj_idx'].unique())}")


def clean_data(data: pd.DataFrame, version: int) -> pd.DataFrame:
    """Drop rows with NaNs in the columns this version actually needs.

    For rp_z models (17, 18, 20), also drop rows where badtrial == 1.
    """
    df = data.copy()
    if version in [17, 18, 20]:
        before = len(df)
        df = df[df["badtrial"] == 0]
        print(f"  badtrial filter: removed {before - len(df)} trials, {len(df)} remain.")
    req = REQUIRED_COLS[version]
    before = len(df)
    df = df.dropna(subset=req)
    print(f"  dropna on {req}: removed {before - len(df)} trials, {len(df)} remain.")
    return df


# ---------------------------------------------------------------------------
# MCMC chain runner (called in parallel)
# ---------------------------------------------------------------------------

def run_chain(trace_id: int, data: pd.DataFrame, model_dir: Path,
              model_name: str, version: int, samples: int, burn: int) -> tuple:
    import os
    import hddm
    from pathlib import Path

    spec = MODEL_SPECS[version]

    if spec["class"] == "HDDM":
        m = hddm.models.HDDM(data, p_outlier=0.05,
                              include=["a", "t", "v", "z"])
    else:
        m = hddm.models.HDDMRegressor(data, spec["regs"],
                                      p_outlier=0.05,
                                      include=["a", "t", "v"],
                                      group_only_regressors=False,
                                      keep_regressor_trace=True)
    m.find_starting_values()
    infdata = m.sample(
        samples, burn=burn,
        dbname=str(model_dir / f"{model_name}_db{trace_id}"),
        db="pickle",
        return_infdata=True, loglike=True, ppc=True,
    )
    return m, infdata


def fit_model(data: pd.DataFrame, version: int, model_name: str,
              model_dir: Path, n_chains: int, samples: int, burn: int) -> None:
    ensure_dir(model_dir)
    t0 = time.time()
    results = Parallel(n_jobs=n_chains)(
        delayed(run_chain)(i, data, model_dir, model_name, version, samples, burn)
        for i in range(n_chains)
    )
    print(f"Sampling done in {time.time() - t0:.0f}s")

    for i, (model, infdata) in enumerate(results):
        model.save(str(model_dir / f"{model_name}_{i}.hddm"))
        with open(model_dir / f"{model_name}_{i}.pkl", "wb") as fh:
            pickle.dump(model, fh)
        infdata = sanitize_infdata(infdata)
        az.to_netcdf(infdata, str(model_dir / f"{model_name}_{i}.nc"))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit HDDM models for PainReward")
    parser.add_argument("--start-version", type=int, default=0,
                        help="Skip versions before this number (for resuming)")
    parser.add_argument("--n-chains",  type=int, default=4)
    parser.add_argument("--samples",   type=int, default=50000)
    parser.add_argument("--burn",      type=int, default=10000)
    parser.add_argument("--model-dir", type=Path, default=BASE_MODEL_DIR)
    args = parser.parse_args()

    ensure_dir(args.model_dir)

    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Input CSV not found: {INPUT_CSV}\nRun hddm_prep.py first."
        )

    data_full = pd.read_csv(INPUT_CSV)

    for version in MODEL_RUN_ORDER:
        if version < args.start_version:
            continue

        model_name = MODEL_BASE_NAME + f"mod_{version}"
        print(f"\n=== Version {version} | {model_name} ===")

        data_clean = clean_data(data_full, version)

        if data_clean.empty:
            raise ValueError(f"No rows left after cleaning for version {version}.")

        quick_report(data_clean, version, model_name)
        fit_model(data_clean, version, model_name, args.model_dir,
                  args.n_chains, args.samples, args.burn)
