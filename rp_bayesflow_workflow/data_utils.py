from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from config import DEFAULT_COLUMNS


def infer_signed_rt(df: pd.DataFrame, rt_col: str, response_col: str) -> pd.Series:
    """Return signed RT: positive = response 1, negative = response 0.

    Assumes response is coded 1/0 or True/False. Adjust here if your coding differs.
    """
    response = pd.to_numeric(df[response_col], errors="coerce")
    rt = pd.to_numeric(df[rt_col], errors="coerce")
    return np.where(response > 0, rt, -rt)


def load_and_prepare_data(
    csv_path: str | Path,
    columns: Dict[str, str] | None = None,
    min_rt: float = 0.25,
    drop_badtrial_column: str | None = "badtrial",
    zscore_rp_within_subject: bool = False,
) -> pd.DataFrame:
    """Load the real task data and keep the columns needed by the workflow.

    Expected columns by default:
    subj_idx, pain_z, money_z, rp_z, rt, response

    Returns a cleaned dataframe with an added column 'signed_rt'.
    """
    cols = DEFAULT_COLUMNS.copy()
    if columns:
        cols.update(columns)

    df = pd.read_csv(csv_path).copy()

    # Harmonize rt if needed
    if cols["rt"] not in df.columns:
        if "choice_resp.rt" in df.columns:
            cols["rt"] = "choice_resp.rt"
        else:
            raise KeyError(f"Could not find RT column '{cols['rt']}' in {csv_path}")

    required = [cols[k] for k in ["subject", "pain", "money", "rp", "rt", "response"]]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    # Basic RT filtering
    df = df[pd.to_numeric(df[cols["rt"]], errors="coerce") > min_rt].copy()

    # Optional bad-trial exclusion
    if drop_badtrial_column and drop_badtrial_column in df.columns:
        df = df[df[drop_badtrial_column] == 0].copy()

    # Drop missing values in modeling columns
    df = df.dropna(subset=required).copy()

    if zscore_rp_within_subject:
        subj = cols["subject"]
        rp = cols["rp"]
        df[rp] = df.groupby(subj)[rp].transform(
            lambda x: (x - x.mean()) / x.std(ddof=0) if x.std(ddof=0) > 0 else np.nan
        )
        df = df.dropna(subset=[rp]).copy()

    df["signed_rt"] = infer_signed_rt(df, cols["rt"], cols["response"])
    return df


def build_design_bank(
    df: pd.DataFrame,
    subject_col: str = DEFAULT_COLUMNS["subject"],
    pain_col: str = DEFAULT_COLUMNS["pain"],
    money_col: str = DEFAULT_COLUMNS["money"],
) -> List[np.ndarray]:
    """Create a bank of real subject design matrices with columns [pain, money]."""
    bank: List[np.ndarray] = []
    for _, sub_df in df.groupby(subject_col):
        arr = sub_df[[pain_col, money_col]].to_numpy(dtype=np.float32)
        if arr.shape[0] >= 10:
            bank.append(arr)
    if not bank:
        raise ValueError("Design bank is empty. Check filtering / column names.")
    return bank


def build_observed_datasets(
    df: pd.DataFrame,
    subject_col: str = DEFAULT_COLUMNS["subject"],
    pain_col: str = DEFAULT_COLUMNS["pain"],
    money_col: str = DEFAULT_COLUMNS["money"],
    rp_col: str = DEFAULT_COLUMNS["rp"],
) -> Dict[str, np.ndarray]:
    """Return one observed dataset per subject with columns [signed_rt, rp, pain, money]."""
    out: Dict[str, np.ndarray] = {}
    for subject, sub_df in df.groupby(subject_col):
        arr = sub_df[["signed_rt", rp_col, pain_col, money_col]].to_numpy(dtype=np.float32)
        if arr.shape[0] >= 10:
            out[str(subject)] = arr
    if not out:
        raise ValueError("No observed subject datasets available after filtering.")
    return out


def save_metadata(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def posterior_samples_to_mean(samples: np.ndarray) -> np.ndarray:
    """BayesFlow sample output can vary by version. This normalizes to [n_sets, n_params]."""
    arr = np.asarray(samples)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D posterior sample array, got shape {arr.shape}")

    # Common BayesFlow case: [n_draws, n_sets, n_params]
    if arr.shape[0] > 1 and arr.shape[2] <= 64:
        return arr.mean(axis=0)

    # Backup: [n_sets, n_draws, n_params]
    return arr.mean(axis=1)


def posterior_summary(samples: np.ndarray, param_names: List[str]) -> pd.DataFrame:
    """Return a summary table from posterior draws of shape [n_draws, n_params] or [n_params, n_draws]."""
    arr = np.asarray(samples)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D samples, got shape {arr.shape}")

    if arr.shape[0] == len(param_names) and arr.shape[1] != len(param_names):
        arr = arr.T
    elif arr.shape[1] != len(param_names):
        raise ValueError(f"Could not align sample array {arr.shape} with {len(param_names)} parameters")

    rows = []
    for i, name in enumerate(param_names):
        x = arr[:, i]
        rows.append(
            {
                "parameter": name,
                "mean": float(np.mean(x)),
                "sd": float(np.std(x, ddof=1)),
                "median": float(np.median(x)),
                "q2.5": float(np.quantile(x, 0.025)),
                "q97.5": float(np.quantile(x, 0.975)),
            }
        )
    return pd.DataFrame(rows)
