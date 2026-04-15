# hddm_prep.py — data loading and feature engineering for HDDM
#
# Input:  behavioural_sv_cleaned_final_3_with_rp.csv  (pre-merged behav + RP)
# Output: derivatives/behav/hddm_ready.csv
#
# Run: python hddm_prep.py
# Env vars: PROJECT_DIR (default: repo root), DATA_FILE (override input CSV path)

import os
import numpy as np
import pandas as pd
from pathlib import Path


# ---------------------------------------------------------------------------
# Column-computation functions (pure pandas — no hddm dependency)
# ---------------------------------------------------------------------------

def compute_acceptance_pair(df: pd.DataFrame) -> pd.Series:
    """I = equal money/pain, M = money dominant or slight pain, P = pain strongly dominant (diff>=2)."""
    money = df["moneylevel"]
    pain  = df["painlevel"]
    return pd.Series(
        np.select(
            [money == pain, (pain - money) >= 2],
            ["I",           "P"],
            default="M",
        ),
        index=df.index,
    )


def compute_ov_value(df: pd.DataFrame) -> pd.Series:
    """Overall value category based on money + pain sum."""
    s = df["moneylevel"] + df["painlevel"]
    return pd.Series(
        np.select(
            [s <= 5, s > 6],
            ["low_OV", "high_OV"],
            default="mid_OV",
        ),
        index=df.index,
    )


def compute_abs_value(df: pd.DataFrame) -> pd.Series:
    """Absolute difference category."""
    d = (df["moneylevel"] - df["painlevel"]).abs()
    return pd.Series(
        np.select(
            [d < 2, d > 2],
            ["low_abs", "high_abs"],
            default="mid_abs",
        ),
        index=df.index,
    )


def compute_ov_money_pain(df: pd.DataFrame) -> pd.Series:
    """Combined OV × dominant-dimension label."""
    money = df["moneylevel"]
    pain  = df["painlevel"]
    s     = money + pain
    return pd.Series(
        np.select(
            [
                (s > 5) & (money > pain),
                (s > 5) & (pain >= money),
                (s <= 5) & (money > pain),
            ],
            ["h_OV_h_money", "h_OV_h_pain", "low_OV_h_money"],
            default="low_OV_h_pain",
        ),
        index=df.index,
    )


def compute_abs_money_pain(df: pd.DataFrame) -> pd.Series:
    """Combined absolute-difference × dominant-dimension label."""
    money = df["moneylevel"]
    pain  = df["painlevel"]
    d     = (money - pain).abs()
    return pd.Series(
        np.select(
            [
                (d < 2)  & (money > pain),
                (d < 2)  & (pain >= money),
                (d > 2)  & (money > pain),
                (d > 2)  & (pain >= money),
            ],
            [
                "low_abs_h_money", "low_abs_h_pain",
                "high_abs_h_money", "high_abs_h_pain",
            ],
            default="mid_abs",
        ),
        index=df.index,
    )


def compute_z_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Add pain_z and money_z columns: per-subject z-score of painlevel/moneylevel.

    Subjects with zero variance in a predictor get NaN (not a crash).
    """
    df = df.copy()
    for col, z_col in [("painlevel", "pain_z"), ("moneylevel", "money_z")]:
        grp    = df.groupby("subj_idx")[col]
        means  = grp.transform("mean")
        stds   = grp.transform("std")
        df[z_col] = (df[col] - means) / stds   # NaN where std == 0
    return df


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def prepare_hddm_data(input_csv: Path, output_csv: Path) -> pd.DataFrame:
    """Load, clean, and feature-engineer the HDDM input dataframe.

    Raises FileNotFoundError if input_csv is missing (intentional — no graceful
    fallback; the upstream pipeline must have run first).
    """
    if not input_csv.exists():
        raise FileNotFoundError(
            f"Input CSV not found: {input_csv}\n"
            "Run the EEG RP preprocessing pipeline first to generate this file."
        )

    df = pd.read_csv(input_csv, sep=",")

    # Filter to decision phase only
    df = df[df["TaskName"] == "decision"].copy()
    if df.empty:
        raise ValueError("No decision-phase rows found in the input CSV.")

    # Rename for HDDM
    df["rt"]       = pd.to_numeric(df["choice_resp.rt"], errors="coerce")
    df["response"] = pd.to_numeric(df["accepted"],       errors="coerce")
    df["subj_idx"] = df["participant"]

    # Cast predictors
    for col in ["moneylevel", "painlevel"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # RT filter
    df = df[df["rt"] > 0.250].copy()

    # Derived categorical columns (vectorised — no iterrows)
    df["acceptance_pair"] = compute_acceptance_pair(df)
    df["OV_value"]        = compute_ov_value(df)
    df["Abs_value"]       = compute_abs_value(df)
    df["OV_Money_Pain"]   = compute_ov_money_pain(df)
    df["Abs_Money_Pain"]  = compute_abs_money_pain(df)

    # Per-subject z-scores
    df = compute_z_scores(df)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Wrote {len(df):,} rows → {output_csv}")
    return df


if __name__ == "__main__":
    PROJECT_DIR = Path(os.getenv("PROJECT_DIR", Path(__file__).parent.parent)).resolve()

    default_input = (
        PROJECT_DIR / "Hddm_Docker_August_24" / "data_sets"
        / "behavioural_sv_cleaned_final_3_with_rp.csv"
    )
    input_csv  = Path(os.getenv("DATA_FILE", str(default_input)))
    output_csv = PROJECT_DIR / "derivatives" / "behav" / "hddm_ready.csv"

    prepare_hddm_data(input_csv, output_csv)
