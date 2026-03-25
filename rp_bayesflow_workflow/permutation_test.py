from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from bayesflow.models import GenerativeModel

from config import DEFAULT_COLUMNS, DEFAULT_TRAINING, PARAM_NAMES
from data_utils import build_design_bank, build_observed_datasets, load_and_prepare_data, posterior_summary
from model_utils import make_trainer
from simulator import batch_simulator, prior, set_design_bank


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Permutation check for the RP–drift integrative model.")
    p.add_argument("--data", required=True)
    p.add_argument("--checkpoint-dir", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--n-posterior-draws", type=int, default=500)
    p.add_argument("--seed", type=int, default=123)
    return p.parse_args()


def _normalize_subject_samples(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 3 and arr.shape[1] == 1:
        arr = arr[:, 0, :]
    elif arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0, :, :]
    elif arr.ndim != 2:
        raise ValueError(f"Unexpected posterior shape: {arr.shape}")
    return arr


def permute_rp_within_subject(df: pd.DataFrame, subject_col: str, rp_col: str, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = df.copy()
    pieces = []
    for _, sub_df in out.groupby(subject_col):
        sub_df = sub_df.copy()
        sub_df[rp_col] = rng.permutation(sub_df[rp_col].to_numpy())
        sub_df["signed_rt"] = sub_df["signed_rt"]  # explicit no-op for readability
        pieces.append(sub_df)
    return pd.concat(pieces, ignore_index=True)


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_and_prepare_data(args.data)
    set_design_bank(build_design_bank(df))

    generative_model = GenerativeModel(prior, batch_simulator)
    trainer = make_trainer(generative_model, args.checkpoint_dir)
    amortizer = trainer.network

    observed = build_observed_datasets(df)
    perm_df = permute_rp_within_subject(df, DEFAULT_COLUMNS["subject"], DEFAULT_COLUMNS["rp"], seed=args.seed)
    permuted = build_observed_datasets(perm_df)

    rows = []
    for subject in sorted(observed.keys()):
        x_orig = observed[subject].astype(np.float32)[None, :, :]
        x_perm = permuted[subject].astype(np.float32)[None, :, :]

        s_orig = _normalize_subject_samples(amortizer.sample(x_orig, n_samples=args.n_posterior_draws))
        s_perm = _normalize_subject_samples(amortizer.sample(x_perm, n_samples=args.n_posterior_draws))

        summ_orig = posterior_summary(s_orig, PARAM_NAMES).set_index("parameter")
        summ_perm = posterior_summary(s_perm, PARAM_NAMES).set_index("parameter")

        # Focus on the integrative linkage parameters.
        for param in ["v_pain", "v_money", "drift_sd", "rp_loading", "rp_noise"]:
            rows.append(
                {
                    "subject": subject,
                    "parameter": param,
                    "orig_mean": float(summ_orig.loc[param, "mean"]),
                    "perm_mean": float(summ_perm.loc[param, "mean"]),
                    "abs_shift": float(abs(summ_orig.loc[param, "mean"] - summ_perm.loc[param, "mean"])),
                }
            )
        print(f"Permutation done for subject {subject}")

    pd.DataFrame(rows).to_csv(outdir / "permutation_parameter_shifts.csv", index=False)
    print("Saved permutation summary.")


if __name__ == "__main__":
    main()
