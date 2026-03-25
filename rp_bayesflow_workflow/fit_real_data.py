from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from bayesflow.models import GenerativeModel

from config import DEFAULT_TRAINING, PARAM_NAMES
from data_utils import (
    build_design_bank,
    build_observed_datasets,
    load_and_prepare_data,
    posterior_summary,
)
from model_utils import make_trainer
from simulator import batch_simulator, prior, set_design_bank


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fit real subject datasets with a trained RP–drift BayesFlow model.")
    p.add_argument("--data", required=True)
    p.add_argument("--checkpoint-dir", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--n-posterior-draws", type=int, default=DEFAULT_TRAINING["posterior_draws"])
    p.add_argument("--seed", type=int, default=123)
    return p.parse_args()


def _normalize_subject_samples(arr: np.ndarray) -> np.ndarray:
    """Return [n_draws, n_params] from BayesFlow subject-level posterior samples."""
    arr = np.asarray(arr)
    if arr.ndim == 3 and arr.shape[1] == 1:
        # [n_draws, 1, n_params]
        arr = arr[:, 0, :]
    elif arr.ndim == 3 and arr.shape[0] == 1:
        # [1, n_draws, n_params]
        arr = arr[0, :, :]
    elif arr.ndim != 2:
        raise ValueError(f"Unexpected posterior shape: {arr.shape}")
    return arr


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    sample_dir = outdir / "subject_posteriors"
    sample_dir.mkdir(parents=True, exist_ok=True)

    df = load_and_prepare_data(args.data)
    set_design_bank(build_design_bank(df))
    observed = build_observed_datasets(df)

    generative_model = GenerativeModel(prior, batch_simulator)
    trainer = make_trainer(generative_model, args.checkpoint_dir)
    amortizer = trainer.network

    all_rows = []
    for subject, x in observed.items():
        x = x.astype(np.float32)[None, :, :]
        post = amortizer.sample(x, n_samples=args.n_posterior_draws)
        post = _normalize_subject_samples(post)
        np.save(sample_dir / f"subject_{subject}_posterior.npy", post)

        summary = posterior_summary(post, PARAM_NAMES)
        summary.insert(0, "subject", subject)
        summary.to_csv(sample_dir / f"subject_{subject}_summary.csv", index=False)
        all_rows.append(summary)
        print(f"Finished subject {subject}")

    combined = pd.concat(all_rows, ignore_index=True)
    combined.to_csv(outdir / "all_subject_parameter_summaries.csv", index=False)
    print(f"Saved summaries to {outdir}")


if __name__ == "__main__":
    main()
