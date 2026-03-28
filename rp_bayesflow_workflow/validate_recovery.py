from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import bayesflow as bf

from config import DEFAULT_TRAINING, PARAM_NAMES
from data_utils import build_design_bank, load_and_prepare_data, posterior_samples_to_mean, save_metadata
from model_utils import make_trainer
from plotting_utils import plot_true_vs_estimated
from simulator import batch_simulator, prior, set_design_bank


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run parameter recovery for the RP–drift integrative BayesFlow model.")
    p.add_argument("--data", required=True)
    p.add_argument("--checkpoint-dir", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--n-param-sets", type=int, default=DEFAULT_TRAINING["recovery_param_sets"])
    p.add_argument("--n-trials", type=int, default=160)
    p.add_argument("--n-posterior-draws", type=int, default=DEFAULT_TRAINING["posterior_draws"])
    p.add_argument("--seed", type=int, default=123)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_and_prepare_data(args.data)
    set_design_bank(build_design_bank(df))

    generative_model = bf.simulation.GenerativeModel(
        prior,
        batch_simulator,
        simulator_is_batched=True,
    )
    trainer = make_trainer(generative_model, args.checkpoint_dir)
    amortizer = trainer.amortizer

    true_params = np.stack([prior() for _ in range(args.n_param_sets)], axis=0).astype(np.float32)
    x = batch_simulator(true_params, args.n_trials).astype(np.float32)
    posterior = amortizer.sample(
        {"summary_conditions": x.astype(np.float32)},
        n_samples=args.n_posterior_draws,
    )

    print("true_params shape:", true_params.shape)
    print("x shape:", x.shape)
    print("posterior shape:", posterior.shape)

    # Robust handling of BayesFlow posterior layout
    if posterior.ndim != 3:
        raise ValueError(f"Unexpected posterior ndim: {posterior.ndim}, shape={posterior.shape}")

    if posterior.shape[0] == args.n_param_sets:
        # shape: (n_param_sets, n_draws, n_params)
        est = posterior.mean(axis=1)
    elif posterior.shape[1] == args.n_param_sets:
        # shape: (n_draws, n_param_sets, n_params)
        est = posterior.mean(axis=0)
    else:
        raise ValueError(
            f"Could not infer posterior layout from shape {posterior.shape} "
            f"with n_param_sets={args.n_param_sets}"
        )

    print("est shape:", est.shape)


    np.save(outdir / "true_params.npy", true_params)
    np.save(outdir / "posterior_samples.npy", posterior)
    np.save(outdir / "posterior_means.npy", est)
    plot_true_vs_estimated(true_params, est, PARAM_NAMES, outdir / "true_vs_estimated.png")

    corr = {
        name: float(np.corrcoef(true_params[:, i], est[:, i])[0, 1])
        for i, name in enumerate(PARAM_NAMES)
    }
    save_metadata(outdir / "recovery_metrics.json", {"correlation_by_parameter": corr})
    print(corr)


if __name__ == "__main__":
    main()
