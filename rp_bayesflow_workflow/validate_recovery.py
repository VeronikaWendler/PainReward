from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import bayesflow as bf

from config import DEFAULT_TRAINING, PARAM_NAMES
from data_utils import (
    build_design_bank,
    load_and_prepare_data,
    save_metadata,
)
from model_utils import make_trainer
from plotting_utils import plot_true_vs_estimated
from simulator import batch_simulator, prior, set_design_bank


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run parameter recovery for the RP-drift integrative BayesFlow model."
    )
    p.add_argument("--data", required=True)
    p.add_argument("--checkpoint-dir", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument(
        "--n-param-sets",
        type=int,
        default=DEFAULT_TRAINING["recovery_param_sets"],
    )
    p.add_argument("--n-trials", type=int, default=160)
    p.add_argument(
        "--n-posterior-draws",
        type=int,
        default=DEFAULT_TRAINING["posterior_draws"],
    )
    p.add_argument("--seed", type=int, default=123)
    return p.parse_args()


def get_amortizer_from_trainer(trainer):
    """Compatibility across BayesFlow trainer versions."""
    if hasattr(trainer, "amortizer"):
        return trainer.amortizer
    if hasattr(trainer, "network"):
        return trainer.network
    raise AttributeError("Trainer has neither `amortizer` nor `network`.")


def posterior_to_means(posterior: np.ndarray, n_param_sets: int) -> np.ndarray:
    """
    Convert posterior draws to posterior means.

    Handles either:
    - (n_param_sets, n_draws, n_params)
    - (n_draws, n_param_sets, n_params)
    """
    posterior = np.asarray(posterior)

    if posterior.ndim != 3:
        raise ValueError(
            f"Unexpected posterior ndim: {posterior.ndim}, shape={posterior.shape}"
        )

    if posterior.shape[0] == n_param_sets:
        # (n_param_sets, n_draws, n_params)
        est = posterior.mean(axis=1)
    elif posterior.shape[1] == n_param_sets:
        # (n_draws, n_param_sets, n_params)
        est = posterior.mean(axis=0)
    else:
        raise ValueError(
            f"Could not infer posterior layout from shape {posterior.shape} "
            f"with n_param_sets={n_param_sets}"
        )

    return np.asarray(est, dtype=np.float32)


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Rebuild same design bank used during training
    df = load_and_prepare_data(args.data)
    design_bank = build_design_bank(df)
    set_design_bank(design_bank)

    # Rebuild trained model and load checkpoint
    generative_model = bf.simulation.GenerativeModel(
        prior,
        batch_simulator,
        simulator_is_batched=True,
    )
    trainer = make_trainer(generative_model, args.checkpoint_dir)
    amortizer = get_amortizer_from_trainer(trainer)

    # Draw true parameters from prior
    true_params = np.stack(
        [prior() for _ in range(args.n_param_sets)],
        axis=0,
    ).astype(np.float32)

    # Simulate recovery datasets
    x = batch_simulator(true_params, args.n_trials).astype(np.float32)

    # IMPORTANT: this BayesFlow version expects a dict, not a raw ndarray
    posterior = amortizer.sample(
        {"summary_conditions": x},
        n_samples=args.n_posterior_draws,
    )
    posterior = np.asarray(posterior)

    print("true_params shape:", true_params.shape)
    print("x shape:", x.shape)
    print("posterior shape:", posterior.shape)

    # Posterior means
    est = posterior_to_means(posterior, args.n_param_sets)
    print("est shape:", est.shape)

    if est.shape != true_params.shape:
        raise ValueError(
            f"Shape mismatch: true_params {true_params.shape} vs est {est.shape}"
        )

    # Save outputs
    np.save(outdir / "true_params.npy", true_params)
    np.save(outdir / "posterior_samples.npy", posterior)
    np.save(outdir / "posterior_means.npy", est)

    plot_true_vs_estimated(
        true_params,
        est,
        PARAM_NAMES,
        outdir / "true_vs_estimated.png",
    )

    corr = {
        name: float(np.corrcoef(true_params[:, i], est[:, i])[0, 1])
        for i, name in enumerate(PARAM_NAMES)
    }

    save_metadata(
        outdir / "recovery_metrics.json",
        {
            "n_param_sets": args.n_param_sets,
            "n_trials": args.n_trials,
            "n_posterior_draws": args.n_posterior_draws,
            "correlation_by_parameter": corr,
        },
    )

    print(corr)
    print("Parameter recovery finished.")


if __name__ == "__main__":
    main()