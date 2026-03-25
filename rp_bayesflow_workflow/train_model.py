from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from bayesflow.models import GenerativeModel

from config import DEFAULT_TRAINING, PARAM_NAMES
from data_utils import build_design_bank, load_and_prepare_data, save_metadata
from model_utils import make_trainer
from plotting_utils import plot_losses
from simulator import batch_simulator, prior, set_design_bank


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the true integrative RP–drift BayesFlow model.")
    p.add_argument("--data", required=True, help="CSV with subj_idx, pain_z, money_z, rp_z, rt, response")
    p.add_argument("--outdir", required=True, help="Output directory for checkpoints and logs")
    p.add_argument("--epochs", type=int, default=DEFAULT_TRAINING["epochs"])
    p.add_argument("--batch-size", type=int, default=DEFAULT_TRAINING["batch_size"])
    p.add_argument("--iterations-per-epoch", type=int, default=DEFAULT_TRAINING["iterations_per_epoch"])
    p.add_argument("--capacity", type=int, default=DEFAULT_TRAINING["capacity"])
    p.add_argument("--n-trials-min", type=int, default=DEFAULT_TRAINING["n_trials_min"])
    p.add_argument("--n-trials-max", type=int, default=DEFAULT_TRAINING["n_trials_max"])
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--min-rt", type=float, default=0.25)
    p.add_argument("--zscore-rp-within-subject", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = outdir / "checkpoints"

    df = load_and_prepare_data(
        csv_path=args.data,
        min_rt=args.min_rt,
        zscore_rp_within_subject=args.zscore_rp_within_subject,
    )
    design_bank = build_design_bank(df)
    set_design_bank(design_bank)

    generative_model = GenerativeModel(prior, batch_simulator)
    trainer = make_trainer(generative_model, checkpoint_dir)

    def prior_N(n_min: int = args.n_trials_min, n_max: int = args.n_trials_max) -> int:
        return np.random.randint(n_min, n_max + 1)

    print(f"Training on {len(design_bank)} real subject design matrices")
    print(f"Parameter set: {PARAM_NAMES}")
    print(f"Checkpoints: {checkpoint_dir}")

    losses = trainer.train_experience_replay(
        epochs=args.epochs,
        batch_size=args.batch_size,
        iterations_per_epoch=args.iterations_per_epoch,
        capacity=args.capacity,
        n_obs=prior_N,
    )

    losses_arr = np.asarray(losses)
    np.save(outdir / "losses.npy", losses_arr)
    plot_losses(losses_arr, outdir / "loss_curve.png")

    save_metadata(
        outdir / "training_config.json",
        {
            "data": str(args.data),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "iterations_per_epoch": args.iterations_per_epoch,
            "capacity": args.capacity,
            "n_trials_min": args.n_trials_min,
            "n_trials_max": args.n_trials_max,
            "seed": args.seed,
            "n_subject_designs": len(design_bank),
            "param_names": PARAM_NAMES,
        },
    )
    print("Training finished.")


if __name__ == "__main__":
    main()
