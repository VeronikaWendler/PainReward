from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import bayesflow as bf

from config import DEFAULT_TRAINING
from data_utils import build_design_bank, build_observed_datasets, load_and_prepare_data
from model_utils import (
    get_amortizer_from_trainer,
    load_prior_moments,
    make_configurator,
    make_trainer,
    prepare_amortizer_input,
    unstandardize_posterior_samples,
)
from simulator import batch_simulator, prior, set_design_bank, simulate_dataset_from_design


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Posterior predictive checks for the RP-drift BayesFlow model."
    )
    p.add_argument("--data", required=True)
    p.add_argument("--checkpoint-dir", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--n-posterior-draws", type=int, default=250)
    p.add_argument("--n-sim-draws", type=int, default=100)
    p.add_argument("--subject", default=None, help="Optional subject id; default = first subject")
    p.add_argument("--seed", type=int, default=123)
    return p.parse_args()


def _normalize_subject_samples(arr: np.ndarray) -> np.ndarray:
    """
    Return posterior samples as [n_draws, n_params].
    """
    arr = np.asarray(arr)

    if arr.ndim == 3 and arr.shape[0] == 1:
        # [1, n_draws, n_params]
        arr = arr[0, :, :]
    elif arr.ndim == 3 and arr.shape[1] == 1:
        # [n_draws, 1, n_params]
        arr = arr[:, 0, :]
    elif arr.ndim != 2:
        raise ValueError(f"Unexpected posterior shape: {arr.shape}")

    return np.asarray(arr, dtype=np.float32)


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = Path(args.checkpoint_dir)
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint_dir}")

    prior_norm_path = checkpoint_dir / "prior_norm.npz"
    if not prior_norm_path.exists():
        raise FileNotFoundError(
            f"Could not find prior normalization file at {prior_norm_path}."
        )

    # Load real data and rebuild design bank
    df = load_and_prepare_data(args.data)
    set_design_bank(build_design_bank(df))
    observed = build_observed_datasets(df)

    # Load prior normalization + configurator
    prior_mean, prior_std = load_prior_moments(prior_norm_path)
    configurator = make_configurator(prior_mean, prior_std)

    # Rebuild trainer/amortizer exactly like in training/recovery
    generative_model = bf.simulation.GenerativeModel(
        prior,
        batch_simulator,
        simulator_is_batched=True,
    )
    trainer = make_trainer(
        generative_model=generative_model,
        checkpoint_path=checkpoint_dir,
        configurator=configurator,
        input_dim=5,
    )
    amortizer = get_amortizer_from_trainer(trainer)

    subject = args.subject or sorted(observed.keys())[0]
    x = observed[subject].astype(np.float32)

    # Posterior in standardized parameter space
    posterior_std = amortizer.sample(
        prepare_amortizer_input(x),
        n_samples=args.n_posterior_draws,
    )
    posterior_std = np.asarray(posterior_std, dtype=np.float32)

    # Undo standardization
    posterior = unstandardize_posterior_samples(posterior_std, prior_mean, prior_std)
    posterior = _normalize_subject_samples(posterior)

    rng = np.random.default_rng(args.seed)
    draw_idx = rng.choice(
        posterior.shape[0],
        size=min(args.n_sim_draws, posterior.shape[0]),
        replace=False,
    )

    design = x[:, 3:5].astype(np.float32)
    sims = [
        simulate_dataset_from_design(posterior[i].astype(np.float32), design)
        for i in draw_idx
    ]

    obs_rt = x[:, 0]
    obs_choice = x[:, 1]
    obs_rp = x[:, 2]

    # RT
    plt.figure(figsize=(7, 4))
    plt.hist(obs_rt, bins=30, density=True, alpha=0.5, label="observed RT")
    for sim in sims[:20]:
        plt.hist(sim[:, 0], bins=30, density=True, histtype="step", alpha=0.15)
    plt.xlabel("RT")
    plt.ylabel("density")
    plt.title(f"Posterior predictive RT check: subject {subject}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / f"subject_{subject}_ppc_rt.png", dpi=160)
    plt.close()

    # RP
    plt.figure(figsize=(7, 4))
    plt.hist(obs_rp, bins=30, density=True, alpha=0.5, label="observed RP")
    for sim in sims[:20]:
        plt.hist(sim[:, 2], bins=30, density=True, histtype="step", alpha=0.15)
    plt.xlabel("RP")
    plt.ylabel("density")
    plt.title(f"Posterior predictive RP check: subject {subject}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / f"subject_{subject}_ppc_rp.png", dpi=160)
    plt.close()

    # Choice probability
    sim_choice_means = np.array([sim[:, 1].mean() for sim in sims], dtype=np.float32)
    obs_choice_mean = float(obs_choice.mean())

    plt.figure(figsize=(6, 4))
    plt.hist(sim_choice_means, bins=30, alpha=0.7, density=True, label="simulated")
    plt.axvline(obs_choice_mean, linestyle="--", label="observed mean choice")
    plt.xlabel("P(choice = 1)")
    plt.ylabel("density")
    plt.title(f"Posterior predictive choice check: subject {subject}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / f"subject_{subject}_ppc_choice.png", dpi=160)
    plt.close()

    print(f"Saved posterior predictive plots for subject {subject}")


if __name__ == "__main__":
    main()