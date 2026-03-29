from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
    p.add_argument("--subject", default=None, help="Optional single subject id. If omitted, runs all subjects.")
    p.add_argument("--seed", type=int, default=123)
    return p.parse_args()


def _normalize_subject_samples(arr: np.ndarray) -> np.ndarray:
    """
    Return posterior samples as [n_draws, n_params].
    Handles:
      [n_draws, n_params]
      [1, n_draws, n_params]
      [n_draws, 1, n_params]
    """
    arr = np.asarray(arr, dtype=np.float32)

    if arr.ndim == 2:
        return arr
    if arr.ndim == 3 and arr.shape[0] == 1:
        return arr[0, :, :]
    if arr.ndim == 3 and arr.shape[1] == 1:
        return arr[:, 0, :]

    raise ValueError(f"Unexpected posterior shape: {arr.shape}")


def simulate_posterior_predictive_datasets(
    posterior_samples: np.ndarray,
    design: np.ndarray,
    n_sim_draws: int,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    n_available = posterior_samples.shape[0]
    n_use = min(n_sim_draws, n_available)
    idx = rng.choice(n_available, size=n_use, replace=False)

    sims = [
        simulate_dataset_from_design(
            posterior_samples[i].astype(np.float32),
            design.astype(np.float32),
        )
        for i in idx
    ]
    return sims


def save_subject_ppc_plots(
    subject: str,
    observed_x: np.ndarray,
    sims: list[np.ndarray],
    outdir: Path,
) -> dict:
    obs_rt = observed_x[:, 0]
    obs_choice = observed_x[:, 1]
    obs_rp = observed_x[:, 2]

    sim_mean_rt = np.array([sim[:, 0].mean() for sim in sims], dtype=np.float32)
    sim_mean_choice = np.array([sim[:, 1].mean() for sim in sims], dtype=np.float32)
    sim_mean_rp = np.array([sim[:, 2].mean() for sim in sims], dtype=np.float32)

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

    plt.figure(figsize=(7, 4))
    plt.hist(sim_mean_choice, bins=20, density=True, alpha=0.7, label="simulated")
    plt.axvline(obs_choice.mean(), linestyle="--", label="observed mean choice")
    plt.xlabel("P(choice = 1)")
    plt.ylabel("density")
    plt.title(f"Posterior predictive choice check: subject {subject}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / f"subject_{subject}_ppc_choice.png", dpi=160)
    plt.close()

    return {
        "subject": subject,
        "n_trials": int(observed_x.shape[0]),
        "obs_mean_rt": float(obs_rt.mean()),
        "pp_mean_rt": float(sim_mean_rt.mean()),
        "pp_sd_rt": float(sim_mean_rt.std(ddof=1) if len(sim_mean_rt) > 1 else 0.0),
        "obs_mean_choice": float(obs_choice.mean()),
        "pp_mean_choice": float(sim_mean_choice.mean()),
        "pp_sd_choice": float(sim_mean_choice.std(ddof=1) if len(sim_mean_choice) > 1 else 0.0),
        "obs_mean_rp": float(obs_rp.mean()),
        "pp_mean_rp": float(sim_mean_rp.mean()),
        "pp_sd_rp": float(sim_mean_rp.std(ddof=1) if len(sim_mean_rp) > 1 else 0.0),
    }


def save_combined_ppc_plots(
    observed_all: list[np.ndarray],
    simulated_all: list[np.ndarray],
    outdir: Path,
) -> None:
    obs_concat = np.concatenate(observed_all, axis=0)
    sim_concat = np.concatenate(simulated_all, axis=0)

    obs_rt = obs_concat[:, 0]
    obs_choice = obs_concat[:, 1]
    obs_rp = obs_concat[:, 2]

    sim_rt = sim_concat[:, 0]
    sim_choice = sim_concat[:, 1]
    sim_rp = sim_concat[:, 2]

    plt.figure(figsize=(7, 4))
    plt.hist(obs_rt, bins=40, density=True, alpha=0.5, label="observed RT")
    plt.hist(sim_rt, bins=40, density=True, histtype="step", linewidth=2.0, label="simulated RT")
    plt.xlabel("RT")
    plt.ylabel("density")
    plt.title("Posterior predictive RT check: all subjects")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "combined_ppc_rt.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.hist(obs_rp, bins=40, density=True, alpha=0.5, label="observed RP")
    plt.hist(sim_rp, bins=40, density=True, histtype="step", linewidth=2.0, label="simulated RP")
    plt.xlabel("RP")
    plt.ylabel("density")
    plt.title("Posterior predictive RP check: all subjects")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "combined_ppc_rp.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.hist(sim_choice, bins=30, density=True, alpha=0.7, label="simulated choice")
    plt.axvline(obs_choice.mean(), linestyle="--", label="observed mean choice")
    plt.xlabel("choice")
    plt.ylabel("density")
    plt.title("Posterior predictive choice check: all subjects")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "combined_ppc_choice.png", dpi=160)
    plt.close()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    subject_plot_dir = outdir / "subject_plots"
    subject_plot_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = Path(args.checkpoint_dir)
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint_dir}")

    prior_norm_path = checkpoint_dir / "prior_norm.npz"
    if not prior_norm_path.exists():
        raise FileNotFoundError(
            f"Could not find prior normalization file at {prior_norm_path}."
        )

    df = load_and_prepare_data(args.data)
    design_bank = build_design_bank(df)
    observed = build_observed_datasets(df)
    set_design_bank(design_bank)

    prior_mean, prior_std = load_prior_moments(prior_norm_path)
    configurator = make_configurator(prior_mean, prior_std)

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

    if args.subject is not None:
        subject_ids = [str(args.subject)]
        if subject_ids[0] not in observed:
            raise KeyError(f"Subject {args.subject} not found in observed datasets.")
    else:
        subject_ids = sorted(observed.keys())

    rows = []
    observed_all = []
    simulated_all = []

    for subject in subject_ids:
        x = observed[subject].astype(np.float32)
        design = x[:, 3:5].astype(np.float32)

        posterior_std = amortizer.sample(
            prepare_amortizer_input(x),
            n_samples=args.n_posterior_draws,
        )
        posterior_std = _normalize_subject_samples(posterior_std)
        posterior = unstandardize_posterior_samples(posterior_std, prior_mean, prior_std)

        sims = simulate_posterior_predictive_datasets(
            posterior_samples=posterior,
            design=design,
            n_sim_draws=args.n_sim_draws,
            rng=rng,
        )

        row = save_subject_ppc_plots(
            subject=subject,
            observed_x=x,
            sims=sims,
            outdir=subject_plot_dir,
        )
        rows.append(row)

        observed_all.append(x)
        simulated_all.extend(sims)

        print(f"Finished PPC for subject {subject}")

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(outdir / "ppc_summary_by_subject.csv", index=False)

    save_combined_ppc_plots(
        observed_all=observed_all,
        simulated_all=simulated_all,
        outdir=outdir,
    )

    print(f"Saved subject-wise PPC plots to {subject_plot_dir}")
    print(f"Saved combined PPC plots to {outdir}")
    print(f"Saved summary table to {outdir / 'ppc_summary_by_subject.csv'}")


if __name__ == "__main__":
    main()