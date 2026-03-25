from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from bayesflow.models import GenerativeModel

from config import DEFAULT_TRAINING, PARAM_NAMES
from data_utils import build_design_bank, build_observed_datasets, load_and_prepare_data
from model_utils import make_trainer
from simulator import batch_simulator, prior, set_design_bank, simulate_dataset_from_design


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Posterior predictive checks for the RP–drift BayesFlow model.")
    p.add_argument("--data", required=True)
    p.add_argument("--checkpoint-dir", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--n-posterior-draws", type=int, default=250)
    p.add_argument("--n-sim-draws", type=int, default=100)
    p.add_argument("--subject", default=None, help="Optional subject id; default = first subject in the file")
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


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_and_prepare_data(args.data)
    set_design_bank(build_design_bank(df))
    observed = build_observed_datasets(df)

    generative_model = GenerativeModel(prior, batch_simulator)
    trainer = make_trainer(generative_model, args.checkpoint_dir)
    amortizer = trainer.network

    subject = args.subject or sorted(observed.keys())[0]
    x = observed[subject].astype(np.float32)
    samples = amortizer.sample(x[None, :, :], n_samples=args.n_posterior_draws)
    samples = _normalize_subject_samples(samples)

    # simulate a few posterior predictive datasets
    rng = np.random.default_rng(123)
    idx = rng.choice(samples.shape[0], size=min(args.n_sim_draws, samples.shape[0]), replace=False)
    sims = [simulate_dataset_from_design(samples[i].astype(np.float32), x[:, 2:4].astype(np.float32)) for i in idx]

    obs_rt = np.abs(x[:, 0])
    obs_rp = x[:, 1]

    plt.figure(figsize=(7, 4))
    plt.hist(obs_rt, bins=30, density=True, alpha=0.5, label="observed RT")
    for sim in sims[:20]:
        plt.hist(np.abs(sim[:, 0]), bins=30, density=True, histtype="step", alpha=0.15)
    plt.xlabel("RT (absolute value)")
    plt.ylabel("density")
    plt.title(f"Posterior predictive RT check: subject {subject}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / f"subject_{subject}_ppc_rt.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.hist(obs_rp, bins=30, density=True, alpha=0.5, label="observed RP")
    for sim in sims[:20]:
        plt.hist(sim[:, 1], bins=30, density=True, histtype="step", alpha=0.15)
    plt.xlabel("RP")
    plt.ylabel("density")
    plt.title(f"Posterior predictive RP check: subject {subject}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / f"subject_{subject}_ppc_rp.png", dpi=160)
    plt.close()

    print(f"Saved posterior predictive plots for subject {subject}")


if __name__ == "__main__":
    main()
