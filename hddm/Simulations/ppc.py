"""Posterior predictive checks for PainReward HDDM models.

Loads all chains for a given model version, combines them, generates PPC
samples, and saves RT/accuracy plots (overall and per condition).

Usage:
    python ppc.py --version 9 [--n-chains 4] [--samples 500]

Env vars: PROJECT_DIR, MODEL_DIR
"""
import argparse
import gc
import os
import sys
from pathlib import Path

import hddm
import kabuki
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model_specs import MODEL_BASE_NAME

PROJECT_DIR = Path(os.getenv("PROJECT_DIR", str(Path(__file__).resolve().parent.parent.parent))).resolve()
MODEL_DIR   = Path(os.getenv("MODEL_DIR",   str(PROJECT_DIR / "derivatives" / "hddm" / "models"))).resolve()
FIG_DIR     = PROJECT_DIR / "derivatives" / "hddm" / "figures"


def load_and_combine(version: int, n_chains: int, model_dir: Path):
    models = []
    for i in range(n_chains):
        path = model_dir / f"{MODEL_BASE_NAME}mod_{version}_{i}.hddm"
        if not path.exists():
            raise FileNotFoundError(f"Chain not found: {path}")
        print(f"  Loading {path.name}")
        models.append(hddm.load(str(path)))
    if len(models) == 1:
        return models[0]
    return kabuki.utils.concat_models(models)


def _rt_bins(rt: np.ndarray, n: int = 50) -> np.ndarray:
    lo = np.percentile(rt[np.isfinite(rt)], 1)
    hi = np.percentile(rt[np.isfinite(rt)], 99)
    return np.linspace(lo, hi, n + 1)


def plot_rt_marginal(obs_rt, sim_rt, model_name: str, out_dir: Path) -> None:
    obs = obs_rt[np.isfinite(obs_rt)]
    bins = _rt_bins(obs)
    sim_clip = sim_rt[(sim_rt >= bins[0]) & (sim_rt <= bins[-1])]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(obs,      bins=bins, density=True, alpha=0.7, color="steelblue", label="Observed")
    ax.hist(sim_clip, bins=bins, density=True, alpha=0.7, color="tomato",    label="PPC")
    ax.set_xlabel("RT (s)")
    ax.set_ylabel("Density")
    ax.set_title(f"PPC – RT | {model_name}")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = out_dir / f"ppc_rt_{model_name}.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out.name}")


def plot_rt_by_condition(data: pd.DataFrame, ppc_data: pd.DataFrame,
                         cond_col: str, model_name: str, out_dir: Path) -> None:
    if cond_col not in data.columns:
        return
    levels = sorted(data[cond_col].dropna().unique())
    fig, axes = plt.subplots(1, len(levels), figsize=(4 * len(levels), 4), sharey=True)
    if len(levels) == 1:
        axes = [axes]

    for ax, level in zip(axes, levels):
        obs = data.loc[data[cond_col] == level, "rt"].dropna().values
        if cond_col in ppc_data.columns:
            sim = ppc_data.loc[ppc_data[cond_col] == level, "rt_sampled"].dropna().values
        else:
            sim = ppc_data["rt_sampled"].dropna().values

        bins = _rt_bins(obs, n=30)
        sim_clip = sim[(sim >= bins[0]) & (sim <= bins[-1])]
        ax.hist(obs,      bins=bins, density=True, alpha=0.7, color="steelblue", label="Observed")
        ax.hist(sim_clip, bins=bins, density=True, alpha=0.7, color="tomato",    label="PPC")
        ax.set_title(f"{cond_col}={level}")
        ax.set_xlabel("RT (s)")
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("Density")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"PPC by {cond_col} | {model_name}")
    fig.tight_layout()
    out = out_dir / f"ppc_rt_by_{cond_col}_{model_name}.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out.name}")


def plot_accuracy(data: pd.DataFrame, ppc_data: pd.DataFrame,
                  model_name: str, out_dir: Path) -> None:
    obs_acc = data["response"].mean()
    sim_acc = ppc_data["response_sampled"].mean()

    fig, ax = plt.subplots(figsize=(3, 4))
    ax.bar(["Observed", "PPC"], [obs_acc, sim_acc],
           color=["steelblue", "tomato"], width=0.5)
    ax.set_ylim(0, 1)
    ax.set_ylabel("P(accept)")
    ax.set_title(f"PPC – Accuracy | {model_name}")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = out_dir / f"ppc_accuracy_{model_name}.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="PPC for PainReward HDDM models")
    parser.add_argument("--version",   type=int, required=True)
    parser.add_argument("--n-chains",  type=int, default=4)
    parser.add_argument("--samples",   type=int, default=500,
                        help="PPC samples per node (500 is usually sufficient)")
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    args = parser.parse_args()

    model_name = f"{MODEL_BASE_NAME}mod_{args.version}"
    out_dir = FIG_DIR / f"ppc_v{args.version}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model v{args.version} ({args.n_chains} chains)...")
    combined = load_and_combine(args.version, args.n_chains, args.model_dir)

    print(f"Generating PPC ({args.samples} samples per node)...")
    ppc_data = hddm.utils.post_pred_gen(combined, samples=args.samples, append_data=True)

    # Summary stats
    ppc_data2 = hddm.utils.post_pred_gen(combined, samples=args.samples)
    ppc_stats  = hddm.utils.post_pred_stats(combined.data, ppc_data2)
    stats_path = out_dir / f"ppc_stats_{model_name}.csv"
    ppc_stats.to_csv(stats_path)
    print(f"  Stats saved to {stats_path.name}")

    plot_rt_marginal(combined.data["rt"].values, ppc_data["rt_sampled"].values,
                     model_name, out_dir)
    plot_accuracy(combined.data, ppc_data, model_name, out_dir)
    for cond in ["painlevel", "moneylevel"]:
        plot_rt_by_condition(combined.data, ppc_data, cond, model_name, out_dir)

    del combined, ppc_data, ppc_data2
    gc.collect()
    print("Done.")


if __name__ == "__main__":
    main()
