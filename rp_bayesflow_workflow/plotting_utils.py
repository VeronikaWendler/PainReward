from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np


def plot_losses(losses: Sequence[float], out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4))
    plt.plot(np.asarray(losses))
    plt.xlabel("training step")
    plt.ylabel("loss")
    plt.title("BayesFlow training loss")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_true_vs_estimated(
    true_params: np.ndarray,
    est_params: np.ndarray,
    param_names: Iterable[str],
    out_path: str | Path,
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    true_params = np.asarray(true_params)
    est_params = np.asarray(est_params)
    names = list(param_names)

    n_params = len(names)
    n_cols = 3
    n_rows = int(np.ceil(n_params / n_cols))

    fig = plt.figure(figsize=(4.6 * n_cols, 3.8 * n_rows), tight_layout=True)
    for i, name in enumerate(names):
        ax = fig.add_subplot(n_rows, n_cols, i + 1)
        ax.scatter(true_params[:, i], est_params[:, i], alpha=0.5, s=14)
        lo = min(true_params[:, i].min(), est_params[:, i].min())
        hi = max(true_params[:, i].max(), est_params[:, i].max())
        ax.plot([lo, hi], [lo, hi])
        ax.set_title(name)
        ax.set_xlabel("true")
        ax.set_ylabel("posterior mean")
    plt.savefig(out_path, dpi=160)
    plt.close(fig)
