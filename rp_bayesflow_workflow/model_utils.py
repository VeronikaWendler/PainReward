from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import bayesflow as bf

from config import PARAM_NAMES


# -----------------------------
# Prior normalization utilities
# -----------------------------
def estimate_prior_moments(
    prior_fn: Callable[[], np.ndarray],
    n_draws: int = 50000,
    seed: int = 123,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate mean and SD of the prior by repeated sampling.

    Returns
    -------
    prior_mean : np.ndarray, shape (n_params,)
    prior_std  : np.ndarray, shape (n_params,)
    """
    rng_state = np.random.get_state()
    np.random.seed(seed)
    draws = np.stack([prior_fn() for _ in range(n_draws)], axis=0).astype(np.float32)
    np.random.set_state(rng_state)

    prior_mean = draws.mean(axis=0).astype(np.float32)
    prior_std = draws.std(axis=0).astype(np.float32)

    # guard against zero SD
    prior_std = np.where(prior_std < 1e-8, 1.0, prior_std).astype(np.float32)
    return prior_mean, prior_std


def save_prior_moments(path: str | Path, prior_mean: np.ndarray, prior_std: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, prior_mean=np.asarray(prior_mean, dtype=np.float32), prior_std=np.asarray(prior_std, dtype=np.float32))


def load_prior_moments(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    path = Path(path)
    arr = np.load(path)
    return arr["prior_mean"].astype(np.float32), arr["prior_std"].astype(np.float32)


# -----------------------------
# BayesFlow input formatting
# -----------------------------
def _get_key(d: dict, candidates: list[str]):
    for k in candidates:
        if k in d:
            return d[k]
    raise KeyError(f"Could not find any of keys {candidates} in forward_dict. Available keys: {list(d.keys())}")


def make_configurator(prior_mean: np.ndarray, prior_std: np.ndarray):
    """
    Returns a BayesFlow configurator function.

    The configurator:
    - standardizes parameters
    - passes trialwise data as summary_conditions
    - adds log(n_obs) as direct_conditions
    """
    prior_mean = np.asarray(prior_mean, dtype=np.float32)
    prior_std = np.asarray(prior_std, dtype=np.float32)

    def configurator(forward_dict: dict) -> dict:
        sim_data = _get_key(
            forward_dict,
            ["sim_data", "simulations", "simulated_data", "x"],
        )
        prior_draws = _get_key(
            forward_dict,
            ["prior_draws", "parameters", "theta"],
        )

        sim_data = np.asarray(sim_data, dtype=np.float32)
        prior_draws = np.asarray(prior_draws, dtype=np.float32)

        if sim_data.ndim != 3:
            raise ValueError(f"Expected sim_data to have shape [batch, n_obs, n_features], got {sim_data.shape}")

        if prior_draws.ndim != 2:
            raise ValueError(f"Expected prior_draws to have shape [batch, n_params], got {prior_draws.shape}")

        batch_size = sim_data.shape[0]
        n_obs = sim_data.shape[1]

        standardized_params = ((prior_draws - prior_mean) / prior_std).astype(np.float32)
        direct_conditions = np.full((batch_size, 1), np.log(float(n_obs)), dtype=np.float32)

        return {
            "summary_conditions": sim_data,
            "direct_conditions": direct_conditions,
            "parameters": standardized_params,
        }

    return configurator


def prepare_amortizer_input(x: np.ndarray) -> dict:
    """
    Prepare observed or simulated data for amortizer.sample(...).

    Parameters
    ----------
    x : np.ndarray
        Shape [n_sets, n_obs, n_features] or [n_obs, n_features]

    Returns
    -------
    dict with summary_conditions and direct_conditions
    """
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 2:
        x = x[None, :, :]
    if x.ndim != 3:
        raise ValueError(f"Expected x to have 2 or 3 dims, got shape {x.shape}")

    batch_size = x.shape[0]
    n_obs = x.shape[1]
    direct_conditions = np.full((batch_size, 1), np.log(float(n_obs)), dtype=np.float32)

    return {
        "summary_conditions": x.astype(np.float32),
        "direct_conditions": direct_conditions,
    }


def unstandardize_posterior_samples(
    samples: np.ndarray,
    prior_mean: np.ndarray,
    prior_std: np.ndarray,
) -> np.ndarray:
    """
    Undo parameter standardization for posterior samples.

    Accepts:
    - 2D: [n_draws, n_params]
    - 3D: [n_sets, n_draws, n_params] or [n_draws, n_sets, n_params]

    Returns array of same shape as input.
    """
    arr = np.asarray(samples, dtype=np.float32)
    prior_mean = np.asarray(prior_mean, dtype=np.float32)
    prior_std = np.asarray(prior_std, dtype=np.float32)

    if arr.ndim == 2:
        if arr.shape[-1] != prior_mean.shape[0]:
            raise ValueError(
                f"Posterior last dimension {arr.shape[-1]} does not match "
                f"number of parameters {prior_mean.shape[0]}"
            )
        return arr * prior_std[None, :] + prior_mean[None, :]

    if arr.ndim == 3:
        if arr.shape[-1] != prior_mean.shape[0]:
            raise ValueError(
                f"Posterior last dimension {arr.shape[-1]} does not match "
                f"number of parameters {prior_mean.shape[0]}"
            )
        return arr * prior_std[None, None, :] + prior_mean[None, None, :]

    raise ValueError(f"Expected 2D or 3D posterior sample array, got shape {arr.shape}")


def get_amortizer_from_trainer(trainer):
    if hasattr(trainer, "amortizer"):
        return trainer.amortizer
    if hasattr(trainer, "network"):
        return trainer.network
    raise AttributeError("Trainer has neither `amortizer` nor `network`.")


# -----------------------------
# Network builders
# -----------------------------
def make_summary_net(input_dim: int = 5, summary_dim: int = 64):
    """
    Prefer SetTransformer when available, otherwise fall back to InvariantNetwork.
    """
    if not hasattr(bf, "networks"):
        raise RuntimeError("Your BayesFlow installation has no `bf.networks` module.")

    # Preferred: SetTransformer
    if hasattr(bf.networks, "SetTransformer"):
        constructors = [
            lambda: bf.networks.SetTransformer(input_dim=input_dim, summary_dim=summary_dim),
            lambda: bf.networks.SetTransformer(num_features=input_dim, summary_dim=summary_dim),
            lambda: bf.networks.SetTransformer(input_dim=input_dim),
            lambda: bf.networks.SetTransformer(num_features=input_dim),
            lambda: bf.networks.SetTransformer(),
        ]
        for ctor in constructors:
            try:
                return ctor()
            except Exception:
                pass

    # Fallback: InvariantNetwork
    if hasattr(bf.networks, "InvariantNetwork"):
        constructors = [
            lambda: bf.networks.InvariantNetwork(input_dim=input_dim, summary_dim=summary_dim),
            lambda: bf.networks.InvariantNetwork(num_features=input_dim, summary_dim=summary_dim),
            lambda: bf.networks.InvariantNetwork(),
        ]
        for ctor in constructors:
            try:
                return ctor()
            except Exception:
                pass

    raise RuntimeError(
        "Could not construct a compatible summary network. "
        "Expected `SetTransformer` or `InvariantNetwork`."
    )


def make_inference_net(n_params: int):
    if not hasattr(bf, "networks"):
        raise RuntimeError("Your BayesFlow installation has no `bf.networks` module.")

    if hasattr(bf.networks, "InvertibleNetwork"):
        constructors = [
            lambda: bf.networks.InvertibleNetwork({"n_params": n_params}),
            lambda: bf.networks.InvertibleNetwork(num_params=n_params),
            lambda: bf.networks.InvertibleNetwork(n_params=n_params),
        ]
        for ctor in constructors:
            try:
                return ctor()
            except Exception:
                pass

    raise RuntimeError("Could not construct a compatible `InvertibleNetwork`.")


def make_amortizer(input_dim: int = 5):
    if not hasattr(bf, "amortizers"):
        raise RuntimeError("Your BayesFlow installation has no `bf.amortizers` module.")

    summary_net = make_summary_net(input_dim=input_dim, summary_dim=64)
    inference_net = make_inference_net(len(PARAM_NAMES))

    if hasattr(bf.amortizers, "SingleModelAmortizer"):
        return bf.amortizers.SingleModelAmortizer(inference_net, summary_net)

    if hasattr(bf.amortizers, "AmortizedPosterior"):
        return bf.amortizers.AmortizedPosterior(inference_net, summary_net)

    raise RuntimeError(
        "Could not find a compatible amortizer class. "
        "Expected `SingleModelAmortizer` or `AmortizedPosterior`."
    )


# -----------------------------
# Trainer builder
# -----------------------------
def make_trainer(
    generative_model,
    checkpoint_path: str | Path,
    configurator=None,
    input_dim: int = 5,
):
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    amortizer = make_amortizer(input_dim=input_dim)

    if not hasattr(bf, "trainers"):
        raise RuntimeError("Your BayesFlow installation has no `bf.trainers` module.")

    if hasattr(bf.trainers, "ParameterEstimationTrainer"):
        # try with configurator first
        try:
            return bf.trainers.ParameterEstimationTrainer(
                network=amortizer,
                generative_model=generative_model,
                configurator=configurator,
                checkpoint_path=str(checkpoint_path),
            )
        except TypeError:
            return bf.trainers.ParameterEstimationTrainer(
                network=amortizer,
                generative_model=generative_model,
                checkpoint_path=str(checkpoint_path),
            )

    if hasattr(bf.trainers, "Trainer"):
        try:
            return bf.trainers.Trainer(
                amortizer=amortizer,
                generative_model=generative_model,
                configurator=configurator,
                checkpoint_path=str(checkpoint_path),
            )
        except TypeError:
            return bf.trainers.Trainer(
                amortizer=amortizer,
                generative_model=generative_model,
                checkpoint_path=str(checkpoint_path),
            )

    raise RuntimeError(
        "Could not find a compatible trainer class. "
        "Expected `ParameterEstimationTrainer` or `Trainer`."
    )