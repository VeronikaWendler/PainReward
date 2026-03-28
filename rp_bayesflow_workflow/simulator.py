from __future__ import annotations

from typing import Iterable, List

import numpy as np
from numba import njit

from config import PRIOR_HIGH, PRIOR_LOW

DESIGN_BANK: List[np.ndarray] | None = None


def set_design_bank(bank: Iterable[np.ndarray]) -> None:
    global DESIGN_BANK
    DESIGN_BANK = [np.asarray(x, dtype=np.float32) for x in bank]
    if len(DESIGN_BANK) == 0:
        raise ValueError("Design bank is empty.")


def prior() -> np.ndarray:
    """Sample one parameter vector from the prior."""
    low = np.asarray(PRIOR_LOW, dtype=np.float32)
    high = np.asarray(PRIOR_HIGH, dtype=np.float32)
    p = np.random.uniform(low=low, high=high, size=(len(low),))
    return p.astype(np.float32)


@njit
def ddm_trial(
    drift: float,
    boundary: float,
    ndt: float,
    dc: float = 1.0,
    dt: float = 0.005,
):
    evidence = boundary * 0.5
    n_steps = 0.0

    while evidence > 0.0 and evidence < boundary:
        evidence += drift * dt + np.sqrt(dt) * dc * np.random.normal()
        n_steps += 1.0

    rt = n_steps * dt + ndt
    choice = 1.0 if evidence >= boundary else 0.0
    return rt, choice


@njit
def simulate_dataset_from_design(params: np.ndarray, design: np.ndarray, dt: float = 0.005) -> np.ndarray:
    """Simulate one full dataset with columns [signed_rt, rp, pain, money].

    Model:
      mu_drift_i = v_intercept + v_pain * pain_i + v_money * money_i
      delta_i    ~ Normal(mu_drift_i, drift_sd)
      signed_rt  ~ DDM(delta_i, boundary, ndt)
      rp_i       ~ Normal(rp_intercept + rp_loading * delta_i, rp_noise)
    """
    v_intercept, v_pain, v_money, boundary, ndt, drift_sd, rp_intercept, rp_loading, rp_noise = params

    n_trials = design.shape[0]
    out = np.empty((n_trials, 5), dtype=np.float32)

    for i in range(n_trials):
        pain_i = design[i, 0]
        money_i = design[i, 1]

        mu_drift = v_intercept + v_pain * pain_i + v_money * money_i
        latent_drift = np.random.normal(mu_drift, drift_sd)

        rt_i, choice_i = ddm_trial(
            drift=latent_drift,
            boundary=boundary,
            ndt=ndt,
            dt=dt,
        )
        rp_i = np.random.normal(rp_intercept + rp_loading * latent_drift, rp_noise)

        out[i, 0] = rt_i
        out[i, 1] = choice_i
        out[i, 2] = rp_i
        out[i, 3] = pain_i
        out[i, 4] = money_i

    return out


def _sample_design(n_obs: int) -> np.ndarray:
    if DESIGN_BANK is None:
        raise RuntimeError("DESIGN_BANK is None. Call set_design_bank(...) before training or recovery.")

    design = DESIGN_BANK[np.random.randint(0, len(DESIGN_BANK))]
    n_available = design.shape[0]
    replace = n_available < n_obs
    idx = np.random.choice(n_available, size=n_obs, replace=replace)
    return design[idx].astype(np.float32)


def batch_simulator(prior_samples: np.ndarray, n_obs: int = 160) -> np.ndarray:
    """Simulate one BayesFlow batch using randomly drawn real design rows.

    Output shape: [batch_size, n_obs, 4]
    """
    if isinstance(n_obs, (tuple, list, np.ndarray)):
        raise ValueError("n_obs must be a single integer per BayesFlow training step.")

    n_sim = prior_samples.shape[0]
    sim_data = np.empty((n_sim, int(n_obs), 4), dtype=np.float32)
    for i in range(n_sim):
        design = _sample_design(int(n_obs))
        sim_data[i] = simulate_dataset_from_design(prior_samples[i], design)
    return sim_data.astype(np.float32)