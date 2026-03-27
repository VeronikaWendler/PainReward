from __future__ import annotations

from pathlib import Path

import bayesflow as bf

from config import PARAM_NAMES


def make_amortizer():
    """
    Compatibility wrapper for BayesFlow versions.

    Tries:
    1) SingleModelAmortizer + InvariantNetwork + InvertibleNetwork
    2) AmortizedPosterior + InvariantNetwork + InvertibleNetwork
    """
    if not hasattr(bf, "networks"):
        raise RuntimeError("Your BayesFlow installation has no `bf.networks` module.")

    if not hasattr(bf, "amortizers"):
        raise RuntimeError("Your BayesFlow installation has no `bf.amortizers` module.")

    # Networks
    if hasattr(bf.networks, "InvariantNetwork"):
        summary_net = bf.networks.InvariantNetwork()
    else:
        raise RuntimeError(
            "Could not find `InvariantNetwork` in `bf.networks`. "
            "Please inspect `dir(bf.networks)`."
        )

    # Different versions use slightly different constructor signatures
    if hasattr(bf.networks, "InvertibleNetwork"):
        try:
            inference_net = bf.networks.InvertibleNetwork({"n_params": len(PARAM_NAMES)})
        except Exception:
            inference_net = bf.networks.InvertibleNetwork(num_params=len(PARAM_NAMES))
    else:
        raise RuntimeError(
            "Could not find `InvertibleNetwork` in `bf.networks`. "
            "Please inspect `dir(bf.networks)`."
        )

    # Amortizer
    if hasattr(bf.amortizers, "SingleModelAmortizer"):
        return bf.amortizers.SingleModelAmortizer(inference_net, summary_net)

    if hasattr(bf.amortizers, "AmortizedPosterior"):
        return bf.amortizers.AmortizedPosterior(inference_net, summary_net)

    raise RuntimeError(
        "Could not find a compatible amortizer class. "
        "Expected `SingleModelAmortizer` or `AmortizedPosterior`."
    )


def make_trainer(generative_model, checkpoint_path: str | Path):
    """
    Compatibility wrapper for BayesFlow trainer versions.

    Tries:
    1) ParameterEstimationTrainer(network=..., generative_model=..., checkpoint_path=...)
    2) Trainer(amortizer=..., generative_model=..., checkpoint_path=...)
    """
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    amortizer = make_amortizer()

    if not hasattr(bf, "trainers"):
        raise RuntimeError("Your BayesFlow installation has no `bf.trainers` module.")

    if hasattr(bf.trainers, "ParameterEstimationTrainer"):
        return bf.trainers.ParameterEstimationTrainer(
            network=amortizer,
            generative_model=generative_model,
            checkpoint_path=str(checkpoint_path),
        )

    if hasattr(bf.trainers, "Trainer"):
        return bf.trainers.Trainer(
            amortizer=amortizer,
            generative_model=generative_model,
            checkpoint_path=str(checkpoint_path),
        )

    raise RuntimeError(
        "Could not find a compatible trainer class. "
        "Expected `ParameterEstimationTrainer` or `Trainer`."
    )