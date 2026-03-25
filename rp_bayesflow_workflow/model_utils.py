from __future__ import annotations

from pathlib import Path

from bayesflow.amortizers import SingleModelAmortizer
from bayesflow.models import GenerativeModel
from bayesflow.networks import InvariantNetwork, InvertibleNetwork
from bayesflow.trainers import ParameterEstimationTrainer

from config import PARAM_NAMES


def make_amortizer() -> SingleModelAmortizer:
    summary_net = InvariantNetwork()
    inference_net = InvertibleNetwork({"n_params": len(PARAM_NAMES)})
    return SingleModelAmortizer(inference_net, summary_net)


def make_trainer(generative_model: GenerativeModel, checkpoint_path: str | Path) -> ParameterEstimationTrainer:
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    amortizer = make_amortizer()
    trainer = ParameterEstimationTrainer(
        network=amortizer,
        generative_model=generative_model,
        checkpoint_path=str(checkpoint_path),
    )
    return trainer
