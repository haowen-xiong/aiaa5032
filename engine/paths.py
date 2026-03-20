from dataclasses import dataclass
from pathlib import Path
import random

import numpy as np
import torch


@dataclass(frozen=True)
class ExperimentPaths:
    output_root: Path
    experiment_root: Path
    model_root: Path
    tensorboard_dir: Path


def build_experiment_paths(output_dir, exp_name, model_name):
    output_root = Path(output_dir)
    experiment_root = output_root / exp_name
    model_root = experiment_root / model_name
    tensorboard_dir = model_root / "tensorboard"

    for path in [output_root, experiment_root, model_root, tensorboard_dir]:
        path.mkdir(parents=True, exist_ok=True)

    return ExperimentPaths(
        output_root=output_root,
        experiment_root=experiment_root,
        model_root=model_root,
        tensorboard_dir=tensorboard_dir,
    )


def pick_device(device_arg):
    if device_arg != "auto":
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
