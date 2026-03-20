from dataclasses import dataclass
from datetime import datetime
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


def build_experiment_paths(output_dir, exp_name, model_name, overwrite=False):
    output_root = Path(output_dir)
    experiment_root = output_root / exp_name
    model_root = experiment_root / model_name
    if model_root.exists() and any(model_root.iterdir()) and not overwrite:
        suffix = datetime.now().strftime("%y%m%d_%H%M%S")
        experiment_root = output_root / f"{exp_name}_{suffix}"
        model_root = experiment_root / model_name
        print(
            f"Warning: experiment directory {output_root / exp_name / model_name} already exists and is not empty; "
            f"writing to {model_root} instead. Use --overwrite true to reuse the existing directory."
        )
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
