from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn

from .common import BaselineOutputConfig, flatten_spatiotemporal, reshape_node_outputs


class TemporalMLPBaseline(nn.Module):
    def __init__(
        self,
        time_steps: int,
        input_channels: int = 1,
        hidden_dims: Iterable[int] = (128, 64),
        n_pred: int = 1,
        direct_multi_step: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.output_cfg = BaselineOutputConfig(n_pred=n_pred, direct_multi_step=direct_multi_step)
        self.time_steps = int(time_steps)
        self.input_channels = int(input_channels)

        layers: list[nn.Module] = []
        in_dim = self.time_steps * self.input_channels
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        self.backbone = nn.Sequential(*layers) if layers else nn.Identity()
        self.head = nn.Linear(in_dim, self.output_cfg.steps)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        flat, batch_size, num_nodes = flatten_spatiotemporal(inputs)
        if flat.shape[-1] != self.time_steps * self.input_channels:
            raise ValueError(
                "TemporalMLPBaseline received incompatible input shape: "
                f"expected T*C={self.time_steps * self.input_channels}, got {flat.shape[-1]}"
            )
        hidden = self.backbone(flat)
        outputs = self.head(hidden)
        return reshape_node_outputs(outputs, batch_size, num_nodes, self.output_cfg.steps)
