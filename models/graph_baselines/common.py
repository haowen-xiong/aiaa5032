from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from models.baselines.common import BaselineOutputConfig, validate_inputs


@dataclass(frozen=True)
class GraphOutputConfig:
    n_pred: int = 1
    direct_multi_step: bool = False

    @property
    def steps(self) -> int:
        if self.direct_multi_step and self.n_pred > 1:
            return int(self.n_pred)
        return 1


class NodeTimeEncoder(nn.Module):
    def __init__(self, time_steps: int, input_channels: int, hidden_dim: int, input_dropout: float = 0.0):
        super().__init__()
        self.time_steps = int(time_steps)
        self.input_channels = int(input_channels)
        self.hidden_dim = int(hidden_dim)
        self.dropout = nn.Dropout(float(input_dropout)) if input_dropout > 0 else nn.Identity()
        self.proj = nn.Linear(self.time_steps * self.input_channels, self.hidden_dim)
        self.act = nn.ReLU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size, time_steps, num_nodes, channels = validate_inputs(inputs, expected_channels=self.input_channels)
        if time_steps != self.time_steps:
            raise ValueError(f"Expected {self.time_steps} time steps, got {time_steps}")
        flat = inputs.permute(0, 2, 1, 3).reshape(batch_size, num_nodes, time_steps * channels)
        flat = self.dropout(flat)
        return self.act(self.proj(flat))


class GraphPredictionHead(nn.Module):
    def __init__(self, hidden_dim: int, output_cfg: BaselineOutputConfig):
        super().__init__()
        self.output_cfg = output_cfg
        self.proj = nn.Linear(int(hidden_dim), self.output_cfg.steps)

    def forward(self, node_repr: torch.Tensor) -> torch.Tensor:
        if node_repr.ndim != 3:
            raise ValueError(f"Expected node representations [B, N, H], got {tuple(node_repr.shape)}")
        outputs = self.proj(node_repr)
        if self.output_cfg.steps == 1:
            return outputs
        return outputs.permute(0, 2, 1).unsqueeze(-1)


class ResidualMLP(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        layers = [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        if dropout > 0:
            layers.append(nn.Dropout(float(dropout)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
