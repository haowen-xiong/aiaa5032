from __future__ import annotations

import torch
import torch.nn as nn

from models.baselines.common import BaselineOutputConfig
from .common import GraphPredictionHead, NodeTimeEncoder, ResidualMLP


class GraphSAGELayer(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.0, residual: bool = True, normalize_embeddings: bool = False):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.residual = bool(residual)
        self.normalize_embeddings = bool(normalize_embeddings)
        self.self_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.neigh_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.out_proj = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.dropout = nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity()
        self.act = nn.ReLU()
        self.norm = nn.LayerNorm(self.hidden_dim)

    def forward(self, x: torch.Tensor, normalized_adjacency: torch.Tensor) -> torch.Tensor:
        neighbor_mean = torch.matmul(normalized_adjacency.unsqueeze(0), x)
        fused = torch.cat([self.self_proj(x), self.neigh_proj(neighbor_mean)], dim=-1)
        updated = self.out_proj(self.dropout(self.act(fused)))
        if self.residual:
            updated = updated + x
        updated = self.norm(updated)
        if self.normalize_embeddings:
            updated = torch.nn.functional.normalize(updated, p=2, dim=-1)
        return updated


class GraphSAGEBaseline(nn.Module):
    def __init__(self, time_steps: int, normalized_adjacency: torch.Tensor, input_channels: int = 1, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.0, input_dropout: float = 0.0, residual: bool = True, normalize_embeddings: bool = False, n_pred: int = 1, direct_multi_step: bool = False):
        super().__init__()
        self.output_cfg = BaselineOutputConfig(n_pred=n_pred, direct_multi_step=direct_multi_step)
        self.encoder = NodeTimeEncoder(time_steps, input_channels, hidden_dim, input_dropout=input_dropout)
        self.register_buffer("normalized_adjacency", normalized_adjacency.to(dtype=torch.float32), persistent=False)
        self.layers = nn.ModuleList([
            GraphSAGELayer(hidden_dim, dropout=dropout, residual=residual, normalize_embeddings=normalize_embeddings)
            for _ in range(int(num_layers))
        ])
        self.post = ResidualMLP(hidden_dim, dropout=dropout)
        self.head = GraphPredictionHead(hidden_dim, self.output_cfg)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.encoder(inputs)
        for layer in self.layers:
            x = layer(x, self.normalized_adjacency)
        x = x + self.post(x)
        return self.head(x)
