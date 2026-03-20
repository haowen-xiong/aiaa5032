from __future__ import annotations

import torch
import torch.nn as nn

from models.baselines.common import BaselineOutputConfig
from .common import GraphPredictionHead, NodeTimeEncoder, ResidualMLP


class DenseGATLayer(nn.Module):
    def __init__(self, hidden_dim: int, heads: int = 2, concat_heads: bool = True, dropout: float = 0.0, attention_dropout: float = 0.0, leaky_relu_slope: float = 0.2, residual: bool = True):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.heads = int(heads)
        self.concat_heads = bool(concat_heads)
        self.residual = bool(residual)
        if self.hidden_dim % self.heads != 0:
            raise ValueError(f"hidden_dim={self.hidden_dim} must be divisible by heads={self.heads}")
        self.head_dim = self.hidden_dim // self.heads
        self.input_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.attn_src = nn.Parameter(torch.empty(self.heads, self.head_dim))
        self.attn_dst = nn.Parameter(torch.empty(self.heads, self.head_dim))
        self.attn_dropout = nn.Dropout(float(attention_dropout)) if attention_dropout > 0 else nn.Identity()
        self.out_dropout = nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity()
        self.act = nn.LeakyReLU(float(leaky_relu_slope))
        self.out_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.norm = nn.LayerNorm(self.hidden_dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.xavier_uniform_(self.attn_src)
        nn.init.xavier_uniform_(self.attn_dst)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: torch.Tensor, adjacency_mask: torch.Tensor) -> torch.Tensor:
        batch_size, num_nodes, _ = x.shape
        projected = self.input_proj(x).view(batch_size, num_nodes, self.heads, self.head_dim)
        projected = projected.permute(0, 2, 1, 3)

        alpha_src = (projected * self.attn_src.view(1, self.heads, 1, self.head_dim)).sum(dim=-1)
        alpha_dst = (projected * self.attn_dst.view(1, self.heads, 1, self.head_dim)).sum(dim=-1)
        scores = self.act(alpha_src.unsqueeze(-1) + alpha_dst.unsqueeze(-2))

        mask = adjacency_mask.unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        attention = torch.softmax(scores, dim=-1)
        attention = torch.where(mask, attention, torch.zeros_like(attention))
        attention = self.attn_dropout(attention)

        aggregated = torch.matmul(attention, projected)
        aggregated = aggregated.permute(0, 2, 1, 3).reshape(batch_size, num_nodes, self.hidden_dim)
        updated = self.out_proj(self.out_dropout(aggregated))
        if self.residual:
            updated = updated + x
        return self.norm(updated)


class GATBaseline(nn.Module):
    def __init__(self, time_steps: int, adjacency_mask: torch.Tensor, input_channels: int = 1, hidden_dim: int = 64, num_layers: int = 2, heads: int = 2, concat_heads: bool = True, dropout: float = 0.0, input_dropout: float = 0.0, attention_dropout: float = 0.0, leaky_relu_slope: float = 0.2, residual: bool = True, n_pred: int = 1, direct_multi_step: bool = False):
        super().__init__()
        self.output_cfg = BaselineOutputConfig(n_pred=n_pred, direct_multi_step=direct_multi_step)
        self.encoder = NodeTimeEncoder(time_steps, input_channels, hidden_dim, input_dropout=input_dropout)
        self.register_buffer("adjacency_mask", adjacency_mask.to(dtype=torch.bool), persistent=False)
        self.layers = nn.ModuleList([
            DenseGATLayer(hidden_dim, heads=heads, concat_heads=concat_heads, dropout=dropout, attention_dropout=attention_dropout, leaky_relu_slope=leaky_relu_slope, residual=residual)
            for _ in range(int(num_layers))
        ])
        self.post = ResidualMLP(hidden_dim, dropout=dropout)
        self.head = GraphPredictionHead(hidden_dim, self.output_cfg)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.encoder(inputs)
        for layer in self.layers:
            x = layer(x, self.adjacency_mask)
        x = x + self.post(x)
        return self.head(x)
