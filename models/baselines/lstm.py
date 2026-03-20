from __future__ import annotations

import torch
import torch.nn as nn

from .common import BaselineOutputConfig, validate_inputs


class LSTMBaseline(nn.Module):
    def __init__(
        self,
        input_channels: int = 1,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = False,
        n_pred: int = 1,
        direct_multi_step: bool = False,
    ):
        super().__init__()
        self.output_cfg = BaselineOutputConfig(n_pred=n_pred, direct_multi_step=direct_multi_step)
        self.input_channels = int(input_channels)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.bidirectional = bool(bidirectional)

        lstm_dropout = dropout if self.num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=self.input_channels,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=lstm_dropout,
            bidirectional=self.bidirectional,
        )
        directions = 2 if self.bidirectional else 1
        self.head = nn.Linear(self.hidden_size * directions, self.output_cfg.steps)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size, time_steps, num_nodes, channels = validate_inputs(inputs)
        if channels != self.input_channels:
            raise ValueError(
                f"LSTMBaseline expected input_channels={self.input_channels}, got {channels}"
            )

        seq = inputs.permute(0, 2, 1, 3).reshape(batch_size * num_nodes, time_steps, channels)
        _, (hidden_state, _) = self.lstm(seq)
        if self.bidirectional:
            hidden_state = hidden_state.view(self.num_layers, 2, batch_size * num_nodes, self.hidden_size)
            node_repr = torch.cat([hidden_state[-1, 0], hidden_state[-1, 1]], dim=-1)
        else:
            node_repr = hidden_state[-1]
        outputs = self.head(node_repr)
        if self.output_cfg.steps == 1:
            return outputs.reshape(batch_size, num_nodes, 1)
        return outputs.reshape(batch_size, num_nodes, self.output_cfg.steps).permute(0, 2, 1).unsqueeze(-1)
