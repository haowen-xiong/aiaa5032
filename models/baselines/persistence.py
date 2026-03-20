from __future__ import annotations

import torch
import torch.nn as nn

from .common import BaselineOutputConfig, extract_last_value, validate_inputs


class PersistenceBaseline(nn.Module):
    def __init__(self, n_pred: int = 1, direct_multi_step: bool = False, feature_index: int = -1):
        super().__init__()
        self.output_cfg = BaselineOutputConfig(n_pred=n_pred, direct_multi_step=direct_multi_step)
        self.feature_index = feature_index

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size, _, num_nodes, _ = validate_inputs(inputs)
        last_value = extract_last_value(inputs, self.feature_index)
        if self.output_cfg.steps == 1:
            return last_value
        return last_value.unsqueeze(1).expand(batch_size, self.output_cfg.steps, num_nodes, 1).contiguous()
