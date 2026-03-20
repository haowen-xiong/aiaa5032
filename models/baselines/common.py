from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class BaselineOutputConfig:
    n_pred: int = 1
    direct_multi_step: bool = False

    @property
    def steps(self) -> int:
        if self.direct_multi_step and self.n_pred > 1:
            return int(self.n_pred)
        return 1


def validate_inputs(inputs: torch.Tensor, expected_channels: int | None = None) -> tuple[int, int, int, int]:
    if inputs.ndim != 4:
        raise ValueError(f"Expected inputs with shape [B, T, N, C], got {tuple(inputs.shape)}")
    shape = tuple(int(v) for v in inputs.shape)
    if expected_channels is not None and shape[-1] != expected_channels:
        raise ValueError(f"Expected {expected_channels} input channels, got {shape[-1]}")
    return shape


def flatten_spatiotemporal(inputs: torch.Tensor) -> tuple[torch.Tensor, int, int]:
    batch_size, time_steps, num_nodes, channels = validate_inputs(inputs)
    flat = inputs.permute(0, 2, 1, 3).reshape(batch_size * num_nodes, time_steps * channels)
    return flat, batch_size, num_nodes


def reshape_node_outputs(outputs: torch.Tensor, batch_size: int, num_nodes: int, steps: int) -> torch.Tensor:
    if steps == 1:
        return outputs.reshape(batch_size, num_nodes, 1)
    return outputs.reshape(batch_size, num_nodes, steps).permute(0, 2, 1).unsqueeze(-1)


def extract_last_value(inputs: torch.Tensor, feature_index: int = -1) -> torch.Tensor:
    validate_inputs(inputs)
    last_frame = inputs[:, -1, :, :]
    if last_frame.shape[-1] == 1:
        return last_frame
    idx = feature_index % last_frame.shape[-1]
    return last_frame[..., idx : idx + 1]
