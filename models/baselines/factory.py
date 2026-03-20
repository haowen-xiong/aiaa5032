from __future__ import annotations

from .lstm import LSTMBaseline
from .persistence import PersistenceBaseline
from .temporal_mlp import TemporalMLPBaseline


BASELINE_REGISTRY = {
    "persistence": PersistenceBaseline,
    "last_value": PersistenceBaseline,
    "temporal_mlp": TemporalMLPBaseline,
    "mlp": TemporalMLPBaseline,
    "lstm": LSTMBaseline,
}


def build_baseline(name: str, **kwargs):
    key = name.lower()
    if key not in BASELINE_REGISTRY:
        available = ", ".join(sorted(BASELINE_REGISTRY))
        raise KeyError(f"Unknown baseline '{name}'. Available: {available}")
    return BASELINE_REGISTRY[key](**kwargs)
