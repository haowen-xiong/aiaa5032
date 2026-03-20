from dataclasses import dataclass
from importlib import import_module
from typing import Callable


@dataclass(frozen=True)
class ModelRuntime:
    name: str
    description: str
    build_fn: Callable
    supports_rollout: bool = True
    supports_graph: bool = False
    supports_direct_multistep: bool = False


_REGISTRY: dict[str, ModelRuntime] = {}


def register_model(runtime: ModelRuntime):
    _REGISTRY[runtime.name.lower()] = runtime
    return runtime


def _load_runtime(module_name: str):
    module = import_module(module_name)
    runtime = getattr(module, "MODEL_RUNTIME", None)
    if not isinstance(runtime, ModelRuntime):
        raise ValueError(f"Module {module_name} does not expose a valid MODEL_RUNTIME.")
    register_model(runtime)


def _register_builtin_models():
    if _REGISTRY:
        return
    _load_runtime("models.stgcn.runtime")
    _load_runtime("models.baselines.runtime")
    _load_runtime("models.graph_baselines.runtime")


def get_model_runtime(model_name: str) -> ModelRuntime:
    _register_builtin_models()
    key = model_name.lower()
    if key not in _REGISTRY:
        raise KeyError(f'Unknown model "{model_name}". Available models: {", ".join(available_models())}')
    return _REGISTRY[key]


def available_models():
    _register_builtin_models()
    return sorted(_REGISTRY.keys())
