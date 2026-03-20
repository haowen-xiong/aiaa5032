import torch

from engine.model_registry import ModelRuntime
from models.stgcn.config import DEFAULT_BLOCK_SPECS
from models.stgcn.model import build_stgcn


def build_model(args, graph_data, device):
    kernel = torch.as_tensor(graph_data["stgcn_kernel"], dtype=torch.float32, device=device)
    return build_stgcn(
        n_his=args.n_his,
        Ks=args.ks,
        Kt=args.kt,
        n_route=args.n_route,
        graph_kernel=kernel,
        block_specs=DEFAULT_BLOCK_SPECS,
        drop_prob=args.drop_prob,
        graph_conv_type=args.graph_approx,
        use_spatial=args.use_spatial,
        direct_multi_step=args.direct_multi_step,
        n_pred=args.n_pred,
    ).to(device)


MODEL_RUNTIME = ModelRuntime(
    name="stgcn",
    description="Spatio-Temporal Graph Convolutional Network with ablation hooks.",
    build_fn=build_model,
    supports_rollout=True,
    supports_graph=True,
    supports_direct_multistep=True,
)
