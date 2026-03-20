import torch

from engine.model_registry import ModelRuntime, register_model
from models.graph_baselines.gat import GATBaseline
from models.graph_baselines.graphsage import GraphSAGEBaseline


def _graph_hidden_dim(args):
    return int(getattr(args, "graph_hidden_dim", 64))


def _graph_num_layers(args):
    return int(getattr(args, "graph_num_layers", 2))


def build_gat(args, graph_data, device):
    if graph_data is None:
        raise ValueError("GAT requires graph_data")
    return GATBaseline(
        time_steps=args.n_his,
        adjacency_mask=torch.as_tensor(graph_data["adjacency_mask"], dtype=torch.bool, device=device),
        input_channels=1,
        hidden_dim=_graph_hidden_dim(args),
        num_layers=_graph_num_layers(args),
        heads=args.gat_heads,
        concat_heads=args.gat_concat_heads,
        dropout=args.graph_dropout,
        input_dropout=args.graph_input_dropout,
        attention_dropout=args.gat_attention_dropout,
        leaky_relu_slope=args.gat_leaky_relu_slope,
        residual=args.graph_residual,
        n_pred=args.n_pred,
        direct_multi_step=args.direct_multi_step,
    ).to(device)


def build_graphsage(args, graph_data, device):
    if graph_data is None:
        raise ValueError("GraphSAGE requires graph_data")
    aggregator = getattr(args, "sage_aggregator", "mean").lower()
    if aggregator != "mean":
        raise ValueError(f"Unsupported GraphSAGE aggregator: {aggregator}")
    return GraphSAGEBaseline(
        time_steps=args.n_his,
        normalized_adjacency=torch.as_tensor(graph_data["normalized_adjacency"], dtype=torch.float32, device=device),
        input_channels=1,
        hidden_dim=_graph_hidden_dim(args),
        num_layers=_graph_num_layers(args),
        dropout=args.graph_dropout,
        input_dropout=args.graph_input_dropout,
        residual=args.graph_residual,
        normalize_embeddings=args.sage_normalize_embeddings,
        n_pred=args.n_pred,
        direct_multi_step=args.direct_multi_step,
    ).to(device)


MODEL_RUNTIME = ModelRuntime(
    name="gat",
    description="Dense PyTorch graph attention baseline.",
    build_fn=build_gat,
    supports_rollout=True,
    supports_graph=True,
    supports_direct_multistep=True,
)

register_model(MODEL_RUNTIME)
register_model(
    ModelRuntime(
        name="graphsage",
        description="Dense PyTorch GraphSAGE baseline.",
        build_fn=build_graphsage,
        supports_rollout=True,
        supports_graph=True,
        supports_direct_multistep=True,
    )
)
