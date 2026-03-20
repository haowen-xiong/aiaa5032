from engine.model_registry import ModelRuntime, register_model
from models.baselines.factory import build_baseline


def parse_hidden_dims(hidden_dims: str):
    values = [v.strip() for v in hidden_dims.split(",") if v.strip()]
    return tuple(int(v) for v in values) if values else (128, 64)


def build_persistence(args, graph_data, device):
    return build_baseline("persistence", n_pred=args.n_pred, direct_multi_step=args.direct_multi_step).to(device)


def build_temporal_mlp(args, graph_data, device):
    return build_baseline(
        "temporal_mlp",
        time_steps=args.n_his,
        hidden_dims=parse_hidden_dims(args.mlp_hidden_dims),
        n_pred=args.n_pred,
        direct_multi_step=args.direct_multi_step,
        dropout=args.drop_prob,
    ).to(device)


def build_lstm(args, graph_data, device):
    return build_baseline(
        "lstm",
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.drop_prob,
        bidirectional=args.bidirectional,
        n_pred=args.n_pred,
        direct_multi_step=args.direct_multi_step,
    ).to(device)


MODEL_RUNTIME = ModelRuntime(
    name="persistence",
    description="Last-value persistence baseline.",
    build_fn=build_persistence,
    supports_rollout=True,
    supports_graph=False,
    supports_direct_multistep=True,
)

register_model(MODEL_RUNTIME)
register_model(ModelRuntime("temporal_mlp", "Temporal MLP baseline.", build_temporal_mlp, True, False, True))
register_model(ModelRuntime("lstm", "LSTM baseline.", build_lstm, True, False, True))
