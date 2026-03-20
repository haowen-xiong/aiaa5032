import json
from pathlib import Path

from engine.data import build_graph_data, load_dataset, summarize_dataset
from engine.experiment import train_and_test_model
from engine.model_registry import available_models, get_model_runtime
from engine.paths import build_experiment_paths, pick_device, set_random_seed


GRAPH_MODELS = {"stgcn", "gat", "graphsage"}


def prepare_experiment(args):
    paths = build_experiment_paths(args.output_dir, args.exp_name, args.model_name, overwrite=getattr(args, "overwrite", False))
    args.save_dir = str(paths.model_root)
    args.sum_dir = str(paths.tensorboard_dir)
    args.model_root = str(paths.model_root)
    args.experiment_root = str(paths.experiment_root)
    args.output_root = str(paths.output_root)
    return paths


def build_hyperparameter_snapshot(args):
    snapshot = {
        "direct_multi_step": bool(getattr(args, "direct_multi_step", False)),
        "graph_file": getattr(args, "graph", "default"),
    }
    if args.model_name == "stgcn":
        snapshot.update(
            {
                "ks": args.ks,
                "kt": args.kt,
                "graph_approx": getattr(args, "graph_approx", "cheb"),
                "use_spatial": bool(getattr(args, "use_spatial", True)),
                "drop_prob": float(getattr(args, "drop_prob", 0.0)),
            }
        )
    elif args.model_name in {"gat", "graphsage"}:
        snapshot.update(
            {
                "graph_hidden_dim": int(getattr(args, "graph_hidden_dim", 64)),
                "graph_num_layers": int(getattr(args, "graph_num_layers", 2)),
                "graph_dropout": float(getattr(args, "graph_dropout", 0.0)),
                "graph_input_dropout": float(getattr(args, "graph_input_dropout", 0.0)),
                "graph_residual": bool(getattr(args, "graph_residual", True)),
                "graph_self_loops": bool(getattr(args, "graph_self_loops", True)),
            }
        )
        if args.model_name == "gat":
            snapshot.update(
                {
                    "gat_heads": int(getattr(args, "gat_heads", 2)),
                    "gat_concat_heads": bool(getattr(args, "gat_concat_heads", True)),
                    "gat_leaky_relu_slope": float(getattr(args, "gat_leaky_relu_slope", 0.2)),
                    "gat_attention_dropout": float(getattr(args, "gat_attention_dropout", 0.0)),
                }
            )
        if args.model_name == "graphsage":
            snapshot.update(
                {
                    "sage_aggregator": getattr(args, "sage_aggregator", "mean"),
                    "sage_normalize_embeddings": bool(getattr(args, "sage_normalize_embeddings", False)),
                }
            )
    return snapshot


def write_experiment_manifest(args, dataset, paths, runtime, graph_data):
    manifest = {
        "model_name": args.model_name,
        "exp_name": args.exp_name,
        "output_root": str(paths.output_root),
        "experiment_root": str(paths.experiment_root),
        "model_root": str(paths.model_root),
        "tensorboard_dir": str(paths.tensorboard_dir),
        "dataset": summarize_dataset(dataset),
        "available_models": available_models(),
        "graph_approx": getattr(args, "graph_approx", "cheb"),
        "graph_file": graph_data["graph_file"] if graph_data is not None else None,
        "model_family": "graph" if args.model_name in GRAPH_MODELS else "baseline",
        "runtime": {
            "supports_rollout": runtime.supports_rollout,
            "supports_graph": runtime.supports_graph,
            "supports_direct_multistep": runtime.supports_direct_multistep,
        },
        "graph_settings": None
        if graph_data is None
        else {
            "graph_file": graph_data["graph_file"],
            "graph_approx": graph_data["graph_approx"],
            "graph_self_loops": bool(getattr(args, "graph_self_loops", True)),
        },
        "hyperparameters": build_hyperparameter_snapshot(args),
    }
    with open(Path(paths.model_root) / "experiment_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def run_experiment(args):
    set_random_seed(args.seed)
    device = pick_device(args.device)
    runtime = get_model_runtime(args.model_name)
    paths = prepare_experiment(args)
    dataset = load_dataset(args)
    graph_data = build_graph_data(args) if runtime.supports_graph else None
    model = runtime.build_fn(args, graph_data, device)

    write_experiment_manifest(args, dataset, paths, runtime, graph_data)

    print(f"Training configs: {args}")
    print(f">> Using device: {device}")
    print(f">> Experiment root: {paths.model_root}")
    print(f">> Loading dataset with Mean: {dataset.mean:.2f}, STD: {dataset.std:.2f}")
    if graph_data is not None:
        print(f">> Graph file: {graph_data['graph_file']}")

    train_and_test_model(model, dataset, args, device, paths.model_root)
