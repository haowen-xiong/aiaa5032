import json
from pathlib import Path

from engine.data import build_graph_kernel, load_dataset, summarize_dataset
from engine.experiment import train_and_test_model
from engine.model_registry import available_models, get_model_runtime
from engine.paths import build_experiment_paths, pick_device, set_random_seed


def prepare_experiment(args):
    paths = build_experiment_paths(args.output_dir, args.exp_name, args.model_name)
    args.save_dir = str(paths.model_root)
    args.sum_dir = str(paths.tensorboard_dir)
    args.model_root = str(paths.model_root)
    args.experiment_root = str(paths.experiment_root)
    args.output_root = str(paths.output_root)
    return paths


def write_experiment_manifest(args, dataset, paths, runtime):
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
        "runtime": {
            "supports_rollout": runtime.supports_rollout,
            "supports_graph": runtime.supports_graph,
            "supports_direct_multistep": runtime.supports_direct_multistep,
        },
    }
    with open(Path(paths.model_root) / "experiment_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def run_experiment(args):
    set_random_seed(args.seed)
    device = pick_device(args.device)
    runtime = get_model_runtime(args.model_name)
    paths = prepare_experiment(args)
    dataset = load_dataset(args)
    graph_kernel, _ = build_graph_kernel(args)
    model = runtime.build_fn(args, graph_kernel, device)

    write_experiment_manifest(args, dataset, paths, runtime)

    print(f"Training configs: {args}")
    print(f">> Using device: {device}")
    print(f">> Experiment root: {paths.model_root}")
    print(f">> Loading dataset with Mean: {dataset.mean:.2f}, STD: {dataset.std:.2f}")

    train_and_test_model(model, dataset, args, device, paths.model_root)
