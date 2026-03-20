from pathlib import Path

import numpy as np

from data_loader.data_utils import data_gen
from utils.math_graph import cheb_poly_approx, first_approx, scaled_laplacian, weight_matrix


def resolve_dataset_file(dataset_dir, filename):
    candidates = [
        Path(dataset_dir) / filename,
        Path(dataset_dir) / "PeMSD7_Full" / filename,
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Cannot find dataset file {filename} under {dataset_dir}.")


def build_graph_kernel(args):
    n_route = args.n_route
    if args.graph == "default":
        graph_file = resolve_dataset_file(args.dataset_dir, f"PeMSD7_W_{n_route}.csv")
    else:
        graph_file = resolve_dataset_file(args.dataset_dir, args.graph)

    W = weight_matrix(str(graph_file))
    graph_mode = getattr(args, "graph_approx", "cheb").lower()
    if graph_mode == "first":
        return first_approx(W, n_route), W

    L = scaled_laplacian(W)
    return cheb_poly_approx(L, args.ks, n_route), W


def load_dataset(args):
    data_file = resolve_dataset_file(args.dataset_dir, f"PeMSD7_V_{args.n_route}.csv")
    n_train, n_val, n_test = 34, 5, 5
    dataset = data_gen(str(data_file), (n_train, n_val, n_test), args.n_route, args.n_his + args.n_pred)
    return dataset


def summarize_dataset(dataset):
    return {
        "mean": float(dataset.mean),
        "std": float(dataset.std),
        "train_samples": int(dataset.get_len("train")),
        "val_samples": int(dataset.get_len("val")),
        "test_samples": int(dataset.get_len("test")),
    }
