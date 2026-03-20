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


def resolve_graph_file(args):
    if args.graph == "default":
        return resolve_dataset_file(args.dataset_dir, f"PeMSD7_W_{args.n_route}.csv")
    return resolve_dataset_file(args.dataset_dir, args.graph)


def normalize_dense_adjacency(W, add_self_loops=True):
    W = np.asarray(W, dtype=np.float64)
    A = np.copy(W)
    if add_self_loops:
        A = A + np.identity(W.shape[0], dtype=np.float64)
    degree = np.sum(A, axis=1)
    inv_degree = np.where(degree > 0, 1.0 / degree, 0.0)
    return inv_degree[:, None] * A


def build_graph_data(args):
    n_route = args.n_route
    graph_file = resolve_graph_file(args)
    W = weight_matrix(str(graph_file))
    if W.shape[0] != n_route:
        raise ValueError(f"Graph CSV has {W.shape[0]} nodes but --n_route={n_route}")

    graph_mode = getattr(args, "graph_approx", "cheb").lower()
    if graph_mode == "first":
        stgcn_kernel = first_approx(W, n_route)
    else:
        L = scaled_laplacian(W)
        stgcn_kernel = cheb_poly_approx(L, args.ks, n_route)

    adjacency_dense = np.asarray(W, dtype=np.float64)
    graph_self_loops = bool(getattr(args, "graph_self_loops", True))
    adjacency_mask = adjacency_dense > 0
    if graph_self_loops:
        adjacency_mask = np.logical_or(adjacency_mask, np.eye(n_route, dtype=bool))
    isolated_nodes = ~adjacency_mask.any(axis=1)
    if np.any(isolated_nodes):
        adjacency_mask[isolated_nodes, isolated_nodes] = True
    normalized_adjacency = normalize_dense_adjacency(adjacency_dense, add_self_loops=graph_self_loops)

    return {
        "graph_file": str(graph_file),
        "graph_approx": graph_mode,
        "raw_weight_matrix": adjacency_dense,
        "adjacency_dense": adjacency_dense,
        "adjacency_mask": adjacency_mask.astype(bool),
        "normalized_adjacency": normalized_adjacency,
        "stgcn_kernel": np.asarray(stgcn_kernel, dtype=np.float64),
    }


def build_graph_kernel(args):
    graph_data = build_graph_data(args)
    return graph_data["stgcn_kernel"], graph_data["raw_weight_matrix"]


def load_dataset(args):
    data_file = resolve_dataset_file(args.dataset_dir, f"PeMSD7_V_{args.n_route}.csv")
    n_train = getattr(args, "n_train", 34)
    n_val = getattr(args, "n_val", 5)
    n_test = getattr(args, "n_test", 5)
    day_slot = getattr(args, "day_slot", 288)
    dataset = data_gen(str(data_file), (n_train, n_val, n_test), args.n_route, args.n_his + args.n_pred, day_slot=day_slot)
    return dataset


def summarize_dataset(dataset):
    return {
        "mean": float(dataset.mean),
        "std": float(dataset.std),
        "train_samples": int(dataset.get_len("train")),
        "val_samples": int(dataset.get_len("val")),
        "test_samples": int(dataset.get_len("test")),
    }
