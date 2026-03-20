import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_loader.data_utils import data_gen, gen_batch
from models.base_model import build_model
from models.tester import resolve_checkpoint
from utils.math_graph import cheb_poly_approx, scaled_laplacian, weight_matrix
from utils.math_utils import MAE, MAPE, RMSE, z_inverse


def parse_args():
    parser = argparse.ArgumentParser(description="Generate visualization figures for trained STGCN runs.")
    parser.add_argument("--run_meta", type=str, default="./output/models/run_meta.json")
    parser.add_argument("--history", type=str, default="./output/models/history.json")
    parser.add_argument("--test_results", type=str, default="./output/models/test_results.json")
    parser.add_argument("--checkpoint_dir", type=str, default="./output/models")
    parser.add_argument("--dataset_dir", type=str, default="./dataset")
    parser.add_argument("--output_dir", type=str, default="./output/visualizations")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--sensor_idx", type=int, default=-1)
    parser.add_argument("--time_points", type=int, default=240)
    parser.add_argument("--heatmap_nodes", type=int, default=64)
    return parser.parse_args()


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_dataset_file(dataset_dir, filename):
    candidates = [
        Path(dataset_dir) / filename,
        Path(dataset_dir) / "PeMSD7_Full" / filename,
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Cannot find dataset file {filename} under {dataset_dir}.")


def load_run_args(run_meta_path):
    with open(run_meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return SimpleNamespace(**meta["args"])


def build_graph_kernel(args):
    if args.graph == "default":
        graph_file = resolve_dataset_file(args.dataset_dir, f"PeMSD7_W_{args.n_route}.csv")
    else:
        graph_file = resolve_dataset_file(args.dataset_dir, args.graph)
    W = weight_matrix(str(graph_file))
    L = scaled_laplacian(W)
    return cheb_poly_approx(L, args.ks, args.n_route), W


def load_dataset(args):
    data_file = resolve_dataset_file(args.dataset_dir, f"PeMSD7_V_{args.n_route}.csv")
    return data_gen(str(data_file), (34, 5, 5), args.n_route, args.n_his + args.n_pred)


def load_model(args, blocks, graph_kernel, device, checkpoint_dir):
    checkpoint_path = resolve_checkpoint(checkpoint_dir)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = build_model(args, blocks, graph_kernel, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint_path


@torch.no_grad()
def predict_all_steps(model, seq, batch_size, n_his, n_pred, device):
    all_batches = []
    for batch in gen_batch(seq, min(batch_size, len(seq)), dynamic_batch=True):
        test_seq = np.copy(batch[:, 0:n_his + 1, :, :])
        step_preds = []
        for _ in range(n_pred):
            pred = model(torch.as_tensor(test_seq, dtype=torch.float32, device=device)).cpu().numpy()
            test_seq[:, 0:n_his - 1, :, :] = test_seq[:, 1:n_his, :, :]
            test_seq[:, n_his - 1, :, :] = pred
            step_preds.append(pred)
        all_batches.append(step_preds)
    return np.concatenate(all_batches, axis=1)


def collect_step_metrics(x_test, preds, stats, n_his):
    metrics = {"MAPE": [], "MAE": [], "RMSE": []}
    targets = []
    for step in range(preds.shape[0]):
        y_true = x_test[:, step + n_his, :, :]
        y_pred = preds[step]
        v_true = z_inverse(y_true, stats["mean"], stats["std"])
        v_pred = z_inverse(y_pred, stats["mean"], stats["std"])
        metrics["MAPE"].append(MAPE(v_true, v_pred))
        metrics["MAE"].append(MAE(v_true, v_pred))
        metrics["RMSE"].append(RMSE(v_true, v_pred))
        targets.append(v_true)
    return metrics, np.stack(targets, axis=0)


def pick_sensor(targets, sensor_idx):
    if sensor_idx >= 0:
        return sensor_idx
    variability = targets[0, :, :, 0].std(axis=0)
    return int(np.argmax(variability))


def save_summary(metrics, checkpoint_path, out_dir, sensor_idx):
    payload = {
        "checkpoint": str(checkpoint_path),
        "selected_sensor_idx": sensor_idx,
        "step_metrics": {
            metric: {f"step_{i + 1}": float(value) for i, value in enumerate(values)}
            for metric, values in metrics.items()
        },
    }
    with open(out_dir / "visualization_summary.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def plot_training_curves(history, best_epoch, out_dir):
    epochs = [item["epoch"] for item in history]
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    metric_names = ["MAPE", "MAE", "RMSE"]
    for metric_idx, metric_name in enumerate(metric_names):
        val_curves = np.array([item["val_metrics"][metric_idx::3] for item in history])
        test_curves = np.array([item["test_metrics"][metric_idx::3] for item in history])
        axes[metric_idx].plot(epochs, val_curves.mean(axis=1), label="Validation", linewidth=2)
        axes[metric_idx].plot(epochs, test_curves.mean(axis=1), label="Test", linewidth=2)
        axes[metric_idx].axvline(best_epoch, color="tab:red", linestyle="--", label=f"Best Epoch {best_epoch}")
        axes[metric_idx].set_ylabel(metric_name)
        axes[metric_idx].grid(alpha=0.3)
        axes[metric_idx].legend()
    axes[-1].set_xlabel("Epoch")
    fig.suptitle("Training Dynamics Across Epochs")
    fig.tight_layout()
    fig.savefig(out_dir / "training_curves.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_horizon_bars(test_results, out_dir):
    horizons = ["15 min", "30 min", "45 min"]
    metrics = ["MAPE", "MAE", "RMSE"]
    values = {
        metric: [
            test_results["test_metrics"]["step_3"][metric] if idx == 0 else
            test_results["test_metrics"]["step_6"][metric] if idx == 1 else
            test_results["test_metrics"]["step_9"][metric]
            for idx in range(3)
        ]
        for metric in metrics
    }

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    colors = ["#0f766e", "#ca8a04", "#b91c1c"]
    for ax, metric in zip(axes, metrics):
        y = values[metric]
        if metric == "MAPE":
            y = [v * 100 for v in y]
        ax.bar(horizons, y, color=colors)
        ax.set_title(metric)
        ax.grid(axis="y", alpha=0.3)
        if metric == "MAPE":
            ax.set_ylabel("Percent")
    fig.suptitle("Forecast Error by Horizon")
    fig.tight_layout()
    fig.savefig(out_dir / "horizon_error_bars.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_rollout_curve(step_metrics, out_dir):
    steps = np.arange(1, len(step_metrics["MAE"]) + 1)
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    for ax, metric_name in zip(axes, ["MAPE", "MAE", "RMSE"]):
        values = np.array(step_metrics[metric_name])
        if metric_name == "MAPE":
            values = values * 100
        ax.plot(steps, values, marker="o", linewidth=2, color="#1d4ed8")
        ax.set_ylabel(metric_name)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("Autoregressive Step")
    fig.suptitle("Error Growth During Multi-Step Rollout")
    fig.tight_layout()
    fig.savefig(out_dir / "rollout_error_curve.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_sensor_forecast(targets, preds, sensor_idx, time_points, out_dir):
    time_points = min(time_points, targets.shape[1])
    x_axis = np.arange(time_points)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(x_axis, targets[0, :time_points, sensor_idx, 0], label="Ground Truth (15 min)", color="#111827", linewidth=2)
    ax.plot(x_axis, preds[2, :time_points, sensor_idx, 0], label="Predicted (15 min)", color="#0f766e", linewidth=1.8)
    ax.plot(x_axis, preds[5, :time_points, sensor_idx, 0], label="Predicted (30 min)", color="#ca8a04", linewidth=1.6)
    ax.plot(x_axis, preds[8, :time_points, sensor_idx, 0], label="Predicted (45 min)", color="#b91c1c", linewidth=1.4)
    ax.set_title(f"Single-Sensor Forecast Trajectory (Sensor {sensor_idx})")
    ax.set_xlabel("Sliding Test Window Index")
    ax.set_ylabel("Speed")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"sensor_{sensor_idx}_forecast.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_error_heatmaps(targets, preds, heatmap_nodes, time_points, out_dir):
    time_points = min(time_points, targets.shape[1])
    heatmap_nodes = min(heatmap_nodes, targets.shape[2])
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    horizons = [(2, "15 min"), (5, "30 min"), (8, "45 min")]
    for ax, (idx, title) in zip(axes, horizons):
        error = np.abs(targets[idx, :time_points, :heatmap_nodes, 0] - preds[idx, :time_points, :heatmap_nodes, 0]).T
        im = ax.imshow(error, aspect="auto", cmap="magma")
        ax.set_title(title)
        ax.set_xlabel("Sliding Test Window Index")
        ax.set_ylabel("Sensor Index")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Spatio-Temporal Absolute Error Heatmaps")
    fig.tight_layout()
    fig.savefig(out_dir / "error_heatmaps.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_adjacency_heatmap(W, out_dir):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(W, aspect="auto", cmap="viridis")
    ax.set_title("Weighted Adjacency Matrix")
    ax.set_xlabel("Sensor Index")
    ax.set_ylabel("Sensor Index")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_dir / "adjacency_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    cli_args = parse_args()
    out_dir = ensure_dir(cli_args.output_dir)

    run_args = load_run_args(cli_args.run_meta)
    run_args.dataset_dir = cli_args.dataset_dir
    device = torch.device(cli_args.device)
    blocks = [[1, 32, 64], [64, 32, 128]]

    graph_kernel, W = build_graph_kernel(run_args)
    dataset = load_dataset(run_args)
    model, checkpoint_path = load_model(run_args, blocks, graph_kernel, device, cli_args.checkpoint_dir)

    with open(cli_args.history, "r", encoding="utf-8") as f:
        history = json.load(f)
    with open(cli_args.test_results, "r", encoding="utf-8") as f:
        test_results = json.load(f)
    with open(Path(cli_args.checkpoint_dir) / "best_meta.json", "r", encoding="utf-8") as f:
        best_meta = json.load(f)

    x_test = dataset.get_data("test")
    preds = predict_all_steps(model, x_test, run_args.batch_size, run_args.n_his, run_args.n_pred, device)
    step_metrics, targets = collect_step_metrics(x_test, preds, dataset.get_stats(), run_args.n_his)
    preds_real = z_inverse(preds, dataset.mean, dataset.std)

    sensor_idx = pick_sensor(targets, cli_args.sensor_idx)
    save_summary(step_metrics, checkpoint_path, out_dir, sensor_idx)

    plot_training_curves(history, best_meta["best_epoch"], out_dir)
    plot_horizon_bars(test_results, out_dir)
    plot_rollout_curve(step_metrics, out_dir)
    plot_sensor_forecast(targets, preds_real, sensor_idx, cli_args.time_points, out_dir)
    plot_error_heatmaps(targets, preds_real, cli_args.heatmap_nodes, cli_args.time_points, out_dir)
    plot_adjacency_heatmap(W, out_dir)

    print(f"Saved visualizations to {out_dir}")


if __name__ == "__main__":
    main()
