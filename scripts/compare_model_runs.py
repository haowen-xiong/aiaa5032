import argparse
import csv
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

from data_loader.data_utils import gen_batch
from engine.data import build_graph_data, load_dataset
from engine.experiment import resolve_checkpoint
from engine.model_registry import get_model_runtime
from utils.math_utils import MAE, MAPE, RMSE, z_inverse


def parse_args():
    parser = argparse.ArgumentParser(description="Compare multiple experiment runs.")
    parser.add_argument("--run_dir", action="append", required=True, help="Run directory containing run_meta/history/test_results/checkpoints")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--labels", type=str, nargs="*")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dataset_dir", type=str, default="./dataset")
    parser.add_argument("--sensor_idx", type=int, default=-1)
    parser.add_argument("--time_points", type=int, default=240)
    parser.add_argument("--heatmap_nodes", type=int, default=64)
    parser.add_argument("--mode", type=str, default="artifact-only", choices=["artifact-only", "full-prediction"])
    parser.add_argument("--baseline_label", type=str, default=None)
    return parser.parse_args()


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_run_bundle(run_dir, label=None):
    run_dir = Path(run_dir)
    run_meta = load_json(run_dir / "run_meta.json")
    history = load_json(run_dir / "history.json")
    test_results = load_json(run_dir / "test_results.json")
    best_meta = load_json(run_dir / "best_meta.json")
    manifest_path = run_dir / "experiment_manifest.json"
    manifest = load_json(manifest_path) if manifest_path.exists() else None
    args = SimpleNamespace(**run_meta["args"])
    return {
        "run_dir": str(run_dir),
        "label": label or args.model_name,
        "args": args,
        "run_meta": run_meta,
        "history": history,
        "test_results": test_results,
        "best_meta": best_meta,
        "manifest": manifest,
    }


def compatibility_signature(bundle):
    args = bundle["args"]
    return {
        "n_route": int(args.n_route),
        "n_his": int(args.n_his),
        "n_pred": int(args.n_pred),
        "n_train": int(getattr(args, "n_train", bundle["run_meta"].get("train_samples"))),
        "n_val": int(getattr(args, "n_val", bundle["run_meta"].get("val_samples"))),
        "n_test": int(getattr(args, "n_test", bundle["run_meta"].get("test_samples"))),
        "day_slot": int(getattr(args, "day_slot", 288)),
        "graph": getattr(args, "graph", "default"),
        "dataset_dir": getattr(args, "dataset_dir", None),
    }


def validate_compatibility(bundles, strict):
    reference = compatibility_signature(bundles[0])
    mismatches = []
    for bundle in bundles[1:]:
        signature = compatibility_signature(bundle)
        diff = {k: (reference[k], signature[k]) for k in reference if reference[k] != signature[k]}
        if diff:
            mismatches.append({"label": bundle["label"], "differences": diff})
    if mismatches and strict:
        raise ValueError(f"Incompatible runs for comparison: {mismatches}")
    return mismatches


@torch.no_grad()
def predict_all_steps(model, seq, batch_size, n_his, n_pred, device, direct_multi_step=False):
    pred_list = []
    model.eval()
    for batch in gen_batch(seq, min(batch_size, len(seq)), dynamic_batch=True):
        history = np.copy(batch[:, 0:n_his, :, :])
        if direct_multi_step:
            pred = model(torch.as_tensor(history, dtype=torch.float32, device=device)).cpu().numpy()
            pred_list.append(pred)
            continue
        step_preds = []
        test_seq = history
        for _ in range(n_pred):
            pred = model(torch.as_tensor(test_seq, dtype=torch.float32, device=device)).cpu().numpy()
            test_seq[:, 0:n_his - 1, :, :] = test_seq[:, 1:n_his, :, :]
            test_seq[:, n_his - 1, :, :] = pred
            step_preds.append(pred)
        pred_list.append(step_preds)
    if direct_multi_step:
        pred_array = np.concatenate(pred_list, axis=0)
        return np.transpose(pred_array, (1, 0, 2, 3))
    return np.concatenate(pred_list, axis=1)


def collect_step_metrics(x_test, preds, stats, n_his):
    metrics = {"MAPE": [], "MAE": [], "RMSE": []}
    targets = []
    for step in range(preds.shape[0]):
        y_true = x_test[:, step + n_his, :, :]
        y_pred = preds[step]
        v_true = z_inverse(y_true, stats["mean"], stats["std"])
        v_pred = z_inverse(y_pred, stats["mean"], stats["std"])
        metrics["MAPE"].append(float(MAPE(v_true, v_pred)))
        metrics["MAE"].append(float(MAE(v_true, v_pred)))
        metrics["RMSE"].append(float(RMSE(v_true, v_pred)))
        targets.append(v_true)
    return metrics, np.stack(targets, axis=0)


def collect_artifact_metrics(bundle):
    step_items = sorted(bundle["test_results"]["test_metrics"].items(), key=lambda item: int(item[0].split("_")[1]))
    return {
        metric: [float(step_metrics[metric]) for _, step_metrics in step_items]
        for metric in ["MAPE", "MAE", "RMSE"]
    }


def enrich_with_predictions(bundle, device, dataset_dir):
    args = bundle["args"]
    args.dataset_dir = dataset_dir
    runtime = get_model_runtime(args.model_name)
    graph_data = build_graph_data(args) if runtime.supports_graph else None
    dataset = load_dataset(args)
    checkpoint_path = resolve_checkpoint(bundle["run_dir"])
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = runtime.build_fn(args, graph_data, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    x_test = dataset.get_data("test")
    preds = predict_all_steps(model, x_test, args.batch_size, args.n_his, args.n_pred, device, getattr(args, "direct_multi_step", False))
    step_metrics, targets = collect_step_metrics(x_test, preds, dataset.get_stats(), args.n_his)
    preds_real = z_inverse(preds, dataset.mean, dataset.std)
    bundle["full_prediction"] = {
        "checkpoint": str(checkpoint_path),
        "step_metrics": step_metrics,
        "targets": targets,
        "preds_real": preds_real,
        "graph_data": graph_data,
    }
    return bundle


def choose_sensor(bundles, sensor_idx):
    if sensor_idx >= 0:
        return sensor_idx
    first = bundles[0].get("full_prediction")
    if first is None:
        return 0
    variability = first["targets"][0, :, :, 0].std(axis=0)
    return int(np.argmax(variability))


def summarize_bundle(bundle, mode):
    metrics = collect_artifact_metrics(bundle) if mode == "artifact-only" else bundle["full_prediction"]["step_metrics"]
    return {
        "label": bundle["label"],
        "run_dir": bundle["run_dir"],
        "model_name": bundle["args"].model_name,
        "best_epoch": int(bundle["best_meta"]["best_epoch"]),
        "best_val_score": float(bundle["best_meta"]["best_val_score"]),
        "avg_mae": float(np.mean(metrics["MAE"])),
        "avg_rmse": float(np.mean(metrics["RMSE"])),
        "avg_mape": float(np.mean(metrics["MAPE"])),
        "horizon_metrics": {
            metric: {f"step_{i + 1}": float(value) for i, value in enumerate(values)}
            for metric, values in metrics.items()
        },
    }


def write_summary_files(summary, out_dir):
    with open(out_dir / "comparison_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(out_dir / "comparison_summary.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "model_name", "best_epoch", "best_val_score", "avg_mape", "avg_mae", "avg_rmse"])
        for item in summary["runs"]:
            writer.writerow([
                item["label"],
                item["model_name"],
                item["best_epoch"],
                item["best_val_score"],
                item["avg_mape"],
                item["avg_mae"],
                item["avg_rmse"],
            ])


def plot_horizon_metric(bundles, metric_name, out_dir, mode):
    fig, ax = plt.subplots(figsize=(10, 5))
    for bundle in bundles:
        metrics = collect_artifact_metrics(bundle) if mode == "artifact-only" else bundle["full_prediction"]["step_metrics"]
        steps = np.arange(1, len(metrics[metric_name]) + 1)
        values = np.array(metrics[metric_name], dtype=float)
        if metric_name == "MAPE":
            values = values * 100
        ax.plot(steps, values, marker="o", linewidth=2, label=bundle["label"])
    ax.set_xlabel("Forecast Horizon Step")
    ax.set_ylabel("Percent" if metric_name == "MAPE" else metric_name)
    ax.set_title(f"Compare Horizon {metric_name}")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    filename = f"compare_horizon_{metric_name.lower()}.png"
    fig.savefig(out_dir / filename, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_training_dynamics(bundles, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    metrics_map = {"MAPE": 0, "MAE": 1, "RMSE": 2}
    
    for bundle in bundles:
        history = bundle.get("history", [])
        if not history:
            print(f"DEBUG: {bundle['label']} history is empty!")
            continue
            
        # 1. 确保 epochs 是纯数字列表
        epochs = []
        for item in history:
            try:
                epochs.append(int(item["epoch"]))
            except (KeyError, ValueError):
                continue
        
        for name, offset in metrics_map.items():
            ax = axes[metrics_map[name]]
            
            # 2. 提取每一个 epoch 的平均指标，并处理异常值
            val_curve = []
            for item in history:
                m = item.get("val_metrics", [])
                # 提取该指标在所有 step 的值 (例如 offset, offset+3, offset+6...)
                step_values = []
                for i in range(offset, len(m), 3):
                    val = m[i]
                    # 过滤掉 NaN, Inf 或非数字
                    if np.isfinite(val):
                        step_values.append(float(val))
                
                if step_values:
                    val_curve.append(np.mean(step_values))
                else:
                    val_curve.append(None) # 占位，保持长度一致

            # 3. 过滤掉 None 值用于绘图
            plot_epochs = [e for e, v in zip(epochs, val_curve) if v is not None]
            plot_values = [v for v in val_curve if v is not None]

            if plot_values:
                ax.plot(plot_epochs, plot_values, linewidth=2, label=bundle["label"], marker='.')
                print(f"DEBUG: Plotted {len(plot_values)} points for {bundle['label']} - {name}")
            else:
                print(f"DEBUG: No valid values for {bundle['label']} - {name}")

            ax.set_title(name)
            ax.set_xlabel("Epoch")
            ax.grid(alpha=0.3)
            
    axes[0].legend()
    fig.suptitle("Validation Dynamics Comparison")
    fig.tight_layout()
    
    save_path = out_dir / "compare_training_dynamics.png"
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"DEBUG: Save dynamic plot to {save_path}")
    plt.close(fig)


def plot_rollout_mae(bundles, out_dir):
    fig, ax = plt.subplots(figsize=(10, 5))
    for bundle in bundles:
        metrics = bundle["full_prediction"]["step_metrics"]
        steps = np.arange(1, len(metrics["MAE"]) + 1)
        ax.plot(steps, metrics["MAE"], marker="o", linewidth=2, label=bundle["label"])
    ax.set_xlabel("Autoregressive Step")
    ax.set_ylabel("MAE")
    ax.set_title("Rollout MAE Comparison")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "compare_rollout_mae.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_sensor_forecast(bundles, sensor_idx, time_points, out_dir):
    fig, ax = plt.subplots(figsize=(12, 5))
    first = bundles[0]["full_prediction"]
    target = first["targets"][0, : min(time_points, first["targets"].shape[1]), sensor_idx, 0]
    x_axis = np.arange(len(target))
    ax.plot(x_axis, target, color="#111827", linewidth=2.2, label="Ground Truth")
    for bundle in bundles:
        preds = bundle["full_prediction"]["preds_real"]
        pred = preds[0, : len(target), sensor_idx, 0]
        ax.plot(x_axis, pred, linewidth=1.8, label=bundle["label"])
    ax.set_title(f"Sensor {sensor_idx} Forecast Comparison")
    ax.set_xlabel("Sliding Test Window Index")
    ax.set_ylabel("Speed")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"compare_sensor_{sensor_idx}_forecast.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_relative_improvement(bundles, baseline_label, out_dir, mode):
    baseline = next((bundle for bundle in bundles if bundle["label"] == baseline_label), None)
    if baseline is None:
        raise ValueError(f"Unknown baseline_label: {baseline_label}")
    baseline_mae = np.array((collect_artifact_metrics(baseline) if mode == "artifact-only" else baseline["full_prediction"]["step_metrics"])["MAE"], dtype=float)
    fig, ax = plt.subplots(figsize=(10, 5))
    for bundle in bundles:
        if bundle["label"] == baseline_label:
            continue
        current_mae = np.array((collect_artifact_metrics(bundle) if mode == "artifact-only" else bundle["full_prediction"]["step_metrics"])["MAE"], dtype=float)
        improvement = (baseline_mae - current_mae) / np.maximum(baseline_mae, 1e-8) * 100.0
        ax.plot(np.arange(1, len(improvement) + 1), improvement, marker="o", linewidth=2, label=bundle["label"])
    ax.axhline(0.0, color="#6b7280", linestyle="--", linewidth=1)
    ax.set_xlabel("Forecast Horizon Step")
    ax.set_ylabel("MAE Improvement (%)")
    ax.set_title(f"Relative Improvement vs {baseline_label}")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "compare_relative_improvement.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    out_dir = ensure_dir(args.output_dir)
    labels = args.labels or []
    if labels and len(labels) != len(args.run_dir):
        raise ValueError("--labels count must match --run_dir count")

    bundles = [load_run_bundle(run_dir, labels[idx] if labels else None) for idx, run_dir in enumerate(args.run_dir)]
    mismatches = validate_compatibility(bundles, strict=args.mode == "full-prediction")

    device = torch.device(args.device)
    if args.mode == "full-prediction":
        bundles = [enrich_with_predictions(bundle, device, args.dataset_dir) for bundle in bundles]

    sensor_idx = choose_sensor(bundles, args.sensor_idx)
    summary = {
        "mode": args.mode,
        "warnings": [f"Compatibility mismatch for {item['label']}: {item['differences']}" for item in mismatches],
        "selected_sensor_idx": sensor_idx,
        "runs": [summarize_bundle(bundle, args.mode) for bundle in bundles],
    }
    write_summary_files(summary, out_dir)

    plot_horizon_metric(bundles, "MAE", out_dir, args.mode)
    plot_horizon_metric(bundles, "RMSE", out_dir, args.mode)
    plot_training_dynamics(bundles, out_dir)
    if args.mode == "full-prediction":
        plot_rollout_mae(bundles, out_dir)
        plot_sensor_forecast(bundles, sensor_idx, args.time_points, out_dir)
    if args.baseline_label is not None:
        plot_relative_improvement(bundles, args.baseline_label, out_dir, args.mode)

    print(f"Saved comparison outputs to {out_dir}")
    if mismatches and args.mode == "artifact-only":
        print("Warnings:")
        for item in summary["warnings"]:
            print(f"- {item}")


if __name__ == "__main__":
    main()
