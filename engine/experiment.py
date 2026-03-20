import json
import time
from pathlib import Path

import numpy as np
import torch

from data_loader.data_utils import gen_batch
from utils.math_utils import evaluation

try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:
    SummaryWriter = None


def log_message(message, log_file=None):
    print(message)
    if log_file is not None:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(message + "\n")


def l2_loss(y_pred, y_true):
    return 0.5 * torch.sum((y_pred - y_true) ** 2)


def build_optimizer(args, model):
    opt = args.opt.upper()
    if opt == "RMSPROP":
        return torch.optim.RMSprop(model.parameters(), lr=args.lr)
    if opt == "ADAM":
        return torch.optim.Adam(model.parameters(), lr=args.lr)
    raise ValueError(f'ERROR: optimizer "{args.opt}" is not defined.')


def resolve_checkpoint(load_path):
    load_dir = Path(load_path)
    for candidate in [load_dir / "best.pt", load_dir / "latest.pt"]:
        if candidate.exists():
            return candidate
    candidates = sorted(load_dir.glob("epoch_*.pt"))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint found under {load_path}.")
    return candidates[-1]


def save_checkpoint(model, optimizer, epoch, args, save_dir, is_best=False, best_metric=None):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "best_metric": best_metric,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "args": vars(args),
    }
    torch.save(payload, save_dir / "latest.pt")
    torch.save(payload, save_dir / f"epoch_{epoch}.pt")
    if is_best:
        torch.save(payload, save_dir / "best.pt")


def select_history_and_target(x_batch, args, device):
    if getattr(args, "direct_multi_step", False):
        seq = torch.as_tensor(x_batch[:, 0 : args.n_his + args.n_pred, :, :], dtype=torch.float32, device=device)
        history = seq[:, 0 : args.n_his, :, :]
        target = seq[:, args.n_his : args.n_his + args.n_pred, :, :]
    else:
        seq = torch.as_tensor(x_batch[:, 0 : args.n_his + 1, :, :], dtype=torch.float32, device=device)
        history = seq[:, 0 : args.n_his, :, :]
        target = seq[:, args.n_his, :, :]
    copy_target = history[:, -1, :, :]
    return history, target, copy_target


def model_predict(model, history, args):
    return model(history)


@torch.no_grad()
def multi_step_predict(model, seq, batch_size, n_his, n_pred, step_idx, device, args, dynamic_batch=True):
    pred_list = []
    model.eval()
    for batch in gen_batch(seq, min(batch_size, len(seq)), dynamic_batch=dynamic_batch):
        history = np.copy(batch[:, 0:n_his, :, :])
        if getattr(args, "direct_multi_step", False):
            pred = model(torch.as_tensor(history, dtype=torch.float32, device=device)).cpu().numpy()
            pred_list.append(pred)
            continue
        step_list = []
        test_seq = history
        for _ in range(n_pred):
            pred = model(torch.as_tensor(test_seq, dtype=torch.float32, device=device)).cpu().numpy()
            test_seq[:, 0:n_his - 1, :, :] = test_seq[:, 1:n_his, :, :]
            test_seq[:, n_his - 1, :, :] = pred
            step_list.append(pred)
        pred_list.append(step_list)

    if getattr(args, "direct_multi_step", False):
        pred_array = np.concatenate(pred_list, axis=0)
        pred_array = np.transpose(pred_array, (1, 0, 2, 3))
    else:
        pred_array = np.concatenate(pred_list, axis=1)
    return pred_array[step_idx], pred_array.shape[1]


def evaluate_split(model, split_data, x_stats, batch_size, args, device, step_idx):
    y_pred, seq_len = multi_step_predict(model, split_data, batch_size, args.n_his, args.n_pred, step_idx, device, args)
    evl = evaluation(split_data[0:seq_len, step_idx + args.n_his, :, :], y_pred, x_stats)
    return evl


def summarize_metric(metric_array, metric_name="mae"):
    metric_array = np.asarray(metric_array, dtype=np.float64)
    offset = {"mape": 0, "mae": 1, "rmse": 2}[metric_name.lower()]
    return float(metric_array[offset::3].mean())


def train_and_test_model(model, dataset, args, device, save_dir):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    log_file = save_dir / "train.log"
    log_file.write_text("", encoding="utf-8")

    writer = None
    if SummaryWriter is not None and args.enable_tensorboard:
        writer = SummaryWriter(log_dir=str(Path(args.sum_dir)))

    trainable = any(p.requires_grad for p in model.parameters())
    optimizer = build_optimizer(args, model) if trainable else None
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7) if optimizer is not None else None

    if args.inf_mode == "sep":
        step_idx = args.n_pred - 1
        tmp_idx = [step_idx]
    elif args.inf_mode == "merge":
        step_idx = tmp_idx = np.arange(3, args.n_pred + 1, 3) - 1
    else:
        raise ValueError(f'ERROR: test mode "{args.inf_mode}" is not defined.')

    run_meta = {
        "args": vars(args),
        "device": str(device),
        "train_samples": int(dataset.get_len("train")),
        "val_samples": int(dataset.get_len("val")),
        "test_samples": int(dataset.get_len("test")),
    }
    with open(save_dir / "run_meta.json", "w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2)

    global_step = 0
    history = []
    best_val_score = float("inf")
    best_epoch = 0
    x_stats = dataset.get_stats()

    for i in range(args.epoch):
        model.train()
        start_time = time.time()
        for j, x_batch in enumerate(gen_batch(dataset.get_data("train"), args.batch_size, dynamic_batch=True, shuffle=True)):
            history_tensor, target, copy_target = select_history_and_target(x_batch, args, device)

            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            pred = model_predict(model, history_tensor, args)
            train_loss = l2_loss(pred, target)
            if optimizer is not None:
                train_loss.backward()
                optimizer.step()

            ref_target = target[:, 0, :, :] if getattr(args, "direct_multi_step", False) else target
            copy_loss = l2_loss(copy_target, ref_target)
            if writer is not None and optimizer is not None:
                writer.add_scalar("train_loss", train_loss.item(), global_step)
                writer.add_scalar("copy_loss", copy_loss.item(), global_step)
                writer.add_scalar("learning_rate", optimizer.param_groups[0]["lr"], global_step)

            if j % 50 == 0:
                log_message(f"Epoch {i:2d}, Step {j:3d}: [{train_loss.item():.3f}, {copy_loss.item():.3f}]", log_file)
            global_step += 1

        if scheduler is not None:
            scheduler.step()
        train_time = time.time() - start_time
        log_message(f"Epoch {i:2d} Training Time {train_time:.3f}s", log_file)

        start_time = time.time()
        evl_val = evaluate_split(model, dataset.get_data("val"), x_stats, args.batch_size, args, device, step_idx)
        evl_test = evaluate_split(model, dataset.get_data("test"), x_stats, args.batch_size, args, device, step_idx)
        infer_time = time.time() - start_time

        for ix in tmp_idx:
            va, te = evl_val[ix - 2:ix + 1], evl_test[ix - 2:ix + 1]
            log_message(
                f"Time Step {ix + 1}: MAPE {va[0]:7.3%}, {te[0]:7.3%}; MAE  {va[1]:4.3f}, {te[1]:4.3f}; RMSE {va[2]:6.3f}, {te[2]:6.3f}.",
                log_file,
            )
        log_message(f"Epoch {i:2d} Inference Time {infer_time:.3f}s", log_file)

        history.append(
            {
                "epoch": i + 1,
                "learning_rate": optimizer.param_groups[0]["lr"] if optimizer is not None else 0.0,
                "train_time_sec": train_time,
                "infer_time_sec": infer_time,
                "val_metrics": evl_val.tolist(),
                "test_metrics": evl_test.tolist(),
            }
        )

        current_val_score = summarize_metric(evl_val, metric_name="mae")
        if current_val_score < best_val_score:
            best_val_score = current_val_score
            best_epoch = i + 1
            save_checkpoint(model, optimizer, i + 1, args, save_dir, is_best=True, best_metric=best_val_score)
            log_message(f"Best model updated at epoch {i + 1}: validation score {best_val_score:.6f}", log_file)
        elif optimizer is not None and (i + 1) % args.save == 0:
            save_checkpoint(model, optimizer, i + 1, args, save_dir)

    if optimizer is not None and args.epoch % args.save != 0:
        save_checkpoint(model, optimizer, args.epoch, args, save_dir)

    with open(save_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    with open(save_dir / "best_meta.json", "w", encoding="utf-8") as f:
        json.dump({"best_epoch": best_epoch, "best_val_score": best_val_score}, f, indent=2)
    if writer is not None:
        writer.close()

    test_model(model, dataset, args, device, save_dir)


def test_model(model, dataset, args, device, save_dir):
    start_time = time.time()
    save_dir = Path(save_dir)
    log_file = save_dir / "test.log"
    log_file.write_text("", encoding="utf-8")

    ckpt_path = resolve_checkpoint(save_dir)
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    log_message(f">> Loading saved model from {ckpt_path} ...", log_file)

    if args.inf_mode == "sep":
        step_idx = args.n_pred - 1
        tmp_idx = [step_idx]
    elif args.inf_mode == "merge":
        step_idx = tmp_idx = np.arange(3, args.n_pred + 1, 3) - 1
    else:
        raise ValueError(f'ERROR: test mode "{args.inf_mode}" is not defined.')

    x_test, x_stats = dataset.get_data("test"), dataset.get_stats()
    y_test, len_test = multi_step_predict(model, x_test, args.batch_size, args.n_his, args.n_pred, step_idx, device, args)
    evl = evaluation(x_test[0:len_test, step_idx + args.n_his, :, :], y_test, x_stats)

    result = {"checkpoint": str(ckpt_path), "test_metrics": {}}
    for ix in tmp_idx:
        te = evl[ix - 2:ix + 1]
        result["test_metrics"][f"step_{ix + 1}"] = {"MAPE": float(te[0]), "MAE": float(te[1]), "RMSE": float(te[2])}
        log_message(f"Time Step {ix + 1}: MAPE {te[0]:7.3%}; MAE  {te[1]:4.3f}; RMSE {te[2]:6.3f}.", log_file)
    log_message(f"Model Test Time {time.time() - start_time:.3f}s", log_file)
    with open(save_dir / "test_results.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    log_message("Testing model finished!", log_file)
