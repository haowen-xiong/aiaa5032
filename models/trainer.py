import json
import time
from pathlib import Path

import numpy as np
import torch
try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:
    SummaryWriter = None

from data_loader.data_utils import gen_batch
from models.base_model import build_model, model_save, model_save_best
from models.tester import model_inference


def l2_loss(y_pred, y_true):
    return 0.5 * torch.sum((y_pred - y_true) ** 2)


def build_optimizer(args, model):
    opt = args.opt.upper()
    if opt == "RMSPROP":
        return torch.optim.RMSprop(model.parameters(), lr=args.lr)
    if opt == "ADAM":
        return torch.optim.Adam(model.parameters(), lr=args.lr)
    raise ValueError(f'ERROR: optimizer "{args.opt}" is not defined.')


def log_message(message, log_file=None):
    print(message)
    if log_file is not None:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(message + "\n")


def summarize_val_metric(metric_array):
    metric_array = np.asarray(metric_array, dtype=np.float64)
    return float(metric_array.mean())


def model_train(inputs, blocks, args, graph_kernel, device):
    n_his, n_pred = args.n_his, args.n_pred
    batch_size, epoch, inf_mode = args.batch_size, args.epoch, args.inf_mode

    model = build_model(args, blocks, graph_kernel, device)
    optimizer = build_optimizer(args, model)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    log_file = save_dir / "train.log"
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("")

    Path(args.sum_dir).mkdir(parents=True, exist_ok=True)
    writer = None
    if SummaryWriter is not None and args.enable_tensorboard:
        writer = SummaryWriter(log_dir=str(Path(args.sum_dir) / "train"))

    if inf_mode == "sep":
        step_idx = n_pred - 1
        tmp_idx = [step_idx]
        min_val = min_va_val = np.array([4e1, 1e5, 1e5], dtype=np.float64)
    elif inf_mode == "merge":
        step_idx = tmp_idx = np.arange(3, n_pred + 1, 3) - 1
        min_val = min_va_val = np.array([4e1, 1e5, 1e5] * len(step_idx), dtype=np.float64)
    else:
        raise ValueError(f'ERROR: test mode "{inf_mode}" is not defined.')

    run_meta = {
        "args": vars(args),
        "device": str(device),
        "train_samples": int(inputs.get_len("train")),
        "val_samples": int(inputs.get_len("val")),
        "test_samples": int(inputs.get_len("test")),
    }
    with open(save_dir / "run_meta.json", "w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2)

    global_step = 0
    history = []
    best_val_score = float("inf")
    best_epoch = 0
    for i in range(epoch):
        model.train()
        start_time = time.time()
        for j, x_batch in enumerate(gen_batch(inputs.get_data("train"), batch_size, dynamic_batch=True, shuffle=True)):
            x_tensor = torch.as_tensor(x_batch[:, 0:n_his + 1, :, :], dtype=torch.float32, device=device)
            target = x_tensor[:, n_his, :, :]
            copy_target = x_tensor[:, n_his - 1, :, :]

            optimizer.zero_grad(set_to_none=True)
            pred = model(x_tensor)
            train_loss = l2_loss(pred, target)
            train_loss.backward()
            optimizer.step()

            copy_loss = l2_loss(copy_target, target)
            if writer is not None:
                writer.add_scalar("train_loss", train_loss.item(), global_step)
                writer.add_scalar("copy_loss", copy_loss.item(), global_step)
                writer.add_scalar("learning_rate", optimizer.param_groups[0]["lr"], global_step)

            if j % 50 == 0:
                log_message(f"Epoch {i:2d}, Step {j:3d}: [{train_loss.item():.3f}, {copy_loss.item():.3f}]", log_file)
            global_step += 1

        scheduler.step()
        train_time = time.time() - start_time
        log_message(f"Epoch {i:2d} Training Time {train_time:.3f}s", log_file)

        start_time = time.time()
        min_va_val, min_val = model_inference(
            model, inputs, batch_size, n_his, n_pred, step_idx, min_va_val, min_val, device
        )
        infer_time = time.time() - start_time
        for ix in tmp_idx:
            va, te = min_va_val[ix - 2:ix + 1], min_val[ix - 2:ix + 1]
            log_message(
                f"Time Step {ix + 1}: "
                f"MAPE {va[0]:7.3%}, {te[0]:7.3%}; "
                f"MAE  {va[1]:4.3f}, {te[1]:4.3f}; "
                f"RMSE {va[2]:6.3f}, {te[2]:6.3f}.",
                log_file,
            )
        log_message(f"Epoch {i:2d} Inference Time {infer_time:.3f}s", log_file)

        history.append(
            {
                "epoch": i + 1,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "train_time_sec": train_time,
                "infer_time_sec": infer_time,
                "val_metrics": min_va_val.tolist(),
                "test_metrics": min_val.tolist(),
            }
        )

        current_val_score = summarize_val_metric(min_va_val)
        if current_val_score < best_val_score:
            best_val_score = current_val_score
            best_epoch = i + 1
            model_save_best(model, optimizer, i + 1, args, best_val_score, args.save_dir)
            log_message(
                f"Best model updated at epoch {i + 1}: validation score {best_val_score:.6f}",
                log_file,
            )

        if (i + 1) % args.save == 0:
            model_save(model, optimizer, i + 1, args, args.save_dir)

    if args.epoch % args.save != 0:
        model_save(model, optimizer, args.epoch, args, args.save_dir)

    with open(save_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    with open(save_dir / "best_meta.json", "w", encoding="utf-8") as f:
        json.dump({"best_epoch": best_epoch, "best_val_score": best_val_score}, f, indent=2)

    if writer is not None:
        writer.close()
    log_message("Training model finished!", log_file)
