import json
import time
from pathlib import Path

import numpy as np
import torch

from data_loader.data_utils import gen_batch
from models.base_model import build_model
from utils.math_utils import evaluation


@torch.no_grad()
def multi_pred(model, seq, batch_size, n_his, n_pred, step_idx, device, dynamic_batch=True):
    pred_list = []
    model.eval()
    for batch in gen_batch(seq, min(batch_size, len(seq)), dynamic_batch=dynamic_batch):
        test_seq = np.copy(batch[:, 0:n_his + 1, :, :])
        step_list = []
        for _ in range(n_pred):
            pred = model(torch.as_tensor(test_seq, dtype=torch.float32, device=device)).cpu().numpy()
            test_seq[:, 0:n_his - 1, :, :] = test_seq[:, 1:n_his, :, :]
            test_seq[:, n_his - 1, :, :] = pred
            step_list.append(pred)
        pred_list.append(step_list)
    pred_array = np.concatenate(pred_list, axis=1)
    return pred_array[step_idx], pred_array.shape[1]


def model_inference(model, inputs, batch_size, n_his, n_pred, step_idx, min_va_val, min_val, device):
    x_val, x_test, x_stats = inputs.get_data("val"), inputs.get_data("test"), inputs.get_stats()

    if n_his + n_pred > x_val.shape[1]:
        raise ValueError(f'ERROR: the value of n_pred "{n_pred}" exceeds the length limit.')

    y_val, len_val = multi_pred(model, x_val, batch_size, n_his, n_pred, step_idx, device)
    evl_val = evaluation(x_val[0:len_val, step_idx + n_his, :, :], y_val, x_stats)

    chks = evl_val < min_va_val
    if np.sum(chks):
        min_va_val[chks] = evl_val[chks]
        y_pred, len_pred = multi_pred(model, x_test, batch_size, n_his, n_pred, step_idx, device)
        min_val = evaluation(x_test[0:len_pred, step_idx + n_his, :, :], y_pred, x_stats)
    return min_va_val, min_val


def resolve_checkpoint(load_path):
    load_dir = Path(load_path)
    best = load_dir / "STGCN_best.pt"
    if best.exists():
        return best
    latest = load_dir / "STGCN_latest.pt"
    if latest.exists():
        return latest
    candidates = sorted(load_dir.glob("STGCN_epoch_*.pt"))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint found under {load_path}.")
    return candidates[-1]


def log_message(message, log_file=None):
    print(message)
    if log_file is not None:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(message + "\n")


def model_test(inputs, blocks, args, graph_kernel, device, load_path=None):
    start_time = time.time()
    save_dir = Path(load_path or args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    log_file = save_dir / "test.log"
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("")

    ckpt_path = resolve_checkpoint(save_dir)
    checkpoint = torch.load(ckpt_path, map_location=device)

    model = build_model(args, blocks, graph_kernel, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    log_message(f">> Loading saved model from {ckpt_path} ...", log_file)

    if args.inf_mode == "sep":
        step_idx = args.n_pred - 1
        tmp_idx = [step_idx]
    elif args.inf_mode == "merge":
        step_idx = tmp_idx = np.arange(3, args.n_pred + 1, 3) - 1
    else:
        raise ValueError(f'ERROR: test mode "{args.inf_mode}" is not defined.')

    x_test, x_stats = inputs.get_data("test"), inputs.get_stats()
    y_test, len_test = multi_pred(model, x_test, args.batch_size, args.n_his, args.n_pred, step_idx, device)
    evl = evaluation(x_test[0:len_test, step_idx + args.n_his, :, :], y_test, x_stats)

    result = {"checkpoint": str(ckpt_path), "test_metrics": {}}
    for ix in tmp_idx:
        te = evl[ix - 2:ix + 1]
        result["test_metrics"][f"step_{ix + 1}"] = {
            "MAPE": float(te[0]),
            "MAE": float(te[1]),
            "RMSE": float(te[2]),
        }
        log_message(f"Time Step {ix + 1}: MAPE {te[0]:7.3%}; MAE  {te[1]:4.3f}; RMSE {te[2]:6.3f}.", log_file)
    log_message(f"Model Test Time {time.time() - start_time:.3f}s", log_file)
    with open(save_dir / "test_results.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    log_message("Testing model finished!", log_file)
