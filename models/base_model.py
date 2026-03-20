from pathlib import Path

import torch
import torch.nn as nn

from models.layers import OutputLayer, STConvBlock


class STGCN(nn.Module):
    def __init__(self, n_his, Ks, Kt, blocks, n_route, graph_kernel, drop_prob=0.0):
        super().__init__()
        self.n_his = n_his
        self.blocks = nn.ModuleList(
            [
                STConvBlock(Ks, Kt, channels, n_route, graph_kernel, drop_prob=drop_prob, act_func="GLU")
                for channels in blocks
            ]
        )

        Ko = n_his
        for _ in blocks:
            Ko -= 2 * (Kt - 1)
        if Ko <= 1:
            raise ValueError(f'ERROR: kernel size Ko must be greater than 1, but received "{Ko}".')
        self.output = OutputLayer(Ko, n_route, blocks[-1][-1], act_func="GLU")

    def forward(self, inputs):
        x = inputs[:, 0:self.n_his, :, :]
        for block in self.blocks:
            x = block(x)
        y = self.output(x)
        return y[:, 0, :, :]


def build_model(args, blocks, graph_kernel, device):
    graph_kernel = torch.as_tensor(graph_kernel, dtype=torch.float32, device=device)
    return STGCN(
        n_his=args.n_his,
        Ks=args.ks,
        Kt=args.kt,
        blocks=blocks,
        n_route=args.n_route,
        graph_kernel=graph_kernel,
        drop_prob=args.drop_prob,
    ).to(device)


def model_save(model, optimizer, epoch, args, save_path="./output/models"):
    save_dir = Path(save_path)
    save_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "args": vars(args),
    }
    latest_path = save_dir / "STGCN_latest.pt"
    epoch_path = save_dir / f"STGCN_epoch_{epoch}.pt"
    torch.save(payload, latest_path)
    torch.save(payload, epoch_path)
    print(f"<< Saving model to {epoch_path} ...")


def model_save_best(model, optimizer, epoch, args, best_metric, save_path="./output/models"):
    save_dir = Path(save_path)
    save_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "epoch": epoch,
        "best_metric": best_metric,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "args": vars(args),
    }
    best_path = save_dir / "STGCN_best.pt"
    torch.save(payload, best_path)
    print(f"<< Saving best model to {best_path} ...")
