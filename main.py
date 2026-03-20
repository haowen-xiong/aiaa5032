import argparse

from engine.runner import run_experiment


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="stgcn")
    parser.add_argument("--exp_name", type=str, default="default")
    parser.add_argument("--output_dir", type=str, default="./output/experiments")

    parser.add_argument("--n_route", type=int, default=228)
    parser.add_argument("--n_his", type=int, default=12)
    parser.add_argument("--n_pred", type=int, default=9)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--save", type=int, default=10)
    parser.add_argument("--ks", type=int, default=3)
    parser.add_argument("--kt", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--opt", type=str, default="RMSProp")
    parser.add_argument("--graph", type=str, default="default")
    parser.add_argument("--graph_approx", type=str, default="cheb")
    parser.add_argument("--inf_mode", type=str, default="merge")
    parser.add_argument("--dataset_dir", type=str, default="./dataset")
    parser.add_argument("--save_dir", type=str, default="")
    parser.add_argument("--sum_dir", type=str, default="")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--drop_prob", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--enable_tensorboard", type=str2bool, default=True)

    parser.add_argument("--use_spatial", type=str2bool, default=True)
    parser.add_argument("--direct_multi_step", type=str2bool, default=False)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--bidirectional", type=str2bool, default=False)
    parser.add_argument("--mlp_hidden_dims", type=str, default="128,64")
    return parser.parse_args()


if __name__ == "__main__":
    run_experiment(parse_args())
