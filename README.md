# STGCN Traffic Forecasting in PyTorch

PyTorch reimplementation and coursework-oriented extension of STGCN for traffic forecasting on PeMSD7.

This repository provides a clean PyTorch experiment framework for spatio-temporal traffic prediction, with:

- `stgcn`: graph-based spatio-temporal forecasting model
- `persistence`: last-value baseline
- `temporal_mlp`: non-graph temporal baseline
- `lstm`: recurrent baseline
- unified experiment runner
- per-model experiment folders
- visualization tools for curves, forecasts, and heatmaps
- group project planning support

## Highlights

- Pure PyTorch codebase
- No TensorFlow dependency
- Multiple model families under separate folders
- Ablation-ready STGCN implementation
- Direct support for group coursework extension

## Repository Layout

```text
.
|-- data_loader/
|-- dataset/
|-- engine/
|-- models/
|   |-- baselines/
|   `-- stgcn/
|-- output/
|   |-- experiments/
|   `-- visualizations/
|-- scripts/
|-- utils/
|-- group_project_plan.md
|-- main.py
|-- requirements.txt
`-- README.md
```

## Install

```bash
pip install -r requirements.txt
```

Main dependencies:

- PyTorch
- NumPy
- SciPy
- Pandas
- Matplotlib
- TensorBoard

## Dataset

Supported data locations:

- `dataset/PeMSD7_V_228.csv`
- `dataset/PeMSD7_W_228.csv`
- `dataset/PeMSD7_V_1026.csv`
- `dataset/PeMSD7_W_1026.csv`

or:

- `dataset/PeMSD7_Full/PeMSD7_V_228.csv`
- `dataset/PeMSD7_Full/PeMSD7_W_228.csv`
- `dataset/PeMSD7_Full/PeMSD7_V_1026.csv`
- `dataset/PeMSD7_Full/PeMSD7_W_1026.csv`

Default split:

- train: 34 days
- validation: 5 days
- test: 5 days

## Quick Start

### STGCN baseline

```bash
python main.py --model_name stgcn --exp_name baseline --n_route 228 --epoch 50 --batch_size 16 --device cuda
```

### Fast smoke test

```bash
python main.py --model_name stgcn --exp_name smoke --n_route 228 --epoch 1 --batch_size 8 --device cpu
```

### Baseline examples

```bash
python main.py --model_name persistence --exp_name persistence_run --n_route 228 --epoch 1 --device cpu
python main.py --model_name temporal_mlp --exp_name mlp_run --n_route 228 --epoch 50 --batch_size 32 --device cuda
python main.py --model_name lstm --exp_name lstm_run --n_route 228 --epoch 50 --batch_size 32 --device cuda
```

## STGCN Ablations

### First-order graph approximation

```bash
python main.py --model_name stgcn --exp_name ablation_first --graph_approx first --n_route 228 --epoch 50 --batch_size 16 --device cuda
```

### Remove spatial graph convolution

```bash
python main.py --model_name stgcn --exp_name ablation_no_spatial --use_spatial false --n_route 228 --epoch 50 --batch_size 16 --device cuda
```

### Direct multi-step prediction

```bash
python main.py --model_name temporal_mlp --exp_name direct_mlp --direct_multi_step true --n_route 228 --epoch 50 --batch_size 32 --device cuda
```

## Output Structure

Each run writes to:

```text
output/experiments/<exp_name>/<model_name>/
```

Typical outputs:

- `best.pt`
- `latest.pt`
- `epoch_*.pt`
- `train.log`
- `test.log`
- `history.json`
- `test_results.json`
- `run_meta.json`
- `best_meta.json`
- `experiment_manifest.json`
- `tensorboard/`

## Visualization

Generate plots from an experiment directory:

```bash
python scripts/visualize_results.py   --run_meta output/experiments/baseline/stgcn/run_meta.json   --history output/experiments/baseline/stgcn/history.json   --test_results output/experiments/baseline/stgcn/test_results.json   --checkpoint_dir output/experiments/baseline/stgcn   --output_dir output/visualizations
```

Supported figures:

- training curves
- forecast error by horizon
- rollout error curve
- single-sensor forecast plot
- spatio-temporal error heatmaps
- adjacency matrix heatmap

## Recommended Group Work Scope

If this repository is used for group coursework, a strong project version should include:

- STGCN reproduction
- 2 to 3 baseline comparisons
- 2 or more ablations
- one scaling experiment on PeMSD7(L)
- visualization and result interpretation

See [group_project_plan.md](./group_project_plan.md) for suggested team split and experiment matrix.

## Active Code Path

The maintained code path is:

- `main.py`
- `engine/`
- `models/stgcn/`
- `models/baselines/`
- `scripts/visualize_results.py`

## Citation

If you use the original STGCN method, please cite:

```bibtex
@inproceedings{yu2018spatio,
  title={Spatio-temporal Graph Convolutional Networks: A Deep Learning Framework for Traffic Forecasting},
  author={Yu, Bing and Yin, Haoteng and Zhu, Zhanxing},
  booktitle={Proceedings of the 27th International Joint Conference on Artificial Intelligence},
  year={2018}
}
```
