# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development commands

### Environment setup
```bash
pip install -r requirements.txt
```

### Run training
Primary entrypoint:
```bash
python main.py --model_name stgcn --exp_name baseline --n_route 228 --epoch 50 --batch_size 16 --device cuda
```

Fast CPU smoke test:
```bash
python main.py --model_name stgcn --exp_name smoke --n_route 228 --epoch 1 --batch_size 8 --device cpu
```

Baseline examples:
```bash
python main.py --model_name persistence --exp_name persistence_run --n_route 228 --epoch 1 --device cpu
python main.py --model_name temporal_mlp --exp_name mlp_run --n_route 228 --epoch 50 --batch_size 32 --device cuda
python main.py --model_name lstm --exp_name lstm_run --n_route 228 --epoch 50 --batch_size 32 --device cuda
```

Ablation examples:
```bash
python main.py --model_name stgcn --exp_name ablation_first --graph_approx first --n_route 228 --epoch 50 --batch_size 16 --device cuda
python main.py --model_name stgcn --exp_name ablation_no_spatial --use_spatial false --n_route 228 --epoch 50 --batch_size 16 --device cuda
python main.py --model_name temporal_mlp --exp_name direct_mlp --direct_multi_step true --n_route 228 --epoch 50 --batch_size 32 --device cuda
```

### Generate visualizations
```bash
python scripts/visualize_results.py \
  --run_meta output/experiments/baseline/stgcn/run_meta.json \
  --history output/experiments/baseline/stgcn/history.json \
  --test_results output/experiments/baseline/stgcn/test_results.json \
  --checkpoint_dir output/experiments/baseline/stgcn \
  --output_dir output/visualizations
```

### Tests / lint / formatting
No dedicated test, lint, or formatter configuration was found in the repository. For validation, use a smoke training run and, when relevant, rerun visualization generation against the produced artifacts.

Closest equivalent to a single-test run:
```bash
python main.py --model_name stgcn --exp_name smoke --n_route 228 --epoch 1 --batch_size 8 --device cpu
```

## High-level architecture

### Main execution flow
- `main.py` parses CLI flags and calls `engine.runner.run_experiment(...)`.
- `engine/runner.py` is the orchestration layer: it sets the seed/device, builds experiment output paths, loads the dataset, constructs the graph kernel, resolves the selected model runtime, writes `experiment_manifest.json`, then calls training/testing.
- `engine/experiment.py` contains the shared training loop, validation logic, checkpointing, metric/history logging, and final test evaluation.

### Model system
The repository is structured around a runtime registry rather than hardcoding a single model path.

- `engine/model_registry.py` defines `ModelRuntime` and lazy-loads built-in runtimes.
- `models/stgcn/runtime.py` registers the graph model.
- `models/baselines/runtime.py` registers baseline families (`persistence`, `temporal_mlp`, `lstm`).
- To add a new model family, follow the same pattern: expose a `MODEL_RUNTIME` with a `build_fn`, then ensure the runtime module is loaded by the registry.

### Data and graph pipeline
- `engine/data.py` resolves dataset files from either `dataset/` or `dataset/PeMSD7_Full/`.
- Traffic values come from `PeMSD7_V_<n_route>.csv`; graph weights come from `PeMSD7_W_<n_route>.csv` unless `--graph` overrides the file.
- Dataset splitting is fixed in code to 34 train days, 5 validation days, and 5 test days.
- `data_loader/data_utils.py` converts the raw CSV into sliding windows of shape `[samples, n_frame, n_route, channels]`, where `n_frame = n_his + n_pred`.
- Normalization uses train-set z-score stats and stores them on the dataset object.
- `utils/math_graph.py` builds either the Chebyshev graph kernel (`--graph_approx cheb`) or first-order approximation (`--graph_approx first`).

### Shared training contract
The training loop in `engine/experiment.py` expects all models to accept history shaped like `[batch, time, nodes, channels]` and return:
- `[batch, nodes, 1]` for one-step / autoregressive mode, or
- `[batch, n_pred, nodes, 1]` when `--direct_multi_step true`.

This output contract is what lets STGCN and non-graph baselines reuse the same train/eval pipeline.

### STGCN implementation
- `models/stgcn/model.py` assembles the model from repeated `STConvBlock`s plus either `OutputLayer` or `DirectMultiStepOutput`.
- `models/stgcn/config.py` holds the default block specification tuple `((1, 32, 64), (64, 32, 128))`.
- `models/stgcn/layers.py` contains the actual temporal convolution, graph convolution, normalization, and output heads.
- `--use_spatial false` swaps the spatial graph step for an identity layer, which is the intended “remove spatial convolution” ablation.
- The temporal receptive field is sensitive to `n_his`, `Kt`, and the number of ST blocks; invalid combinations fail when the computed output kernel length becomes too small.

### Baseline implementation
- `models/baselines/factory.py` maps model names to implementation classes.
- `persistence` is parameter-free and repeats the last observed value.
- `temporal_mlp` flattens each node’s time/channel history independently and predicts per-node outputs.
- `lstm` also operates per node by reshaping `[batch, nodes]` into the batch dimension before sequence modeling.
- Baselines intentionally do not use the graph kernel even though the shared runner still builds one.

### Outputs and artifacts
Each run writes to:
```text
output/experiments/<exp_name>/<model_name>/
```

Important artifacts produced by the shared runner:
- `experiment_manifest.json`: high-level run metadata and runtime capabilities
- `run_meta.json`: serialized CLI args, device, split sizes
- `history.json`: per-epoch training/validation/test summaries
- `best_meta.json`: best epoch and validation score
- `best.pt`, `latest.pt`, `epoch_*.pt`: checkpoints
- `train.log`, `test.log`, `test_results.json`
- `tensorboard/` if TensorBoard is enabled and available

### Visualization flow
`scripts/visualize_results.py` reconstructs the model from `run_meta.json` + checkpoint, reloads the dataset, recomputes predictions for all forecast steps, and generates plots plus `visualization_summary.json`. It depends on the same runtime registry and graph/data loaders as training, so changes to model IO shape or runtime registration usually also require updating the visualization path.

## Repository-specific notes
- The maintained code path is `main.py`, `engine/`, `models/stgcn/`, `models/baselines/`, and `scripts/visualize_results.py`.
- This is a PyTorch-only codebase; older TensorFlow-era structure is not the active path.
- Boolean CLI flags use string parsing (`true/false`, `yes/no`, `1/0`) via `str2bool` in `main.py`, so when changing docs or commands keep boolean examples in that style.
- `engine/experiment.py` uses MAE averaged across forecast horizons as the criterion for `best.pt` selection.
