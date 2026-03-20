# Group Project Plan for STGCN Traffic Forecasting

This project is suitable for a group assignment only if the scope goes beyond a single-model reproduction. A strong version should include reproduction, baselines, ablations, efficiency analysis, and visualization.

## Suggested Scope

- Reproduce the PyTorch STGCN baseline on PeMSD7(M).
- Add at least 3 simple baselines.
- Run at least 2 core ablations.
- Compare Chebyshev graph convolution with first-order approximation.
- Add one scaling experiment on PeMSD7(L).
- Produce a compact but complete visualization package.

## Experiment Matrix

### Main Models

- `stgcn` with Chebyshev graph convolution.
- `stgcn` with first-order graph approximation.
- `persistence`.
- `lstm`.
- `temporal_mlp`.
- `baseline_gcn_only` if time permits.

### Core Ablations

- `n_his = 6 / 12 / 18`.
- `Ks = 2 / 3 / 5`.
- `Kt = 2 / 3 / 4`.
- `rollout` versus `direct multi-step` if implemented.
- `with graph` versus `without graph`.

### Scaling and Cost

- PeMSD7(M), 228 nodes.
- PeMSD7(L), 1026 nodes.
- Train time per epoch.
- Inference time.
- Parameter count.
- GPU memory if available.

## 3-Person Split

### Member A

- Reproduction and experiment management.
- Unified CLI, output paths, seed control, checkpointing.
- Make sure all runs are reproducible.

### Member B

- Baselines and comparison runs.
- Implement and evaluate simple temporal baselines.
- Keep the evaluation interface aligned with STGCN.

### Member C

- Ablation and visualization.
- Run graph approximation and hyperparameter sweeps.
- Generate plots and interpret the results.

## 4-Person Split

### Member A

- Core pipeline and runtime management.

### Member B

- Baseline models.

### Member C

- Ablation studies.

### Member D

- Visualization, reporting, and slide deck.

## 5-Person Split

### Member A

- Core pipeline and experiment registry.

### Member B

- Baseline models.

### Member C

- Graph-related ablations.

### Member D

- Direct multi-step and efficiency analysis.

### Member E

- Visualization and final presentation.

## Deliverables

- A reproducible codebase with per-model output folders.
- A short results table and a few key figures.
- A report section explaining why graph structure matters.
- A comparison section showing where STGCN helps and where it fails.

## Recommended Folder Layout

- `output/experiments/<exp_name>/<model_name>/`
- `output/experiments/<exp_name>/<model_name>/tensorboard/`
- `output/experiments/<exp_name>/<model_name>/experiment_manifest.json`
