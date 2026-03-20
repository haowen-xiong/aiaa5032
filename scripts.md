# CPU 运行指南

## 1. 安装依赖

在仓库根目录执行：

```bash
pip install -r requirements.txt
```

如果你用的是虚拟环境，先激活再安装。

---

## 2. 最基础的 CPU 烟雾测试

### STGCN
```bash
python main.py --model_name stgcn --exp_name smoke_stgcn_cpu --n_route 228 --epoch 1 --batch_size 8 --device cpu
```

### GAT
```bash
python main.py --model_name gat --exp_name smoke_gat_cpu --n_route 228 --epoch 1 --batch_size 8 --device cpu
```

### GraphSAGE
```bash
python main.py --model_name graphsage --exp_name smoke_sage_cpu --n_route 228 --epoch 1 --batch_size 8 --device cpu
```

### Temporal MLP
```bash
python main.py --model_name temporal_mlp --exp_name smoke_mlp_cpu --n_route 228 --epoch 1 --batch_size 8 --device cpu
```

### LSTM
```bash
python main.py --model_name lstm --exp_name smoke_lstm_cpu --n_route 228 --epoch 1 --batch_size 8 --device cpu
```

---

## 3. Direct multi-step 模式

如果你想测试直接多步预测：

### GAT
```bash
python main.py --model_name gat --exp_name gat_direct_cpu --n_route 228 --epoch 1 --batch_size 8 --device cpu --direct_multi_step true
```

### GraphSAGE
```bash
python main.py --model_name graphsage --exp_name sage_direct_cpu --n_route 228 --epoch 1 --batch_size 8 --device cpu --direct_multi_step true
```

### STGCN
```bash
python main.py --model_name stgcn --exp_name stgcn_direct_cpu --n_route 228 --epoch 1 --batch_size 8 --device cpu --direct_multi_step true
```

---

## 4. 图模型常用 CPU 参数

### GAT
```bash
python main.py \
  --model_name gat \
  --exp_name gat_cpu \
  --n_route 228 \
  --epoch 1 \
  --batch_size 8 \
  --device cpu \
  --graph_hidden_dim 64 \
  --graph_num_layers 2 \
  --graph_dropout 0.1 \
  --graph_input_dropout 0.0 \
  --graph_residual true \
  --graph_self_loops true \
  --gat_heads 2 \
  --gat_concat_heads true \
  --gat_leaky_relu_slope 0.2 \
  --gat_attention_dropout 0.0
```

### GraphSAGE
```bash
python main.py \
  --model_name graphsage \
  --exp_name sage_cpu \
  --n_route 228 \
  --epoch 1 \
  --batch_size 8 \
  --device cpu \
  --graph_hidden_dim 64 \
  --graph_num_layers 2 \
  --graph_dropout 0.1 \
  --graph_input_dropout 0.0 \
  --graph_residual true \
  --graph_self_loops true \
  --sage_aggregator mean \
  --sage_normalize_embeddings false
```

---

## 5. 输出文件位置

每次运行结果会写到：

```text
output/experiments/<exp_name>/<model_name>/
```

例如：

```text
output/experiments/smoke_gat_cpu/gat/
```

常见文件有：

- `run_meta.json`
- `history.json`
- `best_meta.json`
- `best.pt`
- `latest.pt`
- `test_results.json`
- `train.log`
- `test.log`

---

## 6. 单个实验可视化

比如可视化一个 STGCN run：

```bash
python scripts/visualize_results.py \
  --run_meta output/experiments/smoke_stgcn_cpu/stgcn/run_meta.json \
  --history output/experiments/smoke_stgcn_cpu/stgcn/history.json \
  --test_results output/experiments/smoke_stgcn_cpu/stgcn/test_results.json \
  --checkpoint_dir output/experiments/smoke_stgcn_cpu/stgcn \
  --output_dir output/visualizations/stgcn_cpu \
  --device cpu
```

GAT 示例：

```bash
python scripts/visualize_results.py \
  --run_meta output/experiments/smoke_gat_cpu/gat/run_meta.json \
  --history output/experiments/smoke_gat_cpu/gat/history.json \
  --test_results output/experiments/smoke_gat_cpu/gat/test_results.json \
  --checkpoint_dir output/experiments/smoke_gat_cpu/gat \
  --output_dir output/visualizations/gat_cpu \
  --device cpu
```

---

## 7. 多模型比较

先至少跑出多个实验目录，再比较。

### artifact-only 模式
只读取现有产物，不重算预测：

```bash
python scripts/compare_model_runs.py \
  --run_dir output/experiments/smoke_stgcn_cpu/stgcn \
  --run_dir output/experiments/smoke_gat_cpu/gat \
  --run_dir output/experiments/smoke_sage_cpu/graphsage \
  --labels stgcn gat graphsage \
  --output_dir output/comparisons/cpu_artifact \
  --mode artifact-only \
  --device cpu
```

### full-prediction 模式
重建模型并在测试集上重算预测：

```bash
python scripts/compare_model_runs.py \
  --run_dir output/experiments/smoke_stgcn_cpu/stgcn \
  --run_dir output/experiments/smoke_gat_cpu/gat \
  --run_dir output/experiments/smoke_sage_cpu/graphsage \
  --labels stgcn gat graphsage \
  --output_dir output/comparisons/cpu_full \
  --mode full-prediction \
  --device cpu
```

如果你想指定基线做相对提升图：

```bash
python scripts/compare_model_runs.py \
  --run_dir output/experiments/smoke_stgcn_cpu/stgcn \
  --run_dir output/experiments/smoke_gat_cpu/gat \
  --run_dir output/experiments/smoke_sage_cpu/graphsage \
  --labels stgcn gat graphsage \
  --output_dir output/comparisons/cpu_full \
  --mode full-prediction \
  --baseline_label stgcn \
  --device cpu
```

---

## 8. CPU 运行建议

CPU 下建议先这样配：

- `--epoch 1`
- `--batch_size 8`
- `--device cpu`

如果还是慢，可以进一步减小：

```bash
--batch_size 4
```

GAT 在 CPU 上通常比 MLP/LSTM 更慢；如果想稳一点，可以先用：

```bash
--graph_hidden_dim 32 --graph_num_layers 1 --gat_heads 2
```

GraphSAGE 也可以先降为：

```bash
--graph_hidden_dim 32 --graph_num_layers 1
```

---

## 9. 推荐的最小验证流程

按这个顺序最稳：

### 第一步：跑 3 个 smoke run
```bash
python main.py --model_name stgcn --exp_name smoke_stgcn_cpu --n_route 228 --epoch 1 --batch_size 8 --device cpu
python main.py --model_name gat --exp_name smoke_gat_cpu --n_route 228 --epoch 1 --batch_size 8 --device cpu
python main.py --model_name graphsage --exp_name smoke_sage_cpu --n_route 228 --epoch 1 --batch_size 8 --device cpu
```

### 第二步：检查输出目录
确认这些目录存在：

```text
output/experiments/smoke_stgcn_cpu/stgcn/
output/experiments/smoke_gat_cpu/gat/
output/experiments/smoke_sage_cpu/graphsage/
```

### 第三步：做比较
```bash
python scripts/compare_model_runs.py \
  --run_dir output/experiments/smoke_stgcn_cpu/stgcn \
  --run_dir output/experiments/smoke_gat_cpu/gat \
  --run_dir output/experiments/smoke_sage_cpu/graphsage \
  --labels stgcn gat graphsage \
  --output_dir output/comparisons/cpu_full \
  --mode full-prediction \
  --device cpu
```

---

## 10. 常见问题

### 1) `Graph CSV has xxx nodes but --n_route=228`
说明图文件和节点数不一致。
检查：

- `dataset/PeMSD7_W_228.csv`
- 或你传入的 `--graph` 文件

### 2) 目录已存在，输出被写到新目录
这是正常行为。默认不覆盖已有实验目录。
如果你想复用原目录：

```bash
--overwrite true
```

### 3) CPU 太慢
先把参数降到最小：

```bash
--epoch 1 --batch_size 4 --graph_hidden_dim 32 --graph_num_layers 1
```
