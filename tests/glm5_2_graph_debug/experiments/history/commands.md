# 图模式实验命令账本

所有命令均在容器 `glm5-npu-dev` 的仓库根目录执行。common 会自行建立干净环境，
不要求先 `conda activate` 或在当前 shell source CANN。

## 1. 环境快照

```bash
tests/glm5_2_graph_debug/run_graph_mode.sh inductor env
tests/glm5_2_graph_debug/run_graph_mode.sh npugraphs env
```

最终报告：

```text
graph_debug_runs/launcher-env-inductor-20260825-172537-4114462/reports/report.md
graph_debug_runs/launcher-env-npugraphs-20260825-172537-4114523/reports/report.md
```

## 2. 原始 train 接口

```bash
tests/glm5_2_graph_debug/run_graph_mode.sh inductor train \
  --compile.enable \
  --compile.components=model \
  --compile.backend=inductor \
  --training.steps=10
```

已有参数会原样透传；缺失的三个 compile 参数由 launcher 补齐。

## 3. smoke 单拓扑命令

```bash
tests/glm5_2_graph_debug/run_graph_mode.sh inductor smoke \
  --topology single
tests/glm5_2_graph_debug/run_graph_mode.sh npugraphs smoke \
  --topology single
```

任意正式 topology 使用同一模板：

```bash
tests/glm5_2_graph_debug/run_graph_mode.sh BACKEND smoke \
  --topology TOPOLOGY
```

`TOPOLOGY` 的实际集合：

```text
single ddp2 ddp8 fsdp8 tp8 cp8 pp8 ep8
fsdp2-tp4 fsdp2-cp4 tp2-cp4 fsdp4-tp2
fsdp2-pp4 fsdp2-tp2-pp2 fsdp2-tp4-ep8
```

调试过程中使用单 topology 命令修复失败，再利用 passed manifest 断点续跑；没有在
每次修复后删除已经通过的其他 topology。

## 4. 全矩阵命令

```bash
tests/glm5_2_graph_debug/run_graph_mode.sh inductor smoke
tests/glm5_2_graph_debug/run_graph_mode.sh npugraphs smoke
```

等价底层 smoke 参数分别为：

```bash
python tests/glm5_2_smoke/train_smoke.py \
  --device npu --graph inductor --topology all
python tests/glm5_2_smoke/train_smoke.py \
  --device npu --graph npugraphs --topology all
```

最终汇总调用跳过了 15 个已有 passed manifest。真实训练由此前的定向命令完成；
每个 manifest 保存了解析后的完整 `run_train.sh` argv。例如 Inductor 固定包含：

```text
--training.steps=10
--training.disable_cuda_graphs
--training.num_tokens_per_microbatch_per_dp_rank=1024
--training.num_tokens_per_train_step=8192
--training.max_context_length=128
--debug.seed=61
--metrics.log_freq=1
--compile.enable
--compile.components=model
--compile.backend=inductor
```

NPUGraphs 仅把最后一项改为 `--compile.backend=npugraphs`，并由环境 profile 禁用
replay。

## 5. graph benchmark 与组合实验

```bash
tests/glm5_2_graph_debug/run_graph_mode.sh inductor compile-probe \
  --capture candidate --topology single
tests/glm5_2_graph_debug/run_graph_mode.sh inductor precision \
  --capture candidate --topology fsdp8
tests/glm5_2_graph_debug/run_graph_mode.sh inductor performance \
  --capture candidate --topology fsdp8
tests/glm5_2_graph_debug/run_graph_mode.sh inductor combination \
  --capture candidate --topology fsdp8
```

launcher 只在原参数缺失时注入 `--reference-graph=eager` 和
`--candidate-graph=inductor`，不会改写 benchmark 自己的目录和验收规则。

## 6. 原生 NPUGraph capture 回归

下面命令用于复现未解决的原生 replay 问题，不是通过配置：

```bash
GRAPH_NPUGRAPH_SKIP_ALL=0 \
GRAPH_TASK_QUEUE_ENABLE=1 \
tests/glm5_2_graph_debug/run_graph_mode.sh npugraphs smoke \
  --topology single
```

TP 回归可把 topology 改为 `tp8`。当前预期分别遇到 grouped-mm 107030/107027 或
graph tree 的 `DeviceMesh` 输入错误。

## 7. 单项兼容开关回归

```bash
# 复现 CANN 路径问题：不要通过该 launcher；使用默认 shell 会混入 9.0。

GRAPH_NPU_INDUCTOR_FALLBACK_LIST= \
  tests/glm5_2_graph_debug/run_graph_mode.sh inductor smoke --topology single

GRAPH_REGISTER_COMPLEX_DTENSOR_STRATEGY=0 \
  tests/glm5_2_graph_debug/run_graph_mode.sh inductor smoke --topology tp8

GRAPH_PIPELINE_META_USE_BATCH=1 \
  tests/glm5_2_graph_debug/run_graph_mode.sh inductor smoke --topology pp8

GRAPH_SAFE_EMPTY_GROUPED_MM=0 \
GRAPH_SAFE_ZERO_NUMEL_TRITON=0 \
  tests/glm5_2_graph_debug/run_graph_mode.sh inductor smoke \
  --topology fsdp2-tp4-ep8
```

这些命令会命中已有同 contract 的 passed manifest；真正做回归时需要另设训练
contract（例如不同 seed）或谨慎使用 `--force`。`--force` 会移走/重建该 topology
结果，不应在普通复现中随意使用。

## 8. 最终静态与单元验证

```bash
bash -n tests/glm5_2_graph_debug/*.sh
git diff --check
tests/glm5_2_graph_debug/run_graph_mode.sh inductor command \
  pytest -q tests/unit_tests/test_glm5_2_graph.py \
  tests/unit_tests/test_glm5_2_smoke.py
```

最终单测为 17/17 通过，对应报告：

```text
graph_debug_runs/launcher-command-inductor-20260825-175353-4115183/reports/report.md
```
