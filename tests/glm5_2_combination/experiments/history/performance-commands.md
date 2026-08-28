# eager/Inductor 组合性能命令账本

以下命令均在容器 `glm5-npu-dev` 的
`/workspace/y50064852_yyb/torchtitan-test` 执行。执行顺序为环境核对、固定输入生成、
eager 矩阵、Inductor 矩阵、离线汇总。所有大日志和指标保存在 Git 忽略目录。

## 1. 环境和三仓身份

```bash
tests/glm5_2_graph_debug/run_graph_mode.sh inductor env
git -C /workspace/y50064852_yyb/torchtitan rev-parse HEAD
git -C /workspace/y50064852_yyb/TorchTitanTurbo rev-parse HEAD
git -C /workspace/y50064852_yyb/torchtitan-test rev-parse HEAD
npu-smi info
```

环境：CANN 9.1.0、torch `2.14.0.dev20260805+cpu`、torch_npu `2.14.0`、
triton_ascend `3.2.1`。物理 NPU0/1/3/4 当时为 Alarm，因此八卡结果只作诊断。

## 2. 固定输入

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 tests/glm5_2_graph_debug/run_graph_mode.sh inductor combination --data --data-device npu --topology all --objectives performance --reference-graph eager --candidate-graph eager --profiler-preset off --steps 30 --performance-skip-steps 10 --performance-nondeterministic
```

输出：

```text
precision_fixtures/self-npu-bf16-random-s30-b64-seq128-seed61-976dfb03/
```

## 3. eager 全拓扑

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 tests/glm5_2_graph_debug/run_graph_mode.sh inductor combination --capture candidate --topology all --objectives performance --reference-graph eager --candidate-graph eager --profiler-preset off --steps 30 --performance-skip-steps 10 --performance-nondeterministic --compiler-diagnostics
```

launcher 与正式运行根目录：

```text
graph_debug_runs/launcher-combination-inductor-20260826-112842-462169/
combination_runs/self-npu-bf16-random-s30-b64-seq128-seed61-eager-eager-performance-no-prof-skip10-nondet-8282f1d8/
```

在正式 batch 前还执行过 single-r1 定向探针，保留于：

```text
graph_debug_runs/launcher-combination-inductor-20260826-111649-4189928/
```

## 4. Inductor 全拓扑

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 tests/glm5_2_graph_debug/run_graph_mode.sh inductor combination --capture candidate --topology all --objectives performance --reference-graph eager --candidate-graph inductor --profiler-preset off --steps 30 --performance-skip-steps 10 --performance-nondeterministic --compiler-diagnostics
```

launcher 与正式运行根目录：

```text
graph_debug_runs/launcher-combination-inductor-20260826-124035-1095243/
combination_runs/self-npu-bf16-random-s30-b64-seq128-seed61-eager-inductor-performance-no-prof-skip10-nondet-9d360eea/
```

在正式 batch 前还执行过 single-r1 定向探针，保留于：

```text
graph_debug_runs/launcher-combination-inductor-20260826-111912-165642/
```

正式 Inductor launcher 用时约 8329 秒，15 个 topology × 2 repeat 全部完成；
没有 graph break、backend failure、基础设施错误或非有限值。

## 5. 离线生成 Markdown/HTML/JSON

```bash
python tests/glm5_2_combination/experiments/tools/summarize_performance.py \
  --eager-root combination_runs/self-npu-bf16-random-s30-b64-seq128-seed61-eager-eager-performance-no-prof-skip10-nondet-8282f1d8 \
  --inductor-root combination_runs/self-npu-bf16-random-s30-b64-seq128-seed61-eager-inductor-performance-no-prof-skip10-nondet-9d360eea \
  --artifact-root combination_artifacts
```

该命令读取每个 `raw_metrics.jsonl`、`runtime.log`、artifact manifest 和训练契约，
校验 60 个运行均有 30 条 step 指标，以每个 repeat 的 steps 11-30 中位数为权威值，
再以两个 repeat 中位数汇总每种模式。输出见：

```text
tests/glm5_2_combination/experiments/runs/performance/
tests/glm5_2_combination/experiments/reports/performance/
```

## 6. 定向复跑

单拓扑复跑时把 `--topology all` 替换为拓扑名，例如：

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  tests/glm5_2_graph_debug/run_graph_mode.sh inductor combination \
  --capture candidate --topology fsdp2-tp4 --objectives performance \
  --reference-graph eager --candidate-graph inductor --profiler-preset off \
  --steps 30 --performance-skip-steps 10 --performance-nondeterministic \
  --compiler-diagnostics
```

有效 artifact 会按完整实验身份复用；要保留新的 repeat，使用 workflow 的 repeat 配置，
不要覆盖已有 `candidate-r1/r2` 目录。
