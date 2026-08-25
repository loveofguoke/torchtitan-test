# GLM-5.2 graph-mode debug

The consolidated current conclusion and three-repository fix ownership are in
[`../glm5_2_graph/NPU_GRAPH_DEBUG_REPORT.md`](../glm5_2_graph/NPU_GRAPH_DEBUG_REPORT.md).
This directory preserves the detailed chronological debugging evidence.

本目录只承载 NPU 图模式的软件栈调试入口、运行说明、环境适配报告和后续 bug
调试文档，不承载正式 precision、performance、stability 或 combination 验收。
推荐统一使用 `run_graph_mode.sh`：它让正常训练、已有 smoke、graph benchmark 和
combination 都经过同一个 CANN 9.1/Conda/HCCL/cache common 环境，并保留各入口的
原参数和结果规范。

## 内容

- `run_npu_inductor.sh`：隔离 CANN 9.1、复用 `.cache`、运行并自动生成报告。
- `graph_env_common.sh`：所有图模式入口共用的纯环境层，不直接启动实验。
- `run_graph_mode.sh`：train、smoke、graph benchmark、combination 和任意命令的
  统一入口。
- `run_smoke_graph.sh`：在同一隔离环境中运行已有 smoke 的 Inductor/NPUGraphs
  单卡、指定拓扑或全部拓扑，并保留 `smoke_runs` 结果规范。
- `RUN_NPU_INDUCTOR.md`：smoke/train/probe/env 用法、缓存和故障排查。
- `SMOKE_GRAPH_SUITE.md`：现有 smoke 图模式的运行方法、覆盖范围和实测结果。
- `GRAPH_MODE_COMMON.md`：统一入口、完整 export、参数透传、覆盖项和结果目录。
- `COMMON_GRAPH_MODE_VALIDATION_REPORT.md`：common、smoke 矩阵与修复过程的历史中断快照。
- `FINAL_GRAPH_DEBUG_REPORT.md`：中断恢复后的完整步骤、15×2 矩阵、修复与最终结论。
- `experiments/index.md`：仿照 performance exploration 的分层实验档案入口，包含
  总结、失败历史、命令账本和按卡数/拓扑索引。
- `CANN_9_1_INSTALLATION.md`：CANN 9.1.0 独立安装、完整性校验与按进程激活。
- `NPU_INDUCTOR_ENV_ADAPTATION_REPORT.md`：首轮环境兼容问题的完整证据与结论。
- `ARTIFACT_LAYOUT.md`：运行结果和 `.cache` 编译中间产物的目录规范。

## 目录约定

```text
tests/glm5_2_graph_debug/        # 仅脚本和文档，纳入 Git
graph_debug_runs/                # 调试运行结果、报告和日志，Git 忽略
~/.cache/torchtitan-test/graph_mode/
                                # Inductor/Triton/PCH/compile debug 中间产物
```

容器内 `~/.cache` 的持久化位置使用
`/workspace/y50064852_yyb/.cache/torchtitan-test/graph_mode/`，对应宿主机
`/home/y50064852_yyb/.cache/torchtitan-test/graph_mode/`。

快速运行：

```bash
tests/glm5_2_graph_debug/run_graph_mode.sh inductor env
tests/glm5_2_graph_debug/run_graph_mode.sh inductor train --training.steps=10
```

已有完整 compile 参数也可以原样放在 `train` 后面；common 只补缺失项，不重复
追加。显式 backend 必须与第一个参数一致：

```bash
tests/glm5_2_graph_debug/run_graph_mode.sh inductor train \
  --compile.enable \
  --compile.components=model \
  --compile.backend=inductor \
  --training.steps=10
```

运行已有 smoke 的单个拓扑或全部拓扑：

```bash
tests/glm5_2_graph_debug/run_graph_mode.sh inductor smoke --topology single
tests/glm5_2_graph_debug/run_graph_mode.sh inductor smoke --topology fsdp8
tests/glm5_2_graph_debug/run_graph_mode.sh inductor smoke
tests/glm5_2_graph_debug/run_graph_mode.sh npugraphs smoke
```

最后两条未指定拓扑，默认运行 smoke 定义中 world size 不超过 8 的全部拓扑。
当前实测结果为 Inductor 15/15 通过；NPUGraphs profile 15/15 在保留 AOT graph、
显式禁用 replay 的兼容降级下通过。两者的边界与报告路径见
`FINAL_GRAPH_DEBUG_REPORT.md`。
不要绕过封装直接从容器默认 shell 启动 NPU 图模式；默认 shell 的 CANN 9.0 路径
会在 Python 导入 `torch_npu` 前泄漏，且共享主机上的 HCCL 默认端口可能冲突。

graph benchmark 和组合实验无需改写原参数：

```bash
tests/glm5_2_graph_debug/run_graph_mode.sh inductor performance \
  --capture candidate --topology fsdp8
tests/glm5_2_graph_debug/run_graph_mode.sh inductor combination \
  --capture candidate --topology fsdp8
```

完整 common 参数和手工 export 见 `GRAPH_MODE_COMMON.md`。下面的
`run_npu_inductor.sh` 参数只为早期专用入口兼容保留。

## 旧专用入口参数与默认值

调用形式为 `run_npu_inductor.sh [action] [extra arguments]`：

| action | 功能 | 默认 |
|---|---|---|
| `smoke` | 运行 10 steps、batch 1、sequence length 32 的 NPU Inductor 小训练。 | 未指定 action 时使用 |
| `train` | 使用正常 `run_train.sh` 配置，并追加 model Inductor 编译参数。 | 非默认 |
| `probe` | 依次执行 data、NPU eager reference、NPU Inductor candidate 和 CPU compare。 | 非默认 |
| `env` | 只打印隔离后的 CANN、Python、torch_npu、Triton 和缓存路径。 | 非默认 |

action 后的参数会传给训练命令；`probe` 会把它们传给四个阶段，因此这些
参数必须在全部阶段都有效。完整 CLI 说明也可以运行
`run_npu_inductor.sh --help` 查看。

| 环境变量 | 功能 | 默认 |
|---|---|---|
| `ASCEND_RT_VISIBLE_DEVICES` | 物理 NPU 编号。 | `2` |
| `GRAPH_CANN_ROOT` | 隔离使用的 CANN 根目录。 | `/usr/local/Ascend/cann-9.1.0` |
| `GRAPH_CONDA_ENV` | 图模式专用 Conda 环境路径。 | `/root/miniconda3/envs/torchtitan-0803-graph-adapt` |
| `GRAPH_ATB_ROOT` | ATB C++ ABI 运行目录。 | `/usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_1` |
| `GRAPH_CACHE_ROOT` | 持久化编译缓存根目录。 | 工作区 `.cache/torchtitan-test/graph_mode/...` |
| `GRAPH_INDUCTOR_CACHE_DIR` | 精确覆盖 Inductor 缓存目录。 | `<GRAPH_CACHE_ROOT>/inductor` |
| `GRAPH_TRITON_CACHE_DIR` | 精确覆盖 Triton 缓存目录。 | `<GRAPH_CACHE_ROOT>/triton` |
| `GRAPH_COMPILE_DEBUG_DIR` | `torch_compile_debug` 输出目录。 | `<GRAPH_CACHE_ROOT>/torch_compile_debug` |
| `GRAPH_RUN_ROOT` | 调试日志和报告根目录。 | `graph_debug_runs` |
| `TORCH_COMPILE_DEBUG` | `1` 时生成详细编译调试文件。 | `0` |

## 结果与判定

每次运行创建独立时间戳目录，保存完整 `runtime.log` 和 `report.md`。
报告记录 action、退出码、耗时、物理设备、CANN/Python 包版本、缓存路径、
训练 step 数、正常完成标记和编译器错误计数。退出码为 0 时该调试运行标记为
`PASSED`。它只证明当前软件栈和入口能够运行；正式图模式精度与性能结论仍由
`glm5_2_graph`/`glm5_2_combination` 报告给出。

## 探索性补充

本地多卡图模式排障记录保留在 `MULTI_NPU_DEBUG.md`，CANN 9.1 隔离安装过程保留在
`CANN_9_1_INSTALLATION.md`。这些文档是环境探索证据，不改变上面的正式入口和判定规则。
