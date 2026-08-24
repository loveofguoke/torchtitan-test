# GLM-5.2 graph-mode debug

本目录只承载 NPU Inductor 图模式的软件栈调试入口、运行说明、环境适配报告和后续
bug 调试文档，不承载正式 precision、performance、stability 或 combination 验收。
该脚本固定为单卡软件栈调试；指定分布式拓扑（下面以 `fsdp8` 为例）或运行
`all` 图模式跑通请直接使用
`tests/glm5_2_smoke/train_smoke.py --device npu --graph inductor`。

## 内容

- `run_npu_inductor.sh`：隔离 CANN 9.1、复用 `.cache`、运行并自动生成报告。
- `RUN_NPU_INDUCTOR.md`：smoke/train/probe/env 用法、缓存和故障排查。
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
tests/glm5_2_graph_debug/run_npu_inductor.sh env
tests/glm5_2_graph_debug/run_npu_inductor.sh smoke
```

分布式图模式 smoke 不使用本目录的单卡封装，直接运行：

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
unset CUDA_VISIBLE_DEVICES

python tests/glm5_2_smoke/train_smoke.py \
  --device npu --topology fsdp8 --graph inductor
python tests/glm5_2_smoke/train_smoke.py \
  --device npu --topology all --graph inductor
```

## 参数与默认值

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
