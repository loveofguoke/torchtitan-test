# 图模式适配代码实现说明

本文回答“具体改了什么”。所有 graph 兼容项都通过独立 launcher 环境显式开启，
没有直接修改 Conda 中的 torch、torch_npu、Triton 或 CANN 安装文件。

## `graph_env_common.sh`

文件：`tests/glm5_2_graph_debug/graph_env_common.sh`

职责：建立所有图模式 target 共用的隔离进程。

### 软件栈隔离

1. 校验 CANN、Conda、ATB 路径和严格递增的可见设备列表。
2. 使用 `env -i` 丢弃父 shell 的 CANN 9.0、Conda 和动态库污染。
3. 只透传代理、编译诊断等白名单变量。
4. 在子进程中先 source CANN 9.1 `set_env.sh`，然后将 clone Conda 和 ATB 放到
   `PATH`/`LD_LIBRARY_PATH` 前部。
5. unset `CUDA_VISIBLE_DEVICES`，固定 `TORCHTITAN_DEVICE=npu`。

### 当前默认 profile

| 设置 | Inductor | NPUGraphs |
|---|---|---|
| `NPU_INDUCTOR_FALLBACK_LIST` | `aten.sum,_c10d_functional.all_reduce` | 空 |
| `TASK_QUEUE_ENABLE` | `0` | `0` |
| `TORCHTITAN_NPUGRAPH_SKIP_ALL` | `0` | `1` |
| grouped-mm unsafe-op list | 空 | 原始 op + safe custom op |
| complex DTensor strategy | `1` | `1` |
| PP metadata batched P2P | `0` | `0` |
| empty grouped-mm guard | `1` | `1` |
| zero-numel Triton guard | `1` | `1` |
| compile threads/rank | `1` | `1` |

两者共享：

```text
HCCL_NPU_SOCKET_PORT_RANGE=auto
HCCL_IF_BASE_PORT=<62000-63584 dynamic>
HCCL_CONNECT_TIMEOUT=600
TORCHTITAN_COMM_INIT_TIMEOUT_SECONDS=2400
TORCHINDUCTOR_NPU_BACKEND=default
ASCEND_LAUNCH_BLOCKING=0
```

### 缓存和运行目录

```text
GRAPH_CACHE_ROOT=~/.cache/torchtitan-test/graph_mode/cann91-torch214-triton321
TORCHINDUCTOR_CACHE_DIR=<root>/inductor
TRITON_CACHE_DIR=<root>/triton
TORCH_COMPILE_DEBUG_DIR=<root>/torch_compile_debug
GRAPH_RUN_ROOT=<repo>/graph_debug_runs
```

缓存根目录由 workspace root 计算，对当前容器是
`/workspace/y50064852_yyb/.cache/...`，宿主机对应 `/home/y50064852_yyb/.cache/...`。

## `run_graph_mode.sh`

文件：`tests/glm5_2_graph_debug/run_graph_mode.sh`

职责：统一入口，不重写原实验 contract。

- `train`：只在缺失时补 `--compile.enable`、`--compile.components=model` 和 backend；
  显式冲突 backend 会拒绝。
- `smoke`：交给 `run_smoke_graph.sh`。
- `compile-probe`、`precision`、`performance`、`combination`：只在缺失时补 eager
  reference 和所选 candidate backend，其余参数原样传递。
- `command`：任意命令在相同隔离环境中执行，便于单测和诊断。
- `env`：输出软件版本、CANN 标记、所有开关和 cache 路径。

每次非 smoke 调用写入：

```text
graph_debug_runs/launcher-<target>-<backend>-<stamp>-<pid>/
  logs/runtime.log
  reports/report.md
```

报告依据退出码、compiler/backend 关键字、资源错误和非有限数值分类结果。

## `run_smoke_graph.sh`

文件：`tests/glm5_2_graph_debug/run_smoke_graph.sh`

职责：在 common 环境中调用已经支持图模式的 `train_smoke.py`。

- 固定 `--device npu --graph <backend>`，防止 wrapper 与用户参数冲突。
- 用户未指定 topology 时补 `--topology all`。
- 不接管 per-topology 目录；manifest、trainer output 和 runtime log 仍由 smoke 管理。
- 额外生成 invocation report，记录环境、命令、started/passed/skipped 数量以及错误分类。

## `train_npu.py`

文件：仓库根目录 `train_npu.py`。

它仍是 TorchTitan NPU bootstrap，但新增的行为全部受 `TORCHTITAN_*` 环境变量控制。

### task queue 恢复

校验 `TORCHTITAN_TASK_QUEUE_ENABLE=0|1|2`。导入 TorchTitanTurbo 后重新设置
`TASK_QUEUE_ENABLE`，避免 Turbo 覆盖 launcher profile。

### process-group timeout

当 launcher 提供 `TORCHTITAN_COMM_INIT_TIMEOUT_SECONDS` 且 CLI 未显式指定时，向
TorchTitan argv 追加 `--comm.init_timeout_seconds=<value>`。

### zero-numel Triton guard

当 `TORCHTITAN_SAFE_ZERO_NUMEL_TRITON=1`：

- 检查 generated kernel 的所有非 reduction `*numel` 参数；
- 参数为 0 时在 autotune/launch 前直接返回；
- 同时包装普通 caching autotuner 和覆盖了 `run()` 的 symbolic grouped autotuner。

### safe grouped-mm

当 `TORCHTITAN_SAFE_EMPTY_GROUPED_MM=1`：

- 注册 `torchtitan_graph_debug::safe_grouped_mm` custom op 和 fake implementation；
- 无空组时直接调用原 `_grouped_mm`；
- 部分空 expert 时补一个零行并在输出中移除；
- 全部输入为空时完全绕过 CANN kernel；
- offsets 始终保持 int32；
- 注册显式 backward，补行不贡献参数梯度；
- 仅在该隔离进程替换 `GroupedExperts._grouped_mm`。

### TP complex strategy

当 `TORCHTITAN_REGISTER_COMPLEX_DTENSOR_STRATEGY=1`，为
`torch.ops.aten.complex.default` 注册 PyTorch 已有的 broadcast-aware 单维 pointwise
strategy。

### PP metadata P2P

当 `TORCHTITAN_PIPELINE_META_USE_BATCH=0`，只替换 `PipelineStage` 的 metadata
send/recv，使 `send_object_list`、`recv_object_list` 使用普通 P2P。模型 activation、
gradient 和 schedule 不变。

### NPUGraph skip policy

- `TORCHTITAN_NPUGRAPH_UNSAFE_OPS` 支持扫描 FX graph 并返回显式 skip reason；这是
  早期局部兼容和后续 backend 回归工具。
- `TORCHTITAN_NPUGRAPH_SKIP_ALL=1` 直接返回全局 replay skip reason；AOT graph 仍执行。
- `TORCHTITAN_NPUGRAPH_CAPTURE_ERROR_MODE` 只用于显式诊断，不在最终 profile 强制放宽。

## `train_smoke.py` 没有被 graph-debug 重写

`tests/glm5_2_smoke/train_smoke.py` 原生接受 `--graph eager|inductor|npugraphs`，通过
共享 `GraphFeatureConfig` 生成 compile 参数，并把 graph contract 写入 suite 名称和
manifest。graph-debug wrapper 只补软件环境和兼容 profile。

## 影响范围与回滚

- 不通过 `run_graph_mode.sh`/`run_smoke_graph.sh` 启动时，普通流程不会设置上述
  `TORCHTITAN_*` 变量，多数 patch 不启用。
- 所有选项都有 `GRAPH_*` 覆盖项，可以逐项关闭做工具链升级回归。
- CANN 9.0 全局软链、原 Conda 环境及已有正式实验目录没有被改写。
- 删除 clone Conda、独立 CANN 目录和仓库外 cache 即可回收适配环境；具体步骤见
  `CANN_9_1_INSTALLATION.md`。
