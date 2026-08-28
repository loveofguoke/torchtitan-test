# NPU 图模式 common 环境与统一入口

## 1. 为什么必须有 common 层

当前容器默认 shell 仍指向 CANN 9.0，而图模式适配环境使用 CANN 9.1.0、独立
Conda 环境和仓库外的持久化编译缓存。图模式启动还涉及 HCCL 端口、task queue、
Inductor fallback、NPUGraph 捕获限制以及若干已定位的多卡兼容项。由用户每次手工
拼接这些变量既容易遗漏，也无法保证 smoke、graph benchmark 和 combination 使用
同一软件栈。

`graph_env_common.sh` 因此只负责一件事：在任何 Python 导入 `torch_npu` 之前，
用干净子进程选择完整的软件环境。`run_graph_mode.sh` 在此基础上提供统一入口，
保留各原有测试程序的 CLI 和结果规范。

## 2. 最短用法

在 `glm5-npu-dev` 容器的仓库根目录执行：

```bash
cd /workspace/y50064852_yyb/torchtitan-test

# 确认实际环境，不启动训练
tests/glm5_2_graph_debug/run_graph_mode.sh inductor env

# 正常 run_train 参数只需放在后面；三个 compile 参数由入口注入
ASCEND_RT_VISIBLE_DEVICES=0 \
  tests/glm5_2_graph_debug/run_graph_mode.sh inductor train \
  --training.steps=10

# smoke 原参数保持不变
ASCEND_RT_VISIBLE_DEVICES=0 \
  tests/glm5_2_graph_debug/run_graph_mode.sh inductor smoke \
  --topology single

# graph benchmark 原参数保持不变
tests/glm5_2_graph_debug/run_graph_mode.sh inductor performance \
  --capture candidate --topology fsdp8

# combination 原参数保持不变
tests/glm5_2_graph_debug/run_graph_mode.sh inductor combination \
  --capture candidate --topology fsdp8
```

将第一个参数改为 `npugraphs` 即可选择 NPUGraphs profile：

```bash
tests/glm5_2_graph_debug/run_graph_mode.sh npugraphs smoke --topology ddp2
tests/glm5_2_graph_debug/run_graph_mode.sh npugraphs precision \
  --capture candidate --topology single
```

## 3. 统一入口支持范围

调用形式：

```text
run_graph_mode.sh BACKEND TARGET [TARGET 原有参数]
```

`BACKEND` 为 `inductor` 或 `npugraphs`。`TARGET` 映射如下：

| target | 原入口 | common 自动补充 |
|---|---|---|
| `train` | `./run_train.sh` | 缺失时补 `--compile.enable --compile.components=model --compile.backend=<backend>` |
| `smoke` | `tests/glm5_2_smoke/train_smoke.py` | `--device npu --graph <backend>` |
| `compile-probe` | `tests/glm5_2_graph/compile_probe.py` | 缺失时补 `--reference-graph eager --candidate-graph <backend>` |
| `precision` | `tests/glm5_2_graph/precision_benchmark.py` | 同上 |
| `performance` | `tests/glm5_2_graph/performance_benchmark.py` | 同上 |
| `combination` | `tests/glm5_2_combination/combination_benchmark.py` | 同上 |
| `command` | `--` 后的任意命令 | 不注入业务参数，只提供 common 环境 |
| `env` | 环境检查 | 不运行训练 |

benchmark 和 combination 如果已经显式传入 `--reference-graph` 或
`--candidate-graph`，common 不覆盖它们。其余参数按原顺序逐项转发，因此原来的
`--capture`、`--compare`、`--topology`、`--repeat`、`--objectives`、profiler 和
输出参数均可继续使用。

`train` 同样不会重复追加已经给出的 compile 参数；显式
`--compile.backend` 必须与第一个 `BACKEND` 参数一致，否则在训练前报错。这样原来的
完整 `run_train.sh --compile...` 参数可以直接放到 `train` target 后面，也可以只传
普通训练参数让 common 补齐最小图模式参数。

`command` 适合尚未登记为 target 的仓库脚本：

```bash
tests/glm5_2_graph_debug/run_graph_mode.sh inductor command -- \
  ./run_train.sh \
  --compile.enable \
  --compile.components=model \
  --compile.backend=inductor \
  --training.steps=10
```

## 4. CANN 9.1.0 必须在 Python 前激活

`torch_npu` 在 import 阶段就会解析 ACL、HCCL 和算子库。先启动 Python，再在
Python 内修改 `PATH`/`LD_LIBRARY_PATH`，无法撤销已加载的 CANN 9.0 动态库。
common 的执行顺序固定为：

1. 读取并校验 `GRAPH_*` 配置；
2. 用 `env -i` 重新执行调用脚本，清除默认 CANN 9.0 泄漏；
3. `source /usr/local/Ascend/cann-9.1.0/set_env.sh`；
4. 追加 Conda 与 ATB 路径；
5. 导入 Python 并运行目标测试。

推荐始终使用统一入口。如果需要排查，下面是与 common 等价的手工基础环境；它
只用于诊断，不替代脚本中的完整 profile：

```bash
env -i \
  HOME=/root \
  USER=root \
  LANG=C.UTF-8 \
  PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  /bin/bash --noprofile --norc

set +u
source /usr/local/Ascend/cann-9.1.0/set_env.sh
set -u

# `set_env.sh` 在干净子进程中的实际 CANN 9.1.0 exports：
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.1.0
export ASCEND_TOOLKIT_HOME=/usr/local/Ascend/cann-9.1.0
export ASCEND_AICPU_PATH=/usr/local/Ascend/cann-9.1.0
export ASCEND_OPP_PATH=/usr/local/Ascend/cann-9.1.0/opp
export TOOLCHAIN_HOME=/usr/local/Ascend/cann-9.1.0/toolkit
export PYTHONPATH=/usr/local/Ascend/cann-9.1.0/python/site-packages:\
/usr/local/Ascend/cann-9.1.0/opp/built-in/op_impl/ai_core/tbe
export PATH=/usr/local/Ascend/cann-9.1.0/bin:\
/usr/local/Ascend/cann-9.1.0/tools/ccec_compiler/bin:\
/usr/local/Ascend/cann-9.1.0/tools/profiler/bin:\
/usr/local/Ascend/cann-9.1.0/tools/ascend_system_advisor/asys:\
/usr/local/Ascend/cann-9.1.0/tools/show_kernel_debug_data:\
/usr/local/Ascend/cann-9.1.0/tools/msobjdump:$PATH
export LD_LIBRARY_PATH=/usr/local/Ascend/cann-9.1.0/lib64:\
/usr/local/Ascend/cann-9.1.0/lib64/plugin/opskernel:\
/usr/local/Ascend/cann-9.1.0/lib64/plugin/nnengine:\
/usr/local/Ascend/cann-9.1.0/opp/built-in/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64:\
/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:\
/usr/local/Ascend/driver/lib64/driver

export PATH=/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin:\
/usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_1/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_1/lib:\
${LD_LIBRARY_PATH:-}
export TORCHTITAN_DEVICE=npu
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
unset CUDA_VISIBLE_DEVICES
```

完整 CANN 9.1.0 独立安装、来源容器复制、官方 run 包安装和校验方式见
`CANN_9_1_INSTALLATION.md`。common 不修改 `/usr/local/Ascend/cann` 或
`ascend-toolkit/latest` 的全局软链，因此不会改变其他实验的默认 CANN 9.0 环境。

## 5. common 实际导出的变量

以下是默认 8 卡 Inductor profile 的完整逻辑配置。端口和绝对 cache 路径由每次
调用解析后写入报告：

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TORCHTITAN_DEVICE=npu

export HCCL_NPU_SOCKET_PORT_RANGE=auto
export HCCL_IF_BASE_PORT=<本次调用的 62000-63584 端口>
export HCCL_CONNECT_TIMEOUT=600
export TORCHTITAN_COMM_INIT_TIMEOUT_SECONDS=2400

export TORCHINDUCTOR_NPU_BACKEND=default
export NPU_INDUCTOR_FALLBACK_LIST=aten.sum,_c10d_functional.all_reduce
export TASK_QUEUE_ENABLE=0
export TORCHTITAN_TASK_QUEUE_ENABLE=0
export TORCHINDUCTOR_COMPILE_THREADS=1

export TORCHTITAN_REGISTER_COMPLEX_DTENSOR_STRATEGY=1
export TORCHTITAN_PIPELINE_META_USE_BATCH=0
export TORCHTITAN_SAFE_EMPTY_GROUPED_MM=1
export TORCHTITAN_SAFE_ZERO_NUMEL_TRITON=1
export ASCEND_LAUNCH_BLOCKING=0

export TORCHINDUCTOR_CACHE_DIR=/workspace/y50064852_yyb/.cache/torchtitan-test/\
graph_mode/cann91-torch214-triton321/inductor
export TRITON_CACHE_DIR=/workspace/y50064852_yyb/.cache/torchtitan-test/\
graph_mode/cann91-torch214-triton321/triton
export TORCH_COMPILE_DEBUG_DIR=/workspace/y50064852_yyb/.cache/torchtitan-test/\
graph_mode/cann91-torch214-triton321/torch_compile_debug
export PYTHONFAULTHANDLER=1
export PYTHONUNBUFFERED=1
```

NPUGraphs profile 与上面共享 CANN、Conda、HCCL、cache 和兼容项，差异为：

```bash
export NPU_INDUCTOR_FALLBACK_LIST=
export TASK_QUEUE_ENABLE=0
export TORCHTITAN_TASK_QUEUE_ENABLE=0
export TORCHTITAN_NPUGRAPH_SKIP_ALL=1
export TORCHTITAN_NPUGRAPH_UNSAFE_OPS=aten._grouped_mm.default,\
torchtitanturbo_graph.safe_grouped_mm.default
```

当前 CANN 9.1/torch 2.14/torch_npu 2.14 组合同时存在 grouped-mm capture 失败和 TP
`DeviceMesh` 录制失败，因此默认 profile 跳过全部 NPUGraph replay，但 AOT graph 仍
执行。报告中出现 `NPUGraph: skipped` 时应写成“图编译通过、捕获安全降级”，不能
写成完整 NPUGraph replay。要继续原生捕获调试，显式设置
`GRAPH_NPUGRAPH_SKIP_ALL=0 GRAPH_TASK_QUEUE_ENABLE=1`。

## 6. 可覆盖参数

所有用户可调项使用 `GRAPH_*` 前缀，common 再映射为工具链变量：

| 用户变量 | 默认 | 作用 |
|---|---|---|
| `ASCEND_RT_VISIBLE_DEVICES` | `0..7` | 可见物理 NPU，必须严格递增 |
| `GRAPH_CANN_ROOT` | `/usr/local/Ascend/cann-9.1.0` | 独立 CANN 根目录 |
| `GRAPH_CONDA_ENV` | `/root/miniconda3/envs/torchtitan-0803-graph-adapt` | 图模式 Conda prefix |
| `GRAPH_ATB_ROOT` | `/usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_1` | ATB C++ ABI runtime |
| `GRAPH_CACHE_ROOT` | 工作区 `.cache/.../cann91-torch214-triton321` | 所有编译缓存根目录 |
| `GRAPH_RUN_ROOT` | 仓库 `graph_debug_runs` | common 调用报告根目录 |
| `GRAPH_HCCL_NPU_SOCKET_PORT_RANGE` | `auto` | NPU NIC socket 端口 |
| `GRAPH_HCCL_IF_BASE_PORT` | 每次调用生成的 `62000-63584` | host socket 基准端口 |
| `GRAPH_HCCL_CONNECT_TIMEOUT` | `600` | HCCL 建链超时，秒 |
| `GRAPH_COMM_INIT_TIMEOUT_SECONDS` | `2400` | PyTorch process-group 超时，秒 |
| `GRAPH_TORCHINDUCTOR_NPU_BACKEND` | `default` | NPU Inductor codegen |
| `GRAPH_NPU_INDUCTOR_FALLBACK_LIST` | Inductor 两项，NPUGraphs 空 | 逗号分隔的最小 fallback |
| `GRAPH_TASK_QUEUE_ENABLE` | 图 profile `0` | task queue，原生 NPUGraph capture 实验需显式用 `1` |
| `GRAPH_NPUGRAPH_CAPTURE_ERROR_MODE` | 不覆盖 | 仅用于捕获诊断 |
| `GRAPH_NPUGRAPH_UNSAFE_OPS` | NPUGraphs 原始及安全 grouped-mm | 捕获前跳过的不安全 op |
| `GRAPH_NPUGRAPH_SKIP_ALL` | `1` | 保留 AOT graph，禁用不兼容的 replay；`0` 用于原生捕获调试 |
| `GRAPH_REGISTER_COMPLEX_DTENSOR_STRATEGY` | `1` | TP 的 `aten.complex` strategy |
| `GRAPH_PIPELINE_META_USE_BATCH` | `0` | PP metadata 是否使用 batched P2P |
| `GRAPH_SAFE_EMPTY_GROUPED_MM` | `1` | 保护空 expert 分组，防止 CANN 零核启动 |
| `GRAPH_SAFE_ZERO_NUMEL_TRITON` | `1` | 动态 pointwise 输出为 0 时跳过 grid=0 launch |
| `GRAPH_VETTED_POINTWISE_AUTOTUNE` | `1` | deterministic 下仅允许已分类的 pointwise autotune benchmark；reduction 仍禁止 |
| `GRAPH_COMPILE_THREADS` | `1` | 每 rank Inductor worker 数 |
| `GRAPH_ASCEND_LAUNCH_BLOCKING` | `0` | 同步 launch，仅用于定位异步错误 |
| `GRAPH_LOG_RANK` | `0` | 测试日志 rank |
| `TORCH_COMPILE_DEBUG` | `0` | 是否生成详细 compile debug |

示例：只用设备 2、3，固定端口并开启详细编译材料：

```bash
ASCEND_RT_VISIBLE_DEVICES=2,3 \
GRAPH_HCCL_NPU_SOCKET_PORT_RANGE=61000-61050 \
GRAPH_HCCL_IF_BASE_PORT=62160 \
TORCH_COMPILE_DEBUG=1 \
  tests/glm5_2_graph_debug/run_graph_mode.sh inductor smoke \
  --topology ddp2
```

不要使用 `ASCEND_RT_VISIBLE_DEVICES=3,2`。当前 torch_npu/CANN 组合对重排 mask 的
设备枚举不稳定，common 会在启动前拒绝非严格递增列表。

## 7. 参数透传与环境隔离边界

命令行参数不会经过字符串拼接或 `eval`，而是 Bash 数组逐项转发，所以包含空格或
等号的原参数不会被重新拆分。环境变量方面，clean re-exec 有意只继承：

- `GLM5_*` 实验变量；
- `MASTER_ADDR`、`MASTER_PORT`、`NNODES`、`NODE_RANK`；
- `MODULE`、`CONFIG`、`NGPU`、`LOG_RANK`；
- 大小写 proxy 变量；
- `TORCH_LOGS`、`TORCHDYNAMO_VERBOSE`。

所有 `GRAPH_*` 项由 common 显式解析后传递。其他默认 shell 变量不会进入图模式
进程，这是防止 CANN 9.0 路径和旧编译配置泄漏的关键。确实需要新增透传变量时，
应在 `graph_env_common.sh` 的 allowlist 中显式登记并在报告中说明，不要取消 `env -i`。

## 8. 结果、日志与缓存

业务结果仍由原测试拥有：

```text
smoke_runs/<contract>/<topology>/
tests/glm5_2_graph/...原有结果位置...
tests/glm5_2_combination/...原有结果位置...
```

common 只追加调用级报告：

```text
graph_debug_runs/
  smoke-suite-<backend>-<timestamp>-<pid>/
    logs/runtime.log
    reports/report.md
  launcher-<target>-<backend>-<timestamp>-<pid>/
    logs/runtime.log
    reports/report.md
```

编译中间结果全部位于 Git 仓库外，可跨运行复用：

```text
/workspace/y50064852_yyb/.cache/torchtitan-test/graph_mode/
  cann91-torch214-triton321/
    inductor/
    triton/
    torch_compile_debug/
```

对应宿主机目录为
`/home/y50064852_yyb/.cache/torchtitan-test/graph_mode/`。不要把这些文件复制到
`tests/glm5_2_graph_debug` 或提交到 Git。

## 9. 验证与故障定位

先执行：

```bash
tests/glm5_2_graph_debug/run_graph_mode.sh inductor env
```

必须确认 CANN、Conda、torch、torch_npu、triton_ascend 和三个 cache 路径都来自
预期位置。再按 `single -> ddp2 -> 目标多卡拓扑` 逐级验证。失败时保留
`graph_debug_runs/.../reports/report.md`、对应 runtime log 和原测试结果目录；需要
编译器 IR 时临时设置 `TORCH_COMPILE_DEBUG=1`，材料仍会进入 `.cache`。

正式精度会开启 deterministic algorithms。当前 torch_npu 2.14 的 pointwise autotuner
没有把安全 benchmark 声明为 vetted，common 因此默认启用 Turbo 的
`TORCHTITAN_VETTED_POINTWISE_AUTOTUNE`。它只放行 `HeuristicType.POINTWISE`，不能放行
reduction，也不能用 `--performance-nondeterministic` 代替精度验收。根因和 torch_npu
patch 位置见 `tests/glm5_2_graph/LOWER_LAYER_ISSUE_HANDOFF.md` 的 G020。

`ASCEND_LAUNCH_BLOCKING=1` 会改变执行时序，只用于把异步 NPU 错误定位到更接近的
算子，不能作为性能数据或最终通过条件。所有 benchmark 的 eager reference 与图模式
candidate 会共享同一套 CANN/Conda/HCCL common 环境，但各自仍由原 CLI 的
`--reference-graph` 和 `--candidate-graph` 决定执行模式。

多卡图模式要求目标设备有足够的 HBM、stream 和 HCCL 资源。启动前执行
`npu-smi info`；如果已有其他任务占用同一组卡，应等待或选择一组空闲且严格递增的
设备，不要把 `EE1023 Too many streams`、507033 或设备初始化失败记为编译后端
失败。common 不会自动结束现有进程，也不会抢占其他实验。

## 10. common 与原实验参数的关系

common 是图模式运行时 profile，不是新的实验定义。它只做三类工作：在 Python
之前选择 CANN/Conda/ATB，设置图后端已验证的环境变量，把编译中间产物导向仓库外
`.cache`。smoke、graph benchmark 和 combination 的 topology、steps、batch、
sequence、seed、precision、capture/compare、repeat、profiler、force 和业务结果目录
仍由原入口及其原参数决定。

因此迁移已有命令时，保留原参数，只替换命令前缀。例如原命令：

```bash
python tests/glm5_2_graph/performance_benchmark.py \
  --capture candidate --topology fsdp8 \
  --reference-graph eager --candidate-graph inductor
```

迁移后为：

```bash
tests/glm5_2_graph_debug/run_graph_mode.sh inductor performance \
  --capture candidate --topology fsdp8 \
  --reference-graph eager --candidate-graph inductor
```

如果原命令已经显式给出 reference/candidate graph，common 原样保留；如果没有，才
自动补 `eager` 与所选 backend。`command --` 可为尚未登记的任意实验提供同一环境，
但不会自动注入业务参数。
