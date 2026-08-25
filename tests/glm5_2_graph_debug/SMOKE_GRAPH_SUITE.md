# Smoke 图模式调试与实测

## 目标

复用 `tests/glm5_2_smoke/train_smoke.py` 的训练参数、拓扑定义、断点续跑和结果目录，
验证它提供的 `inductor` 与 `npugraphs` 两种 NPU 图模式。这里不另造 smoke case，
只补充图模式所需的软件栈隔离、持久化编译缓存、HCCL 端口配置和调用级报告。

## 推荐入口

在 `glm5-npu-dev` 容器中执行：

```bash
cd /workspace/y50064852_yyb/torchtitan-test

# 单卡先验验证
ASCEND_RT_VISIBLE_DEVICES=0 \
  tests/glm5_2_graph_debug/run_graph_mode.sh inductor smoke --topology single
ASCEND_RT_VISIBLE_DEVICES=0 \
  tests/glm5_2_graph_debug/run_graph_mode.sh npugraphs smoke --topology single

# 8 卡全部拓扑；不传 --topology/--topologies 时默认 all
tests/glm5_2_graph_debug/run_graph_mode.sh inductor smoke
tests/glm5_2_graph_debug/run_graph_mode.sh npugraphs smoke
```

也可以只运行一个集合：

```bash
tests/glm5_2_graph_debug/run_graph_mode.sh inductor smoke \
  --topologies ddp8,fsdp8,tp8
```

后续参数原样交给现有 `train_smoke.py`，例如 `--compiler-diagnostics`、`--steps`、
`--sequence-length`。封装固定 `--device npu` 和所选图后端，因此不要重复传入
`--device` 或 `--graph`。

## 覆盖范围

`--topology all` 当前选择 world size 不超过 8 的全部 15 种标准拓扑：

```text
single
ddp2, ddp8, fsdp8
tp8, cp8, pp8, ep8
fsdp2-tp4, fsdp2-cp4, tp2-cp4, fsdp4-tp2
fsdp2-pp4, fsdp2-tp2-pp2, fsdp2-tp4-ep8
```

默认训练 contract 沿用 smoke：10 steps、global batch 64、sequence length 128、
seed 61、`glm5_debugmodel`。成功结果会被后续同 contract 调用自动跳过；失败目录会
由现有 runner 加时间戳保留，便于修复后续跑。

## 必要环境配置

封装会在导入 Python 前通过 `env -i` 隔离以下软件栈：

```text
CANN=/usr/local/Ascend/cann-9.1.0
Conda=/root/miniconda3/envs/torchtitan-0803-graph-adapt
ATB=/usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_1
HCCL_NPU_SOCKET_PORT_RANGE=auto
HCCL_IF_BASE_PORT=<per-run 62000-63584>
HCCL_CONNECT_TIMEOUT=600
TORCHTITAN_COMM_INIT_TIMEOUT_SECONDS=2400
TORCHINDUCTOR_NPU_BACKEND=default
NPU_INDUCTOR_FALLBACK_LIST=aten.sum,_c10d_functional.all_reduce
TASK_QUEUE_ENABLE=0                  # 当前两个 graph profile 均使用 0
TORCHTITAN_PIPELINE_META_USE_BATCH=0 # PP metadata uses regular P2P
TORCHTITAN_REGISTER_COMPLEX_DTENSOR_STRATEGY=1
TORCHTITAN_NPUGRAPH_UNSAFE_OPS=aten._grouped_mm.default,\
torchtitanturbo_graph.safe_grouped_mm.default  # NPUGraphs
TORCHTITAN_NPUGRAPH_SKIP_ALL=1       # 最终 profile：保留 AOT，关闭 replay
TORCHTITAN_SAFE_EMPTY_GROUPED_MM=1
TORCHTITAN_SAFE_ZERO_NUMEL_TRITON=1
TORCHINDUCTOR_COMPILE_THREADS=1
```

其中 CANN 必须在 Python 启动前切换；只在 `train_smoke.py` 内修改环境变量已经太晚，
因为模块加载期间可能已经导入 `torch_npu`。`HCCL_NPU_SOCKET_PORT_RANGE=auto`
用于避免共享宿主上其他容器占用默认 NPU NIC 端口 16666。需要固定范围时设置：

```bash
GRAPH_HCCL_NPU_SOCKET_PORT_RANGE=61000-61050 \
  tests/glm5_2_graph_debug/run_graph_mode.sh inductor smoke
```

底层 Inductor codegen 可通过 `GRAPH_TORCHINDUCTOR_NPU_BACKEND` 显式切换；封装
会在 Python 启动前把它映射为 `TORCHINDUCTOR_NPU_BACKEND` 并写入报告。默认使用
正式 `default` 实现，只在某个已定位的默认后端缺陷需要验证候选修复时使用
`triton_experimental`，不能无记录地混用两种 codegen 结果。

默认 Inductor smoke 设置
`NPU_INDUCTOR_FALLBACK_LIST=aten.sum,_c10d_functional.all_reduce`，规避默认 smoke
的 1024×256 反向双 reduction 被融合为单核后所有 tile 均发生 UB overflow 的问题。
它只让 `aten.sum` 回退到 ACLNN，其余模型仍由 Inductor/Triton 编译。NPUGraphs
默认不设置该 fallback。`_c10d_functional.all_reduce` 回退用于当前 NPU Inductor
多卡 collective 路径，通信语义仍由原 HCCL process group 执行。

可以用 `GRAPH_NPU_INDUCTOR_FALLBACK_LIST` 指定其他完整 ATen 名称；设置显式空值可
复现无 fallback 行为。报告会记录实际列表。优先使用最小算子集合，不要把
`allfallback` 当作图模式通过结论。

NPUGraphs 捕获不支持 CANN 默认环境中的 `TASK_QUEUE_ENABLE=2`。PP8 Inductor 对照
还显示异步 task queue 会把通信/算子错误延迟到无关位置。common 因此对 Inductor
使用 `TASK_QUEUE_ENABLE=0`，便于稳定编译和准确报错；当前 NPUGraphs 兼容 profile
禁用 replay，因此同样使用 `0`。若关闭兼容降级做原生 capture，需显式覆盖为 `1`。
如工具链版本变化需要复核，可用 `GRAPH_TASK_QUEUE_ENABLE` 覆盖，实际值会进入每次
调用报告。

入口传递 `TORCHTITAN_TASK_QUEUE_ENABLE`；TorchTitanTurbo 的环境初始化现在直接尊重
该显式值，不会再无条件覆盖为 `2`。普通训练不设置该变量时仍使用原默认值 2。

PyTorch 2.14 的 `PipelineStage` 默认通过 batched P2P 发送序列化的 shape/dtype
元数据。PP8 首次并行编译时，HCCL 路径收到损坏的 object size，随后表现为
`Tried to allocate more than 1EB` 和下游 `EOFError`；同一 contract 的 eager PP8
可以通过，实际 NPU 空闲显存也超过 57 GiB，因此这不是正常的显存不足。封装默认
设置 `TORCHTITAN_PIPELINE_META_USE_BATCH=0`，仅把一次性的 pipeline 元数据交换改为
普通 P2P，不改变后续 activation/gradient 通信。可用
`GRAPH_PIPELINE_META_USE_BATCH=1` 恢复上游路径做复现对比。

普通 P2P 会为相邻 stage 建立独立通信链路。冷编译导致 send/recv 到达时间不一致，
rank7 的 CANN plog 已明确记录 `Wait timeout for sockets recv ... timeout[120 s]`、
`remote userrank[6]` 和 `HcclBatchSendRecvGroup`，对应 HCCL 默认 120 秒建链超时。
封装按昇腾对“部分 rank 被耗时任务阻塞”的建议设置 `HCCL_CONNECT_TIMEOUT=600`；
同时把 `comm.init_timeout_seconds` 提到 2400，使 PyTorch watchdog 大于 600 秒建链
等待和 CANN 默认 1836 秒执行超时。分别可由 `GRAPH_HCCL_CONNECT_TIMEOUT` 与
`GRAPH_COMM_INIT_TIMEOUT_SECONDS` 覆盖，实际值进入调用报告。

TP 图编译还会经过 NPU RoPE 的 `aten.complex.default`。当前 PyTorch 2.14 没有为该
算子注册 DTensor pointwise strategy，原始 TP8 会报 `does not have a sharding
strategy`。隔离入口通过 `TORCHTITAN_REGISTER_COMPLEX_DTENSOR_STRATEGY=1` 给它注册
与二元 pointwise 算子一致的 broadcast-aware strategy；普通训练不设置该变量，
不会修改全局行为。可用 `GRAPH_REGISTER_COMPLEX_DTENSOR_STRATEGY=0` 关闭并复现。

共享宿主上还可能有其他容器占用 HCCL host socket 默认端口 60000。该端口与
`HCCL_NPU_SOCKET_PORT_RANGE` 管理的 NPU NIC 端口不是同一项。封装为每次调用
生成 62000–63584 范围内的 `HCCL_IF_BASE_PORT` 并写入报告；如管理员已有端口规划，使用
`GRAPH_HCCL_IF_BASE_PORT` 显式覆盖。

GLM5 MoE 的 `_grouped_mm` 路径在捕获期间会同步 device offset tensor。`thread_local`
模式首先报 107030；即使改为 `relaxed`，仍会在 `CopyKernelOpApi` 的 stream
synchronize 处报 107027，因此放宽 capture error mode 不是有效修复。torch 2.14
当前也没有给 `aten._grouped_mm.default` 添加 `cudagraph_unsafe` 标记。

第一阶段曾通过 `TORCHTITAN_NPUGRAPH_UNSAFE_OPS` 扩展 capture 前置检查，只跳过含
grouped-mm 的 FX 子图。随后 TP8 暴露 graph tree 不接受运行时 `DeviceMesh` 输入，
说明局部 unsafe-op 列表不足以覆盖当前 GLM-5.2 图。因此最终兼容 profile 设置
`TORCHTITAN_NPUGRAPH_SKIP_ALL=1`：整个模型保留 Dynamo/AOT 编译执行，但不进入
NPUGraph replay。`TORCHTITAN_NPUGRAPH_UNSAFE_OPS` 仍保留给后续原生 capture 的定向
诊断，不代表最终 profile 做了局部 replay。

这只作用于隔离进程，不修改 torch/torch_npu 安装。必须区分两种结论：进程成功只
代表 AOT 兼容降级跑通，不代表 NPUGraph 捕获成功。设置
`GRAPH_NPUGRAPH_SKIP_ALL=0 GRAPH_TASK_QUEUE_ENABLE=1` 可恢复原生 capture 调试；当前
版本会复现 grouped-mm 107030/107027 或 TP `DeviceMesh` 输入错误。

编译缓存固定在仓库外的持久化目录：

8 卡默认每个 rank 只使用 1 个 Inductor 编译 worker。未限制时，每个 rank 会创建
大量 worker 并同时调用 `SetDevice`；实测在与其他测试共享设备时触发 CANN E39007
`Inner_Error_Device_Subprocess_Startup_Timeout` / 507033。该项可由
`GRAPH_COMPILE_THREADS` 覆盖。它会降低冷编译并发，但不会降低训练阶段吞吐，且已经
生成的缓存仍可复用。

EP 路由允许某个 expert 分组在某个 step 收到 0 个 token。CANN 9.1 的 grouped-mm
会为该分组计算出 `coreDim=0`，queue=0 时明确报 EE1003，异步模式可能直接 SIGSEGV。
common 默认启用 `TORCHTITAN_SAFE_EMPTY_GROUPED_MM=1`：通过图内 opaque 自定义算子
在运行时为每个空 expert 分组临时补一个零行，执行后只抽取原始行；整组输入为空时
完全绕过 CANN grouped-mm。反向始终按原 offsets 计算，补行不会产生参数梯度。
没有空 expert 分组时直接调用原 grouped-mm。可用
`GRAPH_SAFE_EMPTY_GROUPED_MM=0` 复现原始工具链问题。

在 grouped-mm 之前，token permutation 还会生成动态长度的
`aten.arange/add/sub/_to_copy` pointwise 内核。当该 rank 本步接收 0 个 token 时，
torch_npu Triton runtime 仍会向 CANN launch 零 grid，报同一个 EE1003。common 默认
启用 `TORCHTITAN_SAFE_ZERO_NUMEL_TRITON=1`，在 autotune/launch 之前检查非 reduction
轴的 `*_numel`，为 0 时直接 no-op；空输出 pointwise 本来就没有元素需要计算。
`GRAPH_SAFE_ZERO_NUMEL_TRITON=0` 可关闭该防护做工具链回归验证。

```text
/workspace/y50064852_yyb/.cache/torchtitan-test/graph_mode/
  cann91-torch214-triton321/inductor
  cann91-torch214-triton321/triton
  cann91-torch214-triton321/torch_compile_debug
```

不要把这些编译中间文件移入 Git 仓库。

## 为什么普通 train 已跑通，smoke 仍然很久

早先的 `run_train.sh --compile...` 通过，证明的是当时那一个配置、一个拓扑和一组
shape 能完成训练；它不等价于 smoke 的完整矩阵已经通过。默认 smoke 不传拓扑时
会串行验证 15 种并行组合，而且每种都要求 10/10 steps 和独立 passed manifest。

耗时主要来自以下几部分：

1. Inductor 首次遇到一个新 graph/shape/并行布局时要完成 Dynamo、AOTAutograd、
   lowering、Triton autotune、CANN 编译和二进制装载。第一步常比后续 step 慢几个
   数量级。
2. 8 卡拓扑是 8 个 rank 分别编译本 rank 的 forward/backward 子图。TP、CP、PP、
   EP、FSDP 及它们的组合有不同 DTensor placement、collective 和动态 shape，不能
   因为 single 的 cache 已热就全部复用。
3. 为避免 8 rank 同时创建大量编译子进程并触发 CANN 507033/stream 枯竭，common
   默认每 rank 只用 1 个编译 worker。这提高稳定性，但冷编译墙钟时间会更长。
4. 10 steps 不是单纯重复。EP 的零 token 分支直到 `fsdp2-tp4-ep8` step 8 才出现；
   只跑 1–2 steps 会漏掉已实际定位的 `coreDim=0` 问题。数值检查也需要多步才能
   发现 collective 被错误融合后逐步放大的梯度。
5. 调试阶段每发现一个独立问题都要保留失败报告、修复并重跑对应拓扑。缓存能复用
   已经相同的编译结果，但 fallback、graph 结构、shape 或后端策略改变后必须产生
   新结果，不能把旧成功当成新配置的证据。
6. NPUGraphs 还包含 AOT graph 编译、首次 capture/replay 建立和不安全子图检查；
   graph benchmark/combination 通常还会额外运行 eager reference、candidate、多个
   repeat、profiler 和 CPU compare，因此会比单次 train 更久。

实测也能看到冷/热差异：single Inductor 正式 10-step 运行约 172 秒；TP8 冷编译约
683 秒；已有同 contract 的 passed manifest 被跳过时只需约 1 秒。完整矩阵应使用
断点续跑，不要每次加 `--force` 重做已经通过的拓扑。只有需要替换同 contract 的
既有结果时才使用 `--force`，因为它会删除该拓扑当前结果后重新生成。

## 结果目录

每个实际 topology 继续完全使用 smoke 的目录和 manifest：

```text
smoke_runs/<device-config-contract-graph>/<topology>/
  manifest.json
  runtime.log
  trainer_output/
```

隔离封装另外记录一次调用级别的环境和汇总，不复制训练中间产物：

```text
graph_debug_runs/smoke-suite-<backend>-<timestamp>-<pid>/
  logs/runtime.log
  reports/report.md
```

二者均由仓库 `.gitignore` 排除；本目录只保留脚本和文档。

## 实测状态

### Inductor 单卡

默认 smoke contract 已完成 10/10 steps，最终结论为 **PASSED**：

```text
graph_debug_runs/smoke-suite-inductor-20260824-174917-1691961/reports/report.md
smoke_runs/npu-glm5_debugmodel-s10-b64-seq128-seed61-inductor-model/single/manifest.json
```

原始无 fallback 运行在 1024×256 反向双 reduction 上失败。默认 NPU codegen 为每个
候选 tile 计算出的 UB 需求都是 2,162,688 bit，而设备可用 1,572,864 bit，最终报
`No valid triton configs`。`triton_experimental` 虽绕过该编译点，但 autotune 期间在
softmax kernel 触发 507035 vector core exception，不能作为稳定替代。保持正式
`default` codegen 并最小回退 `aten.sum` 后通过。

### NPUGraphs 单卡

默认 smoke contract 已完成 10/10 steps，最终结论为 **PASSED（含安全降级）**：

```text
graph_debug_runs/smoke-suite-npugraphs-20260824-181603-1824452/reports/report.md
smoke_runs/npu-glm5_debugmodel-s10-b64-seq128-seed61-npugraphs-model/single/manifest.json
```

日志明确记录：

```text
[torchtitan-npu] NPUGraph capture skipped for unsafe ops: aten._grouped_mm.default
NPUGraph: skipped — configured incompatible op
```

因此训练和 AOT graph 已跑通，但含 `_grouped_mm` 的子图没有做 NPUGraph replay。
排障中依次确认了以下独立问题：

1. Turbo 把 `TASK_QUEUE_ENABLE` 重写为 2，捕获立即失败；入口会在 Turbo patch 后
   恢复 profile 指定值。当前兼容降级使用 0；原生捕获实验使用 1。
2. 共享宿主的 HCCL host socket 60000 已被其他容器占用；入口现使用每次运行独立
   的 62000–63584 `HCCL_IF_BASE_PORT`。
3. `_grouped_mm` 在 `thread_local` 和 `relaxed` 捕获下分别报 107030/107027；现改为
   捕获前识别不安全子图并由后端安全跳过，不再放宽 capture mode。

### 最终多卡矩阵（2026-08-25）

最终复验结果为：Inductor 15/15 个拓扑完成 10/10 steps；NPUGraphs profile 15/15
个拓扑在保留 AOT graph、显式禁用 replay 的兼容降级下完成 10/10 steps。完整矩阵、
manifest 规范和报告路径见 `FINAL_GRAPH_DEBUG_REPORT.md`。最终汇总报告为：

```text
graph_debug_runs/smoke-suite-inductor-20260825-172213-4113992/reports/report.md
graph_debug_runs/smoke-suite-npugraphs-20260825-172207-4113845/reports/report.md
```

NPUGraphs 原生 replay 仍不算通过：关闭 `GRAPH_NPUGRAPH_SKIP_ALL` 后，grouped-mm
捕获返回 107030/107027，TP AOT wrapper 还会因 `DeviceMesh` 输入被 graph tree 拒绝。

### 历史中断快照（已被上面的最终结果取代）

在该中断快照时，Inductor 已有 14/15 个标准拓扑完成 10/10 steps 并写入 passed
manifest：

| 拓扑 | Inductor | NPUGraphs |
|---|---|---|
| `single` | PASSED | PASSED（含安全降级） |
| `ddp2` | PASSED | PASSED（含安全降级） |
| `ddp8` | PASSED | 待实跑 |
| `fsdp8` | PASSED | 待实跑 |
| `tp8` | PASSED | 待实跑 |
| `cp8` | PASSED | 待实跑 |
| `pp8` | PASSED | 待实跑 |
| `ep8` | PASSED | 待实跑 |
| `fsdp2-tp4` | PASSED | 待实跑 |
| `fsdp2-cp4` | PASSED | 待实跑 |
| `tp2-cp4` | PASSED | 待实跑 |
| `fsdp4-tp2` | PASSED | 待实跑 |
| `fsdp2-pp4` | PASSED | 待实跑 |
| `fsdp2-tp2-pp2` | PASSED | 待实跑 |
| `fsdp2-tp4-ep8` | 修复后待 8 卡复验 | 待实跑 |

Inductor 的 14 个通过结果位于：

```text
smoke_runs/npu-glm5_debugmodel-s10-b64-seq128-seed61-inductor-model/<topology>/manifest.json
```

`fsdp2-tp4-ep8` 在旧 launcher 上稳定运行到 step 7，step 8 的 rank 7 因动态接收
token 数为 0，触发 CANN `coreDim=0`/SIGSEGV。最新代码已同时保护普通
`NPUCachingAutotuner` 与覆盖 `run()` 的 `NPUSymbolicGroupedAutotuner`；不使用 NPU
的 dispatch 探针已验证两条路径都会在 launch 前 no-op：

```text
graph_debug_runs/launcher-command-inductor-20260825-091227-3808822/reports/report.md
```

最新 8 卡复验没有进入编译：当时每张卡都有约 8 个其他进程，device 0 在创建 FSDP
stream 时返回 `EE1023 Too many streams`。该次基础设施失败单独保留为：

```text
graph_debug_runs/smoke-suite-inductor-20260825-090754-3805779/reports/report.md
```

后续只读追踪确认，这些占用来自并行运行的 checkpoint
`--failure-mode all --topology=fsdp4-tp2`。其 `sigkill` 场景出现旧 rank 仍存活而恢复
torchrun 已启动的重叠现场。图模式调试不会自动结束这些进程；必须先由 checkpoint
任务自身或用户侧清理并确认 `npu-smi info` 无残留，再继续 8 卡验证。

后续复查时总 checkpoint 调度器仍以 `--topology all --failure-mode all` 运行，每卡
仍有 7 个相关 Python 进程和约 22 GiB HBM 占用；因此该次恢复没有启动 smoke。

因此在该中断快照中不能把 `fsdp2-tp4-ep8` 写成 PASSED，也不能声称 NPUGraphs
15 种拓扑全部跑通。当时计划在设备空闲后依次执行：

```bash
tests/glm5_2_graph_debug/run_graph_mode.sh inductor smoke \
  --topology fsdp2-tp4-ep8
tests/glm5_2_graph_debug/run_graph_mode.sh npugraphs smoke
```

PP8 通过结果同时验证了普通 metadata P2P、600 秒 HCCL 建链等待、2400 秒
process-group timeout 和 Inductor `TASK_QUEUE_ENABLE=0`。最终判定只按实际 manifest；
资源占用或设备初始化失败单独标为基础设施问题，不计作编译后端缺陷。
