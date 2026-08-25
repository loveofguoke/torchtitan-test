# NPU 图模式底层问题定位与提单交接

本文把 GLM-5.2 图模式实验中的应用侧兼容办法，映射到 PyTorch、torch_npu、
op-plugin、CANN/HCCL 中可能真正修复的位置。目标是支持人工源码核查、构造最小复现、
向底层团队提单，以及在独立 torch_npu 分支上验证 patch。

这不是“所有根因均已证明”的声明。文中使用三种置信度：

- **已定位**：错误条件和实际抛错/同步/launch 调用点能够由源码与日志直接对应；
- **高置信候选**：调用链已定位，但需要底层 trace 或最小复现确认最终责任模块；
- **待分界**：只知道经过该模块，不能仅凭训练日志认定是该模块的 bug。

## 1. 版本基线与行号规则

服务器实验基线为 Ascend 910B2、CANN 9.1.0、torch/torch_npu 2.14 nightly、
triton-ascend 3.2.1。Python 包基准目录为：

```text
/root/miniconda3/envs/torchtitan-0803-graph-adapt/lib/python3.12/site-packages
```

可修改的 Ascend PyTorch 源码 checkout 为：

```text
/home/y50064852_yyb/pytorch
```

记录本文时该源码为 `cc65aaf7f`，且工作树已有其他未提交/生成内容。本文没有修改它。
源码 checkout 与已安装 nightly wheel 不保证逐字节一致，因此：

1. Python 行号以已安装环境为实验事实；
2. C++ 文件/函数以本地 Ascend PyTorch checkout 为 patch 导航；
3. 提单时应同时附提交号、wheel 版本和函数名；后续版本行号漂移时以函数名为准；
4. CANN 算子内部为闭源时，只记录 ACLNN API 边界和错误码，不虚构内部函数名。

## 2. 总览：绕过位置与真正修复位置

| ID | 当前兼容办法 | 根治候选仓库与模块 | 置信度 | 根治后必须移除的开关 |
|---|---|---|---|---|
| G001/G002 | 干净进程只加载 CANN 9.1 | torch_npu–CANN 版本矩阵、ACL header/API 装包检查 | 已定位 | 仍应保留环境隔离；不再允许错误版本静默启动 |
| G003 | `aten.sum` 回退 ACLNN | `torch_npu._inductor` reduction tile 生成、UB 合法性过滤与 codegen | 高置信候选 | `NPU_INDUCTOR_FALLBACK_LIST` 中的 `aten.sum` |
| G011 | `_c10d_functional.all_reduce` 回退 | PyTorch Inductor collective lowering ↔ ProcessGroupHCCL stream/wait/alias 语义 | 待分界 | fallback 中的 `_c10d_functional.all_reduce` |
| G007 | NPUGraph replay 全局 skip | op-plugin grouped-mm shape 推导的 `.item()`/同步；NPUGraph unsafe-op 判定 | 已定位到同步链 | `TORCHTITAN_NPUGRAPH_SKIP_ALL` |
| G016 | NPUGraph replay 全局 skip | torch_npu graph-tree runtime input contract，或 PyTorch AOT/DTensor 输入封装 | 已定位到抛错点 | `TORCHTITAN_NPUGRAPH_SKIP_ALL` |
| G013/G015 | 空组 padding、全空 bypass、INT32 offsets | op-plugin grouped-mm empty contract；CANN `aclnnGroupedMatmulV4/V5` | 已定位到调用边界 | `TORCHTITAN_SAFE_EMPTY_GROUPED_MM` |
| G014 | runtime launch 前检查 `*numel==0` | torch_npu Inductor autotuner/grid policy | 已定位 | `TORCHTITAN_SAFE_ZERO_NUMEL_TRITON` |
| G008 | 运行时注册 `aten.complex` strategy | PyTorch DTensor pointwise operator registration | 已定位 | `TORCHTITAN_REGISTER_COMPLEX_DTENSOR_STRATEGY` |
| G009 | PP metadata 使用 `use_batch=False` | PyTorch PipelineStage object P2P ↔ ProcessGroupHCCL batched P2P | 高置信候选 | `TORCHTITAN_PIPELINE_META_USE_BATCH=0` |

G004/G005/G010/G012/G017 是端口、timeout、编译并发或调试 profile 的工程配置，不应
混同为单个后端代码 bug。它们仍应保留为可配置 launcher policy；只有底层默认值或错误
诊断需要另行改进。

## 3. G001/G002：torch_npu 2.14 与 CANN 9.0/9.1 不兼容

### 3.1 已确认调用点

失败签名是 Triton launcher/PCH 编译期间 `aclmdlRICondHandle`、
`aclmdlRICondTaskParams` 未定义。当前 Ascend PyTorch 源码中：

- `third_party/acl/inc/acl/acl_rt.h:1027-1044` 定义这两个类型；
- 同文件 `:5304-5325` 声明 `aclmdlRICondHandleCreate`、
  `aclmdlRICondHandleGetCondPtr`、`aclmdlRIAddCondTask`；
- `torch_npu/csrc/core/npu/interface/AclInterface.h:304-315` 暴露 torch_npu wrapper；
- `torch_npu/csrc/core/npu/interface/AclInterface.cpp:1461-1515` 动态加载这些符号；
- `torch_npu/csrc/core/npu/NPUGraph.cpp:495-566` 在 graph 条件任务路径使用它们。

CANN 9.0 header 不含这些类型，而 2.14 源码已经在编译接口中引用它们，所以这是明确的
版本契约不匹配，不是 GLM 模型或 TorchTitan 配置错误。

隔离安装和复验步骤见
[CANN 9.1.0 安装文档](../glm5_2_graph_debug/CANN_9_1_INSTALLATION.md)。

### 3.2 建议根治

- torch_npu 发布物应声明并在安装/导入时检查最低 CANN 版本；不能等到运行时 PCH
  编译才以无关的 `InductorError` 暴露；
- wheel 构建使用的 ACL header 与运行时支持矩阵必须可查询；
- 若产品要求兼容 9.0，需要通过编译期 feature guard 和动态符号探测隔离新 NPUGraph
  API，而不是直接引用旧 header 中不存在的类型；
- CANN 安装包应避免 `latest`、全局环境变量或多个 toolkit 的 include/lib 静默混装。

### 3.3 提单最小材料与验收

记录 `torch.__version__`、`torch_npu.__version__`、`ASCEND_HOME_PATH`、实际解析到的
`libascendcl.so`、编译命令中的 include 路径，并只编译一个包含 `acl/acl_rt.h` 且声明
上述类型的 C++ 文件。验收标准是错误组合在启动前给出明确版本错误；支持组合能够冷
缓存编译最小 Triton kernel 和 NPUGraph 条件任务。

## 4. G003：reduction tile 全部超过 UB

### 4.1 已知事实和代码路径

失败 kernel 为 `triton_per_fused_add_mul_sum_29`。全部候选需要 2,162,688 bit UB，设备
上限为 1,572,864 bit，最终报 `No valid triton configs for kernel`。当前仅将
`aten.sum` 回退 ACLNN。

已安装 torch_npu 的关键位置：

- `torch_npu/_inductor/lowering.py::make_reduction`（约 137 行）：NPU reduction
  lowering 入口；
- `torch_npu/_inductor/lowering.py::_register_npu_inductor_fallbacks`（约 176 行）：
  fallback 注册边界；
- `torch_npu/_inductor/runtime/tile_generator.py::TileGenerator`（约 20 行）；
- `torch_npu/_inductor/runtime/fasta_autotune.py::FastATileGenerator`（约 212 行）与
  `NPUFastAutotuner`（约 841 行）：tile/config 候选产生与 autotune；
- `torch_npu/_inductor/runtime/triton_heuristics.py::_precompile_configs_with_vf_retry`
  （约 1093-1120 行）：过滤后无候选并抛出该错误；
- `torch_npu/_inductor/codegen/triton.py::should_use_persistent_reduction`
  （约 923 行）：persistent reduction 选择边界。

`triton_experimental` 能绕过这个 reduction，但随后 softmax autotune 出现 507035
vector-core exception，所以不能把“切 experimental backend”当成修复。

### 4.2 推荐 patch 方向

优先在 tile/config 生成层解决，而不是在 TorchTitan 中继续扩大 fallback：

1. 候选生成时以目标 SoC 的 UB 上限过滤，并保留至少一个小 tile 的安全候选；
2. 当 fused persistent reduction 无合法候选时，自动降低 fusion、拆分 reduction，或
   切换非 persistent reduction；
3. 错误信息输出 shape、dtype、每个候选的 UB 估算、设备上限和被拒原因；
4. 把本次 1024×256 形状的 fused add/mul/sum 加入冷缓存回归测试。

### 4.3 分界实验和验收

- `GRAPH_NPU_INDUCTOR_FALLBACK_LIST=` 冷缓存运行 single，保存 generated code 和
  autotune candidate dump；
- 禁止只利用旧 cache 判定通过；分别验证 default 与 experimental codegen；
- 根治验收：删除 `aten.sum` fallback 后，single、TP、PP、EP 和最复杂组合均通过，
  并与 eager 做逐 step loss/gradient 数值比较，而非只看 10-step 退出码。

## 5. G011：compiled functional all-reduce 数值异常

### 5.1 当前只能定位到交界，尚不能判定最终责任方

观察到 PP/组合拓扑在 step 1 出现非有限 loss/gradient；仅回退
`_c10d_functional.all_reduce` 后恢复。调用路径为：

- PyTorch `torch/distributed/_functional_collectives.py::all_reduce`（161 行）
  dispatch 到 `_c10d_functional.all_reduce`；
- PyTorch `torch/_inductor/_functionalize_collectives.py::_emit_collective_chain` 和
  `_rewrite_allreduce_` 把 functional/in-place collective 串联；
- PyTorch `torch/_inductor/comm_lowering.py::register_comm_lowerings`（197 行）内的
  `_all_reduce`（225 行）clone/contiguous 输入，并创建 `ir._AllReduce_Kernel`，
  最终使用 `_c10d_functional.all_reduce_.default`；
- torch_npu `torch_npu/csrc/distributed/ProcessGroupHCCL.cpp::allreduce`（4691 行），
  约 4724 行进入 `hcclAllReduce`/`OpCommand::RunOpApiV3`。

这里可能是 Inductor clone/alias/in-place functionalization、collective wait/调度顺序、
contiguous/dtype 转换，也可能是 ProcessGroupHCCL stream/event 语义。训练日志尚不足以
断言 HCCL 数学内核错误，提单时应标为**待分界**。

### 5.2 必须完成的最小分界矩阵

固定 2 rank、固定 tensor、dtype、reduce op 和 process group，逐项比较：

1. `dist.all_reduce` eager；
2. `_c10d_functional.all_reduce` eager + 显式 wait；
3. 只 compile 该函数；
4. compile 前后分别强制 contiguous/clone；
5. 开关 `reorder_for_compute_comm_overlap`；
6. 每次 collective 后立刻同步，与延迟消费输出两种情况；
7. dump FX graph、generated wrapper、collective sequence number 和 HCCL stream/event。

若 2 正确、3 错误，优先提 PyTorch Inductor；若 2 已错误，优先进入 torch_npu
ProcessGroupHCCL；若仅异步消费错误，重点核查 wait tensor 与 stream lifetime。验收必须
删除 fallback，重复多轮对比 eager/compiled 的逐元素结果，并覆盖 PP 与组合拓扑。

## 6. G007：grouped-mm 在 NPUGraph capture 中同步 device tensor

### 6.1 已确认同步调用链

日志表现为 `thread_local` capture 下 host/device copy 107030，`relaxed` 下 stream
synchronize 107027。源码链条可以直接对应：

- op-plugin
  `third_party/op-plugin/op_plugin/ops/opapi/GroupedMatmulKernelNpuOpApi.cpp`
  `::npu_grouped_matmul`（101 行）；
- 同函数约 227/234 行对 NPU `group_list_real[i]` 调用 `.item<int64_t>()` 来计算输出
  shape；
- `torch_npu/csrc/aten/ops/op_api/CopyKernelOpApi.cpp`
  `::copy_between_host_and_device_opapi`（70 行）处理 D2H；同步分支在 89 行调用
  `aclrtSynchronizeStream`，约 100 行执行/报告 memcpy；
- 随后 grouped-mm wrapper 在约 335/340 行进入 `aclnnGroupedMatmulV4/V5`；
- torch_npu `torch_npu/utils/_graph_tree.py::check_for_skip`（约 256 行）当前检查 mutation、
  device 与已知不兼容节点，但没有自动识别此同步语义。

因此“放宽 capture error mode”无法根治：同步操作本身仍然存在。

### 6.2 推荐 patch 方向

- 首选：op-plugin 不通过 device tensor `.item()` 推导输出 shape。使用 fake/meta 阶段
  已知的符号 shape、host-side routing metadata，或修改 op contract 显式传入输出布局；
- 在真正 capture-safe 前，torch_npu graph-tree 应将对应 grouped-mm op 纳入内建 unsafe
  registry，带清楚原因跳过该子图；这只是安全降级，不算 native replay 根治；
- CANN 若能支持 device offsets 的 capture-safe grouped matmul，也仍需先移除 wrapper 中
  的 D2H 同步；仅改 CANN capture mode 不够。

验收要求 `TORCHTITAN_NPUGRAPH_SKIP_ALL=0`，单卡和 EP native capture/replay 无
107030/107027；动态 offsets、多次 replay、forward/backward 都与 eager 一致。

## 7. G016：NPUGraph tree 拒绝 `DeviceMesh` runtime input

### 7.1 精确抛错点

TP8 的 AOT wrapper 有 21 个运行时 `DeviceMesh(tp=8)` 输入。已安装环境：

- PyTorch `torch/_inductor/utils.py` 约 155 行将 `InputType` 定义为
  `Tensor | int | SymInt | None`；
- torch_npu `torch_npu/npu/_graph_tree.py::_allocate_and_copy_recording_inputs`
  在 1680 行开始；1699-1703 行对非 Tensor 输入只接受 `int`，否则抛
  `RuntimeError("check isinstance(inp, int) fail")`；
- `torch_npu/utils/_graph_tree.py::check_for_skip` 约 256 行没有在 capture 前检查这种
  runtime object，因此不是清晰 skip，而是在记录输入时硬失败。

### 7.2 两条根治路线

1. **torch_npu graph-tree**：若 `DeviceMesh` 是 immutable/static metadata，将其固定在
   captured callable 中，记录/replay 时校验 identity/mesh invariant，而不是复制；若不
   支持，preflight 必须带类型和 input index 清晰 skip。
2. **PyTorch AOT/DTensor**：检查为何 mesh metadata 被提升为 runtime input；若不参与
   数值计算，应在 export/partition 中闭包化或转换为受支持的静态 metadata。

第一条中的“清晰 skip”只改善健壮性；只有 `GRAPH_NPUGRAPH_SKIP_ALL=0` 下 TP8 真正
capture/replay 才算根治。验收还应改变 mesh 或 shape，确认 invariant guard 不会错误复用
旧图。

## 8. G013/G015：空 expert、grouped-mm 与 offsets dtype

### 8.1 已确认边界

EP 某 rank/expert 收到 0 token 时，CANN 报 `coreDim=0`/EE1003，异步路径可能
SIGSEGV。实际 op 路径为：

- torch_npu Inductor `torch_npu/_inductor/kernel/mm_grouped.py`
  `::_tuned_grouped_mm_common`（约 157 行）和
  `::_register_npu_inductor_grouped_mm`（约 274 行）；
- fallback/ACLNN 路径进入 op-plugin
  `GroupedMatmulKernelNpuOpApi.cpp::npu_grouped_matmul`；
- 约 335/340 行调用 CANN `aclnnGroupedMatmulV4/V5`。

当前 Turbo 在 `graph_compat.py::_install_safe_empty_grouped_mm` 中对部分空组补一行、全空
直接返回、显式实现 backward，并强制 padding offsets 的 `cumsum(..., dtype=int32)`。
这保护了训练，但增加内存/算子并改变应用执行路径。

### 8.2 根治建议

- op-plugin 在 total M=0 时按 schema 返回正确 shape 的空 tensor，不能发起 zero-core
  kernel；
- 对部分空组/repeated cumulative offsets 定义稳定语义，不要求上层伪造 token；
- CANN `aclnnGroupedMatmulV4/V5` 明确定义 total-empty 与 partial-empty contract；若内核
  暂不支持，应由 ACLNN wrapper 安全分流，不能 EE1003 或 SIGSEGV；
- schema/meta/kernel 对 offsets dtype 保持一致；若只支持 INT32，应在入口明确检查并
  给出可读错误，相关构造算子不要无意提升为 INT64。

验收需覆盖 total empty、首/中/尾 expert empty、多个连续 empty、非空基线、INT32/错误
INT64，以及 eager/Inductor、forward/backward。通过后关闭 Turbo safe grouped-mm patch。

## 9. G014：零长度 Triton kernel 仍被 launch

### 9.1 精确 patch 对象

动态 EP 在 step 8 产生 `xnumel=0`，默认 runtime 仍 launch，造成 EE1003/SIGSEGV：

- `torch_npu/_inductor/runtime/triton_heuristics.py::NPUCachingAutotuner.run`
  （已安装环境约 1839 行）；
- 同文件 `::NPUSymbolicGroupedAutotuner.run`（约 2458 行）。

Turbo 同时包装这两个 `run`，匹配所有非 reduction 的 `*numel` 参数并在 0 时 no-op。
只 patch base autotuner 或只匹配 `_numel` 都会漏掉实际路径。

同一包的实验 runtime
`torch_npu/_inductor/triton_experimental/npu_triton_heuristics.py` 约 448 行已经包含针对
`xnumel==0`/CANN `coreDim==0 (EE1003)` 的处理：将 grid 保底为 1，让 masked program
no-op。这可以作为向默认 runtime 移植/统一策略的参考，但要验证所有 kernel 都有正确
mask；否则 runtime 直接跳过仅适用于确认无输出副作用的非 reduction kernel。

### 9.2 验收

底层实现应统一处理静态 0、symbolic 0、普通 autotuner 和 symbolic grouped autotuner；
对 reduction/有 identity 输出的 kernel 不能盲目跳过。测试至少包括空 output pointwise、
空输入 reduction、dynamic shape 0↔非0 多次切换，以及 EP 真实零 token step。通过后关闭
`TORCHTITAN_SAFE_ZERO_NUMEL_TRITON`。

## 10. G008：`aten.complex` 缺 DTensor strategy

已安装 PyTorch 的注册位置是：

- `torch/distributed/tensor/_ops/_pointwise_ops.py::_register_single_dim_pointwise`
  （约 100 行）；
- 同文件 `_extra_pointwise_ops`（481 行）；
- `_get_pointwise_ops_from_tag`（556 行）和最终 pointwise 注册循环（574-579 行）。

`aten.complex.default` 未在当前表中，所以 TP RoPE 报没有 sharding strategy。Turbo 只是
调用现有 broadcast-aware pointwise 注册器。

根治应在 PyTorch DTensor 中完成：如果 `aten::complex` 语义满足 pointwise tag 规则，
补 operator tag；否则显式加入 `_extra_pointwise_ops`，并测试 real/imag 的 Replicate、
Shard、broadcast、uneven shard 与 complex dtype 输出。验收是在不运行 Turbo 注册 patch
时 TP8 compile 成功，且 PyTorch DTensor 单测覆盖该 op。

## 11. G009：PipelineStage batched metadata P2P

### 11.1 调用边界

PP8 初始化出现 `Tried to allocate more than 1EB`，随后 `EOFError`，说明 serialized
object size/header 已损坏，并非正常模型 HBM OOM：

- PyTorch `torch/distributed/pipelining/stage.py::PipelineStage._recv_meta`
  （已安装环境约 1816 行）和 `::_send_meta`（约 1832 行）默认 `use_batch=True`；
- PyTorch distributed object collective 负责长度 tensor 与 payload 序列化；
- torch_npu `torch_npu/csrc/distributed/ProcessGroupHCCL.cpp::pointToPoint`
  位于约 4401/4675 行，是 HCCL P2P 边界。

当前只把一次性 metadata 的 object send/recv 改为 `use_batch=False`，activation/gradient
P2P 不变。它证明非 batched 路径可用，但还不能仅凭此认定 bug 一定在
ProcessGroupHCCL；也可能是 PyTorch batched object 协议与该 backend capability 不匹配。

### 11.2 分界与根治

- 用 2 rank 只发送一个含 shape/dtype 的 object，分别比较 `use_batch=False/True`；
- 分别记录长度 tensor、payload bytes、src/dst global rank、group-local rank 和 P2P op
  顺序；
- 如果 HCCL 明确不支持这类 batched object P2P，PyTorch PipelineStage 应按 backend
  capability 自动选择普通 P2P；
- 如果宣称支持，则修复 batch isend/irecv 的 header/payload ordering、rank 映射或
  work/wait lifetime；接收端在分配前还应校验长度上限，避免损坏 header 导致 1EB 分配。

验收为 `TORCHTITAN_PIPELINE_META_USE_BATCH=1` 的 PP8 多轮冷启动，无 1EB/EOF，metadata
一致且训练 activation/gradient 路径无回归。

## 12. G004-G006/G010/G012/G017：运行环境和调试 profile 问题

这些问题必须记录，但目前没有证据支持把它们全部作为 torch_npu/CANN 代码缺陷提单：

| ID | 已确认位置 | 性质与后续动作 |
|---|---|---|
| G004 | `graph_env_common.sh` 的 `HCCL_NPU_SOCKET_PORT_RANGE` | NPU NIC 端口是宿主共享资源。launcher 保持可配置/auto；若 HCCL `auto` 仍冲突，再携带占用端口和 communicator 日志提 HCCL 端口分配问题。 |
| G005 | `graph_env_common.sh` 的 `HCCL_IF_BASE_PORT` | host socket 与 NPU NIC 是两个端口面。实验每次生成 base port 是隔离策略，不是模型 patch。可推动 HCCL 提供命名空间感知的动态端口或冲突时明确报告两类端口。 |
| G006 | `torchtitanturbo/patch.py::set_environ_variable`（30-39 行） | 旧实现无条件覆盖 queue；当前实现读取并校验 `TORCHTITAN_TASK_QUEUE_ENABLE`，属于 Turbo 已修复问题。三仓源码安装复验后方可标“已验证”。 |
| G010 | `train_npu.py`（10-31 行）的 `--comm.init_timeout_seconds` 注入与 `HCCL_CONNECT_TIMEOUT` | 冷编译使 PP rank 到达通信点的时间差变大。timeout 是实验 policy；若已经进入同一 collective 仍超时，才按通信问题提单。 |
| G012 | PyTorch `torch/_inductor/config.py` 约 1452-1464 行、`async_compile.py` 与 `compile_worker/subproc_pool.py` | 多 rank × 多 compiler worker 同时 `SetDevice` 造成 E39007。当前每 rank 1 worker 是资源控制；底层可改进 NPU compile worker 的 device 初始化、并发上限和错误诊断。 |
| G017 | `graph_env_common.sh` 的 `TASK_QUEUE_ENABLE` profile | replay 已 skip 时 queue 1 的组合通信不稳定，queue 0 是降级 profile。只有 native capture 开启时才单独验证 queue 1；不能据此声称 queue runtime 根因已确定。 |
| G018 | checkpoint failure-mode 任务的进程生命周期 | 残余 rank 占用 streams/HBM，是共享测试资源污染，不是图编译 root cause。应由 checkpoint runner 保存 PID/token 并只清理自己创建的进程；不能在图 launcher 中模糊 kill。 |
| G019 | 验证命令自身 | shell 被误传给 `py_compile`，已改为 shell 用 `bash -n`、Python 用 `py_compile`。这是调试历史，不向底层提单。 |

这类配置的验收重点是“默认值不覆盖显式用户选择、报告能打印最终有效值、并发运行互不
污染”。它们不应被偷偷硬编码进 device-neutral TorchTitan。

## 13. 提单模板

每个底层问题单独提单，不把 reduction、collective、NPUGraph、grouped-mm 混成一个单。
建议固定包含：

```text
标题：<backend>/<module>: <最小错误签名>
硬件：Ascend 910B2 × <rank 数>
软件：CANN / torch / torch_npu / triton-ascend 完整版本与 commit
环境：关键 CANN、HCCL、TASK_QUEUE、compile/cache 变量
最小复现：单文件或最小 topology 命令
期望：eager 或非 capture 路径的语义
实际：首个错误码、stack、首次非有限值或错误结果
分界矩阵：哪些 eager/compile/capture/batch 开关正确或失败
源码定位：仓库、文件、函数、当前版本行号
当前绕过：环境变量或 Turbo 函数，以及为何不算根治
附件：runtime.log、plog、FX graph、generated code、环境报告
验收：关闭哪个 workaround，跑哪些单测/拓扑/精度比较
```

原始训练证据和历史 invocation 对应 G001-G019，见
[失败与修复历史](../glm5_2_graph_debug/experiments/reports/failures.md)。不要把包含编译 cache
的整个 `.cache` 提交到 Git；只抽取复现所需 generated source、配置和错误片段放入提单
附件。涉及数值问题时同时保存 eager 与 compiled tensor 摘要、首个不一致 index 和误差，
不能只附最终 loss。

## 14. 根治完成定义

某问题只有同时满足以下条件，才可从“降级通过/待复验”改成“根治”：

1. 修复落在责任层并有该层单元/集成回归测试；
2. 在全新编译 cache 下重现旧失败用例不再失败；
3. 对应 Turbo/test workaround 显式关闭或删除后仍通过；
4. 不只跑 10-step smoke，还完成 eager-vs-graph 数值验收；
5. 涉及通信或动态 shape 时覆盖多轮、不同 rank/shape、0↔非0 边界；
6. NPUGraph 必须确认发生 native capture/replay，不能把 `skip` 日志当成通过；
7. 结果报告记录 patch commit、命令、日志路径和 cache 是否冷启动。
