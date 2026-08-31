# GLM-5.2 NPU 图模式编译分析与面试问答

本文解释当前三仓实现中，GLM-5.2 训练如何从 Python eager 程序变成 Ascend NPU 上的
Inductor 图执行。重点回答四类问题：整图到底指什么、图断裂如何判断、动态 shape 如何
处理、算子兼容性如何分层。最后给出可直接用于大模型训推岗位面试的问答和排障方法。

本文以 2026-08-28 的确定性单卡冷缓存验证为主要证据。它已经通过 10 个训练 step，
但不能替代正在执行的 5000-step eager/Inductor 精度矩阵。实验结论和编译结构结论在文中
严格分开。

## 1. 先给结论

1. 当前所谓“整图”是 **一个 TransformerBlock 内部的前向或反向完整 FX 图**，不是把
   整个模型、优化器和训练循环合成一个图。
2. TorchTitan 对 `model.layers` 中每个 block 调用
   `compile(backend="inductor", fullgraph=True)`。`fullgraph=True` 下真正的图断裂会令
   首次捕获直接失败，不会静默切成多个 eager/compiled 小段。
3. 冷缓存单卡证据产生四类主编译单元：dense forward、MoE forward、MoE backward、
   dense backward。重复层结构复用相同代码和 cache，所以不是每一层都生成一套不同代码。
4. 本次 10-step trace 没有 graph break、guard failure 或 step 2 以后 recompile 证据；
   `[0/0]` 与 `[0/1]` 是同一 block 代码的 dense/MoE 两种静态 specialization，不是运行中
   反复断图。
5. `extern_kernels.mm/bmm`、NPU fused op、custom op 和 Inductor fallback 都可以保留在
   compiled graph 中。**没有生成 Triton kernel不等于发生图断裂**。
6. 固定 batch/sequence/model hidden shape 是静态的；MoE 每个 expert 的 token 数是数据
   相关的。当前实现通过 scalar capture、unbacked symbolic integer、显式 `output_size` 和
   带 fake/autograd 注册的安全 grouped-mm custom op，把这部分动态性留在完整图中。
7. 当前不是“零降级”交付：`aten.sum` 和 compiled all-reduce 仍使用外部 fallback；
   grouped-mm、零 numel、TP complex strategy、PP metadata P2P 等依赖 opt-in 兼容补丁。
8. NPUGraph 与 Inductor 不是同一个概念。当前 NPUGraph profile 默认关闭原生 replay，
   因而不能把它的通过解释为 NPU Graph capture/replay 性能已经根治。

## 2. 代码入口和编译边界

### 2.1 三仓职责

| 仓库 | 当前职责 | 关键位置 |
|---|---|---|
| TorchTitan | 模型、并行化顺序、device-neutral `torch.compile` 入口 | `torchtitan/distributed/compile.py::apply_compile`、`torchtitan/models/glm5/parallelize.py::parallelize_glm5` |
| TorchTitanTurbo | Ascend 特有算子、DTensor、runtime 和编译兼容 | `torchtitanturbo/tools/graph_compat.py`、`torchtitanturbo/tools/compile.py` |
| torchtitan-test | 环境隔离、实验编排、精度/性能判定、报告 | `tests/glm5_2_graph_debug/run_graph_mode.sh`、`tests/glm5_2_combination/` |

图模式公共环境、CANN 9.1 安装和历史问题分别见
[GRAPH_MODE_COMMON.md](../glm5_2_graph_debug/GRAPH_MODE_COMMON.md)、
[CANN_9_1_INSTALLATION.md](../glm5_2_graph_debug/CANN_9_1_INSTALLATION.md) 和
[LOWER_LAYER_ISSUE_HANDOFF.md](LOWER_LAYER_ISSUE_HANDOFF.md)。

### 2.2 为什么不是编译整个模型

TorchTitan 的 `apply_compile` 遍历 `model.layers.named_children()`，逐 block 调用：

```python
transformer_block.compile(backend=backend, fullgraph=True)
```

这是 regional compilation：编译区域小于整个模型，但每个区域内部要求 full graph。选择
这个边界有三个实际收益：

- Transformer 层高度重复，同类 block 可以命中相同 Dynamo/Inductor cache；
- 避免把训练循环、数据加载、优化器、checkpoint I/O 和分布式控制流一并纳入超大图；
- 出错时能把问题定位到 dense/MoE forward/backward，而不是一个难以分析的单体图。

因此面试时应准确表达为：**“对 TransformerBlock 做区域整图编译”**，不要只说“整模型
整图编译”。

### 2.3 并行化与 compile 的先后顺序

`parallelize_glm5` 的顺序是：

```text
CP forward wrapping
    -> TP/EP parallelize
    -> activation checkpointing
    -> compile each TransformerBlock
    -> DDP/FSDP wrapping
```

这意味着：

- 编译器能看到 TP/EP/CP 注入 block 内部的局部算子和通信；
- activation checkpointing 已经在被编译区域内；
- FSDP/DDP 包装发生在 compile 之后，参数分片/聚合主要位于编译区域外层；
- PP 对模型 stage 的切分发生在更上层，各 stage 只编译自己实际持有的 block。

理解这个顺序对定位问题很重要。例如 TP 的 `aten.complex` DTensor strategy 缺失发生在
捕获/传播阶段；PP metadata P2P 则属于 pipeline runtime，不是某个 Triton kernel 的问题。

## 3. 从 Python 到 NPU kernel 的完整链路

一次首次训练迭代大致经历：

```text
Python TransformerBlock.forward
  -> TorchDynamo bytecode tracing + guards
  -> FX forward graph
  -> AOTAutograd functionalization/decomposition
  -> FX forward/backward graphs
  -> Inductor lowering and IR
  -> scheduler fusion + memory planning
  -> Triton-Ascend codegen / vendor extern call / custom op / fallback
  -> autotune and binary compile
  -> cache write
  -> launch generated wrapper
```

### 3.1 Dynamo 做什么

Dynamo 从 Python frame 捕获 Tensor 运算，生成 FX graph，同时记录对 Python 值、dtype、
device、rank、stride、shape 等条件的 guards。下次调用若 guards 成立就复用编译结果；guards
失效可能触发新的 specialization/recompile。

当前使用 `fullgraph=True`。如果存在无法捕获的 Python 行为，正常结果是首次调用抛出
unsupported/graph-break 类错误，而不是静默回 eager 后继续训练。因此成功运行只能证明
已实际走到的输入路径可整图捕获，不能证明所有未来输入路径都没有数据相关 Python 分支。

### 3.2 AOTAutograd 做什么

AOTAutograd 把需要梯度的 forward 转换为适合编译的 functional graph，并生成 backward
graph。activation checkpointing 会影响保存/重算边界。当前还设置：

```python
torch._dynamo.config.skip_fwd_side_effects_in_bwd_under_checkpoint = True
```

其目的不是提高性能，而是处理 checkpoint backward 重算时 Python side effect 与 compiled
autograd 的语义差异。它是一个明确的行为策略，不能和普通 kernel fusion 混为一谈。

### 3.3 Inductor 做什么

Inductor 对 FX graph 做 decomposition、layout/stride 推导、fusion、buffer reuse 和 codegen。
在当前 NPU backend 中，最终节点大致分为：

| 类别 | 当前示例 | 执行方式 | 是否 graph break |
|---|---|---|---|
| 可融合 pointwise/reduction | add、mul、silu、softmax 片段、layer norm 片段 | Triton-Ascend kernel | 否 |
| 高性能库算子 | `mm`、`bmm` | `extern_kernels.mm/bmm`，转入 NPU vendor 实现 | 否 |
| NPU fused op | `npu_rotary_mul` 及 backward | 部分模式被 Triton 融合，部分保留为 wrapper 内 torch_npu/op-plugin 调用 | 否 |
| 自定义兼容 op | `torchtitanturbo_graph.safe_grouped_mm` | opaque custom op，runtime 内执行安全逻辑 | 否 |
| 显式 fallback | `aten.sum`、`_c10d_functional.all_reduce` | 外部 lowering/ACLNN/collective 路径 | 否 |
| 无 lowering 且不能保留 | 未支持 Python/算子/schema | 捕获或 lowering 失败 | 是失败，不会通过 |

这里最容易混淆的是 fallback。fallback 会减少融合范围并可能有性能/同步代价，但只要它以
外部调用节点留在 compiled wrapper 内，就仍是一张可执行的完整 FX 图。

本次 `output_code.py` 还说明同一 op family 不一定只有一种 lowering：部分
`npu_rotary_mul` 被吸收到名称含 `npu_rotary` 的 Triton fused kernel，另一些 forward/
backward 调用仍以 `torch.ops.npu.*` 和 metadata assertion 留在 wrapper 中。两种形态都没有
离开 compiled region。

### 3.4 Triton-Ascend 做什么

Inductor 为可融合区域生成 Triton kernel，由 Triton-Ascend 编译成 NPU 可执行代码。首次
运行还会搜索 block/tile 等配置并 benchmark 候选。这解释了冷启动很慢，而后续 step 很快。

当前 deterministic 精度模式只允许被明确标记为 vetted 的 pointwise autotune。Turbo 的
兼容逻辑仅对 `HeuristicType.POINTWISE` 向 PyTorch benchmarker 传入
`is_vetted_benchmarking=True`，reduction 仍不能借此绕过确定性约束。这个边界很重要：修复
的是调用方遗漏的安全声明，不是关闭 deterministic。

## 4. 实际冷缓存编译证据

### 4.1 实验配置

证据报告：

```text
graph_debug_runs/submission-readiness-20260828/
  launcher-train-inductor-20260828-155306-2460699/
    reports/report.md
    logs/runtime.log
```

关键配置：CANN 9.1.0、独立 graph Conda、单 NPU、10 steps、固定 sequence length 128、
deterministic、模型组件 Inductor、全新仓库外 cache、每 rank 一个 compiler worker。launcher
总时长 564 秒，10/10 steps 通过。

### 4.2 四个主编译单元

生成的 `torch_compile_debug/.../torchinductor/` 下有四个主要单元：

| 编译单元 | 语义 | 生成时刻 | 约含 Triton kernel | 主要外部调用 |
|---|---|---:|---:|---|
| `model__0_forward_1.0` | dense block forward | 15:55:59 | 19 | 10 `mm`、4 `bmm` |
| `model__1_forward_4.1` | MoE block forward | 15:57:26 | 31 | 11 `mm`、4 `bmm`、3 safe grouped-mm |
| `model__1_backward_6.2` | MoE block backward | 15:59:57 | 31 | grouped-mm backward、gather/rotary backward |
| `model__0_backward_7.3` | dense block backward | 16:01:39 | 21 | matmul/rotary backward 等 |

表中 kernel 数是对该次 `output_code.py` 的静态统计，适合说明编译结构，不是性能 kernel
总调用次数。一个训练 step 会因层数和 gradient accumulation 多次调用这些已编译函数。

四个可读 FX graph 合计出现较多的算子包括：109 个 tensor mul、89 个 permute、46 个
view、44 个 add、38 个 expand、37 个 unsqueeze、28 个 sum、24 个 pow、23 个 mm、
21 个 matmul backward、5 个 topk、5 个 sigmoid、4 个 complex，以及 safe grouped-mm
forward/backward、NPU rotary forward/backward 和 deterministic scatter。大量 view/permute
通常是 layout 表达，不等价于同样数量的独立 runtime kernel；Inductor 会消除或融合其中
一部分。

### 4.3 冷启动时间去了哪里

从训练开始到第一个真实 step 完成，主要阶段为：

| 时间点 | 事件 | 与前一主事件的间隔 |
|---:|---|---:|
| 15:53:48 左右 | 首次 graph trace/compile 开始 | - |
| 15:55:59 | dense forward debug artifact 写出 | 约 131 秒 |
| 15:57:26 | MoE forward 写出 | 约 87 秒 |
| 15:59:57 | MoE backward 写出 | 约 151 秒 |
| 16:01:39 | dense backward 写出 | 约 102 秒 |
| 16:01:57 | step 1 输出 | 约 18 秒 |

首次有效 step 之前约消耗 8 分钟。step 2 到 step 10 的间隔约 2.4–2.5 秒，没有再次出现
新的 compile debug 单元。这正是“冷编译慢、稳态快”的直接证据。

autotune 日志还显示单个 pointwise kernel 可有几十到数百个候选，prune 后再 benchmark；
单个 benchmark 常耗时 0.2–1.7 秒，个别 backward 候选超过 2.6 秒。多 rank × 多 topology
会分别产生与本地 shape、DTensor layout、collective 有关的图，不能假设所有拓扑共用单卡
cache。因此完整 15 拓扑、2 repeat、5000-step 任务天然以天计。

### 4.4 为什么 `[0/0]` 和 `[0/1]` 不是图断裂

日志中的 `[0/0]` 对应 dense block specialization，`[0/1]` 对应 MoE block
specialization。GLM 模型不同层类型通过同一 TransformerBlock 入口走到不同静态结构，
Dynamo 为两种结构各编译一次。这属于有界 specialization。

判定“运行中重编译风暴”需要看到 guards 失败、recompile reason、同一路径 specialization
持续增加，或 step 2 以后出现新的 compile 单元。本次证据均未出现，所以不能把两个变体
误报为 graph break。

## 5. 图断裂：定义、识别和本次结论

### 5.1 什么是图断裂

图断裂是 Dynamo 无法把连续 Python 执行捕获成同一个 FX graph，于是在某个边界结束当前
图，执行一段 Python/eager，再尝试捕获下一段。常见原因包括：

- Tensor 数据控制 Python 分支或循环，并通过 `.item()` 物化为 Python scalar；
- unsupported Python builtin、第三方 C extension 或副作用；
- 无法代理的对象进入 compiled frame；
- 算子 schema、fake tensor 或 decomposition 缺失；
- 显式 `torch.compiler.disable` 或不支持的 mutation/alias。

### 5.2 fullgraph 模式下有什么不同

默认 `torch.compile` 可能容忍 graph break，导致“能跑但碎图”。当前 block 使用
`fullgraph=True`，所以遇到真实断点时该 block 编译失败。这把 silent performance
degradation 变成显式 correctness/compatibility failure，是本项目能声明 block 整图的前提。

但 fullgraph 通过仍有边界：它只覆盖实际运行过的 Python path、shape、dtype 和 topology。
换 sequence length、切换 TP/EP/PP、遇到以前未出现的空 expert，都可能产生新 guard 或走到
新路径，必须通过拓扑矩阵补充证据。

### 5.3 如何系统检查

建议同时收集：

```text
TORCH_LOGS=graph_breaks,recompiles,guards
TORCH_TRACE=<external-cache-path>
TORCH_COMPILE_DEBUG=1
TORCH_COMPILE_DEBUG_DIR=<external-cache-path>
```

项目入口可直接使用 `--compiler-diagnostics`。检查顺序是：

1. launcher 报告确认 diagnostics 路径、版本和 cache 身份；
2. runtime log 搜索 graph break、unsupported、recompile、guard failure；
3. 看 `TORCH_TRACE` 是否持续产生新 frame/specialization；
4. 看 `torch_compile_debug` 中 FX readable、IR pre/post fusion 和 `output_code.py`；
5. 对照 step 时间，确认编译只发生在预期 warmup，而不是周期性插入训练。

### 5.4 本次可以和不可以声称什么

可以声称：固定单卡输入、dense/MoE 实际路径在 block `fullgraph=True` 下完成前后向编译，
10 steps 内没有观测到图断裂或重新编译。

不可以声称：所有 15 拓扑、所有 sequence length、所有 MoE 路由分布都永远不重编译；也
不可以仅凭单卡 trace 声称 distributed collective 已经全部原生 lowering。

## 6. 动态 shape：固定外形与数据相关内形

### 6.1 哪些 shape 是静态的

正式精度实验固定 checkpoint、token plan、batch、sequence length、model config 和 dtype。
单卡 debug 中常见张量维度如 `[1024, 256]` 是静态的。生成 kernel metadata 也记录了多处
`axis_static_values`。因此该实验没有验证任意 sequence length 的单图复用。

### 6.2 MoE 为什么仍然有动态 shape

即使输入 token 总数固定，top-k router 分给每个 expert 的 token 数也随数据变化。EP 模式
下 all-to-all 后每个 rank 收到的 routed rows `R` 也是动态的。典型依赖链是：

```text
router top-k
  -> per-expert token counts
  -> all-to-all split sizes
  -> routed_input.shape[0] = R
  -> per-expert grouped-mm offsets
```

这是大模型 MoE 编译里最典型的数据相关 shape，不能简单通过把 batch/seq 固定就消失。

### 6.3 当前如何保留动态性

第一层策略是允许 scalar output capture：

```python
torch._dynamo.config.capture_scalar_outputs = True
```

第二层策略是在 token dispatcher 的 `_permute` 中，把 all-to-all 输出的动态总行数 `total`
作为 `repeat_interleave(..., output_size=total)`，并复用由它产生的 unbacked symint，避免
无谓地再次读取 device scalar。

第三层策略是 `GroupedExperts` 在 EP dynamic rows 下把 DTensor weight 转为 local tensor，
因为当前 DTensor 很难表达这种动态局部行数。

第四层策略是把不适合 Dynamo 展开的空 expert 判断放进
`torchtitanturbo_graph.safe_grouped_mm` custom op。它注册了 fake 实现：输出 shape 为
`(A.shape[0], B_t.shape[-1])`，也注册了 autograd。编译器把它视作形状契约明确的 opaque
节点；runtime 内部再处理 `.item()`、空 group padding、全空输入和 backward。

这是一种常用工程模式：**把动态控制流收敛到有 fake/meta/autograd 契约的 custom op
边界，而不是让 Python 数据分支切碎主图。**

### 6.4 动态 shape 仍有哪些风险

- 一个以前未见过的 symbolic relation 可能让 guards 失效并产生新 specialization；
- `tolist()`/CPU split list 会产生 device-host sync，影响性能且对 capture backend 敏感；
- 空 expert 可能使 grouped-mm offset 重复，触发底层算子不支持；
- 全空 routed tensor 会形成 `numel=0` kernel grid；
- 不同 EP/TP degree 改变 local expert 数和 weight layout，通常需要不同 cache key；
- 动态 sequence/batch 尚未由本次固定输入实验覆盖。

所以面试中不应把“支持 dynamic shape”说成“完全无 guard、任意 shape 单图”。更准确的说法
是：对 MoE routed rows 做了局部符号化和 custom-op 封装，同时保持外层训练 shape 固定。

## 7. 算子兼容性的五层模型

遇到“不支持算子”时，建议按下面五层区分，不能只问“这个 op 支不支持”。

### 7.1 捕获层

问题表现：Dynamo graph break、fake tensor/meta function 缺失、custom object 无法代理。
修复位置通常在 PyTorch Dynamo/FakeTensor、operator fake implementation，或模型代码中的
Python 数据控制流。

本项目例子：safe grouped-mm 必须注册 fake implementation；NPUGraph tree 曾拒绝
`DeviceMesh` runtime input。

### 7.2 自动微分和分解层

问题表现：forward 能捕获，AOTAutograd 生成 backward 失败；custom op 没有 autograd；
functionalization 不能处理 mutation/alias。

本项目例子：safe grouped-mm 同时注册 backward custom op；NPU rotary 需要对应 backward。

### 7.3 DTensor/并行传播层

问题表现：单卡通过，TP/EP 才报 strategy 或 placement 错误。算子本身可执行，但 DTensor
不知道如何传播 sharding。

本项目例子：`aten.complex` 缺单维 pointwise DTensor strategy。Turbo 当前运行时调用
PyTorch `_register_single_dim_pointwise(torch.ops.aten.complex.default)`；真正根治应进入
PyTorch DTensor pointwise operator registry。

### 7.4 Inductor lowering/codegen 层

问题表现：FX graph 已生成，但 lowering、tile、scheduler 或 Triton codegen 失败。

本项目例子：`aten.sum` reduction tile 全部超过 UB；当前通过
`NPU_INDUCTOR_FALLBACK_LIST` 转外部实现。真正根治位置在 torch_npu Inductor reduction
tile 合法性过滤/codegen，而不是模型里把 sum 改写掉。

### 7.5 runtime/vendor kernel/communication 层

问题表现：代码已生成，launch 时 CANN/op-plugin/HCCL 报错、hang 或数值错误。

本项目例子：

- grouped-mm 空 group/offset contract 在 op-plugin 与 CANN grouped matmul 边界；
- 零 numel Triton kernel 不应进入 NPU launcher；
- compiled functional all-reduce 的 stream/wait/alias 语义仍需 PyTorch Inductor 与 HCCL 分界；
- PP metadata batched P2P 在 PipelineStage 与 ProcessGroupHCCL 交界。

这个分层方法能直接决定 patch 应提到模型仓、PyTorch、torch_npu、op-plugin、CANN 还是 HCCL。

## 8. 当前兼容项对整图的影响

| 兼容项 | 当前做法 | 对图的影响 | 真正根治方向 |
|---|---|---|---|
| CANN 9.0 污染 | `env -i` 后仅 source CANN 9.1 | 编译前环境，不改变 FX 图 | torch_npu/CANN 严格版本检查与清晰报错 |
| `aten.sum` | Inductor fallback | 节点留在图内，减少 fusion | torch_npu reduction tiling/UB 过滤 |
| compiled all-reduce | fallback | collective 外部调用，仍在图内 | Inductor collective lowering ↔ HCCL stream/wait 分界 |
| TP `aten.complex` | 注册 DTensor strategy | 让图能通过 sharding propagation | PyTorch DTensor registry upstream |
| PP metadata | `use_batch=False` | pipeline runtime policy，非 kernel fusion | PipelineStage object P2P/HCCL batched P2P |
| EP 空 expert | safe grouped-mm custom op | opaque 节点保持主图完整 | op-plugin/CANN 定义空 group contract |
| 零长度 Triton | launch 前 `*numel==0` skip | 生成图不变，runtime 不 launch 空 grid | torch_npu autotuner/grid policy |
| deterministic autotune | pointwise 标记 vetted | 允许确定性首次编译 | torch_npu 调用 PyTorch benchmarker 时传正确标记 |
| NPUGraph replay | 默认全局 skip | Dynamo/AOT 可运行，但无原生 replay | graph-tree input contract 和 unsafe-op 支持 |

完整模块、函数、建议 patch 和提单验收见
[LOWER_LAYER_ISSUE_HANDOFF.md](LOWER_LAYER_ISSUE_HANDOFF.md)。

### 8.1 源码阅读地图

以下位置按本文验证 revision 记录。后续 rebase 时以函数名和调用关系为准，不把行号当成
永久 ABI：

| 要回答的问题 | 仓库内路径与模块/函数 | 当前关键行 |
|---|---|---:|
| compile 边界和 fullgraph | TorchTitan `torchtitan/distributed/compile.py::apply_compile` | 39–72 |
| CP/TP/EP/AC/compile/FSDP 顺序 | TorchTitan `torchtitan/models/glm5/parallelize.py::parallelize_glm5` | 167–229 |
| EP count exchange 和 CPU split | TorchTitan `torchtitan/models/common/token_dispatcher.py::AllToAllTokenDispatcher` | 245–370、449–499 |
| unbacked symint 的复用 | 同文件 `AllToAllTokenDispatcher._permute` | 501–553 |
| dynamic row 下 DTensor local weight | TorchTitan `torchtitan/models/common/moe.py::GroupedExperts.forward` | 55–108 |
| grouped-mm 原始 seam | 同文件 `GroupedExperts._grouped_mm` | 110–120 |
| deterministic autotune | Turbo `torchtitanturbo/tools/graph_compat.py::_install_vetted_pointwise_autotune` | 42–91 |
| 零 numel launch guard | 同文件 `_install_zero_numel_triton_guard` | 94–134 |
| safe grouped-mm fake/autograd | 同文件 `_install_safe_empty_grouped_mm` | 137–233 |
| TP complex strategy | 同文件 `_register_complex_dtensor_strategy` | 236–250 |
| PP metadata P2P | 同文件 `_install_pipeline_metadata_p2p` | 253–283 |
| NPUGraph skip/capture policy | 同文件 `_install_npugraph_skip_policy`、`_install_npugraph_capture_mode` | 286–353 |

对于 torch_npu/PyTorch/CANN 安装目录内的真正 patch 候选，不在此表复制可能随 wheel 变化的
绝对行号；应使用 lower-layer handoff 中每个 G 编号的模块、类、函数、最小分界实验和
workaround-off acceptance。

## 9. 精度实验为什么必须跑 5000 steps

10-step smoke 主要验证“能启动、能编译、短期无错误”；它不能覆盖：

- 小数值偏差经优化器状态和参数更新长期累积；
- 不同路由分布晚些时候才触发空 expert 或动态 shape 边界；
- distributed reduction 顺序导致的长期漂移；
- cache 命中后稳定运行阶段的异步错误；
- 两个 repeat 的可重复性。

正式组合实验固定同一个 step-0 checkpoint 和 token plan，以 single NPU eager 为 reference，
每个 Inductor topology 跑 5000 steps、2 repeats，再逐 step 比对 loss、grad norm、finite 状态、
输入身份和 artifact 完整性。只有最终 `--compare --require-all` 通过，才能声明该拓扑的
eager/graph 精度验收通过。

正式入口：

```bash
tests/glm5_2_combination/run_graph_precision_5000.sh inductor all
```

当前执行状态见
[precision-5000.md](../glm5_2_combination/experiments/reports/precision-5000.md)。

## 10. 从编译证据到性能结论

### 10.1 为什么要把性能实验拆成两条线

本项目没有用一份 profiler trace 同时宣称吞吐加速。权威 A/B 和问题归因是两条不同的
实验线：

| 实验线 | profiler | 统计口径 | 回答的问题 |
|---|---|---|---|
| eager/Inductor A/B | 关闭 | 跳过 cold compile 后的 step time 中位数，两个 repeat | 图模式在相同输入、拓扑和 token budget 下是否真的更快 |
| all-preset attribution | 有界窗口 | 每个 preset 独立 capture，不能当无扰动吞吐 | 时间花在 host、kernel、通信、内存还是系统 I/O，为什么快或慢 |

这避免了两个常见错误：把 profiler 自身开销当成模型回退，以及用单个活跃窗口的速度代替
稳态训练吞吐。

### 10.2 60-run eager/Inductor 实测

2026-08-26 的组合性能矩阵固定 step-0 checkpoint 和 token plan，使用 30 steps、跳过前
10 steps、每种模式两个 repeat、每 step 8,192 tokens。15 个验收拓扑共完成
`15 × 2 modes × 2 repeats = 60` 个运行，每个运行均有 30 条指标。完整数据见
[组合性能报告](../glm5_2_combination/experiments/reports/performance/summary.md) 和
[机器可读数据](../glm5_2_combination/experiments/reports/performance/data.json)。

主要结果按结构分组如下：

| 拓扑族 | eager → Inductor 代表结果 | 解释边界 |
|---|---:|---|
| single | `3.0583s → 3.0048s`, `1.0178x` | 小幅变化，不能单凭此值证明融合收益 |
| DDP/FSDP | DDP8 `0.9838x`；FSDP8 `1.0183x` | 通信/同步占比不因 block compile 自动消失 |
| TP | TP8 `6.4241s → 4.2402s`, `1.5150x` | TP 路径呈一致大幅收益，是 host/model launch overhead 假设的强信号 |
| TP 组合 | FSDP2-TP4 `1.5332x`；FSDP4-TP2 `1.5023x`；TP2-CP4 `1.4948x` | 多个独立拓扑方向一致，比单点结果更可信 |
| TP+PP/EP | FSDP2-TP2-PP2 `1.5400x`；FSDP2-TP4-EP8 `1.4552x` | compile 收益仍受 pipeline/EP communication 限制 |
| PP | PP8 `1.0873x`；FSDP2-PP4 `1.0539x` | stage readiness 和 bubble 位于 block kernel 优化之外 |
| CP/EP | CP8 `1.0239x`；EP8 `1.0177x`；FSDP2-CP4 `1.0382x` | collective 与动态 dispatch 仍占主导 |

结构性结论是：所有含 TP 的拓扑获得约 **1.455x–1.540x**，而不含 TP 的多数路径在
`-1.62%` 到 `+3.82%`。这支持“区域编译显著降低 TP 路径的 model/host launch 成分”，但
仍是推断，不是 kernel 级因果证明；因果链要由本轮 all-preset 的 Host/NPU Timeline、
kernel count 和 communication readiness 进一步验证。

实验边界同样必须主动说明：当时物理 NPU0/1/3/4 报 Alarm，八卡 Inductor batch 还与外部
NPU0 作业重叠。因此 45% 以上、跨多个 TP 组合一致的信号值得优先复验，而低于 5% 的差异
不能直接作为生产验收。

### 10.3 一个典型通信瓶颈案例

此前双卡 DDP FP32 reduction profile 的稳态基线约为 741.67 ms/step、
5,522.66 tok/s/device。每 step 有 50 次 AllReduce、约 266–267 MB/rank。底层 HCCS 物理
传输只有约 13.8 ms，带宽约 19.35 GB/s，但 rank0/rank1 在框架归因上的暴露通信时间约为
234.61/31.27 ms。`communication_bottleneck` 将 Top-10 慢 AllReduce 判为 Host-bound，
同时看到 `aten::to`、`GroupedMmBackward0`、`aten::linalg_vector_norm` 等前序任务的 rank
启动偏斜。

这个案例说明：collective 很慢不一定是链路带宽差。要把下面三项分开：

1. payload：逻辑上参与 collective 的字节数；
2. physical transit：HCCS/HCCL 真正传输时间；
3. exposed wait：某 rank 先到 collective 后等待其他 rank 的时间。

所以优先优化方向不是盲目换 ring，而是先定位 rank readiness skew、host launch gap 和
collective 分组粒度。BF16 reduction 原型把 payload 精确减半、物理传输降约 41%–44%，但
当时受另一八卡作业竞争，未把吞吐当正式加速结论。这种“保留微观机制结论、拒绝污染后的
端到端数字”是性能报告可信度的重要部分。完整历史见
[NPU 性能探索总览](../glm5_2_performance/explorations/reports/summary.md)。

### 10.4 Ascend 官方三阶段如何落地

本轮完整性能矩阵严格按三阶段组织：

```text
Ascend PyTorch Profiler
  -> framework/CANN/NPU/HCCL/memory 原始数据
msprof-analyze
  -> advisor / cluster / cluster_time / free / communication_bottleneck
MindStudio Insight
  -> Host-PyTorch-CANN-NPU Timeline、stream、通信、内存可视化
```

`--preset all` 不是“把所有开关塞进一次采集”，而是 8 个独立 acquisition policy：

| preset | 主要问题 |
|---|---|
| overview | 哪个阶段最慢，是否值得下钻 |
| distributed | rank readiness、collective、HCCS interconnection |
| kernel | AICore 利用率、算术和 L2 行为 |
| operator | shape/dtype/op args 与外部算子边界 |
| memory | allocation、active/reserved、memory timeline |
| flamegraph | Python/C++ host 调用栈和 launch 热点 |
| runtime | Level 2 深度算子/内存/host 联合证据 |
| system | CPU、NUMA、disk/network、HCCS/PCIe 系统侧证据 |

性能 runner 的 `--topology all` 还包含 1/2/4/8 卡共 27 个实验拓扑，所以完整矩阵是
216 次独立 capture。精确命令、队列、隔离策略和输出层级见
[all-preset 命令账本](../glm5_2_performance/explorations/history/all-preset-commands.md)。矩阵尚未
生成全部 manifest 前，不能把 `QUEUED` 写成 `PASS`。

### 10.5 如何把这段实习讲成 90 秒

可以按“目标—困难—方法—结果—边界”组织：

> 我的工作是把 GLM-5.2 在 Ascend NPU 上的动态图训练，扩展为 eager 与 Inductor 的多并行
> 拓扑性能和精度实验。难点不是简单打开 `torch.compile`，而是同时跨 TorchTitan、Turbo
> 兼容层和测试仓处理 Dynamo fullgraph、MoE 动态路由、DTensor、grouped-mm、HCCL 以及
> CANN 版本隔离。我把编译边界定为 TransformerBlock 区域整图，使用固定 checkpoint 和
> token plan 做 15 拓扑、两模式、两重复的 60-run A/B；含 TP 的组合稳定获得约
> 1.46–1.54x，而其他路径大多低于 4%。我没有直接把相关性写成因果，而是继续用 Ascend
> Profiler、msprof-analyze 和 MindStudio，按 27 拓扑 × 8 preset 做通信、kernel、内存和
> host launch 归因。过程中我把问题分到 capture、autograd、DTensor、Inductor lowering、
> runtime 五层，所有 workaround 都 opt-in，并给出关闭 workaround 后的上游验收条件。
> 当前 smoke 和性能结构证据完整，5000-step deterministic graph precision 仍以
> `require-all` 为最终门槛，我不会把短跑通过提前说成正式精度通过。

### 10.6 五分钟深挖时的展开顺序

1. 先画 `Dynamo -> FX -> AOTAutograd -> Inductor -> Triton-Ascend/extern -> CANN/NPU`；
2. 解释 per-block `fullgraph=True`，区分 graph break、specialization、recompile 和 fallback；
3. 用 MoE routed rows 解释外层静态、内层动态 shape，以及 fake/autograd custom op；
4. 用 60-run 表说明 TP 家族一致收益和 A/B 统计口径；
5. 用双卡 AllReduce 案例说明 payload、physical transit、exposed wait 的区别；
6. 说明三阶段 profiler 归因与 216-run 独立 preset 设计；
7. 最后主动讲硬件 Alarm/竞争、NPUGraph replay skip、Inductor fallback 和 5000-step 未完成
   的边界。主动披露边界通常比给一个无法复现的漂亮数字更专业。

## 11. 面试问答

### Q1：`torch.compile` 的完整流水线是什么？

答：Dynamo 捕获 Python frame 并建立 guards，生成 FX graph；AOTAutograd 做 functionalization、
decomposition，并拆出可编译 forward/backward；Inductor lowering 到内部 IR，做 fusion、
layout、memory planning 和调度；NPU backend 把融合区域交给 Triton-Ascend，矩阵乘等保留为
vendor extern call，最后 autotune、编译、缓存并由 generated wrapper 调度执行。

### Q2：这个项目的“整图”到底有多整？

答：边界是单个 TransformerBlock，不是训练全流程。区域内使用 `fullgraph=True`，所以该
block 的实际 forward/backward path 必须完整捕获；模型层间、FSDP 外壳、优化器、数据加载
和训练循环不在同一个图中。更准确的术语是“TransformerBlock 区域整图编译”。

### Q3：为什么按 block 编译通常优于整模型一张图？

答：Transformer 层重复度高，同类 block 可共享 cache；较小区域降低 trace/compile 峰值和
诊断复杂度；训练控制流、I/O 与分布式外壳不进入巨图。同时它牺牲了跨 block fusion 的
机会，所以是编译成本、cache 复用、稳定性和潜在性能之间的工程权衡。

### Q4：图断裂与 recompile 有什么区别？

答：graph break 是一次 Python 执行被切成多个图和 eager 片段；recompile 是已有图的 guards
对新输入不成立，生成另一个 specialization。前者破坏连续捕获，后者可能仍是每个
specialization 的 full graph。recompile 数量有限可能合理，持续增长才是 recompile storm。

### Q5：怎么证明没有图断裂？

答：不能只看程序退出码。要确认入口确实是 `fullgraph=True`，开启 graph-break/recompile/
guard 日志与 `TORCH_TRACE`，检查 compile debug 的 FX/IR/code，并观察 warmup 后是否继续
编译。本次固定单卡 10-step 路径满足这些条件，结论范围仅限已运行输入和 topology。

### Q6：外部算子或 fallback 是否意味着 graph break？

答：不一定。若 Inductor wrapper 中保留一个 extern/custom/fallback call，它仍是图节点，
图没有断；只是该节点没有被 Triton fusion。只有捕获退出 compiled region、回 Python/eager
再进入另一张图，才是 graph break。

### Q7：`fullgraph=True` 为什么还能看到 `extern_kernels.mm`？

答：full graph 约束的是捕获边界，不要求所有节点都生成 Triton。GEMM 交给设备高性能库
通常比 Triton 通用实现更合理。Inductor 生成的完整 wrapper 可以同时调 Triton fused
kernel、vendor GEMM、collective 和 custom op。

### Q8：固定 sequence length 后为什么 MoE 仍然是动态 shape？

答：outer token 数固定，但 router 给各 expert 的 token 数随数据变化；EP all-to-all 后每个
rank 的 routed rows 也不同。grouped-mm 的每组 offset、空 expert 和输出行数因此是数据相关
的，这是内层动态 shape。

### Q9：什么是 unbacked SymInt？

答：它是从运行时 Tensor 数据推导、编译时没有固定具体值的符号整数。编译器用它表达动态
维度并传播 shape relation。本项目把 routed input 的动态总行数传给
`repeat_interleave(output_size=total)` 并复用对应符号，避免重复 host scalar 物化。

### Q10：为什么 custom op 有助于保持整图？

答：复杂的数据相关 Python 控制流可以封装在 custom op runtime 实现中，同时向编译器提供
fake/meta shape 契约和 autograd。主 FX 图只保留一个形状可推导的 opaque 节点，避免
`.item()` 和 Python if 在主 frame 中造成 graph break。代价是编译器不能跨 custom op 内部
做 fusion，需要单独保证数值、alias、autograd 和 capture 安全。

### Q11：动态 shape 的常见代价是什么？

答：更多 guards/specializations、调度保守、内存规划困难、autotune 组合变多，以及
device-host sync。若 shape 范围无限且每次都不满足旧 guard，会出现 recompile storm；可用
shape bucketing、padding、符号范围约束或 custom op 稳定边界。

### Q12：为什么编译第一次要八分钟？

答：首次需要捕获 dense/MoE forward，生成两类 backward，完成 lowering/fusion/codegen，
为几十个 Triton kernel 搜索和 benchmark tile，再编译二进制并写 cache。实测四个主单元
约依次耗时 131、87、151、102 秒，随后 step 2–10 约 2.4–2.5 秒。

### Q13：为什么多卡编译比单卡更慢？

答：每个 rank 都可能执行编译，图内还包含 DTensor layout/collective；不同 topology 的 local
shape 和 sharding 不同，cache 不能全部共享。若每 rank 同时启动多个 compiler worker，还会
放大 CPU、内存和 NPU device initialization 压力。本项目限制每 rank 一个 compile worker。

### Q14：cache key 改变的常见原因有哪些？

答：代码/bytecode、FX graph、输入 dtype/device/shape/stride、dynamic guards、backend 配置、
Inductor/Triton/torch_npu 版本、编译 flags 和部分环境身份。TP/EP degree 改变 local layout 时
通常会生成不同 key。所有 cache 必须写到仓库外 `.cache`，既避免污染 Git，也便于冷/热
cache 对照和复用。

### Q15：如何区分 cache miss、recompile 和 graph break？

答：cache miss 是进程没有找到可复用 artifact，可能编译一个与历史相同的图；recompile 是
当前进程已有 specialization 的 guard 失败；graph break 是同一次 frame 捕获被切开。三者
分别要看 cache log、guard/recompile log 和 graph-break/frame trace，不能只凭 step 变慢判断。

### Q16：算子支持应该从哪些层面回答？

答：至少分捕获/fake、AOTAutograd/decomposition、DTensor sharding、Inductor lowering/codegen、
设备 runtime/vendor kernel 五层。单卡 eager 可运行只证明最后设备 op 的某条路径可用，不
证明它有 fake、backward、DTensor strategy 或 Inductor lowering。

### Q17：`aten.complex` 的警告代表什么？

答：Inductor 当前不能为 complex operator 生成高效代码，可能作为边界/外部路径执行并影响
性能；它不自动代表 graph break。TP 下还需要 DTensor placement strategy，这是另一个层面。
当前 RoPE 把 real/imag gather 分开后重建 complex，并用 NPU fused rotary；真正上游修复还应
补齐 PyTorch DTensor 的 complex pointwise 注册。

### Q18：为什么 `aten.sum` fallback 仍需要根治？

答：fallback 让程序可运行，却阻断跨该节点 fusion，可能多一次 kernel、buffer 或 sync；也
掩盖 NPU Inductor reduction tile 对 UB 约束处理不足。生产交付应在 torch_npu reduction
tiling/codegen 过滤非法候选，并通过关闭 fallback 的回归验证。

### Q19：空 expert 为什么会击穿 grouped-mm？

答：某 expert 没 token 时 offsets 会重复，甚至 routed matrix 全部为零行。底层 grouped
matmul 可能假定每组非空或拒绝零 grid。当前 custom op 把空组临时 pad 到 1，完成计算后再
index-select 回真实 rows；全空直接返回正确形状的空 tensor。最终应由 op-plugin/CANN 明确定义
并支持 empty group contract。

### Q20：为什么要阻止零长度 Triton kernel launch？

答：逻辑上空 Tensor 的 elementwise 输出可以为空，不需要 device kernel；但某些 NPU
launcher 不接受 grid/numel 为 0。正确位置是在 runtime/autotuner launch policy 检测
`*numel==0` 并跳过，同时返回值和依赖关系必须已由 wrapper 正确表达。

### Q21：编译图里的 collective 最难在哪里？

答：不仅要 lowering 出通信调用，还要保证 stream、wait、alias、buffer lifetime 与计算图
依赖一致。eager 正确而 compiled all-reduce 数值异常时，要分别验证 collective op、Inductor
调度、AsyncCollectiveTensor unwrap 和 HCCL stream semantics，不能直接归因 CANN。

### Q22：NPUGraph 与 Inductor 的关系是什么？

答：Inductor 是 graph compiler/codegen；NPUGraph 更接近固定 runtime launch 序列的 capture
和 replay。可先经 Dynamo/AOT/Inductor 生成执行，再尝试 graph replay，但两者优化层次不同。
当前 profile 默认 skip NPUGraph replay，所以只能证明降级路径可运行，不能证明 native replay。

### Q23：为什么 NPUGraph 对动态 shape 和 host sync 更敏感？

答：capture/replay 假定 runtime 输入、地址、launch 序列和很多控制条件稳定。MoE 的 split
size、`.item()`、CPU list、空 group 和 DeviceMesh object 都可能破坏这种稳定契约。Inductor
能表达的 symbolic shape 不代表 graph replay backend 一定能接受相同动态性。

### Q24：如何设计 eager/compiled 精度实验？

答：固定同一个 step-0 checkpoint、token plan、seed、dtype 和 deterministic policy；eager
reference 只生成一次，candidate 按 topology 和 repeat 捕获完整 5000-step 指标；逐 step
检查 loss、grad norm、finite、输入身份和 artifact 数量，最后 `require-all`。不要用不同随机
输入比较，也不要只比最后一个 loss。

### Q25：为什么不能用 10-step smoke 宣布精度通过？

答：smoke 主要覆盖首次编译和短路径；数值漂移、优化器状态累积、动态路由稀有边界和异步
通信错误可能数百或数千 step 后才出现。10-step 是 bring-up 证据，5000-step 组合矩阵才是
正式精度验收。

### Q26：面试中如何回答“遇到 graph compile 失败怎么办”？

答：先固定版本、输入和最小 topology，隔离 cache 做冷启动复现；判断失败在 capture、AOT、
DTensor、lowering、codegen 还是 runtime；打开 graph-break/recompile/guard 与 compile debug；
从 FX readable 到 IR、output code、runtime stack 顺向定位；用最小 fallback/custom op 验证
责任边界；最后把 workaround 与真正 upstream patch、关闭 workaround 的验收命令一起记录。

### Q27：如何判断一个 workaround 是否适合长期保留？

答：它应 opt-in、作用域最小、失败时 fail closed、带版本/signature 检查、有单测和实际拓扑
回归，并明确不会改变 eager 默认行为。仍需记录真正根治模块和“关闭 workaround 后通过”的
验收标准，避免兼容层永久掩盖底层缺陷。

### Q28：这次编译分析最重要的工程经验是什么？

答：把“能跑”拆成环境兼容、可捕获、可求导、可分片、可 lowering、可 launch、数值对齐
和稳态性能八个维度；把“整图”限定到明确边界；把 graph break、fallback、extern call、
specialization 和 recompile 分开度量。这样报告才能真正指导 patch 和跨团队提单。

### Q29：为什么 TP 拓扑的 Inductor 收益远大于 DDP/FSDP？

答：从 60-run 数据看，TP 及 TP 组合一致获得 1.455x–1.540x，而 DDP/FSDP 多数接近 1x。
合理假设是 TP 把 block 拆成更多小 local op、layout conversion 和 collective 边界，eager 的
host launch/dispatcher 成分更高，区域编译能融合并减少这一部分；DDP/FSDP 的通信仍位于
block 外层或外部 collective，compile 不能直接消除。最终要用 all-preset Timeline 对比
kernel count、Host gap 和 rank readiness，不能只凭 speedup 反推根因。

### Q30：为什么 profiler preset 要分开采集？

答：Level、shape、stack、memory、L2、op args、system I/O 和全 rank 通信同时开启会显著增加
开销、数据量和相互干扰，还可能超过 profiler 支持组合。独立 preset 让每次回答一个明确
问题，保持采集窗口有界，也使失败可以按 topology/preset 断点恢复。

### Q31：怎么判断通信是带宽瓶颈还是 rank 不对齐？

答：同时看 payload、HCCS physical transit/bandwidth、collective exposed duration 和各 rank
前序 task。若物理传输短、带宽合理，但一个 rank 的 collective 暴露时间远高于其他 rank，
且 advisor 标为 Host-bound，优先调查 launch skew、计算负载和 event wait，而不是直接优化
链路算法。

### Q32：怎样避免性能数字被 profiler 或其他作业污染？

答：吞吐 authority 使用 profiler-off、跳过 cold compile、至少两个 repeat，并保存 pre/post
`npu-smi`；active profile 只做归因。发现同卡外部作业或硬件 Alarm 时保留原始结果，但把
绝对值和小幅差异降级为 diagnostic，等待空闲健康节点复验。本轮 216-run 队列也严格等待
已有资源 owner 正常退出，没有并发抢卡。

### Q33：如果让你基于当前证据提出优化顺序，会怎么排？

答：先完成数据归因，再按“收益上限 × 实现风险 × 可验证性”排序。当前候选是：减少 FSDP/
DDP collective 次数和 rank readiness skew；降低 host launch gap；修复 reduction/collective
lowering 以关闭 fallback；对 DSA top-k/mask/scatter 和空 expert 路径做融合；最后再评估
NPUGraph native replay。每个优化必须有 opt-in patch、同 contract A/B、精度门槛和 rollback，
不能把多个改动混在一个数字里。

### Q34：为什么工具在 shell 中可用，进入统一 launcher 后仍可能“找不到”？

答：生产级 launcher 往往用 `env -i` 重建最小环境，防止 Conda、CANN、代理和缓存变量从
交互 shell 泄漏。这样能提高可复现性，但也意味着新增工具变量必须进入显式白名单。本轮
首个 all-preset run 就完成了训练、capture 和 CANN 解析，却在 advisor 阶段找不到
`msprof-analyze`：不是 CLI 未安装，而是
`TORCHTITAN_MSPROF_ANALYZE{,_WORKERS}` 被环境边界丢弃。修复方式是只透传这两个变量，
再用 launcher 执行 `/usr/bin/env` 验证实际子进程环境，并依靠 capture manifest 只重试
后处理。面试中这个案例能说明：性能工程不仅是 kernel，也包括环境契约、fail-fast、幂等
恢复和证据完整性。

## 12. 推荐排障清单

1. 用 launcher `env` 报告确认只有 CANN 9.1、独立 Conda/ATB，且 cache 在仓库外。
2. 固定单卡、固定 checkpoint/token、10 steps，先验证 deterministic cold cache。
3. 开 `--compiler-diagnostics`，保留 runtime log、TORCH_TRACE、FX、IR、output code。
4. 检查 `fullgraph=True` 的实际 compile 边界，不凭命令行猜测。
5. 搜 graph break、unsupported、guard failure、recompile，并按 step 时间定位发生阶段。
6. 把失败算子归入捕获、autograd、DTensor、lowering、runtime 五层。
7. 对动态 shape 记录符号来源、guard、host sync、空 Tensor 和极值输入。
8. 对 fallback 分别跑开启/关闭，确认它是编译可用性还是数值正确性的必要条件。
9. 逐步扩展 single -> 单一分布式 topology -> 15 topology smoke。
10. 最后跑固定输入 5000-step、2 repeats 的 eager/Inductor `require-all` 精度验收。

## 13. 证据与结论边界

本文的编译结构、四个主单元、冷启动时序和单卡无重编译结论来自 2026-08-28 的实际
artifact；动态 shape、operator category 和修复位置来自当前三仓源码与历史失败记录。

正式 5000-step 精度矩阵尚未完成时，本文不会提前给出“全部 topology 精度通过”的结论。
实验状态必须以
[precision-5000.md](../glm5_2_combination/experiments/reports/precision-5000.md) 和最终生成的
`combination_reports/.../report.md` 为准。
