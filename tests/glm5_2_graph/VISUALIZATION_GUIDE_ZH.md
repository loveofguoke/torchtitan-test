# GLM-5.2 图模式诊断与可视化逐步阅读手册

本文回答的不是“模型结构能不能画成一张图”，而是：TorchTitan 训练开启
`torch.compile` 后，Python 程序如何被分成编译区域，编译器对每个区域做了什么，
最终生成的任务怎样在 NPU 上执行，以及为什么有时“能跑”却没有加速。

命令、参数和验收流程见 [graph README](README.md)；Python 包、Ascend 软件、GUI、
Git 仓库与安装命令统一见
[性能与图模式依赖清单](../glm5_2_common/PERFORMANCE_GRAPH_DEPENDENCIES_ZH.md)。

官方入口：

- [PyTorch torch.compile troubleshooting](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_troubleshooting.html)
- [meta-pytorch/tlparse](https://github.com/meta-pytorch/tlparse)
- [PyTorch FX 技术概览](https://github.com/pytorch/pytorch/blob/main/torch/fx/README.md)
- [Ascend PyTorch 编译模式](https://www.hiascend.com/document/detail/zh/Pytorch/2600/ptmoddevg/Frameworkfeatures/docs/zh/framework_feature_guide_pytorch/pytorch_compilation_mode.md)
- [Ascend 图编译 Debug 信息保存](https://www.hiascend.com/document/detail/zh/Pytorch/2600/modthirdparty/torchairuseguide/docs/zh/ascend_ir/features/basic/debug_save.md)
- [MindStudio Insight](https://gitcode.com/Ascend/msinsight)

## 1. 先区分三种“图”

### 1.1 模型结构图

描述 `nn.Module` 如何嵌套，例如 embedding、TransformerBlock、attention、MoE、head。
torchview/Netron/TensorBoard Graph 常用于此类展示。它适合讲模型架构，不等于实际
编译图，因为 Python 控制流、分布式 wrapper、autograd 和后端融合可能改变边界。

### 1.2 编译器图

`torch.compile` 运行时捕获的 FX graph，以及 AOTAutograd/Inductor 的后续 IR。一次
训练可能产生多个 graph：forward、backward、optimizer、不同 Python frame、不同
shape/guard 都可能形成独立 region。

这是 tlparse、FX readable、Inductor IR 和 generated code 关注的对象。

### 1.3 设备执行图/时间线

编译后的 kernel、Memcpy、event、HCCL task 在各 NPU stream 上的实际执行。MindStudio
Insight、Ascend Profiler 和 Perfetto 看的是这一层。NPUGraph/ACLGraph 还会 capture/
replay 一组设备任务，这个 graph 也不等于 Dynamo FX graph。

所以不存在一张“万能图”同时准确表达模型语义、编译器变换、运行时时间和多 rank
通信。专业工具链的做法是保留层级并建立链接。

## 2. torch.compile 从 Python 到 NPU 经历什么

```text
Python model/training frame
  |
TorchDynamo
  |-- bytecode observation
  |-- guards
  |-- graph breaks / multiple FX GraphModules
  v
FX Graph
  v
AOTAutograd / AOTDispatcher
  |-- functionalization
  |-- joint graph
  |-- forward/backward partition
  v
Inductor
  |-- lowering
  |-- scheduler IR
  |-- fusion / layout / memory planning
  v
Triton-Ascend / external NPU ops / CANN
  v
NPU kernels, streams, events and communication
```

### 2.1 Dynamo

Dynamo 观察 Python frame，在不改变可观察语义的前提下捕获 Tensor 运算。为保证下次
复用编译结果，它会生成 guards，例如：

- tensor shape、stride、dtype、device；
- Python 常量或对象 identity；
- module training 状态；
- 全局变量、函数版本或容器长度；
- 动态维度范围。

guard 不满足时，可能重编译；某项语义无法安全捕获时，可能 graph break 回到 eager。

### 2.2 FX Graph

FX 是一个有序 node 图。常见 `node.op`：

| op | 含义 | 阅读方式 |
|---|---|---|
| `placeholder` | 图输入 | 看 shape/dtype/stride 和输入顺序 |
| `get_attr` | 参数、buffer、常量 | 确认权重/常量来源 |
| `call_function` | 调用函数/operator overload | 看 target、args、users |
| `call_method` | Tensor/对象方法 | 看接收者和参数 |
| `call_module` | 调用子 `nn.Module` | Dynamo/AOT 后不一定保留很多模块边界 |
| `output` | 图输出 | 确认返回 tuple、梯度需要保存的 tensor |

每个 node 的 `users` 表示谁消费它；shape metadata 表示编译器当时看到的样例输入。
FX graph 是数学/数据依赖表示，不包含 NPU kernel 的真实运行耗时。

### 2.3 AOTAutograd

训练不只编译 forward。AOTAutograd 会把 autograd 需要的逻辑函数化，并生成/切分
forward 和 backward graph。某个 forward tensor 被保存到 backward，会影响内存和
fusion。看到多份 FX 文件不一定是重复编译，可能是 forward/backward/推理分区。

### 2.4 Inductor IR

Inductor 把 FX operator lower 成 scheduler/buffer/loop 等表示：

- `ir_pre_fusion.txt`：融合前候选节点与依赖；
- `ir_post_fusion.txt`：调度和融合决策之后；
- buffer/dependency：中间量的读写关系；
- scheduler node/fused node：最终将一起生成或调用的工作。

比较 pre/post fusion 能回答“哪些逐元素/归一化/索引操作被合并，哪些没有”。它仍不
代表实际硬件时间，必须和 generated code、Profiler 对照。

### 2.5 Triton-Ascend 和 generated code

`output_code.py` 是后端生成的 Python wrapper/编译调用，可能包含：

- Triton kernel 定义/launch；
- 外部 aten/aclnn/CANN op 调用；
- buffer 分配、复用和释放；
- shape/stride assert；
- stream/synchronization；
- async compilation/cache 逻辑。

生成一个 `output_code.py` 不能证明所有 op 都被融合；需要看有多少 external calls、
多少 Triton launch，以及运行时 kernel 数是否下降。

## 3. 本项目支持的三种执行模式

### 3.1 eager

不加 `--compile.enable`。PyTorch 每次按动态图执行。它是当前 NPU 图模式精度实验的
reference，而不是“低级错误路径”。

### 3.2 inductor

`torch.compile(backend="inductor")`：Dynamo 捕获 + AOTAutograd + Inductor，并在 NPU
环境通过 Triton-Ascend/torch_npu lowering 生成或调用 NPU 实现。重点收益通常是减少
Python/launch overhead、融合算子和优化内存调度。

### 3.3 npugraphs

NPUGraph backend 侧重捕获并 replay 设备执行序列，降低重复 launch 开销。它对输入
地址、shape、stream、unsafe op 和同步行为有更强约束。本项目当前只允许编译 model
component，不能把它和 Inductor 的 FX/代码生成语义完全等同。

### 3.4 “程序跑完”不等于图模式成功

以下都可能让 smoke 退出码为 0，但不满足图模式目标：

- backend error 被 suppress 后全量 eager fallback；
- graph break 太多，每个小图收益抵不过编译/launch；
- 每 step 因 guard 变化重编译；
- 只编译了不重要的小 component；
- NPUGraph 被环境开关整体 skip；
- 编译成功但生成更多/更慢 kernel；
- 精度或 checkpoint 行为改变。

必须同时看 compile diagnostics、正式 precision 和 profiler-off performance。

## 4. 为什么主入口是 tlparse，而不是一张 DAG 图片

PyTorch 官方建议大模型问题先用 `TORCH_TRACE + tlparse`，因为它能同时展示：

- Python frame/stack trie；
- 哪些 frame 被编译、跳过或失败；
- graph break 原因和源码位置；
- recompile 与 guard failure；
- compilation region/compile ID；
- FX graph 和后端产物链接；
- 编译耗时与错误。

单张 SVG 只能展示一份 FX `GraphModule`，不能告诉你同一 frame 为什么生成第 7 份图。
这也是我们不把 Netron/torchview 当 `torch.compile` 首选诊断器的原因。

本框架在 `--compiler-diagnostics` 时，为每个 rank 独立设置：

```text
TORCH_TRACE=<run>/graph_visualization/rank_<rank>/torch_trace
TORCH_COMPILE_DEBUG=1
TORCH_COMPILE_DEBUG_DIR=<run>/graph_visualization/rank_<rank>/inductor
```

独立目录很重要：八个 rank 同写默认 `torch_compile_debug` 会覆盖、混合或产生无法归属
当前实验的文件。

## 5. 十分钟阅读路线

1. 先看 combination/graph 顶层报告的 reference/candidate、topology、repeat；
2. 确认 candidate 明确记录 `inductor` 或 `npugraphs`，不是 eager；
3. 看 precision PASS/FAIL，先保证数学和训练轨迹；
4. 看 graph 表的 Status 和 Raw trace 是否存在；
5. 打开 tlparse 总览，找 error、graph break、recompile；
6. 选择一个主要 compile region，打开 FX readable；
7. 对比 pre/post-fusion IR；
8. 打开 generated code，看最终 kernel/external op；
9. 打开 performance report/MindStudio，验证 kernel 数、空洞和耗时；
10. profiler-off 比 eager/graph 多次重复，最后判断是否加速。

## 6. 顶层组合报告怎么读

图实验复用 combination/precision/performance 报告，不另造一套结论。

### 6.1 Overview/实验身份

确认：

- reference graph mode；
- candidate graph mode；
- compile components（model，必要时 loss）；
- topology/world size；
- dtype、steps、batch、sequence、seed；
- objectives 是 precision、performance 还是二者；
- compiler diagnostics 是否开启。

capture 和 compare 的这些字段不一致时，不能把产物拼在一起。

### 6.2 Precision suite

正式判断仍看 loss/grad norm 的多 step 标准、重复运行和固定 token plan。图编译改变执行
次序、融合和 reduction 路径，BF16 不要求 bitwise，但必须满足既定迁移/自洽标准。

smoke 只证明接口/拓扑能跑，不能替代 precision。

### 6.3 Performance table

典型列是 topology、endpoint、repeat、median step time、throughput、speedup、details。

- speedup 必须来自相同 token/dtype/topology 口径；
- profiler-active endpoint 只做归因；
- graph 第一次 compile step 不进入稳态速度结论；
- reference single 与 candidate distributed 的精度对比可以成立，但性能 speedup 要明确
  是扩展收益还是编译收益，不能混写。

### 6.4 `torch.compile visualization and TensorBoard` 表

| 列 | 表示什么 | 怎么读 |
|---|---|---|
| Status | diagnostics 是否请求、是否发现产物/解析成功 | 不是图模式性能 PASS |
| Interactive | tlparse `index.html` | 第一入口，找多 frame、break、recompile、error |
| FX | 捕获图文件 | 看数学数据流和 shape，不看设备耗时 |
| IR | pre/post-fusion | 看 lowering/fusion/scheduler 变化 |
| Code | `output_code.py` | 看真正生成/调用哪些 kernel/op |
| Raw trace | `TORCH_TRACE` 原始日志 | 重新解析、提交上游 issue；包含源码信息 |
| TensorBoard | event 文件和启动命令 | 看长时间训练曲线，不看编译 DAG |

一行没有 FX/IR/code 不一定代表失败：npugraphs 与 inductor 的后端产物并不相同；先看
mode、tlparse 和 runtime log。若 inductor 声称成功但所有诊断均空，则需要检查诊断环境
是否在 import/compile 前设置。

## 7. tlparse 一步步怎么读

### 7.1 首页/stack trie

页面通常按 Python 调用栈组织编译 frame。先做：

1. 搜索红色/错误 frame；
2. 搜索 graph break；
3. 找同一 frame 是否出现多个 compile ID；
4. 看主要 forward/backward region，而不是先点最小 helper；
5. 记录源码位置和 rank。

不同 PyTorch/tlparse 版本的 UI 名称会变，但核心对象是 frame、compile region 和事件。

### 7.2 graph break

graph break 表示 Dynamo 在一个 frame 内停止当前图，执行一段 eager 或开始另一个图。
常见原因：

- data-dependent Python control flow；
- `.item()` 后把 tensor 值用于 Python；
- 不支持/无法追踪的 Python/C 扩展；
- side effect、generator、异常语义；
- 自定义 op 没有正确注册 fake/meta/autograd；
- 显式 `torch._dynamo.disable`/graph boundary。

不是所有 break 都必须消灭。冷路径或一次性逻辑的 break 成本可能很低；训练主循环中
每层/每 token 高频 break 才值得优先处理。先看 frequency 和 runtime gap。

### 7.3 recompile

recompile 表示已有编译结果的 guard 不再满足。阅读顺序：

1. 找首次编译与后续 recompile 的同一 frame；
2. 展开 guard failure；
3. 判断变化来自 shape、stride、Python 常量、module state 还是对象 identity；
4. 对照输入 contract 和具体 step/rank；
5. 选择固定输入、消除 Python 变化或合理 dynamic shape。

不要看到 recompile 就盲目 `dynamic=True`。动态代码可能降低融合/特化质量；固定训练
shape 时应优先修不必要的 Python/shape 波动。

### 7.4 guards

guard 是 cache 安全条件，不是错误。大量 guards 只有在检查成本高、频繁失败或导致
cache explosion 时才是问题。把失败 guard 和输入变化对应起来，比单纯数 guard 个数
更有意义。

### 7.5 compile ID/frame ID

`[frame/compile]` 一类标识用于区分哪个 Python frame 的第几次编译。不能把 “compile
1” 当训练 step 1。多 rank 还要结合 rank 目录，不同 rank 的编号不一定能直接一一对应。

### 7.6 error frame

后端错误定位：

1. 记录最内层原始异常，不只看 `BackendCompilerFailed` 包装；
2. 看 FX 输入 shape/dtype；
3. 确定 Dynamo、AOT、Inductor、Triton-Ascend、torch_npu 还是 CANN 层；
4. 用该 region 的 FX/code 做最小复现；
5. 检查是否 eager fallback；
6. 将 runtime log、trace、三仓 commit 和环境版本一起保存。

## 8. FX readable 一步步怎么读

打开 `fx_graph_readable.py` 或 transformed graph：

1. 从 function 参数/placeholder 看输入；
2. 标记 shape、dtype、stride、requires_grad；
3. 顺序读 `aten.*` target；
4. 注意 `view/reshape/permute/contiguous`，它们决定 layout 和潜在 copy；
5. 找 reduction、topk、index/gather/scatter 和通信边界；
6. 看 output tuple，尤其 backward 要保存的中间量；
7. transformed graph 与原图比较是否发生 functionalization/decomposition。

GLM/DSA 重点观察：

- indexer score/top-k 是否留在图内；
- top-k 后 gather 是否真缩小 K 维计算；
- RoPE/normalization 是否被拆成很多小 op；
- MoE dispatch/grouped GEMM 边界；
- DTensor local/redistribute 是否形成 break 或 external call；
- Python 标量同步是否出现在主路径。

FX 数学图相同不能证明 kernel 实现相同；精度看正式实验，性能看 IR/code/Timeline。

## 9. Inductor IR 一步步怎么读

### 9.1 pre-fusion

先找：

- 有多少 scheduler node；
- 哪些 node 生产/消费同一 buffer；
- 哪些逐元素链有融合机会；
- reduction/extern kernel 为什么形成边界；
- layout/alias/dependency 是否阻止融合。

### 9.2 post-fusion

与 pre-fusion 对照：

- node 数是否减少；
- 哪些 op 被组合为 fused scheduler node；
- 是否仍有大量单 op 小 kernel；
- fusion 是否扩大临时量生命周期或导致重复计算；
- 是否保留 external op（matmul/grouped matmul/collective）。

“融合越多越好”不成立。过度融合可能增加寄存器/UB 压力、降低并行度、阻止高性能
库算子。最后看真实 kernel duration 和 memory。

## 10. generated code 一步步怎么读

建议按下面搜索：

```text
triton_
extern_kernels
aten.
aclnn
empty_strided
reinterpret_tensor
assert_size_stride
synchronize / event / stream
```

回答：

- 生成了多少自定义 Triton-Ascend kernel；
- 哪些仍调用外部算子；
- 中间 buffer 是否复用；
- 是否有额外 contiguous/copy/cast；
- kernel launch 的 grid/shape 如何；
- 是否存在 sync；
- forward/backward 各产生什么。

不要手改 `output_code.py` 当生产修复；它是生成证据。优化应回到模型、lowering、custom
op 或 Turbo 的 NPU 实现，并保留 reference 路径。

## 11. 图编译与 MindStudio Timeline 联合定位

编译器回答“为什么生成这些任务”，Profiler 回答“这些任务如何运行”。典型链路：

### 11.1 图很碎、NPU 有空洞

1. tlparse 找高频 graph break；
2. FX 看每个 region 很小；
3. generated code 看 kernel launch 数；
4. Timeline 验证 Host launch gap；
5. 修 break/custom op 后 profiler-off A/B。

### 11.2 无 graph break，但没有加速

1. tlparse 确认只编译一次；
2. pre/post IR 看融合是否发生；
3. output code 看是否仍是相同 external op；
4. Timeline 比 eager/graph kernel count、duration、空洞；
5. 分离 compile startup 与 steady state。

### 11.3 重编译风暴

1. tlparse 看失败 guard；
2. 对照每 step shape/stride/Python state；
3. TensorBoard/runtime log 找开始变慢的 step；
4. Timeline 看编译期间 NPU idle；
5. 固定 contract 或选择合理 dynamic shape。

### 11.4 编译后内存增大

1. FX/AOT 看保存到 backward 的 tensor；
2. IR 看 fusion 后 buffer 生命周期；
3. output code 看临时 allocation/reuse；
4. memory timeline 找峰值事件；
5. 对照 eager 与 graph 的 active/reserved。

### 11.5 分布式只有某个 rank 出问题

1. 每 rank tlparse 比 graph break/recompile/error；
2. 确认 rank-specific control flow/shape；
3. Timeline 对齐 collective sequence；
4. 判断是编译差异导致 readiness skew，还是通信本身；
5. 保存出问题 rank 的 trace/FX/code，不只看 rank 0。

## 12. Ascend 官方图可视化能力的边界

MindStudio Insight 能导入 Ascend Profiler 数据，查看 Host/PyTorch/CANN/NPU 层级；新版
还支持导入 ACLGraph 构图过程 JSON，展示 Record/Wait 等图运行关系。这非常适合
NPUGraph/ACLGraph runtime。

但它不是 Dynamo/AOT/Inductor 的统一 compiler explorer：

- 不替代 tlparse 的 guards/recompile/graph break；
- 不替代 FX readable 的数学图；
- 不替代 Inductor pre/post-fusion IR；
- 不替代 generated Triton-Ascend code。

Ascend NPU Inductor 在部分版本提供 `INDUCTOR_ASCEND_DUMP_FX_GRAPH` 等专有 dump 开关；
这些开关必须按当前 TorchNPU 文档和代码确认，不能跨版本硬编码。本框架默认先使用
PyTorch 官方 `TORCH_TRACE`/`TORCH_COMPILE_DEBUG`，再索引后端实际产生的文件。

## 13. 其他图工具为什么只作为补充

| 工具 | 来源 | 擅长 | 不解决什么 |
|---|---|---|---|
| FX `FxGraphDrawer` | [PyTorch 源码](https://github.com/pytorch/pytorch/blob/main/torch/fx/passes/graph_drawer.py) | 把一份 FX GraphModule 画成 Graphviz SVG | 多 frame、guards、recompile、运行时 |
| Netron | [lutzroeder/netron](https://github.com/lutzroeder/netron) | ONNX/TorchScript 等模型文件 | 动态 PT2 编译过程 |
| torchview | [mert-kurttutan/torchview](https://github.com/mert-kurttutan/torchview) | module/tensor 前向结构 | AOT backward、fusion、kernel |
| torchviz | [szagoruyko/pytorchviz](https://github.com/szagoruyko/pytorchviz) | autograd DAG | Dynamo/Inductor 多图 |
| TensorBoard Graph | [TensorBoard Graphs](https://www.tensorflow.org/tensorboard/graphs) | 教学和静态图浏览 | PT2 compiler 多层 IR |
| Perfetto | [ui.perfetto.dev](https://ui.perfetto.dev/) | 通用 runtime trace 时间线 | compiler guard/fusion 原因 |
| MindStudio Insight | [Ascend/msinsight](https://gitcode.com/Ascend/msinsight) | Ascend runtime、通信、内存、ACLGraph | 完整 Dynamo/Inductor 编译历史 |

主流训练框架也不是靠一张静态图解决编译和性能。例如
[Megatron Bridge](https://docs.nvidia.com/nemo/megatron-bridge/nightly/training/profiling.html)
把 Nsight Systems 与 PyTorch Profiler 分开，限制采集 step/rank；编译器问题仍回到框架
日志/IR。我们的 NPU 对应关系是 MindStudio/Ascend Profiler + tlparse/FX/IR/code。

如果未来需要“对某个已确定的 FX region 画 SVG”，可以在受控 backend/debug hook 中
调用 `FxGraphDrawer`，依赖 `pydot + Graphviz`。它是可选增强，不应取代当前多 rank
compile trace。

## 14. 常见问题决策树

### 14.1 编译失败

```text
runtime.log 最内层异常
  -> Dynamo unsupported/graph break? 看 tlparse/TORCH_LOGS
  -> AOT fake/meta/autograd? 看 FX 输入与 custom op 注册
  -> Inductor lowering? 看 region FX 和 backend stack
  -> Triton-Ascend compile? 保存 output code、编译命令、C++/Triton error
  -> CANN/runtime? 保存 kernel/task、CANN 日志和最小 shape
```

### 14.2 能跑但没有快

```text
是否真的 compiled、无全量 fallback?
  -> 是否只编译一次?
  -> graph 是否足够大?
  -> IR 是否有有效 fusion/内存优化?
  -> generated kernel 数是否减少?
  -> Timeline Host 空洞/设备耗时是否改善?
  -> profiler-off 多次 A/B 是否超过噪声?
```

### 14.3 图模式精度失败

1. 确认相同 fixture/token/checkpoint/dtype；
2. 找第一个 loss/grad norm 超限 step；
3. 用较小固定输入做 eager/graph component 对比；
4. 检查 backend fallback、fusion、reduction、custom op/autograd；
5. 不因性能收益降低正式精度标准。

### 14.4 dynamic shape

只有证据显示 shape guard 导致有害重编译时才调整 dynamic policy。训练通常 shape 固定，
不必要的动态化可能牺牲特化和 fusion；MoE token count、top-k routing、empty group 等确实
可能带来动态 shape，应为具体维度定义 contract，而不是整模型一键动态。

## 15. 产物目录和复现

```text
<run>/graph_visualization/
  rank_0/
    torch_trace/
      dedicated_log_torch_trace.log
    inductor/
      torchinductor_*/
        fx_graph_readable.py
        fx_graph_transformed.py
        ir_pre_fusion.txt
        ir_post_fusion.txt
        output_code.py
  rank_1/...
  tlparse/
    <trace-id>/index.html
    <trace-id>/tlparse.log
  tlparse_manifest.json
```

- structured trace：可重新运行 tlparse，含源码/路径，外发前审阅；
- tlparse HTML：最方便的人读入口；
- FX：编译器捕获的数学/数据图；
- IR：lowering/fusion/scheduler；
- code：后端生成/调用；
- runtime log：错误、环境和完整命令；
- performance profile：真实 NPU 执行。

同步给本地阅读或 Codex 分析时，不要只下载 combined report HTML：

```bash
python release_artifacts.py upload <experiment> --content analysis
```

分析包保留 structured trace、`tlparse`、FX、IR、generated code、runtime log、性能
可视化和紧凑指标；`--content full` 另外保留完整原始 CANN profile、checkpoint、trainer
state 和 run 内原始编译产物，适合底层重新解析或恢复实验。位于仓库外
`GRAPH_CACHE_ROOT` 的可再生成编译缓存不进入 Release。

手工重跑解析：

```bash
tlparse <run>/graph_visualization/rank_0/torch_trace/*.log \
  -o /tmp/glm5-tlparse-rank0
```

框架自动输出使用内容 hash/独立目录，避免不同 trace 互相覆盖。`--force` 的实验生命周期
由 combination/precision 公共逻辑管理；Ambient `torch_compile_debug` 不属于当前实验。

## 16. 完整验收标准

图模式交付至少包含：

1. smoke：目标 single/topology/all 能启动并完成；
2. native evidence：没有不被记录的全量 eager fallback/skip；
3. compiler evidence：每 rank diagnostics 可追溯，错误/重编译有解释；
4. precision：eager reference 与 graph candidate 满足正式 loss/grad norm 标准；
5. performance：profiler-off 多次 A/B，分离 compile startup；
6. attribution：MindStudio/Profiler 证明 kernel/launch/内存/通信变化；
7. checkpoint/stability：需要交付的组合模式下能保存恢复并长稳运行；
8. rollback：所有 NPU workaround/优化默认可关闭，保留 reference 路径。

## 17. 最后检查清单

- [ ] 我区分了模型结构图、FX 编译图和 NPU runtime 图。
- [ ] 我确认 candidate 真的使用目标 backend。
- [ ] 我没有把 smoke 成功当成精度/性能通过。
- [ ] 我先看 tlparse 总览，再看单个 FX/IR/code。
- [ ] 我知道 graph break、guard 和 recompile 的区别。
- [ ] 我对照了 pre/post-fusion，而不是只数 FX op。
- [ ] 我在 Timeline 验证了 kernel、空洞、通信和重叠。
- [ ] 我使用 profiler-off 多次重复判断加速。
- [ ] 我检查了每 rank，而不是只看 rank 0。
- [ ] 我保存了环境版本、三仓 commit、完整命令、日志和原始 trace。
