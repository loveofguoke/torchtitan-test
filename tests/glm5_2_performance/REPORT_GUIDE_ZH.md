# GLM-5.2 性能报告与可视化逐步阅读手册

本文不是指标词典，而是一份从零开始的性能分析教程。目标是让第一次接触 Ascend
Profiler 的同学能够回答三件事：

1. 时间到底消耗在 Host、框架、CANN、NPU 计算还是通信；
2. HTML 中每一章、每张表和每个指标应该怎样读；
3. 发现异常后应打开哪个原始产物继续定位，而不是直接猜优化方案。

实验命令和参数在 [README](README.md)，所有环境、软件、Python 包、仓库及安装方式在
[统一依赖清单](../glm5_2_common/PERFORMANCE_GRAPH_DEPENDENCIES_ZH.md)。

官方参考：

- [Ascend PyTorch Profiler 用户指南](https://gitcode.com/Ascend/pytorch/blob/master/docs/zh/developer_notes/ascend_pytorch_profiler_user_guide.md)
- [MindStudio Insight System Tuning](https://gitcode.com/Ascend/msinsight/blob/master/docs/zh/user_guide/system_tuning.md)
- [msprof-analyze](https://gitcode.com/Ascend/msprof-analyze)
- [MindStudio 官方火焰图](https://gitcode.com/Ascend/msinsight/tree/master/scripts/flame_graph)

## 1. 先建立正确的执行模型

### 1.1 一步训练不是“一条 NPU 指令”

一条 Python 训练语句通常经过下面的链路：

```text
Python Trainer / model.forward
  -> PyTorch dispatcher / autograd / distributed wrapper
  -> torch_npu op-plugin
  -> CANN: AscendCL / Graph Engine / Runtime / HCCL
  -> NPU stream 上的 task
  -> AI Core kernel、AI CPU task、Memcpy 或 HCCL communication
```

这些层级不能混为一谈：

- Python 看到的是 Module、Tensor 和 optimizer step；
- PyTorch 层看到的是 `aten::matmul`、`record_function`、autograd node；
- CANN 层负责把框架调用转成设备任务、管理 stream/event、执行算子和通信；
- NPU 层才是真正运行 kernel、搬运内存和通信的地方。

同一个“矩阵乘”可能在报告中出现多个名字：Python 模块名、PyTorch op、aclnn/CANN
op、最终 NPU kernel。它们是同一调用链的不同层，不是重复计算的充分证据。

### 1.2 Host 是什么

本文中的 Host 通常指运行训练 Python 进程的 CPU 侧，包括：

- Python 解释器和 TorchTitan Trainer；
- DataLoader、token/shape 处理和日志；
- PyTorch dispatcher、autograd 调度；
- kernel launch、stream/event 操作；
- 通信调用的发起和等待；
- 图模式下 Dynamo/Inductor 编译与 cache 查询。

Host 时间高不等于 CPU 计算本身很复杂。常见原因还有：

- Python 小算子/循环过多，NPU 一直等下一次 launch；
- 某处 `.item()`、打印 tensor 或同步 API 强制 Host 等 NPU；
- 各 rank 到 collective 的时刻不同，先到者等待；
- profiler 解析、checkpoint 或日志落盘侵入了训练边界；
- 动态 shape 导致反复编译。

### 1.3 NPU、AI Core 和 AI CPU 是什么

Ascend NPU 上承担大规模张量计算的主要单元是 AI Core。概念上可把它理解为：

- Cube/矩阵计算管线：矩阵乘、卷积等高吞吐计算；
- Vector 管线：逐元素、归一化、激活、reduction 等；
- Scalar/控制单元：地址、循环和标量控制；
- 片上缓存/缓冲区：L0、Unified Buffer 等；
- 芯片共享缓存和 HBM：L2 与高带宽外部显存。

AI Core 指标不是简单的“GPU utilization”。某个 kernel 可能：

- 矩阵管线很满，是 compute-bound；
- 向量管线繁忙但矩阵管线低，是 vector-heavy；
- 算术单元等待 HBM/L2/UB，是 memory-bound；
- 数据 layout、bank/resource conflict 使流水线空转；
- kernel 很短，真正瓶颈是 Host 发射间隙。

AI CPU 是设备侧通用处理单元，适合控制或不适合 AI Core 的工作。大量未知算子落到
AI CPU 往往意味着算子支持/下沉不理想，但少量 AI CPU 事件不能直接判错，必须看其
累计耗时和是否在关键路径。

### 1.4 stream、task、event 和异步执行

- stream 是设备任务的有序队列；同一 stream 内按序执行，不同 stream 可以并发；
- task 是提交到设备的具体 kernel、Memcpy、event 或通信任务；
- event 用于跨 stream 建立依赖；
- Host launch 通常是异步的：Python 发出调用后不必等 kernel 完成。

因此 Host API 区间和 NPU kernel 区间不一定同长，也不一定上下严格对齐。Timeline
中的连接关系、stream 顺序和同步事件比“肉眼看两条横条是否平齐”更可靠。

### 1.5 HCCL 和 collective

HCCL 是 Ascend 分布式通信库。常见 collective：

| collective | 直观含义 | 常见并行用途 |
|---|---|---|
| AllReduce | 每 rank 输入聚合后，每 rank 都得到结果 | DDP 梯度、TP partial 汇总 |
| ReduceScatter | 聚合并把结果切片给各 rank | FSDP/分布式 optimizer 梯度 |
| AllGather | 收集各 rank 分片，让每 rank 得到完整/更大张量 | FSDP 参数、TP 激活 |
| AllToAll/AllToAllV | 每 rank 向每个 rank 发送不同分片/长度 | MoE EP token dispatch/combine |
| Send/Recv | 点对点传输 | PP stage 之间 activation/gradient |

通信总时间不等于性能损失。若通信和计算重叠，只有未被覆盖的部分进入关键路径。报告
因此优先看 exposed/not-overlapped communication，而不是只看 collective 总和。

### 1.6 critical path 是什么

一个 step 完成时间由最慢的依赖链决定。例如 rank 0 很早进入 AllReduce，但 rank 7
还在计算，rank 0 的 Wait 很长。此时：

- 通信库显示等待时间很高；
- 根因却可能是 rank 7 的上游计算/Host 发射偏慢；
- 直接“优化 HCCL 带宽”可能没有作用。

性能分析的核心不是找报告里最大的数，而是重建关键路径。

## 2. 十分钟阅读路线

第一次打开 HTML，不要从最后一张大表开始。按以下顺序：

1. 看标题，确认 device、topology、preset、repeat 和实验身份；
2. 看顶部阶段徽标，它只说明产物是否存在；
3. 看 step time/throughput，并确认当前是不是 profiler-off 基线；
4. 看 `Profiler phase overhead`，判断 active 数据被放大多少；
5. 多卡看 `Distributed critical path` 和 `Collective communication`；
6. 用曲线确认异常是持续、偶发、warmup 还是某个 active window；
7. 看 `Time composition` 判断 Host/compute/communication/free 的大方向；
8. 看 Top operator/shape/L2，把大方向缩到具体 op；
9. 打开 MindStudio Timeline 验证先后、依赖、空洞和重叠；
10. 按问题类型打开火焰图、Memory、TensorBoard、msprof-analyze 或编译诊断。

若只记一条原则：HTML 是导航和摘要，MindStudio 原始时间线是运行时归因依据；
profiler-off 重复实验才是性能数值依据。

## 3. preset 到底是什么

preset 是一次有明确成本和问题范围的采集策略，不是模型精度/性能等级。

| preset | 首要问题 | 主要额外数据 | 代价/注意 |
|---|---|---|---|
| `overview` | 慢发生在哪个大层级 | Level0 基础 CPU/NPU 时间 | 首次采集，开销最低 |
| `comparison` | 给 A/B 提供较轻归因证据 | Level0、简化控制 | 仍不能替代 profiler-off |
| `standard` | AI Core 管线是否健康 | Level1、PipeUtilization、rank 0 | 单 rank 不代表分布式全部 rank |
| `distributed` | 哪个 rank/collective 暴露 | 全 rank、通信、互联、离线解析 | 多卡数据大，优先 `repeat=1` 导入 Insight |
| `kernel` | 热点 kernel 是算力还是缓存问题 | shape、ArithmeticUtilization、L2 | counter 本身会增加开销 |
| `operator` | 哪个 op/参数值得融合或改 layout | shapes、op attr/args、FLOPs 字段 | 元数据大，注意敏感信息 |
| `memory` | 峰值由哪类 tensor/事件产生 | memory、stack/module、timeline export | 开销和产物都大 |
| `flamegraph` | Host/框架累计调用路径在哪 | CPU/NPU folded stack、Ascend DB | 火焰图无时间先后 |
| `runtime` | 需要跨模块/stack/内存深挖 | Level2、stack、module、memory、shape | 高开销，只用于短窗口 |
| `system` | CPU/NUMA/网络/磁盘是否拖慢 | Host system、I/O、interconnect、MSTX | 全 rank 最重，问题明确后再开 |
| `all` | 一次编排得到完整证据套件 | 依次执行上述非冗余策略 | 多次独立训练，不是一锅全开 |

为什么 `all` 必须分目录、分 capture：AI Core counter 一次通常只能选一个指标族；
stack、memory、system 和全 rank 同开会极大扰动训练，甚至让数据无法解析。独立 capture
才能知道每份数据的语义和开销。

## 4. 顶部状态和指标卡逐项解释

### 4.1 阶段徽标

`Capture / Parse / Advisor / Cluster / Compare / Insight` 表示：

- Capture：采集目录中发现文件；
- Parse：存在 text/DB 等解析产物；
- Advisor：advisor 命令成功执行；
- Cluster：集群分析命令成功执行；
- Compare：A/B 离线比较命令成功执行；
- Insight：目录满足 MindStudio 导入的最低形态。

绿色只代表阶段完成，不代表性能 PASS，不代表 advisor 结论是“无问题”，更不代表
精度通过。真正结论要看正文和 formal precision report。

### 4.2 Median step time

定义：所选稳定 step 的耗时中位数。

为什么用中位数：编译、第一次 allocator 扩容、日志、OS 抢占会制造少量长尾；中位数
比 mean 更能描述典型 step。但还要同时看 p90/p99 和曲线，防止中位数掩盖周期性卡顿。

读法：

1. profiler-off 的两次或三次重复先比较；
2. 排除 warmup/compile step；
3. 同模型、token、dtype、topology 比较；
4. active profile 内 step 只用于定位，不用于宣布加速比。

### 4.3 Mean throughput

本框架的基础口径是每设备 token/s：

```text
per_device_tokens_per_second = tokens_per_step / step_time / world_size
job_tokens_per_second = per_device_tokens_per_second * world_size
```

必须确认 `tokens_per_step` 的定义一致。DP 增大常同时改变全局 batch；TP/PP/CP/EP
不一定改变样本数量。比较拓扑时若 token contract 不同，吞吐数字不可直接作自洽结论。

### 4.4 TFLOPS 和 MFU

报告中的 TFLOPS 是模型 FLOPs 估算除以时间：

```text
estimated_TFLOPS = estimated_model_FLOPs_per_step / step_time / 1e12
MFU = estimated_TFLOPS / theoretical_peak_TFLOPS
```

它不是硬件 counter。FLOPs 公式是否计入 activation checkpoint 重算、MoE 激活专家、
attention 稀疏度，会改变数值。MFU 适合同一公式、同硬件、同精度下做趋势比较，不能
把不同项目的 MFU 百分比不加说明地横向比较。

TFLOPS 下降可能是通信/Host 等待，不一定是 matmul 变慢。要结合 AI Core 和 Timeline。

### 4.5 Peak active memory

常见三种内存：

- active/allocated：当前 tensor 实际占用；
- reserved：allocator 向设备申请并保留的池；
- device/process used：还包括 runtime、通信 buffer、编译 cache 或其他进程。

报告卡片的 active peak 不等于 `npu-smi` 看到的整卡占用。若 OOM，但 active 不高，
继续看 reserved、碎片、临时 workspace、HCCL buffer 和其他进程。

### 4.6 Active collection overhead

定义：Profiler active window 的典型 step 相对普通稳定 step 的膨胀：

```text
overhead = active_window_step_time / baseline_step_time - 1
```

例如普通 step 100ms，active step 180ms，则 overhead 约 80%。这不代表模型变慢 80%，
而是 stack/shape/counter/memory 采集带来的观测成本。处理方法：缩短 active steps、减少
ranks、换浅 preset；不能拿 active 数值当优化后的吞吐。

### 4.7 Sync parse stall

同步解析在训练进程退出/边界阻塞的时间。它通常不属于稳态 step，但会影响“整个任务
墙钟时间”。大规模多 rank profile 推荐 offline parse，避免训练结束阶段长时间占用设备
或被外层 timeout 误判。

## 5. HTML 每个章节怎么读

下面按报告实际顺序说明。

### 5.1 中文阅读路线

这是内嵌速查，只负责提醒入口。真正逐项定义以本文为准。它不会复制所有原始 Timeline，
因为压成静态表后会丢失依赖、层级、并发和缩放信息。

### 5.2 Profiler phase overhead

回答：“skip、warmup、active、post-active 四段分别多慢，采集是否严重扰动训练？”

常见字段：

- phase：调度阶段；
- step count：落入该段的 step 数；
- median/mean/p90：该段 step 时间统计；
- ratio/overhead：相对稳定基线的倍率或增量。

阅读步骤：

1. 确认 active 窗口真的落在训练稳定段，而不是首个 compile/warmup；
2. active 显著放大是正常现象，但放大数倍时只做结构归因；
3. post-active 仍持续变慢，检查同步解析、导出 memory/stacks 和 GC；
4. skip 本身很慢，说明模型 warmup/编译还没结束，应把 skip 后移。

### 5.3 Distributed critical path

回答：“哪个 rank 最慢，慢在计算、通信、等待还是空闲？”

表/图通常按 rank 展示：

- Computing：设备计算 task 时间；
- Communication：通信 task 总时间；
- Overlapped：计算与通信同时存在的时间；
- Communication not overlapped/exposed：真正暴露在关键路径上的通信；
- Free/Idle：设备没有可执行 task 的空洞；
- min/median/max：跨 rank 分布。

读法：

1. 先找 step/rank 的最大总时间；
2. 看最大 rank 是否也有最高 compute；
3. 若某 rank compute 正常但 wait/free 高，查看它在等谁；
4. 若所有 rank collective 物理时间接近，但 wait 差异大，优先查 readiness skew；
5. 若所有 rank transit 都高，再查 payload、链路、拓扑映射和 collective 算法。

误区：把每个 rank 的计算+通信直接相加得到 step time。重叠区会重复计数，必须使用
官方 overlap/exposed 口径或 Timeline 的临界链。

### 5.4 Collective communication

回答：“是数据真的传得慢，还是先到的 rank 在等后到者？”

关键概念：

- transit：数据实际传输/collective 执行；
- wait：等待其他 rank/依赖就绪；
- synchronization：同步相关时间；
- payload：逻辑通信数据量；
- bandwidth：传输量/传输时间的派生量；
- calls：调用次数。

读法：

```text
transit 高、wait 低
  -> 先查 payload、链路带宽、collective 次数/算法、跨 NUMA/设备映射

transit 正常、wait 高
  -> 先查上游 compute/Host launch/rank skew，不要先改 HCCL

calls 很多、单次 payload 很小
  -> 可能是通信粒度太碎、bucket/placement/层边界问题

calls 少、单次 payload 很大且暴露
  -> 可能需要 overlap、分块、prefetch 或并行配置调整
```

AllToAllV 还要看每个 peer payload 是否均衡。MoE top-k 总 token 数相同，也可能因 router
偏斜造成某些 rank 收到更多 token，拖慢整个 collective 和 grouped GEMM。

### 5.5 Step time 曲线

回答：“慢是从什么时候开始，是否周期性，是否只发生在 active window？”

- 横轴是训练 step，不是 wall-clock 秒；
- warmup/active 应有标记；
- 单点尖峰看日志/checkpoint/GC/编译；
- 周期性尖峰看 log_freq、checkpoint interval、profile schedule；
- 后半程逐渐上升看内存增长、动态 shape cache、数据/序列变化、温控降频。

曲线必须从所有 step 展示，不能只看 101 以后或只看均值。报告中的 warmup 标记用于
解释，不会把早期数据从原始记录删除。

### 5.6 Throughput 曲线

它通常和 step time 反向变化，但不是完全镜像，因为 token 数可能变化。固定 token plan
下仍出现吞吐突降，回到同 step 的 runtime log 和 Timeline；动态数据下先确认有效 token
或 padding 是否变化。

### 5.7 TFLOPS/MFU 曲线

它们把 step time 映射成一个计算效率口径。若 step time 变化但 FLOPs 估算没反映动态
专家/token，曲线只能做趋势提示。看到 MFU 异常高于合理范围时，先审 FLOPs 公式和
峰值口径（BF16/FP16/稀疏/矩阵单元），不要宣布突破硬件峰值。

### 5.8 Active memory 曲线

回答：“峰值在哪个 step，是否随时间单调增长，active window 是否额外占内存？”

- 稳定锯齿通常对应 activation 生命周期；
- 每 step 基线持续上升可能是 tensor 被引用、cache 增长或日志保存对象；
- 只在第一次升高后稳定可能是 allocator/cache warmup；
- profile-memory/stack 本身会增加内存，和 profiler-off 对照。

想知道峰值由什么组成，要打开 `Ascend memory timeline`，不能只看一条总量曲线。

### 5.9 NPU runtime diagnostics / torch.compile health

报告会从日志统计或索引：CPU fallback、AI CPU fallback、graph break、recompile、backend
failure、parse time 等。

这些计数是线索，不是完整事件数据库：同一错误可能打印多行，某些事件可能不打印。
图模式应打开 tlparse；fallback 应在 Timeline 看累计时间；backend failure 还要确认是否
发生 eager fallback。训练“跑完”不等于编译路径成功。

### 5.10 Time composition probe

回答：“一个 profile 窗口的粗粒度时间构成是什么？”

它适合选择下一层工具：

- Host 高：火焰图、Python stack、launch gap、DataLoader/日志；
- Compute 高：Top kernel、shape、AI Core counter、融合可能性；
- Exposed communication 高：collective、overlap、rank readiness；
- Free 高：依赖、PP bubble、Host 未及时发射、同步；
- Memory copy 高：layout/device copy、Host-Device copy 专项。

不同 CSV 的层级可能重叠，不要把所有分类百分比机械相加到 100%。报告会优先使用
可识别的官方口径，但最后仍由 Timeline 验证。

### 5.11 Top duration entries

回答：“哪些 op/kernel/调用累计占时最多？”

常见列：

- name/type：事件名与层级；
- calls：调用次数；
- total duration：所有调用累计；
- average：平均一次；
- self duration：扣除子调用后的本身时间；
- ratio：占所选表总时间比例。

读法：

- `total` 高、`self` 低：主要时间在子调用，继续展开；
- `self` 高：当前事件本身是热点；
- calls 极多、average 很小：考虑融合/批处理/减少 launch；
- calls 少、average 极大：考虑 kernel 算法、shape、并行切分；
- 名字相同但 shape 不同：必须分 shape，不要把它当一个 kernel。

Top-N 没出现不代表事件不存在；它可能被截断、在另一层或只影响关键路径但累计时间小。

### 5.12 Top kernel shape signatures

回答：“热点到底运行在什么 shape/dtype/layout，是否大量碎片化？”

关注：

- input/output shapes；
- dtype；
- 调用次数；
- 同名 kernel 的 shape 数；
- 总/均值耗时。

例：一个 matmul 总时长高，但每次 shape 都很小，问题更像切分过细和 launch overhead；
一个固定大 matmul 算术利用率低，更像 layout、tiling 或内存访问问题。写 Triton kernel
之前，先把这里的真实 shape 记录为 benchmark contract。

### 5.13 AI Core 指标和 L2 表

一次 profile 通常只选一个 `AiCMetrics` 家族：

| 指标族 | 用来回答 | 不能单独证明 |
|---|---|---|
| PipeUtilization | 各计算/搬运管线是否忙、是否有明显空转 | 根因一定在 kernel |
| ArithmeticUtilization | 算术单元利用程度 | 整个 step 都 compute-bound |
| Memory / MemoryAccess | HBM/缓存访问压力和访问特征 | 单看带宽就能决定优化 |
| MemoryL0 / MemoryUB | 片上 L0/UB 使用与搬运压力 | 具体哪行源码导致 |
| ResourceConflictRatio | 资源冲突是否显著 | 改一个参数必然加速 |
| L2Cache | L2 hit/miss/访问行为 | hit 越高一定越快 |

正确流程：热点 kernel -> 固定 shape -> 选择一个 counter 家族 -> 结合耗时和生成代码 ->
A/B 验证。不要在一次 capture 中期待所有硬件计数器。

### 5.14 msprof-analyze output index

每一行索引一次外部 recipe：

- name：advisor/cluster/compare/free analysis 等；
- return code：工具是否成功执行，不是优化结论；
- files/size：实际产生多少文件；
- outputs and command：可复现命令和 HTML/CSV/JSON 链接。

阅读：先确认 return code=0，再打开 recipe 自己的 summary/advice；任何“建议”都要回到
Timeline 和 profiler-off A/B 验证。advisor 是候选根因生成器，不是自动判决器。

### 5.15 Official Ascend deliverables

每一行对应一个 `*_ascend_pt` profile 根：

- Raw：底层原始采集；
- Text：CSV/JSON 等解析输出；
- Database：统一 DB；
- Insight：目录是否满足官方导入形态；
- Timeline：可直接打开的 trace JSON。

不要只同步 HTML 报告。用于本地阅读、Codex 分析和文档整理时，使用
`release_artifacts.py upload <experiment> --content analysis`，它会同步所有 rank 的
manifest、runtime log、analysis、metrics、火焰图、memory timeline、TensorBoard、
parsed CSV/JSON/DB、advisor/cluster/compare 和图编译可视化，同时排除 checkpoint、原始
CANN 采集树和编译缓存。需要 MindStudio 导入完整原始 profile 或重新解析时，使用
`--content full`。多 rank/cluster 时尤其不能只拿 rank 0。

### 5.16 Interactive Timeline and hierarchy

这是最重要的下钻入口。MindStudio 中按下面顺序操作：

1. 导入完整 `*_ascend_pt` 或集群目录；
2. 先缩放到一个稳定 `ProfilerStep`；
3. 从 Host process/thread 找 PyTorch API；
4. 沿关联关系查看 CANN enqueue/runtime；
5. 在 NPU stream 找对应 task/kernel；
6. 横向检查 stream 空洞、Memcpy、HCCL 和 overlap；
7. 多 rank 对齐同一 collective sequence，比较谁先到、谁后到；
8. 点击热点查看名称、持续时间、stream/task、shape/附加字段。

MindStudio 常见层级：

- Host/process/thread：哪个 Python 进程/线程发起；
- PyTorch：op、autograd、ProfilerStep、用户 range；
- CANN：AscendCL、GE、Runtime、HCCL API；
- NPU：stream、task、kernel、communication、memory；
- AI Core Freq/System：硬件/系统采集可用时显示。

时间线判读模式：

- Host 横条密集但 NPU 有大空洞：launch-bound；
- NPU stream 连续计算、Host 已提前发射：device-bound；
- compute stream 与 HCCL stream 重叠好：通信总时间可能不暴露；
- 所有 stream 同时空，Host 在同步：查 `.item()`/event/log/allocator；
- PP stage 周期性无任务：bubble 或 microbatch/schedule 问题；
- 一个 rank 比其他 rank 晚进入 collective：查该 rank 上游路径。

Perfetto 可快速打开 `trace_view.json`，但它是通用时间线浏览器；Summary、Communication、
Memory 和 Ascend 关联语义以 MindStudio 为准。

### 5.17 CPU/NPU flame graphs

火焰图横轴是累计样本/时间宽度，不是 wall-clock 时间；纵轴是调用深度。

本报告提供两条路径：

1. MindStudio 官方 HTML：从 `ascend_pytorch_profiler_<rank>.db` 重建 Host PyTorch API
   调用层级，支持线程切换、搜索、悬浮、自身/总耗时、点击缩放；
2. folded-stack SVG：从 `export_stacks` 生成 CPU/NPU 聚合栈，便携且补充 NPU 侧视角。

阅读方法：

- 先选训练主线程；
- 搜索 `forward`、`backward`、`optimizer`、`collective` 或具体 op；
- 宽而浅：某个大阶段本身占比高；
- 宽而深：大量时间累积在这条调用路径；
- total 高但 self 低：子节点是热点；
- self 高：该节点自身工作/等待值得查。

火焰图看不出某个调用发生在 step 开头还是结尾，也看不出两个 stream 是否重叠；回到
Timeline 查看先后关系。

### 5.18 Ascend memory timeline

交互 HTML/JSON 用来回答：“峰值那一刻有哪些 tensor/类别存在，什么时候创建和释放？”

典型类别：参数、梯度、optimizer state、activation、input、temporary/unknown。具体分类
取决于 PyTorch/Profiler 版本和 stack/module 元数据。

阅读：

1. 找总量峰值的 step/时间；
2. 看哪种颜色/类别增长；
3. 查看峰值前最后一批 CREATE；
4. 看本应释放的 activation 是否跨 step 保留；
5. 对照 checkpoint/recompute 策略；
6. 再对照 allocator reserved 和 npu-smi。

`*_memory_timeline.html` 直接阅读；分类 JSON 可二次绘图；raw JSON.gz 保存 CREATE/
DESTROY 等原始事件。只看峰值数字无法判断泄漏、碎片还是正常 workspace。

### 5.19 TensorBoard training dashboard

报告会给出事件目录和启动命令。建议选择：

- loss：是否稳定、是否出现 NaN/异常尖峰；
- grad norm：数值稳定性和溢出线索；
- step time/throughput：长时间抖动；
- MFU/TFLOPS：同口径趋势；
- memory：随 step 是否增长。

TensorBoard 是秒/step 级长期视图，Profiler 是微秒级短窗口。先用 TensorBoard 找“哪段
时间异常”，再把 Profiler active window 放到该段；不能让 Profiler 覆盖几小时训练。

### 5.20 Profiler data inventory

按文件后缀统计数量和体积，用于：

- 判断 DB/JSON/CSV 是否真的生成；
- 估算同步/Release 大小；
- 发现异常的超大 stack/DB；
- 复核 data simplification 是否按预期。

它不评价文件内容正确性。

### 5.21 Profiling preflight

记录采集前后的存储、预计空间、profile 窗口和警告。性能目录可能很大，尤其是全 rank
Level2/stack/memory。空间不足可能在训练结束解析时才暴露，因此要在启动前处理。

### 5.22 Effective Ascend profiler controls

这是“本次究竟开了什么”的权威清单，包括 level、ranks、AiCMetrics、shapes、stack、
memory、modules、L2、host system、export type 等。分析任何结果前都先确认对应能力已
开启：没采 shape 的报告无法用空白 shape 列证明 shape 相同。

### 5.23 Experiment configuration / Additional training arguments

配置表回答两次 A/B 是否真正可比：模型、dtype、steps、batch、seq、topology、seed、
compile、extra args。报告名字相似不等于配置相同；manifest/config hash 和这里的实际
参数才是依据。

## 6. 一个完整的诊断例子：PP8 很慢

下面是方法示例，不代表某个具体报告的固定结论。

现象：PP8 step time 高，HCCL P2P 行也很宽。

错误推理：“P2P 很宽，所以网络太慢。”

正确步骤：

1. profiler-off 确认 PP8 重复运行都慢，排除采集开销；
2. `Profiler phase` 确认 active window 没把 step 放大到不可用；
3. `Distributed critical path` 看各 stage free/compute 分布；
4. 若首尾 stage free 很多、中间 stage compute 集中，怀疑 pipeline bubble/stage 不均衡；
5. `Collective/P2P` 分开看 transit 和 wait；
6. transit 正常、wait 高，说明接收方尚未准备好，不是链路带宽不足；
7. MindStudio 对齐 microbatch，观察哪个 stage 最晚发 send/recv；
8. 回到层分配、每 stage 计算量、microbatch 数和 schedule；
9. 调整一个变量，例如 microbatch 或 stage partition；
10. profiler-off 至少三次 A/B 比 median/p90，再用短 profile 解释收益来自 bubble 缩小。

只有 transit 明显异常，才进一步查 P2P payload、HCCL 链路、rank mapping、NUMA 和系统
interconnect。

## 7. 常见现象到下一步工具的决策表

| 现象 | 首选下一步 | 暂时不要做 |
|---|---|---|
| NPU 时间线大片空白 | Host flame graph + sync/event + main thread | 直接写 matmul kernel |
| kernel 连续且 AI Core 算术高 | 固定 shape benchmark、算法/并行切分 | 只调 DataLoader |
| kernel 连续但算术低、memory 高 | layout/融合/访存/Triton 原型 | 只看 MFU |
| collective wait 高、transit 正常 | rank readiness、上游 compute/Host | 归因于 HCCL 带宽 |
| collective transit 高 | payload、calls、互联、算法、rank mapping | 忽略跨 rank 对照 |
| AllToAllV rank payload 偏斜 | router/token distribution、capacity/load balance | 只优化通信 kernel |
| step 周期性尖峰 | checkpoint/log/profile/GC interval | 用中位数掩盖尖峰 |
| active memory 单调增长 | raw memory events、引用/cache、snapshot | 只调 allocator 参数 |
| 图模式跑完但无加速 | tlparse + FX/IR/code + Timeline kernel count | 认为“compile=True”即成功 |
| compile 重复 | tlparse guards/recompiles、shape contract | 盲开 dynamic=True |

## 8. 如何看 AiCMetrics，而不被一个百分比误导

一次选择一个热点和一个问题：

1. `overview` 找到热点 kernel；
2. `operator/kernel` 拿到 shape、dtype、calls；
3. 判断假设：算力、HBM、片上搬运、L2、冲突；
4. 选择对应 AiCMetrics；
5. 同 shape 重复采集；
6. 查看 counter 与 duration 是否共同变化；
7. 做代码 A/B，最终以 profiler-off step time 验收。

例：L2 hit 低不一定坏。流式读取本来就可能没有复用；为了提高 hit 而增加拷贝反而更慢。
只有算法预期存在复用、miss 引发显著带宽等待且 A/B 有收益时，才算证实。

## 9. 报告不能直接回答的问题

- 某个 kernel 数学上是否正确：看 precision/parity；
- 某个模块一定是根因：检查点只表示“到这里误差/耗时表现为何”；
- 生产模型真实 MFU：debug model 的 shape/通信粒度可能不代表生产规模；
- 单次 A/B 的微小差异：至少多次重复并报告分布；
- profiler-active 的真实吞吐：采集本身改变执行；
- 所有图编译原因：需要 tlparse/FX/IR/code；
- 所有内存占用：active allocator 不含所有 runtime/其他进程内存。

## 10. 推荐的完整分析流程

### 10.1 性能基线

1. smoke 确认 topology 能跑；
2. profiler-off 重复至少三次；
3. 记录 median、p90、吞吐、内存和 rank min/median/max；
4. 保存环境、三仓 commit、完整命令。

### 10.2 逐层下钻

1. `overview` 定 Host/compute/communication/memory 大方向；
2. 多卡用 `distributed`；
3. 算子问题用 `kernel`/`operator`；
4. Host 路径用 `flamegraph`；
5. 内存问题用 `memory`；
6. CPU/NUMA/互联问题用 `system`；
7. `msprof-analyze` 输出候选建议；
8. MindStudio 时间线做因果验证。

### 10.3 优化验收

1. 每次只改一个优化点，默认关闭；
2. 保留 reference 路径；
3. precision 先通过；
4. profiler-off A/B 至少三次；
5. active profile 只解释为什么快；
6. 比较 step median/p90、吞吐、collective count/payload/transit/wait、HBM 和 rank skew；
7. 失败原型也保存命令和日志。

## 11. 工具启动与产物定位

### 11.1 完整可选依赖

```bash
python -m pip install -r \
  tests/glm5_2_performance/requirements-visualization.txt

export TORCHTITAN_MSPROF_ANALYZE=/path/to/msprof-analyze
export TORCHTITAN_MSINSIGHT_FLAMEGRAPH=/path/to/flamegraph.py
export TORCHTITAN_FLAMEGRAPH_PL=/path/to/FlameGraph/flamegraph.pl
```

### 11.2 TensorBoard

```bash
tensorboard --logdir <run>/trainer_output/tensorboard \
  --port 6006 --bind_all
```

### 11.3 Perfetto

浏览器打开 [ui.perfetto.dev](https://ui.perfetto.dev/) 后导入报告链接的
`trace_view.json`。敏感环境使用本地部署，避免上传内部 trace。

### 11.4 MindStudio Insight

复制完整 `*_ascend_pt` 目录，在 Insight 中导入 profile 根。多 rank 推荐把同次 capture
按官方目录结构一起导入；只有单 rank 时 Summary/Communication 页可能不出现，这是
工具数据范围限制，不一定是采集失败。

### 11.5 典型目录

```text
performance_runs/<card-scope>/<topology>/<run>/
  runtime.log
  raw_metrics.jsonl
  trainer_output/
    tensorboard/events.out.tfevents.*
    profiling/traces/
      *_ascend_pt/
        ASCEND_PROFILER_OUTPUT/
          ascend_pytorch_profiler_<rank>.db
          trace_view.json
          *.csv
      stacks/*_stacks.log
      stacks/*_flamegraph.svg
      mindstudio_flamegraphs/*/flamegraph.html
      mindstudio_flamegraphs/*/mindstudio_flamegraph.log
      memory_timeline/*_memory_timeline.html
      memory_timeline/*_memory_timeline.json.gz
      memory_timeline/*_memory_timeline_raw.json.gz
  advisor*/
  cluster*/
  compare*/
  graph_visualization/        # compiled endpoint only
```

## 12. 最后检查清单

- [ ] 我确认了 device/topology/preset/repeat/commit。
- [ ] 我没有把阶段徽标当成性能 PASS。
- [ ] 我用 profiler-off 重复实验判断速度。
- [ ] 我检查了 active collection overhead。
- [ ] 多卡时我比较了 rank min/median/max。
- [ ] 我区分了 transit 与 wait、总通信与暴露通信。
- [ ] 我用 shape/dtype/calls 约束了热点 kernel。
- [ ] 我在 MindStudio 验证了先后、依赖、重叠和空洞。
- [ ] 我知道火焰图横轴不是时间线。
- [ ] 我知道 active/allocated/reserved/device-used 不同。
- [ ] 图模式问题我打开了 tlparse/FX/IR/code。
- [ ] 优化前后 precision 通过，并做了多次 profiler-off A/B。
