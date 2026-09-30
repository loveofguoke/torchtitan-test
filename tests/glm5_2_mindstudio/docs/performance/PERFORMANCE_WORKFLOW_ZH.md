# MindStudio 官方性能工作流

本流程首先完成每个拓扑自身的性能健康分析和瓶颈定位，其次才做版本、拓扑或
GPU/NPU 对比。它不重新实现 Profiler，而是把官方工具按职责串起来：

```text
同一训练契约
  +-- 每个拓扑 profiler-off 重复运行 ------------> 自身性能数值基线
  +-- 每个拓扑系统级采集 ------------------------> 自身瓶颈分析
  |     +-- NPU: Ascend PyTorch Profiler/msProf
  |     +-- GPU: Nsight Systems
  |     +-- Summary -> Timeline -> 专项页面/工具
  +-- 单拓扑完成后可选 A/B ----------------------> 版本/拓扑/GPU-NPU 对比
        +-- Advisor/Cluster/Compare/calibrate_npu_gpu
        +-- TopN 分支验证与 profiler-off 回归
```

采集、分析、可视化是三个阶段。采集或分析命令成功不等于性能通过；真实速度始终
以 profiler-off 重复 A/B 的 step time、throughput 和资源指标为准。

## 1. 官方依据

- [MindStudio 总入口](https://www.hiascend.com/document/detail/zh/mindstudio/latest/index/index.html)
- [MindStudio Insight 概述](https://www.hiascend.com/document/detail/zh/mindstudio/latest/GUI_baseddevelopmenttool/MindStudioInsight/docs/zh/user_guide/overview.md)
- [msProf 快速入门](https://www.hiascend.com/document/detail/zh/mindstudio/latest/msTT_msIT/msProf/docs/zh/quick_start/msprof_quick_start.md)
- [Ascend PyTorch Profiler](https://www.hiascend.com/document/detail/zh/mindstudio/latest/msTT_msIT/ascend_pytorch_profiler/docs/zh/ascend_pytorch_profiler/ascend_pytorch_profiler_user_guide.md)
- [msprof-analyze 快速入门](https://www.hiascend.com/document/detail/zh/mindstudio/latest/msTT_msIT/msprof_analyze/docs/zh/quick_start/msprof-analyze_quick_start.md)
- [msprof-analyze compare](https://www.hiascend.com/document/detail/zh/mindstudio/latest/msTT_msIT/msprof_analyze/docs/zh/user_guide/compare_tool_instruct.md)
- [msprof-analyze advisor](https://www.hiascend.com/document/detail/zh/mindstudio/latest/msTT_msIT/msprof_analyze/docs/zh/user_guide/advisor_instruct.md)
- [msprof-analyze cluster](https://www.hiascend.com/document/detail/zh/mindstudio/latest/msTT_msIT/msprof_analyze/docs/zh/user_guide/cluster_analyse_instruct.md)
- [msprof-analyze 进阶分析](https://www.hiascend.com/document/detail/zh/mindstudio/latest/msTT_msIT/msprof_analyze/docs/zh/advanced_features/README.md)
- [Insight 系统调优快速入门](https://www.hiascend.com/document/detail/zh/mindstudio/latest/GUI_baseddevelopmenttool/MindStudioInsight/docs/zh/quick_start/system_tuning_quick_start.md)
- [msMemScope 快速入门](https://www.hiascend.com/document/detail/zh/mindstudio/latest/msTT_msIT/msMemScope/docs/zh/quick_start/quick_start.md)
- [msOpProf 使用场景](https://www.hiascend.com/document/detail/zh/mindstudio/latest/msOT/Operatordevelopmenttools/docs/zh/user_guide/msopprof_usage.md)
- [msServiceProfiler 源码与文档](https://gitcode.com/Ascend/msserviceprofiler)
- [MindStudio 实践案例](https://www.hiascend.com/document/detail/zh/mindstudio/latest/practicalcases/index.html)
- [大模型训练性能瓶颈定位官方案例](https://www.hiascend.com/document/detail/zh/mindstudio/latest/practicalcases/Largemodeltraining/MindStudio/26.1.0/zh/cases/case_of_troubleshooting_performance_bottleneck_in_llm_training.md)
- [性能问题通用定位官方指南](https://www.hiascend.com/document/detail/zh/mindstudio/latest/practicalcases/GeneralPerformanceIssue/MindStudio/26.1.0/zh/cases/general_performance_issue_troubleshooting_guide/overview.md)
- [NVIDIA Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/)
- [NVIDIA Nsight Systems Post-Collection Analysis](https://docs.nvidia.com/nsight-systems/AnalysisGuide/index.html)
- [NVIDIA Nsight Compute CLI](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html)
- [NVIDIA Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)
- [MLPerf Training Rules](https://github.com/mlcommons/training_policies/blob/master/training_rules.adoc)

`latest` 用于阅读。正式实验必须按 Driver/Firmware/CANN/PyTorch/torch_npu 兼容
矩阵选择版本，并把 lock、resolved manifest、可执行文件和源码 commit 写入 manifest。

### 1.1 官方资料与本文覆盖关系

| 官方资料 | 核心内容 | 本流程中的位置 |
|---|---|---|
| [性能问题定位流程](https://www.hiascend.com/document/detail/zh/mindstudio/latest/practicalcases/GeneralPerformanceIssue/MindStudio/26.1.0/zh/cases/general_performance_issue_troubleshooting_guide/positioning_process_for_performance_issues.md) | 问题信息、目标来源、快速/详细定位、单变量验证 | 第 2 节总流程 |
| [性能工具使用](https://www.hiascend.com/document/detail/zh/mindstudio/latest/practicalcases/GeneralPerformanceIssue/MindStudio/26.1.0/zh/cases/general_performance_issue_troubleshooting_guide/performance_tool_usage.md) | msProf/框架 Profiler、msprof-analyze、Insight 的职责 | 第 4 至第 7 节 |
| [大模型训练性能瓶颈案例](https://www.hiascend.com/document/detail/zh/mindstudio/latest/practicalcases/Largemodeltraining/MindStudio/26.1.0/zh/cases/case_of_troubleshooting_performance_bottleneck_in_llm_training.md) | Summary 定界计算/通信/调度，Timeline/Operator/Communication 下钻 | 第 2.4、2.5 节 |
| [Ascend PyTorch Profiler](https://www.hiascend.com/document/detail/zh/mindstudio/latest/msTT_msIT/ascend_pytorch_profiler/docs/zh/ascend_pytorch_profiler/ascend_pytorch_profiler_user_guide.md) | 静态、动态采集，MSTX、环境变量、标记、内存、子线程、离线解析 | 第 4、5 节 |
| [msProf](https://www.hiascend.com/document/detail/zh/mindstudio/latest/msTT_msIT/msProf/docs/zh/quick_start/msprof_quick_start.md) | 无框架语义的 CANN/NPU 命令行采集与 Insight | 第 4、5 节 |
| [msprof-analyze 快速入门](https://www.hiascend.com/document/detail/zh/mindstudio/latest/msTT_msIT/msprof_analyze/docs/zh/quick_start/msprof-analyze_quick_start.md) | Advisor 最小闭环与 HTML/XLSX | 第 6 节 |
| [Advisor](https://www.hiascend.com/document/detail/zh/mindstudio/latest/msTT_msIT/msprof_analyze/docs/zh/user_guide/advisor_instruct.md) | overall/computation/schedule 自动诊断 | 第 1.3、6 节 |
| [Cluster](https://www.hiascend.com/document/detail/zh/mindstudio/latest/msTT_msIT/msprof_analyze/docs/zh/user_guide/cluster_analyse_instruct.md) | 慢卡、慢节点、慢链路、通信时间和矩阵 | 第 2.4、6 节 |
| [Compare](https://www.hiascend.com/document/detail/zh/mindstudio/latest/msTT_msIT/msprof_analyze/docs/zh/user_guide/compare_tool_instruct.md) | NPU-NPU/GPU-NPU 的时间、通信、调度和内存对比 | 第 2.3、6 节 |
| [进阶分析总表](https://www.hiascend.com/document/detail/zh/mindstudio/latest/msTT_msIT/msprof_analyze/docs/zh/advanced_features/README.md) | 拆解、计算、通信、Host、导出和自定义 Recipe | 第 1.2、3、6 节 |
| [TopN 性能问题总览](https://www.hiascend.com/document/detail/zh/mindstudio/latest/practicalcases/GeneralPerformanceIssue/MindStudio/26.1.0/zh/cases/general_performance_issue_troubleshooting_guide/solution_to_topn_overview.md) | 高频根因的专项解决方案 | 第 1.4、2.5 节 |

### 1.2 msprof-analyze 完整能力目录

不能把 `msprof-analyze` 简化成一个 Advisor。官方功能按输入和目标分为：

| 类别 | 官方功能 | 回答的问题 |
|---|---|---|
| 核心分析 | `advisor all/computation/schedule` | 自动筛查常见计算、通信、下发、数据和内存问题 |
| 核心分析 | `cluster -m all/communication_time/communication_matrix` | 慢 Rank、慢节点、慢链路、等待与真实传输 |
| 核心分析 | `compare` | 当前与 baseline 的算子、Module、通信、调度和内存差异 |
| 拆解与比对 | `cluster_time_summary`、`cluster_time_compare_summary` | step 各时间组成以及与基线的增量来源 |
| 拆解与比对 | `module_statistic` | 按 PyTorch Module 层次定位性能热点 |
| 跨平台 | `calibrate_npu_gpu` | 用 NVTX/MSTX 对齐 GPU/NPU Module 并比较 kernel 时间 |
| 计算 | `compute_op_sum`、`freq_analysis` | 设备计算热点与 AI Core 降频/空闲 |
| 计算 | `ep_load_balance` | MoE token、专家和 Rank 负载不均 |
| 计算 | `computational_op_masking` | 计算算子对通信的掩盖关系 |
| 计算 | `operator_mfu` | kernel/module FLOPs、实际 TFLOPS 与 dtype 峰值 MFU |
| 通信 | `communication_group_map`、`communication_time_sum`、`communication_matrix_sum`、`hccl_sum` | 通信域、时间、payload、带宽、矩阵与 collective 汇总 |
| 通信 | `slow_rank`、`slow_link`、`communication_bottleneck` | 快慢卡/链路及通信等待来自 Host 还是 Device |
| PP | `pp_chart` | 每 Rank 前向、反向、收发与流水空泡 |
| Host | `cann_api_sum`、`mstx_sum`、`free_analysis` | CANN API、用户区间、大块 Free 及下发原因 |
| 导出 | `export_summary` | 按 Rank 导出 API 和 Kernel 明细 |
| 数据处理 | `mstx2commop`、`p2p_pairing` | 转换通信标记、配对 P2P；会修改输入 DB，不能自动执行 |
| 扩展 | 自定义 Recipe | 在官方字段上增加项目规则，结果仍需保留来源与公式 |
| 图融合专项 | [Inductor+Triton 融合比较](https://www.hiascend.com/document/detail/zh/mindstudio/latest/msTT_msIT/msprof_analyze/misc/inductor_triton_performance_comparison/README.md) | FX 图中融合前后算子组成和耗时 |

输入能力并不相同：Advisor 和 Compare 需要框架 Profiler 数据；Cluster 支持的输入更广；
进阶 Recipe 主要读取 Ascend PyTorch Profiler 或 msMonitor 的 DB。必须先验证输入契约，
不能因为命令名称存在就声称当前 capture 支持该分析。

### 1.3 Advisor 实际自动检查什么

官方 `advisor all` 已包含大量适合自动化的规则，项目报告应索引它们，而不是再写一套
冲突阈值：

| 方向 | 官方检查项 |
|---|---|
| 总体 | 计算/通信/空闲拆解、环境变量、慢 Rank、慢链路 |
| 计算 | AI CPU、动态 shape、MatMul/FlashAttention/Vector/MIX_AIV、Block Dim、算子瓶颈、融合图、AI Core 降频 |
| 通信 | 小包、通算带宽争抢、重传、SDMA 数据量 512 Byte 对齐 |
| 调度 | 亲和 API、Path 3/Path 5 下发、SyncBatchNorm、SynchronizeStream、GC、可融合算子序列 |
| 数据 | DataLoader 过慢 |
| 内存 | 异常申请与释放操作 |
| 比较 | 无基准时比较快慢 Rank；有基准时比较同 Rank 的 Kernel/API 总时长、自耗时、均值和调用次数 |

Advisor HTML 适合先看优先级；完整明细在 XLSX。`inf` 可能只是分母为零或一侧没有数据，
不能直接解释为无限性能退化。

### 1.4 官方实践案例如何进入流程

| 实践分支 | 官方入口 | 进入条件与正确下钻 |
|---|---|---|
| 通信问题 | [通信调优方案](https://www.hiascend.com/document/detail/zh/mindstudio/latest/practicalcases/GeneralPerformanceIssue/MindStudio/26.1.0/zh/cases/general_performance_issue_troubleshooting_guide/solution_to_top1.md) | 暴露通信或单个 collective 长；先区分快慢 Rank 与真实传输，再查小包、重传、对齐和带宽争抢 |
| 算子问题 | [算子性能调优方案](https://www.hiascend.com/document/detail/zh/mindstudio/latest/practicalcases/GeneralPerformanceIssue/MindStudio/26.1.0/zh/cases/general_performance_issue_troubleshooting_guide/solution_to_top2.md) | 计算占关键路径；TopN -> shape/MFU/调用栈 -> 替换、融合或单算子分析 |
| Host Bound | [Host Bound 定位](https://www.hiascend.com/document/detail/zh/mindstudio/latest/practicalcases/GeneralPerformanceIssue/MindStudio/26.1.0/zh/cases/general_performance_issue_troubleshooting_guide/solution_to_top3.md) | Free 高、HostToDevice 近竖直；查小算子、同步、CPU 亲和、GIL、I/O、后台抢占和 task queue |
| 集群长稳波动 | [集群性能异常波动方法论](https://www.hiascend.com/document/detail/zh/mindstudio/latest/practicalcases/GeneralPerformanceIssue/MindStudio/26.1.0/zh/cases/general_performance_issue_troubleshooting_guide/solution_to_top4.md) | 先关联变更与硬件监控做粗定位，再对异常窗口做细 Profiling |
| ONNX 离线推理 | [ONNX 离线推理方案](https://www.hiascend.com/document/detail/zh/mindstudio/latest/practicalcases/GeneralPerformanceIssue/MindStudio/26.1.0/zh/cases/general_performance_issue_troubleshooting_guide/solution_to_top5.md) | 属于推理交付，不进入 TorchTitan 训练自动流程 |
| MindIE 推理 | [MindIE 推理性能方案](https://www.hiascend.com/document/detail/zh/mindstudio/latest/practicalcases/GeneralPerformanceIssue/MindStudio/26.1.0/zh/cases/general_performance_issue_troubleshooting_guide/solution_to_top6.md) | 属于服务化调优，使用 msServiceProfiler 等服务工具 |
| 版本升级退化 | [版本升级性能劣化方法论](https://www.hiascend.com/document/detail/zh/mindstudio/latest/practicalcases/GeneralPerformanceIssue/MindStudio/26.1.0/zh/cases/general_performance_issue_troubleshooting_guide/solution_to_top7.md) | 固定业务合同，版本二分，并用 Compare/Advisor baseline 定位增量 |

官方“大模型训练性能瓶颈”案例还展示了一个重要反例：通信耗时高可能来自 Rank 不同步，
而不是网络传输慢。必须从 Communication 页的 wait/transmit 和 collective 上游 Timeline
继续定位，不能见到 HCCL 时间长就直接更换网络配置。

## 2. 标准性能诊断流程

性能实验不是“一上来开 Profiler”，也不是只看一次 `tokens/s`。精度不劣化是性能调优
的前置条件；如果优化改变数值算法，必须明确精度代价是否可接受。标准流程固定为：

```text
定义问题、性能指标和目标来源
  -> 精度与训练语义前置检查
  -> 同一训练合同与环境检查
  -> profiler-off 重复正常训练，建立无侵入基线
  -> 判断问题属于绝对性能低、NPU/GPU 差异、回归、长稳波动还是 Rank 不均衡
  -> 对稳定区间做低开销系统级采集
  -> Overview/Cluster 将时间定界为计算、暴露通信、空闲/调度、内存搬运或流水空泡
  -> 只对异常分支做 Timeline/Operator/Communication/Memory 深入分析
  -> 必要时进入算子、Host、内存、编译融合或硬件专项工具
  -> 单变量修复，回到 profiler-off 同合同重复 A/B，确认收益和副作用
```

这与[大模型训练性能瓶颈定位官方案例](https://www.hiascend.com/document/detail/zh/mindstudio/latest/practicalcases/Largemodeltraining/MindStudio/26.1.0/zh/cases/case_of_troubleshooting_performance_bottleneck_in_llm_training.md)
和[性能问题通用定位指南](https://www.hiascend.com/document/detail/zh/mindstudio/latest/practicalcases/GeneralPerformanceIssue/MindStudio/26.1.0/zh/cases/general_performance_issue_troubleshooting_guide/guide.md)
的“由面及点”原则一致。官方 TopN 案例覆盖通信、算子、Host Bound、集群长稳波动、
版本升级回归等高频问题；它们是定界后的分支手册，不替代前面的正常训练基线和总览。

### 2.1 第零步：固定可比较的训练合同

开始运行前先写清楚问题属于迁移后开箱性能低、历史版本回归、长稳随机波动、扩容
线性度不足还是某个 Rank/节点异常；目标必须说明来自 GPU/竞品标杆、历史健康版本、
理论线性扩展还是产品约束。如果通过增大 batch 提升吞吐，就不能再拿单 step time
作为唯一优化指标。

同一次实验必须固定模型与 checkpoint、token plan/数据顺序、global/local batch、
sequence length、dtype、拓扑及各并行度、重计算、优化器、图/eager 模式、融合开关、
设备集合、软件栈和环境变量。GPU/NPU 允许使用各自合法的实现，但任何语义差异都必须
进入 manifest；不能把 batch、并行策略或有效 token 数不同的两次运行称为平台对比。

同时记录机器占用、频率、温度、ECC/链路状态、CPU/NUMA、后台进程和数据盘状态。
共享机器上一次慢跑只能作为线索，不能直接归因到某张卡。

### 2.2 第一步：profiler-off 正常训练

先关闭 Profiler，完成 warmup 后采集足够多的稳态 step，并至少独立重复三次。主要证据是：

- 每 step 原始耗时、median/p90/p95、变异系数和长尾 step；
- rank 级 min/median/max step time 与慢 Rank；
- 每 rank 与整作业 tokens/s、估算 TFLOPS/MFU；
- active/reserved/峰值显存、OOM/重试/碎片迹象；
- loss/grad norm，用于排除“更快但训练语义已经变化”。

这一层回答“是否真的慢、慢多少、是否稳定、从什么时候开始慢”。Profiler-active
时间只能用于归因，不能替代 profiler-off 性能结论。

### 2.3 第二步：先完成单拓扑自身分析，再做性能对比

性能分析的主任务不是“证明 NPU 比 GPU 快或慢”，而是解释一个拓扑自身的时间花在
哪里、是否存在异常、关键路径是什么以及如何优化。每个 NPU/GPU 拓扑都必须先形成
独立分析结论；只有两端都完成这一步，比较才不会把一侧自身的慢 Rank、Host 抢占或
异常 shape 错当成平台差异。

单拓扑必须依次回答：

1. profiler-off 的稳态吞吐、step 分布和显存是否稳定；
2. 同类 Rank、stage、通信域之间是否均衡；
3. 关键路径主要是计算、暴露通信、Free、内存搬运还是 bubble；
4. 最耗时 Module、算子、kernel、collective 和 Host API 分别是什么；
5. 这些热点是模型必需成本、合理高占比，还是低 MFU、小算子、等待、重传、下发空洞；
6. 哪个官方页面/文件/行支持结论；
7. 哪个单变量实验能证伪或验证根因；
8. profiler-off 复测是否得到稳定收益且精度不退化。

完成后再选择比较对象：

| 场景 | 基准 | 先回答的问题 |
|---|---|---|
| NPU 单端 | 同一作业各 Rank、相邻稳态 step、历史健康运行 | 慢卡、慢 step、Host/通信/计算/内存哪一类异常 |
| GPU 单端 | 同上，使用 Nsight Systems | CUDA launch、GPU gap、NCCL、kernel 或内存哪一类异常 |
| NPU-NPU 回归 | 旧版本/旧提交/旧配置 | 哪个时间组成或模块发生增量 |
| GPU-NPU 迁移 | 同训练合同的 GPU 标杆 | 总吞吐差异来自哪个模块、算子、通信或调度阶段 |

NPU 标准采集使用 Ascend PyTorch Profiler；GPU 标准采集使用 Nsight Systems。
GPU/NPU 模块级对比使用官方
[`calibrate_npu_gpu`](https://www.hiascend.com/document/detail/zh/mindstudio/latest/msTT_msIT/msprof_analyze/docs/zh/advanced_features/calibrate_npu_gpu_instruct.md)：
GPU 输入为 Nsys SQLite，NPU 输入为 Ascend PyTorch Profiler DB，并要求完整的
NVTX/MSTX Module 标记和 kernel 数据。工具通过规则和模糊匹配对齐模块，输出两侧
算子数、kernel、总/平均 kernel 时间、模块耗时比和耗时差。匹配结果仍需人工检查；
fuzzy match 不是语义等价证明。

### 2.3.1 不同拓扑自己的合理分析重点

不能拿 DDP 的“正常图形”要求 FSDP、TP、PP 或 EP。报告必须从 topology contract
生成不同的检查项：

| 拓扑 | 正常情况下应看到 | 优先排查的异常 |
|---|---|---|
| single | 主要由前反向计算和优化器构成，无跨卡 collective | Host Free、低 MFU、AICPU/fallback、小算子、显存和 DataLoader |
| DDP | 各 Rank 计算接近，反向梯度 AllReduce 可与反向计算重叠 | 慢 Rank、bucket 过小/过大、AllReduce 暴露、数据或 Host 不均衡 |
| FSDP | 分层参数 AllGather、反向 ReduceScatter，参数按 shard 驻留 | AllGather/RS 无法掩盖、prefetch 时机、reshard、碎片和峰值显存 |
| TP | 每层较频繁的 TP collective，单卡 GEMM shape 随 TP 改变 | 小 GEMM 导致 MFU 下降、collective 高频、切分不整齐、通信暴露 |
| CP | Attention 序列切分及对应 P2P/collective | 长序列通信、负载不均、FA/attention kernel shape 与重叠 |
| PP | stage 计算、P2P send/recv、前反向流水 | stage 不均、microbatch 不足、warmup/cooldown 和 bubble |
| EP | token dispatch/permute、AllToAll、专家计算、combine | token/专家不均、AllToAll payload 不均、小专家 GEMM、热点 Rank |
| HSDP/复合并行 | 各 mesh 轴对应的通信同时存在 | 先按通信域归属 DP/TP/PP/CP/EP，再分析，不能汇总成一个“通信慢” |

“应看到”描述的是结构，不是固定百分比。是否合理仍需结合模型理论量、同组 Rank、
历史健康基线和 profiler-off 性能验证。

### 2.4 第三步：低开销系统采集与一级定界

只采集 warmup 后的短稳态窗口。单卡用 `standard`，多卡用 `distributed`；全拓扑
初筛用 `overview`。首先看官方 Summary/cluster time decomposition：

```text
T_step ~= T_compute
        + T_communication_not_overlapped
        + T_memory_not_overlapped
        + T_free
        + T_pipeline_bubble
```

各项是时间区间的分类/并集，不应把每个 stream 或 kernel 的 duration 简单相加。
`communicationOverlapComputation` 是已经被计算掩盖的通信，不能再次加到关键路径。

| 一级现象 | 主要证据 | 下一步 |
|---|---|---|
| 计算慢或 Rank 计算不均 | computation、Operator TopN、shape、kernel MFU | Operator、卡间算子对比、单算子 |
| 暴露通信长 | communicationNotOverlapComputation、wait/transmit、group/matrix | Communication、慢 Rank/慢链路 |
| Free/下发长 | free、taskLaunchDelay、HostToDevice 连线、CPU/GIL | Timeline、Function Monitor、GIL Tracer |
| 内存/搬运长 | memoryNotOverlap、Memcpy、active/reserved、反复申请释放 | Memory、msMemScope |
| PP 空泡或 stage 不均 | stage、bubble、各 stage/rank 时长 | PP chart、切分与 microbatch |
| EP 负载不均 | token/shape、专家负载、AllToAll payload | ep_load_balance、路由与专家放置 |
| 图编译/融合退化 | graph break、recompile、融合前后 kernel 数与耗时 | 编译日志、融合专项比较 |

官方大模型案例把异常概括为计算、通信和调度三大类；这里把内存、PP/EP 和图编译
作为可直接观测的下钻分支展开，但不改变官方定界逻辑。

### 2.5 第四步：按问题分支逐层下钻

- **计算**：先按类型、名称和 shape 查看 Operator TopN，再看 AI Core/AI CPU、
  Cube/Vector、kernel duration 与 MFU。不要因为某算子总耗时高就认定它低效；高频核心
  GEMM 本来就应占主要时间，需要结合 FLOPs、shape 和 MFU。
- **通信**：先区分等待、同步和真实传输；再按同一通信域、同一 collective、同一链路
  类型比较 payload、带宽和 Rank 到达时间。等待长不等于网络慢。
- **Host Bound**：Timeline 中密集的竖直 HostToDevice 连线和大块 Free 表示 Device
  很快消费完下发任务。继续检查 CPU 亲和、后台抢占、GIL、I/O、同步 API、小算子过多、
  task queue；`ASCEND_LAUNCH_BLOCKING=1` 会关闭异步队列，只能用于定位，不能作为性能配置。
- **内存**：同时看 active、reserved、峰值、碎片/重试和时间轴。显存占用高本身不等于
  异常；如果它来自可复用缓存且没有挤压 batch，可能是合理行为。
- **长稳波动**：先关联近期变更、硬件与系统监控，再对异常窗口做 Profiling；大集群采用
  “粗定位异常时段/机组 -> 细定位调用栈、I/O、锁、计算和通信”的两阶段策略。
- **图与融合**：官方
  [Inductor+Triton 融合算子性能对比](https://www.hiascend.com/document/detail/zh/mindstudio/latest/msTT_msIT/msprof_analyze/misc/inductor_triton_performance_comparison/README.md)
  读取 `INDUCTOR_ASCEND_FX_GRAPH_CACHE` 中的 FX 图，输出融合前后算子与耗时；融合后耗时
  占融合前总耗时小于 100% 才表示该融合本身获得收益。它不是整网 NPU/GPU 比较。

### 2.5.1 可视化结果的固定阅读顺序

```text
正常训练趋势
  -> Summary/Cluster 热力图
  -> 选定异常 step、rank、stage、通信域
  -> Timeline 找关键路径和首个分叉点
  -> Operator/Communication/Memory 验证组成
  -> Advisor/进阶 Recipe 提供自动规则和明细
  -> 源码、配置或单算子实验验证根因
```

具体页面不能孤立阅读：

| 页面/交付件 | 先看什么 | 不能直接得出的结论 |
|---|---|---|
| Summary/cluster time | 各 Rank 的 compute、暴露通信、free、memory、bubble | 占比高不自动说明实现错误 |
| Timeline | 异常 Rank 与正常 Rank 的首个时间分叉、HostToDevice、stream 空洞、collective 前序 | 某通信条长不等于网络慢 |
| Operator | TopN type/name/shape、count、总/平均耗时、AI Core/AICPU、MFU | 总耗时最高不等于单次低效 |
| Communication | wait/transmit/sync、group、payload、同类链路带宽 | wait 高不等于传输带宽差 |
| Memory | active/reserved、峰值、反复申请释放、拷贝和碎片迹象 | reserved 高不等于泄漏 |
| Advisor HTML/XLSX | 先看高优先级，再看 XLSX 全量原始项 | 建议命中不等于根因已证明 |
| Compare/calibrate | match type、Module/算子数、绝对耗时和 diff/ratio | 模糊匹配不证明两侧语义相同 |

### 2.5.2 TopN 实践不是附录，而是标准流程的诊断分支

一级定界后必须进入对应分支，并执行“发现 -> 证据 -> 根因分叉 -> 修复 -> 复测”：

| 分支 | 自动/人工发现 | 可视化与官方证据 | 主要根因分叉 | 常见解决动作 | 必须复测 |
|---|---|---|---|---|---|
| 通信 | 暴露通信、单 collective、Rank skew 或链路带宽异常 | Summary、Communication、matrix、slow_rank/link、communication_bottleneck | 快慢 Rank，或真实传输；再分小包、重传、512B 对齐、带宽争抢、拓扑映射 | 平衡上游计算/数据；调整 bucket/并行与 overlap；修复链路/重传/对齐 | 同规模 profiler-off、Rank skew、暴露通信和精度 |
| 算子 | computation 高、TopN 热点、低 MFU、AICPU/fallback | Operator、shape、调用栈、operator_mfu、Advisor、msOpProf/NCU | 必需大算子但利用率低；动态 shape；小算子碎片；非亲和/AI CPU；融合退化 | shape/tiling、亲和算子替换、融合、减少转换、升级算子、定点 kernel 调优 | 整网 profiler-off；不能只报告单算子 microbenchmark |
| Host Bound | Free 高、竖直 HostToDevice、小 kernel 间隙、launch delay | Timeline、free_analysis、cann_api_sum、Function Monitor、GIL Tracer、CPU/线程栈 | 下发过碎、同步 API、GC/GIL、CPU 亲和/抢占、I/O/DataLoader、task queue 关闭 | 融合/批量下发、异步化、绑核、消除同步、优化 DataLoader、正确开启 task queue | Free、step time、CPU 利用与显存峰值；task queue 可能增加峰值 |
| 长稳波动 | step p95/max、CV、MAD、某时段多 Rank 同时劣化 | 训练趋势、系统监控、硬件告警、异常窗口 Profile、火焰图/调用链 | 近期配置/版本、CPU/内存/网络波动、硬件告警、锁/I/O、特定周期任务 | 回滚或二分变更、隔离异常节点、修复系统任务/链路/锁 | 长时间相同负载复跑，不能只验证短窗口 |
| 版本回归 | 同合同历史 baseline 变慢 | profiler-off A/B、Advisor `-bp`、Compare、cluster_time_compare | 算子/kernel、API 调用次数、通信、调度、内存或编译策略变化 | 版本/提交二分，替换具体回归组件或调整新增策略 | 目标版本完整 A/B，记录工具链和 commit |
| 图融合 | graph break/recompile、kernel 数和融合块耗时异常 | 编译日志、FX/IR/code、Inductor+Triton 融合比较、Timeline | 未融合、错误切图、融合块过大、tiling/kernel 退化、fallback | 修复 graph break、调整分解/融合、定点优化生成 kernel | eager/graph 精度 + profiler-off 性能 + 编译稳定性 |
| 内存 | OOM、峰值异常、反复申请释放、Memcpy 暴露 | Memory、memory timeline、Advisor、msMemScope | 参数/优化器/激活理论量，碎片，生命周期，临时 workspace，通信 buffer | 重计算、分片、生命周期/缓存、bucket、allocator 或 workspace 优化 | 峰值、吞吐、重计算成本和精度 |

ONNX 与 MindIE 是同一 TopN 总览中的推理专项，不自动进入 TorchTitan 训练流程；如果
任务切换为服务化推理，再改用其预检、服务调度和 msServiceProfiler 链路。

### 2.6 第五步：修复与闭环

每次只改变一个候选因素；使用同一 profiler-off 合同重新做多次 A/B。至少同时检查
step time/tokens/s、显存、稳定性和 loss/grad norm。优化只在被诊断的指标改善、没有把
瓶颈转移到其他 Rank/阶段、且正常训练语义保持时成立。

## 3. 自动化分析的证据合同

自动化应减少人工翻表，不应发明统一阈值。报告可自动计算并展示：

| 指标 | 计算或来源 | 能回答什么 |
|---|---|---|
| 稳态 step 分布 | p50/p90/p95/max、CV、MAD 异常点 | 是否有长尾和长稳波动 |
| Rank skew | `(max_rank - min_rank) / median_rank` | 是否存在快慢 Rank；再按时间组成归因 |
| 计算占比 | `computation / stepTime` | 当前窗口 Device 计算份额 |
| 暴露通信占比 | `communicationNotOverlapComputation / stepTime` | 真正进入关键路径的通信 |
| 通算覆盖率 | `communicationOverlapComputation / totalCommunication` | 通信有多少被计算掩盖 |
| 空闲占比 | `free / stepTime` | Host 下发、同步或未分类空洞候选 |
| 内存暴露占比 | `memoryNotOverlapComputationCommunication / stepTime` | 未被通算覆盖的搬运成本 |
| PP 空泡率 | `bubble / stage_or_step_time` | microbatch/schedule/stage 切分候选 |
| 有效链路带宽 | payload / transit time，同链路类型内比较 | 慢链路或小包候选 |
| kernel/module MFU | 官方 `operator_mfu` 的 FLOPs、duration、dtype peak | 核心计算是否接近芯片相应 dtype 峰值 |
| 显存组成 | 参数/梯度/优化器/激活理论量 + active/reserved 实测 | 容量是否符合并行和重计算合同 |
| 算子 TopN | count、total/avg duration、shape、AI Core/AI CPU | 热点、碎片化、小算子和 fallback 候选 |
| GPU/NPU 模块差 | 官方 calibrate 的 ratio/diff/match type | 首个或最大平台性能差异模块 |

“正常比例”不能脱离模型和拓扑给一个常数。正确做法是三层参照：

1. **理论合同**：由模型 FLOPs、参数/梯度/优化器/激活大小、DP/TP/PP/CP/EP/FSDP
   通信量和设备 dtype 峰值计算期望量级；
2. **同组内部**：同一 step、同一通信域和同类 Rank 比较，自动标出离群 Rank/链路/shape；
3. **健康基线**：和同硬件同软件栈的历史运行或 GPU 标杆做 A/B，显示绝对值和差值。

例如 DDP 通常在反向末端出现梯度 AllReduce；FSDP 在前向/反向有参数 AllGather 和
反向 ReduceScatter；TP 每层通信频繁；PP 重点看 stage 平衡和 bubble；EP 重点看
AllToAll 与 token/专家负载。报告应据拓扑选择解释和下一步，而不是要求所有拓扑满足
同一个“计算 70%、通信 20%”比例。

自动输出只允许使用以下结论等级：`observed`（原始/官方字段）、`derived`（明确公式）、
`suspect`（有证据的候选）和 `not available`。不得把启发式阈值变成 PASS/FAIL。

### 3.1 当前实现状态与下一步边界

现有能力不能笼统写成“性能流程已全部支持”：

| 能力 | 当前状态 | 说明 |
|---|---|---|
| NPU profiler-off、Ascend PyTorch Profiler、msProf | 已实现 | 共享 topology/训练配置，采集与分析可独立续跑 |
| NPU Advisor、基础 Cluster、必要进阶 recipe、Insight handoff | 已实现 | 保留官方 DB/CSV/JSON/XLSX，不重算官方结论 |
| 单拓扑证据化诊断与 TopN 路由 | 已实现第一阶段 | 每次 `--analyze`/`--probe` 生成 `diagnosis/self/diagnosis.json` 和 README；只输出 observed/suspect/not_available，并另记 capture semantics，不输出 PASS/FAIL |
| GPU Nsight Systems 与本地 stats 诊断 | 已实现，独立入口 | `tests/glm5_2_nvidia`，与 NPU 共享拓扑定义但输出树独立 |
| GPU Nsight Compute 定点 kernel 下钻 | 已实现，独立入口 | 只对已筛选 kernel replay，不能代替自然训练时间线 |
| NPU/GPU 同合同自动编排与 profiler-off 汇总比较 | 尚未统一 | 目前需分别运行并人工核对 manifest |
| 官方 `calibrate_npu_gpu` 一键编排 | 尚未接通 | recipe 已登记但会明确跳过，直到 Nsys SQLite、NPU DB、Module 标记和跨机输入生命周期完整接入 |
| 理论显存/通信量与实测组成的统一报告 | 部分具备原始数据，尚未统一 | 不能先写固定比例阈值；应按模型和 topology 生成期望量级 |
| Inductor+Triton 融合专项比较 | 尚未接入 | 它需要独立 FX cache 与官方脚本输出，不能伪装成普通 cluster recipe |

因此下一阶段的正确实现顺序是：先建立 GPU/NPU 共用的性能实验合同和 profiler-off
对比索引，再接官方 `calibrate_npu_gpu`，随后把 cluster/NSys 原始字段转成上述
`observed/derived/suspect` 报告，最后接图融合、算子和内存专项分支。不能为了得到一张
“统一报告”而复制官方算法或混合两种 profiler 的不等价字段。

官方 [`operator_mfu`](https://www.hiascend.com/document/detail/zh/mindstudio/latest/msTT_msIT/msprof_analyze/docs/zh/advanced_features/operator_mfu_instruct.md)
只在采集包含 FLOPs 时生成 kernel MFU；模块 MFU 还要求 `Module` domain 的 MSTX。
框架缺少这些输入时必须展示 `not available`，不能用总 step TFLOPS 反推并冒充算子 MFU。

### 3.2 输出结构与所有权

性能数据按“平台类别 -> 卡数范围 -> topology -> 一次不可变 capture”组织。原始采集、
生命周期元数据和可读报告分属 runs、artifacts、reports；不能把官方 DB/CSV 搬进
artifact，也不能用项目分析覆盖官方输出：

```text
mindstudio_runs/performance/system/
└── <card-scope>/
    └── <topology>/
        └── <run-name>/
            ├── experiment.json                 # 人可读实验合同
            ├── runtime.log                     # 训练与采集日志
            ├── metrics.jsonl                   # TorchTitan 每 step 原始指标
            ├── trainer_output/
            │   └── profiling/traces/
            │       ├── rank_*_ascend_pt/       # 官方各 Rank Profiler 根
            │       └── cluster_analysis_output/# 官方 Cluster DB/CSV/JSON
            ├── advisor/                        # 独立可续跑官方 Advisor
            ├── compare/                        # 独立可续跑官方 A/B 或 Rank compare
            └── diagnosis/
                └── self/
                    ├── diagnosis.json          # 项目单拓扑证据、状态和下一步
                    └── README.md               # 可快速阅读的分支摘要

mindstudio_artifacts/performance/system/<card-scope>/<topology>/<run-name>/
├── manifest.json                               # 不可变 capture 身份和环境
├── analysis.json                               # 项目归一化索引，含 self_diagnosis
├── analysis_state.json                         # 派生分析续跑状态
└── mindstudio_insight_handoff.json             # 可移植 Insight 导入目标

mindstudio_reports/performance/system/<card-scope>/<topology>/
└── <run-name>.html                             # 单拓扑完整阅读入口
```

GPU 使用平行的 `nvidia_{runs,artifacts,reports}/performance/system/` 根，官方
`.nsys-rep/.sqlite/stats` 仍由 run 所有。后续 GPU/NPU 比较是同一实验合同下的派生对象，
必须引用两端 manifest 和输入摘要，不能替换任一端的 `diagnosis/self/`。一个 run 的
`diagnosis/self` 回答“自己哪里慢”；`compare/` 回答“相对基线多花在哪里”，二者不是
baseline/candidate 的目录别名关系。

GPU 侧的标准下钻顺序是 `profiler-off -> NSys standard -> NSys
communication/host/memory -> 定点 NCU`。NSys 的 `.nsys-rep` 和 Timeline 用于观察
CPU/PyTorch/NVTX、CUDA API、stream、kernel、memcpy 和 NCCL 的时间关系；`nsys stats`
仅是官方 SQLite/报告脚本生成的轻量聚合。NCU 会通过 kernel/application/range replay
收集硬件计数器，因此只能对 NSys 已选出的稳定 kernel 和调用区间使用，不能用回放时间
解释自然训练中的跨 Rank 偏斜或通信覆盖。

MLPerf 不属于瓶颈定位工具，本项目也不能把普通 GLM 运行称作 MLPerf 成绩。这里只采用
它的测量纪律：固定系统与软件配置、保留完整日志、明确数据预处理和质量目标、运行规定的
独立重复，并且只比较兼容 benchmark/scenario。正式使用 MLPerf 名称还必须满足对应版本
规则、目标质量、最少 run 数、checker 和提交包要求。

## 4. 采集器与边界

| collector | 当前执行 | 层级 | 正确用途 |
|---|---:|---|---|
| `torch_npu_profiler` | 是，默认 | PyTorch/CANN/NPU | 在训练进程内按 step schedule 采集 module、shape、stack、memory 和设备证据。|
| `msprof` | 是，显式可选 | CANN/NPU | 用于命令行包裹整进程的底层或黑盒采集，得到 Insight 和 cluster 输入。|
| `msopprof` | 否 | 单算子/Kernel | 从整网定位热点后做上板或仿真下钻，不是整网训练 launcher。|
| `msmemscope` | 否 | 专项内存 | 内存泄漏、生命周期、低效内存；待目标 CANN 完成独立接入验证。|
| `service_profiler` | 否 | 在线推理服务 | 面向 MindIE/vLLM/SGLang 请求链路，不适用于离线训练。|

msProf 自身也不是只有一条整进程命令。官方能力还包括 AI 任务运行数据、AI Processor
系统数据、Host 系统数据、MSTX/msproftx 标记、动态采集、`delay/duration` 延迟与定时
采集、离线解析/查询/导出，以及 Function Monitor、GIL Tracer、Host 诊断调优等扩展。
标准 TorchTitan 训练优先使用带框架语义和 step schedule 的 Ascend PyTorch Profiler；
只有无法改代码、需要 CANN/NPU 黑盒证据，或已经由一级定界确认是 Host/GIL 专项问题时，
才进入这些 msProf 分支。动态采集与延迟采集等互斥能力必须遵守安装版本的官方约束。

Ascend PyTorch Profiler 同样包含多个模式：`torch_npu.profiler.profile` 静态采集、
`dynamic_profile` 动态启停、MSTX、环境变量采集、用户标记、Device 内存可视化、Profiler
子线程和离线解析。同一进程不能同时启用多种采集模式，也不应与精度 dump 同时开启；
dump 会改变性能并使 profiling 指标失真。

五者均注册在 `PerformanceCollector`。后三者作为训练 collector 会明确抛出用途与
server-validation/not-implemented 错误，不会虚构命令。

这里的“注册”表示 CLI 能识别其职责并阻止误用，不表示三者已经接入 TorchTitan
整网训练。`msmemscope` 官方既有 Python API，也有包装应用的命令行模式；完成当前
CANN、分布式子进程和输出生命周期验收后，才会从登记状态升级为可执行 collector。
`msopprof` 要求先隔离出待调优算子/Kernel；`service_profiler` 要求已经部署
MindIE/vLLM-Ascend/SGLang 服务。二者不能通过把整网训练命令硬塞进参数来冒充支持。

继续下钻时应回到各工具自己的官方入口，而不是把单算子、内存或在线服务参数塞进
整网训练命令：

- [msOpProf 26.1 simulator 指南](https://www.hiascend.com/document/detail/zh/mindstudio/2610/msOT/Operatordevelopmenttools/docs/zh/user_guide/msopprof_simulator_user_guide.md)：单算子上板/仿真、流水和热点；
- [msMemScope 源码](https://gitcode.com/Ascend/msmemscope)：整网显存采集、诊断和优化分析；
- [msServiceProfiler 源码与文档](https://gitcode.com/Ascend/msserviceprofiler)：MindIE、vLLM-Ascend、SGLang 等在线推理服务链路。

这些链接是学习和后续接入入口，不代表当前 TorchTitan 训练 harness 已验证它们。

msProf 是 CANN/NPU 的通用底层采集入口；Ascend PyTorch Profiler 在训练代码内
接入同一底层 profiling 能力，并补充 PyTorch 语义和 step schedule。PyTorch/
TorchTitan 标准流程默认使用 `torch_npu_profiler`，需要命令行包裹、无法修改程序
或专门做底层黑盒排障时才显式选择 `msprof`；测速时两者都关闭。

## 5. 采集命令

物理设备选择必须可追溯。`--devices` 接受有序物理设备 ID；每个拓扑使用前
`world_size` 个 ID。非默认卡组进入可读目录名、配置 SHA、manifest 和续跑判定，
例如 `--devices 6,7 --topology ddp2` 会带有 `dev6-7`。默认前缀卡组保持旧实验
身份，以便无重采集地继续使用已有结果。`--visible-devices` 是兼容别名。

```bash
python tests/glm5_2_mindstudio/performance_benchmark.py \
  --probe --device npu --devices 0,1 --topology ddp2 \
  --preset distributed --record-shapes \
  --analysis-tools all --cluster-recipes necessary

python tests/glm5_2_mindstudio/performance_benchmark.py \
  --probe --device npu --devices 6,7 --topology ddp2 \
  --preset distributed --record-shapes \
  --analysis-tools all --cluster-recipes necessary
```

入口：

```bash
# 默认 Ascend PyTorch Profiler 采集机
python -m tests.glm5_2_mindstudio.toolchain doctor \
  --scope performance-capture

# 兼容别名：Ascend PyTorch Profiler 采集机
python -m tests.glm5_2_mindstudio.toolchain doctor \
  --scope performance-torch-npu-capture

# 独立离线分析机
python -m tests.glm5_2_mindstudio.toolchain doctor \
  --scope performance-analysis

# 只有同一环境承担采集和分析时才运行
python -m tests.glm5_2_mindstudio.toolchain doctor --scope performance

python tests/glm5_2_mindstudio/performance_benchmark.py --help
```

### 5.1 profiler-off 数值基线

先关闭采集器重复测速。`--replicate` 是独立运行编号，不是 profiler schedule 的
repeat；三次运行分别落到独立目录。

```bash
# 单卡 3 次
export ASCEND_RT_VISIBLE_DEVICES=4
for r in 1 2 3; do
  python tests/glm5_2_mindstudio/performance_benchmark.py \
    --capture --device npu --profiler-off \
    --topology single --graph eager --npu-codegen ascend-triton \
    --replicate "$r"
done

# 一个分布式拓扑 3 次
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
for r in 1 2 3; do
  python tests/glm5_2_mindstudio/performance_benchmark.py \
    --capture --device npu --profiler-off \
    --topology fsdp8 --graph eager --npu-codegen ascend-triton \
    --replicate "$r"
done

# 所有不超过 8 卡的拓扑各 3 次
for r in 1 2 3; do
  python tests/glm5_2_mindstudio/performance_benchmark.py \
    --capture --device npu --profiler-off \
    --topology all --graph eager --npu-codegen ascend-triton \
    --replicate "$r"
done
```

以三次 profiler-off 的 median/p90 step time、tokens/s、peak HBM 和 rank
min/median/max 作为性能数值。下面 profiler-active 的结果只做归因。

### 5.2 profiler-active 标准采集与分析

日常标准入口是 `--probe`：同一条命令先完成 bounded capture，所有 rank 退出后再
离线解析，随后把 Advisor、Cluster 和 Insight handoff 写入各自目录。`--capture`
与 `--analyze` 只作为跨机器或补跑分析的高级接口。

单卡默认采集：

```bash
export ASCEND_RT_VISIBLE_DEVICES=4
python tests/glm5_2_mindstudio/performance_benchmark.py \
  --probe --device npu --collector torch_npu_profiler \
  --topology single --preset standard --analysis-tools all \
  --graph eager --npu-codegen ascend-triton
```

一个分布式拓扑：

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python tests/glm5_2_mindstudio/performance_benchmark.py \
  --probe --device npu --collector torch_npu_profiler \
  --topology fsdp8 --preset distributed --analysis-tools all \
  --graph eager --npu-codegen ascend-triton
```

所有不超过八卡的拓扑：

```bash
python tests/glm5_2_mindstudio/performance_benchmark.py \
  --probe --device npu --collector torch_npu_profiler \
  --topology all --preset overview --analysis-tools all
```

`all` 只是提供统一编排，并不表示应在共享服务器上一开始就全量采集。即使默认
Ascend PyTorch Profiler 使用 step schedule，重型 preset 与多拓扑、多 rank 做
笛卡尔积仍可能产生数百 GiB。正式矩阵应先跑 profiler-off 全拓扑取得可比基线，
再对 DP、TP、CP、PP、EP 和复合并行各选代表拓扑做深度 capture；只有存储预算、
采集窗口和保留策略明确时才执行上面的 `all`。

`standard` 默认启用 Level1、PipeUtilization 和 `profile_memory=True`，因此同一份
单卡 capture 可在 Insight 中查看 Timeline、Memory、Operator。`distributed`
在此基础上采集全部 rank 的通信与互联数据，同一份多卡 capture 可进一步查看
Summary 和 Communication。Memory 页面要求 `memory_record.csv` 与
`operator_memory.csv` 同时存在；框架通过 `profile_memory=True` 生成它们。

`overview` 是 Ascend PyTorch Profiler 的轻量策略，不保证 Memory 页面完整。
`--preset all` 会真正展开
多套框架内采集策略，必须评估运行次数和存储。显式 `msprof` 不使用这些 preset 的
level/shape/stack 配置，也不允许 `--preset all`。

默认命令使用官方 `--type=text`。26.1 文档下该模式提供 JSON/CSV 和 DB，适合
可读表格、cluster 和 Insight，但容量高于 db-only。按所装 CANN 的官方参数
可增加版本相关选项：

```bash
python tests/glm5_2_mindstudio/performance_benchmark.py \
  --capture --device npu --collector msprof --topology single \
  --collector-arg=--type=db
```

`--collector-arg=...` 原样放在 application 前，并进入实验 identity；`--output`、
`--application`、`--dynamic` 由生命周期管理，不能覆盖。其他如 task-time、runtime-api、
storage-limit 只有在当前 CANN 官方文档确认后才传，框架不硬编码易变参数。

Ascend PyTorch Profiler 深度采集：

```bash
python tests/glm5_2_mindstudio/performance_benchmark.py \
  --capture --device npu --collector torch_npu_profiler \
  --topology fsdp8 --preset distributed
```

只有这个入口使用 `--profiler-level`、`--profile-ranks`、`--record-shapes`、
`--profile-memory`、`--with-stack`、`--aic-metrics`。每个参数的采集层级、开销和
对应 Insight 页面见本文第 2 节和 [官方文档矩阵](../toolchain/OFFICIAL_DOCUMENTATION_MATRIX_ZH.md)。

当前 MindStudio performance 只实现 NPU。`--device cuda` 保留接口并明确报未实现；
不能让 GPU 静默落入另一套未确认的采集语义。

GPU 标杆由独立的 `tests/glm5_2_nvidia` 采集。官方 `calibrate_npu_gpu` 读取 Nsys
SQLite 和 Ascend PyTorch Profiler DB，并依赖 NVTX/MSTX Module 标记；GPU 内网
部署和数据汇合命令见
[GPU 采集与内网离线比较环境](../toolchain/GPU_COLLECTION_AND_OFFLINE_ANALYSIS_ZH.md)。

## 6. 高级分阶段分析命令

分析已存在 capture，不重跑训练。该入口用于采集与分析环境分离、补跑工具或改变
某一个工具的参数。`offline_parse`、`advisor`、`cluster`、`compare` 分别写入独立
状态文件；新增一个阶段不会覆盖其他阶段，`--force` 只替换本次请求的阶段。先按
官方输入矩阵选择 recipe：

```bash
# msProf：多 rank cluster；单卡直接生成 Insight handoff
python tests/glm5_2_mindstudio/performance_benchmark.py \
  --analyze --device npu --collector msprof \
  --topology fsdp8 --preset overview \
  --cluster --cluster-mode all

# Ascend PyTorch Profiler：advisor
python tests/glm5_2_mindstudio/performance_benchmark.py \
  --analyze --device npu --collector torch_npu_profiler \
  --topology single --preset runtime --advisor

# Ascend PyTorch Profiler：GPU-NPU 或 NPU-NPU compare
python tests/glm5_2_mindstudio/performance_benchmark.py \
  --analyze --device npu --collector torch_npu_profiler \
  --topology single --preset runtime \
  --compare-baseline /path/to/baseline/profile
```

官方命令形态：

```bash
msprof-analyze advisor all -d PROFILE -o OUTPUT/advisor
msprof-analyze cluster -m all -d PROFILE -o OUTPUT/cluster
msprof-analyze compare -d PROFILE -bp BASELINE --output_path OUTPUT/compare
```

- advisor 生成终端建议、HTML 和 XLSX，先看 High，再回到原始证据验证；
- cluster 生成 `cluster_analysis_output`；框架将它同步到包含全部 rank profile 的
  profiler 根目录。导入该统一根目录可关联五个系统调优页面；单独导入 cluster
  目录只适合 Summary/Communication 聚合视图；
- compare 把训练耗时拆为算子/通信/调度，并比较算子耗时、通信和内存；XLSX 的
  差异是候选根因，不是自动 PASS/FAIL。

MindStudio 标准入口默认追加 `--cluster-recipes necessary`，对同一 DB 执行细粒度
拆解、通信/慢 Rank/慢链路、Host 下发和空闲原因分析；EP 拓扑还会分析专家负载。
指定 `--cluster-summary-baseline` 时会先拆解两份 profile，再做集群指标 A/B。
完整命令、输入约束、字段和全部官方特性边界见
[进阶分析操作指南](MSPROF_ANALYZE_ADVANCED_ZH.md)。

官方 26.1 的能力边界必须保留：advisor 只读取 Ascend PyTorch Profiler
`*_ascend_pt` 或 MindSpore `*_ascend_ms`；compare 的 NPU 端同样要求 Ascend
PyTorch Profiler；cluster 才支持 msProf db、Ascend PyTorch Profiler text/db、
MindSpore Profiler text/db 和 msMonitor db。框架会在 `--force` 清理前拒绝不兼容
组合，不会把 `PROF_*` 伪装成 `*_ascend_pt`。

cluster 还支持官方 `communication_time`、`communication_matrix` 和 `--agent`：

```bash
python tests/glm5_2_mindstudio/performance_benchmark.py \
  --analyze --device npu --collector msprof \
  --topology fsdp8 --preset overview \
  --cluster --cluster-mode communication_time \
  --cluster-agent-output
```

benchmark 的 `--force` 负责实验 generation；只有确认属主、权限和超大输入都可信时，
才使用 `--cluster-bypass-input-safety-checks` 传递官方 cluster `--force`。完整交付件、
字段和诊断顺序见 [集群分析操作与判读](MSPROF_ANALYZE_CLUSTER_ZH.md)。

## 7. MindStudio Insight 阅读路径

`analyze` 生成 `mindstudio_insight_handoff.json`。新 capture 只有一个首选 import
root：完整 profiler 根目录，其中同时包含全部 `*_ascend_pt` rank 目录和
`cluster_analysis_output`。服务器无 GUI 时把该目录整体同步到 Windows/macOS，
然后在 Insight 中选择目录导入。多卡不能只同步 rank 0。历史 capture 若尚未生成
同目录交付，handoff 才会兼容性地列出 profile 与 cluster 两个目标。

按现象阅读：

1. **Summary**：先看计算、通信、空闲/Bubble 占比和 rank/stage 差异；
2. **Communication**：看通信域、collective、payload、等待、带宽、链路矩阵、
   non-overlapped communication；
3. **Timeline**：沿 Python -> CANN -> Runtime -> Stream -> Kernel/Collective 查看
   Host 下发、Device 执行和 rank 到达 collective 的时间；
4. **Operator**：按类型/名称/shape 看 count、总/平均耗时、利用率和 fallback；
5. **Memory**：区分 active、reserved、峰值、碎片与反复申请释放；
6. **RL**：只有 Verl/MindSpeed 等受支持框架数据与 MSTX 控制流打点时，才展示
   rollout/inference/reward/train 流水，普通 GLM 预训练不会自动产生 RL 视图。

Timeline 是时间顺序证据；火焰图是按调用栈聚合的耗时证据，二者互补。

## 8. 输出、同步与生命周期

### 8.1 Profiler-off 重复实验和跨平台对比

性能数值不从 profiler-active Timeline 直接下结论。NPU 与 GPU 都先用相同模型、
拓扑、batch、sequence、seed 和 dtype 跑至少三次 profiler-off；每次用不同
`--replicate` 保留独立 generation。共享聚合器读取每个 run 根目录的
`experiment.json` 和 `metrics.jsonl`，跳过暖机 steps 后计算 step time 的
median/p90/p95、吞吐 median/mean、TFLOPS、MFU、峰值显存，以及重复运行间 CV。

聚合器会先校验实验契约。任何模型、拓扑、batch、sequence、seed 或精度配置不一致，
都会拒绝生成平台差值；它不会把不同实验包装成可比结果。只有 reference 时生成单平台
稳定性报告，同时配置 candidate 才生成 before/after、版本或 NPU/GPU 对比：

```bash
python -m tests.glm5_2_performance.comparison \
  --reference-label GPU \
  --reference-run /path/to/gpu-r1 \
  --reference-run /path/to/gpu-r2 \
  --reference-run /path/to/gpu-r3 \
  --candidate-label NPU \
  --candidate-run /path/to/npu-r1 \
  --candidate-run /path/to/npu-r2 \
  --candidate-run /path/to/npu-r3 \
  --skip-steps 10 \
  --output performance_reports/comparisons/gpu-npu-fsdp8
```

输出包含自包含 `comparison.html`、机器可读 `comparison.json` 和入口
`README.md`。报告明确显示每组 repeat 数和 CV；不足三次只标记证据不足，不伪造
PASS/FAIL。这里借鉴 MLPerf 的申报与重复测量纪律，但 GLM 本地实验不是 MLPerf
benchmark，也不得称为 MLPerf 结果。相同输入会安全复用已有报告；输入变化默认拒绝
覆盖，只有显式 `--force` 才替换所选 comparison 目录，不会删除任何原始 capture。

项目自有性能 HTML 使用与精度实验相同的 Panel + pyecharts/ECharts 离线栈：单端
profiler-off 报告展示暖机/稳态、逐 Step 耗时、吞吐、TFLOPS、MFU、显存和诊断分支；
profiler-active 报告保留同一训练视图，同时索引 Insight/Timeline、数据库和官方统计；
重复实验报告叠加每次运行曲线，展示 median/p90/p95、CV 与候选相对基准变化。
外部工具的原生 Timeline、数据库和工作簿不被重新包装成“官方结论”，只作为可追溯入口。

```text
mindstudio_runs/performance/system/<card-scope>/<topology>/<run>/
  runtime.log / run_state.json
  trainer_output/profiling/msprof/       # msProf
  trainer_output/profiling/traces/       # torch_npu.profiler，也是 Insight 唯一导入根
    rank_0_*_ascend_pt/ ...              # 每个 rank 的原始/解析数据
    cluster_analysis_output/             # DB + CSV/JSON 集群聚合交付件
  advisor*/ cluster*/ compare*/
  cluster/advanced/                    # 官方进阶 recipe 与逐 Rank 结果

mindstudio_artifacts/performance/system/<card-scope>/<topology>/<run>/
  manifest.json / metrics.jsonl / analysis.json
  analysis_offline_parse_state.json
  analysis_advisor_state.json
  analysis_cluster_state.json
  analysis_compare_state.json
  analysis_state.json                 # 汇总报告生成状态
  mindstudio_insight_handoff.json

mindstudio_reports/performance/system/<card-scope>/<topology>/<run>.html

performance_reports/comparisons/<comparison-name>/
  README.md
  comparison.json
  comparison.html
```

采集 manifest 只记录影响采集语义的身份。msProf 路径记录 collector 参数、lock、
resolved source、msProf、CANN、torch/torch_npu；Ascend PyTorch Profiler 路径记录
torch、torch_npu、CANN，以及 TorchTitan profiler、TorchTitanTurbo NPU adapter 和
本仓 capture adapter 的源码哈希。它们都不要求采集服务器安装 msprof-analyze。
离线分析在自己的 operation/analysis manifest 中记录 msprof-analyze 版本/源码，
以及实际用于渲染的 Insight/FlameGraph 脚本哈希。采集栈升级需要新 capture；只升级
analyzer、可视化脚本或分析选项时，重复原命令会自动清除并重建身份失配的单个派生
阶段，不会重跑 capture，也不会把旧派生结果与新工具混在一起。
`--compare-baseline` 对目录递归记录相对文件名、大小和纳秒 mtime 的树摘要；基准树
内部 DB/JSON/XLSX 被改写后，不会错误复用旧 compare。

旧版 Ascend PyTorch Profiler manifest 若没有 `collector_toolchain`，仍可作为历史 raw
证据离线阅读和重新生成报告，但不能被新 capture 命令当作同身份结果静默跳过；若要
形成正式的新 capture，应显式 `--force` 或使用新的实验 identity。

- `--force` 在整个选中 suite 开始前清除旧 generation；
- 不加 force 只跳过完整且 identity/toolchain 一致的成员，重试未完成成员；
- 活跃 PID 阻止删除；每个子进程记录 exact command 和 log；
- raw profile 可含路径、算子名和 shape，公开前必须审查；
- Release `analysis` 用于经审查的 DB/XLSX/HTML/JSON，原始大数据按需 `full`。

`mindstudio_insight_handoff.json` 同时保存唯一导入根的服务器绝对路径和相对仓库根的 portable
路径。Release `analysis` 保留经审查的 `msprof_*.db`、解析表、Timeline、XLSX/HTML
和报告，但不保留原始 device payload；如果 Insight 的某个视图要求完整原始 profile，
使用经安全审查的 `full` archive。handoff 中每个 Insight 视图还会给出
`ready`、`inspect_database` 或 `missing` 证据状态，避免把 GUI 菜单存在误写成数据已采集。

## 9. 三仓边界与服务器验证

- TorchTitan：不改；负责模型、Trainer、并行语义；
- TorchTitanTurbo：不在本分支改；现有 adapter 提供 torch_npu profiler；
- torchtitan-test：选择 topology/collector，记录 provenance，调用官方 analyzer，
  生成 Insight handoff。

CPU 单测只验证命令、边界、命名与索引。正式运行前必须验证：

1. `which msprof && msprof --help` 与 CANN 匹配；
2. 单卡 `msprof --type=text` 包装 torchrun 并生成 DB/JSON/CSV；
3. 多 rank 子进程都进入采集，Insight 能看到所有 rank；
4. msProf db 能运行 cluster；Ascend PyTorch Profiler `*_ascend_pt` 能运行
   advisor/compare；
5. Insight 版本兼容当前 CANN；
6. profiler-off 与 profiler-active 分开；
7. 深度采集前估算磁盘并限制 rank/window。

未完成这些验证，只能称官方工作流和命令契约已实现，不能称 NPU 性能实验已通过。
