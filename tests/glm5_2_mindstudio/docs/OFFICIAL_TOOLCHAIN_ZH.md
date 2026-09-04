# 昇腾 MindStudio 官方工具链全景与本项目选型

> 状态：工具角色与选型说明。
>
> MindStudio、CANN 和开源工具的版本变化较快。本文给出稳定的职责边界，不把某个版本的内部 CLI 参数当成永久接口。实际使用前必须阅读所选版本文档、运行 doctor，并在实验 manifest 中记录版本和源码 commit。

## 1. 先理解“工具链”而不是“一个工具”

MindStudio 不是单一 profiler。官方生态覆盖训练、推理和算子开发，每条链路又包含迁移、精度、性能、调试和可视化工具。

```text
模型/训练脚本
  ├─ 迁移：msTransplant
  ├─ 精度：msProbe
  ├─ 性能采集：Ascend PyTorch Profiler / msprof
  ├─ 性能分析：msprof-analyze
  └─ 可视化：MindStudio Insight

推理系统
  ├─ 模型转换/编译：ATC
  ├─ 推理调优：AOE、msIT 相关工具
  ├─ 压缩：msModelSlim 等
  ├─ 精度：msProbe（具体框架覆盖以版本为准）
  ├─ Serving 性能：msServiceProfiler 等
  └─ 可视化：MindStudio Insight

算子/Kernel
  ├─ 设计与性能上界：msKPP
  ├─ 工程生成：msOpGen
  ├─ 快速调用验证：msKL
  ├─ 内存/竞争检测：msSanitizer
  ├─ 原生调试：msDebug
  ├─ 性能采集分析：msOpProf
  └─ 可视化：MindStudio Insight
```

本项目当前主要是 TorchTitan/GLM5 **训练迁移**。推理工具和算子工具需要理解并保留扩展入口，但不能把“工具链存在”写成“本仓库已实现推理部署或算子交付”。

官方总入口：

- [MindStudio 文档中心](https://www.hiascend.com/document/detail/zh/mindstudio/latest/index/index.html)
- [MindStudio 产品页](https://www.hiascend.com/developer/software/mindstudio)
- [MindStudio Training Tools（msTT）](https://gitcode.com/Ascend/mstt)
- [MindStudio Inference Tools（msIT）](https://gitcode.com/Ascend/msit)
- [MindStudio Operator Tools（msOT）](https://gitcode.com/Ascend/msot)

## 2. 训练开发工具链

### 2.1 msTransplant：迁移分析，不是精度比较器

`msTransplant` 用于分析和辅助把 PyTorch 训练程序迁移到昇腾，例如识别 CUDA/NCCL 相关接口并给出迁移建议。它回答的是“程序怎样迁移和跑起来”，不是“迁移后数值是否一致”。

对于三仓项目：

- TorchTitan 本身已经把大部分设备无关逻辑放在通用仓库；
- NPU patch、图模式和 profiler adapter 位于 TorchTitanTurbo；
- 因此不能机械地把 msTransplant 输出直接应用到 TorchTitan 上；
- 可以把它作为新上游版本的 API 差异审计工具，但任何修改仍要遵守三仓边界。

入口：[msTT 训练工具链](https://gitcode.com/Ascend/mstt)

### 2.2 msProbe：精度采集、监控、比较和定位

`msProbe` 是本次官方标准精度流程的核心。其价值不是再画一条 loss 曲线，而是把精度问题从端到端指标下钻到模块/API/算子数据。

典型流程：

1. **配置核对**：检查两端影响精度的环境与训练配置；
2. **训练状态监控**：观察溢出、NaN/Inf 等计算异常；
3. **精度数据采集**：在 forward/backward 的模块或 API 级采集输入输出、统计量或 tensor；
4. **精度预检**：把 API 数据交给 `acc_check`/`multi_acc_check`，先做单 API 风险筛查；
5. **离线比较**：比较 GPU/NPU，或同一 NPU 的 eager/compile；
6. **分级图可视化**：用 L0/mix 的 construct 数据生成 `.vis.db`，在 TensorBoard
   Ascend Graph 插件中按层级查看两端结构/统计；
7. **缩小范围重采**：从异常层继续定位到更细 API，而不是永久保留全模型全量 dump。

应当区分三种比较：

| 比较 | reference | candidate | 主要回答 |
| --- | --- | --- | --- |
| GPU/NPU 模块/API | GPU eager | NPU eager | 迁移后哪个模块/API 首先偏离 |
| NPU 编译精度 | NPU eager | NPU Inductor/NPUGraph | 图编译是否改变数值行为 |
| 同设备自检 | 同设备重复或等价路径 | 同设备另一运行 | 工具配置、确定性和采集链路是否正确 |

本项目对 msProbe 的使用原则：

- 官方 dump 与 compare 文件原样保存；
- 项目 adapter 只生成配置、调用官方入口、校验产物和建立索引；
- 不把任何非 msProbe 产物伪装成官方模块/API 结果；
- 不硬编码未经当前源码确认的内部 Python API；
- 默认限制 step、rank、模块和 dump 类型，防止存储爆炸；
- 与固定 token plan、seed checkpoint、状态监测和多 step 训练现象共同构成证据链。

官方入口：

- [msProbe 源码](https://gitcode.com/Ascend/msprobe)
- [msProbe PyTorch 快速入门](https://gitcode.com/Ascend/msprobe/blob/master/docs/en/quick_start/pytorch_quick_start.md)
- [PyTorch 精度预检指导](https://www.hiascend.com/document/detail/zh/mindstudio/latest/msTT_msIT/msProbe/docs/zh/user_guide/accuracy_checker/pytorch_accuracy_checker_instruct.md)
- [PyTorch 精度比对指导](https://www.hiascend.com/document/detail/zh/mindstudio/latest/msTT_msIT/msProbe/docs/zh/user_guide/accuracy_compare/pytorch_accuracy_compare_instruct.md)
- [PyTorch 编译精度比对指导](https://www.hiascend.com/document/detail/zh/mindstudio/latest/msTT_msIT/msProbe/docs/zh/user_guide/accuracy_compare/pytorch_compile_accuracy_compare_instruct.md)
- [Monitor V2 指导](https://mindstudio-docs-master.readthedocs.io/zh-cn/latest/msprobe/docs/zh/user_guide/monitor_v2_instruct/)
- [PyTorch 快速入门与 graph_visualize](https://github.com/Ascend/msprobe/blob/master/docs/en/quick_start/pytorch_quick_start.md)

### 2.3 Ascend PyTorch Profiler：训练代码内的深度采集入口

`torch_npu.profiler` 面向 PyTorch 训练/在线推理，在训练脚本内通过 schedule 控制 warmup、active 和 repeat 窗口。它可以关联多个层级：

```text
Python/Torch 模块与算子
  → CANN runtime / task launch
  → NPU kernel / stream
  → 内存、通信和可选系统数据
```

它的底层依赖 CANN profiling 能力，并不是绕过 CANN 重新实现硬件采集。本项目
在需要 TorchTitan step 生命周期、rank 选择、profiler schedule、module、shape、
stack 或 memory 语义时使用它。它是 MindStudio PyTorch 训练标准入口的默认
collector；msProf 作为底层或黑盒入口保留，两者不是互斥替代关系。

当前三仓对应关系：

- TorchTitan：提供 profiler 生命周期调用；
- TorchTitanTurbo：构造 `torch_npu.profiler.profile`；
- torchtitan-test：选择 preset、rank、schedule、输出目录并做离线分析。

官方入口：

- [Ascend PyTorch Profiler 文档](https://mindstudio-profiler-docs.readthedocs.io/zh-cn/latest/torch_npu_profiler/)
- [torch_npu profiler 源码](https://github.com/Ascend/pytorch/tree/master/torch_npu/profiler)
- [Ascend PyTorch 仓库](https://gitcode.com/Ascend/pytorch)

### 2.4 msprof：CANN 侧基础采集/导出工具

`msprof` 是 CANN/MindStudio profiler 的基础命令行工具，可在不能方便改训练代码的黑盒场景采集，也可解析/导出底层 profiling 数据。它和 `torch_npu.profiler` 都能触达 CANN profiling 能力，但入口层级不同：

| 项目 | `torch_npu.profiler` | `msprof` |
| --- | --- | --- |
| 入口 | PyTorch 代码内 | 命令行包裹进程或处理 profile 数据 |
| step 窗口 | 容易和训练 step 精确对齐 | 更适合黑盒/底层排障，窗口控制方式不同 |
| PyTorch 语义 | 天然关联框架算子 | 依采集和导出配置而定 |
| 本项目定位 | PyTorch 训练默认采集入口 | 底层、黑盒或跨框架通用采集 |

不能说“msprof 基于 torch_npu”。更准确的关系是：两者都是 profiling 入口，
`torch_npu.profiler` 通过 torch_npu/C 扩展接入 CANN，`msprof` 则是
CANN/MindStudio 的命令行工具。当前 GLM PyTorch 标准流程默认
`torch_npu_profiler`；需要命令行包裹、无法修改训练程序或底层黑盒排障时显式
切换 `msprof`。具体命令和选择边界见
[PERFORMANCE_WORKFLOW_ZH.md](PERFORMANCE_WORKFLOW_ZH.md)。

“msProf”有时也作为产品/组件品牌出现；脚本自动化中应以实际可执行文件 `msprof`、版本输出和所选官方文档为准，不能只靠大小写推断接口。

官方入口：[MindStudio Profiling 工具文档](https://mindstudio-profiler-docs.readthedocs.io/zh-cn/latest/)

### 2.5 msprof-analyze：只分析已有数据

`msprof-analyze` 不启动模型、不负责主要采集。它读取已采集和解析的 profiling 数据，提供：

- advisor 规则诊断；
- 算子、调度和集群通信分析；
- 多 rank/集群对比；
- HTML、Excel/CSV/JSON 等版本相关输出。

它不是“任意 collector 的通用解析器”。26.1 官方输入契约是按 recipe 区分的：

- `cluster` 支持 msProf db、Ascend PyTorch Profiler text/db、MindSpore Profiler
  text/db 和 msMonitor db；
- `advisor` 的训练输入是 Ascend PyTorch Profiler `*_ascend_pt` 或 MindSpore
  `*_ascend_ms`；
- `compare` 的 NPU 输入要求 Ascend PyTorch Profiler/MindSpore 规定格式，GPU 输入
  使用 PyTorch Profiler trace。

因此默认 Ascend PyTorch Profiler capture 可按官方输入契约进入 advisor、cluster、
compare 和 Insight；显式 msProf capture 主要进入 cluster 和 Insight。工具同名不
表示输入格式相同。

这解释了为什么不应把它与 msprof 合并：

- 采集必须在目标硬件和目标作业运行时发生；
- 分析可以离线、多次、按不同 recipe 重跑；
- 两者的依赖、耗时、产物生命周期和权限要求不同。

本项目现有 performance workflow 已调用相关分析能力。MindStudio 标准工作流只索引和审计，不复制 analyzer。

官方入口：

- [msprof-analyze 源码](https://gitcode.com/Ascend/msprof-analyze)
- [msTT 中的 msprof-analyze 文档/源码目录](https://gitcode.com/Ascend/mstt/tree/master/profiler/msprof_analyze)

### 2.6 MindStudio Insight：交互式可视化，不采集训练

MindStudio Insight 读取符合格式的 profiling 数据，提供交互式视图，例如：

- Summary：计算、通信、空闲、资源利用和 rank/stage 差异；
- Communication：通信域、collective、payload、等待、带宽和链路关系；
- Timeline：Host、runtime、stream、kernel、通信事件的时间关系；
- Operator：算子类型、名称、shape、次数、总/平均耗时和相关硬件指标；
- Memory：分配、释放、活跃量、峰值、碎片和生命周期；
- RL：受支持 RL 框架和 MSTX 数据中的 rollout、推理、reward 与训练流水；
- 算子指令流水和源代码关联（依输入数据类型与版本能力）；
- serving 请求端到端 timeline（推理场景）。

Insight 不替代采集器，也不应被当成 CI 判定器。无 GUI 的 NPU 服务器负责生成可导入数据；Windows/桌面端安装与 CANN 数据格式兼容的 Insight 版本进行交互分析。

不同视图依赖不同采集数据。普通 GLM 预训练不会因为导入 msProf 数据就自动出现
RL 流水；单算子源码热点也要求 msOpProf 对应产物。项目生成的 handoff 会列出实际
存在的文件和适用视图，不能把 UI 中存在的页签写成本次实验已经采到的证据。

官方入口：

- [MindStudio Insight 源码](https://gitcode.com/Ascend/msinsight)
- [系统调优说明](https://gitcode.com/Ascend/msinsight/blob/master/docs/en/user_guide/system_tuning.md)
- [算子调优说明](https://gitcode.com/Ascend/msinsight/blob/master/docs/en/user_guide/operator_tuning.md)

完整的官方案例复现顺序见
[OFFICIAL_PRACTICE_ROADMAP_ZH.md](OFFICIAL_PRACTICE_ROADMAP_ZH.md)。

### 2.7 msOpProf：算子/Kernel 专项性能采集

`msOpProf` 面向已经缩小到少量算子或 Kernel 的深度调优。它不负责回答整次
TorchTitan 训练的吞吐、通信占比或 rank 长尾，而是在系统调优已经定位热点之后，
继续采集该 Kernel 的流水、内存访问、Cache、资源冲突、Occupancy、Roofline 和
可选源码映射等指标。其原生 `OPPROF_*` 结果可作为独立的算子调优数据导入
MindStudio Insight，不能用系统调优目录冒充。

本项目提供独立入口：

```bash
python tests/glm5_2_mindstudio/operator_tuning_benchmark.py \
  --kernel-name 'TopK*' \
  --metrics Default \
  -- python /abs/path/to/glm5_dsa_topk_probe.py
```

具体的 onboard/simulator 选择、Kernel 过滤、采集次数、输出目录、Insight 导入和
指标阅读顺序见 [OPERATOR_TUNING_WORKFLOW_ZH.md](OPERATOR_TUNING_WORKFLOW_ZH.md)。
不同芯片和版本支持的扩展指标并不完全相同，正式实验必须先运行 toolchain doctor，
并以当前版本 `msopprof --help` 和官方文档为准。

官方入口：

- [msOpProf 快速入门](https://www.hiascend.com/document/detail/zh/mindstudio/latest/msOT/Operatordevelopmenttools/docs/zh/quick_start/msopprof_quick_start.md)
- [MindStudio Insight 算子调优快速入门](https://www.hiascend.com/document/detail/zh/mindstudio/latest/GUI_baseddevelopmenttool/MindStudioInsight/docs/zh/quick_start/operator_tuning_quick_start.md)
- [MindStudio Insight 算子调优](https://www.hiascend.com/document/detail/zh/mindstudio/latest/GUI_baseddevelopmenttool/MindStudioInsight/docs/zh/user_guide/operator_tuning.md)
- [msOpProf 源码](https://gitcode.com/Ascend/msopprof)

### 2.8 msTX：给性能时间线添加业务语义

msTX 是 instrumentation 标记能力。它让 timeline 不只显示 kernel 名称，还能显示自定义训练阶段、layer、microbatch 或 collective sequence 等 range/mark。

本项目可用它标记：

- data loading；
- forward/backward/optimizer；
- pipeline microbatch；
- MoE dispatch/combine；
- DSA indexer 与 sparse attention；
- checkpoint save/load。

标记本身不优化性能，也不替代 profiler；只有在采集配置启用对应能力时才会进入结果。

入口：[msTX 源码与说明](https://gitcode.com/Ascend/mstx)

### 2.9 msMemScope：内存专项采集与分析

msMemScope 面向内存事件、泄漏、低效内存、内存块监测和生命周期分解。它提供
Python `config/start/stop/step` 接口，也提供包装应用的命令行模式。它回答的是
“内存为什么没有释放、为何反复申请、哪些块低效”，不替代整网 step time、通信和
算子性能采集。

当前项目通过 `memory_tuning_benchmark.py` 提供独立入口，复用 common 拓扑和训练
参数，原样保存官方 DB/CSV，并支持两份采集的官方 step 对比。它不复用系统 profiler
数据。CANN 版本、LD_PRELOAD、多 rank 子进程和具体分析项仍必须在目标服务器 doctor
及最小 single capture 中验证，未验证不能写成 PASS。

官方入口：

- [msMemScope 快速入门](https://www.hiascend.com/document/detail/zh/mindstudio/latest/msTT_msIT/msMemScope/docs/zh/quick_start/quick_start.md)
- [msMemScope 源码](https://gitcode.com/Ascend/msmemscope)
- [本项目内存调优流程](MEMORY_TUNING_WORKFLOW_ZH.md)

## 3. 图编译工具链

### 3.1 PyTorch/NPU 图编译阶段

当前 PyTorch 主流编译路径可以概括为：

```text
Python eager 程序
  → TorchDynamo 捕获
  → FX Graph
  → AOTAutograd/AOTDispatcher 拆分 forward/backward
  → backend
       ├─ Inductor → 调度/融合/代码生成 → Triton-Ascend/CANN
       └─ NPUGraph backend → NPU 图捕获/回放路径
  → NPU 执行
```

不同问题必须用不同工具定位：

| 问题 | 证据 | 工具 |
| --- | --- | --- |
| Dynamo 没捕获某段 Python | graph break、explain | `TORCH_LOGS`、`torch._dynamo.explain`、tlparse |
| shape/guard 变化重复编译 | guard、recompile | `TORCH_LOGS=recompiles,dynamic`、tlparse |
| FX 已捕获但后端无 lowering | backend error/fallback | compiler debug dump、TorchNPU/Inductor 日志 |
| 编译前后数值不同 | 模块/API/输出误差 | msProbe compile accuracy + 多 step precision |
| 编译后反而更慢 | step/kernel/launch/memory | torch_npu.profiler + msprof-analyze + Insight |

### 3.2 tlparse、TensorBoard 与 Insight 的关系

- **tlparse**：解析 `TORCH_TRACE`，关注 Dynamo/Inductor 编译事件、graph、guard、recompile；
- **TensorBoard**：可以查看某些 profiler trace 或训练标量，取决于 exporter/plugin；
- **MindStudio Insight**：官方 NPU 系统/算子/通信交互式可视化；
- **msProbe compile accuracy**：比较 eager 与 compile 精度。

它们不是互相替代的四套 profiler，而是回答不同问题。

官方/上游入口：

- [TorchNPU torch.compile 文档](https://www.hiascend.com/document/detail/zh/Pytorch/latest/userguide/torchcompile/docs/zh/torch_compile/pytorch_compilation_mode.md)
- [PyTorch torch.compile](https://docs.pytorch.org/docs/stable/generated/torch.compile.html)
- [tlparse](https://github.com/meta-pytorch/tlparse)
- [Triton-Ascend](https://github.com/triton-lang/triton-ascend)

## 4. 推理开发工具链

推理工具链在本项目中是知识与后续扩展范围，当前训练实验不能冒充推理验收。

常见角色包括：

| 工具/能力 | 作用 | 与当前 GLM5 训练工作的关系 |
| --- | --- | --- |
| ATC | 模型转换/编译到昇腾离线模型等目标形态 | 未来部署阶段；当前 TorchTitan 训练不直接使用 |
| AOE | 图/算子调优搜索 | 未来推理或特定图调优；不能替代训练 profiler |
| msModelSlim | 量化、压缩等模型瘦身能力 | 未来推理压缩；需单独精度与性能验收 |
| msIT | 推理开发、benchmark、分析、转换、profile 等统一工具 | 未来推理链路入口；具体组件随版本变化 |
| msServiceProfiler | serving 请求端到端性能采集/分析 | 在线服务；与训练 step profiler 不同 |
| MindStudio Insight | serving timeline 可视化 | 读取对应 serving 数据 |

官方入口：

- [MindStudio Inference Tools 源码](https://gitcode.com/Ascend/msit)
- [msIT 安装说明](https://gitcode.com/Ascend/msit/blob/master/docs/en/msit_install_guide.md)
- [MindStudio Insight Serving 调优](https://gitcode.com/Ascend/msinsight/blob/master/docs/en/user_guide/service_optimization.md)

需要注意：官方工具会日落或迁移组件。例如 msIT 文档已提示部分旧 debug/LLM 能力迁移到 msProbe。设计 adapter 时必须绑定明确版本，不能永远解析 `master` 的目录结构。

## 5. 算子开发工具链

### 5.1 完整闭环

```text
性能证据证明瓶颈在某个算子
  → msKPP 估算/分析性能上界与方案
  → msOpGen 生成算子工程
  → 编写 Ascend C / Triton-Ascend / 其他受支持实现
  → msKL 快速调用做功能验证
  → msSanitizer 检测越界、竞争、未初始化、同步问题
  → msDebug 上板单步/变量检查
  → msOpProf 采集算子性能
  → MindStudio Insight 查看流水、源代码和硬件利用
  → 回到模型做精度、性能、稳定性 A/B
```

### 5.2 工具角色

| 工具 | 主要问题 | 不应被误用为 |
| --- | --- | --- |
| msKPP | 设计阶段的性能预测/上界分析 | 真实训练 benchmark |
| msOpGen | 生成规范的算子工程骨架 | 自动生成正确高性能算法 |
| msKL | 快速启动 kernel 做功能验证 | 完整模型集成测试 |
| msSanitizer | 内存、竞争、未初始化和同步检测 | 数值精度比较 |
| msDebug | NPU 原生调试、单步和变量观察 | profiler |
| msOpProf | 算子级性能采集与分析 | 模型端到端吞吐结论 |
| Insight | 指令流水、算子源码和运行负载可视化 | 数据采集器 |

对于 GLM5，只有 profiler 先证明瓶颈在 DSA top-k gather/scatter、MoE grouped GEMM、RoPE/norm 或其他算子，才进入这一层。不能因为“融合听起来会更快”就绕过证据直接替换参考实现。

官方入口：

- [MindStudio Operator Tools 源码](https://gitcode.com/Ascend/msot)
- [msOT 工具全景](https://gitcode.com/Ascend/msot/blob/master/README.md)
- [算子开发工具链快速入门](https://gitcode.com/Ascend/msot/blob/master/docs/zh/quick_start/op_tool_quick_start.md)

## 6. GLM-5.2 官方工具选型

| 目标 | 默认工具 | 补充工具 | 当前状态边界 |
| --- | --- | --- | --- |
| GPU/NPU 训练问题现象 | loss/grad norm 与重复运行 | fixture/config manifest | 输入和训练契约已接入，服务器矩阵待验收 |
| GPU/NPU 训练前配置 | msProbe ConfigChecker | fixture/config manifest | dynamic 逐 rank capture/compare 已实现 |
| GPU/NPU 模块/API 精度 | msProbe PrecisionDebugger/compare | API 预检和 graph_visualize | capture/compare 编排已实现，服务器矩阵待验收 |
| GPU/NPU API 精度预检 | msProbe acc_check/api_precision_compare | L1 statistics/tensor dump | 逐 step/逐 rank 编排已实现，服务器矩阵待验收 |
| 训练状态筛查 | msProbe TrainerMonitorV2 | loss/grad norm | 逐 rank CSV 已接入；无官方 cross-device verdict，PP/分片 optimizer 待验收 |
| 模块层级交互比较 | msProbe graph_visualize + TensorBoard Ascend Graph | 官方 compare CSV | `.vis.db`、hash 索引和启动命令已接入，服务器待验收 |
| NPU eager/compile 精度 | msProbe PrecisionChecker single-pass | checker graph dump | 逐 rank capture/汇总已实现，backend/拓扑待验收 |
| NPU 训练性能采集 | msProf | `torch_npu.profiler` 深度归因 | 标准入口已接入；服务器矩阵待验收 |
| 离线性能诊断 | msprof-analyze | 自定义摘要 | advisor/cluster/compare 已编排 |
| 交互可视化 | MindStudio Insight | TensorBoard/Perfetto | 已生成整目录 handoff 与产物清单 |
| 编译调试 | PrecisionChecker graph dump 与 Torch compiler logs | FX/IR/code | 服务器待验收 |
| 算子优化 | Triton-Ascend/Ascend C + msOT | 自定义 op | 仅在 profiler 证据后进入 |
| 推理部署 | ATC/AOE/msIT/msModelSlim 等 | serving profiler | 当前项目未实现 |

## 7. 一条可讲清楚的标准排障路径

```text
1. 最小训练：模型在目标设备/拓扑可运行
2. 固定契约：相同 checkpoint、token plan、dtype、拓扑
3. 现象确认：loss / grad norm 判断异常 step 和问题类别
4. Monitor V2：长运行中筛出异常 step/rank/module
5. msProbe pre-check：对 L1 API 做 CPU 标杆单测并比较两端预检结论
6. msProbe compare：模块/API 原始数据比较，定位首个异常位置
7. graph_visualize：交互查看模块层级、结构与统计关系
8. 编译工具：若只在 compile 出现，检查 break/guard/lowering
9. profiler：若正确但慢，分解 Host/Device/通信/内存
10. msprof-analyze/Insight：离线诊断与交互验证
11. 算子工具：证据指向具体 kernel 后再设计、调试和优化
12. A/B：按相同官方契约重跑并验收
```

这一流程的核心不是“把所有工具都跑一遍”，而是在每一层回答一个明确问题，并把输入、版本和产物串成可复现证据链。
