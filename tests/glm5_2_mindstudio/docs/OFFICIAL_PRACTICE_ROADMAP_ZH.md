# MindStudio 官方实践学习与 GLM5 复现路线

本文把官方工具文档转换成一条可执行的 GLM5 AI Infra 学习路线。目标不是把所有
工具无差别跑一遍，而是针对一个清楚的现象，选择正确层级的工具，保留输入契约、
版本、命令、原始证据和结论，再回到训练做 A/B 验收。

官方总入口：

- [MindStudio 文档中心](https://www.hiascend.com/document/detail/zh/mindstudio/latest/index/index.html)
- [msProbe PyTorch 快速入门](https://www.hiascend.com/document/detail/zh/mindstudio/latest/msTT_msIT/msProbe/docs/zh/quick_start/pytorch_quick_start.md)
- [msProf 快速入门](https://www.hiascend.com/document/detail/zh/mindstudio/latest/msTT_msIT/msProf/docs/zh/quick_start/msprof_quick_start.md)
- [MindStudio Insight 系统调优快速入门](https://www.hiascend.com/document/detail/zh/mindstudio/latest/GUI_baseddevelopmenttool/MindStudioInsight/docs/zh/quick_start/system_tuning_quick_start.md)
- [大模型训练性能瓶颈定位案例](https://www.hiascend.com/document/detail/zh/mindstudio/latest/practicalcases/Largemodeltraining/MindStudio/latest/zh/cases/case_of_troubleshooting_performance_bottleneck_in_llm_training.md)
- [性能问题通用定位指南](https://www.hiascend.com/document/detail/zh/mindstudio/latest/practicalcases/GeneralPerformanceIssue/MindStudio/latest/zh/cases/general_performance_issue_troubleshooting_guide/overview.md)

## 1. 每次实验共同的零阶段

无论调查精度、性能还是图模式，都先完成同一组前置工作：

```text
锁定三仓源码和官方工具版本
  -> doctor 检查 Python/PyTorch/torch_npu/CANN/CLI
  -> smoke 验证目标设备和拓扑可运行
  -> 生成一次固定 token plan 和 seed checkpoint
  -> 记录 dtype、batch、sequence length、topology 和环境
  -> 再进入专项工具
```

必须避免的伪对照包括：两端 checkpoint 不同、token 顺序不同、global batch 不同、
训练精度不同、拓扑不同、工具版本不明，以及把一次旧 artifact 与一次新 capture
拼在同一报告里。

执行入口：

```bash
python -m tests.glm5_2_mindstudio.toolchain doctor --scope full
python tests/glm5_2_smoke/smoke_benchmark.py --help
```

## 2. 精度实践：从端到端异常下钻到首个异常模块

### 2.1 建立问题现象

在相同 checkpoint、token、超参和拓扑下记录多个 step 的 loss 与 grad norm，确认
问题属于首 step 差异、长稳漂移、尖刺、NaN/Inf 还是重复运行不确定。该现象决定要
采集哪个 step 和前向/反向方向，但不能单独定位具体模块。

### 2.2 训练前配置检查

使用 msProbe ConfigChecker 比较两端可能影响数值的配置。优先消除 seed、dtype、
deterministic、TF32/混合精度、通信规约精度、优化器和数据契约差异，再讨论算子误差。

```bash
python tests/glm5_2_mindstudio/configuration_check_benchmark.py --help
```

### 2.3 训练状态监测

使用 TrainerMonitorV2 观察长训练中的 NaN/Inf、溢出、梯度和模块统计异常。Monitor
用于筛出异常 step/rank/module，不负责给出 GPU/NPU 的最终迁移 PASS/FAIL。

```bash
python tests/glm5_2_mindstudio/training_monitor_benchmark.py --help
```

### 2.4 模块/API 采集与比较

先用 L0 statistics 做低成本模块级定位；发现第一个异常区域后，再缩小模块列表、step
和 rank，切换到 L1 或 tensor。不要一开始就对所有拓扑、所有 rank、所有 step 保存
真实 tensor。

```text
GPU reference capture
  + NPU candidate capture
  -> 同步两端 artifact
  -> 官方 msprobe compare
  -> 先看 Result / Err_Message
  -> 再看具体指标与首个异常位置
```

完整命令见 [ACCURACY_WORKFLOW_ZH.md](ACCURACY_WORKFLOW_ZH.md)。

### 2.5 API 精度预检

对于可疑 API，使用 L1 dump 后运行 `acc_check`。官方流程会重放 API，并相对 CPU
高精度标杆计算适合该 API 的判定；最后用 `api_precision_compare` 比较两端预检结果。
随机 statistics 适合筛查，真实 tensor 适合确认，二者不能混为同一精度等级。

### 2.6 分级图可视化

L0 或 mix capture 的 `construct.json` 可交给 `graph_visualize` 生成 `.vis.db`，再通过
TensorBoard Ascend Graph 插件查看模块层级、匹配关系和统计量。它回答“两个模型结构
与模块数据如何对应”，不替代端到端训练判定。

### 2.7 图编译精度

同一 NPU 上用 msProbe PrecisionChecker 比较 eager 与 compile。single-pass 使用同一
输入重放 eager 路径，适合减少两次独立训练带来的输入差异；其结果还要和目标训练
区间、graph break/recompile 证据共同解释。

```bash
python tests/glm5_2_mindstudio/compile_accuracy_benchmark.py --help
```

### 2.8 API 与算子下钻

标准流程定位到模块后，缩小 L1/tensor 采集范围，结合 stack 找到具体 API；再使用
API 预检、最小单算子复现、Profiler 或算子工具确认根因。对 router/indexer 等离散
节点，应同时保存输入 score、top-k 和 dispatch 映射，区分上游累计误差与当前算子
新增误差。

## 3. 性能实践：从稳定基线到系统、通信、算子和内存

### 3.1 先取得无采集开销的性能基线

关闭 profiler，固定训练契约并重复运行。用 median/p90 step time、tokens/s、peak HBM
和 rank min/median/max 建立基线。Profiler-active 的 step time 包含采集开销，不能直接
当真实性能数字。

### 3.2 默认 msProf 做系统级短采集

msProf 包裹一个短小训练作业，采集 CANN/NPU 系统数据。默认 text 会输出可读表格和
DB；只需要 Insight DB 时可明确选择 db。采集前估算空间，并限制作业时长、rank 和
`--storage-limit`。

```bash
python tests/glm5_2_mindstudio/performance_benchmark.py \
  --capture --device npu --collector msprof \
  --topology single --preset overview
```

全拓扑轻量 overview 可以做横向筛查；深度采集只选择代表性拓扑。不要把所有拓扑与
所有重型 preset 做笛卡尔积，否则 raw profiler 数据会达到数百 GiB。

### 3.3 msprof-analyze 按输入类型做可重复的离线诊断

对同一份 capture 可以多次运行：

- msProf db 可做 cluster：汇总多 rank 时间与通信，定位快慢卡和不均衡；
- Ascend PyTorch Profiler `*_ascend_pt` 可做 advisor：给出候选问题和建议，仍需
  回到原始证据确认；
- Ascend PyTorch Profiler `*_ascend_pt` 可做 compare：比较 GPU/NPU 或 NPU/NPU，
  把训练耗时拆成计算、通信、调度，并比较算子耗时与算子级内存。

分析阶段不重跑模型，因此应与 capture 使用不同的生命周期和输出目录。recipe
必须匹配官方输入矩阵；不能因为三个子命令都属于 msprof-analyze，就假设 msProf
`PROF_*` 能作为 advisor/compare 输入。

### 3.4 按现象进入 Insight 视图

Insight 导入的是完整 profile 根或 `cluster_analysis_output`，不是某个 rank 的单个
CSV。推荐按以下顺序：

1. Summary：确认计算、通信、空闲和 rank/stage 差异；
2. Communication：确认 collective、payload、带宽、等待及未被计算掩盖的通信；
3. Timeline：沿 Host 下发、Runtime、Stream、Kernel/Collective 的时间关系验证根因；
4. Operator：按类型、名称、shape、次数和总/平均耗时找热点或 fallback；
5. Memory：区分 active、reserved、峰值、碎片和反复申请释放；
6. RL：只有受支持 RL 框架并带有对应 MSTX 语义时才进入 rollout/reward/train 视图。

“某个 collective 持续很久”不自动等于网络慢。若多个 rank 到达 collective 的时间不同，
先到的 rank 会等待；应沿 Timeline 回看慢 rank 在通信前执行的 Host 或计算路径。

### 3.5 Ascend PyTorch Profiler 做框架级深度归因

当 msProf 已表明瓶颈落在特定 step/rank/阶段，而需要 `nn.Module -> ATen/ACL ->
NPU Kernel`、shape、stack、memory 或 schedule 窗口时，再切换 collector：

```bash
python tests/glm5_2_mindstudio/performance_benchmark.py \
  --capture --device npu --collector torch_npu_profiler \
  --topology fsdp8 --preset distributed
```

### 3.6 内存和算子专项下钻

- msMemScope：面向泄漏、低效内存、内存块生命周期与分解；当前登记为专项工具，完成
  目标 CANN 的独立服务器验证后再接入正式自动化；
- msOpProf：面向已经隔离出来的 Ascend C/Triton/框架单算子，可做上板或仿真分析；
- MindStudio Insight Operator：读取 msOpProf 数据，查看流水、热点源码和硬件指标。

这两层不能替代整网 profiler。只有整网证据指向 DSA gather/scatter、MoE grouped GEMM、
norm/RoPE 或其他具体热点，才构造隔离算子输入并下钻。

## 4. 图模式实践：区分捕获、后端、精度和性能

```text
eager smoke
  -> compile smoke
  -> fullgraph 暴露首个 graph break
  -> TORCH_LOGS / TORCH_TRACE / tlparse 看 break、guard、recompile
  -> FX 已捕获时检查 backend lowering/fallback
  -> msProbe eager/compile 精度
  -> profiler-off eager/graph 性能 A/B
  -> profiler/Insight 验证 launch、fusion、kernel 和 memory 是否改善
```

图断裂发生在 Dynamo 捕获阶段；FX 已生成但后端没有 lowering 是另一类问题；shape
变化引发 guard miss/recompile 也不必然是 graph break。必须先分类，再决定改 Python
控制流、自定义 op fake/meta、动态 shape 策略，还是后端 lowering。

图模式的标准调试入口、证据目录和判断顺序统一由本目录的编译精度流程维护；执行
时必须把 Dynamo 捕获证据、后端编译证据、运行时 Profiler 证据和数值证据分层保存，
避免把 graph break、recompile 与后端 fallback 混成同一类结论。

## 5. 后续扩展：算子与推理工具链

完成训练迁移闭环后，可以按同一证据方法扩展，而不是直接宣称项目已覆盖：

### 5.1 算子开发

```text
整网 profiler 定位热点
  -> msKPP 评估瓶颈和上界
  -> msOpGen 建工程
  -> Ascend C / Triton-Ascend 实现
  -> msKL 功能验证
  -> msSanitizer / msDebug 正确性调试
  -> msOpProf + Insight 性能下钻
  -> 回到 GLM5 做精度、性能、稳定性 A/B
```

### 5.2 推理部署

按部署目标选择 ATC/AOE、msIT、msModelSlim 和 msServiceProfiler。服务化 profiler
关注请求调度、prefill/decode、KV Cache、batching、TTFT/TPOT 和吞吐，与离线训练
step profiler 是不同实验；必须另建推理契约和报告。

## 6. 每个官方实践的交付清单

一次可复现、可讲述的实验至少包含：

- 问题现象与假设；
- 三仓 commit/dirty tree、官方工具 source commit 与安装产物身份；
- Python、torch、torch_npu、CANN、driver/firmware 信息；
- token plan、checkpoint、训练配置、拓扑和 generation ID；
- 完整命令、环境、runtime log 和退出状态；
- 官方 raw/DB/CSV/XLSX/HTML 或 `.vis.db` 的完整性索引；
- 中文报告：先解释指标，再写证据、推断、下一步；
- profiler-off 或数值基准 A/B；
- 优化/修复后的回归：精度、性能、checkpoint、稳定性和组合实验。

这套清单让“会运行工具”升级为完整 AI Infra 能力：能固定实验、分层定位、理解证据、
做最小修改、验证收益，并说明结论适用的硬件、软件版本和拓扑边界。
