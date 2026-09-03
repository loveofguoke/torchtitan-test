# MindStudio 26.1 官方文档与 GLM-5.2 能力矩阵

本文是本目录的文档审计入口。每一项都回答六个问题：官方工具解决什么问题、输入是
什么、GLM 命令是什么、输出在哪里、下一步用什么页面或命令分析、当前是否完成自动化。

状态含义：

- `SUPPORTED`：已存在 GLM 命令、生命周期、产物校验和文档；仍需目标服务器实跑验收；
- `HANDOFF`：本目录能生成或索引官方输入，但后续操作由官方 GUI/CLI 完成；
- `SPECIALIZED`：工具适用，但需要独立的服务、算子工程或推理模型，不能包装整网训练；
- `NOT_APPLICABLE`：不属于当前 PyTorch 训练场景。

工具运行成功只表示该阶段完成，不表示数值或性能结论通过。

## 1. 总入口与实践案例

| 官方章节 | 用途 | GLM 对应 | 状态 |
| --- | --- | --- | --- |
| [MindStudio 使用导读](https://www.hiascend.com/document/detail/zh/mindstudio/latest/index/index.html) | 训练、推理、算子三条工具链总览 | 本矩阵和 README 阅读顺序 | SUPPORTED |
| [训练推理开发工具介绍](https://www.hiascend.com/document/detail/zh/mindstudio/2610/msTT_msIT/msIT_msTT_user_guide/MindStudio/26.1.0/zh/user_guide/mstt_msit_user_guide.md) | 按迁移、精度、性能、内存、服务场景选择工具 | `toolchain.py doctor`、各 benchmark | SUPPORTED |
| [大模型训练精度定位指南](https://www.hiascend.com/document/detail/zh/mindstudio/2610/practicalcases/LargeModelTrainingAccuracy/docs/zh/best_practices/train_debug_guide.md) | CheckList、复现、NaN、首 step、长稳、随机性、模块/API 下钻 | configuration、monitor、migration、compile benchmark | SUPPORTED |
| [大模型训练性能瓶颈定位案例](https://www.hiascend.com/document/detail/zh/mindstudio/2610/practicalcases/Largemodeltraining/MindStudio/26.1.0/zh/cases/case_of_troubleshooting_performance_bottleneck_in_llm_training.md) | 采集后按计算/调度/通信定界并 A/B 验收 | performance benchmark、Insight handoff | SUPPORTED |
| [性能问题通用定位](https://www.hiascend.com/document/detail/zh/mindstudio/2610/practicalcases/GeneralPerformanceIssue/MindStudio/26.1.0/zh/cases/general_performance_issue_troubleshooting_guide/guide.md) | 从现象选择采集和分析层级 | `OFFICIAL_PRACTICE_ROADMAP_ZH.md` | HANDOFF |
| [大模型推理精度定位](https://www.hiascend.com/document/detail/zh/mindstudio/2610/practicalcases/Analysiscaseofreasoningaccuracy/docs/zh/best_practices/infer_debug_guide.md) | 推理端 dump、比较和根因定位 | 需要独立推理入口和模型导出 | SPECIALIZED |
| [大模型推理量化调试调优](https://www.hiascend.com/document/detail/zh/mindstudio/2610/practicalcases/InferenceQuantization/MindStudio/26.1.0/zh/cases/foundation_model_inference_quantization_debugging_and_tuning_guide.md) | 量化、精度与性能联合调试 | 需要量化模型和推理服务 | SPECIALIZED |

## 2. 训练精度工具

| 官方章节/工具 | 官方输入与输出 | GLM 命令 | 分析入口 | 状态 |
| --- | --- | --- | --- | --- |
| [msProbe PyTorch 快速入门](https://www.hiascend.com/document/detail/zh/mindstudio/2610/msTT_msIT/msProbe/docs/zh/quick_start/pytorch_quick_start.md) | 模型训练进程 -> dump/construct/stack/config | `migration_benchmark.py --capture ...` | `official_summary.json`、官方 CSV/XLSX | SUPPORTED |
| [PyTorch 精度比对](https://www.hiascend.com/document/detail/zh/mindstudio/latest/msTT_msIT/msProbe/docs/zh/user_guide/accuracy_compare/pytorch_accuracy_compare_instruct.md) | target/golden dump -> compare result、advisor | `migration_benchmark.py --compare` | Result、Err_Message、MeanRelativeErr、双千指标 | SUPPORTED |
| ConfigChecker | 两端环境、库、参数、权重、数据 -> zip/result.xlsx | `configuration_check_benchmark.py` | summary sheet 后逐项 sheet | SUPPORTED |
| API 精度预检 | L1 API dump -> acc_check result/details | `migration_benchmark.py --precheck ROLE` | 单端 API 对 CPU 高精度结果 | SUPPORTED |
| API 预检比较 | 两端 details -> api_precision_compare | `migration_benchmark.py --precheck-compare` | `precheck_report.html` 和官方 CSV | SUPPORTED |
| graph_visualize | L0/mix construct -> `.vis.db` | `migration_benchmark.py --graph-visualize` | TensorBoard Ascend Graph | SUPPORTED |
| structure/overflow/nan capture | 结构、软件统计溢出或 NPU 寄存器状态 -> construct/dump | `migration_benchmark.py --capture ... --dump-task TASK` | construct、首异常节点和 is_nan | SUPPORTED |
| TrainerMonitorV2 | 训练模块、优化器、配置 -> 多 step CSV | `training_monitor_benchmark.py` | 激活/梯度/权重/优化器状态趋势 | SUPPORTED |
| 趋势可视化 | 大规模 monitor/dump -> `.trend.db` | `migration_benchmark.py --trend ROLE` 或 monitor 同名入口 | TensorBoard Trend Analyzer | SUPPORTED |
| [编译精度比对](https://www.hiascend.com/document/detail/zh/mindstudio/latest/msTT_msIT/msProbe/docs/zh/user_guide/accuracy_compare/pytorch_compile_accuracy_compare_instruct.md) | 同一 NPU 的 eager/compile -> PrecisionChecker CSV | `compile_accuracy_benchmark.py` | 每 rank、model part、module 的 fwd/bwd 结果 | SUPPORTED |

### 2.1 标准命令顺序

```bash
python -m tests.glm5_2_mindstudio.toolchain doctor --scope full

python tests/glm5_2_mindstudio/migration_benchmark.py --data --topology single
python tests/glm5_2_mindstudio/configuration_check_benchmark.py --capture reference --topology single
python tests/glm5_2_mindstudio/configuration_check_benchmark.py --capture candidate --topology single
python tests/glm5_2_mindstudio/configuration_check_benchmark.py --compare --topology single

python tests/glm5_2_mindstudio/migration_benchmark.py \
  --capture reference --topology single --level L0 --dump-task statistics
python tests/glm5_2_mindstudio/migration_benchmark.py \
  --capture candidate --topology single --level L0 --dump-task statistics
python tests/glm5_2_mindstudio/migration_benchmark.py --compare --topology single
python tests/glm5_2_mindstudio/migration_benchmark.py --graph-visualize --topology single
```

定位到模块后才把 `--level` 提升到 L1/mix，缩小 step、rank 和 module；确认 API 时才
使用 tensor dump 和预检。这样符合官方“先粗后细”的数据量控制原则。

## 3. 性能采集、分析和可视化

| 官方章节/工具 | 适用输入 | GLM 命令 | 关键输出 | 状态 |
| --- | --- | --- | --- | --- |
| [msProf 快速入门](https://www.hiascend.com/document/detail/zh/mindstudio/2610/msTT_msIT/msProf/docs/zh/quick_start/msprof_quick_start.md) | 进程外包装 PyTorch 作业 | `performance_benchmark.py --collector msprof --capture` | PROF/DB/communication/step trace | SUPPORTED |
| [Ascend PyTorch Profiler](https://www.hiascend.com/document/detail/zh/mindstudio/2610/msTT_msIT/ascend_pytorch_profiler/docs/zh/ascend_pytorch_profiler/ascend_pytorch_profiler_user_guide.md) | 训练 step 内 profile API | `--collector torch_npu_profiler --capture` | `*_ascend_pt`、FRAMEWORK、CANN、Device、DB/Text | SUPPORTED |
| [msprof-analyze 快速入门](https://www.hiascend.com/document/detail/zh/mindstudio/2610/msTT_msIT/msprof_analyze/docs/zh/quick_start/msprof-analyze_quick_start.md) | 已采集 profile | `--analyze --analysis-tools ...` | advisor/cluster/compare | SUPPORTED |
| [集群分析](https://www.hiascend.com/document/detail/zh/mindstudio/latest/msTT_msIT/msprof_analyze/docs/zh/user_guide/cluster_analyse_instruct.md) | 同次采集的完整多 rank db/text | `--cluster --cluster-mode ...` | cluster DB、step trace、communication、matrix、group | SUPPORTED |
| [性能比对](https://www.hiascend.com/document/detail/zh/mindstudio/latest/msTT_msIT/msprof_analyze/docs/zh/user_guide/compare_tool_instruct.md) | 官方支持的 baseline 与 comparison profile | `--compare-baseline PATH` | 计算/通信/调度、算子、内存差异 | SUPPORTED |
| [专家建议](https://www.hiascend.com/document/detail/zh/mindstudio/latest/msTT_msIT/msprof_analyze/docs/zh/user_guide/advisor_instruct.md) | `*_ascend_pt` | `--advisor` | HTML/XLSX/terminal suggestions | SUPPORTED |
| [Insight 系统调优快速入门](https://www.hiascend.com/document/detail/zh/mindstudio/2610/GUI_baseddevelopmenttool/MindStudioInsight/docs/zh/quick_start/system_tuning_quick_start.md) | 完整 profile 或 cluster output | handoff JSON 给出导入根 | Summary/Timeline/Communication/Operator/Memory | HANDOFF |
| msMemScope | 指定进程和内存采集配置 | 需确定目标工具版本和注入方式 | 内存申请、释放、泄漏与碎片证据 | SPECIALIZED |
| msMonitor | 长时在线性能与健康监测 | 需部署 monitor 服务/数据库 | 长时异常和集群指标 | SPECIALIZED |
| msPTI/msTX | 框架/应用埋点接口 | 当前训练已有 MSTX 开关；其他 API 按需接入 | 自定义 range、事件和关联 | HANDOFF |

标准性能闭环：

```text
短采集 -> Insight Summary 定界
  计算异常 -> Operator 按类型/名称/shape 比较
  调度异常 -> Timeline 看 HostToDevice、Free、CPU 下发
  通信异常 -> Communication 看通信域、等待/同步、矩阵和链路
  内存异常 -> Memory 看分配释放、峰值、碎片和内存重整
  -> advisor/cluster/compare 验证候选根因
  -> 单变量修改 -> 相同窗口 A/B
```

集群分析的所有参数、字段和诊断例子见 [MSPROF_ANALYZE_CLUSTER_ZH.md](MSPROF_ANALYZE_CLUSTER_ZH.md)。

## 4. MindStudio Insight 页面

| 页面 | 先看什么 | 能回答什么 | 不能直接推出什么 |
| --- | --- | --- | --- |
| Summary | 并行分组、各 rank/stage 的计算、通信、空闲、Bubble | 问题首先属于计算、调度还是通信 | 某个 kernel 源码为何慢 |
| Timeline | Host、CANN、Device、HCCL、HostToDevice 连线、Free gap | 下发空洞、同步、stream overlap、长尾起点 | 单凭一条长线断言网络故障 |
| Communication | 通信域、通信时长、同步/等待、通信矩阵 | 慢 rank、慢链路、负载不均和未覆盖通信 | 上游计算为何晚到 collective |
| Operator | 类型、名称、输入 shape、调用次数、总/平均耗时 | 哪类算子或 shape 劣化 | 不做单算子复现就断言算子 bug |
| Memory | active/reserved、申请释放、算子内存、时间线 | 峰值、碎片、内存重整和泄漏候选 | 仅凭 reserved 高就认定 OOM 根因 |
| RL | rollout、推理、训练等 RL 阶段 | RL 工作负载阶段耗时与资源分布 | 普通预训练没有相应语义时强行解释 |

官方 Insight 的采集器映射不能混用：系统调优主要导入 msProf/Ascend PyTorch
Profiler，算子调优导入 msOpProf，服务化调优导入服务化 Profiling，内存调优导入
msMemScope。当前训练闭环只自动生成系统调优 handoff；其他场景必须使用对应采集器。

官方 26.1 规格提示：多个性能文件总量建议不超过 20GB；JSON 建议单文件不超过 1GB，
系统调优 DB 建议单文件不超过 1GB，CSV 建议单文件不超过 500MB。工具的硬限制更高，
但超过建议值会显著降低导入和交互效率。大集群应优先 DB、短窗口和代表性拓扑，不能
机械展开“全部拓扑 x 全部重型采集项”。

## 5. 推理工具链

| 工具 | 官方用途 | 当前 GLM 训练目录处理方式 | 状态 |
| --- | --- | --- | --- |
| msTransplant | 分析 GPU/PyTorch 到昇腾的 API 迁移问题 | 可用于源码迁移审计，不属于训练 capture | HANDOFF |
| msPrechecker | 推理模型/算子支持与环境预检 | 需要推理模型和后端契约 | SPECIALIZED |
| msModeling | 建模、仿真和吞吐估算 | 需要推理部署规格 | SPECIALIZED |
| msModelSlim | 权重量化、校准、评估 | 需要量化目标、校准集和推理评测 | SPECIALIZED |
| msServiceProfiler | 在线推理服务的请求、调度和执行分析 | 需要服务进程和压测请求 | SPECIALIZED |
| AISBench | 推理性能/精度评测 | 需要导出模型和推理后端 | SPECIALIZED |

## 6. 算子工具链

| 工具 | 官方用途 | 何时从 GLM 训练下钻 | 状态 |
| --- | --- | --- | --- |
| msKPP | 算子设计与性能建模 | profile 已证明瓶颈落在自定义 kernel 设计 | SPECIALIZED |
| msOpGen | 生成算子工程 | 已确定要交付独立 Ascend 算子 | SPECIALIZED |
| msDebug | 功能和内核调试 | 单算子复现失败或 kernel 行为异常 | SPECIALIZED |
| msSanitizer | 越界、竞态、异常检测 | NaN/随机失效/内存踩踏指向 kernel | SPECIALIZED |
| msOpProf | 单算子多维性能采集 | Operator 页面已定位到具体 op/shape | SPECIALIZED |
| msKL | 算子调用与验证 | 单算子复现和基准 | SPECIALIZED |
| msInsight 算子调优 | 展示 Pipe、带宽、指令和内存访问 | msOpProf 产物完成后 | HANDOFF |

整网训练不能直接替代单算子工程。正确路径是先用系统 profiler 定位 op 和 shape，
再提取最小输入进入算子工具链，最后回到完整训练验证收益与精度。

## 7. 文档审计规则

MindStudio 版本升级时必须重新检查：

1. 总入口新增/删除的训练、推理、算子工具；
2. 官方输入格式和目录结构；
3. CLI 子命令、参数名和默认值；
4. 关键输出文件与 Insight 支持页面；
5. 实践案例推荐的定位顺序和阈值；
6. `toolchain.lock.json` 的 tag/commit 与目标 CANN/torch_npu 配套关系；
7. 本矩阵每一行的状态、命令和服务器验收记录。

正式运行必须把 lock 中的浮动分支改成已评审 tag 或完整 commit；否则只能称为探索性
环境，不能用来复现正式结论。
