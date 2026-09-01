# MindStudio 官方性能工作流

本流程不重新实现 Profiler，而是把官方工具按职责串起来：

```text
同一训练契约
  +-- profiler-off 重复运行 ----------------------> 性能数值基线
  +-- msProf（默认） -----------------------------> CANN/NPU 系统证据
  |                                                    +-- cluster（db）
  |                                                    +-- MindStudio Insight
  +-- Ascend PyTorch Profiler --------------------> PyTorch/CANN/NPU 深度证据
                                                       +-- offline parse
                                                       +-- advisor/cluster/compare
                                                       +-- MindStudio Insight
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
- [Insight 系统调优快速入门](https://www.hiascend.com/document/detail/zh/mindstudio/latest/GUI_baseddevelopmenttool/MindStudioInsight/docs/zh/quick_start/system_tuning_quick_start.md)
- [msMemScope 快速入门](https://www.hiascend.com/document/detail/zh/mindstudio/latest/msTT_msIT/msMemScope/docs/zh/quick_start/quick_start.md)
- [msOpProf 使用场景](https://www.hiascend.com/document/detail/zh/mindstudio/latest/msOT/Operatordevelopmenttools/docs/zh/user_guide/msopprof_usage.md)
- [msServiceProfiler 源码与文档](https://gitcode.com/Ascend/msserviceprofiler)
- [MindStudio 实践案例](https://www.hiascend.com/document/detail/zh/mindstudio/latest/practicalcases/index.html)
- [大模型训练性能瓶颈定位官方案例](https://www.hiascend.com/document/detail/zh/mindstudio/latest/practicalcases/Largemodeltraining/MindStudio/latest/zh/cases/case_of_troubleshooting_performance_bottleneck_in_llm_training.md)
- [性能问题通用定位官方指南](https://www.hiascend.com/document/detail/zh/mindstudio/latest/practicalcases/GeneralPerformanceIssue/MindStudio/latest/zh/cases/general_performance_issue_troubleshooting_guide/overview.md)

`latest` 用于阅读。正式实验必须按 Driver/Firmware/CANN/PyTorch/torch_npu 兼容
矩阵选择版本，并把 lock、resolved manifest、可执行文件和源码 commit 写入 manifest。

## 2. 采集器与边界

| collector | 当前执行 | 层级 | 正确用途 |
|---|---:|---|---|
| `msprof` | 是，默认 | CANN/NPU | 包装完整但短小的 TorchTitan 作业，得到 Insight 和 analyzer 输入。|
| `torch_npu_profiler` | 是 | PyTorch/CANN/NPU | 需要 module、shape、stack、memory、schedule 窗口时深度采集。|
| `msopprof` | 否 | 单算子/Kernel | 从整网定位热点后做上板或仿真下钻，不是整网训练 launcher。|
| `msmemscope` | 否 | 专项内存 | 内存泄漏、生命周期、低效内存；待目标 CANN 完成独立接入验证。|
| `service_profiler` | 否 | 在线推理服务 | 面向 MindIE/vLLM/SGLang 请求链路，不适用于离线训练。|

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

msProf 是 CANN/NPU 的通用底座；Ascend PyTorch Profiler 在其能力上补充 PyTorch
语义和 step schedule。默认使用 msProf 建立跨模型统一入口；需要
`nn.Module -> ATen/ACL -> NPU Kernel` 关联时切到 `torch_npu_profiler`；测速时两者
都关闭。

## 3. 采集命令

入口：

```bash
# 默认 msProf 采集机
python -m tests.glm5_2_mindstudio.toolchain doctor \
  --scope performance-capture

# Ascend PyTorch Profiler 深度采集机
python -m tests.glm5_2_mindstudio.toolchain doctor \
  --scope performance-torch-npu-capture

# 独立离线分析机
python -m tests.glm5_2_mindstudio.toolchain doctor \
  --scope performance-analysis

# 只有同一环境承担采集和分析时才运行
python -m tests.glm5_2_mindstudio.toolchain doctor --scope performance

python tests/glm5_2_mindstudio/performance_benchmark.py --help
```

### 3.1 profiler-off 数值基线

先关闭采集器重复测速。`--replicate` 是独立运行编号，不是 profiler schedule 的
repeat；三次运行分别落到独立目录。

```bash
# 单卡 3 次
export ASCEND_RT_VISIBLE_DEVICES=4
for r in 1 2 3; do
  python tests/glm5_2_mindstudio/performance_benchmark.py \
    --capture --device npu --profiler-off \
    --topology single --replicate "$r"
done

# 一个分布式拓扑 3 次
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
for r in 1 2 3; do
  python tests/glm5_2_mindstudio/performance_benchmark.py \
    --capture --device npu --profiler-off \
    --topology fsdp8 --replicate "$r"
done

# 所有不超过 8 卡的拓扑各 3 次
for r in 1 2 3; do
  python tests/glm5_2_mindstudio/performance_benchmark.py \
    --capture --device npu --profiler-off \
    --topology all --replicate "$r"
done
```

以三次 profiler-off 的 median/p90 step time、tokens/s、peak HBM 和 rank
min/median/max 作为性能数值。下面 profiler-active 的结果只做归因。

### 3.2 profiler-active 证据采集

单卡 msProf：

```bash
export ASCEND_RT_VISIBLE_DEVICES=4
python tests/glm5_2_mindstudio/performance_benchmark.py \
  --capture --device npu --collector msprof \
  --topology single --preset overview
```

一个分布式拓扑：

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python tests/glm5_2_mindstudio/performance_benchmark.py \
  --capture --device npu --collector msprof \
  --topology fsdp8 --preset overview
```

所有不超过八卡的拓扑：

```bash
python tests/glm5_2_mindstudio/performance_benchmark.py \
  --capture --device npu --collector msprof \
  --topology all --preset overview
```

`all` 只是提供统一编排，并不表示应在共享服务器上一开始就全量采集。msProf 会采集
完整作业，不受旧 `skip/warmup/active` schedule 控制；多拓扑、多 rank、text 导出
可能产生数百 GiB。正式矩阵应先跑 profiler-off 全拓扑取得可比基线，再对 DP、TP、
CP、PP、EP 和复合并行各选代表拓扑做 msProf；只有存储预算、采集窗口和保留策略
明确时才执行上面的 `all`。需要精确 steady-state step 窗口时，优先使用
`torch_npu_profiler`，或仅在当前 CANN 文档确认后用 `--collector-arg` 传入
`delay/duration` 类选项。

这里 `overview` 只是复用旧 performance CLI 的轻量训练配置槽位，msProf 运行名
明确写成 `basic-msprof`；它不会套用 torch_npu preset 的 level/shape/stack 配置。
msProf 不允许 `--preset all`，因为重复八次完整作业不等于八种框架内采集策略。

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
对应 Insight 页面见本文第 2 节和 [官方文档矩阵](OFFICIAL_DOCUMENTATION_MATRIX_ZH.md)。

当前 MindStudio performance 只实现 NPU。`--device cuda` 保留接口并明确报未实现；
不能让 GPU 静默落入另一套未确认的采集语义。

## 4. 分析命令

分析已存在 capture，不重跑训练。先按官方输入矩阵选择 recipe：

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
- cluster 生成 `cluster_analysis_output`，应整目录导入 Insight；
- compare 把训练耗时拆为算子/通信/调度，并比较算子耗时、通信和内存；XLSX 的
  差异是候选根因，不是自动 PASS/FAIL。

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

## 5. MindStudio Insight 阅读路径

`analyze` 生成 `mindstudio_insight_handoff.json`，列出完整 profile/import root、
`cluster_analysis_output`、DB/Timeline/CSV/XLSX/HTML 清单和视图入口。服务器无 GUI
时把 import target 整目录同步到 Windows/macOS。多卡不能只同步 rank 0。

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

## 6. 输出、同步与生命周期

```text
mindstudio_runs/<card-scope>/<topology>/<run>/
  runtime.log / run_state.json
  trainer_output/profiling/msprof/       # msProf
  trainer_output/profiling/traces/       # torch_npu.profiler
  advisor*/ cluster*/ compare*/

mindstudio_artifacts/<card-scope>/<topology>/<run>/
  manifest.json / metrics.jsonl / analysis.json
  mindstudio_insight_handoff.json

mindstudio_reports/<card-scope>/<topology>/<run>.html
```

采集 manifest 只记录影响采集语义的身份。msProf 路径记录 collector 参数、lock、
resolved source、msProf、CANN、torch/torch_npu；Ascend PyTorch Profiler 路径记录
torch、torch_npu、CANN，以及 TorchTitan profiler、TorchTitanTurbo NPU adapter 和
本仓 capture adapter 的源码哈希。它们都不要求采集服务器安装 msprof-analyze。
离线分析在自己的 operation/analysis manifest 中记录 msprof-analyze 版本/源码，
以及实际用于渲染的 Insight/FlameGraph 脚本哈希。采集栈升级需要新 capture；只升级
analyzer 或可视化脚本只需 `--force --analyze`，不能把旧派生结果与新工具混在一起。
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

`mindstudio_insight_handoff.json` 同时保存服务器绝对路径和相对仓库根的 portable
路径。Release `analysis` 保留经审查的 `msprof_*.db`、解析表、Timeline、XLSX/HTML
和报告，但不保留原始 device payload；如果 Insight 的某个视图要求完整原始 profile，
使用经安全审查的 `full` archive。handoff 中每个 Insight 视图还会给出
`ready`、`inspect_database` 或 `missing` 证据状态，避免把 GUI 菜单存在误写成数据已采集。

## 7. 三仓边界与服务器验证

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
