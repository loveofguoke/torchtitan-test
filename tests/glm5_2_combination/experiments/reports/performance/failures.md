# 组合性能失败与限制

## 确定性 Inductor autotune 失败

首次 single-r1 使用默认确定性训练策略。当前 NPU Triton autotuner 拒绝在确定性算法
开启时对未经 deterministic 标注的候选 kernel 做 benchmark，运行在编译阶段失败。这不是
模型数值失败，也不是 eager 失败。

证据：

```text
graph_debug_runs/launcher-combination-inductor-20260826-111207-4142884/reports/report.md
combination_runs/self-npu-bf16-random-s30-b64-seq128-seed61-eager-inductor-performance-no-prof-skip10-eeb6bf36/single/single-r1/runtime.log
```

当时为了完成纯性能诊断矩阵，新增了仅限 `--objectives performance` 的
`--performance-nondeterministic`；正式 eager/Inductor 性能矩阵两侧都使用相同的
nondeterministic 合约。该开关不允许用于 precision/mixed acceptance，因此没有放宽精度
标准。

随后正式 deterministic precision 暴露了同一 torch_npu 调用缺少
`is_vetted_benchmarking` 的根因。当前配套 Turbo G020 workaround 只把 pointwise heuristic
声明为 vetted，reduction 仍受 deterministic guard 保护；common launcher 默认启用该
opt-in。当前 deterministic single cold-cache 与 smoke 已通过，因此新实验应优先保留
deterministic，不能继续把 performance nondeterministic 当成图模式必需条件。详细调用链与
底层根治位置见
[`LOWER_LAYER_ISSUE_HANDOFF.md`](../../../../glm5_2_graph/LOWER_LAYER_ISSUE_HANDOFF.md#10-g020确定性-pointwise-autotune-未声明-vetted)。

## 环境限制

- profiler 关闭；本矩阵衡量 steady-state 吞吐，不分解通信、计算、内存时间。
- NPU0/1/3/4 在实验前报告 Alarm。
- 八卡 Inductor batch 与外部 NPU0 单卡 5000-step 作业重叠；未终止、暂停或修改该作业。
- 因此八卡的绝对值和低于 5% 的变化不得直接作为验收结果。TP 家族跨 6 个组合一致获得
  45.52%-54.00% 改善，是强结构性信号，但仍建议在空闲健康节点复验。
- 本次 graph 指 Inductor/AOT/Triton，不包括 NPUGraphs native replay。
