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

处理方式是新增仅限 `--objectives performance` 的
`--performance-nondeterministic`。该开关不允许用于 precision/mixed acceptance，因此没有
放宽精度标准。正式 eager/Inductor 性能矩阵都使用相同的 nondeterministic 合约。

## 环境限制

- profiler 关闭；本矩阵衡量 steady-state 吞吐，不分解通信、计算、内存时间。
- NPU0/1/3/4 在实验前报告 Alarm。
- 八卡 Inductor batch 与外部 NPU0 单卡 5000-step 作业重叠；未终止、暂停或修改该作业。
- 因此八卡的绝对值和低于 5% 的变化不得直接作为验收结果。TP 家族跨 6 个组合一致获得
  45.52%-54.00% 改善，是强结构性信号，但仍建议在空闲健康节点复验。
- 本次 graph 指 Inductor/AOT/Triton，不包括 NPUGraphs native replay。
