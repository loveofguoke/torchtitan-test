# msprof-analyze 进阶分析：GLM-5.2 支持矩阵与操作指南

官方总入口：[msprof-analyze 进阶分析](https://www.hiascend.com/document/detail/zh/mindstudio/latest/msTT_msIT/msprof_analyze/docs/zh/advanced_features/README.md)。
本框架只编排官方命令，不复制其分析算法，也不根据 CSV/DB 自行伪造结论。

## 1. 默认执行策略

MindStudio 的 Ascend PyTorch Profiler 性能入口中，`--cluster-recipes` 默认值为
`necessary`。一次多卡
`--probe --analysis-tools all` 会按顺序执行：

1. Ascend PyTorch Profiler 采集并导出 text + db；
2. `msprof-analyze advisor all`；
3. `msprof-analyze cluster -m all`，生成 Insight 的标准集群交付件；
4. 对同一份 DB 执行必要的只读进阶 recipe；
5. 将命令、return code、stdout、stderr、工具版本和文件清单写入索引。

`necessary` 默认包含：

```text
cluster_time_summary
compute_op_sum
freq_analysis
communication_group_map
communication_time_sum
communication_matrix_sum
hccl_sum
slow_rank
slow_link
communication_bottleneck（逐实际 Rank）
cann_api_sum
free_analysis
export_summary
ep_load_balance（仅 EP 拓扑）
cluster_time_compare_summary（提供 baseline 时）
```

这组能力要求 Ascend PyTorch Profiler 或 msMonitor 的 DB 输入；显式选择 `msprof`
采集器时仍只运行其官方基础 cluster，不把 msProf DB 冒充进阶 recipe 输入。这组能力
覆盖计算、通信、快慢卡/慢链路、Host 下发、空闲时间和原始
API/Kernel 导出。它们是诊断证据，不是性能 PASS/FAIL 标准。
标准 `distributed` preset 会采集全部 Rank、Level1、DB、通信、内存和 shape，因此能够
支撑上述通用能力以及 EP 的 GroupedMatmul token-shape 负载分析。

## 2. CLI

### 2.1 默认完整诊断

```bash
python tests/glm5_2_mindstudio/performance_benchmark.py \
  --probe --device npu --collector torch_npu_profiler \
  --topology ddp2 --preset distributed \
  --analysis-tools all
```

`--topology all` 会对每个已采集拓扑分别执行拓扑适用的 recipe：

```bash
python tests/glm5_2_mindstudio/performance_benchmark.py \
  --probe --device npu --collector torch_npu_profiler \
  --topology all --preset distributed \
  --analysis-tools all
```

### 2.2 只保留基础 cluster

```bash
python tests/glm5_2_mindstudio/performance_benchmark.py \
  --analyze --device npu --topology ddp2 --preset distributed \
  --analysis-tools cluster --cluster-recipes none
```

### 2.3 指定 recipe

```bash
python tests/glm5_2_mindstudio/performance_benchmark.py \
  --analyze --device npu --topology ddp2 --preset distributed \
  --analysis-tools cluster \
  --cluster-recipes cluster_time_summary,slow_rank,communication_bottleneck,free_analysis
```

### 2.4 集群细粒度 A/B 比对

baseline 必须是另一份同类、多 Rank、DB 格式的 profiler 根目录：

```bash
python tests/glm5_2_mindstudio/performance_benchmark.py \
  --analyze --device npu --topology ddp2 --preset distributed \
  --analysis-tools cluster --cluster-recipes necessary \
  --cluster-summary-baseline /path/to/baseline/profiling/traces
```

框架先分别运行 `cluster_time_summary`，再把两份含
`ClusterTimeSummary` 表的输出传给 `cluster_time_compare_summary`。不能把原始
profile 直接冒充拆解结果。

## 3. 四条核心诊断链

### 3.1 集群性能细粒度拆解

官方文档：[cluster_time_summary](https://www.hiascend.com/document/detail/zh/mindstudio/latest/msTT_msIT/msprof_analyze/docs/zh/advanced_features/cluster_time_summary_instruct.md)。

每个 rank/step 的主要字段单位为 us：

| 字段 | 含义 | 诊断重点 |
| --- | --- | --- |
| `stepTime` | 完整迭代时间 | 先找最大/最小 Rank |
| `computation` | Device 计算总时间 | shape、负载或计算卡差异 |
| `communicationNotOverlapComputation` | 未被计算掩盖的通信 | 真正暴露在关键路径上的通信 |
| `communicationOverlapComputation` | 通算重叠时间 | 与通信总时间一起看覆盖率 |
| `communicationWaitStageTime` | 通信等待阶段 | Rank 到达 collective 是否不齐 |
| `communicationTransmitStageTime` | 实际传输阶段 | 链路或 payload 是否异常 |
| `memory` | 异步拷贝总时间 | SDMA/数据搬运压力 |
| `memoryNotOverlapComputationCommunication` | 未被通算覆盖的拷贝 | 暴露在关键路径上的拷贝 |
| `free` | 非通算、非异步拷贝时间 | Host 未下发、同步或其他空闲 |
| `taskLaunchDelayAvgTime` | Host API 开始到 Device task 开始的平均延迟 | 下发积压或调度长尾 |

### 3.2 集群性能细粒度比对

官方文档：[cluster_time_compare_summary](https://www.hiascend.com/document/detail/zh/mindstudio/latest/msTT_msIT/msprof_analyze/docs/zh/advanced_features/cluster_time_compare_summary_instruct.md)。

每个拆解指标同时给出当前值、`Base` 和 `Diff`。`Diff = 当前 - baseline`，正值
表示当前更慢。先按 `stepTimeDiff` 找异常 Rank，再在 computation、未覆盖通信、
等待、传输、memory、free 和 task launch delay 中找到主要增量来源。

### 3.3 通信瓶颈与慢 Rank

官方文档：[communication_bottleneck](https://www.hiascend.com/document/detail/zh/mindstudio/latest/msTT_msIT/msprof_analyze/docs/zh/advanced_features/communication_bottleneck_instruct.md)。

框架对每个实际 Rank 执行一次 Top-20，而不是只分析默认 Rank 0。输出记录目标通信
算子的时间、slow/fast Rank 及 `reason`。官方算法继续区分：

```text
长耗时 collective
  -> 同一通信操作跨 Rank 比较
  -> 快慢差异超过阈值
     -> Host Bound：框架/CANN 下发或上游到达时间异常
     -> Device Bound：设备等待、传输或设备操作异常
```

默认阈值来自安装版本的
`communication_bottleneck/config.json`；正式报告必须记录工具版本，不应在测试仓库
复制一份阈值后让两者漂移。

`slow_rank` 汇总某 Rank 影响通信的次数；`communication_bottleneck` 下钻具体算子和
原因。两者应一起读，不能把“某卡上的 collective 显示很长”直接解释成网络慢。

### 3.4 空闲时间与 Host 下发

官方文档：[free_analysis](https://www.hiascend.com/document/detail/zh/mindstudio/latest/msTT_msIT/msprof_analyze/docs/zh/advanced_features/free_analysis_instruct.md)。

每个 Rank 输出 Top-20 大块 Free，并拆出 `Pytorch Idle Time`、`Cann Idle Time` 和
文字原因，能够区分：

- Device 仍在执行未被计算/通信统计覆盖的任务；
- PyTorch 长时间没有下发；
- CANN 两个 `node@launch` 间隔大；
- 单次 launch 自身耗时异常。

结合 `cann_api_sum` 查看哪些 ACL/Runtime API 高频或长尾，再回 Insight Timeline
定位对应 Host 区间。

## 4. 全部官方特性与框架策略

| 类别 | recipe | 策略 |
| --- | --- | --- |
| 拆解比对 | `cluster_time_summary` | necessary |
| 拆解比对 | `cluster_time_compare_summary` | 有 baseline 时 necessary |
| 拆解比对 | `module_statistic` | 需 Module-domain MSTX；`all` 按采集契约启用 |
| 拆解比对 | `calibrate_npu_gpu` | 需 Nsys GPU DB 与模块打点；独立性能校准流程 |
| 计算 | `compute_op_sum`、`freq_analysis` | necessary |
| 计算 | `ep_load_balance` | EP 拓扑 necessary |
| 计算 | `computational_op_masking` | 可显式选择；并行通信域名称必须与数据一致 |
| 计算 | `operator_mfu` | 需 `with_flops`、MSTX、shape 和 Level1+ |
| 通信 | `communication_group_map`、`communication_time_sum`、`communication_matrix_sum`、`hccl_sum` | necessary |
| 通信 | `slow_rank`、`slow_link`、`communication_bottleneck` | necessary |
| 通信 | `pp_chart` | 需 TorchTitan PP 前反向 MSTX 与 `pp_info` metadata；只显式启用 |
| Host | `cann_api_sum`、`free_analysis` | necessary |
| Host | `mstx_sum` | 仅 MSTX capture |
| 导出 | `export_summary` | necessary；在各 Rank 导出 API 与 Kernel CSV |
| 数据处理 | `mstx2commop`、`p2p_pairing` | 会修改输入 Rank DB，只允许显式启用 |

`--cluster-recipes all` 表示执行所有满足当前采集契约的只读 recipe，不会偷偷执行
会修改原始 DB 的 recipe。要执行后者必须明确写名称，并保留原始 capture 备份。

## 5. 输出层级

```text
mindstudio_runs/.../<run>/
└── cluster/
    ├── cluster_analysis_output/       # 基础 cluster，Insight 集群入口
    ├── advanced/
    │   ├── index.json                 # recipe、跳过原因、Rank、文件清单
    │   ├── cluster_time_summary/
    │   ├── slow_rank/
    │   ├── slow_link/
    │   ├── communication_bottleneck/
    │   │   ├── rank_0/
    │   │   └── rank_1/
    │   ├── free_analysis/
    │   └── ...
    └── ...
```

每个 recipe 目录保持官方 `cluster_analysis_output` 子结构；同级 JSON 保存准确命令、
stdout/stderr 和工具链元数据。原始 profiler 根不会因只读 recipe 被删除或改写。

## 6. 版本和失败语义

- 正式实验使用 `toolchain.lock.json` 固定的 msprof-analyze commit/version；
- 当前安装版本没有某 recipe 时，命令失败并保留该 recipe 的 JSON，不会伪装成功；
- `--force` 是实验分析阶段重建；`--cluster-bypass-input-safety-checks` 才是传给官方
  cluster 的输入安全绕过，两者不能混淆；
- 中断后不加 `--force` 会沿用同一 capture；失败的 cluster 分析阶段重新生成，不能
  混用旧 recipe 输出；
- recipe 完成只证明工具成功生成交付件，不代表已经找到或解决性能根因。
