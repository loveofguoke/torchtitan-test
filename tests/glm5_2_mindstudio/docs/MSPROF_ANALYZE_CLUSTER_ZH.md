# msprof-analyze cluster：GLM-5.2 操作、字段与定位指南

官方依据：[集群分析](https://www.hiascend.com/document/detail/zh/mindstudio/latest/msTT_msIT/msprof_analyze/docs/zh/user_guide/cluster_analyse_instruct.md)。
本文按 MindStudio 26.1 的命令、输入和交付件编写。

## 1. 它解决什么问题

`cluster` 聚合同一次多 rank profiling，回答三类问题：

1. **慢卡/负载不均**：同一通信域或 stage 的计算、通信、空闲时间是否明显不一致；
2. **慢链路/带宽异常**：同类 LOCAL/HCCS/PCIE/RDMA 链路的带宽是否存在异常低值；
3. **通信热点**：哪一个 HCCL 算子或通信域占用了主要通信时间，等待和同步发生在哪些 rank。

它不能单独证明网络硬件故障。某 rank 可能因为上游计算、Host 下发或数据负载较慢，
更晚进入 collective；这时其他 rank 的等待时间会增加，但链路传输带宽可能正常。

## 2. 输入契约

官方支持四类输入：msProf db、Ascend PyTorch Profiler text/db、MindSpore Profiler
text/db、msMonitor db。本工作流使用前两类。

### 2.1 Ascend PyTorch Profiler db

```text
profiling-root/
├── rank0_*_ascend_pt/
│   ├── ASCEND_PROFILER_OUTPUT/
│   │   ├── analysis.db
│   │   └── ascend_pytorch_profiler_0.db
│   └── profiler_info_*.json
└── rank1_*_ascend_pt/
    └── ...
```

db 格式处理效率更高。大集群转存时可只保留每 rank 的 `analysis.db` 和
`profiler_info_*.json`，但必须保持原目录层级。

### 2.2 Ascend PyTorch Profiler text

```text
rankN_*_ascend_pt/
├── ASCEND_PROFILER_OUTPUT/
│   ├── step_trace_time.csv
│   ├── communication.json
│   └── communication_matrix.json
└── profiler_info_*.json
```

要获得通信小算子、带宽和矩阵，`profiler_level` 应为 Level1 或更高。Level0 通常只能
汇总 step trace，不足以做完整通信诊断。

### 2.3 同一次采集约束

`-d` 必须指向同一次作业、全量已选择 rank 的公共根目录。不能混入另一次 capture，
也不能把 rank0 来自本轮、rank1 来自旧轮。否则 src/dst rank 映射和通信域会失真。
本工作流通过 capture identity、attempt ID、同 generation manifest 和文件 hash 防止混搭。

## 3. GLM 命令

### 3.1 采集并执行完整分析

```bash
python tests/glm5_2_mindstudio/performance_benchmark.py \
  --probe --device npu --collector msprof \
  --topology fsdp8 --preset overview \
  --cluster --cluster-mode all
```

`--probe` 等于本次执行 capture + analyze。已有完整 capture 时只运行：

```bash
python tests/glm5_2_mindstudio/performance_benchmark.py \
  --analyze --device npu --collector msprof \
  --topology fsdp8 --preset overview \
  --cluster --cluster-mode all
```

### 3.2 只分析通信耗时

```bash
python tests/glm5_2_mindstudio/performance_benchmark.py \
  --analyze --device npu --collector msprof \
  --topology fsdp8 --preset overview \
  --cluster --cluster-mode communication_time
```

### 3.3 只分析通信矩阵

```bash
python tests/glm5_2_mindstudio/performance_benchmark.py \
  --analyze --device npu --collector msprof \
  --topology fsdp8 --preset overview \
  --cluster --cluster-mode communication_matrix
```

### 3.4 Agent JSON 输出

```bash
python tests/glm5_2_mindstudio/performance_benchmark.py \
  --analyze --device npu --collector msprof \
  --topology fsdp8 --preset overview \
  --cluster --cluster-mode all --cluster-agent-output
```

官方 `--agent` 把 JSON 摘要写到标准输出；本工作流把 stdout 同时写入 `cluster.json`，
供自动化系统读取。JSON 是分析摘要，不取代 DB/CSV/JSON 交付件。

### 3.5 分析器安全检查与实验生命周期不是同一个 `force`

- benchmark `--force`：删除选中实验 generation 的旧结果后重跑；
- `--cluster-bypass-input-safety-checks`：向官方 cluster 传递 `--force`，绕过属主、权限
  以及 csv/json/db 大文件限制。

只有确认输入可信、磁盘和内存足够时才使用后者：

```bash
python tests/glm5_2_mindstudio/performance_benchmark.py \
  --analyze --device npu --collector msprof \
  --topology fsdp8 --preset overview \
  --cluster --cluster-mode all \
  --cluster-bypass-input-safety-checks
```

## 4. 实际执行的官方命令

默认等价于：

```bash
msprof-analyze cluster \
  -m all \
  -d <same-capture-multi-rank-profile-root> \
  -o <run-directory>/cluster
```

官方模式：

| mode | 生成重点 | 使用场景 |
| --- | --- | --- |
| `all` | communication time + matrix | 首次完整分析，默认 |
| `communication_time` | 通信算子耗时、等待、同步、通信域 | 通信暴露时间高或 rank 等待明显 |
| `communication_matrix` | src/dst、链路类型、传输量、带宽 | 慢链路或跨节点带宽异常 |

## 5. 输出目录

```text
mindstudio_runs/<experiment>/<topology>/<run>/
├── cluster.json                 # 命令、return code、stdout/stderr、工具版本
└── cluster/
    └── cluster_analysis_output/
        ├── cluster_analysis.db
        ├── cluster_step_trace_time.csv
        ├── cluster_communication.json
        ├── cluster_communication_matrix.json
        └── communication_group.json
```

具体文件是否出现取决于输入格式、采集级别、mode 和原始数据完整度。某文件缺失不能
自动解释为“没有通信”，应先检查采集级别和输入目录。

## 6. cluster_step_trace_time.csv

先按同一 `Step` 分析，再先看 `Type=stage`、后看 `Type=rank`。

| 字段 | 含义 | 诊断方式 |
| --- | --- | --- |
| `Step` | 被分析的训练 step | 多 step 时先固定目标 step，避免把 warmup 与 steady-state 混合 |
| `Type` | `rank` 或 `stage` | PP 先比较 stage，再下钻 rank |
| `Index` | rank ID 或 stage ID | 找最大/最小和异常长尾 |
| `Computing` | Device 计算时间 | 同组差异大可能是负载、shape、算子或硬件差异 |
| `Communication` | 全部通信时间 | 包含能被计算覆盖和未覆盖部分 |
| `Overlapped` | 计算与通信同时执行的时长 | 越大不一定越差；关键看未覆盖通信是否下降 |
| `Communication(Not Overlapped)` | 暴露在 step 关键路径上的通信 | 直接拖慢 step 的通信部分 |
| `Free` | Device 既不计算也不通信 | 可能是 Host 下发慢、等待、SDMA、内存重整 |
| `Preparing` | step 开始到首个计算/通信前的准备 | 数据、Host 或调度准备时间 |
| `Stage` | PP stage 时间，不含 receive | 比较各 stage 负载 |
| `Bubble` | receive 时间总和 | PP 空泡/等待的重要信号 |
| `...Exclude Receive` | 排除 receive 后的未覆盖通信 | 区分 PP receive 与其他通信瓶颈 |
| `DP/PP/TP Index` | rank 所属并行组 | 把异常限定到具体 mesh 轴/通信域 |

官方案例建议：同类 rank/stage 时间最大值与最小值差异超过约 5% 时重点排查。5% 是
诊断触发点，不是所有模型的统一性能 PASS 阈值。

### 6.1 例子：等待不等于网络慢

```text
rank0: 上游 GEMM 10 ms -> 进入 AllReduce -> 等待 0 ms -> 传输 2 ms
rank1: 上游 GEMM 18 ms -> 进入 AllReduce -> 等待 0 ms -> 传输 2 ms

时间轴：
rank0  [GEMM 10] [等待 rank1 8] [AllReduce 2]
rank1  [GEMM        18         ] [AllReduce 2]
```

rank0 的通信记录可能包含较大同步/等待，但两个 rank 的同类链路带宽都正常。根因是
rank1 上游计算或 Host 下发慢，而不是网络传输慢。应回到 Timeline/Operator 比较
collective 之前的最后一个长算子和 HostToDevice 间隙。

## 7. cluster_communication_matrix.json

典型结构：

```json
{
  "0-1": {
    "Transport Type": "HCCS",
    "Transit Time(ms)": 0.42,
    "Transit Size(MB)": 64.0,
    "Bandwidth(GB/s)": 152.38
  }
}
```

| 字段 | 含义 | 判断顺序 |
| --- | --- | --- |
| `src_rank-dst_rank` | 链路两端 | 先按同一通信域和链路类型分组 |
| `Transport Type` | LOCAL/HCCS/PCIE/RDMA | 不跨类型直接比较带宽 |
| `Transit Size(MB)` | 实际传输量 | 先确认 token/参数负载是否均衡 |
| `Transit Time(ms)` | 传输耗时 | 结合 size 解释，不能只看绝对时间 |
| `Bandwidth(GB/s)` | 有效传输带宽 | 同类链路明显偏低才是慢链路候选 |

先比较传输量，再比较同类链路带宽。EP 的 token 路由不均可能导致 size 不同；此时
带宽正常但总时长不同，首先是负载分配问题，不是链路异常。

## 8. cluster_communication.json 与 communication_group.json

`cluster_communication.json` 记录 HCCL 算子、通信域、起始时间、总耗时、传输、等待、
同步和空闲；`communication_group.json` 把 collective/P2P 通信域映射到 rank 集合。

诊断顺序：

1. 找 `elapsed_time` 或未覆盖通信占比最高的 op/group；
2. 看 `transit_time`：真实数据传输是否长；
3. 看 `wait_time`：是否在等待算子/数据就绪；
4. 看 `synchronization_time`：rank 到达是否不一致；
5. 用 group mapping 确认它属于 DP、TP、PP、CP 或 EP 轴；
6. 回 Timeline 查看该 collective 前各 rank 的上游事件；
7. 回矩阵确认 payload、链路类型和带宽。

## 9. cluster_analysis.db 表

| 表 | 关键字段 | 用途 |
| --- | --- | --- |
| `ClusterBaseInfo` | key/value | 并行策略和集群基础信息 |
| `ClusterStepTraceTime` | computing、communication、overlapped、free、bubble | rank/stage 迭代拆解 |
| `CommunicationGroupMapping` | type、rank_set、group_name/id、pg_name | 通信域归属 |
| `ClusterCommunicationTime` | op、group、elapsed/transit/wait/sync | 通信时间根因拆解 |
| `ClusterCommunicationBandwidth` | band_type、size、time、bandwidth、packet | 带宽和包大小分布 |
| `ClusterCommunicationMatrix` | src/dst、size、time、bandwidth、transport | 链路矩阵 |

不要直接修改 DB。将整个 `cluster_analysis_output` 导入版本匹配的 MindStudio Insight。

## 10. MindStudio Insight 阅读顺序

1. **Summary / 并行策略分析**：选择 DP/TP/PP/CP/EP 通信域，看热力图；
2. **计算/通信概览**：比较纯计算、通信重叠、未覆盖通信、Free 和 Bubble；
3. **Communication / 通信时长**：找同步、等待和耗时异常 rank；
4. **Communication / 通信矩阵**：检查 payload、链路类型、时间和带宽；
5. **Timeline**：回看异常 collective 前的算子、HostToDevice 和 Free gap；
6. **Operator**：若上游计算异常，按 op type/name/input shape 比较正常卡和异常卡；
7. 记录证据后再提出单一根因假设。

## 11. 从现象到调优动作

| 证据组合 | 根因候选 | 下一步验证 | 可能动作 |
| --- | --- | --- | --- |
| payload 相同，同类链路带宽某一对明显低 | 慢链路/拓扑/网络配置 | 重复采集、换卡/换节点、检查 HCCL/链路 | 修复拓扑、网络或通信配置 |
| payload 不同，带宽正常 | DP/EP token 或工作量不均 | 查看数据分配/router token count | 数据平衡、容量/路由策略、负载均衡 |
| wait/sync 高，transit 正常 | rank readiness skew | Timeline 看 collective 前最后事件 | 优化上游算子、Host 下发、绑核或负载 |
| 未覆盖通信高，但 overlapped 低 | overlap 不足 | Timeline 验证计算/通信 stream 排布 | bucket、prefetch、异步 collective、调度 |
| PP Bubble 高且 stage 时间不均 | stage partition 或 microbatch 不合理 | 比较各 stage 层数/算量与 schedule | 重分层、增加 microbatch、调整 schedule |
| Free 高且出现内存事件 | 内存碎片/重整或 Host 调度 | Memory + Timeline | allocator 配置、减少临时分配、融合 |

每个动作都必须回到相同输入、相同采集窗口、相同 topology 做 A/B；同时比较 step
time、未覆盖通信、payload、带宽、Free、峰值内存和数值结果，不能只看一个百分比。

## 12. 常见错误

- 把不同作业的 rank 目录放到同一个 `-d`；
- Level0 却期待通信矩阵；
- 只同步 `cluster_analysis.db` 而丢失同目录其他交付件；
- 比较 HCCS 与 RDMA 的绝对带宽；
- 看见 wait 高就宣布网络慢；
- 用 profiler-active step time 作为最终吞吐；
- 使用 benchmark `--force` 时误以为会传给 cluster；
- 在未确认磁盘和输入可信时使用 `--cluster-bypass-input-safety-checks`。
