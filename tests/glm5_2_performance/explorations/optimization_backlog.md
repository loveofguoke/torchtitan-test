# GLM-5.2 性能优化任务清单

本文件把已经保存的 NPU profiler 证据转换成可独立交付的优化任务。它不是性能承诺，
也不会把 `ops_candidate` 暴露成训练参数。每个任务必须单独做 profiler-off A/B、短窗口
归因和正式精度验证；失败原型同样保留 run、log 和结论。

## 统一边界

| 仓库 | 允许承担的内容 |
|---|---|
| `torchtitan` | 通用 GLM-5 数学实现、placement、公共并行边界；未验证实现只放 `models/glm5/ops_candidate` |
| `TorchTitanTurbo` | Triton-Ascend、CANN/HCCL、NPU stream/graph 适配；未验证实现只放 `models/glm5/ops_candidate` |
| `torchtitan-test` | A/B 开关、Profiler、不可变 run、分析、报告和验收；不能替代生产训练逻辑 |

公共约束：默认关闭；不调用私有 HCCL 符号；不同时修改多个拓扑机制；不改变 reduction
dtype、token plan、checkpoint 或 loss 语义来换取速度；Profiler-active 只用于归因，
profiler-off 才用于吞吐比较。

## 证据基线

- 总结：[`reports/summary.md`](reports/summary.md) 和
  [`reports/8-card/summary.md`](reports/8-card/summary.md)。
- TP8：1,177 collective/step，物理传输约 19-22 ms，暴露通信 2.31-3.82 s；见
  [`reports/8-card/tp8/analysis.md`](reports/8-card/tp8/analysis.md)。
- CP8 对 FSDP2-CP4：590 对 299 calls/step，约 1,194 对 580 MB/rank/step，
  CP8 clean step 慢 1.93 倍；见 [`cp8`](reports/8-card/cp8/analysis.md) 和
  [`fsdp2-cp4`](reports/8-card/fsdp2-cp4/analysis.md)。
- PP8：P2P 物理传输不超过 0.43 ms/rank/step，暴露等待最高 3.50 s；见
  [`reports/8-card/pp8/analysis.md`](reports/8-card/pp8/analysis.md)。
- EP8：AllToAllV payload 最大/最小为 1.56 倍，物理传输约 1.86-2.56 ms，
  暴露通信 335-811 ms；见 [`reports/8-card/ep8/analysis.md`](reports/8-card/ep8/analysis.md)。

这些 W0 数据包含异常物理 NPU0，且多数 profiler-off 只有一次成功重复，因此只能决定
“下一步测什么和原型放哪里”，不能作为生产验收数字。

## 当前原型状态

| 任务 | 状态 | 已落地代码 |
|---|---|---|
| PERF-001 Host launch skew | 待实现 instrumentation | 暂不在没有统一 collective ID 的情况下埋零散 range |
| PERF-002 TP redistribution audit | isolated prototype | `torchtitan/.../ops_candidate/communication/redistribution_audit.py` |
| PERF-003 FSDP layer grouping | isolated prototype | `torchtitan/.../ops_candidate/communication/fsdp_layer_grouping.py` |
| PERF-004 CP degree/granularity | experiment backlog | 复用现有 topology profiler，不修改模型 |
| PERF-005 PP partition/bubble | isolated prototype | `torchtitan/.../ops_candidate/pipeline/stage_partition.py` |
| PERF-006 EP placement/overlap | isolated prototype | `torchtitan/.../ops_candidate/moe/expert_placement.py` 和 Turbo `hccl_overlap_plan.py` |
| PERF-007 INT32 top-k mask | unregistered Triton candidate | Turbo `ops_candidate/compute/triton_topk_mask.py` |
| PERF-008 recompute/workspace | isolated prototype | TorchTitan `selective_recompute.py` 和 Turbo `graph_workspace_plan.py` |

`isolated prototype` 表示代码可审阅、纯规划逻辑可单测，但没有被 model、override、
benchmark 或普通 patch 导入；`unregistered Triton candidate` 还必须在服务器执行 operator
probe，不能因本地 `compileall` 通过就视为后端可用。

## P0：先补齐因果证据

### PERF-001 Host launch skew instrumentation

- 证据：TP8 的 160 条慢 collective 中 151 条 Host-bound；CP8/EP8 的 160 条均为
  Host-bound，附近反复出现 copy/view、GroupedMM、norm 和 event。
- 目标仓库：`torchtitan-test`。
- 准确注入边界：训练 step 中 backward、optimizer、metrics；GLM DSA backward；
  `torchtitan.distributed.utils::clip_grad_norm_`；
  `torchtitan.distributed.fsdp::apply_fsdp_to_decoder` 生成的 FSDP 单元。
- 原型：给每个 rank 写相同 collective sequence ID、单调时钟、range name 和 tensor
  摘要；Ascend 侧使用 MSTX range，JSONL 是无 profiler 时的低成本对照。
- 风险：range 自身扰动 host launch；rank 发生条件分支后 sequence ID 不一致。
- 通过：定位第一个稳定的 rank divergence，并且 instrumentation-off 的训练行为不变。
- 失败：只能看到“某 rank 更慢”，但无法映射到相同语义 collective。

### PERF-002 TP/组合拓扑 DTensor redistribution audit

- 证据：FSDP4-TP2、FSDP2-TP4、TP8 的 AllReduce 为 224、440、865 次/step；运行时
  明确警告多轴 `Partial -> Replicate` 使用两个串行 AllReduce。
- 目标仓库：先在 `torchtitan/models/glm5/ops_candidate/communication/`
  使用 `redistribution_audit.py`；证明后才考虑公共 placement 改动。
- 准确源码边界：
  `torchtitan.models.glm5.parallelize::parallelize_glm5`、
  `torchtitan.models.glm5.sharding` 的 attention/MoE sharding config、
  `torchtitan.distributed.spmd_types::spmd_redistribute_per_axis`。
- 原型：审计所有 active partial mesh axis 是否在同一依赖点变为 Replicate，估算扁平
  process group 能否把 N 次 reduction 变成一次；不执行 collective。
- 风险：不同轴的 reduction order、process group 和 gradient placement 可能不等价。
- 通过：语义 trace 确认候选点；正式 patch 后相关序列从 2 次变 1 次，placement、loss、
  grad norm 和 checkpoint 不变。

## P1：通信粒度、拓扑和流水

### PERF-003 FSDP contiguous layer grouping

- 证据：FSDP8 仅 39 collective/step 并以 37,406.66 tok/s/job 领先；细粒度 FSDP/CP
  组合具有更多 AllGather/ReduceScatter 和 host wait。
- 目标仓库：`torchtitan/models/glm5/ops_candidate/communication/`
  的 `fsdp_layer_grouping.py`；NPU A/B 开关最终放 Turbo/test。
- 准确源码边界：`torchtitan.distributed.fsdp::apply_fsdp_to_decoder` 中逐 block
  `fully_shard`；参考
  `megatron/core/distributed/fsdp/src/megatron_fsdp/param_and_grad_buffer.py::BucketingPolicy`、
  `Bucket`、`ParamAndGradBuffer`，以及
  `deepspeed/runtime/zero/stage3.py::IPGBucketZ3`。
- 原型：只合并相邻且 `policy_key` 相同的 dense/MoE FSDP 单元，输出预计 calls 和峰值
  unsharded bytes。
- 风险：更高 HBM、较晚 reshard、损失 backward/communication overlap；不能跨 dense/
  expert mesh 或 pipeline stage。
- 通过：calls 至少下降 2 倍，profiler-off median/p90 改善，rank max/median exposed
  communication 下降，且 HBM 在预算内。

### PERF-004 CP degree and ReduceScatter granularity

- 证据：seq=128 时 FSDP2-CP4 将 CP8 的 calls/bytes 约减半，clean step 快 1.93 倍；
  CP8 ReduceScatter 仅约 4.09-5.19 GB/s，且 overlap 只有约 2.37-3.00%。
- 目标仓库：拓扑 sweep 在 `torchtitan-test`；只有证明冗余 placement 后，公共改动才进入
  `torchtitan.models.glm5.parallelize::parallelize_glm5` 或 sharding config。
- 准确边界：GLM attention 的 CP input/output wrapper、
  `torchtitan.distributed.spmd_types::spmd_redistribute_per_axis`。
- 实验：CP2/4/8 x seq 128/512/容量长度，固定 global tokens、模型、checkpoint 和
  profiler schedule。
- 通过：报告性能/HBM Pareto frontier；不能只因 W0 的 seq=128 就默认 CP4。

### PERF-005 PP measured stage partition and microbatch sweep

- 证据：PP8 P2P 传输小于 0.43 ms，而暴露等待可达 3.50 s，说明问题是 fill/drain、
  readiness 和 stage work，不是链路带宽。
- 目标仓库：`torchtitan/models/glm5/ops_candidate/pipeline/stage_partition.py`
  做离线规划；A/B 参数和 trace 解析放 `torchtitan-test`。
- 准确参考：TorchTitan `torchtitan/distributed/pipeline_parallel.py::pipeline_llm`、
  `_build_pipeline_schedule`、`_pipeline_module_split`；
  `megatron/core/pipeline_parallel/schedules.py::forward_backward_pipelining_without_interleaving`、
  `forward_backward_pipelining_with_interleaving`；
  `deepspeed/runtime/utils.py::partition_balanced` 和
  `deepspeed/runtime/pipe/engine.py::PipelineEngine`。
- 原型：使用实测 forward+backward cost 做 contiguous dynamic-programming partition；
  同时输出理想 non-interleaved 1F1B bubble fraction。
- 风险：参数量不等于计算量；embedding/loss、通信和 activation HBM 必须计入。
- 通过：固定 global tokens 下做 microbatch 8/16/32 和受支持 schedule A/B，降低实测
  bubble、median/p90，且不靠增加总计算量伪造吞吐。

### PERF-006 EP load evidence and overlap

- 证据：EP8 AllToAllV payload 差 1.56 倍，但物理传输仅约 2 ms，暴露等待 335-811 ms，
  overlap 仅约 2.04-2.38%。
- 目标仓库：离线 placement what-if 在
  `torchtitan/models/glm5/ops_candidate/moe/expert_placement.py`；NPU stream 双缓冲在
  `TorchTitanTurbo/.../ops_candidate/communication/hccl_overlap_plan.py`。
- 准确参考：TorchTitan
  `experiments/graph_trainer/ep_eager_chunk.py::maybe_apply_ep_overlap_eager_chunking`、
  `ep_overlap_pass.py::ep_overlap_schedule_pass`；Megatron
  `megatron/core/transformer/moe/token_dispatcher.py::MoEAlltoAllTokenDispatcher`、
  `moe_utils.py::permute`/`unpermute`。
- 原型：先记录每 expert/rank token；离线用 hot-expert greedy placement 估算理论下界；
  再对 dispatch/compute/combine 做固定 chunk、双缓冲、显式 event 的 isolated A/B。
- 风险：动态改变 expert owner 会破坏 checkpoint/optimizer/process-group contract，
  所以 placement 原型绝不能直接变成每 step 运行策略。
- 通过：token ordering 和 grad 不变，rank payload max/median、exposed communication、
  profiler-off step 同时改善；只提高 overlap 百分比不算通过。

## P2：确定 shape 后再写 Triton/CANN 候选

### PERF-007 INT32 top-k to additive mask

- 证据：运行时出现 INT32/INT64 ArgSort AiCPU 和 Scatter/Cast/ReduceSum 家族；当前还没
  证明它们在 critical path 上，因此优先级低于 launch skew。
- 目标仓库：`TorchTitanTurbo/models/glm5/ops_candidate/compute/`
  的 `triton_topk_mask.py`，不注册 override。
- 准确模型边界：`torchtitan.models.glm5.model::DSAIndexerTopK.forward` 生成 top-k，
  `Glm5DsaIndexer.forward` 形成 mask；工程参考
  `slime_plugins/models/glm5/glm5.py` 内 `fused_select_topk` 和
  `slime_plugins/models/glm5/ops/indexer.py::lighting_indexer`。
- 原型：保留 INT32 indices，以两个 Triton-Ascend kernel 完成 mask fill 和 selected
  position 写入，避免 CANN INT64 scatter；仍未融合 TopK 自身。
- 风险：packed-document mask、重复 index、dynamic K、Triton-Ascend graph capture、
  kernel launch 数。
- 通过：operator probe 完全匹配；ArgSort/Scatter fallback 和 cast launch 减少；
  DSA 区间及端到端 step 改善。若只把一个慢 op 换成两个更慢 kernel，则淘汰。

### PERF-008 selective recompute and graph-stable workspace

- 证据：W0 active HBM 约 1 GiB，当前不能据此宣称 AC 有收益；正式模型才需要测
  no/selective/full AC Pareto frontier。
- 目标仓库：通用 cost model 在
  `torchtitan/models/glm5/ops_candidate/recompute/selective_recompute.py`；NPU 静态地址
  计划在 `TorchTitanTurbo/.../ops_candidate/memory/graph_workspace_plan.py`。
- 准确参考：`torchtitan.distributed.activation_checkpoint::FullAC`、`SelectiveAC`、
  `MemoryBudgetAC`；Megatron `megatron/core/recompute.py::checkpointed_forward`；
  DeepSpeed `deepspeed/runtime/activation_checkpointing/checkpointing.py::partition_activations`
  和 `checkpoint`。
- 约束：DSA top-k、MoE routing 和通信结果属于离散/分布式控制流，默认保存而不是重算。
- 通过：正式模型 peak HBM 降到目标预算，额外 FLOPs/通信可解释，loss/grad 和离散
  路径满足精度标准，graph 不新增 recompile。

## 每张任务卡的执行模板

```text
1. 固定三个仓库 branch/commit、环境、设备健康和基线 run。
2. profiler-off 按 A/B/A 或 B/A/B 交错重复至少 3 次。
3. 另跑 profiler-active 归因，不把 active-window throughput 当收益。
4. 比较 median、p90、collective count/payload/transit、exposed communication、
   HBM、rank min/median/max、loss/grad norm。
5. 保存命令、runtime.log、manifest、analysis.json、HTML 和失败记录。
6. 机制没有出现在 trace、数值语义变化、或收益小于噪声时，原型保持 candidate/淘汰，
   不迁入 ops。
```

基础复现命令沿用各 topology `analysis.md` 中的完整命令。新运行必须增加 replicate，
不能覆盖本目录已有证据。
