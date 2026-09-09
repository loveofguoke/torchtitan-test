# Shared experiment primitives

模型配置的来源与各实验检查结果：[模型配置审计](MODEL_CONFIG_AUDIT.md)。

`glm5_2_common` is dependency-free with respect to individual experiments. It
owns canonical accelerator selection, topology definitions, topology selection,
and conflict-checked execution feature composition.

Dependency direction is one way:

```text
glm5_2_common
  <- precision
  <- performance
  <- graph execution feature
  <- precision + graph <- checkpoint / stability
  <- graph <- smoke
  <- precision + performance + graph <- combination
```

Experiment modules must not be imported by `glm5_2_common`. Features contribute
arguments and environment variables through `TrainingFeature`; only the central
combination workflow assembles them into a training command.

Performance and graph experiments share one external dependency inventory:
[性能与图模式环境、外部工具和依赖总表](PERFORMANCE_GRAPH_DEPENDENCIES_ZH.md).
It separates training runtime requirements from optional analysis/GUI tools and
is the installation authority for performance, graph, and combination reports.

## Standard experiment lifecycle

Training experiments use one reproducible input contract whenever numerical
results or restart behavior are compared:

1. `--data` creates a step-0 model checkpoint and topology-independent fixed
   token plan.
2. `--capture ...` or the experiment's training action consumes that fixture.
3. `--compare` reads portable artifacts and writes the report without an
   accelerator.

The parity, formal precision, combination/graph, checkpoint, and stability
experiments follow this contract. A fixture can be generated on either backend
and is reused across topologies when its training settings are unchanged.

The standalone performance profiler is the deliberate exception. It measures
profiling overhead and runtime behavior and does not claim numerical
comparability. Use the combination runner when fixed inputs, graph mode,
distributed execution, precision comparison, and profiling must be enabled in
the same training process.

Graph and profiler execution are currently implemented only for Ascend NPU.
Their CUDA device values are reserved public interfaces that fail explicitly
until the corresponding CUDA policies are implemented. Device-neutral eager
precision, checkpoint, stability, and smoke experiments continue to support
both CUDA and NPU.

## Shared topology selectors

Experiment CLIs that support distributed execution use the same vocabulary:

| Name | Ranks | Parallel decomposition |
|---|---:|---|
| `single` | 1 | no distributed degree |
| `ddp2`, `ddp8` | 2, 8 | DP replicate 2 or 8 |
| `fsdp8` | 8 | DP shard 8 |
| `hsdp2x4` | 8 | DP replicate 2 x DP shard 4 |
| `hsdp4x2` | 8 | DP replicate 4 x DP shard 2 |
| `tp8` | 8 | TP 8 |
| `cp8` | 8 | CP 8 |
| `pp8` | 8 | PP 8 with the shared pipeline schedule |
| `ep8` | 8 | dense FSDP 8 plus EP 8 |
| `fsdp2-tp4` | 8 | FSDP 2 x TP 4 |
| `fsdp2-cp4` | 8 | FSDP 2 x CP 4 |
| `tp2-cp4` | 8 | TP 2 x CP 4 |
| `fsdp4-tp2` | 8 | FSDP 4 x TP 2 |
| `fsdp2-pp4` | 8 | FSDP 2 x PP 4 |
| `fsdp2-tp2-pp2` | 8 | FSDP 2 x TP 2 x PP 2 |
| `fsdp2-tp4-ep8` | 8 | dense FSDP 2 x TP 4; expert region EP 8 |

`--topology NAME` selects one member, `--topology all` selects every member
advertised by that experiment, and `--topologies A,B,C` selects a subset. The
singular and plural selectors are mutually exclusive. `ddp16` and `fsdp16`
exist in the common registry as future multi-node definitions but are excluded
from current single-node `all` suites.

```bash
# Single card.
python <experiment-entry.py> <action> --topology single

# One representative distributed topology.
python <experiment-entry.py> <action> --topology fsdp8

# A focused subset.
python <experiment-entry.py> <action> --topologies ddp8,fsdp8,tp8

# Every topology advertised by that experiment.
python <experiment-entry.py> <action> --topology all
```

Each experiment README states its own default: smoke/checkpoint/stability and
standalone performance default to `single`; formal precision and the central
combination suite default to `all`; graph convenience entry points default to
`single`.

## HSDP：组内分片、组间复制

`hsdpRxS` 的两个数字表示副本组数 R 和每组分片数 S，不是版本号。
当前注册了两种八卡布局；都使用 TorchTitan 的 FSDP2 路径，
TP/CP/PP/EP degree 均为 1。它与 `fsdp2-tp4` 不同：后者的 2
表示 FSDP 分片度，4 表示张量并行度，不包含 HSDP 的副本维度。

以普通二维参数 `W` 的 global shape `[8, 4]` 为例，存储时的
placement 为 `(Replicate(), Shard(0))`，对应 `(dp_replicate, dp_shard)`。
下面按连续编号给出逻辑 mesh 示例；实际 rank 组以运行时 DeviceMesh 为准。

```text
hsdp2x4：mesh shape [2, 4]，每卡参数分片 [2, 4]
                 shard0   shard1   shard2   shard3
replicate0       rank0    rank1    rank2    rank3
replicate1       rank4    rank5    rank6    rank7

rank0 / rank4：W[0:2, :]
rank1 / rank5：W[2:4, :]
rank2 / rank6：W[4:6, :]
rank3 / rank7：W[6:8, :]

hsdp4x2：mesh shape [4, 2]，每卡参数分片 [4, 4]
                 shard0   shard1
replicate0       rank0    rank1
replicate1       rank2    rank3
replicate2       rank4    rank5
replicate3       rank6    rank7
```

前向前，在每一行的 shard 组中 AllGather 所需参数；每个 rank 用自己的
数据计算。反向梯度在 shard 组内 ReduceScatter，再沿 replicate 维度
同步对应的梯度分片（AllReduce）。例如 `hsdp2x4` 的 rank0 与 rank4
同步同一部分参数的梯度，而不是交换整份模型。这里描述逻辑通信语义，
实际 buffer 分组、异步重叠及参数重分片时机由 FSDP2 配置决定。
优化器更新本地分片后，同一列的模型状态继续一致。

| 八卡布局 | 参数分片大小（普通均匀参数） | 数据并行度 | 特点 |
|---|---|---:|---|
| `ddp8` | 完整参数 | 8 | 参数复制，梯度同步 |
| `fsdp8` | 参数的1/8 | 8 | 八卡组内分片 |
| `hsdp2x4` | 参数的1/4，每片复制2份 | 8 | 四卡分片组，加副本间同步 |
| `hsdp4x2` | 参数的1/2，每片复制4份 | 8 | 两卡分片组，加副本间同步 |

表中是参数分片比例，不是峰值显存比例；激活、临时完整参数、通信缓冲等
另外占内存。HSDP常用于将频繁参数收集限制在较快的组内链路，但单机八卡
实验不能证明多机收益，也不能预先认定它比 FSDP8 快。

两种预设的数据并行度都是 `R*S=8`。默认 local batch=8、global batch=64，
因此梯度累积次数为 `64/(8*8)=1`；序列长度128时，每步总token数为8192。

运行方法见 [smoke HSDP示例](../glm5_2_smoke/README.md#hsdp-smoke-coverage)。
所有使用共用注册表的实验均能选择这两个名称；`all`按各实验可用列表展开。
新增拓扑不修改旧成员的配置或结果身份，补跑不用加 `--force`。
每个成员照常记录配置、起止时间、耗时、状态和日志；功能通过不代表精度
或性能通过。NPU当前仍需先解决单卡FlexAttention设备接入失败。

## Output names

Each experiment root already identifies the artifact type, so child names do
not repeat `fixture`, `combo`, `stability`, `checkpoint`, or the model name.
Readable settings are followed by an eight-character digest of only the values
that can change capture results. Report thresholds and presentation settings
are excluded, so report-only changes never require another training run.

When a pre-digest output directory contains the same stored training contract,
the workflow renames it to the current config-digested name and continues. A
different contract is never adopted silently.

## Rerun and generation rules

Experiment output is resumed per suite member:

- Without `--force`, a completed member is reused only when its stored
  contract belongs to the current fixture generation. Failed, incomplete, or
  incompatible output is archived and that member is retried.
- `--force` starts a new generation for the complete selected range. Every
  selected member is removed before the first training process starts; this
  prevents a mid-suite failure from mixing newly captured members with
  untouched output from an older run.
- After a forced suite stops part way through, rerun the same command without
  `--force`. Completed members from that generation are skipped and execution
  continues at the first incomplete member.
- `--data --force` creates a new fixture generation. Captures tied to an older
  generation are not finalized or reused, even when their files are complete.

Long-running mutable experiments use the shared `RunAttempt` lifecycle. Each
run directory contains `run_state.json` with an attempt ID, orchestrator PID,
context, and `running`/`failed`/`completed` status. Runtime logs record the same
attempt ID where the experiment owns log creation. Before `--force` removes
anything, the complete selected range is preflighted for live orchestrators;
all selected run/artifact/report/input-contract paths and their exact-name
`.previous-*`/`.failed-*` siblings are then removed, printed, and verified
absent before the first new process starts. An interrupted command
is resumed without `--force`: completed members are retained, while incomplete
members are archived or replaced as one unit.

This policy applies to precision (including graph and combination captures),
performance/profiler, stability, checkpoint, and smoke. Parity artifacts remain
immutable and graph-debug probes create timestamped directories, so those two
workflows never overwrite an existing generation and intentionally do not add a
mutable `--force` lifecycle.

Use `--topology NAME --force` to replace only one topology. Use `--topology
all --force` only when the whole suite should start over.
