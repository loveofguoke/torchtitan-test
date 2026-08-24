# GLM-5.2 NPU performance exploration

Date: 2026-08-24

This report covers the W0 `glm5_debugmodel` experiments only. It establishes
the profiler and analysis workflow and produces optimization prototypes; it is
not evidence for the formal A100/910B2 G3 target.

## Tool flow and experiment conditions

The completed flow follows the official three stages:

1. Ascend PyTorch Profiler: scheduled all-rank collection, offline parsing.
2. `msprof-analyze`: cluster summary, cluster-time breakdown,
   communication-bottleneck analysis, and free-time analysis.
3. MindStudio Insight: both complete `*_ascend_pt` roots and the generated
   `cluster_analysis_output` are ready to import.

NPU0 was excluded from communication experiments because `npu-smi` reported
`Health Status: Warning`, error `819B8605`, and an HCCS lane-drop safety state.
The clean two-rank capture used physical NPU1 and NPU2. During later repeat and
BF16 attribution runs, another eight-card job increased HBM on NPU1/2 from
about 3.5 GiB to about 27 GiB and drove AICore utilization close to 100%.
Those later throughput samples are marked as contended; their profile payload
and call-count data remain useful.

Primary outputs:

- [FP32-reduction run and complete process](../runs/2-card/ddp2/npu-ddp2-bf16-s30-l8-b64-seq128-seed61-distributed-23bade6d/readme.md)
- FP32 cluster DB: `performance_runs/2-card/ddp2/npu-ddp2-bf16-s30-l8-b64-seq128-seed61-distributed-23bade6d/cluster/cluster_analysis_output/cluster_analysis.db`
- [BF16-reduction run and complete process](../runs/2-card/ddp2/npu-ddp2-bf16-s30-l8-b64-seq128-seed61-distributed-reduce-bf16-33b5a0e5/readme.md)
- BF16 cluster DB: `performance_runs/2-card/ddp2/npu-ddp2-bf16-s30-l8-b64-seq128-seed61-distributed-reduce-bf16-33b5a0e5/cluster/cluster_analysis_output/cluster_analysis.db`

## Clean FP32-reduction diagnosis

Profiler-off is the throughput authority. The clean profiler run is used for
attribution only:

| Metric | Value |
| --- | ---: |
| Baseline throughput per device | 5,522.66 tok/s |
| Baseline job throughput | 11,045.31 tok/s |
| Baseline median step | 741.67 ms |
| Active profiler overhead | 31.13% |
| Rank median compute | 162.53 / 162.71 ms |
| Rank median exposed communication | 234.61 / 31.27 ms |
| Rank median device free time | 592.91 / 777.71 ms |

Compute is balanced to about 0.1%, while exposed communication differs by
7.5x. The critical problem is therefore rank timing and communication launch
skew, not unequal model compute.

The communication files show:

| Metric | Rank 0 | Rank 1 |
| --- | ---: | ---: |
| AllReduce calls per step | 50 | 50 |
| HCCS payload per step | 265.93 MB | 267.40 MB |
| Physical transit per step | 13.72 ms | 13.82 ms |
| Effective bandwidth | 19.38 GB/s | 19.35 GB/s |

`cluster_time_summary` independently reports about 15.3-15.6 ms of transmit
stage time per step, while communication-wait stage time reaches 240-262 ms on
rank 0 in several steps. Link transfer is a small part of the exposed delay.

The official `communication_bottleneck` Top-10 results classify every selected
AllReduce as host-bound. Reported causes include tasks not aligned from the
beginning and start-time differences around `aten::to`,
`GroupedMmBackward0`, and `aten::linalg_vector_norm` of roughly 9.7-10.9 ms.
`free_analysis` also finds PyTorch intervals with no task dispatched for about
4.5-4.9 ms, plus 13-19 ms holes containing event record/wait and asynchronous
memory-copy tasks.

Memory-copy time in the clean cluster-time summary is about 0.8-1.3 ms per
step and non-overlapped memory time is below 0.81 ms. Memory is not the W0
critical path. The runtime still reports the CANN 32-byte-padding allocation
warning, so allocation count and fragmentation should be revisited on the
formal model where HBM pressure is representative.

## BF16 reduction prototype

TorchTitan's public config currently restricts `mixed_precision_reduce` to
FP32. The performance harness now widens only the runtime dataclass metadata in
the isolated capture process; it does not modify the generic Trainer or
TorchTitan config source. The value reaches GLM-5's existing
`MixedPrecisionPolicy(reduce_dtype=...)` path.

The profiled candidate proves the expected communication reduction:

| Metric | FP32 reduction | BF16 reduction | Change |
| --- | ---: | ---: | ---: |
| AllReduce calls per step | 50 | 50 | unchanged |
| Rank 0 payload per step | 265.93 MB | 132.96 MB | -50.0% |
| Rank 1 payload per step | 267.40 MB | 133.70 MB | -50.0% |
| Rank 0 physical transit | 13.72 ms | 8.05 ms | -41.3% |
| Rank 1 physical transit | 13.82 ms | 7.75 ms | -43.9% |

One uncontended profiler-off BF16 run reached a median 5,617.09 tok/s/device.
FP32 profiler-off repeats ranged from 2,838.38 to 4,524.68 tok/s/device because
the eight-card job started during the A/B/A sequence. This is insufficient for
a final speedup claim. Repeat at least three interleaved FP32/BF16 runs with
stable pre/post `npu-smi` snapshots before accepting a throughput delta.

## Health-safe four-card topology screening

The next exploration used four healthy cards and a fixed global budget of
8,192 tokens/step (`local_batch=8`, `global_batch=64`, `seq=128`). NPU0 remained
excluded. Profiler-off runs supply the throughput numbers below; no model
optimization was enabled.

| Topology | Median step | tok/s/device | tok/s/job | Peak active HBM | vs DDP4 job throughput |
| --- | ---: | ---: | ---: | ---: | ---: |
| DDP4 | 389.92 ms | 5,252.34 | 21,009.35 | 0.843 GiB | reference |
| FSDP4 | 401.75 ms | 5,097.71 | 20,390.85 | 0.660 GiB | -2.94% |
| EP4 | 477.19 ms | 4,291.94 | 17,167.77 | 0.658 GiB | -18.29% |
| CP4 | 1,615.86 ms | 1,267.44 | 5,069.75 | 0.232 GiB | -75.87% |
| FSDP2 x TP2 | 2,204.28 ms | 929.22 | 3,716.87 | 0.404 GiB | -82.31% |
| PP4 | 3,175.36 ms | 644.97 | 2,579.88 | 0.066 GiB | -87.72% |
| TP4 | 4,330.00 ms | 472.98 | 1,891.92 | 0.290 GiB | -90.99% |

These values characterize the W0 short-sequence debug workload, not the formal
model. TP/CP/PP each use a data-parallel degree of one, so the fixed global
budget requires eight microbatches per step; DDP4 and FSDP4 require two. The
result therefore exposes both communication granularity and insufficient work
per model-parallel rank. It does not show that model parallelism is generally
inferior at the sequence lengths and model sizes for which it is intended.

The complete comparable table, formulas, evidence paths, and generated chart
are in [4-card/comparison.md](4-card/comparison.md) and
`performance_reports/4-card/comparison.html`.

### DDP4 versus FSDP4 attribution

Both captures have balanced device compute: DDP4 spans 81.67–83.30 ms/rank and
FSDP4 spans 81.64–82.87 ms/rank. Exposed communication is highly skewed in
both: 42.22–223.38 ms for DDP4 and 19.42–294.37 ms for FSDP4.

| Topology / collective | Calls/step | Payload MB/rank/step | Physical transit/step |
| --- | ---: | ---: | ---: |
| DDP4 AllReduce | 28 | 199.82 | 12.14 ms |
| FSDP4 AllGather | 42 | 103.99 | 5.75 ms |
| FSDP4 ReduceScatter | 22 | 100.00 | 8.75 ms |
| FSDP4 scalar AllReduce | 7 | <0.002 | about 0.015 ms |

FSDP4 reduces active HBM by about 21.7% and loses only 2.94% profiler-off job
throughput on W0. It does not reduce communication-launch count: the profile
contains 64 data collectives plus seven tiny scalar AllReduces per step. The
official bottleneck reports classify the selected DDP4 and FSDP4 collectives
as host-bound. DDP4 shows start skew around `MatmulBackward0`, `Event::wait`,
`aten::add_`, and `FSDP::pre_forward`; FSDP4 shows `aten::_grouped_mm`,
`aten::copy_`, `aten::empty`, `AddBackward0`, and `ToCopyBackward0`, with
start differences reaching roughly 18–20 ms.

`free_analysis` finds recurring PyTorch no-dispatch gaps around 4.2–5.7 ms and
event/memcpy gaps up to roughly 7–9 ms. These measurements reinforce that rank
arrival and host launch order dominate over raw HCCS transit.

### FSDP4 physical-card failure isolation

FSDP4 profiler capture failed twice when physical NPU4 was present. With
`1,2,3,4`, NPU4/rank3 was the only unconnected rank (`EI0015`). With
`4,5,6,7`, NPU4 moved to rank0 and the remaining ranks timed out waiting for
rank0's communicator ID. The same experiment succeeded on `1,2,3,5`.

This failure follows physical NPU4 rather than logical rank. Exact commands,
tracebacks, and the successful control are indexed in
[failures.md](failures.md). NPU4 must be excluded from formal FSDP data
until its HCCL/link path is diagnosed.

## Implementable optimization ladder

### P0: reduced-precision gradient communication

Status: prototype implemented and payload reduction measured.

Injection:

```bash
python tests/glm5_2_performance/profiler_benchmark.py \
  --probe --device npu --topology ddp2 --preset distributed \
  --profiler-off --mixed-precision-reduce bfloat16 \
  --visible-devices 1,2 --replicate 1
```

Promotion gate: stable profiler-off repeats first, then a separate numerical
acceptance run. The latter is outside this exploration's scope.

### P1: reduce fine-grained collective launches

Payload compression does not change collective count. DDP2 has 50 AllReduces
per step and DDP4 has 28; FSDP4 changes the pattern to 42 AllGathers, 22
ReduceScatters, and seven tiny AllReduces. Add a performance-only or
TorchTitanTurbo FSDP granularity control at
`torchtitan/distributed/fsdp.py::apply_fsdp_to_decoder`, for example
`fsdp_group_layers={1,2,4,8}`. Group consecutive transformer blocks into one
`fully_shard` unit, then measure:

- AllGather and ReduceScatter calls per step, initially targeting at least a
  2x launch-count reduction;
- total bytes per step for a fixed reduction dtype;
- rank max/median exposed communication;
- additional peak HBM and loss of backward/communication overlap.

This is safer than a raw HCCL replacement because FSDP retains ownership of
gradient scaling and stream ordering. If a lower-level prototype is needed,
use `FSDPModule.set_all_reduce_hook` to dispatch on a dedicated HCCL stream and
return only after an event is recorded on the compute stream. Do not delay a
buffer beyond its FSDP lifetime merely to coalesce it.

### P1: remove host-side launch skew

The automatic analysis points to per-rank start-time skew rather than HCCS
bandwidth. Instrument `optim`, `collect_dist_metrics`, DSA backward, and each
FSDP AllReduce with MSTX ranges. Then prototype, in order:

1. cache device-resident scalar tensors instead of constructing or copying
   them with per-step `aten::to`;
2. separate gradient-norm computation from clipping and overlap local norm
   work with the tail of backward before its world reduction;
3. keep metrics `.item()` and logging after the optimizer's critical path;
4. compare a dedicated HCCL stream plus compute-stream event dependency with
   the default FSDP stream policy.

Acceptance requires lower rank max/median exposed communication without
increasing stage time or producing unmatched collective order.

### P1: remove DSA index/scatter AiCPU fallback

The model returns TopK indices as INT32 and later converts them to INT64 for
`scatter`. CANN reports INT32/INT64 `ArgSort` on AiCPU, and the overview trace
shows large Cast, ReduceSum, Scatter, RepeatInterleave, and TopK families.

Prototype an NPU-only function in TorchTitanTurbo with this interface:

```python
def dsa_topk_mask(
    index_scores_qk: torch.Tensor,
    topk: int,
    attention_mask_qk: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return INT32 top-k indices and the additive sparse mask."""
```

Implementation choices, in increasing effort:

1. keep INT32 indices and use `torch_npu.npu_scatter` with tensor updates;
2. register the above as a `torch.library.custom_op` so compile/export sees a
   single boundary;
3. fuse TopK index production, boolean-mask construction, and additive-mask
   application in an Ascend C/custom kernel.

The first success criterion is disappearance of the ArgSort AiCPU warning and
fewer Cast/Scatter launches. The second is lower DSA indexer plus sparse-mask
time in a kernel preset, with identical output shapes and dtypes.

### P2: fusion, recomputation, and topology matrix

- Fuse adjacent Cast/ReduceSum and mask/scatter chains only after shapes are
  stable in the kernel preset. Keep a Python reference and dispatch the custom
  path through TorchTitanTurbo rather than forking the generic Trainer.
- W0 active memory is only about 0.95-0.97 GiB, so activation checkpointing is
  not useful here. On the formal model, sweep no AC, selective AC, and full AC;
  report throughput, peak HBM, recompute time, and exposed communication as a
  Pareto frontier.
- TP4, CP4, and PP4 need scale sweeps before kernel work. Sweep sequence length,
  global token budget, and microbatch count while holding one variable at a
  time. For PP, compare `1F1B` with schedules supported by the current Trainer
  and report the theoretical bubble fraction beside measured idle time. For
  CP, attribute DSA ring/P2P time. For TP and FSDP2 x TP2, verify whether the
  DTensor warning about two sequential norm AllReduces can be removed by a
  flattened `fsdp` x `tp` mesh without changing reduction order.
- EP4 should be re-profiled at realistic tokens-per-expert. Attribute router,
  AllToAll, permute, grouped GEMM, and unpermute separately before proposing a
  fused operator or communication overlap.
- The diagnostic eight-card matrix is complete, but repeat acceptance runs
  only when all required devices have healthy links. NPU0's current HCCS state
  prevents treating this node's eight-card comparison as acceptance evidence.

## Cross-rank synthesis

The hierarchy above now covers all recorded 1/2/4/8-card experiments. The
single-card set has profiler attribution only and therefore no authoritative
profiler-off throughput. Two- and four-card values use the healthy-card
experiments; eight-card values are diagnostic because NPU0 is included.

| Scope | Representative topology | Median step | tok/s/job | Interpretation |
| --- | --- | ---: | ---: | --- |
| 1 card | single | - | - | profiler attribution only |
| 2 cards | DDP2 | 741.67 ms | 11,045.31 | clean FP32-reduction baseline |
| 4 cards | DDP4 | 389.92 ms | 21,009.35 | healthy-card screening leader |
| 4 cards | FSDP4 | 401.75 ms | 20,390.85 | 21.7% lower active HBM than DDP4 |
| 8 cards | FSDP8 | 219.00 ms | 37,406.66 | stable diagnostic leader |
| 8 cards | EP8 | 267.04 ms | 30,677.33 | stable diagnostic runner-up |
| 8 cards | DDP8 | 773.35 ms | 10,960.70 | unresolved; 73.52% p90 drift |

The DDP2→DDP4 job-throughput ratio is 1.90x for a 2x rank increase. The
FSDP4→FSDP8 diagnostic ratio is 1.83x. These ratios suggest useful scaling for
data sharding, while TP/CP/PP fixed overhead dominates this small, short-
sequence model. They are not a hardware-acceptance or full-model scaling
claim. See the [four-card comparison](4-card/comparison.md),
[eight-card comparison](8-card/comparison.md), and each topology's
`experiment.md` for evidence and run-level reproduction.

## Measurement rules retained by the harness

- profiler-off steady state is authoritative throughput;
- profiler active windows are attribution-only;
- per-device and job throughput are both reported;
- distributed tables use rank min/median/max and non-overlapped communication;
- physical transport payload excludes mirrored SDMA and duplicate CANN output;
- `--replicate N` creates repeat identities without overwriting captures;
- pre/post device snapshots record HBM, AICore utilization, health, and process
  occupancy for future runs;
- complete `*_ascend_pt` roots and cluster DBs remain the MindStudio Insight
  handoff, not the generated HTML alone.
