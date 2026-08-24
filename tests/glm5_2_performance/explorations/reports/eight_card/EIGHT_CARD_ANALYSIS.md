# GLM-5.2 eight-card NPU topology analysis

## Executive summary

All 13 declared eight-rank performance topologies have completed one
profiler-off screening run. FSDP8 and EP8 are the strongest clean signals:
37,406.66 and 30,677.33 tok/s/job, with p90 step drift of 1.21% and 2.11%.
Tensor-parallel degree is the clearest negative scaling signal. Pipeline-only
and context-only shapes are also inefficient at this small debug-model/token
scale. DDP8 throughput is unresolved because its profiler-off run was highly
variable and conflicts with the profiler run's non-active window.

These are diagnostic comparisons. They include physical NPU0, which currently
reports a health warning, and most topologies have only one successful
profiler-off replicate. No optimization or acceptance claim is made here.

## Controlled configuration

Every row uses the same model and token budget:

- `glm5_debugmodel`, 8 layers;
- 20 steps, with steps 9-20 forming the 12-step steady-state summary;
- local batch 8, global batch 64, sequence length 128;
- 1024 tokens/rank/step and 8192 tokens/job/step;
- FP32 training, BF16 parameter storage, FP32 collective reduction;
- physical devices `0,1,2,3,4,5,6,7`;
- Profiler completely disabled for screening throughput.

Job throughput is derived as `8192 / median_step_seconds`, equivalently eight
times the reported per-rank throughput. p90 drift is
`(p90_step / median_step - 1) × 100%`. Peak active HBM is the maximum metrics
sample on the metrics rank, not a sum across ranks.

The generated complete table and visualization are in
`EIGHT_CARD_TOPOLOGY_REPORT.md` and
`performance_reports/eight_card/npu-eight-card-topology-comparison.html`.

## Main topology signals

### FSDP and EP are the leading candidates

FSDP8 reaches 37,406.66 tok/s/job at 219.00 ms/step. EP8 reaches 30,677.33
tok/s/job at 267.04 ms/step, so FSDP8 is 1.22x faster. Both are stable and use
less than 0.63 GiB active HBM on this debug model. They are the first two
topologies to confirm with idle-system repeats and all-rank attribution.

EP8 here is not evidence that all-to-all is cheap at production scale. The
debug model, short sequence, and small payload can make expert sharding look
favorable. The EP profiler run must report AllToAll count, payload, transit,
and rank imbalance before extrapolation.

### Tensor parallelism is the strongest structural bottleneck

Holding eight ranks while increasing TP degree produces an almost geometric
loss:

| Shape | TP degree | Job throughput | Median step |
| --- | ---: | ---: | ---: |
| FSDP4-TP2 | 2 | 7,250.90 tok/s | 1,129.79 ms |
| FSDP2-TP4 | 4 | 3,671.06 tok/s | 2,231.55 ms |
| TP8 | 8 | 1,852.98 tok/s | 4,421.03 ms |

Each doubling of TP degree nearly halves throughput. Runtime warnings report
two sequential AllReduce operations while flattening partial DTensor mesh
dimensions. This gives a testable attribution hypothesis: excessive small,
serialized TP reductions and redistributions dominate the debug-model work.
The TP8 profiler must separate AllReduce transit from host enqueue gaps before
any fusion or HCCL change is proposed.

Adding EP8 to FSDP2-TP4 lowers throughput from 3,671.06 to 3,526.44 tok/s/job
(3.94% lower). At this scale, EP does not hide the TP4 critical path.

### Context parallelism benefits from an FSDP dimension

FSDP2-CP4 reaches 9,743.17 tok/s/job versus 5,037.26 for CP8, a 1.93x gain.
This suggests the pure CP8 collective granularity exceeds useful attention
work at sequence length 128. The next profiler comparison should measure CP
AllGather/ReduceScatter count and payload for both shapes, with identical
active windows.

TP2-CP4 falls to 1,813.47 tok/s/job, 5.37x below FSDP2-CP4. The difference
isolates TP2 as a larger risk than CP4 in this workload.

### Pipeline parallelism is dominated by bubble and composition overhead

PP8 reaches 4,020.93 tok/s/job at 2,037.35 ms/step. FSDP2-PP4 improves this by
only 1.17x to 4,697.69 tok/s/job. Both are slower than CP-based shapes despite
very low rank-local active HBM (0.018 and 0.044 GiB on the metrics rank).

FSDP2-TP2-PP2 is the slowest topology at 895.08 tok/s/job and 9.15 s/step. Its
p90 is stable, so this is repeatable structural overhead rather than a single
outlier. A PP profiler should decompose fill/drain bubble, send/recv, TP
collectives, and stage imbalance. Increasing microbatch count is only a future
A/B plan; it is not justified until that decomposition exists.

### Memory is not the current limiter

All recorded peak active HBM values are below 0.8 GiB/rank because this is the
debug model. The fastest FSDP8 row uses 0.621 GiB, while several slow topologies
use less. Current rankings therefore reflect scheduling, collective frequency,
and fixed overhead more than capacity pressure. Activation checkpointing and
memory scheduling need a larger-model/longer-sequence experiment; this matrix
cannot justify them.

## DDP8 attribution

The completed all-rank DDP8 profile reports:

- compute balanced at 41.51-42.17 ms/rank;
- exposed communication at 14.07-131.17 ms/rank;
- 17 AllReduce calls/step and 116.49-117.90 MB/rank/step;
- physical transit of 8.58-8.89 ms/step at 13.11-13.58 GB/s;
- official slow-collective classification: Host-bound, with 10-21 ms launch
  skew around backward, GroupedMM, cast/view/add, event, and collective enqueue.

The complete derivation is in `ddp8/DDP8_ANALYSIS.md`. The immediate action is
measurement, not optimization: repeat profiler-off DDP8 on an idle system and
test whether high-wait ranks follow logical rank or physical device mapping.

## Remaining profiler matrix

The profiler-off matrix covers all declared eight-rank shapes. All-rank
Profiler + official analysis is complete for DDP8. The minimum attribution set
to cover distinct communication patterns is:

| Priority | Topology | Primary question |
| ---: | --- | --- |
| 1 | FSDP8 | ReduceScatter/AllGather count, overlap, and why it leads throughput |
| 2 | TP8 | serialized AllReduce/redistribution cost and host launch gaps |
| 3 | EP8 | AllToAll count, payload, routing imbalance, and AiCPU routing work |
| 4 | CP8 and FSDP2-CP4 | CP collective granularity and the 1.93x topology gap |
| 5 | PP8 | stage balance, send/recv time, and pipeline bubble |
| 6 | FSDP2-TP2-PP2 | compound PP/TP critical path behind the 9.15 s step |

Each run uses the same all-rank three-step active window and runs the official
cluster, time-summary, free, and per-rank communication-bottleneck analyses.

## Potential optimization plan after attribution

No item below is implemented in this phase. Each is deliberately paired with a
measurement gate and a narrow injection point.

1. **FSDP collective granularity:** configure layer grouping/wrapping at the
   source repository's model parallelization boundary. Gate on fewer
   ReduceScatter/AllGather launches, unchanged numerical behavior, and lower
   profiler-off median/p90.
2. **Communication overlap:** use FSDP2's public collective hook or a dedicated
   communication stream with explicit compute-stream events. Gate on lower
   exposed communication without higher memory, retries, or physical transit.
3. **TP redistribution removal:** patch the DTensor placement transition that
   warns about two sequential AllReduce operations. Compare a flattened mesh or
   direct target placement. Gate on one collective sequence replacing two and
   unchanged shard semantics.
4. **EP routing path:** retain INT32-compatible routing where possible, replace
   proven AiCPU ArgSort/scatter segments, and prototype TopK-mask fusion only
   for trace-confirmed shapes. Gate on AiCPU disappearance and end-to-end gain.
5. **PP schedule/microbatches:** expose microbatch count and schedule in the
   performance harness. Gate on lower measured bubble at unchanged global
   tokens, not on stage-local kernel time alone.
6. **Host launch alignment:** add per-rank JSONL timestamps around backward
   GroupedMM, norm/finite checks, collective enqueue, and optimizer boundaries.
   Gate on locating and then reducing the first rank divergence.
7. **Reduction dtype prototype:** use the existing
   `--mixed-precision-reduce bfloat16` performance-only path. Treat physical
   transit as the upper-bound opportunity and keep it separate from accuracy
   acceptance.
8. **Activation checkpoint trade-off:** defer to a capacity-relevant model.
   Sweep checkpoint modes only when HBM pressure is measurable, and report
   compute, communication, and peak-HBM deltas together.

## Reproducibility and evidence

- Every execution gets a unique run directory under `explorations/runs/`.
- `command_history.jsonl` preserves the exact driver command, container path,
  environment overrides, and generated `torchrun` command.
- `manifest.json`, `metrics.jsonl`, and `analysis.json` preserve configuration,
  raw step metrics, and structured derivations.
- `tool_commands/*.json` preserves every official analysis invocation and its
  stdout/stderr/return status.
- `artifacts.json` inventories MindStudio-importable roots and official output.
- The failed DDP8 TSD initialization attempt is retained, not overwritten.
- Aggregate report-generation commands are appended under
  `explorations/history/`.

