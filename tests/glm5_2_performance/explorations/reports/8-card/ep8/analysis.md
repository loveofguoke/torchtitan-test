# EP8 NPU performance analysis

## Status

Profiler-active replicate 4 completed the Ascend PyTorch Profiler → offline
parse → `msprof-analyze` → MindStudio workflow for all eight ranks. Replicates
1-3 are retained as failed environment/resource attempts. The independent
profiler-off run remains throughput authority at 30,677.33 tok/s/job; this
profile is attribution evidence only.

The successful run uses `HCCL_NPU_SOCKET_PORT_RANGE=auto` and an isolated
`HCCL_IF_BASE_PORT=63200`, because other containers owned the default NPU NIC
port 16666. Both values are recorded in `command_history.jsonl` and apply only
to the child process.

## Configuration and timing

| Item | Value |
| --- | --- |
| Model | `glm5_debugmodel`, 8 layers |
| Topology | FSDP8 + EP8 (`dp_shard=8`, `ep=8`) |
| Tokens | 8192 tokens/job/step; sequence length 128 |
| Precision | FP32 training, BF16 parameters, FP32 reduction |
| Profile | all ranks; skip 8, warmup 2, active 3; offline parse |

| Source | Median step | Job throughput | Use |
| --- | ---: | ---: | --- |
| profiler-off screen | 267.04 ms | 30,677.33 tok/s | screening authority |
| profile pre-window | 874.31 ms | 9,369.64 tok/s | concurrency-contaminated |
| active window | 1,019.82 ms | 8,032.79 tok/s | attribution only |
| post-window | 882.95 ms | 9,278.00 tok/s | phase cross-check |

The profiled run is much slower than the earlier screen, so it cannot validate
the screen's absolute throughput. Within its active window, however, stage
times are tightly aligned at 1,020.99-1,039.43 ms.

## Critical path and imbalance

Compute is balanced at 41.47-45.02 ms/rank. Exposed communication ranges from
334.89 ms on rank 7 to 810.78 ms on rank 3. Low-wait ranks spend the difference
in `Free` time (154.04-643.61 ms), leaving stage time balanced. Communication
overlap is only 2.04%-2.38%.

| Collective | Calls/step | Payload/rank/step | Transit/rank/step | Observation |
| --- | ---: | ---: | ---: | --- |
| AllToAllV | 35 | 21.17-33.11 MB | 1.86-2.56 ms | 1.56x payload spread indicates routed-token imbalance |
| AllGather | 21 | 22.15 MB | 1.47-1.55 ms | stable, about 14.3-15.0 GB/s |
| ReduceScatter | 11 | 19.80 MB | 4.44-5.80 ms | lower 3.4-4.5 GB/s |
| AllReduce | 8 | 0.0001-0.0015 MB | 0.013-0.018 ms | tiny payload, but 57%-99% wait |

Physical collective transit totals only about 7.8-9.9 ms/rank/step, far below
335-811 ms of exposed communication. EP payload imbalance is real, but link
time is not the main critical path in this debug workload. The dominant signal
is host/synchronization alignment around many collectives, especially the tiny
AllReduces.

All 160 Top-20 rows from official `communication_bottleneck` are Host-bound.
Reason text repeatedly reports tasks unaligned at the beginning; named nearby
operators include `aten::as_strided` (14 rows), `aten::copy_` (6), and
`GroupedMmBackward0` (3). The runtime also reports CPU fallback for
`_assert_async.msg` and AiCPU ArgSort for INT32/INT64.

## Potential experiments

No optimization is implemented in this phase.

1. Emit routed-token send/receive counts per expert and rank beside the trace,
   then correlate the 1.56x AllToAllV payload spread with exposed wait.
2. Name the eight tiny AllReduces and group only semantically identical
   synchronization points through the model/FSDP boundary.
3. Prototype dispatch/combine overlap on a dedicated communication stream in
   Turbo, with explicit compute-stream events and unchanged token ordering.
4. Evaluate INT32-compatible routing or fused TopK/dispatch only after the
   ArgSort duration is confirmed in kernel tables.
5. Repeat EP8 on W1/W2 shapes: this debug model is too small to extrapolate
   expert communication efficiency to production.

## Reproduction and evidence

- [Complete run process](../../../runs/8-card/ep8/npu-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-r4-ccb6f22c/readme.md)
- [Structured analysis](../../../runs/8-card/ep8/npu-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-r4-ccb6f22c/analysis.json)
- Raw/MindStudio inventory: the same run's `artifacts.json`
- HTML: `performance_reports/8-card/ep8/npu-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-r4-ccb6f22c.html`

Use replicate 5 for any repeat; never overwrite the successful replicate 4 or
the retained failed attempts.
