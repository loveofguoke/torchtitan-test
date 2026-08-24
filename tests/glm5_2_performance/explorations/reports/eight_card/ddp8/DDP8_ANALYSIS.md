# DDP8 NPU performance analysis

## Status

This is diagnostic evidence, not an acceptance result. The run uses physical
devices `0,1,2,3,4,5,6,7`, and NPU0 currently reports a health warning. The
profiler-off throughput repeat is also not converged, as described below.

## Experiment settings

| Item | Value |
| --- | --- |
| Model | `glm5_debugmodel`, 8 layers |
| Topology | DDP8, 8 local HCCL ranks |
| Device mapping | logical rank 0-7 to physical NPU 0-7 |
| Tokens | local batch 8, global batch 64, sequence length 128; 8192 tokens/job/step |
| Precision | FP32 training, BF16 parameters, FP32 reduction |
| Schedule | 20 steps; skip 8, warmup 2, active 3 |
| Profiler | Ascend PyTorch Profiler level 1, all ranks, pipe utilization, offline parse |
| Analysis | `msprof-analyze cluster -m all`, `cluster_time_summary`, `free_analysis`, and `communication_bottleneck` for ranks 0-7 |

The exact driver and `torchrun` argv are in the run's `command_history.jsonl`
and `manifest.json`. Every official analysis command, stdout/stderr, and return
status is in `tool_commands/`.

## Reproduction

Run inside `glm5-npu-dev` after activating `torchtitan-0803`:

```bash
cd /workspace/y50064852_yyb/torchtitan-test
unset CUDA_VISIBLE_DEVICES
TORCHTITAN_MSPROF_ANALYZE=/workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli \
python tests/glm5_2_performance/profiler_benchmark.py \
  --probe --device npu --topology ddp8 --preset distributed \
  --visible-devices 0,1,2,3,4,5,6,7 \
  --steps 20 --skip-steps 8 --warmup-steps 2 --active-steps 3 \
  --replicate 1 --cluster
```

Profiler-off authority run:

```bash
python tests/glm5_2_performance/profiler_benchmark.py \
  --probe --device npu --topology ddp8 --preset distributed --profiler-off \
  --visible-devices 0,1,2,3,4,5,6,7 \
  --steps 20 --skip-steps 8 --warmup-steps 2 --active-steps 3 \
  --replicate 3
```

Use a new replicate number for every rerun. The `replicate 3` command is the
next planned idle-system confirmation; it has not yet been used in this report.

## Throughput validity

| Source | Median step | p90 step | Job throughput | Interpretation |
| --- | ---: | ---: | ---: | --- |
| profiler-off replicate 2 | 773.35 ms | 1,341.89 ms | 10,960.70 tok/s | Contaminated: p90 is 73.52% above median |
| profiler run, non-active steps | 219.14 ms | - | 37,382.71 tok/s | Useful cross-check, not throughput authority |
| profiler active window | 300.45 ms | - | 27,265.42 tok/s | Attribution only; 37.11% instrumentation overhead |

The fourfold disagreement means DDP8 throughput is unresolved. No speedup or
topology ranking should use the profiler-off DDP8 row until at least two
idle-system profiler-off repeats agree. The active trace remains valid for
critical-path attribution because all ranks were captured in the same window.

The first profiler-off attempt failed at concurrent device initialization on
physical NPU4/6/7 with `507033: TsdOpen failed`. Sequential `set_device` probes
for all three devices passed immediately afterwards, so this is retained as a
transient TSD-startup failure rather than a permanent card exclusion.

## Rank critical path

The following values are medians over captured steps 10-12 from the official
cluster step trace.

| Rank | Stage (ms) | Compute (ms) | Exposed communication (ms) | Free (ms) | Exposed / stage |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 302.08 | 42.17 | 14.07 | 231.63 | 4.66% |
| 1 | 301.97 | 41.63 | 22.85 | 226.65 | 7.57% |
| 2 | 298.13 | 41.51 | 34.21 | 222.61 | 11.47% |
| 3 | 302.26 | 42.09 | 45.72 | 200.93 | 15.13% |
| 4 | 304.20 | 41.82 | 131.17 | 131.21 | 43.12% |
| 5 | 301.64 | 41.63 | 88.65 | 168.19 | 29.39% |
| 6 | 298.73 | 42.04 | 122.64 | 133.64 | 41.05% |
| 7 | 296.44 | 41.51 | 121.83 | 136.49 | 41.10% |

Compute is balanced within 0.66 ms, while exposed communication differs by
117.10 ms (9.32x). Rank 4 is the worst exposed-communication rank; ranks 4-7
form a distinct high-wait group. The stage times remain close because ranks
with less exposed communication spend more time in the `Free` category. This
is a rank scheduling/launch alignment problem, not a compute-load imbalance.

## Collective accounting

| Metric | Rank range |
| --- | ---: |
| AllReduce calls | 17/step on every rank |
| Payload | 116.49-117.90 MB/rank/step |
| Physical transit | 8.58-8.89 ms/step |
| Effective bandwidth | 13.11-13.58 GB/s |
| Median exposed communication | 14.07-131.17 ms/step |

Physical transit varies by only 0.31 ms. Therefore payload reduction can save
at most the physical transit component unless it also changes launch ordering;
it cannot explain or remove the current 117 ms rank skew by itself.

## Official-tool findings

`communication_bottleneck` classifies the top slow AllReduce operations as
`Host-bound`. All 160 exported Top-20 rows (20 per rank) receive that
classification; none is classified as network-bound. Reason text contains
`GroupedMmBackward0` in 72 rows, `AddBackward0` in 61, and
`MatmulBackward0` in 35 (categories can overlap within a row). Across ranks the
tool reports 10-21 ms collective start-time skew near:

- `AddBackward0` and `MatmulBackward0`;
- `GroupedMmBackward0`;
- `ToCopyBackward0`, `aten::empty`, `aten::add_`, and `aten::as_strided`;
- `Event::record`, `Event::wait`, and `c10d::allreduce_`;
- the first `FSDP::pre_forward (layers.2)` boundary.

`free_analysis` exposes repeated approximately 4.55-4.70 ms PyTorch periods
with no task dispatch and 5.9-11.2 ms device-side event/memcpy/wait gaps. These
are concrete launch-pipeline targets. The raw CSV is retained, because the
current official free-analysis export selected rank 6 and should not be
generalized to all ranks without per-rank confirmation.

## Potential optimization experiments (not implemented)

Each item below is a separate patch/API experiment. It must first be measured
with profiler-off repeats and then attributed with the same all-rank window.

1. **Bucket-count and launch-order control.** Add an experiment-only DDP/FSDP
   configuration hook immediately after model parallelization. Sweep bucket
   cap and layer grouping while recording collective sequence IDs. Acceptance:
   no loss/gradient change, fewer than 17 launches/step or lower start skew,
   and improved idle-system median/p90.
2. **Dedicated HCCL stream overlap.** Inject through the public FSDP2
   `FSDPModule.set_all_reduce_hook` path, or a narrowly scoped distributed hook
   in the source repository. Record events on the compute stream, enqueue HCCL
   on one communication stream, and wait only at optimizer consumption. Do not
   call private HCCL symbols from the test repository. Acceptance: exposed
   communication falls without increasing physical transit or memory retries.
3. **Reduction dtype A/B.** The performance harness already exposes
   `--mixed-precision-reduce bfloat16`, which reaches
   `MixedPrecisionPolicy(reduce_dtype=...)`. Use it only as a performance
   prototype. It halves FP32 payload, but the expected upper bound is about
   8.9 ms/step unless launch skew also changes.
4. **Host-launch skew instrumentation.** Add record functions around
   `GroupedMmBackward0`, gradient norm, DDP/FSDP pre-forward, and collective
   enqueue. Emit per-rank monotonic timestamps and collective sequence IDs to
   one JSONL file per rank. This identifies the first divergence rather than
   only observing the final collective wait.
5. **Launch-gap kernel/fusion candidates.** Prototype only the operator chains
   immediately preceding a proven gap: GroupedMM backward epilogues,
   cast/add/view chains, and gradient-norm finite checks. A custom NPU op,
   `torch.library` decomposition, or Triton-NPU kernel is acceptable only after
   its exact shapes appear in the trace. Reject it if kernel time improves but
   collective start skew or end-to-end p90 does not.
6. **Topology-aware rank mapping.** Repeat DDP8 with a cyclic rank-to-device
   mapping while keeping the same physical cards. If the high-wait group follows
   physical devices, investigate HCCS/card health; if it follows logical ranks,
   investigate host launch order and process scheduling. This is diagnostic and
   must not be presented as a production optimization.

## Evidence locations

- Structured analysis: `explorations/runs/npu-ddp8-bf16-s20-l8-b64-seq128-seed61-distributed-r1-488247ce/analysis.json`
- Reproduction and tool commands: the same run directory's
  `command_history.jsonl`, `manifest.json`, and `tool_commands/`
- MindStudio import roots and official output inventory: `artifacts.json`
- Raw official CSVs: the matching directory under `performance_runs/`
- HTML performance report:
  `performance_reports/npu-ddp8-bf16-s20-l8-b64-seq128-seed61-distributed-r1-488247ce.html`
- Failed TSD-startup attempt:
  `explorations/runs/npu-ddp8-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-270760c1/`
