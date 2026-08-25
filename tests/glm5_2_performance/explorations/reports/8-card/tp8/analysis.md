# TP8 NPU performance analysis

## Status and validity

The TP8 all-rank profile completed the full Ascend PyTorch Profiler → offline
parse → `msprof-analyze` → MindStudio handoff. It is diagnostic evidence, not
an acceptance run. The profiler-off run is the throughput authority; the
profiled run changed phase behavior substantially, so its active-window timing
must not be used as a speed comparison.

The profiler-off screen reaches 1,852.98 tok/s/job at a 4,421.03 ms median
step. Its p90 is 4,668.07 ms, 5.59% above the median. The profiled run changes
from 9,097.91 ms before profiling to 6,289.25 ms in the active window and
4,797.79 ms afterwards. This inversion is evidence of a non-stationary run,
not negative profiler overhead. Collective structure and within-window rank
relationships remain useful for attribution.

## Experiment settings

| Item | Value |
| --- | --- |
| Model | `glm5_debugmodel`, 8 layers |
| Topology | TP8, one eight-rank tensor-parallel mesh |
| Device mapping | logical rank 0-7 to physical NPU 0-7 |
| Tokens | local batch 8, global batch 64, sequence length 128; 8192 tokens/job/step |
| Precision | FP32 training, BF16 parameters, FP32 reduction |
| Schedule | 20 steps; skip 8, warmup 2, active 3 |
| Profiler | Ascend PyTorch Profiler level 1, all ranks, pipe utilization, offline parse |
| Analysis | cluster, time summary, free analysis, and communication bottleneck for ranks 0-7 |

The complete commands, resolved `torchrun` argv, and official tool return
codes are in the [run readme](../../../runs/8-card/tp8/npu-tp8-bf16-s20-l8-b64-seq128-seed61-distributed-r1-dba2190c/readme.md).

## Throughput cross-check

| Source | Median step | Job throughput | Interpretation |
| --- | ---: | ---: | --- |
| profiler-off steady window | 4,421.03 ms | 1,852.98 tok/s | screening authority; p90 drift 5.59% |
| profiler run, pre-profile | 9,097.91 ms | 900.43 tok/s | contaminated/non-stationary |
| profiler active window | 6,289.25 ms | 1,302.54 tok/s | attribution only |
| profiler run, post-profile | 4,797.79 ms | 1,707.45 tok/s | confirms phase drift |

TP8 is 20.19x slower than the FSDP8 screen and 3.91x slower than FSDP4-TP2.
Across the eight-rank screen, doubling TP degree from 2 → 4 → 8 nearly halves
job throughput at each step. The profile below explains this as collective
launch granularity and synchronization, not device compute imbalance.

## Rank critical path

The official cluster summary covers captured steps 10-12.

| Rank | Stage (ms) | Compute (ms) | Exposed communication (ms) | Free (ms) | Overlap |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 6,305.11 | 280.50 | 3,495.36 | 2,529.26 | 0.012% |
| 1 | 6,261.07 | 285.20 | 3,284.34 | 2,691.52 | 0.013% |
| 2 | 6,310.40 | 278.88 | 3,001.64 | 3,978.03 | 0.014% |
| 3 | 6,306.15 | 280.64 | 2,306.83 | 3,737.44 | 0.012% |
| 4 | 6,301.23 | 284.13 | 3,453.23 | 2,229.69 | 0.014% |
| 5 | 6,306.14 | 280.71 | 3,816.10 | 2,209.33 | 0.015% |
| 6 | 6,302.85 | 278.20 | 3,720.32 | 2,304.33 | 0.014% |
| 7 | 6,308.12 | 285.63 | 3,661.59 | 2,360.90 | 0.013% |

Compute differs by only 7.43 ms (2.67%), while exposed communication spans
2,306.83-3,816.10 ms (1.65x). Stage time remains within 49.33 ms because ranks
with less exposed communication spend more time in `Free`. Communication and
free time exchange positions across ranks; compute is not the source of the
cross-rank imbalance.

## Collective accounting

| Collective | Calls/step | Payload/rank/step | Physical transit/rank/step | Effective bandwidth |
| --- | ---: | ---: | ---: | ---: |
| AllReduce | 865 | 146.80-171.17 MB | 11.24-14.08 ms | 12.15-13.06 GB/s |
| ReduceScatter | 192 | 55.05 MB | 4.11-4.30 ms | 12.80-13.40 GB/s |
| AllGather | 120 | 55.05 MB | 3.61-3.67 ms | 14.98-15.25 GB/s |
| Total | 1,177 | 256.90-281.27 MB | 18.96-22.05 ms | - |

The measured physical HCCS transit is only about 19-22 ms/step, while exposed
communication is 2.31-3.82 seconds. Payload or link bandwidth alone therefore
cannot explain TP8. The dominant opportunity is reducing 1,177 collective
launches per step and their synchronization/placement-transition waits.

Rank 0 carries 24.37 MB less AllReduce payload than ranks 1-7. This is retained
as a topology-specific asymmetry requiring semantic mapping; it must not be
silently averaged away.

## Official-tool findings

Across the 160 exported Top-20 slow-collective rows, 151 are classified as
Host-bound and 9 as Device-bound. Slow operations span 17.41-43.81 ms, with a
25.83 ms mean. The reason text identifies task-start misalignment and host-side
operators near collective enqueue:

- `aten::copy_` in 19 rows and `aten::as_strided` in 10;
- `GroupedMmBackward0` in 17 rows and `ViewBackward0` in 5;
- optimizer/device-side `ApplyAdamW` context in the 9 Device-bound rows;
- repeated cases where tasks are already unaligned at the beginning of the
  comparison interval.

`free_analysis` selects rank 6. Its largest finding is 28.52 ms of PyTorch
idle time with no task dispatch. It also reports a 14.85 ms CANN gap between
launches and repeated 10.6-11.8 ms launch/event/memcpy gaps. These values are
direct evidence for a host launch pipeline problem, but they apply to the
selected rank and should not be generalized to all ranks without per-rank
free-analysis support.

The runtime log also records two known fallback signals: `_assert_async.msg`
falls back to CPU and INT32/INT64 `ArgSort` runs on AiCPU. They are valid future
targets only after their duration is mapped to the TP8 critical path.

## Potential experiments after the measurement matrix

No optimization is implemented in this phase. The following are narrow,
measurable follow-ups.

1. **DTensor redistribution audit.** Instrument placement transitions in the
   GLM5 TP parallelization plan and assign a semantic label to each collective.
   Target the warning path that performs two sequential AllReduce operations
   while flattening partial mesh dimensions. Acceptance is a lower collective
   count with identical placements and training numerics.
2. **Collective coalescing prototype.** At the model-specific TP plan or a
   Turbo patch boundary, group only adjacent reductions with identical
   dependency points. Do not call private HCCL symbols from the test harness.
   Gate on fewer than 1,177 launches/step and lower profiler-off median/p90.
3. **Sequence-parallel A/B.** Expose a topology-safe experiment toggle and
   compare TP2/4/8 with identical global tokens. Record redistribution count,
   activation HBM, and loss/gradient parity before accepting any result.
4. **Host launch alignment.** Emit per-rank monotonic timestamps and collective
   sequence IDs around `GroupedMmBackward0`, copy/view chains, optimizer, and
   collective enqueue. The first diverging sequence is the optimization point.
5. **AiCPU routing cleanup.** Prototype a supported dtype or a fused TopK/index
   path in Turbo only after kernel tables confirm ArgSort duration. Gate on
   AiCPU disappearance and end-to-end gain, not kernel replacement alone.
6. **Work-size scaling.** Repeat TP degree scaling on W1/W2 shapes. The current
   debug model has too little compute to amortize 1,177 launches, so it is a
   launch-stress diagnostic rather than a production TP verdict.

## Reproduction

Run inside `glm5-npu-dev` with `torchtitan-0803` activated:

```bash
cd /workspace/y50064852_yyb/torchtitan-test
unset CUDA_VISIBLE_DEVICES
TORCHTITAN_MSPROF_ANALYZE=/workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli \
python tests/glm5_2_performance/profiler_benchmark.py \
  --probe --device npu --topology tp8 --preset distributed \
  --visible-devices 0,1,2,3,4,5,6,7 \
  --steps 20 --skip-steps 8 --warmup-steps 2 --active-steps 3 \
  --replicate 2 --cluster
```

Use a new replicate number for every rerun. Replicate 2 is the next idle-system
confirmation; this report uses profiler-active replicate 1 and profiler-off
replicate 1.

## Evidence locations

- Structured analysis: [analysis.json](../../../runs/8-card/tp8/npu-tp8-bf16-s20-l8-b64-seq128-seed61-distributed-r1-dba2190c/analysis.json)
- Exact process and tool commands: [run readme](../../../runs/8-card/tp8/npu-tp8-bf16-s20-l8-b64-seq128-seed61-distributed-r1-dba2190c/readme.md)
- MindStudio roots and official output inventory: the same run's `artifacts.json`
- HTML report: `performance_reports/8-card/tp8/npu-tp8-bf16-s20-l8-b64-seq128-seed61-distributed-r1-dba2190c.html`
- Profiler-off authority: [run readme](../../../runs/8-card/tp8/npu-tp8-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-64da983b/readme.md)
