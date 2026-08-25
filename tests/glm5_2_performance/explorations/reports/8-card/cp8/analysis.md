# CP8 NPU performance analysis

## Status and validity

CP8 completed the full Ascend PyTorch Profiler → offline parse →
`msprof-analyze` → MindStudio handoff on all eight ranks. The profiler-off run
is the throughput authority. The profiler-active run is used only to explain
the critical path because its active window adds 68.02% step time over its own
pre-profile median.

The clean screen reaches 5,037.26 tok/s/job at 1,626.28 ms/step. This is 1.93x
slower than FSDP2-CP4 and 7.43x slower than FSDP8 on the same debug-model token
budget. The result is diagnostic rather than an acceptance claim: it is one
profiler-off replicate, includes the warned physical NPU0, and uses sequence
length 128 where CP has little attention work to amortize fixed collectives.

## Experiment settings

| Item | Value |
| --- | --- |
| Model | `glm5_debugmodel`, 8 layers |
| Topology | CP8, one eight-rank context-parallel mesh |
| Device mapping | logical rank 0-7 to physical NPU 0-7 |
| Tokens | local batch 8, global batch 64, sequence length 128; 8192 tokens/job/step |
| Precision | FP32 training, BF16 parameters, FP32 reduction |
| Schedule | 20 steps; skip 8, warmup 2, active 3 |
| Profiler | Ascend PyTorch Profiler level 1, all ranks, pipe utilization, offline parse |
| Analysis | cluster, time summary, free analysis, and communication bottleneck for ranks 0-7 |

The successful run's exact driver command, generated `torchrun` command,
environment overrides, official tool calls, outputs, and return codes are in
the [run readme](../../../runs/8-card/cp8/npu-cp8-bf16-s20-l8-b64-seq128-seed61-distributed-r2-03d875ae/readme.md).

## Throughput cross-check

| Source | Median step | Job throughput | Interpretation |
| --- | ---: | ---: | --- |
| profiler-off steady window | 1,626.28 ms | 5,037.26 tok/s | screening authority; p90 drift 4.17% |
| profiler run, pre-profile | 1,723.89 ms | 4,752.03 tok/s | attribution-run baseline |
| profiler active window | 2,896.46 ms | 2,828.28 tok/s | attribution only; +68.02% step time |
| profiler run, post-profile | 2,078.31 ms | 3,941.67 tok/s | profile run remains non-stationary |

The post-profile phase does not return to the pre-profile median. Collective
counts, payloads, transit and same-window rank relationships remain valid, but
the active-window throughput is not an optimization baseline.

## Rank critical path

| Rank | Stage (ms) | Compute (ms) | Exposed communication (ms) | Free (ms) | Overlap |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 2,875.93 | 216.35 | 1,233.57 | 1,646.22 | 2.95% |
| 1 | 2,877.30 | 216.23 | 1,317.52 | 1,581.11 | 2.86% |
| 2 | 2,879.52 | 221.30 | 1,342.76 | 1,437.21 | 2.37% |
| 3 | 2,877.73 | 217.11 | 1,513.43 | 1,135.21 | 2.75% |
| 4 | 2,878.02 | 217.90 | 1,023.25 | 1,624.48 | 2.82% |
| 5 | 2,877.50 | 221.31 | 1,325.93 | 1,318.80 | 3.00% |
| 6 | 2,884.17 | 215.51 | 1,214.34 | 1,443.67 | 2.78% |
| 7 | 2,878.79 | 217.85 | 1,114.07 | 1,423.87 | 2.97% |

Stage time differs by only 8.24 ms and compute by 5.80 ms, while exposed
communication spans 1,023.25-1,513.43 ms. Ranks compensate with 1,135.21-
1,646.22 ms of `Free`, so the step is synchronized but not productively
balanced. Rank 3 is the communication tail; rank 4 is the communication floor.

## Collective accounting

| Collective | Calls/step | Payload/rank/step | Physical transit/rank/step | Effective bandwidth |
| --- | ---: | ---: | ---: | ---: |
| AllGather | 368 | 610.34 MB | 43.01-44.76 ms | 13.63-14.19 GB/s |
| ReduceScatter | 216 | 584.10 MB | 112.45-142.98 ms | 4.09-5.19 GB/s |
| tiny AllReduce | 6 | 0.00006-0.00150 MB | 0.0067-0.0154 ms | not bandwidth-relevant |
| Total | 590 | about 1,194.44 MB | 156.27-187.76 ms | - |

The physical collective transit is roughly 156-188 ms/step, far below the
1.02-1.51 s of exposed communication. CP8 therefore has two separate costs:
a real 1.19 GB/rank payload and a much larger launch/dependency wait. The
AllGather path sustains normal HCCS bandwidth; ReduceScatter is both slower and
more variable and should be analyzed first when CP optimization begins.

## Official-tool findings

All 160 exported Top-20 slow-collective rows are classified Host-bound. The
reasons repeatedly identify ranks already unaligned at the interval start and
host-side operators around enqueue, including `aten::copy_`,
`aten::as_strided`, `aten::select`, `aten::empty_strided`, grouped MM, and
post-backward registration. The longest rank-0 record is a 52.41 ms AllGather;
its reason says the compared ranks were unaligned from the beginning.

`free_analysis` exposes the same mechanism from the idle side. It reports
PyTorch no-dispatch intervals up to 91.22 ms and CANN launch gaps up to
63.90 ms, plus repeated event-wait/event-record windows. These are much larger
than an individual HCCS transit fragment and explain why exposed communication
cannot be estimated from bytes divided by link bandwidth.

The runtime also records CPU fallback for `_assert_async.msg` and AiCPU
ArgSort for INT32/INT64. They are retained as trace hypotheses, not declared
bottlenecks until their contribution to the CP8 critical path is measured.

## Potential experiments after the measurement matrix

No optimization is implemented in this phase.

1. **CP collective semantic map.** Label all 590 launches/step at the CP
   attention and DTensor placement boundaries. Separate required sequence
   exchange from redundant redistributions before changing kernels.
2. **CP degree/sequence sweep.** Compare CP2/4/8 at sequence lengths 128,
   512, and a capacity-relevant length with identical global tokens. Gate on
   profiler-off median/p90, HBM, calls, bytes, and loss/gradient behavior.
3. **ReduceScatter granularity.** Prototype grouping only for adjacent shards
   with the same dependency point through the public distributed API or the
   model parallelization boundary. Acceptance requires fewer launches and
   lower exposed time, not bandwidth alone.
4. **Host launch alignment.** Add per-rank sequence IDs and monotonic stamps
   around attention exchange, grouped MM, copy/view chains, backward hooks and
   optimizer boundaries. Optimize the first reproducible divergence.
5. **Overlap experiment.** Use a dedicated communication stream only after
   dependencies are explicit. Gate on more than the current 2.37-3.00%
   overlap without increasing HBM, allocation retries, or step variability.
6. **FSDP2-CP4 controlled comparison.** Profile it with the identical active
   window. Compare CP collective counts and process-group sizes to explain the
   measured 1.93x profiler-off gap before selecting the production topology.

## Reproduction and evidence

Run inside `glm5-npu-dev` after activating `torchtitan-0803`. Use a fresh
replicate number and an isolated HCCL child-process port range:

```bash
cd /workspace/y50064852_yyb/torchtitan-test
unset CUDA_VISIBLE_DEVICES
TORCHTITAN_MSPROF_ANALYZE=/workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli \
TORCHTITAN_MSPROF_ANALYZE_WORKERS=2 \
HCCL_NPU_SOCKET_PORT_RANGE=auto HCCL_IF_BASE_PORT=63200 \
python tests/glm5_2_performance/profiler_benchmark.py \
  --probe --device npu --topology cp8 --preset distributed \
  --visible-devices 0,1,2,3,4,5,6,7 \
  --steps 20 --skip-steps 8 --warmup-steps 2 --active-steps 3 \
  --replicate 3 --cluster
```

- Structured analysis: [analysis.json](../../../runs/8-card/cp8/npu-cp8-bf16-s20-l8-b64-seq128-seed61-distributed-r2-03d875ae/analysis.json)
- Exact process and tool commands: [run readme](../../../runs/8-card/cp8/npu-cp8-bf16-s20-l8-b64-seq128-seed61-distributed-r2-03d875ae/readme.md)
- MindStudio roots and official output inventory: the same run's `artifacts.json`
- HTML report: `performance_reports/8-card/cp8/npu-cp8-bf16-s20-l8-b64-seq128-seed61-distributed-r2-03d875ae.html`
- Profiler-off authority: [run readme](../../../runs/8-card/cp8/npu-cp8-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-486604c0/readme.md)
