# PP8 NPU performance analysis

## Status and validity

PP8 completed all-rank Ascend PyTorch Profiler collection, offline parsing,
official `msprof-analyze`, and MindStudio handoff. Its profiler-off screen is
the throughput authority: 4,020.93 tok/s/job at 2,037.35 ms/step with only
1.18% p90 drift. The active profile uses a 3,634.53 ms median step and is for
attribution only; its own pre-profile and post-profile medians are 2,849.37
and 2,775.31 ms, respectively.

This debug workload deliberately stresses fixed pipeline cost: eight model
layers are split across eight stages and use eight microbatches. The result
does not establish production PP efficiency, but it does expose bubble,
point-to-point scheduling, and stage imbalance clearly.

## Experiment settings

| Item | Value |
| --- | --- |
| Model | `glm5_debugmodel`, 8 layers |
| Topology | PP8, one layer per pipeline stage |
| Schedule | 1F1B, 8 microbatches |
| Device mapping | logical rank 0-7 to physical NPU 0-7 |
| Tokens | local batch 8, global batch 64, sequence length 128; 8192 tokens/job/step |
| Precision | FP32 training, BF16 parameters, FP32 reduction |
| Profile window | 20 steps; skip 8, warmup 2, active 3 |
| Analysis | cluster, time summary, free analysis, and communication bottleneck ranks 0-7 |

Exact commands and output inventories are in the [run readme](../../../runs/8-card/pp8/npu-pp8-bf16-s20-l8-b64-seq128-seed61-distributed-r2-462adeae/readme.md).

## Throughput cross-check

| Source | Median step | Job throughput | Interpretation |
| --- | ---: | ---: | --- |
| profiler-off steady window | 2,037.35 ms | 4,020.93 tok/s | screening authority |
| profiler run, pre-profile | 2,849.37 ms | 2,875.02 tok/s | attribution-run baseline |
| profiler active window | 3,634.53 ms | 2,253.94 tok/s | attribution only; +27.56% |
| profiler run, post-profile | 2,775.31 ms | 2,951.75 tok/s | profile run remains slower than clean screen |

FSDP2-PP4 reaches 4,697.69 tok/s/job in the clean screen, only 1.17x faster;
FSDP8 is 9.30x faster. The debug model has too little stage-local work to
amortize pipeline orchestration.

## Rank critical path

Across captured steps 10-12, stage time spans 1,894.32-3,551.44 ms, compute
77.36-401.46 ms, exposed communication 131.78-3,497.76 ms, and `Free`
55.80-3,082.25 ms. This is qualitatively different from DDP/FSDP/CP: pipeline
ranks are expected to execute different schedules, so rank equality is not the
goal. The useful signal is that device compute is a small fraction of the
stage window while exposed point-to-point wait and free bubble exchange across
stages.

Rank 1 has the longest stage and highest compute/free combination; rank 7 has
the largest exposed communication. A stage timeline in MindStudio is needed
to distinguish fill/drain bubble from a genuinely slow stage before changing
the schedule.

## Point-to-point accounting

The parsed communication table distinguishes `BatchSendRecv`, `Send`, and
`Receive` per stage. Because stage roles differ, counts are intentionally not
averaged into one misleading job value.

| Direction | Calls/step by rank | Payload/rank/step | Transit/rank/step | Effective bandwidth |
| --- | --- | ---: | ---: | ---: |
| BatchSendRecv | 9.33, 25.33, 41.33, 57.33, 73.33, 89.33, 105.33, 57.33 | 0.52-6.82 MB | 0.032-0.430 ms | 14.58-16.16 GB/s |
| Receive | 56, 104, 88, 72, 56, 40, 24, 8 | 0.52-6.82 MB | 0.033-0.416 ms | 15.73-16.37 GB/s |
| tiny AllReduce | 2 on every rank | at most 0.000016 MB | at most 0.0042 ms | not bandwidth-relevant |

Physical point-to-point transit is below half a millisecond/rank/step, while
exposed communication reaches seconds. The bottleneck is dependency and
pipeline readiness, not HCCS bandwidth or payload volume. Wait percentages
are usually above 92% for receive/batched operations, which is consistent with
stages waiting for peers rather than links transmitting slowly.

## Official-tool findings

The official communication-bottleneck exports contain 26 applicable slow rows
and classify all of them Host-bound. `free_analysis` identifies rank 0 and
reports PyTorch no-dispatch gaps of 37.35 and 30.55 ms, a 5.85 ms CANN launch
gap, and repeated event wait/record windows. Those local gaps are real, but
they are still much smaller than the multi-second cross-stage bubble; schedule
structure must be quantified first.

## Potential experiments after the measurement matrix

No optimization is implemented in this phase.

1. **Bubble decomposition.** Derive per-microbatch forward/backward/send/recv
   intervals from the trace and report fill, steady 1F1B, and drain separately.
2. **Microbatch sweep.** Expose 8/16/32 microbatches at fixed global tokens.
   Gate on lower bubble fraction and profiler-off median/p90 while tracking
   stage activation HBM.
3. **Schedule A/B.** Compare supported 1F1B/interleaved schedules only after
   the eight-layer stage assignment is explicit. Preserve the same model,
   tokens and optimizer semantics.
4. **Stage rebalance.** Use measured compute per stage to move layer or
   embedding/loss work; do not infer balance from parameter count alone.
5. **P2P overlap.** Use public pipeline APIs/streams and explicit events only
   after the dependency graph is known. Gate on lower exposed wait without a
   larger bubble or additional copies.
6. **FSDP2-PP4 attribution.** Compare stage bubble and FSDP collectives in the
   matching active window to explain why its clean gain is only 1.17x.

## Reproduction and evidence

```bash
cd /workspace/y50064852_yyb/torchtitan-test
unset CUDA_VISIBLE_DEVICES
TORCHTITAN_MSPROF_ANALYZE=/workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli \
TORCHTITAN_MSPROF_ANALYZE_WORKERS=2 \
HCCL_NPU_SOCKET_PORT_RANGE=auto HCCL_IF_BASE_PORT=63232 \
python tests/glm5_2_performance/profiler_benchmark.py \
  --probe --device npu --topology pp8 --preset distributed \
  --visible-devices 0,1,2,3,4,5,6,7 \
  --steps 20 --skip-steps 8 --warmup-steps 2 --active-steps 3 \
  --replicate 3 --cluster
```

- Structured analysis: [analysis.json](../../../runs/8-card/pp8/npu-pp8-bf16-s20-l8-b64-seq128-seed61-distributed-r2-462adeae/analysis.json)
- Exact process and tool commands: [run readme](../../../runs/8-card/pp8/npu-pp8-bf16-s20-l8-b64-seq128-seed61-distributed-r2-462adeae/readme.md)
- MindStudio inventory: the same run's `artifacts.json`
- HTML report: `performance_reports/8-card/pp8/npu-pp8-bf16-s20-l8-b64-seq128-seed61-distributed-r2-462adeae.html`
- Profiler-off authority: [run readme](../../../runs/8-card/pp8/npu-pp8-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-02473d85/readme.md)
