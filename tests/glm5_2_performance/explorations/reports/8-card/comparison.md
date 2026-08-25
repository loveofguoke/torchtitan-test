# GLM-5.2 NPU 8-card topology report

Generated: 2026-08-25T14:13:21.040559+08:00

## Scope and comparability

This table selects NPU distributed runs with `steps=20`, `world_size=8`, and the distributed preset; profiler-off screening uses `latest successful` replicate, while attribution uses the latest successful replicate. Exact physical device mappings are retained per run. The eight-card runs include NPU0, which currently reports a health warning; therefore those results are diagnostic comparisons, not healthy-hardware acceptance evidence. Profiler-off runs are throughput evidence; profiler-active runs are attribution only.

Job throughput is `world_size × rank throughput`, equivalently the configured global token budget divided by median step time. Peak HBM is the maximum `memory/max_active(GiB)` sample on the metrics rank.

## Validity constraints

- The first DDP8 profiler-off attempt (`replicate=1`) failed during concurrent TSD initialization on physical NPU4/6/7 (`507033: TsdOpen failed`). Sequential set-device probes then passed; the table therefore records successful `replicate=2` and retains the failed run directory as evidence.
- DDP8 replicate 2 has 73.52% p90 drift and disagrees with the non-active portion of the profiler run. Treat its throughput as contaminated until an idle-system repeat converges; the rank/collective attribution remains useful.
- Every topology currently has one authoritative profiler-off replicate at most. Rankings are screening results and need repeated idle-system confirmation before acceptance.

## Profiler-off topology screening

| Topology | Degrees (DP-repl/DP-shard/TP/CP/PP/EP) | Replicate | Median / p90 / max step (ms) | p90 drift | tok/s/device | tok/s/job | Peak active HBM (GiB) | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| fsdp8 | 1/8/1/1/1/1 | 1 | 219.00 / 221.66 / 221.90 | 1.21% | 4,675.83 | 37,406.66 | 0.621 | [analysis.json](../../runs/8-card/fsdp8/npu-fsdp8-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-4a88f56a/analysis.json) |
| ep8 | 1/8/1/1/1/8 | 1 | 267.04 / 272.66 / 315.53 | 2.11% | 3,834.67 | 30,677.33 | 0.609 | [analysis.json](../../runs/8-card/ep8/npu-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-50c6107d/analysis.json) |
| ddp8 | 8/1/1/1/1/1 | 2 | 773.35 / 1,341.89 / 1,767.16 | 73.52% | 1,370.09 | 10,960.70 | 0.783 | [analysis.json](../../runs/8-card/ddp8/npu-ddp8-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r2-80606b65/analysis.json) |
| fsdp2-cp4 | 1/2/1/4/1/1 | 1 | 840.80 / 858.67 / 869.82 | 2.13% | 1,217.90 | 9,743.17 | 0.201 | [analysis.json](../../runs/8-card/fsdp2-cp4/npu-fsdp2-cp4-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-d7029385/analysis.json) |
| fsdp4-tp2 | 1/4/2/1/1/1 | 1 | 1,129.79 / 1,157.37 / 1,160.71 | 2.44% | 906.36 | 7,250.90 | 0.372 | [analysis.json](../../runs/8-card/fsdp4-tp2/npu-fsdp4-tp2-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-e3971d37/analysis.json) |
| cp8 | 1/1/1/8/1/1 | 1 | 1,626.28 / 1,694.16 / 1,726.96 | 4.17% | 629.66 | 5,037.26 | 0.130 | [analysis.json](../../runs/8-card/cp8/npu-cp8-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-486604c0/analysis.json) |
| fsdp2-pp4 | 1/2/1/1/4/1 | 1 | 1,744.04 / 1,875.22 / 1,889.89 | 7.52% | 587.21 | 4,697.69 | 0.044 | [analysis.json](../../runs/8-card/fsdp2-pp4/npu-fsdp2-pp4-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-7900f287/analysis.json) |
| pp8 | 1/1/1/1/8/1 | 1 | 2,037.35 / 2,061.31 / 2,064.00 | 1.18% | 502.62 | 4,020.93 | 0.018 | [analysis.json](../../runs/8-card/pp8/npu-pp8-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-02473d85/analysis.json) |
| fsdp2-tp4 | 1/2/4/1/1/1 | 1 | 2,231.55 / 2,302.92 / 2,333.07 | 3.20% | 458.88 | 3,671.06 | 0.257 | [analysis.json](../../runs/8-card/fsdp2-tp4/npu-fsdp2-tp4-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-ecf6fd22/analysis.json) |
| fsdp2-tp4-ep8 | 1/2/4/1/1/8 | 1 | 2,323.02 / 2,347.76 / 2,349.01 | 1.06% | 440.80 | 3,526.44 | 0.216 | [analysis.json](../../runs/8-card/fsdp2-tp4-ep8/npu-fsdp2-tp4-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-53db2e7c/analysis.json) |
| tp8 | 1/1/8/1/1/1 | 1 | 4,421.03 / 4,668.07 / 4,763.60 | 5.59% | 231.62 | 1,852.98 | 0.208 | [analysis.json](../../runs/8-card/tp8/npu-tp8-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-64da983b/analysis.json) |
| tp2-cp4 | 1/1/2/4/1/1 | 1 | 4,517.32 / 4,629.49 / 4,673.33 | 2.48% | 226.68 | 1,813.47 | 0.130 | [analysis.json](../../runs/8-card/tp2-cp4/npu-tp2-cp4-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-3b7f547c/analysis.json) |
| fsdp2-tp2-pp2 | 1/2/2/1/2/1 | 1 | 9,152.28 / 9,183.01 / 9,273.85 | 0.34% | 111.88 | 895.08 | 0.074 | [analysis.json](../../runs/8-card/fsdp2-tp2-pp2/npu-fsdp2-tp2-pp2-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-5759ec05/analysis.json) |

## Profiler attribution

| Topology | Compute range (ms) | Exposed communication range (ms) | Collective calls/step | Payload MB/rank/step | Physical transit ms/step | Effective GB/s | Evidence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| cp8 `allgather` | 215.51–221.31 | 1,023.25–1,513.43 | 368.00 | 610.34 | 43.81 | 13.93 | [analysis.json](../../runs/8-card/cp8/npu-cp8-bf16-s20-l8-b64-seq128-seed61-distributed-r2-03d875ae/analysis.json) |
|  `allreduce` | – | – | 6.00 | 0.00 | 0.01 | 0.10 | [analysis.json](../../runs/8-card/cp8/npu-cp8-bf16-s20-l8-b64-seq128-seed61-distributed-r2-03d875ae/analysis.json) |
|  `reducescatter` | – | – | 216.00 | 584.10 | 129.84 | 4.50 | [analysis.json](../../runs/8-card/cp8/npu-cp8-bf16-s20-l8-b64-seq128-seed61-distributed-r2-03d875ae/analysis.json) |
| ddp8 `allreduce` | 41.51–42.17 | 14.07–131.17 | 17.00 | 116.49 | 8.79 | 13.25 | [analysis.json](../../runs/8-card/ddp8/npu-ddp8-bf16-s20-l8-b64-seq128-seed61-distributed-r1-488247ce/analysis.json) |
| ep8 `allgather` | 41.47–45.02 | 334.89–810.78 | 21.00 | 22.15 | 1.50 | 14.75 | [analysis.json](../../runs/8-card/ep8/npu-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-r4-ccb6f22c/analysis.json) |
|  `allreduce` | – | – | 8.00 | 0.00 | 0.02 | 0.09 | [analysis.json](../../runs/8-card/ep8/npu-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-r4-ccb6f22c/analysis.json) |
|  `alltoallv` | – | – | 35.00 | 24.64 | 2.07 | 11.87 | [analysis.json](../../runs/8-card/ep8/npu-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-r4-ccb6f22c/analysis.json) |
|  `reducescatter` | – | – | 11.00 | 19.80 | 5.06 | 3.92 | [analysis.json](../../runs/8-card/ep8/npu-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-r4-ccb6f22c/analysis.json) |
| fsdp2-cp4 `allgather` | 123.75–125.90 | 53.23–753.87 | 184.00 | 296.25 | 21.28 | 13.92 | [analysis.json](../../runs/8-card/fsdp2-cp4/npu-fsdp2-cp4-bf16-s20-l8-b64-seq128-seed61-distributed-r2-60aa6993/analysis.json) |
|  `allreduce` | – | – | 7.00 | 0.00 | 0.02 | 0.10 | [analysis.json](../../runs/8-card/fsdp2-cp4/npu-fsdp2-cp4-bf16-s20-l8-b64-seq128-seed61-distributed-r2-60aa6993/analysis.json) |
|  `reducescatter` | – | – | 108.00 | 283.66 | 65.42 | 4.34 | [analysis.json](../../runs/8-card/fsdp2-cp4/npu-fsdp2-cp4-bf16-s20-l8-b64-seq128-seed61-distributed-r2-60aa6993/analysis.json) |
| fsdp2-pp4 `allgather` | 137.53–299.42 | 436.97–1,689.48 | 12.00 | 21.26 | 1.10 | 19.26 | [analysis.json](../../runs/8-card/fsdp2-pp4/npu-fsdp2-pp4-bf16-s20-l8-b64-seq128-seed61-distributed-r2-dde8b599/analysis.json) |
|  `allreduce` | – | – | 8.00 | 0.00 | 0.01 | 0.01 | [analysis.json](../../runs/8-card/fsdp2-pp4/npu-fsdp2-pp4-bf16-s20-l8-b64-seq128-seed61-distributed-r2-dde8b599/analysis.json) |
|  `batchsendrecv` | – | – | 37.33 | 2.36 | 0.15 | 15.86 | [analysis.json](../../runs/8-card/fsdp2-pp4/npu-fsdp2-pp4-bf16-s20-l8-b64-seq128-seed61-distributed-r2-dde8b599/analysis.json) |
|  `receive` | – | – | 12.00 | 0.79 | 0.05 | 16.01 | [analysis.json](../../runs/8-card/fsdp2-pp4/npu-fsdp2-pp4-bf16-s20-l8-b64-seq128-seed61-distributed-r2-dde8b599/analysis.json) |
|  `reducescatter` | – | – | 12.00 | 41.51 | 2.36 | 17.84 | [analysis.json](../../runs/8-card/fsdp2-pp4/npu-fsdp2-pp4-bf16-s20-l8-b64-seq128-seed61-distributed-r2-dde8b599/analysis.json) |
|  `send` | – | – | 12.00 | 0.00 | 0.00 | - | [analysis.json](../../runs/8-card/fsdp2-pp4/npu-fsdp2-pp4-bf16-s20-l8-b64-seq128-seed61-distributed-r2-dde8b599/analysis.json) |
| fsdp2-tp2-pp2 `allgather` | 347.17–437.62 | 1,714.66–6,588.58 | 292.00 | 34.21 | 2.00 | 17.01 | [analysis.json](../../runs/8-card/fsdp2-tp2-pp2/npu-fsdp2-tp2-pp2-bf16-s20-l8-b64-seq128-seed61-distributed-r2-02a4a78f/analysis.json) |
|  `allreduce` | – | – | 1,065.50 | 23.74 | 1.74 | 4.90 | [analysis.json](../../runs/8-card/fsdp2-tp2-pp2/npu-fsdp2-tp2-pp2-bf16-s20-l8-b64-seq128-seed61-distributed-r2-02a4a78f/analysis.json) |
|  `batchsendrecv` | – | – | 29.33 | 1.38 | 0.09 | 14.40 | [analysis.json](../../runs/8-card/fsdp2-tp2-pp2/npu-fsdp2-tp2-pp2-bf16-s20-l8-b64-seq128-seed61-distributed-r2-02a4a78f/analysis.json) |
|  `receive` | – | – | 4.00 | 0.20 | 0.01 | 14.62 | [analysis.json](../../runs/8-card/fsdp2-tp2-pp2/npu-fsdp2-tp2-pp2-bf16-s20-l8-b64-seq128-seed61-distributed-r2-02a4a78f/analysis.json) |
|  `reducescatter` | – | – | 420.00 | 49.50 | 3.21 | 15.37 | [analysis.json](../../runs/8-card/fsdp2-tp2-pp2/npu-fsdp2-tp2-pp2-bf16-s20-l8-b64-seq128-seed61-distributed-r2-02a4a78f/analysis.json) |
|  `send` | – | – | 4.00 | 0.00 | 0.00 | - | [analysis.json](../../runs/8-card/fsdp2-tp2-pp2/npu-fsdp2-tp2-pp2-bf16-s20-l8-b64-seq128-seed61-distributed-r2-02a4a78f/analysis.json) |
| fsdp2-tp4 `allgather` | 152.70–155.75 | 523.16–1,941.06 | 144.00 | 63.54 | 3.49 | 18.22 | [analysis.json](../../runs/8-card/fsdp2-tp4/npu-fsdp2-tp4-bf16-s20-l8-b64-seq128-seed61-distributed-r2-7088c3b5/analysis.json) |
|  `allreduce` | – | – | 440.00 | 75.10 | 5.28 | 14.23 | [analysis.json](../../runs/8-card/fsdp2-tp4/npu-fsdp2-tp4-bf16-s20-l8-b64-seq128-seed61-distributed-r2-7088c3b5/analysis.json) |
|  `reducescatter` | – | – | 140.00 | 59.80 | 3.77 | 15.87 | [analysis.json](../../runs/8-card/fsdp2-tp4/npu-fsdp2-tp4-bf16-s20-l8-b64-seq128-seed61-distributed-r2-7088c3b5/analysis.json) |
| fsdp2-tp4-ep8 `allgather` | 145.92–151.26 | 858.81–1,738.75 | 220.00 | 71.41 | 4.13 | 17.31 | [analysis.json](../../runs/8-card/fsdp2-tp4-ep8/npu-fsdp2-tp4-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-r2-e6d2b2bd/analysis.json) |
|  `allreduce` | – | – | 362.00 | 11.96 | 1.28 | 9.27 | [analysis.json](../../runs/8-card/fsdp2-tp4-ep8/npu-fsdp2-tp4-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-r2-e6d2b2bd/analysis.json) |
|  `alltoallv` | – | – | 140.00 | 23.65 | 2.02 | 11.70 | [analysis.json](../../runs/8-card/fsdp2-tp4-ep8/npu-fsdp2-tp4-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-r2-e6d2b2bd/analysis.json) |
|  `reducescatter` | – | – | 212.00 | 66.09 | 4.42 | 14.96 | [analysis.json](../../runs/8-card/fsdp2-tp4-ep8/npu-fsdp2-tp4-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-r2-e6d2b2bd/analysis.json) |
| fsdp4-tp2 `allgather` | 81.14–83.39 | 169.70–936.23 | 72.00 | 62.50 | 4.15 | 15.07 | [analysis.json](../../runs/8-card/fsdp4-tp2/npu-fsdp4-tp2-bf16-s20-l8-b64-seq128-seed61-distributed-r2-a955eda9/analysis.json) |
|  `allreduce` | – | – | 224.00 | 24.02 | 1.49 | 16.39 | [analysis.json](../../runs/8-card/fsdp4-tp2/npu-fsdp4-tp2-bf16-s20-l8-b64-seq128-seed61-distributed-r2-a955eda9/analysis.json) |
|  `reducescatter` | – | – | 70.00 | 59.30 | 4.98 | 11.91 | [analysis.json](../../runs/8-card/fsdp4-tp2/npu-fsdp4-tp2-bf16-s20-l8-b64-seq128-seed61-distributed-r2-a955eda9/analysis.json) |
| fsdp8 `allgather` | 40.76–41.94 | 13.39–201.34 | 21.00 | 60.69 | 4.52 | 13.44 | [analysis.json](../../runs/8-card/fsdp8/npu-fsdp8-bf16-s20-l8-b64-seq128-seed61-distributed-r1-8501ffd6/analysis.json) |
|  `allreduce` | – | – | 7.00 | 0.00 | 0.02 | 0.09 | [analysis.json](../../runs/8-card/fsdp8/npu-fsdp8-bf16-s20-l8-b64-seq128-seed61-distributed-r1-8501ffd6/analysis.json) |
|  `reducescatter` | – | – | 11.00 | 58.33 | 16.47 | 3.55 | [analysis.json](../../runs/8-card/fsdp8/npu-fsdp8-bf16-s20-l8-b64-seq128-seed61-distributed-r1-8501ffd6/analysis.json) |
| pp8 `allreduce` | 77.36–401.46 | 131.78–3,497.76 | 2.00 | 0.00 | 0.00 | 0.00 | [analysis.json](../../runs/8-card/pp8/npu-pp8-bf16-s20-l8-b64-seq128-seed61-distributed-r2-462adeae/analysis.json) |
|  `batchsendrecv` | – | – | 57.33 | 3.67 | 0.23 | 15.90 | [analysis.json](../../runs/8-card/pp8/npu-pp8-bf16-s20-l8-b64-seq128-seed61-distributed-r2-462adeae/analysis.json) |
|  `receive` | – | – | 56.00 | 3.67 | 0.22 | 16.32 | [analysis.json](../../runs/8-card/pp8/npu-pp8-bf16-s20-l8-b64-seq128-seed61-distributed-r2-462adeae/analysis.json) |
|  `send` | – | – | 56.00 | 0.00 | 0.00 | - | [analysis.json](../../runs/8-card/pp8/npu-pp8-bf16-s20-l8-b64-seq128-seed61-distributed-r2-462adeae/analysis.json) |
| tp2-cp4 `allgather` | 244.52–257.95 | 1,765.77–4,192.39 | 488.00 | 283.09 | 18.90 | 14.98 | [analysis.json](../../runs/8-card/tp2-cp4/npu-tp2-cp4-bf16-s20-l8-b64-seq128-seed61-distributed-r2-ddc3f9b1/analysis.json) |
|  `allreduce` | – | – | 871.00 | 29.59 | 2.27 | 6.55 | [analysis.json](../../runs/8-card/tp2-cp4/npu-tp2-cp4-bf16-s20-l8-b64-seq128-seed61-distributed-r2-ddc3f9b1/analysis.json) |
|  `reducescatter` | – | – | 408.00 | 263.94 | 22.31 | 11.84 | [analysis.json](../../runs/8-card/tp2-cp4/npu-tp2-cp4-bf16-s20-l8-b64-seq128-seed61-distributed-r2-ddc3f9b1/analysis.json) |
| tp8 `allgather` | 278.20–285.63 | 2,306.83–3,816.10 | 120.00 | 55.05 | 3.64 | 15.12 | [analysis.json](../../runs/8-card/tp8/npu-tp8-bf16-s20-l8-b64-seq128-seed61-distributed-r1-dba2190c/analysis.json) |
|  `allreduce` | – | – | 865.00 | 171.17 | 13.97 | 12.26 | [analysis.json](../../runs/8-card/tp8/npu-tp8-bf16-s20-l8-b64-seq128-seed61-distributed-r1-dba2190c/analysis.json) |
|  `reducescatter` | – | – | 192.00 | 55.05 | 4.25 | 12.95 | [analysis.json](../../runs/8-card/tp8/npu-tp8-bf16-s20-l8-b64-seq128-seed61-distributed-r1-dba2190c/analysis.json) |

## Evidence chain per conclusion

1. Experiment identity and exact `torchrun` argv: each run's `manifest.json`.
2. Driver invocations and environment: `command_history.jsonl`.
3. Step metrics and HBM: `metrics.jsonl` and `analysis.json`.
4. Rank critical path: parsed `step_trace_time.csv` summarized in `distributed_step_trace`.
5. Collective count/payload/transit: canonical `ASCEND_PROFILER_OUTPUT/communication.json` summarized in `communication_summary`.
6. Official offline tools: `tool_commands/*.json`; MindStudio imports are listed in `artifacts.json`.

No optimization is accepted from these measurements alone. Each proposed change needs profiler-off repeats plus a separate profiler attribution run.
