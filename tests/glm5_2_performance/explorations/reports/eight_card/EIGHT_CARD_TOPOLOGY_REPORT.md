# GLM-5.2 NPU 8-card topology report

Generated: 2026-08-24T18:39:11.801351+08:00

## Scope and comparability

This table selects NPU distributed runs with `steps=20`, `world_size=8`, and the distributed preset; profiler-off screening uses `latest successful` replicate, while attribution uses the latest successful replicate. Exact physical device mappings are retained per run. The eight-card runs include NPU0, which currently reports a health warning; therefore those results are diagnostic comparisons, not healthy-hardware acceptance evidence. Profiler-off runs are throughput evidence; profiler-active runs are attribution only.

Job throughput is `world_size × rank throughput`, equivalently the configured global token budget divided by median step time. Peak HBM is the maximum `memory/max_active(GiB)` sample on the metrics rank.

## Validity constraints

- The first DDP8 profiler-off attempt (`replicate=1`) failed during concurrent TSD initialization on physical NPU4/6/7 (`507033: TsdOpen failed`). Sequential set-device probes then passed; the table therefore records successful `replicate=2` and retains the failed run directory as evidence.
- DDP8 replicate 2 has 73.52% p90 drift and disagrees with the non-active portion of the profiler run. Treat its throughput as contaminated until an idle-system repeat converges; the rank/collective attribution remains useful.
- Every other topology currently has one profiler-off replicate. Rankings are screening results and need repeated idle-system confirmation before acceptance.

## Profiler-off topology screening

| Topology | Degrees (DP-repl/DP-shard/TP/CP/PP/EP) | Replicate | Median / p90 / max step (ms) | p90 drift | tok/s/device | tok/s/job | Peak active HBM (GiB) | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| fsdp8 | 1/8/1/1/1/1 | 1 | 219.00 / 221.66 / 221.90 | 1.21% | 4,675.83 | 37,406.66 | 0.621 | `tests/glm5_2_performance/explorations/runs/npu-fsdp8-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-4a88f56a/analysis.json` |
| ep8 | 1/8/1/1/1/8 | 1 | 267.04 / 272.66 / 315.53 | 2.11% | 3,834.67 | 30,677.33 | 0.609 | `tests/glm5_2_performance/explorations/runs/npu-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-50c6107d/analysis.json` |
| ddp8 | 8/1/1/1/1/1 | 2 | 773.35 / 1,341.89 / 1,767.16 | 73.52% | 1,370.09 | 10,960.70 | 0.783 | `tests/glm5_2_performance/explorations/runs/npu-ddp8-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r2-80606b65/analysis.json` |
| fsdp2-cp4 | 1/2/1/4/1/1 | 1 | 840.80 / 858.67 / 869.82 | 2.13% | 1,217.90 | 9,743.17 | 0.201 | `tests/glm5_2_performance/explorations/runs/npu-fsdp2-cp4-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-d7029385/analysis.json` |
| fsdp4-tp2 | 1/4/2/1/1/1 | 1 | 1,129.79 / 1,157.37 / 1,160.71 | 2.44% | 906.36 | 7,250.90 | 0.372 | `tests/glm5_2_performance/explorations/runs/npu-fsdp4-tp2-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-e3971d37/analysis.json` |
| cp8 | 1/1/1/8/1/1 | 1 | 1,626.28 / 1,694.16 / 1,726.96 | 4.17% | 629.66 | 5,037.26 | 0.130 | `tests/glm5_2_performance/explorations/runs/npu-cp8-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-486604c0/analysis.json` |
| fsdp2-pp4 | 1/2/1/1/4/1 | 1 | 1,744.04 / 1,875.22 / 1,889.89 | 7.52% | 587.21 | 4,697.69 | 0.044 | `tests/glm5_2_performance/explorations/runs/npu-fsdp2-pp4-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-7900f287/analysis.json` |
| pp8 | 1/1/1/1/8/1 | 1 | 2,037.35 / 2,061.31 / 2,064.00 | 1.18% | 502.62 | 4,020.93 | 0.018 | `tests/glm5_2_performance/explorations/runs/npu-pp8-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-02473d85/analysis.json` |
| fsdp2-tp4 | 1/2/4/1/1/1 | 1 | 2,231.55 / 2,302.92 / 2,333.07 | 3.20% | 458.88 | 3,671.06 | 0.257 | `tests/glm5_2_performance/explorations/runs/npu-fsdp2-tp4-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-ecf6fd22/analysis.json` |
| fsdp2-tp4-ep8 | 1/2/4/1/1/8 | 1 | 2,323.02 / 2,347.76 / 2,349.01 | 1.06% | 440.80 | 3,526.44 | 0.216 | `tests/glm5_2_performance/explorations/runs/npu-fsdp2-tp4-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-53db2e7c/analysis.json` |
| tp8 | 1/1/8/1/1/1 | 1 | 4,421.03 / 4,668.07 / 4,763.60 | 5.59% | 231.62 | 1,852.98 | 0.208 | `tests/glm5_2_performance/explorations/runs/npu-tp8-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-64da983b/analysis.json` |
| tp2-cp4 | 1/1/2/4/1/1 | 1 | 4,517.32 / 4,629.49 / 4,673.33 | 2.48% | 226.68 | 1,813.47 | 0.130 | `tests/glm5_2_performance/explorations/runs/npu-tp2-cp4-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-3b7f547c/analysis.json` |
| fsdp2-tp2-pp2 | 1/2/2/1/2/1 | 1 | 9,152.28 / 9,183.01 / 9,273.85 | 0.34% | 111.88 | 895.08 | 0.074 | `tests/glm5_2_performance/explorations/runs/npu-fsdp2-tp2-pp2-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-5759ec05/analysis.json` |

## Profiler attribution

| Topology | Compute range (ms) | Exposed communication range (ms) | Collective calls/step | Payload MB/rank/step | Physical transit ms/step | Effective GB/s | Evidence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| ddp8 `allreduce` | 41.51–42.17 | 14.07–131.17 | 17.00 | 116.49 | 8.79 | 13.25 | `tests/glm5_2_performance/explorations/runs/npu-ddp8-bf16-s20-l8-b64-seq128-seed61-distributed-r1-488247ce/analysis.json` |
| fsdp8 `allgather` | 40.76–41.94 | 13.39–201.34 | 21.00 | 60.69 | 4.52 | 13.44 | `tests/glm5_2_performance/explorations/runs/npu-fsdp8-bf16-s20-l8-b64-seq128-seed61-distributed-r1-8501ffd6/analysis.json` |
|  `allreduce` | – | – | 7.00 | 0.00 | 0.02 | 0.09 | `tests/glm5_2_performance/explorations/runs/npu-fsdp8-bf16-s20-l8-b64-seq128-seed61-distributed-r1-8501ffd6/analysis.json` |
|  `reducescatter` | – | – | 11.00 | 58.33 | 16.47 | 3.55 | `tests/glm5_2_performance/explorations/runs/npu-fsdp8-bf16-s20-l8-b64-seq128-seed61-distributed-r1-8501ffd6/analysis.json` |

## Evidence chain per conclusion

1. Experiment identity and exact `torchrun` argv: each run's `manifest.json`.
2. Driver invocations and environment: `command_history.jsonl`.
3. Step metrics and HBM: `metrics.jsonl` and `analysis.json`.
4. Rank critical path: parsed `step_trace_time.csv` summarized in `distributed_step_trace`.
5. Collective count/payload/transit: canonical `ASCEND_PROFILER_OUTPUT/communication.json` summarized in `communication_summary`.
6. Official offline tools: `tool_commands/*.json`; MindStudio imports are listed in `artifacts.json`.

No optimization is accepted from these measurements alone. Each proposed change needs profiler-off repeats plus a separate profiler attribution run.
