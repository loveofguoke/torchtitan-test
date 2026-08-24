# GLM-5.2 NPU multi-card topology report

Generated: 2026-08-24T17:24:19.826490+08:00

## Scope and comparability

This table selects NPU distributed runs with `steps=20` and the distributed preset; profiler-off screening uses `replicate=1`, while attribution uses the latest successful replicate. Exact physical device mappings are retained per run; NPU0 is excluded because it reports an HCCS lane-drop warning. Profiler-off runs are throughput evidence; profiler-active runs are attribution only.

Job throughput is `world_size × rank throughput`, equivalently the configured global token budget divided by median step time. Peak HBM is the maximum `memory/max_active(GiB)` sample on the metrics rank.

## Profiler-off topology screening

| Topology | Degrees (DP-repl/DP-shard/TP/CP/PP/EP) | Median step (ms) | tok/s/device | tok/s/job | Peak active HBM (GiB) | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| ddp4 | 4/1/1/1/1/1 | 389.92 | 5,252.34 | 21,009.35 | 0.843 | `tests/glm5_2_performance/explorations/runs/npu-ddp4-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-5da7b9c5/analysis.json` |
| fsdp4 | 1/4/1/1/1/1 | 401.75 | 5,097.71 | 20,390.85 | 0.660 | `tests/glm5_2_performance/explorations/runs/npu-fsdp4-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-0ba5e24c/analysis.json` |
| ep4 | 1/4/1/1/1/4 | 477.19 | 4,291.94 | 17,167.77 | 0.658 | `tests/glm5_2_performance/explorations/runs/npu-ep4-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-4521dc9c/analysis.json` |
| cp4 | 1/1/1/4/1/1 | 1,615.86 | 1,267.44 | 5,069.75 | 0.232 | `tests/glm5_2_performance/explorations/runs/npu-cp4-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-f83c9d6c/analysis.json` |
| fsdp2-tp2 | 1/2/2/1/1/1 | 2,204.28 | 929.22 | 3,716.87 | 0.404 | `tests/glm5_2_performance/explorations/runs/npu-fsdp2-tp2-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-823a6a00/analysis.json` |
| pp4 | 1/1/1/1/4/1 | 3,175.36 | 644.97 | 2,579.88 | 0.066 | `tests/glm5_2_performance/explorations/runs/npu-pp4-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-d56db519/analysis.json` |
| tp4 | 1/1/4/1/1/1 | 4,330.00 | 472.98 | 1,891.92 | 0.290 | `tests/glm5_2_performance/explorations/runs/npu-tp4-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-8aa36e1b/analysis.json` |

## Profiler attribution

| Topology | Compute range (ms) | Exposed communication range (ms) | Collective calls/step | Payload MB/rank/step | Physical transit ms/step | Effective GB/s | Evidence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| ddp4 `allreduce` | 81.67–83.30 | 42.22–223.38 | 28.00 | 199.82 | 12.14 | 16.49 | `tests/glm5_2_performance/explorations/runs/npu-ddp4-bf16-s20-l8-b64-seq128-seed61-distributed-r1-64a465c1/analysis.json` |
| fsdp4 `allgather` | 81.64–82.87 | 19.42–294.37 | 42.00 | 103.99 | 5.75 | 18.07 | `tests/glm5_2_performance/explorations/runs/npu-fsdp4-bf16-s20-l8-b64-seq128-seed61-distributed-r3-d3b92684/analysis.json` |
|  `allreduce` | – | – | 7.00 | 0.00 | 0.01 | 0.10 | `tests/glm5_2_performance/explorations/runs/npu-fsdp4-bf16-s20-l8-b64-seq128-seed61-distributed-r3-d3b92684/analysis.json` |
|  `reducescatter` | – | – | 22.00 | 100.00 | 8.75 | 11.44 | `tests/glm5_2_performance/explorations/runs/npu-fsdp4-bf16-s20-l8-b64-seq128-seed61-distributed-r3-d3b92684/analysis.json` |

## Evidence chain per conclusion

1. Experiment identity and exact `torchrun` argv: each run's `manifest.json`.
2. Driver invocations and environment: `command_history.jsonl`.
3. Step metrics and HBM: `metrics.jsonl` and `analysis.json`.
4. Rank critical path: parsed `step_trace_time.csv` summarized in `distributed_step_trace`.
5. Collective count/payload/transit: canonical `ASCEND_PROFILER_OUTPUT/communication.json` summarized in `communication_summary`.
6. Official offline tools: `tool_commands/*.json`; MindStudio imports are listed in `artifacts.json`.

No optimization is accepted from these measurements alone. Each proposed change needs profiler-off repeats plus a separate profiler attribution run.
