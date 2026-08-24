# GLM-5.2 NPU 4-card topology report

Generated: 2026-08-24T19:20:50.961766+08:00

## Scope and comparability

This table selects NPU distributed runs with `steps=20`, `world_size=4`, and the distributed preset; profiler-off screening uses `latest successful` replicate, while attribution uses the latest successful replicate. Exact physical device mappings are retained per run. Profiler-off runs are throughput evidence; profiler-active runs are attribution only.

Job throughput is `world_size × rank throughput`, equivalently the configured global token budget divided by median step time. Peak HBM is the maximum `memory/max_active(GiB)` sample on the metrics rank.

## Validity constraints

- Every topology currently has one authoritative profiler-off replicate at most. Rankings are screening results and need repeated idle-system confirmation before acceptance.

## Profiler-off topology screening

| Topology | Degrees (DP-repl/DP-shard/TP/CP/PP/EP) | Replicate | Median / p90 / max step (ms) | p90 drift | tok/s/device | tok/s/job | Peak active HBM (GiB) | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| ddp4 | 4/1/1/1/1/1 | 1 | 389.92 / 408.48 / 422.21 | 4.76% | 5,252.34 | 21,009.35 | 0.843 | [analysis.json](../../runs/4-card/ddp4/npu-ddp4-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-5da7b9c5/analysis.json) |
| fsdp4 | 1/4/1/1/1/1 | 1 | 401.75 / 413.37 / 415.93 | 2.89% | 5,097.71 | 20,390.85 | 0.660 | [analysis.json](../../runs/4-card/fsdp4/npu-fsdp4-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-0ba5e24c/analysis.json) |
| ep4 | 1/4/1/1/1/4 | 1 | 477.19 / 484.04 / 525.94 | 1.44% | 4,291.94 | 17,167.77 | 0.653 | [analysis.json](../../runs/4-card/ep4/npu-ep4-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-4521dc9c/analysis.json) |
| cp4 | 1/1/1/4/1/1 | 1 | 1,615.86 / 1,739.84 / 1,792.23 | 7.67% | 1,267.44 | 5,069.75 | 0.232 | [analysis.json](../../runs/4-card/cp4/npu-cp4-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-f83c9d6c/analysis.json) |
| fsdp2-tp2 | 1/2/2/1/1/1 | 1 | 2,204.28 / 2,293.93 / 2,366.87 | 4.07% | 929.22 | 3,716.87 | 0.404 | [analysis.json](../../runs/4-card/fsdp2-tp2/npu-fsdp2-tp2-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-823a6a00/analysis.json) |
| pp4 | 1/1/1/1/4/1 | 1 | 3,175.36 / 3,243.78 / 3,250.70 | 2.15% | 644.97 | 2,579.88 | 0.066 | [analysis.json](../../runs/4-card/pp4/npu-pp4-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-d56db519/analysis.json) |
| tp4 | 1/1/4/1/1/1 | 1 | 4,330.00 / 4,385.52 / 4,395.95 | 1.28% | 472.98 | 1,891.92 | 0.290 | [analysis.json](../../runs/4-card/tp4/npu-tp4-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-8aa36e1b/analysis.json) |

## Profiler attribution

| Topology | Compute range (ms) | Exposed communication range (ms) | Collective calls/step | Payload MB/rank/step | Physical transit ms/step | Effective GB/s | Evidence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| ddp4 `allreduce` | 81.67–83.30 | 42.22–223.38 | 28.00 | 199.82 | 12.14 | 16.49 | [analysis.json](../../runs/4-card/ddp4/npu-ddp4-bf16-s20-l8-b64-seq128-seed61-distributed-r1-64a465c1/analysis.json) |
| fsdp4 `allgather` | 81.64–82.87 | 19.42–294.37 | 42.00 | 103.99 | 5.75 | 18.07 | [analysis.json](../../runs/4-card/fsdp4/npu-fsdp4-bf16-s20-l8-b64-seq128-seed61-distributed-r3-d3b92684/analysis.json) |
|  `allreduce` | – | – | 7.00 | 0.00 | 0.01 | 0.10 | [analysis.json](../../runs/4-card/fsdp4/npu-fsdp4-bf16-s20-l8-b64-seq128-seed61-distributed-r3-d3b92684/analysis.json) |
|  `reducescatter` | – | – | 22.00 | 100.00 | 8.75 | 11.44 | [analysis.json](../../runs/4-card/fsdp4/npu-fsdp4-bf16-s20-l8-b64-seq128-seed61-distributed-r3-d3b92684/analysis.json) |

## Evidence chain per conclusion

1. Experiment identity and exact `torchrun` argv: each run's `manifest.json`.
2. Driver invocations and environment: `command_history.jsonl`.
3. Step metrics and HBM: `metrics.jsonl` and `analysis.json`.
4. Rank critical path: parsed `step_trace_time.csv` summarized in `distributed_step_trace`.
5. Collective count/payload/transit: canonical `ASCEND_PROFILER_OUTPUT/communication.json` summarized in `communication_summary`.
6. Official offline tools: `tool_commands/*.json`; MindStudio imports are listed in `artifacts.json`.

No optimization is accepted from these measurements alone. Each proposed change needs profiler-off repeats plus a separate profiler attribution run.
