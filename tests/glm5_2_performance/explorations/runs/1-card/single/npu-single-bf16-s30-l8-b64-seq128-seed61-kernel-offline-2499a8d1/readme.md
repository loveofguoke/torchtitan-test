# npu-single-bf16-s30-l8-b64-seq128-seed61-kernel-offline-2499a8d1

## experiment

| field | value |
| --- | --- |
| status | command-only |
| topology | single |
| world size | 1 |
| mode | profiler-active |
| device | npu |
| preset | - |
| steps | - |
| local / global batch | - / - |
| sequence length | - |
| parameter / reduction dtype | - / - |
| replicate | - |
| visible devices | - |
| started | 2026-08-24T16:32:02.286244+08:00 |

## process

1. Run the `analyze` driver:

   ```bash
   /root/miniconda3/envs/torchtitan-0803/bin/python tests/glm5_2_performance/profiler_benchmark.py --analyze --device npu --topology single --preset kernel --offline --visible-devices 1
   ```

2. Produce compact evidence (`manifest.json`, `metrics.jsonl`, `analysis.json`, tool status files) and the HTML report when analysis completed.

## results

| metric | value |
| --- | ---: |
| median step | - ms |
| p90 step | - ms |
| throughput / device | - tok/s |
| throughput / job | - tok/s |
| peak active hbm | - GiB |

## outputs

| output | purpose |
| --- | --- |
| [command_history.jsonl](command_history.jsonl) | exact driver history and environment |


Profiler-off results are throughput evidence. Profiler-active results are used for attribution only and must not be compared as authoritative throughput.
