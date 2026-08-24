# tp4 experiments

## topology

| field | value |
| --- | --- |
| world size | 4 |
| dp replicate / shard | 1 / 1 |
| tp / cp / pp / ep | 4 / 1 / 1 / 1 |
| recorded runs | 1 |

## run results

| run | status | mode | preset | median step | tok/s/device | tok/s/job | peak hbm |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| [npu-tp4-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-8aa36e1b](../../../runs/4-card/tp4/npu-tp4-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-8aa36e1b/readme.md) | completed | profiler-off | distributed | 4,330.00 ms | 472.98 | 1,891.92 | 0.290 GiB |

## experiment sequence

### 1. npu-tp4-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-8aa36e1b

- Status: `completed`
- Started: `2026-08-24T16:35:47.669094+08:00`
- Full process and outputs: [run readme](../../../runs/4-card/tp4/npu-tp4-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-8aa36e1b/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/4-card/tp4/npu-tp4-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-8aa36e1b.html`

```bash
/root/miniconda3/envs/torchtitan-0803/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology tp4 --preset distributed --profiler-off --visible-devices 1,2,3,4 --steps 20 --skip-steps 8 --warmup-steps 2 --active-steps 3 --replicate 1
```

## current summary

The highest recorded job throughput is 1,891.92 tok/s from `npu-tp4-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-8aa36e1b`. Profiler-off runs are throughput evidence; profiler-active runs are attribution evidence only.
