# ep4 experiments

## topology

| field | value |
| --- | --- |
| world size | 4 |
| dp replicate / shard | 1 / 4 |
| tp / cp / pp / ep | 1 / 1 / 1 / 4 |
| recorded runs | 1 |

## run results

| run | status | mode | preset | median step | tok/s/device | tok/s/job | peak hbm |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| [npu-ep4-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-4521dc9c](../../../runs/4-card/ep4/npu-ep4-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-4521dc9c/readme.md) | completed | profiler-off | distributed | 477.19 ms | 4,291.94 | 17,167.77 | 0.653 GiB |

## experiment sequence

### 1. npu-ep4-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-4521dc9c

- Status: `completed`
- Started: `2026-08-24T16:41:21.029153+08:00`
- Full process and outputs: [run readme](../../../runs/4-card/ep4/npu-ep4-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-4521dc9c/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/4-card/ep4/npu-ep4-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-4521dc9c.html`

```bash
/root/miniconda3/envs/torchtitan-0803/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology ep4 --preset distributed --profiler-off --visible-devices 1,2,3,4 --steps 20 --skip-steps 8 --warmup-steps 2 --active-steps 3 --replicate 1
```

## current summary

The highest recorded job throughput is 17,167.77 tok/s from `npu-ep4-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-4521dc9c`. Profiler-off runs are throughput evidence; profiler-active runs are attribution evidence only.
