# cp4 experiments

## topology

| field | value |
| --- | --- |
| world size | 4 |
| dp replicate / shard | 1 / 1 |
| tp / cp / pp / ep | 1 / 4 / 1 / 1 |
| recorded runs | 1 |

## run results

| run | status | mode | preset | median step | tok/s/device | tok/s/job | peak hbm |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| [npu-cp4-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-f83c9d6c](../../../runs/4-card/cp4/npu-cp4-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-f83c9d6c/readme.md) | completed | profiler-off | distributed | 1,615.86 ms | 1,267.44 | 5,069.75 | 0.232 GiB |

## experiment sequence

### 1. npu-cp4-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-f83c9d6c

- Status: `completed`
- Started: `2026-08-24T16:38:04.728711+08:00`
- Full process and outputs: [run readme](../../../runs/4-card/cp4/npu-cp4-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-f83c9d6c/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/4-card/cp4/npu-cp4-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-f83c9d6c.html`

```bash
/root/miniconda3/envs/torchtitan-0803/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology cp4 --preset distributed --profiler-off --visible-devices 1,2,3,4 --steps 20 --skip-steps 8 --warmup-steps 2 --active-steps 3 --replicate 1
```

## current summary

The highest recorded job throughput is 5,069.75 tok/s from `npu-cp4-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-f83c9d6c`. Profiler-off runs are throughput evidence; profiler-active runs are attribution evidence only.
