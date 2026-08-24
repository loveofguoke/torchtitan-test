# ep8 experiments

## topology

| field | value |
| --- | --- |
| world size | 8 |
| dp replicate / shard | 1 / 8 |
| tp / cp / pp / ep | 1 / 1 / 1 / 8 |
| recorded runs | 1 |

## run results

| run | status | mode | preset | median step | tok/s/device | tok/s/job | peak hbm |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| [npu-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-50c6107d](../../../runs/8-card/ep8/npu-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-50c6107d/readme.md) | completed | profiler-off | distributed | 267.04 ms | 3,834.67 | 30,677.33 | 0.609 GiB |

## experiment sequence

### 1. npu-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-50c6107d

- Status: `completed`
- Started: `2026-08-24T17:56:02.331131+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/ep8/npu-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-50c6107d/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/ep8/npu-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-50c6107d.html`

```bash
/root/miniconda3/envs/torchtitan-0803/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology ep8 --preset distributed --profiler-off --visible-devices 0,1,2,3,4,5,6,7 --steps 20 --skip-steps 8 --warmup-steps 2 --active-steps 3 --replicate 1
```

## current summary

The highest recorded job throughput is 30,677.33 tok/s from `npu-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-50c6107d`. Profiler-off runs are throughput evidence; profiler-active runs are attribution evidence only.
