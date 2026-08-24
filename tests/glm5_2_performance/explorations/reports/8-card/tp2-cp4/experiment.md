# tp2-cp4 experiments

## topology

| field | value |
| --- | --- |
| world size | 8 |
| dp replicate / shard | 1 / 1 |
| tp / cp / pp / ep | 2 / 4 / 1 / 1 |
| recorded runs | 1 |

## run results

| run | status | mode | preset | median step | tok/s/device | tok/s/job | peak hbm |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| [npu-tp2-cp4-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-3b7f547c](../../../runs/8-card/tp2-cp4/npu-tp2-cp4-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-3b7f547c/readme.md) | completed | profiler-off | distributed | 4,517.32 ms | 226.68 | 1,813.47 | 0.130 GiB |

## experiment sequence

### 1. npu-tp2-cp4-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-3b7f547c

- Status: `completed`
- Started: `2026-08-24T18:00:29.592527+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/tp2-cp4/npu-tp2-cp4-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-3b7f547c/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/tp2-cp4/npu-tp2-cp4-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-3b7f547c.html`

```bash
/root/miniconda3/envs/torchtitan-0803/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology tp2-cp4 --preset distributed --profiler-off --visible-devices 0,1,2,3,4,5,6,7 --steps 20 --skip-steps 8 --warmup-steps 2 --active-steps 3 --replicate 1
```

## current summary

The highest recorded job throughput is 1,813.47 tok/s from `npu-tp2-cp4-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-3b7f547c`. Profiler-off runs are throughput evidence; profiler-active runs are attribution evidence only.
