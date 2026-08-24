# pp4 experiments

## topology

| field | value |
| --- | --- |
| world size | 4 |
| dp replicate / shard | 1 / 1 |
| tp / cp / pp / ep | 1 / 1 / 4 / 1 |
| recorded runs | 1 |

## run results

| run | status | mode | preset | median step | tok/s/device | tok/s/job | peak hbm |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| [npu-pp4-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-d56db519](../../../runs/4-card/pp4/npu-pp4-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-d56db519/readme.md) | completed | profiler-off | distributed | 3,175.36 ms | 644.97 | 2,579.88 | 0.066 GiB |

## experiment sequence

### 1. npu-pp4-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-d56db519

- Status: `completed`
- Started: `2026-08-24T16:39:22.693361+08:00`
- Full process and outputs: [run readme](../../../runs/4-card/pp4/npu-pp4-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-d56db519/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/4-card/pp4/npu-pp4-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-d56db519.html`

```bash
/root/miniconda3/envs/torchtitan-0803/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology pp4 --preset distributed --profiler-off --visible-devices 1,2,3,4 --steps 20 --skip-steps 8 --warmup-steps 2 --active-steps 3 --replicate 1
```

## current summary

The highest recorded job throughput is 2,579.88 tok/s from `npu-pp4-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-d56db519`. Profiler-off runs are throughput evidence; profiler-active runs are attribution evidence only.
