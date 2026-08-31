# cp2 experiments

## topology

| field | value |
| --- | --- |
| world size | 2 |
| dp replicate / shard | 1 / 1 |
| tp / cp / pp / ep | 1 / 2 / 1 / 1 |
| recorded runs | 1 |

## run results

| run | status | mode | preset | median step | tok/s/device | tok/s/job | peak hbm |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| [npu-cp2-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-538f588d](../../../runs/2-card/cp2/npu-cp2-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-538f588d/readme.md) | completed | profiler-active | overview | 2,821.60 ms | 1,451.66 | 2,903.32 | - GiB |

## experiment sequence

### 1. npu-cp2-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-538f588d

- Status: `completed`
- Started: `2026-08-31T12:10:01.756270+08:00`
- Full process and outputs: [run readme](../../../runs/2-card/cp2/npu-cp2-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-538f588d/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/2-card/cp2/npu-cp2-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-538f588d.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology all --preset all --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all
```

## current summary

The highest recorded job throughput is 2,903.32 tok/s from `npu-cp2-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-538f588d`. Profiler-off runs are throughput evidence; profiler-active runs are attribution evidence only.
