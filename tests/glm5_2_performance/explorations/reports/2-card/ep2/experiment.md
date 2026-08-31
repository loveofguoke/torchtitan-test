# ep2 experiments

## topology

| field | value |
| --- | --- |
| world size | 2 |
| dp replicate / shard | 1 / 2 |
| tp / cp / pp / ep | 1 / 1 / 1 / 2 |
| recorded runs | 1 |

## run results

| run | status | mode | preset | median step | tok/s/device | tok/s/job | peak hbm |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| [npu-ep2-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-2e297c09](../../../runs/2-card/ep2/npu-ep2-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-2e297c09/readme.md) | completed | profiler-active | overview | 1,932.64 ms | 2,119.38 | 4,238.77 | - GiB |

## experiment sequence

### 1. npu-ep2-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-2e297c09

- Status: `completed`
- Started: `2026-08-31T12:47:28.179963+08:00`
- Full process and outputs: [run readme](../../../runs/2-card/ep2/npu-ep2-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-2e297c09/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/2-card/ep2/npu-ep2-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-2e297c09.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology all --preset all --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all
```

## current summary

The highest recorded job throughput is 4,238.77 tok/s from `npu-ep2-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-2e297c09`. Profiler-off runs are throughput evidence; profiler-active runs are attribution evidence only.
