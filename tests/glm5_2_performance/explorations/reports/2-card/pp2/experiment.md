# pp2 experiments

## topology

| field | value |
| --- | --- |
| world size | 2 |
| dp replicate / shard | 1 / 1 |
| tp / cp / pp / ep | 1 / 1 / 2 / 1 |
| recorded runs | 1 |

## run results

| run | status | mode | preset | median step | tok/s/device | tok/s/job | peak hbm |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| [npu-pp2-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-e9783e4b](../../../runs/2-card/pp2/npu-pp2-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-e9783e4b/readme.md) | completed | profiler-active | overview | 9,946.33 ms | 411.81 | 823.62 | - GiB |

## experiment sequence

### 1. npu-pp2-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-e9783e4b

- Status: `completed`
- Started: `2026-08-31T12:22:17.071116+08:00`
- Full process and outputs: [run readme](../../../runs/2-card/pp2/npu-pp2-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-e9783e4b/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/2-card/pp2/npu-pp2-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-e9783e4b.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology all --preset all --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all
```

## current summary

The highest recorded job throughput is 823.62 tok/s from `npu-pp2-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-e9783e4b`. Profiler-off runs are throughput evidence; profiler-active runs are attribution evidence only.
