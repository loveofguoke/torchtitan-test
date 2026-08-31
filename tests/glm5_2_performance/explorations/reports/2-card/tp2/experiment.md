# tp2 experiments

## topology

| field | value |
| --- | --- |
| world size | 2 |
| dp replicate / shard | 1 / 1 |
| tp / cp / pp / ep | 2 / 1 / 1 / 1 |
| recorded runs | 1 |

## run results

| run | status | mode | preset | median step | tok/s/device | tok/s/job | peak hbm |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| [npu-tp2-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-e698990c](../../../runs/2-card/tp2/npu-tp2-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-e698990c/readme.md) | completed | profiler-active | overview | 6,060.82 ms | 675.82 | 1,351.63 | - GiB |

## experiment sequence

### 1. npu-tp2-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-e698990c

- Status: `completed`
- Started: `2026-08-31T11:52:15.754222+08:00`
- Full process and outputs: [run readme](../../../runs/2-card/tp2/npu-tp2-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-e698990c/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/2-card/tp2/npu-tp2-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-e698990c.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology all --preset all --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all
```

## current summary

The highest recorded job throughput is 1,351.63 tok/s from `npu-tp2-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-e698990c`. Profiler-off runs are throughput evidence; profiler-active runs are attribution evidence only.
