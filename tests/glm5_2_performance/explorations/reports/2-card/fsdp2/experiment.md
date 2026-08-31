# fsdp2 experiments

## topology

| field | value |
| --- | --- |
| world size | 2 |
| dp replicate / shard | 1 / 2 |
| tp / cp / pp / ep | 1 / 1 / 1 / 1 |
| recorded runs | 1 |

## run results

| run | status | mode | preset | median step | tok/s/device | tok/s/job | peak hbm |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| [npu-fsdp2-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-aa77d62c](../../../runs/2-card/fsdp2/npu-fsdp2-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-aa77d62c/readme.md) | completed | profiler-active | overview | 1,400.59 ms | 2,924.49 | 5,848.98 | - GiB |

## experiment sequence

### 1. npu-fsdp2-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-aa77d62c

- Status: `completed`
- Started: `2026-08-31T11:47:00.243062+08:00`
- Full process and outputs: [run readme](../../../runs/2-card/fsdp2/npu-fsdp2-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-aa77d62c/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/2-card/fsdp2/npu-fsdp2-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-aa77d62c.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology all --preset all --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all
```

## current summary

The highest recorded job throughput is 5,848.98 tok/s from `npu-fsdp2-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-aa77d62c`. Profiler-off runs are throughput evidence; profiler-active runs are attribution evidence only.
