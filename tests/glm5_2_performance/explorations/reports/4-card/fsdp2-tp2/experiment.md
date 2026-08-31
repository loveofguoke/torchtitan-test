# fsdp2-tp2 experiments

## topology

| field | value |
| --- | --- |
| world size | 4 |
| dp replicate / shard | 1 / 2 |
| tp / cp / pp / ep | 2 / 1 / 1 / 1 |
| recorded runs | 2 |

## run results

| run | status | mode | preset | median step | tok/s/device | tok/s/job | peak hbm |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| [npu-fsdp2-tp2-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-823a6a00](../../../runs/4-card/fsdp2-tp2/npu-fsdp2-tp2-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-823a6a00/readme.md) | completed | profiler-off | distributed | 2,204.28 ms | 929.22 | 3,716.87 | 0.404 GiB |
| [npu-fsdp2-tp2-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-0a488f2b](../../../runs/4-card/fsdp2-tp2/npu-fsdp2-tp2-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-0a488f2b/readme.md) | completed | profiler-active | overview | 3,292.14 ms | 622.09 | 2,488.35 | - GiB |

## experiment sequence

### 1. npu-fsdp2-tp2-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-823a6a00

- Status: `completed`
- Started: `2026-08-24T16:42:18.708928+08:00`
- Full process and outputs: [run readme](../../../runs/4-card/fsdp2-tp2/npu-fsdp2-tp2-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-823a6a00/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/4-card/fsdp2-tp2/npu-fsdp2-tp2-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-823a6a00.html`

```bash
/root/miniconda3/envs/torchtitan-0803/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology fsdp2-tp2 --preset distributed --profiler-off --visible-devices 1,2,3,4 --steps 20 --skip-steps 8 --warmup-steps 2 --active-steps 3 --replicate 1
```

### 2. npu-fsdp2-tp2-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-0a488f2b

- Status: `completed`
- Started: `2026-08-31T12:53:21.943468+08:00`
- Full process and outputs: [run readme](../../../runs/4-card/fsdp2-tp2/npu-fsdp2-tp2-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-0a488f2b/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/4-card/fsdp2-tp2/npu-fsdp2-tp2-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-0a488f2b.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology all --preset all --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all
```

## current summary

The highest recorded job throughput is 3,716.87 tok/s from `npu-fsdp2-tp2-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-823a6a00`. Profiler-off runs are throughput evidence; profiler-active runs are attribution evidence only.
