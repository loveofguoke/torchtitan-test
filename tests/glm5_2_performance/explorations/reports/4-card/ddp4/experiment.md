# ddp4 experiments

## topology

| field | value |
| --- | --- |
| world size | 4 |
| dp replicate / shard | 4 / 1 |
| tp / cp / pp / ep | 1 / 1 / 1 / 1 |
| recorded runs | 3 |

## run results

| run | status | mode | preset | median step | tok/s/device | tok/s/job | peak hbm |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| [npu-ddp4-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-5da7b9c5](../../../runs/4-card/ddp4/npu-ddp4-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-5da7b9c5/readme.md) | completed | profiler-off | distributed | 389.92 ms | 5,252.34 | 21,009.35 | 0.843 GiB |
| [npu-ddp4-bf16-s20-l8-b64-seq128-seed61-distributed-r1-64a465c1](../../../runs/4-card/ddp4/npu-ddp4-bf16-s20-l8-b64-seq128-seed61-distributed-r1-64a465c1/readme.md) | completed | profiler-active | distributed | 398.32 ms | 5,141.56 | 20,566.23 | - GiB |
| [npu-ddp4-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-198b87aa](../../../runs/4-card/ddp4/npu-ddp4-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-198b87aa/readme.md) | completed | profiler-active | overview | 758.56 ms | 2,699.87 | 10,799.48 | - GiB |

## experiment sequence

### 1. npu-ddp4-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-5da7b9c5

- Status: `completed`
- Started: `2026-08-24T16:33:00.017343+08:00`
- Full process and outputs: [run readme](../../../runs/4-card/ddp4/npu-ddp4-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-5da7b9c5/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/4-card/ddp4/npu-ddp4-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-5da7b9c5.html`

```bash
/root/miniconda3/envs/torchtitan-0803/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology ddp4 --preset distributed --profiler-off --visible-devices 1,2,3,4 --steps 20 --skip-steps 8 --warmup-steps 2 --active-steps 3 --replicate 1
```

```bash
/root/miniconda3/envs/torchtitan-0803/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology ddp4 --preset distributed --profiler-off --visible-devices 1,2,3,4 --steps 20 --skip-steps 8 --warmup-steps 2 --active-steps 3 --replicate 1
```

### 2. npu-ddp4-bf16-s20-l8-b64-seq128-seed61-distributed-r1-64a465c1

- Status: `completed`
- Started: `2026-08-24T16:44:11.408897+08:00`
- Full process and outputs: [run readme](../../../runs/4-card/ddp4/npu-ddp4-bf16-s20-l8-b64-seq128-seed61-distributed-r1-64a465c1/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/4-card/ddp4/npu-ddp4-bf16-s20-l8-b64-seq128-seed61-distributed-r1-64a465c1.html`

```bash
/root/miniconda3/envs/torchtitan-0803/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology ddp4 --preset distributed --visible-devices 1,2,3,4 --steps 20 --skip-steps 8 --warmup-steps 2 --active-steps 3 --replicate 1 --cluster
```

```bash
/root/miniconda3/envs/torchtitan-0803/bin/python tests/glm5_2_performance/profiler_benchmark.py --analyze --device npu --topology ddp4 --preset distributed --visible-devices 1,2,3,4 --steps 20 --skip-steps 8 --warmup-steps 2 --active-steps 3 --replicate 1 --cluster
```

```bash
/root/miniconda3/envs/torchtitan-0803/bin/python tests/glm5_2_performance/profiler_benchmark.py --analyze --device npu --topology ddp4 --preset distributed --visible-devices 1,2,3,4 --steps 20 --skip-steps 8 --warmup-steps 2 --active-steps 3 --replicate 1
```

### 3. npu-ddp4-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-198b87aa

- Status: `completed`
- Started: `2026-08-31T11:44:52.760329+08:00`
- Full process and outputs: [run readme](../../../runs/4-card/ddp4/npu-ddp4-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-198b87aa/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/4-card/ddp4/npu-ddp4-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-198b87aa.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology all --preset all --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all
```

## current summary

The highest recorded job throughput is 21,009.35 tok/s from `npu-ddp4-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-5da7b9c5`. Profiler-off runs are throughput evidence; profiler-active runs are attribution evidence only.
