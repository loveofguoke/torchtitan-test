# fsdp4 experiments

## topology

| field | value |
| --- | --- |
| world size | 4 |
| dp replicate / shard | 1 / 4 |
| tp / cp / pp / ep | 1 / 1 / 1 / 1 |
| recorded runs | 5 |

## run results

| run | status | mode | preset | median step | tok/s/device | tok/s/job | peak hbm |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| [npu-fsdp4-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-0ba5e24c](../../../runs/4-card/fsdp4/npu-fsdp4-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-0ba5e24c/readme.md) | completed | profiler-off | distributed | 401.75 ms | 5,097.71 | 20,390.85 | 0.660 GiB |
| [npu-fsdp4-bf16-s20-l8-b64-seq128-seed61-distributed-r1-965e89d3](../../../runs/4-card/fsdp4/npu-fsdp4-bf16-s20-l8-b64-seq128-seed61-distributed-r1-965e89d3/readme.md) | failed | profiler-active | - | - ms | - | - | - GiB |
| [npu-fsdp4-bf16-s20-l8-b64-seq128-seed61-distributed-r2-49400b80](../../../runs/4-card/fsdp4/npu-fsdp4-bf16-s20-l8-b64-seq128-seed61-distributed-r2-49400b80/readme.md) | failed | profiler-active | - | - ms | - | - | - GiB |
| [npu-fsdp4-bf16-s20-l8-b64-seq128-seed61-distributed-r3-d3b92684](../../../runs/4-card/fsdp4/npu-fsdp4-bf16-s20-l8-b64-seq128-seed61-distributed-r3-d3b92684/readme.md) | completed | profiler-active | distributed | 409.04 ms | 5,006.86 | 20,027.44 | - GiB |
| [npu-fsdp4-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-2cf55db3](../../../runs/4-card/fsdp4/npu-fsdp4-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-2cf55db3/readme.md) | completed | profiler-active | overview | 709.53 ms | 2,886.43 | 11,545.73 | - GiB |

## experiment sequence

### 1. npu-fsdp4-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-0ba5e24c

- Status: `completed`
- Started: `2026-08-24T16:34:54.341851+08:00`
- Full process and outputs: [run readme](../../../runs/4-card/fsdp4/npu-fsdp4-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-0ba5e24c/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/4-card/fsdp4/npu-fsdp4-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-0ba5e24c.html`

```bash
/root/miniconda3/envs/torchtitan-0803/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology fsdp4 --preset distributed --profiler-off --visible-devices 1,2,3,4 --steps 20 --skip-steps 8 --warmup-steps 2 --active-steps 3 --replicate 1
```

### 2. npu-fsdp4-bf16-s20-l8-b64-seq128-seed61-distributed-r1-965e89d3

- Status: `failed`
- Started: `2026-08-24T16:53:45.152666+08:00`
- Full process and outputs: [run readme](../../../runs/4-card/fsdp4/npu-fsdp4-bf16-s20-l8-b64-seq128-seed61-distributed-r1-965e89d3/readme.md)

```bash
/root/miniconda3/envs/torchtitan-0803/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology fsdp4 --preset distributed --visible-devices 1,2,3,4 --steps 20 --skip-steps 8 --warmup-steps 2 --active-steps 3 --replicate 1 --cluster
```

### 3. npu-fsdp4-bf16-s20-l8-b64-seq128-seed61-distributed-r2-49400b80

- Status: `failed`
- Started: `2026-08-24T16:56:37.297551+08:00`
- Full process and outputs: [run readme](../../../runs/4-card/fsdp4/npu-fsdp4-bf16-s20-l8-b64-seq128-seed61-distributed-r2-49400b80/readme.md)

```bash
/root/miniconda3/envs/torchtitan-0803/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology fsdp4 --preset distributed --visible-devices 4,5,6,7 --steps 20 --skip-steps 8 --warmup-steps 2 --active-steps 3 --replicate 2 --cluster
```

### 4. npu-fsdp4-bf16-s20-l8-b64-seq128-seed61-distributed-r3-d3b92684

- Status: `completed`
- Started: `2026-08-24T17:02:27.528676+08:00`
- Full process and outputs: [run readme](../../../runs/4-card/fsdp4/npu-fsdp4-bf16-s20-l8-b64-seq128-seed61-distributed-r3-d3b92684/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/4-card/fsdp4/npu-fsdp4-bf16-s20-l8-b64-seq128-seed61-distributed-r3-d3b92684.html`

```bash
/root/miniconda3/envs/torchtitan-0803/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology fsdp4 --preset distributed --visible-devices 1,2,3,5 --steps 20 --skip-steps 8 --warmup-steps 2 --active-steps 3 --replicate 3 --cluster
```

### 5. npu-fsdp4-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-2cf55db3

- Status: `completed`
- Started: `2026-08-31T11:50:17.888430+08:00`
- Full process and outputs: [run readme](../../../runs/4-card/fsdp4/npu-fsdp4-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-2cf55db3/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/4-card/fsdp4/npu-fsdp4-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-2cf55db3.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology all --preset all --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all
```

## current summary

The highest recorded job throughput is 20,390.85 tok/s from `npu-fsdp4-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-0ba5e24c`. Profiler-off runs are throughput evidence; profiler-active runs are attribution evidence only.
