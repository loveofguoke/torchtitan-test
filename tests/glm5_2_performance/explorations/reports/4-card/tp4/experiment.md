# tp4 experiments

## topology

| field | value |
| --- | --- |
| world size | 4 |
| dp replicate / shard | 1 / 1 |
| tp / cp / pp / ep | 4 / 1 / 1 / 1 |
| recorded runs | 9 |

## run results

| run | status | mode | preset | median step | tok/s/device | tok/s/job | peak hbm |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| [npu-tp4-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-8aa36e1b](../../../runs/4-card/tp4/npu-tp4-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-8aa36e1b/readme.md) | completed | profiler-off | distributed | 4,330.00 ms | 472.98 | 1,891.92 | 0.290 GiB |
| [npu-tp4-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-53fedaa4](../../../runs/4-card/tp4/npu-tp4-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-53fedaa4/readme.md) | completed | profiler-active | overview | 6,277.27 ms | 326.26 | 1,305.03 | - GiB |
| [npu-tp4-bf16-s30-l8-b64-seq128-seed61-distributed-r31-a63f17d4](../../../runs/4-card/tp4/npu-tp4-bf16-s30-l8-b64-seq128-seed61-distributed-r31-a63f17d4/readme.md) | completed | profiler-active | distributed | 6,184.96 ms | 331.13 | 1,324.50 | - GiB |
| [npu-tp4-bf16-s30-l8-b64-seq128-seed61-kernel-r31-offline-e59509fc](../../../runs/4-card/tp4/npu-tp4-bf16-s30-l8-b64-seq128-seed61-kernel-r31-offline-e59509fc/readme.md) | completed | profiler-active | kernel | 6,139.48 ms | 333.58 | 1,334.31 | - GiB |
| [npu-tp4-bf16-s30-l8-b64-seq128-seed61-operator-r32-offline-6dd9c1a7](../../../runs/4-card/tp4/npu-tp4-bf16-s30-l8-b64-seq128-seed61-operator-r32-offline-6dd9c1a7/readme.md) | completed | profiler-active | operator | 6,030.11 ms | 339.63 | 1,358.51 | - GiB |
| [npu-tp4-bf16-s30-l8-b64-seq128-seed61-memory-r31-offline-eff88ea0](../../../runs/4-card/tp4/npu-tp4-bf16-s30-l8-b64-seq128-seed61-memory-r31-offline-eff88ea0/readme.md) | completed | profiler-active | memory | 6,543.66 ms | 312.97 | 1,251.90 | - GiB |
| [npu-tp4-bf16-s30-l8-b64-seq128-seed61-flamegraph-r31-a72b8daf](../../../runs/4-card/tp4/npu-tp4-bf16-s30-l8-b64-seq128-seed61-flamegraph-r31-a72b8daf/readme.md) | completed | profiler-active | flamegraph | 6,061.85 ms | 337.85 | 1,351.40 | - GiB |
| [npu-tp4-bf16-s30-l8-b64-seq128-seed61-runtime-r31-offline-51a86748](../../../runs/4-card/tp4/npu-tp4-bf16-s30-l8-b64-seq128-seed61-runtime-r31-offline-51a86748/readme.md) | completed | profiler-active | runtime | 5,965.75 ms | 343.29 | 1,373.17 | - GiB |
| [npu-tp4-bf16-s30-l8-b64-seq128-seed61-system-r31-036d5d91](../../../runs/4-card/tp4/npu-tp4-bf16-s30-l8-b64-seq128-seed61-system-r31-036d5d91/readme.md) | captured | profiler-active | system | - ms | - | - | - GiB |

## experiment sequence

### 1. npu-tp4-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-8aa36e1b

- Status: `completed`
- Started: `2026-08-24T16:35:47.669094+08:00`
- Full process and outputs: [run readme](../../../runs/4-card/tp4/npu-tp4-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-8aa36e1b/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/4-card/tp4/npu-tp4-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-8aa36e1b.html`

```bash
/root/miniconda3/envs/torchtitan-0803/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology tp4 --preset distributed --profiler-off --visible-devices 1,2,3,4 --steps 20 --skip-steps 8 --warmup-steps 2 --active-steps 3 --replicate 1
```

### 2. npu-tp4-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-53fedaa4

- Status: `completed`
- Started: `2026-08-31T12:01:06.238930+08:00`
- Full process and outputs: [run readme](../../../runs/4-card/tp4/npu-tp4-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-53fedaa4/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/4-card/tp4/npu-tp4-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-53fedaa4.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology all --preset all --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all
```

### 3. npu-tp4-bf16-s30-l8-b64-seq128-seed61-distributed-r31-a63f17d4

- Status: `completed`
- Started: `2026-08-31T17:35:18.290882+08:00`
- Full process and outputs: [run readme](../../../runs/4-card/tp4/npu-tp4-bf16-s30-l8-b64-seq128-seed61-distributed-r31-a63f17d4/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/4-card/tp4/npu-tp4-bf16-s30-l8-b64-seq128-seed61-distributed-r31-a63f17d4.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topologies fsdp2-tp4-ep8,ddp4,fsdp2,fsdp4,tp2,tp4,cp2,cp4,pp2,pp4,ep2,ep4,fsdp2-tp2 --preset distributed --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all
```

### 4. npu-tp4-bf16-s30-l8-b64-seq128-seed61-kernel-r31-offline-e59509fc

- Status: `completed`
- Started: `2026-08-31T21:32:03.085076+08:00`
- Full process and outputs: [run readme](../../../runs/4-card/tp4/npu-tp4-bf16-s30-l8-b64-seq128-seed61-kernel-r31-offline-e59509fc/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/4-card/tp4/npu-tp4-bf16-s30-l8-b64-seq128-seed61-kernel-r31-offline-e59509fc.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology all --preset kernel --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all
```

### 5. npu-tp4-bf16-s30-l8-b64-seq128-seed61-operator-r32-offline-6dd9c1a7

- Status: `completed`
- Started: `2026-09-01T00:08:42.226144+08:00`
- Full process and outputs: [run readme](../../../runs/4-card/tp4/npu-tp4-bf16-s30-l8-b64-seq128-seed61-operator-r32-offline-6dd9c1a7/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/4-card/tp4/npu-tp4-bf16-s30-l8-b64-seq128-seed61-operator-r32-offline-6dd9c1a7.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --capture --device npu --topology tp4 --preset operator --visible-devices 0,1,2,3,4,5,6,7 --replicate 32 --analysis-tools none
```

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --analyze --device npu --topology tp4 --preset operator --visible-devices 0,1,2,3,4,5,6,7 --replicate 32 --analysis-tools all --parse-workers 8
```

### 6. npu-tp4-bf16-s30-l8-b64-seq128-seed61-memory-r31-offline-eff88ea0

- Status: `completed`
- Started: `2026-09-01T09:49:04.493190+08:00`
- Full process and outputs: [run readme](../../../runs/4-card/tp4/npu-tp4-bf16-s30-l8-b64-seq128-seed61-memory-r31-offline-eff88ea0/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/4-card/tp4/npu-tp4-bf16-s30-l8-b64-seq128-seed61-memory-r31-offline-eff88ea0.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --capture --device npu --topology tp4 --preset memory --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools none
```

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --analyze --device npu --topology tp4 --preset memory --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all --parse-workers 8
```

### 7. npu-tp4-bf16-s30-l8-b64-seq128-seed61-flamegraph-r31-a72b8daf

- Status: `completed`
- Started: `2026-09-01T11:08:28.784484+08:00`
- Full process and outputs: [run readme](../../../runs/4-card/tp4/npu-tp4-bf16-s30-l8-b64-seq128-seed61-flamegraph-r31-a72b8daf/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/4-card/tp4/npu-tp4-bf16-s30-l8-b64-seq128-seed61-flamegraph-r31-a72b8daf.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --capture --device npu --topology tp4 --preset flamegraph --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools none
```

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --analyze --device npu --topology tp4 --preset flamegraph --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all --parse-workers 8
```

### 8. npu-tp4-bf16-s30-l8-b64-seq128-seed61-runtime-r31-offline-51a86748

- Status: `completed`
- Started: `2026-09-01T12:28:18.937347+08:00`
- Full process and outputs: [run readme](../../../runs/4-card/tp4/npu-tp4-bf16-s30-l8-b64-seq128-seed61-runtime-r31-offline-51a86748/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/4-card/tp4/npu-tp4-bf16-s30-l8-b64-seq128-seed61-runtime-r31-offline-51a86748.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --capture --device npu --topology tp4 --preset runtime --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools none
```

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --analyze --device npu --topology tp4 --preset runtime --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all --parse-workers 8
```

### 9. npu-tp4-bf16-s30-l8-b64-seq128-seed61-system-r31-036d5d91

- Status: `captured`
- Started: `2026-09-01T13:53:59.832522+08:00`
- Full process and outputs: [run readme](../../../runs/4-card/tp4/npu-tp4-bf16-s30-l8-b64-seq128-seed61-system-r31-036d5d91/readme.md)

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --capture --device npu --topology tp4 --preset system --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools none
```

## current summary

The highest recorded job throughput is 1,891.92 tok/s from `npu-tp4-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-8aa36e1b`. Profiler-off runs are throughput evidence; profiler-active runs are attribution evidence only.
