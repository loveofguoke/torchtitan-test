# pp4 experiments

## topology

| field | value |
| --- | --- |
| world size | 4 |
| dp replicate / shard | 1 / 1 |
| tp / cp / pp / ep | 1 / 1 / 4 / 1 |
| recorded runs | 9 |

## run results

| run | status | mode | preset | median step | tok/s/device | tok/s/job | peak hbm |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| [npu-pp4-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-d56db519](../../../runs/4-card/pp4/npu-pp4-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-d56db519/readme.md) | completed | profiler-off | distributed | 3,175.36 ms | 644.97 | 2,579.88 | 0.066 GiB |
| [npu-pp4-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-84d3f848](../../../runs/4-card/pp4/npu-pp4-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-84d3f848/readme.md) | completed | profiler-active | overview | 6,567.07 ms | 311.86 | 1,247.44 | - GiB |
| [npu-pp4-bf16-s30-l8-b64-seq128-seed61-distributed-r31-2f5e0b40](../../../runs/4-card/pp4/npu-pp4-bf16-s30-l8-b64-seq128-seed61-distributed-r31-2f5e0b40/readme.md) | completed | profiler-active | distributed | 6,210.33 ms | 329.77 | 1,319.09 | - GiB |
| [npu-pp4-bf16-s30-l8-b64-seq128-seed61-kernel-r31-offline-132b4917](../../../runs/4-card/pp4/npu-pp4-bf16-s30-l8-b64-seq128-seed61-kernel-r31-offline-132b4917/readme.md) | completed | profiler-active | kernel | 6,438.26 ms | 318.10 | 1,272.39 | - GiB |
| [npu-pp4-bf16-s30-l8-b64-seq128-seed61-operator-r32-offline-b6e30419](../../../runs/4-card/pp4/npu-pp4-bf16-s30-l8-b64-seq128-seed61-operator-r32-offline-b6e30419/readme.md) | completed | profiler-active | operator | 6,330.68 ms | 323.50 | 1,294.02 | - GiB |
| [npu-pp4-bf16-s30-l8-b64-seq128-seed61-memory-r31-offline-88551f56](../../../runs/4-card/pp4/npu-pp4-bf16-s30-l8-b64-seq128-seed61-memory-r31-offline-88551f56/readme.md) | completed | profiler-active | memory | 6,355.23 ms | 322.25 | 1,289.02 | - GiB |
| [npu-pp4-bf16-s30-l8-b64-seq128-seed61-flamegraph-r31-493f5e38](../../../runs/4-card/pp4/npu-pp4-bf16-s30-l8-b64-seq128-seed61-flamegraph-r31-493f5e38/readme.md) | completed | profiler-active | flamegraph | 6,400.29 ms | 319.99 | 1,279.94 | - GiB |
| [npu-pp4-bf16-s30-l8-b64-seq128-seed61-runtime-r31-offline-3794b8a2](../../../runs/4-card/pp4/npu-pp4-bf16-s30-l8-b64-seq128-seed61-runtime-r31-offline-3794b8a2/readme.md) | completed | profiler-active | runtime | 6,315.47 ms | 324.28 | 1,297.13 | - GiB |
| [npu-pp4-bf16-s30-l8-b64-seq128-seed61-system-r31-4951b5db](../../../runs/4-card/pp4/npu-pp4-bf16-s30-l8-b64-seq128-seed61-system-r31-4951b5db/readme.md) | captured | profiler-active | system | - ms | - | - | - GiB |

## experiment sequence

### 1. npu-pp4-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-d56db519

- Status: `completed`
- Started: `2026-08-24T16:39:22.693361+08:00`
- Full process and outputs: [run readme](../../../runs/4-card/pp4/npu-pp4-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-d56db519/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/4-card/pp4/npu-pp4-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-d56db519.html`

```bash
/root/miniconda3/envs/torchtitan-0803/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology pp4 --preset distributed --profiler-off --visible-devices 1,2,3,4 --steps 20 --skip-steps 8 --warmup-steps 2 --active-steps 3 --replicate 1
```

### 2. npu-pp4-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-84d3f848

- Status: `completed`
- Started: `2026-08-31T12:38:05.212941+08:00`
- Full process and outputs: [run readme](../../../runs/4-card/pp4/npu-pp4-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-84d3f848/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/4-card/pp4/npu-pp4-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-84d3f848.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology all --preset all --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all
```

### 3. npu-pp4-bf16-s30-l8-b64-seq128-seed61-distributed-r31-2f5e0b40

- Status: `completed`
- Started: `2026-08-31T18:54:28.775057+08:00`
- Full process and outputs: [run readme](../../../runs/4-card/pp4/npu-pp4-bf16-s30-l8-b64-seq128-seed61-distributed-r31-2f5e0b40/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/4-card/pp4/npu-pp4-bf16-s30-l8-b64-seq128-seed61-distributed-r31-2f5e0b40.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topologies fsdp2-tp4-ep8,ddp4,fsdp2,fsdp4,tp2,tp4,cp2,cp4,pp2,pp4,ep2,ep4,fsdp2-tp2 --preset distributed --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all
```

### 4. npu-pp4-bf16-s30-l8-b64-seq128-seed61-kernel-r31-offline-132b4917

- Status: `completed`
- Started: `2026-08-31T22:17:06.452457+08:00`
- Full process and outputs: [run readme](../../../runs/4-card/pp4/npu-pp4-bf16-s30-l8-b64-seq128-seed61-kernel-r31-offline-132b4917/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/4-card/pp4/npu-pp4-bf16-s30-l8-b64-seq128-seed61-kernel-r31-offline-132b4917.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology all --preset kernel --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all
```

### 5. npu-pp4-bf16-s30-l8-b64-seq128-seed61-operator-r32-offline-b6e30419

- Status: `completed`
- Started: `2026-09-01T00:25:20.930357+08:00`
- Full process and outputs: [run readme](../../../runs/4-card/pp4/npu-pp4-bf16-s30-l8-b64-seq128-seed61-operator-r32-offline-b6e30419/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/4-card/pp4/npu-pp4-bf16-s30-l8-b64-seq128-seed61-operator-r32-offline-b6e30419.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --capture --device npu --topology pp4 --preset operator --visible-devices 0,1,2,3,4,5,6,7 --replicate 32 --analysis-tools none
```

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --analyze --device npu --topology pp4 --preset operator --visible-devices 0,1,2,3,4,5,6,7 --replicate 32 --analysis-tools all --parse-workers 8
```

### 6. npu-pp4-bf16-s30-l8-b64-seq128-seed61-memory-r31-offline-88551f56

- Status: `completed`
- Started: `2026-09-01T10:03:53.665842+08:00`
- Full process and outputs: [run readme](../../../runs/4-card/pp4/npu-pp4-bf16-s30-l8-b64-seq128-seed61-memory-r31-offline-88551f56/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/4-card/pp4/npu-pp4-bf16-s30-l8-b64-seq128-seed61-memory-r31-offline-88551f56.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --capture --device npu --topology pp4 --preset memory --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools none
```

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --analyze --device npu --topology pp4 --preset memory --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all --parse-workers 8
```

### 7. npu-pp4-bf16-s30-l8-b64-seq128-seed61-flamegraph-r31-493f5e38

- Status: `completed`
- Started: `2026-09-01T11:22:46.448223+08:00`
- Full process and outputs: [run readme](../../../runs/4-card/pp4/npu-pp4-bf16-s30-l8-b64-seq128-seed61-flamegraph-r31-493f5e38/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/4-card/pp4/npu-pp4-bf16-s30-l8-b64-seq128-seed61-flamegraph-r31-493f5e38.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --capture --device npu --topology pp4 --preset flamegraph --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools none
```

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --analyze --device npu --topology pp4 --preset flamegraph --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all --parse-workers 8
```

### 8. npu-pp4-bf16-s30-l8-b64-seq128-seed61-runtime-r31-offline-3794b8a2

- Status: `completed`
- Started: `2026-09-01T12:45:21.780172+08:00`
- Full process and outputs: [run readme](../../../runs/4-card/pp4/npu-pp4-bf16-s30-l8-b64-seq128-seed61-runtime-r31-offline-3794b8a2/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/4-card/pp4/npu-pp4-bf16-s30-l8-b64-seq128-seed61-runtime-r31-offline-3794b8a2.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --capture --device npu --topology pp4 --preset runtime --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools none
```

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --analyze --device npu --topology pp4 --preset runtime --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all --parse-workers 8
```

### 9. npu-pp4-bf16-s30-l8-b64-seq128-seed61-system-r31-4951b5db

- Status: `captured`
- Started: `2026-09-01T14:09:01.223183+08:00`
- Full process and outputs: [run readme](../../../runs/4-card/pp4/npu-pp4-bf16-s30-l8-b64-seq128-seed61-system-r31-4951b5db/readme.md)

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --capture --device npu --topology pp4 --preset system --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools none
```

## current summary

The highest recorded job throughput is 2,579.88 tok/s from `npu-pp4-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-d56db519`. Profiler-off runs are throughput evidence; profiler-active runs are attribution evidence only.
