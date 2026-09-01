# fsdp2 experiments

## topology

| field | value |
| --- | --- |
| world size | 2 |
| dp replicate / shard | 1 / 2 |
| tp / cp / pp / ep | 1 / 1 / 1 / 1 |
| recorded runs | 8 |

## run results

| run | status | mode | preset | median step | tok/s/device | tok/s/job | peak hbm |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| [npu-fsdp2-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-aa77d62c](../../../runs/2-card/fsdp2/npu-fsdp2-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-aa77d62c/readme.md) | completed | profiler-active | overview | 1,400.59 ms | 2,924.49 | 5,848.98 | - GiB |
| [npu-fsdp2-bf16-s30-l8-b64-seq128-seed61-distributed-r31-cae33347](../../../runs/2-card/fsdp2/npu-fsdp2-bf16-s30-l8-b64-seq128-seed61-distributed-r31-cae33347/readme.md) | completed | profiler-active | distributed | 1,352.81 ms | 3,027.77 | 6,055.54 | - GiB |
| [npu-fsdp2-bf16-s30-l8-b64-seq128-seed61-kernel-r31-offline-6804134e](../../../runs/2-card/fsdp2/npu-fsdp2-bf16-s30-l8-b64-seq128-seed61-kernel-r31-offline-6804134e/readme.md) | completed | profiler-active | kernel | 1,401.92 ms | 2,921.71 | 5,843.43 | - GiB |
| [npu-fsdp2-bf16-s30-l8-b64-seq128-seed61-operator-r32-offline-2019ca0e](../../../runs/2-card/fsdp2/npu-fsdp2-bf16-s30-l8-b64-seq128-seed61-operator-r32-offline-2019ca0e/readme.md) | completed | profiler-active | operator | 1,392.32 ms | 2,941.85 | 5,883.70 | - GiB |
| [npu-fsdp2-bf16-s30-l8-b64-seq128-seed61-memory-r31-offline-7e7ffacc](../../../runs/2-card/fsdp2/npu-fsdp2-bf16-s30-l8-b64-seq128-seed61-memory-r31-offline-7e7ffacc/readme.md) | completed | profiler-active | memory | 1,394.92 ms | 2,936.36 | 5,872.72 | - GiB |
| [npu-fsdp2-bf16-s30-l8-b64-seq128-seed61-flamegraph-r31-f0be3973](../../../runs/2-card/fsdp2/npu-fsdp2-bf16-s30-l8-b64-seq128-seed61-flamegraph-r31-f0be3973/readme.md) | completed | profiler-active | flamegraph | 1,411.84 ms | 2,901.18 | 5,802.36 | - GiB |
| [npu-fsdp2-bf16-s30-l8-b64-seq128-seed61-runtime-r31-offline-6122539c](../../../runs/2-card/fsdp2/npu-fsdp2-bf16-s30-l8-b64-seq128-seed61-runtime-r31-offline-6122539c/readme.md) | completed | profiler-active | runtime | 1,394.72 ms | 2,936.79 | 5,873.58 | - GiB |
| [npu-fsdp2-bf16-s30-l8-b64-seq128-seed61-system-r31-b50da2fd](../../../runs/2-card/fsdp2/npu-fsdp2-bf16-s30-l8-b64-seq128-seed61-system-r31-b50da2fd/readme.md) | completed | profiler-active | system | 1,408.13 ms | 2,908.82 | 5,817.65 | - GiB |

## experiment sequence

### 1. npu-fsdp2-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-aa77d62c

- Status: `completed`
- Started: `2026-08-31T11:47:00.243062+08:00`
- Full process and outputs: [run readme](../../../runs/2-card/fsdp2/npu-fsdp2-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-aa77d62c/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/2-card/fsdp2/npu-fsdp2-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-aa77d62c.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology all --preset all --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all
```

### 2. npu-fsdp2-bf16-s30-l8-b64-seq128-seed61-distributed-r31-cae33347

- Status: `completed`
- Started: `2026-08-31T17:07:07.506551+08:00`
- Full process and outputs: [run readme](../../../runs/2-card/fsdp2/npu-fsdp2-bf16-s30-l8-b64-seq128-seed61-distributed-r31-cae33347/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/2-card/fsdp2/npu-fsdp2-bf16-s30-l8-b64-seq128-seed61-distributed-r31-cae33347.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topologies fsdp2-tp4-ep8,ddp4,fsdp2,fsdp4,tp2,tp4,cp2,cp4,pp2,pp4,ep2,ep4,fsdp2-tp2 --preset distributed --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all
```

### 3. npu-fsdp2-bf16-s30-l8-b64-seq128-seed61-kernel-r31-offline-6804134e

- Status: `completed`
- Started: `2026-08-31T21:16:00.678057+08:00`
- Full process and outputs: [run readme](../../../runs/2-card/fsdp2/npu-fsdp2-bf16-s30-l8-b64-seq128-seed61-kernel-r31-offline-6804134e/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/2-card/fsdp2/npu-fsdp2-bf16-s30-l8-b64-seq128-seed61-kernel-r31-offline-6804134e.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology all --preset kernel --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all
```

### 4. npu-fsdp2-bf16-s30-l8-b64-seq128-seed61-operator-r32-offline-2019ca0e

- Status: `completed`
- Started: `2026-09-01T00:01:27.236036+08:00`
- Full process and outputs: [run readme](../../../runs/2-card/fsdp2/npu-fsdp2-bf16-s30-l8-b64-seq128-seed61-operator-r32-offline-2019ca0e/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/2-card/fsdp2/npu-fsdp2-bf16-s30-l8-b64-seq128-seed61-operator-r32-offline-2019ca0e.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --capture --device npu --topology fsdp2 --preset operator --visible-devices 0,1,2,3,4,5,6,7 --replicate 32 --analysis-tools none
```

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --analyze --device npu --topology fsdp2 --preset operator --visible-devices 0,1,2,3,4,5,6,7 --replicate 32 --analysis-tools all --parse-workers 8
```

### 5. npu-fsdp2-bf16-s30-l8-b64-seq128-seed61-memory-r31-offline-7e7ffacc

- Status: `completed`
- Started: `2026-09-01T09:42:32.766037+08:00`
- Full process and outputs: [run readme](../../../runs/2-card/fsdp2/npu-fsdp2-bf16-s30-l8-b64-seq128-seed61-memory-r31-offline-7e7ffacc/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/2-card/fsdp2/npu-fsdp2-bf16-s30-l8-b64-seq128-seed61-memory-r31-offline-7e7ffacc.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --capture --device npu --topology fsdp2 --preset memory --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools none
```

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --analyze --device npu --topology fsdp2 --preset memory --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all --parse-workers 8
```

### 6. npu-fsdp2-bf16-s30-l8-b64-seq128-seed61-flamegraph-r31-f0be3973

- Status: `completed`
- Started: `2026-09-01T11:02:06.630134+08:00`
- Full process and outputs: [run readme](../../../runs/2-card/fsdp2/npu-fsdp2-bf16-s30-l8-b64-seq128-seed61-flamegraph-r31-f0be3973/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/2-card/fsdp2/npu-fsdp2-bf16-s30-l8-b64-seq128-seed61-flamegraph-r31-f0be3973.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --capture --device npu --topology fsdp2 --preset flamegraph --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools none
```

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --analyze --device npu --topology fsdp2 --preset flamegraph --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all --parse-workers 8
```

### 7. npu-fsdp2-bf16-s30-l8-b64-seq128-seed61-runtime-r31-offline-6122539c

- Status: `completed`
- Started: `2026-09-01T12:21:03.571206+08:00`
- Full process and outputs: [run readme](../../../runs/2-card/fsdp2/npu-fsdp2-bf16-s30-l8-b64-seq128-seed61-runtime-r31-offline-6122539c/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/2-card/fsdp2/npu-fsdp2-bf16-s30-l8-b64-seq128-seed61-runtime-r31-offline-6122539c.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --capture --device npu --topology fsdp2 --preset runtime --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools none
```

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --analyze --device npu --topology fsdp2 --preset runtime --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all --parse-workers 8
```

### 8. npu-fsdp2-bf16-s30-l8-b64-seq128-seed61-system-r31-b50da2fd

- Status: `completed`
- Started: `2026-09-01T13:47:12.222721+08:00`
- Full process and outputs: [run readme](../../../runs/2-card/fsdp2/npu-fsdp2-bf16-s30-l8-b64-seq128-seed61-system-r31-b50da2fd/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/2-card/fsdp2/npu-fsdp2-bf16-s30-l8-b64-seq128-seed61-system-r31-b50da2fd.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --capture --device npu --topology fsdp2 --preset system --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools none
```

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --analyze --device npu --topology fsdp2 --preset system --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all --parse-workers 8
```

## current summary

The highest recorded job throughput is 6,055.54 tok/s from `npu-fsdp2-bf16-s30-l8-b64-seq128-seed61-distributed-r31-cae33347`. Profiler-off runs are throughput evidence; profiler-active runs are attribution evidence only.
