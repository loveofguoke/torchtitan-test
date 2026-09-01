# cp2 experiments

## topology

| field | value |
| --- | --- |
| world size | 2 |
| dp replicate / shard | 1 / 1 |
| tp / cp / pp / ep | 1 / 2 / 1 / 1 |
| recorded runs | 8 |

## run results

| run | status | mode | preset | median step | tok/s/device | tok/s/job | peak hbm |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| [npu-cp2-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-538f588d](../../../runs/2-card/cp2/npu-cp2-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-538f588d/readme.md) | completed | profiler-active | overview | 2,821.60 ms | 1,451.66 | 2,903.32 | - GiB |
| [npu-cp2-bf16-s30-l8-b64-seq128-seed61-distributed-r31-87c046d2](../../../runs/2-card/cp2/npu-cp2-bf16-s30-l8-b64-seq128-seed61-distributed-r31-87c046d2/readme.md) | completed | profiler-active | distributed | 2,821.25 ms | 1,451.84 | 2,903.68 | - GiB |
| [npu-cp2-bf16-s30-l8-b64-seq128-seed61-kernel-r31-offline-d49e5897](../../../runs/2-card/cp2/npu-cp2-bf16-s30-l8-b64-seq128-seed61-kernel-r31-offline-d49e5897/readme.md) | completed | profiler-active | kernel | 2,724.98 ms | 1,503.13 | 3,006.26 | - GiB |
| [npu-cp2-bf16-s30-l8-b64-seq128-seed61-operator-r32-offline-dcdd1fd1](../../../runs/2-card/cp2/npu-cp2-bf16-s30-l8-b64-seq128-seed61-operator-r32-offline-dcdd1fd1/readme.md) | completed | profiler-active | operator | 2,791.53 ms | 1,467.30 | 2,934.59 | - GiB |
| [npu-cp2-bf16-s30-l8-b64-seq128-seed61-memory-r31-offline-0df4d1d3](../../../runs/2-card/cp2/npu-cp2-bf16-s30-l8-b64-seq128-seed61-memory-r31-offline-0df4d1d3/readme.md) | completed | profiler-active | memory | 2,724.11 ms | 1,503.61 | 3,007.22 | - GiB |
| [npu-cp2-bf16-s30-l8-b64-seq128-seed61-flamegraph-r31-6a83e9c3](../../../runs/2-card/cp2/npu-cp2-bf16-s30-l8-b64-seq128-seed61-flamegraph-r31-6a83e9c3/readme.md) | completed | profiler-active | flamegraph | 2,685.47 ms | 1,525.25 | 3,050.49 | - GiB |
| [npu-cp2-bf16-s30-l8-b64-seq128-seed61-runtime-r31-offline-287fd715](../../../runs/2-card/cp2/npu-cp2-bf16-s30-l8-b64-seq128-seed61-runtime-r31-offline-287fd715/readme.md) | completed | profiler-active | runtime | 2,734.81 ms | 1,497.73 | 2,995.46 | - GiB |
| [npu-cp2-bf16-s30-l8-b64-seq128-seed61-system-r31-b0195d24](../../../runs/2-card/cp2/npu-cp2-bf16-s30-l8-b64-seq128-seed61-system-r31-b0195d24/readme.md) | captured | profiler-active | system | - ms | - | - | - GiB |

## experiment sequence

### 1. npu-cp2-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-538f588d

- Status: `completed`
- Started: `2026-08-31T12:10:01.756270+08:00`
- Full process and outputs: [run readme](../../../runs/2-card/cp2/npu-cp2-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-538f588d/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/2-card/cp2/npu-cp2-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-538f588d.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology all --preset all --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all
```

### 2. npu-cp2-bf16-s30-l8-b64-seq128-seed61-distributed-r31-87c046d2

- Status: `completed`
- Started: `2026-08-31T17:53:08.635532+08:00`
- Full process and outputs: [run readme](../../../runs/2-card/cp2/npu-cp2-bf16-s30-l8-b64-seq128-seed61-distributed-r31-87c046d2/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/2-card/cp2/npu-cp2-bf16-s30-l8-b64-seq128-seed61-distributed-r31-87c046d2.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topologies fsdp2-tp4-ep8,ddp4,fsdp2,fsdp4,tp2,tp4,cp2,cp4,pp2,pp4,ep2,ep4,fsdp2-tp2 --preset distributed --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all
```

### 3. npu-cp2-bf16-s30-l8-b64-seq128-seed61-kernel-r31-offline-d49e5897

- Status: `completed`
- Started: `2026-08-31T21:42:01.428498+08:00`
- Full process and outputs: [run readme](../../../runs/2-card/cp2/npu-cp2-bf16-s30-l8-b64-seq128-seed61-kernel-r31-offline-d49e5897/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/2-card/cp2/npu-cp2-bf16-s30-l8-b64-seq128-seed61-kernel-r31-offline-d49e5897.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology all --preset kernel --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all
```

### 4. npu-cp2-bf16-s30-l8-b64-seq128-seed61-operator-r32-offline-dcdd1fd1

- Status: `completed`
- Started: `2026-09-01T00:13:06.602624+08:00`
- Full process and outputs: [run readme](../../../runs/2-card/cp2/npu-cp2-bf16-s30-l8-b64-seq128-seed61-operator-r32-offline-dcdd1fd1/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/2-card/cp2/npu-cp2-bf16-s30-l8-b64-seq128-seed61-operator-r32-offline-dcdd1fd1.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --capture --device npu --topology cp2 --preset operator --visible-devices 0,1,2,3,4,5,6,7 --replicate 32 --analysis-tools none
```

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --analyze --device npu --topology cp2 --preset operator --visible-devices 0,1,2,3,4,5,6,7 --replicate 32 --analysis-tools all --parse-workers 8
```

### 5. npu-cp2-bf16-s30-l8-b64-seq128-seed61-memory-r31-offline-0df4d1d3

- Status: `completed`
- Started: `2026-09-01T09:53:16.961069+08:00`
- Full process and outputs: [run readme](../../../runs/2-card/cp2/npu-cp2-bf16-s30-l8-b64-seq128-seed61-memory-r31-offline-0df4d1d3/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/2-card/cp2/npu-cp2-bf16-s30-l8-b64-seq128-seed61-memory-r31-offline-0df4d1d3.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --capture --device npu --topology cp2 --preset memory --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools none
```

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --analyze --device npu --topology cp2 --preset memory --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all --parse-workers 8
```

### 6. npu-cp2-bf16-s30-l8-b64-seq128-seed61-flamegraph-r31-6a83e9c3

- Status: `completed`
- Started: `2026-09-01T11:12:24.826859+08:00`
- Full process and outputs: [run readme](../../../runs/2-card/cp2/npu-cp2-bf16-s30-l8-b64-seq128-seed61-flamegraph-r31-6a83e9c3/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/2-card/cp2/npu-cp2-bf16-s30-l8-b64-seq128-seed61-flamegraph-r31-6a83e9c3.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --capture --device npu --topology cp2 --preset flamegraph --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools none
```

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --analyze --device npu --topology cp2 --preset flamegraph --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all --parse-workers 8
```

### 7. npu-cp2-bf16-s30-l8-b64-seq128-seed61-runtime-r31-offline-287fd715

- Status: `completed`
- Started: `2026-09-01T12:32:47.291982+08:00`
- Full process and outputs: [run readme](../../../runs/2-card/cp2/npu-cp2-bf16-s30-l8-b64-seq128-seed61-runtime-r31-offline-287fd715/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/2-card/cp2/npu-cp2-bf16-s30-l8-b64-seq128-seed61-runtime-r31-offline-287fd715.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --capture --device npu --topology cp2 --preset runtime --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools none
```

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --analyze --device npu --topology cp2 --preset runtime --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all --parse-workers 8
```

### 8. npu-cp2-bf16-s30-l8-b64-seq128-seed61-system-r31-b0195d24

- Status: `captured`
- Started: `2026-09-01T13:58:03.035407+08:00`
- Full process and outputs: [run readme](../../../runs/2-card/cp2/npu-cp2-bf16-s30-l8-b64-seq128-seed61-system-r31-b0195d24/readme.md)

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --capture --device npu --topology cp2 --preset system --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools none
```

## current summary

The highest recorded job throughput is 3,050.49 tok/s from `npu-cp2-bf16-s30-l8-b64-seq128-seed61-flamegraph-r31-6a83e9c3`. Profiler-off runs are throughput evidence; profiler-active runs are attribution evidence only.
