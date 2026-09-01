# pp2 experiments

## topology

| field | value |
| --- | --- |
| world size | 2 |
| dp replicate / shard | 1 / 1 |
| tp / cp / pp / ep | 1 / 1 / 2 / 1 |
| recorded runs | 8 |

## run results

| run | status | mode | preset | median step | tok/s/device | tok/s/job | peak hbm |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| [npu-pp2-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-e9783e4b](../../../runs/2-card/pp2/npu-pp2-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-e9783e4b/readme.md) | completed | profiler-active | overview | 9,946.33 ms | 411.81 | 823.62 | - GiB |
| [npu-pp2-bf16-s30-l8-b64-seq128-seed61-distributed-r31-f0724e00](../../../runs/2-card/pp2/npu-pp2-bf16-s30-l8-b64-seq128-seed61-distributed-r31-f0724e00/readme.md) | completed | profiler-active | distributed | 10,353.57 ms | 395.61 | 791.22 | - GiB |
| [npu-pp2-bf16-s30-l8-b64-seq128-seed61-kernel-r31-offline-50aea613](../../../runs/2-card/pp2/npu-pp2-bf16-s30-l8-b64-seq128-seed61-kernel-r31-offline-50aea613/readme.md) | completed | profiler-active | kernel | 10,298.16 ms | 397.74 | 795.48 | - GiB |
| [npu-pp2-bf16-s30-l8-b64-seq128-seed61-operator-r32-offline-6d100381](../../../runs/2-card/pp2/npu-pp2-bf16-s30-l8-b64-seq128-seed61-operator-r32-offline-6d100381/readme.md) | completed | profiler-active | operator | 10,193.85 ms | 401.81 | 803.62 | - GiB |
| [npu-pp2-bf16-s30-l8-b64-seq128-seed61-memory-r31-offline-6f0bd5ed](../../../runs/2-card/pp2/npu-pp2-bf16-s30-l8-b64-seq128-seed61-memory-r31-offline-6f0bd5ed/readme.md) | completed | profiler-active | memory | 9,999.09 ms | 409.64 | 819.27 | - GiB |
| [npu-pp2-bf16-s30-l8-b64-seq128-seed61-flamegraph-r31-74d27d73](../../../runs/2-card/pp2/npu-pp2-bf16-s30-l8-b64-seq128-seed61-flamegraph-r31-74d27d73/readme.md) | completed | profiler-active | flamegraph | 9,958.14 ms | 411.32 | 822.64 | - GiB |
| [npu-pp2-bf16-s30-l8-b64-seq128-seed61-runtime-r31-offline-24c76351](../../../runs/2-card/pp2/npu-pp2-bf16-s30-l8-b64-seq128-seed61-runtime-r31-offline-24c76351/readme.md) | completed | profiler-active | runtime | 10,156.54 ms | 403.29 | 806.57 | - GiB |
| [npu-pp2-bf16-s30-l8-b64-seq128-seed61-system-r31-ff4e4447](../../../runs/2-card/pp2/npu-pp2-bf16-s30-l8-b64-seq128-seed61-system-r31-ff4e4447/readme.md) | captured | profiler-active | system | - ms | - | - | - GiB |

## experiment sequence

### 1. npu-pp2-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-e9783e4b

- Status: `completed`
- Started: `2026-08-31T12:22:17.071116+08:00`
- Full process and outputs: [run readme](../../../runs/2-card/pp2/npu-pp2-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-e9783e4b/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/2-card/pp2/npu-pp2-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-e9783e4b.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology all --preset all --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all
```

### 2. npu-pp2-bf16-s30-l8-b64-seq128-seed61-distributed-r31-f0724e00

- Status: `completed`
- Started: `2026-08-31T18:17:03.631812+08:00`
- Full process and outputs: [run readme](../../../runs/2-card/pp2/npu-pp2-bf16-s30-l8-b64-seq128-seed61-distributed-r31-f0724e00/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/2-card/pp2/npu-pp2-bf16-s30-l8-b64-seq128-seed61-distributed-r31-f0724e00.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topologies fsdp2-tp4-ep8,ddp4,fsdp2,fsdp4,tp2,tp4,cp2,cp4,pp2,pp4,ep2,ep4,fsdp2-tp2 --preset distributed --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all
```

### 3. npu-pp2-bf16-s30-l8-b64-seq128-seed61-kernel-r31-offline-50aea613

- Status: `completed`
- Started: `2026-08-31T21:56:20.210582+08:00`
- Full process and outputs: [run readme](../../../runs/2-card/pp2/npu-pp2-bf16-s30-l8-b64-seq128-seed61-kernel-r31-offline-50aea613/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/2-card/pp2/npu-pp2-bf16-s30-l8-b64-seq128-seed61-kernel-r31-offline-50aea613.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology all --preset kernel --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all
```

### 4. npu-pp2-bf16-s30-l8-b64-seq128-seed61-operator-r32-offline-6d100381

- Status: `completed`
- Started: `2026-09-01T00:18:16.770427+08:00`
- Full process and outputs: [run readme](../../../runs/2-card/pp2/npu-pp2-bf16-s30-l8-b64-seq128-seed61-operator-r32-offline-6d100381/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/2-card/pp2/npu-pp2-bf16-s30-l8-b64-seq128-seed61-operator-r32-offline-6d100381.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --capture --device npu --topology pp2 --preset operator --visible-devices 0,1,2,3,4,5,6,7 --replicate 32 --analysis-tools none
```

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --analyze --device npu --topology pp2 --preset operator --visible-devices 0,1,2,3,4,5,6,7 --replicate 32 --analysis-tools all --parse-workers 8
```

### 5. npu-pp2-bf16-s30-l8-b64-seq128-seed61-memory-r31-offline-6f0bd5ed

- Status: `completed`
- Started: `2026-09-01T09:57:43.909659+08:00`
- Full process and outputs: [run readme](../../../runs/2-card/pp2/npu-pp2-bf16-s30-l8-b64-seq128-seed61-memory-r31-offline-6f0bd5ed/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/2-card/pp2/npu-pp2-bf16-s30-l8-b64-seq128-seed61-memory-r31-offline-6f0bd5ed.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --capture --device npu --topology pp2 --preset memory --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools none
```

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --analyze --device npu --topology pp2 --preset memory --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all --parse-workers 8
```

### 6. npu-pp2-bf16-s30-l8-b64-seq128-seed61-flamegraph-r31-74d27d73

- Status: `completed`
- Started: `2026-09-01T11:16:40.967005+08:00`
- Full process and outputs: [run readme](../../../runs/2-card/pp2/npu-pp2-bf16-s30-l8-b64-seq128-seed61-flamegraph-r31-74d27d73/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/2-card/pp2/npu-pp2-bf16-s30-l8-b64-seq128-seed61-flamegraph-r31-74d27d73.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --capture --device npu --topology pp2 --preset flamegraph --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools none
```

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --analyze --device npu --topology pp2 --preset flamegraph --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all --parse-workers 8
```

### 7. npu-pp2-bf16-s30-l8-b64-seq128-seed61-runtime-r31-offline-24c76351

- Status: `completed`
- Started: `2026-09-01T12:38:03.132052+08:00`
- Full process and outputs: [run readme](../../../runs/2-card/pp2/npu-pp2-bf16-s30-l8-b64-seq128-seed61-runtime-r31-offline-24c76351/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/2-card/pp2/npu-pp2-bf16-s30-l8-b64-seq128-seed61-runtime-r31-offline-24c76351.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --capture --device npu --topology pp2 --preset runtime --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools none
```

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --analyze --device npu --topology pp2 --preset runtime --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all --parse-workers 8
```

### 8. npu-pp2-bf16-s30-l8-b64-seq128-seed61-system-r31-ff4e4447

- Status: `captured`
- Started: `2026-09-01T14:02:26.871773+08:00`
- Full process and outputs: [run readme](../../../runs/2-card/pp2/npu-pp2-bf16-s30-l8-b64-seq128-seed61-system-r31-ff4e4447/readme.md)

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --capture --device npu --topology pp2 --preset system --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools none
```

## current summary

The highest recorded job throughput is 823.62 tok/s from `npu-pp2-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-e9783e4b`. Profiler-off runs are throughput evidence; profiler-active runs are attribution evidence only.
