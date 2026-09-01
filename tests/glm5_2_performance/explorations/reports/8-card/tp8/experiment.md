# tp8 experiments

## topology

| field | value |
| --- | --- |
| world size | 8 |
| dp replicate / shard | 1 / 1 |
| tp / cp / pp / ep | 8 / 1 / 1 / 1 |
| recorded runs | 12 |

## run results

| run | status | mode | preset | median step | tok/s/device | tok/s/job | peak hbm |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| [npu-tp8-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-64da983b](../../../runs/8-card/tp8/npu-tp8-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-64da983b/readme.md) | completed | profiler-off | distributed | 4,421.03 ms | 231.62 | 1,852.98 | 0.208 GiB |
| [npu-tp8-bf16-s20-l8-b64-seq128-seed61-distributed-r1-dba2190c](../../../runs/8-card/tp8/npu-tp8-bf16-s20-l8-b64-seq128-seed61-distributed-r1-dba2190c/readme.md) | completed | profiler-active | distributed | 9,097.91 ms | 112.55 | 900.43 | - GiB |
| [npu-tp8-bf16-s30-l8-b64-seq128-seed61-overview-r31-dbbcd393](../../../runs/8-card/tp8/npu-tp8-bf16-s30-l8-b64-seq128-seed61-overview-r31-dbbcd393/readme.md) | failed | profiler-active | - | - ms | - | - | - GiB |
| [npu-tp8-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-7d88d2fc](../../../runs/8-card/tp8/npu-tp8-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-7d88d2fc/readme.md) | completed | profiler-active | overview | 6,274.99 ms | 163.19 | 1,305.50 | - GiB |
| [npu-tp8-bf16-s30-l8-b64-seq128-seed61-distributed-r31-5154b546](../../../runs/8-card/tp8/npu-tp8-bf16-s30-l8-b64-seq128-seed61-distributed-r31-5154b546/readme.md) | completed | profiler-active | distributed | 6,244.72 ms | 163.98 | 1,311.83 | - GiB |
| [npu-tp8-bf16-s30-l8-b64-seq128-seed61-kernel-r31-offline-e0944fdb](../../../runs/8-card/tp8/npu-tp8-bf16-s30-l8-b64-seq128-seed61-kernel-r31-offline-e0944fdb/readme.md) | completed | profiler-active | kernel | 6,262.76 ms | 163.51 | 1,308.05 | - GiB |
| [npu-tp8-bf16-s30-l8-b64-seq128-seed61-operator-r32-offline-b094ba01](../../../runs/8-card/tp8/npu-tp8-bf16-s30-l8-b64-seq128-seed61-operator-r32-offline-b094ba01/readme.md) | completed | profiler-active | operator | 6,209.44 ms | 164.91 | 1,319.28 | - GiB |
| [npu-tp8-bf16-s30-l8-b64-seq128-seed61-memory-r31-offline-1e46461a](../../../runs/8-card/tp8/npu-tp8-bf16-s30-l8-b64-seq128-seed61-memory-r31-offline-1e46461a/readme.md) | command-only | profiler-active | - | - ms | - | - | - GiB |
| [npu-tp8-bf16-s30-l8-b64-seq128-seed61-memory-r31-offline-5d81d702](../../../runs/8-card/tp8/npu-tp8-bf16-s30-l8-b64-seq128-seed61-memory-r31-offline-5d81d702/readme.md) | completed | profiler-active | memory | 6,135.18 ms | 166.91 | 1,335.25 | - GiB |
| [npu-tp8-bf16-s30-l8-b64-seq128-seed61-flamegraph-r31-3499a905](../../../runs/8-card/tp8/npu-tp8-bf16-s30-l8-b64-seq128-seed61-flamegraph-r31-3499a905/readme.md) | completed | profiler-active | flamegraph | 6,998.23 ms | 146.32 | 1,170.58 | - GiB |
| [npu-tp8-bf16-s30-l8-b64-seq128-seed61-runtime-r31-offline-5ec8e8dd](../../../runs/8-card/tp8/npu-tp8-bf16-s30-l8-b64-seq128-seed61-runtime-r31-offline-5ec8e8dd/readme.md) | completed | profiler-active | runtime | 6,418.67 ms | 159.53 | 1,276.28 | - GiB |
| [npu-tp8-bf16-s30-l8-b64-seq128-seed61-system-r31-59670d46](../../../runs/8-card/tp8/npu-tp8-bf16-s30-l8-b64-seq128-seed61-system-r31-59670d46/readme.md) | completed | profiler-active | system | 6,100.42 ms | 167.86 | 1,342.86 | - GiB |

## experiment sequence

### 1. npu-tp8-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-64da983b

- Status: `completed`
- Started: `2026-08-24T17:49:57.398320+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/tp8/npu-tp8-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-64da983b/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/tp8/npu-tp8-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-64da983b.html`

```bash
/root/miniconda3/envs/torchtitan-0803/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology tp8 --preset distributed --profiler-off --visible-devices 0,1,2,3,4,5,6,7 --steps 20 --skip-steps 8 --warmup-steps 2 --active-steps 3 --replicate 1
```

### 2. npu-tp8-bf16-s20-l8-b64-seq128-seed61-distributed-r1-dba2190c

- Status: `completed`
- Started: `2026-08-24T20:42:29.508722+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/tp8/npu-tp8-bf16-s20-l8-b64-seq128-seed61-distributed-r1-dba2190c/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/tp8/npu-tp8-bf16-s20-l8-b64-seq128-seed61-distributed-r1-dba2190c.html`

```bash
/root/miniconda3/envs/torchtitan-0803/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology tp8 --preset distributed --visible-devices 0,1,2,3,4,5,6,7 --steps 20 --skip-steps 8 --warmup-steps 2 --active-steps 3 --replicate 1 --cluster
```

### 3. npu-tp8-bf16-s30-l8-b64-seq128-seed61-overview-r31-dbbcd393

- Status: `failed`
- Started: `2026-08-31T10:11:02.193769+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/tp8/npu-tp8-bf16-s30-l8-b64-seq128-seed61-overview-r31-dbbcd393/readme.md)

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology all --preset all --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all
```

### 4. npu-tp8-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-7d88d2fc

- Status: `completed`
- Started: `2026-08-31T10:19:57.599445+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/tp8/npu-tp8-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-7d88d2fc/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/tp8/npu-tp8-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-7d88d2fc.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology tp8 --preset overview --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all
```

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology all --preset all --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all
```

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology all --preset all --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all
```

### 5. npu-tp8-bf16-s30-l8-b64-seq128-seed61-distributed-r31-5154b546

- Status: `completed`
- Started: `2026-08-31T13:24:01.758579+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/tp8/npu-tp8-bf16-s30-l8-b64-seq128-seed61-distributed-r31-5154b546/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/tp8/npu-tp8-bf16-s30-l8-b64-seq128-seed61-distributed-r31-5154b546.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology all --preset all --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all
```

### 6. npu-tp8-bf16-s30-l8-b64-seq128-seed61-kernel-r31-offline-e0944fdb

- Status: `completed`
- Started: `2026-08-31T19:56:34.727609+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/tp8/npu-tp8-bf16-s30-l8-b64-seq128-seed61-kernel-r31-offline-e0944fdb/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/tp8/npu-tp8-bf16-s30-l8-b64-seq128-seed61-kernel-r31-offline-e0944fdb.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology all --preset kernel --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all
```

### 7. npu-tp8-bf16-s30-l8-b64-seq128-seed61-operator-r32-offline-b094ba01

- Status: `completed`
- Started: `2026-08-31T23:09:35.677451+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/tp8/npu-tp8-bf16-s30-l8-b64-seq128-seed61-operator-r32-offline-b094ba01/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/tp8/npu-tp8-bf16-s30-l8-b64-seq128-seed61-operator-r32-offline-b094ba01.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology tp8 --preset operator --visible-devices 0,1,2,3,4,5,6,7 --replicate 32 --analysis-tools all
```

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --analyze --device npu --topology tp8 --preset operator --visible-devices 0,1,2,3,4,5,6,7 --replicate 32 --analysis-tools all --parse-workers 8
```

### 8. npu-tp8-bf16-s30-l8-b64-seq128-seed61-memory-r31-offline-1e46461a

- Status: `command-only`
- Started: `2026-09-01T00:42:33.268995+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/tp8/npu-tp8-bf16-s30-l8-b64-seq128-seed61-memory-r31-offline-1e46461a/readme.md)

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --capture --device npu --topology tp8 --preset memory --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools none
```

### 9. npu-tp8-bf16-s30-l8-b64-seq128-seed61-memory-r31-offline-5d81d702

- Status: `completed`
- Started: `2026-09-01T09:07:59.072166+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/tp8/npu-tp8-bf16-s30-l8-b64-seq128-seed61-memory-r31-offline-5d81d702/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/tp8/npu-tp8-bf16-s30-l8-b64-seq128-seed61-memory-r31-offline-5d81d702.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --capture --device npu --topology tp8 --preset memory --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools none
```

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --analyze --device npu --topology tp8 --preset memory --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all --parse-workers 8
```

### 10. npu-tp8-bf16-s30-l8-b64-seq128-seed61-flamegraph-r31-3499a905

- Status: `completed`
- Started: `2026-09-01T10:26:08.783480+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/tp8/npu-tp8-bf16-s30-l8-b64-seq128-seed61-flamegraph-r31-3499a905/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/tp8/npu-tp8-bf16-s30-l8-b64-seq128-seed61-flamegraph-r31-3499a905.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --capture --device npu --topology tp8 --preset flamegraph --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools none
```

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --analyze --device npu --topology tp8 --preset flamegraph --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all --parse-workers 8
```

### 11. npu-tp8-bf16-s30-l8-b64-seq128-seed61-runtime-r31-offline-5ec8e8dd

- Status: `completed`
- Started: `2026-09-01T11:43:40.908771+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/tp8/npu-tp8-bf16-s30-l8-b64-seq128-seed61-runtime-r31-offline-5ec8e8dd/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/tp8/npu-tp8-bf16-s30-l8-b64-seq128-seed61-runtime-r31-offline-5ec8e8dd.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --capture --device npu --topology tp8 --preset runtime --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools none
```

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --analyze --device npu --topology tp8 --preset runtime --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all --parse-workers 8
```

### 12. npu-tp8-bf16-s30-l8-b64-seq128-seed61-system-r31-59670d46

- Status: `completed`
- Started: `2026-09-01T13:11:57.266861+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/tp8/npu-tp8-bf16-s30-l8-b64-seq128-seed61-system-r31-59670d46/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/tp8/npu-tp8-bf16-s30-l8-b64-seq128-seed61-system-r31-59670d46.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --capture --device npu --topology tp8 --preset system --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools none
```

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --analyze --device npu --topology tp8 --preset system --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all --parse-workers 8
```

## current summary

The highest recorded job throughput is 1,852.98 tok/s from `npu-tp8-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-64da983b`. Profiler-off runs are throughput evidence; profiler-active runs are attribution evidence only.

## analysis

- [Detailed topology analysis](analysis.md)
