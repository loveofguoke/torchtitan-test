# ep8 experiments

## topology

| field | value |
| --- | --- |
| world size | 8 |
| dp replicate / shard | 1 / 8 |
| tp / cp / pp / ep | 1 / 1 / 1 / 8 |
| recorded runs | 13 |

## run results

| run | status | mode | preset | median step | tok/s/device | tok/s/job | peak hbm |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| [npu-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-50c6107d](../../../runs/8-card/ep8/npu-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-50c6107d/readme.md) | completed | profiler-off | distributed | 267.04 ms | 3,834.67 | 30,677.33 | 0.609 GiB |
| [npu-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-r1-17dbf09f](../../../runs/8-card/ep8/npu-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-r1-17dbf09f/readme.md) | failed | profiler-active | - | - ms | - | - | - GiB |
| [npu-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-r2-7d87cd5f](../../../runs/8-card/ep8/npu-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-r2-7d87cd5f/readme.md) | failed | profiler-active | - | - ms | - | - | - GiB |
| [npu-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-r3-8f7adec9](../../../runs/8-card/ep8/npu-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-r3-8f7adec9/readme.md) | failed | profiler-active | - | - ms | - | - | - GiB |
| [npu-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-r4-ccb6f22c](../../../runs/8-card/ep8/npu-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-r4-ccb6f22c/readme.md) | completed | profiler-active | distributed | 874.31 ms | 1,171.21 | 9,369.64 | - GiB |
| [npu-ep8-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-5d19af07](../../../runs/8-card/ep8/npu-ep8-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-5d19af07/readme.md) | completed | profiler-active | overview | 688.15 ms | 1,488.04 | 11,904.36 | - GiB |
| [npu-ep8-bf16-s30-l8-b64-seq128-seed61-distributed-r31-bca393ef](../../../runs/8-card/ep8/npu-ep8-bf16-s30-l8-b64-seq128-seed61-distributed-r31-bca393ef/readme.md) | completed | profiler-active | distributed | 693.03 ms | 1,477.57 | 11,820.56 | - GiB |
| [npu-ep8-bf16-s30-l8-b64-seq128-seed61-kernel-r31-offline-7a56058a](../../../runs/8-card/ep8/npu-ep8-bf16-s30-l8-b64-seq128-seed61-kernel-r31-offline-7a56058a/readme.md) | completed | profiler-active | kernel | 734.44 ms | 1,394.26 | 11,154.05 | - GiB |
| [npu-ep8-bf16-s30-l8-b64-seq128-seed61-operator-r32-offline-43246350](../../../runs/8-card/ep8/npu-ep8-bf16-s30-l8-b64-seq128-seed61-operator-r32-offline-43246350/readme.md) | completed | profiler-active | operator | 697.90 ms | 1,467.26 | 11,738.10 | - GiB |
| [npu-ep8-bf16-s30-l8-b64-seq128-seed61-memory-r31-offline-0e498f1d](../../../runs/8-card/ep8/npu-ep8-bf16-s30-l8-b64-seq128-seed61-memory-r31-offline-0e498f1d/readme.md) | completed | profiler-active | memory | 720.73 ms | 1,420.79 | 11,366.32 | - GiB |
| [npu-ep8-bf16-s30-l8-b64-seq128-seed61-flamegraph-r31-9e4d77ce](../../../runs/8-card/ep8/npu-ep8-bf16-s30-l8-b64-seq128-seed61-flamegraph-r31-9e4d77ce/readme.md) | completed | profiler-active | flamegraph | 746.17 ms | 1,372.33 | 10,978.67 | - GiB |
| [npu-ep8-bf16-s30-l8-b64-seq128-seed61-runtime-r31-offline-4ac46849](../../../runs/8-card/ep8/npu-ep8-bf16-s30-l8-b64-seq128-seed61-runtime-r31-offline-4ac46849/readme.md) | completed | profiler-active | runtime | 700.09 ms | 1,462.67 | 11,701.37 | - GiB |
| [npu-ep8-bf16-s30-l8-b64-seq128-seed61-system-r31-d330ad74](../../../runs/8-card/ep8/npu-ep8-bf16-s30-l8-b64-seq128-seed61-system-r31-d330ad74/readme.md) | completed | profiler-active | system | 691.07 ms | 1,481.77 | 11,854.14 | - GiB |

## experiment sequence

### 1. npu-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-50c6107d

- Status: `completed`
- Started: `2026-08-24T17:56:02.331131+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/ep8/npu-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-50c6107d/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/ep8/npu-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-50c6107d.html`

```bash
/root/miniconda3/envs/torchtitan-0803/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology ep8 --preset distributed --profiler-off --visible-devices 0,1,2,3,4,5,6,7 --steps 20 --skip-steps 8 --warmup-steps 2 --active-steps 3 --replicate 1
```

### 2. npu-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-r1-17dbf09f

- Status: `failed`
- Started: `2026-08-25T09:06:17.768552+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/ep8/npu-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-r1-17dbf09f/readme.md)

```bash
/root/miniconda3/envs/torchtitan-0803/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology ep8 --preset distributed --visible-devices 0,1,2,3,4,5,6,7 --steps 20 --skip-steps 8 --warmup-steps 2 --active-steps 3 --replicate 1 --cluster
```

### 3. npu-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-r2-7d87cd5f

- Status: `failed`
- Started: `2026-08-25T11:35:48.239630+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/ep8/npu-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-r2-7d87cd5f/readme.md)

```bash
/root/miniconda3/envs/torchtitan-0803/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology ep8 --preset distributed --visible-devices 0,1,2,3,4,5,6,7 --steps 20 --skip-steps 8 --warmup-steps 2 --active-steps 3 --replicate 2 --cluster
```

### 4. npu-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-r3-8f7adec9

- Status: `failed`
- Started: `2026-08-25T11:36:46.450872+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/ep8/npu-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-r3-8f7adec9/readme.md)

```bash
/root/miniconda3/envs/torchtitan-0803/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology ep8 --preset distributed --visible-devices 0,1,2,3,4,5,6,7 --steps 20 --skip-steps 8 --warmup-steps 2 --active-steps 3 --replicate 3 --cluster
```

### 5. npu-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-r4-ccb6f22c

- Status: `completed`
- Started: `2026-08-25T11:40:40.938165+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/ep8/npu-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-r4-ccb6f22c/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/ep8/npu-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-r4-ccb6f22c.html`

```bash
/root/miniconda3/envs/torchtitan-0803/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology ep8 --preset distributed --visible-devices 0,1,2,3,4,5,6,7 --steps 20 --skip-steps 8 --warmup-steps 2 --active-steps 3 --replicate 4 --cluster
```

### 6. npu-ep8-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-5d19af07

- Status: `completed`
- Started: `2026-08-31T10:57:19.138268+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/ep8/npu-ep8-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-5d19af07/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/ep8/npu-ep8-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-5d19af07.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology all --preset all --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all
```

### 7. npu-ep8-bf16-s30-l8-b64-seq128-seed61-distributed-r31-bca393ef

- Status: `completed`
- Started: `2026-08-31T14:27:20.506074+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/ep8/npu-ep8-bf16-s30-l8-b64-seq128-seed61-distributed-r31-bca393ef/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/ep8/npu-ep8-bf16-s30-l8-b64-seq128-seed61-distributed-r31-bca393ef.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology all --preset all --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all
```

### 8. npu-ep8-bf16-s30-l8-b64-seq128-seed61-kernel-r31-offline-7a56058a

- Status: `completed`
- Started: `2026-08-31T20:20:08.781641+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/ep8/npu-ep8-bf16-s30-l8-b64-seq128-seed61-kernel-r31-offline-7a56058a/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/ep8/npu-ep8-bf16-s30-l8-b64-seq128-seed61-kernel-r31-offline-7a56058a.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology all --preset kernel --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all
```

### 9. npu-ep8-bf16-s30-l8-b64-seq128-seed61-operator-r32-offline-43246350

- Status: `completed`
- Started: `2026-08-31T23:34:12.797261+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/ep8/npu-ep8-bf16-s30-l8-b64-seq128-seed61-operator-r32-offline-43246350/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/ep8/npu-ep8-bf16-s30-l8-b64-seq128-seed61-operator-r32-offline-43246350.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --capture --device npu --topology ep8 --preset operator --visible-devices 0,1,2,3,4,5,6,7 --replicate 32 --analysis-tools none
```

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --analyze --device npu --topology ep8 --preset operator --visible-devices 0,1,2,3,4,5,6,7 --replicate 32 --analysis-tools all --parse-workers 8
```

### 10. npu-ep8-bf16-s30-l8-b64-seq128-seed61-memory-r31-offline-0e498f1d

- Status: `completed`
- Started: `2026-09-01T09:17:07.811820+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/ep8/npu-ep8-bf16-s30-l8-b64-seq128-seed61-memory-r31-offline-0e498f1d/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/ep8/npu-ep8-bf16-s30-l8-b64-seq128-seed61-memory-r31-offline-0e498f1d.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --capture --device npu --topology ep8 --preset memory --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools none
```

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --analyze --device npu --topology ep8 --preset memory --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all --parse-workers 8
```

### 11. npu-ep8-bf16-s30-l8-b64-seq128-seed61-flamegraph-r31-9e4d77ce

- Status: `completed`
- Started: `2026-09-01T10:36:08.695608+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/ep8/npu-ep8-bf16-s30-l8-b64-seq128-seed61-flamegraph-r31-9e4d77ce/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/ep8/npu-ep8-bf16-s30-l8-b64-seq128-seed61-flamegraph-r31-9e4d77ce.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --capture --device npu --topology ep8 --preset flamegraph --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools none
```

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --analyze --device npu --topology ep8 --preset flamegraph --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all --parse-workers 8
```

### 12. npu-ep8-bf16-s30-l8-b64-seq128-seed61-runtime-r31-offline-4ac46849

- Status: `completed`
- Started: `2026-09-01T11:54:08.436088+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/ep8/npu-ep8-bf16-s30-l8-b64-seq128-seed61-runtime-r31-offline-4ac46849/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/ep8/npu-ep8-bf16-s30-l8-b64-seq128-seed61-runtime-r31-offline-4ac46849.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --capture --device npu --topology ep8 --preset runtime --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools none
```

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --analyze --device npu --topology ep8 --preset runtime --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all --parse-workers 8
```

### 13. npu-ep8-bf16-s30-l8-b64-seq128-seed61-system-r31-d330ad74

- Status: `completed`
- Started: `2026-09-01T13:21:22.809140+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/ep8/npu-ep8-bf16-s30-l8-b64-seq128-seed61-system-r31-d330ad74/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/ep8/npu-ep8-bf16-s30-l8-b64-seq128-seed61-system-r31-d330ad74.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --capture --device npu --topology ep8 --preset system --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools none
```

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --analyze --device npu --topology ep8 --preset system --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all --parse-workers 8
```

## current summary

The highest recorded job throughput is 30,677.33 tok/s from `npu-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-50c6107d`. Profiler-off runs are throughput evidence; profiler-active runs are attribution evidence only.

## analysis

- [Detailed topology analysis](analysis.md)
