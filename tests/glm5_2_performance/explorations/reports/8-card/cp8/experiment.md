# cp8 experiments

## topology

| field | value |
| --- | --- |
| world size | 8 |
| dp replicate / shard | 1 / 1 |
| tp / cp / pp / ep | 1 / 8 / 1 / 1 |
| recorded runs | 11 |

## run results

| run | status | mode | preset | median step | tok/s/device | tok/s/job | peak hbm |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| [npu-cp8-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-486604c0](../../../runs/8-card/cp8/npu-cp8-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-486604c0/readme.md) | completed | profiler-off | distributed | 1,626.28 ms | 629.66 | 5,037.26 | 0.130 GiB |
| [npu-cp8-bf16-s20-l8-b64-seq128-seed61-distributed-r1-d80e36ca](../../../runs/8-card/cp8/npu-cp8-bf16-s20-l8-b64-seq128-seed61-distributed-r1-d80e36ca/readme.md) | failed | profiler-active | - | - ms | - | - | - GiB |
| [npu-cp8-bf16-s20-l8-b64-seq128-seed61-distributed-r2-03d875ae](../../../runs/8-card/cp8/npu-cp8-bf16-s20-l8-b64-seq128-seed61-distributed-r2-03d875ae/readme.md) | completed | profiler-active | distributed | 1,723.89 ms | 594.00 | 4,752.03 | - GiB |
| [npu-cp8-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-d595df87](../../../runs/8-card/cp8/npu-cp8-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-d595df87/readme.md) | completed | profiler-active | overview | 2,782.33 ms | 368.04 | 2,944.30 | - GiB |
| [npu-cp8-bf16-s30-l8-b64-seq128-seed61-distributed-r31-ba7e0fe4](../../../runs/8-card/cp8/npu-cp8-bf16-s30-l8-b64-seq128-seed61-distributed-r31-ba7e0fe4/readme.md) | completed | profiler-active | distributed | 2,774.08 ms | 369.13 | 2,953.06 | - GiB |
| [npu-cp8-bf16-s30-l8-b64-seq128-seed61-kernel-r31-offline-a5d38160](../../../runs/8-card/cp8/npu-cp8-bf16-s30-l8-b64-seq128-seed61-kernel-r31-offline-a5d38160/readme.md) | completed | profiler-active | kernel | 2,897.27 ms | 353.44 | 2,827.49 | - GiB |
| [npu-cp8-bf16-s30-l8-b64-seq128-seed61-operator-r32-offline-19885bc8](../../../runs/8-card/cp8/npu-cp8-bf16-s30-l8-b64-seq128-seed61-operator-r32-offline-19885bc8/readme.md) | completed | profiler-active | operator | 2,806.15 ms | 364.91 | 2,919.31 | - GiB |
| [npu-cp8-bf16-s30-l8-b64-seq128-seed61-memory-r31-offline-e006f421](../../../runs/8-card/cp8/npu-cp8-bf16-s30-l8-b64-seq128-seed61-memory-r31-offline-e006f421/readme.md) | completed | profiler-active | memory | 2,768.21 ms | 369.91 | 2,959.32 | - GiB |
| [npu-cp8-bf16-s30-l8-b64-seq128-seed61-flamegraph-r31-be0297a7](../../../runs/8-card/cp8/npu-cp8-bf16-s30-l8-b64-seq128-seed61-flamegraph-r31-be0297a7/readme.md) | completed | profiler-active | flamegraph | 3,118.41 ms | 328.37 | 2,626.98 | - GiB |
| [npu-cp8-bf16-s30-l8-b64-seq128-seed61-runtime-r31-offline-7be269a4](../../../runs/8-card/cp8/npu-cp8-bf16-s30-l8-b64-seq128-seed61-runtime-r31-offline-7be269a4/readme.md) | completed | profiler-active | runtime | 2,760.39 ms | 370.96 | 2,967.70 | - GiB |
| [npu-cp8-bf16-s30-l8-b64-seq128-seed61-system-r31-9a5745ba](../../../runs/8-card/cp8/npu-cp8-bf16-s30-l8-b64-seq128-seed61-system-r31-9a5745ba/readme.md) | completed | profiler-active | system | 2,778.68 ms | 368.52 | 2,948.16 | - GiB |

## experiment sequence

### 1. npu-cp8-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-486604c0

- Status: `completed`
- Started: `2026-08-24T17:52:46.682516+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/cp8/npu-cp8-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-486604c0/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/cp8/npu-cp8-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-486604c0.html`

```bash
/root/miniconda3/envs/torchtitan-0803/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology cp8 --preset distributed --profiler-off --visible-devices 0,1,2,3,4,5,6,7 --steps 20 --skip-steps 8 --warmup-steps 2 --active-steps 3 --replicate 1
```

### 2. npu-cp8-bf16-s20-l8-b64-seq128-seed61-distributed-r1-d80e36ca

- Status: `failed`
- Started: `2026-08-25T09:07:49.872209+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/cp8/npu-cp8-bf16-s20-l8-b64-seq128-seed61-distributed-r1-d80e36ca/readme.md)

```bash
/root/miniconda3/envs/torchtitan-0803/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology cp8 --preset distributed --visible-devices 0,1,2,3,4,5,6,7 --steps 20 --skip-steps 8 --warmup-steps 2 --active-steps 3 --replicate 1 --cluster
```

### 3. npu-cp8-bf16-s20-l8-b64-seq128-seed61-distributed-r2-03d875ae

- Status: `completed`
- Started: `2026-08-25T11:49:31.538912+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/cp8/npu-cp8-bf16-s20-l8-b64-seq128-seed61-distributed-r2-03d875ae/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/cp8/npu-cp8-bf16-s20-l8-b64-seq128-seed61-distributed-r2-03d875ae.html`

```bash
/root/miniconda3/envs/torchtitan-0803/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology cp8 --preset distributed --visible-devices 0,1,2,3,4,5,6,7 --steps 20 --skip-steps 8 --warmup-steps 2 --active-steps 3 --replicate 2 --cluster
```

### 4. npu-cp8-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-d595df87

- Status: `completed`
- Started: `2026-08-31T10:46:30.476841+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/cp8/npu-cp8-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-d595df87/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/cp8/npu-cp8-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-d595df87.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology all --preset all --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all
```

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology all --preset all --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all
```

### 5. npu-cp8-bf16-s30-l8-b64-seq128-seed61-distributed-r31-ba7e0fe4

- Status: `completed`
- Started: `2026-08-31T13:50:13.747507+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/cp8/npu-cp8-bf16-s30-l8-b64-seq128-seed61-distributed-r31-ba7e0fe4/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/cp8/npu-cp8-bf16-s30-l8-b64-seq128-seed61-distributed-r31-ba7e0fe4.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology all --preset all --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all
```

### 6. npu-cp8-bf16-s30-l8-b64-seq128-seed61-kernel-r31-offline-a5d38160

- Status: `completed`
- Started: `2026-08-31T20:07:14.752030+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/cp8/npu-cp8-bf16-s30-l8-b64-seq128-seed61-kernel-r31-offline-a5d38160/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/cp8/npu-cp8-bf16-s30-l8-b64-seq128-seed61-kernel-r31-offline-a5d38160.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology all --preset kernel --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all
```

### 7. npu-cp8-bf16-s30-l8-b64-seq128-seed61-operator-r32-offline-19885bc8

- Status: `completed`
- Started: `2026-08-31T23:21:31.979041+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/cp8/npu-cp8-bf16-s30-l8-b64-seq128-seed61-operator-r32-offline-19885bc8/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/cp8/npu-cp8-bf16-s30-l8-b64-seq128-seed61-operator-r32-offline-19885bc8.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology cp8 --preset operator --visible-devices 0,1,2,3,4,5,6,7 --replicate 32 --analysis-tools all
```

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --analyze --device npu --topology cp8 --preset operator --visible-devices 0,1,2,3,4,5,6,7 --replicate 32 --analysis-tools all --parse-workers 8
```

### 8. npu-cp8-bf16-s30-l8-b64-seq128-seed61-memory-r31-offline-e006f421

- Status: `completed`
- Started: `2026-09-01T09:12:05.720365+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/cp8/npu-cp8-bf16-s30-l8-b64-seq128-seed61-memory-r31-offline-e006f421/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/cp8/npu-cp8-bf16-s30-l8-b64-seq128-seed61-memory-r31-offline-e006f421.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --capture --device npu --topology cp8 --preset memory --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools none
```

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --analyze --device npu --topology cp8 --preset memory --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all --parse-workers 8
```

### 9. npu-cp8-bf16-s30-l8-b64-seq128-seed61-flamegraph-r31-be0297a7

- Status: `completed`
- Started: `2026-09-01T10:30:38.445107+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/cp8/npu-cp8-bf16-s30-l8-b64-seq128-seed61-flamegraph-r31-be0297a7/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/cp8/npu-cp8-bf16-s30-l8-b64-seq128-seed61-flamegraph-r31-be0297a7.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --capture --device npu --topology cp8 --preset flamegraph --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools none
```

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --analyze --device npu --topology cp8 --preset flamegraph --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all --parse-workers 8
```

### 10. npu-cp8-bf16-s30-l8-b64-seq128-seed61-runtime-r31-offline-7be269a4

- Status: `completed`
- Started: `2026-09-01T11:48:26.071722+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/cp8/npu-cp8-bf16-s30-l8-b64-seq128-seed61-runtime-r31-offline-7be269a4/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/cp8/npu-cp8-bf16-s30-l8-b64-seq128-seed61-runtime-r31-offline-7be269a4.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --capture --device npu --topology cp8 --preset runtime --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools none
```

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --analyze --device npu --topology cp8 --preset runtime --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all --parse-workers 8
```

### 11. npu-cp8-bf16-s30-l8-b64-seq128-seed61-system-r31-9a5745ba

- Status: `completed`
- Started: `2026-09-01T13:16:05.884994+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/cp8/npu-cp8-bf16-s30-l8-b64-seq128-seed61-system-r31-9a5745ba/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/cp8/npu-cp8-bf16-s30-l8-b64-seq128-seed61-system-r31-9a5745ba.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --capture --device npu --topology cp8 --preset system --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools none
```

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --analyze --device npu --topology cp8 --preset system --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all --parse-workers 8
```

## current summary

The highest recorded job throughput is 5,037.26 tok/s from `npu-cp8-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-486604c0`. Profiler-off runs are throughput evidence; profiler-active runs are attribution evidence only.

## analysis

- [Detailed topology analysis](analysis.md)
