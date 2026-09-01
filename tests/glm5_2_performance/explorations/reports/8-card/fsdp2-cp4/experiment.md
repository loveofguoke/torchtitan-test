# fsdp2-cp4 experiments

## topology

| field | value |
| --- | --- |
| world size | 8 |
| dp replicate / shard | 1 / 2 |
| tp / cp / pp / ep | 1 / 4 / 1 / 1 |
| recorded runs | 11 |

## run results

| run | status | mode | preset | median step | tok/s/device | tok/s/job | peak hbm |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| [npu-fsdp2-cp4-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-d7029385](../../../runs/8-card/fsdp2-cp4/npu-fsdp2-cp4-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-d7029385/readme.md) | completed | profiler-off | distributed | 840.80 ms | 1,217.90 | 9,743.17 | 0.201 GiB |
| [npu-fsdp2-cp4-bf16-s20-l8-b64-seq128-seed61-distributed-r1-7bed3060](../../../runs/8-card/fsdp2-cp4/npu-fsdp2-cp4-bf16-s20-l8-b64-seq128-seed61-distributed-r1-7bed3060/readme.md) | failed | profiler-active | - | - ms | - | - | - GiB |
| [npu-fsdp2-cp4-bf16-s20-l8-b64-seq128-seed61-distributed-r2-60aa6993](../../../runs/8-card/fsdp2-cp4/npu-fsdp2-cp4-bf16-s20-l8-b64-seq128-seed61-distributed-r2-60aa6993/readme.md) | completed | profiler-active | distributed | 879.64 ms | 1,164.11 | 9,312.91 | - GiB |
| [npu-fsdp2-cp4-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-c8e2cb7e](../../../runs/8-card/fsdp2-cp4/npu-fsdp2-cp4-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-c8e2cb7e/readme.md) | completed | profiler-active | overview | 1,407.37 ms | 727.60 | 5,820.78 | - GiB |
| [npu-fsdp2-cp4-bf16-s30-l8-b64-seq128-seed61-distributed-r31-7929572e](../../../runs/8-card/fsdp2-cp4/npu-fsdp2-cp4-bf16-s30-l8-b64-seq128-seed61-distributed-r31-7929572e/readme.md) | completed | profiler-active | distributed | 1,380.72 ms | 741.64 | 5,933.13 | - GiB |
| [npu-fsdp2-cp4-bf16-s30-l8-b64-seq128-seed61-kernel-r31-offline-4440d47b](../../../runs/8-card/fsdp2-cp4/npu-fsdp2-cp4-bf16-s30-l8-b64-seq128-seed61-kernel-r31-offline-4440d47b/readme.md) | completed | profiler-active | kernel | 1,475.32 ms | 694.09 | 5,552.71 | - GiB |
| [npu-fsdp2-cp4-bf16-s30-l8-b64-seq128-seed61-operator-r32-offline-d8911674](../../../runs/8-card/fsdp2-cp4/npu-fsdp2-cp4-bf16-s30-l8-b64-seq128-seed61-operator-r32-offline-d8911674/readme.md) | completed | profiler-active | operator | 1,478.19 ms | 692.74 | 5,541.92 | - GiB |
| [npu-fsdp2-cp4-bf16-s30-l8-b64-seq128-seed61-memory-r31-offline-f26543cd](../../../runs/8-card/fsdp2-cp4/npu-fsdp2-cp4-bf16-s30-l8-b64-seq128-seed61-memory-r31-offline-f26543cd/readme.md) | completed | profiler-active | memory | 1,400.68 ms | 731.07 | 5,848.57 | - GiB |
| [npu-fsdp2-cp4-bf16-s30-l8-b64-seq128-seed61-flamegraph-r31-f17a6a9d](../../../runs/8-card/fsdp2-cp4/npu-fsdp2-cp4-bf16-s30-l8-b64-seq128-seed61-flamegraph-r31-f17a6a9d/readme.md) | completed | profiler-active | flamegraph | 1,493.05 ms | 685.84 | 5,486.75 | - GiB |
| [npu-fsdp2-cp4-bf16-s30-l8-b64-seq128-seed61-runtime-r31-offline-3212b969](../../../runs/8-card/fsdp2-cp4/npu-fsdp2-cp4-bf16-s30-l8-b64-seq128-seed61-runtime-r31-offline-3212b969/readme.md) | completed | profiler-active | runtime | 1,381.69 ms | 741.12 | 5,928.97 | - GiB |
| [npu-fsdp2-cp4-bf16-s30-l8-b64-seq128-seed61-system-r31-9b50326c](../../../runs/8-card/fsdp2-cp4/npu-fsdp2-cp4-bf16-s30-l8-b64-seq128-seed61-system-r31-9b50326c/readme.md) | completed | profiler-active | system | 1,500.68 ms | 682.36 | 5,458.86 | - GiB |

## experiment sequence

### 1. npu-fsdp2-cp4-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-d7029385

- Status: `completed`
- Started: `2026-08-24T17:59:23.655844+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/fsdp2-cp4/npu-fsdp2-cp4-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-d7029385/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/fsdp2-cp4/npu-fsdp2-cp4-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-d7029385.html`

```bash
/root/miniconda3/envs/torchtitan-0803/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology fsdp2-cp4 --preset distributed --profiler-off --visible-devices 0,1,2,3,4,5,6,7 --steps 20 --skip-steps 8 --warmup-steps 2 --active-steps 3 --replicate 1
```

### 2. npu-fsdp2-cp4-bf16-s20-l8-b64-seq128-seed61-distributed-r1-7bed3060

- Status: `failed`
- Started: `2026-08-25T09:17:37.101662+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/fsdp2-cp4/npu-fsdp2-cp4-bf16-s20-l8-b64-seq128-seed61-distributed-r1-7bed3060/readme.md)

```bash
/root/miniconda3/envs/torchtitan-0803/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology fsdp2-cp4 --preset distributed --visible-devices 0,1,2,3,4,5,6,7 --steps 20 --skip-steps 8 --warmup-steps 2 --active-steps 3 --replicate 1 --cluster
```

### 3. npu-fsdp2-cp4-bf16-s20-l8-b64-seq128-seed61-distributed-r2-60aa6993

- Status: `completed`
- Started: `2026-08-25T12:37:58.515192+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/fsdp2-cp4/npu-fsdp2-cp4-bf16-s20-l8-b64-seq128-seed61-distributed-r2-60aa6993/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/fsdp2-cp4/npu-fsdp2-cp4-bf16-s20-l8-b64-seq128-seed61-distributed-r2-60aa6993.html`

```bash
/root/miniconda3/envs/torchtitan-0803/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topologies pp8,fsdp2-tp4,fsdp2-cp4,tp2-cp4,fsdp4-tp2,fsdp2-pp4,fsdp2-tp2-pp2,fsdp2-tp4-ep8 --preset distributed --visible-devices 0,1,2,3,4,5,6,7 --steps 20 --skip-steps 8 --warmup-steps 2 --active-steps 3 --replicate 2 --cluster
```

### 4. npu-fsdp2-cp4-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-c8e2cb7e

- Status: `completed`
- Started: `2026-08-31T11:03:53.982686+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/fsdp2-cp4/npu-fsdp2-cp4-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-c8e2cb7e/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/fsdp2-cp4/npu-fsdp2-cp4-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-c8e2cb7e.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology all --preset all --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all
```

### 5. npu-fsdp2-cp4-bf16-s30-l8-b64-seq128-seed61-distributed-r31-7929572e

- Status: `completed`
- Started: `2026-08-31T14:44:05.958361+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/fsdp2-cp4/npu-fsdp2-cp4-bf16-s30-l8-b64-seq128-seed61-distributed-r31-7929572e/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/fsdp2-cp4/npu-fsdp2-cp4-bf16-s30-l8-b64-seq128-seed61-distributed-r31-7929572e.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology all --preset all --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all
```

### 6. npu-fsdp2-cp4-bf16-s30-l8-b64-seq128-seed61-kernel-r31-offline-4440d47b

- Status: `completed`
- Started: `2026-08-31T20:27:44.505512+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/fsdp2-cp4/npu-fsdp2-cp4-bf16-s30-l8-b64-seq128-seed61-kernel-r31-offline-4440d47b/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/fsdp2-cp4/npu-fsdp2-cp4-bf16-s30-l8-b64-seq128-seed61-kernel-r31-offline-4440d47b.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology all --preset kernel --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all
```

### 7. npu-fsdp2-cp4-bf16-s30-l8-b64-seq128-seed61-operator-r32-offline-d8911674

- Status: `completed`
- Started: `2026-08-31T23:38:20.668383+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/fsdp2-cp4/npu-fsdp2-cp4-bf16-s30-l8-b64-seq128-seed61-operator-r32-offline-d8911674/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/fsdp2-cp4/npu-fsdp2-cp4-bf16-s30-l8-b64-seq128-seed61-operator-r32-offline-d8911674.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --capture --device npu --topology fsdp2-cp4 --preset operator --visible-devices 0,1,2,3,4,5,6,7 --replicate 32 --analysis-tools none
```

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --analyze --device npu --topology fsdp2-cp4 --preset operator --visible-devices 0,1,2,3,4,5,6,7 --replicate 32 --analysis-tools all --parse-workers 8
```

### 8. npu-fsdp2-cp4-bf16-s30-l8-b64-seq128-seed61-memory-r31-offline-f26543cd

- Status: `completed`
- Started: `2026-09-01T09:20:39.210550+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/fsdp2-cp4/npu-fsdp2-cp4-bf16-s30-l8-b64-seq128-seed61-memory-r31-offline-f26543cd/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/fsdp2-cp4/npu-fsdp2-cp4-bf16-s30-l8-b64-seq128-seed61-memory-r31-offline-f26543cd.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --capture --device npu --topology fsdp2-cp4 --preset memory --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools none
```

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --analyze --device npu --topology fsdp2-cp4 --preset memory --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all --parse-workers 8
```

### 9. npu-fsdp2-cp4-bf16-s30-l8-b64-seq128-seed61-flamegraph-r31-f17a6a9d

- Status: `completed`
- Started: `2026-09-01T10:39:57.412450+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/fsdp2-cp4/npu-fsdp2-cp4-bf16-s30-l8-b64-seq128-seed61-flamegraph-r31-f17a6a9d/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/fsdp2-cp4/npu-fsdp2-cp4-bf16-s30-l8-b64-seq128-seed61-flamegraph-r31-f17a6a9d.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --capture --device npu --topology fsdp2-cp4 --preset flamegraph --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools none
```

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --analyze --device npu --topology fsdp2-cp4 --preset flamegraph --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all --parse-workers 8
```

### 10. npu-fsdp2-cp4-bf16-s30-l8-b64-seq128-seed61-runtime-r31-offline-3212b969

- Status: `completed`
- Started: `2026-09-01T11:58:02.121719+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/fsdp2-cp4/npu-fsdp2-cp4-bf16-s30-l8-b64-seq128-seed61-runtime-r31-offline-3212b969/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/fsdp2-cp4/npu-fsdp2-cp4-bf16-s30-l8-b64-seq128-seed61-runtime-r31-offline-3212b969.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --capture --device npu --topology fsdp2-cp4 --preset runtime --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools none
```

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --analyze --device npu --topology fsdp2-cp4 --preset runtime --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all --parse-workers 8
```

### 11. npu-fsdp2-cp4-bf16-s30-l8-b64-seq128-seed61-system-r31-9b50326c

- Status: `completed`
- Started: `2026-09-01T13:25:06.907781+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/fsdp2-cp4/npu-fsdp2-cp4-bf16-s30-l8-b64-seq128-seed61-system-r31-9b50326c/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/fsdp2-cp4/npu-fsdp2-cp4-bf16-s30-l8-b64-seq128-seed61-system-r31-9b50326c.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --capture --device npu --topology fsdp2-cp4 --preset system --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools none
```

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --analyze --device npu --topology fsdp2-cp4 --preset system --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all --parse-workers 8
```

## current summary

The highest recorded job throughput is 9,743.17 tok/s from `npu-fsdp2-cp4-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-d7029385`. Profiler-off runs are throughput evidence; profiler-active runs are attribution evidence only.

## analysis

- [Detailed topology analysis](analysis.md)
