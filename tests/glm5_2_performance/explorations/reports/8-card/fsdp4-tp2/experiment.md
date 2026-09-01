# fsdp4-tp2 experiments

## topology

| field | value |
| --- | --- |
| world size | 8 |
| dp replicate / shard | 1 / 4 |
| tp / cp / pp / ep | 2 / 1 / 1 / 1 |
| recorded runs | 11 |

## run results

| run | status | mode | preset | median step | tok/s/device | tok/s/job | peak hbm |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| [npu-fsdp4-tp2-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-e3971d37](../../../runs/8-card/fsdp4-tp2/npu-fsdp4-tp2-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-e3971d37/readme.md) | completed | profiler-off | distributed | 1,129.79 ms | 906.36 | 7,250.90 | 0.372 GiB |
| [npu-fsdp4-tp2-bf16-s20-l8-b64-seq128-seed61-distributed-r1-49448ffb](../../../runs/8-card/fsdp4-tp2/npu-fsdp4-tp2-bf16-s20-l8-b64-seq128-seed61-distributed-r1-49448ffb/readme.md) | failed | profiler-active | - | - ms | - | - | - GiB |
| [npu-fsdp4-tp2-bf16-s20-l8-b64-seq128-seed61-distributed-r2-a955eda9](../../../runs/8-card/fsdp4-tp2/npu-fsdp4-tp2-bf16-s20-l8-b64-seq128-seed61-distributed-r2-a955eda9/readme.md) | completed | profiler-active | distributed | 1,251.04 ms | 818.52 | 6,548.15 | - GiB |
| [npu-fsdp4-tp2-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-54e05c30](../../../runs/8-card/fsdp4-tp2/npu-fsdp4-tp2-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-54e05c30/readme.md) | completed | profiler-active | overview | 1,604.13 ms | 638.35 | 5,106.83 | - GiB |
| [npu-fsdp4-tp2-bf16-s30-l8-b64-seq128-seed61-distributed-r31-be85efc7](../../../runs/8-card/fsdp4-tp2/npu-fsdp4-tp2-bf16-s30-l8-b64-seq128-seed61-distributed-r31-be85efc7/readme.md) | completed | profiler-active | distributed | 1,623.04 ms | 630.91 | 5,047.31 | - GiB |
| [npu-fsdp4-tp2-bf16-s30-l8-b64-seq128-seed61-kernel-r31-offline-685e4369](../../../runs/8-card/fsdp4-tp2/npu-fsdp4-tp2-bf16-s30-l8-b64-seq128-seed61-kernel-r31-offline-685e4369/readme.md) | completed | profiler-active | kernel | 1,627.88 ms | 629.04 | 5,032.33 | - GiB |
| [npu-fsdp4-tp2-bf16-s30-l8-b64-seq128-seed61-operator-r32-offline-f1f35402](../../../runs/8-card/fsdp4-tp2/npu-fsdp4-tp2-bf16-s30-l8-b64-seq128-seed61-operator-r32-offline-f1f35402/readme.md) | completed | profiler-active | operator | 1,600.86 ms | 639.66 | 5,117.26 | - GiB |
| [npu-fsdp4-tp2-bf16-s30-l8-b64-seq128-seed61-memory-r31-offline-f45a5732](../../../runs/8-card/fsdp4-tp2/npu-fsdp4-tp2-bf16-s30-l8-b64-seq128-seed61-memory-r31-offline-f45a5732/readme.md) | completed | profiler-active | memory | 1,588.92 ms | 644.46 | 5,155.69 | - GiB |
| [npu-fsdp4-tp2-bf16-s30-l8-b64-seq128-seed61-flamegraph-r31-4d3428eb](../../../runs/8-card/fsdp4-tp2/npu-fsdp4-tp2-bf16-s30-l8-b64-seq128-seed61-flamegraph-r31-4d3428eb/readme.md) | completed | profiler-active | flamegraph | 1,677.82 ms | 610.32 | 4,882.52 | - GiB |
| [npu-fsdp4-tp2-bf16-s30-l8-b64-seq128-seed61-runtime-r31-offline-3cc0301b](../../../runs/8-card/fsdp4-tp2/npu-fsdp4-tp2-bf16-s30-l8-b64-seq128-seed61-runtime-r31-offline-3cc0301b/readme.md) | completed | profiler-active | runtime | 1,561.30 ms | 655.86 | 5,246.92 | - GiB |
| [npu-fsdp4-tp2-bf16-s30-l8-b64-seq128-seed61-system-r31-fe3b7142](../../../runs/8-card/fsdp4-tp2/npu-fsdp4-tp2-bf16-s30-l8-b64-seq128-seed61-system-r31-fe3b7142/readme.md) | completed | profiler-active | system | 1,565.13 ms | 654.26 | 5,234.08 | - GiB |

## experiment sequence

### 1. npu-fsdp4-tp2-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-e3971d37

- Status: `completed`
- Started: `2026-08-24T18:02:44.721148+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/fsdp4-tp2/npu-fsdp4-tp2-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-e3971d37/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/fsdp4-tp2/npu-fsdp4-tp2-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-e3971d37.html`

```bash
/root/miniconda3/envs/torchtitan-0803/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology fsdp4-tp2 --preset distributed --profiler-off --visible-devices 0,1,2,3,4,5,6,7 --steps 20 --skip-steps 8 --warmup-steps 2 --active-steps 3 --replicate 1
```

### 2. npu-fsdp4-tp2-bf16-s20-l8-b64-seq128-seed61-distributed-r1-49448ffb

- Status: `failed`
- Started: `2026-08-25T09:20:41.412735+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/fsdp4-tp2/npu-fsdp4-tp2-bf16-s20-l8-b64-seq128-seed61-distributed-r1-49448ffb/readme.md)

```bash
/root/miniconda3/envs/torchtitan-0803/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology fsdp4-tp2 --preset distributed --visible-devices 0,1,2,3,4,5,6,7 --steps 20 --skip-steps 8 --warmup-steps 2 --active-steps 3 --replicate 1 --cluster
```

### 3. npu-fsdp4-tp2-bf16-s20-l8-b64-seq128-seed61-distributed-r2-a955eda9

- Status: `completed`
- Started: `2026-08-25T13:21:30.519573+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/fsdp4-tp2/npu-fsdp4-tp2-bf16-s20-l8-b64-seq128-seed61-distributed-r2-a955eda9/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/fsdp4-tp2/npu-fsdp4-tp2-bf16-s20-l8-b64-seq128-seed61-distributed-r2-a955eda9.html`

```bash
/root/miniconda3/envs/torchtitan-0803/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topologies pp8,fsdp2-tp4,fsdp2-cp4,tp2-cp4,fsdp4-tp2,fsdp2-pp4,fsdp2-tp2-pp2,fsdp2-tp4-ep8 --preset distributed --visible-devices 0,1,2,3,4,5,6,7 --steps 20 --skip-steps 8 --warmup-steps 2 --active-steps 3 --replicate 2 --cluster
```

### 4. npu-fsdp4-tp2-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-54e05c30

- Status: `completed`
- Started: `2026-08-31T11:16:34.450854+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/fsdp4-tp2/npu-fsdp4-tp2-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-54e05c30/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/fsdp4-tp2/npu-fsdp4-tp2-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-54e05c30.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology all --preset all --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all
```

### 5. npu-fsdp4-tp2-bf16-s30-l8-b64-seq128-seed61-distributed-r31-be85efc7

- Status: `completed`
- Started: `2026-08-31T15:20:25.313826+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/fsdp4-tp2/npu-fsdp4-tp2-bf16-s30-l8-b64-seq128-seed61-distributed-r31-be85efc7/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/fsdp4-tp2/npu-fsdp4-tp2-bf16-s30-l8-b64-seq128-seed61-distributed-r31-be85efc7.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology all --preset all --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all
```

### 6. npu-fsdp4-tp2-bf16-s30-l8-b64-seq128-seed61-kernel-r31-offline-685e4369

- Status: `completed`
- Started: `2026-08-31T20:42:27.348133+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/fsdp4-tp2/npu-fsdp4-tp2-bf16-s30-l8-b64-seq128-seed61-kernel-r31-offline-685e4369/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/fsdp4-tp2/npu-fsdp4-tp2-bf16-s30-l8-b64-seq128-seed61-kernel-r31-offline-685e4369.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology all --preset kernel --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all
```

### 7. npu-fsdp4-tp2-bf16-s30-l8-b64-seq128-seed61-operator-r32-offline-f1f35402

- Status: `completed`
- Started: `2026-08-31T23:44:41.897811+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/fsdp4-tp2/npu-fsdp4-tp2-bf16-s30-l8-b64-seq128-seed61-operator-r32-offline-f1f35402/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/fsdp4-tp2/npu-fsdp4-tp2-bf16-s30-l8-b64-seq128-seed61-operator-r32-offline-f1f35402.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --capture --device npu --topology fsdp4-tp2 --preset operator --visible-devices 0,1,2,3,4,5,6,7 --replicate 32 --analysis-tools none
```

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --analyze --device npu --topology fsdp4-tp2 --preset operator --visible-devices 0,1,2,3,4,5,6,7 --replicate 32 --analysis-tools all --parse-workers 8
```

### 8. npu-fsdp4-tp2-bf16-s30-l8-b64-seq128-seed61-memory-r31-offline-f45a5732

- Status: `completed`
- Started: `2026-09-01T09:26:23.707233+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/fsdp4-tp2/npu-fsdp4-tp2-bf16-s30-l8-b64-seq128-seed61-memory-r31-offline-f45a5732/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/fsdp4-tp2/npu-fsdp4-tp2-bf16-s30-l8-b64-seq128-seed61-memory-r31-offline-f45a5732.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --capture --device npu --topology fsdp4-tp2 --preset memory --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools none
```

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --analyze --device npu --topology fsdp4-tp2 --preset memory --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all --parse-workers 8
```

### 9. npu-fsdp4-tp2-bf16-s30-l8-b64-seq128-seed61-flamegraph-r31-4d3428eb

- Status: `completed`
- Started: `2026-09-01T10:46:03.485688+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/fsdp4-tp2/npu-fsdp4-tp2-bf16-s30-l8-b64-seq128-seed61-flamegraph-r31-4d3428eb/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/fsdp4-tp2/npu-fsdp4-tp2-bf16-s30-l8-b64-seq128-seed61-flamegraph-r31-4d3428eb.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --capture --device npu --topology fsdp4-tp2 --preset flamegraph --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools none
```

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --analyze --device npu --topology fsdp4-tp2 --preset flamegraph --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all --parse-workers 8
```

### 10. npu-fsdp4-tp2-bf16-s30-l8-b64-seq128-seed61-runtime-r31-offline-3cc0301b

- Status: `completed`
- Started: `2026-09-01T12:04:22.224935+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/fsdp4-tp2/npu-fsdp4-tp2-bf16-s30-l8-b64-seq128-seed61-runtime-r31-offline-3cc0301b/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/fsdp4-tp2/npu-fsdp4-tp2-bf16-s30-l8-b64-seq128-seed61-runtime-r31-offline-3cc0301b.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --capture --device npu --topology fsdp4-tp2 --preset runtime --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools none
```

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --analyze --device npu --topology fsdp4-tp2 --preset runtime --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all --parse-workers 8
```

### 11. npu-fsdp4-tp2-bf16-s30-l8-b64-seq128-seed61-system-r31-fe3b7142

- Status: `completed`
- Started: `2026-09-01T13:30:59.307169+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/fsdp4-tp2/npu-fsdp4-tp2-bf16-s30-l8-b64-seq128-seed61-system-r31-fe3b7142/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/fsdp4-tp2/npu-fsdp4-tp2-bf16-s30-l8-b64-seq128-seed61-system-r31-fe3b7142.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --capture --device npu --topology fsdp4-tp2 --preset system --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools none
```

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --analyze --device npu --topology fsdp4-tp2 --preset system --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all --parse-workers 8
```

## current summary

The highest recorded job throughput is 7,250.90 tok/s from `npu-fsdp4-tp2-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-e3971d37`. Profiler-off runs are throughput evidence; profiler-active runs are attribution evidence only.
