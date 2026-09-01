# tp2-cp4 experiments

## topology

| field | value |
| --- | --- |
| world size | 8 |
| dp replicate / shard | 1 / 1 |
| tp / cp / pp / ep | 2 / 4 / 1 / 1 |
| recorded runs | 11 |

## run results

| run | status | mode | preset | median step | tok/s/device | tok/s/job | peak hbm |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| [npu-tp2-cp4-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-3b7f547c](../../../runs/8-card/tp2-cp4/npu-tp2-cp4-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-3b7f547c/readme.md) | completed | profiler-off | distributed | 4,517.32 ms | 226.68 | 1,813.47 | 0.130 GiB |
| [npu-tp2-cp4-bf16-s20-l8-b64-seq128-seed61-distributed-r1-7c32d3ee](../../../runs/8-card/tp2-cp4/npu-tp2-cp4-bf16-s20-l8-b64-seq128-seed61-distributed-r1-7c32d3ee/readme.md) | failed | profiler-active | - | - ms | - | - | - GiB |
| [npu-tp2-cp4-bf16-s20-l8-b64-seq128-seed61-distributed-r2-ddc3f9b1](../../../runs/8-card/tp2-cp4/npu-tp2-cp4-bf16-s20-l8-b64-seq128-seed61-distributed-r2-ddc3f9b1/readme.md) | completed | profiler-active | distributed | 4,773.67 ms | 214.51 | 1,716.08 | - GiB |
| [npu-tp2-cp4-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-c9bc2408](../../../runs/8-card/tp2-cp4/npu-tp2-cp4-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-c9bc2408/readme.md) | completed | profiler-active | overview | 6,338.00 ms | 161.57 | 1,292.52 | - GiB |
| [npu-tp2-cp4-bf16-s30-l8-b64-seq128-seed61-distributed-r31-d5fe582c](../../../runs/8-card/tp2-cp4/npu-tp2-cp4-bf16-s30-l8-b64-seq128-seed61-distributed-r31-d5fe582c/readme.md) | completed | profiler-active | distributed | 6,267.41 ms | 163.38 | 1,307.08 | - GiB |
| [npu-tp2-cp4-bf16-s30-l8-b64-seq128-seed61-kernel-r31-offline-ea076464](../../../runs/8-card/tp2-cp4/npu-tp2-cp4-bf16-s30-l8-b64-seq128-seed61-kernel-r31-offline-ea076464/readme.md) | completed | profiler-active | kernel | 6,466.05 ms | 158.37 | 1,266.93 | - GiB |
| [npu-tp2-cp4-bf16-s30-l8-b64-seq128-seed61-operator-r32-offline-569d06f6](../../../runs/8-card/tp2-cp4/npu-tp2-cp4-bf16-s30-l8-b64-seq128-seed61-operator-r32-offline-569d06f6/readme.md) | completed | profiler-active | operator | 6,357.75 ms | 161.06 | 1,288.51 | - GiB |
| [npu-tp2-cp4-bf16-s30-l8-b64-seq128-seed61-memory-r31-offline-9145c319](../../../runs/8-card/tp2-cp4/npu-tp2-cp4-bf16-s30-l8-b64-seq128-seed61-memory-r31-offline-9145c319/readme.md) | completed | profiler-active | memory | 6,226.25 ms | 164.47 | 1,315.72 | - GiB |
| [npu-tp2-cp4-bf16-s30-l8-b64-seq128-seed61-flamegraph-r31-44d6a7dd](../../../runs/8-card/tp2-cp4/npu-tp2-cp4-bf16-s30-l8-b64-seq128-seed61-flamegraph-r31-44d6a7dd/readme.md) | completed | profiler-active | flamegraph | 6,844.98 ms | 149.60 | 1,196.79 | - GiB |
| [npu-tp2-cp4-bf16-s30-l8-b64-seq128-seed61-runtime-r31-offline-e166ffb3](../../../runs/8-card/tp2-cp4/npu-tp2-cp4-bf16-s30-l8-b64-seq128-seed61-runtime-r31-offline-e166ffb3/readme.md) | completed | profiler-active | runtime | 6,230.85 ms | 164.34 | 1,314.75 | - GiB |
| [npu-tp2-cp4-bf16-s30-l8-b64-seq128-seed61-system-r31-92faba2a](../../../runs/8-card/tp2-cp4/npu-tp2-cp4-bf16-s30-l8-b64-seq128-seed61-system-r31-92faba2a/readme.md) | completed | profiler-active | system | 6,185.32 ms | 165.55 | 1,324.43 | - GiB |

## experiment sequence

### 1. npu-tp2-cp4-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-3b7f547c

- Status: `completed`
- Started: `2026-08-24T18:00:29.592527+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/tp2-cp4/npu-tp2-cp4-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-3b7f547c/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/tp2-cp4/npu-tp2-cp4-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-3b7f547c.html`

```bash
/root/miniconda3/envs/torchtitan-0803/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology tp2-cp4 --preset distributed --profiler-off --visible-devices 0,1,2,3,4,5,6,7 --steps 20 --skip-steps 8 --warmup-steps 2 --active-steps 3 --replicate 1
```

### 2. npu-tp2-cp4-bf16-s20-l8-b64-seq128-seed61-distributed-r1-7c32d3ee

- Status: `failed`
- Started: `2026-08-25T09:18:52.589571+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/tp2-cp4/npu-tp2-cp4-bf16-s20-l8-b64-seq128-seed61-distributed-r1-7c32d3ee/readme.md)

```bash
/root/miniconda3/envs/torchtitan-0803/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology tp2-cp4 --preset distributed --visible-devices 0,1,2,3,4,5,6,7 --steps 20 --skip-steps 8 --warmup-steps 2 --active-steps 3 --replicate 1 --cluster
```

### 3. npu-tp2-cp4-bf16-s20-l8-b64-seq128-seed61-distributed-r2-ddc3f9b1

- Status: `completed`
- Started: `2026-08-25T12:54:54.004064+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/tp2-cp4/npu-tp2-cp4-bf16-s20-l8-b64-seq128-seed61-distributed-r2-ddc3f9b1/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/tp2-cp4/npu-tp2-cp4-bf16-s20-l8-b64-seq128-seed61-distributed-r2-ddc3f9b1.html`

```bash
/root/miniconda3/envs/torchtitan-0803/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topologies pp8,fsdp2-tp4,fsdp2-cp4,tp2-cp4,fsdp4-tp2,fsdp2-pp4,fsdp2-tp2-pp2,fsdp2-tp4-ep8 --preset distributed --visible-devices 0,1,2,3,4,5,6,7 --steps 20 --skip-steps 8 --warmup-steps 2 --active-steps 3 --replicate 2 --cluster
```

### 4. npu-tp2-cp4-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-c9bc2408

- Status: `completed`
- Started: `2026-08-31T11:07:21.636361+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/tp2-cp4/npu-tp2-cp4-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-c9bc2408/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/tp2-cp4/npu-tp2-cp4-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-c9bc2408.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology all --preset all --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all
```

### 5. npu-tp2-cp4-bf16-s30-l8-b64-seq128-seed61-distributed-r31-d5fe582c

- Status: `completed`
- Started: `2026-08-31T14:54:43.252940+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/tp2-cp4/npu-tp2-cp4-bf16-s30-l8-b64-seq128-seed61-distributed-r31-d5fe582c/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/tp2-cp4/npu-tp2-cp4-bf16-s30-l8-b64-seq128-seed61-distributed-r31-d5fe582c.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology all --preset all --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all
```

### 6. npu-tp2-cp4-bf16-s30-l8-b64-seq128-seed61-kernel-r31-offline-ea076464

- Status: `completed`
- Started: `2026-08-31T20:31:54.740765+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/tp2-cp4/npu-tp2-cp4-bf16-s30-l8-b64-seq128-seed61-kernel-r31-offline-ea076464/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/tp2-cp4/npu-tp2-cp4-bf16-s30-l8-b64-seq128-seed61-kernel-r31-offline-ea076464.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology all --preset kernel --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all
```

### 7. npu-tp2-cp4-bf16-s30-l8-b64-seq128-seed61-operator-r32-offline-569d06f6

- Status: `completed`
- Started: `2026-08-31T23:40:07.456809+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/tp2-cp4/npu-tp2-cp4-bf16-s30-l8-b64-seq128-seed61-operator-r32-offline-569d06f6/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/tp2-cp4/npu-tp2-cp4-bf16-s30-l8-b64-seq128-seed61-operator-r32-offline-569d06f6.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --capture --device npu --topology tp2-cp4 --preset operator --visible-devices 0,1,2,3,4,5,6,7 --replicate 32 --analysis-tools none
```

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --analyze --device npu --topology tp2-cp4 --preset operator --visible-devices 0,1,2,3,4,5,6,7 --replicate 32 --analysis-tools all --parse-workers 8
```

### 8. npu-tp2-cp4-bf16-s30-l8-b64-seq128-seed61-memory-r31-offline-9145c319

- Status: `completed`
- Started: `2026-09-01T09:22:13.643791+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/tp2-cp4/npu-tp2-cp4-bf16-s30-l8-b64-seq128-seed61-memory-r31-offline-9145c319/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/tp2-cp4/npu-tp2-cp4-bf16-s30-l8-b64-seq128-seed61-memory-r31-offline-9145c319.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --capture --device npu --topology tp2-cp4 --preset memory --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools none
```

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --analyze --device npu --topology tp2-cp4 --preset memory --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all --parse-workers 8
```

### 9. npu-tp2-cp4-bf16-s30-l8-b64-seq128-seed61-flamegraph-r31-44d6a7dd

- Status: `completed`
- Started: `2026-09-01T10:41:38.228063+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/tp2-cp4/npu-tp2-cp4-bf16-s30-l8-b64-seq128-seed61-flamegraph-r31-44d6a7dd/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/tp2-cp4/npu-tp2-cp4-bf16-s30-l8-b64-seq128-seed61-flamegraph-r31-44d6a7dd.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --capture --device npu --topology tp2-cp4 --preset flamegraph --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools none
```

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --analyze --device npu --topology tp2-cp4 --preset flamegraph --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all --parse-workers 8
```

### 10. npu-tp2-cp4-bf16-s30-l8-b64-seq128-seed61-runtime-r31-offline-e166ffb3

- Status: `completed`
- Started: `2026-09-01T11:59:43.665461+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/tp2-cp4/npu-tp2-cp4-bf16-s30-l8-b64-seq128-seed61-runtime-r31-offline-e166ffb3/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/tp2-cp4/npu-tp2-cp4-bf16-s30-l8-b64-seq128-seed61-runtime-r31-offline-e166ffb3.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --capture --device npu --topology tp2-cp4 --preset runtime --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools none
```

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --analyze --device npu --topology tp2-cp4 --preset runtime --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all --parse-workers 8
```

### 11. npu-tp2-cp4-bf16-s30-l8-b64-seq128-seed61-system-r31-92faba2a

- Status: `completed`
- Started: `2026-09-01T13:26:46.658345+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/tp2-cp4/npu-tp2-cp4-bf16-s30-l8-b64-seq128-seed61-system-r31-92faba2a/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/tp2-cp4/npu-tp2-cp4-bf16-s30-l8-b64-seq128-seed61-system-r31-92faba2a.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --capture --device npu --topology tp2-cp4 --preset system --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools none
```

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --analyze --device npu --topology tp2-cp4 --preset system --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all --parse-workers 8
```

## current summary

The highest recorded job throughput is 1,813.47 tok/s from `npu-tp2-cp4-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-3b7f547c`. Profiler-off runs are throughput evidence; profiler-active runs are attribution evidence only.
