# fsdp8 experiments

## topology

| field | value |
| --- | --- |
| world size | 8 |
| dp replicate / shard | 1 / 8 |
| tp / cp / pp / ep | 1 / 1 / 1 / 1 |
| recorded runs | 12 |

## run results

| run | status | mode | preset | median step | tok/s/device | tok/s/job | peak hbm |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| [npu-fsdp8-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-4a88f56a](../../../runs/8-card/fsdp8/npu-fsdp8-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-4a88f56a/readme.md) | completed | profiler-off | distributed | 219.00 ms | 4,675.83 | 37,406.66 | 0.621 GiB |
| [npu-fsdp8-bf16-s20-l8-b64-seq128-seed61-distributed-r1-8501ffd6](../../../runs/8-card/fsdp8/npu-fsdp8-bf16-s20-l8-b64-seq128-seed61-distributed-r1-8501ffd6/readme.md) | completed | profiler-active | distributed | 232.32 ms | 4,407.63 | 35,261.06 | - GiB |
| [npu-fsdp8-bf16-s30-l8-b64-seq128-seed61-overview-r31-b51e969a](../../../runs/8-card/fsdp8/npu-fsdp8-bf16-s30-l8-b64-seq128-seed61-overview-r31-b51e969a/readme.md) | completed | profiler-active | overview | 389.23 ms | 2,630.85 | 21,046.83 | - GiB |
| [npu-fsdp8-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-8a3891e5](../../../runs/8-card/fsdp8/npu-fsdp8-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-8a3891e5/readme.md) | completed | profiler-active | overview | 385.95 ms | 2,653.16 | 21,225.30 | - GiB |
| [npu-fsdp8-bf16-s30-l8-b64-seq128-seed61-distributed-r31-8c98fbbc](../../../runs/8-card/fsdp8/npu-fsdp8-bf16-s30-l8-b64-seq128-seed61-distributed-r31-8c98fbbc/readme.md) | completed | profiler-active | distributed | 386.94 ms | 2,646.42 | 21,171.37 | - GiB |
| [npu-fsdp8-bf16-s30-l8-b64-seq128-seed61-kernel-r31-offline-aa275892](../../../runs/8-card/fsdp8/npu-fsdp8-bf16-s30-l8-b64-seq128-seed61-kernel-r31-offline-aa275892/readme.md) | completed | profiler-active | kernel | 387.05 ms | 2,645.68 | 21,165.44 | - GiB |
| [npu-fsdp8-bf16-s30-l8-b64-seq128-seed61-operator-r32-offline-092ecc68](../../../runs/8-card/fsdp8/npu-fsdp8-bf16-s30-l8-b64-seq128-seed61-operator-r32-offline-092ecc68/readme.md) | completed | profiler-active | operator | 408.28 ms | 2,508.10 | 20,064.81 | - GiB |
| [npu-fsdp8-bf16-s30-l8-b64-seq128-seed61-memory-r31-offline-a16b98cf](../../../runs/8-card/fsdp8/npu-fsdp8-bf16-s30-l8-b64-seq128-seed61-memory-r31-offline-a16b98cf/readme.md) | failed | profiler-active | - | - ms | - | - | - GiB |
| [npu-fsdp8-bf16-s30-l8-b64-seq128-seed61-memory-r31-offline-7948d5a2](../../../runs/8-card/fsdp8/npu-fsdp8-bf16-s30-l8-b64-seq128-seed61-memory-r31-offline-7948d5a2/readme.md) | completed | profiler-active | memory | 391.43 ms | 2,616.02 | 20,928.19 | - GiB |
| [npu-fsdp8-bf16-s30-l8-b64-seq128-seed61-flamegraph-r31-1cf6eefa](../../../runs/8-card/fsdp8/npu-fsdp8-bf16-s30-l8-b64-seq128-seed61-flamegraph-r31-1cf6eefa/readme.md) | completed | profiler-active | flamegraph | 393.67 ms | 2,601.18 | 20,809.45 | - GiB |
| [npu-fsdp8-bf16-s30-l8-b64-seq128-seed61-runtime-r31-offline-534b554a](../../../runs/8-card/fsdp8/npu-fsdp8-bf16-s30-l8-b64-seq128-seed61-runtime-r31-offline-534b554a/readme.md) | completed | profiler-active | runtime | 386.86 ms | 2,646.98 | 21,175.84 | - GiB |
| [npu-fsdp8-bf16-s30-l8-b64-seq128-seed61-system-r31-951c3ec4](../../../runs/8-card/fsdp8/npu-fsdp8-bf16-s30-l8-b64-seq128-seed61-system-r31-951c3ec4/readme.md) | completed | profiler-active | system | 389.08 ms | 2,631.83 | 21,054.61 | - GiB |

## experiment sequence

### 1. npu-fsdp8-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-4a88f56a

- Status: `completed`
- Started: `2026-08-24T17:49:07.801535+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/fsdp8/npu-fsdp8-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-4a88f56a/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/fsdp8/npu-fsdp8-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-4a88f56a.html`

```bash
/root/miniconda3/envs/torchtitan-0803/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology fsdp8 --preset distributed --profiler-off --visible-devices 0,1,2,3,4,5,6,7 --steps 20 --skip-steps 8 --warmup-steps 2 --active-steps 3 --replicate 1
```

### 2. npu-fsdp8-bf16-s20-l8-b64-seq128-seed61-distributed-r1-8501ffd6

- Status: `completed`
- Started: `2026-08-24T18:26:01.593338+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/fsdp8/npu-fsdp8-bf16-s20-l8-b64-seq128-seed61-distributed-r1-8501ffd6/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/fsdp8/npu-fsdp8-bf16-s20-l8-b64-seq128-seed61-distributed-r1-8501ffd6.html`

```bash
/root/miniconda3/envs/torchtitan-0803/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology fsdp8 --preset distributed --visible-devices 0,1,2,3,4,5,6,7 --steps 20 --skip-steps 8 --warmup-steps 2 --active-steps 3 --replicate 1 --cluster
```

### 3. npu-fsdp8-bf16-s30-l8-b64-seq128-seed61-overview-r31-b51e969a

- Status: `completed`
- Started: `2026-08-31T10:09:30.486260+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/fsdp8/npu-fsdp8-bf16-s30-l8-b64-seq128-seed61-overview-r31-b51e969a/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/fsdp8/npu-fsdp8-bf16-s30-l8-b64-seq128-seed61-overview-r31-b51e969a.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology all --preset all --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all
```

### 4. npu-fsdp8-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-8a3891e5

- Status: `completed`
- Started: `2026-08-31T10:39:48.051672+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/fsdp8/npu-fsdp8-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-8a3891e5/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/fsdp8/npu-fsdp8-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-8a3891e5.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology all --preset all --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all
```

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology all --preset all --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all
```

### 5. npu-fsdp8-bf16-s30-l8-b64-seq128-seed61-distributed-r31-8c98fbbc

- Status: `completed`
- Started: `2026-08-31T13:16:11.522720+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/fsdp8/npu-fsdp8-bf16-s30-l8-b64-seq128-seed61-distributed-r31-8c98fbbc/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/fsdp8/npu-fsdp8-bf16-s30-l8-b64-seq128-seed61-distributed-r31-8c98fbbc.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology all --preset all --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all
```

### 6. npu-fsdp8-bf16-s30-l8-b64-seq128-seed61-kernel-r31-offline-aa275892

- Status: `completed`
- Started: `2026-08-31T19:54:50.420889+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/fsdp8/npu-fsdp8-bf16-s30-l8-b64-seq128-seed61-kernel-r31-offline-aa275892/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/fsdp8/npu-fsdp8-bf16-s30-l8-b64-seq128-seed61-kernel-r31-offline-aa275892.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology all --preset kernel --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all
```

### 7. npu-fsdp8-bf16-s30-l8-b64-seq128-seed61-operator-r32-offline-092ecc68

- Status: `completed`
- Started: `2026-08-31T23:07:32.385667+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/fsdp8/npu-fsdp8-bf16-s30-l8-b64-seq128-seed61-operator-r32-offline-092ecc68/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/fsdp8/npu-fsdp8-bf16-s30-l8-b64-seq128-seed61-operator-r32-offline-092ecc68.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology fsdp8 --preset operator --visible-devices 0,1,2,3,4,5,6,7 --replicate 32 --analysis-tools all
```

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --analyze --device npu --topology fsdp8 --preset operator --visible-devices 0,1,2,3,4,5,6,7 --replicate 32 --analysis-tools all --parse-workers 8
```

### 8. npu-fsdp8-bf16-s30-l8-b64-seq128-seed61-memory-r31-offline-a16b98cf

- Status: `failed`
- Started: `2026-09-01T00:41:26.614463+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/fsdp8/npu-fsdp8-bf16-s30-l8-b64-seq128-seed61-memory-r31-offline-a16b98cf/readme.md)

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --capture --device npu --topology fsdp8 --preset memory --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools none
```

### 9. npu-fsdp8-bf16-s30-l8-b64-seq128-seed61-memory-r31-offline-7948d5a2

- Status: `completed`
- Started: `2026-09-01T09:07:05.222355+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/fsdp8/npu-fsdp8-bf16-s30-l8-b64-seq128-seed61-memory-r31-offline-7948d5a2/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/fsdp8/npu-fsdp8-bf16-s30-l8-b64-seq128-seed61-memory-r31-offline-7948d5a2.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --capture --device npu --topology fsdp8 --preset memory --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools none
```

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --analyze --device npu --topology fsdp8 --preset memory --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all --parse-workers 8
```

### 10. npu-fsdp8-bf16-s30-l8-b64-seq128-seed61-flamegraph-r31-1cf6eefa

- Status: `completed`
- Started: `2026-09-01T10:25:14.498587+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/fsdp8/npu-fsdp8-bf16-s30-l8-b64-seq128-seed61-flamegraph-r31-1cf6eefa/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/fsdp8/npu-fsdp8-bf16-s30-l8-b64-seq128-seed61-flamegraph-r31-1cf6eefa.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --capture --device npu --topology fsdp8 --preset flamegraph --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools none
```

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --analyze --device npu --topology fsdp8 --preset flamegraph --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all --parse-workers 8
```

### 11. npu-fsdp8-bf16-s30-l8-b64-seq128-seed61-runtime-r31-offline-534b554a

- Status: `completed`
- Started: `2026-09-01T11:42:37.746901+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/fsdp8/npu-fsdp8-bf16-s30-l8-b64-seq128-seed61-runtime-r31-offline-534b554a/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/fsdp8/npu-fsdp8-bf16-s30-l8-b64-seq128-seed61-runtime-r31-offline-534b554a.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --capture --device npu --topology fsdp8 --preset runtime --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools none
```

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --analyze --device npu --topology fsdp8 --preset runtime --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all --parse-workers 8
```

### 12. npu-fsdp8-bf16-s30-l8-b64-seq128-seed61-system-r31-951c3ec4

- Status: `completed`
- Started: `2026-09-01T13:10:57.641209+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/fsdp8/npu-fsdp8-bf16-s30-l8-b64-seq128-seed61-system-r31-951c3ec4/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/fsdp8/npu-fsdp8-bf16-s30-l8-b64-seq128-seed61-system-r31-951c3ec4.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --capture --device npu --topology fsdp8 --preset system --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools none
```

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --analyze --device npu --topology fsdp8 --preset system --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all --parse-workers 8
```

## current summary

The highest recorded job throughput is 37,406.66 tok/s from `npu-fsdp8-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-4a88f56a`. Profiler-off runs are throughput evidence; profiler-active runs are attribution evidence only.

## analysis

- [Detailed topology analysis](analysis.md)
