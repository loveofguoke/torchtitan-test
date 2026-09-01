# ep4 experiments

## topology

| field | value |
| --- | --- |
| world size | 4 |
| dp replicate / shard | 1 / 4 |
| tp / cp / pp / ep | 1 / 1 / 1 / 4 |
| recorded runs | 9 |

## run results

| run | status | mode | preset | median step | tok/s/device | tok/s/job | peak hbm |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| [npu-ep4-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-4521dc9c](../../../runs/4-card/ep4/npu-ep4-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-4521dc9c/readme.md) | completed | profiler-off | distributed | 477.19 ms | 4,291.94 | 17,167.77 | 0.653 GiB |
| [npu-ep4-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-d8ef2c7f](../../../runs/4-card/ep4/npu-ep4-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-d8ef2c7f/readme.md) | completed | profiler-active | overview | 1,066.81 ms | 1,919.74 | 7,678.95 | - GiB |
| [npu-ep4-bf16-s30-l8-b64-seq128-seed61-distributed-r31-c23f867c](../../../runs/4-card/ep4/npu-ep4-bf16-s30-l8-b64-seq128-seed61-distributed-r31-c23f867c/readme.md) | completed | profiler-active | distributed | 1,020.74 ms | 2,006.39 | 8,025.56 | - GiB |
| [npu-ep4-bf16-s30-l8-b64-seq128-seed61-kernel-r31-offline-51e814a3](../../../runs/4-card/ep4/npu-ep4-bf16-s30-l8-b64-seq128-seed61-kernel-r31-offline-51e814a3/readme.md) | completed | profiler-active | kernel | 1,041.49 ms | 1,966.41 | 7,865.65 | - GiB |
| [npu-ep4-bf16-s30-l8-b64-seq128-seed61-operator-r32-offline-4509c387](../../../runs/4-card/ep4/npu-ep4-bf16-s30-l8-b64-seq128-seed61-operator-r32-offline-4509c387/readme.md) | completed | profiler-active | operator | 1,019.78 ms | 2,008.27 | 8,033.08 | - GiB |
| [npu-ep4-bf16-s30-l8-b64-seq128-seed61-memory-r31-offline-c62e7e67](../../../runs/4-card/ep4/npu-ep4-bf16-s30-l8-b64-seq128-seed61-memory-r31-offline-c62e7e67/readme.md) | completed | profiler-active | memory | 1,038.63 ms | 1,971.82 | 7,887.29 | - GiB |
| [npu-ep4-bf16-s30-l8-b64-seq128-seed61-flamegraph-r31-d83da0d2](../../../runs/4-card/ep4/npu-ep4-bf16-s30-l8-b64-seq128-seed61-flamegraph-r31-d83da0d2/readme.md) | completed | profiler-active | flamegraph | 1,016.35 ms | 2,015.06 | 8,060.23 | - GiB |
| [npu-ep4-bf16-s30-l8-b64-seq128-seed61-runtime-r31-offline-0a0a76e7](../../../runs/4-card/ep4/npu-ep4-bf16-s30-l8-b64-seq128-seed61-runtime-r31-offline-0a0a76e7/readme.md) | completed | profiler-active | runtime | 1,037.64 ms | 1,973.70 | 7,894.82 | - GiB |
| [npu-ep4-bf16-s30-l8-b64-seq128-seed61-system-r31-a9347568](../../../runs/4-card/ep4/npu-ep4-bf16-s30-l8-b64-seq128-seed61-system-r31-a9347568/readme.md) | captured | profiler-active | system | - ms | - | - | - GiB |

## experiment sequence

### 1. npu-ep4-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-4521dc9c

- Status: `completed`
- Started: `2026-08-24T16:41:21.029153+08:00`
- Full process and outputs: [run readme](../../../runs/4-card/ep4/npu-ep4-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-4521dc9c/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/4-card/ep4/npu-ep4-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-4521dc9c.html`

```bash
/root/miniconda3/envs/torchtitan-0803/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology ep4 --preset distributed --profiler-off --visible-devices 1,2,3,4 --steps 20 --skip-steps 8 --warmup-steps 2 --active-steps 3 --replicate 1
```

### 2. npu-ep4-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-d8ef2c7f

- Status: `completed`
- Started: `2026-08-31T12:51:09.197359+08:00`
- Full process and outputs: [run readme](../../../runs/4-card/ep4/npu-ep4-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-d8ef2c7f/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/4-card/ep4/npu-ep4-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-d8ef2c7f.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology all --preset all --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all
```

### 3. npu-ep4-bf16-s30-l8-b64-seq128-seed61-distributed-r31-c23f867c

- Status: `completed`
- Started: `2026-08-31T19:29:36.337370+08:00`
- Full process and outputs: [run readme](../../../runs/4-card/ep4/npu-ep4-bf16-s30-l8-b64-seq128-seed61-distributed-r31-c23f867c/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/4-card/ep4/npu-ep4-bf16-s30-l8-b64-seq128-seed61-distributed-r31-c23f867c.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topologies fsdp2-tp4-ep8,ddp4,fsdp2,fsdp4,tp2,tp4,cp2,cp4,pp2,pp4,ep2,ep4,fsdp2-tp2 --preset distributed --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all
```

### 4. npu-ep4-bf16-s30-l8-b64-seq128-seed61-kernel-r31-offline-51e814a3

- Status: `completed`
- Started: `2026-08-31T22:31:42.742131+08:00`
- Full process and outputs: [run readme](../../../runs/4-card/ep4/npu-ep4-bf16-s30-l8-b64-seq128-seed61-kernel-r31-offline-51e814a3/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/4-card/ep4/npu-ep4-bf16-s30-l8-b64-seq128-seed61-kernel-r31-offline-51e814a3.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology all --preset kernel --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all
```

### 5. npu-ep4-bf16-s30-l8-b64-seq128-seed61-operator-r32-offline-4509c387

- Status: `completed`
- Started: `2026-09-01T00:31:37.395924+08:00`
- Full process and outputs: [run readme](../../../runs/4-card/ep4/npu-ep4-bf16-s30-l8-b64-seq128-seed61-operator-r32-offline-4509c387/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/4-card/ep4/npu-ep4-bf16-s30-l8-b64-seq128-seed61-operator-r32-offline-4509c387.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --capture --device npu --topology ep4 --preset operator --visible-devices 0,1,2,3,4,5,6,7 --replicate 32 --analysis-tools none
```

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --analyze --device npu --topology ep4 --preset operator --visible-devices 0,1,2,3,4,5,6,7 --replicate 32 --analysis-tools all --parse-workers 8
```

### 6. npu-ep4-bf16-s30-l8-b64-seq128-seed61-memory-r31-offline-c62e7e67

- Status: `completed`
- Started: `2026-09-01T10:09:57.081180+08:00`
- Full process and outputs: [run readme](../../../runs/4-card/ep4/npu-ep4-bf16-s30-l8-b64-seq128-seed61-memory-r31-offline-c62e7e67/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/4-card/ep4/npu-ep4-bf16-s30-l8-b64-seq128-seed61-memory-r31-offline-c62e7e67.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --capture --device npu --topology ep4 --preset memory --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools none
```

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --analyze --device npu --topology ep4 --preset memory --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all --parse-workers 8
```

### 7. npu-ep4-bf16-s30-l8-b64-seq128-seed61-flamegraph-r31-d83da0d2

- Status: `completed`
- Started: `2026-09-01T11:28:37.444627+08:00`
- Full process and outputs: [run readme](../../../runs/4-card/ep4/npu-ep4-bf16-s30-l8-b64-seq128-seed61-flamegraph-r31-d83da0d2/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/4-card/ep4/npu-ep4-bf16-s30-l8-b64-seq128-seed61-flamegraph-r31-d83da0d2.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --capture --device npu --topology ep4 --preset flamegraph --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools none
```

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --analyze --device npu --topology ep4 --preset flamegraph --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all --parse-workers 8
```

### 8. npu-ep4-bf16-s30-l8-b64-seq128-seed61-runtime-r31-offline-0a0a76e7

- Status: `completed`
- Started: `2026-09-01T12:51:42.634544+08:00`
- Full process and outputs: [run readme](../../../runs/4-card/ep4/npu-ep4-bf16-s30-l8-b64-seq128-seed61-runtime-r31-offline-0a0a76e7/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/4-card/ep4/npu-ep4-bf16-s30-l8-b64-seq128-seed61-runtime-r31-offline-0a0a76e7.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --capture --device npu --topology ep4 --preset runtime --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools none
```

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --analyze --device npu --topology ep4 --preset runtime --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all --parse-workers 8
```

### 9. npu-ep4-bf16-s30-l8-b64-seq128-seed61-system-r31-a9347568

- Status: `captured`
- Started: `2026-09-01T14:15:31.578687+08:00`
- Full process and outputs: [run readme](../../../runs/4-card/ep4/npu-ep4-bf16-s30-l8-b64-seq128-seed61-system-r31-a9347568/readme.md)

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --capture --device npu --topology ep4 --preset system --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools none
```

## current summary

The highest recorded job throughput is 17,167.77 tok/s from `npu-ep4-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-4521dc9c`. Profiler-off runs are throughput evidence; profiler-active runs are attribution evidence only.
