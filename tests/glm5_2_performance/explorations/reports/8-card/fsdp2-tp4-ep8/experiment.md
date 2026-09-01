# fsdp2-tp4-ep8 experiments

## topology

| field | value |
| --- | --- |
| world size | 8 |
| dp replicate / shard | 1 / 2 |
| tp / cp / pp / ep | 4 / 1 / 1 / 8 |
| recorded runs | 11 |

## run results

| run | status | mode | preset | median step | tok/s/device | tok/s/job | peak hbm |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| [npu-fsdp2-tp4-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-53db2e7c](../../../runs/8-card/fsdp2-tp4-ep8/npu-fsdp2-tp4-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-53db2e7c/readme.md) | completed | profiler-off | distributed | 2,323.02 ms | 440.80 | 3,526.44 | 0.216 GiB |
| [npu-fsdp2-tp4-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-r1-bf02b5d0](../../../runs/8-card/fsdp2-tp4-ep8/npu-fsdp2-tp4-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-r1-bf02b5d0/readme.md) | failed | profiler-active | - | - ms | - | - | - GiB |
| [npu-fsdp2-tp4-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-r2-e6d2b2bd](../../../runs/8-card/fsdp2-tp4-ep8/npu-fsdp2-tp4-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-r2-e6d2b2bd/readme.md) | completed | profiler-active | distributed | 2,658.32 ms | 385.21 | 3,081.64 | - GiB |
| [npu-fsdp2-tp4-ep8-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-a644c0f7](../../../runs/8-card/fsdp2-tp4-ep8/npu-fsdp2-tp4-ep8-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-a644c0f7/readme.md) | completed | profiler-active | overview | 3,549.01 ms | 288.53 | 2,308.25 | - GiB |
| [npu-fsdp2-tp4-ep8-bf16-s30-l8-b64-seq128-seed61-distributed-r31-e45371d7](../../../runs/8-card/fsdp2-tp4-ep8/npu-fsdp2-tp4-ep8-bf16-s30-l8-b64-seq128-seed61-distributed-r31-e45371d7/readme.md) | completed | profiler-active | distributed | 3,690.06 ms | 277.50 | 2,220.02 | - GiB |
| [npu-fsdp2-tp4-ep8-bf16-s30-l8-b64-seq128-seed61-kernel-r31-offline-bef5b8da](../../../runs/8-card/fsdp2-tp4-ep8/npu-fsdp2-tp4-ep8-bf16-s30-l8-b64-seq128-seed61-kernel-r31-offline-bef5b8da/readme.md) | completed | profiler-active | kernel | 3,568.15 ms | 286.98 | 2,295.86 | - GiB |
| [npu-fsdp2-tp4-ep8-bf16-s30-l8-b64-seq128-seed61-operator-r32-offline-6809909b](../../../runs/8-card/fsdp2-tp4-ep8/npu-fsdp2-tp4-ep8-bf16-s30-l8-b64-seq128-seed61-operator-r32-offline-6809909b/readme.md) | completed | profiler-active | operator | 3,526.58 ms | 290.37 | 2,322.93 | - GiB |
| [npu-fsdp2-tp4-ep8-bf16-s30-l8-b64-seq128-seed61-memory-r31-offline-3721b7ab](../../../runs/8-card/fsdp2-tp4-ep8/npu-fsdp2-tp4-ep8-bf16-s30-l8-b64-seq128-seed61-memory-r31-offline-3721b7ab/readme.md) | completed | profiler-active | memory | 3,538.02 ms | 289.43 | 2,315.42 | - GiB |
| [npu-fsdp2-tp4-ep8-bf16-s30-l8-b64-seq128-seed61-flamegraph-r31-66962740](../../../runs/8-card/fsdp2-tp4-ep8/npu-fsdp2-tp4-ep8-bf16-s30-l8-b64-seq128-seed61-flamegraph-r31-66962740/readme.md) | completed | profiler-active | flamegraph | 3,496.04 ms | 292.90 | 2,343.22 | - GiB |
| [npu-fsdp2-tp4-ep8-bf16-s30-l8-b64-seq128-seed61-runtime-r31-offline-bfb981e2](../../../runs/8-card/fsdp2-tp4-ep8/npu-fsdp2-tp4-ep8-bf16-s30-l8-b64-seq128-seed61-runtime-r31-offline-bfb981e2/readme.md) | completed | profiler-active | runtime | 3,467.63 ms | 295.30 | 2,362.42 | - GiB |
| [npu-fsdp2-tp4-ep8-bf16-s30-l8-b64-seq128-seed61-system-r31-5e4c72ac](../../../runs/8-card/fsdp2-tp4-ep8/npu-fsdp2-tp4-ep8-bf16-s30-l8-b64-seq128-seed61-system-r31-5e4c72ac/readme.md) | completed | profiler-active | system | 3,511.87 ms | 291.58 | 2,332.66 | - GiB |

## experiment sequence

### 1. npu-fsdp2-tp4-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-53db2e7c

- Status: `completed`
- Started: `2026-08-24T18:09:26.542934+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/fsdp2-tp4-ep8/npu-fsdp2-tp4-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-53db2e7c/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/fsdp2-tp4-ep8/npu-fsdp2-tp4-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-53db2e7c.html`

```bash
/root/miniconda3/envs/torchtitan-0803/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology fsdp2-tp4-ep8 --preset distributed --profiler-off --visible-devices 0,1,2,3,4,5,6,7 --steps 20 --skip-steps 8 --warmup-steps 2 --active-steps 3 --replicate 1
```

### 2. npu-fsdp2-tp4-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-r1-bf02b5d0

- Status: `failed`
- Started: `2026-08-25T09:25:07.206333+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/fsdp2-tp4-ep8/npu-fsdp2-tp4-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-r1-bf02b5d0/readme.md)

```bash
/root/miniconda3/envs/torchtitan-0803/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology fsdp2-tp4-ep8 --preset distributed --visible-devices 0,1,2,3,4,5,6,7 --steps 20 --skip-steps 8 --warmup-steps 2 --active-steps 3 --replicate 1 --cluster
```

### 3. npu-fsdp2-tp4-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-r2-e6d2b2bd

- Status: `completed`
- Started: `2026-08-25T13:57:47.973837+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/fsdp2-tp4-ep8/npu-fsdp2-tp4-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-r2-e6d2b2bd/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/fsdp2-tp4-ep8/npu-fsdp2-tp4-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-r2-e6d2b2bd.html`

```bash
/root/miniconda3/envs/torchtitan-0803/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topologies pp8,fsdp2-tp4,fsdp2-cp4,tp2-cp4,fsdp4-tp2,fsdp2-pp4,fsdp2-tp2-pp2,fsdp2-tp4-ep8 --preset distributed --visible-devices 0,1,2,3,4,5,6,7 --steps 20 --skip-steps 8 --warmup-steps 2 --active-steps 3 --replicate 2 --cluster
```

### 4. npu-fsdp2-tp4-ep8-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-a644c0f7

- Status: `completed`
- Started: `2026-08-31T11:39:29.613880+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/fsdp2-tp4-ep8/npu-fsdp2-tp4-ep8-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-a644c0f7/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/fsdp2-tp4-ep8/npu-fsdp2-tp4-ep8-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-a644c0f7.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology all --preset all --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all
```

### 5. npu-fsdp2-tp4-ep8-bf16-s30-l8-b64-seq128-seed61-distributed-r31-e45371d7

- Status: `completed`
- Started: `2026-08-31T16:50:46.742996+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/fsdp2-tp4-ep8/npu-fsdp2-tp4-ep8-bf16-s30-l8-b64-seq128-seed61-distributed-r31-e45371d7/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/fsdp2-tp4-ep8/npu-fsdp2-tp4-ep8-bf16-s30-l8-b64-seq128-seed61-distributed-r31-e45371d7.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topologies fsdp2-tp4-ep8,ddp4,fsdp2,fsdp4,tp2,tp4,cp2,cp4,pp2,pp4,ep2,ep4,fsdp2-tp2 --preset distributed --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all
```

### 6. npu-fsdp2-tp4-ep8-bf16-s30-l8-b64-seq128-seed61-kernel-r31-offline-bef5b8da

- Status: `completed`
- Started: `2026-08-31T21:07:39.832507+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/fsdp2-tp4-ep8/npu-fsdp2-tp4-ep8-bf16-s30-l8-b64-seq128-seed61-kernel-r31-offline-bef5b8da/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/fsdp2-tp4-ep8/npu-fsdp2-tp4-ep8-bf16-s30-l8-b64-seq128-seed61-kernel-r31-offline-bef5b8da.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology all --preset kernel --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all
```

### 7. npu-fsdp2-tp4-ep8-bf16-s30-l8-b64-seq128-seed61-operator-r32-offline-6809909b

- Status: `completed`
- Started: `2026-08-31T23:57:29.115573+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/fsdp2-tp4-ep8/npu-fsdp2-tp4-ep8-bf16-s30-l8-b64-seq128-seed61-operator-r32-offline-6809909b/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/fsdp2-tp4-ep8/npu-fsdp2-tp4-ep8-bf16-s30-l8-b64-seq128-seed61-operator-r32-offline-6809909b.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --capture --device npu --topology fsdp2-tp4-ep8 --preset operator --visible-devices 0,1,2,3,4,5,6,7 --replicate 32 --analysis-tools none
```

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --analyze --device npu --topology fsdp2-tp4-ep8 --preset operator --visible-devices 0,1,2,3,4,5,6,7 --replicate 32 --analysis-tools all --parse-workers 8
```

### 8. npu-fsdp2-tp4-ep8-bf16-s30-l8-b64-seq128-seed61-memory-r31-offline-3721b7ab

- Status: `completed`
- Started: `2026-09-01T09:38:33.082315+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/fsdp2-tp4-ep8/npu-fsdp2-tp4-ep8-bf16-s30-l8-b64-seq128-seed61-memory-r31-offline-3721b7ab/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/fsdp2-tp4-ep8/npu-fsdp2-tp4-ep8-bf16-s30-l8-b64-seq128-seed61-memory-r31-offline-3721b7ab.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --capture --device npu --topology fsdp2-tp4-ep8 --preset memory --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools none
```

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --analyze --device npu --topology fsdp2-tp4-ep8 --preset memory --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all --parse-workers 8
```

### 9. npu-fsdp2-tp4-ep8-bf16-s30-l8-b64-seq128-seed61-flamegraph-r31-66962740

- Status: `completed`
- Started: `2026-09-01T10:58:20.207063+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/fsdp2-tp4-ep8/npu-fsdp2-tp4-ep8-bf16-s30-l8-b64-seq128-seed61-flamegraph-r31-66962740/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/fsdp2-tp4-ep8/npu-fsdp2-tp4-ep8-bf16-s30-l8-b64-seq128-seed61-flamegraph-r31-66962740.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --capture --device npu --topology fsdp2-tp4-ep8 --preset flamegraph --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools none
```

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --analyze --device npu --topology fsdp2-tp4-ep8 --preset flamegraph --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all --parse-workers 8
```

### 10. npu-fsdp2-tp4-ep8-bf16-s30-l8-b64-seq128-seed61-runtime-r31-offline-bfb981e2

- Status: `completed`
- Started: `2026-09-01T12:16:57.342899+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/fsdp2-tp4-ep8/npu-fsdp2-tp4-ep8-bf16-s30-l8-b64-seq128-seed61-runtime-r31-offline-bfb981e2/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/fsdp2-tp4-ep8/npu-fsdp2-tp4-ep8-bf16-s30-l8-b64-seq128-seed61-runtime-r31-offline-bfb981e2.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --capture --device npu --topology fsdp2-tp4-ep8 --preset runtime --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools none
```

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --analyze --device npu --topology fsdp2-tp4-ep8 --preset runtime --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all --parse-workers 8
```

### 11. npu-fsdp2-tp4-ep8-bf16-s30-l8-b64-seq128-seed61-system-r31-5e4c72ac

- Status: `completed`
- Started: `2026-09-01T13:43:23.139137+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/fsdp2-tp4-ep8/npu-fsdp2-tp4-ep8-bf16-s30-l8-b64-seq128-seed61-system-r31-5e4c72ac/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/fsdp2-tp4-ep8/npu-fsdp2-tp4-ep8-bf16-s30-l8-b64-seq128-seed61-system-r31-5e4c72ac.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --capture --device npu --topology fsdp2-tp4-ep8 --preset system --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools none
```

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --analyze --device npu --topology fsdp2-tp4-ep8 --preset system --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all --parse-workers 8
```

## current summary

The highest recorded job throughput is 3,526.44 tok/s from `npu-fsdp2-tp4-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-53db2e7c`. Profiler-off runs are throughput evidence; profiler-active runs are attribution evidence only.
