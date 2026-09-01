# fsdp2-tp2-pp2 experiments

## topology

| field | value |
| --- | --- |
| world size | 8 |
| dp replicate / shard | 1 / 2 |
| tp / cp / pp / ep | 2 / 1 / 2 / 1 |
| recorded runs | 11 |

## run results

| run | status | mode | preset | median step | tok/s/device | tok/s/job | peak hbm |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| [npu-fsdp2-tp2-pp2-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-5759ec05](../../../runs/8-card/fsdp2-tp2-pp2/npu-fsdp2-tp2-pp2-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-5759ec05/readme.md) | completed | profiler-off | distributed | 9,152.28 ms | 111.88 | 895.08 | 0.074 GiB |
| [npu-fsdp2-tp2-pp2-bf16-s20-l8-b64-seq128-seed61-distributed-r1-3f710e94](../../../runs/8-card/fsdp2-tp2-pp2/npu-fsdp2-tp2-pp2-bf16-s20-l8-b64-seq128-seed61-distributed-r1-3f710e94/readme.md) | failed | profiler-active | - | - ms | - | - | - GiB |
| [npu-fsdp2-tp2-pp2-bf16-s20-l8-b64-seq128-seed61-distributed-r2-02a4a78f](../../../runs/8-card/fsdp2-tp2-pp2/npu-fsdp2-tp2-pp2-bf16-s20-l8-b64-seq128-seed61-distributed-r2-02a4a78f/readme.md) | completed | profiler-active | distributed | 9,675.98 ms | 105.83 | 846.63 | - GiB |
| [npu-fsdp2-tp2-pp2-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-6cdcb093](../../../runs/8-card/fsdp2-tp2-pp2/npu-fsdp2-tp2-pp2-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-6cdcb093/readme.md) | completed | profiler-active | overview | 13,387.05 ms | 76.49 | 611.93 | - GiB |
| [npu-fsdp2-tp2-pp2-bf16-s30-l8-b64-seq128-seed61-distributed-r31-28affced](../../../runs/8-card/fsdp2-tp2-pp2/npu-fsdp2-tp2-pp2-bf16-s30-l8-b64-seq128-seed61-distributed-r31-28affced/readme.md) | completed | profiler-active | distributed | 13,350.39 ms | 76.70 | 613.62 | - GiB |
| [npu-fsdp2-tp2-pp2-bf16-s30-l8-b64-seq128-seed61-kernel-r31-offline-e2d27edb](../../../runs/8-card/fsdp2-tp2-pp2/npu-fsdp2-tp2-pp2-bf16-s30-l8-b64-seq128-seed61-kernel-r31-offline-e2d27edb/readme.md) | completed | profiler-active | kernel | 13,781.13 ms | 74.30 | 594.44 | - GiB |
| [npu-fsdp2-tp2-pp2-bf16-s30-l8-b64-seq128-seed61-operator-r32-offline-9a812951](../../../runs/8-card/fsdp2-tp2-pp2/npu-fsdp2-tp2-pp2-bf16-s30-l8-b64-seq128-seed61-operator-r32-offline-9a812951/readme.md) | completed | profiler-active | operator | 13,526.82 ms | 75.70 | 605.61 | - GiB |
| [npu-fsdp2-tp2-pp2-bf16-s30-l8-b64-seq128-seed61-memory-r31-offline-cdead6a4](../../../runs/8-card/fsdp2-tp2-pp2/npu-fsdp2-tp2-pp2-bf16-s30-l8-b64-seq128-seed61-memory-r31-offline-cdead6a4/readme.md) | completed | profiler-active | memory | 13,326.52 ms | 76.84 | 614.71 | - GiB |
| [npu-fsdp2-tp2-pp2-bf16-s30-l8-b64-seq128-seed61-flamegraph-r31-c081c381](../../../runs/8-card/fsdp2-tp2-pp2/npu-fsdp2-tp2-pp2-bf16-s30-l8-b64-seq128-seed61-flamegraph-r31-c081c381/readme.md) | completed | profiler-active | flamegraph | 13,865.97 ms | 73.85 | 590.80 | - GiB |
| [npu-fsdp2-tp2-pp2-bf16-s30-l8-b64-seq128-seed61-runtime-r31-offline-912b7a1c](../../../runs/8-card/fsdp2-tp2-pp2/npu-fsdp2-tp2-pp2-bf16-s30-l8-b64-seq128-seed61-runtime-r31-offline-912b7a1c/readme.md) | completed | profiler-active | runtime | 12,955.86 ms | 79.04 | 632.30 | - GiB |
| [npu-fsdp2-tp2-pp2-bf16-s30-l8-b64-seq128-seed61-system-r31-0d2fd600](../../../runs/8-card/fsdp2-tp2-pp2/npu-fsdp2-tp2-pp2-bf16-s30-l8-b64-seq128-seed61-system-r31-0d2fd600/readme.md) | completed | profiler-active | system | 13,187.91 ms | 77.65 | 621.17 | - GiB |

## experiment sequence

### 1. npu-fsdp2-tp2-pp2-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-5759ec05

- Status: `completed`
- Started: `2026-08-24T18:05:29.037922+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/fsdp2-tp2-pp2/npu-fsdp2-tp2-pp2-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-5759ec05/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/fsdp2-tp2-pp2/npu-fsdp2-tp2-pp2-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-5759ec05.html`

```bash
/root/miniconda3/envs/torchtitan-0803/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology fsdp2-tp2-pp2 --preset distributed --profiler-off --visible-devices 0,1,2,3,4,5,6,7 --steps 20 --skip-steps 8 --warmup-steps 2 --active-steps 3 --replicate 1
```

### 2. npu-fsdp2-tp2-pp2-bf16-s20-l8-b64-seq128-seed61-distributed-r1-3f710e94

- Status: `failed`
- Started: `2026-08-25T09:23:34.781135+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/fsdp2-tp2-pp2/npu-fsdp2-tp2-pp2-bf16-s20-l8-b64-seq128-seed61-distributed-r1-3f710e94/readme.md)

```bash
/root/miniconda3/envs/torchtitan-0803/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology fsdp2-tp2-pp2 --preset distributed --visible-devices 0,1,2,3,4,5,6,7 --steps 20 --skip-steps 8 --warmup-steps 2 --active-steps 3 --replicate 1 --cluster
```

### 3. npu-fsdp2-tp2-pp2-bf16-s20-l8-b64-seq128-seed61-distributed-r2-02a4a78f

- Status: `completed`
- Started: `2026-08-25T13:40:27.735606+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/fsdp2-tp2-pp2/npu-fsdp2-tp2-pp2-bf16-s20-l8-b64-seq128-seed61-distributed-r2-02a4a78f/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/fsdp2-tp2-pp2/npu-fsdp2-tp2-pp2-bf16-s20-l8-b64-seq128-seed61-distributed-r2-02a4a78f.html`

```bash
/root/miniconda3/envs/torchtitan-0803/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topologies pp8,fsdp2-tp4,fsdp2-cp4,tp2-cp4,fsdp4-tp2,fsdp2-pp4,fsdp2-tp2-pp2,fsdp2-tp4-ep8 --preset distributed --visible-devices 0,1,2,3,4,5,6,7 --steps 20 --skip-steps 8 --warmup-steps 2 --active-steps 3 --replicate 2 --cluster
```

### 4. npu-fsdp2-tp2-pp2-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-6cdcb093

- Status: `completed`
- Started: `2026-08-31T11:24:30.761789+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/fsdp2-tp2-pp2/npu-fsdp2-tp2-pp2-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-6cdcb093/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/fsdp2-tp2-pp2/npu-fsdp2-tp2-pp2-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-6cdcb093.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology all --preset all --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all
```

### 5. npu-fsdp2-tp2-pp2-bf16-s30-l8-b64-seq128-seed61-distributed-r31-28affced

- Status: `completed`
- Started: `2026-08-31T15:42:23.767336+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/fsdp2-tp2-pp2/npu-fsdp2-tp2-pp2-bf16-s30-l8-b64-seq128-seed61-distributed-r31-28affced/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/fsdp2-tp2-pp2/npu-fsdp2-tp2-pp2-bf16-s30-l8-b64-seq128-seed61-distributed-r31-28affced.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology all --preset all --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all
```

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --analyze --device npu --topology fsdp2-tp2-pp2 --preset distributed --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all
```

### 6. npu-fsdp2-tp2-pp2-bf16-s30-l8-b64-seq128-seed61-kernel-r31-offline-e2d27edb

- Status: `completed`
- Started: `2026-08-31T20:51:25.328070+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/fsdp2-tp2-pp2/npu-fsdp2-tp2-pp2-bf16-s30-l8-b64-seq128-seed61-kernel-r31-offline-e2d27edb/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/fsdp2-tp2-pp2/npu-fsdp2-tp2-pp2-bf16-s30-l8-b64-seq128-seed61-kernel-r31-offline-e2d27edb.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology all --preset kernel --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all
```

### 7. npu-fsdp2-tp2-pp2-bf16-s30-l8-b64-seq128-seed61-operator-r32-offline-9a812951

- Status: `completed`
- Started: `2026-08-31T23:49:16.356157+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/fsdp2-tp2-pp2/npu-fsdp2-tp2-pp2-bf16-s30-l8-b64-seq128-seed61-operator-r32-offline-9a812951/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/fsdp2-tp2-pp2/npu-fsdp2-tp2-pp2-bf16-s30-l8-b64-seq128-seed61-operator-r32-offline-9a812951.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --capture --device npu --topology fsdp2-tp2-pp2 --preset operator --visible-devices 0,1,2,3,4,5,6,7 --replicate 32 --analysis-tools none
```

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --analyze --device npu --topology fsdp2-tp2-pp2 --preset operator --visible-devices 0,1,2,3,4,5,6,7 --replicate 32 --analysis-tools all --parse-workers 8
```

### 8. npu-fsdp2-tp2-pp2-bf16-s30-l8-b64-seq128-seed61-memory-r31-offline-cdead6a4

- Status: `completed`
- Started: `2026-09-01T09:30:48.565641+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/fsdp2-tp2-pp2/npu-fsdp2-tp2-pp2-bf16-s30-l8-b64-seq128-seed61-memory-r31-offline-cdead6a4/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/fsdp2-tp2-pp2/npu-fsdp2-tp2-pp2-bf16-s30-l8-b64-seq128-seed61-memory-r31-offline-cdead6a4.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --capture --device npu --topology fsdp2-tp2-pp2 --preset memory --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools none
```

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --analyze --device npu --topology fsdp2-tp2-pp2 --preset memory --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all --parse-workers 8
```

### 9. npu-fsdp2-tp2-pp2-bf16-s30-l8-b64-seq128-seed61-flamegraph-r31-c081c381

- Status: `completed`
- Started: `2026-09-01T10:50:20.905276+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/fsdp2-tp2-pp2/npu-fsdp2-tp2-pp2-bf16-s30-l8-b64-seq128-seed61-flamegraph-r31-c081c381/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/fsdp2-tp2-pp2/npu-fsdp2-tp2-pp2-bf16-s30-l8-b64-seq128-seed61-flamegraph-r31-c081c381.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --capture --device npu --topology fsdp2-tp2-pp2 --preset flamegraph --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools none
```

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --analyze --device npu --topology fsdp2-tp2-pp2 --preset flamegraph --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all --parse-workers 8
```

### 10. npu-fsdp2-tp2-pp2-bf16-s30-l8-b64-seq128-seed61-runtime-r31-offline-912b7a1c

- Status: `completed`
- Started: `2026-09-01T12:08:49.996082+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/fsdp2-tp2-pp2/npu-fsdp2-tp2-pp2-bf16-s30-l8-b64-seq128-seed61-runtime-r31-offline-912b7a1c/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/fsdp2-tp2-pp2/npu-fsdp2-tp2-pp2-bf16-s30-l8-b64-seq128-seed61-runtime-r31-offline-912b7a1c.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --capture --device npu --topology fsdp2-tp2-pp2 --preset runtime --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools none
```

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --analyze --device npu --topology fsdp2-tp2-pp2 --preset runtime --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all --parse-workers 8
```

### 11. npu-fsdp2-tp2-pp2-bf16-s30-l8-b64-seq128-seed61-system-r31-0d2fd600

- Status: `completed`
- Started: `2026-09-01T13:35:20.581936+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/fsdp2-tp2-pp2/npu-fsdp2-tp2-pp2-bf16-s30-l8-b64-seq128-seed61-system-r31-0d2fd600/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/fsdp2-tp2-pp2/npu-fsdp2-tp2-pp2-bf16-s30-l8-b64-seq128-seed61-system-r31-0d2fd600.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --capture --device npu --topology fsdp2-tp2-pp2 --preset system --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools none
```

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --analyze --device npu --topology fsdp2-tp2-pp2 --preset system --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all --parse-workers 8
```

## current summary

The highest recorded job throughput is 895.08 tok/s from `npu-fsdp2-tp2-pp2-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-5759ec05`. Profiler-off runs are throughput evidence; profiler-active runs are attribution evidence only.
