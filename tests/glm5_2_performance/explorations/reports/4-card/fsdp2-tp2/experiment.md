# fsdp2-tp2 experiments

## topology

| field | value |
| --- | --- |
| world size | 4 |
| dp replicate / shard | 1 / 2 |
| tp / cp / pp / ep | 2 / 1 / 1 / 1 |
| recorded runs | 9 |

## run results

| run | status | mode | preset | median step | tok/s/device | tok/s/job | peak hbm |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| [npu-fsdp2-tp2-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-823a6a00](../../../runs/4-card/fsdp2-tp2/npu-fsdp2-tp2-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-823a6a00/readme.md) | completed | profiler-off | distributed | 2,204.28 ms | 929.22 | 3,716.87 | 0.404 GiB |
| [npu-fsdp2-tp2-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-0a488f2b](../../../runs/4-card/fsdp2-tp2/npu-fsdp2-tp2-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-0a488f2b/readme.md) | completed | profiler-active | overview | 3,292.14 ms | 622.09 | 2,488.35 | - GiB |
| [npu-fsdp2-tp2-bf16-s30-l8-b64-seq128-seed61-distributed-r31-cc86808e](../../../runs/4-card/fsdp2-tp2/npu-fsdp2-tp2-bf16-s30-l8-b64-seq128-seed61-distributed-r31-cc86808e/readme.md) | completed | profiler-active | distributed | 3,067.43 ms | 667.66 | 2,670.64 | - GiB |
| [npu-fsdp2-tp2-bf16-s30-l8-b64-seq128-seed61-kernel-r31-offline-bb31d578](../../../runs/4-card/fsdp2-tp2/npu-fsdp2-tp2-bf16-s30-l8-b64-seq128-seed61-kernel-r31-offline-bb31d578/readme.md) | completed | profiler-active | kernel | 3,126.33 ms | 655.08 | 2,620.33 | - GiB |
| [npu-fsdp2-tp2-bf16-s30-l8-b64-seq128-seed61-operator-r32-offline-8dba4884](../../../runs/4-card/fsdp2-tp2/npu-fsdp2-tp2-bf16-s30-l8-b64-seq128-seed61-operator-r32-offline-8dba4884/readme.md) | completed | profiler-active | operator | 3,314.79 ms | 617.84 | 2,471.35 | - GiB |
| [npu-fsdp2-tp2-bf16-s30-l8-b64-seq128-seed61-memory-r31-offline-12273c4b](../../../runs/4-card/fsdp2-tp2/npu-fsdp2-tp2-bf16-s30-l8-b64-seq128-seed61-memory-r31-offline-12273c4b/readme.md) | completed | profiler-active | memory | 3,116.35 ms | 657.18 | 2,628.71 | - GiB |
| [npu-fsdp2-tp2-bf16-s30-l8-b64-seq128-seed61-flamegraph-r31-601ebecb](../../../runs/4-card/fsdp2-tp2/npu-fsdp2-tp2-bf16-s30-l8-b64-seq128-seed61-flamegraph-r31-601ebecb/readme.md) | completed | profiler-active | flamegraph | 3,081.47 ms | 664.62 | 2,658.47 | - GiB |
| [npu-fsdp2-tp2-bf16-s30-l8-b64-seq128-seed61-runtime-r31-offline-9b412abc](../../../runs/4-card/fsdp2-tp2/npu-fsdp2-tp2-bf16-s30-l8-b64-seq128-seed61-runtime-r31-offline-9b412abc/readme.md) | completed | profiler-active | runtime | 3,082.26 ms | 664.45 | 2,657.79 | - GiB |
| [npu-fsdp2-tp2-bf16-s30-l8-b64-seq128-seed61-system-r31-0898ddfd](../../../runs/4-card/fsdp2-tp2/npu-fsdp2-tp2-bf16-s30-l8-b64-seq128-seed61-system-r31-0898ddfd/readme.md) | captured | profiler-active | system | - ms | - | - | - GiB |

## experiment sequence

### 1. npu-fsdp2-tp2-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-823a6a00

- Status: `completed`
- Started: `2026-08-24T16:42:18.708928+08:00`
- Full process and outputs: [run readme](../../../runs/4-card/fsdp2-tp2/npu-fsdp2-tp2-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-823a6a00/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/4-card/fsdp2-tp2/npu-fsdp2-tp2-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-823a6a00.html`

```bash
/root/miniconda3/envs/torchtitan-0803/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology fsdp2-tp2 --preset distributed --profiler-off --visible-devices 1,2,3,4 --steps 20 --skip-steps 8 --warmup-steps 2 --active-steps 3 --replicate 1
```

### 2. npu-fsdp2-tp2-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-0a488f2b

- Status: `completed`
- Started: `2026-08-31T12:53:21.943468+08:00`
- Full process and outputs: [run readme](../../../runs/4-card/fsdp2-tp2/npu-fsdp2-tp2-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-0a488f2b/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/4-card/fsdp2-tp2/npu-fsdp2-tp2-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-0a488f2b.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology all --preset all --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all
```

### 3. npu-fsdp2-tp2-bf16-s30-l8-b64-seq128-seed61-distributed-r31-cc86808e

- Status: `completed`
- Started: `2026-08-31T19:33:30.741978+08:00`
- Full process and outputs: [run readme](../../../runs/4-card/fsdp2-tp2/npu-fsdp2-tp2-bf16-s30-l8-b64-seq128-seed61-distributed-r31-cc86808e/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/4-card/fsdp2-tp2/npu-fsdp2-tp2-bf16-s30-l8-b64-seq128-seed61-distributed-r31-cc86808e.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topologies fsdp2-tp4-ep8,ddp4,fsdp2,fsdp4,tp2,tp4,cp2,cp4,pp2,pp4,ep2,ep4,fsdp2-tp2 --preset distributed --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all
```

### 4. npu-fsdp2-tp2-bf16-s30-l8-b64-seq128-seed61-kernel-r31-offline-bb31d578

- Status: `completed`
- Started: `2026-08-31T22:34:11.917720+08:00`
- Full process and outputs: [run readme](../../../runs/4-card/fsdp2-tp2/npu-fsdp2-tp2-bf16-s30-l8-b64-seq128-seed61-kernel-r31-offline-bb31d578/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/4-card/fsdp2-tp2/npu-fsdp2-tp2-bf16-s30-l8-b64-seq128-seed61-kernel-r31-offline-bb31d578.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology all --preset kernel --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all
```

### 5. npu-fsdp2-tp2-bf16-s30-l8-b64-seq128-seed61-operator-r32-offline-8dba4884

- Status: `completed`
- Started: `2026-09-01T00:33:00.154379+08:00`
- Full process and outputs: [run readme](../../../runs/4-card/fsdp2-tp2/npu-fsdp2-tp2-bf16-s30-l8-b64-seq128-seed61-operator-r32-offline-8dba4884/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/4-card/fsdp2-tp2/npu-fsdp2-tp2-bf16-s30-l8-b64-seq128-seed61-operator-r32-offline-8dba4884.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --capture --device npu --topology fsdp2-tp2 --preset operator --visible-devices 0,1,2,3,4,5,6,7 --replicate 32 --analysis-tools none
```

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --analyze --device npu --topology fsdp2-tp2 --preset operator --visible-devices 0,1,2,3,4,5,6,7 --replicate 32 --analysis-tools all --parse-workers 8
```

### 6. npu-fsdp2-tp2-bf16-s30-l8-b64-seq128-seed61-memory-r31-offline-12273c4b

- Status: `completed`
- Started: `2026-09-01T10:11:24.368333+08:00`
- Full process and outputs: [run readme](../../../runs/4-card/fsdp2-tp2/npu-fsdp2-tp2-bf16-s30-l8-b64-seq128-seed61-memory-r31-offline-12273c4b/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/4-card/fsdp2-tp2/npu-fsdp2-tp2-bf16-s30-l8-b64-seq128-seed61-memory-r31-offline-12273c4b.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --capture --device npu --topology fsdp2-tp2 --preset memory --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools none
```

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --analyze --device npu --topology fsdp2-tp2 --preset memory --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all --parse-workers 8
```

### 7. npu-fsdp2-tp2-bf16-s30-l8-b64-seq128-seed61-flamegraph-r31-601ebecb

- Status: `completed`
- Started: `2026-09-01T11:29:51.145445+08:00`
- Full process and outputs: [run readme](../../../runs/4-card/fsdp2-tp2/npu-fsdp2-tp2-bf16-s30-l8-b64-seq128-seed61-flamegraph-r31-601ebecb/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/4-card/fsdp2-tp2/npu-fsdp2-tp2-bf16-s30-l8-b64-seq128-seed61-flamegraph-r31-601ebecb.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --capture --device npu --topology fsdp2-tp2 --preset flamegraph --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools none
```

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --analyze --device npu --topology fsdp2-tp2 --preset flamegraph --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all --parse-workers 8
```

### 8. npu-fsdp2-tp2-bf16-s30-l8-b64-seq128-seed61-runtime-r31-offline-9b412abc

- Status: `completed`
- Started: `2026-09-01T12:53:07.537573+08:00`
- Full process and outputs: [run readme](../../../runs/4-card/fsdp2-tp2/npu-fsdp2-tp2-bf16-s30-l8-b64-seq128-seed61-runtime-r31-offline-9b412abc/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/4-card/fsdp2-tp2/npu-fsdp2-tp2-bf16-s30-l8-b64-seq128-seed61-runtime-r31-offline-9b412abc.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --capture --device npu --topology fsdp2-tp2 --preset runtime --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools none
```

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --analyze --device npu --topology fsdp2-tp2 --preset runtime --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all --parse-workers 8
```

### 9. npu-fsdp2-tp2-bf16-s30-l8-b64-seq128-seed61-system-r31-0898ddfd

- Status: `captured`
- Started: `2026-09-01T14:16:53.945881+08:00`
- Full process and outputs: [run readme](../../../runs/4-card/fsdp2-tp2/npu-fsdp2-tp2-bf16-s30-l8-b64-seq128-seed61-system-r31-0898ddfd/readme.md)

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --capture --device npu --topology fsdp2-tp2 --preset system --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools none
```

## current summary

The highest recorded job throughput is 3,716.87 tok/s from `npu-fsdp2-tp2-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-823a6a00`. Profiler-off runs are throughput evidence; profiler-active runs are attribution evidence only.
