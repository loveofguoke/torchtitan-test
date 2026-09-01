# tp2 experiments

## topology

| field | value |
| --- | --- |
| world size | 2 |
| dp replicate / shard | 1 / 1 |
| tp / cp / pp / ep | 2 / 1 / 1 / 1 |
| recorded runs | 8 |

## run results

| run | status | mode | preset | median step | tok/s/device | tok/s/job | peak hbm |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| [npu-tp2-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-e698990c](../../../runs/2-card/tp2/npu-tp2-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-e698990c/readme.md) | completed | profiler-active | overview | 6,060.82 ms | 675.82 | 1,351.63 | - GiB |
| [npu-tp2-bf16-s30-l8-b64-seq128-seed61-distributed-r31-e4b07492](../../../runs/2-card/tp2/npu-tp2-bf16-s30-l8-b64-seq128-seed61-distributed-r31-e4b07492/readme.md) | completed | profiler-active | distributed | 6,169.70 ms | 663.89 | 1,327.78 | - GiB |
| [npu-tp2-bf16-s30-l8-b64-seq128-seed61-kernel-r31-offline-acf304e8](../../../runs/2-card/tp2/npu-tp2-bf16-s30-l8-b64-seq128-seed61-kernel-r31-offline-acf304e8/readme.md) | completed | profiler-active | kernel | 6,340.36 ms | 646.02 | 1,292.04 | - GiB |
| [npu-tp2-bf16-s30-l8-b64-seq128-seed61-operator-r32-offline-e04c189e](../../../runs/2-card/tp2/npu-tp2-bf16-s30-l8-b64-seq128-seed61-operator-r32-offline-e04c189e/readme.md) | completed | profiler-active | operator | 6,104.71 ms | 670.96 | 1,341.92 | - GiB |
| [npu-tp2-bf16-s30-l8-b64-seq128-seed61-memory-r31-offline-175f14b6](../../../runs/2-card/tp2/npu-tp2-bf16-s30-l8-b64-seq128-seed61-memory-r31-offline-175f14b6/readme.md) | completed | profiler-active | memory | 6,085.60 ms | 673.06 | 1,346.13 | - GiB |
| [npu-tp2-bf16-s30-l8-b64-seq128-seed61-flamegraph-r31-3b7f386d](../../../runs/2-card/tp2/npu-tp2-bf16-s30-l8-b64-seq128-seed61-flamegraph-r31-3b7f386d/readme.md) | completed | profiler-active | flamegraph | 6,081.41 ms | 673.53 | 1,347.06 | - GiB |
| [npu-tp2-bf16-s30-l8-b64-seq128-seed61-runtime-r31-offline-5dd70c69](../../../runs/2-card/tp2/npu-tp2-bf16-s30-l8-b64-seq128-seed61-runtime-r31-offline-5dd70c69/readme.md) | completed | profiler-active | runtime | 5,912.71 ms | 692.74 | 1,385.49 | - GiB |
| [npu-tp2-bf16-s30-l8-b64-seq128-seed61-system-r31-262ee194](../../../runs/2-card/tp2/npu-tp2-bf16-s30-l8-b64-seq128-seed61-system-r31-262ee194/readme.md) | captured | profiler-active | system | - ms | - | - | - GiB |

## experiment sequence

### 1. npu-tp2-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-e698990c

- Status: `completed`
- Started: `2026-08-31T11:52:15.754222+08:00`
- Full process and outputs: [run readme](../../../runs/2-card/tp2/npu-tp2-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-e698990c/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/2-card/tp2/npu-tp2-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-e698990c.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology all --preset all --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all
```

### 2. npu-tp2-bf16-s30-l8-b64-seq128-seed61-distributed-r31-e4b07492

- Status: `completed`
- Started: `2026-08-31T17:19:46.269024+08:00`
- Full process and outputs: [run readme](../../../runs/2-card/tp2/npu-tp2-bf16-s30-l8-b64-seq128-seed61-distributed-r31-e4b07492/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/2-card/tp2/npu-tp2-bf16-s30-l8-b64-seq128-seed61-distributed-r31-e4b07492.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topologies fsdp2-tp4-ep8,ddp4,fsdp2,fsdp4,tp2,tp4,cp2,cp4,pp2,pp4,ep2,ep4,fsdp2-tp2 --preset distributed --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all
```

### 3. npu-tp2-bf16-s30-l8-b64-seq128-seed61-kernel-r31-offline-acf304e8

- Status: `completed`
- Started: `2026-08-31T21:22:12.830158+08:00`
- Full process and outputs: [run readme](../../../runs/2-card/tp2/npu-tp2-bf16-s30-l8-b64-seq128-seed61-kernel-r31-offline-acf304e8/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/2-card/tp2/npu-tp2-bf16-s30-l8-b64-seq128-seed61-kernel-r31-offline-acf304e8.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology all --preset kernel --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all
```

### 4. npu-tp2-bf16-s30-l8-b64-seq128-seed61-operator-r32-offline-e04c189e

- Status: `completed`
- Started: `2026-09-01T00:04:19.355554+08:00`
- Full process and outputs: [run readme](../../../runs/2-card/tp2/npu-tp2-bf16-s30-l8-b64-seq128-seed61-operator-r32-offline-e04c189e/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/2-card/tp2/npu-tp2-bf16-s30-l8-b64-seq128-seed61-operator-r32-offline-e04c189e.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --capture --device npu --topology tp2 --preset operator --visible-devices 0,1,2,3,4,5,6,7 --replicate 32 --analysis-tools none
```

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --analyze --device npu --topology tp2 --preset operator --visible-devices 0,1,2,3,4,5,6,7 --replicate 32 --analysis-tools all --parse-workers 8
```

### 5. npu-tp2-bf16-s30-l8-b64-seq128-seed61-memory-r31-offline-175f14b6

- Status: `completed`
- Started: `2026-09-01T09:45:04.294553+08:00`
- Full process and outputs: [run readme](../../../runs/2-card/tp2/npu-tp2-bf16-s30-l8-b64-seq128-seed61-memory-r31-offline-175f14b6/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/2-card/tp2/npu-tp2-bf16-s30-l8-b64-seq128-seed61-memory-r31-offline-175f14b6.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --capture --device npu --topology tp2 --preset memory --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools none
```

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --analyze --device npu --topology tp2 --preset memory --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all --parse-workers 8
```

### 6. npu-tp2-bf16-s30-l8-b64-seq128-seed61-flamegraph-r31-3b7f386d

- Status: `completed`
- Started: `2026-09-01T11:04:33.826936+08:00`
- Full process and outputs: [run readme](../../../runs/2-card/tp2/npu-tp2-bf16-s30-l8-b64-seq128-seed61-flamegraph-r31-3b7f386d/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/2-card/tp2/npu-tp2-bf16-s30-l8-b64-seq128-seed61-flamegraph-r31-3b7f386d.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --capture --device npu --topology tp2 --preset flamegraph --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools none
```

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --analyze --device npu --topology tp2 --preset flamegraph --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all --parse-workers 8
```

### 7. npu-tp2-bf16-s30-l8-b64-seq128-seed61-runtime-r31-offline-5dd70c69

- Status: `completed`
- Started: `2026-09-01T12:23:51.712921+08:00`
- Full process and outputs: [run readme](../../../runs/2-card/tp2/npu-tp2-bf16-s30-l8-b64-seq128-seed61-runtime-r31-offline-5dd70c69/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/2-card/tp2/npu-tp2-bf16-s30-l8-b64-seq128-seed61-runtime-r31-offline-5dd70c69.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --capture --device npu --topology tp2 --preset runtime --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools none
```

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --analyze --device npu --topology tp2 --preset runtime --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all --parse-workers 8
```

### 8. npu-tp2-bf16-s30-l8-b64-seq128-seed61-system-r31-262ee194

- Status: `captured`
- Started: `2026-09-01T13:49:51.136451+08:00`
- Full process and outputs: [run readme](../../../runs/2-card/tp2/npu-tp2-bf16-s30-l8-b64-seq128-seed61-system-r31-262ee194/readme.md)

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --capture --device npu --topology tp2 --preset system --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools none
```

## current summary

The highest recorded job throughput is 1,385.49 tok/s from `npu-tp2-bf16-s30-l8-b64-seq128-seed61-runtime-r31-offline-5dd70c69`. Profiler-off runs are throughput evidence; profiler-active runs are attribution evidence only.
