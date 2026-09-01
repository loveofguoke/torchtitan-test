# ep2 experiments

## topology

| field | value |
| --- | --- |
| world size | 2 |
| dp replicate / shard | 1 / 2 |
| tp / cp / pp / ep | 1 / 1 / 1 / 2 |
| recorded runs | 8 |

## run results

| run | status | mode | preset | median step | tok/s/device | tok/s/job | peak hbm |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| [npu-ep2-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-2e297c09](../../../runs/2-card/ep2/npu-ep2-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-2e297c09/readme.md) | completed | profiler-active | overview | 1,932.64 ms | 2,119.38 | 4,238.77 | - GiB |
| [npu-ep2-bf16-s30-l8-b64-seq128-seed61-distributed-r31-3a451b46](../../../runs/2-card/ep2/npu-ep2-bf16-s30-l8-b64-seq128-seed61-distributed-r31-3a451b46/readme.md) | completed | profiler-active | distributed | 1,869.16 ms | 2,191.36 | 4,382.72 | - GiB |
| [npu-ep2-bf16-s30-l8-b64-seq128-seed61-kernel-r31-offline-fc087bf0](../../../runs/2-card/ep2/npu-ep2-bf16-s30-l8-b64-seq128-seed61-kernel-r31-offline-fc087bf0/readme.md) | completed | profiler-active | kernel | 1,830.38 ms | 2,237.78 | 4,475.56 | - GiB |
| [npu-ep2-bf16-s30-l8-b64-seq128-seed61-operator-r32-offline-54f354ff](../../../runs/2-card/ep2/npu-ep2-bf16-s30-l8-b64-seq128-seed61-operator-r32-offline-54f354ff/readme.md) | completed | profiler-active | operator | 1,830.58 ms | 2,237.54 | 4,475.07 | - GiB |
| [npu-ep2-bf16-s30-l8-b64-seq128-seed61-memory-r31-offline-0a778cb7](../../../runs/2-card/ep2/npu-ep2-bf16-s30-l8-b64-seq128-seed61-memory-r31-offline-0a778cb7/readme.md) | completed | profiler-active | memory | 1,849.62 ms | 2,214.51 | 4,429.03 | - GiB |
| [npu-ep2-bf16-s30-l8-b64-seq128-seed61-flamegraph-r31-e32f6515](../../../runs/2-card/ep2/npu-ep2-bf16-s30-l8-b64-seq128-seed61-flamegraph-r31-e32f6515/readme.md) | completed | profiler-active | flamegraph | 1,848.30 ms | 2,216.10 | 4,432.19 | - GiB |
| [npu-ep2-bf16-s30-l8-b64-seq128-seed61-runtime-r31-offline-f2b29d72](../../../runs/2-card/ep2/npu-ep2-bf16-s30-l8-b64-seq128-seed61-runtime-r31-offline-f2b29d72/readme.md) | completed | profiler-active | runtime | 1,814.60 ms | 2,257.25 | 4,514.50 | - GiB |
| [npu-ep2-bf16-s30-l8-b64-seq128-seed61-system-r31-687b6c2b](../../../runs/2-card/ep2/npu-ep2-bf16-s30-l8-b64-seq128-seed61-system-r31-687b6c2b/readme.md) | captured | profiler-active | system | - ms | - | - | - GiB |

## experiment sequence

### 1. npu-ep2-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-2e297c09

- Status: `completed`
- Started: `2026-08-31T12:47:28.179963+08:00`
- Full process and outputs: [run readme](../../../runs/2-card/ep2/npu-ep2-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-2e297c09/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/2-card/ep2/npu-ep2-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-2e297c09.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology all --preset all --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all
```

### 2. npu-ep2-bf16-s30-l8-b64-seq128-seed61-distributed-r31-3a451b46

- Status: `completed`
- Started: `2026-08-31T19:22:14.012652+08:00`
- Full process and outputs: [run readme](../../../runs/2-card/ep2/npu-ep2-bf16-s30-l8-b64-seq128-seed61-distributed-r31-3a451b46/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/2-card/ep2/npu-ep2-bf16-s30-l8-b64-seq128-seed61-distributed-r31-3a451b46.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topologies fsdp2-tp4-ep8,ddp4,fsdp2,fsdp4,tp2,tp4,cp2,cp4,pp2,pp4,ep2,ep4,fsdp2-tp2 --preset distributed --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all
```

### 3. npu-ep2-bf16-s30-l8-b64-seq128-seed61-kernel-r31-offline-fc087bf0

- Status: `completed`
- Started: `2026-08-31T22:27:32.673138+08:00`
- Full process and outputs: [run readme](../../../runs/2-card/ep2/npu-ep2-bf16-s30-l8-b64-seq128-seed61-kernel-r31-offline-fc087bf0/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/2-card/ep2/npu-ep2-bf16-s30-l8-b64-seq128-seed61-kernel-r31-offline-fc087bf0.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology all --preset kernel --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all
```

### 4. npu-ep2-bf16-s30-l8-b64-seq128-seed61-operator-r32-offline-54f354ff

- Status: `completed`
- Started: `2026-09-01T00:29:47.438214+08:00`
- Full process and outputs: [run readme](../../../runs/2-card/ep2/npu-ep2-bf16-s30-l8-b64-seq128-seed61-operator-r32-offline-54f354ff/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/2-card/ep2/npu-ep2-bf16-s30-l8-b64-seq128-seed61-operator-r32-offline-54f354ff.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --capture --device npu --topology ep2 --preset operator --visible-devices 0,1,2,3,4,5,6,7 --replicate 32 --analysis-tools none
```

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --analyze --device npu --topology ep2 --preset operator --visible-devices 0,1,2,3,4,5,6,7 --replicate 32 --analysis-tools all --parse-workers 8
```

### 5. npu-ep2-bf16-s30-l8-b64-seq128-seed61-memory-r31-offline-0a778cb7

- Status: `completed`
- Started: `2026-09-01T10:08:09.056532+08:00`
- Full process and outputs: [run readme](../../../runs/2-card/ep2/npu-ep2-bf16-s30-l8-b64-seq128-seed61-memory-r31-offline-0a778cb7/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/2-card/ep2/npu-ep2-bf16-s30-l8-b64-seq128-seed61-memory-r31-offline-0a778cb7.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --capture --device npu --topology ep2 --preset memory --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools none
```

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --analyze --device npu --topology ep2 --preset memory --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all --parse-workers 8
```

### 6. npu-ep2-bf16-s30-l8-b64-seq128-seed61-flamegraph-r31-e32f6515

- Status: `completed`
- Started: `2026-09-01T11:26:57.746847+08:00`
- Full process and outputs: [run readme](../../../runs/2-card/ep2/npu-ep2-bf16-s30-l8-b64-seq128-seed61-flamegraph-r31-e32f6515/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/2-card/ep2/npu-ep2-bf16-s30-l8-b64-seq128-seed61-flamegraph-r31-e32f6515.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --capture --device npu --topology ep2 --preset flamegraph --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools none
```

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --analyze --device npu --topology ep2 --preset flamegraph --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all --parse-workers 8
```

### 7. npu-ep2-bf16-s30-l8-b64-seq128-seed61-runtime-r31-offline-f2b29d72

- Status: `completed`
- Started: `2026-09-01T12:49:49.884396+08:00`
- Full process and outputs: [run readme](../../../runs/2-card/ep2/npu-ep2-bf16-s30-l8-b64-seq128-seed61-runtime-r31-offline-f2b29d72/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/2-card/ep2/npu-ep2-bf16-s30-l8-b64-seq128-seed61-runtime-r31-offline-f2b29d72.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --capture --device npu --topology ep2 --preset runtime --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools none
```

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --analyze --device npu --topology ep2 --preset runtime --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all --parse-workers 8
```

### 8. npu-ep2-bf16-s30-l8-b64-seq128-seed61-system-r31-687b6c2b

- Status: `captured`
- Started: `2026-09-01T14:13:48.790588+08:00`
- Full process and outputs: [run readme](../../../runs/2-card/ep2/npu-ep2-bf16-s30-l8-b64-seq128-seed61-system-r31-687b6c2b/readme.md)

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --capture --device npu --topology ep2 --preset system --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools none
```

## current summary

The highest recorded job throughput is 4,514.50 tok/s from `npu-ep2-bf16-s30-l8-b64-seq128-seed61-runtime-r31-offline-f2b29d72`. Profiler-off runs are throughput evidence; profiler-active runs are attribution evidence only.
