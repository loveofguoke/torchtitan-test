# cp4 experiments

## topology

| field | value |
| --- | --- |
| world size | 4 |
| dp replicate / shard | 1 / 1 |
| tp / cp / pp / ep | 1 / 4 / 1 / 1 |
| recorded runs | 9 |

## run results

| run | status | mode | preset | median step | tok/s/device | tok/s/job | peak hbm |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| [npu-cp4-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-f83c9d6c](../../../runs/4-card/cp4/npu-cp4-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-f83c9d6c/readme.md) | completed | profiler-off | distributed | 1,615.86 ms | 1,267.44 | 5,069.75 | 0.232 GiB |
| [npu-cp4-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-bf78e0a0](../../../runs/4-card/cp4/npu-cp4-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-bf78e0a0/readme.md) | completed | profiler-active | overview | 2,829.84 ms | 723.72 | 2,894.86 | - GiB |
| [npu-cp4-bf16-s30-l8-b64-seq128-seed61-distributed-r31-d4688ab3](../../../runs/4-card/cp4/npu-cp4-bf16-s30-l8-b64-seq128-seed61-distributed-r31-d4688ab3/readme.md) | completed | profiler-active | distributed | 2,818.15 ms | 726.72 | 2,906.88 | - GiB |
| [npu-cp4-bf16-s30-l8-b64-seq128-seed61-kernel-r31-offline-18bbc004](../../../runs/4-card/cp4/npu-cp4-bf16-s30-l8-b64-seq128-seed61-kernel-r31-offline-18bbc004/readme.md) | completed | profiler-active | kernel | 2,718.08 ms | 753.47 | 3,013.89 | - GiB |
| [npu-cp4-bf16-s30-l8-b64-seq128-seed61-operator-r32-offline-2c3ab07f](../../../runs/4-card/cp4/npu-cp4-bf16-s30-l8-b64-seq128-seed61-operator-r32-offline-2c3ab07f/readme.md) | completed | profiler-active | operator | 2,692.21 ms | 760.71 | 3,042.85 | - GiB |
| [npu-cp4-bf16-s30-l8-b64-seq128-seed61-memory-r31-offline-ee5fecfe](../../../runs/4-card/cp4/npu-cp4-bf16-s30-l8-b64-seq128-seed61-memory-r31-offline-ee5fecfe/readme.md) | completed | profiler-active | memory | 2,697.56 ms | 759.20 | 3,036.82 | - GiB |
| [npu-cp4-bf16-s30-l8-b64-seq128-seed61-flamegraph-r31-84957da5](../../../runs/4-card/cp4/npu-cp4-bf16-s30-l8-b64-seq128-seed61-flamegraph-r31-84957da5/readme.md) | completed | profiler-active | flamegraph | 2,665.51 ms | 768.33 | 3,073.33 | - GiB |
| [npu-cp4-bf16-s30-l8-b64-seq128-seed61-runtime-r31-offline-7d349e1b](../../../runs/4-card/cp4/npu-cp4-bf16-s30-l8-b64-seq128-seed61-runtime-r31-offline-7d349e1b/readme.md) | completed | profiler-active | runtime | 2,660.79 ms | 769.70 | 3,078.78 | - GiB |
| [npu-cp4-bf16-s30-l8-b64-seq128-seed61-system-r31-069b8a0b](../../../runs/4-card/cp4/npu-cp4-bf16-s30-l8-b64-seq128-seed61-system-r31-069b8a0b/readme.md) | captured | profiler-active | system | - ms | - | - | - GiB |

## experiment sequence

### 1. npu-cp4-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-f83c9d6c

- Status: `completed`
- Started: `2026-08-24T16:38:04.728711+08:00`
- Full process and outputs: [run readme](../../../runs/4-card/cp4/npu-cp4-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-f83c9d6c/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/4-card/cp4/npu-cp4-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-f83c9d6c.html`

```bash
/root/miniconda3/envs/torchtitan-0803/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology cp4 --preset distributed --profiler-off --visible-devices 1,2,3,4 --steps 20 --skip-steps 8 --warmup-steps 2 --active-steps 3 --replicate 1
```

### 2. npu-cp4-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-bf78e0a0

- Status: `completed`
- Started: `2026-08-31T12:16:07.249115+08:00`
- Full process and outputs: [run readme](../../../runs/4-card/cp4/npu-cp4-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-bf78e0a0/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/4-card/cp4/npu-cp4-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-bf78e0a0.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology all --preset all --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all
```

### 3. npu-cp4-bf16-s30-l8-b64-seq128-seed61-distributed-r31-d4688ab3

- Status: `completed`
- Started: `2026-08-31T18:04:13.936828+08:00`
- Full process and outputs: [run readme](../../../runs/4-card/cp4/npu-cp4-bf16-s30-l8-b64-seq128-seed61-distributed-r31-d4688ab3/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/4-card/cp4/npu-cp4-bf16-s30-l8-b64-seq128-seed61-distributed-r31-d4688ab3.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topologies fsdp2-tp4-ep8,ddp4,fsdp2,fsdp4,tp2,tp4,cp2,cp4,pp2,pp4,ep2,ep4,fsdp2-tp2 --preset distributed --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all
```

### 4. npu-cp4-bf16-s30-l8-b64-seq128-seed61-kernel-r31-offline-18bbc004

- Status: `completed`
- Started: `2026-08-31T21:48:59.397740+08:00`
- Full process and outputs: [run readme](../../../runs/4-card/cp4/npu-cp4-bf16-s30-l8-b64-seq128-seed61-kernel-r31-offline-18bbc004/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/4-card/cp4/npu-cp4-bf16-s30-l8-b64-seq128-seed61-kernel-r31-offline-18bbc004.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology all --preset kernel --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all
```

### 5. npu-cp4-bf16-s30-l8-b64-seq128-seed61-operator-r32-offline-2c3ab07f

- Status: `completed`
- Started: `2026-09-01T00:15:44.197492+08:00`
- Full process and outputs: [run readme](../../../runs/4-card/cp4/npu-cp4-bf16-s30-l8-b64-seq128-seed61-operator-r32-offline-2c3ab07f/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/4-card/cp4/npu-cp4-bf16-s30-l8-b64-seq128-seed61-operator-r32-offline-2c3ab07f.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --capture --device npu --topology cp4 --preset operator --visible-devices 0,1,2,3,4,5,6,7 --replicate 32 --analysis-tools none
```

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --analyze --device npu --topology cp4 --preset operator --visible-devices 0,1,2,3,4,5,6,7 --replicate 32 --analysis-tools all --parse-workers 8
```

### 6. npu-cp4-bf16-s30-l8-b64-seq128-seed61-memory-r31-offline-ee5fecfe

- Status: `completed`
- Started: `2026-09-01T09:55:29.050850+08:00`
- Full process and outputs: [run readme](../../../runs/4-card/cp4/npu-cp4-bf16-s30-l8-b64-seq128-seed61-memory-r31-offline-ee5fecfe/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/4-card/cp4/npu-cp4-bf16-s30-l8-b64-seq128-seed61-memory-r31-offline-ee5fecfe.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --capture --device npu --topology cp4 --preset memory --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools none
```

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --analyze --device npu --topology cp4 --preset memory --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all --parse-workers 8
```

### 7. npu-cp4-bf16-s30-l8-b64-seq128-seed61-flamegraph-r31-84957da5

- Status: `completed`
- Started: `2026-09-01T11:14:31.352341+08:00`
- Full process and outputs: [run readme](../../../runs/4-card/cp4/npu-cp4-bf16-s30-l8-b64-seq128-seed61-flamegraph-r31-84957da5/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/4-card/cp4/npu-cp4-bf16-s30-l8-b64-seq128-seed61-flamegraph-r31-84957da5.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --capture --device npu --topology cp4 --preset flamegraph --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools none
```

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --analyze --device npu --topology cp4 --preset flamegraph --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all --parse-workers 8
```

### 8. npu-cp4-bf16-s30-l8-b64-seq128-seed61-runtime-r31-offline-7d349e1b

- Status: `completed`
- Started: `2026-09-01T12:35:25.996868+08:00`
- Full process and outputs: [run readme](../../../runs/4-card/cp4/npu-cp4-bf16-s30-l8-b64-seq128-seed61-runtime-r31-offline-7d349e1b/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/4-card/cp4/npu-cp4-bf16-s30-l8-b64-seq128-seed61-runtime-r31-offline-7d349e1b.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --capture --device npu --topology cp4 --preset runtime --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools none
```

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --analyze --device npu --topology cp4 --preset runtime --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all --parse-workers 8
```

### 9. npu-cp4-bf16-s30-l8-b64-seq128-seed61-system-r31-069b8a0b

- Status: `captured`
- Started: `2026-09-01T14:00:12.511097+08:00`
- Full process and outputs: [run readme](../../../runs/4-card/cp4/npu-cp4-bf16-s30-l8-b64-seq128-seed61-system-r31-069b8a0b/readme.md)

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --capture --device npu --topology cp4 --preset system --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools none
```

## current summary

The highest recorded job throughput is 5,069.75 tok/s from `npu-cp4-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-f83c9d6c`. Profiler-off runs are throughput evidence; profiler-active runs are attribution evidence only.
