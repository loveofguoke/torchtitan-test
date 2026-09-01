# ddp8 experiments

## topology

| field | value |
| --- | --- |
| world size | 8 |
| dp replicate / shard | 8 / 1 |
| tp / cp / pp / ep | 1 / 1 / 1 / 1 |
| recorded runs | 13 |

## run results

| run | status | mode | preset | median step | tok/s/device | tok/s/job | peak hbm |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| [npu-ddp8-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-270760c1](../../../runs/8-card/ddp8/npu-ddp8-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-270760c1/readme.md) | failed | profiler-off | - | - ms | - | - | - GiB |
| [npu-ddp8-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r2-80606b65](../../../runs/8-card/ddp8/npu-ddp8-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r2-80606b65/readme.md) | completed | profiler-off | distributed | 773.35 ms | 1,370.09 | 10,960.70 | 0.783 GiB |
| [npu-ddp8-bf16-s20-l8-b64-seq128-seed61-distributed-r1-488247ce](../../../runs/8-card/ddp8/npu-ddp8-bf16-s20-l8-b64-seq128-seed61-distributed-r1-488247ce/readme.md) | completed | profiler-active | distributed | 219.14 ms | 4,672.84 | 37,382.71 | - GiB |
| [npu-ddp8-bf16-s30-l8-b64-seq128-seed61-overview-r31-0ce71019](../../../runs/8-card/ddp8/npu-ddp8-bf16-s30-l8-b64-seq128-seed61-overview-r31-0ce71019/readme.md) | completed | profiler-active | overview | 440.49 ms | 2,324.69 | 18,597.53 | - GiB |
| [npu-ddp8-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-18f2f254](../../../runs/8-card/ddp8/npu-ddp8-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-18f2f254/readme.md) | completed | profiler-active | overview | 406.12 ms | 2,521.42 | 20,171.38 | - GiB |
| [npu-ddp8-bf16-s30-l8-b64-seq128-seed61-distributed-r31-aabbbd06](../../../runs/8-card/ddp8/npu-ddp8-bf16-s30-l8-b64-seq128-seed61-distributed-r31-aabbbd06/readme.md) | completed | profiler-active | distributed | 393.30 ms | 2,603.59 | 20,828.69 | - GiB |
| [npu-ddp8-bf16-s30-l8-b64-seq128-seed61-kernel-r31-offline-ff71666e](../../../runs/8-card/ddp8/npu-ddp8-bf16-s30-l8-b64-seq128-seed61-kernel-r31-offline-ff71666e/readme.md) | completed | profiler-active | kernel | 429.69 ms | 2,383.11 | 19,064.90 | - GiB |
| [npu-ddp8-bf16-s30-l8-b64-seq128-seed61-operator-r32-offline-09cab892](../../../runs/8-card/ddp8/npu-ddp8-bf16-s30-l8-b64-seq128-seed61-operator-r32-offline-09cab892/readme.md) | completed | profiler-active | operator | 405.20 ms | 2,527.16 | 20,217.28 | - GiB |
| [npu-ddp8-bf16-s30-l8-b64-seq128-seed61-memory-r31-offline-63922994](../../../runs/8-card/ddp8/npu-ddp8-bf16-s30-l8-b64-seq128-seed61-memory-r31-offline-63922994/readme.md) | failed | profiler-active | - | - ms | - | - | - GiB |
| [npu-ddp8-bf16-s30-l8-b64-seq128-seed61-memory-r31-offline-b59d3541](../../../runs/8-card/ddp8/npu-ddp8-bf16-s30-l8-b64-seq128-seed61-memory-r31-offline-b59d3541/readme.md) | completed | profiler-active | memory | 398.02 ms | 2,572.73 | 20,581.83 | - GiB |
| [npu-ddp8-bf16-s30-l8-b64-seq128-seed61-flamegraph-r31-18f674c8](../../../runs/8-card/ddp8/npu-ddp8-bf16-s30-l8-b64-seq128-seed61-flamegraph-r31-18f674c8/readme.md) | completed | profiler-active | flamegraph | 404.64 ms | 2,530.65 | 20,245.17 | - GiB |
| [npu-ddp8-bf16-s30-l8-b64-seq128-seed61-runtime-r31-offline-7e60dbde](../../../runs/8-card/ddp8/npu-ddp8-bf16-s30-l8-b64-seq128-seed61-runtime-r31-offline-7e60dbde/readme.md) | completed | profiler-active | runtime | 430.39 ms | 2,379.22 | 19,033.76 | - GiB |
| [npu-ddp8-bf16-s30-l8-b64-seq128-seed61-system-r31-5ba774fd](../../../runs/8-card/ddp8/npu-ddp8-bf16-s30-l8-b64-seq128-seed61-system-r31-5ba774fd/readme.md) | completed | profiler-active | system | 397.06 ms | 2,578.94 | 20,631.55 | - GiB |

## experiment sequence

### 1. npu-ddp8-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-270760c1

- Status: `failed`
- Started: `2026-08-24T17:40:01.329634+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/ddp8/npu-ddp8-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-270760c1/readme.md)

```bash
/root/miniconda3/envs/torchtitan-0803/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology ddp8 --preset distributed --profiler-off --visible-devices 0,1,2,3,4,5,6,7 --steps 20 --skip-steps 8 --warmup-steps 2 --active-steps 3 --replicate 1
```

### 2. npu-ddp8-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r2-80606b65

- Status: `completed`
- Started: `2026-08-24T17:46:53.603664+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/ddp8/npu-ddp8-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r2-80606b65/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/ddp8/npu-ddp8-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r2-80606b65.html`

```bash
/root/miniconda3/envs/torchtitan-0803/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology ddp8 --preset distributed --profiler-off --visible-devices 0,1,2,3,4,5,6,7 --steps 20 --skip-steps 8 --warmup-steps 2 --active-steps 3 --replicate 2
```

### 3. npu-ddp8-bf16-s20-l8-b64-seq128-seed61-distributed-r1-488247ce

- Status: `completed`
- Started: `2026-08-24T18:12:10.707076+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/ddp8/npu-ddp8-bf16-s20-l8-b64-seq128-seed61-distributed-r1-488247ce/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/ddp8/npu-ddp8-bf16-s20-l8-b64-seq128-seed61-distributed-r1-488247ce.html`

```bash
/root/miniconda3/envs/torchtitan-0803/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology ddp8 --preset distributed --visible-devices 0,1,2,3,4,5,6,7 --steps 20 --skip-steps 8 --warmup-steps 2 --active-steps 3 --replicate 1 --cluster
```

### 4. npu-ddp8-bf16-s30-l8-b64-seq128-seed61-overview-r31-0ce71019

- Status: `completed`
- Started: `2026-08-31T10:07:47.459184+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/ddp8/npu-ddp8-bf16-s30-l8-b64-seq128-seed61-overview-r31-0ce71019/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/ddp8/npu-ddp8-bf16-s30-l8-b64-seq128-seed61-overview-r31-0ce71019.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology all --preset all --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all
```

### 5. npu-ddp8-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-18f2f254

- Status: `completed`
- Started: `2026-08-31T10:38:21.432514+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/ddp8/npu-ddp8-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-18f2f254/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/ddp8/npu-ddp8-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-18f2f254.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology all --preset all --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all
```

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology all --preset all --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all
```

### 6. npu-ddp8-bf16-s30-l8-b64-seq128-seed61-distributed-r31-aabbbd06

- Status: `completed`
- Started: `2026-08-31T13:11:06.534278+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/ddp8/npu-ddp8-bf16-s30-l8-b64-seq128-seed61-distributed-r31-aabbbd06/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/ddp8/npu-ddp8-bf16-s30-l8-b64-seq128-seed61-distributed-r31-aabbbd06.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology all --preset all --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all
```

### 7. npu-ddp8-bf16-s30-l8-b64-seq128-seed61-kernel-r31-offline-ff71666e

- Status: `completed`
- Started: `2026-08-31T19:53:06.511155+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/ddp8/npu-ddp8-bf16-s30-l8-b64-seq128-seed61-kernel-r31-offline-ff71666e/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/ddp8/npu-ddp8-bf16-s30-l8-b64-seq128-seed61-kernel-r31-offline-ff71666e.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology all --preset kernel --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all
```

### 8. npu-ddp8-bf16-s30-l8-b64-seq128-seed61-operator-r32-offline-09cab892

- Status: `completed`
- Started: `2026-08-31T23:05:19.053168+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/ddp8/npu-ddp8-bf16-s30-l8-b64-seq128-seed61-operator-r32-offline-09cab892/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/ddp8/npu-ddp8-bf16-s30-l8-b64-seq128-seed61-operator-r32-offline-09cab892.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology ddp8 --preset operator --visible-devices 0,1,2,3,4,5,6,7 --replicate 32 --analysis-tools all
```

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --analyze --device npu --topology ddp8 --preset operator --visible-devices 0,1,2,3,4,5,6,7 --replicate 32 --analysis-tools all --parse-workers 8
```

### 9. npu-ddp8-bf16-s30-l8-b64-seq128-seed61-memory-r31-offline-63922994

- Status: `failed`
- Started: `2026-09-01T00:40:21.165412+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/ddp8/npu-ddp8-bf16-s30-l8-b64-seq128-seed61-memory-r31-offline-63922994/readme.md)

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --capture --device npu --topology ddp8 --preset memory --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools none
```

### 10. npu-ddp8-bf16-s30-l8-b64-seq128-seed61-memory-r31-offline-b59d3541

- Status: `completed`
- Started: `2026-09-01T09:06:09.766405+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/ddp8/npu-ddp8-bf16-s30-l8-b64-seq128-seed61-memory-r31-offline-b59d3541/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/ddp8/npu-ddp8-bf16-s30-l8-b64-seq128-seed61-memory-r31-offline-b59d3541.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --capture --device npu --topology ddp8 --preset memory --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools none
```

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --analyze --device npu --topology ddp8 --preset memory --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all --parse-workers 8
```

### 11. npu-ddp8-bf16-s30-l8-b64-seq128-seed61-flamegraph-r31-18f674c8

- Status: `completed`
- Started: `2026-09-01T10:24:08.637403+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/ddp8/npu-ddp8-bf16-s30-l8-b64-seq128-seed61-flamegraph-r31-18f674c8/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/ddp8/npu-ddp8-bf16-s30-l8-b64-seq128-seed61-flamegraph-r31-18f674c8.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --capture --device npu --topology ddp8 --preset flamegraph --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools none
```

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --analyze --device npu --topology ddp8 --preset flamegraph --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all --parse-workers 8
```

### 12. npu-ddp8-bf16-s30-l8-b64-seq128-seed61-runtime-r31-offline-7e60dbde

- Status: `completed`
- Started: `2026-09-01T11:41:20.637927+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/ddp8/npu-ddp8-bf16-s30-l8-b64-seq128-seed61-runtime-r31-offline-7e60dbde/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/ddp8/npu-ddp8-bf16-s30-l8-b64-seq128-seed61-runtime-r31-offline-7e60dbde.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --capture --device npu --topology ddp8 --preset runtime --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools none
```

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --analyze --device npu --topology ddp8 --preset runtime --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all --parse-workers 8
```

### 13. npu-ddp8-bf16-s30-l8-b64-seq128-seed61-system-r31-5ba774fd

- Status: `completed`
- Started: `2026-09-01T13:09:50.495590+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/ddp8/npu-ddp8-bf16-s30-l8-b64-seq128-seed61-system-r31-5ba774fd/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/ddp8/npu-ddp8-bf16-s30-l8-b64-seq128-seed61-system-r31-5ba774fd.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --capture --device npu --topology ddp8 --preset system --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools none
```

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --analyze --device npu --topology ddp8 --preset system --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all --parse-workers 8
```

## current summary

The highest recorded job throughput is 37,382.71 tok/s from `npu-ddp8-bf16-s20-l8-b64-seq128-seed61-distributed-r1-488247ce`. Profiler-off runs are throughput evidence; profiler-active runs are attribution evidence only.

## analysis

- [Detailed topology analysis](analysis.md)
