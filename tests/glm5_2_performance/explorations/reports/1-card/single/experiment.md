# single experiments

## topology

| field | value |
| --- | --- |
| world size | 1 |
| dp replicate / shard | 1 / 1 |
| tp / cp / pp / ep | 1 / 1 / 1 / 1 |
| recorded runs | 8 |

## run results

| run | status | mode | preset | median step | tok/s/device | tok/s/job | peak hbm |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| [npu-single-bf16-s30-l8-b64-seq128-seed61-overview-a7f99b62](../../../runs/1-card/single/npu-single-bf16-s30-l8-b64-seq128-seed61-overview-a7f99b62/readme.md) | command-only | profiler-active | - | - ms | - | - | - GiB |
| [npu-single-bf16-s30-l8-b64-seq128-seed61-overview-offline-bf438d2a](../../../runs/1-card/single/npu-single-bf16-s30-l8-b64-seq128-seed61-overview-offline-bf438d2a/readme.md) | command-only | profiler-active | - | - ms | - | - | - GiB |
| [npu-single-bf16-s30-l8-b64-seq128-seed61-kernel-offline-2499a8d1](../../../runs/1-card/single/npu-single-bf16-s30-l8-b64-seq128-seed61-kernel-offline-2499a8d1/readme.md) | command-only | profiler-active | - | - ms | - | - | - GiB |
| [npu-single-bf16-s30-l8-b64-seq128-seed61-overview-af49a08e](../../../runs/1-card/single/npu-single-bf16-s30-l8-b64-seq128-seed61-overview-af49a08e/readme.md) | completed | profiler-active | overview | 1,839.62 ms | 4,453.08 | 4,453.08 | - GiB |
| [npu-single-bf16-s30-l8-b64-seq128-seed61-overview-offline-b14bdd6a](../../../runs/1-card/single/npu-single-bf16-s30-l8-b64-seq128-seed61-overview-offline-b14bdd6a/readme.md) | completed | profiler-active | overview | 1,739.46 ms | 4,709.52 | 4,709.52 | - GiB |
| [npu-single-bf16-s30-l8-b64-seq128-seed61-kernel-offline-2ed8313b](../../../runs/1-card/single/npu-single-bf16-s30-l8-b64-seq128-seed61-kernel-offline-2ed8313b/readme.md) | completed | profiler-active | kernel | 1,747.83 ms | 4,686.94 | 4,686.94 | - GiB |
| [npu-single-bf16-s30-l8-b64-seq128-seed61-overview-r31-92543cf1](../../../runs/1-card/single/npu-single-bf16-s30-l8-b64-seq128-seed61-overview-r31-92543cf1/readme.md) | completed | profiler-active | overview | 2,853.23 ms | 2,871.14 | 2,871.14 | - GiB |
| [npu-single-bf16-s30-l8-b64-seq128-seed61-distributed-r31-a9358e97](../../../runs/1-card/single/npu-single-bf16-s30-l8-b64-seq128-seed61-distributed-r31-a9358e97/readme.md) | completed | profiler-active | distributed | 2,715.64 ms | 3,016.60 | 3,016.60 | - GiB |

## experiment sequence

### 1. npu-single-bf16-s30-l8-b64-seq128-seed61-overview-a7f99b62

- Status: `command-only`
- Started: `2026-08-24T16:31:51.795492+08:00`
- Full process and outputs: [run readme](../../../runs/1-card/single/npu-single-bf16-s30-l8-b64-seq128-seed61-overview-a7f99b62/readme.md)

```bash
/root/miniconda3/envs/torchtitan-0803/bin/python tests/glm5_2_performance/profiler_benchmark.py --analyze --device npu --topology single --preset overview --visible-devices 1
```

### 2. npu-single-bf16-s30-l8-b64-seq128-seed61-overview-offline-bf438d2a

- Status: `command-only`
- Started: `2026-08-24T16:31:57.043252+08:00`
- Full process and outputs: [run readme](../../../runs/1-card/single/npu-single-bf16-s30-l8-b64-seq128-seed61-overview-offline-bf438d2a/readme.md)

```bash
/root/miniconda3/envs/torchtitan-0803/bin/python tests/glm5_2_performance/profiler_benchmark.py --analyze --device npu --topology single --preset overview --offline --visible-devices 1
```

### 3. npu-single-bf16-s30-l8-b64-seq128-seed61-kernel-offline-2499a8d1

- Status: `command-only`
- Started: `2026-08-24T16:32:02.286244+08:00`
- Full process and outputs: [run readme](../../../runs/1-card/single/npu-single-bf16-s30-l8-b64-seq128-seed61-kernel-offline-2499a8d1/readme.md)

```bash
/root/miniconda3/envs/torchtitan-0803/bin/python tests/glm5_2_performance/profiler_benchmark.py --analyze --device npu --topology single --preset kernel --offline --visible-devices 1
```

### 4. npu-single-bf16-s30-l8-b64-seq128-seed61-overview-af49a08e

- Status: `completed`
- Started: `2026-08-24T17:13:47.115347+08:00`
- Full process and outputs: [run readme](../../../runs/1-card/single/npu-single-bf16-s30-l8-b64-seq128-seed61-overview-af49a08e/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/1-card/single/npu-single-bf16-s30-l8-b64-seq128-seed61-overview-af49a08e.html`

```bash
/root/miniconda3/envs/torchtitan-0803/bin/python tests/glm5_2_performance/profiler_benchmark.py --analyze --device npu --topology single --preset overview --visible-devices 1
```

### 5. npu-single-bf16-s30-l8-b64-seq128-seed61-overview-offline-b14bdd6a

- Status: `completed`
- Started: `2026-08-24T17:13:52.411250+08:00`
- Full process and outputs: [run readme](../../../runs/1-card/single/npu-single-bf16-s30-l8-b64-seq128-seed61-overview-offline-b14bdd6a/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/1-card/single/npu-single-bf16-s30-l8-b64-seq128-seed61-overview-offline-b14bdd6a.html`

```bash
/root/miniconda3/envs/torchtitan-0803/bin/python tests/glm5_2_performance/profiler_benchmark.py --analyze --device npu --topology single --preset overview --offline --visible-devices 1
```

### 6. npu-single-bf16-s30-l8-b64-seq128-seed61-kernel-offline-2ed8313b

- Status: `completed`
- Started: `2026-08-24T17:14:03.421139+08:00`
- Full process and outputs: [run readme](../../../runs/1-card/single/npu-single-bf16-s30-l8-b64-seq128-seed61-kernel-offline-2ed8313b/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/1-card/single/npu-single-bf16-s30-l8-b64-seq128-seed61-kernel-offline-2ed8313b.html`

```bash
/root/miniconda3/envs/torchtitan-0803/bin/python tests/glm5_2_performance/profiler_benchmark.py --analyze --device npu --topology single --preset kernel --offline --visible-devices 1
```

### 7. npu-single-bf16-s30-l8-b64-seq128-seed61-overview-r31-92543cf1

- Status: `completed`
- Started: `2026-08-31T09:52:36.985839+08:00`
- Full process and outputs: [run readme](../../../runs/1-card/single/npu-single-bf16-s30-l8-b64-seq128-seed61-overview-r31-92543cf1/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/1-card/single/npu-single-bf16-s30-l8-b64-seq128-seed61-overview-r31-92543cf1.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology single --preset overview --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all
```

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology single --preset overview --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all
```

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology all --preset all --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all
```

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology single --preset overview --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all
```

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology all --preset all --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all
```

### 8. npu-single-bf16-s30-l8-b64-seq128-seed61-distributed-r31-a9358e97

- Status: `completed`
- Started: `2026-08-31T12:58:12.500199+08:00`
- Full process and outputs: [run readme](../../../runs/1-card/single/npu-single-bf16-s30-l8-b64-seq128-seed61-distributed-r31-a9358e97/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/1-card/single/npu-single-bf16-s30-l8-b64-seq128-seed61-distributed-r31-a9358e97.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology all --preset all --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all
```

## current summary

The highest recorded job throughput is 4,709.52 tok/s from `npu-single-bf16-s30-l8-b64-seq128-seed61-overview-offline-b14bdd6a`. Profiler-off runs are throughput evidence; profiler-active runs are attribution evidence only.
