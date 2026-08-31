# ddp2 experiments

## topology

| field | value |
| --- | --- |
| world size | 2 |
| dp replicate / shard | 2 / 1 |
| tp / cp / pp / ep | 1 / 1 / 1 / 1 |
| recorded runs | 14 |

## run results

| run | status | mode | preset | median step | tok/s/device | tok/s/job | peak hbm |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| [npu-ddp2-bf16-s30-l8-b64-seq128-seed61-overview-offline-efca0dfc](../../../runs/2-card/ddp2/npu-ddp2-bf16-s30-l8-b64-seq128-seed61-overview-offline-efca0dfc/readme.md) | command-only | profiler-active | - | - ms | - | - | - GiB |
| [npu-ddp2-bf16-s30-l8-b64-seq128-seed61-distributed-fe1e682c](../../../runs/2-card/ddp2/npu-ddp2-bf16-s30-l8-b64-seq128-seed61-distributed-fe1e682c/readme.md) | command-only | profiler-active | - | - ms | - | - | - GiB |
| [npu-ddp2-bf16-s30-l8-b64-seq128-seed61-distributed-reduce-bf16-33b5a0e5](../../../runs/2-card/ddp2/npu-ddp2-bf16-s30-l8-b64-seq128-seed61-distributed-reduce-bf16-33b5a0e5/readme.md) | completed | profiler-active | distributed | 735.31 ms | 5,570.42 | 11,140.84 | - GiB |
| [npu-ddp2-bf16-s30-l8-b64-seq128-seed61-distributed-profiler-off-5b768f16](../../../runs/2-card/ddp2/npu-ddp2-bf16-s30-l8-b64-seq128-seed61-distributed-profiler-off-5b768f16/readme.md) | command-only | profiler-off | - | - ms | - | - | - GiB |
| [npu-ddp2-bf16-s30-l8-b64-seq128-seed61-distributed-profiler-off-d0093fa5](../../../runs/2-card/ddp2/npu-ddp2-bf16-s30-l8-b64-seq128-seed61-distributed-profiler-off-d0093fa5/readme.md) | command-only | profiler-off | - | - ms | - | - | - GiB |
| [npu-ddp2-bf16-s30-l8-b64-seq128-seed61-distributed-reduce-bf16-profiler-off-29003e6c](../../../runs/2-card/ddp2/npu-ddp2-bf16-s30-l8-b64-seq128-seed61-distributed-reduce-bf16-profiler-off-29003e6c/readme.md) | command-only | profiler-off | - | - ms | - | - | - GiB |
| [npu-ddp2-bf16-s30-l8-b64-seq128-seed61-overview-offline-240cc2ba](../../../runs/2-card/ddp2/npu-ddp2-bf16-s30-l8-b64-seq128-seed61-overview-offline-240cc2ba/readme.md) | completed | profiler-active | overview | 720.41 ms | 5,685.67 | 11,371.35 | - GiB |
| [npu-ddp2-bf16-s30-l8-b64-seq128-seed61-distributed-23bade6d](../../../runs/2-card/ddp2/npu-ddp2-bf16-s30-l8-b64-seq128-seed61-distributed-23bade6d/readme.md) | completed | profiler-active | distributed | 741.67 ms | 5,522.66 | 11,045.31 | - GiB |
| [npu-ddp2-bf16-s30-l8-b64-seq128-seed61-distributed-profiler-off-e51e73ea](../../../runs/2-card/ddp2/npu-ddp2-bf16-s30-l8-b64-seq128-seed61-distributed-profiler-off-e51e73ea/readme.md) | completed | profiler-off | distributed | 1,443.12 ms | 2,838.38 | 5,676.77 | 0.843 GiB |
| [npu-ddp2-bf16-s30-l8-b64-seq128-seed61-distributed-profiler-off-225f5586](../../../runs/2-card/ddp2/npu-ddp2-bf16-s30-l8-b64-seq128-seed61-distributed-profiler-off-225f5586/readme.md) | completed | profiler-off | distributed | 907.99 ms | 4,524.68 | 9,049.35 | 0.843 GiB |
| [npu-ddp2-bf16-s30-l8-b64-seq128-seed61-distributed-reduce-bf16-profiler-off-051ff000](../../../runs/2-card/ddp2/npu-ddp2-bf16-s30-l8-b64-seq128-seed61-distributed-reduce-bf16-profiler-off-051ff000/readme.md) | completed | profiler-off | distributed | 729.20 ms | 5,617.09 | 11,234.19 | 0.843 GiB |
| [npu-ddp2-bf16-s30-l8-b64-seq128-seed61-overview-r31-fe910bf5](../../../runs/2-card/ddp2/npu-ddp2-bf16-s30-l8-b64-seq128-seed61-overview-r31-fe910bf5/readme.md) | completed | profiler-active | overview | 1,472.68 ms | 2,781.33 | 5,562.66 | - GiB |
| [npu-ddp2-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-63813547](../../../runs/2-card/ddp2/npu-ddp2-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-63813547/readme.md) | completed | profiler-active | overview | 1,513.76 ms | 2,705.85 | 5,411.71 | - GiB |
| [npu-ddp2-bf16-s30-l8-b64-seq128-seed61-distributed-r31-af31bb05](../../../runs/2-card/ddp2/npu-ddp2-bf16-s30-l8-b64-seq128-seed61-distributed-r31-af31bb05/readme.md) | completed | profiler-active | distributed | 1,447.67 ms | 2,829.37 | 5,658.74 | - GiB |

## experiment sequence

### 1. npu-ddp2-bf16-s30-l8-b64-seq128-seed61-overview-offline-efca0dfc

- Status: `command-only`
- Started: `2026-08-24T16:32:13.189441+08:00`
- Full process and outputs: [run readme](../../../runs/2-card/ddp2/npu-ddp2-bf16-s30-l8-b64-seq128-seed61-overview-offline-efca0dfc/readme.md)

```bash
/root/miniconda3/envs/torchtitan-0803/bin/python tests/glm5_2_performance/profiler_benchmark.py --analyze --device npu --topology ddp2 --preset overview --offline --visible-devices 1,2
```

### 2. npu-ddp2-bf16-s30-l8-b64-seq128-seed61-distributed-fe1e682c

- Status: `command-only`
- Started: `2026-08-24T16:32:16.011590+08:00`
- Full process and outputs: [run readme](../../../runs/2-card/ddp2/npu-ddp2-bf16-s30-l8-b64-seq128-seed61-distributed-fe1e682c/readme.md)

```bash
/root/miniconda3/envs/torchtitan-0803/bin/python tests/glm5_2_performance/profiler_benchmark.py --analyze --device npu --topology ddp2 --preset distributed --visible-devices 1,2
```

### 3. npu-ddp2-bf16-s30-l8-b64-seq128-seed61-distributed-reduce-bf16-33b5a0e5

- Status: `completed`
- Started: `2026-08-24T16:32:26.944800+08:00`
- Full process and outputs: [run readme](../../../runs/2-card/ddp2/npu-ddp2-bf16-s30-l8-b64-seq128-seed61-distributed-reduce-bf16-33b5a0e5/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/2-card/ddp2/npu-ddp2-bf16-s30-l8-b64-seq128-seed61-distributed-reduce-bf16-33b5a0e5.html`

```bash
/root/miniconda3/envs/torchtitan-0803/bin/python tests/glm5_2_performance/profiler_benchmark.py --analyze --device npu --topology ddp2 --preset distributed --mixed-precision-reduce bfloat16 --visible-devices 1,2
```

```bash
/root/miniconda3/envs/torchtitan-0803/bin/python tests/glm5_2_performance/profiler_benchmark.py --analyze --device npu --topology ddp2 --preset distributed --mixed-precision-reduce bfloat16 --visible-devices 1,2
```

### 4. npu-ddp2-bf16-s30-l8-b64-seq128-seed61-distributed-profiler-off-5b768f16

- Status: `command-only`
- Started: `2026-08-24T16:32:37.815253+08:00`
- Full process and outputs: [run readme](../../../runs/2-card/ddp2/npu-ddp2-bf16-s30-l8-b64-seq128-seed61-distributed-profiler-off-5b768f16/readme.md)

```bash
/root/miniconda3/envs/torchtitan-0803/bin/python tests/glm5_2_performance/profiler_benchmark.py --analyze --device npu --topology ddp2 --preset distributed --profiler-off --visible-devices 1,2
```

### 5. npu-ddp2-bf16-s30-l8-b64-seq128-seed61-distributed-profiler-off-d0093fa5

- Status: `command-only`
- Started: `2026-08-24T16:32:38.144235+08:00`
- Full process and outputs: [run readme](../../../runs/2-card/ddp2/npu-ddp2-bf16-s30-l8-b64-seq128-seed61-distributed-profiler-off-d0093fa5/readme.md)

```bash
/root/miniconda3/envs/torchtitan-0803/bin/python tests/glm5_2_performance/profiler_benchmark.py --analyze --device npu --topology ddp2 --preset distributed --profiler-off --extra-train-arg=--metrics.log_freq=1 --visible-devices 1,2
```

### 6. npu-ddp2-bf16-s30-l8-b64-seq128-seed61-distributed-reduce-bf16-profiler-off-29003e6c

- Status: `command-only`
- Started: `2026-08-24T16:32:38.519783+08:00`
- Full process and outputs: [run readme](../../../runs/2-card/ddp2/npu-ddp2-bf16-s30-l8-b64-seq128-seed61-distributed-reduce-bf16-profiler-off-29003e6c/readme.md)

```bash
/root/miniconda3/envs/torchtitan-0803/bin/python tests/glm5_2_performance/profiler_benchmark.py --analyze --device npu --topology ddp2 --preset distributed --profiler-off --mixed-precision-reduce bfloat16 --visible-devices 1,2
```

### 7. npu-ddp2-bf16-s30-l8-b64-seq128-seed61-overview-offline-240cc2ba

- Status: `completed`
- Started: `2026-08-24T17:14:06.268813+08:00`
- Full process and outputs: [run readme](../../../runs/2-card/ddp2/npu-ddp2-bf16-s30-l8-b64-seq128-seed61-overview-offline-240cc2ba/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/2-card/ddp2/npu-ddp2-bf16-s30-l8-b64-seq128-seed61-overview-offline-240cc2ba.html`

```bash
/root/miniconda3/envs/torchtitan-0803/bin/python tests/glm5_2_performance/profiler_benchmark.py --analyze --device npu --topology ddp2 --preset overview --offline --visible-devices 1,2
```

### 8. npu-ddp2-bf16-s30-l8-b64-seq128-seed61-distributed-23bade6d

- Status: `completed`
- Started: `2026-08-24T17:14:17.105664+08:00`
- Full process and outputs: [run readme](../../../runs/2-card/ddp2/npu-ddp2-bf16-s30-l8-b64-seq128-seed61-distributed-23bade6d/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/2-card/ddp2/npu-ddp2-bf16-s30-l8-b64-seq128-seed61-distributed-23bade6d.html`

```bash
/root/miniconda3/envs/torchtitan-0803/bin/python tests/glm5_2_performance/profiler_benchmark.py --analyze --device npu --topology ddp2 --preset distributed --visible-devices 1,2
```

### 9. npu-ddp2-bf16-s30-l8-b64-seq128-seed61-distributed-profiler-off-e51e73ea

- Status: `completed`
- Started: `2026-08-24T17:14:28.439011+08:00`
- Full process and outputs: [run readme](../../../runs/2-card/ddp2/npu-ddp2-bf16-s30-l8-b64-seq128-seed61-distributed-profiler-off-e51e73ea/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/2-card/ddp2/npu-ddp2-bf16-s30-l8-b64-seq128-seed61-distributed-profiler-off-e51e73ea.html`

```bash
/root/miniconda3/envs/torchtitan-0803/bin/python tests/glm5_2_performance/profiler_benchmark.py --analyze --device npu --topology ddp2 --preset distributed --profiler-off --visible-devices 1,2
```

### 10. npu-ddp2-bf16-s30-l8-b64-seq128-seed61-distributed-profiler-off-225f5586

- Status: `completed`
- Started: `2026-08-24T17:14:28.817261+08:00`
- Full process and outputs: [run readme](../../../runs/2-card/ddp2/npu-ddp2-bf16-s30-l8-b64-seq128-seed61-distributed-profiler-off-225f5586/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/2-card/ddp2/npu-ddp2-bf16-s30-l8-b64-seq128-seed61-distributed-profiler-off-225f5586.html`

```bash
/root/miniconda3/envs/torchtitan-0803/bin/python tests/glm5_2_performance/profiler_benchmark.py --analyze --device npu --topology ddp2 --preset distributed --profiler-off --extra-train-arg=--metrics.log_freq=1 --visible-devices 1,2
```

### 11. npu-ddp2-bf16-s30-l8-b64-seq128-seed61-distributed-reduce-bf16-profiler-off-051ff000

- Status: `completed`
- Started: `2026-08-24T17:14:29.208548+08:00`
- Full process and outputs: [run readme](../../../runs/2-card/ddp2/npu-ddp2-bf16-s30-l8-b64-seq128-seed61-distributed-reduce-bf16-profiler-off-051ff000/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/2-card/ddp2/npu-ddp2-bf16-s30-l8-b64-seq128-seed61-distributed-reduce-bf16-profiler-off-051ff000.html`

```bash
/root/miniconda3/envs/torchtitan-0803/bin/python tests/glm5_2_performance/profiler_benchmark.py --analyze --device npu --topology ddp2 --preset distributed --profiler-off --mixed-precision-reduce bfloat16 --visible-devices 1,2
```

### 12. npu-ddp2-bf16-s30-l8-b64-seq128-seed61-overview-r31-fe910bf5

- Status: `completed`
- Started: `2026-08-31T10:04:06.917176+08:00`
- Full process and outputs: [run readme](../../../runs/2-card/ddp2/npu-ddp2-bf16-s30-l8-b64-seq128-seed61-overview-r31-fe910bf5/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/2-card/ddp2/npu-ddp2-bf16-s30-l8-b64-seq128-seed61-overview-r31-fe910bf5.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology all --preset all --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all
```

### 13. npu-ddp2-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-63813547

- Status: `completed`
- Started: `2026-08-31T10:34:44.583972+08:00`
- Full process and outputs: [run readme](../../../runs/2-card/ddp2/npu-ddp2-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-63813547/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/2-card/ddp2/npu-ddp2-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-63813547.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology all --preset all --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all
```

### 14. npu-ddp2-bf16-s30-l8-b64-seq128-seed61-distributed-r31-af31bb05

- Status: `completed`
- Started: `2026-08-31T13:04:32.283188+08:00`
- Full process and outputs: [run readme](../../../runs/2-card/ddp2/npu-ddp2-bf16-s30-l8-b64-seq128-seed61-distributed-r31-af31bb05/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/2-card/ddp2/npu-ddp2-bf16-s30-l8-b64-seq128-seed61-distributed-r31-af31bb05.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology all --preset all --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all
```

## current summary

The highest recorded job throughput is 11,371.35 tok/s from `npu-ddp2-bf16-s30-l8-b64-seq128-seed61-overview-offline-240cc2ba`. Profiler-off runs are throughput evidence; profiler-active runs are attribution evidence only.
