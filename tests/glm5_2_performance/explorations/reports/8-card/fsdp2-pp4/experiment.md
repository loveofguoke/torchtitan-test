# fsdp2-pp4 experiments

## topology

| field | value |
| --- | --- |
| world size | 8 |
| dp replicate / shard | 1 / 2 |
| tp / cp / pp / ep | 1 / 1 / 4 / 1 |
| recorded runs | 11 |

## run results

| run | status | mode | preset | median step | tok/s/device | tok/s/job | peak hbm |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| [npu-fsdp2-pp4-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-7900f287](../../../runs/8-card/fsdp2-pp4/npu-fsdp2-pp4-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-7900f287/readme.md) | completed | profiler-off | distributed | 1,744.04 ms | 587.21 | 4,697.69 | 0.044 GiB |
| [npu-fsdp2-pp4-bf16-s20-l8-b64-seq128-seed61-distributed-r1-283c59b2](../../../runs/8-card/fsdp2-pp4/npu-fsdp2-pp4-bf16-s20-l8-b64-seq128-seed61-distributed-r1-283c59b2/readme.md) | failed | profiler-active | - | - ms | - | - | - GiB |
| [npu-fsdp2-pp4-bf16-s20-l8-b64-seq128-seed61-distributed-r2-dde8b599](../../../runs/8-card/fsdp2-pp4/npu-fsdp2-pp4-bf16-s20-l8-b64-seq128-seed61-distributed-r2-dde8b599/readme.md) | completed | profiler-active | distributed | 2,316.39 ms | 442.07 | 3,536.53 | - GiB |
| [npu-fsdp2-pp4-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-7fbfaed3](../../../runs/8-card/fsdp2-pp4/npu-fsdp2-pp4-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-7fbfaed3/readme.md) | completed | profiler-active | overview | 3,270.23 ms | 313.13 | 2,505.02 | - GiB |
| [npu-fsdp2-pp4-bf16-s30-l8-b64-seq128-seed61-distributed-r31-eee8f86a](../../../runs/8-card/fsdp2-pp4/npu-fsdp2-pp4-bf16-s30-l8-b64-seq128-seed61-distributed-r31-eee8f86a/readme.md) | completed | profiler-active | distributed | 3,255.81 ms | 314.51 | 2,516.12 | - GiB |
| [npu-fsdp2-pp4-bf16-s30-l8-b64-seq128-seed61-kernel-r31-offline-78f91db6](../../../runs/8-card/fsdp2-pp4/npu-fsdp2-pp4-bf16-s30-l8-b64-seq128-seed61-kernel-r31-offline-78f91db6/readme.md) | completed | profiler-active | kernel | 3,385.31 ms | 302.48 | 2,419.87 | - GiB |
| [npu-fsdp2-pp4-bf16-s30-l8-b64-seq128-seed61-operator-r32-offline-eacd46fd](../../../runs/8-card/fsdp2-pp4/npu-fsdp2-pp4-bf16-s30-l8-b64-seq128-seed61-operator-r32-offline-eacd46fd/readme.md) | completed | profiler-active | operator | 3,292.23 ms | 311.04 | 2,488.28 | - GiB |
| [npu-fsdp2-pp4-bf16-s30-l8-b64-seq128-seed61-memory-r31-offline-a9877a31](../../../runs/8-card/fsdp2-pp4/npu-fsdp2-pp4-bf16-s30-l8-b64-seq128-seed61-memory-r31-offline-a9877a31/readme.md) | completed | profiler-active | memory | 3,276.60 ms | 312.52 | 2,500.15 | - GiB |
| [npu-fsdp2-pp4-bf16-s30-l8-b64-seq128-seed61-flamegraph-r31-313d4269](../../../runs/8-card/fsdp2-pp4/npu-fsdp2-pp4-bf16-s30-l8-b64-seq128-seed61-flamegraph-r31-313d4269/readme.md) | completed | profiler-active | flamegraph | 3,280.58 ms | 312.14 | 2,497.12 | - GiB |
| [npu-fsdp2-pp4-bf16-s30-l8-b64-seq128-seed61-runtime-r31-offline-9278dac8](../../../runs/8-card/fsdp2-pp4/npu-fsdp2-pp4-bf16-s30-l8-b64-seq128-seed61-runtime-r31-offline-9278dac8/readme.md) | completed | profiler-active | runtime | 3,239.81 ms | 316.07 | 2,528.55 | - GiB |
| [npu-fsdp2-pp4-bf16-s30-l8-b64-seq128-seed61-system-r31-5f9ec8ed](../../../runs/8-card/fsdp2-pp4/npu-fsdp2-pp4-bf16-s30-l8-b64-seq128-seed61-system-r31-5f9ec8ed/readme.md) | completed | profiler-active | system | 3,248.10 ms | 315.26 | 2,522.09 | - GiB |

## experiment sequence

### 1. npu-fsdp2-pp4-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-7900f287

- Status: `completed`
- Started: `2026-08-24T18:04:03.260384+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/fsdp2-pp4/npu-fsdp2-pp4-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-7900f287/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/fsdp2-pp4/npu-fsdp2-pp4-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-7900f287.html`

```bash
/root/miniconda3/envs/torchtitan-0803/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology fsdp2-pp4 --preset distributed --profiler-off --visible-devices 0,1,2,3,4,5,6,7 --steps 20 --skip-steps 8 --warmup-steps 2 --active-steps 3 --replicate 1
```

### 2. npu-fsdp2-pp4-bf16-s20-l8-b64-seq128-seed61-distributed-r1-283c59b2

- Status: `failed`
- Started: `2026-08-25T09:22:13.454994+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/fsdp2-pp4/npu-fsdp2-pp4-bf16-s20-l8-b64-seq128-seed61-distributed-r1-283c59b2/readme.md)

```bash
/root/miniconda3/envs/torchtitan-0803/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology fsdp2-pp4 --preset distributed --visible-devices 0,1,2,3,4,5,6,7 --steps 20 --skip-steps 8 --warmup-steps 2 --active-steps 3 --replicate 1 --cluster
```

### 3. npu-fsdp2-pp4-bf16-s20-l8-b64-seq128-seed61-distributed-r2-dde8b599

- Status: `completed`
- Started: `2026-08-25T13:29:01.985723+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/fsdp2-pp4/npu-fsdp2-pp4-bf16-s20-l8-b64-seq128-seed61-distributed-r2-dde8b599/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/fsdp2-pp4/npu-fsdp2-pp4-bf16-s20-l8-b64-seq128-seed61-distributed-r2-dde8b599.html`

```bash
/root/miniconda3/envs/torchtitan-0803/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topologies pp8,fsdp2-tp4,fsdp2-cp4,tp2-cp4,fsdp4-tp2,fsdp2-pp4,fsdp2-tp2-pp2,fsdp2-tp4-ep8 --preset distributed --visible-devices 0,1,2,3,4,5,6,7 --steps 20 --skip-steps 8 --warmup-steps 2 --active-steps 3 --replicate 2 --cluster
```

### 4. npu-fsdp2-pp4-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-7fbfaed3

- Status: `completed`
- Started: `2026-08-31T11:19:22.598708+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/fsdp2-pp4/npu-fsdp2-pp4-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-7fbfaed3/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/fsdp2-pp4/npu-fsdp2-pp4-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-7fbfaed3.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology all --preset all --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all
```

### 5. npu-fsdp2-pp4-bf16-s30-l8-b64-seq128-seed61-distributed-r31-eee8f86a

- Status: `completed`
- Started: `2026-08-31T15:27:41.784308+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/fsdp2-pp4/npu-fsdp2-pp4-bf16-s30-l8-b64-seq128-seed61-distributed-r31-eee8f86a/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/fsdp2-pp4/npu-fsdp2-pp4-bf16-s30-l8-b64-seq128-seed61-distributed-r31-eee8f86a.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology all --preset all --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all
```

### 6. npu-fsdp2-pp4-bf16-s30-l8-b64-seq128-seed61-kernel-r31-offline-78f91db6

- Status: `completed`
- Started: `2026-08-31T20:45:42.061409+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/fsdp2-pp4/npu-fsdp2-pp4-bf16-s30-l8-b64-seq128-seed61-kernel-r31-offline-78f91db6/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/fsdp2-pp4/npu-fsdp2-pp4-bf16-s30-l8-b64-seq128-seed61-kernel-r31-offline-78f91db6.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology all --preset kernel --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all
```

### 7. npu-fsdp2-pp4-bf16-s30-l8-b64-seq128-seed61-operator-r32-offline-eacd46fd

- Status: `completed`
- Started: `2026-08-31T23:46:35.822606+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/fsdp2-pp4/npu-fsdp2-pp4-bf16-s30-l8-b64-seq128-seed61-operator-r32-offline-eacd46fd/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/fsdp2-pp4/npu-fsdp2-pp4-bf16-s30-l8-b64-seq128-seed61-operator-r32-offline-eacd46fd.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --capture --device npu --topology fsdp2-pp4 --preset operator --visible-devices 0,1,2,3,4,5,6,7 --replicate 32 --analysis-tools none
```

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --analyze --device npu --topology fsdp2-pp4 --preset operator --visible-devices 0,1,2,3,4,5,6,7 --replicate 32 --analysis-tools all --parse-workers 8
```

### 8. npu-fsdp2-pp4-bf16-s30-l8-b64-seq128-seed61-memory-r31-offline-a9877a31

- Status: `completed`
- Started: `2026-09-01T09:28:04.875701+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/fsdp2-pp4/npu-fsdp2-pp4-bf16-s30-l8-b64-seq128-seed61-memory-r31-offline-a9877a31/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/fsdp2-pp4/npu-fsdp2-pp4-bf16-s30-l8-b64-seq128-seed61-memory-r31-offline-a9877a31.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --capture --device npu --topology fsdp2-pp4 --preset memory --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools none
```

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --analyze --device npu --topology fsdp2-pp4 --preset memory --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all --parse-workers 8
```

### 9. npu-fsdp2-pp4-bf16-s30-l8-b64-seq128-seed61-flamegraph-r31-313d4269

- Status: `completed`
- Started: `2026-09-01T10:47:47.057712+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/fsdp2-pp4/npu-fsdp2-pp4-bf16-s30-l8-b64-seq128-seed61-flamegraph-r31-313d4269/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/fsdp2-pp4/npu-fsdp2-pp4-bf16-s30-l8-b64-seq128-seed61-flamegraph-r31-313d4269.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --capture --device npu --topology fsdp2-pp4 --preset flamegraph --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools none
```

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --analyze --device npu --topology fsdp2-pp4 --preset flamegraph --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all --parse-workers 8
```

### 10. npu-fsdp2-pp4-bf16-s30-l8-b64-seq128-seed61-runtime-r31-offline-9278dac8

- Status: `completed`
- Started: `2026-09-01T12:06:06.825329+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/fsdp2-pp4/npu-fsdp2-pp4-bf16-s30-l8-b64-seq128-seed61-runtime-r31-offline-9278dac8/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/fsdp2-pp4/npu-fsdp2-pp4-bf16-s30-l8-b64-seq128-seed61-runtime-r31-offline-9278dac8.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --capture --device npu --topology fsdp2-pp4 --preset runtime --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools none
```

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --analyze --device npu --topology fsdp2-pp4 --preset runtime --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all --parse-workers 8
```

### 11. npu-fsdp2-pp4-bf16-s30-l8-b64-seq128-seed61-system-r31-5f9ec8ed

- Status: `completed`
- Started: `2026-09-01T13:32:37.255259+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/fsdp2-pp4/npu-fsdp2-pp4-bf16-s30-l8-b64-seq128-seed61-system-r31-5f9ec8ed/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/fsdp2-pp4/npu-fsdp2-pp4-bf16-s30-l8-b64-seq128-seed61-system-r31-5f9ec8ed.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --capture --device npu --topology fsdp2-pp4 --preset system --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools none
```

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --analyze --device npu --topology fsdp2-pp4 --preset system --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all --parse-workers 8
```

## current summary

The highest recorded job throughput is 4,697.69 tok/s from `npu-fsdp2-pp4-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-7900f287`. Profiler-off runs are throughput evidence; profiler-active runs are attribution evidence only.
