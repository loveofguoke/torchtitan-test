# fsdp2-tp4 experiments

## topology

| field | value |
| --- | --- |
| world size | 8 |
| dp replicate / shard | 1 / 2 |
| tp / cp / pp / ep | 4 / 1 / 1 / 1 |
| recorded runs | 11 |

## run results

| run | status | mode | preset | median step | tok/s/device | tok/s/job | peak hbm |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| [npu-fsdp2-tp4-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-ecf6fd22](../../../runs/8-card/fsdp2-tp4/npu-fsdp2-tp4-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-ecf6fd22/readme.md) | completed | profiler-off | distributed | 2,231.55 ms | 458.88 | 3,671.06 | 0.257 GiB |
| [npu-fsdp2-tp4-bf16-s20-l8-b64-seq128-seed61-distributed-r1-754de573](../../../runs/8-card/fsdp2-tp4/npu-fsdp2-tp4-bf16-s20-l8-b64-seq128-seed61-distributed-r1-754de573/readme.md) | failed | profiler-active | - | - ms | - | - | - GiB |
| [npu-fsdp2-tp4-bf16-s20-l8-b64-seq128-seed61-distributed-r2-7088c3b5](../../../runs/8-card/fsdp2-tp4/npu-fsdp2-tp4-bf16-s20-l8-b64-seq128-seed61-distributed-r2-7088c3b5/readme.md) | completed | profiler-active | distributed | 2,365.60 ms | 432.87 | 3,462.97 | - GiB |
| [npu-fsdp2-tp4-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-bc797ece](../../../runs/8-card/fsdp2-tp4/npu-fsdp2-tp4-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-bc797ece/readme.md) | completed | profiler-active | overview | 3,088.21 ms | 331.58 | 2,652.67 | - GiB |
| [npu-fsdp2-tp4-bf16-s30-l8-b64-seq128-seed61-distributed-r31-309fa354](../../../runs/8-card/fsdp2-tp4/npu-fsdp2-tp4-bf16-s30-l8-b64-seq128-seed61-distributed-r31-309fa354/readme.md) | completed | profiler-active | distributed | 3,257.00 ms | 314.40 | 2,515.20 | - GiB |
| [npu-fsdp2-tp4-bf16-s30-l8-b64-seq128-seed61-kernel-r31-offline-e9dab65f](../../../runs/8-card/fsdp2-tp4/npu-fsdp2-tp4-bf16-s30-l8-b64-seq128-seed61-kernel-r31-offline-e9dab65f/readme.md) | completed | profiler-active | kernel | 3,233.54 ms | 316.68 | 2,533.45 | - GiB |
| [npu-fsdp2-tp4-bf16-s30-l8-b64-seq128-seed61-operator-r32-offline-d01c9a30](../../../runs/8-card/fsdp2-tp4/npu-fsdp2-tp4-bf16-s30-l8-b64-seq128-seed61-operator-r32-offline-d01c9a30/readme.md) | completed | profiler-active | operator | 3,239.96 ms | 316.05 | 2,528.42 | - GiB |
| [npu-fsdp2-tp4-bf16-s30-l8-b64-seq128-seed61-memory-r31-offline-120fd885](../../../runs/8-card/fsdp2-tp4/npu-fsdp2-tp4-bf16-s30-l8-b64-seq128-seed61-memory-r31-offline-120fd885/readme.md) | completed | profiler-active | memory | 3,089.38 ms | 331.46 | 2,651.67 | - GiB |
| [npu-fsdp2-tp4-bf16-s30-l8-b64-seq128-seed61-flamegraph-r31-8ffd32b1](../../../runs/8-card/fsdp2-tp4/npu-fsdp2-tp4-bf16-s30-l8-b64-seq128-seed61-flamegraph-r31-8ffd32b1/readme.md) | completed | profiler-active | flamegraph | 3,178.35 ms | 322.18 | 2,577.44 | - GiB |
| [npu-fsdp2-tp4-bf16-s30-l8-b64-seq128-seed61-runtime-r31-offline-0e0f9969](../../../runs/8-card/fsdp2-tp4/npu-fsdp2-tp4-bf16-s30-l8-b64-seq128-seed61-runtime-r31-offline-0e0f9969/readme.md) | completed | profiler-active | runtime | 3,072.23 ms | 333.31 | 2,666.47 | - GiB |
| [npu-fsdp2-tp4-bf16-s30-l8-b64-seq128-seed61-system-r31-8b5ec3dc](../../../runs/8-card/fsdp2-tp4/npu-fsdp2-tp4-bf16-s30-l8-b64-seq128-seed61-system-r31-8b5ec3dc/readme.md) | completed | profiler-active | system | 3,017.31 ms | 339.38 | 2,715.00 | - GiB |

## experiment sequence

### 1. npu-fsdp2-tp4-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-ecf6fd22

- Status: `completed`
- Started: `2026-08-24T17:57:46.382537+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/fsdp2-tp4/npu-fsdp2-tp4-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-ecf6fd22/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/fsdp2-tp4/npu-fsdp2-tp4-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-ecf6fd22.html`

```bash
/root/miniconda3/envs/torchtitan-0803/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology fsdp2-tp4 --preset distributed --profiler-off --visible-devices 0,1,2,3,4,5,6,7 --steps 20 --skip-steps 8 --warmup-steps 2 --active-steps 3 --replicate 1
```

### 2. npu-fsdp2-tp4-bf16-s20-l8-b64-seq128-seed61-distributed-r1-754de573

- Status: `failed`
- Started: `2026-08-25T09:14:54.761071+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/fsdp2-tp4/npu-fsdp2-tp4-bf16-s20-l8-b64-seq128-seed61-distributed-r1-754de573/readme.md)

```bash
/root/miniconda3/envs/torchtitan-0803/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology fsdp2-tp4 --preset distributed --visible-devices 0,1,2,3,4,5,6,7 --steps 20 --skip-steps 8 --warmup-steps 2 --active-steps 3 --replicate 1 --cluster
```

### 3. npu-fsdp2-tp4-bf16-s20-l8-b64-seq128-seed61-distributed-r2-7088c3b5

- Status: `completed`
- Started: `2026-08-25T12:26:31.693143+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/fsdp2-tp4/npu-fsdp2-tp4-bf16-s20-l8-b64-seq128-seed61-distributed-r2-7088c3b5/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/fsdp2-tp4/npu-fsdp2-tp4-bf16-s20-l8-b64-seq128-seed61-distributed-r2-7088c3b5.html`

```bash
/root/miniconda3/envs/torchtitan-0803/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topologies pp8,fsdp2-tp4,fsdp2-cp4,tp2-cp4,fsdp4-tp2,fsdp2-pp4,fsdp2-tp2-pp2,fsdp2-tp4-ep8 --preset distributed --visible-devices 0,1,2,3,4,5,6,7 --steps 20 --skip-steps 8 --warmup-steps 2 --active-steps 3 --replicate 2 --cluster
```

### 4. npu-fsdp2-tp4-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-bc797ece

- Status: `completed`
- Started: `2026-08-31T10:58:55.987301+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/fsdp2-tp4/npu-fsdp2-tp4-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-bc797ece/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/fsdp2-tp4/npu-fsdp2-tp4-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-bc797ece.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology all --preset all --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all
```

### 5. npu-fsdp2-tp4-bf16-s30-l8-b64-seq128-seed61-distributed-r31-309fa354

- Status: `completed`
- Started: `2026-08-31T14:31:07.241121+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/fsdp2-tp4/npu-fsdp2-tp4-bf16-s30-l8-b64-seq128-seed61-distributed-r31-309fa354/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/fsdp2-tp4/npu-fsdp2-tp4-bf16-s30-l8-b64-seq128-seed61-distributed-r31-309fa354.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology all --preset all --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all
```

### 6. npu-fsdp2-tp4-bf16-s30-l8-b64-seq128-seed61-kernel-r31-offline-e9dab65f

- Status: `completed`
- Started: `2026-08-31T20:22:06.795080+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/fsdp2-tp4/npu-fsdp2-tp4-bf16-s30-l8-b64-seq128-seed61-kernel-r31-offline-e9dab65f/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/fsdp2-tp4/npu-fsdp2-tp4-bf16-s30-l8-b64-seq128-seed61-kernel-r31-offline-e9dab65f.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology all --preset kernel --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all
```

### 7. npu-fsdp2-tp4-bf16-s30-l8-b64-seq128-seed61-operator-r32-offline-d01c9a30

- Status: `completed`
- Started: `2026-08-31T23:35:29.889491+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/fsdp2-tp4/npu-fsdp2-tp4-bf16-s30-l8-b64-seq128-seed61-operator-r32-offline-d01c9a30/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/fsdp2-tp4/npu-fsdp2-tp4-bf16-s30-l8-b64-seq128-seed61-operator-r32-offline-d01c9a30.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --capture --device npu --topology fsdp2-tp4 --preset operator --visible-devices 0,1,2,3,4,5,6,7 --replicate 32 --analysis-tools none
```

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --analyze --device npu --topology fsdp2-tp4 --preset operator --visible-devices 0,1,2,3,4,5,6,7 --replicate 32 --analysis-tools all --parse-workers 8
```

### 8. npu-fsdp2-tp4-bf16-s30-l8-b64-seq128-seed61-memory-r31-offline-120fd885

- Status: `completed`
- Started: `2026-09-01T09:18:14.014943+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/fsdp2-tp4/npu-fsdp2-tp4-bf16-s30-l8-b64-seq128-seed61-memory-r31-offline-120fd885/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/fsdp2-tp4/npu-fsdp2-tp4-bf16-s30-l8-b64-seq128-seed61-memory-r31-offline-120fd885.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --capture --device npu --topology fsdp2-tp4 --preset memory --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools none
```

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --analyze --device npu --topology fsdp2-tp4 --preset memory --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all --parse-workers 8
```

### 9. npu-fsdp2-tp4-bf16-s30-l8-b64-seq128-seed61-flamegraph-r31-8ffd32b1

- Status: `completed`
- Started: `2026-09-01T10:37:20.392381+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/fsdp2-tp4/npu-fsdp2-tp4-bf16-s30-l8-b64-seq128-seed61-flamegraph-r31-8ffd32b1/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/fsdp2-tp4/npu-fsdp2-tp4-bf16-s30-l8-b64-seq128-seed61-flamegraph-r31-8ffd32b1.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --capture --device npu --topology fsdp2-tp4 --preset flamegraph --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools none
```

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --analyze --device npu --topology fsdp2-tp4 --preset flamegraph --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all --parse-workers 8
```

### 10. npu-fsdp2-tp4-bf16-s30-l8-b64-seq128-seed61-runtime-r31-offline-0e0f9969

- Status: `completed`
- Started: `2026-09-01T11:55:21.179020+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/fsdp2-tp4/npu-fsdp2-tp4-bf16-s30-l8-b64-seq128-seed61-runtime-r31-offline-0e0f9969/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/fsdp2-tp4/npu-fsdp2-tp4-bf16-s30-l8-b64-seq128-seed61-runtime-r31-offline-0e0f9969.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --capture --device npu --topology fsdp2-tp4 --preset runtime --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools none
```

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --analyze --device npu --topology fsdp2-tp4 --preset runtime --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all --parse-workers 8
```

### 11. npu-fsdp2-tp4-bf16-s30-l8-b64-seq128-seed61-system-r31-8b5ec3dc

- Status: `completed`
- Started: `2026-09-01T13:22:33.589156+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/fsdp2-tp4/npu-fsdp2-tp4-bf16-s30-l8-b64-seq128-seed61-system-r31-8b5ec3dc/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/fsdp2-tp4/npu-fsdp2-tp4-bf16-s30-l8-b64-seq128-seed61-system-r31-8b5ec3dc.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --capture --device npu --topology fsdp2-tp4 --preset system --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools none
```

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --analyze --device npu --topology fsdp2-tp4 --preset system --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all --parse-workers 8
```

## current summary

The highest recorded job throughput is 3,671.06 tok/s from `npu-fsdp2-tp4-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-ecf6fd22`. Profiler-off runs are throughput evidence; profiler-active runs are attribution evidence only.
