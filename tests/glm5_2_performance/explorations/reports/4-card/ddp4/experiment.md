# ddp4 experiments

## topology

| field | value |
| --- | --- |
| world size | 4 |
| dp replicate / shard | 4 / 1 |
| tp / cp / pp / ep | 1 / 1 / 1 / 1 |
| recorded runs | 10 |

## run results

| run | status | mode | preset | median step | tok/s/device | tok/s/job | peak hbm |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| [npu-ddp4-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-5da7b9c5](../../../runs/4-card/ddp4/npu-ddp4-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-5da7b9c5/readme.md) | completed | profiler-off | distributed | 389.92 ms | 5,252.34 | 21,009.35 | 0.843 GiB |
| [npu-ddp4-bf16-s20-l8-b64-seq128-seed61-distributed-r1-64a465c1](../../../runs/4-card/ddp4/npu-ddp4-bf16-s20-l8-b64-seq128-seed61-distributed-r1-64a465c1/readme.md) | completed | profiler-active | distributed | 398.32 ms | 5,141.56 | 20,566.23 | - GiB |
| [npu-ddp4-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-198b87aa](../../../runs/4-card/ddp4/npu-ddp4-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-198b87aa/readme.md) | completed | profiler-active | overview | 758.56 ms | 2,699.87 | 10,799.48 | - GiB |
| [npu-ddp4-bf16-s30-l8-b64-seq128-seed61-distributed-r31-da378369](../../../runs/4-card/ddp4/npu-ddp4-bf16-s30-l8-b64-seq128-seed61-distributed-r31-da378369/readme.md) | completed | profiler-active | distributed | 733.58 ms | 2,791.80 | 11,167.18 | - GiB |
| [npu-ddp4-bf16-s30-l8-b64-seq128-seed61-kernel-r31-offline-9f4621cf](../../../runs/4-card/ddp4/npu-ddp4-bf16-s30-l8-b64-seq128-seed61-kernel-r31-offline-9f4621cf/readme.md) | completed | profiler-active | kernel | 735.31 ms | 2,785.21 | 11,140.84 | - GiB |
| [npu-ddp4-bf16-s30-l8-b64-seq128-seed61-operator-r32-offline-c2c19178](../../../runs/4-card/ddp4/npu-ddp4-bf16-s30-l8-b64-seq128-seed61-operator-r32-offline-c2c19178/readme.md) | completed | profiler-active | operator | 740.34 ms | 2,766.31 | 11,065.23 | - GiB |
| [npu-ddp4-bf16-s30-l8-b64-seq128-seed61-memory-r31-offline-5c0081ac](../../../runs/4-card/ddp4/npu-ddp4-bf16-s30-l8-b64-seq128-seed61-memory-r31-offline-5c0081ac/readme.md) | completed | profiler-active | memory | 745.44 ms | 2,747.36 | 10,989.43 | - GiB |
| [npu-ddp4-bf16-s30-l8-b64-seq128-seed61-flamegraph-r31-8b0bdbfd](../../../runs/4-card/ddp4/npu-ddp4-bf16-s30-l8-b64-seq128-seed61-flamegraph-r31-8b0bdbfd/readme.md) | completed | profiler-active | flamegraph | 727.79 ms | 2,814.01 | 11,256.05 | - GiB |
| [npu-ddp4-bf16-s30-l8-b64-seq128-seed61-runtime-r31-offline-d605fa5d](../../../runs/4-card/ddp4/npu-ddp4-bf16-s30-l8-b64-seq128-seed61-runtime-r31-offline-d605fa5d/readme.md) | completed | profiler-active | runtime | 738.94 ms | 2,771.53 | 11,086.11 | - GiB |
| [npu-ddp4-bf16-s30-l8-b64-seq128-seed61-system-r31-9eedf37a](../../../runs/4-card/ddp4/npu-ddp4-bf16-s30-l8-b64-seq128-seed61-system-r31-9eedf37a/readme.md) | completed | profiler-active | system | 733.29 ms | 2,792.90 | 11,171.59 | - GiB |

## experiment sequence

### 1. npu-ddp4-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-5da7b9c5

- Status: `completed`
- Started: `2026-08-24T16:33:00.017343+08:00`
- Full process and outputs: [run readme](../../../runs/4-card/ddp4/npu-ddp4-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-5da7b9c5/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/4-card/ddp4/npu-ddp4-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-5da7b9c5.html`

```bash
/root/miniconda3/envs/torchtitan-0803/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology ddp4 --preset distributed --profiler-off --visible-devices 1,2,3,4 --steps 20 --skip-steps 8 --warmup-steps 2 --active-steps 3 --replicate 1
```

```bash
/root/miniconda3/envs/torchtitan-0803/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology ddp4 --preset distributed --profiler-off --visible-devices 1,2,3,4 --steps 20 --skip-steps 8 --warmup-steps 2 --active-steps 3 --replicate 1
```

### 2. npu-ddp4-bf16-s20-l8-b64-seq128-seed61-distributed-r1-64a465c1

- Status: `completed`
- Started: `2026-08-24T16:44:11.408897+08:00`
- Full process and outputs: [run readme](../../../runs/4-card/ddp4/npu-ddp4-bf16-s20-l8-b64-seq128-seed61-distributed-r1-64a465c1/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/4-card/ddp4/npu-ddp4-bf16-s20-l8-b64-seq128-seed61-distributed-r1-64a465c1.html`

```bash
/root/miniconda3/envs/torchtitan-0803/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology ddp4 --preset distributed --visible-devices 1,2,3,4 --steps 20 --skip-steps 8 --warmup-steps 2 --active-steps 3 --replicate 1 --cluster
```

```bash
/root/miniconda3/envs/torchtitan-0803/bin/python tests/glm5_2_performance/profiler_benchmark.py --analyze --device npu --topology ddp4 --preset distributed --visible-devices 1,2,3,4 --steps 20 --skip-steps 8 --warmup-steps 2 --active-steps 3 --replicate 1 --cluster
```

```bash
/root/miniconda3/envs/torchtitan-0803/bin/python tests/glm5_2_performance/profiler_benchmark.py --analyze --device npu --topology ddp4 --preset distributed --visible-devices 1,2,3,4 --steps 20 --skip-steps 8 --warmup-steps 2 --active-steps 3 --replicate 1
```

### 3. npu-ddp4-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-198b87aa

- Status: `completed`
- Started: `2026-08-31T11:44:52.760329+08:00`
- Full process and outputs: [run readme](../../../runs/4-card/ddp4/npu-ddp4-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-198b87aa/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/4-card/ddp4/npu-ddp4-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-198b87aa.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology all --preset all --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all
```

### 4. npu-ddp4-bf16-s30-l8-b64-seq128-seed61-distributed-r31-da378369

- Status: `completed`
- Started: `2026-08-31T17:01:56.211244+08:00`
- Full process and outputs: [run readme](../../../runs/4-card/ddp4/npu-ddp4-bf16-s30-l8-b64-seq128-seed61-distributed-r31-da378369/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/4-card/ddp4/npu-ddp4-bf16-s30-l8-b64-seq128-seed61-distributed-r31-da378369.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topologies fsdp2-tp4-ep8,ddp4,fsdp2,fsdp4,tp2,tp4,cp2,cp4,pp2,pp4,ep2,ep4,fsdp2-tp2 --preset distributed --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all
```

### 5. npu-ddp4-bf16-s30-l8-b64-seq128-seed61-kernel-r31-offline-9f4621cf

- Status: `completed`
- Started: `2026-08-31T21:13:39.140636+08:00`
- Full process and outputs: [run readme](../../../runs/4-card/ddp4/npu-ddp4-bf16-s30-l8-b64-seq128-seed61-kernel-r31-offline-9f4621cf/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/4-card/ddp4/npu-ddp4-bf16-s30-l8-b64-seq128-seed61-kernel-r31-offline-9f4621cf.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology all --preset kernel --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all
```

### 6. npu-ddp4-bf16-s30-l8-b64-seq128-seed61-operator-r32-offline-c2c19178

- Status: `completed`
- Started: `2026-09-01T00:00:16.659425+08:00`
- Full process and outputs: [run readme](../../../runs/4-card/ddp4/npu-ddp4-bf16-s30-l8-b64-seq128-seed61-operator-r32-offline-c2c19178/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/4-card/ddp4/npu-ddp4-bf16-s30-l8-b64-seq128-seed61-operator-r32-offline-c2c19178.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --capture --device npu --topology ddp4 --preset operator --visible-devices 0,1,2,3,4,5,6,7 --replicate 32 --analysis-tools none
```

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --analyze --device npu --topology ddp4 --preset operator --visible-devices 0,1,2,3,4,5,6,7 --replicate 32 --analysis-tools all --parse-workers 8
```

### 7. npu-ddp4-bf16-s30-l8-b64-seq128-seed61-memory-r31-offline-5c0081ac

- Status: `completed`
- Started: `2026-09-01T09:41:11.438193+08:00`
- Full process and outputs: [run readme](../../../runs/4-card/ddp4/npu-ddp4-bf16-s30-l8-b64-seq128-seed61-memory-r31-offline-5c0081ac/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/4-card/ddp4/npu-ddp4-bf16-s30-l8-b64-seq128-seed61-memory-r31-offline-5c0081ac.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --capture --device npu --topology ddp4 --preset memory --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools none
```

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --analyze --device npu --topology ddp4 --preset memory --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all --parse-workers 8
```

### 8. npu-ddp4-bf16-s30-l8-b64-seq128-seed61-flamegraph-r31-8b0bdbfd

- Status: `completed`
- Started: `2026-09-01T11:01:01.050200+08:00`
- Full process and outputs: [run readme](../../../runs/4-card/ddp4/npu-ddp4-bf16-s30-l8-b64-seq128-seed61-flamegraph-r31-8b0bdbfd/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/4-card/ddp4/npu-ddp4-bf16-s30-l8-b64-seq128-seed61-flamegraph-r31-8b0bdbfd.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --capture --device npu --topology ddp4 --preset flamegraph --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools none
```

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --analyze --device npu --topology ddp4 --preset flamegraph --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all --parse-workers 8
```

### 9. npu-ddp4-bf16-s30-l8-b64-seq128-seed61-runtime-r31-offline-d605fa5d

- Status: `completed`
- Started: `2026-09-01T12:19:49.815105+08:00`
- Full process and outputs: [run readme](../../../runs/4-card/ddp4/npu-ddp4-bf16-s30-l8-b64-seq128-seed61-runtime-r31-offline-d605fa5d/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/4-card/ddp4/npu-ddp4-bf16-s30-l8-b64-seq128-seed61-runtime-r31-offline-d605fa5d.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --capture --device npu --topology ddp4 --preset runtime --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools none
```

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --analyze --device npu --topology ddp4 --preset runtime --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all --parse-workers 8
```

### 10. npu-ddp4-bf16-s30-l8-b64-seq128-seed61-system-r31-9eedf37a

- Status: `completed`
- Started: `2026-09-01T13:46:03.582050+08:00`
- Full process and outputs: [run readme](../../../runs/4-card/ddp4/npu-ddp4-bf16-s30-l8-b64-seq128-seed61-system-r31-9eedf37a/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/4-card/ddp4/npu-ddp4-bf16-s30-l8-b64-seq128-seed61-system-r31-9eedf37a.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --capture --device npu --topology ddp4 --preset system --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools none
```

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --analyze --device npu --topology ddp4 --preset system --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all --parse-workers 8
```

## current summary

The highest recorded job throughput is 21,009.35 tok/s from `npu-ddp4-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-5da7b9c5`. Profiler-off runs are throughput evidence; profiler-active runs are attribution evidence only.
