# fsdp4 experiments

## topology

| field | value |
| --- | --- |
| world size | 4 |
| dp replicate / shard | 1 / 4 |
| tp / cp / pp / ep | 1 / 1 / 1 / 1 |
| recorded runs | 12 |

## run results

| run | status | mode | preset | median step | tok/s/device | tok/s/job | peak hbm |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| [npu-fsdp4-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-0ba5e24c](../../../runs/4-card/fsdp4/npu-fsdp4-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-0ba5e24c/readme.md) | completed | profiler-off | distributed | 401.75 ms | 5,097.71 | 20,390.85 | 0.660 GiB |
| [npu-fsdp4-bf16-s20-l8-b64-seq128-seed61-distributed-r1-965e89d3](../../../runs/4-card/fsdp4/npu-fsdp4-bf16-s20-l8-b64-seq128-seed61-distributed-r1-965e89d3/readme.md) | failed | profiler-active | - | - ms | - | - | - GiB |
| [npu-fsdp4-bf16-s20-l8-b64-seq128-seed61-distributed-r2-49400b80](../../../runs/4-card/fsdp4/npu-fsdp4-bf16-s20-l8-b64-seq128-seed61-distributed-r2-49400b80/readme.md) | failed | profiler-active | - | - ms | - | - | - GiB |
| [npu-fsdp4-bf16-s20-l8-b64-seq128-seed61-distributed-r3-d3b92684](../../../runs/4-card/fsdp4/npu-fsdp4-bf16-s20-l8-b64-seq128-seed61-distributed-r3-d3b92684/readme.md) | completed | profiler-active | distributed | 409.04 ms | 5,006.86 | 20,027.44 | - GiB |
| [npu-fsdp4-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-2cf55db3](../../../runs/4-card/fsdp4/npu-fsdp4-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-2cf55db3/readme.md) | completed | profiler-active | overview | 709.53 ms | 2,886.43 | 11,545.73 | - GiB |
| [npu-fsdp4-bf16-s30-l8-b64-seq128-seed61-distributed-r31-238dce9f](../../../runs/4-card/fsdp4/npu-fsdp4-bf16-s30-l8-b64-seq128-seed61-distributed-r31-238dce9f/readme.md) | completed | profiler-active | distributed | 744.07 ms | 2,752.42 | 11,009.68 | - GiB |
| [npu-fsdp4-bf16-s30-l8-b64-seq128-seed61-kernel-r31-offline-5578ac6a](../../../runs/4-card/fsdp4/npu-fsdp4-bf16-s30-l8-b64-seq128-seed61-kernel-r31-offline-5578ac6a/readme.md) | completed | profiler-active | kernel | 722.48 ms | 2,834.68 | 11,338.70 | - GiB |
| [npu-fsdp4-bf16-s30-l8-b64-seq128-seed61-operator-r32-offline-c03a3c30](../../../runs/4-card/fsdp4/npu-fsdp4-bf16-s30-l8-b64-seq128-seed61-operator-r32-offline-c03a3c30/readme.md) | completed | profiler-active | operator | 713.76 ms | 2,869.30 | 11,477.19 | - GiB |
| [npu-fsdp4-bf16-s30-l8-b64-seq128-seed61-memory-r31-offline-567866ed](../../../runs/4-card/fsdp4/npu-fsdp4-bf16-s30-l8-b64-seq128-seed61-memory-r31-offline-567866ed/readme.md) | completed | profiler-active | memory | 729.66 ms | 2,806.78 | 11,227.10 | - GiB |
| [npu-fsdp4-bf16-s30-l8-b64-seq128-seed61-flamegraph-r31-63986e8b](../../../runs/4-card/fsdp4/npu-fsdp4-bf16-s30-l8-b64-seq128-seed61-flamegraph-r31-63986e8b/readme.md) | completed | profiler-active | flamegraph | 714.03 ms | 2,868.22 | 11,472.87 | - GiB |
| [npu-fsdp4-bf16-s30-l8-b64-seq128-seed61-runtime-r31-offline-50b36c14](../../../runs/4-card/fsdp4/npu-fsdp4-bf16-s30-l8-b64-seq128-seed61-runtime-r31-offline-50b36c14/readme.md) | completed | profiler-active | runtime | 709.38 ms | 2,887.01 | 11,548.06 | - GiB |
| [npu-fsdp4-bf16-s30-l8-b64-seq128-seed61-system-r31-3c22a1eb](../../../runs/4-card/fsdp4/npu-fsdp4-bf16-s30-l8-b64-seq128-seed61-system-r31-3c22a1eb/readme.md) | completed | profiler-active | system | 709.48 ms | 2,886.62 | 11,546.47 | - GiB |

## experiment sequence

### 1. npu-fsdp4-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-0ba5e24c

- Status: `completed`
- Started: `2026-08-24T16:34:54.341851+08:00`
- Full process and outputs: [run readme](../../../runs/4-card/fsdp4/npu-fsdp4-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-0ba5e24c/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/4-card/fsdp4/npu-fsdp4-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-0ba5e24c.html`

```bash
/root/miniconda3/envs/torchtitan-0803/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology fsdp4 --preset distributed --profiler-off --visible-devices 1,2,3,4 --steps 20 --skip-steps 8 --warmup-steps 2 --active-steps 3 --replicate 1
```

### 2. npu-fsdp4-bf16-s20-l8-b64-seq128-seed61-distributed-r1-965e89d3

- Status: `failed`
- Started: `2026-08-24T16:53:45.152666+08:00`
- Full process and outputs: [run readme](../../../runs/4-card/fsdp4/npu-fsdp4-bf16-s20-l8-b64-seq128-seed61-distributed-r1-965e89d3/readme.md)

```bash
/root/miniconda3/envs/torchtitan-0803/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology fsdp4 --preset distributed --visible-devices 1,2,3,4 --steps 20 --skip-steps 8 --warmup-steps 2 --active-steps 3 --replicate 1 --cluster
```

### 3. npu-fsdp4-bf16-s20-l8-b64-seq128-seed61-distributed-r2-49400b80

- Status: `failed`
- Started: `2026-08-24T16:56:37.297551+08:00`
- Full process and outputs: [run readme](../../../runs/4-card/fsdp4/npu-fsdp4-bf16-s20-l8-b64-seq128-seed61-distributed-r2-49400b80/readme.md)

```bash
/root/miniconda3/envs/torchtitan-0803/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology fsdp4 --preset distributed --visible-devices 4,5,6,7 --steps 20 --skip-steps 8 --warmup-steps 2 --active-steps 3 --replicate 2 --cluster
```

### 4. npu-fsdp4-bf16-s20-l8-b64-seq128-seed61-distributed-r3-d3b92684

- Status: `completed`
- Started: `2026-08-24T17:02:27.528676+08:00`
- Full process and outputs: [run readme](../../../runs/4-card/fsdp4/npu-fsdp4-bf16-s20-l8-b64-seq128-seed61-distributed-r3-d3b92684/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/4-card/fsdp4/npu-fsdp4-bf16-s20-l8-b64-seq128-seed61-distributed-r3-d3b92684.html`

```bash
/root/miniconda3/envs/torchtitan-0803/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology fsdp4 --preset distributed --visible-devices 1,2,3,5 --steps 20 --skip-steps 8 --warmup-steps 2 --active-steps 3 --replicate 3 --cluster
```

### 5. npu-fsdp4-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-2cf55db3

- Status: `completed`
- Started: `2026-08-31T11:50:17.888430+08:00`
- Full process and outputs: [run readme](../../../runs/4-card/fsdp4/npu-fsdp4-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-2cf55db3/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/4-card/fsdp4/npu-fsdp4-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-2cf55db3.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology all --preset all --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all
```

### 6. npu-fsdp4-bf16-s30-l8-b64-seq128-seed61-distributed-r31-238dce9f

- Status: `completed`
- Started: `2026-08-31T17:13:40.137693+08:00`
- Full process and outputs: [run readme](../../../runs/4-card/fsdp4/npu-fsdp4-bf16-s30-l8-b64-seq128-seed61-distributed-r31-238dce9f/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/4-card/fsdp4/npu-fsdp4-bf16-s30-l8-b64-seq128-seed61-distributed-r31-238dce9f.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topologies fsdp2-tp4-ep8,ddp4,fsdp2,fsdp4,tp2,tp4,cp2,cp4,pp2,pp4,ep2,ep4,fsdp2-tp2 --preset distributed --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all
```

### 7. npu-fsdp4-bf16-s30-l8-b64-seq128-seed61-kernel-r31-offline-5578ac6a

- Status: `completed`
- Started: `2026-08-31T21:19:46.890825+08:00`
- Full process and outputs: [run readme](../../../runs/4-card/fsdp4/npu-fsdp4-bf16-s30-l8-b64-seq128-seed61-kernel-r31-offline-5578ac6a/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/4-card/fsdp4/npu-fsdp4-bf16-s30-l8-b64-seq128-seed61-kernel-r31-offline-5578ac6a.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology all --preset kernel --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all
```

### 8. npu-fsdp4-bf16-s30-l8-b64-seq128-seed61-operator-r32-offline-c03a3c30

- Status: `completed`
- Started: `2026-09-01T00:03:08.761420+08:00`
- Full process and outputs: [run readme](../../../runs/4-card/fsdp4/npu-fsdp4-bf16-s30-l8-b64-seq128-seed61-operator-r32-offline-c03a3c30/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/4-card/fsdp4/npu-fsdp4-bf16-s30-l8-b64-seq128-seed61-operator-r32-offline-c03a3c30.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --capture --device npu --topology fsdp4 --preset operator --visible-devices 0,1,2,3,4,5,6,7 --replicate 32 --analysis-tools none
```

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --analyze --device npu --topology fsdp4 --preset operator --visible-devices 0,1,2,3,4,5,6,7 --replicate 32 --analysis-tools all --parse-workers 8
```

### 9. npu-fsdp4-bf16-s30-l8-b64-seq128-seed61-memory-r31-offline-567866ed

- Status: `completed`
- Started: `2026-09-01T09:43:58.602451+08:00`
- Full process and outputs: [run readme](../../../runs/4-card/fsdp4/npu-fsdp4-bf16-s30-l8-b64-seq128-seed61-memory-r31-offline-567866ed/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/4-card/fsdp4/npu-fsdp4-bf16-s30-l8-b64-seq128-seed61-memory-r31-offline-567866ed.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --capture --device npu --topology fsdp4 --preset memory --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools none
```

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --analyze --device npu --topology fsdp4 --preset memory --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all --parse-workers 8
```

### 10. npu-fsdp4-bf16-s30-l8-b64-seq128-seed61-flamegraph-r31-63986e8b

- Status: `completed`
- Started: `2026-09-01T11:03:30.866490+08:00`
- Full process and outputs: [run readme](../../../runs/4-card/fsdp4/npu-fsdp4-bf16-s30-l8-b64-seq128-seed61-flamegraph-r31-63986e8b/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/4-card/fsdp4/npu-fsdp4-bf16-s30-l8-b64-seq128-seed61-flamegraph-r31-63986e8b.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --capture --device npu --topology fsdp4 --preset flamegraph --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools none
```

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --analyze --device npu --topology fsdp4 --preset flamegraph --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all --parse-workers 8
```

### 11. npu-fsdp4-bf16-s30-l8-b64-seq128-seed61-runtime-r31-offline-50b36c14

- Status: `completed`
- Started: `2026-09-01T12:22:42.317328+08:00`
- Full process and outputs: [run readme](../../../runs/4-card/fsdp4/npu-fsdp4-bf16-s30-l8-b64-seq128-seed61-runtime-r31-offline-50b36c14/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/4-card/fsdp4/npu-fsdp4-bf16-s30-l8-b64-seq128-seed61-runtime-r31-offline-50b36c14.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --capture --device npu --topology fsdp4 --preset runtime --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools none
```

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --analyze --device npu --topology fsdp4 --preset runtime --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all --parse-workers 8
```

### 12. npu-fsdp4-bf16-s30-l8-b64-seq128-seed61-system-r31-3c22a1eb

- Status: `completed`
- Started: `2026-09-01T13:48:43.621791+08:00`
- Full process and outputs: [run readme](../../../runs/4-card/fsdp4/npu-fsdp4-bf16-s30-l8-b64-seq128-seed61-system-r31-3c22a1eb/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/4-card/fsdp4/npu-fsdp4-bf16-s30-l8-b64-seq128-seed61-system-r31-3c22a1eb.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --capture --device npu --topology fsdp4 --preset system --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools none
```

## current summary

The highest recorded job throughput is 20,390.85 tok/s from `npu-fsdp4-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-0ba5e24c`. Profiler-off runs are throughput evidence; profiler-active runs are attribution evidence only.
