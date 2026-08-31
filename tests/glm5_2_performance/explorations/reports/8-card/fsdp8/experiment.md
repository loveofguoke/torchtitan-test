# fsdp8 experiments

## topology

| field | value |
| --- | --- |
| world size | 8 |
| dp replicate / shard | 1 / 8 |
| tp / cp / pp / ep | 1 / 1 / 1 / 1 |
| recorded runs | 5 |

## run results

| run | status | mode | preset | median step | tok/s/device | tok/s/job | peak hbm |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| [npu-fsdp8-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-4a88f56a](../../../runs/8-card/fsdp8/npu-fsdp8-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-4a88f56a/readme.md) | completed | profiler-off | distributed | 219.00 ms | 4,675.83 | 37,406.66 | 0.621 GiB |
| [npu-fsdp8-bf16-s20-l8-b64-seq128-seed61-distributed-r1-8501ffd6](../../../runs/8-card/fsdp8/npu-fsdp8-bf16-s20-l8-b64-seq128-seed61-distributed-r1-8501ffd6/readme.md) | completed | profiler-active | distributed | 232.32 ms | 4,407.63 | 35,261.06 | - GiB |
| [npu-fsdp8-bf16-s30-l8-b64-seq128-seed61-overview-r31-b51e969a](../../../runs/8-card/fsdp8/npu-fsdp8-bf16-s30-l8-b64-seq128-seed61-overview-r31-b51e969a/readme.md) | completed | profiler-active | overview | 389.23 ms | 2,630.85 | 21,046.83 | - GiB |
| [npu-fsdp8-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-8a3891e5](../../../runs/8-card/fsdp8/npu-fsdp8-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-8a3891e5/readme.md) | completed | profiler-active | overview | 385.95 ms | 2,653.16 | 21,225.30 | - GiB |
| [npu-fsdp8-bf16-s30-l8-b64-seq128-seed61-distributed-r31-8c98fbbc](../../../runs/8-card/fsdp8/npu-fsdp8-bf16-s30-l8-b64-seq128-seed61-distributed-r31-8c98fbbc/readme.md) | completed | profiler-active | distributed | 386.94 ms | 2,646.42 | 21,171.37 | - GiB |

## experiment sequence

### 1. npu-fsdp8-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-4a88f56a

- Status: `completed`
- Started: `2026-08-24T17:49:07.801535+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/fsdp8/npu-fsdp8-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-4a88f56a/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/fsdp8/npu-fsdp8-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-4a88f56a.html`

```bash
/root/miniconda3/envs/torchtitan-0803/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology fsdp8 --preset distributed --profiler-off --visible-devices 0,1,2,3,4,5,6,7 --steps 20 --skip-steps 8 --warmup-steps 2 --active-steps 3 --replicate 1
```

### 2. npu-fsdp8-bf16-s20-l8-b64-seq128-seed61-distributed-r1-8501ffd6

- Status: `completed`
- Started: `2026-08-24T18:26:01.593338+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/fsdp8/npu-fsdp8-bf16-s20-l8-b64-seq128-seed61-distributed-r1-8501ffd6/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/fsdp8/npu-fsdp8-bf16-s20-l8-b64-seq128-seed61-distributed-r1-8501ffd6.html`

```bash
/root/miniconda3/envs/torchtitan-0803/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology fsdp8 --preset distributed --visible-devices 0,1,2,3,4,5,6,7 --steps 20 --skip-steps 8 --warmup-steps 2 --active-steps 3 --replicate 1 --cluster
```

### 3. npu-fsdp8-bf16-s30-l8-b64-seq128-seed61-overview-r31-b51e969a

- Status: `completed`
- Started: `2026-08-31T10:09:30.486260+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/fsdp8/npu-fsdp8-bf16-s30-l8-b64-seq128-seed61-overview-r31-b51e969a/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/fsdp8/npu-fsdp8-bf16-s30-l8-b64-seq128-seed61-overview-r31-b51e969a.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology all --preset all --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all
```

### 4. npu-fsdp8-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-8a3891e5

- Status: `completed`
- Started: `2026-08-31T10:39:48.051672+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/fsdp8/npu-fsdp8-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-8a3891e5/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/fsdp8/npu-fsdp8-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-8a3891e5.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology all --preset all --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all
```

### 5. npu-fsdp8-bf16-s30-l8-b64-seq128-seed61-distributed-r31-8c98fbbc

- Status: `completed`
- Started: `2026-08-31T13:16:11.522720+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/fsdp8/npu-fsdp8-bf16-s30-l8-b64-seq128-seed61-distributed-r31-8c98fbbc/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/fsdp8/npu-fsdp8-bf16-s30-l8-b64-seq128-seed61-distributed-r31-8c98fbbc.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology all --preset all --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all
```

## current summary

The highest recorded job throughput is 37,406.66 tok/s from `npu-fsdp8-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-4a88f56a`. Profiler-off runs are throughput evidence; profiler-active runs are attribution evidence only.

## analysis

- [Detailed topology analysis](analysis.md)
