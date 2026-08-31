# tp8 experiments

## topology

| field | value |
| --- | --- |
| world size | 8 |
| dp replicate / shard | 1 / 1 |
| tp / cp / pp / ep | 8 / 1 / 1 / 1 |
| recorded runs | 5 |

## run results

| run | status | mode | preset | median step | tok/s/device | tok/s/job | peak hbm |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| [npu-tp8-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-64da983b](../../../runs/8-card/tp8/npu-tp8-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-64da983b/readme.md) | completed | profiler-off | distributed | 4,421.03 ms | 231.62 | 1,852.98 | 0.208 GiB |
| [npu-tp8-bf16-s20-l8-b64-seq128-seed61-distributed-r1-dba2190c](../../../runs/8-card/tp8/npu-tp8-bf16-s20-l8-b64-seq128-seed61-distributed-r1-dba2190c/readme.md) | completed | profiler-active | distributed | 9,097.91 ms | 112.55 | 900.43 | - GiB |
| [npu-tp8-bf16-s30-l8-b64-seq128-seed61-overview-r31-dbbcd393](../../../runs/8-card/tp8/npu-tp8-bf16-s30-l8-b64-seq128-seed61-overview-r31-dbbcd393/readme.md) | failed | profiler-active | - | - ms | - | - | - GiB |
| [npu-tp8-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-7d88d2fc](../../../runs/8-card/tp8/npu-tp8-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-7d88d2fc/readme.md) | completed | profiler-active | overview | 6,274.99 ms | 163.19 | 1,305.50 | - GiB |
| [npu-tp8-bf16-s30-l8-b64-seq128-seed61-distributed-r31-5154b546](../../../runs/8-card/tp8/npu-tp8-bf16-s30-l8-b64-seq128-seed61-distributed-r31-5154b546/readme.md) | completed | profiler-active | distributed | 6,244.72 ms | 163.98 | 1,311.83 | - GiB |

## experiment sequence

### 1. npu-tp8-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-64da983b

- Status: `completed`
- Started: `2026-08-24T17:49:57.398320+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/tp8/npu-tp8-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-64da983b/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/tp8/npu-tp8-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-64da983b.html`

```bash
/root/miniconda3/envs/torchtitan-0803/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology tp8 --preset distributed --profiler-off --visible-devices 0,1,2,3,4,5,6,7 --steps 20 --skip-steps 8 --warmup-steps 2 --active-steps 3 --replicate 1
```

### 2. npu-tp8-bf16-s20-l8-b64-seq128-seed61-distributed-r1-dba2190c

- Status: `completed`
- Started: `2026-08-24T20:42:29.508722+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/tp8/npu-tp8-bf16-s20-l8-b64-seq128-seed61-distributed-r1-dba2190c/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/tp8/npu-tp8-bf16-s20-l8-b64-seq128-seed61-distributed-r1-dba2190c.html`

```bash
/root/miniconda3/envs/torchtitan-0803/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology tp8 --preset distributed --visible-devices 0,1,2,3,4,5,6,7 --steps 20 --skip-steps 8 --warmup-steps 2 --active-steps 3 --replicate 1 --cluster
```

### 3. npu-tp8-bf16-s30-l8-b64-seq128-seed61-overview-r31-dbbcd393

- Status: `failed`
- Started: `2026-08-31T10:11:02.193769+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/tp8/npu-tp8-bf16-s30-l8-b64-seq128-seed61-overview-r31-dbbcd393/readme.md)

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology all --preset all --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all
```

### 4. npu-tp8-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-7d88d2fc

- Status: `completed`
- Started: `2026-08-31T10:19:57.599445+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/tp8/npu-tp8-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-7d88d2fc/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/tp8/npu-tp8-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-7d88d2fc.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology tp8 --preset overview --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all
```

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology all --preset all --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all
```

### 5. npu-tp8-bf16-s30-l8-b64-seq128-seed61-distributed-r31-5154b546

- Status: `completed`
- Started: `2026-08-31T13:24:01.758579+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/tp8/npu-tp8-bf16-s30-l8-b64-seq128-seed61-distributed-r31-5154b546/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/tp8/npu-tp8-bf16-s30-l8-b64-seq128-seed61-distributed-r31-5154b546.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology all --preset all --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all
```

## current summary

The highest recorded job throughput is 1,852.98 tok/s from `npu-tp8-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-64da983b`. Profiler-off runs are throughput evidence; profiler-active runs are attribution evidence only.

## analysis

- [Detailed topology analysis](analysis.md)
