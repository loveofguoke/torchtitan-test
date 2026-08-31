# ddp8 experiments

## topology

| field | value |
| --- | --- |
| world size | 8 |
| dp replicate / shard | 8 / 1 |
| tp / cp / pp / ep | 1 / 1 / 1 / 1 |
| recorded runs | 6 |

## run results

| run | status | mode | preset | median step | tok/s/device | tok/s/job | peak hbm |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| [npu-ddp8-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-270760c1](../../../runs/8-card/ddp8/npu-ddp8-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-270760c1/readme.md) | failed | profiler-off | - | - ms | - | - | - GiB |
| [npu-ddp8-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r2-80606b65](../../../runs/8-card/ddp8/npu-ddp8-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r2-80606b65/readme.md) | completed | profiler-off | distributed | 773.35 ms | 1,370.09 | 10,960.70 | 0.783 GiB |
| [npu-ddp8-bf16-s20-l8-b64-seq128-seed61-distributed-r1-488247ce](../../../runs/8-card/ddp8/npu-ddp8-bf16-s20-l8-b64-seq128-seed61-distributed-r1-488247ce/readme.md) | completed | profiler-active | distributed | 219.14 ms | 4,672.84 | 37,382.71 | - GiB |
| [npu-ddp8-bf16-s30-l8-b64-seq128-seed61-overview-r31-0ce71019](../../../runs/8-card/ddp8/npu-ddp8-bf16-s30-l8-b64-seq128-seed61-overview-r31-0ce71019/readme.md) | completed | profiler-active | overview | 440.49 ms | 2,324.69 | 18,597.53 | - GiB |
| [npu-ddp8-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-18f2f254](../../../runs/8-card/ddp8/npu-ddp8-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-18f2f254/readme.md) | completed | profiler-active | overview | 406.12 ms | 2,521.42 | 20,171.38 | - GiB |
| [npu-ddp8-bf16-s30-l8-b64-seq128-seed61-distributed-r31-aabbbd06](../../../runs/8-card/ddp8/npu-ddp8-bf16-s30-l8-b64-seq128-seed61-distributed-r31-aabbbd06/readme.md) | completed | profiler-active | distributed | 393.30 ms | 2,603.59 | 20,828.69 | - GiB |

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

### 6. npu-ddp8-bf16-s30-l8-b64-seq128-seed61-distributed-r31-aabbbd06

- Status: `completed`
- Started: `2026-08-31T13:11:06.534278+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/ddp8/npu-ddp8-bf16-s30-l8-b64-seq128-seed61-distributed-r31-aabbbd06/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/ddp8/npu-ddp8-bf16-s30-l8-b64-seq128-seed61-distributed-r31-aabbbd06.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology all --preset all --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all
```

## current summary

The highest recorded job throughput is 37,382.71 tok/s from `npu-ddp8-bf16-s20-l8-b64-seq128-seed61-distributed-r1-488247ce`. Profiler-off runs are throughput evidence; profiler-active runs are attribution evidence only.

## analysis

- [Detailed topology analysis](analysis.md)
