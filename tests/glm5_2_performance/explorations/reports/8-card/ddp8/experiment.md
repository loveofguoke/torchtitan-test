# ddp8 experiments

## topology

| field | value |
| --- | --- |
| world size | 8 |
| dp replicate / shard | 8 / 1 |
| tp / cp / pp / ep | 1 / 1 / 1 / 1 |
| recorded runs | 3 |

## run results

| run | status | mode | preset | median step | tok/s/device | tok/s/job | peak hbm |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| [npu-ddp8-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-270760c1](../../../runs/8-card/ddp8/npu-ddp8-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-270760c1/readme.md) | failed | profiler-off | - | - ms | - | - | - GiB |
| [npu-ddp8-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r2-80606b65](../../../runs/8-card/ddp8/npu-ddp8-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r2-80606b65/readme.md) | completed | profiler-off | distributed | 773.35 ms | 1,370.09 | 10,960.70 | 0.783 GiB |
| [npu-ddp8-bf16-s20-l8-b64-seq128-seed61-distributed-r1-488247ce](../../../runs/8-card/ddp8/npu-ddp8-bf16-s20-l8-b64-seq128-seed61-distributed-r1-488247ce/readme.md) | completed | profiler-active | distributed | 219.14 ms | 4,672.84 | 37,382.71 | - GiB |

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

## current summary

The highest recorded job throughput is 37,382.71 tok/s from `npu-ddp8-bf16-s20-l8-b64-seq128-seed61-distributed-r1-488247ce`. Profiler-off runs are throughput evidence; profiler-active runs are attribution evidence only.
