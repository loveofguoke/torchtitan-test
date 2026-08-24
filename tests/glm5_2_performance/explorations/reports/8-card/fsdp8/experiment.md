# fsdp8 experiments

## topology

| field | value |
| --- | --- |
| world size | 8 |
| dp replicate / shard | 1 / 8 |
| tp / cp / pp / ep | 1 / 1 / 1 / 1 |
| recorded runs | 2 |

## run results

| run | status | mode | preset | median step | tok/s/device | tok/s/job | peak hbm |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| [npu-fsdp8-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-4a88f56a](../../../runs/8-card/fsdp8/npu-fsdp8-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-4a88f56a/readme.md) | completed | profiler-off | distributed | 219.00 ms | 4,675.83 | 37,406.66 | 0.621 GiB |
| [npu-fsdp8-bf16-s20-l8-b64-seq128-seed61-distributed-r1-8501ffd6](../../../runs/8-card/fsdp8/npu-fsdp8-bf16-s20-l8-b64-seq128-seed61-distributed-r1-8501ffd6/readme.md) | completed | profiler-active | distributed | 232.32 ms | 4,407.63 | 35,261.06 | - GiB |

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

## current summary

The highest recorded job throughput is 37,406.66 tok/s from `npu-fsdp8-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-4a88f56a`. Profiler-off runs are throughput evidence; profiler-active runs are attribution evidence only.
