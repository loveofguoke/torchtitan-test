# fsdp2-tp4-ep8 experiments

## topology

| field | value |
| --- | --- |
| world size | 8 |
| dp replicate / shard | 1 / 2 |
| tp / cp / pp / ep | 4 / 1 / 1 / 8 |
| recorded runs | 3 |

## run results

| run | status | mode | preset | median step | tok/s/device | tok/s/job | peak hbm |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| [npu-fsdp2-tp4-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-53db2e7c](../../../runs/8-card/fsdp2-tp4-ep8/npu-fsdp2-tp4-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-53db2e7c/readme.md) | completed | profiler-off | distributed | 2,323.02 ms | 440.80 | 3,526.44 | 0.216 GiB |
| [npu-fsdp2-tp4-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-r1-bf02b5d0](../../../runs/8-card/fsdp2-tp4-ep8/npu-fsdp2-tp4-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-r1-bf02b5d0/readme.md) | failed | profiler-active | - | - ms | - | - | - GiB |
| [npu-fsdp2-tp4-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-r2-e6d2b2bd](../../../runs/8-card/fsdp2-tp4-ep8/npu-fsdp2-tp4-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-r2-e6d2b2bd/readme.md) | completed | profiler-active | distributed | 2,658.32 ms | 385.21 | 3,081.64 | - GiB |

## experiment sequence

### 1. npu-fsdp2-tp4-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-53db2e7c

- Status: `completed`
- Started: `2026-08-24T18:09:26.542934+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/fsdp2-tp4-ep8/npu-fsdp2-tp4-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-53db2e7c/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/fsdp2-tp4-ep8/npu-fsdp2-tp4-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-53db2e7c.html`

```bash
/root/miniconda3/envs/torchtitan-0803/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology fsdp2-tp4-ep8 --preset distributed --profiler-off --visible-devices 0,1,2,3,4,5,6,7 --steps 20 --skip-steps 8 --warmup-steps 2 --active-steps 3 --replicate 1
```

### 2. npu-fsdp2-tp4-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-r1-bf02b5d0

- Status: `failed`
- Started: `2026-08-25T09:25:07.206333+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/fsdp2-tp4-ep8/npu-fsdp2-tp4-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-r1-bf02b5d0/readme.md)

```bash
/root/miniconda3/envs/torchtitan-0803/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology fsdp2-tp4-ep8 --preset distributed --visible-devices 0,1,2,3,4,5,6,7 --steps 20 --skip-steps 8 --warmup-steps 2 --active-steps 3 --replicate 1 --cluster
```

### 3. npu-fsdp2-tp4-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-r2-e6d2b2bd

- Status: `completed`
- Started: `2026-08-25T13:57:47.973837+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/fsdp2-tp4-ep8/npu-fsdp2-tp4-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-r2-e6d2b2bd/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/fsdp2-tp4-ep8/npu-fsdp2-tp4-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-r2-e6d2b2bd.html`

```bash
/root/miniconda3/envs/torchtitan-0803/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topologies pp8,fsdp2-tp4,fsdp2-cp4,tp2-cp4,fsdp4-tp2,fsdp2-pp4,fsdp2-tp2-pp2,fsdp2-tp4-ep8 --preset distributed --visible-devices 0,1,2,3,4,5,6,7 --steps 20 --skip-steps 8 --warmup-steps 2 --active-steps 3 --replicate 2 --cluster
```

## current summary

The highest recorded job throughput is 3,526.44 tok/s from `npu-fsdp2-tp4-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-53db2e7c`. Profiler-off runs are throughput evidence; profiler-active runs are attribution evidence only.
