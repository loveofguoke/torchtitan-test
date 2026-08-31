# fsdp2-pp4 experiments

## topology

| field | value |
| --- | --- |
| world size | 8 |
| dp replicate / shard | 1 / 2 |
| tp / cp / pp / ep | 1 / 1 / 4 / 1 |
| recorded runs | 5 |

## run results

| run | status | mode | preset | median step | tok/s/device | tok/s/job | peak hbm |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| [npu-fsdp2-pp4-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-7900f287](../../../runs/8-card/fsdp2-pp4/npu-fsdp2-pp4-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-7900f287/readme.md) | completed | profiler-off | distributed | 1,744.04 ms | 587.21 | 4,697.69 | 0.044 GiB |
| [npu-fsdp2-pp4-bf16-s20-l8-b64-seq128-seed61-distributed-r1-283c59b2](../../../runs/8-card/fsdp2-pp4/npu-fsdp2-pp4-bf16-s20-l8-b64-seq128-seed61-distributed-r1-283c59b2/readme.md) | failed | profiler-active | - | - ms | - | - | - GiB |
| [npu-fsdp2-pp4-bf16-s20-l8-b64-seq128-seed61-distributed-r2-dde8b599](../../../runs/8-card/fsdp2-pp4/npu-fsdp2-pp4-bf16-s20-l8-b64-seq128-seed61-distributed-r2-dde8b599/readme.md) | completed | profiler-active | distributed | 2,316.39 ms | 442.07 | 3,536.53 | - GiB |
| [npu-fsdp2-pp4-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-7fbfaed3](../../../runs/8-card/fsdp2-pp4/npu-fsdp2-pp4-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-7fbfaed3/readme.md) | completed | profiler-active | overview | 3,270.23 ms | 313.13 | 2,505.02 | - GiB |
| [npu-fsdp2-pp4-bf16-s30-l8-b64-seq128-seed61-distributed-r31-eee8f86a](../../../runs/8-card/fsdp2-pp4/npu-fsdp2-pp4-bf16-s30-l8-b64-seq128-seed61-distributed-r31-eee8f86a/readme.md) | completed | profiler-active | distributed | 3,255.81 ms | 314.51 | 2,516.12 | - GiB |

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

## current summary

The highest recorded job throughput is 4,697.69 tok/s from `npu-fsdp2-pp4-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-7900f287`. Profiler-off runs are throughput evidence; profiler-active runs are attribution evidence only.
