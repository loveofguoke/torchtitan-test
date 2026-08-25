# ep8 experiments

## topology

| field | value |
| --- | --- |
| world size | 8 |
| dp replicate / shard | 1 / 8 |
| tp / cp / pp / ep | 1 / 1 / 1 / 8 |
| recorded runs | 5 |

## run results

| run | status | mode | preset | median step | tok/s/device | tok/s/job | peak hbm |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| [npu-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-50c6107d](../../../runs/8-card/ep8/npu-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-50c6107d/readme.md) | completed | profiler-off | distributed | 267.04 ms | 3,834.67 | 30,677.33 | 0.609 GiB |
| [npu-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-r1-17dbf09f](../../../runs/8-card/ep8/npu-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-r1-17dbf09f/readme.md) | failed | profiler-active | - | - ms | - | - | - GiB |
| [npu-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-r2-7d87cd5f](../../../runs/8-card/ep8/npu-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-r2-7d87cd5f/readme.md) | failed | profiler-active | - | - ms | - | - | - GiB |
| [npu-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-r3-8f7adec9](../../../runs/8-card/ep8/npu-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-r3-8f7adec9/readme.md) | failed | profiler-active | - | - ms | - | - | - GiB |
| [npu-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-r4-ccb6f22c](../../../runs/8-card/ep8/npu-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-r4-ccb6f22c/readme.md) | completed | profiler-active | distributed | 874.31 ms | 1,171.21 | 9,369.64 | - GiB |

## experiment sequence

### 1. npu-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-50c6107d

- Status: `completed`
- Started: `2026-08-24T17:56:02.331131+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/ep8/npu-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-50c6107d/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/ep8/npu-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-50c6107d.html`

```bash
/root/miniconda3/envs/torchtitan-0803/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology ep8 --preset distributed --profiler-off --visible-devices 0,1,2,3,4,5,6,7 --steps 20 --skip-steps 8 --warmup-steps 2 --active-steps 3 --replicate 1
```

### 2. npu-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-r1-17dbf09f

- Status: `failed`
- Started: `2026-08-25T09:06:17.768552+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/ep8/npu-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-r1-17dbf09f/readme.md)

```bash
/root/miniconda3/envs/torchtitan-0803/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology ep8 --preset distributed --visible-devices 0,1,2,3,4,5,6,7 --steps 20 --skip-steps 8 --warmup-steps 2 --active-steps 3 --replicate 1 --cluster
```

### 3. npu-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-r2-7d87cd5f

- Status: `failed`
- Started: `2026-08-25T11:35:48.239630+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/ep8/npu-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-r2-7d87cd5f/readme.md)

```bash
/root/miniconda3/envs/torchtitan-0803/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology ep8 --preset distributed --visible-devices 0,1,2,3,4,5,6,7 --steps 20 --skip-steps 8 --warmup-steps 2 --active-steps 3 --replicate 2 --cluster
```

### 4. npu-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-r3-8f7adec9

- Status: `failed`
- Started: `2026-08-25T11:36:46.450872+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/ep8/npu-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-r3-8f7adec9/readme.md)

```bash
/root/miniconda3/envs/torchtitan-0803/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology ep8 --preset distributed --visible-devices 0,1,2,3,4,5,6,7 --steps 20 --skip-steps 8 --warmup-steps 2 --active-steps 3 --replicate 3 --cluster
```

### 5. npu-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-r4-ccb6f22c

- Status: `completed`
- Started: `2026-08-25T11:40:40.938165+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/ep8/npu-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-r4-ccb6f22c/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/ep8/npu-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-r4-ccb6f22c.html`

```bash
/root/miniconda3/envs/torchtitan-0803/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology ep8 --preset distributed --visible-devices 0,1,2,3,4,5,6,7 --steps 20 --skip-steps 8 --warmup-steps 2 --active-steps 3 --replicate 4 --cluster
```

## current summary

The highest recorded job throughput is 30,677.33 tok/s from `npu-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-50c6107d`. Profiler-off runs are throughput evidence; profiler-active runs are attribution evidence only.

## analysis

- [Detailed topology analysis](analysis.md)
