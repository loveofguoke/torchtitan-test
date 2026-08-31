# pp8 experiments

## topology

| field | value |
| --- | --- |
| world size | 8 |
| dp replicate / shard | 1 / 1 |
| tp / cp / pp / ep | 1 / 1 / 8 / 1 |
| recorded runs | 5 |

## run results

| run | status | mode | preset | median step | tok/s/device | tok/s/job | peak hbm |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| [npu-pp8-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-02473d85](../../../runs/8-card/pp8/npu-pp8-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-02473d85/readme.md) | completed | profiler-off | distributed | 2,037.35 ms | 502.62 | 4,020.93 | 0.018 GiB |
| [npu-pp8-bf16-s20-l8-b64-seq128-seed61-distributed-r1-cab841c5](../../../runs/8-card/pp8/npu-pp8-bf16-s20-l8-b64-seq128-seed61-distributed-r1-cab841c5/readme.md) | failed | profiler-active | - | - ms | - | - | - GiB |
| [npu-pp8-bf16-s20-l8-b64-seq128-seed61-distributed-r2-462adeae](../../../runs/8-card/pp8/npu-pp8-bf16-s20-l8-b64-seq128-seed61-distributed-r2-462adeae/readme.md) | completed | profiler-active | distributed | 2,849.37 ms | 359.38 | 2,875.02 | - GiB |
| [npu-pp8-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-6775b48f](../../../runs/8-card/pp8/npu-pp8-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-6775b48f/readme.md) | completed | profiler-active | overview | 3,716.28 ms | 275.54 | 2,204.35 | - GiB |
| [npu-pp8-bf16-s30-l8-b64-seq128-seed61-distributed-r31-e9147576](../../../runs/8-card/pp8/npu-pp8-bf16-s30-l8-b64-seq128-seed61-distributed-r31-e9147576/readme.md) | completed | profiler-active | distributed | 3,982.32 ms | 257.14 | 2,057.09 | - GiB |

## experiment sequence

### 1. npu-pp8-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-02473d85

- Status: `completed`
- Started: `2026-08-24T17:54:15.561755+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/pp8/npu-pp8-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-02473d85/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/pp8/npu-pp8-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-02473d85.html`

```bash
/root/miniconda3/envs/torchtitan-0803/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology pp8 --preset distributed --profiler-off --visible-devices 0,1,2,3,4,5,6,7 --steps 20 --skip-steps 8 --warmup-steps 2 --active-steps 3 --replicate 1
```

### 2. npu-pp8-bf16-s20-l8-b64-seq128-seed61-distributed-r1-cab841c5

- Status: `failed`
- Started: `2026-08-25T09:10:14.593033+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/pp8/npu-pp8-bf16-s20-l8-b64-seq128-seed61-distributed-r1-cab841c5/readme.md)

```bash
/root/miniconda3/envs/torchtitan-0803/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology pp8 --preset distributed --visible-devices 0,1,2,3,4,5,6,7 --steps 20 --skip-steps 8 --warmup-steps 2 --active-steps 3 --replicate 1 --cluster
```

### 3. npu-pp8-bf16-s20-l8-b64-seq128-seed61-distributed-r2-462adeae

- Status: `completed`
- Started: `2026-08-25T12:16:15.844813+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/pp8/npu-pp8-bf16-s20-l8-b64-seq128-seed61-distributed-r2-462adeae/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/pp8/npu-pp8-bf16-s20-l8-b64-seq128-seed61-distributed-r2-462adeae.html`

```bash
/root/miniconda3/envs/torchtitan-0803/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topologies pp8,fsdp2-tp4,fsdp2-cp4,tp2-cp4,fsdp4-tp2,fsdp2-pp4,fsdp2-tp2-pp2,fsdp2-tp4-ep8 --preset distributed --visible-devices 0,1,2,3,4,5,6,7 --steps 20 --skip-steps 8 --warmup-steps 2 --active-steps 3 --replicate 2 --cluster
```

### 4. npu-pp8-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-6775b48f

- Status: `completed`
- Started: `2026-08-31T10:52:43.386498+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/pp8/npu-pp8-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-6775b48f/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/pp8/npu-pp8-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-6775b48f.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology all --preset all --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all
```

### 5. npu-pp8-bf16-s30-l8-b64-seq128-seed61-distributed-r31-e9147576

- Status: `completed`
- Started: `2026-08-31T14:10:38.597515+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/pp8/npu-pp8-bf16-s30-l8-b64-seq128-seed61-distributed-r31-e9147576/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/pp8/npu-pp8-bf16-s30-l8-b64-seq128-seed61-distributed-r31-e9147576.html`

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology all --preset all --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all
```

## current summary

The highest recorded job throughput is 4,020.93 tok/s from `npu-pp8-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-02473d85`. Profiler-off runs are throughput evidence; profiler-active runs are attribution evidence only.

## analysis

- [Detailed topology analysis](analysis.md)
