# single experiments

## topology

| field | value |
| --- | --- |
| world size | 1 |
| dp replicate / shard | 1 / 1 |
| tp / cp / pp / ep | 1 / 1 / 1 / 1 |
| recorded runs | 6 |

## run results

| run | status | mode | preset | median step | tok/s/device | tok/s/job | peak hbm |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| [npu-single-bf16-s30-l8-b64-seq128-seed61-overview-a7f99b62](../../../runs/1-card/single/npu-single-bf16-s30-l8-b64-seq128-seed61-overview-a7f99b62/readme.md) | command-only | profiler-active | - | - ms | - | - | - GiB |
| [npu-single-bf16-s30-l8-b64-seq128-seed61-overview-offline-bf438d2a](../../../runs/1-card/single/npu-single-bf16-s30-l8-b64-seq128-seed61-overview-offline-bf438d2a/readme.md) | command-only | profiler-active | - | - ms | - | - | - GiB |
| [npu-single-bf16-s30-l8-b64-seq128-seed61-kernel-offline-2499a8d1](../../../runs/1-card/single/npu-single-bf16-s30-l8-b64-seq128-seed61-kernel-offline-2499a8d1/readme.md) | command-only | profiler-active | - | - ms | - | - | - GiB |
| [npu-single-bf16-s30-l8-b64-seq128-seed61-overview-af49a08e](../../../runs/1-card/single/npu-single-bf16-s30-l8-b64-seq128-seed61-overview-af49a08e/readme.md) | completed | profiler-active | overview | 1,839.62 ms | 4,453.08 | 4,453.08 | - GiB |
| [npu-single-bf16-s30-l8-b64-seq128-seed61-overview-offline-b14bdd6a](../../../runs/1-card/single/npu-single-bf16-s30-l8-b64-seq128-seed61-overview-offline-b14bdd6a/readme.md) | completed | profiler-active | overview | 1,739.46 ms | 4,709.52 | 4,709.52 | - GiB |
| [npu-single-bf16-s30-l8-b64-seq128-seed61-kernel-offline-2ed8313b](../../../runs/1-card/single/npu-single-bf16-s30-l8-b64-seq128-seed61-kernel-offline-2ed8313b/readme.md) | completed | profiler-active | kernel | 1,747.83 ms | 4,686.94 | 4,686.94 | - GiB |

## experiment sequence

### 1. npu-single-bf16-s30-l8-b64-seq128-seed61-overview-a7f99b62

- Status: `command-only`
- Started: `2026-08-24T16:31:51.795492+08:00`
- Full process and outputs: [run readme](../../../runs/1-card/single/npu-single-bf16-s30-l8-b64-seq128-seed61-overview-a7f99b62/readme.md)

```bash
/root/miniconda3/envs/torchtitan-0803/bin/python tests/glm5_2_performance/profiler_benchmark.py --analyze --device npu --topology single --preset overview --visible-devices 1
```

### 2. npu-single-bf16-s30-l8-b64-seq128-seed61-overview-offline-bf438d2a

- Status: `command-only`
- Started: `2026-08-24T16:31:57.043252+08:00`
- Full process and outputs: [run readme](../../../runs/1-card/single/npu-single-bf16-s30-l8-b64-seq128-seed61-overview-offline-bf438d2a/readme.md)

```bash
/root/miniconda3/envs/torchtitan-0803/bin/python tests/glm5_2_performance/profiler_benchmark.py --analyze --device npu --topology single --preset overview --offline --visible-devices 1
```

### 3. npu-single-bf16-s30-l8-b64-seq128-seed61-kernel-offline-2499a8d1

- Status: `command-only`
- Started: `2026-08-24T16:32:02.286244+08:00`
- Full process and outputs: [run readme](../../../runs/1-card/single/npu-single-bf16-s30-l8-b64-seq128-seed61-kernel-offline-2499a8d1/readme.md)

```bash
/root/miniconda3/envs/torchtitan-0803/bin/python tests/glm5_2_performance/profiler_benchmark.py --analyze --device npu --topology single --preset kernel --offline --visible-devices 1
```

### 4. npu-single-bf16-s30-l8-b64-seq128-seed61-overview-af49a08e

- Status: `completed`
- Started: `2026-08-24T17:13:47.115347+08:00`
- Full process and outputs: [run readme](../../../runs/1-card/single/npu-single-bf16-s30-l8-b64-seq128-seed61-overview-af49a08e/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/1-card/single/npu-single-bf16-s30-l8-b64-seq128-seed61-overview-af49a08e.html`

```bash
/root/miniconda3/envs/torchtitan-0803/bin/python tests/glm5_2_performance/profiler_benchmark.py --analyze --device npu --topology single --preset overview --visible-devices 1
```

### 5. npu-single-bf16-s30-l8-b64-seq128-seed61-overview-offline-b14bdd6a

- Status: `completed`
- Started: `2026-08-24T17:13:52.411250+08:00`
- Full process and outputs: [run readme](../../../runs/1-card/single/npu-single-bf16-s30-l8-b64-seq128-seed61-overview-offline-b14bdd6a/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/1-card/single/npu-single-bf16-s30-l8-b64-seq128-seed61-overview-offline-b14bdd6a.html`

```bash
/root/miniconda3/envs/torchtitan-0803/bin/python tests/glm5_2_performance/profiler_benchmark.py --analyze --device npu --topology single --preset overview --offline --visible-devices 1
```

### 6. npu-single-bf16-s30-l8-b64-seq128-seed61-kernel-offline-2ed8313b

- Status: `completed`
- Started: `2026-08-24T17:14:03.421139+08:00`
- Full process and outputs: [run readme](../../../runs/1-card/single/npu-single-bf16-s30-l8-b64-seq128-seed61-kernel-offline-2ed8313b/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/1-card/single/npu-single-bf16-s30-l8-b64-seq128-seed61-kernel-offline-2ed8313b.html`

```bash
/root/miniconda3/envs/torchtitan-0803/bin/python tests/glm5_2_performance/profiler_benchmark.py --analyze --device npu --topology single --preset kernel --offline --visible-devices 1
```

## current summary

The highest recorded job throughput is 4,709.52 tok/s from `npu-single-bf16-s30-l8-b64-seq128-seed61-overview-offline-b14bdd6a`. Profiler-off runs are throughput evidence; profiler-active runs are attribution evidence only.
