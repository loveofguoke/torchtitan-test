# GLM-5.2 performance profiler

This directory is independent from `tests/glm5_2_precision`. It does not
replace the precision data loader, checkpoint, metrics, or PASS/FAIL logic.
Profiling is enabled only for commands launched by this directory.

The workflow is deliberately top-down:

1. `overview`: Level0 timing with low collection overhead.
2. `distributed`: Level1 on every rank with pipe-utilization metrics.
3. `kernel`: Level1 shapes, arithmetic utilization, and L2 information.
4. `runtime`: Level2 stack, memory, operator attributes, and shapes.

Start at `overview`. Move to a deeper preset only when the previous report
locates the bottleneck below that layer.

## Quick visual probe

The probe runs ten tiny training steps, profiles a three-step window, and
immediately writes a self-contained HTML report.

Ascend NPU:

```bash
conda activate torchtitan
export ASCEND_RT_VISIBLE_DEVICES=4
python tests/glm5_2_performance/profiler_probe.py --probe
```

CUDA GPU:

```bash
conda activate torchtitan
export CUDA_VISIBLE_DEVICES=4
python tests/glm5_2_performance/profiler_probe.py --probe
```

The device comes from the exported visibility variable. Use `--device npu`
or `--device cuda` only when both variables exist in one environment.

Generated paths are short and explicit:

```text
performance_runs/glm5-probe-npu-single-overview/       # raw heavy data
performance_artifacts/glm5-probe-npu-single-overview/  # manifest + summary
performance_reports/glm5-probe-npu-single-overview.html
```

`performance_runs/` is ignored by Git. Artifacts and reports are portable and
are not ignored by this repository.

## Capture and analyze separately

For a normal single-card capture:

```bash
export ASCEND_RT_VISIBLE_DEVICES=4
python tests/glm5_2_performance/profiler_benchmark.py \
  --capture --topology single --preset overview

python tests/glm5_2_performance/profiler_benchmark.py \
  --analyze --device npu --topology single --preset overview
```

For an eight-card communication probe:

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python tests/glm5_2_performance/profiler_benchmark.py \
  --probe --topology fsdp8 --preset distributed
```

Available topology names match the precision experiment vocabulary:
`single`, `ddp2`, `ddp8`, `fsdp8`, `tp8`, `pp8`, `ep8`,
`fsdp2-tp4`, `fsdp4-tp2`, `fsdp2-pp4`, `fsdp2-tp2-pp2`, and
`fsdp2-tp4-ep8`.

The command rejects a topology when fewer visible local devices are exported.
A failed partial run is archived as `.failed-TIMESTAMP` and the next capture
can retry without manual deletion. A completed artifact is skipped unless
`--force` is passed.

## Offline Ascend parsing and advisor

Online parsing is convenient for the probe. For lower runtime interference,
capture raw Ascend data and parse it after training:

```bash
export ASCEND_RT_VISIBLE_DEVICES=4
python tests/glm5_2_performance/profiler_benchmark.py \
  --capture --topology single --preset kernel --offline

python tests/glm5_2_performance/profiler_benchmark.py \
  --analyze --device npu --topology single --preset kernel \
  --offline-parse --advisor
```

`--offline-parse` calls `torch_npu.profiler.profiler.analyse` for each captured
`*_ascend_pt` directory. `--advisor` invokes:

```bash
msprof-analyze advisor all -d PROFILER_DIR -o ADVISOR_DIR
```

The advisor option is explicit: missing `msprof-analyze` is reported as an
error instead of silently changing the analysis.

## Detailed trace viewers

The generated HTML is the first-level dashboard. It includes training step
time, throughput, TFLOPS, MFU/memory summaries, profiler file inventory, and
recognized operator/kernel duration CSVs. The blue region on time-series
charts is the actual active profile window.

Use the raw outputs for interactive event-level inspection:

```bash
tensorboard --logdir performance_runs/RUN/trainer_output/profiling/traces
```

Chrome trace JSON can also be opened in Perfetto. Ascend database and CSV
outputs remain available to msprof and msprof-analyze.

This is a bounded in-process probe, not an external sampler attached to an
arbitrary PID. It observes the real training process at a scheduled step
window, which gives reliable operator correlation while keeping overhead
controlled.
