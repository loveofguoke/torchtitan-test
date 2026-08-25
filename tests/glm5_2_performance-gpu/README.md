# GLM-5.2 GPU performance profiler

This suite is the CUDA counterpart of `tests/glm5_2_performance`. It uses
TorchTitan's profiler lifecycle with `torch.profiler` CPU and CUDA activities,
captures numeric training metrics, parses per-rank Chrome traces, and writes a
self-contained HTML report. It does not import TorchTitanTurbo and does not
depend on Ascend, `msprof-analyze`, or MindStudio.

The directory name intentionally contains a hyphen. Run its entry scripts by
path; do not try to import it as a Python package.

## Quick probe

```bash
export CUDA_VISIBLE_DEVICES=0
python tests/glm5_2_performance-gpu/profiler_probe.py --probe
```

The ten-step probe skips two startup steps, warms up for one step, and records
three steps. Generated output is separated from the NPU suite:

```text
performance_gpu_runs/<topology>/<run>/
  metrics.jsonl
  runtime.log
  trainer_output/profiling/traces/iteration_*/rank*_trace.json
performance_gpu_artifacts/<topology>/<run>/
  manifest.json
  metrics.jsonl
  analysis.json
performance_gpu_reports/<topology>/<run>.html
```

Open `rank*_trace.json` with Perfetto (`ui.perfetto.dev`) or Chrome tracing for
the full CPU/CUDA timeline. The generated report summarizes training metrics,
CUDA kernels, CUDA runtime calls, CPU operators, and NCCL/c10d communication.

## Capture and analyze separately

```bash
export CUDA_VISIBLE_DEVICES=0
python tests/glm5_2_performance-gpu/profiler_benchmark.py --capture
python tests/glm5_2_performance-gpu/profiler_benchmark.py --analyze
```

For an eight-GPU DDP profile:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python tests/glm5_2_performance-gpu/profiler_benchmark.py \
  --probe --topology ddp8 --profile-ranks all
```

Useful options:

- `--profile-ranks 0` profiles only rank 0 and minimizes overhead.
- `--profile-ranks all` is useful for NCCL imbalance investigations.
- `--profile-memory` enables profiler memory events.
- `--with-stack` captures stacks and has substantial overhead.
- `--extra-train-arg=...` passes one additional TorchTitan option.
- `--force` replaces only the exact generated run and artifact directories.

The report's category durations overlap; they are attribution totals, not wall
clock time. Use `metrics.jsonl` for authoritative step time and throughput.
Use Nsight Systems/Compute when system-wide NCCL timelines or hardware-counter
analysis beyond the PyTorch trace is required.
