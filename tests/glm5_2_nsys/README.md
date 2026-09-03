# GLM CUDA Nsight Systems performance probe

This experiment is the GPU counterpart of the NPU MindStudio performance
workflow. It reuses the canonical GLM topology and token-budget definitions,
but its collection and analysis are completely independent from the NPU path.
It does not modify TorchTitan or TorchTitanTurbo.

The first implementation follows NVIDIA's standard PyTorch workflow:

1. `nsys profile` captures CUDA, NVTX, OS runtime, cuBLAS, cuDNN, PyTorch
   function/shape, autograd, CUDA memory, and CUDA Graph events;
2. `nsys export` produces SQLite for programmatic analysis;
3. `nsys stats` produces CSV summaries for CUDA APIs, kernels, memory
   operations, NVTX ranges, and OS runtime calls;
4. the `.nsys-rep` remains the authoritative input for the interactive
   Nsight Systems UI timeline.

Official references:

- [Nsight Systems getting started](https://developer.nvidia.com/nsight-systems/get-started)
- [Nsight Systems user guide](https://docs.nvidia.com/nsight-systems/UserGuide/)
- [Post-collection analysis guide](https://docs.nvidia.com/nsight-systems/AnalysisGuide/)

## Preflight

```bash
nsys --version
nsys status --environment
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
```

The Nsight Systems target CLI must run inside the same container/environment as
TorchTitan. No additional Python package is required for the standard path.

## Standard commands

Single GPU:

```bash
python tests/glm5_2_nsys/performance_benchmark.py \
  --probe --topology single
```

One distributed topology (replace `fsdp8` with any common topology):

```bash
python tests/glm5_2_nsys/performance_benchmark.py \
  --probe --topology fsdp8
```

Every registered topology through eight GPUs:

```bash
python tests/glm5_2_nsys/performance_benchmark.py \
  --probe --topology all
```

Capture and analysis may be resumed independently:

```bash
python tests/glm5_2_nsys/performance_benchmark.py \
  --capture --topology ddp2
python tests/glm5_2_nsys/performance_benchmark.py \
  --analyze --topology ddp2
```

For a newer NCCL that supports Nsight Systems advanced NCCL tracing, add it
explicitly instead of silently changing the compatible default:

```bash
python tests/glm5_2_nsys/performance_benchmark.py \
  --probe --topology ddp2 \
  --trace cuda,nvtx,osrt,cublas,cudnn,nccl
```

## Main options

| Option | Meaning | Default |
|---|---|---|
| `--capture` / `--analyze` / `--probe` | Capture only, analyze an existing capture, or do both. No action flag means probe. | probe |
| `--topology` | One common topology or `all`. | `single` |
| `--topologies` | Comma-separated topology subset. | unset |
| `--steps` | Training optimizer steps. | `30` |
| `--local-batch-size` / `--global-batch-size` | Shared sample-batch contract. | `8` / `64` |
| `--sequence-length` | Tokens per sample. | `128` |
| `--trace` | Nsight Systems trace domains. Add `nccl` only when supported by the installed NCCL. | `cuda,nvtx,osrt,cublas,cudnn` |
| `--pytorch` | Automatic PyTorch NVTX annotations. | `functions-trace-shapes,autograd-nvtx` |
| `--sample` | CPU sampling scope; `none` avoids permission and overhead surprises in the standard run. | `none` |
| `--cuda-memory-usage` | Record CUDA allocation/residency information. | enabled |
| `--cuda-graph-trace` | CUDA Graph granularity. | `node` |
| `--delay` / `--duration` | Optional collection time window in seconds. | unset |
| `--stats-reports` | Comma-separated `nsys stats` reports. | six standard summaries |
| `--force` | Remove all selected old run/artifact/report generations before capture. | disabled |
| `--dry-run` | Validate selection and print resolved identities without invoking Nsight Systems. | disabled |

Without `--force`, an exact completed capture/analysis is skipped. A mismatched
or incomplete selected member is reset before capture, so new and old data are
never mixed. With `--force`, every selected member is removed before the first
capture; after an interruption, rerun without `--force` to continue unfinished
members while retaining completed ones.

## Outputs

```text
nsys_runs/<N-card>/<topology>/<experiment>/
  nsys_profile.log
  training.log
  trainer_output/profiling/nsys/
    profile.nsys-rep
    profile.sqlite
    export_sqlite.log
    stats/*.csv
    stats/*.log
  run_state.json

nsys_artifacts/<N-card>/<topology>/<experiment>/
  manifest.json

nsys_reports/<N-card>/<topology>/<experiment>.html
```

The `nsys_artifacts` tree contains only orchestration metadata. Official
Nsight capture and export products stay together under the run's
`trainer_output/profiling/nsys/` directory. Existing captures made with the
older artifact-owned layout are moved there automatically after their bytes
are checked; no profiling rerun is required.

Open `trainer_output/profiling/nsys/profile.nsys-rep` in Nsight Systems UI to inspect CPU/PyTorch/NVTX,
CUDA API launches, GPU streams, kernels, memory operations, and NCCL events
when collected. The CSV files are lightweight inputs for Codex and automated
GPU/NPU comparison. Profiler-active timings are attribution evidence, not an
unperturbed throughput baseline; performance claims still require repeated
profiler-off runs with the same model, topology, batch, sequence, and dtype.
