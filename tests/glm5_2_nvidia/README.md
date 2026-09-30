# GLM NVIDIA GPU performance experiments

This package is the NVIDIA GPU counterpart of the NPU MindStudio experiment
family. It reuses the canonical GLM topology and token-budget definitions, but
its collection and analysis are independent from the NPU path. It does not
modify TorchTitan or TorchTitanTurbo.

The package is named after the NVIDIA experiment family instead of one tool.
Nsight Systems is the system collector. Nsight Compute is implemented as a
separate, targeted kernel-analysis workflow with its own replay and lifecycle
contract. Nsight Systems and Nsight Compute outputs
remain separate because they answer different questions and use different file
formats.

The shared output namespace follows the same category split as MindStudio:

```text
nvidia_{runs,artifacts,reports}/
  accuracy/             # future NVIDIA-side accuracy endpoints and reports
  performance/
    system/             # Nsight Systems, implemented here
    operator/           # Nsight Compute, reserved until implemented
  graph/                # eager/torch.compile capture and compiler evidence
```

For CUDA `torch.compile` experiments, PyTorch Inductor normally uses Triton for
GPU kernel code generation. A formal graph run must still record the resolved
compiler/backend and generated evidence; the directory name or an ambient
default is not sufficient proof that a specific region used Triton.

The implementation follows NVIDIA's standard PyTorch workflow:

1. `nsys profile` captures CUDA, NVTX, OS runtime, cuBLAS, cuDNN, PyTorch
   function/shape, autograd, CUDA memory, and CUDA Graph events;
2. `nsys export` produces SQLite for programmatic analysis;
3. `nsys stats` produces CSV summaries for CUDA APIs, kernels, memory
   operations, NVTX ranges, and OS runtime calls;
4. the `.nsys-rep` remains the authoritative input for the interactive
   Nsight Systems UI timeline;
5. deterministic local triage reads only official CSV outputs and produces
   evidence-linked findings and next actions;
6. `kernel_benchmark.py` reruns narrowly selected kernels under Nsight
   Compute after the system trace has identified it.

This separation is intentional. Nsight Compute replays kernels, can be very
slow, and is not a safe automatic continuation of a distributed training run.
An automatic finding is triage evidence, never a pass/fail verdict.

Official references:

- [Nsight Systems getting started](https://developer.nvidia.com/nsight-systems/get-started)
- [Nsight Systems user guide](https://docs.nvidia.com/nsight-systems/UserGuide/)
- [Post-collection analysis guide](https://docs.nvidia.com/nsight-systems/AnalysisGuide/)
- [Nsight Compute CLI](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html)
- [Nsight Compute profiling guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)
- [MLPerf Training rules](https://github.com/mlcommons/training_policies/blob/master/training_rules.adoc)
- [MLPerf result messaging guidelines](https://github.com/mlcommons/policies/blob/master/MLPerf_Results_Messaging_Guidelines.adoc)

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
python tests/glm5_2_nvidia/performance_benchmark.py \
  --probe --topology single
```

One distributed topology (replace `fsdp8` with any common topology):

```bash
python tests/glm5_2_nvidia/performance_benchmark.py \
  --probe --topology fsdp8
```

Every registered topology through eight GPUs:

```bash
python tests/glm5_2_nvidia/performance_benchmark.py \
  --probe --topology all
```

## Layered diagnosis

First collect at least three profiler-off repeats. These runs are the authority
for step time, throughput, TFLOPS, MFU, and peak memory; Nsys and NCU explain
causes but do not replace the uninstrumented measurement:

```bash
for REPEAT in 1 2 3; do
  python tests/glm5_2_nvidia/performance_benchmark.py \
    --probe --profiler-off --replicate "${REPEAT}" \
    --topology fsdp8 --steps 50
done
```

Start with the standard pass. It captures CUDA/PyTorch/NVTX, memory allocation,
kernel, API, and OS-runtime evidence. The current integration profiles the
selected training process; use `--delay`/`--duration` when a bounded wall-clock
window is needed. Exact step-controlled `cudaProfilerApi` capture is intentionally
not claimed until the TorchTitan training loop owns matching start/stop calls:

```bash
python tests/glm5_2_nvidia/performance_benchmark.py \
  --probe --profile standard --topology ddp2
```

Read `diagnosis/self/diagnosis.md` and open `profile.nsys-rep`. Rerun only the
relevant deeper policy; each policy has a distinct capture identity:

```bash
# Collective arrival, NCCL execution, rank skew, and compute overlap.
python tests/glm5_2_nvidia/performance_benchmark.py \
  --probe --profile communication --topology ddp2

# CUDA synchronization, GPU gaps, and CPU process-tree attribution.
# This requires a working perf_event environment.
python tests/glm5_2_nvidia/performance_benchmark.py \
  --probe --profile host --topology single

# Allocation timeline and synchronous/asynchronous memory-copy evidence.
python tests/glm5_2_nvidia/performance_benchmark.py \
  --probe --profile memory --topology single

# Expensive combined capture; use only after the standard pass justifies it.
python tests/glm5_2_nvidia/performance_benchmark.py \
  --probe --profile deep --topology ddp2
```

After a stable CUDA kernel is selected from the Nsys timeline/statistics, run
NCU narrowly. Begin with `basic`; use `detailed` or `full` only when needed:

```bash
python tests/glm5_2_nvidia/kernel_benchmark.py \
  --topology single \
  --kernel-name 'regex:.*gemm.*' \
  --launch-skip 0 --launch-count 1 \
  --set basic
```

NCU accepts every registered topology and uses `--target-processes=all` plus
the official `%i` report placeholder so child-rank reports cannot overwrite one
another. Distributed stragglers and NCCL timing must still be diagnosed using
the communication Nsys capture: NCU serializes/replays selected kernels, so its
timestamps are not valid evidence of natural rank skew or communication overlap.

Capture and analysis may be resumed independently:

```bash
python tests/glm5_2_nvidia/performance_benchmark.py \
  --capture --topology ddp2
python tests/glm5_2_nvidia/performance_benchmark.py \
  --analyze --topology ddp2
```

For a newer NCCL that supports Nsight Systems advanced NCCL tracing, add it
explicitly instead of silently changing the compatible default:

```bash
python tests/glm5_2_nvidia/performance_benchmark.py \
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
| `--skip-steps` | Warmup steps excluded from profiler-off steady-state conclusions. | `10` |
| `--local-batch-size` / `--global-batch-size` | Shared sample-batch contract. | `8` / `64` |
| `--sequence-length` | Tokens per sample. | `128` |
| `--trace` | Nsight Systems trace domains. Add `nccl` only when supported by the installed NCCL. | `cuda,nvtx,osrt,cublas,cudnn` |
| `--profile` | Collection policy: `standard`, `communication`, `host`, `memory`, or `deep`. | `standard` |
| `--profiler-off` | Run normal training without Nsys; this is the performance-number baseline. | disabled |
| `--replicate` | Optional independent run index in the identity. Use 1, 2, 3 for formal baselines. | `0` |
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
nvidia_runs/performance/system/<N-card>/<topology>/<experiment>/
  # profiler-off members use runtime.log + metrics.jsonl
  nsys_profile.log
  training.log
  trainer_output/profiling/nsys/
    profile.nsys-rep
    profile.sqlite
    export_sqlite.log
    stats/*.csv
    stats/*.log
  diagnosis/self/
    diagnosis.json
    diagnosis.md
  run_state.json

nvidia_artifacts/performance/system/<N-card>/<topology>/<experiment>/
  manifest.json

nvidia_reports/performance/system/<N-card>/<topology>/<experiment>.html

nvidia_runs/performance/operator/<N-card>/<topology>/<experiment>/
  ncu_profile.log
  trainer_output/profiling/ncu/profile-*.ncu-rep

nvidia_artifacts/performance/operator/<N-card>/<topology>/<experiment>/
  manifest.json

nvidia_reports/performance/operator/<N-card>/<topology>/<experiment>.html

# Shared NPU/GPU or before/after profiler-off aggregation:
<selected-output>/
  README.md
  comparison.json
  comparison.html
```

The system report is one self-contained interactive HTML. It contains the
per-step training curves, warmup/steady-state split, summary tables, diagnostic
branches, and an inventory linking the native `.nsys-rep`, SQLite database,
official CSV statistics, metrics JSONL, and logs. The HTML does not replace the
native Nsight Systems timeline; it provides the portable reading entry point.
The shared comparison HTML overlays all repeats and supports hover, zoom,
filtering, and candidate/reference relative-change inspection.

Aggregate one platform or compare two contract-compatible groups after capture:

```bash
# Single-platform repeat stability.
python -m tests.glm5_2_performance.comparison \
  --reference-label GPU \
  --reference-run /path/to/gpu-r1 \
  --reference-run /path/to/gpu-r2 \
  --reference-run /path/to/gpu-r3 \
  --skip-steps 10 \
  --output nvidia_reports/performance/comparisons/gpu-fsdp8

# GPU reference versus NPU candidate. Both groups must have the same model,
# topology, batch, sequence, seed, dtype, and mixed-precision contract.
python -m tests.glm5_2_performance.comparison \
  --reference-label GPU --reference-run /path/to/gpu-r1 \
  --reference-run /path/to/gpu-r2 --reference-run /path/to/gpu-r3 \
  --candidate-label NPU --candidate-run /path/to/npu-r1 \
  --candidate-run /path/to/npu-r2 --candidate-run /path/to/npu-r3 \
  --skip-steps 10 \
  --output performance_reports/comparisons/gpu-npu-fsdp8
```

The `nvidia_artifacts/performance/system` tree contains only orchestration metadata. Official
Nsight capture and export products stay together under the run's
`trainer_output/profiling/nsys/` directory. Existing captures made with the
older artifact-owned layout or top-level `nsys_*` roots are moved there
automatically after their bytes are checked; no profiling rerun is required.

Open `trainer_output/profiling/nsys/profile.nsys-rep` in Nsight Systems UI to inspect CPU/PyTorch/NVTX,
CUDA API launches, GPU streams, kernels, memory operations, and NCCL events
when collected. The CSV files are lightweight inputs for Codex and automated
GPU/NPU comparison. Profiler-active timings are attribution evidence, not an
unperturbed throughput baseline; performance claims still require repeated
profiler-off runs with the same model, topology, batch, sequence, and dtype.

## Automatic diagnosis boundary

The local diagnosis indexes top CUDA APIs, kernels, memory operations, and
NVTX ranges. Synchronization APIs, NCCL kernels, memory operations, and kernel
hotspots create explicit investigation branches with source rows and next-step
commands. It does not infer that a long NCCL kernel means the network is slow:
the rank may simply have arrived early. It likewise does not infer tensor or
allocator ownership from CUDA allocation events; use a PyTorch memory snapshot
for that question.

Nsight Systems `stats`/SQLite and Nsight Compute `.ncu-rep` remain the source of
truth. NVIDIA's Nsys recipes and NCU rules can be added as official derived
stages after their installed-version command and output contracts are validated;
their data must not be rewritten into invented formats.

## MLPerf boundary

MLPerf is not a profiler and this GLM experiment is not an MLPerf submission.
The framework adopts its measurement discipline only: declare the system and
software configuration, preserve complete logs, keep preprocessing and quality
semantics explicit, use multiple independent runs, and do not compare unlike
benchmarks or scenarios. Local step-time, throughput, MFU, Nsys, or NCU results
must therefore be described as project measurements. They must not be labelled
as MLPerf results unless the exact MLCommons benchmark, rules, checker, required
run count, target quality, and submission package have all been satisfied.
