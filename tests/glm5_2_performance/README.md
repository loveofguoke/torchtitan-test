# GLM-5.2 performance profiler

This directory is independent from `tests/glm5_2_precision`. It does not
replace the precision data loader, checkpoint, metrics, or PASS/FAIL logic.
Profiling is enabled only for commands launched by this directory.

For opt-in Full DSA compute, communication, memory, and recomputation A/B
profiles, see [OPTIMIZATION.md](OPTIMIZATION.md).
For a Chinese explanation of the implemented profiles, metrics, commands, and
interview concepts, see [性能优化实验与面试速查.md](性能优化实验与面试速查.md).

Current executable support is Ascend NPU only. The CLI retains `--device cuda`
as a reserved interface, but raises `NotImplementedError` until the CUDA
`torch.profiler` activities, parsing, and report contract are defined. Use the
formal precision runner for CUDA/NPU eager accuracy and the NPU combination
runner for graph-aware performance comparison.

Exploratory NPU measurements are deliberately separated from this formal
contract. See the [exploration index](explorations/index.md), the
[cross-topology analysis](explorations/reports/summary.md), and the
[1/2/4/8-card report index](explorations/reports/index.md). Each topology owns
an `experiment.md`, and each immutable run owns a readable `readme.md` beside
its machine-readable evidence.
The evidence-backed candidate work is decomposed in
[explorations/optimization_backlog.md](explorations/optimization_backlog.md);
none of those tasks becomes a benchmark option until it passes its promotion
gates.
Those reports follow the three-stage Ascend
Profiler → `msprof-analyze` → MindStudio Insight workflow and do not redefine
the acceptance topology matrix.

The workflow is deliberately top-down:

1. `overview`: Level0 timing with low collection overhead.
2. `comparison`: Level0 with comparison-safe low-overhead controls.
3. `standard`: Level1 pipe-utilization detail on rank 0.
4. `distributed`: Level1 on every rank with pipe-utilization and interconnect
   data; raw data is parsed offline by default.
5. `kernel`: Level1 shapes, arithmetic utilization, and L2 information.
6. `runtime`: Level2 stack, module hierarchy, memory, operator attributes,
   host CPU/memory, and shapes.

Start at `overview`. Move to a deeper preset only when the previous report
locates the bottleneck below that layer.

## Command-line parameters

The main entry point is `profiler_benchmark.py`. Values shown below are its
defaults; `profiler_probe.py` uses the same CLI but overrides the training
profile to 10 steps, skip 2, warmup 1, active 3, local/global batch 1, and
sequence length 32.

| Parameter | Purpose and choices | `profiler_benchmark.py` default |
|---|---|---|
| `--probe` | Capture and immediately analyze selected topologies. | no action; one action is required |
| `--capture` | Collect the bounded Ascend profile and runtime data only. | no action; one action is required |
| `--analyze` | Parse existing output and generate reports without training. | no action; one action is required |
| `--device` | `auto`, `cuda`, or `npu`. `cuda` is reserved and currently raises `NotImplementedError`. | `auto`, resolved from visibility variables |
| `--topology` | One common topology or `all`. | `single` |
| `--topologies` | Comma-separated subset such as `single,fsdp8,cp8`; mutually exclusive with `--topology`. | unset |
| `--preset` | `overview`, `comparison`, `standard`, `distributed`, `kernel`, or `runtime`. | `overview` |
| `--visible-devices` | Override `ASCEND_RT_VISIBLE_DEVICES` for automation. Environment export is preferred. | unset |
| `--steps` | Total training optimizer steps. Must reach the full profiler schedule. | `30` |
| `--skip-steps` | Training steps before Profiler warmup. | `10` |
| `--warmup-steps` | Profiler warmup steps. | `2` |
| `--active-steps` | Profiler recording steps. | `5` |
| `--local-batch-size` | Per-DP-rank local batch. | `8` |
| `--global-batch-size` | Global samples per optimizer step. | `64` |
| `--sequence-length` | Tokens per sample. | `128` |
| `--run-root` | Raw capture root. Put this on node-local storage for large profiles. | `performance_runs` |
| `--parse-mode` | `sync`, `async`, or `offline`; overrides the preset parse mode. | preset-specific (`distributed` uses `offline`, others `sync`) |
| `--offline` | Compatibility alias for `--parse-mode offline`. | disabled |
| `--offline-parse` | Run the official offline parser during analysis. | disabled |
| `--parse-workers` | Process limit passed to offline parsing. | torch_npu parser default |
| `--advisor` | Run `msprof-analyze advisor all` after parsing. | disabled |
| `--cluster` | Run `msprof-analyze cluster -m all`; intended for multi-rank captures. | disabled |
| `--compare-baseline` | Baseline profiler path passed to `msprof-analyze compare`. | unset |
| `--extra-train-arg` | Append one raw TorchTitan argument; repeat the flag for multiple arguments. | none |
| `--force` | Start a new generation for the selected topology range. All selected captures are removed before execution; rerun without it after interruption to resume. | disabled |

The fixed seed is 61, parameter dtype is BF16, reduction/training master dtype
is FP32, and deterministic algorithms are disabled in the profiler benchmark
to avoid measuring a different execution policy. These values live in
`PerformanceConfig` rather than runtime flags.

## Report contents

Each HTML report identifies the device, topology, preset, collection schedule,
software/source metadata, and raw-profile location. It summarizes step time,
throughput, model FLOPS utilization when available, memory, and the exact active
Profiler window; charts mark that active window instead of presenting startup
as steady-state performance. It also inventories parsed operator/kernel files
and recognized duration tables, records storage/time warnings, and links the
official `msprof-analyze` advisor, compare, or cluster outputs when requested.

This suite is diagnostic and does not impose a universal performance PASS/FAIL
threshold. A formal regression decision needs an explicit baseline and target
outside the profiler preset. Use the combination report when reference and
candidate throughput must be shown together with precision acceptance.

Topology definitions come from `tests/glm5_2_common`. The benchmark supports
`--topology all` and `--topologies single,fsdp8,cp8` so one command can capture
or analyze every selected topology sequentially. The all-topology benchmark
profile uses local batch 8 and global batch 64; the tiny probe intentionally
remains a single-card smoke test.

## Quick visual probe

The probe runs ten tiny training steps, profiles a three-step window, and
immediately writes a self-contained HTML report.

Ascend NPU:

```bash
conda activate torchtitan
export ASCEND_RT_VISIBLE_DEVICES=4
python tests/glm5_2_performance/profiler_probe.py --probe
```

The device comes from `ASCEND_RT_VISIBLE_DEVICES`. Use `--device npu` only when
both CUDA and NPU visibility variables exist in one environment.

Generated paths are short and explicit:

```text
performance_runs/glm5-probe-npu-single-overview/       # raw heavy data
performance_artifacts/glm5-probe-npu-single-overview/  # manifest + summary
performance_reports/glm5-probe-npu-single-overview.html
```

All generated performance roots are ignored by Git. Transfer portable
artifacts/reports, and raw runs when official profiler data is needed, with
`release_artifacts.py`.

Do not enable msProbe/precision tensor dumping in the same training process as
profiling. The official profiler guide notes that dump hooks and tensor writes
distort performance metrics. Precision and performance remain separate
experiments in this repository for that reason.

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

Available topology names match the common precision vocabulary:
`single`, `ddp2`, `ddp8`, `fsdp8`, `tp8`, `cp8`, `pp8`, `ep8`,
`fsdp2-tp4`, `fsdp2-cp4`, `tp2-cp4`, `fsdp4-tp2`, `fsdp2-pp4`,
`fsdp2-tp2-pp2`, and `fsdp2-tp4-ep8`.

The command rejects a topology when fewer visible local devices are exported.
A failed partial run is archived as `.failed-TIMESTAMP` and the next capture
can retry without manual deletion. A completed artifact is skipped unless
`--force` is passed.

To capture and analyze every registered topology through eight ranks in one
invocation, use the same probe stage with `all`:

```bash
unset CUDA_VISIBLE_DEVICES
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python tests/glm5_2_performance/profiler_benchmark.py \
  --probe --device npu --topology all --preset distributed
```

For formal collection/analyze separation, replace each `--probe` with these two
commands while keeping the topology and preset identical:

```bash
python tests/glm5_2_performance/profiler_benchmark.py \
  --capture --device npu --topology fsdp8 --preset distributed
python tests/glm5_2_performance/profiler_benchmark.py \
  --analyze --device npu --topology fsdp8 --preset distributed \
  --offline-parse --cluster
```

CUDA performance profiling is a reserved interface. For example,
`--device cuda --topology single` currently raises `NotImplementedError`
instead of silently producing an incomparable report.

The capture records umask, user privilege, free space, output size, and CANN
environment markers. For large multi-rank jobs, place raw output on node-local
storage to avoid shared-storage write amplification:

```bash
python tests/glm5_2_performance/profiler_benchmark.py \
  --capture --topology fsdp8 --preset distributed \
  --run-root /data/local_profile_runs
```

Use the same `--run-root` during `--analyze`. The report warns when the active
profile window exceeds five minutes or the pre-capture free space is below 20
times the resulting run size. These implement official operating guidance;
they do not estimate storage before the first representative capture.

## Ascend parsing modes

The framework follows the official Ascend PyTorch Profiler modes:

- `sync`: parse when a scheduled trace becomes ready. This is simplest, but
  parsing extends the training process and can distort the profiled boundary.
- `async`: parse in a background process. This reduces blocking, but parsed
  outputs may finish after the training process.
- `offline`: collect raw data during training and parse it afterwards. This is
  the default for the all-rank `distributed` preset and the recommended formal
  cluster workflow.

Select a mode explicitly with `--parse-mode`. `--offline` remains a short
compatibility alias for `--parse-mode offline`.

```bash
export ASCEND_RT_VISIBLE_DEVICES=4
python tests/glm5_2_performance/profiler_benchmark.py \
  --capture --topology single --preset kernel --parse-mode offline

python tests/glm5_2_performance/profiler_benchmark.py \
  --analyze --device npu --topology single --preset kernel \
  --offline-parse --advisor
```

`--offline-parse` calls `torch_npu.profiler.profiler.analyse` for the capture
root containing all `*_ascend_pt` directories. All ranks are submitted in one call so the
official parser can process them concurrently. Use `--parse-workers N` to set
its process limit. `--advisor` invokes:

```bash
msprof-analyze advisor all -d PROFILER_DIR -o ADVISOR_DIR
```

The advisor option is explicit: missing `msprof-analyze` is reported as an
error instead of silently changing the analysis.

## Official diagnosis, comparison, and cluster analysis

The HTML report is the experiment summary. The official deep-analysis entry
points remain explicit CLI actions:

```bash
# Single-run tuning suggestions
python tests/glm5_2_performance/profiler_benchmark.py \
  --analyze --device npu --topology single --preset standard --advisor

# Multi-rank communication summary and cluster_analysis_output
python tests/glm5_2_performance/profiler_benchmark.py \
  --analyze --device npu --topology fsdp8 --preset distributed \
  --offline-parse --cluster

# NPU/NPU performance comparison
python tests/glm5_2_performance/profiler_benchmark.py \
  --analyze --device npu --topology single --preset comparison \
  --compare-baseline /path/to/baseline/profile/or/gpu_trace.json
```

These actions invoke, respectively:

```text
msprof-analyze advisor all -d PROFILE_DIR -o ADVISOR_DIR
msprof-analyze cluster -m all -d PROFILE_DIR -o CLUSTER_DIR
msprof-analyze compare -d PROFILE -bp BASELINE -o COMPARE_DIR
```

## MindStudio Insight handoff

The generated HTML is the first-level dashboard. It includes training step
time, throughput, TFLOPS, MFU/memory summaries, profiler file inventory, and
recognized operator/kernel duration CSVs. It also separates startup, steady
baseline, profiler warmup, active collection, synchronous parse boundary, and
post-profile steps so parser stalls do not pollute the steady-state summary.
The blue region on time-series charts is the actual active profile window.

Profiler-control overrides are part of the run identity. For example,
`overview` with `sync` parsing and `overview` with `offline` parsing use
different run/artifact/report paths and can be compared without `--force` or
overwriting either capture.

For Ascend data, copy the complete `*_ascend_pt` directory to local storage and
import that root into MindStudio Insight. Do not import only its
`ASCEND_PROFILER_OUTPUT` child. The report verifies raw metadata and the
official parsed Text/DB deliverables, then marks whether the profile is ready
for Insight. Import `cluster_analysis_output` for the official cluster views.

GPU trace JSON can be used as the baseline for `msprof-analyze compare` and can
also be inspected in Perfetto. Despite the upstream API name
`tensorboard_trace_handler`, TensorBoard is not the preferred Ascend trace
viewer.

This is a bounded in-process probe, not an external sampler attached to an
arbitrary PID. It observes the real training process at a scheduled step
window, which gives reliable operator correlation while keeping overhead
controlled.

## Dynamic profiler for a running training job

The versioned Ascend documentation also supports non-intrusive dynamic
profiling through `PROF_CONFIG_PATH`. It patches native PyTorch optimizer steps,
so TorchTitan's training loop does not need a profiling hook. Initialize a
disabled configuration before starting a long training job:

```bash
python tests/glm5_2_performance/dynamic_profiler_control.py \
  --init --config-dir /shared/glm5_profiler_control \
  --profile-dir /data/local_glm5_profiles

export PROF_CONFIG_PATH=/shared/glm5_profiler_control
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NGPU=8 LOG_RANK=0 MODULE=glm5 CONFIG=glm5_debugmodel ./run_train.sh
```

While training is running, enable a bounded collection from another terminal:

```bash
python tests/glm5_2_performance/dynamic_profiler_control.py \
  --enable --config-dir /shared/glm5_profiler_control \
  --preset distributed --parse-mode offline \
  --start-step 500 --warmup-steps 1 --active-steps 5
```

The controller writes `profiler_config.json` atomically because
`dynamic_profile` polls the file while training. `--show` prints the effective
configuration and `--disable` prevents a later pending collection. The default
dynamic parse mode is asynchronous; rank-filtered dynamic profiling does not
support synchronous parsing.

Do not combine `PROF_CONFIG_PATH` with this directory's scheduled `--capture`
or `--probe` commands. The fixed workflow rejects that combination explicitly.
The environment-only dynamic mode relies on native `torch.optim.Optimizer.step`
and therefore does not apply to a custom optimizer that bypasses that API.

Dynamic controller parameters are independent of the scheduled benchmark:

| Parameter | Purpose and choices | Default |
|---|---|---|
| `--init` | Write a disabled configuration before training starts. | no action; one action is required |
| `--enable` | Atomically enable one future bounded window. | no action; one action is required |
| `--disable` | Mark the existing configuration invalid. | no action; one action is required |
| `--show` | Print the current JSON configuration. | no action; one action is required |
| `--config-dir` | Directory polled through `PROF_CONFIG_PATH`. | `performance_dynamic/glm5_profiler` |
| `--profile-dir` | Raw profile destination. | existing value, otherwise `<config-dir>/profiles` |
| `--preset` | Same six Ascend presets as the scheduled benchmark. | `overview` |
| `--parse-mode` | `async` or `offline`; dynamic rank filtering cannot use sync parsing. | `async` |
| `--start-step` | Optimizer step at which profiling becomes active. | `10` |
| `--warmup-steps` | Dynamic Profiler warmup steps. | `1` |
| `--active-steps` | Dynamic Profiler recording steps. | `3` |
