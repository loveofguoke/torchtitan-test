# GLM-5.2 performance profiler

This directory is independent from `tests/glm5_2_precision`. It does not
replace the precision data loader, checkpoint, metrics, or PASS/FAIL logic.
Profiling is enabled only for commands launched by this directory.

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
Those reports follow the three-stage Ascend
Profiler → `msprof-analyze` → MindStudio Insight workflow and do not redefine
the acceptance topology matrix.

## Documentation map

This README is the primary performance-experiment entry point. The complete
workflow is split by responsibility as follows:

| Layer | Document or implementation | Purpose |
|---|---|---|
| Experiment interface | This README | Commands, parameters, presets, output layout, report contents, and interpretation rules. |
| Environment and external tools | [性能与图模式统一依赖清单](../glm5_2_common/PERFORMANCE_GRAPH_DEPENDENCIES_ZH.md) | Required runtime, optional Python packages, Ascend tools, GUI viewers, repositories, installation, verification, and version boundaries. |
| Report interpretation | [中文性能报告与可视化阅读指南](REPORT_GUIDE_ZH.md) | Every HTML section, table, metric, official artifact, viewer, and PP-specific reading path. |
| Evidence layout | [exploration index](explorations/index.md) and [report index](explorations/reports/index.md) | Immutable run evidence and navigation by card count/topology. |
| Current analysis | [cross-topology summary](explorations/reports/summary.md) and [failed attempts](explorations/reports/failures.md) | Measured bottlenecks, hardware caveats, optimization backlog, and failed experiments. |
| Ascend collection implementation | [TorchTitanTurbo profiler document](https://github.com/loveofguoke/TorchTitanTurbo/blob/glm-dev/torchtitanturbo/tools/PROFILER.md), [profiler patch](https://github.com/loveofguoke/TorchTitanTurbo/blob/glm-dev/torchtitanturbo/tools/profiler.py), and [patch inventory](https://github.com/loveofguoke/TorchTitanTurbo/blob/glm-dev/PATCHES.md) | Translation of TorchTitan's lifecycle to `torch_npu.profiler`, NPU-only controls, and memory snapshots. |
| Device-neutral framework | `torchtitan/tools/profiler.py` in the source-installed TorchTitan checkout | Profiler schedule, lifecycle, step calls, and public configuration. |

The test repository is authoritative for commands, measurements, analysis,
reports, and optimization acceptance. Turbo is authoritative only for the
Ascend collection implementation. TorchTitan remains the device-neutral owner
of the profiler lifecycle. A proposed optimization in an exploration report
is not an implemented optimization unless its target repository, switch,
tests, and A/B evidence are linked explicitly.

The workflow is deliberately top-down:

1. `overview`: Level0 timing with low collection overhead.
2. `comparison`: Level0 with comparison-safe low-overhead controls.
3. `standard`: Level1 pipe-utilization detail on rank 0.
4. `distributed`: Level1 on every rank with pipe-utilization and interconnect
   data; raw data is parsed offline by default.
5. `kernel`: Level1 shapes, arithmetic utilization, and L2 information.
6. `operator`: shapes, operator attributes/arguments, and raw FLOPs capture.
7. `memory`: official categorized memory timeline in HTML/JSON/raw JSON.
8. `flamegraph`: Level0 CPU/NPU call stacks and Ascend profiler DB; analysis
   renders the official MindStudio Host HTML when its script is configured and
   portable CPU/NPU SVGs when Brendan Gregg's `flamegraph.pl` is available.
9. `runtime`: Level2 stack, module hierarchy, memory, operator attributes,
   host CPU/memory, and shapes.
10. `system`: Level2 Host CPU/memory/disk/network/OS runtime/NUMA, I/O,
    interconnection, GC, and MSTX collection on all ranks.

`all` is a meta-preset. It runs the non-redundant policies above as independent
captures and creates one suite index; it never enables every high-overhead
switch inside a single training process. See the
[Chinese report guide](REPORT_GUIDE_ZH.md#2-preset-到底是什么) for the exact
coverage matrix and overhead model.

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
| `--preset` | `overview`, `comparison`, `standard`, `distributed`, `kernel`, `operator`, `memory`, `flamegraph`, `runtime`, `system`, or the multi-capture meta-option `all`. | `overview` |
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
| `--analysis-tools` | `none`, `offline`, `advisor`, `cluster`, or `all`. `all` parses offline presets, runs advisor, and runs cluster only for multi-rank captures. | `none` |
| `--compare-baseline` | Baseline profiler path passed to `msprof-analyze compare`. | unset |
| `--extra-train-arg` | Append one raw TorchTitan argument; repeat the flag for multiple arguments. | none |
| `--force` | Remove and recapture completed output. Incomplete output is archived and retried without it. | disabled |

Advanced collection overrides apply to one concrete preset and are rejected
with `--preset all`, because silently applying one override to every independent
capture would destroy the suite's meaning:

| Parameter | Purpose | Default |
|---|---|---|
| `--profiler-level` | `level_none`, `level0`, `level1`, or `level2`. | selected preset |
| `--profile-ranks` | `all` or comma-separated global ranks. | selected preset |
| `--aic-metrics` | `none`, `pipe_utilization`, `arithmetic_utilization`, `memory`, `memory_l0`, `memory_ub`, `resource_conflict_ratio`, `l2_cache`, or `memory_access`. Only one metric family can be collected per profile. | selected preset |
| `--record-shapes` / `--no-record-shapes` | Input shape and dtype metadata. | selected preset |
| `--profile-memory` / `--no-profile-memory` | Framework/CANN memory records. | selected preset |
| `--with-stack` / `--no-with-stack` | Python/framework call stacks. | selected preset |
| `--with-modules` / `--no-with-modules` | Module hierarchy. | selected preset |
| `--with-flops` / `--no-with-flops` | Raw FLOPs field. The current official parser does not support it. | selected preset |
| `--export-stacks` / `--no-export-stacks` | CPU/NPU folded-stack callback export. Requires stack capture and sync parse. | selected preset |
| `--export-memory-timeline` / `--no-export-memory-timeline` | Official HTML, categorized JSON, and raw JSON memory timeline. Requires shape, memory, and stack/module data. | selected preset |
| `--l2-cache`, `--op-attr`, `--record-op-args` (and corresponding `--no-*`) | L2 counters, aclnn attributes, and argument statistics. | selected preset |
| `--data-simplification` / `--no-data-simplification` | Delete or retain redundant CANN raw data after parsing. | selected preset |
| `--gc-detect-threshold` | Record Python GC events lasting at least this many milliseconds. | selected preset |
| `--msprof-tx` / `--no-msprof-tx` | Legacy msprof TX range collection. Keep it separate from MSTX when comparing marker mechanisms. | selected preset |
| `--mstx`, `--mstx-domain-include`, `--mstx-domain-exclude` | Current MSTX collection and optional domain filters. Include/exclude are mutually exclusive. | selected preset |
| `--host-system` | Comma-separated `cpu,mem,disk,network,osrt,numa`. | selected preset |
| `--system-io`, `--system-interconnection` (and corresponding `--no-*`) | NIC/RoCE and HCCS/PCIe data. | selected preset |
| `--export-types` | Comma-separated `text,db`. | `text,db` |

For example, isolate the AI Core memory-access counter without defining a new
preset:

```bash
python tests/glm5_2_performance/profiler_benchmark.py \
  --probe --device npu --topology single --preset kernel \
  --aic-metrics memory_access --no-l2-cache
```

Each mutable capture writes `run_state.json`, and `runtime.log` begins with the
same attempt ID. Forced suite execution preflights all selected topologies for
live orchestrators, then prints and verifies removal of run, artifact, and
report output before starting the first capture.

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
It also indexes the complete interactive evidence instead of flattening it:

- `trace_view.json` and the complete `*_ascend_pt` root for the expandable
  PyTorch/CANN/NPU Timeline in MindStudio Insight; the JSON also opens in
  Perfetto or `chrome://tracing`;
- the official MindStudio Host flame graph HTML from the Ascend profiler DB,
  plus folded CPU/NPU stacks and portable click-to-zoom SVGs from the
  `flamegraph` or `runtime` preset;
- official memory timeline HTML, categorized JSON series, and raw event JSON
  from the `memory` or `runtime` preset;
- TorchTitan TensorBoard event files plus the exact `tensorboard --logdir`
  command for loss, grad norm, throughput, MFU, and memory curves;
- when the endpoint is compiled, `TORCH_TRACE`/`tlparse` and Inductor FX/IR/code
   artifacts captured by the graph diagnostics path.

The generated report starts with a Chinese reading order and metric glossary.
For a chapter-by-chapter explanation, including StepTrace, HCCL wait/transit,
operator/shape/L2 tables, flame graphs, memory categories, TensorBoard, and the
PP8-specific reading path, use [REPORT_GUIDE_ZH.md](REPORT_GUIDE_ZH.md).

## Complete profiler feature suite

One NPU, every non-redundant collection policy, every applicable official
offline analysis, and one suite index:

```bash
export ASCEND_RT_VISIBLE_DEVICES=4
unset CUDA_VISIBLE_DEVICES

python tests/glm5_2_performance/profiler_benchmark.py \
  --probe --device npu --topology single \
  --preset all --analysis-tools all
```

One distributed topology (replace `tp8` with any registered topology):

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python tests/glm5_2_performance/profiler_benchmark.py \
  --probe --device npu --topology tp8 \
  --preset all --analysis-tools all
```

All registered topologies and all profiler dimensions:

```bash
python tests/glm5_2_performance/profiler_benchmark.py \
  --probe --device npu --topology all \
  --preset all --analysis-tools all
```

The final command is intentionally expensive: it is a formal full matrix, not
a quick probe. `comparison` is excluded because it requires an explicit
`--compare-baseline`; `standard` is covered by deeper Level1 captures. Every
topology × preset run has a separate run/artifact/report path.

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
performance_runs/1-card/single/<experiment>/       # raw heavy data
performance_artifacts/1-card/single/<experiment>/  # manifest + summary
performance_reports/1-card/single/<experiment>.html
```

When `--topology all` or `--preset all` selects more than one member, every
member keeps the same tree above and the navigation index is written to:

```text
performance_reports/suites/<experiment>-<device>-suite.html
performance_reports/suites/<experiment>-<device>-all-presets-suite.html
```

The second form indexes the full `topology x preset` matrix. There is no shared
raw directory whose files mix different acquisition policies.

Heavy `performance_runs`, compact `performance_artifacts`, and dynamic scratch
outputs are ignored by Git; lightweight `performance_reports` are tracked for
review under the repository-wide output policy. For release transfer, the
following command includes HTML reports, compact artifacts, flame graphs,
memory timelines, TensorBoard, parsed Ascend outputs, advisor/cluster/compare
results, and graph visualization while excluding raw collection trees:

```bash
python release_artifacts.py upload <experiment> --content analysis
```

Use `--content full` when MindStudio must import the complete original profile
root or another host must resume/re-parse the experiment.

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
viewer. On Ascend, MindStudio Insight owns device-trace visualization;
TensorBoard remains the scalar training dashboard.

### Expandable Timeline, memory timeline, flame graph, and TensorBoard

Capture the lightweight stack preset on one NPU, then analyze it:

```bash
export ASCEND_RT_VISIBLE_DEVICES=4
unset CUDA_VISIBLE_DEVICES

python tests/glm5_2_performance/profiler_benchmark.py \
  --capture --device npu --topology single --preset flamegraph

# Optional official MindStudio interactive Host flame graph. Point to the
# flamegraph.py shipped by MindStudio Insight or cloned from Ascend/msinsight.
export TORCHTITAN_MSINSIGHT_FLAMEGRAPH=\
/opt/msinsight/scripts/flame_graph/flamegraph.py

# Optional portable CPU/NPU folded-stack SVG fallback.
export TORCHTITAN_FLAMEGRAPH_PL=/opt/FlameGraph/flamegraph.pl
python tests/glm5_2_performance/profiler_benchmark.py \
  --analyze --device npu --topology single --preset flamegraph
```

The report links every `trace_view.json`, official MindStudio flame graph,
folded stack, generated SVG, and TensorBoard event file. Open the complete
`*_ascend_pt` directory in
MindStudio Insight for the authoritative multi-level Timeline. For a quick
browser-only Timeline, load `trace_view.json` in `https://ui.perfetto.dev/`.
The report prints the exact run-specific TensorBoard command; a root-level view
can also be started with:

```bash
tensorboard --logdir performance_runs --port 6006 --bind_all
```

Stack collection and parsing add material overhead. Use `overview` or a
profiler-off run for performance baselines and use `flamegraph` only for
attribution. The dedicated `flamegraph` preset is available to scheduled
in-process capture, not the non-intrusive dynamic-profiler controller; dynamic
`runtime` still exposes stack hierarchy in its Timeline.

This is a bounded in-process probe, not an external sampler attached to an
arbitrary PID. It observes the real training process at a scheduled step
window, which gives reliable operator correlation while keeping overhead
controlled.

Capture the official categorized memory timeline separately:

```bash
python -m pip install matplotlib
python tests/glm5_2_performance/profiler_benchmark.py \
  --probe --device npu --topology single --preset memory
```

The report links the interactive HTML, categorized `.json.gz`, and raw
`_raw.json.gz` event stream. Parameter, optimizer, input, activation, gradient,
temporary, autograd-detail, and unknown categories follow the official Ascend
Profiler definition.

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
| `--preset` | Any scheduled preset except `flamegraph`, `memory`, and the `all` meta-preset. Dynamic mode records stack/memory hierarchy but cannot run the scheduled callback exports. | `overview` |
| `--parse-mode` | `async` or `offline`; dynamic rank filtering cannot use sync parsing. | `async` |
| `--start-step` | Optimizer step at which profiling becomes active. | `10` |
| `--warmup-steps` | Dynamic Profiler warmup steps. | `1` |
| `--active-steps` | Dynamic Profiler recording steps. | `3` |
