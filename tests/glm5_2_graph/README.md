# GLM-5.2 graph-mode entry points

For the isolated CANN 9.1 NPU Inductor launcher, persistent compiler cache,
automatic per-run reports, and Chinese operating instructions, see:

- `../glm5_2_graph_debug/run_npu_inductor.sh`
- `../glm5_2_graph_debug/RUN_NPU_INDUCTOR.md`
- `../glm5_2_graph_debug/NPU_INDUCTOR_ENV_ADAPTATION_REPORT.md`
- `../glm5_2_graph_debug/HUAWEI_DELIVERY.md`

This launcher is an environment/debugging aid. It does not replace the formal
precision, performance, stability, or central combination acceptance tests.

Graph mode is an independent feature owned by the combination experiment. It
does not modify TorchTitan and does not configure precision or performance
modules. Ascend-only backend validation lives in TorchTitanTurbo.

For direct execution without the debug wrapper, place every compiler-generated
file in the persistent user cache before running any graph command:

```bash
GRAPH_CACHE_ROOT=/workspace/y50064852_yyb/.cache/torchtitan-test/graph_mode/cann91-torch214-triton321
mkdir -p "$GRAPH_CACHE_ROOT"/{inductor,triton,torch_compile_debug}
export TORCHINDUCTOR_CACHE_DIR="$GRAPH_CACHE_ROOT/inductor"
export TRITON_CACHE_DIR="$GRAPH_CACHE_ROOT/triton"
export TORCH_COMPILE_DEBUG_DIR="$GRAPH_CACHE_ROOT/torch_compile_debug"
```

The `glm5_2_graph_debug/run_npu_inductor.sh` launcher applies this policy
automatically, including for its `probe` action that calls `compile_probe.py`.

The supported execution policies match TorchTitan's current compile contract:

- `eager`: no compile arguments;
- `inductor`: `torch.compile` with the Inductor backend;
- `npugraphs`: NPU-only NPUGraph backend, model component only.

There is no `inductor-dynamic` option because the installed TorchTitan
`CompileConfig` does not expose a dynamic-shape field.

## Normal training entry point

Graph mode is available to ordinary training, not only to the test workflow.
Eager is the default and requires no compile option:

```bash
NGPU=1 LOG_RANK=0 MODULE=glm5 CONFIG=glm5_debugmodel ./run_train.sh
```

Enable TorchTitan's `torch.compile` path with Inductor:

```bash
NGPU=1 LOG_RANK=0 MODULE=glm5 CONFIG=glm5_debugmodel ./run_train.sh \
  --compile.enable \
  --compile.components=model \
  --compile.backend=inductor
```

On NPU, `run_train.sh` selects `train_npu.py`, which imports TorchTitanTurbo
before starting TorchTitan. The NPUGraph prototype uses the same public
TorchTitan configuration:

```bash
export ASCEND_RT_VISIBLE_DEVICES=4
unset CUDA_VISIBLE_DEVICES

NGPU=1 LOG_RANK=0 MODULE=glm5 CONFIG=glm5_debugmodel ./run_train.sh \
  --compile.enable \
  --compile.components=model \
  --compile.backend=npugraphs
```

These commands establish functional graph-mode training. The experiment below
then checks accuracy, graph diagnostics, distributed composition, and
performance using the same underlying options.

## Multi-card graph-mode training

Run from the repository root. The shared eight-card profile uses local batch 8,
global batch 64, and sequence length 128, expressed through TorchTitan's current
token-budget CLI:

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
unset CUDA_VISIBLE_DEVICES
export TORCHTITAN_DEVICE=npu

NGPU=8 LOG_RANK=0 MODULE=glm5 CONFIG=glm5_debugmodel ./run_train.sh \
  --training.steps=10 \
  --training.disable_cuda_graphs \
  --training.num_tokens_per_microbatch_per_dp_rank=1024 \
  --training.num_tokens_per_train_step=8192 \
  --training.max_context_length=128 \
  --compile.enable \
  --compile.components=model \
  --compile.backend=inductor \
  --parallelism.spmd_backend=partial_dtensor \
  --parallelism.data_parallel_replicate_degree=1 \
  --parallelism.data_parallel_shard_degree=8 \
  --parallelism.context_parallel_degree=1 \
  --parallelism.tensor_parallel_degree=1 \
  --parallelism.pipeline_parallel_degree=1 \
  --parallelism.expert_parallel_degree=1
```

This first command validates FSDP8. `NGPU` must equal the product of the dense
parallel degrees. Keep every degree explicit when changing topology so values
from the TOML configuration cannot leak into the experiment. The authoritative
topologies and their generated arguments live in
`tests/glm5_2_common/topology.py`; common focused choices are:

| Topology | Non-unit degree and required extra option |
|----------|-------------------------------------------|
| DDP8 | `data_parallel_replicate_degree=8` |
| FSDP8 | `data_parallel_shard_degree=8` |
| TP8 | `tensor_parallel_degree=8`, plus `--parallelism.no-enable-sequence-parallel` |
| CP8 | `context_parallel_degree=8` |
| FSDP2 + TP4 | `data_parallel_shard_degree=2`, `tensor_parallel_degree=4`, plus `--parallelism.no-enable-sequence-parallel` |

`--training.disable_cuda_graphs` disables TorchTitan Trainer's whole-step CUDA
Graph capture, which is independent from this experiment's `torch.compile`
path. Keep it set for PP, because the Trainer CUDA Graph path does not support
pipeline parallelism. PP8 also requires
`--parallelism.num_pp_microbatches=8` with the shared batch profile.

The 10-step command is a functional smoke run. After all ranks complete compile,
forward, backward, optimizer, and collectives, increase the configured steps for
formal precision and steady-state performance tests. Complete output is written
to `train_runs/<run-name>/runtime.log`.

## Fast NPU eager versus graph probe

The probe uses the same fixed seed checkpoint and token plan as the precision
framework. Therefore `--data` is required before the first capture. The
fixture may be generated on either endpoint; this example generates it on the
NPU server:

```bash
export ASCEND_RT_VISIBLE_DEVICES=4
unset CUDA_VISIBLE_DEVICES

python tests/glm5_2_graph/compile_probe.py \
  --data --data-device npu --topology single \
  --objectives precision \
  --reference-graph eager --candidate-graph inductor

python tests/glm5_2_graph/compile_probe.py \
  --capture reference --topology single \
  --objectives precision \
  --reference-graph eager --candidate-graph inductor \
  --compiler-diagnostics

python tests/glm5_2_graph/compile_probe.py \
  --capture candidate --topology single \
  --objectives precision \
  --reference-graph eager --candidate-graph inductor \
  --compiler-diagnostics

python tests/glm5_2_graph/compile_probe.py \
  --compare --topology single \
  --objectives precision \
  --reference-graph eager --candidate-graph inductor \
  --compiler-diagnostics --require-all
```

The first command creates a config-digested directory below
`precision_fixtures/`. The two captures run NPU eager as the reference and NPU
Inductor as the candidate with the same checkpoint and token plan. A matching
pre-digest fixture is renamed automatically and reused. Each capture prints the
exact run directory and always writes the complete command, stdout, stderr,
graph breaks, recompiles, and backend failures to:

```text
combination_runs/<combination-id>/single/candidate-r1/runtime.log
combination_runs/<combination-id>/single/reference-r1/runtime.log
```

All configured repeats are captured when `--repeat` is omitted. Use
`--repeat 1` for one quick smoke run. To probe NPUGraph instead, replace
`--candidate-graph inductor` with `--candidate-graph npugraphs` in all
commands.

`torch_compile_debug/` is PyTorch compiler output, not the runtime log. Enable
and place it explicitly when detailed Dynamo/Inductor artifacts are needed:

```bash
export TORCH_COMPILE_DEBUG=1
export TORCH_COMPILE_DEBUG_DIR=/workspace/y50064852_yyb/.cache/torchtitan-test/graph_mode/cann91-torch214-triton321/torch_compile_debug
```

## Full NPU graph precision experiment

The full experiment uses the same four stages but the 5000-step configuration
from `precision_benchmark.py`:

```bash
export ASCEND_RT_VISIBLE_DEVICES=4
unset CUDA_VISIBLE_DEVICES

python tests/glm5_2_graph/precision_benchmark.py \
  --data --data-device npu --topology single \
  --objectives precision \
  --reference-graph eager --candidate-graph inductor

python tests/glm5_2_graph/precision_benchmark.py \
  --capture reference --topology single \
  --objectives precision \
  --reference-graph eager --candidate-graph inductor \
  --compiler-diagnostics

python tests/glm5_2_graph/precision_benchmark.py \
  --capture candidate --topology single \
  --objectives precision \
  --reference-graph eager --candidate-graph inductor \
  --compiler-diagnostics

python tests/glm5_2_graph/precision_benchmark.py \
  --compare --topology single \
  --objectives precision \
  --reference-graph eager --candidate-graph inductor \
  --compiler-diagnostics --require-all
```

Use the central combination runner for graph + distributed + precision +
performance experiments. The graph scripts above are single-topology
NPU eager-versus-graph entry points. The central combination experiment keeps
its independent CUDA-reference/NPU-candidate endpoint configuration.

See `tests/glm5_2_combination/README.md` for complete two-endpoint and
all-topology commands.

## Multi-card precision and performance tests

The entry points in `tests/glm5_2_graph` intentionally register only the
`single` topology. Multi-card acceptance must use the combination runner, which
owns the common topology definitions and composes graph, precision, and
performance settings at one execution boundary.

The formal multi-card workflow uses the combination experiment's independent
GPU eager reference and NPU graph candidate endpoints. Generate the fixed
checkpoint and token plan once, then synchronize the generated
`precision_fixtures/<fixture-id>/` directory to the other server:

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
unset CUDA_VISIBLE_DEVICES

python tests/glm5_2_combination/combination_benchmark.py \
  --data --data-device npu --topology all \
  --objectives precision,performance \
  --reference-graph eager --candidate-graph inductor
```

Capture the eager reference on the GPU server:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
unset ASCEND_RT_VISIBLE_DEVICES

python tests/glm5_2_combination/combination_benchmark.py \
  --capture reference --topology all \
  --objectives precision,performance \
  --reference-graph eager --candidate-graph inductor \
  --compiler-diagnostics
```

Capture the graph candidate on the NPU server:

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
unset CUDA_VISIBLE_DEVICES

python tests/glm5_2_combination/combination_benchmark.py \
  --capture candidate --topology all \
  --objectives precision,performance \
  --reference-graph eager --candidate-graph inductor \
  --compiler-diagnostics
```

Synchronize `combination_artifacts` and the profiler-containing
`combination_runs`, then compare on either server or a CPU-only report host:

```bash
python tests/glm5_2_combination/combination_benchmark.py \
  --compare --topology all \
  --objectives precision,performance \
  --reference-graph eager --candidate-graph inductor \
  --compiler-diagnostics --require-all
```

Start with `--topology fsdp8` for one focused multi-card test, or use
`--topologies ddp8,fsdp8,tp8` for a subset. The data, reference, candidate, and
compare stages must use the same topology, graph, objectives, profiler,
diagnostics, and training configuration because these fields define the
combination identity.

Precision acceptance compares GPU eager and NPU graph loss and grad-norm traces
produced from the same initial state and fixed tokens. Performance acceptance
compares profiler-backed throughput and step-time statistics from both
endpoints. Run at least 10 steps so startup and warm-up do not dominate the
result. Use `--objectives precision` or
`--objectives performance` to isolate one dimension. Omitting `--repeat` runs
every configured repeat; `--repeat 1` is only a quick diagnostic.

Every topology and role keeps a complete runtime log:

```text
combination_runs/<combination-id>/<topology>/reference-r<repeat>/runtime.log
combination_runs/<combination-id>/<topology>/candidate-r<repeat>/runtime.log
```

Review graph breaks, recompiles, and backend failures together with the
numerical and performance reports. A zero exit status alone is not sufficient
graph-mode acceptance.
