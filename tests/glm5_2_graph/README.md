# GLM-5.2 graph-mode entry points

Graph mode is an independent feature owned by the combination experiment. It
does not modify TorchTitan and does not configure precision or performance
modules. Ascend-only backend validation lives in TorchTitanTurbo.

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

The first command creates `graph-probe-fixture` below
`precision_fixtures/`. The two captures run NPU eager as the reference and NPU
Inductor as the candidate with the same checkpoint and token plan. Each capture
prints the exact run directory and always writes the complete command, stdout,
stderr, graph breaks, recompiles, and backend failures to:

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
export TORCH_COMPILE_DEBUG_DIR="$PWD/graph_compile_debug/npu-inductor"
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
