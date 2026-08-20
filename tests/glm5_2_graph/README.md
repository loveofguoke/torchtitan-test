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

Use the central combination runner for graph + distributed + precision +
performance experiments. The graph scripts are single-topology convenience
aliases:

```bash
python tests/glm5_2_graph/compile_probe.py \
  --data --objectives performance --candidate-graph inductor

python tests/glm5_2_graph/compile_probe.py \
  --capture candidate --objectives performance --candidate-graph inductor
```

See `tests/glm5_2_combination/README.md` for complete two-endpoint and
all-topology commands.
