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
