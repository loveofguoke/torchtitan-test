# GLM-5.2 combination experiments

This package composes topology, graph mode, profiling, and precision capture at
one execution boundary. Precision and performance modules do not configure one
another. Each feature contributes only command arguments, environment variables,
and metadata; the common execution plan rejects conflicting contributions.

## Complete workflow

Every capture consumes one shared random seed checkpoint and fixed token plan.
Run `--data` exactly once before any capture, then synchronize the generated
`precision_fixtures/<fixture-id>/` directory to the other endpoint. The fixture may
be generated on either device. For example, generate it on the NPU server:

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
unset CUDA_VISIBLE_DEVICES

python tests/glm5_2_combination/combination_benchmark.py \
  --data --data-device npu --topology all \
  --objectives precision,performance \
  --reference-graph eager --candidate-graph inductor
```

`--data-device npu` is optional when only
`ASCEND_RT_VISIBLE_DEVICES` is exported. To generate the same fixture on GPU,
export only `CUDA_VISIBLE_DEVICES` and use `--data-device cuda`.

After synchronizing the fixture, capture every topology on the GPU server:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
unset ASCEND_RT_VISIBLE_DEVICES

python tests/glm5_2_combination/combination_benchmark.py \
  --capture reference --topology all \
  --objectives precision,performance \
  --reference-graph eager --candidate-graph inductor \
  --compiler-diagnostics
```

Capture the same topology suite on the NPU server:

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
unset CUDA_VISIBLE_DEVICES

python tests/glm5_2_combination/combination_benchmark.py \
  --capture candidate --topology all \
  --objectives precision,performance \
  --reference-graph eager --candidate-graph inductor \
  --compiler-diagnostics
```

After synchronizing `combination_artifacts` and the profiler-containing
`combination_runs`, generate the unified report on CPU:

```bash
python tests/glm5_2_combination/combination_benchmark.py \
  --compare --topology all \
  --objectives precision,performance \
  --reference-graph eager --candidate-graph inductor \
  --compiler-diagnostics --require-all
```

The training configuration used by `--data` must match the captures. The graph,
objective, profiler, and diagnostics selections must remain identical across
both capture commands and `--compare`; they are part of the combination
identity. All configured repeats run automatically when `--repeat` is omitted;
use `--repeat N` only to capture one specific repeat.

Before each training process starts, the framework prints its topology and
runtime-log path. Every capture writes the full command, stdout, and stderr to:

```text
combination_runs/<combination-id>/<topology>/<role>-r<repeat>/runtime.log
```

Each combination identity has one compact suite directory. Topology captures
are stored below `single/`, `ddp8/`, `fsdp8/`, and the other selected topology
subdirectories. A valid published artifact is skipped on the next capture;
stale run output without a valid artifact is replaced and rerun. `--force`
reruns completed captures. Graph modes, objectives, and profiler settings are
part of the suite identity, so captures from different feature combinations
cannot be reused accidentally.

Use `--objectives precision` or `--objectives performance` for an isolated
experiment. Use `--topology single`, `--topology fsdp8`, or
`--topologies single,fsdp8,tp8` to select a subset. NPUGraph is selected with
`--candidate-graph npugraphs` and is valid only for an NPU endpoint.
