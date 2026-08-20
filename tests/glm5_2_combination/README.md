# GLM-5.2 combination experiments

This package composes topology, graph mode, profiling, and precision capture at
one execution boundary. Precision and performance modules do not configure one
another. Each feature contributes only command arguments, environment variables,
and metadata; the common execution plan rejects conflicting contributions.

Prepare one checkpoint and fixed token plan from either endpoint environment:

```bash
python tests/glm5_2_combination/combination_benchmark.py --data
```

Capture every topology on the GPU server and NPU server:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python tests/glm5_2_combination/combination_benchmark.py \
  --capture reference --topology all \
  --objectives precision,performance \
  --reference-graph eager --candidate-graph inductor

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python tests/glm5_2_combination/combination_benchmark.py \
  --capture candidate --topology all \
  --objectives precision,performance \
  --reference-graph eager --candidate-graph inductor
```

After synchronizing `combination_artifacts` and the profiler-containing
`combination_runs`, generate the unified report on CPU:

```bash
python tests/glm5_2_combination/combination_benchmark.py \
  --compare --topology all \
  --objectives precision,performance \
  --reference-graph eager --candidate-graph inductor \
  --require-all
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
