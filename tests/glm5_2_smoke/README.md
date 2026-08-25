# GLM-5.2 training smoke tests

This suite verifies that the current source-installed TorchTitan and
TorchTitanTurbo versions can launch short single-device and distributed GLM
training jobs. It reuses the canonical topology definitions in
`tests/glm5_2_common`.

Install the dependencies declared by the current TorchTitan checkout. If
TorchTitan was installed with `--no-deps`, install its pinned Grain version:

```bash
python -m pip install 'grain==0.2.18'
```

Export exactly one backend's visibility variable. Device detection is
automatic, or pass `--device` explicitly.

```bash
# GPU
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# NPU, in the NPU environment
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
```

## Eager smoke tests

GPU single-card, one distributed-topology example, and the complete suite:

```bash
unset ASCEND_RT_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python tests/glm5_2_smoke/train_smoke.py \
  --device gpu --topology single
# Replace fsdp8 with any registered distributed topology.
python tests/glm5_2_smoke/train_smoke.py \
  --device gpu --topology fsdp8
python tests/glm5_2_smoke/train_smoke.py \
  --device gpu --topology all
```

NPU single-card, one distributed-topology example, and the complete suite:

```bash
unset CUDA_VISIBLE_DEVICES
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python tests/glm5_2_smoke/train_smoke.py \
  --device npu --topology single
# Replace fsdp8 with any registered distributed topology.
python tests/glm5_2_smoke/train_smoke.py \
  --device npu --topology fsdp8
python tests/glm5_2_smoke/train_smoke.py \
  --device npu --topology all
```

Either backend can run a focused subset:

```bash
python tests/glm5_2_smoke/train_smoke.py \
  --topologies ddp8,fsdp8,tp8
```

On the dedicated full-DSA branches, cross-layer index sharing is selected by
an explicit experiment flag. It does not replace the default smoke
configuration:

```bash
python tests/glm5_2_smoke/train_smoke.py \
  --device npu --topology single --full-dsa
python tests/glm5_2_smoke/train_smoke.py \
  --device npu --topologies ddp8,fsdp8,tp8,ep8 \
  --full-dsa
python tests/glm5_2_smoke/train_smoke.py \
  --device npu --topology all --full-dsa
```

For Full DSA, `--topology all` automatically selects every supported non-PP
layout. PP layouts are omitted because GLM-5 does not move shared top-k tensors
across pipeline-stage boundaries; an explicit Full DSA PP selection is rejected.
The default `glm5_debugmodel` retains an indexer on every layer and continues to
support the complete smoke topology suite, including PP.

The Ascend SparseMLA kernel is a second, independent opt-in. The reduced debug
shape may be rejected by the released operator, so validate its production
geometry with the operator probe below before passing
`torchtitanturbo.models.glm5.ops.sparse_mla.npu_sparse_mla` through
`--override-imports` to a compatible production configuration.

The equivalent explicit selector is:

```bash
# reference, auto, triton, or npu-sparse
python tests/glm5_2_smoke/train_smoke.py \
  --device npu --topology single --full-dsa \
  --full-dsa-kernel auto
```

`auto` maps GPU to CUDA Triton and NPU to Triton-Ascend indexer/SparseMLA.
`npu-sparse` selects the Ascend-native SparseMLA while retaining the reference
indexer. Unregistered `ops_candidate` prototypes are intentionally not exposed
as smoke options. Repeat
`--extra-train-arg` to test an isolated TorchTitan tuning option; these values
are included in the manifest and run identity.

See [FULL_DSA.md](FULL_DSA.md) for the three-repository implementation and
test correspondence, including the HF-compatible no-sharing mode.

Use the same-device production-geometry operator probe before enabling an
optimized operator in a training run:

```bash
python tests/glm5_2_smoke/full_dsa_operator_probe.py \
  --device gpu --candidate triton
python tests/glm5_2_smoke/full_dsa_operator_probe.py \
  --device npu --candidate triton
python tests/glm5_2_smoke/full_dsa_operator_probe.py \
  --device npu --candidate npu-sparse
```

## NPU graph-mode smoke tests

The same topology selector can add the shared graph execution feature. Compiled
graph execution is currently implemented for NPU only; CUDA eager smoke tests
remain available, while `--device gpu --graph inductor|npugraphs` raises an
explicit `NotImplementedError` until a CUDA graph policy is defined.

```bash
unset CUDA_VISIBLE_DEVICES
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Single-card Inductor.
python tests/glm5_2_smoke/train_smoke.py \
  --device npu --topology single --graph inductor

# One distributed topology with compiler diagnostics.
python tests/glm5_2_smoke/train_smoke.py \
  --device npu --topology fsdp8 --graph inductor \
  --compiler-diagnostics

# A subset or every registered topology through eight ranks.
python tests/glm5_2_smoke/train_smoke.py \
  --device npu --topologies ddp8,fsdp8,tp8 --graph inductor
python tests/glm5_2_smoke/train_smoke.py \
  --device npu --topology all --graph inductor

# NPUGraph model capture. Add --compile-loss only when that backend supports
# the requested component set; the current NPUGraph contract is model-only.
python tests/glm5_2_smoke/train_smoke.py \
  --device npu --topology single --graph npugraphs
```

`--graph eager` is the default and preserves the original smoke behavior.
Compiled runs add graph mode, compiled components, and optional diagnostics to
the suite directory name, so different execution contracts neither overwrite
nor falsely reuse each other.

The default local batch size is 8 and the default global batch size is 64.
This shared profile is valid for every built-in topology through eight ranks,
including the eight microbatches required by PP8 with the 1F1B schedule.

Successful topologies are skipped on the next invocation. Incomplete output is
preserved with a `.failed-<timestamp>` suffix and retried. Use `--force` to
replace successful output as well.

## Command-line parameters

| Parameter | Purpose and choices | Default |
|---|---|---|
| `--device` | Backend selection: `auto`, `gpu`, or `npu`. `auto` selects NPU when `ASCEND_RT_VISIBLE_DEVICES` is set; otherwise it selects GPU. | `auto` |
| `--topology` | Run one common topology, or `all` for every registered topology of at most eight ranks. | `single` |
| `--topologies` | Run a comma-separated subset such as `ddp2,fsdp8,tp8`. Mutually exclusive with `--topology`. | unset |
| `--steps` | Number of optimizer steps in each smoke run. | `10` |
| `--local-batch-size` | Per-DP-rank local batch used to derive the token and PP microbatch schedule. | `8` |
| `--global-batch-size` | Global samples per optimizer step. Must satisfy the selected DP topology. | `64` |
| `--sequence-length` | Tokens per sample. | `128` |
| `--seed` | TorchTitan deterministic seed. | `61` |
| `--module` | TorchTitan model module passed to `run_train.sh`. | `glm5` |
| `--config` | TorchTitan registered model/training config. | `glm5_debugmodel` |
| `--override-imports` | Comma-separated explicit TorchTitan component overrides. No override is loaded unless requested. | unset |
| `--graph` | Execution mode: `eager`, `inductor`, or `npugraphs`. Compiled choices are NPU-only. | `eager` |
| `--compile-loss` | Compile both `model` and `loss`; without it only `model` is compiled. NPUGraph currently rejects this option. | disabled |
| `--compiler-diagnostics` | Set the shared compiler diagnostic environment for graph breaks, recompiles, and dynamic-shape events. | disabled |
| `--force` | Start a new generation for the selected topology range. All selected members are removed before execution; rerun without it after interruption to resume. | disabled |

Use either `--topology` or `--topologies`, not both. The available names are
defined centrally in `tests/glm5_2_common/topology.py`.

## Result contents and acceptance

Smoke is a launch/run-through gate, not a numerical comparison report. Each
topology directory contains the complete `runtime.log`, TorchTitan
`trainer_output`, and `manifest.json`. The manifest records the exact command,
device, topology, graph policy, training contract, return code, and final
`passed` or `failed` status. A topology passes only when the real training
process completes all requested steps with exit code zero. Precision,
performance, checkpoint equivalence, and long-duration stability are evaluated
by their dedicated suites.

Every run writes `runtime.log`, `manifest.json`, and TorchTitan output below:

```text
smoke_runs/<device-and-training-config>/<topology>/
```

Direct `run_train.sh` invocations also tee complete terminal output to:

```text
train_runs/<module-config-timestamp-pid>/runtime.log
```

Set `TORCHTITAN_RUN_LOG` when a caller needs an exact log path.
