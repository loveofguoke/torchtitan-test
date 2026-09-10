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

For the CP8 FlexAttention backward investigation, select TorchNPU's two
lowering paths explicitly. This does not enable whole-model compilation:
FlexAttention still invokes its internal compile while `--graph eager` keeps
the surrounding GLM model eager.

```bash
# Control: reproduce TorchNPU's default mask-out path.
python tests/glm5_2_smoke/train_smoke.py \
  --device npu --topology cp8 --graph eager \
  --npu-codegen ascend-triton \
  --npu-flexattention-mask-mode mask-out --steps 2 --force

# Root-cause A/B: bypass the mask-out metadata and persistent dK/dV path.
python tests/glm5_2_smoke/train_smoke.py \
  --device npu --topology cp8 --graph eager \
  --npu-codegen ascend-triton \
  --npu-flexattention-mask-mode mask-in --steps 2 --force
```

Both choices have distinct suite identities and manifests. A mask-in pass
paired with a mask-out failure localizes the defect to TorchNPU's mask-out
lowering/kernel family. It does not alone distinguish saved LSE, compact
backward metadata, dQ, and dK/dV corruption.

To diagnose the observed CP8 backward corruption, enable the focused capture
on the rank and layer that first showed the large finite dQ/dK values:

```bash
python tests/glm5_2_smoke/train_smoke.py \
  --device npu --topology cp8 --graph eager \
  --npu-codegen ascend-triton \
  --npu-flexattention-mask-mode mask-out \
  --nonfinite-diagnostics \
  --diagnostic-rank 6 \
  --diagnostic-layer layers.6.attention.inner_attention \
  --steps 2 --force
```

The selected topology directory contains
`nonfinite_replay/rank6/callNNN/{forward,backward,actual_gradients}.pt`.
`forward.pt` records an immediate CPU snapshot of Q/K/V, mask, selected
indices, output, tensor storage/version/stride/stream metadata, and the compiler
environment that created the call.
`backward.pt` is written when the public attention output first receives its
gradient, before the compiled FlexAttention backward executes. It contains the
output read again at that boundary, its maximum difference from the forward
snapshot, allocator state, per-rank compiler trace/debug directories, and the independent FP32 reference
`delta_ref_QN = sum(output_snapshot * grad_output, dim=-1)`. A nonzero output
difference identifies saved-output overwrite or aliasing before entering the
backward kernel. A finite, plausible reference DELTA moves the investigation
inside TorchNPU lowering/runtime; it does not by itself prove that the DELTA
actually consumed by the generated kernel is correct. `actual_gradients.pt`
captures the dQ/dK/dV returned by that exact distributed backward invocation.

After the distributed process exits, smoke automatically selects one capture per
rank: the call with the largest observed dQ/dK magnitude, with non-finite values
ranked first. This retains every raw capture while avoiding 64 sequential
compilations for an 8-rank, 8-call CP run. Before replay starts it creates
`nonfinite_replay/replay_status.json` and `replay.log`; `replay_summary.json` is
initialized immediately and replaced atomically after every completed call.
Per-call `replay_result.json` files compare the actual distributed gradients with
the isolated replay. Automatic replay runs before a failed training result is
raised, so it is not skipped when the failure being diagnosed terminates the
training step. Replay compiler evidence is stored separately under
`nonfinite_replay/compiler/replay`; `compiler_comparison.{json,md}` inventories
per-rank and replay files, hashes, kernel names, and relevant source/log snippets.
The diagnostic deliberately does not change Inductor or Triton cache directories,
because doing so could hide a cache/concurrency failure. Use the manual command
below to replay additional saved calls.

Replay one or more captures on a single device with:

```bash
python tests/glm5_2_smoke/replay_glm5_flex.py \
  smoke_runs/<suite>/cp8/nonfinite_replay/rank6/call000 \
  --device npu:0
```

The replay report includes captured and replayed DELTA statistics, actual versus
replayed dQ/dK/dV statistics and maximum differences. Capture schema version 4
is recorded in the smoke contract, so an older diagnostic run is not silently
reused.

Either backend can run a focused subset:

```bash
python tests/glm5_2_smoke/train_smoke.py \
  --topologies ddp8,fsdp8,tp8
```

### HSDP smoke coverage

For mesh/placement examples, gradient communication and batch semantics, see
the [shared topology guide](../glm5_2_common/README.md#hsdp组内分片组间复制).

HSDP combines replicated groups with parameter sharding inside each group.
`hsdp2x4` means `dp_replicate=2, dp_shard=4`; `hsdp4x2` means
`dp_replicate=4, dp_shard=2`. Both use eight devices, DP degree eight,
and TP/CP/PP/EP degree one. With local batch eight and global batch 64,
both require one gradient accumulation step. These are FSDP2 layouts, not
a separate model implementation or an FSDP1 backend.

The shared registry exposes both layouts to smoke, precision, checkpoint,
stability, graph/combination, MindStudio and performance consumers. Their
eight-device `all` selections now include these additional runs. Existing
topology identities are unchanged; no `--force` is needed merely to add HSDP.

```bash
# GPU: run only the newly added members after the previous all smoke.
python tests/glm5_2_smoke/train_smoke.py \
  --device gpu --graph eager --topologies hsdp2x4,hsdp4x2 --steps 2

# NPU: run after the single-device FlexAttention compatibility issue is fixed.
python tests/glm5_2_smoke/train_smoke.py \
  --device npu --graph eager --topologies hsdp2x4,hsdp4x2 --steps 2
```

The normal suite report includes each HSDP member's status, elapsed time and
runtime log. Registration is not evidence of successful device execution.

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
For ordinary pipeline schedules, the launcher selects the first rank of the
last pipeline stage as `LOG_RANK`, because that stage owns the real loss.
Logging rank 0 instead would aggregate the non-loss-stage `-1` placeholders;
for PP8 this is displayed as `-8` even though the last stage computed a normal
loss. An explicit `LOG_RANK` environment value still overrides this default.

Successful topologies are skipped on the next invocation. Incomplete output is
preserved with a `.failed-<timestamp>` suffix and retried. `--force` removes
every selected topology before the first run starts; if that fresh suite is
interrupted, rerun without `--force` to continue from its incomplete member.

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
| `--graph` | Execution mode: `eager`, `inductor`, or `npugraphs`. Compiled choices are NPU-only. | `eager` |
| `--compile-loss` | Compile both `model` and `loss`; without it only `model` is compiled. NPUGraph currently rejects this option. | disabled |
| `--compiler-diagnostics` | Set the shared compiler diagnostic environment for graph breaks, recompiles, and dynamic-shape events. | disabled |
| `--nonfinite-diagnostics` | Capture GLM FlexAttention inputs, saved-output lifetime evidence, and the independent FP32 backward DELTA reference. NPU only. | disabled |
| `--diagnostic-rank` | Global rank whose selected FlexAttention layer is captured, or `all` to compare every rank in one run. An integer rank must exist in every selected topology. | `6` |
| `--diagnostic-layer` | Module-FQN substring selecting the captured GLM FlexAttention layer. | `layers.6.attention.inner_attention` |
| `--npu-flexattention-mask-mode` | Select TorchNPU `mask-in` or `mask-out` FlexAttention lowering, including internally compiled FlexAttention under `--graph eager`. | unset |
| `--force` | Remove and rerun completed topology output. Without it, completed runs are skipped and incomplete runs are archived before retry. | disabled |

Use either `--topology` or `--topologies`, not both. The available names are
defined centrally in `tests/glm5_2_common/topology.py`.

## Result contents and acceptance

The suite root contains `README.md` (a topology result table with relative log
links) and `summary.json` (full selected-member records). They are refreshed
after each member, including failures. The existing fail-fast behavior remains:
members not reached are `not_run`, not passed. Run the same command without
`--force` to resume. The summary describes the latest invocation's selection,
not every historical topology under the directory.

New manifests record UTC start/end timestamps, monotonic elapsed seconds,
exported device visibility, and launch errors/return codes. Elapsed time includes
process startup and compilation and is not steady-state training performance.
Reused runs retain their original timestamps; historical manifests lacking time
fields display `unknown` and do not require recollection. The visibility list
is the exported device pool, not an independently verified rank-to-card map.

For a short eager gate use `--device gpu --graph eager --topology all --steps 2`
or the same arguments with `--device npu`. Existing graph selection semantics
are unchanged. Smoke success is not proof of full-graph compilation or numerical
alignment; use the dedicated tooling for those claims.

Smoke is a launch/run-through gate, not a numerical comparison report. Each
topology directory contains the complete `runtime.log`, TorchTitan
`trainer_output`, and `manifest.json`. The manifest records the exact command,
device, topology, graph policy, training contract, return code, and final
`passed` or `failed` status. A topology passes only when the real training
process completes all requested steps with exit code zero. Precision,
performance, checkpoint equivalence, and long-duration stability are evaluated
by their dedicated suites.

Every run writes `runtime.log`, `manifest.json`, and TorchTitan output below:

The topology directory also writes `run_state.json`. Its attempt ID and
orchestrator PID distinguish a newly forced run from an older log at the same
path. `--force` prints and verifies every removed topology directory and refuses
to overwrite a run whose orchestrator is still alive.

```text
smoke_runs/<device-and-training-config>/<topology>/
```

Direct `run_train.sh` invocations also tee complete terminal output to:

```text
train_runs/<module-config-timestamp-pid>/runtime.log
```

Set `TORCHTITAN_RUN_LOG` when a caller needs an exact log path.
