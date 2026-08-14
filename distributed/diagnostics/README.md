# DDP step-1 GPU/NPU diagnostic

This diagnostic narrows a distributed precision failure before enabling the
full single-device layer parity suite. It runs the existing `ddp4_fp32`
fixture for exactly one optimizer step with the original batch size 4 and
sequence length 128.

The run is gated by a pre-materialized global batch. The tensor file contains
the exact 16 input, position, and label rows for step 1; its JSON sidecar binds
the bytes to the scenario and fixture SHA-256 identities. Every rank replays a
contiguous four-row slice, and capture fails before producing `capture.json` if
any captured tensor differs from that slice.

Each rank stores its exact input tensors, selected GLM-5 module outputs, and
local loss in safetensors. Rank 0 additionally stores selected loaded
parameters, post-DDP/post-clipping gradients, updated parameters, and optimizer
deltas. The regular TorchTitan source tree is not modified; the diagnostic
entry point injects temporary Trainer hooks from this test repository.

Generate the fixed input artifact once, on the machine that owns the canonical
fixture. Commit or copy both the `.safetensors` file and its generated
`.safetensors.json` sidecar; do not regenerate them independently on GPU and
NPU:

```bash
python -m distributed.diagnostics.ddp_step prepare-inputs \
  --scenario distributed/scenarios/glm5_debugmodel_4card/ddp4_fp32.json \
  --fixture distributed_runs/glm5_debugmodel_4card/fixtures/ddp4_fp32 \
  --output distributed/fixed_inputs/glm5_debugmodel_ddp4_step1.safetensors
```

On the GPU machine:

```bash
python -m distributed.diagnostics.ddp_step capture \
  --scenario distributed/scenarios/glm5_debugmodel_4card/ddp4_fp32.json \
  --fixture distributed_runs/glm5_debugmodel_4card/fixtures/ddp4_fp32 \
  --fixed-inputs distributed/fixed_inputs/glm5_debugmodel_ddp4_step1.safetensors \
  --backend gpu \
  --output distributed_diagnostics/ddp4_step1_gpu
```

On the NPU machine:

```bash
python -m distributed.diagnostics.ddp_step capture \
  --scenario distributed/scenarios/glm5_debugmodel_4card/ddp4_fp32.json \
  --fixture distributed_runs/glm5_debugmodel_4card/fixtures/ddp4_fp32 \
  --fixed-inputs distributed/fixed_inputs/glm5_debugmodel_ddp4_step1.safetensors \
  --backend npu \
  --output distributed_diagnostics/ddp4_step1_npu
```

Put both immutable capture directories on either machine and compare on CPU:

```bash
python -m distributed.diagnostics.ddp_step compare \
  --gpu distributed_diagnostics/ddp4_step1_gpu \
  --npu distributed_diagnostics/ddp4_step1_npu \
  --report distributed_diagnostics/ddp4_step1_report.json
```

The command also writes `ddp4_step1_report.md`. The first failure follows
actual execution order. Integer inputs and routing outputs require exact
equality. FP32 inputs, parameters, local loss, and activations use `1e-5`
absolute/relative tolerance; gradients and optimizer deltas use `1e-4`
absolute and `1e-3` relative tolerance.

Comparison also requires the two `capture.json` files to contain the same
`fixed_inputs_digest`. This prevents otherwise-valid GPU and NPU captures made
with different input files from being compared.

Interpret the first failing observation as follows:

- `input.*` or `labels`: rank/data assignment differs.
- `parameter.before.*`: checkpoint loading or parameter placement differs.
- `forward.*`: the named GLM component is the first observed forward split.
- `loss.local`: forward tensors were within tolerance but loss split.
- `gradient.post_reduce_clip.*`: backward or DDP reduction split.
- `parameter.delta.*`: optimizer update split.

Captures are intentionally immutable. Move an old output aside before a rerun.
