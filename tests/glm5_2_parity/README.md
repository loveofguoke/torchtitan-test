# Portable GLM-5.2 parity workflow

This framework lives in `torchtitan-test` and imports the source-installed
`torchtitan` package. Run every command below from the `torchtitan-test`
repository root. NPU captures also import source-installed `torchtitanturbo`
before PyTorch selects its device backend.

The GLM-5.2 parity suite supports four execution modes:

- `paired` (default): run two endpoints in one process and immediately report.
- `prepare`: create one immutable fixture containing exact model state and data.
- `capture`: run one endpoint and write a portable, checksummed artifact.
- `compare`: compare two artifacts on CPU and write the same HTML diagnostics.

Artifacts contain a versioned JSON manifest, exact fixture tensors, module
activation and gradient traces, discrete routing selections, logits, loss,
canonical parameters, and canonical gradients. Tensor data is stored in
checksummed safetensors shards. Python stdout/stderr is attached as
`attachments/runtime.log`. A completed artifact is an immutable directory. If
execution fails after capture starts, the partial observations, runtime log,
and exception are written with `status=failed`; failed runs cannot be compared.
Because parameters and gradients are retained for offline comparison, budget at
least two model-state sizes per artifact, plus activation traces and fixtures.

End-to-end captures also retain full indexer selection scores and full router
choice scores. When a top-k decision differs, compare reports candidate scores,
the K/K+1 cutoff margin, and a CPU replay of legacy indexer captures. Existing
artifacts remain comparable through the replay fallback; newly captured
artifacts expose the device-side score trace directly.

Every offline indexer and router row reports the selected `id:score` list for
every batch/query, together with the BF16 ULP at the cutoff scale, the nominal
rounding band, the observed cross-device score band, and both K/K+1 margins.
Discrete results have three states: `PASS` for identical selected sets,
`BOUNDARY_PASS` for BF16-only cutoff changes explained by the measured score
band whose complete score row passes scale-aware BF16 tolerance and whose
available downstream continuous checkpoints remain acceptable, and `FAIL` for
stable, out-of-band, or harmful propagated differences. ULP values remain in
the report as scale diagnostics but are not a fixed pass/fail limit. FP32
selection remains exact.

HTML result tables keep their header visible inside a bounded scroll area.
Buttons above each table independently fold path, dtype, value, metric, and
diagnostic column groups without hiding the component or status columns.

## Existing paired command

torchtitan glm5.2 vs hf glm5.2
- gpu
- fp32

The paired scenario keeps all editable settings in one file:

```bash
python tests/glm5_2_parity/titan_hf_gpu_fp32_random_paired.py --run
```

Copy and rename that file to define another paired experiment. The filename is
the scenario ID and therefore controls the default report and log directory.
Only edit its `CONFIG` block.

## Offline GPU/NPU comparison

torchtitan glm5.2 gpu vs npu
- fp32&bf16

The offline scenario is also one editable file. Its filename is the scenario
ID; its `CONFIG` block defines both endpoints, devices, data, model size, cases,
components, output names, and directories. Inspect the effective configuration
and resolved paths without executing a test:

```bash
python tests/glm5_2_parity/titan_gpu_npu_fp32_random.py --print-config
```

The offline engine calls the two sides `actual` and `expected`; it does not
assume a GPU/NPU pair. Each `OfflineEndpointConfig` independently specifies the
model endpoint, device type, visibility environment variable, device index, and
artifact name. The portable stage names are `--actual-capture` and
`--expected-capture`. Scenario-specific aliases are generated from endpoint
names, such as `--npu-capture`, `--gpu-capture`, `--titan-gpu-capture`, or
`--hf-gpu-capture`.

Generate the fixture once on CPU. This stores the exact FP32 TorchTitan state,
every case tensor, test ordinal, seed, test plan, and effective configuration:

```bash
python tests/glm5_2_parity/titan_gpu_npu_fp32_random.py --data
```

Copy the unchanged scenario file and its complete fixture directory to both
servers. Each capture refuses to run without that fixture or when any scenario,
model, data, case-order, or configuration digest differs:

```bash
python tests/glm5_2_parity/titan_gpu_npu_fp32_random.py --gpu-capture
python tests/glm5_2_parity/titan_gpu_npu_fp32_random.py --npu-capture
```

Copy the NPU capture beside the GPU capture and generate the report on CPU:

```bash
python tests/glm5_2_parity/titan_gpu_npu_fp32_random.py --compare
```

Copy and rename the scenario file for BF16 or another data case, then edit only
its `CONFIG` block. BF16 uses the same exact FP32 fixture state and performs the
BF16 cast on CPU before moving tensors to GPU or NPU.

## Offline Titan/HF validation on GPU

Use the supplied dual-GPU-endpoint scenario to validate that offline capture
reproduces the existing paired Titan/HF comparison. The two captures are
sequential and may use the same physical GPU:

```bash
python tests/glm5_2_parity/titan_hf_gpu_fp32_random.py --data
python tests/glm5_2_parity/titan_hf_gpu_fp32_random.py --actual-capture
python tests/glm5_2_parity/titan_hf_gpu_fp32_random.py --expected-capture
python tests/glm5_2_parity/titan_hf_gpu_fp32_random.py --compare
```

The named aliases `--titan-gpu-capture` and `--hf-gpu-capture` are equivalent.

Comparison rejects different test plans, fixture tensors, effective
configuration, incomplete artifacts, and corrupt shards. Test-repository,
TorchTitan, and TorchTitanTurbo source metadata is recorded for diagnostics but
does not gate fixture loading, capture, or comparison. This allows one
experiment to run from uncommitted or differently committed checkouts while its
scenario configuration and exact fixture contents remain identical.

## Framework self-check

`tests/unit_tests/test_glm5_2_parity_artifacts.py` is not part of capture or
compare execution. It is a fast CPU regression suite for the artifact protocol:
dtype-preserving round trips, checksums, incomplete or failed runs, configuration
and fixture mismatch rejection, offline tensor comparison, the report contents
links, and the default model-size budget. Run it after changing the framework:

```bash
python -m pytest tests/unit_tests/test_glm5_2_parity_artifacts.py -q
```
