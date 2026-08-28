# Portable GLM-5.2 parity workflow

This framework lives in `torchtitan-test` and imports the source-installed
`torchtitan` package. Run every command below from the `torchtitan-test`
repository root. NPU captures also import source-installed `torchtitanturbo`
before PyTorch selects its device backend.

Parity scenarios are single-endpoint exploratory runs. They do not expose
`--topology` or `--topology all`; distributed long-training acceptance belongs
to the formal precision suite. The GPU/NPU and Titan/HF command sets below are
therefore the complete device matrix for this package.

The GLM-5.2 parity suite supports four execution modes:

- `paired` (default): run two endpoints in one process and immediately report.
- `prepare`: create one immutable fixture containing exact model state and data.
- `capture`: run one endpoint and write a portable, checksummed artifact.
- `compare`: compare two artifacts on CPU and write the same HTML diagnostics.

## Command-line parameters

Offline scenario files expose one required stage at a time:

| Parameter | Purpose | Default |
|---|---|---|
| `--data` | Materialize the exact model state, test plan, and every case tensor. | no stage; one stage is required |
| `--actual-capture` | Run the configured `actual` endpoint and save its portable trace artifact. | no stage; one stage is required |
| `--expected-capture` | Run the configured `expected` endpoint and save its portable trace artifact. | no stage; one stage is required |
| endpoint alias | Scenario-generated alias such as `--npu-capture`, `--gpu-capture`, `--titan-gpu-capture`, or `--hf-gpu-capture`; equivalent to its actual/expected stage. | scenario-dependent |
| `--compare` | Compare the two complete artifacts on CPU and render the HTML report. | no stage; one stage is required |
| `--print-config` | Print the effective scenario, digests, and resolved fixture/artifact/report paths without running pytest. | no stage; one stage is required |
| `--force` | Start a new generation for the selected stage. With `--data`, remove the fixture and every dependent capture/report/log first; with capture or compare, remove only that stage and stale report output. | disabled |

Paired scenarios expose `--run` and `--print-config`; `--run` constructs both
endpoints in one process and immediately reports. There are intentionally no
CLI overrides for precision, data case, layers, components, model size, seeds,
endpoint, or device. Those values live in the copied scenario file's `CONFIG`
block so its filename and configuration digest identify one reproducible
experiment. The current scenario values are therefore the defaults and the
only values for that file.

Parity follows the same generation-safe rerun contract as the formal suites.
Use `--force` once to start a new selected generation. If that command is
interrupted, rerun the same stage without `--force`: a complete fixture or
capture is skipped, while an incomplete/failed output is archived and retried.
Forcing `--data` removes both captures and the report before regenerating the
fixture, so old and new generations cannot be mixed. Captures additionally
validate the exact fixture digest, test plan, scenario configuration, and suite
version. Source commits and dirty-worktree metadata remain diagnostic only.

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
same-layer direct attention or MoE/FFN output remains acceptable, and `FAIL`
for stable, out-of-band, or direct-output differences outside tolerance. ULP
values remain in the report as scale diagnostics but are not a fixed pass/fail
limit. FP32 selection remains exact.

Relative error growth is diagnostic only because growth measured from a
near-zero baseline can be arbitrarily large. A locally explainable boundary
change is downgraded to `FAIL` only when the same-layer direct module output
exceeds its configured numerical tolerance at the changed token. This marks
where a discrepancy becomes important; it does not claim that the top-k node
is the root cause.

Report rows follow GLM execution semantics instead of path hierarchy. Forward
rows are ordered from input to output, internal module outputs precede their
parent output, and each decoder-layer output is last in that layer. Activation
gradient rows use reverse execution order. New captures retain hook completion
order to resolve events inside a semantic stage; old artifacts use the same
GLM semantic ordering as a compatible fallback. Error trend connects only the
major numeric dataflow checkpoints (norm, attention output, norm, FFN/MoE
output, layer output), not unrelated branches or every trace leaf. A numeric
checkpoint's PASS/FAIL is always determined by its configured tolerance;
neither trend growth nor a neighboring component changes that verdict.
Existing artifacts do not need to be recaptured for this report change. A CPU
`compare` rerun is sufficient; old artifacts simply use semantic ordering where
the optional execution-order tags are absent.

HTML result tables keep their header visible inside a bounded scroll area.
Buttons above each table independently fold path, dtype, value, metric, and
diagnostic column groups without hiding the component or status columns.
The exploratory report now follows the formal precision report's visual
hierarchy: linked contents, a summary hero and status cards, card-style charts,
consistent status colors, and sticky headers for every table. Trace
content and top-k boundary diagnostics are unchanged. Existing captures can be
reused; rerun only `compare` to render the updated HTML.

## Report contents and acceptance

The exploratory HTML begins with the effective configuration and summary,
then presents forward values, activation gradients, parameter gradients,
canonical parameters, logits, loss, and discrete indexer/router decisions in
semantic execution order. Numerical rows include dtype/shape, max and mean
absolute error, relative metrics, cosine similarity, mismatch count, trend,
and status. Top-k rows include readable ranked scores for all positions,
cutoff/precision bands, changed candidates, boundary classification, and the
same-layer downstream numerical check.

`PASS` means every decisive requested component satisfies its configured
tolerance. `BOUNDARY_PASS` is the documented BF16 discrete cutoff case and is
kept visually distinct from exact PASS. `TRACE` is diagnostic-only evidence;
it does not independently fail the suite. `FAIL` identifies the first observed
checkpoint outside its contract, not necessarily the root-cause module. This
report is for localization; long-training delivery acceptance remains owned by
`glm5_2_precision`.

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

The reusable harness lives in `tests/glm5_2_parity/suite.py`; the historical
`tests/unit_tests/test_glm5_parity.py` module is now only the stable pytest
discovery and compatibility entry. The CPU regressions are not part of capture
or compare execution. They cover the artifact protocol and lifecycle:
dtype-preserving round trips, checksums, incomplete or failed runs, configuration
and fixture mismatch rejection, offline tensor comparison, the report contents
links, default model-size budget, forced generation reset, completed-stage skip,
and failed-stage retry. Run them after changing the framework:

```bash
python -m pytest \
  tests/unit_tests/test_glm5_2_parity_artifacts.py \
  tests/unit_tests/test_glm5_2_parity_workflow.py -q
```
