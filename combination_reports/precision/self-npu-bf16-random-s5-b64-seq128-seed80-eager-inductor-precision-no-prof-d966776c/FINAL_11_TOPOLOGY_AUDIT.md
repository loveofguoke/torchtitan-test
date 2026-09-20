# GLM-5.2 distributed eager/Inductor precision audit

- Date: 2026-09-20
- Device: Ascend NPU, 8 devices
- Training: BF16 parameters, FP32 reduction, global batch 64, sequence length 128
- Seed: 80
- Captures: 2 eager repeats and 2 Inductor repeats per topology
- Steps: 5 optimizer steps per capture
- Comparison mode: `--require-all`
- Result: **11/11 PASS**

| Topology | Mean loss absolute error | Mean loss absolute relative error | Grad norm signed mean relative error | Report |
|---|---:|---:|---:|---|
| PP8 | 0.001251602 | 0.021466% | -0.037448% | [HTML](pp8/self-npu-pp8-vs-npu-pp8-bf16.html) |
| TP8 | 0.000603676 | 0.011393% | +0.785913% | [HTML](tp8/self-npu-tp8-vs-npu-tp8-bf16.html) |
| FSDP2×TP4 | 0.000643730 | 0.012401% | +0.764728% | [HTML](fsdp2-tp4/self-npu-fsdp2-tp4-vs-npu-fsdp2-tp4-bf16.html) |
| FSDP2×TP2×PP2 | 0.000812244 | 0.014721% | +0.750427% | [HTML](fsdp2-tp2-pp2/self-npu-fsdp2-tp2-pp2-vs-npu-fsdp2-tp2-pp2-bf16.html) |
| FSDP2×TP4×EP8 | 0.000470638 | 0.008351% | -0.003025% | [HTML](fsdp2-tp4-ep8/self-npu-fsdp2-tp4-ep8-vs-npu-fsdp2-tp4-ep8-bf16.html) |
| FSDP4×TP2 | 0.000680542 | 0.012442% | +0.759506% | [HTML](fsdp4-tp2/self-npu-fsdp4-tp2-vs-npu-fsdp4-tp2-bf16.html) |
| DDP8 | 0.000640106 | 0.011637% | +0.052688% | [HTML](ddp8/self-npu-ddp8-vs-npu-ddp8-bf16.html) |
| FSDP8 | 0.000624561 | 0.011302% | +0.053797% | [HTML](fsdp8/self-npu-fsdp8-vs-npu-fsdp8-bf16.html) |
| HSDP2×4 | 0.000689602 | 0.012457% | +0.055527% | [HTML](hsdp2x4/self-npu-hsdp2x4-vs-npu-hsdp2x4-bf16.html) |
| HSDP4×2 | 0.000737381 | 0.013407% | +0.053616% | [HTML](hsdp4x2/self-npu-hsdp4x2-vs-npu-hsdp4x2-bf16.html) |
| EP8 (FSDP8×EP8) | 0.000517273 | 0.009366% | +0.049745% | [HTML](ep8/self-npu-fsdp8-ep8-vs-npu-fsdp8-ep8-bf16.html) |

## Artifact audit

- 11 precision summaries exist and all have `passed=true`.
- All 11 topologies have bitwise-identical eager repeat loss/grad-norm series.
- All 11 topologies have bitwise-identical Inductor repeat loss/grad-norm series.
- 44/44 artifacts contain exactly steps `[1, 2, 3, 4, 5]`.
- 44/44 `metrics.jsonl` SHA-256 digests match their manifests.
- 44/44 manifests record clean `torch_npu` and TorchTitanTurbo source trees.

## Source provenance

- `torch_npu`: `eeded0663bb6eebd98ed65dc31ae499d1d3932c9`
- TorchTitanTurbo baseline runs: `638cef79a6f0371c1a542f367860528d319a1a64`
- TorchTitanTurbo HSDP2×4, HSDP4×2, and standalone EP8 reruns: `4af0147c0d6457edf503896f8272c915ca634bb7`
- Test harness timeout forwarding: `b759b2fd923f9be7933484fc91b9d25def9fad52`
- Test harness compatibility regression tests: `7f4e522091558bd551e60e89d4e779dff0d54bc3`

The Turbo delta between the two recorded commits only stabilizes serialization
order for NPU compile-option metadata and does not change model arithmetic.
The three topologies that exercised that path were rerun after the fix was
committed so their manifests record a clean source tree.
