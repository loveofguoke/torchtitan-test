# Ascend GLM on TorchTitan 0.4.0

`worker.py` calls the Ascend GLM recipe and unmodified Trainer under torchrun.
It reuses the canonical topology registry; the additional two/four-card cases
are minimal diagnostic compositions, not changes to existing experiment presets.

Use the fixed TorchTitan 0.4.0 source and the Ascend integration branch through
`PYTHONPATH`. Set `GLM_TOPOLOGY`, `GLM_OUTPUT`, `GLM_REMAT=selective|full`, and
optionally `GLM_MTP=1`. Launch from the fixed TorchTitan root so its tokenizer
fixture is resolved. Every rank writes JSON with the explicit degrees, remat
policy, step count, full-precision metrics and verdict. A run passes only after
all expected rank files report two completed steps and finite metrics.

The worker does not disable remat, install dependencies, select graph backends,
or modify the production recipe. PP+MTP is rejected. Runs are short functional
checks, not cross-device accuracy, deterministic equivalence or performance tests.

`GLM_RUNTIME=turbo` instead imports Turbo before constructing the current
TorchTitan GLM recipe. Use the current TorchTitan checkout, not the 0.4.0
snapshot, for this route. `TORCHTITAN_TURBO_PATCHES` selects fresh-process
patch A/B; each rank records Turbo's actual selected/applied groups. This is a
separate experiment from the Ascend fused implementation above.

For native initializer and TP loss checks, add this directory to `PYTHONPATH`
and use `-p turbo_plugin --noconftest` to import Turbo before collecting
the existing native-op tests. It does not silently import the Ascend bootstrap.
Compare `all`, `none`, and one-group removals in independent subprocesses.

`run_turbo_ab.sh` supports `GLM_AB_PHASE=components|smoke|all`. Smoke endpoints
restore a single shared full-model fixture through PyTorch's public distributed
state-dict API and record its SHA256. The comparison reports exact metric
equality and deltas without inventing a tolerance or comparing different
topologies as numerical equivalents. Component-only evidence is not promoted
to an end-to-end patch-removal verdict.

`GLM_AB_DEVICES` restricts execution to an explicit physical-device list. Each
endpoint checks those devices against the full npu-smi process table, without
requiring or taking other users' devices. `GLM_AB_COMMON_PATCHES` holds an
explicit compiler compatibility group constant across all endpoints. If it is
set, `none` means no legacy groups, not zero patches; inspect the recorded
selection/applied list rather than the short directory label.

`GLM_PRECISION=bfloat16|float32` selects the mixed-precision parameter/compute
dtype, retaining the recipe's master-state and reduction dtype. Each rank
records all three dtypes. `GLM_DETERMINISTIC=1` explicitly enables strict
deterministic algorithms; no warn-only escape is used. Ordinary compatibility
smoke defaults to `0` and must not be presented as deterministic numerical
equivalence evidence. Same-fixture deltas without determinism are observations,
not a pass criterion for a non-computation change.

`GLM_NPU_OVERLAY` explicitly prepends an independently prepared TorchNPU package
copy. It never modifies the installed package; the suite records its root and
each rank records the actual import path, version and Flex kernel SHA256. An
overlay run is distinct from installed-wheel acceptance and must be labeled as
such when discussing patch removal.

`GLM_AB_PHASE=matrix` runs the remaining canonical two/four/eight-rank topologies,
including PP, HSDP, combined Full remat, MTP and an FP32 FSDP8 endpoint. It uses
native seeded initialization rather than claiming same-fixture equivalence; PP
partitions and MTP parameters are not loaded from a base-model single-rank
fixture. Each case retains its own runtime and rank evidence, even on failure.

`GLM_AB_SHARED_DEVICES=1` is an explicit user-authorized sharing policy, not the
default. It preserves the complete process table before each endpoint and does
not terminate existing processes. Such runs provide functional smoke evidence,
not isolated throughput/latency comparisons. Omit it for the default idle-only
policy. Use `GLM_AB_DEVICES=0,1,2,3,4,5,6,7` for the eight-card matrix.

`GLM_AB_MATRIX_CASES='fsdp8:8 cp4:4'` selects explicit diagnostic/resume cases;
`GLM_AB_MATRIX_EXTRAS=0` omits the default Full/MTP/FP32 tail for that probe.
Use the manifest's actual applied groups: if `remat` is held in common after a
multi-rank null-storage crash, the endpoint is not legacy-patch-free even if its
short directory label begins with `none`.

For current TorchTitan local Inductor probes, set
`GLM_LOCAL_COMPILE_REGIONS=rope,fused_binary_activation,moe_aux_loss,fp32_to_bf16_split`.
The worker uses the public `model.local_compile.regions` API and records the
actual region list. Select the existing TorchNPU loader contract before process
startup: `TORCHINDUCTOR_NPU_BACKEND=default` for Ascend Triton or `dvm` for DVM.
An empty region list is the ordinary Trainer path; FlexAttention itself remains
an always-compiled helper. These are local-compile tests, not a claim that
GraphTrainer whole-step compilation was exercised. Strict deterministic probes
use `GLM_DETERMINISTIC=1`, never warn-only mode.

The worker also accepts `GLM_STEPS` (default 2), `GLM_SEQ_LEN` (default 32),
`GLM_MICROBATCH_TOKENS` (default 256), and `GLM_SEED` (default 42). For a
500-step NPU capture, use a fresh output directory, retain recomputation, set
`GLM_STEPS=500`, and preserve the shared checkpoint fixture for the later GPU
reference. Each rank writes full-precision metrics incrementally to
`rank-N-metrics.jsonl`, the resolved config to `rank-N-config.txt`, and final
step-count/finite-value evidence to `rank-N.json`. Normal launcher exit is also
required. This capture alone is not a CUDA/NPU accuracy comparison.

The capture script also saves the actual CPU microbatches under each run's
`inputs/`, with an incremental SHA256 manifest. For a same-input reference,
`GLM_INPUT_REPLAY=/path/to/inputs` verifies and reconstructs those exact
microbatches instead of retokenizing or trusting equal RNG seeds. Standalone
`GLM_INPUT_CAPTURE=/fresh/path` and replay are mutually exclusive and support
single-rank validation only. Existing capture directories are never overwritten.

After the upstream gathered-attention migration, `GLM_NPU_FLEX=1` explicitly
applies Turbo's `NpuGlm5FlexTransform`; it is not a global model patch or an
implicit fallback. Keep it unset for older TorchTitan revisions that already
build FlexAttention configs. The actual selection is recorded in rank evidence.
