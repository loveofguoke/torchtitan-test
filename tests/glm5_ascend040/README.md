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
