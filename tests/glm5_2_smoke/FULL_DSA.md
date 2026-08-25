# GLM-5 Full DSA test mapping

The Full DSA feature is split across three repositories:

| Responsibility | Repository and document |
|---|---|
| PyTorch reference model, sharing schedule, distributed contracts, GPU ops, and general candidates | TorchTitan `torchtitan/models/glm5/FULL_DSA.md`, `ops/README.md`, and `ops_candidate/README.md` |
| Triton-Ascend/native SparseMLA, graph/HCCL compatibility, and NPU candidates | [TorchTitanTurbo Full DSA](https://github.com/loveofguoke/TorchTitanTurbo/blob/feat/glm5-full-dsa-npu/torchtitanturbo/models/glm5/FULL_DSA.md), [ops](https://github.com/loveofguoke/TorchTitanTurbo/blob/feat/glm5-full-dsa-npu/torchtitanturbo/models/glm5/ops/README.md), and [ops_candidate](https://github.com/loveofguoke/TorchTitanTurbo/blob/feat/glm5-full-dsa-npu/torchtitanturbo/models/glm5/ops_candidate/README.md) |
| Smoke, parity, precision, graph/performance A/B, reports, failure artifacts, and promotion decisions | torchtitan-test; [performance optimization matrix](../glm5_2_performance/OPTIMIZATION.md) |

## Test layers

1. CPU unit tests prove absorbed SparseMLA equals dense K/V expansion and that
   shared layers reuse the exact source top-k tensor.
2. Eager single-device smoke proves the complete training loop runs with the
   reference implementation.
3. HF parity uses the no-sharing `glm5_debugmodel` to validate the mathematical
   implementation with identical weights and inputs.
4. Full DSA reference parity compares GPU and NPU with the same sharing
   schedule and token plan.
5. Operator parity compares each optional GPU/NPU implementation against its
   same-device PyTorch reference before cross-device comparison.
6. Distributed smoke and precision tests expand only after single-device
   forward, backward, loss, and grad norm pass.

The formal cross-device entry point is
`tests/glm5_2_precision/full_dsa_migration_benchmark.py`. It reuses the formal
checkpoint, fixed-token plan, capture artifact, error standards, and report
modules without changing the established migration benchmark. Its `all`
selection contains only non-PP topologies because top-k is not transported
between pipeline stages.

The HF-compatible mode is `glm5_debugmodel`, which owns an indexer on every
layer and loads no operator overrides. `glm5_full_dsa_debugmodel` has the same
small dimensions but enables cross-layer sharing, so its HF counterpart must
use the same sharing schedule.

Every invocation must preserve its command, environment, source revisions,
runtime log, manifest, and failure traceback. A failed run directory is an
artifact to diagnose, not evidence that the next topology should be skipped.

Use `--full-dsa` and `--full-dsa-kernel` instead of manually coupling model and
operator choices. See `tests/glm5_2_performance/OPTIMIZATION.md` for the formal
kernel, communication, memory, and recomputation A/B matrix.
