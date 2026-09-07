# DP+PP multi-microbatch final-norm gradient issue

## Status marker

- Issue ID: `GLM5-DIST-FINAL-NORM-MULTIMB`
- Status: **OPEN**
- Backend attribution: **UNRESOLVED**
- GPU validation: **DEFERRED**
- Scope: eager mode only; `torch.compile` is excluded

Do not describe this issue as an NPU-specific defect. The divergence is
confirmed on the tested NPU runtime, but there is no valid matched GPU result
for the decisive DP+PP microbatch crossing. It may be NPU-specific or it may be
a backend-independent PyTorch/TorchTitan behavior.

## Confirmed observation

The final RMSNorm input, output, output gradient, and independently
reconstructed weight gradient align with the single-card reference. The real
`norm.weight.grad` diverges before gradient clipping when data parallelism is
combined with pipeline parallelism and one pipeline schedule owns multiple
microbatches.

| Topology/control | PP microbatches per schedule | Pre-clip gradient cosine | Result |
| --- | ---: | ---: | --- |
| PP8 | 2 | 1.000000 | aligned |
| FSDP8 | none | 1.000000 | aligned |
| DDP2-PP4 | 2 | 0.992461 | divergent |
| FSDP2-PP4 | 2 | 0.992340 | divergent |
| FSDP2-PP2, global batch 4 | 2 | 0.975016 | divergent |
| FSDP2-PP2, global batch 4 | 1 | 1.000000 | aligned |
| FSDP2-PP2, outer accumulation 4 | 1 | 1.000000 | aligned |

For FSDP2-PP4, the reconstructed gradient norm is 0.100117 versus reference
0.100118, while the real pre-clip parameter-gradient norm is 0.075296 versus
reference 0.100118. The mismatch is therefore present before clipping.

## Localized boundary

The first observed divergent boundary is after correct per-microbatch backward
tensors have been produced and before the optimizer sees the final parameter
gradient:

```text
aligned per-microbatch backward tensors
    -> pipeline microbatch accumulation
    -> data-parallel REDUCE_GRAD transition
    -> divergent norm.weight.grad
    -> gradient clipping
```

In the inspected runtime, this corresponds to the transition from pipeline
stage `backward_maybe_with_nosync()` to `perform_reduce_grad()`. This identifies
where the observed value first changes.

For the FSDP2-PP2, global-batch-4 case, the immediate mechanism is now
confirmed. Ignoring one initialization probe call, the two real final-norm
forwards see local weight sizes 256 and 128 respectively. The first microbatch
therefore uses the complete unsharded weight, while the second uses only the
local FSDP shard. The first microbatch's complete gradient is saved in
`unsharded_accumulated_grad`; that buffer remains bitwise unchanged after the
second backward. The second microbatch instead appears only as the local
128-element `parameter.grad` shard. At the final reduction, each output shard
contains the globally reduced first-microbatch contribution plus only its own
rank's second-microbatch contribution.

The matched one-microbatch control has one real final-norm forward with local
weight size 256 and remains aligned with the single-card reconstruction
(cosine 0.9999991, L2 residual 1.49864e-4). The two-microbatch case has cosine
0.9749982 and L2 residual 3.12679e-2.

The ordered FSDP lifecycle trace is identical on both final-stage ranks. At
the second `STAGE F begin`, the final-norm parameter is already `SHARDED` with
128 local elements. No target-group `pre_forward`, `unshard`, or
`wait_for_unshard` event occurs before `FINAL NORM begin`; the final norm runs
and returns with that shard. The `SHARDED -> UNSHARDED` transition occurs only
after the stage forward has returned, making the all-gather too late for the
second microbatch's computation. All three observed final-norm invocations
occur outside an autograd graph task, so the 128-element call is not activation
checkpoint backward recomputation.

Three diagnostic ablations further constrain the mechanism:

- Synchronizing the NPU immediately before `perform_reduce_grad()` produces a
  bitwise-identical divergent gradient, excluding a simple unfinished-kernel
  race before reduction.
- Enabling native FSDP synchronization for `last_backward=True` is too late:
  the second forward has already used the sharded parameter, and the final
  gradient remains bitwise identical.
- Directly all-reducing the existing local shards is invalid because ranks own
  different parameter coordinates; it worsens the final-norm L2 residual to
  3.81001e-2 and is not a fix.

## What is not established

- It is not established that the defect is NPU-specific.
- It is not established that the defect is backend-independent.
- It is not yet established which enclosing FSDP state's training-state
  transition causes the target parameter group's pre-forward materialization
  to occur after the second stage forward, or whether the same lifecycle occurs
  on GPU.
- The exact FSDP2-PP2 mechanism above must not yet be generalized to the DDP+PP
  result without an equivalent parameter-lifecycle trace.
- MindStudio's native result column reports these saved tensors as `pass`; the
  stricter investigation is based on the non-unit cosine and material gradient
  norm discrepancy.

Earlier GPU distributed runs do not close this question. The matching
PP2-FSDP2 run did not complete, and the successful historical GPU runs did not
capture the final-norm boundary reconstruction and real pre-clip gradient.

## Deferred validation

When GPU execution is resumed, run the matched eager, one-step, fixed-fixture
crossing:

1. FSDP2-PP2, global batch 4, two PP microbatches per schedule.
2. FSDP2-PP2, global batch 4, one PP microbatch per schedule.
3. FSDP2-PP2, global batch 16, outer accumulation 4, one PP microbatch per
   schedule.

Only after those results are available should the attribution be changed:

- GPU aligned and NPU divergent: classify as NPU-specific and fix in Turbo.
- GPU divergent with the same signature: classify as backend-independent and
  address the PyTorch/TorchTitan path.

Until then, retain the labels **OPEN**, **BACKEND ATTRIBUTION UNRESOLVED**, and
**GPU VALIDATION DEFERRED**.
