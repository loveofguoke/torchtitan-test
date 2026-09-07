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
where the observed value first changes; it does not yet identify the faulty
backend or prove the exact implementation defect.

## What is not established

- It is not established that the defect is NPU-specific.
- It is not established that the defect is backend-independent.
- It is not established whether a microbatch contribution is overwritten,
  incorrectly accumulated, incorrectly scaled, or incorrectly reduced.
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
