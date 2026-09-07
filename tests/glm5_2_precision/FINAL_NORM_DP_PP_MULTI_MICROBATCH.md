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

The enclosing-state trace identifies why the target pre-forward hook is
skipped. TorchTitan places `model.norm` and `model.lm_head` in one grouped FSDP
unit, but `ChunkedLossWrapper` sets `_skip_lm_head=True`: the stage model
forward runs the norm and returns, then the loss wrapper invokes the LM head
separately. PyTorch's grouped FSDP forward hooks keep the group open until both
members have run. At the second `STAGE F begin`, the owner and group training
states are already `IDLE` and the parameter is `SHARDED`/128, but
`iter_forward_root` still points to the grouped owner. At `FINAL NORM begin`,
the root state changes to `FORWARD` while the group itself remains `IDLE`; no
target-group pre-forward or unshard event is issued. The delayed grouped hook
therefore spans pipeline schedule actions instead of enclosing one coherent
forward.

A test-only state-reset ablation closes this causal chain. Immediately before
the affected stage forward, the probe clears the grouped unit's stale
`_modules_to_run_forward` tracking and its matching `iter_forward_root`, then
allows the standard FSDP hooks to run. The three observed final-norm weight
sizes become `[256, 256, 256]` (initialization probe plus two real
microbatches), both microbatch contributions enter
`unsharded_accumulated_grad`, and the final parameter gradient aligns with the
single-card reconstruction: cosine 0.9999991, L2 residual 1.44015e-4, and
maximum absolute residual 3.8838e-5. This is mechanism validation, not a
production fix.

A stronger structural ablation replaces the single grouped
`fully_shard([model.norm, model.lm_head], ...)` call with independent
`fully_shard(model.norm, ...)` and `fully_shard(model.lm_head, ...)` calls.
It does not clear or otherwise mutate any FSDP runtime state. The capture
confirms `ungrouped_fsdp_unit=1`, `forced_reset_group_forward_state=0`, and
final-norm local weight sizes `[256, 256, 256]`. The second-microbatch change
in `unsharded_accumulated_grad` matches its boundary reconstruction on the two
final-stage DP ranks with cosines 0.9999975 and 0.9999973. The candidate and
single-card runs also report the same loss 8.21366 and pre-clip global gradient
norm 1.5566.

The structural run captures all 127 trainable parameters, split across the two
pipeline stage owners, and compares them with native `msprobe compare -m auto`.
All 508 parameter-state rows pass: gradient presence, optimizer-consumed
post-clip gradient, one-step parameter update, and updated parameter for every
parameter. Concatenating the 127 logical tensors gives:

| Tensor family | Cosine | L2 residual | Maximum absolute residual |
| --- | ---: | ---: | ---: |
| Post-clip parameter gradient | 0.9999979854 | 2.10921e-3 | 1.99124e-4 |
| One-step parameter update | 0.9998117490 | 6.20892e-2 | 1.59578e-3 |
| Updated parameter | 0.9999999958 | 6.20892e-2 | 1.59578e-3 |

The one-step update is compatible but not bitwise identical. Its lowest
per-parameter cosine is 0.9967207, the already documented first-step AdamW
sign-amplification behavior for near-zero BF16 gradient differences; every
update row remains a native MindStudio `pass`. The final-norm pre-clip gradient
itself has cosine 0.9999987, L2 residual 1.76637e-4, and maximum absolute
residual 4.57764e-5, while its boundary reconstruction is bitwise identical to
the single-card reference. This establishes independent norm/LM-head FSDP
units as the preferred production-fix shape over an internal-state reset,
subject to the still-deferred GPU attribution check.

## PP4 replicated and sharded DP structural validation

The independent norm/LM-head unit ablation was repeated on the previously
divergent DDP2-PP4 and FSDP2-PP4 cases. Both use the same seed-61,
global-batch-16 fixed fixture, outer accumulation 4, GPipe, and two
microbatches per pipeline schedule. Thus the comparison crosses pipeline depth
and separately exercises replicated and sharded data parallelism without
changing the input or optimizer step.

| Structural PP4 case | Loss | Global pre-clip grad norm | MindStudio rows | Final-norm pre-clip cosine |
| --- | ---: | ---: | ---: | ---: |
| DDP2-PP4 | 8.14223862 | 1.40195525 | 518/518 pass | 0.9999995317 |
| FSDP2-PP4 | 8.14223862 | 1.40195525 | 518/518 pass | 0.9999995317 |

The four native `msprobe compare -m auto` stage reports contain 116, 192,
128, and 82 rows. All 508 parameter-state rows pass and cover all 127
parameters across gradient presence, optimizer-consumed post-clip gradient,
one-step update, and updated parameter. Concatenating each logical tensor
family gives the same result for DDP2-PP4 and FSDP2-PP4:

| Tensor family | Cosine | L2 residual | Maximum absolute residual |
| --- | ---: | ---: | ---: |
| Post-clip parameter gradient | 0.9999990563 | 1.37386e-3 | 1.13346e-4 |
| One-step parameter update | 0.9997472272 | 7.19929e-2 | 1.59825e-3 |
| Updated parameter | 0.9999999952 | 7.19929e-2 | 1.59825e-3 |

The final-norm pre-clip gradient has candidate/reference norms
0.10011485/0.10011815, L2 residual 9.69441e-5, and maximum absolute residual
3.43323e-5. Both final-stage DP ranks report `ungrouped_fsdp_unit=1`, and all
nine observed calls (one initialization probe plus eight training forwards)
use the complete 256-element final-norm weight. DDP2-PP4 and FSDP2-PP4 are
also bitwise identical across all 381 parameter gradient/update/updated-value
tensors, the real final-norm pre-clip gradient, and its boundary
reconstruction.

This result shows that removing the erroneous grouped unit covers PP4 and both
replicated and sharded DP paths. The issue remains open until the production
change and its regression coverage are landed, and backend attribution remains
unresolved until the matched GPU experiment runs.

## TP-composed structural validation

The final uncovered matrix row, FSDP2-TP2-PP2, was repeated with independent
norm and LM-head FSDP units. It uses the same seed-61, global-batch-16 fixed
fixture, outer accumulation 4, and two microbatches per 1F1B schedule. The
two stage-owner reports contain 244 and 274 rows, all `pass`; 508 rows cover
gradient presence, post-clip gradient, update, and updated value for all 127
parameters.

| Metric | Structural result |
| --- | ---: |
| Loss | 8.14235306 |
| Global pre-clip gradient norm | 1.40190768 |
| Final-norm pre-clip cosine | 0.9999982582 |
| Final-norm candidate/reference norm | 0.10011943 / 0.10011815 |
| Final-norm L2 / maximum absolute residual | 1.86870e-4 / 6.14673e-5 |

The original matrix's `norm.weight` gradient cosine was 0.992356. After the
split, the concatenated logical parameter-gradient cosine is 0.9999813810,
the one-step update cosine is 0.9948072954, and the updated-parameter cosine is
0.9999999006. Their L2 residuals are 6.10228e-3, 3.26305e-1, and 3.26305e-1.
The minimum individual update cosine of 0.9104859 retains the already confirmed
TP first-step AdamW sign-amplification signature; every native comparison row
passes.

All four final-stage ranks report `ungrouped_fsdp_unit=1`. Every rank observes
nine final-norm calls, and every call uses the complete 256-element weight.
This closes the structural coverage gap: the correction now works with TP in
addition to PP2/PP4 and replicated/sharded DP. It remains a test-only ablation
until the production change is landed, and GPU/backend attribution is still
unresolved.

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
- It is not yet established whether the same grouped-FSDP lifecycle occurs on
  GPU. The tested NPU mechanism is confirmed, but backend attribution remains
  unresolved.
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
