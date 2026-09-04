# MindStudio eager precision alignment

This test deliberately excludes `torch.compile`. It uses the public interfaces
shipped by `mindstudio-probe` rather than private comparison helpers.

## Standard used

- Capture the same complete logical step on both sides with
  `msprobe.pytorch.PrecisionDebugger` (`start`, `stop`, and `step`). The selected
  step is zero-based. A ten-step run is not required by msProbe; this diagnostic
  captures only `step0`.
- Hold the model checkpoint and input batch constant. The fixture records a
  SHA-256 digest for the topology-invariant global batch.
- Use `statistics` plus `mix` for cross-parallel-strategy graph merging. This
  mode is diagnostic: it compares Max, Min, Mean, and L2 norm and does not prove
  real-tensor equivalence.
- Use `tensor` plus `msprobe compare` when both sides have the same rank layout
  or when comparing tensors saved at a topology-invariant logical boundary.

Official references:

- [PyTorch data dump](https://www.hiascend.com/document/detail/zh/mindstudio/2610/msTT_msIT/msProbe/docs/zh/user_guide/dump/pytorch_data_dump_instruct.md)
- [PyTorch accuracy comparison](https://www.hiascend.com/document/detail/zh/mindstudio/2610/msTT_msIT/msProbe/docs/zh/user_guide/accuracy_compare/pytorch_accuracy_compare_instruct.md)
- [Cross-partition graph visualization](https://www.hiascend.com/document/detail/zh/mindstudio/2610/msTT_msIT/msProbe/docs/zh/user_guide/accuracy_compare/pytorch_visualization_instruct.md)

## Supported topology claims

| Reference versus candidate | Native msProbe route | Claim allowed |
| --- | --- | --- |
| Same rank layout | `msprobe compare` | Real-tensor accuracy when task is `tensor` |
| Single versus TP/PP/VPP, with identical DP | `msprobe graph_visualize` | Statistics-level compatibility diagnostic |
| Single versus DDP/FSDP/HSDP | None for whole-network cross-layout merge | Not a supported cross-topology claim because DP differs |
| Any comparison involving CP or EP | None in graph merge | Not supported |

The 26.1 graph-merging documentation names Megatron and MindSpeed-LLM as the
validated frameworks. A successful TorchTitan invocation does not by itself
establish that every TorchTitan parallel operator was merged correctly.

## Current single versus TP4 run

The fixed global batch digest is
`6ed39a056b154d219e84a56d329ad9b5d0c99cb31f1f71477c3b54d9dfc45065`.
The pre-embedding input nodes compare exactly, confirming that the two runs
consume the same data. The one-step training summaries are:

| Metric | Single | TP4 | Absolute difference |
| --- | ---: | ---: | ---: |
| Loss | 8.14224624633789 | 8.1422700881958 | 0.00002384185791 |
| Grad norm | 1.401854395866394 | 1.4018086194992065 | 0.0000457763671875 |

The native graph database contains 13,490 pass nodes, 519 warning nodes, and 64
error nodes. All 64 errors repeat at topology-dependent sites: eight
vocabulary-parallel embedding calls and seven `RoutedExperts` modules across
eight gradient-accumulation micro-batches. The embedding merger compares
rank-local remapped vocabulary IDs and combines rank-local norms as though the
values were ordinary TP shards. `RoutedExperts` compares the local padded-token
count (64) with the unpartitioned count (256). These are graph-merging semantic
mismatches, not sufficient evidence of a numerical model failure.

Consequently, this run does **not** certify single/TP4 precision alignment and
also does **not** demonstrate a TP4 accuracy failure. It establishes that the
public msProbe workflow runs end to end, the inputs match, and the step-level
outputs are close, while exposing a TorchTitan GLM5 graph-merging compatibility
gap. A strict pass requires either a same-topology `tensor` comparison or
package-level `PrecisionDebugger.save()` probes placed after TP reconstruction
at explicitly chosen topology-invariant boundaries.

## Block-boundary tensor result

The follow-up experiment uses the latter strict route. It saves the global
logical input and output of every transformer block with the public
`PrecisionDebugger.save(..., save_backward=False)` interface and compares the
resulting `debug.json` files with the native `msprobe compare` command. No
private msProbe API or project-local numerical comparator participates in the
pass/fail decision.

The run still contains exactly one optimizer step (`step0`). Its global batch is
split into eight gradient-accumulation micro-batches, so each of the eight
blocks contributes eight input tensors and eight output tensors. These repeated
suffixes are micro-batch indices, not training-step indices. Both single and TP4
dumps contain all 128 expected tensors, and the fixed global batch digest is
`606e71ce36450063c2af7db551d14ba5d153ce006c24c1b59a3d23e805cd649a`.

| Block output | Compared micro-batches | Minimum cosine | Maximum Euclidean distance | Maximum absolute error |
| --- | ---: | ---: | ---: | ---: |
| 00 | 8 | 1.000000 | 0.180933 | 0.0234375 |
| 01 | 8 | 0.999999 | 0.444547 | 0.0312500 |
| 02 | 8 | 0.999998 | 0.555112 | 0.0390625 |
| 03 | 8 | 0.999997 | 0.615510 | 0.0390625 |
| 04 | 8 | 0.999996 | 0.684876 | 0.0390625 |
| 05 | 8 | 0.999996 | 0.750007 | 0.0468750 |
| 06 | 8 | 0.999995 | 0.782777 | 0.0468750 |
| 07 | 8 | 0.999995 | 0.812935 | 0.0468750 |

The official CSV contains 128 `pass`, zero failures, and zero unsupported
comparisons. Block 00 inputs are bit-identical; small BF16 differences appear
after the first TP block and accumulate gradually, but no block boundary fails
the msProbe criterion. The one-step summaries remain loss
`8.14224624633789` versus `8.1422700881958` and gradient norm
`1.401854395866394` versus `1.4018086194992065` for single versus TP4.

This result certifies forward tensor alignment at the selected reconstructed
block boundaries for this fixture and seed. It does not independently certify
every internal operator, backward tensor, optimizer state, or another topology.

## Four-card block matrix

The matrix run aggregates all local micro-batches belonging to `step0`, restores
the fixed-fixture sample order across the data-parallel mesh, and saves one
`[16, 128, 256]` input and output tensor per block. Pipeline candidates save
their blocks from one leader rank per stage and are compared stage by stage.

| Candidate | Compared boundaries | Result | Minimum cosine | Maximum absolute error | Loss | Gradient norm |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| TP4 | 16 | 16 pass | 0.999995 | 0.046875 | 8.1422700881958 | 1.4018086194992065 |
| DDP4 | 16 | 16 pass | 1.000000 | 0 | 8.14224624633789 | 1.401854395866394 |
| FSDP4 | 16 | 16 pass | 1.000000 | 0 | 8.14224624633789 | 1.4018546342849731 |
| HSDP2x2 | 16 | 16 pass | 1.000000 | 0 | 8.14224624633789 | 1.4018546342849731 |
| FSDP2-TP2 | 16 | 16 pass | 0.999996 | 0.04443359375 | 8.142351150512695 | 1.4018051624298096 |
| FSDP4-EP2 | 16 | 16 pass | 1.000000 | 0.015625 | 8.14224624633789 | 1.4018564224243164 |
| FSDP4-EP4 | 16 | 16 pass | 1.000000 | 0.015625 | 8.142251968383789 | 1.4018584489822388 |
| FSDP2-TP2-EP2 | 16 | 16 pass | 0.999996 | 0.04443359375 | 8.142288208007812 | 1.4018845558166504 |
| PP4-GPipe | 16 | 16 pass | 1.000000 | 0.015625 | 8.14223861694336 | 1.4023042917251587 |
| PP2-FSDP2 | 16 | 16 pass | 1.000000 | 0.015625 | 8.14223861694336 | 1.400683879852295 |
| PP2-TP2 | 16 | 16 pass | 0.999996 | 0.04443359375 | 8.142352104187012 | 1.4021679162979126 |
| PP2-FSDP2-EP2 | 16 | 16 pass | 1.000000 | 0.015625 | 8.14223861694336 | 1.4006836414337158 |
| PP2-TP2-EP2 | 16 | 16 pass | 0.999996 | 0.04443359375 | 8.142268180847168 | 1.4023077487945557 |

The single-card reference loss is `8.14224624633789` and its gradient norm is
`1.401854395866394`. Across the final official CSV files there are 208 pass
rows, zero failures, and zero unsupported rows. No candidate has a first failing
block. DDP4 is bit-identical to the reference; the only visible forward drift
families are TP (up to `0.046875`) and EP/PP (up to `0.015625`).

The first PP4 diagnostic capture contained 17 rows because the pipeline
scheduler performs a leading shape-propagation call. The final capture enforces
the configured per-step sample count and excludes that non-training call; all
stage tensors then have the required `[16, 128, 256]` shape. The rejected
17-row CSVs are not included in the matrix result.

## Block backward matrix

The highest-priority follow-up captures both `grad_input` and `grad_output` for
every transformer block. Gradients are associated with their forward invocation
index so pipeline backward scheduling cannot reorder samples. Partial TP
gradients are reduced to their global logical values, DP rows are restored to
fixed-fixture order, and stage owners save the reconstructed tensors through
the public `PrecisionDebugger.save()` interface. Native `msprobe compare`
remains the sole pass/fail comparator.

| Candidate | Gradient boundaries | Result | Minimum cosine | Maximum absolute error |
| --- | ---: | --- | ---: | ---: |
| TP4 | 16 | 16 pass | 0.999901 | 3.33786e-6 |
| DDP4 | 16 | 16 pass | 1.000000 | 0 |
| FSDP4 | 16 | 16 pass | 1.000000 | 0 |
| HSDP2x2 | 16 | 16 pass | 1.000000 | 0 |
| FSDP2-TP2 | 16 | 16 pass | 0.999943 | 2.384186e-6 |
| FSDP4-EP2 | 16 | 16 pass | 1.000000 | 1.192093e-6 |
| FSDP4-EP4 | 16 | 16 pass | 1.000000 | 9.53674e-7 |
| FSDP2-TP2-EP2 | 16 | 16 pass | 0.999989 | 2.384186e-6 |
| PP4-GPipe | 16 | 16 pass | 1.000000 | 1.192093e-6 |
| PP2-FSDP2 | 16 | 16 pass | 1.000000 | 1.192093e-6 |
| PP2-TP2 | 16 | 16 pass | 0.999943 | 2.384186e-6 |
| PP2-FSDP2-EP2 | 16 | 16 pass | 1.000000 | 9.53674e-7 |
| PP2-TP2-EP2 | 16 | 16 pass | 0.999989 | 2.384186e-6 |

The final backward report tree contains 416 comparison rows: 208 forward and
208 backward. All 416 rows pass, with zero failures and zero unsupported rows.
No topology has a first failing backward block. As expected for TP, the largest
gradient drift appears toward the bottom of the network after errors accumulate
through more backward blocks, but its absolute magnitude remains below
`3.34e-6` and passes the native criterion.

This closes forward and block-boundary backward alignment for `step0`. Parameter
gradient reconstruction and post-optimizer parameter comparison remain separate
tests; aggregate gradient norm alone does not replace them.

## Parameter-gradient and optimizer-step diagnostic

The next experiment snapshots every trainable parameter before `step0` and,
after the optimizer step, saves its gradient-presence flag, reconstructed global
gradient (when present), update delta, and updated value. The capture uses the
public `PrecisionDebugger.save()` interface and native `msprobe compare` in
`tensor/debug` mode. It keeps the same eager-only fixed fixture, seed 61, global
batch 16, sequence length 128, and one optimizer step.

There are 167 trainable parameters. All tested topologies agree that 127 receive
a gradient and 40 do not. The 40 inactive parameters are the five indexer
parameters in each of the eight attention blocks. The float32 two-element
gradient-presence probes compare exactly and avoid unsupported scalar-cosine
results.

| Candidate | Gradient min cosine / max abs | Update min cosine / max abs | Updated parameter min cosine / max abs |
| --- | ---: | ---: | ---: |
| DDP4 | 1.000000 / 3.725e-9 | 1.000000 / 1.863e-9 | 1.000000 / 1.863e-9 |
| FSDP4 | 1.000000 / 1.4901e-8 | 1.000000 / 1.19209e-7 | 1.000000 / 1.19209e-7 |
| FSDP2-TP2 | 0.188443 / 1.50656e-4 | 0.099647 / 1.59979e-3 | 0.980722 / 1.59979e-3 |
| TP4 | 0.119879 / 1.45134e-4 | 0.008036 / 1.59971e-3 | 0.976565 / 1.59971e-3 |

The DP-only controls are numerically identical within float32 noise. Both TP
cases show the same qualitatively different behavior: small gradient drift is
concentrated in the MoE router gates, with additional `ffn_norm` sensitivity at
TP4. Adam's first update is sign-sensitive, so these small gradients produce
opposite-sign updates for many router elements. For example, the layer-2 TP4
router update has only 50.34% sign agreement with the reference and cosine
0.008036, while its maximum absolute update difference is approximately twice
the step magnitude (`0.00159968`). The updated parameter hides much of that
effect because the common initial value dominates its cosine.

The generated CSV files contain 628 matched rows, no unsupported metrics, and
their `Result` column says `pass`. That status must not be interpreted as an
absolute standalone-tensor acceptance decision in this msProbe version. The
installed tensor indicator calculator applies numerical degradation rules to
grouped module inputs and outputs; a value saved as an independent `debug`
probe has no output group and therefore retains the initialized status while
still receiving valid Cosine, EucDist, and MaxAbsErr values. As a compatibility
diagnostic only, applying the package's legacy `check_accuracy` constants
(`cosine >= 0.99`, `max abs <= 0.001`, with its combined and hard-limit rules)
would reject 7 gradients, 58 updates, and 4 updated parameters for FSDP2-TP2;
and 14 gradients, 59 updates, and 3 updated parameters for TP4. DDP4 and FSDP4
would reject none.

Consequently, DP-only one-step parameter state is aligned, but TP parameter
state is not certified by this experiment. Expanding the same parameter dump
blindly to every four-card composition is lower priority than localizing the TP
router-gradient sensitivity first.

## TP router-gradient localization

The router follow-up saves 11 topology-invariant tensors for each of the seven
MoE layers: gate input and logits, post-activation scores, order-independent
selection and weighted-routing maps, top-k margin, and the backward tensors at
the top-k scores, scores, gate logits, and gate input. It uses the same public
`PrecisionDebugger.save()` plus native `msprobe compare` workflow. A second set
of backward probes all-reduces detached diagnostic copies over the TP mesh; it
does not mutate autograd or the optimizer inputs.

Forward routing is close. TP2 changes one selected expert for 24 of the 14,336
layer-token pairs (0.1674%); TP4 changes 25 (0.1744%). The first difference is
at MoE layer 1. No layer changes more than eight of its 2,048 tokens, and layer
7 changes none. Gate inputs and logits retain minimum cosine 0.999995. The
changed tokens generally sit close to the top-2/top-3 decision boundary, so
small TP forward drift can flip their discrete selection.

The large backward discrepancy has a different signature. Before diagnostic
TP summation, router-score gradient norms are approximately `1/sqrt(TP)` of
the single-card norm: roughly 0.66--0.75 for TP2 and 0.47--0.57 for TP4. The
minimum score/logit gradient cosine is 0.582 for TP2 and 0.324 for TP4. After
summing the detached copies over TP, the minimum cosines become:

| Boundary | FSDP2-TP2 local | FSDP2-TP2 TP-sum | TP4 local | TP4 TP-sum |
| --- | ---: | ---: | ---: | ---: |
| top-k score gradient | 0.602294 | 0.994727 | 0.395796 | 0.996259 |
| post-activation score gradient | 0.581997 | 0.997556 | 0.324262 | 0.997443 |
| gate-logit gradient | 0.582493 | 0.997552 | 0.323928 | 0.997447 |

The weight-gradient identity closes the diagnosis. Reconstructing the gate
gradient as `sum(grad_logits^T @ gate_input)` from the unsummed rank-0 probes
matches the gradient actually passed to the optimizer with cosine 1.0 in every
MoE layer. Reconstructing it from the TP-summed probes instead matches the
single-card gate gradient with cosine 0.999685--0.999996 for TP2 and
0.999366--0.999992 for TP4. The constant norm ratio between the reconstructed
pre-clip gradient and saved optimizer gradient is the recorded global gradient
norm (about 1.4018), as expected from gradient clipping.

This demonstrates that the dominant parameter-state failure is not caused by
the small number of changed top-k selections. The replicated router gate is
receiving a TP-rank-local partial weight gradient at the optimizer boundary.
The likely implementation site is the TP-only `RoutedExperts` local-map input
gradient placement: the code computes a Partial input-gradient layout but sets
`in_grad_placements` to `None` when EP is disabled. A correction should be
validated as a separate ablation before changing production adaptation code.

## Turbo TP router Partial-gradient fix ablation

The NPU adaptation now corrects only the modern, EP-disabled `x_BLD`
RoutedExperts interface. After the upstream GLM5 sharding builder runs, the
Turbo wrapper replaces the missing local-map gradient annotation with four
ordered entries: TP-Partial activation layouts for `x_BLD`, `topk_scores_BLK`,
and the routing-index metadata position, plus the existing routing-count
layout. EP paths and the legacy `x_TD` compatibility rewrite are unchanged.
The implementation is contained in
`TorchTitanTurbo/torchtitanturbo/models/glm5/patch.py`; TorchTitan is not
modified.

The configuration test suite checks the exact ordered placement tuple. The
combined NPU regression then captures 733 tensors per topology (628 parameter
state plus 105 router tensors) with the same eager step-0 fixture.

| Candidate | Metric | Before fix | After fix |
| --- | --- | ---: | ---: |
| FSDP2-TP2 | Parameter-gradient min cosine | 0.188443 | 0.998216 |
| FSDP2-TP2 | Compatibility gradient failures | 7 | 0 |
| FSDP2-TP2 | Updated-parameter min cosine | 0.980722 | 0.999618 |
| FSDP2-TP2 | Compatibility updated-parameter failures | 4 | 0 |
| TP4 | Parameter-gradient min cosine | 0.119879 | 0.998410 |
| TP4 | Compatibility gradient failures | 14 | 0 |
| TP4 | Updated-parameter min cosine | 0.976565 | 0.999645 |
| TP4 | Compatibility updated-parameter failures | 3 | 0 |

At the router boundary, the real post-fix gradient and the detached diagnostic
TP-sum have identical cosine. Gate-logit gradient minimum cosine is 0.997565
for TP2 and 0.997467 for TP4. Router gate parameter-gradient minimum cosine is
0.999654 for TP2 and 0.999330 for TP4. This confirms that DTensor now performs
the missing reduction before the replicated gate parameter gradient reaches
the optimizer.

The isolated optimizer-delta diagnostic still has 45 compatibility failures
for TP2 and 48 for TP4. These are mostly attention parameters (40 in each
topology), with smaller attention-norm and FFN-norm sets and one layer-4 router
gate. They are first-step AdamW sign amplification of residual, already-aligned
gradients rather than missing TP reduction: all 127 parameter gradients and all
167 updated parameters satisfy the compatibility rule after the fix. The
router layer-4 update remains below the combined delta threshold, but its
updated parameter cosine is at least 0.999684 and no updated parameter fails.
The patched single-card run is exactly identical to the pre-fix reference for
all 628 parameter-state tensors (minimum cosine 1.0 and maximum absolute error
0), confirming that the normalized degree-one TP placement is an identity.

### AdamW first-step update-chain confirmation

A focused single-card versus TP4 run captures 662 tensors through the public
`PrecisionDebugger.save()` interface and compares them with native
`msprobe compare -m auto`. All 662 rows match and the MindStudio `Result`
column is `pass`. The diagnostic freezes the optimizer hyperparameters before
the scheduler runs and records pre-clip gradient, post-clip gradient, first and
second moments, reconstructed FP32 AdamW deltas, and the actual parameter
delta. The global norm differs by only 9.30e-6 (single 1.4018544, TP4
1.4018451), so clipping applies effectively the same coefficient.

| Parameter | Pre/post-clip cosine | `exp_avg` / `exp_avg_sq` cosine | Adaptive/intended delta cosine | Gradient sign differences | Actual minus intended max abs |
| --- | ---: | ---: | ---: | ---: | ---: |
| layer 3 attention `kv_norm.weight` (control) | 0.999903 / 0.999903 | 0.999903 / 0.999930 | 1.000000 / 1.000000 | 0 / 64 | 6.92e-8 |
| layer 6 attention `kv_norm.weight` | 0.999820 / 0.999820 | 0.999820 / 0.999815 | 0.873088 / 0.872576 | 5 / 64 | 6.95e-8 |
| layer 4 router `gate.weight` | 0.999330 / 0.999330 | 0.999330 / 0.999566 | 0.979337 / 0.979337 | 24 / 2048 | 1.75e-9 |

The control parameter has no sign differences and its update remains aligned.
For layer 6, four elements receive an opposite near-full first-step adaptive
update; the maximum single-versus-TP4 difference is 0.001581, close to twice
the 0.0008 learning rate. The router has 16 such elements and a maximum
difference of 0.001589. This is the expected first-step AdamW behavior: with
zero-initialized moments, `m / sqrt(v)` is approximately the gradient sign, so
a small residual around zero can change the update by about `2 * lr` even when
the gradient cosine is above 0.999. The actual FP32 parameter delta tracks the
reconstructed intended delta within numerical noise, excluding clipping,
optimizer-state corruption, and parameter write-back as independent TP bugs.

The final combined regression captures 765 tensors per topology. All 16 block
forward and 16 block backward boundaries continue to satisfy the compatibility
rule with zero unsupported metrics. FSDP2-TP2 block-backward minimum cosine
improves from 0.999943 to 0.999989 (maximum absolute error 1.66893e-6), while
TP4 improves from 0.999901 to 0.999985 (maximum absolute error 2.14577e-6).
Six weighted-routing-map diagnostics retain maximum absolute error above 1
because the same small set of near-tie tokens selects a different expert; this
is a forward discrete-routing diagnostic and is unchanged by the backward-only
placement fix.

### PP2-TP2-EP2 full-composition regression

The final four-card composition enables PP2, TP2, and EP2 together. The Turbo
TP-Partial correction is intentionally inactive because EP owns the routed
gradient layout. Pipeline rank 0 saves 365 stage-0 tensors and rank 2 saves 400
stage-1 tensors; together they cover the same 765 logical diagnostics as the
single-card reference. Both stages are compared independently with native
`msprobe compare -m auto`.

All 737 semantically common tensors pass the MindStudio `Result` criterion:

| Metric | Tensors | Minimum cosine | Maximum absolute error | Compatibility failures |
| --- | ---: | ---: | ---: | ---: |
| Block forward | 16 | 0.999996 | 0.0444336 | 0 |
| Block backward | 16 | 0.999989 | 2.38419e-6 | 0 |
| Parameter gradient | 127 | 0.998442 | 1.57662e-4 | 0 |
| Optimizer delta | 167 | 0.910258 | 0.00159914 | 44 |
| Updated parameter | 167 | 0.999645 | 0.00159914 | 0 |
| Router state | 77 | 0.995199 | 1.25001 | 0 |

The raw stage reports contain 744 `pass` rows and 21 shape errors. Every error
belongs to a diagnostic `_tp_sum` tensor. All 28 `_tp_sum` tensors are excluded,
including seven whose shapes happen to match, because an all-reduce across the
TP mesh is not the correct logical reconstruction when EP shares that mesh.
This is the same predefined exclusion used for FSDP2-TP2-EP2, not a numerical
failure discovered after comparison. The remaining 44 low-cosine isolated
optimizer deltas have the already-confirmed first-step AdamW sign-amplification
signature; no parameter gradient or updated parameter fails compatibility.

The single-card loss/gradient norm are 8.14224625/1.40185440; PP2-TP2-EP2 gives
8.14226818/1.40230775. Router gate parameter-gradient minimum cosine is
0.999666, and router gate updated-parameter minimum cosine is 0.999778. Six
weighted routing maps retain maximum absolute error above 1 due to near-tie
expert choices, consistent with the other TP combinations.

### Eight-card TP scale-out regression

The same topology-invariant fixture is extended from TP4 to TP8 without
changing the eager, one-step MindStudio contract. The debug model has eight
attention heads and eight experts, so TP8 exercises the maximum meaningful TP
degree for this model. Both the single-card reference and TP8 candidate capture
all 765 tensors, and native `msprobe compare -m auto` reports 765 `pass`, zero
errors, and no missing tensors.

| Metric | Tensors | Minimum cosine | Maximum absolute error | Compatibility failures |
| --- | ---: | ---: | ---: | ---: |
| Block forward | 16 | 0.999995 | 0.078125 | 0 |
| Block backward | 16 | 0.999985 | 1.90735e-6 | 0 |
| Parameter gradient | 127 | 0.998309 | 1.13498e-4 | 0 |
| Optimizer delta | 167 | 0.887072 | 0.00159879 | 46 |
| Updated parameter | 167 | 0.999590 | 0.00159879 | 0 |
| Router state | 105 | 0.987392 | 1.24979 | 0 |

The single-card loss/gradient norm are 8.14224625/1.40185440; TP8 gives
8.14215469/1.40182996. Router gate parameter-gradient minimum cosine is
0.999294 and its updated-parameter minimum cosine is 0.999759. Unlike EP
combinations, all 28 TP-summed router gradients are semantically valid here
and pass. Router top-k score-gradient cosine reaches 0.987392, but the maximum
absolute error is only 4.68990e-6. Five weighted routing maps have maximum
absolute error above 1 because of near-tie expert choices. The 46 isolated
optimizer-delta compatibility failures retain the previously confirmed
first-step AdamW sign-amplification signature; all parameter gradients and
updated parameters pass compatibility.
