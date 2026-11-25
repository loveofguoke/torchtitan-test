"""DSV4-Flash CSA core control, not the GB300 six-layer training benchmark."""

import json
import hashlib
import statistics
import time
from pathlib import Path


def run_control(run, phase):
    import attn_gym
    import torch
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention
    from attn_gym.sparse.gather_attn import gather_attn

    assert Path(attn_gym.__file__).resolve().parent == run / phase / "attn_gym"
    # #4858 sequence length; DSV4 Flash CSA geometry (one unpadded document).
    T, H, D, W, K, R = 4096, 64, 512, 128, 512, 4
    S = T // R
    torch.manual_seed(61)
    def rand(*shape):
        return torch.randn(shape, device="cuda", dtype=torch.bfloat16)

    q = rand(T, H, D).transpose(0, 1)[None].requires_grad_()
    local = rand(1, 1, T, D).requires_grad_()
    pool = rand(1, 1, S, D).requires_grad_()
    sink = torch.randn(H, device="cuda")
    grad = rand(*q.shape)
    pos = torch.arange(T, device="cuda")
    available = torch.arange(S, device="cuda")[None, :] < ((pos + 1) // R)[:, None]
    # Synthetic, fixed indexer output; select only fully compressed causal blocks.
    scores = torch.rand((T, S), device="cuda").masked_fill(~available, -torch.inf)
    vals, ids = scores.topk(K)
    ids = ids.masked_fill(~vals.isfinite(), -1).int()[None]
    digest = hashlib.sha256()
    for tensor in (q, local, pool, sink, ids, grad):
        digest.update(tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
    fingerprint = digest.hexdigest()
    print("input_sha256", fingerprint, flush=True)
    print(f"{phase} {torch.cuda.get_device_name()} torch={torch.__version__} "
          f"T={T} H={H} D={D} topk={K} window={W} ratio={R}", flush=True)
    direct = phase in ("gather_bench", "dense_bench")
    dense = phase == "dense_bench"
    if phase in ("csa_gather", "gather_bench"):
        def call():
            return gather_attn(q, local, pool, ids, attention_sink=sink,
                               sliding_window_size=W, impl="fused",
                               kernel_options={"backend": "triton"})
    elif dense:
        # Plain causal attention: no indexer, compressed pool, sink or window.
        bm = create_block_mask(lambda b, h, i, j: i >= j, 1, 1, T, T, device="cuda")
        flex = torch.compile(flex_attention)
        def call():
            return flex(q, local, local, block_mask=bm, enable_gqa=True)
    else:
        selected = torch.zeros((T, S + 1), device="cuda", dtype=torch.bool)
        selected.scatter_(1, ids[0].long().masked_fill(ids[0] < 0, S), True)
        def mask(b, h, i, j):
            return (((j <= i) & (j > i - W)) |
                    ((j >= T) & (j < T + S) & selected[i % T, (j - T).clamp(0, S - 1)]) |
                    (j == T + S))
        def score(s, b, h, i, j):
            return torch.where(j == T + S, sink[h], s)
        bm = create_block_mask(mask, 1, 1, T, T + S + 1, device="cuda")
        flex = torch.compile(flex_attention)
        def call():
            kv = torch.cat((local, pool, local.new_zeros(1, 1, 1, D)), dim=2)
            return flex(q, kv, kv, score_mod=score, block_mask=bm, enable_gqa=True)

    if direct:
        import triton

        # Match attention-gym's gather_attn_benchmark: no-grad forward and
        # backward alone on a retained graph, using median do_bench timings.
        with torch.no_grad():
            out = call()
            for r in (0, 127, 2047, T - 1):
                kv = local[0, 0, :r+1].float()
                if not dense:
                    idx = ids[0, r].long()
                    kv = torch.cat((kv[-W:], pool[0, 0, idx[idx >= 0]].float()))
                logits = q[0, :, r].float() @ kv.T / D**.5
                if not dense:
                    logits = torch.cat((logits, sink[:, None]), -1)
                ref = logits.softmax(-1)[:, :kv.size(0)] @ kv
                torch.testing.assert_close(out[0, :, r].float(), ref, atol=.025, rtol=.04)
            del out
            fwd = triton.testing.do_bench(call, warmup=100, rep=500, return_mode="median")
        out = call()
        inputs = (q, local) if dense else (q, local, pool)
        def backward():
            return torch.autograd.grad(out, inputs, grad, retain_graph=True)
        assert all(g.isfinite().all() for g in backward())
        bwd = triton.testing.do_bench(backward, warmup=100, rep=500, return_mode="median")
        print("RESULT", phase, json.dumps(dict(input_sha256=fingerprint, fwd_ms=fwd,
              bwd_ms=bwd, oracle_rows=4)), flush=True)
        return
    samples = []
    for i in range(7):
        torch.cuda.synchronize()
        start = time.perf_counter()
        out = call()
        torch.cuda.synchronize()
        mid = time.perf_counter()
        grads = torch.autograd.grad(out, (q, local, pool), grad)
        torch.cuda.synchronize()
        end = time.perf_counter()
        if i == 0:
            # Independent fp32 oracle: same sink denominator, window and chosen keys.
            with torch.no_grad():
                for r in (0, 127, 2047, T - 1):
                    idx = ids[0, r].long()
                    kv = torch.cat((local[0, 0, max(0, r-W+1):r+1],
                                    pool[0, 0, idx[idx >= 0]])).float()
                    logits = (q[0, :, r].float()[:, None] * kv).sum(-1) / D**.5
                    probs = torch.cat((logits, sink[:, None]), -1).softmax(-1)
                    torch.testing.assert_close(out[0, :, r].float(), probs[:, :-1] @ kv,
                                               atol=.025, rtol=.04)
                assert all(g.isfinite().all() for g in grads)
                sample = [g.detach().reshape(-1)[::257].float().cpu() for g in grads]
                saved = run / "csa_gather_gradient_sample.pt"
                if phase == "csa_gather":
                    torch.save((fingerprint, sample), saved)
                elif saved.exists():
                    other_hash, other = torch.load(saved)
                    assert other_hash == fingerprint
                    for a, b in zip(sample, other, strict=True):
                        torch.testing.assert_close(a, b, atol=.04, rtol=.05)
                    print("RESULT matched input SHA and sampled dQ/dLocalKV/dCompressedKV", flush=True)
        ms = ((mid-start)*1000, (end-mid)*1000)
        print(f"{phase} iter={i} fwd_ms={ms[0]:.3f} bwd_ms={ms[1]:.3f}", flush=True)
        if i >= 2:
            samples.append(ms)
        del out, grads
    print("RESULT", phase, json.dumps(dict(fwd_ms=statistics.median(x for x,y in samples),
          bwd_ms=statistics.median(y for x,y in samples),
          fwd_bwd_ms=statistics.median(x+y for x,y in samples), oracle_rows=4)), flush=True)
