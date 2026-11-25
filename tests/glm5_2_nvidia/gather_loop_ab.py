"""Linux/H20 diagnostic. One input snapshot, two isolated attn_gym imports."""

import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import time
import traceback


def dynamic_source(src):
    # Already-patched H20 fixture: accept the exact previously measured kernel.
    if hashlib.sha256(src.encode()).hexdigest() == "4039586e5d65fcb9c5d35fab72039e9dcfefe781cca5a30603a22963bcd3b2b2":
        return src
    lines = src.splitlines(keepends=True)
    kernel, = (n for n in ast.parse(src).body
               if isinstance(n, ast.FunctionDef) and n.name == "_gather_attn_fwd")
    loops = [n for n in ast.walk(kernel) if isinstance(n, ast.For)
             and ast.unparse(n.iter) == "tl.static_range(0, TOPK)"]
    if len(loops) != 1:
        raise ValueError(f"unsupported gather source {hashlib.sha256(src.encode()).hexdigest()}: found {len(loops)} matching static loops")
    loop, = loops
    lines[loop.lineno - 1] = lines[loop.lineno - 1].replace("tl.static_range", "tl.range")
    return "".join(lines)


def prepare(run):
    import torch
    from unittest.mock import patch
    from torchtitan.models.glm5 import dsa
    from tests.glm5_2_nvidia import attention_ab
    from tests.glm5_2_nvidia.gather_probe import ProbeComplete

    def save(q, local, pool, ids, **kw):
        assert ids.shape[-2:] == (8192, 512) and kw["sliding_window_size"] == 0
        assert kw["impl"] == "fused" and not kw["return_aux"].lse
        torch.save((q.detach(), local.detach(), pool.detach(), ids,
                    kw["cu_seqlens"], kw["cu_seqlens_k"], kw["scale"],
                    q.requires_grad, pool.requires_grad), run / "inputs.pt")
        raise ProbeComplete

    try:
        with patch.object(dsa, "gather_attn", save):
            attention_ab.main()
    except ProbeComplete:
        return
    raise RuntimeError("no production gather captured")


def measure(run, arm):
    import torch
    import attn_gym
    from torchtitan.models.glm5.dsa import gather_attn, AuxRequest

    assert Path(attn_gym.__file__).resolve().parent == run / arm / "attn_gym"
    q, local, pool, ids, cu, cuk, scale, q_grad, pool_grad = torch.load(run / "inputs.pt")
    q.requires_grad_(q_grad)
    pool.requires_grad_(pool_grad)
    assert pool.shape[1] == 1 and cu.tolist() == cuk.tolist() == [0, 8192]
    print(arm, torch.cuda.get_device_name(), "q/pool/ids", q.shape, pool.shape, ids.shape, flush=True)
    times = []
    for i in range(11):
        torch.cuda.synchronize()
        start = time.perf_counter()
        out, _ = gather_attn(q, local, pool, ids, cu_seqlens=cu, cu_seqlens_k=cuk,
                             sliding_window_size=0, scale=scale, impl="fused",
                             kernel_options={"backend": "triton"}, return_aux=AuxRequest(lse=False))
        torch.cuda.synchronize()
        ms = (time.perf_counter() - start) * 1000
        times.append(ms)
        print(f"{arm} call={i} ms={ms:.3f}", flush=True)
        if i == 0:
            # One packed document, no window or sink. Check early/late rows on all heads.
            with torch.no_grad():
                for row in (0, 7, 511, 4096, 8191):
                    selected = ids[0, row].long()
                    kv = pool[0, 0, selected[selected >= 0]].float()
                    query = q[0, :, row].float()
                    ref = (((query[:, None] * kv).sum(-1) * scale).softmax(-1) @ kv
                           if kv.numel() else torch.zeros_like(query))
                    torch.testing.assert_close(out[0, :, row].float(), ref, atol=.02, rtol=.02)
        del out
    print("RESULT", arm, json.dumps(dict(cold_ms=times[0], warm_median_ms=statistics.median(times[1:]),
                                         oracle_rows=5)), flush=True)


def verify(run, phase):
    import attn_gym
    import torch
    from torchtitan.models.glm5.dsa import gather_attn, AuxRequest

    assert Path(attn_gym.__file__).resolve().parent == run / phase / "attn_gym"
    torch.manual_seed(61)
    if phase == "step":
        from tests.glm5_2_nvidia import attention_ab as ab
        cfg = ab.glm5_scaled_model_config(seq_len=ab.T)
        tokens = torch.randint(cfg.vocab_size, (ab.T + 1,)).cuda()
        return ab.measure(cfg, True, {}, tokens, torch.arange(ab.T, device="cuda"), one_step=True)

    call_ab = phase.startswith("call_")
    snapshot = Path(os.environ["GLM_GATHER_INPUTS"]) if call_ab else run / "inputs.pt"
    q, local, pool, ids, cu, cuk, scale, *_ = torch.load(snapshot)
    q.requires_grad_()
    pool.requires_grad_()
    rows = [0, 7, 511, 4096, 8191]
    out, _ = gather_attn(q, local, pool, ids, cu_seqlens=cu, cu_seqlens_k=cuk,
                         sliding_window_size=0, scale=scale, impl="fused",
                         kernel_options={"backend": "triton"}, return_aux=AuxRequest(lse=False))
    v = torch.randn_like(out[0, :, rows, :])
    dq, dp = torch.autograd.grad((out[0, :, rows, :].float() * v.float()).sum(), (q, pool))
    qr, pr = (t.detach().float().requires_grad_() for t in (q, pool))
    terms = []
    for j, row in enumerate(rows):
        idx = ids[0, row].long()
        kv = pr[0, 0, idx[idx >= 0]]
        if len(kv):
            logits = (qr[0, :, row, None, :] * kv).sum(-1) * scale
            terms.append(((logits.softmax(-1) @ kv) * v[:, j].float()).sum())
    refs = torch.autograd.grad(sum(terms), (qr, pr))
    errors = {}
    for name, grad, ref in zip(("dq", "dpool"), (dq, dp), refs):
        assert ref.norm() > 0
        err = (grad.float() - ref).norm() / ref.norm()
        errors[name] = err.item()
        print(f"{name} relative_l2={err.item():.6g}", flush=True)
        assert err < .03, (name, errors)
        torch.testing.assert_close(grad.float(), ref, atol=.03, rtol=.05)
    print("RESULT gradient five query rows, all heads; Q/shared pool relative_l2", errors, flush=True)


def glm_core(run, phase):
    import torch
    import triton
    import attn_gym
    from attn_gym.sparse.gather_attn import gather_attn
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention

    assert Path(attn_gym.__file__).resolve().parent == run / phase / "attn_gym"
    q, local, pool, ids, cu, cuk, scale, *_ = torch.load(run / "inputs.pt")
    assert tuple(q.shape) == (1, 8, 8192, 256) and ids.shape[-1] == 512
    if "_heads" in phase:
        heads = int(phase.split("_heads")[1])
        # Tile the same captured queries; allocate before either timed region.
        q = q.repeat(1, heads // 8, 1, 1).contiguous()
    assert cu.tolist() == cuk.tolist() == [0, 8192]
    if phase == "glm_recent":
        valid = ids >= 0
        rows = torch.arange(8192, device=ids.device)[None, :, None]
        count = valid.sum(-1, keepdim=True)
        assert (count <= rows + 1).all()
        # Keep padding in exactly the same slots; valid ids ascend to the query.
        recent = rows - count + valid.cumsum(-1)
        ids = torch.where(valid, recent.to(ids.dtype), ids)
        assert torch.equal(ids >= 0, valid)
    q.requires_grad_()
    pool.requires_grad_()
    torch.manual_seed(61)
    grad = torch.randn_like(q)
    dense = phase.split("_heads")[0] == "glm_dense"
    if call_ab:
        from torchtitan.models.glm5.dsa import AuxRequest, gather_attn as production_gather
        kw = dict(scale=scale, sliding_window_size=0, impl="fused", return_aux=AuxRequest(lse=False))
        if phase != "call_default_offsets":
            kw["kernel_options"] = {"backend": "triton"}
        if phase != "call_triton_no_offsets":
            kw.update(cu_seqlens=cu, cu_seqlens_k=cuk)
        def call():
            return production_gather(q, local, pool, ids, **kw)[0]
    elif dense:
        mask = create_block_mask(lambda b, h, i, j: i >= j, 1, 1, 8192, 8192, device="cuda")
        flex = torch.compile(flex_attention)
        def call():
            return flex(q, pool, pool, scale=scale, block_mask=mask, enable_gqa=True)
    else:
        def call():
            return gather_attn(q, local, pool, ids, scale=scale, sliding_window_size=0,
                               cu_seqlens=cu, cu_seqlens_k=cuk, impl="fused",
                               kernel_options={"backend": "triton"})
    with torch.no_grad():
        out = call()
        for r in (0, 127, 2047, 8191):
            ix = ids[0, r].long()
            kv = (pool[0, 0, :r+1] if dense else pool[0, 0, ix[ix >= 0]]).float()
            qr = q[0, :, r].float()
            ref = ((qr @ kv.T * scale).softmax(-1) @ kv if len(kv) else torch.zeros_like(qr))
            torch.testing.assert_close(out[0, :, r].float(), ref, atol=.025, rtol=.04)
        del out
        fwd = triton.testing.do_bench(call, warmup=100, rep=500, return_mode="median")
    out = call()
    def backward():
        return torch.autograd.grad(out, (q, pool), grad, retain_graph=True)
    assert all(g.isfinite().all() for g in backward())
    if call_ab:
        values = tuple(x.detach().cpu() for x in (out, *backward()))
        reference = run / "default-output-gradients.pt"
        if phase == "call_default_offsets":
            torch.save(values, reference)
        else:
            for value, expected in zip(values, torch.load(reference), strict=True):
                torch.testing.assert_close(value, expected, atol=.04, rtol=.05)
    bwd = triton.testing.do_bench(backward, warmup=100, rep=500, return_mode="median")
    print("RESULT", phase, json.dumps(dict(fwd_ms=fwd, bwd_ms=bwd, oracle_rows=4)), flush=True)


def main():
    if len(sys.argv) == 3:
        phase, run = sys.argv[1], Path(sys.argv[2])
        if phase.startswith("call_") or phase.split("_heads")[0] in ("glm_gather", "glm_dense", "glm_recent"):
            return glm_core(run, phase)
        if phase in ("csa_gather", "csa_flex", "gather_bench", "dense_bench"):
            from tests.glm5_2_nvidia.dsv4_csa_control import run_control
            return run_control(run, phase)
        if phase in ("gradient", "step"):
            return verify(run, phase)
        return prepare(run) if phase == "prepare" else measure(run, phase)
    check = sys.argv[1:] == ["--verify-training"]
    csa = sys.argv[1:] == ["--dsv4-csa-control"]
    direct = sys.argv[1:] == ["--dsv4-gather-vs-dense"]
    glm = sys.argv[1:] == ["--glm-gather-vs-dense"]
    locality = sys.argv[1:] == ["--glm-index-locality"]
    sweep = sys.argv[1:] == ["--glm-head-sweep"]
    calls = sys.argv[1:] == ["--glm-call-ab"]
    if sys.argv[1:] and not (check or csa or direct or glm or locality or sweep or calls):
        raise ValueError("expected --verify-training or no arguments")
    root = Path(__file__).resolve().parents[2]
    outputs = root / "outputs"
    outputs.mkdir(exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix="pr5578-gather-loop-", dir=outputs))
    log = run / "regression.log"
    pkg = Path(importlib.util.find_spec("attn_gym").origin).parent
    module = "tests.glm5_2_nvidia.gather_loop_ab"
    codes = {}
    phase = "setup"
    try:
        with log.open("a") as f:
            print("attn_gym source:", pkg, file=f)
        phases = ("prepare", "gradient", "step") if check else ("prepare", "static", "dynamic")
        if direct:
            phases = ("gather_bench", "dense_bench")
        if glm:
            phases = ("prepare", "glm_gather", "glm_dense")
        if locality:
            phases = ("prepare", "glm_gather", "glm_recent")
        if sweep:
            phases = ("prepare", *(f"glm_{backend}_heads{h}" for h in (8, 16, 32, 64)
                                    for backend in ("gather", "dense")))
        if calls:
            phases = ("call_default_offsets", "call_triton_offsets", "call_triton_no_offsets")
        for phase in (("csa_gather", "csa_flex") if csa else phases):
            env = os.environ.copy()
            if phase != "prepare":
                dest = run / phase / "attn_gym"
                shutil.copytree(pkg, dest, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
                path = dest / "sparse/gather_attn/impl/triton/forward.py"
                src = path.read_text()
                if phase.split("_heads")[0] not in ("static", "csa_flex", "dense_bench", "glm_dense"):
                    path.write_text(dynamic_source(src))
                print(phase, "forward.py SHA256", hashlib.sha256(path.read_bytes()).hexdigest(), flush=True)
                env["PYTHONPATH"] = os.pathsep.join([str(dest.parent), env.get("PYTHONPATH", "")])
                env["TRITON_CACHE_DIR"] = str(dest.parent / "cache")
            limit = 300 if calls or direct or csa or locality or sweep or phase in ("gradient", "step", "glm_gather", "glm_dense") else 180
            print(f"{phase}: start (limit {limit}s per process)", flush=True)
            with log.open("a") as f:
                p = subprocess.Popen([sys.executable, "-u", "-m", module, phase, str(run)],
                                     cwd=root, env=env, stdout=f, stderr=subprocess.STDOUT,
                                     start_new_session=True)
                try:
                    for _ in range(limit // 15):
                        try:
                            codes[phase] = p.wait(timeout=15)
                            break
                        except subprocess.TimeoutExpired:
                            print(f"{phase} elapsed={15*(_+1)}s", flush=True)
                    else:
                        codes[phase] = 124
                finally:
                    if p.poll() is None:
                        os.killpg(p.pid, signal.SIGKILL)
                        p.wait(timeout=5)
            if (phase in ("prepare", "call_default_offsets") or check) and codes[phase]:
                break
    except Exception:
        codes[phase] = 1
        with log.open("a") as f:
            traceback.print_exc(file=f)
    finally:
        lines = log.read_text(errors="replace").splitlines() if log.exists() else []
        summary = ["========== SUMMARY ==========", str(codes) + " (124=timeout)",
                   *[s for s in lines if s.startswith("RESULT ") or "Error:" in s],
                   "========== END SUMMARY ==========", f"{root.name}/{log.relative_to(root)}"]
        with log.open("a") as f:
            for line in summary:
                print(line, flush=True)
                print(line, file=f)
    return int(any(codes.values()))


if __name__ == "__main__":
    sys.exit(main())
