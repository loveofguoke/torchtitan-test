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


def dynamic_source(src):
    lines = src.splitlines(keepends=True)
    kernel, = (n for n in ast.parse(src).body
               if isinstance(n, ast.FunctionDef) and n.name == "_gather_attn_fwd")
    loop, = (n for n in ast.walk(kernel) if isinstance(n, ast.For)
             and ast.unparse(n.iter) == "tl.static_range(0, TOPK)")
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

    q, local, pool, ids, cu, cuk, scale, *_ = torch.load(run / "inputs.pt")
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


def main():
    if len(sys.argv) == 3:
        phase, run = sys.argv[1], Path(sys.argv[2])
        if phase in ("csa_gather", "csa_flex"):
            from tests.glm5_2_nvidia.dsv4_csa_control import run_control
            return run_control(run, phase)
        if phase in ("gradient", "step"):
            return verify(run, phase)
        return prepare(run) if phase == "prepare" else measure(run, phase)
    check = sys.argv[1:] == ["--verify-training"]
    csa = sys.argv[1:] == ["--dsv4-csa-control"]
    if sys.argv[1:] and not (check or csa):
        raise ValueError("expected --verify-training or no arguments")
    root = Path(__file__).resolve().parents[2]
    outputs = root / "outputs"
    outputs.mkdir(exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix="pr5578-gather-loop-", dir=outputs))
    log = run / "regression.log"
    pkg = Path(importlib.util.find_spec("attn_gym").origin).parent
    module = "tests.glm5_2_nvidia.gather_loop_ab"
    codes = {}
    try:
        phases = ("prepare", "gradient", "step") if check else ("prepare", "static", "dynamic")
        for phase in (("csa_gather", "csa_flex") if csa else phases):
            env = os.environ.copy()
            if phase != "prepare":
                dest = run / phase / "attn_gym"
                shutil.copytree(pkg, dest, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
                path = dest / "sparse/gather_attn/impl/triton/forward.py"
                src = path.read_text()
                if phase not in ("static", "csa_flex"):
                    path.write_text(dynamic_source(src))
                print(phase, "forward.py SHA256", hashlib.sha256(path.read_bytes()).hexdigest(), flush=True)
                env["PYTHONPATH"] = os.pathsep.join([str(dest.parent), env.get("PYTHONPATH", "")])
                env["TRITON_CACHE_DIR"] = str(dest.parent / "cache")
            limit = 300 if csa or phase in ("gradient", "step") else 180
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
            if (phase == "prepare" or check) and codes[phase]:
                break
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
