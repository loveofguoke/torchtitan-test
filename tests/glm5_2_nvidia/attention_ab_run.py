"""Linux foreground supervisor: keep terminal signals out of CUDA and its compiler."""

import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time


def supervise(*, diagnose=False, gather=False):
    root = Path(__file__).resolve().parents[2]
    outputs = root / "outputs"
    outputs.mkdir(exist_ok=True)
    path = Path(tempfile.mkdtemp(prefix="pr5578-h20-attention-8k-", dir=outputs)) / "regression.log"
    display = f"{root.name}/{path.relative_to(root)}"
    interrupted = False

    def stop(*_):
        nonlocal interrupted
        interrupted = True

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, stop)
    rc = 1
    result = ""
    marker = "GATHER_8K " if gather else "FIRST_FORWARD " if diagnose else "ATTENTION_AB "
    with path.open("a", buffering=1) as log, path.open(errors="replace") as reader:
        def note(line):
            print(line, file=log, flush=True)

        def drain():
            nonlocal result
            for line in reader:
                print(line, end="", flush=True)
                if line.startswith(marker):
                    result = line.strip()

        started = time.monotonic()
        note(f"supervisor_pid={os.getpid()} | Ctrl+C to stop | {display}")
        drain()
        proc = None
        try:
            worker = (f"from tests.glm5_2_nvidia.attention_ab import main; main(diagnose={diagnose!r})")
            env = os.environ.copy()
            if gather:
                worker = "from tests.glm5_2_nvidia.gather_probe import main; main()"
                env["TRITON_CACHE_DIR"] = str(path.parent / "triton-cache")
                note(f"private Triton cache: {env['TRITON_CACHE_DIR']}")
            proc = subprocess.Popen(
                [sys.executable, "-u", "-c", worker], env=env,
                cwd=root, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
            )
            note(f"worker_pid={proc.pid} | phase=import torch and initialize CUDA")
            beat = started
            while proc.poll() is None and not interrupted:
                drain()
                now = time.monotonic()
                if now - beat >= 15:
                    note(f"heartbeat | worker alive | elapsed={now-started:.0f}s")
                    beat = now
                time.sleep(0.2)
            rc = 130 if interrupted else proc.returncode
        finally:
            if proc is not None and (interrupted or proc.poll() is None):
                note("STOP requested: sending TERM to this run's process group")
                drain()
                for sig in (signal.SIGTERM, signal.SIGKILL):
                    try:
                        os.killpg(proc.pid, sig)
                    except ProcessLookupError:
                        break
                    if sig == signal.SIGTERM:
                        time.sleep(2)
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    note("Worker still pending kernel exit; check PID before restarting.")
            drain()
            if rc == 0 and not result:
                rc = 1
            note("========== SUMMARY ==========")
            label = "8K first-forward diagnostic" if diagnose else "8K attention A/B"
            if gather:
                label = "8K fused gather forward cold/warm"
            note(f"{label} | exit_code={rc}")
            if rc == 0:
                note(result)
            else:
                note("INTERRUPTED" if interrupted else "FAILED: see worker output above")
            note("========== END SUMMARY ==========")
            note(display)
            drain()
    return rc
