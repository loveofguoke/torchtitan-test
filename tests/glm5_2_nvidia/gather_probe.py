"""Time the first production gather call; do not time the surrounding decoder."""

import faulthandler
import json
import statistics
import time
from unittest.mock import patch

import torch
from torchtitan.models.glm5 import dsa

from tests.glm5_2_nvidia import attention_ab


class ProbeComplete(Exception):
    pass


def main():
    real_gather = dsa.gather_attn
    result = {}

    def probe(*args, **kwargs):
        print("layer=0", [(tuple(t.shape), str(t.dtype), t.stride()) for t in args],
              "impl=", kwargs["impl"], "window=", kwargs["sliding_window_size"], flush=True)
        assert kwargs["impl"] == "fused" and args[3].shape[-2:] == (8192, 512)
        torch.cuda.synchronize()
        durations = []
        faulthandler.dump_traceback_later(60, repeat=True)
        try:
            for i in range(11):
                if i == 0:
                    print("phase=first fused gather (cold Triton cache)", flush=True)
                start = time.perf_counter()
                out, aux = real_gather(*args, **kwargs)
                torch.cuda.synchronize()
                ms = (time.perf_counter() - start) * 1000
                durations.append(ms)
                print(f"call={i} synchronized_wall_ms={ms:.3f}", flush=True)
                if i == 0:
                    if not torch.isfinite(out).all().item():
                        raise ValueError("nonfinite first gather output")
                del out, aux
        finally:
            faulthandler.cancel_dump_traceback_later()
        result.update(first_call_ms=durations[0], warm_median_ms=statistics.median(durations[1:]),
                      warm_samples_ms=durations[1:])
        raise ProbeComplete

    try:
        with patch.object(dsa, "gather_attn", probe):
            attention_ab.main()
    except ProbeComplete:
        print("GATHER_8K " + json.dumps(result), flush=True)
    else:
        raise RuntimeError("did not intercept production gather")
