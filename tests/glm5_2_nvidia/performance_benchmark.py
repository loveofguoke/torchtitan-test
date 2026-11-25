#!/usr/bin/env python3
"""Profile GLM training with the NVIDIA performance toolchain."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.glm5_2_nvidia.workflow import run_cli  # noqa: E402


if __name__ == "__main__":
    if sys.argv[1:] in (["--compare-attention-8k"], ["--diagnose-attention-8k"],
                        ["--measure-gather-8k"], ["--compare-attention-8k-dynamic"],
                        ["--compare-blockmask-8k"]):
        from tests.glm5_2_nvidia.attention_ab_run import supervise

        raise SystemExit(supervise(diagnose=sys.argv[1] == "--diagnose-attention-8k",
                                   gather=sys.argv[1] == "--measure-gather-8k",
                                   dynamic=sys.argv[1] == "--compare-attention-8k-dynamic",
                                   blockmask=sys.argv[1] == "--compare-blockmask-8k"))
    raise SystemExit(run_cli())
