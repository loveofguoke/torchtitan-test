#!/usr/bin/env python3
"""Create or update an official torch_npu dynamic profiler configuration."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.glm5_2_performance.config import profiler_presets  # noqa: E402
from tests.glm5_2_performance.dynamic_profile import (  # noqa: E402
    build_dynamic_profile_config,
    write_dynamic_profile_config,
)


def main() -> None:
    presets = profiler_presets()
    dynamic_presets = tuple(
        name for name in presets if name not in {"flamegraph", "memory"}
    )
    parser = argparse.ArgumentParser(
        description="Control non-intrusive torch_npu dynamic profiling"
    )
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--init", action="store_true")
    actions.add_argument("--enable", action="store_true")
    actions.add_argument("--disable", action="store_true")
    actions.add_argument("--show", action="store_true")
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path("performance_dynamic/glm5_profiler"),
    )
    parser.add_argument("--profile-dir", type=Path)
    parser.add_argument("--preset", choices=dynamic_presets, default="overview")
    parser.add_argument(
        "--parse-mode",
        choices=("async", "offline"),
        default="async",
        help="rank-filtered dynamic profiling cannot use synchronous parsing",
    )
    parser.add_argument("--start-step", type=int, default=10)
    parser.add_argument("--active-steps", type=int, default=3)
    parser.add_argument("--warmup-steps", type=int, default=1)
    args = parser.parse_args()

    config_path = args.config_dir / "profiler_config.json"
    if args.show:
        if not config_path.is_file():
            raise FileNotFoundError(config_path)
        print(config_path.read_text(encoding="utf-8"), end="")
        return
    if args.disable:
        if not config_path.is_file():
            raise FileNotFoundError(config_path)
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["is_valid"] = False
    else:
        existing = (
            json.loads(config_path.read_text(encoding="utf-8"))
            if config_path.is_file()
            else {}
        )
        profile_directory = (
            args.profile_dir
            or existing.get("prof_dir")
            or args.config_dir / "profiles"
        )
        profile_directory = Path(profile_directory)
        preset = replace(presets[args.preset], parse_mode=args.parse_mode)
        config = build_dynamic_profile_config(
            preset=preset,
            profile_directory=profile_directory,
            start_step=args.start_step,
            active_steps=args.active_steps,
            warmup_steps=args.warmup_steps,
            enabled=args.enable,
        )
    destination = write_dynamic_profile_config(args.config_dir, config)
    state = "enabled" if config["is_valid"] else "disabled"
    print(f"Dynamic profiler configuration ({state}): {destination.resolve()}")
    print(f"export PROF_CONFIG_PATH={args.config_dir.resolve()}")


if __name__ == "__main__":
    main()
