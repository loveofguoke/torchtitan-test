"""Run a matrix of distributed GPU/NPU comparison scenarios."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from distributed.artifacts import read_json
from distributed.workflow import _load_scenario, capture, compare, prepare_fixture


def _scenario_paths(suite_path: Path) -> list[Path]:
    suite = read_json(suite_path)
    configured = suite.get("scenarios")
    if not isinstance(configured, list) or not configured:
        raise ValueError("suite scenarios must be a non-empty list")
    paths = []
    for value in configured:
        if not isinstance(value, str):
            raise ValueError("suite scenario paths must be strings")
        path = Path(value)
        if not path.is_absolute():
            path = (suite_path.parent / path).resolve()
        paths.append(path)
    if len({path.stem for path in paths}) != len(paths):
        raise ValueError("suite scenario filenames must have unique stems")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    stages = parser.add_subparsers(dest="stage", required=True)

    prepare_parser = stages.add_parser("prepare")
    prepare_parser.add_argument("--backend", choices=("gpu", "npu"), default="gpu")

    capture_parser = stages.add_parser("capture")
    capture_parser.add_argument("--backend", choices=("gpu", "npu"), required=True)

    stages.add_parser("compare")
    stages.add_parser("list")

    arguments = parser.parse_args()
    suite_path = arguments.suite.resolve()
    output_root = arguments.output_root.resolve()
    scenario_paths = _scenario_paths(suite_path)
    all_passed = True

    for index, scenario_path in enumerate(scenario_paths, start=1):
        scenario, scenario_digest = _load_scenario(scenario_path)
        scenario_id = scenario_path.stem
        fixture = output_root / "fixtures" / scenario_id
        print(f"\n[{index}/{len(scenario_paths)}] {scenario_id}: {scenario['name']}")

        if arguments.stage == "list":
            print(
                f"world_size={scenario['world_size']} steps={scenario['steps']} "
                f"deterministic={scenario['deterministic']}"
            )
        elif arguments.stage == "prepare":
            if arguments.skip_existing and fixture.exists():
                print(f"skip existing fixture: {fixture}")
                continue
            prepare_fixture(
                scenario_path,
                scenario,
                scenario_digest,
                fixture,
                arguments.backend,
            )
        elif arguments.stage == "capture":
            backend = arguments.backend
            output = output_root / "captures" / f"{scenario_id}_{backend}.json"
            if arguments.skip_existing and output.exists():
                existing = read_json(output)
                if existing.get("status") != "success":
                    if arguments.continue_on_error:
                        print(f"skip existing failed capture: {output}")
                        all_passed = False
                        continue
                    raise RuntimeError(
                        f"refusing to skip unsuccessful capture: {output}"
                    )
                print(f"skip existing successful capture: {output}")
                continue
            try:
                capture(
                    scenario,
                    scenario_digest,
                    fixture,
                    backend,
                    output,
                )
            except Exception as error:
                if not arguments.continue_on_error:
                    raise
                all_passed = False
                print(f"capture failed: {scenario_id}: {type(error).__name__}: {error}")
        elif arguments.stage == "compare":
            passed = compare(
                scenario,
                scenario_digest,
                fixture,
                output_root / "captures" / f"{scenario_id}_gpu.json",
                output_root / "captures" / f"{scenario_id}_npu.json",
                output_root / "reports" / f"{scenario_id}.json",
            )
            all_passed &= passed

    if arguments.stage in {"capture", "compare"} and not all_passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
