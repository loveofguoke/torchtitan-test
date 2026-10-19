"""Generate the reviewed TorchTitan feature acceptance manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

from tests.glm5_2_feature_acceptance.matrix import build_feature_matrix, write_outputs
from tests.glm5_2_feature_acceptance.optimizations import (
    write_optimization_plans,
    write_optimization_registry,
)
from tests.glm5_2_feature_acceptance.plans import write_plan_outputs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("feature_acceptance_reports/current"),
    )
    args = parser.parse_args()
    json_path, markdown_path = write_outputs(args.output_directory)
    plan_json_path, plan_markdown_path = write_plan_outputs(
        build_feature_matrix(), args.output_directory
    )
    optimization_path = write_optimization_registry(
        args.output_directory / "optimization_registry.json"
    )
    optimization_plan_path = write_optimization_plans(
        args.output_directory / "optimization_plans.json"
    )
    print(f"Feature matrix JSON: {json_path}")
    print(f"Feature matrix Markdown: {markdown_path}")
    print(f"Acceptance plans JSON: {plan_json_path}")
    print(f"Acceptance plans Markdown: {plan_markdown_path}")
    print(f"Optimization registry JSON: {optimization_path}")
    print(f"Optimization A/B plans JSON: {optimization_plan_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

