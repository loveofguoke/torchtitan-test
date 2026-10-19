"""Generate honest minimal acceptance plans from the feature manifest.

A topology command is emitted only when selecting that topology actually
activates the feature. Configuration-only prototypes remain integration work;
the planner never prints a baseline command and labels it as feature evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import json
from pathlib import Path
from typing import Iterable

from .matrix import AcceptanceStatus, FeatureSpec, Platform


class PlanDisposition(StrEnum):
    RUNNABLE = "runnable"
    REQUIRES_INTEGRATION = "requires-integration"
    BLOCKED = "blocked"


@dataclass(frozen=True, slots=True)
class AcceptancePlan:
    feature_id: str
    platform: Platform
    disposition: PlanDisposition
    smoke_command: tuple[str, ...] | None
    performance_command: tuple[str, ...] | None
    explanation: str


# These features are activated by the topology/graph selectors themselves.
# Other features require a recipe or provider selector that must be added and
# tested before a launcher command can constitute evidence.
TOPOLOGY_ACTIVATED_FEATURES = frozenset(
    {
        "ddp",
        "fsdp2",
        "hsdp",
        "tp",
        "cp",
        "pp",
        "ep",
        "local_compile",
        "observability",
    }
)


def _topologies(feature: FeatureSpec) -> str:
    return ",".join(feature.required_topologies)


def build_acceptance_plan(
    feature: FeatureSpec,
    platform: Platform,
) -> AcceptancePlan:
    """Build the minimal smoke/performance plan without inventing activation."""
    acceptance = feature.acceptance(platform)
    if acceptance.status in {
        AcceptanceStatus.BLOCKED_BY_CAPABILITY,
        AcceptanceStatus.BLOCKED_BY_VERSION,
    }:
        return AcceptancePlan(
            feature.feature_id,
            platform,
            PlanDisposition.BLOCKED,
            None,
            None,
            acceptance.reason,
        )

    if feature.feature_id not in TOPOLOGY_ACTIVATED_FEATURES:
        return AcceptancePlan(
            feature.feature_id,
            platform,
            PlanDisposition.REQUIRES_INTEGRATION,
            None,
            None,
            (
                "Add or select the feature-specific recipe/provider first; a "
                "baseline topology command would not prove activation. "
                + acceptance.reason
            ),
        )

    smoke_device = platform.value
    performance_device = "cuda" if platform is Platform.GPU else "npu"
    graph = "inductor" if feature.feature_id == "local_compile" else "eager"
    smoke = (
        "python",
        "tests/glm5_2_smoke/train_smoke.py",
        "--device",
        smoke_device,
        "--topologies",
        _topologies(feature),
        "--graph",
        graph,
        "--module",
        "glm5",
        "--config",
        "glm5_debugmodel",
        "--steps",
        "2",
        "--sequence-length",
        "128",
    )
    if platform is Platform.NPU:
        smoke += ("--npu-codegen", "ascend-triton")

    performance = (
        "python",
        "tests/glm5_2_mindstudio/performance_benchmark.py",
        "--capture",
        "--device",
        performance_device,
        "--topologies",
        _topologies(feature),
        "--graph",
        graph,
        "--steps",
        "30",
        "--replicate",
        "3",
        "--profiler-off",
    )
    if platform is Platform.NPU:
        performance += ("--npu-codegen", "ascend-triton")

    return AcceptancePlan(
        feature.feature_id,
        platform,
        PlanDisposition.RUNNABLE,
        smoke,
        performance,
        "Topology or graph selection activates this feature.",
    )


def shell_command(command: tuple[str, ...] | None) -> str | None:
    """Render commands whose arguments contain no shell-sensitive values."""
    return None if command is None else " ".join(command)


def write_plan_outputs(
    features: Iterable[FeatureSpec], output_dir: Path
) -> tuple[Path, Path]:
    """Write all GPU/NPU plans, including explicit non-runnable reasons."""
    plans = [
        build_acceptance_plan(feature, platform)
        for feature in features
        for platform in Platform
    ]
    payload = {
        "schema_version": 1,
        "plans": [
            {
                "feature_id": plan.feature_id,
                "platform": plan.platform.value,
                "disposition": plan.disposition.value,
                "smoke_command": shell_command(plan.smoke_command),
                "performance_command": shell_command(plan.performance_command),
                "explanation": plan.explanation,
            }
            for plan in plans
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "acceptance_plans.json"
    markdown_path = output_dir / "acceptance_plans.md"
    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    rows = [
        "# TorchTitan feature acceptance plans",
        "",
        "| Feature | Platform | Disposition | Smoke | Performance | Reason |",
        "|---|---|---|---|---|---|",
    ]
    for plan in plans:
        smoke = shell_command(plan.smoke_command) or "—"
        performance = shell_command(plan.performance_command) or "—"
        rows.append(
            f"| `{plan.feature_id}` | `{plan.platform.value}` | "
            f"`{plan.disposition.value}` | `{smoke}` | `{performance}` | "
            f"{plan.explanation} |"
        )
    markdown_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return json_path, markdown_path
