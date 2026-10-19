from __future__ import annotations

import unittest
from pathlib import Path
import json
import tempfile

from tests.glm5_2_feature_acceptance.matrix import Platform, build_feature_matrix
from tests.glm5_2_feature_acceptance.plans import (
    PlanDisposition,
    build_acceptance_plan,
    write_plan_outputs,
)


class FeatureAcceptancePlanTest(unittest.TestCase):
    def setUp(self) -> None:
        self.features = {item.feature_id: item for item in build_feature_matrix()}

    def test_topology_feature_has_runnable_gpu_and_npu_plans(self) -> None:
        for platform in Platform:
            plan = build_acceptance_plan(self.features["fsdp2"], platform)
            self.assertEqual(plan.disposition, PlanDisposition.RUNNABLE)
            topology_index = plan.smoke_command.index("--topologies")
            self.assertIn("fsdp2", plan.smoke_command[topology_index + 1].split(","))
            self.assertIn("--profiler-off", plan.performance_command)

    def test_local_compile_selects_inductor(self) -> None:
        plan = build_acceptance_plan(self.features["local_compile"], Platform.NPU)
        self.assertEqual(plan.disposition, PlanDisposition.RUNNABLE)
        graph_index = plan.smoke_command.index("--graph")
        self.assertEqual(plan.smoke_command[graph_index + 1], "inductor")
        self.assertIn("ascend-triton", plan.smoke_command)

    def test_configuration_prototype_does_not_emit_false_evidence_command(self) -> None:
        plan = build_acceptance_plan(self.features["async_tp"], Platform.NPU)
        self.assertEqual(plan.disposition, PlanDisposition.REQUIRES_INTEGRATION)
        self.assertIsNone(plan.smoke_command)
        self.assertIsNone(plan.performance_command)

    def test_capability_block_has_no_command(self) -> None:
        plan = build_acceptance_plan(self.features["hybridep"], Platform.NPU)
        self.assertEqual(plan.disposition, PlanDisposition.BLOCKED)
        self.assertIsNone(plan.smoke_command)

    def test_plan_reports_preserve_non_runnable_reasons(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            json_path, markdown_path = write_plan_outputs(
                self.features.values(), Path(directory)
            )
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            plans = {
                (item["feature_id"], item["platform"]): item
                for item in payload["plans"]
            }
            blocked = plans[("hybridep", "npu")]
            self.assertEqual(blocked["disposition"], "blocked")
            self.assertIsNone(blocked["smoke_command"])
            self.assertIn("`hybridep`", markdown_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
