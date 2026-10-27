from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from tests.glm5_2_feature_acceptance.communication import (
    CommunicationEvidence,
    diagnose_communication,
)
from tests.glm5_2_feature_acceptance.optimizations import (
    OptimizationReadiness,
    build_optimization_plans,
    build_optimization_registry,
    validate_optimization_registry,
    write_optimization_registry,
)
from tests.glm5_2_feature_acceptance.compile_contracts import (
    CompileContractKind,
    build_compile_contracts,
    validate_compile_contracts,
)


class OptimizationRegistryTest(unittest.TestCase):
    def test_registry_is_unique_and_complete(self) -> None:
        items = validate_optimization_registry(build_optimization_registry())
        self.assertEqual(len(items), len({item.optimization_id for item in items}))
        self.assertTrue(all(item.metrics and item.topologies for item in items))

    def test_graph_only_glm_optimizations_are_not_falsely_runnable(self) -> None:
        items = {item.optimization_id: item for item in build_optimization_registry()}
        for optimization_id in (
            "async_tp",
            "fsdp_ag_rs_overlap",
            "fsdp_dense_overlap",
            "ep_overlap",
        ):
            self.assertEqual(
                items[optimization_id].npu,
                OptimizationReadiness.REQUIRES_INTEGRATION,
            )

    def test_hyperparallel_inspired_candidates_are_not_falsely_runnable(self) -> None:
        items = {item.optimization_id: item for item in build_optimization_registry()}
        for optimization_id in (
            "async_context_parallel",
            "pp_forward_backward_overlap",
            "topology_aware_collectives",
            "hot_expert_replication",
            "activation_swap_prefetch",
            "autoparallel_cost_model",
            "ascend_multicore_moe",
        ):
            self.assertNotEqual(
                items[optimization_id].npu,
                OptimizationReadiness.RUNNABLE,
            )

    def test_compile_contracts_separate_graph_and_dynamic_shape_claims(self) -> None:
        contracts = validate_compile_contracts(build_compile_contracts())
        self.assertEqual(len(contracts), 4)
        kinds = {item.kind for item in contracts}
        self.assertEqual(
            kinds,
            {
                CompileContractKind.END_TO_END_GRAPH,
                CompileContractKind.OPERATOR_DYNAMIC_SHAPE,
            },
        )
        graph_contracts = [
            item for item in contracts
            if item.kind is CompileContractKind.END_TO_END_GRAPH
        ]
        self.assertTrue(
            all("--compiler-diagnostics" in item.command for item in graph_contracts)
        )

    def test_platform_plans_only_emit_activation_when_runnable(self) -> None:
        plans = {
            (item.optimization_id, item.platform.value): item
            for item in build_optimization_plans()
        }
        self.assertEqual(
            plans[("fsdp_bf16_reduce", "npu")].candidate_args,
            ("--training.mixed_precision_reduce=bfloat16",),
        )
        self.assertEqual(plans[("async_tp", "npu")].candidate_args, ())
        self.assertEqual(
            plans[("async_tp", "npu")].readiness,
            OptimizationReadiness.REQUIRES_INTEGRATION,
        )

    def test_communication_triage_selects_collective_family(self) -> None:
        recommendations = diagnose_communication(
            CommunicationEvidence(
                exposed_percent=35.0,
                overlap_percent=30.0,
                rank_ready_skew_ms=0.2,
                effective_bandwidth_ratio=0.4,
                repeated_fsdp_all_gathers=3,
                all_to_all_percent=25.0,
            )
        )
        selected = {
            optimization_id
            for item in recommendations
            for optimization_id in item.optimization_ids
        }
        self.assertIn("fsdp_no_reshard", selected)
        self.assertIn("deepep_ascend", selected)
        self.assertIn("fsdp_bf16_reduce", selected)

    def test_low_exposure_does_not_recommend_collective_changes(self) -> None:
        recommendations = diagnose_communication(
            CommunicationEvidence(
                exposed_percent=4.0,
                overlap_percent=80.0,
                rank_ready_skew_ms=0.0,
            )
        )
        self.assertEqual(recommendations[0].classification, "not_primary")
        self.assertEqual(recommendations[0].optimization_ids, ())

    def test_registry_report_is_machine_readable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = write_optimization_registry(Path(directory) / "registry.json")
            payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["schema_version"], 1)
        self.assertTrue(
            any(
                item["optimization_id"] == "fsdp_bf16_reduce"
                for item in payload["optimizations"]
            )
        )


if __name__ == "__main__":
    unittest.main()
