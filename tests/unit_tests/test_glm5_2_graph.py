# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

from pathlib import Path

import pytest

from tests.glm5_2_combination.config import GraphFeatureConfig
from tests.glm5_2_common.execution import TrainingFeature, compose_execution
from tests.glm5_2_common.topology import select_topologies, standard_topologies
from tests.glm5_2_performance.analysis import _compiler_diagnostics


def test_graph_feature_generates_only_compile_arguments() -> None:
    eager = GraphFeatureConfig("eager").feature(device_type="cuda")
    compiled = GraphFeatureConfig("inductor").feature(device_type="cuda")

    assert eager.arguments == ()
    assert compiled.arguments == (
        "--compile.enable",
        "--compile.components=model",
        "--compile.backend=inductor",
    )


def test_npugraphs_is_npu_only() -> None:
    with pytest.raises(ValueError, match="NPU endpoint"):
        GraphFeatureConfig("npugraphs").feature(device_type="cuda")
    assert "--compile.backend=npugraphs" in GraphFeatureConfig(
        "npugraphs"
    ).feature(device_type="npu").arguments


def test_execution_plan_rejects_cross_feature_option_conflicts() -> None:
    topology = standard_topologies()["single"]
    with pytest.raises(ValueError, match="supplied by both"):
        compose_execution(
            topology,
            (
                TrainingFeature("first", ("--compile.backend=inductor",)),
                TrainingFeature("second", ("--compile.backend=npugraphs",)),
            ),
        )


def test_topology_selector_supports_all_and_subsets() -> None:
    available = ("single", "ddp8", "fsdp8")
    assert select_topologies(available=available, topology="all") == available
    assert select_topologies(
        available=available,
        topologies="single,fsdp8",
    ) == ("single", "fsdp8")


def test_compiler_diagnostics_summarize_runtime_log(tmp_path: Path) -> None:
    runtime_log = tmp_path / "runtime.log"
    runtime_log.write_text(
        "Compiling each TransformerBlock with torch.compile\n"
        "Graph break from user code\n"
        "Recompiling function forward\n"
        "BackendCompilerFailed: inductor\n",
        encoding="utf-8",
    )

    diagnostics = _compiler_diagnostics(runtime_log)

    assert diagnostics["compile_markers"] == 1
    assert diagnostics["graph_breaks"] == 1
    assert diagnostics["recompiles"] == 1
    assert diagnostics["backend_failures"] == 1
