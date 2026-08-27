# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

from pathlib import Path

import pytest

from tests.glm5_2_combination.combination_benchmark import (
    CONFIG as COMBINATION_CONFIG,
    TOPOLOGY_NAMES,
)
from tests.glm5_2_combination.workflow import (
    CombinationSelection,
    _apply_selection,
    _relative_link,
)
from tests.glm5_2_common.execution import TrainingFeature, compose_execution
from tests.glm5_2_common.topology import select_topologies, standard_topologies
from tests.glm5_2_graph.config import (
    GraphFeatureConfig,
    validate_graph_training_args,
)
from tests.glm5_2_graph.precision_benchmark import CONFIG as GRAPH_CONFIG
from tests.glm5_2_graph.gpu_compile_probe import PROBE_CONFIG as GPU_PROBE_CONFIG
from tests.glm5_2_graph.gpu_precision_benchmark import CONFIG as GPU_GRAPH_CONFIG
from tests.glm5_2_performance.analysis import _compiler_diagnostics
from tests.glm5_2_precision.topology_suite import topology_config
from tests.glm5_2_precision.workflow import (
    FormalExperimentConfig,
    TrainingEndpoint,
    _artifact_directory,
)


def test_graph_feature_generates_only_compile_arguments() -> None:
    eager = GraphFeatureConfig("eager").feature(device_type="npu")
    compiled = GraphFeatureConfig("inductor").feature(device_type="npu")

    assert eager.arguments == ()
    assert compiled.arguments == (
        "--compile.enable",
        "--compile.components=model",
        "--compile.backend=inductor",
    )


def test_inductor_supports_cuda_and_npu_but_npugraphs_is_npu_only() -> None:
    assert "--compile.backend=inductor" in GraphFeatureConfig(
        "inductor"
    ).feature(device_type="cuda").arguments
    with pytest.raises(NotImplementedError, match="only NPU"):
        GraphFeatureConfig("npugraphs").feature(device_type="cuda")
    assert "--compile.backend=npugraphs" in GraphFeatureConfig(
        "npugraphs"
    ).feature(device_type="npu").arguments


def test_raw_compile_arguments_use_the_same_device_gate() -> None:
    validate_graph_training_args(device_type="cuda", arguments=("--debug.seed=61",))
    validate_graph_training_args(
        device_type="cuda",
        arguments=("--compile.enable", "--compile.backend=inductor"),
    )
    with pytest.raises(NotImplementedError, match="only NPU"):
        validate_graph_training_args(
            device_type="cuda",
            arguments=("--compile.enable", "--compile.backend=npugraphs"),
        )


def test_graph_benchmark_compares_npu_eager_and_npu_graph() -> None:
    assert GRAPH_CONFIG.kind == "self_consistency"
    assert GRAPH_CONFIG.reference.device_type == "npu"
    assert GRAPH_CONFIG.candidate.device_type == "npu"
    assert GRAPH_CONFIG.reference.topology == GRAPH_CONFIG.candidate.topology


def test_gpu_graph_benchmark_is_single_cuda_and_isolates_inductor() -> None:
    assert GPU_GRAPH_CONFIG.kind == "self_consistency"
    assert GPU_GRAPH_CONFIG.reference.device_type == "cuda"
    assert GPU_GRAPH_CONFIG.candidate.device_type == "cuda"
    assert GPU_GRAPH_CONFIG.reference.topology.name == "single"
    assert GPU_GRAPH_CONFIG.candidate.topology.name == "single"
    assert GPU_GRAPH_CONFIG.training.steps == 1000
    assert GPU_GRAPH_CONFIG.training.extra_args == (
        "--training.disable_cuda_graphs",
    )
    assert GPU_PROBE_CONFIG.training.steps == 10

    selection = CombinationSelection(
        objectives=frozenset(("precision",)),
        reference_graph=GraphFeatureConfig("eager"),
        candidate_graph=GraphFeatureConfig("inductor", diagnostics=True),
        profiler=None,
    )
    configured = _apply_selection(
        topology_config(
            GPU_GRAPH_CONFIG,
            GPU_GRAPH_CONFIG.candidate.topology,
            precision="bf16",
        ),
        selection,
    )
    assert not any(
        argument.startswith("--compile.")
        for argument in configured.reference.extra_args
    )
    assert "--compile.backend=inductor" in configured.candidate.extra_args
    assert configured.candidate.environment["TORCH_LOGS"] == (
        "graph_breaks,recompiles,dynamic"
    )


def test_combination_defaults_to_npu_self_consistency() -> None:
    assert COMBINATION_CONFIG.kind == "self_consistency"
    assert COMBINATION_CONFIG.reference.device_type == "npu"
    assert COMBINATION_CONFIG.candidate.device_type == "npu"


def test_combination_all_selector_is_bounded_to_eight_devices() -> None:
    topologies = standard_topologies()

    assert "single" in TOPOLOGY_NAMES
    assert all(topologies[name].world_size <= 8 for name in TOPOLOGY_NAMES)
    assert "ddp16" not in TOPOLOGY_NAMES


def test_self_consistency_keeps_single_reference_for_distributed_candidate() -> None:
    topologies = standard_topologies()
    configured = topology_config(
        COMBINATION_CONFIG,
        topologies["fsdp8"],
        precision="bf16",
    )

    assert configured.reference.topology.name == "single"
    assert configured.candidate.topology.name == "fsdp8"


def test_combined_cuda_performance_is_reserved() -> None:
    topologies = standard_topologies()
    base = FormalExperimentConfig(
        name="cuda-performance",
        kind="self_consistency",
        reference=TrainingEndpoint("reference", "cuda", "0", topologies["single"]),
        candidate=TrainingEndpoint("candidate", "cuda", "0", topologies["single"]),
    )
    selection = CombinationSelection(
        objectives=frozenset(("performance",)),
        reference_graph=GraphFeatureConfig("eager"),
        candidate_graph=GraphFeatureConfig("eager"),
        profiler=None,
    )

    with pytest.raises(NotImplementedError, match="only NPU"):
        _apply_selection(
            topology_config(base, topologies["single"], precision="bf16"),
            selection,
        )


def test_npu_performance_does_not_require_profiler_collection() -> None:
    topologies = standard_topologies()
    selection = CombinationSelection(
        objectives=frozenset(("performance",)),
        reference_graph=GraphFeatureConfig("eager"),
        candidate_graph=GraphFeatureConfig("inductor"),
        profiler=None,
    )

    configured = _apply_selection(
        topology_config(COMBINATION_CONFIG, topologies["single"], precision="bf16"),
        selection,
    )

    assert "--profiler.enable_profiling" not in configured.reference.extra_args
    assert "--profiler.enable_profiling" not in configured.candidate.extra_args


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


def test_combination_report_can_link_to_sibling_precision_report(
    tmp_path: Path,
) -> None:
    report_root = tmp_path / "combination_reports" / "experiment"
    precision_report = (
        tmp_path
        / "combination_reports"
        / "precision"
        / "experiment"
        / "suite.html"
    )

    assert _relative_link(precision_report, report_root) == (
        "../precision/experiment/suite.html"
    )


def test_combination_topologies_share_one_suite_root(tmp_path: Path) -> None:
    topologies = standard_topologies()
    all_devices = "0,1,2,3,4,5,6,7"
    base = FormalExperimentConfig(
        name="combined",
        kind="migration",
        reference=TrainingEndpoint(
            "gpu", "cuda", all_devices, topologies["single"]
        ),
        candidate=TrainingEndpoint(
            "npu", "npu", all_devices, topologies["single"]
        ),
    )
    selection = CombinationSelection(
        objectives=frozenset(("precision",)),
        reference_graph=GraphFeatureConfig("eager"),
        candidate_graph=GraphFeatureConfig("inductor"),
        profiler=None,
    )
    single = _apply_selection(
        topology_config(base, topologies["single"], precision="bf16"), selection
    )
    fsdp = _apply_selection(
        topology_config(base, topologies["fsdp8"], precision="bf16"), selection
    )

    assert single.storage_name == fsdp.storage_name
    assert single.fixture_storage_name == fsdp.fixture_storage_name
    assert single.fixture_storage_name != single.storage_name
    assert not single.storage_name.startswith("combo-")
    single_path = _artifact_directory(
        tmp_path, single, "candidate", single.candidate, 1
    )
    fsdp_path = _artifact_directory(
        tmp_path, fsdp, "candidate", fsdp.candidate, 1
    )
    assert single_path.parents[1] == fsdp_path.parents[1]
    assert single_path.parent.name == "single"
    assert fsdp_path.parent.name == "fsdp8"
