# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import json
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
from tests.glm5_2_graph.compare_gpu_block_traces import compare_traces
from tests.glm5_2_graph.compare_gpu_attention_traces import (
    compare_attention_traces,
)
from tests.glm5_2_graph.compare_gpu_internal_traces import compare_internal_traces
from tests.glm5_2_graph.compare_gpu_minimal_results import compare_minimal_results
from tests.glm5_2_graph.compare_gpu_silu_control_traces import (
    ORDERED_KEYS as SILU_CONTROL_KEYS,
    compare_control_traces,
)
from tests.glm5_2_common.execution import TrainingFeature, compose_execution
from tests.glm5_2_common.topology import select_topologies, standard_topologies
from tests.glm5_2_graph.config import (
    GraphFeatureConfig,
    validate_graph_training_args,
)
from tests.glm5_2_graph.precision_benchmark import CONFIG as GRAPH_CONFIG
from tests.glm5_2_graph.gpu_compile_probe import PROBE_CONFIG as GPU_PROBE_CONFIG
from tests.glm5_2_graph.gpu_attention_internal_trace import ATTENTION_STAGE_NAMES
from tests.glm5_2_graph.gpu_attention_trace_probe import (
    ATTENTION_TRACE_CONFIG as GPU_ATTENTION_TRACE_CONFIG,
)
from tests.glm5_2_graph.gpu_block_trace_probe import TRACE_CONFIG as GPU_TRACE_CONFIG
from tests.glm5_2_graph.gpu_internal_block_trace import (
    STAGE_NAMES,
    _split_primary_and_auxiliary,
)
from tests.glm5_2_graph.gpu_internal_trace_probe import (
    INTERNAL_TRACE_CONFIG as GPU_INTERNAL_TRACE_CONFIG,
)
from tests.glm5_2_graph.gpu_precision_benchmark import CONFIG as GPU_GRAPH_CONFIG
from tests.glm5_2_graph.gpu_silu_materialization_benchmark import (
    MATERIALIZATION_CONFIG as GPU_SILU_MATERIALIZATION_CONFIG,
)
from tests.glm5_2_graph.gpu_silu_materialization_probe import (
    PROBE_CONFIG as GPU_SILU_MATERIALIZATION_PROBE_CONFIG,
)
from tests.glm5_2_graph.gpu_silu_materialization import MATERIALIZATION_POINTS
from tests.glm5_2_graph.gpu_silu_staged_control import (
    STAGED_CONFIG as GPU_SILU_STAGED_CONFIG,
    VARIANT_POINTS as GPU_SILU_VARIANT_POINTS,
)
from tests.glm5_2_graph.gpu_silu_symmetric_probe import (
    SYMMETRIC_CONFIG as GPU_SILU_SYMMETRIC_CONFIG,
)
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
    assert configured.reference.entry_module == (
        "tests.glm5_2_combination.capture_metrics"
    )


def test_gpu_block_trace_uses_dedicated_cuda_capture() -> None:
    assert GPU_TRACE_CONFIG.training.steps == 20
    assert GPU_TRACE_CONFIG.reference.device_type == "cuda"
    assert GPU_TRACE_CONFIG.candidate.device_type == "cuda"
    assert GPU_TRACE_CONFIG.reference.entry_module == (
        "tests.glm5_2_graph.gpu_diagnostic_capture"
    )
    assert GPU_TRACE_CONFIG.candidate.entry_module == (
        "tests.glm5_2_graph.gpu_diagnostic_capture"
    )
    assert GPU_TRACE_CONFIG.reference.environment["GLM5_GPU_DIAGNOSTIC"] == (
        "block-trace-v1"
    )

    selection = CombinationSelection(
        objectives=frozenset(("precision",)),
        reference_graph=GraphFeatureConfig("eager"),
        candidate_graph=GraphFeatureConfig("inductor", diagnostics=True),
        profiler=None,
    )
    configured = _apply_selection(
        topology_config(
            GPU_TRACE_CONFIG,
            GPU_TRACE_CONFIG.candidate.topology,
            precision="fp32",
        ),
        selection,
    )
    assert configured.reference.entry_module == (
        "tests.glm5_2_graph.gpu_diagnostic_capture"
    )
    assert configured.candidate.entry_module == (
        "tests.glm5_2_graph.gpu_diagnostic_capture"
    )


def test_gpu_block_trace_comparison_reports_replays_and_eager_drift(
    tmp_path: Path,
) -> None:
    def write_trace(path: Path, *, output_digest: str, output_mean: float) -> None:
        rows = (
            {
                "step": 1,
                "name": "block_input",
                "kind": "forward",
                "sha256": "input",
                "mean": 0.0,
                "max_abs": 1.0,
                "l2": 2.0,
            },
            {
                "step": 1,
                "name": "block_output",
                "kind": "forward",
                "sha256": output_digest,
                "mean": output_mean,
                "max_abs": 3.0,
                "l2": 4.0,
            },
        )
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )

    paths = {
        name: tmp_path / f"{name}.jsonl"
        for name in ("eager-r1", "eager-r2", "inductor-r1", "inductor-r2")
    }
    write_trace(paths["eager-r1"], output_digest="eager", output_mean=1.0)
    write_trace(paths["eager-r2"], output_digest="eager", output_mean=1.0)
    write_trace(paths["inductor-r1"], output_digest="graph", output_mean=1.25)
    write_trace(paths["inductor-r2"], output_digest="graph", output_mean=1.25)

    result = compare_traces(paths)
    comparisons = {
        (row["left"], row["right"]): row for row in result["comparisons"]
    }

    assert comparisons[("eager-r1", "eager-r2")]["mismatch_records"] == 0
    assert comparisons[("inductor-r1", "inductor-r2")][
        "mismatch_records"
    ] == 0
    eager_graph = comparisons[("eager-r1", "inductor-r1")]
    assert eager_graph["mismatch_records"] == 1
    assert eager_graph["first_mismatches"][0]["name"] == "block_output"
    assert eager_graph["max_mean_abs_delta"] == pytest.approx(0.25)


def test_gpu_internal_trace_preserves_capture_and_orders_first_divergence(
    tmp_path: Path,
) -> None:
    selection = CombinationSelection(
        objectives=frozenset(("precision",)),
        reference_graph=GraphFeatureConfig("eager"),
        candidate_graph=GraphFeatureConfig("inductor", diagnostics=True),
        profiler=None,
    )
    configured = _apply_selection(
        topology_config(
            GPU_INTERNAL_TRACE_CONFIG,
            GPU_INTERNAL_TRACE_CONFIG.candidate.topology,
            precision="bf16",
        ),
        selection,
    )
    assert configured.training.steps == 10
    assert configured.reference.entry_module == (
        "tests.glm5_2_graph.gpu_diagnostic_capture"
    )
    assert configured.candidate.entry_module == (
        "tests.glm5_2_graph.gpu_diagnostic_capture"
    )

    def write_trace(path: Path, *, divergent: bool) -> None:
        rows = []
        for index, stage in enumerate(STAGE_NAMES):
            changed = divergent and stage in STAGE_NAMES[2:]
            rows.append(
                {
                    "step": 1,
                    "name": stage,
                    "kind": "forward",
                    "sha256": f"{stage}-{'different' if changed else 'same'}",
                    "mean": float(index) + (0.25 if changed else 0.0),
                    "max_abs": float(index + 1),
                    "l2": float(index + 2) + (0.5 if changed else 0.0),
                }
            )
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )

    paths = {
        name: tmp_path / f"{name}.jsonl"
        for name in ("eager-r1", "eager-r2", "inductor-r1", "inductor-r2")
    }
    write_trace(paths["eager-r1"], divergent=False)
    write_trace(paths["eager-r2"], divergent=False)
    write_trace(paths["inductor-r1"], divergent=True)
    write_trace(paths["inductor-r2"], divergent=True)

    result = compare_internal_traces(paths)
    eager_replay, inductor_replay, eager_inductor = result["comparisons"][:3]
    assert eager_replay["exact_records"] == len(STAGE_NAMES)
    assert inductor_replay["exact_records"] == len(STAGE_NAMES)
    first = eager_inductor["first_divergence_by_step"][0]["first_divergence"]
    assert first["stage"] == "attention_output"
    assert first["mean_abs_delta"] == pytest.approx(0.25)


def test_gpu_internal_trace_preserves_shared_indexer_auxiliary_output() -> None:
    primary, auxiliary = _split_primary_and_auxiliary(("hidden", "topk"))
    assert primary == "hidden"
    assert auxiliary == ("topk",)

    primary, auxiliary = _split_primary_and_auxiliary("hidden")
    assert primary == "hidden"
    assert auxiliary == ()


def test_gpu_attention_trace_orders_absorbed_mla_stages(tmp_path: Path) -> None:
    assert GPU_ATTENTION_TRACE_CONFIG.training.steps == 1
    assert GPU_ATTENTION_TRACE_CONFIG.reference.entry_module == (
        "tests.glm5_2_graph.gpu_diagnostic_capture"
    )

    def write_trace(path: Path, *, divergent: bool) -> None:
        rows = []
        for index, stage in enumerate(ATTENTION_STAGE_NAMES):
            changed = divergent and stage in ATTENTION_STAGE_NAMES[3:]
            rows.append(
                {
                    "step": 1,
                    "name": stage,
                    "kind": "forward",
                    "sha256": f"{stage}-{'different' if changed else 'same'}",
                    "mean": float(index) + (0.125 if changed else 0.0),
                    "max_abs": float(index + 1),
                    "l2": float(index + 2) + (0.25 if changed else 0.0),
                }
            )
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )

    paths = {
        name: tmp_path / f"{name}.jsonl"
        for name in ("eager-r1", "eager-r2", "inductor-r1", "inductor-r2")
    }
    write_trace(paths["eager-r1"], divergent=False)
    write_trace(paths["eager-r2"], divergent=False)
    write_trace(paths["inductor-r1"], divergent=True)
    write_trace(paths["inductor-r2"], divergent=True)
    result = compare_attention_traces(paths)
    assert result["comparisons"][0]["exact_records"] == len(
        ATTENTION_STAGE_NAMES
    )
    assert result["comparisons"][1]["exact_records"] == len(
        ATTENTION_STAGE_NAMES
    )
    first = result["comparisons"][2]["first_divergence"]
    assert first["stage"] == "q_norm"
    assert first["mean_abs_delta"] == pytest.approx(0.125)


def test_gpu_silu_materialization_is_candidate_only_and_has_two_lengths() -> None:
    assert GPU_SILU_MATERIALIZATION_PROBE_CONFIG.training.steps == 10
    assert GPU_SILU_MATERIALIZATION_CONFIG.training.steps == 1000
    assert "GLM5_GPU_MATERIALIZE_DENSE_SILU" not in (
        GPU_SILU_MATERIALIZATION_CONFIG.reference.environment
    )
    assert GPU_SILU_MATERIALIZATION_CONFIG.candidate.environment[
        "GLM5_GPU_MATERIALIZE_DENSE_SILU"
    ] == "1"
    assert GPU_SILU_MATERIALIZATION_CONFIG.reference.entry_module == (
        "tests.glm5_2_graph.gpu_diagnostic_capture"
    )
    assert GPU_SILU_MATERIALIZATION_CONFIG.candidate.entry_module == (
        "tests.glm5_2_graph.gpu_diagnostic_capture"
    )
    selection = CombinationSelection(
        objectives=frozenset(("precision",)),
        reference_graph=GraphFeatureConfig("eager"),
        candidate_graph=GraphFeatureConfig("inductor", diagnostics=True),
        profiler=None,
    )
    configured = _apply_selection(
        topology_config(
            GPU_SILU_MATERIALIZATION_CONFIG,
            GPU_SILU_MATERIALIZATION_CONFIG.candidate.topology,
            precision="bf16",
        ),
        selection,
    )
    assert configured.candidate.entry_module == (
        "tests.glm5_2_graph.gpu_diagnostic_capture"
    )
    assert configured.candidate.environment[
        "GLM5_GPU_MATERIALIZE_DENSE_SILU"
    ] == "1"


def test_gpu_silu_symmetric_control_patches_both_endpoints() -> None:
    assert GPU_SILU_SYMMETRIC_CONFIG.training.steps == 10
    assert GPU_SILU_SYMMETRIC_CONFIG.reference.environment[
        "GLM5_GPU_MATERIALIZE_DENSE_SILU"
    ] == "1"
    assert GPU_SILU_SYMMETRIC_CONFIG.candidate.environment[
        "GLM5_GPU_MATERIALIZE_DENSE_SILU"
    ] == "1"
    assert GPU_SILU_SYMMETRIC_CONFIG.reference.entry_module == (
        "tests.glm5_2_graph.gpu_diagnostic_capture"
    )


def test_gpu_silu_control_compares_ordered_step1_gradients(tmp_path: Path) -> None:
    def write_trace(path: Path, *, divergent: bool) -> None:
        rows = []
        for index, (step, name, kind) in enumerate(SILU_CONTROL_KEYS):
            changed = divergent and index >= 3
            rows.append(
                {
                    "step": step,
                    "name": name,
                    "kind": kind,
                    "sha256": f"{name}-{kind}-{'different' if changed else 'same'}",
                    "mean": float(index) + (0.25 if changed else 0.0),
                    "max_abs": float(index + 1),
                    "l2": float(index + 2) + (0.5 if changed else 0.0),
                }
            )
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )

    paths = {
        name: tmp_path / f"{name}.jsonl"
        for name in ("eager-r1", "eager-r2", "inductor-r1", "inductor-r2")
    }
    write_trace(paths["eager-r1"], divergent=False)
    write_trace(paths["eager-r2"], divergent=False)
    write_trace(paths["inductor-r1"], divergent=True)
    write_trace(paths["inductor-r2"], divergent=True)
    result = compare_control_traces(paths)
    assert result["comparisons"][0]["exact_records"] == len(SILU_CONTROL_KEYS)
    assert result["comparisons"][1]["exact_records"] == len(SILU_CONTROL_KEYS)
    rows = result["comparisons"][2]["records"]
    assert rows[2]["exact"]
    assert rows[3]["name"] == "parameter.feed_forward.w2.weight"
    assert not rows[3]["exact"]


def test_gpu_silu_staged_control_defines_progressive_boundaries() -> None:
    assert MATERIALIZATION_POINTS == {"w1", "w3", "silu", "product", "down"}
    assert GPU_SILU_VARIANT_POINTS == {
        "silu-product": "silu,product",
        "silu-product-down": "silu,product,down",
        "all": "w1,w3,silu,product,down",
    }
    assert GPU_SILU_STAGED_CONFIG.training.steps == 1
    expected = "silu,product"
    assert GPU_SILU_STAGED_CONFIG.reference.environment[
        "GLM5_GPU_DENSE_SILU_MATERIALIZATION_POINTS"
    ] == expected
    assert GPU_SILU_STAGED_CONFIG.candidate.environment[
        "GLM5_GPU_DENSE_SILU_MATERIALIZATION_POINTS"
    ] == expected


def test_gpu_minimal_comparison_reports_cold_compile_reproducibility(
    tmp_path: Path,
) -> None:
    metadata = {
        "type": "metadata",
        "dtype": "bf16",
        "allow_tf32_matmul": False,
    }

    def write_result(path: Path, *, compiled_digest: str) -> None:
        result = {
            "name": "silu_multiply",
            "exact": False,
            "mismatch_count": 3,
            "max_abs_diff": 0.125,
            "eager_sha256": "eager",
            "inductor_sha256": compiled_digest,
        }
        path.write_text(
            json.dumps(metadata) + "\n" + json.dumps(result) + "\n",
            encoding="utf-8",
        )

    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    write_result(first, compiled_digest="compiled")
    write_result(second, compiled_digest="compiled")

    result = compare_minimal_results(first, second)

    assert result["dtype"] == "bf16"
    assert result["results"][0]["repeat_1_mismatch_count"] == 3
    assert result["results"][0]["cold_eager_reproducible"]
    assert result["results"][0]["cold_inductor_reproducible"]


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
