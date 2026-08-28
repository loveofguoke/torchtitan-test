# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""CPU regression tests for portable GLM-5.2 numerical parity artifacts."""

from pathlib import Path

import pytest
import torch

import tests.unit_tests.test_glm5_parity as glm5_parity
from tests.glm5_2_parity.artifacts import (
    EndpointIdentity,
    ObservationMetadata,
    ParityArtifactError,
    ParityArtifactReader,
    ParityArtifactWriter,
)
from tests.glm5_2_parity.workflow import (
    _capture_options,
    _configure_capture_environment,
    OfflineEndpointConfig,
    ParityModelConfig,
)


def _writer(
    path: Path,
    *,
    fixture_digest: str = "fixture",
    seed: int = 17,
) -> ParityArtifactWriter:
    return ParityArtifactWriter(
        path,
        suite="unit",
        suite_version=1,
        endpoint=EndpointIdentity(
            implementation="native",
            precision="bf16",
            device_type="cpu",
            device_name="cpu",
            run_id=path.name,
        ),
        test_plan={"cases": [{"id": "forward", "ordinal": 0}]},
        configuration={"layers": [0], "seed": seed},
        fixture_digest=fixture_digest,
        environment={"torch": torch.__version__},
        shard_size_bytes=8,
    )


def test_parity_model_dtype_cast_preserves_complex_buffers() -> None:
    model = torch.nn.Module()
    model.weight = torch.nn.Parameter(torch.ones(2, dtype=torch.float32))
    model.register_buffer(
        "rope_cache",
        torch.polar(torch.ones(2), torch.ones(2)),
    )

    glm5_parity._cast_floating_dtype_preserving_complex_buffers(
        model, torch.bfloat16
    )

    assert model.weight.dtype is torch.bfloat16
    assert model.rope_cache.dtype is torch.complex64
    assert torch.equal(
        model.rope_cache,
        torch.polar(torch.ones(2), torch.ones(2)),
    )


def test_auto_parity_does_not_reuse_an_incidentally_loaded_npu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GLM5_PARITY_DEVICE", raising=False)
    monkeypatch.setattr(
        torch._utils, "_get_available_device_type", lambda: "npu"
    )

    with pytest.raises(RuntimeError, match="GLM5_PARITY_DEVICE=npu"):
        glm5_parity._parity_device()


def test_parity_artifact_round_trip_preserves_dtype_and_metadata(
    tmp_path: Path,
) -> None:
    path = tmp_path / "run"
    writer = _writer(path)
    writer.add(
        ObservationMetadata(
            key="forward/activation/layers.0",
            section_id="forward",
            scope="trace",
            component="block",
            layer=0,
            dtype_flow="in=bf16;out=bf16",
        ),
        torch.tensor([[1.0, -2.0]], dtype=torch.bfloat16),
    )
    writer.add(
        ObservationMetadata(
            key="forward/router/indices",
            section_id="forward",
            scope="trace",
            component="router",
            layer=0,
            value_kind="discrete",
        ),
        torch.tensor([[[1, 3]]], dtype=torch.int64),
    )
    log_path = tmp_path / "runtime.log"
    log_path.write_text("capture complete\n", encoding="utf-8")
    writer.add_attachment(log_path, name="runtime.log")
    writer.write()

    reader = ParityArtifactReader(path)
    assert reader.endpoint.precision == "bf16"
    assert reader.metadata("forward/activation/layers.0").dtype_flow == (
        "in=bf16;out=bf16"
    )
    activation = reader.tensor("forward/activation/layers.0")
    indices = reader.tensor("forward/router/indices")
    assert activation.dtype is torch.bfloat16
    assert torch.equal(activation, torch.tensor([[1.0, -2.0]], dtype=torch.bfloat16))
    assert indices.dtype is torch.int64
    assert torch.equal(indices, torch.tensor([[[1, 3]]]))
    assert len(reader.manifest["shards"]) == 2
    assert reader.manifest["attachments"][0]["file"] == "attachments/runtime.log"


def test_parity_artifact_rejects_incompatible_fixture(tmp_path: Path) -> None:
    left_path = tmp_path / "left"
    right_path = tmp_path / "right"
    _writer(left_path, fixture_digest="left").write()
    _writer(right_path, fixture_digest="right").write()

    left = ParityArtifactReader(left_path)
    right = ParityArtifactReader(right_path)
    with pytest.raises(ParityArtifactError, match="fixture_digest"):
        left.validate_compatible(right)


def test_parity_artifact_rejects_incompatible_configuration(
    tmp_path: Path,
) -> None:
    left_path = tmp_path / "left"
    right_path = tmp_path / "right"
    _writer(left_path, seed=17).write()
    _writer(right_path, seed=18).write()

    left = ParityArtifactReader(left_path)
    right = ParityArtifactReader(right_path)
    with pytest.raises(ParityArtifactError, match="configuration_digest"):
        left.validate_compatible(right)


def test_parity_artifact_rejects_incomplete_directory(tmp_path: Path) -> None:
    path = tmp_path / "incomplete"
    path.mkdir()
    (path / "manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ParityArtifactError, match="incomplete"):
        ParityArtifactReader(path)


def test_parity_artifact_rejects_corrupt_shard(tmp_path: Path) -> None:
    path = tmp_path / "run"
    writer = _writer(path)
    writer.add(
        ObservationMetadata(
            key="forward/output",
            section_id="forward",
            scope="output",
            component="output",
            layer="all",
        ),
        torch.arange(8, dtype=torch.float32),
    )
    writer.write()
    shard = next(path.glob("*.safetensors"))
    with shard.open("ab") as file:
        file.write(b"corrupt")
    with pytest.raises(ParityArtifactError, match="shard size mismatch"):
        ParityArtifactReader(path)


def test_parity_artifact_rejects_failed_run_comparison(tmp_path: Path) -> None:
    failed_path = tmp_path / "failed"
    success_path = tmp_path / "success"
    failed_writer = _writer(failed_path)
    failed_writer.mark_failed("unsupported operator")
    failed_writer.write()
    _writer(success_path).write()

    failed = ParityArtifactReader(failed_path)
    success = ParityArtifactReader(success_path)
    with pytest.raises(ParityArtifactError, match="unsuccessful"):
        failed.validate_compatible(success)


def test_glm5_offline_comparator_reads_raw_artifact_tensors(
    tmp_path: Path,
) -> None:
    actual_path = tmp_path / "actual"
    expected_path = tmp_path / "expected"
    for path, value in ((actual_path, 1.0), (expected_path, 1.0)):
        writer = _writer(path)
        writer.add(
            ObservationMetadata(
                key="forward/output/logits",
                section_id="forward",
                scope="configured",
                component="logits",
                layer="all",
            ),
            torch.tensor([value], dtype=torch.float32),
        )
        writer.write()

    actual = ParityArtifactReader(actual_path)
    expected = ParityArtifactReader(expected_path)
    recorder = glm5_parity.ParityRecorder(glm5_parity.FP32)
    suite = glm5_parity.TestGlm5Parity(
        methodName="test_configured_precision_suite"
    )
    suite._compare_artifact_observation(
        key="forward/output/logits",
        actual=actual,
        expected=expected,
        recorder=recorder,
        policy=glm5_parity.FP32,
    )
    assert len(recorder.results) == 1
    assert recorder.results[0].passed


def test_glm5_offline_comparator_ignores_router_tuple_branches() -> None:
    metadata = ObservationMetadata(
        key="component-router/activation/layers.1.moe.router[0]",
        section_id="component-router",
        scope="trace",
        component="router[0]",
        layer=1,
    )
    assert not glm5_parity.TestGlm5Parity._artifact_observation_is_comparable(
        metadata.key,
        metadata,
    )
    exact = ObservationMetadata(
        key="component-router/exact/layers.1.moe.router.indices",
        section_id="component-router",
        scope="component",
        component="router_indices",
        layer=1,
        value_kind="discrete",
    )
    assert glm5_parity.TestGlm5Parity._artifact_observation_is_comparable(
        exact.key,
        exact,
    )


def test_discrete_mismatch_column_reports_every_batch_and_topk() -> None:
    actual = torch.zeros((2, 2, 2), dtype=torch.int32)
    expected = actual.clone()
    expected[1, 1] = torch.tensor([0, 1], dtype=torch.int32)
    positions = torch.tensor([[0, 1], [0, 1]], dtype=torch.int64)
    recorder = glm5_parity.ParityRecorder(glm5_parity.BF16)

    result = recorder.discrete(
        scope="component",
        component="indexer",
        layer=2,
        actual=actual,
        expected=expected,
        positions=positions,
    )

    assert not result.passed
    assert result.mismatch_count == 1
    assert result.mismatch_positions == [1]
    assert result.actual_summary == (
        "Batch 1, query position 1: topk=[0, 0]"
    )
    assert result.expected_summary == (
        "Batch 1, query position 1: topk=[0, 1]"
    )
    assert (
        "1 mismatch"
        in recorder.table()
    )
    assert "Batch 1, query position 1" in recorder.table()
    assert "Actual topk: [0, 0]" in recorder.table()
    assert "Expected topk: [0, 1]" in recorder.table()


def test_token_first_discrete_results_use_fixture_batch_coordinates() -> None:
    recorder = glm5_parity.ParityRecorder(glm5_parity.BF16)
    positions = torch.tensor([[0, 1], [0, 1]])
    actual = torch.tensor([[0], [1], [2], [3]], dtype=torch.int32)
    expected = torch.tensor([[0], [1], [2], [2]], dtype=torch.int32)

    result = recorder.discrete(
        scope="configured",
        component="indexer",
        layer=1,
        actual=actual,
        expected=expected,
        positions=positions,
    )

    assert result.mismatch_count == 1
    assert result.mismatch_coordinates == [(1, 1)]
    assert "Batch 1, query position 1" in result.detail


@pytest.mark.parametrize("component", ["indexer", "router"])
def test_bf16_topk_cutoff_difference_is_boundary_pass(
    component: str,
) -> None:
    recorder = glm5_parity.ParityRecorder(glm5_parity.BF16)
    actual_topk = torch.tensor([[[1, 0]]], dtype=torch.int32)
    expected_topk = torch.tensor([[[1, 2]]], dtype=torch.int32)
    positions = torch.tensor([[2]]) if component == "indexer" else None
    result = recorder.discrete(
        scope="configured",
        component=component,
        layer=2,
        actual=actual_topk,
        expected=expected_topk,
        positions=positions,
    )

    recorder.annotate_topk_scores(
        result,
        actual_topk=actual_topk,
        expected_topk=expected_topk,
        actual_scores=torch.tensor([[[0.9999, 1.1, 0.9998]]]),
        expected_scores=torch.tensor([[[0.9997, 1.1, 0.9999]]]),
        positions=positions,
    )

    assert result.passed
    assert result.boundary_passed
    assert recorder._status(result) == "BOUNDARY_PASS"
    assert "boundary_passed=1" in recorder.summary()
    assert "Batch 0" in result.actual_topk_scores
    assert "Query position 0: [index 1: 1.1, index 0:" in (
        result.actual_topk_scores
    )
    assert "Query position 0: [index 1: 1.1, index 2:" in (
        result.expected_topk_scores
    )
    assert "1 boundary-pass; 0 hard mismatch" in result.precision_band
    assert "BF16 ULP at cutoff scale:" in result.precision_band
    assert "One-value BF16 rounding interval:" in result.precision_band
    assert "ULP role: diagnostic only" in result.precision_band
    assert "Complete score row passes BF16 tolerance: Yes" in (
        result.precision_band
    )
    assert "Observed one-score instability interval:" in result.precision_band
    assert "Scores reproduce recorded top-k: Yes" in result.precision_band
    assert "Index 0 (selected only by Actual)" in result.detail
    assert "Index 2 (selected only by Expected)" in result.detail
    assert "Explanation:" in result.detail
    html = recorder.html_section("topk")
    assert "How to read BF16 top-k results" in html
    assert "View Actual top-k scores" in html
    assert "1 differing query position" in html
    assert "Direct same-layer output: NOT CAPTURED" in html


@pytest.mark.parametrize("precision", [glm5_parity.FP32, glm5_parity.BF16])
def test_topk_difference_outside_boundary_band_fails(
    precision: glm5_parity.PrecisionPolicy,
) -> None:
    recorder = glm5_parity.ParityRecorder(precision)
    actual_topk = torch.tensor([[[0, 1]]], dtype=torch.int32)
    expected_topk = torch.tensor([[[2, 1]]], dtype=torch.int32)
    result = recorder.discrete(
        scope="configured",
        component="router",
        layer=1,
        actual=actual_topk,
        expected=expected_topk,
    )

    recorder.annotate_topk_scores(
        result,
        actual_topk=actual_topk,
        expected_topk=expected_topk,
        actual_scores=torch.tensor([[[1.2, 1.1, 0.5]]]),
        expected_scores=torch.tensor([[[0.5, 1.1, 1.2]]]),
    )

    assert not result.passed
    assert not result.boundary_passed
    assert recorder._status(result) == "FAIL"
    assert "0 boundary-pass; 1 hard mismatch" in result.precision_band
    assert "Classification: Hard mismatch" in result.detail


def test_fp32_topk_cutoff_difference_remains_strict() -> None:
    recorder = glm5_parity.ParityRecorder(glm5_parity.FP32)
    actual_topk = torch.tensor([[[1, 0]]], dtype=torch.int32)
    expected_topk = torch.tensor([[[1, 2]]], dtype=torch.int32)
    result = recorder.discrete(
        scope="configured",
        component="indexer",
        layer=2,
        actual=actual_topk,
        expected=expected_topk,
        positions=torch.tensor([[2]]),
    )

    recorder.annotate_topk_scores(
        result,
        actual_topk=actual_topk,
        expected_topk=expected_topk,
        actual_scores=torch.tensor([[[0.9999, 1.1, 0.9998]]]),
        expected_scores=torch.tensor([[[0.9997, 1.1, 0.9999]]]),
        positions=torch.tensor([[2]]),
    )

    assert not result.passed
    assert not result.boundary_passed
    assert recorder._status(result) == "FAIL"


def test_bf16_ulp_tracks_value_scale() -> None:
    assert glm5_parity._bf16_ulp(1.0) == 2.0**-7
    assert glm5_parity._bf16_ulp(0.5) == 2.0**-8
    assert glm5_parity._bf16_ulp(-0.0184) == 2.0**-13


def test_accumulated_score_error_over_two_ulps_can_be_boundary_pass() -> None:
    recorder = glm5_parity.ParityRecorder(glm5_parity.BF16)
    actual_topk = torch.tensor([[[1, 0]]], dtype=torch.int32)
    expected_topk = torch.tensor([[[1, 2]]], dtype=torch.int32)
    result = recorder.discrete(
        scope="configured",
        component="indexer",
        layer=2,
        actual=actual_topk,
        expected=expected_topk,
        positions=torch.tensor([[2]]),
    )

    recorder.annotate_topk_scores(
        result,
        actual_topk=actual_topk,
        expected_topk=expected_topk,
        actual_scores=torch.tensor([[[0.01802, 0.03, 0.01790]]]),
        expected_scores=torch.tensor([[[0.01800, 0.03, 0.01815]]]),
        positions=torch.tensor([[2]]),
    )

    assert result.boundary_local_passed
    assert result.passed
    assert "Difference in ULPs: 2." in result.detail
    assert "ULP values are diagnostic only" in result.detail
    assert "Pairwise ranking uncertainty" in result.precision_band


def _boundary_result_with_downstream(
    *,
    downstream_passed: bool,
) -> tuple[glm5_parity.ParityRecorder, glm5_parity.ComparisonResult]:
    recorder = glm5_parity.ParityRecorder(glm5_parity.BF16)
    actual_topk = torch.tensor([[[1, 0]]], dtype=torch.int32)
    expected_topk = torch.tensor([[[1, 2]]], dtype=torch.int32)
    boundary = recorder.discrete(
        scope="configured",
        component="indexer",
        layer=2,
        actual=actual_topk,
        expected=expected_topk,
        positions=torch.tensor([[2]]),
        module_path="layers.2.attention.indexer",
    )
    recorder.annotate_topk_scores(
        boundary,
        actual_topk=actual_topk,
        expected_topk=expected_topk,
        actual_scores=torch.tensor([[[0.9999, 1.1, 0.9998]]]),
        expected_scores=torch.tensor([[[0.9997, 1.1, 0.9999]]]),
        positions=torch.tensor([[2]]),
    )
    recorder.tensor(
        scope="configured",
        component="attention",
        layer=2,
        actual=torch.tensor([0.0]),
        expected=(
            torch.tensor([0.0])
            if downstream_passed
            else torch.tensor([1.0])
        ),
        module_path="layers.2.attention",
    )
    recorder.finalize_topk_boundary_verdicts()
    return recorder, boundary


def test_boundary_pass_requires_acceptable_downstream_checkpoints() -> None:
    recorder, boundary = _boundary_result_with_downstream(
        downstream_passed=True
    )

    assert boundary.passed
    assert boundary.boundary_passed
    assert boundary.boundary_propagation_checked
    assert "Direct same-layer output: ACCEPTABLE" in boundary.precision_band
    assert "Final boundary verdict" in recorder._mismatch_display(boundary)
    assert "Final decision: BOUNDARY_PASS" in (
        recorder._mismatch_display(boundary)
    )
    assert recorder._status(boundary) == "BOUNDARY_PASS"


def test_boundary_change_fails_when_downstream_checkpoint_fails() -> None:
    recorder, boundary = _boundary_result_with_downstream(
        downstream_passed=False
    )

    assert not boundary.passed
    assert not boundary.boundary_passed
    assert boundary.boundary_propagation_checked
    assert "Direct same-layer output: NOT ACCEPTABLE" in (
        boundary.precision_band
    )
    assert "layers.2.attention" in boundary.precision_band
    assert "Final decision: FAIL" in recorder._mismatch_display(boundary)
    assert recorder._status(boundary) == "FAIL"


def test_router_boundary_checks_same_layer_moe_output() -> None:
    recorder = glm5_parity.ParityRecorder(glm5_parity.BF16)
    actual_topk = torch.tensor([[[1, 0]]], dtype=torch.int32)
    expected_topk = torch.tensor([[[1, 2]]], dtype=torch.int32)
    boundary = recorder.discrete(
        scope="configured",
        component="router",
        layer=2,
        actual=actual_topk,
        expected=expected_topk,
        module_path="layers.2.moe.router",
    )
    recorder.annotate_topk_scores(
        boundary,
        actual_topk=actual_topk,
        expected_topk=expected_topk,
        actual_scores=torch.tensor([[[0.9999, 1.1, 0.9998]]]),
        expected_scores=torch.tensor([[[0.9997, 1.1, 0.9999]]]),
    )
    recorder.tensor(
        scope="trace",
        component="moe",
        layer=2,
        actual=torch.tensor([[[1.0]]]),
        expected=torch.zeros((1, 1, 1)),
        module_path="layers.2.moe",
    )
    recorder.finalize_topk_boundary_verdicts()

    assert not boundary.passed
    assert not boundary.boundary_passed
    assert "layers.2.moe" in boundary.boundary_propagation


def test_in_tolerance_growth_is_diagnostic_for_boundary_change() -> None:
    recorder, boundary = _boundary_result_with_downstream(
        downstream_passed=True
    )
    recorder.tensor(
        scope="configured",
        component="attention",
        layer=2,
        actual=torch.tensor([1.01]),
        expected=torch.tensor([1.0]),
        module_path="layers.2.attention",
    )
    recorder.finalize_topk_boundary_verdicts()

    assert boundary.passed
    assert boundary.boundary_passed
    assert "diagnostic only" in boundary.precision_band


def test_boundary_does_not_claim_unrelated_batch_failure() -> None:
    recorder, boundary = _boundary_result_with_downstream(
        downstream_passed=True
    )
    recorder.tensor(
        scope="configured",
        component="attention",
        layer=2,
        actual=torch.tensor([[[0.0]], [[1.0]]]),
        expected=torch.zeros((2, 1, 1)),
        module_path="layers.2.attention",
    )
    recorder.finalize_topk_boundary_verdicts()

    assert boundary.passed
    assert boundary.boundary_passed
    assert "other sequence positions" in boundary.precision_band


def test_boundary_ignores_failures_after_the_direct_module_output() -> None:
    recorder, boundary = _boundary_result_with_downstream(
        downstream_passed=True
    )
    recorder.tensor(
        scope="configured",
        component="decoder_block",
        layer=3,
        actual=torch.tensor([1.0]),
        expected=torch.tensor([0.0]),
        module_path="layers.3",
    )
    recorder.finalize_topk_boundary_verdicts()

    assert boundary.passed
    assert boundary.boundary_passed
    assert "Direct same-layer output: ACCEPTABLE" in (
        boundary.precision_band
    )


def test_report_rows_follow_glm_dataflow_not_path_hierarchy() -> None:
    recorder = glm5_parity.ParityRecorder(glm5_parity.FP32)

    def record(path: str, component: str) -> None:
        recorder.tensor(
            scope="trace",
            component=component,
            layer=0,
            actual=torch.zeros(1),
            expected=torch.zeros(1),
            module_path=path,
        )

    # Intentionally insert parent rows before their children.
    record("layers.0", "decoder_block")
    record("layers.0.feed_forward", "feed_forward")
    record("layers.0.attention", "attention")
    record("layers.0.attention.indexer", "indexer_scores")
    record("layers.0.feed_forward.w2", "w2")
    record("layers.0.ffn_norm", "ffn_norm")
    record("layers.0.attention.indexer.weights_proj", "weights_proj")
    record("layers.0.attention.q_norm", "q_norm")
    record("layers.0.attention_norm", "attention_norm")

    paths = [
        recorder._display_path(result)
        for result in recorder.ordered_results()
    ]
    assert paths == [
        "layers.0.attention_norm",
        "layers.0.attention.q_norm",
        "layers.0.attention.indexer.weights_proj",
        "layers.0.attention.indexer.scores",
        "layers.0.attention",
        "layers.0.ffn_norm",
        "layers.0.feed_forward.w2",
        "layers.0.feed_forward",
        "layers.0",
    ]


def test_numeric_trend_uses_major_dataflow_checkpoints_only() -> None:
    recorder = glm5_parity.ParityRecorder(glm5_parity.FP32)
    rows = []
    for index, (path, component) in enumerate(
        (
            ("layers.0.attention_norm", "attention_norm"),
            ("layers.0.attention.q_norm", "q_norm"),
            ("layers.0.attention", "attention"),
            ("layers.0.ffn_norm", "ffn_norm"),
            ("layers.0.feed_forward", "feed_forward"),
            ("layers.0", "decoder_block"),
            ("layers.1.attention_norm", "attention_norm"),
        ),
        start=1,
    ):
        rows.append(
            recorder.tensor(
                scope="trace",
                component=component,
                layer=int(path.split(".")[1]),
                actual=torch.tensor([index * 1.0e-5]),
                expected=torch.zeros(1),
                module_path=path,
            )
        )

    recorder.ordered_results()

    by_path = {recorder._display_path(row): row for row in rows}
    assert by_path["layers.0.attention"].trend_from == (
        "layers.0.attention_norm"
    )
    assert by_path["layers.0.attention.q_norm"].trend_from == ""
    assert by_path["layers.0"].trend_from == "layers.0.feed_forward"
    assert by_path["layers.1.attention_norm"].trend_from == "layers.0"
    assert "from layers.0" in recorder._trend_display(
        by_path["layers.1.attention_norm"]
    )


def test_abrupt_growth_does_not_change_numeric_pass_fail() -> None:
    recorder = glm5_parity.ParityRecorder(glm5_parity.FP32)
    first = recorder.tensor(
        scope="trace",
        component="attention_norm",
        layer=0,
        actual=torch.tensor([1.0000001]),
        expected=torch.ones(1),
        module_path="layers.0.attention_norm",
    )
    second = recorder.tensor(
        scope="trace",
        component="attention",
        layer=0,
        actual=torch.tensor([1.00001]),
        expected=torch.ones(1),
        module_path="layers.0.attention",
    )
    recorder.ordered_results()

    assert first.passed
    assert second.passed
    assert second.explosion
    assert recorder._status(second) == "PASS"
    assert "ABRUPT GROWTH" in recorder._trend_display(second)


def test_activation_gradient_rows_follow_reverse_dataflow() -> None:
    recorder = glm5_parity.ParityRecorder(glm5_parity.FP32)
    for path, component, layer in (
        ("layers.0.attention_norm", "attention_norm", 0),
        ("layers.0", "decoder_block", 0),
        ("layers.1.attention", "attention", 1),
        ("layers.1", "decoder_block", 1),
    ):
        recorder.tensor(
            scope="activation_gradient",
            component=component,
            layer=layer,
            actual=torch.zeros(1),
            expected=torch.zeros(1),
            module_path=path,
        )

    assert [
        recorder._display_path(result)
        for result in recorder.ordered_results()
    ] == [
        "layers.1",
        "layers.1.attention",
        "layers.0",
        "layers.0.attention_norm",
    ]


def test_legacy_activation_gradient_ties_use_reverse_natural_order() -> None:
    recorder = glm5_parity.ParityRecorder(glm5_parity.FP32)
    for output_index in (0, 1):
        recorder.tensor(
            scope="activation_gradient",
            component="rope",
            layer=0,
            actual=torch.zeros(1),
            expected=torch.zeros(1),
            module_path=f"layers.0.attention.rope[{output_index}]",
        )

    assert [
        recorder._display_path(result)
        for result in recorder.ordered_results()
    ] == [
        "layers.0.attention.rope[1]",
        "layers.0.attention.rope[0]",
    ]


def test_captured_order_breaks_ties_inside_one_semantic_stage() -> None:
    recorder = glm5_parity.ParityRecorder(glm5_parity.FP32)
    for output_index, execution_order in ((0, 8), (1, 7)):
        recorder.tensor(
            scope="trace",
            component="rope",
            layer=0,
            actual=torch.zeros(1),
            expected=torch.zeros(1),
            module_path=f"layers.0.attention.rope[{output_index}]",
            execution_order=execution_order,
            execution_order_source="captured forward hook",
        )

    assert [
        recorder._display_path(result)
        for result in recorder.ordered_results()
    ] == [
        "layers.0.attention.rope[1]",
        "layers.0.attention.rope[0]",
    ]


def test_matching_topk_still_reports_every_selected_score() -> None:
    recorder = glm5_parity.ParityRecorder(glm5_parity.BF16)
    topk = torch.tensor([[[2, 0], [1, 2]]], dtype=torch.int32)
    scores = torch.tensor([[[0.7, 0.2, 0.9], [0.1, 0.8, 0.6]]])
    result = recorder.discrete(
        scope="configured",
        component="router",
        layer=1,
        actual=topk,
        expected=topk,
    )

    recorder.annotate_topk_scores(
        result,
        actual_topk=topk,
        expected_topk=topk,
        actual_scores=scores,
        expected_scores=scores,
    )

    assert result.passed
    assert not result.boundary_passed
    assert "Query position 0: [index 2:" in result.actual_topk_scores
    assert ", index 0:" in result.actual_topk_scores
    assert "Query position 1: [index 1:" in result.actual_topk_scores
    assert ", index 2:" in result.actual_topk_scores
    assert "2 query positions; 2 exact; 0 boundary-pass" in (
        result.precision_band
    )


def test_indexer_score_replay_uses_captured_mixed_precision_boundaries() -> None:
    scores = glm5_parity._replay_titan_indexer_scores(
        q_projection=torch.tensor([[0.0, 2.0], [0.0, 4.0]]),
        q_rotated=torch.tensor([[[1.0]], [[3.0]]]),
        k_normalized=torch.tensor([[0.0, 6.0], [0.0, 8.0]]),
        k_rotated=torch.tensor([[[5.0]], [[7.0]]]),
        projected_weights=torch.ones((2, 1)),
        attention_mask=torch.zeros((2, 2)),
        n_heads=1,
        head_dim=2,
        rope_head_dim=1,
    )
    expected = torch.tensor([[17.0, 23.0], [39.0, 53.0]]) / (2.0**0.5)
    torch.testing.assert_close(scores, expected)


def test_indexer_score_replay_accepts_structured_token_first_projection() -> None:
    scores = glm5_parity._replay_titan_indexer_scores(
        q_projection=torch.tensor([[[0.0, 2.0]], [[0.0, 4.0]]]),
        q_rotated=torch.tensor([[[1.0]], [[3.0]]]),
        k_normalized=torch.tensor([[0.0, 6.0], [0.0, 8.0]]),
        k_rotated=torch.tensor([[[5.0]], [[7.0]]]),
        projected_weights=torch.ones((2, 1)),
        attention_mask=torch.zeros((2, 2)),
        n_heads=1,
        head_dim=2,
        rope_head_dim=1,
    )
    expected = torch.tensor([[17.0, 23.0], [39.0, 53.0]]) / (2.0**0.5)
    torch.testing.assert_close(scores, expected)


def test_routed_expert_load_supports_token_first_and_legacy_batches() -> None:
    token_first = torch.tensor(
        [[True, False, True], [False, True, True]],
    )
    legacy_batches = token_first.reshape(1, 2, 3)

    expected = torch.tensor([1, 1, 2])
    assert torch.equal(glm5_parity._routed_expert_load(token_first), expected)
    assert torch.equal(glm5_parity._routed_expert_load(legacy_batches), expected)


def test_offline_score_diagnostics_prefer_captured_scores(
    tmp_path: Path,
) -> None:
    path = tmp_path / "scores"
    writer = _writer(path)
    indexer_key = (
        "end-to-end/checkpoint/layers.1.attention.indexer_scores"
    )
    router_key = "end-to-end/checkpoint/layers.1.moe.router_scores"
    for key, component, value in (
        (indexer_key, "indexer_scores", torch.tensor([[[1.0, 2.0]]])),
        (router_key, "router_scores", torch.tensor([[[0.25, 0.75]]])),
    ):
        writer.add(
            ObservationMetadata(
                key=key,
                section_id="end-to-end",
                scope="configured",
                component=component,
                layer=1,
                checkpoint=False,
                compare=False,
            ),
            value,
        )
    writer.write()
    reader = ParityArtifactReader(path)
    metadata = ObservationMetadata(
        key="end-to-end/checkpoint/layers.1.attention.indexer",
        section_id="end-to-end",
        scope="configured",
        component="indexer",
        layer=1,
        value_kind="discrete",
    )

    indexer_scores, indexer_error = (
        glm5_parity.TestGlm5Parity._replay_artifact_indexer_scores(
            reader, metadata
        )
    )
    router_scores, router_error = (
        glm5_parity.TestGlm5Parity._artifact_router_scores(reader, metadata)
    )

    assert indexer_error == ""
    assert router_error == ""
    assert torch.equal(indexer_scores, torch.tensor([[[1.0, 2.0]]]))
    assert torch.equal(router_scores, torch.tensor([[[0.25, 0.75]]]))
    assert not reader.metadata(indexer_key).compare
    assert not reader.metadata(router_key).compare


def test_legacy_indexer_score_replay_infers_geometry_from_tensors(
    tmp_path: Path,
) -> None:
    path = tmp_path / "legacy-scores"
    writer = _writer(path)
    prefix = "end-to-end/activation/layers.1.attention.indexer"
    tensors = {
        f"{prefix}.wq_b": torch.tensor([[[0.0, 2.0], [0.0, 4.0]]]),
        f"{prefix}.rope[0]": torch.tensor([[[[1.0]], [[3.0]]]]),
        f"{prefix}.k_norm": torch.tensor([[[0.0, 6.0], [0.0, 8.0]]]),
        f"{prefix}.rope[1]": torch.tensor([[[[5.0]], [[7.0]]]]),
        f"{prefix}.weights_proj": torch.ones((1, 2, 1)),
        "fixture/data/end-to-end/causal_mask": torch.zeros((1, 2, 2)),
    }
    for key, value in tensors.items():
        writer.add(
            ObservationMetadata(
                key=key,
                section_id="end-to-end",
                scope="trace",
                component="indexer",
                layer=1,
                compare=False,
            ),
            value,
        )
    writer.write()
    reader = ParityArtifactReader(path)
    metadata = ObservationMetadata(
        key="end-to-end/checkpoint/layers.1.attention.indexer",
        section_id="end-to-end",
        scope="configured",
        component="indexer",
        layer=1,
        value_kind="discrete",
    )

    scores, error = glm5_parity.TestGlm5Parity._replay_artifact_indexer_scores(
        reader, metadata
    )

    assert error == ""
    expected = torch.tensor([[[17.0, 23.0], [39.0, 53.0]]]) / (2.0**0.5)
    torch.testing.assert_close(scores, expected)


def test_endpoint_path_is_visible_without_an_hf_endpoint() -> None:
    recorder = glm5_parity.ParityRecorder(glm5_parity.BF16)
    recorder.tensor(
        scope="configured",
        component="q_norm",
        layer=3,
        actual=torch.zeros(1),
        expected=torch.zeros(1),
        module_path=(
            "titan:bf16:layers.3.attention.q_norm <-> "
            "titan:bf16:layers.3.attention.q_norm"
        ),
    )

    html = recorder.html_table()

    assert "endpoint_path" in html
    assert "<td class='col-path'>layers.3.attention.q_norm</td>" in html


def test_glm5_offline_q_residual_uses_suite_tolerance(
    tmp_path: Path,
) -> None:
    key = "component-indexer/exact/layers.0.attention.q_residual"
    for name, value in (("actual", 0.0), ("expected", 9.536743e-7)):
        writer = _writer(tmp_path / name)
        writer.add(
            ObservationMetadata(
                key=key,
                section_id="component-indexer",
                scope="component",
                component="q_residual",
                layer=0,
                tags={"rtol": "1e-6", "atol": "1e-7"},
            ),
            torch.tensor([value], dtype=torch.float32),
        )
        writer.write()

    actual = ParityArtifactReader(tmp_path / "actual")
    expected = ParityArtifactReader(tmp_path / "expected")
    recorder = glm5_parity.ParityRecorder(glm5_parity.FP32)
    suite = glm5_parity.TestGlm5Parity(
        methodName="test_configured_precision_suite"
    )
    suite._compare_artifact_observation(
        key=key,
        actual=actual,
        expected=expected,
        recorder=recorder,
        policy=glm5_parity.FP32,
    )
    assert len(recorder.results) == 1
    assert recorder.results[0].passed
    assert recorder.results[0].mismatch_count == 0


def test_glm5_2_default_parity_model_is_about_ten_fp32_gb() -> None:
    model_size = glm5_parity.ParityModelSize()
    assert model_size.num_layers == 11
    assert 10.0 <= model_size.estimated_fp32_size_gb <= 10.5


def test_glm5_2_html_contents_is_top_only_and_targets_sections(
    tmp_path: Path,
) -> None:
    recorder = glm5_parity.ParityRecorder(
        glm5_parity.FP32,
        title="Indexer and router",
    )
    report = glm5_parity.ParitySuiteReport("GLM-5.2 offline parity")
    report.add("component:indexer", recorder)
    path = tmp_path / "report.html"
    report.write(str(path))

    html = path.read_text(encoding="utf-8")
    assert html.count("class='report-toc'") == 1
    assert "nav.report-toc {" in html
    toc_css = html.split("nav.report-toc {", 1)[1].split("}", 1)[0]
    assert "position: sticky" not in toc_css
    assert "class='configuration-scroll'" in html
    assert "thead th {" in html
    table_header_css = html.split("thead th {", 1)[1].split("}", 1)[0]
    assert "position: sticky;" in table_header_css
    assert "data-column-group='metric'" in html
    assert "data-column-group='topk'" in html
    assert "aria-pressed='false'" in html
    assert 'button[aria-pressed="true"]' in html
    assert "hide-metric" in html
    assert "hide-topk" in html
    assert "Actual top-k scores" in html
    assert "td.boundary {" in html
    assert "class='parity-table-scroll'" in html
    assert ".error-chart-legend {" in html
    assert "display: grid;" in html
    assert "href='#section-0-component-indexer'" in html
    assert "<section id='section-0-component-indexer'>" in html
    assert html.index("class='report-toc'") < html.index("<section id=")


def test_glm5_2_capture_requires_an_explicit_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(glm5_parity.TestGlm5Parity, "FIXTURE_PATH", None)
    monkeypatch.setattr(
        glm5_parity.TestGlm5Parity,
        "ARTIFACT_PATH",
        str(tmp_path / "capture"),
    )
    with pytest.raises(ValueError, match="GLM5_PARITY_FIXTURE"):
        glm5_parity.TestGlm5Parity._set_up_capture_class()


def test_glm5_2_workflow_model_configuration_maps_to_environment() -> None:
    environment = ParityModelConfig(layers=7, dim=1024).environment()
    assert environment["GLM5_PARITY_MODEL_LAYERS"] == "7"
    assert environment["GLM5_PARITY_MODEL_DIM"] == "1024"


def test_glm5_2_offline_endpoint_configuration_is_device_independent(
    tmp_path: Path,
) -> None:
    endpoint = OfflineEndpointConfig(
        name="hf-gpu",
        endpoint="hf:fp32",
        device_type="cuda",
        visible_device="3",
        visible_devices_env="CUDA_VISIBLE_DEVICES",
        artifact_name="hf_capture",
    )
    assert _capture_options("expected", endpoint) == [
        "--expected-capture",
        "--hf-gpu-capture",
    ]
    environment = {
        "CUDA_VISIBLE_DEVICES": "old",
        "ASCEND_RT_VISIBLE_DEVICES": "old",
    }
    artifact = tmp_path / "hf_capture"
    _configure_capture_environment(environment, endpoint, artifact)
    assert environment["CUDA_VISIBLE_DEVICES"] == "3"
    assert environment["ASCEND_RT_VISIBLE_DEVICES"] == ""
    assert environment["GLM5_PARITY_DEVICE"] == "cuda"
    assert environment["GLM5_PARITY_ENDPOINT"] == "hf:fp32"
    assert environment["GLM5_PARITY_ARTIFACT"] == str(artifact)
