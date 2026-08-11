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
    assert result.actual_summary == "p1: topk=[0, 0]"
    assert result.expected_summary == "p1: topk=[0, 0]"
    assert (
        "1: b1,p1 actual topk=[0, 0] expected topk=[0, 1]"
        in recorder.table()
    )


def test_indexer_score_replay_uses_captured_mixed_precision_boundaries() -> None:
    scores = glm5_parity._replay_titan_indexer_scores(
        q_projection=torch.tensor([[[0.0, 2.0], [0.0, 4.0]]]),
        q_rotated=torch.tensor([[[[1.0]], [[3.0]]]]),
        k_normalized=torch.tensor([[[0.0, 6.0], [0.0, 8.0]]]),
        k_rotated=torch.tensor([[[[5.0]], [[7.0]]]]),
        projected_weights=torch.ones((1, 2, 1)),
        attention_mask=torch.zeros((1, 2, 2)),
        n_heads=1,
        head_dim=2,
        rope_head_dim=1,
    )
    expected = torch.tensor([[[17.0, 23.0], [39.0, 53.0]]]) / (2.0**0.5)
    torch.testing.assert_close(scores, expected)


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
    assert ".report-toc{position:sticky" not in html
    assert ".parity-table thead th{position:sticky" in html
    assert "class='configuration-scroll'" in html
    assert "thead th{position:sticky" in html
    assert "data-column-group='metric'" in html
    assert "aria-pressed='false'" in html
    assert "button[aria-pressed='true']" in html
    assert "hide-metric" in html
    assert "class='parity-table-scroll'" in html
    assert ".error-chart-legend{display:grid" in html
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
