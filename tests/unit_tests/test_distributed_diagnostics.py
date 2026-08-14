from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from distributed.artifacts import file_digest, write_json
from distributed.diagnostics.ddp_step import (
    FIXED_BATCH_SCHEMA,
    FIXED_BATCH_SCHEMA_VERSION,
    _metric,
    compare,
    validate_fixed_inputs,
    verify_captured_inputs,
)
from distributed.diagnostics.runtime import _summary, _trace_module, _trace_parameter


def _capture(
    root: Path,
    *,
    backend: str,
    tensors: dict[str, torch.Tensor],
    kinds: dict[str, str],
) -> None:
    root.mkdir()
    save_file(tensors, root / "rank-000.safetensors")
    write_json(
        root / "rank-000.json",
        {
            "status": "success",
            "rank": 0,
            "world_size": 1,
            "records": [
                {"key": key, "kind": kinds[key]} for key in tensors
            ],
            "tensor_file": "rank-000.safetensors",
        },
    )
    write_json(
        root / "capture.json",
        {
            "backend": backend,
            "world_size": 1,
            "scenario_digest": "scenario",
            "fixture_digest": "fixture",
            "fixed_inputs_digest": "fixed-inputs",
        },
    )


def _fixed_inputs(path: Path) -> dict[str, torch.Tensor]:
    tensors = {
        "input": torch.arange(16 * 128, dtype=torch.int64).reshape(16, 128),
        "positions": torch.arange(128, dtype=torch.int64).repeat(16, 1),
        "labels": torch.arange(16 * 128, dtype=torch.int64).reshape(16, 128) + 1,
    }
    save_file(
        tensors,
        path,
        metadata={
            "steps": "1",
            "global_batch_size": "16",
            "sequence_length": "128",
        },
    )
    write_json(
        Path(f"{path}.json"),
        {
            "schema": FIXED_BATCH_SCHEMA,
            "schema_version": FIXED_BATCH_SCHEMA_VERSION,
            "scenario_digest": "scenario",
            "fixture_digest": "fixture",
            "sha256": file_digest(path),
            "size": path.stat().st_size,
            "steps": 1,
            "global_batch_size": 16,
            "local_batch_size": 4,
            "sequence_length": 128,
            "world_size": 4,
        },
    )
    return tensors


def test_metric_requires_exact_discrete_values() -> None:
    passed = _metric(torch.tensor([1, 2]), torch.tensor([1, 2]))
    failed = _metric(torch.tensor([1, 2]), torch.tensor([1, 3]))
    assert passed == {"shape_match": True, "exact": True, "passed": True}
    assert failed == {"shape_match": True, "exact": False, "passed": False}


def test_scalar_summary_has_stable_bytes() -> None:
    summary = _summary(torch.tensor(1.25))
    assert summary["shape"] == []
    assert summary["numel"] == 1
    assert summary["sample"] == [1.25]
    assert len(summary["sha256"]) == 64


def test_compare_reports_first_observation_failure(tmp_path: Path) -> None:
    kinds = {
        "input.tokens": "input",
        "forward.part0.layers.0.tensor": "activation",
        "forward.part0.layers.1.tensor": "activation",
    }
    reference = {
        "input.tokens": torch.tensor([1, 2]),
        "forward.part0.layers.0.tensor": torch.tensor([1.0, 2.0]),
        "forward.part0.layers.1.tensor": torch.tensor([3.0, 4.0]),
    }
    candidate = {
        **reference,
        "forward.part0.layers.0.tensor": torch.tensor([1.0, 2.01]),
    }
    _capture(tmp_path / "gpu", backend="gpu", tensors=reference, kinds=kinds)
    _capture(tmp_path / "npu", backend="npu", tensors=candidate, kinds=kinds)

    report = tmp_path / "report.json"
    assert not compare(gpu=tmp_path / "gpu", npu=tmp_path / "npu", report=report)
    payload = json.loads(report.read_text())
    assert payload["first_failure"]["key"] == "forward.part0.layers.0.tensor"
    assert payload["first_failure"]["max_abs"] == pytest.approx(0.01)
    assert report.with_suffix(".md").is_file()


def test_compare_rejects_different_fixture(tmp_path: Path) -> None:
    tensors = {"input.tokens": torch.tensor([1, 2])}
    kinds = {"input.tokens": "input"}
    _capture(tmp_path / "gpu", backend="gpu", tensors=tensors, kinds=kinds)
    _capture(tmp_path / "npu", backend="npu", tensors=tensors, kinds=kinds)
    payload = json.loads((tmp_path / "npu" / "capture.json").read_text())
    payload["fixture_digest"] = "different"
    write_json(tmp_path / "npu" / "capture.json", payload)

    with pytest.raises(ValueError, match="fixture_digest"):
        compare(
            gpu=tmp_path / "gpu",
            npu=tmp_path / "npu",
            report=tmp_path / "report.json",
        )


def test_validate_fixed_inputs_checks_identity_and_tensor_contract(
    tmp_path: Path,
) -> None:
    path = tmp_path / "fixed.safetensors"
    _fixed_inputs(path)

    manifest = validate_fixed_inputs(
        path, scenario_digest="scenario", fixture_digest="fixture"
    )

    assert manifest["sha256"] == file_digest(path)
    with pytest.raises(ValueError, match="different fixture"):
        validate_fixed_inputs(
            path, scenario_digest="scenario", fixture_digest="other"
        )


def test_verify_captured_inputs_requires_exact_contiguous_rank_slices(
    tmp_path: Path,
) -> None:
    fixed_path = tmp_path / "fixed.safetensors"
    fixed = _fixed_inputs(fixed_path)
    capture_dir = tmp_path / "capture"
    capture_dir.mkdir()
    mapping = {
        "input.input": "input",
        "input.positions": "positions",
        "labels": "labels",
    }
    for rank in range(4):
        start = rank * 4
        tensors = {
            capture_key: fixed[fixed_key][start : start + 4]
            for capture_key, fixed_key in mapping.items()
        }
        save_file(tensors, capture_dir / f"rank-{rank:03d}.safetensors")
        write_json(
            capture_dir / f"rank-{rank:03d}.json",
            {"tensor_file": f"rank-{rank:03d}.safetensors"},
        )

    result = verify_captured_inputs(
        capture_dir=capture_dir, fixed_inputs=fixed_path
    )

    assert result["passed"]
    assert len(result["rows"]) == 12
    assert result["rows"][3]["global_rows"] == [4, 8]

    rank_one = load_file(capture_dir / "rank-001.safetensors")
    rank_one["input.input"][0, 0] += 1
    save_file(rank_one, capture_dir / "rank-001.safetensors")
    with pytest.raises(RuntimeError, match="differs from fixed global rows"):
        verify_captured_inputs(capture_dir=capture_dir, fixed_inputs=fixed_path)


def test_trace_selection_covers_glm_dataflow() -> None:
    assert _trace_module("tok_embeddings")
    assert _trace_module("layers.0")
    assert _trace_module("layers.1.attention")
    assert _trace_module("layers.1.attention.indexer")
    assert _trace_module("layers.1.moe.router")
    assert not _trace_module("layers.1.attention.wq_a")
    assert _trace_parameter("layers.0.attention.wq_a.weight")
    assert _trace_parameter("layers.1.moe.router.gate.weight")
    assert not _trace_parameter("layers.2.attention.wq_a.weight")
