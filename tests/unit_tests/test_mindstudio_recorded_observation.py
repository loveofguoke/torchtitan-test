import json
from types import SimpleNamespace

import pytest

from tests.glm5_ascend040.test_gpu_replay import evidence
from tests.glm5_2_mindstudio import recorded_observation as workflow
from tests.glm5_2_mindstudio.recorded_training import sha256


@pytest.fixture
def imported(tmp_path):
    source = tmp_path / "mindstudio_runs/accuracy/example/recorded_source"
    inputs = source / "steps-500/inputs"
    inputs.mkdir(parents=True)
    (source / "initial-model.pt").write_bytes(b"fixture")
    rows = []
    for index in range(1, 501):
        name = f"microbatch-{index:06d}.pt"
        (inputs / name).write_bytes(b"batch")
        rows.append({"microbatch": index, "file": name, "sha256": sha256(inputs / name)})
    (inputs / "manifest.jsonl").write_text("\n".join(map(json.dumps, rows)) + "\n")
    record = evidence()
    record.update(fixture_sha256=sha256(source / "initial-model.pt"),
                  input_manifest_sha256=sha256(inputs / "manifest.jsonl"))
    (source / "steps-500/rank-0.json").write_text(json.dumps(record))
    (source / "steps-500/rank-0-config.txt").write_text("original config")
    workflow.import_fixture(tmp_path, source)
    return tmp_path, source


def args(source, action):
    return SimpleNamespace(recorded_source=source, data=action == "data",
                           capture="reference" if action == "capture" else None,
                           compare=action == "compare")


def fake_capture(options):
    value = json.loads((options.reference / "steps-500/rank-0.json").read_text())
    value["runtime"] = "cuda"
    value["replayed_microbatches"] = 500
    destination = options.output / "steps-500"
    destination.mkdir()
    (destination / "rank-0.json").write_text(json.dumps(value))
    (destination / "rank-0-config.txt").write_text("original config")


def test_import_reuse_capture_skip_and_existing_ms_report(imported, monkeypatch):
    root, source = imported
    workflow.import_fixture(root, source)
    calls = []
    def capture(options):
        calls.append(options)
        fake_capture(options)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(workflow, "run", capture)
    workflow.dispatch(root, args(source, "capture"))
    workflow.dispatch(root, args(source, "capture"))
    assert len(calls) == 1
    workflow.dispatch(root, args(source, "compare"))
    reports = list((root / "mindstudio_reports").rglob("training_observation.html"))
    assert len(reports) == 1
    assert list((root / "mindstudio_reports").rglob("single.html"))
    summary = json.loads(reports[0].with_name("summary.json").read_text())
    assert summary["step_count"] == 500
    assert summary["loss"]["mean_relative_error"] == 0


def test_failed_capture_retries_and_preserves_failure(imported, monkeypatch):
    root, source = imported
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    def fail(options):
        raise RuntimeError("synthetic worker failure")
    monkeypatch.setattr(workflow, "run", fail)
    with pytest.raises(RuntimeError, match="synthetic"):
        workflow.dispatch(root, args(source, "capture"))
    monkeypatch.setattr(workflow, "run", fake_capture)
    workflow.dispatch(root, args(source, "capture"))
    assert list((root / "mindstudio_runs").rglob(".reference-r1.previous-*"))
    logs = list((root / "mindstudio_runs").rglob("runtime.log"))
    assert any("synthetic worker failure" in p.read_text() for p in logs)


def test_fixture_mutation_and_corrupt_artifact_rejected(imported):
    root, source = imported
    _, fixture, manifest, _ = workflow.resolve_fixture(root, source)
    artifact = workflow.artifact_location(root, "example", manifest["profile"], "candidate")
    (artifact / "official/training_metrics.jsonl").write_text("corrupt")
    with pytest.raises(ValueError, match="checksum mismatch"):
        workflow.validate_artifact(artifact, manifest["identity"])
    (fixture / "source/initial-model.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="fixture checksum"):
        workflow.resolve_fixture(root, source)


def test_active_capture_refuses_retry(imported):
    root, source = imported
    _, _, manifest, _ = workflow.resolve_fixture(root, source)
    run = root / "mindstudio_runs/accuracy/example/single" / manifest["profile"] / "observations/training/reference-r1"
    attempt = workflow.RunAttempt.start(run, kind="test")
    try:
        with pytest.raises(RuntimeError):
            workflow.dispatch(root, args(source, "capture"))
    finally:
        attempt.update("failed")
