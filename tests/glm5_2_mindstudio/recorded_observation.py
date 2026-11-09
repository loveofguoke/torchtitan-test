"""Import recorded Turbo training into the existing observation stages."""

from contextlib import redirect_stderr, redirect_stdout
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace

from tests.glm5_2_common.cli import (
    RunAttempt, archive_previous_output, assert_run_not_active, print_output_path,
)
from .recorded_training import compare, read_evidence, run, sha256, validate_reference
from .training_observation import compare_training_metrics


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def fixture_location(root, source):
    experiment = source.parent.name
    return experiment, root / "mindstudio_fixtures/accuracy" / experiment / "recorded_training"


def artifact_location(root, experiment, profile, role):
    return (root / "mindstudio_artifacts/accuracy" / experiment / "single" /
            profile / "observations/training" / f"{role}-r1")


def publish_artifact(destination, evidence, config_path, identity):
    if destination.exists():
        validate_artifact(destination, identity)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent, prefix=".import-") as temporary:
        staging = Path(temporary) / "capture"
        (staging / "official").mkdir(parents=True)
        metrics = staging / "official/training_metrics.jsonl"
        with metrics.open("w", encoding="utf-8") as stream:
            for row in evidence["metrics"]:
                stream.write(json.dumps({"step": row["step"], "metrics": {
                    "loss_metrics/global_avg_loss": row["loss"],
                    "loss_metrics/global_max_loss": row["max_loss"],
                    "grad_norm": row["grad_norm"],
                }}) + "\n")
        write_json(staging / "source-evidence.json", evidence)
        shutil.copyfile(config_path, staging / "resolved-config.txt")
        write_json(staging / "manifest.json", {
            "schema": "torchtitan.mindstudio.recorded_observation.v1",
            "fixture_identity": identity,
            "origin": "imported_turbo_training" if evidence["runtime"] == "turbo" else "cuda_recorded_replay",
            "instrumentation": "none", "completed_steps": 500,
            "files": {name: sha256(staging / name) for name in (
                "official/training_metrics.jsonl", "source-evidence.json", "resolved-config.txt")},
        })
        write_json(staging / "complete.json", {"manifest_sha256": sha256(staging / "manifest.json")})
        staging.rename(destination)


def validate_artifact(path, identity):
    manifest = json.loads((path / "manifest.json").read_text())
    complete = json.loads((path / "complete.json").read_text())
    if (complete["manifest_sha256"] != sha256(path / "manifest.json") or
            manifest["fixture_identity"] != identity):
        raise ValueError("Observation artifact generation mismatch")
    for name, digest in manifest["files"].items():
        if sha256(path / name) != digest:
            raise ValueError("Observation artifact checksum mismatch: " + name)


def resolve_fixture(root, source):
    experiment, fixture = fixture_location(root, source)
    manifest = json.loads((fixture / "fixture.json").read_text())
    evidence, _, _ = validate_reference(fixture / "source")
    if sha256(fixture / "source/steps-500/rank-0.json") != manifest["source_evidence_sha256"]:
        raise ValueError("Imported reference metadata changed")
    if sha256(fixture / "source/steps-500/rank-0-config.txt") != manifest["source_config_sha256"]:
        raise ValueError("Imported reference config changed")
    if manifest["fixture_sha256"] != evidence["fixture_sha256"] or manifest["input_manifest_sha256"] != evidence["input_manifest_sha256"]:
        raise ValueError("Imported fixture changed")
    actual_identity = hashlib.sha256(json.dumps({
        "fixture": evidence["fixture_sha256"], "inputs": evidence["input_manifest_sha256"],
        "source_evidence": manifest["source_evidence_sha256"],
        "source_config": manifest["source_config_sha256"],
    }, sort_keys=True).encode()).hexdigest()
    if manifest["identity"] != actual_identity:
        raise ValueError("Imported fixture identity changed")
    return experiment, fixture, manifest, evidence


def import_fixture(root, source):
    evidence, _, _ = validate_reference(source)
    experiment, fixture = fixture_location(root, source)
    identity = hashlib.sha256(json.dumps({
        "fixture": evidence["fixture_sha256"], "inputs": evidence["input_manifest_sha256"],
        "source_evidence": sha256(source / "steps-500/rank-0.json"),
        "source_config": sha256(source / "steps-500/rank-0-config.txt"),
    }, sort_keys=True).encode()).hexdigest()
    if not fixture.exists():
        fixture.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=fixture.parent, prefix=".fixture-") as temporary:
            staging = Path(temporary) / "fixture"
            shutil.copytree(source, staging / "source")
            write_json(staging / "fixture.json", {
                "schema": "torchtitan.mindstudio.recorded_fixture.v1", "identity": identity,
                "profile": "s500-recorded-" + identity[:8], "steps": 500,
                "fixture_sha256": evidence["fixture_sha256"],
                "input_manifest_sha256": evidence["input_manifest_sha256"],
                "source_evidence_sha256": sha256(source / "steps-500/rank-0.json"),
                "source_config_sha256": sha256(source / "steps-500/rank-0-config.txt"),
            })
            staging.rename(fixture)
    _, _, manifest, _ = resolve_fixture(root, source)
    if manifest["identity"] != identity:
        raise ValueError("Existing imported fixture belongs to another generation")
    candidate = artifact_location(root, experiment, manifest["profile"], "candidate")
    publish_artifact(candidate, evidence, source / "steps-500/rank-0-config.txt", identity)
    print_output_path("Imported observation fixture", fixture)
    print_output_path("Imported NPU observation", candidate)


def dispatch(root, args):
    source = args.recorded_source.resolve()
    if args.data:
        import_fixture(root, source)
        return
    experiment, fixture, manifest, reference = resolve_fixture(root, source)
    identity, profile = manifest["identity"], manifest["profile"]
    gpu_artifact = artifact_location(root, experiment, profile, "reference")
    npu_artifact = artifact_location(root, experiment, profile, "candidate")
    run_directory = (root / "mindstudio_runs/accuracy" / experiment / "single" /
                     profile / "observations/training/reference-r1")
    if args.capture:
        if args.capture != "reference":
            raise ValueError("Recorded NPU candidate is imported by --data; capture only reference")
        assert_run_not_active(run_directory)
        if gpu_artifact.exists():
            validate_artifact(gpu_artifact, identity)
            print_output_path("Skip completed observation", gpu_artifact)
            return
        device = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
        if not device.isdecimal():
            raise ValueError("Select one GPU with CUDA_VISIBLE_DEVICES")
        archive_previous_output(run_directory)
        attempt = RunAttempt.start(run_directory, kind="recorded_observation", context={"fixture_identity": identity})
        log_path = run_directory / "runtime.log"
        print_output_path("Observation runtime log", log_path)
        try:
            with log_path.open("w", encoding="utf-8", buffering=1) as log:
                with redirect_stdout(log), redirect_stderr(log):
                    try:
                        run(SimpleNamespace(reference=fixture / "source", output=run_directory,
                                            titan=root.parent / "torchtitan", device=device))
                    except BaseException:
                        import traceback
                        traceback.print_exc()
                        print("exit_code=1")
                        raise
                    print("exit_code=0")
            gpu = read_evidence(run_directory / "steps-500/rank-0.json", "cuda")
            compare(reference, gpu)
            publish_artifact(gpu_artifact, gpu, run_directory / "steps-500/rank-0-config.txt", identity)
            attempt.update("complete")
        except BaseException as error:
            attempt.update("failed", error=repr(error))
            raise
        print_output_path("Completed GPU observation", gpu_artifact)
        return
    if not args.compare:
        raise ValueError("Recorded observations support --data, --capture reference or --compare")
    assert_run_not_active(run_directory)
    for artifact in (gpu_artifact, npu_artifact):
        validate_artifact(artifact, identity)
    report = (root / "mindstudio_reports/accuracy" / experiment / "single" /
              profile / "observations/training/compare-r1")
    report.mkdir(parents=True, exist_ok=True)
    summary_path = compare_training_metrics(
        reference_path=gpu_artifact / "official/training_metrics.jsonl",
        candidate_path=npu_artifact / "official/training_metrics.jsonl",
        output_directory=report, reference_label="GPU", candidate_label="NPU")
    summary = json.loads(summary_path.read_text())
    concise = {key: summary[key] for key in ("step_count", "loss", "grad_norm")}
    (report / "summary.log").write_text(json.dumps(concise, indent=2) + "\n", encoding="utf-8")
    mirror = root / "mindstudio_reports/accuracy" / experiment / "html_reports" / profile / "training/single.html"
    mirror.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(report / "training_observation.html", mirror)
    print_output_path("Observation summary", report / "summary.log")
    print_output_path("Observation HTML", mirror)
