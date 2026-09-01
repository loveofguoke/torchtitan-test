# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Lifecycle-safe msProbe hierarchical graph comparison."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Sequence

from tests.glm5_2_common.cli import (
    RunAttempt,
    archive_previous_output,
    assert_run_not_active,
    reset_output_generation,
)
from tests.glm5_2_common.naming import config_digest
from tests.glm5_2_common.topology import ParallelTopology

from .artifacts import (
    MindStudioArtifactError,
    artifact_is_complete,
    output_index,
    write_json,
)
from .config import MindStudioExperimentConfig
from .msprobe_adapter import graph_visualize_command, write_compare_invocation


def _capture_artifact(
    root: Path,
    config: MindStudioExperimentConfig,
    topology: ParallelTopology,
    role: str,
    repeat: int,
) -> Path:
    return (
        root
        / config.artifact_root
        / config.storage_name
        / topology.slug
        / f"{role}-r{repeat}"
    )


def _visualization_paths(
    root: Path,
    config: MindStudioExperimentConfig,
    topology: ParallelTopology,
    repeat: int,
) -> tuple[Path, Path, Path]:
    relative = (
        Path(config.storage_name)
        / topology.slug
        / f"graph-visualize-r{repeat}"
    )
    return (
        root / config.run_root / relative,
        root / config.artifact_root / relative,
        root / config.report_root / relative,
    )


def _load_capture(
    root: Path,
    config: MindStudioExperimentConfig,
    topology: ParallelTopology,
    role: str,
    repeat: int,
) -> tuple[Path, dict[str, Any]]:
    from .workflow import _experiment_digest, _fixture_manifest

    artifact = _capture_artifact(root, config, topology, role, repeat)
    fixture = _fixture_manifest(root, config)
    generation = str(fixture["generation_id"])
    if not artifact_is_complete(
        artifact,
        experiment_digest=_experiment_digest(
            config,
            topology,
            role,  # type: ignore[arg-type]
        ),
        fixture_generation_id=generation,
    ):
        raise MindStudioArtifactError(
            f"complete {role} dump capture is required: {artifact}"
        )
    manifest = json.loads(
        (artifact / "manifest.json").read_text(encoding="utf-8")
    )
    return artifact, manifest


def _require_construct_files(
    artifact: Path,
    config: MindStudioExperimentConfig,
    topology: ParallelTopology,
) -> None:
    ranks = config.dump.ranks or tuple(range(topology.world_size))
    missing: list[Path] = []
    empty: list[Path] = []
    for step in config.dump.steps:
        for rank in ranks:
            path = artifact / "official" / f"step{step}" / f"rank{rank}" / "construct.json"
            if not path.is_file():
                missing.append(path)
                continue
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                value = None
            if not value:
                empty.append(path)
    if missing or empty:
        details = []
        if missing:
            details.append("missing=" + ", ".join(str(path) for path in missing))
        if empty:
            details.append("empty=" + ", ".join(str(path) for path in empty))
        raise MindStudioArtifactError(
            "graph_visualize requires non-empty construct.json from L0/mix "
            "captures: " + "; ".join(details)
        )


def _graph_input(
    artifact: Path,
    config: MindStudioExperimentConfig,
    topology: ParallelTopology,
) -> Path:
    ranks = config.dump.ranks or tuple(range(topology.world_size))
    if len(config.dump.steps) > 1:
        return artifact / "official"
    step_directory = artifact / "official" / f"step{config.dump.steps[0]}"
    if len(ranks) > 1:
        return step_directory
    return step_directory / f"rank{ranks[0]}"


def _toolchain() -> tuple[dict[str, Any], dict[str, Any]]:
    from .toolchain import collect_toolchain_metadata
    from .workflow import _comparison_toolchain_identity

    metadata = collect_toolchain_metadata(scope="accuracy-npu")
    return metadata, _comparison_toolchain_identity(metadata)


def _write_report(
    *,
    root: Path,
    report_directory: Path,
    artifact_directory: Path,
    run_directory: Path,
    identity: dict[str, Any],
) -> Path:
    official = artifact_directory / "official"
    databases = [
        entry for entry in output_index(official) if entry["path"].endswith(".vis.db")
    ]
    relative_artifact = artifact_directory.resolve().relative_to(root.resolve())
    relative_log = (run_directory / "runtime.log").resolve().relative_to(root.resolve())
    tensorboard = f"tensorboard --logdir {official.resolve()} --bind_all"
    index = {
        "schema": "torchtitan.glm5_2.mindstudio_graph_visualization",
        "schema_version": 1,
        "identity": identity,
        "artifact": relative_artifact.as_posix(),
        "runtime_log": relative_log.as_posix(),
        "databases": databases,
        "tensorboard_command": tensorboard,
        "security": (
            "The .vis.db files remain under the ignored artifact root because "
            "they can contain tensor statistics, model names, and source paths."
        ),
    }
    write_json(report_directory / "index.json", index)
    lines = [
        "# msProbe hierarchical graph comparison",
        "",
        "This is an index only. The official `.vis.db` stays in the ignored ",
        "artifact directory and is not copied into the tracked report.",
        "",
        f"- Artifact: `{relative_artifact.as_posix()}`",
        f"- Runtime log: `{relative_log.as_posix()}`",
        f"- Database count: {len(databases)}",
        "",
        "Start the official TensorBoard plugin on the server:",
        "",
        "```bash",
        tensorboard,
        "```",
        "",
        "With VSCode Remote, port-forward the printed port. Do not expose the ",
        "service publicly. The Release analysis profile can synchronize the ",
        "processed database only after its tensor statistics, model names, and ",
        "source paths have been reviewed for sensitive content.",
    ]
    report_directory.mkdir(parents=True, exist_ok=True)
    (report_directory / "README.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    return report_directory


def reset_graph_visualizations(
    root: Path,
    config: MindStudioExperimentConfig,
    *,
    topologies: Sequence[ParallelTopology],
    repeat: int,
) -> None:
    paths = [
        path
        for topology in topologies
        for path in _visualization_paths(root, config, topology, repeat)
    ]
    for topology in topologies:
        run, _, _ = _visualization_paths(root, config, topology, repeat)
        assert_run_not_active(run)
    reset_output_generation(paths, label="MindStudio graph visualization")


def run_graph_visualization(
    root: Path,
    config: MindStudioExperimentConfig,
    *,
    topology: ParallelTopology,
    repeat: int,
    fuzzy_match: bool = False,
    overflow_check: bool = False,
    tensor_log: bool = False,
    progress_log: bool = False,
    dry_run: bool = False,
) -> Path:
    if config.workflow != "migration":
        raise ValueError("graph visualization is supported only for migration dumps")
    if config.dump.level not in {"L0", "mix"}:
        raise ValueError("graph visualization requires an L0 or mix dump")
    if tensor_log and config.dump.task != "tensor":
        raise ValueError("graph tensor logging requires a tensor dump")

    reference, reference_manifest = _load_capture(
        root, config, topology, "reference", repeat
    )
    candidate, candidate_manifest = _load_capture(
        root, config, topology, "candidate", repeat
    )
    reference_generation = str(reference_manifest["fixture_generation_id"])
    candidate_generation = str(candidate_manifest["fixture_generation_id"])
    if reference_generation != candidate_generation:
        raise MindStudioArtifactError(
            "reference and candidate graph dumps use different fixture generations"
        )
    _require_construct_files(reference, config, topology)
    _require_construct_files(candidate, config, topology)
    toolchain, toolchain_identity = _toolchain()
    identity = {
        "experiment": config.identity,
        "topology": topology.name,
        "repeat": repeat,
        "fixture_generation_id": reference_generation,
        "reference_files": reference_manifest["official_files"],
        "candidate_files": candidate_manifest["official_files"],
        "options": {
            "fuzzy_match": fuzzy_match,
            "overflow_check": overflow_check,
            "tensor_log": tensor_log,
            "progress_log": progress_log,
        },
        "toolchain": toolchain_identity,
    }
    digest = config_digest(identity, length=64)
    run_directory, artifact_directory, report_directory = _visualization_paths(
        root, config, topology, repeat
    )
    if artifact_is_complete(
        artifact_directory,
        experiment_digest=digest,
        fixture_generation_id=reference_generation,
    ):
        print(f"Skip completed graph visualization: {artifact_directory}", flush=True)
        return _write_report(
            root=root,
            report_directory=report_directory,
            artifact_directory=artifact_directory,
            run_directory=run_directory,
            identity=identity,
        )
    target = _graph_input(candidate, config, topology)
    golden = _graph_input(reference, config, topology)
    official_output = artifact_directory / "official"
    command = graph_visualize_command(
        target=target,
        golden=golden,
        output=official_output,
        fuzzy_match=fuzzy_match,
        overflow_check=overflow_check,
        tensor_log=tensor_log,
        progress_log=progress_log,
    )
    if dry_run:
        print(json.dumps({"command": command, "identity": identity}, indent=2))
        return report_directory

    for stale in (run_directory, artifact_directory, report_directory):
        if stale.exists():
            if stale == run_directory:
                assert_run_not_active(stale)
            archived = archive_previous_output(stale)
            print(f"Retry incomplete graph visualization; archived: {archived}")
    run_directory.mkdir(parents=True, exist_ok=True)
    artifact_directory.mkdir(parents=True, exist_ok=True)
    write_compare_invocation(run_directory / "invocation.json", command)
    attempt = RunAttempt.start(
        run_directory,
        kind="mindstudio_graph_visualize",
        context={
            "topology": topology.name,
            "repeat": repeat,
            "fixture_generation_id": reference_generation,
            "experiment_digest": digest,
        },
    )
    from .workflow import _run_process

    try:
        _run_process(
            command,
            root=root,
            environment=os.environ.copy(),
            log_path=run_directory / "runtime.log",
            context=attempt.log_context,
        )
        databases = sorted(official_output.rglob("*.vis.db"))
        if not databases:
            raise MindStudioArtifactError(
                f"graph_visualize produced no .vis.db under {official_output}"
            )
        manifest = {
            "schema": "torchtitan.glm5_2.mindstudio_graph_visualization_artifact",
            "schema_version": 1,
            "experiment_digest": digest,
            "fixture_generation_id": reference_generation,
            "official_output": "official",
            "official_files": output_index(official_output),
            "identity": identity,
            "toolchain": toolchain,
            "runtime_log": str((run_directory / "runtime.log").resolve()),
        }
        write_json(artifact_directory / "manifest.json", manifest)
        write_json(
            artifact_directory / "complete.json",
            {
                "status": "completed",
                "experiment_digest": digest,
                "fixture_generation_id": reference_generation,
                "attempt_id": attempt.attempt_id,
            },
        )
        attempt.update("completed", artifact=str(artifact_directory.resolve()))
    except BaseException as error:
        attempt.update("failed", error=repr(error))
        raise
    return _write_report(
        root=root,
        report_directory=report_directory,
        artifact_directory=artifact_directory,
        run_directory=run_directory,
        identity=identity,
    )


__all__ = [
    "reset_graph_visualizations",
    "run_graph_visualization",
]
