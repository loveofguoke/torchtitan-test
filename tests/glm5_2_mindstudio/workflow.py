# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Lifecycle and launch orchestration around official MindStudio tools."""

from __future__ import annotations

import argparse
from html import escape
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Literal, Sequence

from tests.glm5_2_common.cli import (
    RunAttempt,
    archive_previous_output,
    assert_run_not_active,
    reset_output_generation,
)
from tests.glm5_2_common.naming import config_digest
from tests.glm5_2_common.topology import (
    ParallelTopology,
    select_topologies,
    standard_topologies,
    training_command_args,
)
from tests.glm5_2_precision.workflow import (
    FormalTrainingConfig,
    TrainingEndpoint,
    fixed_input_environment,
    prepare_fixture,
    resolve_fixture_inputs,
)
from tests.glm5_2_precision.token_data import (
    load_token_plan,
    validate_runtime_input_contract,
)

from .artifacts import (
    MindStudioArtifactError,
    artifact_is_complete,
    output_index,
    sha256_file,
    write_json,
)
from .config import MindStudioExperimentConfig
from .msprobe_adapter import (
    aggregate_compile_csv,
    compare_command,
    config_check_compare_command,
    find_dump_compare_input,
    find_precheck_details,
    precheck_command,
    precheck_compare_command,
    summarize_official_results,
    validate_compile_report,
    write_compare_invocation,
)
from .report import write_report_index


Role = Literal["reference", "candidate"]


def _root(script_path: str) -> Path:
    return Path(script_path).resolve().parents[2]


def _paths(
    root: Path,
    config: MindStudioExperimentConfig,
    topology: ParallelTopology,
    role: Role,
    repeat: int,
) -> tuple[Path, Path, Path]:
    relative = Path(config.storage_name) / topology.slug / f"{role}-r{repeat}"
    return (
        root / config.run_root / relative,
        root / config.artifact_root / relative,
        root / config.report_root / config.storage_name / topology.slug,
    )


def _fixture_directory(root: Path, config: MindStudioExperimentConfig) -> Path:
    return root / config.fixture_root / config.storage_name


def _fixture_manifest(root: Path, config: MindStudioExperimentConfig) -> dict[str, Any]:
    path = _fixture_directory(root, config) / "fixture.json"
    if not path.is_file():
        raise FileNotFoundError(f"fixture is missing; run --data first: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _experiment_digest(
    config: MindStudioExperimentConfig,
    topology: ParallelTopology,
    role: Role,
) -> str:
    return config_digest(
        {
            "experiment": config.identity,
            "topology": asdict(topology),
            "role": role,
        },
        length=64,
    )


def _optional_file_identity(path: Path | None) -> dict[str, str] | None:
    if path is None:
        return None
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {"path": str(resolved), "sha256": sha256_file(resolved)}


def _accuracy_toolchain_metadata(device_type: str) -> dict[str, Any]:
    try:
        from .toolchain import collect_toolchain_metadata

        scope = {
            "npu": "accuracy-npu",
            "cuda": "accuracy-gpu",
            "compare": "compare",
        }.get(device_type)
        if scope is None:
            raise ValueError(f"unsupported toolchain device scope: {device_type}")
        return collect_toolchain_metadata(scope=scope)
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        return {"metadata_error": str(error)}


def _comparison_toolchain_identity(metadata: dict[str, Any]) -> dict[str, Any]:
    """Select compare-defining msProbe metadata for cache invalidation."""

    if metadata.get("metadata_error"):
        raise RuntimeError(
            "cannot identify the compare-side MindStudio toolchain: "
            f"{metadata['metadata_error']}"
        )
    sources = metadata.get("sources", {})
    source = sources.get("msprobe", {}) if isinstance(sources, dict) else {}
    doctor = metadata.get("doctor", {})
    checks = doctor.get("checks", {}) if isinstance(doctor, dict) else {}
    msprobe = checks.get("msprobe", {}) if isinstance(checks, dict) else {}
    executable = (
        msprobe.get("executable", {}) if isinstance(msprobe, dict) else {}
    )
    imported = msprobe.get("import", {}) if isinstance(msprobe, dict) else {}
    executable_path = executable.get("path")
    executable_identity = (
        _optional_file_identity(Path(executable_path))
        if executable_path
        else None
    )
    versions = metadata.get("versions", {})
    lock = metadata.get("lock", {})
    resolved = metadata.get("resolved_toolchain", {})
    if not isinstance(resolved, dict):
        resolved = {}
    installations = metadata.get("installations", {})
    installation = (
        installations.get("msprobe", {})
        if isinstance(installations, dict)
        else {}
    )
    if not isinstance(installation, dict):
        installation = {}
    record = installation.get("record", {})
    package_tree = installation.get("package_tree", {})
    console_scripts = installation.get("console_scripts", [])
    return {
        "lock_sha256": lock.get("sha256") if isinstance(lock, dict) else None,
        "profile_name": metadata.get("profile_name"),
        "msprobe_version": (
            versions.get("msprobe") if isinstance(versions, dict) else None
        ),
        "msprobe_import_origin": (
            imported.get("origin") if isinstance(imported, dict) else None
        ),
        "msprobe_executable": executable_identity,
        "resolved_msprobe_build": {
            "source_commit": resolved.get("msprobe_source_commit"),
            "artifact_identity": resolved.get("msprobe_artifact_identity"),
        },
        "installed_msprobe": {
            "distribution": installation.get("distribution"),
            "record": {
                "size": record.get("size"),
                "sha256": record.get("sha256"),
            }
            if isinstance(record, dict)
            else None,
            "package_tree": {
                key: package_tree.get(key)
                for key in ("file_count", "total_size", "sha256")
            }
            if isinstance(package_tree, dict)
            else None,
            "console_scripts": sorted(
                (
                    {
                        "name": script.get("name"),
                        "size": script.get("size"),
                        "sha256": script.get("sha256"),
                    }
                    for script in console_scripts
                    if isinstance(script, dict)
                ),
                key=lambda value: str(value.get("name")),
            ),
            "origin_in_distribution": installation.get(
                "origin_in_distribution"
            ),
        },
        "msprobe_source": {
            key: source.get(key)
            for key in (
                "official_url",
                "selected_ref",
                "locked_commit",
                "locked_version",
                "commit",
                "branch",
                "tag",
                "dirty",
            )
        },
    }


def _git_dirty_tree_sha256(repository: Path) -> str | None:
    """Hash tracked diffs and untracked source files without hashing ignored runs."""

    try:
        diff = subprocess.run(
            ["git", "-C", str(repository), "diff", "--binary", "HEAD"],
            check=True,
            capture_output=True,
        ).stdout
        untracked = subprocess.run(
            [
                "git",
                "-C",
                str(repository),
                "ls-files",
                "--others",
                "--exclude-standard",
                "-z",
            ],
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    digest = hashlib.sha256()
    digest.update(diff)
    for encoded in sorted(value for value in untracked.split(b"\0") if value):
        relative = encoded.decode("utf-8", errors="surrogateescape")
        path = repository / relative
        digest.update(encoded)
        digest.update(b"\0")
        if path.is_file():
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def _capture_execution_identity(root: Path, device_type: str) -> dict[str, Any]:
    """Return path-independent source and framework identity for one capture."""

    from tests.glm5_2_precision.workflow import _source_metadata

    metadata = _source_metadata(root)
    source_values: dict[str, Any] = {}
    sources = metadata.get("sources", {})
    if isinstance(sources, dict):
        for name in ("torchtitan_test", "torchtitan", "torchtitanturbo"):
            value = sources.get(name)
            if not isinstance(value, dict):
                continue
            repository_value = value.get("root")
            repository = Path(repository_value) if repository_value else None
            source_values[name] = {
                "commit": value.get("commit"),
                "branch": value.get("branch"),
                "status": value.get("status"),
                "dirty_tree_sha256": (
                    _git_dirty_tree_sha256(repository)
                    if repository is not None
                    else None
                ),
            }
    packages = metadata.get("packages", {})
    runtime = metadata.get("runtime", {})
    executable = Path(sys.executable).resolve()
    cann_files = (
        runtime.get("cann_version_files", [])
        if isinstance(runtime, dict)
        else []
    )
    return {
        "device_type": device_type,
        "packages": {
            name: packages.get(name)
            for name in ("torch", "torch_npu", "torchtitan", "torchtitanturbo")
        }
        if isinstance(packages, dict)
        else {},
        "sources": source_values,
        "python": {
            "version": (
                runtime.get("python_version")
                if isinstance(runtime, dict)
                else None
            ),
            "executable_sha256": (
                sha256_file(executable) if executable.is_file() else None
            ),
        },
        "cann_version_files": sorted(
            str(value.get("sha256"))
            for value in cann_files
            if isinstance(value, dict) and value.get("sha256")
        ),
    }


def _msprobe_toolchain_compatibility(
    identity: dict[str, Any],
) -> dict[str, Any]:
    """Return the path-independent msProbe identity shared across hosts."""

    source = identity.get("msprobe_source", {})
    if not isinstance(source, dict):
        source = {}
    resolved_build = identity.get("resolved_msprobe_build", {})
    if not isinstance(resolved_build, dict):
        resolved_build = {}
    executable = identity.get("msprobe_executable")
    executable_sha256 = (
        executable.get("sha256") if isinstance(executable, dict) else None
    )
    return {
        "lock_sha256": identity.get("lock_sha256"),
        "profile_name": identity.get("profile_name"),
        "msprobe_version": identity.get("msprobe_version"),
        "msprobe_executable_sha256": executable_sha256,
        "resolved_msprobe_build": {
            "source_commit": resolved_build.get("source_commit"),
            "artifact_identity": resolved_build.get("artifact_identity"),
        },
        "installed_msprobe": identity.get("installed_msprobe"),
        "msprobe_source": {
            key: source.get(key)
            for key in (
                "official_url",
                "selected_ref",
                "locked_commit",
                "locked_version",
                "commit",
                "tag",
                "dirty",
            )
        },
    }


def _capture_toolchain_compatibility(
    identity: dict[str, Any],
) -> dict[str, Any]:
    """Return the path-independent msProbe identity shared across hosts."""

    compatibility = _msprobe_toolchain_compatibility(identity)
    execution = identity.get("execution_environment", {})
    sources = execution.get("sources", {}) if isinstance(execution, dict) else {}
    compatibility["project_sources"] = {
        name: sources.get(name)
        for name in ("torchtitan_test", "torchtitan")
    }
    return compatibility


def _capture_is_complete(
    artifact_directory: Path,
    *,
    experiment_digest: str,
    fixture_generation_id: str,
    toolchain_identity: dict[str, Any],
) -> bool:
    if not artifact_is_complete(
        artifact_directory,
        experiment_digest=experiment_digest,
        fixture_generation_id=fixture_generation_id,
    ):
        return False
    try:
        manifest = json.loads(
            (artifact_directory / "manifest.json").read_text(encoding="utf-8")
        )
    except (OSError, TypeError, ValueError):
        return False
    input_contract = manifest.get("input_contract")
    return (
        manifest.get("toolchain_identity") == toolchain_identity
        and isinstance(input_contract, dict)
        and input_contract.get("valid") is True
    )


def _comparison_is_current(path: Path, digest: str) -> bool:
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
        current_files = _comparison_output_index(path.parent)
    except (OSError, TypeError, ValueError):
        return False
    return (
        state.get("comparison_digest") == digest
        and state.get("output_files") == current_files
    )


def _comparison_output_index(directory: Path) -> list[dict[str, Any]]:
    return [
        entry
        for entry in output_index(directory)
        if entry["path"]
        not in {"comparison_state.json", "compare_run_state.json"}
    ]


def _metrics_rank(topology: ParallelTopology) -> int:
    if (
        topology.pipeline_parallel_degree == 1
        or topology.pipeline_parallel_schedule == "ZBVZeroBubble"
    ):
        return 0
    return (topology.world_size // topology.pipeline_parallel_degree) * (
        topology.pipeline_parallel_degree - 1
    )


def _base_training_args(
    training: FormalTrainingConfig,
    *,
    dump_folder: Path,
    topology: ParallelTopology,
) -> list[str]:
    arguments = [
        f"--dump_folder={dump_folder}",
        f"--training.steps={training.steps}",
        *training_command_args(
            local_batch_size=training.local_batch_size,
            global_batch_size=training.global_batch_size,
            sequence_length=training.sequence_length,
            topology=topology,
        ),
        f"--training.dtype={training.training_dtype}",
        f"--training.mixed_precision_param={training.mixed_precision_param}",
        f"--training.mixed_precision_reduce={training.mixed_precision_reduce}",
        f"--debug.seed={training.seed}",
        "--metrics.log_freq=1",
        "--metrics.disable_color_printing",
        "--debug.no-enable-structured-logging",
    ]
    if training.deterministic:
        arguments.append("--debug.deterministic")
    arguments.extend(training.extra_args)
    return arguments


def _torchrun_command(
    *,
    root: Path,
    config: MindStudioExperimentConfig,
    endpoint: TrainingEndpoint,
    run_directory: Path,
    checkpoint_path: Path,
) -> list[str]:
    if endpoint.num_nodes > 1 and endpoint.rendezvous_endpoint == "localhost:0":
        raise ValueError("multi-node capture requires a shared rendezvous endpoint")
    torchrun = shutil.which("torchrun") or str(
        Path(sys.executable).with_name("torchrun")
    )
    rendezvous_id = (
        f"{config.storage_name}-{endpoint.topology.slug}-{run_directory.name}"
    )
    return [
        torchrun,
        f"--nnodes={endpoint.num_nodes}",
        f"--node_rank={endpoint.node_rank}",
        f"--nproc_per_node={endpoint.num_processes_per_node}",
        "--rdzv_backend=c10d",
        f"--rdzv_endpoint={endpoint.rendezvous_endpoint}",
        f"--rdzv_id={rendezvous_id}",
        "--local-ranks-filter="
        f"{_metrics_rank(endpoint.topology) % endpoint.num_processes_per_node}",
        "--role=rank",
        "--tee=3",
        "-m",
        "tests.glm5_2_mindstudio.capture_training",
        "--module",
        config.training.module,
        "--config",
        config.training.config,
        *_base_training_args(
            config.training,
            dump_folder=run_directory / "trainer_output",
            topology=endpoint.topology,
        ),
        *endpoint.topology.command_args(),
        *endpoint.extra_args,
        "--checkpoint.enable",
        "--checkpoint.load_only",
        f"--checkpoint.initial_load_path={checkpoint_path}",
    ]


def _run_process(
    command: Sequence[str],
    *,
    root: Path,
    environment: dict[str, str],
    log_path: Path,
    context: dict[str, Any],
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Runtime log: {log_path}", flush=True)
    with log_path.open("w", encoding="utf-8") as stream:
        for key, value in context.items():
            stream.write(f"{key}: {value}\n")
        stream.write("\nCommand: " + " ".join(command) + "\n\n")
        stream.flush()
        process = subprocess.run(
            list(command),
            cwd=root,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if process.returncode:
        print(f"Process failed; runtime log: {log_path}", file=sys.stderr)
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        if lines:
            print("\n".join(lines[-80:]), file=sys.stderr)
        raise subprocess.CalledProcessError(process.returncode, list(command))


def _write_launch_contract(
    path: Path,
    *,
    command: Sequence[str],
    environment: dict[str, str],
    endpoint: TrainingEndpoint,
) -> None:
    """Write the shell/config input consumed by official ConfigChecker."""

    exported = {
        endpoint.visible_devices_env: environment[endpoint.visible_devices_env],
        "TORCHTITAN_DEVICE": environment["TORCHTITAN_DEVICE"],
        "LOG_RANK": environment["LOG_RANK"],
    }
    lines = ["#!/usr/bin/env bash", "set -euo pipefail"]
    lines.extend(
        f"export {key}={shlex.quote(value)}" for key, value in exported.items()
    )
    lines.append(shlex.join(command))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _compile_rank_csv_files(
    artifact_directory: Path, world_size: int
) -> list[Path]:
    """Select only the merged per-rank PrecisionChecker reports."""

    return [
        artifact_directory / "official" / f"precision_rank{rank}.csv"
        for rank in range(world_size)
    ]


def _validate_compare_options(
    config: MindStudioExperimentConfig,
    topology: ParallelTopology,
    *,
    fuzzy_match: bool,
    data_mapping: Path | None,
    cell_mapping: Path | None,
    diff_analysis: bool,
    tensor_log: bool,
    xlsx: bool,
) -> None:
    selected_options = (
        fuzzy_match
        or data_mapping is not None
        or cell_mapping is not None
        or diff_analysis
        or tensor_log
        or xlsx
    )
    if config.workflow != "migration" and selected_options:
        raise ValueError(
            "msProbe compare options are only valid for the migration workflow"
        )
    if fuzzy_match and data_mapping is not None:
        raise ValueError("msProbe fuzzy matching is incompatible with data mapping")
    selected_ranks = config.dump.ranks or tuple(range(topology.world_size))
    if any(rank >= topology.world_size for rank in selected_ranks):
        raise ValueError("msProbe dump rank is outside the selected topology")
    if data_mapping is not None and len(selected_ranks) != 1:
        raise ValueError("msProbe data mapping requires rank-by-rank comparison")
    if cell_mapping is not None and config.dump.level != "L0":
        raise ValueError("msProbe cell mapping requires an L0 module dump")
    if tensor_log and config.dump.task != "tensor":
        raise ValueError("msProbe tensor logging requires a tensor dump")


def _validate_rank_outputs(
    official_output: Path,
    *,
    pattern: str,
    world_size: int,
) -> None:
    missing = [
        official_output / pattern.format(rank=rank)
        for rank in range(world_size)
        if not (official_output / pattern.format(rank=rank)).is_file()
    ]
    if missing:
        raise MindStudioArtifactError(
            "official output is missing per-rank files: "
            + ", ".join(str(path) for path in missing)
        )


def _validate_compile_outputs(
    official_output: Path,
    *,
    world_size: int,
    policy: str,
    children_depth: int,
) -> None:
    """Validate per-rank/part policy coverage and decisive checker rows."""

    _validate_rank_outputs(
        official_output,
        pattern="precision_rank{rank}.csv",
        world_size=world_size,
    )
    for rank in range(world_size):
        coverage_path = official_output / f"compile_coverage_rank{rank}.json"
        if not coverage_path.is_file():
            raise MindStudioArtifactError(
                f"PrecisionChecker coverage metadata is missing: {coverage_path}"
            )
        try:
            coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError) as error:
            raise MindStudioArtifactError(
                f"PrecisionChecker coverage metadata is invalid: {coverage_path}"
            ) from error
        if (
            coverage.get("schema")
            != "torchtitan.glm5_2.msprobe_compile_coverage"
            or coverage.get("schema_version") != 1
        ):
            raise MindStudioArtifactError(
                f"PrecisionChecker coverage schema is invalid: {coverage_path}"
            )
        if coverage.get("rank") != rank:
            raise MindStudioArtifactError(
                f"PrecisionChecker coverage rank mismatch: {coverage_path}"
            )
        expected_policy = {"name": policy, "children_depth": children_depth}
        if coverage.get("policy") != expected_policy:
            raise MindStudioArtifactError(
                f"PrecisionChecker coverage policy mismatch: {coverage_path}"
            )
        parts = coverage.get("parts")
        if not isinstance(parts, list) or not parts:
            raise MindStudioArtifactError(
                f"PrecisionChecker coverage has no model parts: {coverage_path}"
            )
        for part_index, part in enumerate(parts):
            if not isinstance(part, dict) or part.get("part") != part_index:
                raise MindStudioArtifactError(
                    "PrecisionChecker coverage has invalid part metadata: "
                    f"{coverage_path}"
                )
            names = part.get("expected_glm_block_names")
            count = part.get("expected_glm_block_count")
            if (
                not isinstance(count, int)
                or count < 1
                or not isinstance(names, list)
                or len(names) != count
                or any(not isinstance(name, str) or not name for name in names)
            ):
                raise MindStudioArtifactError(
                    "PrecisionChecker expected GLM block coverage must be nonzero "
                    f"for every rank/part: {coverage_path}"
                )
            report_path = official_output / str(part.get("report", ""))
            expected_report = official_output / (
                f"precision_rank{rank}_part{part_index}.csv"
            )
            if report_path != expected_report or not report_path.is_file():
                raise MindStudioArtifactError(
                    f"PrecisionChecker part report is missing: {expected_report}"
                )
            actual_rows = validate_compile_report(report_path)
            if part.get("decisive_rows") != actual_rows:
                raise MindStudioArtifactError(
                    f"PrecisionChecker coverage/report mismatch: {coverage_path}"
                )
        validate_compile_report(
            official_output / f"precision_rank{rank}.csv"
        )


def _validate_dump_outputs(
    official_output: Path,
    *,
    steps: Sequence[int],
    configured_ranks: Sequence[int],
    world_size: int,
) -> None:
    ranks = tuple(configured_ranks) or tuple(range(world_size))
    missing = [
        official_output / f"step{step}" / f"rank{rank}" / "dump.json"
        for step in steps
        for rank in ranks
        if not (
            official_output / f"step{step}" / f"rank{rank}" / "dump.json"
        ).is_file()
    ]
    if missing:
        raise MindStudioArtifactError(
            "msProbe output is missing selected step/rank dumps: "
            + ", ".join(str(path) for path in missing)
        )


def _validate_monitor_outputs(
    official_output: Path,
    *,
    configured_ranks: Sequence[int],
    world_size: int,
) -> None:
    ranks = tuple(configured_ranks) or tuple(range(world_size))
    missing = [
        official_output / f"rank_{rank}"
        for rank in ranks
        if not any(
            path.is_file()
            for path in (official_output / f"rank_{rank}").rglob("*.csv")
        )
    ]
    if missing:
        raise MindStudioArtifactError(
            "TrainerMonitorV2 output is missing rank CSV files: "
            + ", ".join(str(path) for path in missing)
        )


def _validate_monitor_execution(
    config: MindStudioExperimentConfig,
    topology: ParallelTopology,
    endpoint: TrainingEndpoint,
) -> None:
    ranks = config.monitor.ranks or tuple(range(topology.world_size))
    if any(rank >= topology.world_size for rank in ranks):
        raise ValueError("monitor rank is outside the selected topology")
    if any(
        argument.lower().startswith("--compile.enable")
        for argument in endpoint.extra_args
    ):
        raise ValueError(
            "TrainerMonitorV2 plus torch.compile is not validated by this "
            "workflow; use the compile accuracy workflow independently"
        )
    if topology.pipeline_parallel_degree > 1:
        print(
            "Warning: TrainerMonitorV2 accepts a model-parts list, but "
            f"pipeline topology {topology.name} still requires server-side "
            "validation of virtual-stage names and optimizer ownership.",
            flush=True,
        )


def _resolve_endpoint_visibility(endpoint: TrainingEndpoint) -> TrainingEndpoint:
    value = os.environ.get(endpoint.visible_devices_env, "").strip()
    return replace(endpoint, visible_devices=value) if value else endpoint


def _endpoint(
    config: MindStudioExperimentConfig,
    role: Role,
    topology: ParallelTopology,
) -> TrainingEndpoint:
    base = config.reference if role == "reference" else config.candidate
    return _resolve_endpoint_visibility(replace(base, topology=topology))


def _data_endpoint(
    config: MindStudioExperimentConfig,
    requested_device: str | None,
) -> TrainingEndpoint:
    endpoints = (config.reference, config.candidate)
    if requested_device is not None:
        matches = [
            value for value in endpoints if value.device_type == requested_device
        ]
        if not matches:
            raise ValueError(f"experiment has no {requested_device} endpoint")
        return _resolve_endpoint_visibility(
            replace(matches[0], topology=ParallelTopology("single", 1))
        )
    visible = {
        "cuda": os.environ.get("CUDA_VISIBLE_DEVICES", "").strip(),
        "npu": os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "").strip(),
    }
    selected = [device for device, value in visible.items() if value]
    if len(selected) != 1:
        raise ValueError(
            "export exactly one visibility variable or pass --data-device"
        )
    return _data_endpoint(config, selected[0])


def _formal_config(
    config: MindStudioExperimentConfig,
    topology: ParallelTopology,
):
    formal = config.formal_fixture_config()
    return replace(
        formal,
        reference=replace(config.reference, topology=topology),
        candidate=replace(config.candidate, topology=topology),
    )


def prepare_shared_fixture(
    root: Path,
    config: MindStudioExperimentConfig,
    *,
    endpoint: TrainingEndpoint,
    force: bool,
) -> Path:
    formal = _formal_config(config, ParallelTopology("single", 1))
    return prepare_fixture(root, formal, endpoint=endpoint, force=force)


def _assert_tree_not_active(path: Path) -> None:
    if not path.is_dir():
        return
    state_paths = {
        *path.rglob("run_state.json"),
        *path.rglob("capture_state.json"),
        *path.rglob("precheck_state.json"),
        *path.rglob("compare_run_state.json"),
    }
    for lock_path in path.rglob("*.json.lock"):
        state_paths.add(lock_path.with_name(lock_path.name.removesuffix(".lock")))
    for state_path in state_paths:
        assert_run_not_active(state_path.parent, state_name=state_path.name)


def reset_selected_outputs(
    root: Path,
    config: MindStudioExperimentConfig,
    *,
    topologies: Sequence[ParallelTopology],
    role: Role | None,
    include_fixture: bool,
) -> None:
    paths: list[Path] = []
    if include_fixture:
        paths.append(_fixture_directory(root, config))
        paths.extend(
            (
                root / config.run_root / config.storage_name,
                root / config.artifact_root / config.storage_name,
                root / config.report_root / config.storage_name,
            )
        )
    else:
        roles: tuple[Role, ...] = (role,) if role is not None else (
            "reference",
            "candidate",
        )
        for topology in topologies:
            for selected_role in roles:
                endpoint = _endpoint(config, selected_role, topology)
                for repeat in range(1, endpoint.repeats + 1):
                    run, artifact, report = _paths(
                        root, config, topology, selected_role, repeat
                    )
                    precheck = _precheck_root(
                        root,
                        config,
                        topology,
                        owner=selected_role,
                        repeat=repeat,
                    )
                    paths.extend((run, artifact, precheck, report))
        aggregate = root / config.report_root / config.storage_name
        paths.extend(
            (
                aggregate / f"{config.storage_name}.html",
                aggregate / "report.json",
                aggregate / "README.md",
            )
        )
    for path in paths:
        _assert_tree_not_active(path)
    reset_output_generation(paths, label="MindStudio official workflow")


def reset_compare_outputs(
    root: Path,
    config: MindStudioExperimentConfig,
    *,
    topologies: Sequence[ParallelTopology],
    repeat: int,
) -> None:
    """Reset one selected comparison generation before any compare starts."""

    aggregate = root / config.report_root / config.storage_name
    paths: list[Path] = [
        aggregate / f"{config.storage_name}.html",
        aggregate / "report.json",
        aggregate / "README.md",
    ]
    active: list[Path] = []
    for topology in topologies:
        _, _, report = _paths(root, config, topology, "reference", repeat)
        compare = report / "official_compare"
        paths.append(compare)
        active.append(compare)
    for directory in active:
        assert_run_not_active(
            directory,
            state_name="compare_run_state.json",
        )
    reset_output_generation(
        paths,
        label="MindStudio official compare",
        active_run_directories=active,
    )


def _aggregate_report_paths(
    root: Path,
    config: MindStudioExperimentConfig,
) -> tuple[Path, ...]:
    aggregate = root / config.report_root / config.storage_name
    return (
        aggregate / f"{config.storage_name}.html",
        aggregate / "report.json",
        aggregate / "README.md",
    )


def _invalidate_aggregate_report(
    root: Path,
    config: MindStudioExperimentConfig,
    *,
    label: str,
) -> None:
    paths = _aggregate_report_paths(root, config)
    if not any(path.exists() for path in paths):
        return
    reset_output_generation(
        paths,
        label=label,
    )


def _invalidate_precheck_dependents(
    root: Path,
    config: MindStudioExperimentConfig,
    *,
    topology: ParallelTopology,
    repeat: int,
) -> None:
    compare_root = _precheck_root(
        root,
        config,
        topology,
        owner="compare",
        repeat=repeat,
    )
    _assert_tree_not_active(compare_root)
    reset_output_generation(
        (compare_root, *_aggregate_report_paths(root, config)),
        label="dependent MindStudio pre-check reports",
    )


def capture_official(
    root: Path,
    config: MindStudioExperimentConfig,
    *,
    topology: ParallelTopology,
    role: Role,
    repeat: int,
    dry_run: bool = False,
) -> Path:
    if config.workflow == "compile" and role != "candidate":
        raise ValueError(
            "compile workflow performs eager-vs-compiled checking inside one "
            "candidate run; it has no separate reference capture"
        )
    endpoint = _endpoint(config, role, topology)
    if config.workflow == "monitor":
        _validate_monitor_execution(config, topology, endpoint)
    if not 1 <= repeat <= endpoint.repeats:
        raise ValueError(f"repeat must be in [1, {endpoint.repeats}]")
    formal = _formal_config(config, topology)
    checkpoint_path, token_plan_path = resolve_fixture_inputs(root, formal)
    fixture = _fixture_manifest(root, config)
    generation = str(fixture["generation_id"])
    digest = _experiment_digest(config, topology, role)
    run_directory, artifact_directory, report_directory = _paths(
        root, config, topology, role, repeat
    )
    toolchain: dict[str, Any] | None = None
    toolchain_identity: dict[str, Any] | None = None
    toolchain_compatibility: dict[str, Any] | None = None
    if not dry_run:
        toolchain = _accuracy_toolchain_metadata(endpoint.device_type)
        toolchain_identity = _comparison_toolchain_identity(toolchain)
        toolchain_identity["execution_environment"] = (
            _capture_execution_identity(root, endpoint.device_type)
        )
        toolchain_compatibility = _capture_toolchain_compatibility(
            toolchain_identity
        )
    if not dry_run and _capture_is_complete(
        artifact_directory,
        experiment_digest=digest,
        fixture_generation_id=generation,
        toolchain_identity=toolchain_identity,
    ):
        print(f"Skip completed official capture: {artifact_directory}", flush=True)
        return artifact_directory
    official_output = artifact_directory / "official"
    input_contract = run_directory / "input_contract"
    runtime_log = run_directory / "runtime.log"
    environment = os.environ.copy()
    environment.update(endpoint.environment)
    environment_overrides = {
        endpoint.visible_devices_env: endpoint.visible_devices,
        "TORCHTITAN_DEVICE": endpoint.torchtitan_device,
        "LOG_RANK": str(_metrics_rank(topology)),
        "GLM5_EXPERIMENT_RUN_DIRECTORY": str(run_directory.resolve()),
        "GLM5_MINDSTUDIO_OUTPUT": str(official_output.resolve()),
        "GLM5_MINDSTUDIO_SEED": str(config.training.seed),
        "GLM5_MINDSTUDIO_DETERMINISTIC": (
            "1" if config.training.deterministic else "0"
        ),
        "NPU_ASD_ENABLE": "0",
        **fixed_input_environment(
            token_plan_path=token_plan_path,
            input_contract_directory=input_contract,
            training=config.training,
            topology=topology,
        ),
    }
    environment.update(environment_overrides)
    if config.workflow == "config-check":
        environment["GLM5_MINDSTUDIO_MODE"] = "config-check"
        official_config = {
            "workflow": "config-check",
            "collection": "dynamic",
            "framework": "pytorch",
        }
    elif config.workflow == "migration":
        environment["GLM5_MINDSTUDIO_MODE"] = "dump"
        official_config = config.dump.official_config(official_output)
    elif config.workflow == "compile":
        environment["GLM5_MINDSTUDIO_MODE"] = "compile"
        official_config = {
            **asdict(config.compile),
            "target_training_step": max(config.dump.steps) + 1,
        }
    else:
        environment["GLM5_MINDSTUDIO_MODE"] = "monitor"
        official_config = config.monitor.official_config(official_output)
    environment_overrides["GLM5_MINDSTUDIO_MODE"] = environment[
        "GLM5_MINDSTUDIO_MODE"
    ]
    config_path = run_directory / "msprobe_config.json"
    environment["GLM5_MINDSTUDIO_CONFIG"] = str(config_path.resolve())
    environment_overrides["GLM5_MINDSTUDIO_CONFIG"] = environment[
        "GLM5_MINDSTUDIO_CONFIG"
    ]
    command = _torchrun_command(
        root=root,
        config=config,
        endpoint=endpoint,
        run_directory=run_directory,
        checkpoint_path=checkpoint_path,
    )
    launch_contract = run_directory / "resolved_launch.sh"
    environment["GLM5_MINDSTUDIO_SHELL_PATH"] = str(launch_contract.resolve())
    environment_overrides["GLM5_MINDSTUDIO_SHELL_PATH"] = environment[
        "GLM5_MINDSTUDIO_SHELL_PATH"
    ]
    if dry_run:
        print(
            json.dumps(
                {
                    "command": command,
                    "endpoint_environment_keys": sorted(endpoint.environment),
                    "environment_overrides": environment_overrides,
                    "official_config": official_config,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return artifact_directory

    precheck_directory = _precheck_root(
        root,
        config,
        topology,
        owner=role,
        repeat=repeat,
    )
    stale_paths = [
        path
        for path in (run_directory, artifact_directory, precheck_directory)
        if path.exists()
    ]
    for stale in stale_paths:
        _assert_tree_not_active(stale)
    if report_directory.exists():
        _assert_tree_not_active(report_directory)
    for stale in stale_paths:
        archived = archive_previous_output(stale)
        print(f"Retry incomplete output; archived: {archived}", flush=True)
    if report_directory.exists():
        archived = archive_previous_output(report_directory)
        print(
            "Capture retry invalidated the previous comparison; archived: "
            f"{archived}",
            flush=True,
        )
    _invalidate_aggregate_report(
        root,
        config,
        label="stale MindStudio aggregate report",
    )
    run_directory.mkdir(parents=True, exist_ok=True)
    artifact_directory.mkdir(parents=True, exist_ok=True)
    input_contract.mkdir(parents=True, exist_ok=True)
    write_json(config_path, official_config)
    _write_launch_contract(
        launch_contract,
        command=command,
        environment=environment,
        endpoint=endpoint,
    )
    write_json(run_directory / "resolved_command.json", {"command": command})

    attempt = RunAttempt.start(
        run_directory,
        kind="mindstudio_official_capture",
        context={
            "workflow": config.workflow,
            "role": role,
            "topology": topology.name,
            "repeat": repeat,
            "fixture_generation_id": generation,
            "experiment_digest": digest,
        },
    )
    try:
        _run_process(
            command,
            root=root,
            environment=environment,
            log_path=runtime_log,
            context=attempt.log_context,
        )
        input_contract_summary = validate_runtime_input_contract(
            contract_directory=input_contract,
            plan=load_token_plan(token_plan_path),
            steps=config.training.steps,
            global_batch_size=config.training.global_batch_size,
            training_local_batch_size=config.training.local_batch_size,
            dp_world_size=topology.data_parallel_degree,
            context_parallel_degree=topology.context_parallel_degree,
            tensor_parallel_degree=topology.tensor_parallel_degree,
            pipeline_parallel_degree=topology.pipeline_parallel_degree,
            node_rank=endpoint.node_rank,
            num_processes_per_node=endpoint.num_processes_per_node,
        )
        if config.workflow == "config-check":
            _validate_rank_outputs(
                official_output,
                pattern="config_check_rank{rank}.zip",
                world_size=topology.world_size,
            )
        elif config.workflow == "migration":
            _validate_dump_outputs(
                official_output,
                steps=config.dump.steps,
                configured_ranks=config.dump.ranks,
                world_size=topology.world_size,
            )
        elif config.workflow == "compile":
            _validate_compile_outputs(
                official_output,
                world_size=topology.world_size,
                policy=config.compile.policy,
                children_depth=config.compile.children_depth,
            )
        else:
            _validate_monitor_outputs(
                official_output,
                configured_ranks=config.monitor.ranks,
                world_size=topology.world_size,
            )
        assert toolchain is not None
        assert toolchain_identity is not None
        assert toolchain_compatibility is not None
        manifest = {
            "schema": "torchtitan.glm5_2.mindstudio_artifact",
            "schema_version": 1,
            "workflow": config.workflow,
            "role": role,
            "repeat": repeat,
            "topology": asdict(topology),
            "experiment_digest": digest,
            "fixture_generation_id": generation,
            "fixture_checkpoint_sha256": fixture["checkpoint_sha256"],
            "fixture_token_plan": fixture["token_plan"],
            "input_contract": input_contract_summary,
            "official_output": "official",
            "official_files": output_index(official_output),
            "toolchain": toolchain,
            "toolchain_identity": toolchain_identity,
            "toolchain_compatibility": toolchain_compatibility,
            "runtime_log": str(runtime_log.resolve()),
        }
        write_json(artifact_directory / "manifest.json", manifest)
        write_json(
            artifact_directory / "complete.json",
            {
                "status": "completed",
                "experiment_digest": digest,
                "fixture_generation_id": generation,
                "attempt_id": attempt.attempt_id,
            },
        )
        attempt.update("completed", artifact=str(artifact_directory.resolve()))
    except BaseException as error:
        attempt.update("failed", error=repr(error))
        raise
    return artifact_directory


def _precheck_root(
    root: Path,
    config: MindStudioExperimentConfig,
    topology: ParallelTopology,
    *,
    owner: Role | Literal["compare"],
    repeat: int,
) -> Path:
    if owner == "compare":
        _, _, report_directory = _paths(
            root,
            config,
            topology,
            "reference",
            repeat,
        )
        base = report_directory
    else:
        base = root / config.artifact_root / config.storage_name / topology.slug
    return base / "precision_precheck" / f"{owner}-r{repeat}"


def _precheck_member(root: Path, *, step: int, rank: int) -> Path:
    return root / f"step{step}" / f"rank{rank}"


def _offline_result_is_complete(
    directory: Path,
    *,
    operation_digest: str,
    fixture_generation_id: str,
) -> bool:
    manifest_path = directory / "manifest.json"
    marker_path = directory / "complete.json"
    if not manifest_path.is_file() or not marker_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        relative_output = Path(manifest["official_output"])
        if relative_output.is_absolute() or ".." in relative_output.parts:
            return False
        official = (directory / relative_output).resolve()
        if not official.is_relative_to(directory.resolve()):
            return False
        current_index = output_index(official)
    except (KeyError, OSError, TypeError, ValueError, MindStudioArtifactError):
        return False
    return (
        manifest.get("schema")
        in {
            "torchtitan.glm5_2.mindstudio_precheck",
            "torchtitan.glm5_2.mindstudio_precheck_compare",
        }
        and manifest.get("schema_version") == 1
        and marker.get("status") == "completed"
        and marker.get("operation_digest") == operation_digest
        and marker.get("fixture_generation_id") == fixture_generation_id
        and manifest.get("operation_digest") == operation_digest
        and manifest.get("fixture_generation_id") == fixture_generation_id
        and manifest.get("official_files") == current_index
    )


def _require_precheck_capture(
    root: Path,
    config: MindStudioExperimentConfig,
    *,
    topology: ParallelTopology,
    role: Role,
    repeat: int,
) -> tuple[Path, dict[str, Any], str]:
    if config.workflow != "migration":
        raise ValueError("precision pre-check requires the migration workflow")
    if config.dump.level not in {"L1", "mix"}:
        raise ValueError(
            "precision pre-check requires an API-level L1 or mix capture"
        )
    _, artifact, _ = _paths(root, config, topology, role, repeat)
    fixture = _fixture_manifest(root, config)
    generation = str(fixture["generation_id"])
    digest = _experiment_digest(config, topology, role)
    if not artifact_is_complete(
        artifact,
        experiment_digest=digest,
        fixture_generation_id=generation,
    ):
        raise MindStudioArtifactError(
            f"official capture is missing, stale, or corrupt: {artifact}"
        )
    manifest = json.loads(
        (artifact / "manifest.json").read_text(encoding="utf-8")
    )
    if not isinstance(manifest.get("toolchain_compatibility"), dict) or not isinstance(
        manifest.get("toolchain_identity"), dict
    ):
        raise MindStudioArtifactError(
            "official capture predates toolchain-bound artifacts; recapture it: "
            f"{artifact}"
        )
    capture_identity = {
        "artifact_key": (
            f"{config.storage_name}/{topology.slug}/{role}-r{repeat}"
        ),
        "manifest_sha256": sha256_file(artifact / "manifest.json"),
        "complete_sha256": sha256_file(artifact / "complete.json"),
        "toolchain_identity": manifest["toolchain_identity"],
        "toolchain_compatibility": manifest["toolchain_compatibility"],
    }
    return artifact, capture_identity, generation


def _configured_dump_members(
    config: MindStudioExperimentConfig,
    topology: ParallelTopology,
) -> tuple[tuple[int, int], ...]:
    ranks = config.dump.ranks or tuple(range(topology.world_size))
    return tuple((step, rank) for step in config.dump.steps for rank in ranks)


def _portable_file_identity(path: Path) -> dict[str, str]:
    """Identify a synchronized input without embedding a host-local path."""

    if not path.is_file():
        raise FileNotFoundError(path)
    return {"name": path.name, "sha256": sha256_file(path)}


def _validated_resume_precheck_source(
    result: Path,
    details: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate a resume CSV against its completed pre-check generation."""

    if not result.is_file() or not details.is_file():
        raise ValueError(
            "--precheck-resume-csv requires both result and details CSV files"
        )
    for member in result.parents:
        manifest_path = member / "manifest.json"
        if not manifest_path.is_file():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (
                manifest.get("schema")
                != "torchtitan.glm5_2.mindstudio_precheck"
                or manifest.get("schema_version") != 1
            ):
                raise ValueError("resume source has an incompatible manifest schema")
            operation_digest = str(manifest["operation_digest"])
            generation = str(manifest["fixture_generation_id"])
            identity = manifest["identity"]
            relative_output = Path(manifest["official_output"])
            if relative_output.is_absolute() or ".." in relative_output.parts:
                raise ValueError("resume source official_output is unsafe")
            official = (member / relative_output).resolve()
            if not official.is_relative_to(member.resolve()):
                raise ValueError("resume source official_output escapes its member")
            if not result.resolve().is_relative_to(official):
                raise ValueError("resume result is outside the completed member")
            if not details.resolve().is_relative_to(official):
                raise ValueError("resume details are outside the completed member")
            if not isinstance(identity, dict) or not _offline_result_is_complete(
                member,
                operation_digest=operation_digest,
                fixture_generation_id=generation,
            ):
                raise ValueError("resume source is incomplete or its index is stale")
        except (KeyError, OSError, TypeError, MindStudioArtifactError) as error:
            raise ValueError(
                "--precheck-resume-csv has invalid provenance metadata"
            ) from error
        return identity, {
            "schema": manifest["schema"],
            "schema_version": manifest["schema_version"],
            "operation_digest": operation_digest,
            "fixture_generation_id": generation,
            "result": _portable_file_identity(result),
            "details": _portable_file_identity(details),
        }
    raise ValueError(
        "--precheck-resume-csv must come from a completed pre-check member "
        "with a provenance manifest"
    )


def reset_precheck_outputs(
    root: Path,
    config: MindStudioExperimentConfig,
    *,
    topologies: Sequence[ParallelTopology],
    owner: Role | Literal["compare"],
    repeat: int,
) -> None:
    paths = [
        _precheck_root(
            root,
            config,
            topology,
            owner=owner,
            repeat=repeat,
        )
        for topology in topologies
    ]
    if owner in {"reference", "candidate"}:
        paths.extend(
            _precheck_root(
                root,
                config,
                topology,
                owner="compare",
                repeat=repeat,
            )
            for topology in topologies
        )
        paths.extend(_aggregate_report_paths(root, config))
    for path in paths:
        _assert_tree_not_active(path)
    reset_output_generation(paths, label=f"MindStudio pre-check {owner}")


def run_precheck(
    root: Path,
    config: MindStudioExperimentConfig,
    *,
    topology: ParallelTopology,
    role: Role,
    repeat: int,
    device_ids: Sequence[int],
    num_splits: int | None,
    jit_compile: bool,
    save_error_data: bool,
    filter_api: bool,
    config_path: Path | None,
    resume_csv: Path | None = None,
    dry_run: bool = False,
) -> Path:
    """Run official API pre-check for every captured step/rank independently."""

    endpoint = _endpoint(config, role, topology)
    artifact, capture_identity, generation = _require_precheck_capture(
        root,
        config,
        topology=topology,
        role=role,
        repeat=repeat,
    )
    members = _configured_dump_members(config, topology)
    if resume_csv is not None and len(members) != 1:
        raise ValueError(
            "--precheck-resume-csv can be used only when exactly one "
            "step/rank member is selected"
        )
    resolved_config = config_path.resolve() if config_path is not None else None
    config_identity = _optional_file_identity(resolved_config)
    resolved_resume = resume_csv.resolve() if resume_csv is not None else None
    resume_details: Path | None = None
    resume_identity: dict[str, Any] | None = None
    resume_payloads: tuple[bytes, bytes] | None = None
    resume_source_identity: dict[str, Any] | None = None
    resume_source_provenance: dict[str, Any] | None = None
    if resolved_resume is not None:
        if not resolved_resume.name.startswith("accuracy_checking_result_"):
            raise ValueError(
                "--precheck-resume-csv must name an "
                "accuracy_checking_result_*.csv file"
            )
        resume_details = resolved_resume.with_name(
            resolved_resume.name.replace(
                "accuracy_checking_result_",
                "accuracy_checking_details_",
                1,
            )
        )
        resume_source_identity, resume_source_provenance = (
            _validated_resume_precheck_source(resolved_resume, resume_details)
        )
        resume_identity = resume_source_provenance
        resume_payloads = (
            resolved_resume.read_bytes(),
            resume_details.read_bytes(),
        )
    if dry_run:
        toolchain = {"planned_from_capture": True}
        toolchain_identity = capture_identity["toolchain_identity"]
        toolchain_compatibility = capture_identity[
            "toolchain_compatibility"
        ]
    else:
        toolchain = _accuracy_toolchain_metadata(endpoint.device_type)
        toolchain_identity = _comparison_toolchain_identity(toolchain)
        toolchain_identity["execution_environment"] = (
            _capture_execution_identity(root, endpoint.device_type)
        )
        toolchain_compatibility = _capture_toolchain_compatibility(
            toolchain_identity
        )
        if (
            toolchain_identity != capture_identity["toolchain_identity"]
            or toolchain_compatibility
            != capture_identity["toolchain_compatibility"]
        ):
            raise MindStudioArtifactError(
                "precision pre-check toolchain or source identity does not "
                "match the selected capture; use the pinned environment and "
                "source revision that produced the capture"
            )
    precheck_root = _precheck_root(
        root,
        config,
        topology,
        owner=role,
        repeat=repeat,
    )
    environment = os.environ.copy()
    environment.update(endpoint.environment)
    environment_overrides = {
        endpoint.visible_devices_env: endpoint.visible_devices,
        "TORCHTITAN_DEVICE": endpoint.torchtitan_device,
    }
    environment.update(environment_overrides)

    dependents_invalidated = False
    for step, rank in members:
        dump_json = artifact / "official" / f"step{step}" / f"rank{rank}" / "dump.json"
        if not dump_json.is_file():
            raise MindStudioArtifactError(f"msProbe dump is missing: {dump_json}")
        identity = {
            "operation": "precision_precheck",
            "experiment": config.identity,
            "topology": asdict(topology),
            "role": role,
            "endpoint_device_type": endpoint.device_type,
            "repeat": repeat,
            "step": step,
            "rank": rank,
            "fixture_generation_id": generation,
            "capture": capture_identity,
            "dump_json_sha256": sha256_file(dump_json),
            "toolchain": toolchain_identity,
            "options": {
                "device_ids": list(device_ids),
                "num_splits": num_splits,
                "jit_compile": jit_compile,
                "save_error_data": save_error_data,
                "filter_api": filter_api,
                "config": config_identity,
                "resume": resume_identity,
            },
        }
        if resume_source_identity is not None:
            expected_source_identity = json.loads(json.dumps(identity))
            expected_source_identity["options"]["resume"] = None
            actual_source_identity = json.loads(
                json.dumps(resume_source_identity)
            )
            actual_options = actual_source_identity.get("options")
            if isinstance(actual_options, dict):
                actual_options["resume"] = None
            if actual_source_identity != expected_source_identity:
                raise ValueError(
                    "--precheck-resume-csv belongs to another capture, "
                    "configuration, toolchain, step, or rank"
                )
        operation_digest = config_digest(identity, length=64)
        member = _precheck_member(precheck_root, step=step, rank=rank)
        if _offline_result_is_complete(
            member,
            operation_digest=operation_digest,
            fixture_generation_id=generation,
        ):
            print(f"Skip completed precision pre-check: {member}", flush=True)
            continue
        official_output = member / "official"
        staged_resume = (
            official_output / resolved_resume.name
            if resolved_resume is not None
            else None
        )
        command = precheck_command(
            api_info=dump_json,
            output=official_output,
            device_ids=device_ids,
            num_splits=num_splits,
            jit_compile=jit_compile,
            save_error_data=save_error_data,
            filter_api=filter_api,
            config_path=resolved_config,
            resume_csv=staged_resume,
        )
        if dry_run:
            print(
                json.dumps(
                    {
                        "command": command,
                        "endpoint_environment_keys": sorted(
                            endpoint.environment
                        ),
                        "environment_overrides": environment_overrides,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            continue
        if not dependents_invalidated:
            _invalidate_precheck_dependents(
                root,
                config,
                topology=topology,
                repeat=repeat,
            )
            dependents_invalidated = True
        if member.exists():
            _assert_tree_not_active(member)
            archived = archive_previous_output(member)
            print(f"Retry incomplete pre-check; archived: {archived}", flush=True)
        if staged_resume is not None:
            official_output.mkdir(parents=True, exist_ok=True)
            assert resume_details is not None
            assert resume_payloads is not None
            staged_resume.write_bytes(resume_payloads[0])
            (official_output / resume_details.name).write_bytes(resume_payloads[1])
        write_compare_invocation(member / "invocation.json", command)
        attempt = RunAttempt.start(
            member,
            kind="mindstudio_precision_precheck",
            state_name="precheck_state.json",
            context={
                "role": role,
                "device_type": endpoint.device_type,
                "topology": topology.name,
                "repeat": repeat,
                "step": step,
                "rank": rank,
                "fixture_generation_id": generation,
                "operation_digest": operation_digest,
            },
        )
        try:
            _run_process(
                command,
                root=root,
                environment=environment,
                log_path=member / "runtime.log",
                context=attempt.log_context,
            )
            result_files = sorted(
                official_output.rglob("accuracy_checking_result_*.csv")
            )
            details_files = sorted(
                official_output.rglob("accuracy_checking_details_*.csv")
            )
            if not result_files or not details_files:
                raise MindStudioArtifactError(
                    "precision pre-check did not emit result/details CSV files: "
                    f"{official_output}"
                )
            summarize_official_results(official_output)
            relative_details = [
                path.relative_to(official_output).as_posix()
                for path in details_files
            ]
            write_json(
                member / "manifest.json",
                {
                    "schema": "torchtitan.glm5_2.mindstudio_precheck",
                    "schema_version": 1,
                    "operation_digest": operation_digest,
                    "fixture_generation_id": generation,
                    "identity": identity,
                    "toolchain": toolchain,
                    "toolchain_identity": toolchain_identity,
                    "toolchain_compatibility": toolchain_compatibility,
                    "resume_source": resume_source_provenance,
                    "official_output": "official",
                    "official_files": output_index(official_output),
                    "details_files": relative_details,
                    "automatic_compare_compatible": len(details_files) == 1,
                    "automatic_compare_note": (
                        "Automatic --precheck-compare requires exactly one "
                        "details CSV per endpoint. Multi-device pre-check output "
                        "is preserved, but must be reduced explicitly or rerun "
                        "with one device before automatic comparison."
                    ),
                    "runtime_log": str((member / "runtime.log").resolve()),
                },
            )
            write_json(
                member / "complete.json",
                {
                    "status": "completed",
                    "operation_digest": operation_digest,
                    "fixture_generation_id": generation,
                    "attempt_id": attempt.attempt_id,
                },
            )
            attempt.update("completed", output=str(member.resolve()))
        except BaseException as error:
            attempt.update("failed", error=repr(error))
            raise
    return precheck_root


def _load_current_precheck_member(
    member: Path,
    *,
    capture_identity: dict[str, Any],
    dump_json_sha256: str,
    fixture_generation_id: str,
) -> tuple[dict[str, Any], Path]:
    try:
        manifest = json.loads((member / "manifest.json").read_text(encoding="utf-8"))
        operation_digest = str(manifest["operation_digest"])
        identity = manifest["identity"]
    except (KeyError, OSError, TypeError, ValueError) as error:
        raise MindStudioArtifactError(
            f"precision pre-check is missing or corrupt: {member}"
        ) from error
    if not _offline_result_is_complete(
        member,
        operation_digest=operation_digest,
        fixture_generation_id=fixture_generation_id,
    ):
        raise MindStudioArtifactError(
            f"precision pre-check is incomplete or corrupt: {member}"
        )
    if (
        identity.get("capture") != capture_identity
        or identity.get("dump_json_sha256") != dump_json_sha256
    ):
        raise MindStudioArtifactError(
            f"precision pre-check belongs to a stale capture: {member}"
        )
    try:
        details = find_precheck_details(member / "official")
    except MindStudioArtifactError as error:
        raise MindStudioArtifactError(
            "automatic --precheck-compare requires exactly one "
            "accuracy_checking_details CSV per endpoint; multi-device "
            "standalone output is preserved, but must be reduced explicitly "
            "or rerun with one device: "
            f"{member}"
        ) from error
    return manifest, details


def _write_precheck_report(
    compare_root: Path,
    *,
    config: MindStudioExperimentConfig,
    topology: ParallelTopology,
    repeat: int,
    fixture_generation_id: str,
) -> Path:
    entries: list[dict[str, Any]] = []
    for step, rank in _configured_dump_members(config, topology):
        member = _precheck_member(compare_root, step=step, rank=rank)
        summary_path = member / "official" / "official_summary.json"
        if not summary_path.is_file():
            raise MindStudioArtifactError(
                f"pre-check comparison summary is missing: {summary_path}"
            )
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        official = member / "official"
        result_files = sorted(
            official.rglob("api_precision_compare_result_*.csv")
        )
        details_files = sorted(
            official.rglob("api_precision_compare_details_*.csv")
        )
        if len(result_files) != 1 or len(details_files) != 1:
            raise MindStudioArtifactError(
                "pre-check report requires one final result and details CSV "
                f"per step/rank: {member}"
            )
        entries.append(
            {
                "step": step,
                "rank": rank,
                "verdict": summary.get("verdict", "unparsed"),
                "status_counts": summary.get("status_counts", {}),
                "summary": summary_path.relative_to(compare_root).as_posix(),
                "result": result_files[0].relative_to(compare_root).as_posix(),
                "details": details_files[0].relative_to(compare_root).as_posix(),
                "complete_sha256": sha256_file(member / "complete.json"),
            }
        )
    index = {
        "schema": "torchtitan.glm5_2.mindstudio_precheck_index",
        "schema_version": 1,
        "experiment": config.storage_name,
        "topology": topology.name,
        "repeat": repeat,
        "fixture_generation_id": fixture_generation_id,
        "meaning": (
            "Each endpoint first runs an API unit pre-check against a CPU "
            "higher-precision reference. The final table compares NPU-vs-CPU "
            "error with GPU-vs-CPU error using msProbe. It does not compare "
            "the two complete training processes directly."
        ),
        "entries": entries,
    }
    write_json(compare_root / "precheck_index.json", index)
    rows = []
    for entry in entries:
        rows.append(
            "<tr>"
            f"<td>{entry['step']}</td>"
            f"<td>{entry['rank']}</td>"
            f"<td>{escape(str(entry['verdict']))}</td>"
            f"<td><a href=\"{escape(entry['result'])}\">API result</a></td>"
            f"<td><a href=\"{escape(entry['details'])}\">metric details</a></td>"
            f"<td><a href=\"{escape(entry['summary'])}\">summary JSON</a></td>"
            "</tr>"
        )
    html = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>MindStudio precision pre-check</title>
<style>
body {{ font-family: sans-serif; margin: 2rem; color: #1f2937; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #d1d5db; padding: .55rem; text-align: left; }}
th {{ background: #f3f4f6; }}
code {{ background: #f3f4f6; padding: .1rem .25rem; }}
</style></head><body>
<h1>MindStudio precision pre-check</h1>
<p><strong>Meaning:</strong> Each endpoint first runs an API unit pre-check
against a CPU higher-precision reference. This table then compares
NPU-vs-CPU error with GPU-vs-CPU error using the official
<code>api_precision_compare</code> standard. It does <strong>not</strong> compare
the two complete training processes directly.</p>
<p>Topology: <code>{topology}</code>; repeat: {repeat}; fixture generation:
<code>{generation}</code>.</p>
<table><thead><tr><th>Step</th><th>Rank</th><th>Verdict</th>
<th>API result</th><th>Metric details</th><th>Summary</th></tr></thead>
<tbody>{rows}</tbody></table></body></html>
""".format(
        topology=escape(topology.name),
        repeat=repeat,
        generation=escape(fixture_generation_id),
        rows="\n".join(rows),
    )
    report = compare_root / "precheck_report.html"
    temporary = report.with_suffix(".html.tmp")
    temporary.write_text(html, encoding="utf-8")
    temporary.replace(report)
    return report


def compare_prechecks(
    root: Path,
    config: MindStudioExperimentConfig,
    *,
    topology: ParallelTopology,
    repeat: int,
    dry_run: bool = False,
) -> Path:
    """Compare current NPU and GPU pre-check details with the official CLI."""

    role_by_device: dict[str, Role] = {}
    for role in ("reference", "candidate"):
        endpoint = _endpoint(config, role, topology)
        if endpoint.device_type in role_by_device:
            raise ValueError(
                "pre-check comparison requires one GPU and one NPU endpoint"
            )
        role_by_device[endpoint.device_type] = role
    if set(role_by_device) != {"cuda", "npu"}:
        raise ValueError(
            "pre-check comparison requires one GPU and one NPU endpoint"
        )

    capture_inputs: dict[Role, tuple[Path, dict[str, Any], str]] = {}
    for role in ("reference", "candidate"):
        capture_inputs[role] = _require_precheck_capture(
            root,
            config,
            topology=topology,
            role=role,
            repeat=repeat,
        )
    generation = capture_inputs["reference"][2]
    if capture_inputs["candidate"][2] != generation:
        raise MindStudioArtifactError(
            "reference and candidate captures use different fixture generations"
        )
    if (
        capture_inputs["reference"][1]["toolchain_compatibility"]
        != capture_inputs["candidate"][1]["toolchain_compatibility"]
    ):
        raise MindStudioArtifactError(
            "reference and candidate pre-check captures used incompatible "
            "msProbe toolchains"
        )
    compare_root = _precheck_root(
        root,
        config,
        topology,
        owner="compare",
        repeat=repeat,
    )
    toolchain = _accuracy_toolchain_metadata("compare")
    toolchain_identity = _comparison_toolchain_identity(toolchain)
    toolchain_compatibility = _msprobe_toolchain_compatibility(
        toolchain_identity
    )
    report_invalidated = False
    for step, rank in _configured_dump_members(config, topology):
        details_by_role: dict[Role, Path] = {}
        precheck_manifests: dict[Role, dict[str, Any]] = {}
        for role in ("reference", "candidate"):
            artifact, capture_identity, _ = capture_inputs[role]
            dump_json = (
                artifact / "official" / f"step{step}" / f"rank{rank}" / "dump.json"
            )
            member_root = _precheck_root(
                root,
                config,
                topology,
                owner=role,
                repeat=repeat,
            )
            precheck_manifest, details = _load_current_precheck_member(
                _precheck_member(member_root, step=step, rank=rank),
                capture_identity=capture_identity,
                dump_json_sha256=sha256_file(dump_json),
                fixture_generation_id=generation,
            )
            precheck_identity = precheck_manifest.get("toolchain_identity")
            precheck_compatibility = precheck_manifest.get(
                "toolchain_compatibility"
            )
            if not isinstance(precheck_identity, dict) or not isinstance(
                precheck_compatibility, dict
            ):
                raise MindStudioArtifactError(
                    "precision pre-check predates strict toolchain provenance; "
                    f"rerun it: {member_root}"
                )
            if (
                precheck_compatibility
                != capture_identity["toolchain_compatibility"]
            ):
                raise MindStudioArtifactError(
                    "precision pre-check toolchain does not match its capture: "
                    f"{member_root}"
                )
            if (
                _msprobe_toolchain_compatibility(precheck_identity)
                != toolchain_compatibility
            ):
                raise MindStudioArtifactError(
                    "compare-side msProbe toolchain is incompatible with the "
                    f"{role} pre-check"
                )
            precheck_manifests[role] = precheck_manifest
            details_by_role[role] = details
        if (
            precheck_manifests["reference"]["toolchain_compatibility"]
            != precheck_manifests["candidate"]["toolchain_compatibility"]
        ):
            raise MindStudioArtifactError(
                "reference and candidate pre-checks used incompatible "
                "toolchains or source revisions"
            )
        npu_role = role_by_device["npu"]
        gpu_role = role_by_device["cuda"]
        identity = {
            "operation": "precision_precheck_compare",
            "experiment": config.identity,
            "topology": asdict(topology),
            "repeat": repeat,
            "step": step,
            "rank": rank,
            "fixture_generation_id": generation,
            "role_by_device": role_by_device,
            "toolchain": toolchain_identity,
            "inputs": {
                role: {
                    "manifest_sha256": sha256_file(
                        _precheck_member(
                            _precheck_root(
                                root,
                                config,
                                topology,
                                owner=role,
                                repeat=repeat,
                            ),
                            step=step,
                            rank=rank,
                        )
                        / "manifest.json"
                    ),
                    "details_sha256": sha256_file(details_by_role[role]),
                }
                for role in ("reference", "candidate")
            },
        }
        operation_digest = config_digest(identity, length=64)
        member = _precheck_member(compare_root, step=step, rank=rank)
        if _offline_result_is_complete(
            member,
            operation_digest=operation_digest,
            fixture_generation_id=generation,
        ):
            print(f"Skip completed pre-check comparison: {member}", flush=True)
            continue
        official_output = member / "official"
        command = precheck_compare_command(
            npu_details=details_by_role[npu_role],
            gpu_details=details_by_role[gpu_role],
            output=official_output,
        )
        if dry_run:
            print(json.dumps({"command": command}, indent=2))
            continue
        if not report_invalidated:
            _assert_tree_not_active(compare_root)
            reset_output_generation(
                (
                    compare_root / "precheck_report.html",
                    compare_root / "precheck_index.json",
                    *_aggregate_report_paths(root, config),
                ),
                label="stale MindStudio pre-check report",
            )
            report_invalidated = True
        if member.exists():
            _assert_tree_not_active(member)
            archived = archive_previous_output(member)
            print(
                f"Retry incomplete pre-check comparison; archived: {archived}",
                flush=True,
            )
        write_compare_invocation(member / "invocation.json", command)
        attempt = RunAttempt.start(
            member,
            kind="mindstudio_precision_precheck_compare",
            state_name="precheck_state.json",
            context={
                "topology": topology.name,
                "repeat": repeat,
                "step": step,
                "rank": rank,
                "fixture_generation_id": generation,
                "operation_digest": operation_digest,
            },
        )
        try:
            _run_process(
                command,
                root=root,
                environment=os.environ.copy(),
                log_path=member / "runtime.log",
                context=attempt.log_context,
            )
            result_files = sorted(
                official_output.rglob("api_precision_compare_result_*.csv")
            )
            details_files = sorted(
                official_output.rglob("api_precision_compare_details_*.csv")
            )
            if not result_files or not details_files:
                raise MindStudioArtifactError(
                    "pre-check comparison did not emit result/details CSV files: "
                    f"{official_output}"
                )
            summarize_official_results(official_output)
            write_json(
                member / "manifest.json",
                {
                    "schema": "torchtitan.glm5_2.mindstudio_precheck_compare",
                    "schema_version": 1,
                    "operation_digest": operation_digest,
                    "fixture_generation_id": generation,
                    "identity": identity,
                    "toolchain": toolchain,
                    "toolchain_identity": toolchain_identity,
                    "toolchain_compatibility": toolchain_compatibility,
                    "official_output": "official",
                    "official_files": output_index(official_output),
                    "runtime_log": str((member / "runtime.log").resolve()),
                },
            )
            write_json(
                member / "complete.json",
                {
                    "status": "completed",
                    "operation_digest": operation_digest,
                    "fixture_generation_id": generation,
                    "attempt_id": attempt.attempt_id,
                },
            )
            attempt.update("completed", output=str(member.resolve()))
        except BaseException as error:
            attempt.update("failed", error=repr(error))
            raise
    if not dry_run:
        report = _write_precheck_report(
            compare_root,
            config=config,
            topology=topology,
            repeat=repeat,
            fixture_generation_id=generation,
        )
        print(f"Pre-check HTML report: {report}", flush=True)
    return compare_root


def compare_official(
    root: Path,
    config: MindStudioExperimentConfig,
    *,
    topology: ParallelTopology,
    repeat: int,
    fuzzy_match: bool,
    data_mapping: Path | None,
    cell_mapping: Path | None,
    diff_analysis: bool,
    tensor_log: bool,
    xlsx: bool,
    force: bool,
    dry_run: bool = False,
) -> dict[str, Any]:
    _validate_compare_options(
        config,
        topology,
        fuzzy_match=fuzzy_match,
        data_mapping=data_mapping,
        cell_mapping=cell_mapping,
        diff_analysis=diff_analysis,
        tensor_log=tensor_log,
        xlsx=xlsx,
    )
    reference_run, reference_artifact, report_directory = _paths(
        root, config, topology, "reference", repeat
    )
    candidate_run, candidate_artifact, _ = _paths(
        root, config, topology, "candidate", repeat
    )
    del reference_run, candidate_run
    fixture = _fixture_manifest(root, config)
    generation = str(fixture["generation_id"])
    selected_artifacts: dict[Role, Path] = {"candidate": candidate_artifact}
    if config.workflow != "compile":
        selected_artifacts["reference"] = reference_artifact
    artifact_inputs: dict[str, Any] = {}
    capture_compatibility: dict[str, dict[str, Any]] = {}
    for selected_role, artifact in selected_artifacts.items():
        digest = _experiment_digest(config, topology, selected_role)
        if not artifact_is_complete(
            artifact,
            experiment_digest=digest,
            fixture_generation_id=generation,
        ):
            raise MindStudioArtifactError(
                f"official capture is missing, stale, or corrupt: {artifact}"
            )
        manifest = json.loads(
            (artifact / "manifest.json").read_text(encoding="utf-8")
        )
        input_contract = manifest.get("input_contract")
        if (
            not isinstance(input_contract, dict)
            or input_contract.get("valid") is not True
        ):
            raise MindStudioArtifactError(
                "official capture predates validated runtime input contracts; "
                f"recapture it: {artifact}"
            )
        compatibility = manifest.get("toolchain_compatibility")
        if not isinstance(compatibility, dict):
            raise MindStudioArtifactError(
                "official capture predates toolchain-bound artifacts; "
                f"recapture it: {artifact}"
            )
        capture_compatibility[selected_role] = compatibility
        artifact_inputs[selected_role] = {
            "manifest_sha256": sha256_file(artifact / "manifest.json"),
            "complete_sha256": sha256_file(artifact / "complete.json"),
            "toolchain_compatibility": compatibility,
        }
    if (
        "reference" in capture_compatibility
        and capture_compatibility["reference"]
        != capture_compatibility["candidate"]
    ):
        raise MindStudioArtifactError(
            "reference and candidate captures used incompatible msProbe "
            "toolchains; install the same pinned source revision and recapture"
        )
    comparison_toolchain = _accuracy_toolchain_metadata("compare")
    comparison_toolchain_identity = _comparison_toolchain_identity(
        comparison_toolchain
    )
    comparison_msprobe_compatibility = _msprobe_toolchain_compatibility(
        comparison_toolchain_identity
    )
    for selected_role, compatibility in capture_compatibility.items():
        capture_msprobe_compatibility = {
            key: value
            for key, value in compatibility.items()
            if key != "project_sources"
        }
        if capture_msprobe_compatibility != comparison_msprobe_compatibility:
            raise MindStudioArtifactError(
                "compare-side msProbe toolchain is incompatible with the "
                f"{selected_role} capture"
            )
    comparison_identity = {
        "workflow": config.workflow,
        "topology": asdict(topology),
        "repeat": repeat,
        "fixture_generation_id": generation,
        "artifacts": artifact_inputs,
        "toolchain": comparison_toolchain_identity,
        "options": {
            "fuzzy_match": fuzzy_match,
            "data_mapping": _optional_file_identity(data_mapping),
            "cell_mapping": _optional_file_identity(cell_mapping),
            "diff_analysis": diff_analysis,
            "tensor_log": tensor_log,
            "xlsx": xlsx,
        },
    }
    comparison_digest = config_digest(comparison_identity, length=64)
    compare_directory = report_directory / "official_compare"
    comparison_state_path = compare_directory / "comparison_state.json"
    assert_run_not_active(
        compare_directory,
        state_name="compare_run_state.json",
    )
    if dry_run:
        commands: list[list[str]] = []
        if config.workflow == "config-check":
            for rank in range(topology.world_size):
                commands.append(
                    config_check_compare_command(
                        golden_pack=(
                            reference_artifact
                            / "official"
                            / f"config_check_rank{rank}.zip"
                        ),
                        target_pack=(
                            candidate_artifact
                            / "official"
                            / f"config_check_rank{rank}.zip"
                        ),
                        output=compare_directory / f"rank{rank}",
                    )
                )
        elif config.workflow == "migration":
            multiple_steps = len(config.dump.steps) > 1
            for step in config.dump.steps:
                commands.append(
                    compare_command(
                        target=find_dump_compare_input(candidate_artifact, step),
                        golden=find_dump_compare_input(reference_artifact, step),
                        output=(
                            compare_directory / f"step{step}"
                            if multiple_steps
                            else compare_directory
                        ),
                        fuzzy_match=fuzzy_match,
                        data_mapping=data_mapping,
                        cell_mapping=cell_mapping,
                        diff_analysis=diff_analysis,
                        tensor_log=tensor_log,
                        xlsx=xlsx,
                    )
                )
        print(
            json.dumps(
                {
                    "operation": "official_compare",
                    "workflow": config.workflow,
                    "topology": topology.name,
                    "output": str(compare_directory.resolve()),
                    "commands": commands,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return {
            "topology": topology.name,
            "verdict": "planned",
            "status_counts": {},
            "official_files": [],
            "official_result": str(
                (compare_directory / "official_summary.json").resolve()
            ),
            "runtime_log": str((compare_directory / "runtime.log").resolve()),
        }
    if (
        not force
        and (compare_directory / "official_summary.json").is_file()
        and _comparison_is_current(comparison_state_path, comparison_digest)
    ):
        summary = json.loads(
            (compare_directory / "official_summary.json").read_text(encoding="utf-8")
        )
        return {
            "topology": topology.name,
            "verdict": summary.get("verdict", "unparsed"),
            "status_counts": summary.get("status_counts", {}),
            "official_files": summary.get("official_files", []),
            "official_result": str(
                (compare_directory / "official_summary.json").resolve()
            ),
            "runtime_log": str((compare_directory / "runtime.log").resolve()),
        }
    _invalidate_aggregate_report(
        root,
        config,
        label="stale MindStudio aggregate report",
    )
    if force and compare_directory.exists():
        reset_output_generation(
            (compare_directory,),
            label="official compare",
            active_run_directories=(compare_directory,),
        )
    elif compare_directory.exists():
        assert_run_not_active(
            compare_directory,
            state_name="compare_run_state.json",
        )
        archive_previous_output(compare_directory)
    compare_directory.mkdir(parents=True, exist_ok=True)
    runtime_log = compare_directory / "runtime.log"
    attempt = RunAttempt.start(
        compare_directory,
        kind="mindstudio_official_compare",
        state_name="compare_run_state.json",
        context={
            "workflow": config.workflow,
            "topology": topology.name,
            "repeat": repeat,
            "fixture_generation_id": generation,
            "comparison_digest": comparison_digest,
        },
    )
    try:
        if config.workflow == "config-check":
            for rank in range(topology.world_size):
                rank_directory = compare_directory / f"rank{rank}"
                command = config_check_compare_command(
                    golden_pack=(
                        reference_artifact
                        / "official"
                        / f"config_check_rank{rank}.zip"
                    ),
                    target_pack=(
                        candidate_artifact
                        / "official"
                        / f"config_check_rank{rank}.zip"
                    ),
                    output=rank_directory,
                )
                write_compare_invocation(
                    rank_directory / "invocation.json", command
                )
                _run_process(
                    command,
                    root=root,
                    environment=os.environ.copy(),
                    log_path=rank_directory / "runtime.log",
                    context={
                        **attempt.log_context,
                        "Rank": rank,
                    },
                )
                summarize_official_results(rank_directory)
            runtime_log.write_text(
                "Official ConfigChecker comparison completed for ranks: "
                + ", ".join(str(rank) for rank in range(topology.world_size))
                + "\n",
                encoding="utf-8",
            )
        elif config.workflow == "migration":
            multiple_steps = len(config.dump.steps) > 1
            for step in config.dump.steps:
                target = find_dump_compare_input(candidate_artifact, step)
                golden = find_dump_compare_input(reference_artifact, step)
                step_directory = (
                    compare_directory / f"step{step}"
                    if multiple_steps
                    else compare_directory
                )
                command = compare_command(
                    target=target,
                    golden=golden,
                    output=step_directory,
                    fuzzy_match=fuzzy_match,
                    data_mapping=data_mapping,
                    cell_mapping=cell_mapping,
                    diff_analysis=diff_analysis,
                    tensor_log=tensor_log,
                    xlsx=xlsx,
                )
                write_compare_invocation(
                    step_directory / "invocation.json", command
                )
                _run_process(
                    command,
                    root=root,
                    environment=os.environ.copy(),
                    log_path=step_directory / "runtime.log",
                    context={
                        **attempt.log_context,
                        "Step": step,
                        "Target": target,
                        "Golden": golden,
                    },
                )
            if multiple_steps:
                runtime_log.write_text(
                    "Official msProbe comparison completed for steps: "
                    + ", ".join(str(step) for step in config.dump.steps)
                    + "\n",
                    encoding="utf-8",
                )
        elif config.workflow == "compile":
            _validate_compile_outputs(
                candidate_artifact / "official",
                world_size=topology.world_size,
                policy=config.compile.policy,
                children_depth=config.compile.children_depth,
            )
            rank_csv = _compile_rank_csv_files(
                candidate_artifact, topology.world_size
            )
            aggregate_compile_csv(
                rank_csv,
                output=compare_directory / "compile_all_ranks.csv",
                scenario=f"{topology.name}-{config.compile.backend}",
            )
            runtime_log.write_text(
                "No second execution is required. PrecisionChecker compared "
                "eager and compiled modules in the candidate training process.\n",
                encoding="utf-8",
            )
        else:
            write_json(
                compare_directory / "monitor_index.json",
                {
                    "schema": "torchtitan.glm5_2.mindstudio_monitor_index",
                    "schema_version": 1,
                    "reference": artifact_inputs["reference"],
                    "candidate": artifact_inputs["candidate"],
                    "note": (
                        "Monitor V2 emits per-rank CSV time series and does not "
                        "define an official cross-device pass/fail comparator. "
                        "Use these traces to select a suspicious step, then run "
                        "the dump/compare workflow at that step."
                    ),
                },
            )
            runtime_log.write_text(
                "Indexed reference and candidate TrainerMonitorV2 captures. "
                "No unofficial numerical verdict was computed.\n",
                encoding="utf-8",
            )
        summary = summarize_official_results(compare_directory)
        write_json(
            comparison_state_path,
            {
                "schema": "torchtitan.glm5_2.mindstudio_comparison",
                "schema_version": 1,
                "comparison_digest": comparison_digest,
                "identity": comparison_identity,
                "toolchain": comparison_toolchain,
                "output_files": _comparison_output_index(compare_directory),
            },
        )
        attempt.update(
            "completed",
            output=str(compare_directory.resolve()),
        )
    except BaseException as error:
        attempt.update("failed", error=repr(error))
        raise
    return {
        "topology": topology.name,
        "verdict": summary["verdict"],
        "status_counts": summary["status_counts"],
        "official_files": summary["official_files"],
        "official_result": str((compare_directory / "official_summary.json").resolve()),
        "runtime_log": str(runtime_log.resolve()),
    }


def _topology_registry() -> dict[str, ParallelTopology]:
    return {
        name: topology
        for name, topology in standard_topologies().items()
        if topology.world_size <= 8
    }


def run_mindstudio_cli(
    base_config: MindStudioExperimentConfig,
    script_path: str,
) -> None:
    registry = _topology_registry()
    parser = argparse.ArgumentParser(
        description="Official MindStudio accuracy workflow for TorchTitan GLM5"
    )
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--data", action="store_true")
    actions.add_argument("--capture", choices=("reference", "candidate"))
    actions.add_argument("--precheck", choices=("reference", "candidate"))
    actions.add_argument("--precheck-compare", action="store_true")
    actions.add_argument("--graph-visualize", action="store_true")
    actions.add_argument("--compare", action="store_true")
    actions.add_argument("--doctor", action="store_true")
    actions.add_argument("--list-topologies", action="store_true")
    parser.add_argument("--topology", default="single")
    parser.add_argument("--topologies")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--data-device", choices=("cuda", "npu"))
    parser.add_argument(
        "--doctor-device",
        choices=("cuda", "npu"),
        help="readiness scope for a GPU or NPU capture server",
    )
    parser.add_argument("--dump-task", choices=("statistics", "tensor"))
    parser.add_argument("--level", choices=("L0", "L1", "mix"))
    parser.add_argument("--dump-step", type=int)
    parser.add_argument(
        "--dump-steps",
        help="comma-separated zero-based msProbe steps",
    )
    parser.add_argument(
        "--dump-ranks",
        help="comma-separated global ranks; empty configuration captures all",
    )
    parser.add_argument(
        "--scope",
        action="append",
        default=None,
        help="repeat to restrict the official module/API capture scope",
    )
    parser.add_argument(
        "--module-or-api",
        action="append",
        default=None,
        help="repeat to select an official module/API list entry",
    )
    parser.add_argument(
        "--tensor-list",
        action="append",
        default=None,
        help="repeat to select a statistics tensor category",
    )
    parser.add_argument(
        "--data-mode",
        help="comma-separated msProbe data_mode values",
    )
    parser.add_argument("--summary-mode", choices=("statistics", "md5"))
    parser.add_argument("--async-dump", action="store_true")
    parser.add_argument("--no-extra-info", action="store_true")
    parser.add_argument("--backend")
    parser.add_argument("--no-dump-graphs", action="store_true")
    parser.add_argument("--no-capture-input", action="store_true")
    parser.add_argument(
        "--compile-policy",
        choices=("glm5-block", "children"),
    )
    parser.add_argument("--children-depth", type=int)
    parser.add_argument("--fuzzy-match", action="store_true")
    parser.add_argument("--data-mapping", type=Path)
    parser.add_argument("--cell-mapping", type=Path)
    parser.add_argument("--diff-analysis", action="store_true")
    parser.add_argument("--tensor-log", action="store_true")
    parser.add_argument("--xlsx", action="store_true")
    parser.add_argument("--graph-overflow-check", action="store_true")
    parser.add_argument("--graph-progress-log", action="store_true")
    parser.add_argument(
        "--precheck-device-ids",
        default="0",
        help="comma-separated device IDs used by acc_check (default: 0)",
    )
    parser.add_argument(
        "--precheck-splits",
        type=int,
        help="use multi_acc_check with this many API splits (1..64)",
    )
    parser.add_argument("--precheck-jit-compile", action="store_true")
    parser.add_argument("--precheck-save-error-data", action="store_true")
    parser.add_argument("--precheck-filter-api", action="store_true")
    parser.add_argument("--precheck-config", type=Path)
    parser.add_argument("--precheck-resume-csv", type=Path)
    parser.add_argument("--training-steps", type=int)
    parser.add_argument(
        "--monitor-ranks",
        help="comma-separated ranks monitored by TrainerMonitorV2",
    )
    parser.add_argument("--monitor-start-step", type=int)
    parser.add_argument("--monitor-stop-step", type=int)
    parser.add_argument("--monitor-step-interval", type=int)
    parser.add_argument("--monitor-record-steps", type=int)
    parser.add_argument("--monitor-collect-times", type=int)
    parser.add_argument(
        "--monitor-ops",
        help="comma-separated Monitor V2 statistics",
    )
    parser.add_argument("--monitor-module", action="store_true")
    parser.add_argument("--no-monitor-weight-grad", action="store_true")
    parser.add_argument("--monitor-optimizer", action="store_true")
    parser.add_argument("--monitor-param", action="store_true")
    parser.add_argument("--monitor-cc", action="store_true")
    parser.add_argument("--monitor-mbs-grad", action="store_true")
    parser.add_argument(
        "--monitor-target",
        action="append",
        default=None,
        help="repeat to limit module monitoring by name substring",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.list_topologies:
        for name, topology in registry.items():
            print(f"{name}: {asdict(topology)}")
        return
    if args.doctor:
        from .toolchain import doctor_report

        doctor_device = args.doctor_device or args.data_device
        if doctor_device is None:
            visible_devices = [
                device
                for device, variable in (
                    ("cuda", "CUDA_VISIBLE_DEVICES"),
                    ("npu", "ASCEND_RT_VISIBLE_DEVICES"),
                )
                if os.environ.get(variable, "").strip()
            ]
            if len(visible_devices) != 1:
                parser.error(
                    "--doctor requires --doctor-device when zero or multiple "
                    "device visibility variables are exported"
                )
            doctor_device = visible_devices[0]
        report = doctor_report(
            scope=(
                "accuracy-gpu" if doctor_device == "cuda" else "accuracy-npu"
            )
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        if not report.get("ok", False):
            raise SystemExit(1)
        return

    config = base_config
    dump = config.dump
    if args.dump_task is not None:
        dump = replace(dump, task=args.dump_task)
    if args.level is not None:
        dump = replace(dump, level=args.level)
    if args.dump_step is not None and args.dump_steps is not None:
        parser.error("use either --dump-step or --dump-steps")
    if args.dump_step is not None:
        dump = replace(dump, steps=(args.dump_step,))
    elif args.dump_steps is not None:
        dump = replace(
            dump,
            steps=tuple(
                sorted(
                    {
                        int(value.strip())
                        for value in args.dump_steps.split(",")
                        if value.strip()
                    }
                )
            ),
        )
    if args.dump_ranks is not None:
        dump = replace(
            dump,
            ranks=tuple(
                sorted(
                    {
                        int(value.strip())
                        for value in args.dump_ranks.split(",")
                        if value.strip()
                    }
                )
            ),
        )
    if args.scope is not None:
        dump = replace(dump, scope=tuple(args.scope))
    if args.module_or_api is not None:
        dump = replace(dump, module_or_api_list=tuple(args.module_or_api))
    if args.tensor_list is not None:
        dump = replace(dump, tensor_list=tuple(args.tensor_list))
    if args.data_mode is not None:
        dump = replace(
            dump,
            data_mode=tuple(
                value.strip()
                for value in args.data_mode.split(",")
                if value.strip()
            ),
        )
    if args.summary_mode is not None:
        dump = replace(dump, summary_mode=args.summary_mode)
    if args.async_dump:
        dump = replace(dump, async_dump=True)
    if args.no_extra_info:
        dump = replace(dump, extra_info=False)
    training = config.training
    if args.training_steps is not None:
        if config.workflow != "monitor":
            parser.error("--training-steps is supported only by monitor workflow")
        if args.training_steps < 1:
            parser.error("--training-steps must be positive")
        training = replace(training, steps=args.training_steps)
    required_steps = max(dump.steps) + 1
    if training.steps < required_steps:
        training = replace(training, steps=required_steps)
    compile_config = config.compile
    if args.backend is not None:
        compile_config = replace(compile_config, backend=args.backend)
    if args.no_dump_graphs:
        compile_config = replace(compile_config, dump_graphs=False)
    if args.no_capture_input:
        compile_config = replace(compile_config, capture_input=False)
    if args.compile_policy is not None:
        compile_config = replace(compile_config, policy=args.compile_policy)
    if args.children_depth is not None:
        compile_config = replace(
            compile_config,
            children_depth=args.children_depth,
        )
    monitor_config = config.monitor
    monitor_arguments_selected = any(
        value
        for value in (
            args.monitor_ranks is not None,
            args.monitor_start_step is not None,
            args.monitor_stop_step is not None,
            args.monitor_step_interval is not None,
            args.monitor_record_steps is not None,
            args.monitor_collect_times is not None,
            args.monitor_ops is not None,
            args.monitor_module,
            args.no_monitor_weight_grad,
            args.monitor_optimizer,
            args.monitor_param,
            args.monitor_cc,
            args.monitor_mbs_grad,
            args.monitor_target is not None,
        )
    )
    if monitor_arguments_selected and config.workflow != "monitor":
        parser.error("--monitor-* options require the monitor workflow")
    if args.monitor_ranks is not None:
        monitor_config = replace(
            monitor_config,
            ranks=tuple(
                sorted(
                    {
                        int(value.strip())
                        for value in args.monitor_ranks.split(",")
                        if value.strip()
                    }
                )
            ),
        )
    for argument, field_name in (
        (args.monitor_start_step, "start_step"),
        (args.monitor_stop_step, "stop_step"),
        (args.monitor_step_interval, "step_interval"),
        (args.monitor_record_steps, "step_count_per_record"),
        (args.monitor_collect_times, "collect_times"),
    ):
        if argument is not None:
            monitor_config = replace(monitor_config, **{field_name: argument})
    if args.monitor_ops is not None:
        monitor_config = replace(
            monitor_config,
            ops=tuple(
                value.strip()
                for value in args.monitor_ops.split(",")
                if value.strip()
            ),
        )
    monitor_config = replace(
        monitor_config,
        module=monitor_config.module or args.monitor_module,
        weight_grad=(
            monitor_config.weight_grad and not args.no_monitor_weight_grad
        ),
        optimizer=monitor_config.optimizer or args.monitor_optimizer,
        param=monitor_config.param or args.monitor_param,
        cc=monitor_config.cc or args.monitor_cc,
        monitor_mbs_grad=(
            monitor_config.monitor_mbs_grad or args.monitor_mbs_grad
        ),
        module_targets=(
            tuple(args.monitor_target)
            if args.monitor_target is not None
            else monitor_config.module_targets
        ),
    )
    config = replace(
        config,
        dump=dump,
        compile=compile_config,
        monitor=monitor_config,
        training=training,
    )

    selected_names = select_topologies(
        available=tuple(registry),
        topology=args.topology if args.topologies is None else None,
        topologies=args.topologies,
        default=("single",),
    )
    selected = tuple(registry[name] for name in selected_names)
    root = _root(script_path)

    if args.data:
        endpoint = _data_endpoint(config, args.data_device)
        if args.dry_run:
            print(
                json.dumps(
                    {
                        "operation": "prepare_shared_fixture",
                        "device_type": endpoint.device_type,
                        "endpoint_environment_keys": sorted(
                            endpoint.environment
                        ),
                        "fixture": str(
                            _fixture_directory(root, config).resolve()
                        ),
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return
        if args.force:
            reset_selected_outputs(
                root,
                config,
                topologies=selected,
                role=None,
                include_fixture=True,
            )
        path = prepare_shared_fixture(
            root,
            config,
            endpoint=endpoint,
            force=False,
        )
        print(f"Prepared MindStudio fixture: {path}")
        return

    if args.capture:
        role: Role = args.capture
        if args.force and not args.dry_run:
            reset_selected_outputs(
                root,
                config,
                topologies=selected,
                role=role,
                include_fixture=False,
            )
        for topology in selected:
            print(
                f"Starting official capture: workflow={config.workflow}, "
                f"role={role}, topology={topology.name}, repeat={args.repeat}",
                flush=True,
            )
            path = capture_official(
                root,
                config,
                topology=topology,
                role=role,
                repeat=args.repeat,
                dry_run=args.dry_run,
            )
            print(f"Official artifact: {path}")
        return

    if args.precheck:
        if config.workflow != "migration":
            parser.error("--precheck is supported only by the migration workflow")
        try:
            precheck_device_ids = tuple(
                int(value.strip())
                for value in args.precheck_device_ids.split(",")
                if value.strip()
            )
        except ValueError as error:
            parser.error(f"invalid --precheck-device-ids: {error}")
        if not precheck_device_ids or any(
            value < 0 for value in precheck_device_ids
        ):
            parser.error(
                "--precheck-device-ids must contain non-negative integers"
            )
        if args.precheck_splits is not None and not 1 <= args.precheck_splits <= 64:
            parser.error("--precheck-splits must be in [1, 64]")
        if args.precheck_splits is None and len(precheck_device_ids) != 1:
            parser.error(
                "multiple --precheck-device-ids require --precheck-splits"
            )
        if args.precheck_resume_csv is not None:
            selected_members = sum(
                len(_configured_dump_members(config, topology))
                for topology in selected
            )
            if selected_members != 1:
                parser.error(
                    "--precheck-resume-csv requires exactly one selected "
                    "topology/step/rank member"
                )
            if args.force:
                parser.error(
                    "--force cannot be combined with --precheck-resume-csv; "
                    "resume first or start a fresh pre-check without the CSV"
                )
        role = args.precheck
        if args.force and not args.dry_run:
            reset_precheck_outputs(
                root,
                config,
                topologies=selected,
                owner=role,
                repeat=args.repeat,
            )
        for topology in selected:
            print(
                f"Starting precision pre-check: role={role}, "
                f"topology={topology.name}, repeat={args.repeat}",
                flush=True,
            )
            path = run_precheck(
                root,
                config,
                topology=topology,
                role=role,
                repeat=args.repeat,
                device_ids=precheck_device_ids,
                num_splits=args.precheck_splits,
                jit_compile=args.precheck_jit_compile,
                save_error_data=args.precheck_save_error_data,
                filter_api=args.precheck_filter_api,
                config_path=args.precheck_config,
                resume_csv=args.precheck_resume_csv,
                dry_run=args.dry_run,
            )
            print(f"Precision pre-check report: {path}")
        return

    if args.precheck_compare:
        if config.workflow != "migration":
            parser.error(
                "--precheck-compare is supported only by the migration workflow"
            )
        endpoint_devices = {
            config.reference.device_type,
            config.candidate.device_type,
        }
        if endpoint_devices != {"cuda", "npu"}:
            parser.error(
                "--precheck-compare requires one GPU and one NPU endpoint"
            )
        if args.force and not args.dry_run:
            reset_precheck_outputs(
                root,
                config,
                topologies=selected,
                owner="compare",
                repeat=args.repeat,
            )
        for topology in selected:
            print(
                f"Starting pre-check comparison: topology={topology.name}, "
                f"repeat={args.repeat}",
                flush=True,
            )
            path = compare_prechecks(
                root,
                config,
                topology=topology,
                repeat=args.repeat,
                dry_run=args.dry_run,
            )
            report = path / "precheck_report.html"
            print(
                "Pre-check comparison report: "
                f"{report if report.is_file() else path}"
            )
        return

    if args.graph_visualize:
        if config.workflow != "migration":
            parser.error(
                "--graph-visualize is supported only by the migration workflow"
            )
        from .visualization import (
            reset_graph_visualizations,
            run_graph_visualization,
        )

        if args.force and not args.dry_run:
            reset_graph_visualizations(
                root,
                config,
                topologies=selected,
                repeat=args.repeat,
            )
        for topology in selected:
            print(
                f"Starting hierarchical graph comparison: "
                f"topology={topology.name}, repeat={args.repeat}",
                flush=True,
            )
            path = run_graph_visualization(
                root,
                config,
                topology=topology,
                repeat=args.repeat,
                fuzzy_match=args.fuzzy_match,
                overflow_check=args.graph_overflow_check,
                tensor_log=args.tensor_log,
                progress_log=args.graph_progress_log,
                dry_run=args.dry_run,
            )
            print(f"Hierarchical graph report: {path}")
        return

    if args.force and not args.dry_run:
        reset_compare_outputs(
            root,
            config,
            topologies=selected,
            repeat=args.repeat,
        )
    rows = [
        compare_official(
            root,
            config,
            topology=topology,
            repeat=args.repeat,
            fuzzy_match=args.fuzzy_match,
            data_mapping=args.data_mapping,
            cell_mapping=args.cell_mapping,
            diff_analysis=args.diff_analysis,
            tensor_log=args.tensor_log,
            xlsx=args.xlsx,
            force=False,
            dry_run=args.dry_run,
        )
        for topology in selected
    ]
    if args.dry_run:
        return
    report_directory = root / config.report_root / config.storage_name
    path = write_report_index(
        repository_root=root,
        report_directory=report_directory,
        experiment_name=config.storage_name,
        workflow=config.workflow,
        rows=rows,
        supplemental_report_patterns=(
            *config.formal_precision_reports,
            *config.performance_reports,
        ),
    )
    print(f"MindStudio report: {path}")
