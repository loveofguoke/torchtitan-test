# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Offline, distributed workflow for formal GLM 5.2 precision experiments."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field, replace
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import uuid
from typing import Any, Literal, Sequence

from tests.glm5_2_common.cli import (
    archive_previous_output,
    assert_run_not_active,
    process_is_running,
    reset_output_generation,
)
from tests.glm5_2_common.naming import config_name, slug
from tests.glm5_2_common.topology import (
    ParallelTopology,
    standard_topologies,
    training_command_args,
)

from .artifacts import (
    PrecisionArtifactError,
    PrecisionArtifactReader,
    PrecisionArtifactWriter,
    read_captured_metrics,
)
from .standards import PrecisionStandard
from .token_data import (
    INPUT_MAPPING,
    load_token_plan,
    sha256_file,
    step_series_digest,
    validate_runtime_input_contract,
)


ExperimentKind = Literal["migration", "self_consistency"]
CheckpointKind = Literal["random_seed", "converged"]


def _slug(value: str) -> str:
    return slug(value)


@dataclass(frozen=True)
class TrainingEndpoint:
    name: str
    device_type: Literal["cuda", "npu"]
    visible_devices: str
    topology: ParallelTopology
    num_nodes: int = 1
    node_rank: int = 0
    rendezvous_endpoint: str = "localhost:0"
    repeats: int = 2
    environment: dict[str, str] = field(default_factory=dict)
    extra_args: tuple[str, ...] = ()
    entry_module: str = "tests.glm5_2_precision.capture_metrics"

    def __post_init__(self) -> None:
        devices = [value.strip() for value in self.visible_devices.split(",") if value]
        if not devices:
            raise ValueError("visible_devices must contain at least one device")
        if self.num_nodes < 1 or not 0 <= self.node_rank < self.num_nodes:
            raise ValueError("invalid multi-node endpoint configuration")
        local_world_size = self.topology.world_size // self.num_nodes
        if self.topology.world_size % self.num_nodes != 0:
            raise ValueError(
                "topology world size must be divisible by num_nodes: "
                f"{self.topology.world_size} % {self.num_nodes} != 0"
            )
        if len(devices) < local_world_size:
            raise ValueError(
                f"endpoint needs {local_world_size} local devices but only "
                f"{len(devices)} are visible"
            )
        if self.repeats < 1:
            raise ValueError("endpoint repeats must be positive")

    @property
    def num_processes_per_node(self) -> int:
        return self.topology.world_size // self.num_nodes

    @property
    def visible_devices_env(self) -> str:
        return (
            "ASCEND_RT_VISIBLE_DEVICES"
            if self.device_type == "npu"
            else "CUDA_VISIBLE_DEVICES"
        )

    @property
    def torchtitan_device(self) -> str:
        return "npu" if self.device_type == "npu" else "gpu"


@dataclass(frozen=True)
class FormalTrainingConfig:
    module: str = "glm5"
    config: str = "glm5_debugmodel"
    steps: int = 1000
    local_batch_size: int = 2
    global_batch_size: int = 16
    sequence_length: int = 128
    seed: int = 61
    deterministic: bool = True
    deterministic_warn_only: bool = False
    training_dtype: Literal["float32", "bfloat16"] = "float32"
    mixed_precision_param: Literal["float32", "bfloat16"] = "bfloat16"
    mixed_precision_reduce: Literal["float32"] = "float32"
    checkpoint_kind: CheckpointKind = "random_seed"
    converged_checkpoint: str | None = None
    data_paths: tuple[str, ...] = ("tests/assets/c4_test", "tests/assets/tokenizer")
    exploratory_steps: tuple[int, ...] = ()
    extra_args: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        positive = (
            self.steps,
            self.local_batch_size,
            self.global_batch_size,
            self.sequence_length,
        )
        if any(value < 1 for value in positive):
            raise ValueError("steps, batch sizes, and sequence length must be positive")
        if self.checkpoint_kind == "converged" and not self.converged_checkpoint:
            raise ValueError("converged checkpoint mode requires converged_checkpoint")
        if self.deterministic_warn_only:
            raise ValueError("deterministic_warn_only is forbidden for precision runs")
        if any(not 1 <= step <= self.steps for step in self.exploratory_steps):
            raise ValueError("exploratory_steps must be within the training step range")
        if tuple(sorted(set(self.exploratory_steps))) != self.exploratory_steps:
            raise ValueError("exploratory_steps must be sorted and unique")

    @property
    def precision_name(self) -> str:
        if self.training_dtype == "bfloat16":
            return "full-bf16"
        return f"mixed-{self.mixed_precision_param}"


@dataclass(frozen=True)
class TrainingTopologyPlan:
    """Derived batch schedule for one training topology."""

    gradient_accumulation_steps: int
    pipeline_microbatches: int


def training_topology_plan(
    training: FormalTrainingConfig,
    topology: ParallelTopology,
) -> TrainingTopologyPlan:
    """Validate and describe the batch schedule before launching workers."""

    data_parallel_batch = training.local_batch_size * topology.data_parallel_degree
    if training.global_batch_size % data_parallel_batch:
        raise ValueError(
            f"topology {topology.name!r} requires global_batch_size to be "
            "divisible by local_batch_size * data_parallel_degree: "
            f"{training.global_batch_size} % ({training.local_batch_size} * "
            f"{topology.data_parallel_degree}) != 0"
        )
    if training.local_batch_size % topology.pipeline_parallel_microbatch_size:
        raise ValueError(
            f"topology {topology.name!r} requires local_batch_size to be "
            "divisible by pipeline_parallel_microbatch_size: "
            f"{training.local_batch_size} % "
            f"{topology.pipeline_parallel_microbatch_size} != 0"
        )
    pipeline_microbatches = (
        training.local_batch_size
        // topology.pipeline_parallel_microbatch_size
    )
    if (
        topology.pipeline_parallel_degree > 1
        and pipeline_microbatches < topology.pipeline_parallel_degree
    ):
        raise ValueError(
            f"topology {topology.name!r} has {pipeline_microbatches} pipeline "
            f"microbatches but {topology.pipeline_parallel_degree} stages; "
            "increase local_batch_size or reduce the PP microbatch size"
        )
    return TrainingTopologyPlan(
        gradient_accumulation_steps=(
            training.global_batch_size // data_parallel_batch
        ),
        pipeline_microbatches=pipeline_microbatches,
    )


@dataclass(frozen=True)
class FormalExperimentConfig:
    name: str
    kind: ExperimentKind
    reference: TrainingEndpoint
    candidate: TrainingEndpoint
    training: FormalTrainingConfig = field(default_factory=FormalTrainingConfig)
    standard: PrecisionStandard = field(default_factory=PrecisionStandard)
    fixture_root: str = "precision_fixtures"
    artifact_root: str = "precision_artifacts"
    report_root: str = "precision_reports"
    run_root: str = "precision_runs"
    exploratory_reports: tuple[str, ...] = ()
    fixture_name: str | None = None
    allow_endpoint_argument_difference: bool = False
    storage_name_override: str | None = None
    topology_subdirectory: bool = False
    legacy_storage_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.kind == "migration":
            if self.reference.topology != self.candidate.topology:
                raise ValueError("migration requires the same parallel topology")
        elif self.reference.device_type != self.candidate.device_type:
            raise ValueError("self-consistency requires the same device type")
        training_topology_plan(self.training, self.reference.topology)
        training_topology_plan(self.training, self.candidate.topology)

    @property
    def scenario_name(self) -> str:
        sides = (
            f"{self.reference.device_type}-{self.reference.topology.slug}-vs-"
            f"{self.candidate.device_type}-{self.candidate.topology.slug}"
        )
        return _slug(
            f"{self.name}-{self.kind}-{sides}-{self.training.precision_name}-"
            f"{self.training.checkpoint_kind}-steps{self.training.steps}-"
            f"global-batch{self.training.global_batch_size}-"
            f"seq{self.training.sequence_length}-seed{self.training.seed}"
        )

    @property
    def storage_base_name(self) -> str:
        """Return the readable part retained by the pre-digest layout."""

        if self.storage_name_override is not None:
            return _slug(self.storage_name_override)
        if self.kind == "migration":
            experiment = (
                f"migration-{self.reference.device_type}-"
                f"{self.candidate.device_type}"
            )
            if not self.topology_subdirectory:
                experiment += f"-{self.reference.topology.slug}"
        else:
            experiment = f"self-{self.reference.device_type}"
        checkpoint = (
            "random" if self.training.checkpoint_kind == "random_seed" else "converged"
        )
        precision = {
            "mixed-bfloat16": "bf16",
            "mixed-float32": "fp32",
            "full-bf16": "full-bf16",
        }.get(self.training.precision_name, self.training.precision_name)
        return _slug(
            f"{experiment}-{precision}-{checkpoint}-s{self.training.steps}-"
            f"b{self.training.global_batch_size}-seq{self.training.sequence_length}-"
            f"seed{self.training.seed}"
        )

    @property
    def capture_identity(self) -> dict[str, Any]:
        """Return settings that can change captured training results."""

        def endpoint_identity(endpoint: TrainingEndpoint) -> dict[str, Any]:
            return {
                "device_type": endpoint.device_type,
                "num_nodes": endpoint.num_nodes,
                "entry_module": endpoint.entry_module,
                "environment": endpoint.environment,
                "extra_args": endpoint.extra_args,
            }

        return {
            "kind": self.kind,
            "training": _normalized_training(self.training),
            "reference": endpoint_identity(self.reference),
            "candidate": endpoint_identity(self.candidate),
        }

    @property
    def storage_name(self) -> str:
        """Return the readable pre-digest name plus its capture config digest."""

        return config_name(self.storage_base_name, self.capture_identity)

    @property
    def fixture_storage_name(self) -> str:
        """Return the experiment identity of synchronized inputs."""

        if self.fixture_name is not None:
            return _slug(self.fixture_name)
        return self.storage_name


def _directory_digest(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    paths = [path] if path.is_file() else sorted(item for item in path.rglob("*") if item.is_file())
    for item in paths:
        relative = item.name if path.is_file() else item.relative_to(path).as_posix()
        digest.update(relative.encode("utf-8"))
        with item.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def _git_metadata(path: Path) -> dict[str, str]:
    commands = {
        "commit": ("rev-parse", "HEAD"),
        "branch": ("branch", "--show-current"),
        "status": ("status", "--short", "--untracked-files=normal"),
    }
    values: dict[str, str] = {"root": str(path)}
    for name, args in commands.items():
        try:
            values[name] = subprocess.run(
                ["git", "-C", str(path), *args],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError) as error:
            values[name] = f"unavailable: {error}"
    return values


def _source_metadata(root: Path) -> dict[str, Any]:
    try:
        from importlib import metadata as package_metadata
    except (ImportError, ModuleNotFoundError) as error:
        try:
            import importlib_metadata as package_metadata
        except (ImportError, ModuleNotFoundError):
            package_metadata = None
        metadata_import_error: Exception | None = error
    else:
        metadata_import_error = None

    try:
        from importlib.util import find_spec
    except (ImportError, ModuleNotFoundError):
        find_spec = None

    sources: dict[str, Any] = {"torchtitan_test": _git_metadata(root)}
    packages: dict[str, str] = {}
    for package in ("torch", "torchtitan", "torchtitanturbo", "torch_npu"):
        if package_metadata is None:
            packages[package] = f"unavailable: {metadata_import_error}"
        else:
            try:
                packages[package] = package_metadata.version(package)
            except package_metadata.PackageNotFoundError:
                packages[package] = "not-installed"
            except Exception as error:
                packages[package] = f"unavailable: {error}"
        if find_spec is None:
            continue
        try:
            spec = find_spec(package)
        except Exception:
            continue
        if spec is None:
            continue
        if spec.submodule_search_locations:
            source = Path(next(iter(spec.submodule_search_locations))).resolve()
        elif spec.origin:
            source = Path(spec.origin).resolve().parent
        else:
            continue
        git_root = next(
            (
                candidate
                for candidate in (source, *source.parents)
                if (candidate / ".git").exists()
            ),
            None,
        )
        if git_root is not None:
            sources[package] = _git_metadata(git_root)
    runtime_environment = {
        name: os.environ[name]
        for name in (
            "ASCEND_HOME_PATH",
            "ASCEND_TOOLKIT_HOME",
            "ASCEND_OPP_PATH",
            "GRAPH_CANN_ROOT",
            "GRAPH_CONDA_ENV",
            "TORCHINDUCTOR_NPU_BACKEND",
        )
        if os.environ.get(name)
    }
    cann_version_files: list[dict[str, str]] = []
    cann_roots = {
        Path(value).resolve()
        for name, value in runtime_environment.items()
        if name in {"ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME", "GRAPH_CANN_ROOT"}
    }
    for cann_root in sorted(cann_roots, key=str):
        for relative_path in ("compiler/version.info", "version.info"):
            version_path = cann_root / relative_path
            if version_path.is_file():
                cann_version_files.append(
                    {
                        "path": str(version_path),
                        "sha256": sha256_file(version_path),
                    }
                )
                break
    return {
        "packages": packages,
        "sources": sources,
        "runtime": {
            "python_version": platform.python_version(),
            "python_executable": sys.executable,
            "environment": runtime_environment,
            "cann_version_files": cann_version_files,
        },
    }


def _completed_capture_can_be_finalized(
    metrics_path: Path,
    input_contract_directory: Path,
    *,
    expected_steps: int,
) -> bool:
    """Return whether a finished run only needs artifact publication."""

    if not (input_contract_directory / "summary.json").is_file():
        return False
    try:
        return len(read_captured_metrics(metrics_path)) == expected_steps
    except (KeyError, OSError, TypeError, ValueError, PrecisionArtifactError):
        return False


def _recorded_capture_attempt_id(capture_state_path: Path) -> str:
    """Return the training attempt that produced an existing capture."""

    if not capture_state_path.is_file():
        return "legacy-unrecorded"
    try:
        state = json.loads(capture_state_path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError):
        return "legacy-unrecorded"
    attempt_id = state.get("capture_attempt_id")
    if not isinstance(attempt_id, str) or not attempt_id:
        return "legacy-unrecorded"
    return attempt_id


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _root(script_path: str) -> Path:
    return Path(script_path).resolve().parents[2]


def _precision_name(training: FormalTrainingConfig) -> str:
    return {
        "mixed-bfloat16": "bf16",
        "mixed-float32": "fp32",
        "full-bf16": "full-bf16",
    }.get(training.precision_name, training.precision_name)


def _legacy_fixture_names(config: FormalExperimentConfig) -> tuple[str, ...]:
    training = _normalized_training(config.training)
    old_digest = hashlib.sha256(
        json.dumps(training, sort_keys=True).encode("utf-8")
    ).hexdigest()[:10]
    old_generated = _slug(
        f"fixture-{config.training.module}-{_precision_name(config.training)}-"
        f"s{config.training.steps}-b{config.training.global_batch_size}-"
        f"seq{config.training.sequence_length}-seed{config.training.seed}-{old_digest}"
    )
    checkpoint = (
        "random" if config.training.checkpoint_kind == "random_seed" else "converged"
    )
    generic_base = _slug(
        f"{_precision_name(config.training)}-{checkpoint}-"
        f"s{config.training.steps}-b{config.training.global_batch_size}-"
        f"seq{config.training.sequence_length}-seed{config.training.seed}"
    )
    generic_digested = config_name(generic_base, training)
    return tuple(
        dict.fromkeys(
            (
                config.storage_base_name,
                generic_digested,
                old_generated,
                "graph-probe-fixture",
                *config.legacy_storage_names,
            )
        )
    )


def _stored_training_matches(path: Path, config: FormalExperimentConfig) -> bool:
    expected = json.loads(
        json.dumps(_normalized_training(config.training), sort_keys=True)
    )
    fixture_manifest = path / "fixture.json"
    if fixture_manifest.is_file():
        try:
            return json.loads(fixture_manifest.read_text(encoding="utf-8")).get(
                "training"
            ) == expected
        except (OSError, TypeError, ValueError):
            return False
    contracts = list(path.rglob("training_contract.json"))
    if contracts:
        allowed_endpoint_args = {
            tuple(config.reference.extra_args),
            tuple(config.candidate.extra_args),
        }
        try:
            values = [
                json.loads(contract.read_text(encoding="utf-8"))
                for contract in contracts
            ]
            return all(
                value.get("training") == expected
                and tuple(value.get("endpoint_args", ())) in allowed_endpoint_args
                for value in values
            )
        except (OSError, TypeError, ValueError):
            return False
    shared_required = (
        f"--training.steps={config.training.steps}",
        f"--training.dtype={config.training.training_dtype}",
        f"--training.mixed_precision_param={config.training.mixed_precision_param}",
        f"--debug.seed={config.training.seed}",
    )
    old_batch_args = (
        f"--training.local_batch_size={config.training.local_batch_size}",
        f"--training.global_batch_size={config.training.global_batch_size}",
        f"--training.seq_len={config.training.sequence_length}",
    )
    new_batch_args = tuple(
        training_command_args(
            local_batch_size=config.training.local_batch_size,
            global_batch_size=config.training.global_batch_size,
            sequence_length=config.training.sequence_length,
            topology=config.reference.topology,
        )[:3]
    )
    logs = list(path.rglob("runtime.log"))
    if not logs:
        return False
    try:
        commands = [log.read_text(encoding="utf-8", errors="replace") for log in logs]
    except OSError:
        return False
    return all(
        all(argument in command for argument in shared_required)
        and (
            all(argument in command for argument in old_batch_args)
            or all(argument in command for argument in new_batch_args)
        )
        for command in commands
    )


def _adopt_legacy_directory(
    parent: Path,
    *,
    destination_name: str,
    legacy_names: Sequence[str],
    config: FormalExperimentConfig,
) -> Path:
    destination = parent / destination_name
    if destination.exists():
        return destination
    matches = []
    for name in dict.fromkeys(legacy_names):
        source = parent / name
        if source != destination and source.is_dir() and _stored_training_matches(
            source, config
        ):
            matches.append(source)
    if len(matches) > 1:
        raise RuntimeError(
            "multiple legacy output directories match the current config: "
            + ", ".join(str(path) for path in matches)
        )
    if matches:
        source = matches[0]
        parent.mkdir(parents=True, exist_ok=True)
        source.rename(destination)
        fixture_manifest = destination / "fixture.json"
        if fixture_manifest.is_file():
            manifest = json.loads(fixture_manifest.read_text(encoding="utf-8"))
            manifest["scenario_name"] = destination_name
            _write_json(fixture_manifest, manifest)
        for contract_path in destination.rglob("training_contract.json"):
            contract = json.loads(contract_path.read_text(encoding="utf-8"))
            if "scenario_name" not in contract:
                continue
            contract["scenario_name"] = destination_name
            _write_json(contract_path, contract)
            manifest_path = contract_path.parent / "manifest.json"
            if manifest_path.is_file():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                files = manifest.get("files", {})
                if "training_contract.json" in files:
                    files["training_contract.json"] = sha256_file(contract_path)
                    _write_json(manifest_path, manifest)
        print(f"Adopted matching legacy output:\n  {source}\n  -> {destination}")
    return destination


def _fixture_directory(root: Path, config: FormalExperimentConfig) -> Path:
    return _adopt_legacy_directory(
        root / config.fixture_root,
        destination_name=config.fixture_storage_name,
        legacy_names=_legacy_fixture_names(config),
        config=config,
    )


def _artifact_directory(
    root: Path,
    config: FormalExperimentConfig,
    role: str,
    endpoint: TrainingEndpoint,
    repeat: int,
) -> Path:
    name = (
        _slug(f"{endpoint.topology.slug}-r{repeat}")
        if config.kind == "self_consistency" and role == "candidate"
        else _slug(f"{role}-r{repeat}")
    )
    directory = _adopt_legacy_directory(
        root / config.artifact_root,
        destination_name=config.storage_name,
        legacy_names=(config.storage_base_name, *config.legacy_storage_names),
        config=config,
    )
    _adopt_legacy_directory(
        root / config.run_root,
        destination_name=config.storage_name,
        legacy_names=(config.storage_base_name, *config.legacy_storage_names),
        config=config,
    )
    if config.topology_subdirectory:
        directory /= endpoint.topology.slug
    return directory / name


def _run_directory(
    root: Path,
    config: FormalExperimentConfig,
    role: str,
    endpoint: TrainingEndpoint,
    repeat: int,
) -> Path:
    name = (
        _slug(f"{endpoint.topology.slug}-r{repeat}")
        if config.kind == "self_consistency" and role == "candidate"
        else _slug(f"{role}-r{repeat}")
    )
    if endpoint.num_nodes > 1:
        name = f"{name}-node-{endpoint.node_rank}"
    directory = _adopt_legacy_directory(
        root / config.run_root,
        destination_name=config.storage_name,
        legacy_names=(config.storage_base_name, *config.legacy_storage_names),
        config=config,
    )
    if config.topology_subdirectory:
        directory /= endpoint.topology.slug
    return directory / name


def _input_contract_directory(
    root: Path,
    config: FormalExperimentConfig,
    role: str,
    endpoint: TrainingEndpoint,
    repeat: int,
) -> Path:
    """Return the rank-level fixed-input evidence owned by one capture.

    This directory is deliberately outside the trainer run directory because
    every rank writes into the same contract. It must nevertheless follow the
    same reset/retry lifecycle as the run and artifact; otherwise rank files
    left by an interrupted generation can be mixed into the next capture.
    """

    directory = root / config.run_root / config.storage_name
    if config.topology_subdirectory:
        directory /= endpoint.topology.slug
    artifact_name = _artifact_directory(root, config, role, endpoint, repeat).name
    return directory / f"{artifact_name}-input-contract"


def _report_directory(root: Path, config: FormalExperimentConfig) -> Path:
    parent = root / config.report_root
    destination = parent / config.storage_name
    if destination.exists():
        return destination
    artifact_directory = root / config.artifact_root / config.storage_name
    if not artifact_directory.is_dir():
        return destination
    for legacy_name in dict.fromkeys(
        (config.storage_base_name, *config.legacy_storage_names)
    ):
        source = parent / legacy_name
        if source.is_dir():
            parent.mkdir(parents=True, exist_ok=True)
            source.rename(destination)
            print(f"Adopted matching legacy output:\n  {source}\n  -> {destination}")
            break
    return destination


def _seed_checkpoint_path(fixture_directory: Path) -> Path:
    manifest_path = fixture_directory / "fixture.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"fixture is not prepared; run --data first: {manifest_path}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checkpoint_path = fixture_directory / manifest["checkpoint_relative_path"]
    if not checkpoint_path.is_dir():
        raise FileNotFoundError(f"fixture checkpoint not found: {checkpoint_path}")
    if _directory_digest(checkpoint_path) != manifest["checkpoint_sha256"]:
        raise RuntimeError(f"fixture checkpoint checksum mismatch: {checkpoint_path}")
    return checkpoint_path


def _token_plan_path(fixture_directory: Path) -> Path:
    manifest_path = fixture_directory / "fixture.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"fixture is not prepared; run --data first: {manifest_path}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    relative_path = manifest.get("token_plan_relative_path")
    if not relative_path:
        raise RuntimeError(
            "fixture has no fixed token plan; recreate it with --data --force"
        )
    token_plan_path = fixture_directory / relative_path
    load_token_plan(token_plan_path)
    return token_plan_path


def resolve_fixture_inputs(
    root: Path,
    config: FormalExperimentConfig,
) -> tuple[Path, Path]:
    """Resolve and verify the shared seed checkpoint and fixed token plan."""

    fixture_directory = _fixture_directory(root, config)
    return (
        _seed_checkpoint_path(fixture_directory),
        _token_plan_path(fixture_directory),
    )


def fixed_input_environment(
    *,
    token_plan_path: Path,
    input_contract_directory: Path,
    training: FormalTrainingConfig,
    topology: ParallelTopology,
) -> dict[str, str]:
    """Return the shared fixed-token contract for one training process."""

    return {
        "GLM5_PRECISION_TOKEN_PLAN_PATH": str(token_plan_path),
        "GLM5_PRECISION_INPUT_CONTRACT_DIR": str(input_contract_directory),
        "GLM5_PRECISION_GLOBAL_BATCH_SIZE": str(training.global_batch_size),
        "GLM5_PRECISION_TRAINING_LOCAL_BATCH_SIZE": str(
            training.local_batch_size
        ),
        "GLM5_PRECISION_TENSOR_PARALLEL_DEGREE": str(
            topology.tensor_parallel_degree
        ),
        "GLM5_PRECISION_CONTEXT_PARALLEL_DEGREE": str(
            topology.context_parallel_degree
        ),
    }


def _base_training_args(
    config: FormalTrainingConfig,
    *,
    dump_folder: Path,
    topology: ParallelTopology,
) -> list[str]:
    args = [
        f"--dump_folder={dump_folder}",
        f"--training.steps={config.steps}",
        *training_command_args(
            local_batch_size=config.local_batch_size,
            global_batch_size=config.global_batch_size,
            sequence_length=config.sequence_length,
            topology=topology,
        ),
        f"--training.dtype={config.training_dtype}",
        f"--training.mixed_precision_param={config.mixed_precision_param}",
        f"--training.mixed_precision_reduce={config.mixed_precision_reduce}",
        f"--debug.seed={config.seed}",
        "--metrics.log_freq=1",
        "--metrics.enable_tensorboard",
        "--metrics.disable_color_printing",
        "--metrics.save_tb_folder=tensorboard",
        "--debug.no-enable-structured-logging",
    ]
    if config.deterministic:
        args.append("--debug.deterministic")
    args.extend(config.extra_args)
    return args


def _normalized_training(config: FormalTrainingConfig) -> dict[str, Any]:
    value = asdict(config)
    # The source location is not part of the experiment identity. Captures load
    # the synchronized fixture checkpoint and compare its digest instead.
    value["converged_checkpoint"] = None
    return value


def _torchrun_command(
    *,
    root: Path,
    config: FormalExperimentConfig,
    endpoint: TrainingEndpoint,
    dump_folder: Path,
    checkpoint_path: Path,
) -> list[str]:
    training_topology_plan(config.training, endpoint.topology)
    if endpoint.num_nodes > 1 and endpoint.rendezvous_endpoint == "localhost:0":
        raise ValueError(
            "multi-node capture requires a shared host:port rendezvous_endpoint"
        )
    metrics_rank = (
        0
        if endpoint.topology.pipeline_parallel_degree == 1
        or endpoint.topology.pipeline_parallel_schedule == "ZBVZeroBubble"
        else (
            endpoint.topology.world_size // endpoint.topology.pipeline_parallel_degree
        )
        * (endpoint.topology.pipeline_parallel_degree - 1)
    )
    torchrun = shutil.which("torchrun") or str(
        Path(sys.executable).with_name("torchrun")
    )
    return [
        torchrun,
        f"--nnodes={endpoint.num_nodes}",
        f"--node_rank={endpoint.node_rank}",
        f"--nproc_per_node={endpoint.num_processes_per_node}",
        "--rdzv_backend=c10d",
        f"--rdzv_endpoint={endpoint.rendezvous_endpoint}",
        f"--rdzv_id={config.storage_name}",
        f"--local-ranks-filter={metrics_rank % endpoint.num_processes_per_node}",
        "--role=rank",
        "--tee=3",
        "-m",
        endpoint.entry_module,
        "--module",
        config.training.module,
        "--config",
        config.training.config,
        *_base_training_args(
            config.training,
            dump_folder=dump_folder,
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
    log_context: dict[str, Any] | None = None,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        for key, value in (log_context or {}).items():
            log.write(f"{key}: {value}\n")
        if log_context:
            log.write("\n")
        log.write("Command: " + " ".join(command) + "\n\n")
        log.flush()
        process = subprocess.run(
            list(command),
            cwd=root,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if process.returncode:
        print(f"Process failed; runtime log: {log_path}", file=sys.stderr)
        try:
            lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            lines = []
        if lines:
            print("Last 80 log lines:", file=sys.stderr)
            print("\n".join(lines[-80:]), file=sys.stderr)
        raise subprocess.CalledProcessError(process.returncode, list(command))


def prepare_fixture(
    root: Path,
    config: FormalExperimentConfig,
    *,
    endpoint: TrainingEndpoint,
    force: bool,
) -> Path:
    fixture_directory = _fixture_directory(root, config)
    if fixture_directory.exists():
        if not force and _stored_training_matches(fixture_directory, config):
            try:
                resolve_fixture_inputs(root, config)
            except (KeyError, OSError, RuntimeError, TypeError, ValueError):
                pass
            else:
                print(f"Reuse completed fixture: {fixture_directory}", flush=True)
                return fixture_directory
        if force:
            previous_generation = None
            fixture_manifest = fixture_directory / "fixture.json"
            if fixture_manifest.is_file():
                try:
                    previous_generation = json.loads(
                        fixture_manifest.read_text(encoding="utf-8")
                    ).get("generation_id")
                except (OSError, TypeError, ValueError):
                    pass
            _assert_fixture_generation_not_running(
                root,
                config,
                previous_generation,
            )
            reset_output_generation(
                (fixture_directory,),
                label="fixture",
            )
        else:
            archived = archive_previous_output(fixture_directory)
            print(
                "Retry incomplete or incompatible fixture; archived previous "
                f"output: {archived}",
                flush=True,
            )
    fixture_directory.mkdir(parents=True)

    data_digests = {
        path: _directory_digest(root / path) for path in config.training.data_paths
    }
    token_plan_relative_path = "token_plan"
    token_plan_path = fixture_directory / token_plan_relative_path
    token_plan_environment = os.environ.copy()
    token_plan_environment.update(endpoint.environment)
    token_plan_environment[endpoint.visible_devices_env] = (
        endpoint.visible_devices.split(",")[0]
    )
    token_plan_environment["TORCHTITAN_DEVICE"] = endpoint.torchtitan_device
    token_plan_environment["GLM5_PRECISION_TOKEN_PLAN_OUTPUT"] = str(token_plan_path)
    token_plan_environment["GLM5_PRECISION_TOKEN_PLAN_STEPS"] = str(
        config.training.steps
    )
    token_plan_environment["GLM5_PRECISION_TOKEN_PLAN_GLOBAL_BATCH_SIZE"] = str(
        config.training.global_batch_size
    )
    token_plan_environment["GLM5_PRECISION_TOKEN_PLAN_SEQUENCE_LENGTH"] = str(
        config.training.sequence_length
    )
    token_plan_command = [
        sys.executable,
        "-m",
        "tests.glm5_2_precision.generate_token_plan",
        "--module",
        config.training.module,
        "--config",
        config.training.config,
        *config.training.extra_args,
    ]
    token_generation_log = fixture_directory / "token_generation.log"
    print(f"Token-plan generation log: {token_generation_log}", flush=True)
    _run_process(
        token_plan_command,
        root=root,
        environment=token_plan_environment,
        log_path=token_generation_log,
    )
    token_plan = load_token_plan(token_plan_path)
    data_digests["fixed_token_plan"] = sha256_file(token_plan_path / "manifest.json")
    if config.training.checkpoint_kind == "converged":
        checkpoint_path = Path(config.training.converged_checkpoint or "").resolve()
        if not checkpoint_path.is_dir():
            raise FileNotFoundError(checkpoint_path)
        relative_checkpoint = "checkpoint"
        shutil.copytree(checkpoint_path, fixture_directory / relative_checkpoint)
    else:
        output = fixture_directory / ".seed_generation"
        environment = os.environ.copy()
        environment.update(endpoint.environment)
        environment[endpoint.visible_devices_env] = endpoint.visible_devices.split(",")[0]
        environment["TORCHTITAN_DEVICE"] = endpoint.torchtitan_device
        environment["LOG_RANK"] = "0"
        torchrun = shutil.which("torchrun") or str(
            Path(sys.executable).with_name("torchrun")
        )
        command = [
            torchrun,
            "--nproc_per_node=1",
            "--rdzv_backend=c10d",
            "--rdzv_endpoint=localhost:0",
            "--role=rank",
            "--tee=3",
            "-m",
            (
                "tests.glm5_2_precision.seed_checkpoint"
                if endpoint.device_type == "npu"
                else "torchtitan.train"
            ),
        ]
        command.extend(
            [
                "--module",
                config.training.module,
                "--config",
                config.training.config,
                *_base_training_args(
                    config.training,
                    dump_folder=output,
                    topology=ParallelTopology("single", 1),
                ),
                "--parallelism.data_parallel_replicate_degree=1",
                "--parallelism.data_parallel_shard_degree=1",
                "--parallelism.tensor_parallel_degree=1",
                "--parallelism.pipeline_parallel_degree=1",
                "--parallelism.expert_parallel_degree=1",
                "--checkpoint.enable",
                "--checkpoint.create_seed_checkpoint",
            ]
        )
        seed_generation_log = fixture_directory / "seed_generation.log"
        print(f"Seed-checkpoint generation log: {seed_generation_log}", flush=True)
        _run_process(
            command,
            root=root,
            environment=environment,
            log_path=seed_generation_log,
        )
        matches = list(output.rglob("step-0"))
        if len(matches) != 1:
            raise RuntimeError(f"expected one step-0 seed checkpoint, found {matches}")
        relative_checkpoint = "checkpoint"
        shutil.move(str(matches[0]), fixture_directory / relative_checkpoint)
        shutil.rmtree(output)

    local_checkpoint = fixture_directory / relative_checkpoint
    manifest = {
        "schema": "torchtitan.glm5_2.precision_fixture",
        "schema_version": 1,
        "generation_id": uuid.uuid4().hex,
        "scenario_name": config.fixture_storage_name,
        "scenario_description": config.scenario_name,
        "checkpoint_kind": config.training.checkpoint_kind,
        "checkpoint_relative_path": relative_checkpoint,
        "checkpoint_sha256": _directory_digest(local_checkpoint),
        "data_sha256": data_digests,
        "token_plan_relative_path": token_plan_relative_path,
        "token_plan": {
            "mapping": INPUT_MAPPING,
            "manifest_sha256": data_digests["fixed_token_plan"],
            "steps": token_plan.steps,
            "global_batch_size": token_plan.global_batch_size,
            "sequence_length": token_plan.sequence_length,
            "step_series_sha256": step_series_digest(token_plan.step_sha256),
        },
        "training": _normalized_training(config.training),
    }
    _write_json(fixture_directory / "fixture.json", manifest)
    return fixture_directory


def _assert_fixture_generation_not_running(
    root: Path,
    config: FormalExperimentConfig,
    generation_id: str | None,
) -> None:
    if generation_id is None:
        return
    run_root = root / config.run_root
    if not run_root.is_dir():
        return
    state_paths = (
        *run_root.rglob("capture_state.json"),
        *run_root.rglob("run_state.json"),
    )
    for state_path in state_paths:
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            state_generation = state.get("fixture_generation_id") or state.get(
                "context", {}
            ).get("fixture_generation_id")
            pid = int(state.get("pid", -1))
        except (OSError, TypeError, ValueError):
            continue
        if (
            state.get("status") == "running"
            and state_generation == generation_id
            and process_is_running(pid)
        ):
            raise RuntimeError(
                "fixture generation is still used by a live experiment: "
                f"PID {pid}, state {state_path}; stop that process before "
                "recreating the fixture"
            )


def capture_endpoint(
    root: Path,
    config: FormalExperimentConfig,
    *,
    role: Literal["reference", "candidate"],
    repeat: int,
    force: bool,
) -> Path | None:
    endpoint = config.reference if role == "reference" else config.candidate
    metrics_rank = (
        0
        if endpoint.topology.pipeline_parallel_degree == 1
        or endpoint.topology.pipeline_parallel_schedule == "ZBVZeroBubble"
        else (
            endpoint.topology.world_size
            // endpoint.topology.pipeline_parallel_degree
        )
        * (endpoint.topology.pipeline_parallel_degree - 1)
    )
    metrics_node_rank = metrics_rank // endpoint.num_processes_per_node
    if not 1 <= repeat <= endpoint.repeats:
        raise ValueError(f"repeat must be in [1, {endpoint.repeats}]")
    artifact_directory = _artifact_directory(root, config, role, endpoint, repeat)
    fixture_directory = _fixture_directory(root, config)
    fixture_manifest = json.loads(
        (fixture_directory / "fixture.json").read_text(encoding="utf-8")
    )
    fixture_generation = fixture_manifest.get("generation_id")
    artifact_complete = False
    artifact_generation_mismatch = False
    if endpoint.node_rank == metrics_node_rank and artifact_directory.exists():
        try:
            artifact = PrecisionArtifactReader(artifact_directory)
            artifact_generation_mismatch = (
                fixture_generation is not None
                and artifact.training_contract.get("fixture_generation_id")
                != fixture_generation
            )
            artifact_complete = not artifact_generation_mismatch
        except (KeyError, OSError, TypeError, ValueError, PrecisionArtifactError):
            artifact_complete = False
        if artifact_complete and not force:
            print(f"Skip completed capture: {artifact_directory}")
            return artifact_directory
    checkpoint_path = _seed_checkpoint_path(fixture_directory)
    token_plan_path = _token_plan_path(fixture_directory)
    run_directory = _run_directory(root, config, role, endpoint, repeat)
    metrics_path = run_directory / "raw_metrics.jsonl"
    runtime_log = run_directory / "runtime.log"
    run_generation_path = run_directory / "fixture_generation.json"
    capture_state_path = run_directory / "capture_state.json"
    input_contract_directory = _input_contract_directory(
        root,
        config,
        role,
        endpoint,
        repeat,
    )
    finalize_existing = False
    if run_directory.exists():
        assert_run_not_active(
            run_directory,
            state_name="capture_state.json",
        )
        run_generation_matches = fixture_generation is None
        if fixture_generation is not None and run_generation_path.is_file():
            try:
                run_generation_matches = (
                    json.loads(run_generation_path.read_text(encoding="utf-8")).get(
                        "fixture_generation_id"
                    )
                    == fixture_generation
                )
            except (OSError, TypeError, ValueError):
                run_generation_matches = False
        retry_incomplete = (
            not artifact_complete
            and endpoint.num_nodes == 1
            and endpoint.node_rank == metrics_node_rank
        )
        if not force and not retry_incomplete:
            raise FileExistsError(f"run output already exists: {run_directory}")
        finalize_existing = (
            retry_incomplete
            and not artifact_generation_mismatch
            and run_generation_matches
            and _completed_capture_can_be_finalized(
                metrics_path,
                input_contract_directory,
                expected_steps=config.training.steps,
            )
        )
        if finalize_existing and not force:
            print(
                "Finalize completed capture without rerunning training: "
                f"{run_directory}"
            )
        elif retry_incomplete and not force:
            print(
                "Retry incomplete capture; replacing stale run output: "
                f"{run_directory}"
            )
        if not finalize_existing:
            shutil.rmtree(run_directory)
            if input_contract_directory.exists():
                shutil.rmtree(input_contract_directory)
    elif input_contract_directory.exists():
        print(
            "Discard orphaned input contract before capture retry: "
            f"{input_contract_directory}",
            flush=True,
        )
        shutil.rmtree(input_contract_directory)
    run_directory.mkdir(parents=True, exist_ok=True)
    if fixture_generation is not None:
        _write_json(
            run_generation_path,
            {"fixture_generation_id": fixture_generation},
        )

    environment = os.environ.copy()
    environment.update(endpoint.environment)
    environment["GLM5_EXPERIMENT_RUN_DIRECTORY"] = str(run_directory.resolve())
    environment[endpoint.visible_devices_env] = endpoint.visible_devices
    environment["TORCHTITAN_DEVICE"] = endpoint.torchtitan_device
    environment["GLM5_PRECISION_METRICS_PATH"] = str(metrics_path)
    # Every node writes unique global-rank files into one shared directory. The
    # metrics-owning node validates the complete world before publishing an
    # artifact, so a missing or incorrectly mapped node cannot pass silently.
    input_contract_directory.mkdir(parents=True, exist_ok=True)
    environment.update(
        fixed_input_environment(
            token_plan_path=token_plan_path,
            input_contract_directory=input_contract_directory,
            training=config.training,
            topology=endpoint.topology,
        )
    )
    environment["LOG_RANK"] = str(metrics_rank)
    command = _torchrun_command(
        root=root,
        config=config,
        endpoint=endpoint,
        dump_folder=run_directory / "trainer_output",
        checkpoint_path=checkpoint_path,
    )
    capture_label = (
        f"role={role}, topology={endpoint.topology.name}, "
        f"repeat={repeat}/{endpoint.repeats}, device={endpoint.device_type}, "
        f"world_size={endpoint.topology.world_size}"
    )
    capture_attempt_id = (
        _recorded_capture_attempt_id(capture_state_path)
        if finalize_existing
        else uuid.uuid4().hex
    )
    if not finalize_existing:
        print(
            f"Starting capture: {capture_label}\nRuntime log: {runtime_log}",
            flush=True,
        )
        capture_state = {
            "schema": "torchtitan.glm5_2.precision_capture_state",
            "status": "running",
            "pid": os.getpid(),
            "capture_attempt_id": capture_attempt_id,
            "fixture_generation_id": fixture_generation,
            "capture": capture_label,
        }
        _write_json(capture_state_path, capture_state)
        try:
            _run_process(
                command,
                root=root,
                environment=environment,
                log_path=runtime_log,
                log_context={
                    "Capture": capture_label,
                    "Capture attempt": capture_attempt_id,
                    "Fixture generation": fixture_generation,
                    "Orchestrator PID": os.getpid(),
                },
            )
        except subprocess.CalledProcessError as error:
            _write_json(
                capture_state_path,
                {
                    **capture_state,
                    "status": "failed",
                    "returncode": error.returncode,
                },
            )
            print(
                f"Capture failed: {capture_label}\nRuntime log: {runtime_log}",
                file=sys.stderr,
                flush=True,
            )
            raise
        _write_json(
            capture_state_path,
            {**capture_state, "status": "training_completed"},
        )

    if endpoint.node_rank != metrics_node_rank:
        return None
    metrics = read_captured_metrics(metrics_path)
    if len(metrics) != config.training.steps:
        raise RuntimeError(
            f"captured {len(metrics)} metric steps, expected {config.training.steps}"
        )
    input_contract = validate_runtime_input_contract(
        contract_directory=input_contract_directory,
        plan=load_token_plan(token_plan_path),
        steps=config.training.steps,
        global_batch_size=config.training.global_batch_size,
        training_local_batch_size=config.training.local_batch_size,
        dp_world_size=endpoint.topology.data_parallel_degree,
        context_parallel_degree=endpoint.topology.context_parallel_degree,
        tensor_parallel_degree=endpoint.topology.tensor_parallel_degree,
        pipeline_parallel_degree=endpoint.topology.pipeline_parallel_degree,
        node_rank=0,
        num_processes_per_node=endpoint.topology.world_size,
    )
    batch_schedule = training_topology_plan(
        config.training,
        endpoint.topology,
    )
    training_contract = {
        "scenario_name": config.storage_name,
        "training": _normalized_training(config.training),
        "checkpoint_sha256": fixture_manifest["checkpoint_sha256"],
        "data_sha256": fixture_manifest["data_sha256"],
        "token_plan": fixture_manifest["token_plan"],
        "endpoint_args": list(endpoint.extra_args),
        "topology": {
            **asdict(endpoint.topology),
            "batch_schedule": asdict(batch_schedule),
            "input_contract": input_contract,
        },
    }
    if fixture_generation is not None:
        training_contract["fixture_generation_id"] = fixture_generation
    metadata = {
        "experiment_kind": config.kind,
        "role": role,
        "endpoint_name": endpoint.name,
        "device_type": endpoint.device_type,
        "repeat": repeat,
        "node_rank": endpoint.node_rank,
        "capture_attempt_id": capture_attempt_id,
        "source": _source_metadata(root),
    }
    if "--profiler.enable_profiling" in endpoint.extra_args:
        metadata["profiler_output"] = str(
            run_directory / "trainer_output" / "profiling" / "traces"
        )
    artifact_path = PrecisionArtifactWriter(artifact_directory).write(
        metadata=metadata,
        training_contract=training_contract,
        metrics=metrics,
        attachments={
            "input_contract.json": input_contract_directory / "summary.json"
        },
    )
    _write_json(
        capture_state_path,
        {
            "schema": "torchtitan.glm5_2.precision_capture_state",
            "status": "completed",
            "pid": os.getpid(),
            "capture_attempt_id": capture_attempt_id,
            "fixture_generation_id": fixture_generation,
            "capture": capture_label,
            "artifact": str(artifact_path),
        },
    )
    return artifact_path


def reset_capture_outputs(
    root: Path,
    config: FormalExperimentConfig,
    *,
    role: Literal["reference", "candidate"],
    repeats: Sequence[int],
) -> None:
    """Remove the complete selected capture generation before rerunning it.

    Suite-level callers must reset every selected member before starting the
    first training process. Otherwise a mid-suite failure can combine newly
    captured members with untouched members from an older generation.
    """

    endpoint = config.reference if role == "reference" else config.candidate
    report_directory = _report_directory(root, config)
    selected_paths = [report_directory]
    run_directories: list[Path] = []
    for repeat in repeats:
        run_directory = _run_directory(root, config, role, endpoint, repeat)
        run_directories.append(run_directory)
        selected_paths.extend(
            (
                _artifact_directory(root, config, role, endpoint, repeat),
                run_directory,
                _input_contract_directory(
                    root, config, role, endpoint, repeat
                ),
            )
        )
    for run_directory in run_directories:
        assert_run_not_active(
            run_directory,
            state_name="capture_state.json",
        )
    reset_output_generation(selected_paths, label="precision capture")


def _apply_endpoint_overrides(
    endpoint: TrainingEndpoint,
    *,
    topology: ParallelTopology | None,
    device_type: str | None,
    visible_devices: str | None,
    num_nodes: int | None,
    node_rank: int | None,
    rendezvous_endpoint: str | None,
) -> TrainingEndpoint:
    values: dict[str, Any] = {}
    if topology is not None:
        values["topology"] = topology
    if device_type is not None:
        values["device_type"] = device_type
    if visible_devices is not None:
        values["visible_devices"] = visible_devices
    if num_nodes is not None:
        values["num_nodes"] = num_nodes
    if node_rank is not None:
        values["node_rank"] = node_rank
    if rendezvous_endpoint is not None:
        values["rendezvous_endpoint"] = rendezvous_endpoint
    return replace(endpoint, **values)


def _endpoint_from_process_environment(
    endpoint: TrainingEndpoint,
) -> TrainingEndpoint:
    """Use the standard accelerator visibility variable when it is exported."""

    visible_devices = os.environ.get(endpoint.visible_devices_env)
    if not visible_devices:
        return endpoint
    return replace(endpoint, visible_devices=visible_devices)


def _fixture_endpoint_from_environment(
    config: FormalExperimentConfig,
    requested_device: str | None,
) -> TrainingEndpoint:
    if requested_device is not None:
        matching = [
            endpoint
            for endpoint in (config.reference, config.candidate)
            if endpoint.device_type == requested_device
        ]
        if not matching:
            raise ValueError(
                f"the experiment has no {requested_device!r} execution backend"
            )
        return matching[0]
    npu_visible = os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if npu_visible and cuda_visible:
        raise ValueError(
            "both ASCEND_RT_VISIBLE_DEVICES and CUDA_VISIBLE_DEVICES are set; "
            "unset one or use --data-device to choose the fixture backend"
        )
    if npu_visible:
        for endpoint in (config.reference, config.candidate):
            if endpoint.device_type == "npu":
                return endpoint
        raise ValueError("the experiment has no NPU endpoint")
    if cuda_visible:
        for endpoint in (config.reference, config.candidate):
            if endpoint.device_type == "cuda":
                return endpoint
        raise ValueError("the experiment has no CUDA endpoint")
    raise ValueError(
        "fixture backend is ambiguous; export CUDA_VISIBLE_DEVICES or "
        "ASCEND_RT_VISIBLE_DEVICES, or pass --data-device"
    )


def run_formal_cli(
    base_config: FormalExperimentConfig,
    script_path: str,
    *,
    topologies: dict[str, ParallelTopology] | None = None,
) -> None:
    topology_registry = topologies or standard_topologies()
    parser = argparse.ArgumentParser(
        description="Formal multi-step TorchTitan precision benchmark"
    )
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--data", action="store_true", help="prepare shared fixture")
    actions.add_argument(
        "--capture",
        choices=("reference", "candidate"),
        help="capture one endpoint",
    )
    actions.add_argument("--compare", action="store_true", help="generate report")
    actions.add_argument("--list-topologies", action="store_true")
    parser.add_argument(
        "--repeat",
        type=int,
        help="capture only one repeat; omitted captures every configured repeat",
    )
    parser.add_argument(
        "--data-device",
        choices=("cuda", "npu"),
        help="fixture backend override; normally inferred from the exported visibility variable",
    )
    parser.add_argument("--topology", choices=sorted(topology_registry))
    parser.add_argument("--reference-topology", choices=sorted(topology_registry))
    parser.add_argument("--candidate-topology", choices=sorted(topology_registry))
    parser.add_argument("--device", choices=("cuda", "npu"))
    parser.add_argument("--reference-device", choices=("cuda", "npu"))
    parser.add_argument("--candidate-device", choices=("cuda", "npu"))
    parser.add_argument("--reference-visible-devices")
    parser.add_argument("--candidate-visible-devices")
    parser.add_argument("--num-nodes", type=int)
    parser.add_argument("--node-rank", type=int)
    parser.add_argument("--rdzv-endpoint")
    parser.add_argument(
        "--precision",
        choices=("fp32", "bf16", "full-bf16"),
        help="override the training and mixed-precision dtypes",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.list_topologies:
        for name, topology in topology_registry.items():
            try:
                batch_schedule: dict[str, Any] = asdict(
                    training_topology_plan(base_config.training, topology)
                )
            except ValueError as error:
                batch_schedule = {"invalid": str(error)}
            print(
                f"{name}: topology={asdict(topology)}, "
                f"batch_schedule={batch_schedule}"
            )
        return
    shared_topology = topology_registry.get(args.topology)
    reference_topology = topology_registry.get(args.reference_topology) or shared_topology
    candidate_topology = topology_registry.get(args.candidate_topology) or shared_topology
    shared_device = args.device
    reference = _apply_endpoint_overrides(
        _endpoint_from_process_environment(base_config.reference),
        topology=reference_topology,
        device_type=args.reference_device or shared_device,
        visible_devices=args.reference_visible_devices,
        num_nodes=args.num_nodes,
        node_rank=args.node_rank,
        rendezvous_endpoint=args.rdzv_endpoint,
    )
    candidate = _apply_endpoint_overrides(
        _endpoint_from_process_environment(base_config.candidate),
        topology=candidate_topology,
        device_type=args.candidate_device or shared_device,
        visible_devices=args.candidate_visible_devices,
        num_nodes=args.num_nodes,
        node_rank=args.node_rank,
        rendezvous_endpoint=args.rdzv_endpoint,
    )
    training = base_config.training
    if args.precision == "fp32":
        training = replace(
            training,
            training_dtype="float32",
            mixed_precision_param="float32",
        )
    elif args.precision == "bf16":
        training = replace(
            training,
            training_dtype="float32",
            mixed_precision_param="bfloat16",
        )
    elif args.precision == "full-bf16":
        training = replace(
            training,
            training_dtype="bfloat16",
            mixed_precision_param="bfloat16",
        )
    config = replace(
        base_config,
        reference=reference,
        candidate=candidate,
        training=training,
    )
    root = _root(script_path)

    if args.data:
        fixture_endpoint = _fixture_endpoint_from_environment(
            config, args.data_device
        )
        path = prepare_fixture(
            root,
            config,
            endpoint=fixture_endpoint,
            force=args.force,
        )
        print(f"Prepared fixture: {path}")
    elif args.capture:
        endpoint = config.reference if args.capture == "reference" else config.candidate
        repeats = tuple(
            (args.repeat,)
            if args.repeat is not None
            else range(1, endpoint.repeats + 1)
        )
        if args.force:
            reset_capture_outputs(
                root,
                config,
                role=args.capture,
                repeats=repeats,
            )
        for repeat in repeats:
            path = capture_endpoint(
                root,
                config,
                role=args.capture,
                repeat=repeat,
                force=False,
            )
            if path is None:
                print(
                    f"Repeat {repeat} completed; this node does not own the metrics artifact."
                )
            else:
                print(f"Captured repeat {repeat}: {path}")
    else:
        from .report import compare_and_write_report

        path = compare_and_write_report(root, config)
        print(f"Precision report: {path}")
