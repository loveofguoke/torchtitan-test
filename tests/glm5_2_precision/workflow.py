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
import re
import shutil
import subprocess
import sys
from typing import Any, Literal, Sequence

from tests.glm5_2_common.topology import ParallelTopology, standard_topologies

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
    result = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    if not result:
        raise ValueError(f"value does not produce a usable name: {value!r}")
    return result


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
    def storage_name(self) -> str:
        """Return a compact, human-readable directory and rendezvous name."""

        if self.kind == "migration":
            experiment = (
                f"migration-{self.reference.device_type}-"
                f"{self.candidate.device_type}-{self.reference.topology.slug}"
            )
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
        value = _slug(
            f"{experiment}-{precision}-{checkpoint}-s{self.training.steps}-"
            f"b{self.training.global_batch_size}-seq{self.training.sequence_length}-"
            f"seed{self.training.seed}"
        )
        if len(value) <= 112:
            return value
        digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:10]
        return f"{value[:101].rstrip('-')}-{digest}"

    @property
    def fixture_storage_name(self) -> str:
        """Return the topology-independent identity of synchronized inputs."""

        if self.fixture_name is not None:
            return _slug(self.fixture_name)
        training = _normalized_training(self.training)
        digest = hashlib.sha256(
            json.dumps(training, sort_keys=True).encode("utf-8")
        ).hexdigest()[:10]
        precision = {
            "mixed-bfloat16": "bf16",
            "mixed-float32": "fp32",
            "full-bf16": "full-bf16",
        }.get(self.training.precision_name, self.training.precision_name)
        return _slug(
            f"fixture-{self.training.module}-{precision}-"
            f"s{self.training.steps}-b{self.training.global_batch_size}-"
            f"seq{self.training.sequence_length}-seed{self.training.seed}-{digest}"
        )


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
    return {"packages": packages, "sources": sources}


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


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _root(script_path: str) -> Path:
    return Path(script_path).resolve().parents[2]


def _fixture_directory(root: Path, config: FormalExperimentConfig) -> Path:
    return root / config.fixture_root / config.fixture_storage_name


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
    return root / config.artifact_root / config.storage_name / name


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
    return root / config.run_root / config.storage_name / name


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


def _base_training_args(
    config: FormalTrainingConfig,
    *,
    dump_folder: Path,
) -> list[str]:
    args = [
        f"--dump_folder={dump_folder}",
        f"--training.steps={config.steps}",
        f"--training.local_batch_size={config.local_batch_size}",
        f"--training.global_batch_size={config.global_batch_size}",
        f"--training.seq_len={config.sequence_length}",
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
        *_base_training_args(config.training, dump_folder=dump_folder),
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
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
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
        if not force:
            raise FileExistsError(
                f"fixture already exists; pass --force to replace it: {fixture_directory}"
            )
        shutil.rmtree(fixture_directory)
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
    _run_process(
        token_plan_command,
        root=root,
        environment=token_plan_environment,
        log_path=fixture_directory / "token_generation.log",
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
                *_base_training_args(config.training, dump_folder=output),
                "--parallelism.data_parallel_replicate_degree=1",
                "--parallelism.data_parallel_shard_degree=1",
                "--parallelism.tensor_parallel_degree=1",
                "--parallelism.pipeline_parallel_degree=1",
                "--parallelism.expert_parallel_degree=1",
                "--checkpoint.enable",
                "--checkpoint.create_seed_checkpoint",
            ]
        )
        _run_process(
            command,
            root=root,
            environment=environment,
            log_path=fixture_directory / "seed_generation.log",
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
    fixture_directory = _fixture_directory(root, config)
    checkpoint_path = _seed_checkpoint_path(fixture_directory)
    token_plan_path = _token_plan_path(fixture_directory)
    artifact_directory = _artifact_directory(root, config, role, endpoint, repeat)
    artifact_complete = False
    if endpoint.node_rank == metrics_node_rank and artifact_directory.exists():
        try:
            PrecisionArtifactReader(artifact_directory)
            artifact_complete = True
        except (KeyError, OSError, TypeError, ValueError, PrecisionArtifactError):
            artifact_complete = False
        if artifact_complete and not force:
            print(f"Skip completed capture: {artifact_directory}")
            return artifact_directory
    run_directory = _run_directory(root, config, role, endpoint, repeat)
    metrics_path = run_directory / "raw_metrics.jsonl"
    runtime_log = run_directory / "runtime.log"
    input_contract_directory = (
        root
        / config.run_root
        / config.storage_name
        / (
            f"{_artifact_directory(root, config, role, endpoint, repeat).name}"
            "-input-contract"
        )
    )
    finalize_existing = False
    if run_directory.exists():
        retry_incomplete = (
            not artifact_complete
            and endpoint.num_nodes == 1
            and endpoint.node_rank == metrics_node_rank
        )
        if not force and not retry_incomplete:
            raise FileExistsError(f"run output already exists: {run_directory}")
        finalize_existing = retry_incomplete and _completed_capture_can_be_finalized(
            metrics_path,
            input_contract_directory,
            expected_steps=config.training.steps,
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
    run_directory.mkdir(parents=True, exist_ok=True)

    environment = os.environ.copy()
    environment.update(endpoint.environment)
    environment[endpoint.visible_devices_env] = endpoint.visible_devices
    environment["TORCHTITAN_DEVICE"] = endpoint.torchtitan_device
    environment["GLM5_PRECISION_METRICS_PATH"] = str(metrics_path)
    # Every node writes unique global-rank files into one shared directory. The
    # metrics-owning node validates the complete world before publishing an
    # artifact, so a missing or incorrectly mapped node cannot pass silently.
    input_contract_directory.mkdir(parents=True, exist_ok=True)
    environment["GLM5_PRECISION_TOKEN_PLAN_PATH"] = str(token_plan_path)
    environment["GLM5_PRECISION_INPUT_CONTRACT_DIR"] = str(
        input_contract_directory
    )
    environment["GLM5_PRECISION_GLOBAL_BATCH_SIZE"] = str(
        config.training.global_batch_size
    )
    environment["GLM5_PRECISION_TRAINING_LOCAL_BATCH_SIZE"] = str(
        config.training.local_batch_size
    )
    environment["GLM5_PRECISION_TENSOR_PARALLEL_DEGREE"] = str(
        endpoint.topology.tensor_parallel_degree
    )
    environment["GLM5_PRECISION_CONTEXT_PARALLEL_DEGREE"] = str(
        endpoint.topology.context_parallel_degree
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
    if not finalize_existing:
        print(
            f"Starting capture: {capture_label}\nRuntime log: {runtime_log}",
            flush=True,
        )
        try:
            _run_process(
                command,
                root=root,
                environment=environment,
                log_path=runtime_log,
            )
        except subprocess.CalledProcessError:
            print(
                f"Capture failed: {capture_label}\nRuntime log: {runtime_log}",
                file=sys.stderr,
                flush=True,
            )
            raise

    if endpoint.node_rank != metrics_node_rank:
        return None
    metrics = read_captured_metrics(metrics_path)
    if len(metrics) != config.training.steps:
        raise RuntimeError(
            f"captured {len(metrics)} metric steps, expected {config.training.steps}"
        )
    fixture_manifest = json.loads(
        (fixture_directory / "fixture.json").read_text(encoding="utf-8")
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
    metadata = {
        "experiment_kind": config.kind,
        "role": role,
        "endpoint_name": endpoint.name,
        "device_type": endpoint.device_type,
        "repeat": repeat,
        "node_rank": endpoint.node_rank,
        "source": _source_metadata(root),
    }
    if "--profiler.enable_profiling" in endpoint.extra_args:
        metadata["profiler_output"] = str(
            run_directory / "trainer_output" / "profiling" / "traces"
        )
    return PrecisionArtifactWriter(artifact_directory).write(
        metadata=metadata,
        training_contract=training_contract,
        metrics=metrics,
        attachments={
            "input_contract.json": input_contract_directory / "summary.json"
        },
    )


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
        repeats = (args.repeat,) if args.repeat is not None else range(1, endpoint.repeats + 1)
        for repeat in repeats:
            path = capture_endpoint(
                root,
                config,
                role=args.capture,
                repeat=repeat,
                force=args.force,
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
