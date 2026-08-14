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

from .artifacts import PrecisionArtifactWriter, read_captured_metrics
from .standards import PrecisionStandard


ExperimentKind = Literal["migration", "self_consistency"]
CheckpointKind = Literal["random_seed", "converged"]


def _slug(value: str) -> str:
    result = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    if not result:
        raise ValueError(f"value does not produce a usable name: {value!r}")
    return result


@dataclass(frozen=True)
class ParallelTopology:
    name: str
    world_size: int
    data_parallel_replicate_degree: int = 1
    data_parallel_shard_degree: int = 1
    tensor_parallel_degree: int = 1
    pipeline_parallel_degree: int = 1
    expert_parallel_degree: int = 1
    pipeline_parallel_schedule: str = "1F1B"
    pipeline_parallel_microbatch_size: int = 1
    extra_args: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        degrees = (
            self.world_size,
            self.data_parallel_replicate_degree,
            self.data_parallel_shard_degree,
            self.tensor_parallel_degree,
            self.pipeline_parallel_degree,
            self.expert_parallel_degree,
            self.pipeline_parallel_microbatch_size,
        )
        if any(degree < 1 for degree in degrees):
            raise ValueError("parallel degrees and world size must be positive")
        dense_world_size = (
            self.data_parallel_replicate_degree
            * self.data_parallel_shard_degree
            * self.tensor_parallel_degree
            * self.pipeline_parallel_degree
        )
        if dense_world_size != self.world_size:
            raise ValueError(
                f"topology {self.name!r} covers {dense_world_size} dense ranks, "
                f"not world_size={self.world_size}"
            )
        sparse_width = (
            self.data_parallel_shard_degree * self.tensor_parallel_degree
        )
        if sparse_width % self.expert_parallel_degree != 0:
            raise ValueError(
                "dp_shard * tp must be divisible by expert_parallel_degree: "
                f"{sparse_width} % {self.expert_parallel_degree} != 0"
            )

    @property
    def slug(self) -> str:
        return _slug(self.name)

    @property
    def data_parallel_degree(self) -> int:
        return (
            self.data_parallel_replicate_degree
            * self.data_parallel_shard_degree
        )

    def command_args(self) -> list[str]:
        args = [
            "--parallelism.data_parallel_replicate_degree="
            f"{self.data_parallel_replicate_degree}",
            "--parallelism.data_parallel_shard_degree="
            f"{self.data_parallel_shard_degree}",
            f"--parallelism.tensor_parallel_degree={self.tensor_parallel_degree}",
            f"--parallelism.pipeline_parallel_degree={self.pipeline_parallel_degree}",
            f"--parallelism.expert_parallel_degree={self.expert_parallel_degree}",
        ]
        if self.tensor_parallel_degree > 1:
            args.append("--parallelism.no-enable-sequence-parallel")
        if self.pipeline_parallel_degree > 1:
            args.extend(
                [
                    "--parallelism.pipeline_parallel_schedule="
                    f"{self.pipeline_parallel_schedule}",
                    "--parallelism.pipeline_parallel_microbatch_size="
                    f"{self.pipeline_parallel_microbatch_size}",
                ]
            )
        args.extend(self.extra_args)
        return args


def standard_topologies() -> dict[str, ParallelTopology]:
    """Return reusable single-card and distributed GLM topology definitions."""

    return {
        "single": ParallelTopology("single", 1),
        "fsdp4": ParallelTopology("fsdp4", 4, data_parallel_shard_degree=4),
        "ddp8": ParallelTopology(
            "ddp8", 8, data_parallel_replicate_degree=8
        ),
        "fsdp8": ParallelTopology("fsdp8", 8, data_parallel_shard_degree=8),
        "tp8": ParallelTopology("tp8", 8, tensor_parallel_degree=8),
        "pp8": ParallelTopology("pp8", 8, pipeline_parallel_degree=8),
        "ep8": ParallelTopology(
            "fsdp8-ep8",
            8,
            data_parallel_shard_degree=8,
            expert_parallel_degree=8,
        ),
        "fsdp2-tp4": ParallelTopology(
            "fsdp2-tp4",
            8,
            data_parallel_shard_degree=2,
            tensor_parallel_degree=4,
        ),
        "fsdp4-tp2": ParallelTopology(
            "fsdp4-tp2",
            8,
            data_parallel_shard_degree=4,
            tensor_parallel_degree=2,
        ),
        "fsdp2-pp4": ParallelTopology(
            "fsdp2-pp4",
            8,
            data_parallel_shard_degree=2,
            pipeline_parallel_degree=4,
        ),
        "fsdp2-tp2-pp2": ParallelTopology(
            "fsdp2-tp2-pp2",
            8,
            data_parallel_shard_degree=2,
            tensor_parallel_degree=2,
            pipeline_parallel_degree=2,
        ),
        "fsdp2-tp4-ep8": ParallelTopology(
            "fsdp2-tp4-ep8",
            8,
            data_parallel_shard_degree=2,
            tensor_parallel_degree=4,
            expert_parallel_degree=8,
        ),
        "ddp16": ParallelTopology(
            "ddp16", 16, data_parallel_replicate_degree=16
        ),
        "fsdp16": ParallelTopology("fsdp16", 16, data_parallel_shard_degree=16),
    }


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
    fixed_global_batches: bool = False
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
    shared_reference_group: str | None = None

    def __post_init__(self) -> None:
        if self.kind == "migration":
            if self.reference.topology != self.candidate.topology:
                raise ValueError("migration requires the same parallel topology")
        elif self.reference.device_type != self.candidate.device_type:
            raise ValueError("self-consistency requires the same device type")

    @property
    def scenario_name(self) -> str:
        if self.shared_reference_group:
            sides = (
                f"{self.reference.device_type}-{self.reference.topology.slug}-vs-"
                f"{self.candidate.device_type}-{self.shared_reference_group}"
            )
        else:
            sides = (
                f"{self.reference.device_type}-{self.reference.topology.slug}-vs-"
                f"{self.candidate.device_type}-{self.candidate.topology.slug}"
            )
        input_suffix = "-fixed-inputs" if self.training.fixed_global_batches else ""
        return _slug(
            f"{self.name}-{self.kind}-{sides}-{self.training.precision_name}-"
            f"{self.training.checkpoint_kind}-steps{self.training.steps}-"
            f"global-batch{self.training.global_batch_size}-"
            f"seq{self.training.sequence_length}-seed{self.training.seed}"
            f"{input_suffix}"
        )

    @property
    def report_subdirectory(self) -> str | None:
        if self.shared_reference_group:
            return self.candidate.topology.slug
        return None


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
    import importlib.metadata
    import importlib.util

    sources: dict[str, Any] = {"torchtitan_test": _git_metadata(root)}
    packages: dict[str, str] = {}
    for package in ("torch", "torchtitan", "torchtitanturbo", "torch_npu"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = "not-installed"
        spec = importlib.util.find_spec(package)
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


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _root(script_path: str) -> Path:
    return Path(script_path).resolve().parents[2]


def _fixture_directory(root: Path, config: FormalExperimentConfig) -> Path:
    return root / config.fixture_root / config.scenario_name


def _artifact_directory(
    root: Path,
    config: FormalExperimentConfig,
    role: str,
    endpoint: TrainingEndpoint,
    repeat: int,
) -> Path:
    name = _slug(f"{role}-{endpoint.name}-{endpoint.topology.slug}-run-{repeat}")
    return root / config.artifact_root / config.scenario_name / name


def _run_directory(
    root: Path,
    config: FormalExperimentConfig,
    role: str,
    endpoint: TrainingEndpoint,
    repeat: int,
) -> Path:
    name = _slug(f"{role}-{endpoint.name}-{endpoint.topology.slug}-run-{repeat}")
    if endpoint.num_nodes > 1:
        name = f"{name}-node-{endpoint.node_rank}"
    return root / config.run_root / config.scenario_name / name


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
    if (
        config.training.global_batch_size
        % (config.training.local_batch_size * endpoint.topology.data_parallel_degree)
        != 0
    ):
        raise ValueError(
            "global batch size must be divisible by local batch size times data "
            f"parallel degree for {endpoint.topology.name}"
        )
    if (
        config.training.local_batch_size
        % endpoint.topology.pipeline_parallel_microbatch_size
        != 0
    ):
        raise ValueError("local batch size must be divisible by PP microbatch size")
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
        f"--rdzv_id={config.scenario_name}",
        f"--local-ranks-filter={metrics_rank % endpoint.num_processes_per_node}",
        "--role=rank",
        "--tee=3",
        "-m",
        "tests.glm5_2_precision.capture_metrics",
        "--module",
        config.training.module,
        "--config",
        config.training.config,
        *_base_training_args(config.training, dump_folder=dump_folder),
        *endpoint.topology.command_args(),
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
    fixed_batches_relative_path = None
    fixed_batches_sha256 = None
    if config.training.fixed_global_batches:
        if len(config.training.data_paths) != 2:
            raise ValueError(
                "fixed global batches require dataset and tokenizer data paths"
            )
        from .fixed_batches import materialize_fixed_batches

        fixed_batches_relative_path = "fixed_global_batches.safetensors"
        fixed_batches_path = fixture_directory / fixed_batches_relative_path
        print(
            "Materializing topology-invariant global batches: "
            f"{fixed_batches_path}"
        )
        materialize_fixed_batches(
            fixed_batches_path,
            dataset_path=root / config.training.data_paths[0],
            tokenizer_path=root / config.training.data_paths[1],
            steps=config.training.steps,
            global_batch_size=config.training.global_batch_size,
            sequence_length=config.training.sequence_length,
        )
        fixed_batches_sha256 = _directory_digest(fixed_batches_path)
        print(f"Materialized fixed batch digest: {fixed_batches_sha256}")
    if config.training.checkpoint_kind == "converged":
        checkpoint_path = Path(config.training.converged_checkpoint or "").resolve()
        if not checkpoint_path.is_dir():
            raise FileNotFoundError(checkpoint_path)
        relative_checkpoint = "checkpoint"
        shutil.copytree(checkpoint_path, fixture_directory / relative_checkpoint)
    else:
        output = fixture_directory / "seed_generation"
        endpoint = config.reference
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
        relative_checkpoint = matches[0].relative_to(fixture_directory).as_posix()

    local_checkpoint = fixture_directory / relative_checkpoint
    manifest = {
        "schema": "torchtitan.glm5_2.precision_fixture",
        "schema_version": 1,
        "scenario_name": config.scenario_name,
        "checkpoint_kind": config.training.checkpoint_kind,
        "checkpoint_relative_path": relative_checkpoint,
        "checkpoint_sha256": _directory_digest(local_checkpoint),
        "data_sha256": data_digests,
        "fixed_batches_relative_path": fixed_batches_relative_path,
        "fixed_batches_sha256": fixed_batches_sha256,
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
    if not 1 <= repeat <= endpoint.repeats:
        raise ValueError(f"repeat must be in [1, {endpoint.repeats}]")
    checkpoint_path = _seed_checkpoint_path(_fixture_directory(root, config))
    artifact_directory = _artifact_directory(root, config, role, endpoint, repeat)
    if artifact_directory.exists() and not force:
        raise FileExistsError(
            f"capture already exists; pass --force to replace it: {artifact_directory}"
        )
    run_directory = _run_directory(root, config, role, endpoint, repeat)
    if run_directory.exists():
        if not force:
            raise FileExistsError(f"run output already exists: {run_directory}")
        shutil.rmtree(run_directory)
    run_directory.mkdir(parents=True)
    metrics_path = run_directory / "raw_metrics.jsonl"
    runtime_log = run_directory / "runtime.log"

    environment = os.environ.copy()
    environment.update(endpoint.environment)
    environment[endpoint.visible_devices_env] = endpoint.visible_devices
    environment["TORCHTITAN_DEVICE"] = endpoint.torchtitan_device
    environment["GLM5_PRECISION_METRICS_PATH"] = str(metrics_path)
    if config.training.fixed_global_batches:
        from .fixed_batches import FIXED_BATCHES_ENV

        fixture_manifest = json.loads(
            (_fixture_directory(root, config) / "fixture.json").read_text(
                encoding="utf-8"
            )
        )
        fixed_batches_path = _fixture_directory(root, config) / str(
            fixture_manifest["fixed_batches_relative_path"]
        )
        expected_digest = fixture_manifest["fixed_batches_sha256"]
        if not fixed_batches_path.is_file():
            raise FileNotFoundError(fixed_batches_path)
        if _directory_digest(fixed_batches_path) != expected_digest:
            raise RuntimeError(f"fixed batch checksum mismatch: {fixed_batches_path}")
        environment[FIXED_BATCHES_ENV] = str(fixed_batches_path)
    metrics_rank = (
        0
        if endpoint.topology.pipeline_parallel_degree == 1
        or endpoint.topology.pipeline_parallel_schedule == "ZBVZeroBubble"
        else (
            endpoint.topology.world_size // endpoint.topology.pipeline_parallel_degree
        )
        * (endpoint.topology.pipeline_parallel_degree - 1)
    )
    environment["LOG_RANK"] = str(metrics_rank)
    command = _torchrun_command(
        root=root,
        config=config,
        endpoint=endpoint,
        dump_folder=run_directory / "trainer_output",
        checkpoint_path=checkpoint_path,
    )
    _run_process(
        command,
        root=root,
        environment=environment,
        log_path=runtime_log,
    )

    metrics_node_rank = metrics_rank // endpoint.num_processes_per_node
    if endpoint.node_rank != metrics_node_rank:
        return None
    metrics = read_captured_metrics(metrics_path)
    if len(metrics) != config.training.steps:
        raise RuntimeError(
            f"captured {len(metrics)} metric steps, expected {config.training.steps}"
        )
    fixture_manifest = json.loads(
        (_fixture_directory(root, config) / "fixture.json").read_text(encoding="utf-8")
    )
    training_contract = {
        "scenario_name": config.scenario_name,
        "training": _normalized_training(config.training),
        "checkpoint_sha256": fixture_manifest["checkpoint_sha256"],
        "data_sha256": fixture_manifest["data_sha256"],
        "fixed_batches_sha256": fixture_manifest.get("fixed_batches_sha256"),
        "topology": asdict(endpoint.topology),
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
    return PrecisionArtifactWriter(artifact_directory).write(
        metadata=metadata,
        training_contract=training_contract,
        metrics=metrics,
        attachments={
            "runtime.log": runtime_log,
            "raw_metrics.jsonl": metrics_path,
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
    parser.add_argument("--repeat", type=int, default=1)
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
            print(f"{name}: {asdict(topology)}")
        return
    shared_topology = topology_registry.get(args.topology)
    reference_topology = topology_registry.get(args.reference_topology) or shared_topology
    candidate_topology = topology_registry.get(args.candidate_topology) or shared_topology
    shared_device = args.device
    reference = _apply_endpoint_overrides(
        base_config.reference,
        topology=reference_topology,
        device_type=args.reference_device or shared_device,
        visible_devices=args.reference_visible_devices,
        num_nodes=args.num_nodes,
        node_rank=args.node_rank,
        rendezvous_endpoint=args.rdzv_endpoint,
    )
    candidate = _apply_endpoint_overrides(
        base_config.candidate,
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
        path = prepare_fixture(root, config, force=args.force)
        print(f"Prepared fixture: {path}")
    elif args.capture:
        path = capture_endpoint(
            root,
            config,
            role=args.capture,
            repeat=args.repeat,
            force=args.force,
        )
        if path is None:
            print("Capture completed; this node does not own the metrics artifact.")
        else:
            print(f"Captured artifact: {path}")
    else:
        from .report import compare_and_write_report

        path = compare_and_write_report(root, config)
        print(f"Precision report: {path}")
