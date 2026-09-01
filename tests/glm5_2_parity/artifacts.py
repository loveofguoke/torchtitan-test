# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Portable, checksummed artifacts for decoupled GLM-5.2 parity runs.

The artifact format intentionally stores only JSON metadata and safetensors
tensor shards.  Comparing artifacts therefore never imports the model that
produced them and never executes pickled Python objects.

Each observation key maps to one tensor plus semantic metadata (component,
layer, forward/backward role, dtype, and shape). Large captures are split into
bounded safetensors shards. The manifest stores exact hashes for every shard and
attachment, making directory synchronization detectable before comparison.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import sys
import uuid
from contextlib import ExitStack
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Iterator

import torch
from safetensors import safe_open
from safetensors.torch import save_file


SCHEMA_NAME = "torchtitan.parity-run"
SCHEMA_VERSION = 1
DEFAULT_SHARD_SIZE_BYTES = 1024**3


class ParityArtifactError(RuntimeError):
    """Raised when a parity artifact is incomplete, corrupt, or incompatible."""


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def json_digest(value: Any) -> str:
    """Return a stable SHA-256 digest for JSON-compatible data."""
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_digest(tensor: torch.Tensor) -> str:
    """Hash tensor dtype, shape, and exact CPU bytes."""
    value = tensor.detach().contiguous().cpu()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(_canonical_json(list(value.shape)))
    # NumPy does not expose every PyTorch dtype (notably bfloat16).  Viewing
    # the storage as bytes preserves the exact representation for all dtypes.
    digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class EndpointIdentity:
    """Identity of one implementation execution on one device backend."""

    implementation: str
    precision: str
    device_type: str
    device_name: str
    run_id: str
    rank: int = 0
    world_size: int = 1

    @property
    def label(self) -> str:
        label = (
            f"{self.implementation}:{self.precision}@{self.device_type}"
            f"[{self.run_id}]"
        )
        if self.world_size > 1:
            label += f"/rank{self.rank}-of-{self.world_size}"
        return label


@dataclass(frozen=True)
class ObservationMetadata:
    """Description used to align and compare one captured tensor."""

    key: str
    section_id: str
    scope: str
    component: str
    layer: int | str
    value_kind: str = "tensor"
    module_path: str = ""
    parent_path: str = ""
    level: int = 0
    node_kind: str = "checkpoint"
    checkpoint: bool = True
    dtype_flow: str = ""
    positions_key: str = ""
    compare: bool = True
    tags: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class StoredObservation:
    """Manifest entry for one tensor in one safetensors shard."""

    metadata: ObservationMetadata
    storage_key: str
    shard: str
    dtype: str
    shape: list[int]
    num_bytes: int
    sha256: str


def runtime_metadata() -> dict[str, str]:
    """Collect environment facts that do not require a device runtime."""
    metadata = {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "hostname": platform.node(),
    }
    for package in (
        "numpy",
        "safetensors",
        "transformers",
        "torchtitan",
        "torchtitanturbo",
        "torch-npu",
    ):
        try:
            metadata[package] = version(package)
        except PackageNotFoundError:
            metadata[package] = "unavailable"
    return metadata


class ParityArtifactWriter:
    """Build one immutable parity artifact directory."""

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        suite: str,
        suite_version: int,
        endpoint: EndpointIdentity,
        test_plan: dict[str, Any],
        configuration: dict[str, Any],
        fixture_digest: str,
        environment: dict[str, Any] | None = None,
        shard_size_bytes: int = DEFAULT_SHARD_SIZE_BYTES,
    ) -> None:
        self.path = Path(path)
        self.suite = suite
        self.suite_version = suite_version
        self.endpoint = endpoint
        self.test_plan = test_plan
        self.configuration = configuration
        self.fixture_digest = fixture_digest
        self.environment = environment or runtime_metadata()
        self.shard_size_bytes = shard_size_bytes
        self._entries: list[tuple[ObservationMetadata, torch.Tensor]] = []
        self._keys: set[str] = set()
        self._attachments: list[tuple[str, Path]] = []
        self.status = "success"
        self.failure = ""

    def add(self, metadata: ObservationMetadata, tensor: torch.Tensor) -> None:
        if metadata.key in self._keys:
            raise ValueError(f"duplicate parity observation key: {metadata.key}")
        if metadata.value_kind not in {"tensor", "discrete", "fixture"}:
            raise ValueError(
                f"unsupported observation value_kind: {metadata.value_kind}"
            )
        self._keys.add(metadata.key)
        value = tensor.detach().cpu().contiguous().clone()
        self._entries.append((metadata, value))

    def add_attachment(
        self,
        path: str | os.PathLike[str],
        *,
        name: str,
    ) -> None:
        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(f"parity attachment does not exist: {source}")
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"invalid parity attachment name: {name!r}")
        if any(existing == name for existing, _path in self._attachments):
            raise ValueError(f"duplicate parity attachment name: {name}")
        self._attachments.append((name, source))

    def mark_failed(self, failure: str) -> None:
        self.status = "failed"
        self.failure = failure

    @staticmethod
    def _num_bytes(tensor: torch.Tensor) -> int:
        return tensor.numel() * tensor.element_size()

    def _shards(self) -> Iterator[list[tuple[ObservationMetadata, torch.Tensor]]]:
        current: list[tuple[ObservationMetadata, torch.Tensor]] = []
        current_bytes = 0
        for entry in self._entries:
            entry_bytes = self._num_bytes(entry[1])
            if current and current_bytes + entry_bytes > self.shard_size_bytes:
                yield current
                current = []
                current_bytes = 0
            current.append(entry)
            current_bytes += entry_bytes
        if current:
            yield current

    def write(self) -> Path:
        if self.path.exists():
            raise FileExistsError(f"parity artifact already exists: {self.path}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.tmp-{uuid.uuid4().hex}")
        temporary.mkdir()
        observations: list[StoredObservation] = []
        shard_records: list[dict[str, Any]] = []
        attachment_records: list[dict[str, Any]] = []
        try:
            for shard_index, entries in enumerate(self._shards()):
                shard_name = f"tensors-{shard_index:05d}.safetensors"
                shard_path = temporary / shard_name
                tensors: dict[str, torch.Tensor] = {}
                pending: list[tuple[ObservationMetadata, torch.Tensor, str]] = []
                for entry_index, (metadata, tensor) in enumerate(entries):
                    storage_key = f"tensor_{shard_index:05d}_{entry_index:06d}"
                    tensors[storage_key] = tensor
                    pending.append((metadata, tensor, storage_key))
                save_file(tensors, str(shard_path))
                shard_records.append(
                    {
                        "file": shard_name,
                        "num_bytes": shard_path.stat().st_size,
                        "sha256": _file_digest(shard_path),
                    }
                )
                for metadata, tensor, storage_key in pending:
                    observations.append(
                        StoredObservation(
                            metadata=metadata,
                            storage_key=storage_key,
                            shard=shard_name,
                            dtype=str(tensor.dtype).removeprefix("torch."),
                            shape=list(tensor.shape),
                            num_bytes=self._num_bytes(tensor),
                            sha256=tensor_digest(tensor),
                        )
                    )

            for name, source in self._attachments:
                destination = temporary / "attachments" / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
                attachment_records.append(
                    {
                        "file": str(Path("attachments") / name).replace("\\", "/"),
                        "num_bytes": destination.stat().st_size,
                        "sha256": _file_digest(destination),
                    }
                )

            manifest = {
                "schema": SCHEMA_NAME,
                "schema_version": SCHEMA_VERSION,
                "suite": self.suite,
                "suite_version": self.suite_version,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "status": self.status,
                "failure": self.failure,
                "endpoint": asdict(self.endpoint),
                "test_plan": self.test_plan,
                "test_plan_digest": json_digest(self.test_plan),
                "fixture_digest": self.fixture_digest,
                "configuration": self.configuration,
                "configuration_digest": json_digest(self.configuration),
                "environment": self.environment,
                "shards": shard_records,
                "attachments": attachment_records,
                "observations": [asdict(entry) for entry in observations],
            }
            (temporary / "manifest.json").write_bytes(_canonical_json(manifest))
            complete = {
                "manifest_sha256": _file_digest(temporary / "manifest.json"),
                "num_observations": len(observations),
            }
            (temporary / "complete.json").write_bytes(_canonical_json(complete))
            temporary.replace(self.path)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        return self.path


class ParityArtifactReader:
    """Validate and lazily read one parity artifact."""

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        verify_shards: bool = True,
    ) -> None:
        self.path = Path(path)
        manifest_path = self.path / "manifest.json"
        complete_path = self.path / "complete.json"
        if not manifest_path.is_file() or not complete_path.is_file():
            raise ParityArtifactError(f"incomplete parity artifact: {self.path}")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self._exit_stack = ExitStack()
        self._shard_handles: dict[str, Any] = {}
        complete = json.loads(complete_path.read_text(encoding="utf-8"))
        if self.manifest.get("schema") != SCHEMA_NAME:
            raise ParityArtifactError(
                f"unsupported parity artifact schema: {self.manifest.get('schema')!r}"
            )
        if self.manifest.get("schema_version") != SCHEMA_VERSION:
            raise ParityArtifactError(
                "unsupported parity artifact schema version: "
                f"{self.manifest.get('schema_version')!r}"
            )
        if complete.get("manifest_sha256") != _file_digest(manifest_path):
            raise ParityArtifactError(f"manifest checksum mismatch: {self.path}")
        self._observations: dict[str, dict[str, Any]] = {}
        for entry in self.manifest.get("observations", []):
            key = entry["metadata"]["key"]
            if key in self._observations:
                raise ParityArtifactError(
                    f"duplicate observation key in manifest: {key}"
                )
            self._observations[key] = entry
        if complete.get("num_observations") != len(self._observations):
            raise ParityArtifactError(
                f"observation count mismatch in artifact: {self.path}"
            )
        if verify_shards:
            self._verify_shards()
            self._verify_attachments()

    @property
    def endpoint(self) -> EndpointIdentity:
        return EndpointIdentity(**self.manifest["endpoint"])

    @property
    def observation_keys(self) -> set[str]:
        return set(self._observations)

    def entry(self, key: str) -> dict[str, Any]:
        try:
            return self._observations[key]
        except KeyError as error:
            raise KeyError(
                f"observation not found in parity artifact: {key}"
            ) from error

    def metadata(self, key: str) -> ObservationMetadata:
        return ObservationMetadata(**self.entry(key)["metadata"])

    def tensor(self, key: str, *, verify: bool = True) -> torch.Tensor:
        entry = self.entry(key)
        shard_path = self.path / entry["shard"]
        handle = self._shard_handles.get(entry["shard"])
        if handle is None:
            handle = self._exit_stack.enter_context(
                safe_open(str(shard_path), framework="pt", device="cpu")
            )
            self._shard_handles[entry["shard"]] = handle
        value = handle.get_tensor(entry["storage_key"])
        if list(value.shape) != entry["shape"]:
            raise ParityArtifactError(f"shape mismatch for observation: {key}")
        if str(value.dtype).removeprefix("torch.") != entry["dtype"]:
            raise ParityArtifactError(f"dtype mismatch for observation: {key}")
        if verify and tensor_digest(value) != entry["sha256"]:
            raise ParityArtifactError(
                f"tensor checksum mismatch for observation: {key}"
            )
        return value

    def close(self) -> None:
        self._exit_stack.close()
        self._shard_handles.clear()

    def __enter__(self) -> "ParityArtifactReader":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()

    def _verify_shards(self) -> None:
        for shard in self.manifest.get("shards", []):
            path = self.path / shard["file"]
            if not path.is_file():
                raise ParityArtifactError(f"missing tensor shard: {path}")
            if path.stat().st_size != shard["num_bytes"]:
                raise ParityArtifactError(f"tensor shard size mismatch: {path}")
            if _file_digest(path) != shard["sha256"]:
                raise ParityArtifactError(f"tensor shard checksum mismatch: {path}")

    def _verify_attachments(self) -> None:
        for attachment in self.manifest.get("attachments", []):
            path = self.path / attachment["file"]
            if not path.is_file():
                raise ParityArtifactError(f"missing artifact attachment: {path}")
            if path.stat().st_size != attachment["num_bytes"]:
                raise ParityArtifactError(
                    f"artifact attachment size mismatch: {path}"
                )
            if _file_digest(path) != attachment["sha256"]:
                raise ParityArtifactError(
                    f"artifact attachment checksum mismatch: {path}"
                )

    def validate_compatible(self, other: "ParityArtifactReader") -> None:
        failed = [
            f"{reader.path}: {reader.manifest.get('failure', '')}"
            for reader in (self, other)
            if reader.manifest.get("status") != "success"
        ]
        if failed:
            raise ParityArtifactError(
                "cannot compare unsuccessful parity runs:\n" + "\n".join(failed)
            )
        checks = (
            ("suite", self.manifest.get("suite"), other.manifest.get("suite")),
            (
                "suite_version",
                self.manifest.get("suite_version"),
                other.manifest.get("suite_version"),
            ),
            (
                "test_plan_digest",
                self.manifest.get("test_plan_digest"),
                other.manifest.get("test_plan_digest"),
            ),
            (
                "fixture_digest",
                self.manifest.get("fixture_digest"),
                other.manifest.get("fixture_digest"),
            ),
            (
                "configuration_digest",
                self.manifest.get("configuration_digest"),
                other.manifest.get("configuration_digest"),
            ),
        )
        mismatches = [
            f"{name}: {left!r} != {right!r}"
            for name, left, right in checks
            if left != right
        ]
        if mismatches:
            raise ParityArtifactError(
                "parity artifacts are not comparable:\n" + "\n".join(mismatches)
            )
