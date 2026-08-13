# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Small, portable artifacts for offline formal precision comparison."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any, Iterable


SCHEMA = "torchtitan.glm5_2.training_precision"
SCHEMA_VERSION = 1
LOSS_KEY = "loss_metrics/global_avg_loss"
MAX_LOSS_KEY = "loss_metrics/global_max_loss"
GRAD_NORM_KEY = "grad_norm"


class PrecisionArtifactError(RuntimeError):
    pass


@dataclass(frozen=True)
class TrainingMetric:
    step: int
    loss: float
    global_max_loss: float
    grad_norm: float
    extras: dict[str, float]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def read_captured_metrics(path: str | Path) -> tuple[TrainingMetric, ...]:
    records: dict[int, TrainingMetric] = {}
    source = Path(path)
    if not source.is_file():
        raise PrecisionArtifactError(f"captured metrics not found: {source}")
    for line_number, line in enumerate(
        source.read_text(encoding="utf-8").splitlines(), start=1
    ):
        payload = json.loads(line)
        step = int(payload["step"])
        values = payload["metrics"]
        missing = [
            key for key in (LOSS_KEY, MAX_LOSS_KEY, GRAD_NORM_KEY) if key not in values
        ]
        if missing:
            raise PrecisionArtifactError(
                f"metrics line {line_number} is missing {missing}: {source}"
            )
        if step in records:
            raise PrecisionArtifactError(f"duplicate captured training step {step}")
        extras = {
            key: float(value)
            for key, value in values.items()
            if key not in {LOSS_KEY, MAX_LOSS_KEY, GRAD_NORM_KEY}
        }
        records[step] = TrainingMetric(
            step=step,
            loss=float(values[LOSS_KEY]),
            global_max_loss=float(values[MAX_LOSS_KEY]),
            grad_norm=float(values[GRAD_NORM_KEY]),
            extras=extras,
        )
    if not records:
        raise PrecisionArtifactError(f"captured metrics are empty: {source}")
    return tuple(records[step] for step in sorted(records))


class PrecisionArtifactWriter:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def write(
        self,
        *,
        metadata: dict[str, Any],
        training_contract: dict[str, Any],
        metrics: Iterable[TrainingMetric],
        attachments: dict[str, str | Path] | None = None,
    ) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(
            tempfile.mkdtemp(prefix=f".{self.path.name}.", dir=self.path.parent)
        )
        try:
            metrics_path = temporary / "metrics.jsonl"
            with metrics_path.open("w", encoding="utf-8") as stream:
                for metric in metrics:
                    stream.write(
                        json.dumps(asdict(metric), sort_keys=True, separators=(",", ":"))
                    )
                    stream.write("\n")
            contract_path = temporary / "training_contract.json"
            _write_json(contract_path, training_contract)
            files = {
                "metrics.jsonl": _sha256(metrics_path),
                "training_contract.json": _sha256(contract_path),
            }
            for name, source_value in (attachments or {}).items():
                if Path(name).name != name:
                    raise ValueError(f"attachment name must be a file name: {name!r}")
                source = Path(source_value)
                if not source.is_file():
                    raise PrecisionArtifactError(f"attachment not found: {source}")
                destination = temporary / "attachments" / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
                files[destination.relative_to(temporary).as_posix()] = _sha256(
                    destination
                )
            manifest = {
                "schema": SCHEMA,
                "schema_version": SCHEMA_VERSION,
                "metadata": metadata,
                "files": files,
            }
            _write_json(temporary / "manifest.json", manifest)
            if self.path.exists():
                shutil.rmtree(self.path)
            temporary.replace(self.path)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        return self.path


class PrecisionArtifactReader:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        manifest_path = self.path / "manifest.json"
        if not manifest_path.is_file():
            raise PrecisionArtifactError(f"artifact manifest not found: {manifest_path}")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("schema") != SCHEMA:
            raise PrecisionArtifactError(
                f"unsupported precision artifact schema: {self.manifest.get('schema')!r}"
            )
        if self.manifest.get("schema_version") != SCHEMA_VERSION:
            raise PrecisionArtifactError(
                "unsupported precision artifact schema version: "
                f"{self.manifest.get('schema_version')!r}"
            )
        for name, expected_digest in self.manifest["files"].items():
            file_path = self.path / name
            if not file_path.is_file():
                raise PrecisionArtifactError(f"artifact file not found: {file_path}")
            observed_digest = _sha256(file_path)
            if observed_digest != expected_digest:
                raise PrecisionArtifactError(
                    f"artifact checksum mismatch for {file_path}: "
                    f"{observed_digest} != {expected_digest}"
                )
        self.metadata: dict[str, Any] = self.manifest["metadata"]
        self.training_contract: dict[str, Any] = json.loads(
            (self.path / "training_contract.json").read_text(encoding="utf-8")
        )
        self.metrics = self._read_metrics()
        steps = [metric.step for metric in self.metrics]
        if steps != sorted(steps) or len(steps) != len(set(steps)):
            raise PrecisionArtifactError(
                f"artifact metric steps are not strictly increasing: {self.path}"
            )

    def _read_metrics(self) -> tuple[TrainingMetric, ...]:
        metrics: list[TrainingMetric] = []
        for line in (self.path / "metrics.jsonl").read_text(
            encoding="utf-8"
        ).splitlines():
            payload = json.loads(line)
            metrics.append(
                TrainingMetric(
                    step=int(payload["step"]),
                    loss=float(payload["loss"]),
                    global_max_loss=float(payload["global_max_loss"]),
                    grad_norm=float(payload["grad_norm"]),
                    extras={
                        key: float(value) for key, value in payload["extras"].items()
                    },
                )
            )
        if not metrics:
            raise PrecisionArtifactError(f"artifact contains no metrics: {self.path}")
        return tuple(metrics)

    def loss_series(self) -> dict[int, float]:
        return {metric.step: metric.loss for metric in self.metrics}

    def grad_norm_series(self) -> dict[int, float]:
        return {metric.step: metric.grad_norm for metric in self.metrics}
