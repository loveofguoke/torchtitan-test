#!/usr/bin/env python3
"""MindStudio Probe capture and TensorBoard visualization helpers.

The formal precision artifacts intentionally remain independent from this
instrumented diagnostic path.  msProbe hooks may introduce synchronization,
so their metrics must not be used as authoritative precision or performance
evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import wraps
from importlib import metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, Literal, Sequence


MSPROBE_CONFIG_PATH_ENV = "GLM5_MSPROBE_CONFIG_PATH"
SCHEMA = "torchtitan.glm5_2.msprobe_tensorboard"
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class MsprobeCaptureConfig:
    """Small, visualization-oriented msProbe capture configuration."""

    steps: tuple[int, ...] = (0,)
    ranks: tuple[int, ...] = ()
    task: Literal["statistics", "tensor"] = "statistics"
    level: Literal["L0", "mix"] = "mix"

    def __post_init__(self) -> None:
        if not self.steps or any(step < 0 for step in self.steps):
            raise ValueError("msProbe steps must contain non-negative step indexes")
        if tuple(sorted(set(self.steps))) != self.steps:
            raise ValueError("msProbe steps must be sorted and unique")
        if any(rank < 0 for rank in self.ranks):
            raise ValueError("msProbe ranks must be non-negative")
        if tuple(sorted(set(self.ranks))) != self.ranks:
            raise ValueError("msProbe ranks must be sorted and unique")

    def payload(self, dump_path: str | Path) -> dict[str, Any]:
        task_options: dict[str, Any] = {
            "scope": [],
            "list": [],
            "data_mode": ["all"],
            "summary_mode": "statistics",
        }
        return {
            "task": self.task,
            "dump_path": str(Path(dump_path).resolve()),
            "rank": list(self.ranks),
            "step": list(self.steps),
            "level": self.level,
            "async_dump": False,
            self.task: task_options,
        }


def write_capture_config(
    path: str | Path,
    *,
    dump_path: str | Path,
    config: MsprobeCaptureConfig,
) -> Path:
    """Write the official ``PrecisionDebugger`` JSON configuration."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(config.payload(dump_path), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination


def install_trainer_capture(config_path: str | Path | None = None) -> Any:
    """Wrap one TorchTitan process' train steps with ``PrecisionDebugger``."""

    value = config_path or os.environ.get(MSPROBE_CONFIG_PATH_ENV)
    if not value:
        raise RuntimeError(f"{MSPROBE_CONFIG_PATH_ENV} must name an msProbe config")
    path = Path(value).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)

    try:
        from msprobe.pytorch import PrecisionDebugger
    except ImportError as error:
        raise RuntimeError(
            "mindstudio-probe with PyTorch support is required for msProbe capture"
        ) from error

    from torchtitan.trainer import Trainer

    marker = "_glm5_msprobe_original_train_step"
    if hasattr(Trainer, marker):
        return getattr(Trainer, "_glm5_msprobe_debugger")

    debugger = PrecisionDebugger(config_path=str(path))
    original_train_step = Trainer.train_step

    @wraps(original_train_step)
    def train_step_with_msprobe(self: Any, *args: Any, **kwargs: Any) -> Any:
        # PrecisionDebugger explicitly accepts a list/tuple of model parts and
        # prefixes their module names with the local part index.
        debugger.start(model=self.model_parts)
        try:
            return original_train_step(self, *args, **kwargs)
        finally:
            try:
                debugger.stop()
            finally:
                debugger.step()

    setattr(Trainer, marker, original_train_step)
    setattr(Trainer, "_glm5_msprobe_debugger", debugger)
    Trainer.train_step = train_step_with_msprobe
    return debugger


def validate_dump_directory(path: str | Path) -> Path:
    """Require a completed L0/mix dump suitable for both visualizers."""

    directory = Path(path)
    dump_files = list(directory.rglob("dump.json")) if directory.is_dir() else []
    construct_files = (
        list(directory.rglob("construct.json")) if directory.is_dir() else []
    )
    if not dump_files:
        raise RuntimeError(f"no msProbe dump.json found under {directory}")
    if not construct_files:
        raise RuntimeError(f"no msProbe construct.json found under {directory}")
    return directory


def _dump_leaf_directories(path: Path) -> tuple[Path, ...]:
    dump_parents = {item.parent for item in path.rglob("dump.json")}
    construct_parents = {item.parent for item in path.rglob("construct.json")}
    return tuple(sorted(dump_parents & construct_parents))


def _resolve_executable(name: str, explicit: str | Path | None) -> str:
    if explicit is not None:
        candidate = Path(explicit)
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
        return str(candidate.resolve())
    candidate = shutil.which(name)
    if candidate is None:
        raise RuntimeError(f"required executable is not on PATH: {name}")
    return candidate


def _package_version() -> str:
    try:
        return metadata.version("mindstudio-probe")
    except metadata.PackageNotFoundError:
        return "unknown"


def tensorboard_plugins() -> tuple[str, ...]:
    """Return installed TensorBoard plugin entry-point names."""

    entry_points = metadata.entry_points()
    if hasattr(entry_points, "select"):
        selected = entry_points.select(group="tensorboard_plugins")
    else:  # pragma: no cover - Python/importlib compatibility path
        selected = entry_points.get("tensorboard_plugins", ())
    return tuple(sorted(entry_point.name for entry_point in selected))


def require_visualization_plugins() -> tuple[str, ...]:
    plugins = tensorboard_plugins()
    by_lower_name = {name.lower() for name in plugins}
    required = {"graph_ascend", "trendvis"}
    missing = sorted(required - by_lower_name)
    if missing:
        raise RuntimeError(
            "mindstudio-probe is missing TensorBoard visualization plugins: "
            f"{missing}; install a build containing tb_graph_ascend and "
            "trend_analyzer"
        )
    return plugins


def _run(command: Sequence[str]) -> None:
    subprocess.run(list(command), check=True)


def _move_single_trend_database(source: Path, destination: Path) -> None:
    matches = list(source.glob("*.trend.db"))
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one .trend.db from msprobe data2db in {source}, got {matches}"
        )
    shutil.move(str(matches[0]), destination)
    shutil.rmtree(source)


def build_tensorboard_assets(
    *,
    reference_dump: str | Path,
    candidate_dump: str | Path,
    output: str | Path,
    force: bool = False,
    resume: bool = False,
    msprobe_executable: str | Path | None = None,
) -> Path:
    """Build hierarchy and trend databases consumed by TensorBoard."""

    if force and resume:
        raise ValueError("force and resume are mutually exclusive")
    reference = validate_dump_directory(reference_dump).resolve()
    candidate = validate_dump_directory(candidate_dump).resolve()
    # A single-rank hierarchy comparison is portable across TorchTitan's
    # single/FSDP/DDP layouts. The Trend Analyzer still receives the complete
    # roots and therefore retains every captured step and rank.
    graph_reference = _dump_leaf_directories(reference)[0]
    graph_candidate = _dump_leaf_directories(candidate)[0]
    destination = Path(output).resolve()
    manifest_path = destination / "msprobe_tensorboard.json"
    expected = (
        destination / "reference.trend.db",
        destination / "candidate.trend.db",
    )
    if destination.exists():
        if resume:
            completed = False
            if manifest_path.is_file() and all(path.is_file() for path in expected):
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                completed = (
                    manifest.get("reference_dump") == str(reference)
                    and manifest.get("candidate_dump") == str(candidate)
                    and bool(list(destination.glob("*.vis.db")))
                )
            if completed:
                return destination
            shutil.rmtree(destination)
        elif not force:
            raise FileExistsError(
                f"TensorBoard output already exists; use force or resume: {destination}"
            )
        else:
            shutil.rmtree(destination)
    destination.mkdir(parents=True)

    executable = _resolve_executable("msprobe", msprobe_executable)
    commands: list[list[str]] = []

    graph_command = [
        executable,
        "graph_visualize",
        "-tp",
        str(graph_candidate),
        "-gp",
        str(graph_reference),
        "-o",
        str(destination),
    ]
    _run(graph_command)
    commands.append(graph_command)

    for label, dump_path in (("reference", reference), ("candidate", candidate)):
        scratch = destination / f".{label}-data2db"
        command = [
            executable,
            "data2db",
            "--data",
            str(dump_path),
            "--db",
            str(scratch),
            "--format",
            "dump",
        ]
        _run(command)
        commands.append(command)
        _move_single_trend_database(
            scratch, destination / f"{label}.trend.db"
        )

    graph_databases = sorted(destination.glob("*.vis.db"))
    if not graph_databases:
        raise RuntimeError(
            f"msprobe graph_visualize produced no .vis.db in {destination}"
        )

    payload = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "mindstudio_probe_version": _package_version(),
        "reference_dump": str(reference),
        "candidate_dump": str(candidate),
        "graph_reference_dump": str(graph_reference),
        "graph_candidate_dump": str(graph_candidate),
        "output": str(destination),
        "assets": sorted(path.name for path in destination.glob("*.db")),
        "commands": commands,
        "tensorboard": {
            "logdir": str(destination),
            "tabs": ["GRAPH_ASCEND", "TREND ANALYZER"],
        },
        "warning": (
            "Diagnostic msProbe capture may add synchronization; do not use its "
            "metrics as formal precision or throughput evidence."
        ),
    }
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return destination


def tensorboard_command(
    logdir: str | Path,
    *,
    port: int = 6006,
    bind_all: bool = False,
    tensorboard_executable: str | Path | None = None,
) -> list[str]:
    if not 1 <= port <= 65535:
        raise ValueError("TensorBoard port must be in [1, 65535]")
    executable = _resolve_executable("tensorboard", tensorboard_executable)
    command = [
        executable,
        "--logdir",
        str(Path(logdir).resolve()),
        "--port",
        str(port),
    ]
    if bind_all:
        command.append("--bind_all")
    return command


def serve_tensorboard(
    logdir: str | Path,
    *,
    port: int = 6006,
    bind_all: bool = False,
    tensorboard_executable: str | Path | None = None,
) -> None:
    require_visualization_plugins()
    _run(
        tensorboard_command(
            logdir,
            port=port,
            bind_all=bind_all,
            tensorboard_executable=tensorboard_executable,
        )
    )


__all__ = [
    "MSPROBE_CONFIG_PATH_ENV",
    "MsprobeCaptureConfig",
    "build_tensorboard_assets",
    "install_trainer_capture",
    "require_visualization_plugins",
    "serve_tensorboard",
    "tensorboard_command",
    "validate_dump_directory",
    "write_capture_config",
]
