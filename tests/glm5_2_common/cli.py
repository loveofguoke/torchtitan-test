# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Lifecycle and CLI helpers shared by independent experiment families.

The central invariant is generation safety. ``--force`` removes the complete
selected generation before any new worker starts. A non-force retry reuses only
outputs whose completion marker and experiment identity are valid; incomplete
directories are archived or retried by the owning workflow. ``RunAttempt``
adds an atomic state file so another launcher never deletes a live run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence
import uuid


def archive_previous_output(path: Path) -> Path | None:
    """Move stale experiment output aside without overwriting prior evidence."""

    if not path.exists():
        return None
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    destination = path.with_name(f".{path.name}.previous-{timestamp}")
    suffix = 1
    while destination.exists():
        destination = path.with_name(
            f".{path.name}.previous-{timestamp}-{suffix}"
        )
        suffix += 1
    path.rename(destination)
    return destination


def process_is_running(pid: int) -> bool:
    """Return whether a recorded orchestrator PID is still alive."""

    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def assert_run_not_active(
    path: Path,
    *,
    state_name: str | None = None,
) -> None:
    """Refuse to replace a run whose recorded orchestrator is still alive."""

    state_names = (
        (state_name,)
        if state_name is not None
        else ("run_state.json", "capture_state.json")
    )
    for candidate in state_names:
        state_path = path / candidate
        lock_path = state_path.with_suffix(state_path.suffix + ".lock")
        if lock_path.is_file():
            try:
                lock = json.loads(lock_path.read_text(encoding="utf-8"))
                lock_pid = int(lock.get("pid", -1))
            except (OSError, TypeError, ValueError) as error:
                raise RuntimeError(
                    f"run lock is unreadable and cannot be replaced safely: "
                    f"{lock_path}"
                ) from error
            if process_is_running(lock_pid):
                raise RuntimeError(
                    f"run is still active with orchestrator PID {lock_pid}: "
                    f"{path}; stop that process before retrying or forcing "
                    "the experiment"
                )
        if not state_path.is_file():
            continue
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            pid = int(state.get("pid", -1))
        except (OSError, TypeError, ValueError):
            continue
        if (
            state.get("status") == "running"
            and process_is_running(pid)
        ):
            raise RuntimeError(
                f"run is still active with orchestrator PID {pid}: {path}; "
                "stop that process before retrying or forcing the experiment"
            )


@dataclass
class RunAttempt:
    """Auditable lifecycle marker shared by long-running experiments.

    ``attempt_id`` identifies one orchestrator invocation, not an experiment
    configuration. The configuration hash names storage; the attempt id tells
    whether metrics/logs in that storage came from the same training process.
    State is atomically replaced so readers never observe partial JSON.
    """

    directory: Path
    kind: str
    context: dict[str, Any] = field(default_factory=dict)
    attempt_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    pid: int = field(default_factory=os.getpid)
    state_name: str = "run_state.json"

    @property
    def lock_path(self) -> Path:
        return self.state_path.with_suffix(self.state_path.suffix + ".lock")

    def _acquire_lock(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            {
                "schema": "torchtitan.glm5_2.run_lock",
                "schema_version": 1,
                "attempt_id": self.attempt_id,
                "pid": self.pid,
            },
            sort_keys=True,
        ).encode("utf-8") + b"\n"
        temporary = self.lock_path.with_name(
            f".{self.lock_path.name}.{self.attempt_id}.tmp"
        )
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        try:
            os.write(descriptor, payload)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        try:
            try:
                # The hard-link create is atomic and publishes only the
                # already complete owner payload. It avoids a crash window
                # where another launcher sees a zero-byte lock.
                os.link(temporary, self.lock_path)
            except FileExistsError:
                assert_run_not_active(
                    self.directory,
                    state_name=self.state_name,
                )
                raise RuntimeError(
                    "a stale run lock remains after an interrupted launcher; "
                    "archive or force-reset the incomplete generation before "
                    f"retrying: {self.lock_path}"
                ) from None
            return
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass

    def _release_lock(self) -> None:
        try:
            lock = json.loads(self.lock_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return
        except (OSError, TypeError, ValueError) as error:
            raise RuntimeError(
                f"run lock is unreadable and cannot be released safely: "
                f"{self.lock_path}"
            ) from error
        if lock.get("attempt_id") != self.attempt_id:
            raise RuntimeError(
                f"run lock belongs to another attempt: {self.lock_path}"
            )
        self.lock_path.unlink()

    @classmethod
    def start(
        cls,
        directory: Path,
        *,
        kind: str,
        context: dict[str, Any] | None = None,
        state_name: str = "run_state.json",
    ) -> "RunAttempt":
        attempt = cls(
            directory=directory,
            kind=kind,
            context=dict(context or {}),
            state_name=state_name,
        )
        attempt._acquire_lock()
        try:
            attempt.update("running")
        except BaseException:
            attempt._release_lock()
            raise
        return attempt

    @property
    def state_path(self) -> Path:
        return self.directory / self.state_name

    @property
    def log_context(self) -> dict[str, Any]:
        return {
            "Run kind": self.kind,
            "Run attempt": self.attempt_id,
            "Orchestrator PID": self.pid,
            **self.context,
        }

    def update(self, status: str, **values: Any) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        state = {
            "schema": "torchtitan.glm5_2.run_state",
            "schema_version": 1,
            "kind": self.kind,
            "status": status,
            "attempt_id": self.attempt_id,
            "pid": self.pid,
            "context": self.context,
            **values,
        }
        temporary = self.state_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(state, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.state_path)
        if status != "running":
            self._release_lock()


def reset_output_generation(
    paths: Sequence[Path],
    *,
    active_run_directories: Sequence[Path] = (),
    label: str = "experiment",
    include_archives: bool = True,
) -> None:
    """Remove and verify every selected output before a new generation.

    Callers pass all coupled roots -- run data, compact artifacts, contracts,
    and reports -- rather than deleting one directory ad hoc. Previous/failed
    archives are included by default so a forced run cannot later adopt stale
    evidence. Live orchestrator state is checked before the first deletion.
    """

    selected_paths = list(dict.fromkeys(paths))
    expanded_paths = list(selected_paths)
    if include_archives:
        for path in selected_paths:
            if not path.parent.exists():
                continue
            patterns = (
                f".{path.name}.previous-*",
                f"{path.name}.previous-*",
                f".{path.name}.failed-*",
                f"{path.name}.failed-*",
            )
            for pattern in patterns:
                expanded_paths.extend(path.parent.glob(pattern))
    unique_paths = list(dict.fromkeys(expanded_paths))
    for run_directory in dict.fromkeys(active_run_directories):
        assert_run_not_active(run_directory)
    for path in unique_paths:
        if path.is_dir():
            assert_run_not_active(path)
    existing_paths = [path for path in unique_paths if path.exists()]
    for path in unique_paths:
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()
    remaining = [path for path in existing_paths if path.exists()]
    if remaining:
        raise RuntimeError(
            f"failed to reset {label} outputs: "
            + ", ".join(str(path) for path in remaining)
        )
    if existing_paths:
        print(f"Removed {label} outputs for forced rerun:", flush=True)
        for path in existing_paths:
            print(f"  {path}", flush=True)
    else:
        print(f"No existing {label} outputs required removal.", flush=True)


def replace_topology(argv: Sequence[str], topology: str) -> list[str]:
    """Return argv with exactly one topology selection."""

    result: list[str] = []
    skip_next = False
    for index, argument in enumerate(argv):
        if skip_next:
            skip_next = False
            continue
        if argument == "--topology":
            if index + 1 >= len(argv):
                raise ValueError("--topology requires a value")
            skip_next = True
            continue
        if argument.startswith("--topology="):
            continue
        result.append(argument)
    result.append(f"--topology={topology}")
    return result


def run_all_topologies(
    script_path: str,
    *,
    argv: Sequence[str],
    topology_names: Sequence[str],
) -> int:
    """Run a lifecycle experiment once per topology in deterministic order."""

    for topology in topology_names:
        command = [
            sys.executable,
            str(Path(script_path).resolve()),
            *replace_topology(argv, topology),
        ]
        print(f"Starting topology suite member: {topology}", flush=True)
        process = subprocess.run(command, check=False)
        if process.returncode:
            return process.returncode
    return 0
