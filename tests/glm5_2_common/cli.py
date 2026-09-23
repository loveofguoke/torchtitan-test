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
import signal
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence
import uuid


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PROCESS_TERMINATION_GRACE_SECONDS = 10.0
MANAGED_PROCESS_GROUP_ENV = "_TORCHTITAN_TEST_MANAGED_PROCESS_GROUP"


def display_repository_path(path: Path) -> str:
    """Render repository-owned paths from the checkout directory name."""

    resolved = path.resolve()
    try:
        relative = resolved.relative_to(REPOSITORY_ROOT)
    except ValueError:
        # Containers can expose one checkout through multiple bind-mount roots
        # (for example, /workspace/... and /home/...). Path.resolve() cannot
        # identify that aliasing, so recover the repository boundary by name.
        matching_indices = [
            index
            for index, part in enumerate(resolved.parts)
            if part.casefold() == REPOSITORY_ROOT.name.casefold()
        ]
        if not matching_indices:
            return str(resolved)
        relative = Path(*resolved.parts[matching_indices[-1] + 1 :])
    return (Path(REPOSITORY_ROOT.name) / relative).as_posix()


class LoggedProcessError(subprocess.CalledProcessError):
    """A failed subprocess whose final exception line identifies its log."""

    def __init__(
        self,
        returncode: int,
        cmd: Sequence[str],
        *,
        log_path: Path,
    ) -> None:
        super().__init__(returncode, list(cmd))
        self.log_path = log_path.resolve()

    def __str__(self) -> str:
        return (
            f"{super().__str__().rstrip('.')}; runtime log: "
            f"{display_repository_path(self.log_path)}"
        )


def print_runtime_log(log_path: Path) -> None:
    """Print a clickable repository-relative log location."""
    print(f"Runtime log: {display_repository_path(log_path)}", flush=True)


def print_output_path(label: str, path: Path) -> None:
    """Print a clickable repository-relative experiment output path."""

    print(f"{label}: {display_repository_path(path)}", flush=True)


def _terminate_process_tree(
    process: subprocess.Popen[Any],
    *,
    grace_seconds: float = PROCESS_TERMINATION_GRACE_SECONDS,
    process_group: bool = True,
) -> None:
    """Terminate and reap a managed subprocess and all of its descendants."""

    if process.poll() is not None:
        process.wait()
        return

    if os.name == "posix" and process_group:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            process.wait()
            return
    else:
        process.terminate()

    try:
        process.wait(timeout=grace_seconds)
        return
    except subprocess.TimeoutExpired:
        pass

    if os.name == "posix" and process_group:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    else:
        process.kill()
    process.wait()


def run_managed_process(
    command: Sequence[str],
    *,
    cwd: Path | str | None = None,
    env: dict[str, str] | None = None,
    stdout: Any = None,
    stderr: Any = None,
    text: bool | None = None,
    check: bool = False,
) -> subprocess.CompletedProcess[Any]:
    """Run a subprocess whose complete process tree follows parent lifetime."""

    popen_arguments: dict[str, Any] = {
        "cwd": cwd,
        "env": env,
        "stdout": stdout,
        "stderr": stderr,
        "text": text,
    }
    owns_process_group = os.environ.get(MANAGED_PROCESS_GROUP_ENV) != "1"
    if os.name == "posix" and owns_process_group:
        # A separate session lets the parent terminate torchrun and every rank
        # without signalling itself.
        popen_arguments["start_new_session"] = True
    elif os.name == "nt" and owns_process_group:
        popen_arguments["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP

    process = subprocess.Popen(list(command), **popen_arguments)
    previous_handlers: dict[int, Any] = {}

    def interrupt(signum: int, _frame: Any) -> None:
        raise SystemExit(128 + signum)

    # SIGINT already becomes KeyboardInterrupt. Convert termination and shell
    # hangup into exceptions so cleanup and RunAttempt finalization still run.
    if os.name == "posix":
        try:
            for signum in (signal.SIGTERM, signal.SIGHUP):
                previous_handlers[signum] = signal.getsignal(signum)
                signal.signal(signum, interrupt)
        except ValueError:
            # Signal handlers can only be installed by the main thread. The
            # process-group cleanup still covers exceptions in helper threads.
            previous_handlers.clear()

    try:
        returncode = process.wait()
    except BaseException:
        _terminate_process_tree(process, process_group=owns_process_group)
        raise
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)

    completed = subprocess.CompletedProcess(list(command), returncode)
    if check and returncode:
        raise subprocess.CalledProcessError(returncode, list(command))
    return completed


def write_experiment_overview(
    directory: Path,
    *,
    title: str,
    summary: dict[str, Any],
    entry_command: Sequence[str] | None = None,
) -> None:
    """Write the human and machine entry points for an experiment directory.

    Directory-name hashes are identity guards, not a user interface.  Every
    run directory should therefore explain itself without requiring readers to
    decode its name or inspect the expanded torchrun command.
    """

    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "torchtitan.glm5_2.experiment_overview",
        "schema_version": 1,
        "title": title,
        **summary,
    }
    if entry_command is not None:
        payload["entry_command"] = [str(part) for part in entry_command]
    (directory / "experiment.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    lines = [f"# {title}", ""]
    for key, value in summary.items():
        rendered = json.dumps(value, ensure_ascii=False, sort_keys=True)
        lines.append(f"- `{key}`: `{rendered}`")
    if entry_command is not None:
        lines.extend(
            (
                "",
                "## Entry command",
                "",
                "```bash",
                " ".join(str(part) for part in entry_command),
                "```",
            )
        )
    lines.extend(
        (
            "",
            "`experiment.json` is the complete machine-readable overview. ",
            "Expanded worker launch details remain in the workflow-specific ",
            "command and runtime-log files.",
            "",
        )
    )
    (directory / "README.md").write_text("\n".join(lines), encoding="utf-8")


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


def _linux_process_start_ticks(pid: int) -> str | None:
    """Return Linux /proc start ticks so reused PIDs are not mistaken as owners."""

    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        fields = stat.rsplit(")", 1)[1].split()
        return fields[19]
    except (IndexError, ValueError):
        return None


def _linux_boot_id() -> str | None:
    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text(
            encoding="utf-8"
        ).strip()
    except OSError:
        return None


def _owner_is_running(payload: dict[str, Any]) -> bool:
    try:
        pid = int(payload.get("pid", -1))
    except (TypeError, ValueError):
        return False
    if not process_is_running(pid):
        return False

    expected_boot_id = payload.get("boot_id")
    actual_boot_id = _linux_boot_id()
    if (
        expected_boot_id is not None
        and actual_boot_id is not None
        and expected_boot_id != actual_boot_id
    ):
        return False

    expected_start_ticks = payload.get("process_start_ticks")
    actual_start_ticks = _linux_process_start_ticks(pid)
    if (
        expected_start_ticks is not None
        and actual_start_ticks is not None
        and str(expected_start_ticks) != actual_start_ticks
    ):
        return False
    return True


def active_run_pid(
    path: Path,
    *,
    state_name: str | None = None,
) -> int | None:
    """Return the live owner of a run, validating its process identity."""

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
            except (OSError, TypeError, ValueError) as error:
                raise RuntimeError(
                    f"run lock is unreadable and cannot be replaced safely: "
                    f"{lock_path}"
                ) from error
            if _owner_is_running(lock):
                return int(lock["pid"])
        if not state_path.is_file():
            continue
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError):
            continue
        if state.get("status") == "running" and _owner_is_running(state):
            return int(state["pid"])
    return None


def assert_run_not_active(
    path: Path,
    *,
    state_name: str | None = None,
) -> None:
    """Refuse to replace a run whose recorded orchestrator is still alive."""

    pid = active_run_pid(path, state_name=state_name)
    if pid is not None:
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
                "boot_id": _linux_boot_id(),
                "process_start_ticks": _linux_process_start_ticks(self.pid),
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
            "boot_id": _linux_boot_id(),
            "process_start_ticks": _linux_process_start_ticks(self.pid),
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
            print(f"  {display_repository_path(path)}", flush=True)
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
        child_environment = {
            **os.environ,
            MANAGED_PROCESS_GROUP_ENV: "1",
        }
        process = run_managed_process(
            command,
            env=child_environment,
            check=False,
        )
        if process.returncode:
            return process.returncode
    return 0
