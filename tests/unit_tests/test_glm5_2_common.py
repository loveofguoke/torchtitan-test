# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import json
import os
from pathlib import Path
import signal
import subprocess
from unittest import mock

import pytest

from tests.glm5_2_common.cli import (
    display_repository_path,
    LoggedProcessError,
    RunAttempt,
    replace_topology,
    reset_output_generation,
    run_managed_process,
    write_experiment_overview,
)
from tests.glm5_2_common.topology import standard_topologies


def test_context_parallel_diagnostic_topologies_scale_world_size() -> None:
    topologies = standard_topologies()

    for degree in (2, 4, 8):
        topology = topologies[f"cp{degree}"]
        assert topology.world_size == degree
        assert topology.context_parallel_degree == degree


def test_logged_process_error_keeps_external_runtime_log_absolute(
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "run" / "runtime.log"

    error = LoggedProcessError(7, ["trainer", "--run"], log_path=log_path)

    assert str(error).endswith(f"runtime log: {log_path.resolve()}")


def test_logged_process_error_starts_repository_log_at_checkout_name() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    log_path = repository_root / "smoke_runs" / "single" / "runtime.log"

    error = LoggedProcessError(7, ["trainer", "--run"], log_path=log_path)

    assert str(error).endswith(
        "runtime log: torchtitan-test/smoke_runs/single/runtime.log"
    )


def test_repository_output_starts_at_checkout_name() -> None:
    repository_root = Path(__file__).resolve().parents[2]

    assert display_repository_path(repository_root / "reports" / "result.json") == (
        "torchtitan-test/reports/result.json"
    )


def test_repository_output_recognizes_alternate_mount_root(tmp_path: Path) -> None:
    alternate = (
        tmp_path
        / "home"
        / "owner"
        / "torchtitan-test"
        / "mindstudio_runs"
        / "accuracy"
        / "runtime.log"
    )

    assert display_repository_path(alternate) == (
        "torchtitan-test/mindstudio_runs/accuracy/runtime.log"
    )


def test_experiment_overview_is_human_and_machine_readable(tmp_path: Path) -> None:
    write_experiment_overview(
        tmp_path,
        title="Example capture",
        summary={"device": "npu", "topology": "tp8", "steps": 30},
        entry_command=["python", "benchmark.py", "--topology", "tp8"],
    )

    payload = json.loads((tmp_path / "experiment.json").read_text("utf-8"))
    readme = (tmp_path / "README.md").read_text("utf-8")
    assert payload["device"] == "npu"
    assert payload["entry_command"][-1] == "tp8"
    assert "Example capture" in readme
    assert "--topology tp8" in readme


def test_all_topology_runner_replaces_both_cli_forms() -> None:
    assert replace_topology(
        ["--device", "cuda", "--topology", "all"], "fsdp8"
    ) == ["--device", "cuda", "--topology=fsdp8"]
    assert replace_topology(
        ["--topology=all", "--force"], "cp8"
    ) == ["--force", "--topology=cp8"]


def test_managed_process_terminates_tree_when_parent_is_interrupted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.glm5_2_common import cli

    class InterruptedProcess:
        pid = 12345

        def wait(self, timeout: float | None = None) -> int:
            raise KeyboardInterrupt

    process = InterruptedProcess()
    terminate = mock.Mock()
    monkeypatch.setattr(cli, "_terminate_process_tree", terminate)
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: process)

    with pytest.raises(KeyboardInterrupt):
        run_managed_process(["trainer"])

    terminate.assert_called_once_with(process, process_group=True)


def test_managed_process_does_not_detach_inside_outer_managed_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.glm5_2_common import cli

    class InterruptedProcess:
        pid = 12345

        def wait(self, timeout: float | None = None) -> int:
            raise KeyboardInterrupt

    process = InterruptedProcess()
    terminate = mock.Mock()
    popen = mock.Mock(return_value=process)
    monkeypatch.setenv(cli.MANAGED_PROCESS_GROUP_ENV, "1")
    monkeypatch.setattr(cli, "_terminate_process_tree", terminate)
    monkeypatch.setattr(subprocess, "Popen", popen)

    with pytest.raises(KeyboardInterrupt):
        run_managed_process(["trainer"])

    assert "start_new_session" not in popen.call_args.kwargs
    terminate.assert_called_once_with(process, process_group=False)


@pytest.mark.skipif(os.name != "posix", reason="POSIX process-group signals")
def test_terminate_process_tree_escalates_and_reaps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.glm5_2_common import cli

    class HungProcess:
        pid = 12345

        def poll(self) -> None:
            return None

        def wait(self, timeout: float | None = None) -> int:
            if timeout is not None:
                raise subprocess.TimeoutExpired(["trainer"], timeout)
            return 0

    killpg = mock.Mock()
    monkeypatch.setattr(os, "killpg", killpg)

    cli._terminate_process_tree(HungProcess(), grace_seconds=0)

    assert killpg.call_args_list == [
        mock.call(12345, signal.SIGTERM),
        mock.call(12345, signal.SIGKILL),
    ]


def test_reset_output_generation_removes_selected_members_and_archives(
    tmp_path: Path,
) -> None:
    completed = tmp_path / "completed"
    pending = tmp_path / "pending"
    completed.mkdir()
    pending.mkdir()
    (completed / "manifest.json").write_text("{}", encoding="utf-8")
    archived = tmp_path / ".completed.previous-20260827-120000"
    failed = tmp_path / "pending.failed-20260827-120001"
    archived.mkdir()
    failed.mkdir()

    reset_output_generation((completed, pending))

    assert not completed.exists()
    assert not pending.exists()
    assert not archived.exists()
    assert not failed.exists()


def test_run_attempt_records_lifecycle_and_guards_force_reset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.glm5_2_common import cli

    run = tmp_path / "run"
    attempt = RunAttempt.start(
        run,
        kind="unit-test",
        context={"topology": "single"},
    )
    state = json.loads((run / "run_state.json").read_text(encoding="utf-8"))
    assert state["status"] == "running"
    assert state["attempt_id"] == attempt.attempt_id
    state["pid"] = 12345
    (run / "run_state.json").write_text(
        json.dumps(state) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(cli, "process_is_running", lambda pid: True)

    with pytest.raises(RuntimeError, match="run is still active"):
        reset_output_generation(
            (run,),
            active_run_directories=(run,),
        )

    attempt.update("completed", return_code=0)
    reset_output_generation(
        (run,),
        active_run_directories=(run,),
    )
    assert not run.exists()


def test_run_attempt_lock_is_exclusive_even_inside_one_process(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    first = RunAttempt.start(run, kind="first")

    with pytest.raises(RuntimeError, match="run is still active"):
        RunAttempt.start(run, kind="second")

    first.update("completed")
    second = RunAttempt.start(run, kind="second")
    second.update("completed")


def test_run_owner_identity_rejects_reused_pid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.glm5_2_common import cli

    monkeypatch.setattr(cli, "process_is_running", lambda pid: True)
    monkeypatch.setattr(cli, "_linux_boot_id", lambda: "same-boot")
    monkeypatch.setattr(
        cli,
        "_linux_process_start_ticks",
        lambda pid: "new-process",
    )

    assert not cli._owner_is_running(
        {
            "pid": 12345,
            "boot_id": "same-boot",
            "process_start_ticks": "old-process",
        }
    )


def test_force_reset_recovers_a_dead_owner_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.glm5_2_common import cli

    run = tmp_path / "run"
    run.mkdir()
    (run / "run_state.json.lock").write_text(
        json.dumps({"attempt_id": "dead", "pid": 12345}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(cli, "process_is_running", lambda pid: False)

    with pytest.raises(RuntimeError, match="stale run lock"):
        RunAttempt.start(run, kind="replacement")

    reset_output_generation(
        (run,),
        active_run_directories=(run,),
    )
    attempt = RunAttempt.start(run, kind="replacement")
    attempt.update("completed")

    assert not (run / "run_state.json.lock").exists()
