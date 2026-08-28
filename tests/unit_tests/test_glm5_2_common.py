# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import json
from pathlib import Path

import pytest

from tests.glm5_2_common.cli import (
    replace_topology,
    reset_output_generation,
    RunAttempt,
)


def test_all_topology_runner_replaces_both_cli_forms() -> None:
    assert replace_topology(
        ["--device", "cuda", "--topology", "all"], "fsdp8"
    ) == ["--device", "cuda", "--topology=fsdp8"]
    assert replace_topology(
        ["--topology=all", "--force"], "cp8"
    ) == ["--force", "--topology=cp8"]


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
