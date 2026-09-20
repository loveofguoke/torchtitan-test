# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import sys

import pytest

from tests.glm5_2_combination.capture_metrics import (
    _install_comm_timeout_override,
)


def test_comm_timeout_override_is_not_added_without_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TORCHTITAN_COMM_INIT_TIMEOUT_SECONDS", raising=False)
    monkeypatch.setattr(sys, "argv", ["capture_metrics.py"])

    _install_comm_timeout_override()

    assert sys.argv == ["capture_metrics.py"]


def test_comm_timeout_override_is_forwarded_as_torchtitan_argument(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TORCHTITAN_COMM_INIT_TIMEOUT_SECONDS", "2400")
    monkeypatch.setattr(sys, "argv", ["capture_metrics.py", "--job.dump_folder=/tmp/run"])

    _install_comm_timeout_override()

    assert sys.argv == [
        "capture_metrics.py",
        "--job.dump_folder=/tmp/run",
        "--comm.init_timeout_seconds=2400",
    ]


@pytest.mark.parametrize(
    "explicit_arguments",
    [
        ["--comm.init_timeout_seconds=777"],
        ["--comm.init_timeout_seconds", "777"],
    ],
)
def test_explicit_comm_timeout_takes_precedence(
    monkeypatch: pytest.MonkeyPatch,
    explicit_arguments: list[str],
) -> None:
    monkeypatch.setenv("TORCHTITAN_COMM_INIT_TIMEOUT_SECONDS", "2400")
    expected = ["capture_metrics.py", *explicit_arguments]
    monkeypatch.setattr(sys, "argv", expected.copy())

    _install_comm_timeout_override()

    assert sys.argv == expected


def test_similarly_prefixed_argument_does_not_hide_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TORCHTITAN_COMM_INIT_TIMEOUT_SECONDS", "2400")
    monkeypatch.setattr(
        sys,
        "argv",
        ["capture_metrics.py", "--comm.init_timeout_seconds_backup=777"],
    )

    _install_comm_timeout_override()

    assert sys.argv[-1] == "--comm.init_timeout_seconds=2400"


@pytest.mark.parametrize("invalid_timeout", ["not-an-integer", "0", "-1"])
def test_invalid_comm_timeout_override_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    invalid_timeout: str,
) -> None:
    monkeypatch.setenv("TORCHTITAN_COMM_INIT_TIMEOUT_SECONDS", invalid_timeout)
    monkeypatch.setattr(sys, "argv", ["capture_metrics.py"])

    with pytest.raises(
        ValueError,
        match="TORCHTITAN_COMM_INIT_TIMEOUT_SECONDS must be a positive integer",
    ):
        _install_comm_timeout_override()
