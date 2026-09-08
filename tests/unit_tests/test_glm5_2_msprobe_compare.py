from __future__ import annotations

from dataclasses import asdict
import csv
import json
from pathlib import Path

from tests.glm5_2_precision import msprobe_compare
from tests.glm5_2_precision.msprobe_compare import compare_msprobe_captures
from tests.glm5_2_precision.msprobe_tensorboard import MsprobeCaptureConfig


def _capture(
    directory: Path,
    *,
    role: str,
    config: MsprobeCaptureConfig,
    ranks: dict[int, tuple[str, ...]],
) -> Path:
    directory.mkdir(parents=True)
    (directory / "msprobe_capture.json").write_text(
        json.dumps(
            {
                "role": role,
                "repeat": 1,
                "fixture_scenario_name": "fixture",
                "msprobe": asdict(config),
            }
        ),
        encoding="utf-8",
    )
    for rank, names in ranks.items():
        rank_directory = directory / "msprobe_dump" / "step0" / f"rank{rank}"
        rank_directory.mkdir(parents=True)
        (rank_directory / "debug.json").write_text(
            json.dumps({"data": {name: {} for name in names}}),
            encoding="utf-8",
        )
    return directory


def _fake_native_compare(
    *,
    executable: str,
    reference: Path,
    candidate: Path,
    output: Path,
) -> tuple[list[str], Path]:
    del executable
    output.mkdir(parents=True)
    reference_names = set(json.loads(reference.read_text())["data"])
    candidate_names = set(json.loads(candidate.read_text())["data"])
    csv_path = output / "debug_compare_result_test.csv"
    fieldnames = [
        "NPU Name",
        "Bench Name",
        "NPU Dtype",
        "Bench Dtype",
        "NPU Tensor Shape",
        "Bench Tensor Shape",
        "Requires_grad Consistent",
        "Cosine",
        "MaxAbsErr",
        "Result",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for name in sorted(candidate_names & reference_names):
            writer.writerow(
                {
                    "NPU Name": name,
                    "Bench Name": name,
                    "NPU Dtype": "torch.bfloat16",
                    "Bench Dtype": "torch.bfloat16",
                    "NPU Tensor Shape": "[2, 4]" if "_tp_sum" not in name else "[1]",
                    "Bench Tensor Shape": "[2, 4]" if "_tp_sum" not in name else "[2]",
                    "Requires_grad Consistent": "True",
                    "Cosine": "0.8" if "bad" in name else "1.0",
                    "MaxAbsErr": "0.0",
                    "Result": "error" if "_tp_sum" in name else "pass",
                }
            )
    return ["msprobe", "compare"], csv_path


def test_native_msprobe_gate_accepts_complete_partitioned_coverage(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = MsprobeCaptureConfig(
        task="tensor",
        level="debug",
        block_boundaries=True,
        block_global_step=True,
    )
    reference = _capture(
        tmp_path / "reference",
        role="reference",
        config=config,
        ranks={0: ("block_0.debug", "block_1.debug", "router_tp_sum.debug")},
    )
    candidate = _capture(
        tmp_path / "candidate",
        role="candidate",
        config=config,
        ranks={
            0: ("block_0.debug", "router_tp_sum.debug"),
            4: ("block_1.debug",),
        },
    )
    monkeypatch.setattr(
        msprobe_compare, "_run_native_compare", _fake_native_compare
    )

    summary_path = compare_msprobe_captures(
        reference_run=reference,
        candidate_run=candidate,
        output_directory=tmp_path / "report",
        repeat=1,
        capture_config=config,
        exclude_patterns=("_tp_sum",),
    )
    summary = json.loads(summary_path.read_text())

    assert summary["passed"]
    assert summary["steps"][0]["expected_tensor_count"] == 2
    assert summary["steps"][0]["compared_tensor_count"] == 2
    assert summary["steps"][0]["native_fail_count"] == 0
    assert summary["steps"][0]["excluded_row_count"] == 1
    assert (summary_path.parent / "msprobe_summary.md").is_file()


def test_native_msprobe_gate_rejects_missing_tensor(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = MsprobeCaptureConfig(task="tensor", level="debug")
    reference = _capture(
        tmp_path / "reference",
        role="reference",
        config=config,
        ranks={0: ("block_0.debug", "block_1.debug")},
    )
    candidate = _capture(
        tmp_path / "candidate",
        role="candidate",
        config=config,
        ranks={0: ("block_0.debug",)},
    )
    monkeypatch.setattr(
        msprobe_compare, "_run_native_compare", _fake_native_compare
    )

    summary_path = compare_msprobe_captures(
        reference_run=reference,
        candidate_run=candidate,
        output_directory=tmp_path / "report",
        repeat=1,
        capture_config=config,
    )
    summary = json.loads(summary_path.read_text())

    assert not summary["passed"]
    assert summary["steps"][0]["missing_tensors"] == ["block_1.debug"]


def test_documented_msprobe_threshold_backstops_stale_native_result(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = MsprobeCaptureConfig(task="tensor", level="debug")
    reference = _capture(
        tmp_path / "reference",
        role="reference",
        config=config,
        ranks={0: ("bad_tensor.debug",)},
    )
    candidate = _capture(
        tmp_path / "candidate",
        role="candidate",
        config=config,
        ranks={0: ("bad_tensor.debug",)},
    )
    monkeypatch.setattr(
        msprobe_compare, "_run_native_compare", _fake_native_compare
    )

    summary_path = compare_msprobe_captures(
        reference_run=reference,
        candidate_run=candidate,
        output_directory=tmp_path / "report",
        repeat=1,
        capture_config=config,
    )
    summary = json.loads(summary_path.read_text())

    assert not summary["passed"]
    assert summary["steps"][0]["native_fail_count"] == 0
    assert summary["steps"][0]["documented_threshold_failures"] == [
        "bad_tensor.debug"
    ]
