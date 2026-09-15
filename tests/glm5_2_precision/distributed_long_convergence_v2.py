# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Batch offline convergence reassessment for the 5000-step topology matrix."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Sequence

from .ddp_long_v2 import (
    DdpLongV2Config,
    compare_distributed_long_convergence_v2,
    write_distributed_long_convergence_v2_report,
)


DEFAULT_TOPOLOGIES = (
    "ddp8",
    "ep8",
    "fsdp2-tp4",
    "fsdp2-tp4-ep8",
    "fsdp4-tp2",
    "fsdp8",
    "pp8",
    "tp8",
)
SCENARIO_PREFIX = "migration-cuda-npu-"
SCENARIO_SUFFIX = "-bf16-random-s5000-b64-seq128-seed61"


@dataclass(frozen=True)
class TopologyResult:
    topology: str
    scenario: str
    status: str
    area_relative_error: float | None
    final_mean_relative_error: float | None
    smoothed_correlation: float | None
    maximum_sustained_window_relative_error: float | None
    raw_loss_mare: float | None
    report: str | None
    error: str | None


def _scenario(topology: str) -> str:
    return f"{SCENARIO_PREFIX}{topology}{SCENARIO_SUFFIX}"


def _format_percent(value: float | None) -> str:
    return "-" if value is None else f"{value:.4%}"


def _format_correlation(value: float | None) -> str:
    return "-" if value is None else f"{value:.6f}"


def _matrix_status(results: Sequence[TopologyResult]) -> str:
    statuses = {result.status for result in results}
    if "INVALID" in statuses or "INCONCLUSIVE" in statuses:
        return "INCONCLUSIVE"
    if "FAIL" in statuses:
        return "FAIL"
    return "PASS"


def _write_matrix_report(
    results: Sequence[TopologyResult],
    config: DdpLongV2Config,
    output_root: Path,
) -> tuple[str, Path, Path]:
    status = _matrix_status(results)
    summary_path = output_root / "distributed_long_convergence_v2_summary.json"
    report_path = output_root / "distributed_long_convergence_v2_report.md"
    summary_path.write_text(
        json.dumps(
            {
                "status": status,
                "config": asdict(config),
                "results": [asdict(result) for result in results],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Distributed long-run convergence alignment V2",
        "",
        f"Matrix result: **{status}**",
        "",
        "| Topology | Result | AUC diff | Final 20% diff | Smoothed corr | "
        "Worst 500-step diff | Raw loss MARE |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        lines.append(
            f"| {result.topology} | {result.status} | "
            f"{_format_percent(result.area_relative_error)} | "
            f"{_format_percent(result.final_mean_relative_error)} | "
            f"{_format_correlation(result.smoothed_correlation)} | "
            f"{_format_percent(result.maximum_sustained_window_relative_error)} | "
            f"{_format_percent(result.raw_loss_mare)} |"
        )

    errors = [result for result in results if result.error]
    if errors:
        lines.extend(["", "## Invalid or inconclusive inputs", ""])
        for result in errors:
            safe_error = (result.error or "").replace("\n", " ")
            lines.append(f"- `{result.topology}`: {safe_error}")

    lines.extend(
        [
            "",
            "Individual reports",
            "",
        ]
    )
    for result in results:
        if result.report:
            lines.append(f"- `{result.topology}`: `{result.report}`")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return status, summary_path, report_path


def compare_topology_matrix(
    artifact_root: str | Path,
    output_root: str | Path,
    *,
    topologies: Sequence[str] = DEFAULT_TOPOLOGIES,
    config: DdpLongV2Config = DdpLongV2Config(),
) -> tuple[str, tuple[TopologyResult, ...], Path, Path]:
    """Reassess existing topology artifacts and write per-topology reports."""

    artifact_root = Path(artifact_root).resolve()
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    results: list[TopologyResult] = []

    for topology in topologies:
        scenario = _scenario(topology)
        scenario_root = artifact_root / scenario
        topology_output = output_root / topology
        try:
            result = compare_distributed_long_convergence_v2(
                (
                    scenario_root / "reference-r1",
                    scenario_root / "reference-r2",
                ),
                (
                    scenario_root / "candidate-r1",
                    scenario_root / "candidate-r2",
                ),
                config=config,
            )
            _, report_path = write_distributed_long_convergence_v2_report(
                result, topology_output
            )
            results.append(
                TopologyResult(
                    topology=topology,
                    scenario=scenario,
                    status=result.status,
                    area_relative_error=result.curve.area_relative_error,
                    final_mean_relative_error=(
                        result.curve.final_mean_relative_error
                    ),
                    smoothed_correlation=result.curve.smoothed_correlation,
                    maximum_sustained_window_relative_error=(
                        result.curve.maximum_sustained_window_relative_error
                    ),
                    raw_loss_mare=result.loss.mean_absolute_relative_error,
                    report=str(report_path),
                    error=(
                        "; ".join(result.inconclusive_reasons)
                        if result.inconclusive_reasons
                        else None
                    ),
                )
            )
        except (OSError, ValueError, RuntimeError) as error:
            results.append(
                TopologyResult(
                    topology=topology,
                    scenario=scenario,
                    status="INVALID",
                    area_relative_error=None,
                    final_mean_relative_error=None,
                    smoothed_correlation=None,
                    maximum_sustained_window_relative_error=None,
                    raw_loss_mare=None,
                    report=None,
                    error=str(error),
                )
            )

    result_tuple = tuple(results)
    status, summary_path, report_path = _write_matrix_report(
        result_tuple, config, output_root
    )
    return status, result_tuple, summary_path, report_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline loss-curve reassessment for all distributed topologies."
    )
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--topology",
        action="append",
        choices=DEFAULT_TOPOLOGIES,
        help="Reassess only the selected topology; repeat for multiple topologies.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    status, _, summary_path, report_path = compare_topology_matrix(
        args.artifact_root,
        args.output_root,
        topologies=tuple(args.topology or DEFAULT_TOPOLOGIES),
    )
    print(f"Distributed long-run convergence V2: {status}")
    print(f"JSON: {summary_path}")
    print(f"Report: {report_path}")
    return {"PASS": 0, "FAIL": 1, "INCONCLUSIVE": 2}[status]


if __name__ == "__main__":
    raise SystemExit(main())
