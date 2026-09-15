# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Create a compact summary from per-topology long-run V2 reports."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Sequence

from .distributed_long_convergence_v2 import DEFAULT_TOPOLOGIES


PER_TOPOLOGY_SUMMARY = "ddp_long_v2_summary.json"


@dataclass(frozen=True)
class CompactResult:
    topology: str
    status: str
    steps: int | None
    step_range: tuple[int, int] | None
    auc_difference: float | None
    final_20_percent_difference: float | None
    smoothed_correlation: float | None
    worst_500_step_difference: float | None
    error: str | None = None


def _number(mapping: dict[str, Any], key: str) -> float:
    value = mapping.get(key)
    if not isinstance(value, (int, float)):
        raise ValueError(f"missing numeric field: {key}")
    return float(value)


def _read_result(report_root: Path, topology: str) -> CompactResult:
    path = report_root / topology / PER_TOPOLOGY_SUMMARY
    if not path.is_file():
        return CompactResult(
            topology=topology,
            status="MISSING",
            steps=None,
            step_range=None,
            auc_difference=None,
            final_20_percent_difference=None,
            smoothed_correlation=None,
            worst_500_step_difference=None,
            error=f"not found: {path}",
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        status = payload.get("status")
        if status not in {"PASS", "FAIL", "INCONCLUSIVE"}:
            raise ValueError(f"unsupported status: {status!r}")
        step_range = payload.get("step_range")
        if not (
            isinstance(step_range, list)
            and len(step_range) == 2
            and all(isinstance(value, int) for value in step_range)
        ):
            raise ValueError("step_range must contain two integers")
        loss = payload.get("loss")
        curve = payload.get("curve")
        if not isinstance(loss, dict) or not isinstance(curve, dict):
            raise ValueError("loss and curve sections are required")
        count = loss.get("count")
        if not isinstance(count, int):
            raise ValueError("missing integer field: loss.count")
        return CompactResult(
            topology=topology,
            status=status,
            steps=count,
            step_range=(step_range[0], step_range[1]),
            auc_difference=_number(curve, "area_relative_error"),
            final_20_percent_difference=_number(
                curve, "final_mean_relative_error"
            ),
            smoothed_correlation=_number(curve, "smoothed_correlation"),
            worst_500_step_difference=_number(
                curve, "maximum_sustained_window_relative_error"
            ),
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        return CompactResult(
            topology=topology,
            status="INVALID",
            steps=None,
            step_range=None,
            auc_difference=None,
            final_20_percent_difference=None,
            smoothed_correlation=None,
            worst_500_step_difference=None,
            error=f"{path}: {error}",
        )


def _overall_status(results: Sequence[CompactResult]) -> str:
    statuses = {result.status for result in results}
    if statuses & {"MISSING", "INVALID"}:
        return "INCOMPLETE"
    if "FAIL" in statuses:
        return "FAIL"
    if "INCONCLUSIVE" in statuses:
        return "INCONCLUSIVE"
    return "PASS"


def _percent(value: float | None) -> str:
    return "-" if value is None else f"{value:.3%}"


def _correlation(value: float | None) -> str:
    return "-" if value is None else f"{value:.6f}"


def _step_text(result: CompactResult) -> str:
    if result.steps is None or result.step_range is None:
        return "-"
    start, end = result.step_range
    return f"{result.steps} ({start}..{end})"


def _table_rows(results: Sequence[CompactResult]) -> list[list[str]]:
    return [
        [
            result.topology,
            result.status,
            _step_text(result),
            _percent(result.auc_difference),
            _percent(result.final_20_percent_difference),
            _correlation(result.smoothed_correlation),
            _percent(result.worst_500_step_difference),
        ]
        for result in results
    ]


def _console_table(results: Sequence[CompactResult]) -> str:
    headers = [
        "Topology",
        "Result",
        "Steps",
        "AUC diff",
        "Final 20%",
        "Smooth corr",
        "Worst 500",
    ]
    rows = _table_rows(results)
    widths = [
        max(len(header), *(len(row[index]) for row in rows))
        for index, header in enumerate(headers)
    ]

    def render(row: Sequence[str]) -> str:
        return "  ".join(value.ljust(width) for value, width in zip(row, widths))

    return "\n".join(
        [
            render(headers),
            render(["-" * width for width in widths]),
            *(render(row) for row in rows),
        ]
    )


def _markdown(
    results: Sequence[CompactResult],
    overall_status: str,
) -> str:
    passed = sum(result.status == "PASS" for result in results)
    lines = [
        "# Distributed long-run key results",
        "",
        f"Overall: **{overall_status}** ({passed}/{len(results)} PASS)",
        "",
        "| Topology | Result | Steps | AUC diff | Final 20% diff | "
        "Smoothed corr | Worst 500-step diff |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in _table_rows(results):
        lines.append("| " + " | ".join(row) + " |")
    errors = [result for result in results if result.error]
    if errors:
        lines.extend(["", "## Missing or invalid reports", ""])
        lines.extend(
            f"- `{result.topology}`: {result.error}" for result in errors
        )
    return "\n".join(lines) + "\n"


def summarize_distributed_long_results(
    report_root: str | Path,
    output_directory: str | Path | None = None,
    *,
    topologies: Sequence[str] = DEFAULT_TOPOLOGIES,
) -> tuple[str, tuple[CompactResult, ...], Path, Path, str]:
    """Read per-topology JSON files and write compact JSON/Markdown summaries."""

    if not topologies:
        raise ValueError("at least one topology is required")
    report_root = Path(report_root).resolve()
    output_directory = (
        report_root
        if output_directory is None
        else Path(output_directory).resolve()
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    results = tuple(_read_result(report_root, topology) for topology in topologies)
    overall_status = _overall_status(results)
    passed = sum(result.status == "PASS" for result in results)

    json_path = output_directory / "distributed_long_key_results.json"
    json_path.write_text(
        json.dumps(
            {
                "overall_status": overall_status,
                "passed": passed,
                "total": len(results),
                "results": [asdict(result) for result in results],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    markdown_path = output_directory / "distributed_long_key_results.md"
    markdown_path.write_text(
        _markdown(results, overall_status),
        encoding="utf-8",
    )
    return (
        overall_status,
        results,
        json_path,
        markdown_path,
        _console_table(results),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize key results from per-topology long-run V2 reports."
    )
    parser.add_argument("--report-root", required=True, type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory; defaults to --report-root.",
    )
    parser.add_argument(
        "--topology",
        action="append",
        choices=DEFAULT_TOPOLOGIES,
        help="Include only this topology; repeat for multiple topologies.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    status, _, json_path, markdown_path, table = (
        summarize_distributed_long_results(
            args.report_root,
            args.output_dir,
            topologies=tuple(args.topology or DEFAULT_TOPOLOGIES),
        )
    )
    print(f"Overall: {status}")
    print(table)
    print(f"JSON: {json_path}")
    print(f"Markdown: {markdown_path}")
    return {"PASS": 0, "FAIL": 1, "INCONCLUSIVE": 2, "INCOMPLETE": 2}[status]


if __name__ == "__main__":
    raise SystemExit(main())
