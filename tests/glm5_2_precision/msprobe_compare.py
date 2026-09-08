# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Native msProbe tensor comparison with an explicit coverage gate."""

from __future__ import annotations

import csv
from dataclasses import asdict
import importlib.metadata
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any, Sequence

from .msprobe_tensorboard import MsprobeCaptureConfig, validate_debug_dump_directory


SCHEMA = "torchtitan.glm5_2.msprobe_native_compare"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected a JSON object: {path}")
    return value


def _debug_names(path: Path) -> set[str]:
    payload = _read_json(path)
    data = payload.get("data")
    if not isinstance(data, dict) or not data:
        raise RuntimeError(f"msProbe debug.json contains no tensor data: {path}")
    return set(data)


def _matches_any(name: str, patterns: Sequence[re.Pattern[str]]) -> bool:
    return any(pattern.search(name) for pattern in patterns)


def _native_result(row: dict[str, str]) -> str:
    for key in ("Result", "Accuracy Reached or Not"):
        if key in row:
            return row[key].strip().lower()
    raise RuntimeError("native msProbe result has no Result column")


def _row_name(row: dict[str, str]) -> str:
    for key in ("NPU Name", "Target Name", "Candidate Name"):
        value = row.get(key, "").strip()
        if value:
            return value
    raise RuntimeError("native msProbe result has no candidate tensor-name column")


def _read_native_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise RuntimeError(f"native msProbe CSV has no header: {path}")
        rows = [dict(row) for row in reader]
    if not rows:
        raise RuntimeError(f"native msProbe CSV has no comparison rows: {path}")
    # Fail early instead of accepting a CSV schema that this gate cannot interpret.
    _native_result(rows[0])
    _row_name(rows[0])
    return rows


def _package_version() -> str:
    try:
        return importlib.metadata.version("mindstudio-probe")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _structural_mismatch(row: dict[str, str]) -> bool:
    pairs = (
        ("NPU Dtype", "Bench Dtype"),
        ("NPU Tensor Shape", "Bench Tensor Shape"),
    )
    for left, right in pairs:
        if left in row and right in row and row[left].strip() != row[right].strip():
            return True
    consistency = row.get("Requires_grad Consistent")
    return consistency is not None and consistency.strip().lower() not in {
        "true",
        "yes",
        "pass",
    }


def _documented_threshold_failure(row: dict[str, str]) -> bool | None:
    """Apply msProbe's documented cosine/max-absolute compatibility rule.

    ``None`` means the native report did not provide usable numeric indicators.
    The caller treats that as unsupported instead of silently accepting it.
    """

    try:
        cosine = float(row["Cosine"])
        max_absolute = float(row["MaxAbsErr"])
    except (KeyError, TypeError, ValueError):
        return None
    if not math.isfinite(cosine) or not math.isfinite(max_absolute):
        return None
    return (
        (cosine < 0.99 and max_absolute > 0.001)
        or cosine < 0.9
        or max_absolute > 1.0
    )


def _validate_capture_manifest(
    run_directory: Path,
    *,
    role: str,
    repeat: int,
    capture_config: MsprobeCaptureConfig,
) -> dict[str, Any]:
    manifest_path = run_directory / "msprobe_capture.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"msProbe capture manifest not found; run --capture-msprobe {role}: "
            f"{manifest_path}"
        )
    manifest = _read_json(manifest_path)
    expected_capture = json.loads(json.dumps(asdict(capture_config)))
    if manifest.get("role") != role:
        raise RuntimeError(f"unexpected capture role in {manifest_path}")
    if manifest.get("repeat") != repeat:
        raise RuntimeError(f"unexpected capture repeat in {manifest_path}")
    if manifest.get("msprobe") != expected_capture:
        raise RuntimeError(
            "msProbe capture profile differs from --compare-msprobe options: "
            f"{manifest_path}"
        )
    validate_debug_dump_directory(run_directory / "msprobe_dump")
    return manifest


def _run_native_compare(
    *,
    executable: str,
    reference: Path,
    candidate: Path,
    output: Path,
) -> tuple[list[str], Path]:
    output.mkdir(parents=True)
    command = [
        executable,
        "compare",
        "-m",
        "auto",
        "-gp",
        str(reference),
        "-tp",
        str(candidate),
        "-o",
        str(output),
    ]
    completed = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    (output / "msprobe_compare.log").write_text(
        completed.stdout or "", encoding="utf-8"
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"native msProbe compare failed with exit code {completed.returncode}; "
            f"see {output / 'msprobe_compare.log'}"
        )
    matches = sorted(output.rglob("debug_compare_result_*.csv"))
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one native msProbe CSV under {output}, got {matches}"
        )
    return command, matches[0]


def _markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Native msProbe tensor comparison",
        "",
        f"Overall: **{'PASS' if summary['passed'] else 'FAIL'}**",
        "",
        f"mindstudio-probe: `{summary['mindstudio_probe_version']}`",
        "",
        "| Step | Result | Expected | Compared | Native fail | Threshold fail | "
        "Unsupported | Excluded | Missing | Unexpected | Duplicated |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
        "---: | ---: |",
    ]
    for step in summary["steps"]:
        lines.append(
            "| {step} | {result} | {expected} | {compared} | {native_fail} | "
            "{threshold_fail} | {unsupported} | {excluded} | {missing} | "
            "{unexpected} | {duplicated} |".format(
                step=step["step"],
                result="PASS" if step["passed"] else "FAIL",
                expected=step["expected_tensor_count"],
                compared=step["compared_tensor_count"],
                native_fail=step["native_fail_count"],
                threshold_fail=len(step["documented_threshold_failures"]),
                unsupported=len(step["unsupported_indicators"]),
                excluded=step["excluded_row_count"],
                missing=len(step["missing_tensors"]),
                unexpected=len(step["unexpected_tensors"]),
                duplicated=len(step["duplicated_tensors"]),
            )
        )
    if summary["exclude_patterns"]:
        lines.extend(
            [
                "",
                "Excluded semantic patterns: "
                + ", ".join(f"`{value}`" for value in summary["exclude_patterns"]),
            ]
        )
    failures = []
    for step in summary["steps"]:
        failures.extend(step["first_native_failures"])
        failures.extend(step["documented_threshold_failures"][:20])
    if failures:
        lines.extend(["", "First native failures:"])
        lines.extend(f"- `{name}`" for name in failures[:20])
    return "\n".join(lines) + "\n"


def compare_msprobe_captures(
    *,
    reference_run: str | Path,
    candidate_run: str | Path,
    output_directory: str | Path,
    repeat: int,
    capture_config: MsprobeCaptureConfig,
    exclude_patterns: Sequence[str] = (),
    executable: str = "msprobe",
    force: bool = False,
    resume: bool = False,
) -> Path:
    """Run native comparisons and require complete topology-invariant coverage."""

    if capture_config.task != "tensor" or capture_config.level != "debug":
        raise ValueError("native msProbe comparison requires task=tensor, level=debug")
    if force and resume:
        raise ValueError("force and resume are mutually exclusive")
    reference_run = Path(reference_run)
    candidate_run = Path(candidate_run)
    output_directory = Path(output_directory)
    summary_path = output_directory / "msprobe_summary.json"
    normalized_capture_config = json.loads(json.dumps(asdict(capture_config)))
    if output_directory.exists():
        if resume and summary_path.is_file():
            summary = _read_json(summary_path)
            if (
                summary.get("schema") == SCHEMA
                and summary.get("reference_run") == str(reference_run)
                and summary.get("candidate_run") == str(candidate_run)
                and summary.get("capture_config") == normalized_capture_config
                and summary.get("exclude_patterns") == list(exclude_patterns)
            ):
                return summary_path
        if not force and not resume:
            raise FileExistsError(
                "msProbe comparison output exists; pass --resume to reuse it or "
                f"--force to replace it: {output_directory}"
            )
        shutil.rmtree(output_directory)
    output_directory.mkdir(parents=True)

    reference_manifest = _validate_capture_manifest(
        reference_run,
        role="reference",
        repeat=repeat,
        capture_config=capture_config,
    )
    candidate_manifest = _validate_capture_manifest(
        candidate_run,
        role="candidate",
        repeat=repeat,
        capture_config=capture_config,
    )
    if reference_manifest.get("fixture_scenario_name") != candidate_manifest.get(
        "fixture_scenario_name"
    ):
        raise RuntimeError("reference and candidate msProbe captures use different fixtures")

    compiled_patterns = tuple(re.compile(value) for value in exclude_patterns)
    step_summaries: list[dict[str, Any]] = []
    all_commands: list[list[str]] = []
    for step in capture_config.steps:
        reference_paths = sorted(
            (reference_run / "msprobe_dump" / f"step{step}").glob(
                "rank*/debug.json"
            )
        )
        if len(reference_paths) != 1:
            raise RuntimeError(
                f"single-card reference step{step} must contain one debug.json, "
                f"got {reference_paths}"
            )
        reference_path = reference_paths[0]
        reference_names = _debug_names(reference_path)
        candidate_paths = sorted(
            (candidate_run / "msprobe_dump" / f"step{step}").glob(
                "rank*/debug.json"
            )
        )
        if not candidate_paths:
            raise RuntimeError(f"candidate step{step} contains no debug.json")

        candidate_names_by_path = {
            path: _debug_names(path) for path in candidate_paths
        }
        candidate_names = set().union(*candidate_names_by_path.values())
        excluded_names = {
            name
            for name in reference_names | candidate_names
            if _matches_any(name, compiled_patterns)
        }
        expected_names = reference_names - excluded_names
        valid_candidate_names = candidate_names - excluded_names
        missing_tensors = sorted(expected_names - valid_candidate_names)
        unexpected_tensors = sorted(valid_candidate_names - expected_names)

        observed_names: list[str] = []
        native_failures: list[str] = []
        structural_failures: list[str] = []
        threshold_failures: list[str] = []
        unsupported_indicators: list[str] = []
        excluded_row_count = 0
        raw_row_count = 0
        native_pass_count = 0
        reports: list[dict[str, Any]] = []
        for candidate_path, names in candidate_names_by_path.items():
            if not ((names & reference_names) - excluded_names):
                continue
            rank_name = candidate_path.parent.name
            invocation_output = output_directory / f"step{step}" / rank_name
            command, csv_path = _run_native_compare(
                executable=executable,
                reference=reference_path,
                candidate=candidate_path,
                output=invocation_output,
            )
            all_commands.append(command)
            rows = _read_native_rows(csv_path)
            raw_row_count += len(rows)
            for row in rows:
                name = _row_name(row)
                if _matches_any(name, compiled_patterns):
                    excluded_row_count += 1
                    continue
                observed_names.append(name)
                result = _native_result(row)
                if result in {"pass", "yes", "true"}:
                    native_pass_count += 1
                else:
                    native_failures.append(name)
                if _structural_mismatch(row):
                    structural_failures.append(name)
                threshold_failure = _documented_threshold_failure(row)
                if threshold_failure is None:
                    unsupported_indicators.append(name)
                elif threshold_failure:
                    threshold_failures.append(name)
            reports.append(
                {
                    "candidate_rank": rank_name,
                    "candidate_debug_json": str(candidate_path),
                    "csv": str(csv_path),
                    "rows": len(rows),
                }
            )

        observed_name_set = set(observed_names)
        missing_native_rows = sorted(
            (expected_names & valid_candidate_names) - observed_name_set
        )
        unexpected_native_rows = sorted(
            observed_name_set - (expected_names & valid_candidate_names)
        )
        counts = {name: observed_names.count(name) for name in observed_name_set}
        duplicated_tensors = sorted(name for name, count in counts.items() if count > 1)
        native_fail_count = len(native_failures)
        passed = not any(
            (
                missing_tensors,
                unexpected_tensors,
                missing_native_rows,
                unexpected_native_rows,
                duplicated_tensors,
                native_failures,
                structural_failures,
                threshold_failures,
                unsupported_indicators,
            )
        ) and len(observed_name_set) == len(expected_names)
        step_summaries.append(
            {
                "step": step,
                "passed": passed,
                "expected_tensor_count": len(expected_names),
                "candidate_tensor_count": len(valid_candidate_names),
                "compared_tensor_count": len(observed_name_set),
                "raw_row_count": raw_row_count,
                "native_pass_count": native_pass_count,
                "native_fail_count": native_fail_count,
                "excluded_tensor_count": len(excluded_names),
                "excluded_row_count": excluded_row_count,
                "missing_tensors": missing_tensors,
                "unexpected_tensors": unexpected_tensors,
                "missing_native_rows": missing_native_rows,
                "unexpected_native_rows": unexpected_native_rows,
                "duplicated_tensors": duplicated_tensors,
                "structural_failures": sorted(set(structural_failures)),
                "documented_threshold_failures": sorted(set(threshold_failures)),
                "unsupported_indicators": sorted(set(unsupported_indicators)),
                "first_native_failures": native_failures[:20],
                "reports": reports,
            }
        )

    summary = {
        "schema": SCHEMA,
        "schema_version": 1,
        "passed": all(step["passed"] for step in step_summaries),
        "mindstudio_probe_version": _package_version(),
        "reference_run": str(reference_run),
        "candidate_run": str(candidate_run),
        "capture_config": asdict(capture_config),
        "exclude_patterns": list(exclude_patterns),
        "commands": all_commands,
        "steps": step_summaries,
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_directory / "msprobe_summary.md").write_text(
        _markdown(summary), encoding="utf-8"
    )
    return summary_path


__all__ = ["SCHEMA", "compare_msprobe_captures"]
