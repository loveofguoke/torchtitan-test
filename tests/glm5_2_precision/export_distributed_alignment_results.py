# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Export the key GPU/NPU distributed-alignment results from saved artifacts.

This command is intentionally offline: it validates and reads existing formal
artifacts, prints a compact Markdown report, and writes the same results as JSON
and Markdown.  It never launches training or changes an artifact.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import sys
from typing import Mapping, Sequence

from .artifacts import PrecisionArtifactReader
from .ddp_long_v2 import (
    CurveDiagnostics,
    DdpLongV2Config,
    _curve_diagnostics,
    _mean_curve,
    _repeat_curve_diagnostics,
)
from .standards import AnyOfErrorLimit, MigrationStandard


SHORT_TOPOLOGIES = (
    "ddp8",
    "ep8",
    "fsdp2-pp4",
    "fsdp2-tp2-pp2",
    "fsdp2-tp4-ep8",
    "fsdp2-tp4",
    "fsdp4-tp2",
    "fsdp8",
    "tp8",
)
LONG_TOPOLOGIES = (
    "ddp8",
    "ep8",
    "fsdp2-tp4",
    "fsdp2-tp4-ep8",
    "fsdp4-tp2",
    "fsdp8",
    "pp8",
    "tp8",
)
SHORT_SCENARIO_SUFFIX = "-bf16-random-s10-b16-seq128-seed61"
LONG_SCENARIO_SUFFIX = "-bf16-random-s5000-b64-seq128-seed61"


@dataclass(frozen=True)
class ShortResult:
    topology: str
    status: str
    gpu_first_loss: float | None
    npu_first_loss: float | None
    relative_error: float | None
    error: str | None = None


@dataclass(frozen=True)
class LongResult:
    topology: str
    status: str
    steps: int | None
    area_relative_error: float | None
    final_mean_relative_error: float | None
    smoothed_correlation: float | None
    maximum_sustained_window_relative_error: float | None
    error: str | None = None


def _scenario(topology: str, suffix: str) -> str:
    return f"migration-cuda-npu-{topology}{suffix}"


def _relative_error(reference: float, candidate: float) -> float:
    if reference == 0.0:
        if candidate == 0.0:
            return 0.0
        return math.inf
    return abs(reference - candidate) / abs(reference)


def _limit_passes(
    limit: AnyOfErrorLimit,
    *,
    absolute_error: float,
    relative_error: float,
) -> bool:
    checks = (
        limit.absolute is not None and absolute_error <= limit.absolute,
        limit.relative_absolute is not None
        and relative_error <= limit.relative_absolute,
        limit.max_absolute is not None and absolute_error <= limit.max_absolute,
    )
    return any(checks)


def _load_artifacts(
    scenario_root: Path,
    topology: str,
) -> tuple[
    tuple[PrecisionArtifactReader, PrecisionArtifactReader],
    tuple[PrecisionArtifactReader, PrecisionArtifactReader],
]:
    specifications = (
        ("reference-r1", "reference", "cuda", 1),
        ("reference-r2", "reference", "cuda", 2),
        ("candidate-r1", "candidate", "npu", 1),
        ("candidate-r2", "candidate", "npu", 2),
    )
    readers: list[PrecisionArtifactReader] = []
    for directory, expected_role, expected_device, expected_repeat in specifications:
        reader = PrecisionArtifactReader(scenario_root / directory)
        role = reader.metadata.get("role")
        device = reader.metadata.get("device_type")
        repeat = reader.metadata.get("repeat")
        if role != expected_role or device != expected_device:
            raise ValueError(
                f"{reader.path}: expected role/device "
                f"{expected_role}/{expected_device}, observed {role}/{device}"
            )
        if int(repeat) != expected_repeat:
            raise ValueError(
                f"{reader.path}: expected repeat {expected_repeat}, observed {repeat}"
            )
        readers.append(reader)

    expected_contract = readers[0].training_contract
    for reader in readers[1:]:
        if reader.training_contract != expected_contract:
            raise ValueError(
                f"{scenario_root}: GPU/NPU training contracts do not match"
            )
    contract_topology = expected_contract.get("topology")
    if not isinstance(contract_topology, Mapping):
        raise ValueError(f"{scenario_root}: topology contract is missing")
    if contract_topology.get("name") != topology:
        raise ValueError(
            f"{scenario_root}: expected topology {topology}, observed "
            f"{contract_topology.get('name')}"
        )
    if int(contract_topology.get("world_size", 1)) <= 1:
        raise ValueError(f"{scenario_root}: topology is not distributed")

    return (readers[0], readers[1]), (readers[2], readers[3])


def _aligned_series(
    readers: Sequence[PrecisionArtifactReader],
    name: str,
) -> tuple[tuple[int, ...], tuple[tuple[float, ...], ...]]:
    mappings: list[dict[int, float]] = []
    for reader in readers:
        if name == "loss":
            mappings.append(reader.loss_series())
        elif name == "grad_norm":
            mappings.append(reader.grad_norm_series())
        else:
            raise ValueError(f"unsupported metric series: {name}")
    steps = tuple(sorted(mappings[0]))
    if any(tuple(sorted(mapping)) != steps for mapping in mappings[1:]):
        raise ValueError(f"{name} steps do not match across GPU/NPU repeats")
    curves = tuple(
        tuple(mapping[step] for step in steps) for mapping in mappings
    )
    return steps, curves


def collect_short_results(
    artifact_root: str | Path,
    *,
    topologies: Sequence[str] = SHORT_TOPOLOGIES,
    standard: MigrationStandard = MigrationStandard(),
) -> tuple[ShortResult, ...]:
    root = Path(artifact_root)
    results: list[ShortResult] = []
    for topology in topologies:
        scenario_root = root / _scenario(topology, SHORT_SCENARIO_SUFFIX)
        try:
            gpu, npu = _load_artifacts(scenario_root, topology)
            gpu_losses = tuple(reader.loss_series()[1] for reader in gpu)
            npu_losses = tuple(reader.loss_series()[1] for reader in npu)
            gpu_first_loss = sum(gpu_losses) / len(gpu_losses)
            npu_first_loss = sum(npu_losses) / len(npu_losses)
            absolute_error = abs(gpu_first_loss - npu_first_loss)
            relative_error = _relative_error(gpu_first_loss, npu_first_loss)
            status = (
                "PASS"
                if _limit_passes(
                    standard.first_loss,
                    absolute_error=absolute_error,
                    relative_error=relative_error,
                )
                else "FAIL"
            )
            results.append(
                ShortResult(
                    topology=topology,
                    status=status,
                    gpu_first_loss=gpu_first_loss,
                    npu_first_loss=npu_first_loss,
                    relative_error=relative_error,
                )
            )
        except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
            results.append(
                ShortResult(
                    topology=topology,
                    status="INVALID",
                    gpu_first_loss=None,
                    npu_first_loss=None,
                    relative_error=None,
                    error=str(error),
                )
            )
    return tuple(results)


def _long_status(
    *,
    steps: Sequence[int],
    curves: Sequence[Sequence[float]],
    curve: CurveDiagnostics,
    config: DdpLongV2Config,
) -> tuple[str, str | None]:
    if len(steps) < config.minimum_steps:
        return (
            "INCONCLUSIVE",
            f"only {len(steps)} steps; {config.minimum_steps} are required",
        )
    if not all(math.isfinite(value) for series in curves for value in series):
        return "FAIL", "loss or global grad norm contains NaN/Inf"

    gpu_repeat = _repeat_curve_diagnostics(curves[0:2], steps, config)
    npu_repeat = _repeat_curve_diagnostics(curves[2:4], steps, config)
    repeat_reasons: list[str] = []
    for platform, repeat in (("GPU", gpu_repeat), ("NPU", npu_repeat)):
        if repeat.maximum_area_relative_error > config.curve_area_relative_limit:
            repeat_reasons.append(f"{platform} repeat AUC is unstable")
        if (
            repeat.maximum_final_mean_relative_error
            > config.final_mean_relative_limit
        ):
            repeat_reasons.append(f"{platform} repeat final 20% is unstable")
        if repeat.minimum_smoothed_correlation < config.smoothed_correlation_minimum:
            repeat_reasons.append(f"{platform} repeat correlation is unstable")
        if (
            repeat.maximum_sustained_window_relative_error
            > config.sustained_window_relative_limit
        ):
            repeat_reasons.append(f"{platform} repeat 500-step window is unstable")
    if repeat_reasons:
        return "INCONCLUSIVE", "; ".join(repeat_reasons)

    passed = (
        curve.area_relative_error <= config.curve_area_relative_limit
        and curve.final_mean_relative_error <= config.final_mean_relative_limit
        and curve.smoothed_correlation >= config.smoothed_correlation_minimum
        and curve.maximum_sustained_window_relative_error
        <= config.sustained_window_relative_limit
    )
    return ("PASS" if passed else "FAIL"), None


def collect_long_results(
    artifact_root: str | Path,
    *,
    topologies: Sequence[str] = LONG_TOPOLOGIES,
    config: DdpLongV2Config = DdpLongV2Config(),
) -> tuple[LongResult, ...]:
    root = Path(artifact_root)
    results: list[LongResult] = []
    for topology in topologies:
        scenario_root = root / _scenario(topology, LONG_SCENARIO_SUFFIX)
        try:
            gpu, npu = _load_artifacts(scenario_root, topology)
            readers = (*gpu, *npu)
            steps, loss_curves = _aligned_series(readers, "loss")
            grad_steps, grad_curves = _aligned_series(readers, "grad_norm")
            if grad_steps != steps:
                raise ValueError("loss and global grad norm steps do not match")
            gpu_loss = _mean_curve(loss_curves[0:2])
            npu_loss = _mean_curve(loss_curves[2:4])
            curve = _curve_diagnostics(gpu_loss, npu_loss, steps, config)
            status, error = _long_status(
                steps=steps,
                curves=(*loss_curves, *grad_curves),
                curve=curve,
                config=config,
            )
            results.append(
                LongResult(
                    topology=topology,
                    status=status,
                    steps=len(steps),
                    area_relative_error=curve.area_relative_error,
                    final_mean_relative_error=curve.final_mean_relative_error,
                    smoothed_correlation=curve.smoothed_correlation,
                    maximum_sustained_window_relative_error=(
                        curve.maximum_sustained_window_relative_error
                    ),
                    error=error,
                )
            )
        except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
            results.append(
                LongResult(
                    topology=topology,
                    status="INVALID",
                    steps=None,
                    area_relative_error=None,
                    final_mean_relative_error=None,
                    smoothed_correlation=None,
                    maximum_sustained_window_relative_error=None,
                    error=str(error),
                )
            )
    return tuple(results)


def _format_number(value: float | None) -> str:
    return "-" if value is None else f"{value:.15g}"


def _format_percent(value: float | None) -> str:
    return "-" if value is None else f"{value:.6%}"


def _format_correlation(value: float | None) -> str:
    return "-" if value is None else f"{value:.8f}"


def _group_status(results: Sequence[ShortResult | LongResult]) -> str:
    statuses = {result.status for result in results}
    if "INVALID" in statuses:
        return "INCOMPLETE"
    if "INCONCLUSIVE" in statuses:
        return "INCONCLUSIVE"
    if "FAIL" in statuses:
        return "FAIL"
    return "PASS"


def render_markdown(
    short_results: Sequence[ShortResult],
    long_results: Sequence[LongResult],
) -> str:
    short_status = _group_status(short_results)
    long_status = _group_status(long_results)
    lines = [
        "# Distributed GPU/NPU alignment key results",
        "",
        "## Standards",
        "",
        "- First step: absolute loss difference <= 0.005 or relative difference "
        "<= 0.5%.",
        "- 5000 steps: AUC difference <= 2%, final-20% mean loss difference <= 2%,",
        "  smoothed correlation >= 0.99, and every 500-step window difference <= 3%.",
        "",
        f"## First-step loss ({short_status})",
        "",
        "| Topology | Result | GPU loss | NPU loss | Relative error |",
        "|---|---:|---:|---:|---:|",
    ]
    for result in short_results:
        lines.append(
            f"| {result.topology} | {result.status} | "
            f"{_format_number(result.gpu_first_loss)} | "
            f"{_format_number(result.npu_first_loss)} | "
            f"{_format_percent(result.relative_error)} |"
        )

    lines.extend(
        [
            "",
            f"## 5000-step loss curves ({long_status})",
            "",
            "| Topology | Result | Steps | AUC diff | Final 20% diff | "
            "Smoothed corr | Worst 500-step diff |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for result in long_results:
        lines.append(
            f"| {result.topology} | {result.status} | {result.steps or '-'} | "
            f"{_format_percent(result.area_relative_error)} | "
            f"{_format_percent(result.final_mean_relative_error)} | "
            f"{_format_correlation(result.smoothed_correlation)} | "
            f"{_format_percent(result.maximum_sustained_window_relative_error)} |"
        )

    errors = [
        (result.topology, result.error)
        for result in (*short_results, *long_results)
        if result.error
    ]
    if errors:
        lines.extend(["", "## Invalid or inconclusive inputs", ""])
        lines.extend(f"- `{topology}`: {error}" for topology, error in errors)
    return "\n".join(lines) + "\n"


def export_results(
    short_artifact_root: str | Path,
    long_artifact_root: str | Path,
    output_directory: str | Path,
    *,
    short_topologies: Sequence[str] = SHORT_TOPOLOGIES,
    long_topologies: Sequence[str] = LONG_TOPOLOGIES,
    long_config: DdpLongV2Config = DdpLongV2Config(),
) -> tuple[str, Path, Path]:
    short_results = collect_short_results(
        short_artifact_root,
        topologies=short_topologies,
    )
    long_results = collect_long_results(
        long_artifact_root,
        topologies=long_topologies,
        config=long_config,
    )
    markdown = render_markdown(short_results, long_results)
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    markdown_path = output / "distributed_alignment_key_results.md"
    json_path = output / "distributed_alignment_key_results.json"
    markdown_path.write_text(markdown, encoding="utf-8")
    json_path.write_text(
        json.dumps(
            {
                "short_status": _group_status(short_results),
                "long_status": _group_status(long_results),
                "standards": {
                    "first_step": asdict(MigrationStandard().first_loss),
                    "long_curve": asdict(long_config),
                },
                "short_results": [asdict(result) for result in short_results],
                "long_results": [asdict(result) for result in long_results],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    overall = _group_status((*short_results, *long_results))
    return overall, json_path, markdown_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export key distributed GPU/NPU precision-alignment results."
    )
    parser.add_argument(
        "--short-artifact-root",
        type=Path,
        default=Path("precision_short_matrix_artifacts"),
    )
    parser.add_argument(
        "--long-artifact-root",
        type=Path,
        default=Path("precision_artifacts"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("precision_reports/distributed-alignment-key-results"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    overall, json_path, markdown_path = export_results(
        args.short_artifact_root,
        args.long_artifact_root,
        args.output_dir,
    )
    print(markdown_path.read_text(encoding="utf-8"), end="")
    print(f"JSON: {json_path}")
    print(f"Markdown: {markdown_path}")
    return {"PASS": 0, "FAIL": 1, "INCONCLUSIVE": 2, "INCOMPLETE": 2}[overall]


if __name__ == "__main__":
    sys.exit(main())
