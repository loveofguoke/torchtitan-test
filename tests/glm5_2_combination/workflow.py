# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Central orchestration for orthogonal precision, graph, and profiler features."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import hashlib
import html
import json
from pathlib import Path
from typing import Sequence

from tests.glm5_2_common.execution import compose_execution
from tests.glm5_2_common.topology import ParallelTopology, select_topologies
from tests.glm5_2_performance.analysis import build_analysis, render_html_report
from tests.glm5_2_performance.config import profiler_presets
from tests.glm5_2_precision.artifacts import PrecisionArtifactReader
from tests.glm5_2_precision.topology_suite import (
    compare_topology_suite,
    topology_config,
)
from tests.glm5_2_precision.workflow import (
    FormalExperimentConfig,
    _artifact_directory,
    _endpoint_from_process_environment,
    _fixture_endpoint_from_environment,
    _root,
    _run_directory,
    capture_endpoint,
    prepare_fixture,
)

from .config import GraphFeatureConfig, ProfilerFeatureConfig


@dataclass(frozen=True)
class CombinationSelection:
    objectives: frozenset[str]
    reference_graph: GraphFeatureConfig
    candidate_graph: GraphFeatureConfig
    profiler: ProfilerFeatureConfig | None

    def __post_init__(self) -> None:
        unknown = self.objectives - {"precision", "performance"}
        if unknown or not self.objectives:
            raise ValueError(f"invalid combination objectives: {sorted(unknown)}")
        if "performance" in self.objectives and self.profiler is None:
            raise ValueError("performance objective requires a profiler feature")


def _combination_storage_base(
    config: FormalExperimentConfig,
    selection: CombinationSelection,
) -> str:
    precision = {
        "mixed-bfloat16": "bf16",
        "mixed-float32": "fp32",
        "full-bf16": "full-bf16",
    }.get(config.training.precision_name, config.training.precision_name)
    objective = "-".join(sorted(selection.objectives))
    profiler = (
        f"prof-{selection.profiler.preset.name}" if selection.profiler else "no-prof"
    )
    experiment = (
        f"migration-{config.reference.device_type}-{config.candidate.device_type}"
        if config.kind == "migration"
        else f"self-{config.reference.device_type}"
    )
    checkpoint = (
        "random" if config.training.checkpoint_kind == "random_seed" else "converged"
    )
    return (
        f"{experiment}-{precision}-{checkpoint}-s{config.training.steps}-"
        f"b{config.training.global_batch_size}-seq{config.training.sequence_length}-"
        f"seed{config.training.seed}-{selection.reference_graph.mode}-"
        f"{selection.candidate_graph.mode}-{objective}-{profiler}"
    )


def _legacy_combination_storage_name(
    config: FormalExperimentConfig,
    selection: CombinationSelection,
) -> str:
    precision = {
        "mixed-bfloat16": "bf16",
        "mixed-float32": "fp32",
        "full-bf16": "full-bf16",
    }.get(config.training.precision_name, config.training.precision_name)
    objective = "-".join(sorted(selection.objectives))
    profiler = selection.profiler.preset.name if selection.profiler else "no-prof"
    identity = {
        "objectives": sorted(selection.objectives),
        "reference_graph": asdict(selection.reference_graph),
        "candidate_graph": asdict(selection.candidate_graph),
        "profiler": asdict(selection.profiler) if selection.profiler else None,
        "reference_extra_args": config.reference.extra_args,
        "reference_environment": config.reference.environment,
        "candidate_extra_args": config.candidate.extra_args,
        "candidate_environment": config.candidate.environment,
        "training": asdict(config.training),
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode("utf-8")
    ).hexdigest()[:8]
    return (
        f"combo-{config.reference.device_type}-{config.candidate.device_type}-"
        f"{precision}-s{config.training.steps}-b{config.training.global_batch_size}-"
        f"q{config.training.sequence_length}-seed{config.training.seed}-"
        f"{selection.reference_graph.mode}-{selection.candidate_graph.mode}-"
        f"{objective}-{profiler}-{digest}"
    )


def _apply_selection(
    config: FormalExperimentConfig,
    selection: CombinationSelection,
) -> FormalExperimentConfig:
    endpoints = {}
    for role, endpoint, graph in (
        ("reference", config.reference, selection.reference_graph),
        ("candidate", config.candidate, selection.candidate_graph),
    ):
        features = [graph.feature(device_type=endpoint.device_type)]
        if selection.profiler is not None:
            features.append(selection.profiler.feature())
        plan = compose_execution(endpoint.topology, features)
        endpoints[role] = replace(
            endpoint,
            name=f"{endpoint.name}-{graph.mode}",
            extra_args=endpoint.extra_args + plan.command_args(),
            environment={**endpoint.environment, **plan.environment()},
            entry_module="tests.glm5_2_combination.capture_metrics",
        )
    storage_base = _combination_storage_base(config, selection)
    legacy_storage_name = _legacy_combination_storage_name(config, selection)
    selected = replace(
        config,
        reference=endpoints["reference"],
        candidate=endpoints["candidate"],
        allow_endpoint_argument_difference=True,
        artifact_root="combination_artifacts",
        run_root="combination_runs",
        storage_name_override=storage_base,
        topology_subdirectory=True,
        legacy_storage_names=(legacy_storage_name,),
    )
    return replace(
        selected,
        fixture_name=selected.storage_name,
        report_root="combination_reports/precision",
    )


def _metric_median(analysis: dict, fragment: str) -> float | None:
    for name, values in analysis["metrics"]["summary"].items():
        if fragment in name.lower():
            return values.get("median")
    return None


def _relative_link(path: Path, root: Path) -> str:
    return html.escape(path.relative_to(root).as_posix())


def _performance_report(
    root: Path,
    config: FormalExperimentConfig,
    *,
    role: str,
    repeat: int,
    selection: CombinationSelection,
) -> tuple[Path, float | None]:
    endpoint = config.reference if role == "reference" else config.candidate
    artifact = PrecisionArtifactReader(
        _artifact_directory(root, config, role, endpoint, repeat)
    )
    if artifact.training_contract.get("endpoint_args", []) != list(
        endpoint.extra_args
    ):
        raise ValueError(
            f"{role} artifact execution features do not match this report command"
        )
    run_directory = _run_directory(root, config, role, endpoint, repeat)
    metrics_path = run_directory / "raw_metrics.jsonl"
    if not metrics_path.is_file():
        raise FileNotFoundError(metrics_path)
    analysis = build_analysis(run_directory, metrics_path=metrics_path)
    profiler = selection.profiler
    assert profiler is not None
    run_name = f"{role}-{endpoint.device_type}-{endpoint.topology.slug}-r{repeat}"
    manifest = {
        "run_name": run_name,
        "device": endpoint.device_type,
        "topology": endpoint.topology.name,
        "preset": profiler.preset.name,
        "config": {
            "steps": config.training.steps,
            "local_batch_size": config.training.local_batch_size,
            "global_batch_size": config.training.global_batch_size,
            "sequence_length": config.training.sequence_length,
            "skip_steps": profiler.skip_steps,
            "warmup_steps": profiler.warmup_steps,
            "active_steps": profiler.active_steps,
            "extra_args": list(endpoint.extra_args),
            "environment": endpoint.environment,
        },
        "profiler_environment": profiler.preset.environment(),
        "preflight": {},
    }
    output = (
        root
        / "combination_reports"
        / config.storage_name
        / "performance"
        / endpoint.topology.slug
        / f"{run_name}-performance.html"
    )
    render_html_report(manifest=manifest, analysis=analysis, output_path=output)
    return output, _metric_median(analysis, "end_to_end")


def _write_combination_report(
    root: Path,
    base: FormalExperimentConfig,
    *,
    topologies: dict[str, ParallelTopology],
    selected: Sequence[str],
    precision: str | None,
    selection: CombinationSelection,
    require_all: bool,
) -> Path:
    first_raw = topology_config(base, topologies[selected[0]], precision=precision)
    first_config = _apply_selection(first_raw, selection)
    report_parent = root / "combination_reports"
    report_root = report_parent / first_config.storage_name
    if not report_root.exists():
        for legacy_name in first_config.legacy_storage_names:
            legacy_root = report_parent / legacy_name
            if legacy_root.is_dir():
                report_parent.mkdir(parents=True, exist_ok=True)
                legacy_root.rename(report_root)
                print(
                    "Adopted matching legacy output:\n"
                    f"  {legacy_root}\n  -> {report_root}"
                )
                break
    precision_report: Path | None = None
    if "precision" in selection.objectives:
        precision_report = compare_topology_suite(
            root,
            replace(
                base,
                artifact_root="combination_artifacts",
                run_root="combination_runs",
                report_root="combination_reports/precision",
                allow_endpoint_argument_difference=True,
            ),
            topologies=topologies,
            selected=selected,
            precision=precision,
            require_all=require_all,
            config_transform=lambda config: _apply_selection(config, selection),
        )

    rows = []
    if "performance" in selection.objectives:
        for name in selected:
            raw = topology_config(base, topologies[name], precision=precision)
            config = _apply_selection(raw, selection)
            for repeat in range(1, config.reference.repeats + 1):
                reference_report, reference_median = _performance_report(
                    root, config, role="reference", repeat=repeat, selection=selection
                )
                candidate_report, candidate_median = _performance_report(
                    root, config, role="candidate", repeat=repeat, selection=selection
                )
                speedup = (
                    reference_median / candidate_median
                    if reference_median is not None
                    and candidate_median not in (None, 0)
                    else None
                )
                rows.append(
                    "<tr>"
                    f"<td>{html.escape(name)}</td><td>{repeat}</td>"
                    "<td>"
                    f"{reference_median if reference_median is not None else '-'}"
                    "</td><td>"
                    f"{candidate_median if candidate_median is not None else '-'}"
                    "</td>"
                    f"<td>{f'{speedup:.4f}x' if speedup is not None else '-'}</td>"
                    f'<td><a href="{_relative_link(reference_report, report_root)}">'
                    "reference</a> | "
                    f'<a href="{_relative_link(candidate_report, report_root)}">'
                    "candidate</a></td></tr>"
                )

    report_root.mkdir(parents=True, exist_ok=True)
    precision_link = (
        f'<a href="{_relative_link(precision_report, report_root)}">'
        "Open precision topology suite</a>"
        if precision_report is not None
        else "Precision objective disabled"
    )
    report = report_root / f"{first_config.storage_name}.html"
    report.write_text(
        f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>GLM-5.2 combined experiment</title>
<style>body{{font:14px/1.5 system-ui,sans-serif;max-width:1400px;margin:auto;padding:24px;color:#172033}}
section{{border:1px solid #dfe5ef;border-radius:12px;padding:20px;margin:16px 0}}table{{border-collapse:collapse;width:100%}}
th,td{{padding:9px;border-bottom:1px solid #dfe5ef;text-align:left}}th{{position:sticky;top:0;background:#f8fafc}}a{{color:#155eef}}</style>
</head><body><h1>GLM-5.2 combined experiment</h1>
<p>Objectives: {html.escape(', '.join(sorted(selection.objectives)))}; reference graph: {selection.reference_graph.mode}; candidate graph: {selection.candidate_graph.mode}.</p>
<section><h2>Precision</h2><p>{precision_link}</p></section>
<section><h2>Performance</h2><table><thead><tr><th>Topology</th><th>Repeat</th><th>Reference median step time</th><th>Candidate median step time</th><th>Candidate speedup</th><th>Details</th></tr></thead>
<tbody>{''.join(rows) or '<tr><td colspan="6">Performance objective disabled</td></tr>'}</tbody></table></section>
</body></html>""",
        encoding="utf-8",
    )
    return report


def run_combination_cli(
    base: FormalExperimentConfig,
    script_path: str,
    *,
    topologies: dict[str, ParallelTopology],
    topology_names: Sequence[str],
) -> None:
    parser = argparse.ArgumentParser(description="Composable GLM-5.2 experiment")
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--data", action="store_true")
    actions.add_argument("--capture", choices=("reference", "candidate"))
    actions.add_argument("--compare", action="store_true")
    actions.add_argument("--list-topologies", action="store_true")
    parser.add_argument("--topology", choices=("all", *topology_names), default="all")
    parser.add_argument("--topologies", help="comma-separated topology subset or all")
    parser.add_argument("--repeat", type=int)
    parser.add_argument("--precision", choices=("fp32", "bf16", "full-bf16"))
    parser.add_argument("--objectives", default="precision,performance")
    graph_choices = ("eager", "inductor", "npugraphs")
    parser.add_argument(
        "--reference-graph", choices=graph_choices, default="eager"
    )
    parser.add_argument(
        "--candidate-graph", choices=graph_choices, default="inductor"
    )
    parser.add_argument("--compile-loss", action="store_true")
    parser.add_argument("--compiler-diagnostics", action="store_true")
    parser.add_argument(
        "--profiler-preset",
        choices=tuple(profiler_presets()),
        default="comparison",
    )
    parser.add_argument("--profile-skip-steps", type=int, default=10)
    parser.add_argument("--profile-warmup-steps", type=int, default=1)
    parser.add_argument("--profile-active-steps", type=int, default=3)
    parser.add_argument("--data-device", choices=("cuda", "npu"))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--require-all", action="store_true")
    args = parser.parse_args()
    selected = select_topologies(
        available=topology_names,
        topology=None if args.topologies else args.topology,
        topologies=args.topologies,
    )
    if args.list_topologies:
        for name in topology_names:
            print(f"{name}: {asdict(topologies[name])}")
        return
    objectives = frozenset(
        value.strip() for value in args.objectives.split(",") if value.strip()
    )
    components = ("model", "loss") if args.compile_loss else ("model",)
    profiler = (
        ProfilerFeatureConfig(
            preset=profiler_presets()[args.profiler_preset],
            skip_steps=args.profile_skip_steps,
            warmup_steps=args.profile_warmup_steps,
            active_steps=args.profile_active_steps,
        )
        if "performance" in objectives
        else None
    )
    selection = CombinationSelection(
        objectives=objectives,
        reference_graph=GraphFeatureConfig(
            args.reference_graph, components, args.compiler_diagnostics
        ),
        candidate_graph=GraphFeatureConfig(
            args.candidate_graph, components, args.compiler_diagnostics
        ),
        profiler=profiler,
    )
    if profiler is not None and base.training.steps < (
        profiler.skip_steps + profiler.warmup_steps + profiler.active_steps
    ):
        raise ValueError("training steps do not reach the profiler window")
    root = _root(script_path)
    raw_configs = {
        name: topology_config(base, topologies[name], precision=args.precision)
        for name in selected
    }
    configs = {
        name: _apply_selection(config, selection)
        for name, config in raw_configs.items()
    }
    if args.data:
        first = next(iter(configs.values()))
        endpoint = _fixture_endpoint_from_environment(first, args.data_device)
        print(
            "Prepared fixture:",
            prepare_fixture(root, first, endpoint=endpoint, force=args.force),
        )
        return
    if args.capture:
        seen: set[Path] = set()
        for name, config in configs.items():
            endpoint = (
                config.reference
                if args.capture == "reference"
                else config.candidate
            )
            endpoint = _endpoint_from_process_environment(endpoint)
            config = replace(config, **{args.capture: endpoint})
            repeats = (
                (args.repeat,)
                if args.repeat
                else range(1, endpoint.repeats + 1)
            )
            for repeat in repeats:
                artifact = _artifact_directory(
                    root, config, args.capture, endpoint, repeat
                )
                if artifact in seen:
                    continue
                seen.add(artifact)
                path = capture_endpoint(
                    root, config, role=args.capture, repeat=repeat, force=args.force
                )
                print(f"Captured {args.capture} {name} repeat {repeat}: {path}")
        return
    print(
        "Combined report:",
        _write_combination_report(
            root,
            base,
            topologies=topologies,
            selected=selected,
            precision=args.precision,
            selection=selection,
            require_all=args.require_all,
        ),
    )
