# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Central orchestration for orthogonal precision, graph, and profiler features.

Combination experiments do not make one subsystem inject hidden options into
another. They compose the same graph/profiler/precision feature objects used by
standalone experiments at one checked execution boundary. This supports eager
vs eager, eager vs graph, and graph vs graph precision; performance can be
enabled independently or in the same runs when the desired collection policy
is explicit.

The combined report links the authoritative precision report, profiler report,
TensorBoard/stack/memory outputs, and compiler diagnostics rather than
reimplementing their analysis logic.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import hashlib
import html
import json
import os
from pathlib import Path
from typing import Sequence

from tests.glm5_2_common.execution import compose_execution
from tests.glm5_2_common.topology import ParallelTopology, select_topologies
from tests.glm5_2_graph.config import GraphFeatureConfig
from tests.glm5_2_graph.visualization import (
    generate_tlparse_reports,
    inspect_graph_visualizations,
)
from tests.glm5_2_performance.analysis import build_analysis, render_html_report
from tests.glm5_2_performance.config import profiler_presets
from tests.glm5_2_performance.visualization import (
    inspect_tensorboard,
    render_flamegraphs,
    render_mindstudio_flamegraphs,
)
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
    reset_capture_outputs,
)

from .config import ProfilerFeatureConfig


@dataclass(frozen=True)
class CombinationSelection:
    objectives: frozenset[str]
    reference_graph: GraphFeatureConfig
    candidate_graph: GraphFeatureConfig
    profiler: ProfilerFeatureConfig | None
    performance_skip_steps: int = 10

    def __post_init__(self) -> None:
        unknown = self.objectives - {"precision", "performance"}
        if unknown or not self.objectives:
            raise ValueError(f"invalid combination objectives: {sorted(unknown)}")
        if self.performance_skip_steps < 0:
            raise ValueError("performance skip steps must be non-negative")


def _combination_storage_prefix(
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


def _combination_storage_base(
    config: FormalExperimentConfig,
    selection: CombinationSelection,
) -> str:
    base = _combination_storage_prefix(config, selection)
    if "performance" in selection.objectives:
        base += f"-skip{selection.performance_skip_steps}"
    if not config.training.deterministic:
        base += "-nondet"
    return base


def _transitional_combination_storage_base(
    config: FormalExperimentConfig,
    selection: CombinationSelection,
) -> str:
    """Return the short-lived storage name used before compatibility review."""

    determinism = "det" if config.training.deterministic else "nondet"
    return (
        f"{_combination_storage_prefix(config, selection)}-"
        f"skip{selection.performance_skip_steps}-{determinism}"
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
        if "performance" in selection.objectives and endpoint.device_type != "npu":
            raise NotImplementedError(
                "combined performance experiments currently support only NPU "
                "endpoints; the CUDA performance interface is reserved"
            )
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
    selected = replace(
        config,
        reference=endpoints["reference"],
        candidate=endpoints["candidate"],
        allow_endpoint_argument_difference=True,
        artifact_root="combination_artifacts",
        run_root="combination_runs",
        storage_name_override=storage_base,
        topology_subdirectory=True,
        legacy_storage_names=(),
    )
    legacy_names = (
        replace(
            selected,
            storage_name_override=_combination_storage_prefix(config, selection),
        ).storage_name,
        replace(
            selected,
            storage_name_override=_transitional_combination_storage_base(
                config, selection
            ),
        ).storage_name,
        _legacy_combination_storage_name(config, selection),
    )
    unique_legacy_names = dict.fromkeys(
        name for name in legacy_names if name != selected.storage_name
    )
    selected = replace(
        selected,
        legacy_storage_names=tuple(unique_legacy_names),
    )
    return replace(
        selected,
        fixture_name=config.fixture_name,
        report_root="combination_reports/precision",
    )


def _metric_median(analysis: dict, fragment: str) -> float | None:
    for name, values in analysis["metrics"]["summary"].items():
        if fragment in name.lower():
            return values.get("median")
    return None


def _performance_median(analysis: dict) -> float | None:
    phase_median = analysis.get("profile_phases", {}).get("comparison", {}).get(
        "baseline_median_step_seconds"
    )
    if phase_median is not None:
        return float(phase_median)
    return _metric_median(analysis, "end_to_end")


def _relative_link(path: Path, root: Path) -> str:
    return html.escape(path.relative_to(root).as_posix())


def _portable_link(path: Path, root: Path) -> str:
    return html.escape(
        Path(os.path.relpath(path, root)).as_posix(),
        quote=True,
    )


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
    generate_tlparse_reports(run_directory)
    profiler_directory = (
        run_directory / "trainer_output" / "profiling" / "traces"
    )
    render_mindstudio_flamegraphs(profiler_directory)
    render_flamegraphs(profiler_directory)
    profiler = selection.profiler
    run_name = f"{role}-{endpoint.device_type}-{endpoint.topology.slug}-r{repeat}"
    analysis_config = {
        "steps": config.training.steps,
        "local_batch_size": config.training.local_batch_size,
        "global_batch_size": config.training.global_batch_size,
        "sequence_length": config.training.sequence_length,
        "topology": endpoint.topology.name,
        "profiler_enabled": profiler is not None,
        "skip_steps": (
            profiler.skip_steps
            if profiler is not None
            else selection.performance_skip_steps
        ),
        "warmup_steps": profiler.warmup_steps if profiler is not None else 0,
        "active_steps": profiler.active_steps if profiler is not None else 0,
        "extra_args": list(endpoint.extra_args),
        "environment": endpoint.environment,
    }
    profiler_environment = (
        profiler.preset.environment() if profiler is not None else {}
    )
    analysis = build_analysis(
        run_directory,
        metrics_path=metrics_path,
        config=analysis_config,
        profiler_environment=profiler_environment,
    )
    manifest = {
        "run_name": run_name,
        "device": endpoint.device_type,
        "topology": endpoint.topology.name,
        "preset": profiler.preset.name if profiler is not None else "off",
        "config": analysis_config,
        "profiler_environment": profiler_environment,
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
    return output, _performance_median(analysis)


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

    graph_rows = []
    seen_graph_runs: set[Path] = set()
    for name in selected:
        raw = topology_config(base, topologies[name], precision=precision)
        config = _apply_selection(raw, selection)
        for role, endpoint, graph in (
            ("reference", config.reference, selection.reference_graph),
            ("candidate", config.candidate, selection.candidate_graph),
        ):
            for repeat in range(1, endpoint.repeats + 1):
                run_directory = _run_directory(
                    root, config, role, endpoint, repeat
                ).resolve()
                if run_directory in seen_graph_runs:
                    continue
                seen_graph_runs.add(run_directory)
                if graph.mode != "eager" and graph.diagnostics:
                    generate_tlparse_reports(run_directory)
                inventory = inspect_graph_visualizations(run_directory)
                graph_root = Path(inventory["root"])
                tensorboard = inspect_tensorboard(run_directory)

                def links(kind: str, label: str) -> str:
                    values = inventory.get(kind, [])[:8]
                    if not values:
                        return "-"
                    return "<br>".join(
                        f'<a href="{_portable_link(graph_root / value, report_root)}">'
                        f"{html.escape(label)} {index + 1}</a>"
                        for index, value in enumerate(values)
                    )

                if graph.mode == "eager":
                    status = "not applicable (eager)"
                elif not graph.diagnostics:
                    status = "not captured; use --compiler-diagnostics"
                elif inventory["tlparse_reports"]:
                    status = "interactive report ready"
                elif inventory["structured_traces"]:
                    status = (
                        "tlparse is not installed; install it and compare again"
                    )
                else:
                    status = "diagnostics requested but no structured trace found"
                tensorboard_links = "<br>".join(
                    f'<a href="{_portable_link(run_directory / value, report_root)}">'
                    f"event {index + 1}</a>"
                    for index, value in enumerate(
                        tensorboard.get("event_files", [])[:8]
                    )
                ) or "-"
                tensorboard_command = html.escape(
                    " ".join(tensorboard.get("command", []))
                )
                tensorboard_cell = (
                    f"<details><summary>{tensorboard.get('event_count', 0)} "
                    f"event file(s)</summary><code>{tensorboard_command}</code>"
                    f"<br>{tensorboard_links}</details>"
                )
                graph_rows.append(
                    "<tr>"
                    f"<td>{html.escape(name)}</td><td>{html.escape(role)}</td>"
                    f"<td>{repeat}</td><td>{html.escape(graph.mode)}</td>"
                    f"<td>{html.escape(status)}</td>"
                    f"<td>{links('tlparse_reports', 'tlparse')}</td>"
                    f"<td>{links('fx_graphs', 'FX graph')}</td>"
                    f"<td>{links('inductor_ir', 'IR')}</td>"
                    f"<td>{links('generated_code', 'code')}</td>"
                    f"<td>{links('structured_traces', 'trace')}</td>"
                    f"<td>{tensorboard_cell}</td></tr>"
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
        f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>GLM-5.2 combined experiment</title>
<style>body{{font:14px/1.5 system-ui,sans-serif;max-width:1400px;margin:auto;padding:24px;color:#172033}}
section{{border:1px solid #dfe5ef;border-radius:12px;padding:20px;margin:16px 0}}table{{border-collapse:collapse;width:100%}}
th,td{{padding:9px;border-bottom:1px solid #dfe5ef;text-align:left}}th{{position:sticky;top:0;background:#f8fafc}}a{{color:#155eef}}code{{background:#edf2ff;padding:2px 5px;border-radius:4px}}</style>
</head><body><h1>GLM-5.2 combined experiment</h1>
<p>Objectives: {html.escape(', '.join(sorted(selection.objectives)))}; reference graph: {selection.reference_graph.mode}; candidate graph: {selection.candidate_graph.mode}.</p>
<section><h2>中文阅读路线</h2>
<p>先打开 Precision 套件确认 loss/grad norm 是否通过，再看 Performance 的 profiler-off 中位 step time 与 speedup。图模式表按 topology、端点和 rank 链接编译诊断：Interactive 是 <code>tlparse</code> 总览，FX 是 Dynamo 捕获图，IR 是融合前后调度表示，Code 是后端生成代码，Raw trace 用于复现和重新解析。TensorBoard 只看训练标量，不替代编译图或 NPU 时间线。</p>
<p>图模式成功的最低证据链是：无未解释 backend failure；graph break/recompile 数量可解释且不会随 step 无界增长；精度通过；再用性能报告证明 steady-state 收益。一次成功启动不等于图模式验收通过。</p></section>
<section><h2>Precision</h2><p>{precision_link}</p></section>
<section><h2>Performance</h2><table><thead><tr><th>Topology</th><th>Repeat</th><th>Reference median step time</th><th>Candidate median step time</th><th>Candidate speedup</th><th>Details</th></tr></thead>
<tbody>{''.join(rows) or '<tr><td colspan="6">Performance objective disabled</td></tr>'}</tbody></table></section>
<section><h2>torch.compile visualization and TensorBoard</h2><p>Compiled endpoints can emit an official <code>TORCH_TRACE</code>. When <code>tlparse</code> is installed on the compare host, compare converts each rank trace into an interactive HTML stack trie with graph breaks, recompiles, FX graphs, and generated code. Inductor debug files remain linked separately because NPUGraph and Inductor do not expose identical backend artifacts. Every endpoint also exposes its TensorBoard scalar event files and exact launch command.</p>
<table><thead><tr><th>Topology</th><th>Role</th><th>Repeat</th><th>Mode</th><th>Status</th><th>Interactive</th><th>FX</th><th>IR</th><th>Code</th><th>Raw trace</th><th>TensorBoard</th></tr></thead><tbody>{''.join(graph_rows) or '<tr><td colspan="11">No graph endpoint selected</td></tr>'}</tbody></table>
<p><code>pip install tlparse</code>. Structured traces contain model source code but no weights; review them before sharing.</p></section>
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
    default_topology: str = "all",
    default_objectives: str = "precision,performance",
) -> None:
    parser = argparse.ArgumentParser(description="Composable GLM-5.2 experiment")
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--data", action="store_true")
    actions.add_argument("--capture", choices=("reference", "candidate"))
    actions.add_argument("--compare", action="store_true")
    actions.add_argument("--list-topologies", action="store_true")
    topology_selection = parser.add_mutually_exclusive_group()
    topology_selection.add_argument(
        "--topology", choices=("all", *topology_names)
    )
    topology_selection.add_argument(
        "--topologies", help="comma-separated topology subset or all"
    )
    parser.add_argument("--repeat", type=int)
    parser.add_argument("--precision", choices=("fp32", "bf16", "full-bf16"))
    parser.add_argument("--objectives", default=default_objectives)
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
        choices=("off", *profiler_presets()),
        default="off",
    )
    parser.add_argument("--profile-skip-steps", type=int, default=10)
    parser.add_argument("--profile-warmup-steps", type=int, default=1)
    parser.add_argument("--profile-active-steps", type=int, default=3)
    parser.add_argument(
        "--performance-skip-steps",
        type=int,
        default=10,
        help="exclude startup/compile steps from profiler-off performance summaries",
    )
    parser.add_argument(
        "--steps",
        type=int,
        help="override the maintained training length for an identified exploration",
    )
    parser.add_argument(
        "--performance-nondeterministic",
        action="store_true",
        help=(
            "disable deterministic algorithms for performance-only Inductor "
            "autotuning; forbidden when precision is selected"
        ),
    )
    parser.add_argument("--data-device", choices=("cuda", "npu"))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--require-all", action="store_true")
    args = parser.parse_args()
    selected = select_topologies(
        available=topology_names,
        topology=args.topology,
        topologies=args.topologies,
        default=(
            tuple(topology_names)
            if default_topology == "all"
            else (default_topology,)
        ),
    )
    if args.list_topologies:
        for name in topology_names:
            print(f"{name}: {asdict(topologies[name])}")
        return
    objectives = frozenset(
        value.strip() for value in args.objectives.split(",") if value.strip()
    )
    if args.performance_nondeterministic:
        if objectives != {"performance"}:
            raise ValueError(
                "--performance-nondeterministic requires performance-only objectives"
            )
        base = replace(base, training=replace(base.training, deterministic=False))
    components = ("model", "loss") if args.compile_loss else ("model",)
    profiler = (
        ProfilerFeatureConfig(
            preset=profiler_presets()[args.profiler_preset],
            skip_steps=args.profile_skip_steps,
            warmup_steps=args.profile_warmup_steps,
            active_steps=args.profile_active_steps,
        )
        if "performance" in objectives and args.profiler_preset != "off"
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
        performance_skip_steps=args.performance_skip_steps,
    )
    if args.steps is not None:
        if objectives != {"performance"}:
            raise ValueError("--steps requires performance-only objectives")
        if args.steps < 10:
            raise ValueError("performance experiments require at least 10 steps")
        base = replace(base, training=replace(base.training, steps=args.steps))
    if (
        "performance" in objectives
        and profiler is None
        and args.performance_skip_steps >= base.training.steps
    ):
        raise ValueError(
            "performance skip steps must leave at least one measured step"
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
        prepared: list[tuple[FormalExperimentConfig, tuple[int, ...]]] = []
        reset_parents: set[Path] = set()
        for config in configs.values():
            endpoint = (
                config.reference
                if args.capture == "reference"
                else config.candidate
            )
            endpoint = _endpoint_from_process_environment(endpoint)
            config = replace(config, **{args.capture: endpoint})
            repeats = tuple(
                (args.repeat,)
                if args.repeat
                else range(1, endpoint.repeats + 1)
            )
            prepared.append((config, repeats))
            if args.force:
                parent = _artifact_directory(
                    root, config, args.capture, endpoint, repeats[0]
                ).parent
                if parent not in reset_parents:
                    reset_parents.add(parent)
                    reset_capture_outputs(
                        root,
                        config,
                        role=args.capture,
                        repeats=repeats,
                    )
        seen: set[Path] = set()
        for (name, _), (config, repeats) in zip(configs.items(), prepared):
            endpoint = (
                config.reference
                if args.capture == "reference"
                else config.candidate
            )
            for repeat in repeats:
                artifact = _artifact_directory(
                    root, config, args.capture, endpoint, repeat
                )
                if artifact in seen:
                    continue
                seen.add(artifact)
                path = capture_endpoint(
                    root, config, role=args.capture, repeat=repeat, force=False
                )
                capture_topology = endpoint.topology.slug
                if path is None:
                    print(
                        f"Completed {args.capture} {capture_topology} repeat "
                        f"{repeat}; this node does not own the metrics artifact."
                    )
                else:
                    print(
                        f"Captured {args.capture} {capture_topology} repeat "
                        f"{repeat}: {path}"
                    )
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
