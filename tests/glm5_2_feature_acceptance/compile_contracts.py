"""Acceptance contracts for graph breaks and dynamic-shape compilation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
import json
from pathlib import Path

from .matrix import Platform


class CompileContractKind(StrEnum):
    END_TO_END_GRAPH = "end-to-end-graph"
    OPERATOR_DYNAMIC_SHAPE = "operator-dynamic-shape"


@dataclass(frozen=True, slots=True)
class CompileAcceptanceContract:
    contract_id: str
    kind: CompileContractKind
    platform: Platform
    command: tuple[str, ...]
    required_evidence: tuple[str, ...]
    acceptance: tuple[str, ...]
    scope: str

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["kind"] = self.kind.value
        payload["platform"] = self.platform.value
        return payload


def build_compile_contracts() -> tuple[CompileAcceptanceContract, ...]:
    contracts: list[CompileAcceptanceContract] = []
    for platform in Platform:
        device = "cuda" if platform is Platform.GPU else "npu"
        contracts.append(
            CompileAcceptanceContract(
                contract_id=f"{platform.value}-glm-graph-breaks",
                kind=CompileContractKind.END_TO_END_GRAPH,
                platform=platform,
                command=(
                    "python", "tests/glm5_2_smoke/train_smoke.py",
                    "--device", device,
                    "--topology", "single",
                    "--graph", "inductor",
                    "--compiler-diagnostics",
                    "--steps", "10",
                ),
                required_evidence=(
                    "runtime.log",
                    "graph_visualization/compilation_report.json",
                    "graph_visualization/compilation_report.html",
                    "per-rank TORCH_TRACE or an explicit unavailable reason",
                ),
                acceptance=(
                    "training exits successfully",
                    "loss and grad_norm remain finite",
                    "every graph break has an attributed source location and reason",
                    "recompiles are explained and bounded for the fixed-shape workload",
                ),
                scope="End-to-end GLM graph capture; not a dynamic-shape proof.",
            )
        )
        contracts.append(
            CompileAcceptanceContract(
                contract_id=f"{platform.value}-operator-dynamic-shape",
                kind=CompileContractKind.OPERATOR_DYNAMIC_SHAPE,
                platform=platform,
                command=(
                    "python", "tests/glm5_2_graph/dynamic_shape_probe.py",
                    "--device", device,
                    "--backend", "inductor",
                    "--output", f"graph_probe_reports/{platform.value}/dynamic_shape.json",
                ),
                required_evidence=(
                    "dynamic_shape.json",
                    "forward error per shape",
                    "input-gradient error per shape",
                    "parameter-gradient error per shape",
                    "Dynamo unique-graph and graph-break counters when available",
                ),
                acceptance=(
                    "one compiled callable accepts every requested batch and sequence shape",
                    "forward and backward match eager within dtype tolerances",
                    "no graph break is counted",
                    "unique graphs remain bounded; specialization must be explained",
                ),
                scope=(
                    "Backend/operator dynamic-shape contract. It does not claim that the "
                    "whole GLM training loop accepts arbitrary shapes."
                ),
            )
        )
    return tuple(contracts)


def validate_compile_contracts(
    contracts: tuple[CompileAcceptanceContract, ...],
) -> tuple[CompileAcceptanceContract, ...]:
    ids = [item.contract_id for item in contracts]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate compile contract ids")
    for item in contracts:
        if not item.command or not item.required_evidence or not item.acceptance:
            raise ValueError(f"incomplete compile contract: {item.contract_id}")
        if item.kind is CompileContractKind.END_TO_END_GRAPH:
            if "--compiler-diagnostics" not in item.command:
                raise ValueError(f"missing diagnostics: {item.contract_id}")
    return contracts


def write_compile_contracts(output: Path) -> Path:
    contracts = validate_compile_contracts(build_compile_contracts())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {"schema_version": 1, "contracts": [item.to_dict() for item in contracts]},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return output
