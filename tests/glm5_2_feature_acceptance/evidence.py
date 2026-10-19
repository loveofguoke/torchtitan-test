"""Evidence contract for promoting a feature beyond its inventory status."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
import json
from pathlib import Path

from .matrix import STANDARD_GATES, Platform


class GateState(StrEnum):
    NOT_RUN = "not-run"
    PASS = "pass"
    FAIL = "fail"


@dataclass(frozen=True, slots=True)
class GateEvidence:
    state: GateState = GateState.NOT_RUN
    artifacts: tuple[str, ...] = ()
    summary: str = ""

    def validate(self, gate: str) -> None:
        if self.state is not GateState.NOT_RUN and not self.artifacts:
            raise ValueError(f"{gate} {self.state.value} requires an artifact path")
        if self.state is GateState.FAIL and not self.summary.strip():
            raise ValueError(f"{gate} failure requires a summary")


@dataclass(frozen=True, slots=True)
class FeatureEvidence:
    feature_id: str
    platform: Platform
    run_identity: str
    gates: dict[str, GateEvidence]

    def validate(self) -> None:
        if set(self.gates) != set(STANDARD_GATES):
            missing = sorted(set(STANDARD_GATES) - set(self.gates))
            extra = sorted(set(self.gates) - set(STANDARD_GATES))
            raise ValueError(f"gate mismatch: missing={missing}, extra={extra}")
        if not self.run_identity.strip():
            raise ValueError("run_identity must be non-empty")
        for gate, evidence in self.gates.items():
            evidence.validate(gate)

    @property
    def accepted(self) -> bool:
        self.validate()
        return all(item.state is GateState.PASS for item in self.gates.values())

    @property
    def failed_gates(self) -> tuple[str, ...]:
        self.validate()
        return tuple(
            gate
            for gate in STANDARD_GATES
            if self.gates[gate].state is GateState.FAIL
        )

    @property
    def pending_gates(self) -> tuple[str, ...]:
        self.validate()
        return tuple(
            gate
            for gate in STANDARD_GATES
            if self.gates[gate].state is GateState.NOT_RUN
        )

    def to_dict(self) -> dict[str, object]:
        self.validate()
        payload = asdict(self)
        payload["platform"] = self.platform.value
        for evidence in payload["gates"].values():
            evidence["state"] = evidence["state"].value
        payload["accepted"] = self.accepted
        payload["failed_gates"] = list(self.failed_gates)
        payload["pending_gates"] = list(self.pending_gates)
        return payload


def empty_feature_evidence(
    feature_id: str, platform: Platform, run_identity: str
) -> FeatureEvidence:
    return FeatureEvidence(
        feature_id=feature_id,
        platform=platform,
        run_identity=run_identity,
        gates={gate: GateEvidence() for gate in STANDARD_GATES},
    )


def write_evidence_report(evidence: FeatureEvidence, output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(evidence.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output
