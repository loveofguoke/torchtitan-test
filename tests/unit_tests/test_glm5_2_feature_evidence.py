from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from tests.glm5_2_feature_acceptance.evidence import (
    FeatureEvidence,
    GateEvidence,
    GateState,
    empty_feature_evidence,
    write_evidence_report,
)
from tests.glm5_2_feature_acceptance.matrix import Platform, STANDARD_GATES


class FeatureEvidenceTest(unittest.TestCase):
    def test_empty_report_is_pending_not_accepted(self) -> None:
        evidence = empty_feature_evidence("fsdp2", Platform.NPU, "run-1")
        self.assertFalse(evidence.accepted)
        self.assertEqual(evidence.pending_gates, STANDARD_GATES)

    def test_every_gate_needs_real_pass_evidence(self) -> None:
        gates = {
            gate: GateEvidence(GateState.PASS, (f"artifacts/{gate}.json",), "ok")
            for gate in STANDARD_GATES
        }
        evidence = FeatureEvidence("fsdp2", Platform.NPU, "run-1", gates)
        self.assertTrue(evidence.accepted)

    def test_pass_without_artifact_is_rejected(self) -> None:
        evidence = empty_feature_evidence("fsdp2", Platform.NPU, "run-1")
        gates = dict(evidence.gates)
        gates["function"] = GateEvidence(GateState.PASS)
        with self.assertRaisesRegex(ValueError, "requires an artifact"):
            FeatureEvidence("fsdp2", Platform.NPU, "run-1", gates).validate()

    def test_failure_is_preserved_in_report(self) -> None:
        gates = {
            gate: GateEvidence() for gate in STANDARD_GATES
        }
        gates["numerics"] = GateEvidence(
            GateState.FAIL, ("runs/runtime.log",), "non-finite gradient"
        )
        evidence = FeatureEvidence("flex_attention", Platform.NPU, "run-2", gates)
        with tempfile.TemporaryDirectory() as directory:
            path = write_evidence_report(evidence, Path(directory) / "evidence.json")
            payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["failed_gates"], ["numerics"])
        self.assertFalse(payload["accepted"])


if __name__ == "__main__":
    unittest.main()
