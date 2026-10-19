from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from tests.glm5_2_feature_acceptance.matrix import (
    AcceptanceStatus,
    FeatureSpec,
    Platform,
    PlatformAcceptance,
    README_FEATURE_IDS,
    STANDARD_GATES,
    build_feature_matrix,
    render_markdown,
    validate_feature_matrix,
    write_outputs,
)


class FeatureAcceptanceTest(unittest.TestCase):
    def test_official_feature_ids_are_unique_and_use_all_gates(self) -> None:
        features = validate_feature_matrix(build_feature_matrix())
        self.assertGreaterEqual(len(features), 30)
        self.assertEqual(len(features), len({item.feature_id for item in features}))
        self.assertTrue(all(item.gates == STANDARD_GATES for item in features))

    def test_current_readme_feature_inventory_is_complete(self) -> None:
        feature_ids = {item.feature_id for item in build_feature_matrix()}
        self.assertEqual(README_FEATURE_IDS - feature_ids, set())

    def test_capability_blocks_are_explicit(self) -> None:
        features = {item.feature_id: item for item in build_feature_matrix()}
        self.assertEqual(
            features["dist_moe"].npu.status,
            AcceptanceStatus.BLOCKED_BY_CAPABILITY,
        )
        self.assertEqual(
            features["mxfp8"].gpu.status,
            AcceptanceStatus.BLOCKED_BY_CAPABILITY,
        )
        self.assertEqual(
            features["flashmla"].npu.status,
            AcceptanceStatus.BLOCKED_BY_CAPABILITY,
        )

    def test_platform_status_selection(self) -> None:
        feature = build_feature_matrix()[0]
        self.assertIs(feature.acceptance(Platform.GPU), feature.gpu)
        self.assertIs(feature.acceptance(Platform.NPU), feature.npu)

    def test_duplicate_ids_are_rejected(self) -> None:
        feature = FeatureSpec(
            feature_id="duplicate",
            category="test",
            name="duplicate",
            upstream_surface="test",
            prototype_owner="test",
            required_topologies=("single",),
            gates=STANDARD_GATES,
            gpu=PlatformAcceptance(AcceptanceStatus.NOT_STARTED, "test"),
            npu=PlatformAcceptance(AcceptanceStatus.NOT_STARTED, "test"),
        )
        with self.assertRaisesRegex(ValueError, "duplicate feature ids"):
            validate_feature_matrix((feature, feature))

    def test_rendered_outputs_are_deterministic(self) -> None:
        features = build_feature_matrix()
        rendered = render_markdown(features)
        self.assertIn("`graph_trainer`", rendered)
        self.assertIn("`blocked-by-capability`", rendered)
        with tempfile.TemporaryDirectory() as directory:
            json_path, markdown_path = write_outputs(Path(directory))
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema_version"], 1)
            self.assertEqual(len(payload["features"]), len(features))
            self.assertEqual(markdown_path.read_text(encoding="utf-8"), rendered)


if __name__ == "__main__":
    unittest.main()
