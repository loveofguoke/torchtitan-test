import unittest

from tests.glm5_2_graph.config import (
    GraphFeatureConfig,
    npu_codegen_environment,
)


class TestNpuCodegen(unittest.TestCase):
    def test_loader_names(self):
        self.assertEqual(npu_codegen_environment("dvm"), {"TORCHINDUCTOR_NPU_BACKEND": "dvm"})
        self.assertEqual(npu_codegen_environment("ascend-triton"), {"TORCHINDUCTOR_NPU_BACKEND": "default"})
        self.assertEqual(npu_codegen_environment(None), {})

    def test_identity_and_compile_backend(self):
        dvm = GraphFeatureConfig("inductor", npu_codegen="dvm").feature(device_type="npu")
        triton = GraphFeatureConfig("inductor", npu_codegen="ascend-triton").feature(device_type="npu")
        self.assertEqual(dvm.arguments, triton.arguments)
        self.assertIn("--compile.backend=inductor", dvm.arguments)
        self.assertNotEqual(dvm.metadata, triton.metadata)
        self.assertNotEqual(
            dvm.environment["TORCHINDUCTOR_NPU_BACKEND"],
            triton.environment["TORCHINDUCTOR_NPU_BACKEND"],
        )
        self.assertEqual(
            dvm.environment["TORCHTITAN_PIPELINE_META_USE_BATCH"], "0"
        )
        self.assertEqual(
            triton.environment["TORCHTITAN_PIPELINE_META_USE_BATCH"], "0"
        )
        self.assertEqual(
            triton.environment["TORCHTITAN_PIPELINE_METADATA_FORCE_EAGER"],
            "1",
        )
        self.assertEqual(
            triton.environment["TORCHTITAN_PIPELINE_REAL_INPUT_PRECOMPILE"],
            "1",
        )
        self.assertEqual(
            triton.environment["TORCHTITAN_FIRST_ALL_REDUCE_HOST_BARRIER"],
            "1",
        )
        self.assertNotIn(
            "TORCHTITAN_PIPELINE_STAGE_PRECOMPILE_SHAPE", triton.environment
        )
        self.assertNotIn("TORCHTITAN_PIPELINE_META_TRANSPORT", triton.environment)
        self.assertEqual(triton.environment["TASK_QUEUE_ENABLE"], "0")
        self.assertEqual(
            triton.environment["TORCHTITAN_TASK_QUEUE_ENABLE"], "0"
        )
        self.assertEqual(
            triton.environment["TORCHINDUCTOR_COMPILE_THREADS"], "1"
        )

    def test_eager_regional_compile(self):
        feature = GraphFeatureConfig(npu_codegen="dvm").feature(device_type="npu")
        self.assertEqual(feature.arguments, ())
        self.assertEqual(feature.environment["TORCHINDUCTOR_NPU_BACKEND"], "dvm")

    def test_gpu_rejects_npu_selection(self):
        with self.assertRaises(ValueError):
            GraphFeatureConfig(npu_codegen="dvm").feature(device_type="cuda")

    def test_flexattention_dispatch_remains_torchnpu_owned(self):
        for backend in (None, "dvm", "ascend-triton"):
            feature = GraphFeatureConfig(
                "eager", npu_codegen=backend
            ).feature(device_type="npu")
            self.assertNotIn(
                "TORCHINDUCTOR_FLEXATTENTION_MASKOUT", feature.environment
            )

if __name__ == "__main__":
    unittest.main()
