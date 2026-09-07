import unittest

from tests.glm5_2_graph.config import GraphFeatureConfig, npu_codegen_environment


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
        self.assertNotEqual(dvm.environment, triton.environment)

    def test_eager_regional_compile(self):
        feature = GraphFeatureConfig(npu_codegen="dvm").feature(device_type="npu")
        self.assertEqual(feature.arguments, ())
        self.assertEqual(feature.environment["TORCHINDUCTOR_NPU_BACKEND"], "dvm")

    def test_gpu_rejects_npu_selection(self):
        with self.assertRaises(ValueError):
            GraphFeatureConfig(npu_codegen="dvm").feature(device_type="cuda")


if __name__ == "__main__":
    unittest.main()
