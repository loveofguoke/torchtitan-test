# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Opt-in performance profiles built from existing TorchTitan capabilities."""

from __future__ import annotations

from dataclasses import dataclass, replace

from tests.glm5_2_common.full_dsa import (
    NPU_SPARSE_OVERRIDES,
    NPU_TRITON_OVERRIDES,
)
from tests.glm5_2_performance.config import PerformanceConfig


OPTIMIZATION_PROFILE_ENV = "GLM5_PERFORMANCE_OPTIMIZATION"


@dataclass(frozen=True)
class OptimizationProfile:
    """One measurable optimization hypothesis, never a default model change."""

    name: str
    category: str
    hypothesis: str
    extra_args: tuple[str, ...] = ()
    module: str = "glm5"
    model_config: str = "glm5_full_dsa_debugmodel"
    allowed_topologies: tuple[str, ...] | None = None

    def apply(self, config: PerformanceConfig) -> PerformanceConfig:
        environment = dict(config.environment)
        environment[OPTIMIZATION_PROFILE_ENV] = self.name
        return replace(
            config,
            name=f"{config.name}-{self.name}",
            module=self.module,
            model_config=self.model_config,
            extra_args=config.extra_args + self.extra_args,
            environment=environment,
        )

    def validate_topologies(self, selected: tuple[str, ...]) -> None:
        if self.allowed_topologies is None:
            return
        unsupported = sorted(set(selected) - set(self.allowed_topologies))
        if unsupported:
            raise ValueError(
                f"optimization {self.name!r} requires a supported TP topology; "
                f"unsupported selections: {unsupported}"
            )


def optimization_profiles() -> dict[str, OptimizationProfile]:
    """Return independently measurable Full DSA optimization profiles."""

    tp_topologies = (
        "tp2",
        "tp4",
        "tp8",
        "fsdp2-tp2",
        "fsdp2-tp4",
        "fsdp4-tp2",
    )
    return {
        "reference": OptimizationProfile(
            "reference",
            "baseline",
            "Readable PyTorch Full DSA without optional operator overrides.",
        ),
        "sparse-mla": OptimizationProfile(
            "sparse-mla",
            "compute",
            "Fuse indexed SparseMLA to reduce launches and intermediate tensors.",
            extra_args=(
                "--override.imports=" + ",".join(NPU_SPARSE_OVERRIDES),
            ),
        ),
        "triton-full-dsa": OptimizationProfile(
            "triton-full-dsa",
            "compute",
            "Use Triton-Ascend index-score and SparseMLA forward/backward kernels.",
            extra_args=(
                "--override.imports=" + ",".join(NPU_TRITON_OVERRIDES),
            ),
        ),
        "async-tp": OptimizationProfile(
            "async-tp",
            "communication",
            "Pipeline TP collectives with compiled matrix multiplication.",
            extra_args=(
                "--compile.enable",
                "--compile.components=model",
                "--compile.backend=inductor",
                "--compile.enable_async_tensor_parallel",
            ),
            allowed_topologies=tp_topologies,
        ),
        "foreach-optimizer": OptimizationProfile(
            "foreach-optimizer",
            "compute",
            "Batch optimizer tensor operations to reduce Python and launch overhead.",
            extra_args=("--optimizer.implementation=foreach",),
        ),
        "chunked-loss": OptimizationProfile(
            "chunked-loss",
            "memory",
            "Use sixteen loss chunks to reduce peak logits materialization.",
            extra_args=("--loss.num_chunks=16",),
        ),
        "cpu-offload": OptimizationProfile(
            "cpu-offload",
            "memory",
            "Trade host transfers for lower device-resident parameter memory.",
            extra_args=("--training.enable_cpu_offload",),
        ),
        "full-recompute": OptimizationProfile(
            "full-recompute",
            "compute-memory-tradeoff",
            "Recompute full transformer blocks during backward to save activations.",
            module="tests.glm5_2_performance",
            model_config="glm5_full_dsa_full_ac_debugmodel",
        ),
    }


__all__ = [
    "OPTIMIZATION_PROFILE_ENV",
    "OptimizationProfile",
    "optimization_profiles",
]
