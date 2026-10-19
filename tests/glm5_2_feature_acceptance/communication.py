"""Evidence-driven communication bottleneck triage."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CommunicationEvidence:
    exposed_percent: float
    overlap_percent: float
    rank_ready_skew_ms: float
    effective_bandwidth_ratio: float | None = None
    repeated_fsdp_all_gathers: int = 0
    all_to_all_percent: float = 0.0
    pipeline_bubble_percent: float = 0.0


@dataclass(frozen=True, slots=True)
class CommunicationRecommendation:
    classification: str
    optimization_ids: tuple[str, ...]
    explanation: str


def diagnose_communication(
    evidence: CommunicationEvidence,
) -> tuple[CommunicationRecommendation, ...]:
    """Select experiments, never automatic fixes, from measured evidence."""
    recommendations: list[CommunicationRecommendation] = []
    if evidence.exposed_percent < 10:
        return (
            CommunicationRecommendation(
                "not_primary",
                (),
                "Exposed communication is below 10%; optimize the measured compute, host, or memory critical path first.",
            ),
        )
    if evidence.rank_ready_skew_ms > 1.0:
        recommendations.append(
            CommunicationRecommendation(
                "readiness_skew",
                (),
                "Ranks enter collectives at different times; inspect input, MoE routing, host scheduling, and preceding kernels before changing the collective backend.",
            )
        )
    if evidence.repeated_fsdp_all_gathers > 0:
        recommendations.append(
            CommunicationRecommendation(
                "fsdp_repeated_prefetch",
                ("fsdp_no_reshard", "fsdp_ag_rs_overlap"),
                "Repeated parameter all-gathers are visible; compare retention and overlap with memory held constant.",
            )
        )
    if evidence.all_to_all_percent >= 20:
        recommendations.append(
            CommunicationRecommendation(
                "expert_exchange",
                ("ep_overlap", "deepep_gpu", "deepep_ascend"),
                "All-to-all dominates exposed communication; test dispatcher and overlap variants for the selected platform.",
            )
        )
    if evidence.pipeline_bubble_percent >= 15:
        recommendations.append(
            CommunicationRecommendation(
                "pipeline_bubble",
                (),
                "Pipeline idle time is high; tune schedule, stages, microbatches, partition balance, and metadata/compile startup separately.",
            )
        )
    if (
        evidence.effective_bandwidth_ratio is not None
        and evidence.effective_bandwidth_ratio < 0.5
        and evidence.rank_ready_skew_ms <= 1.0
    ):
        recommendations.append(
            CommunicationRecommendation(
                "transport_or_payload",
                ("fsdp_bf16_reduce",),
                "Ranks are ready together but effective bandwidth is low; inspect payload size, link path, algorithm, dtype, affinity, and HCCL/NCCL configuration.",
            )
        )
    if evidence.overlap_percent < 50:
        recommendations.append(
            CommunicationRecommendation(
                "insufficient_overlap",
                ("fsdp_ag_rs_overlap", "fsdp_dense_overlap", "ep_overlap"),
                "Less than half of communication overlaps compute; select only the overlap strategy matching the observed collective family.",
            )
        )
    return tuple(recommendations)
