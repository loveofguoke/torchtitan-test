"""Official msprof-analyze advanced recipe selection for GLM experiments."""

from __future__ import annotations

from dataclasses import dataclass

from tests.glm5_2_common.topology import ParallelTopology


# Recipes that only read the captured DB and produce derived output. Recipes
# that mutate rank databases (mstx2commop, p2p_pairing, and pp_chart) remain
# explicit because rerunning them can change the portable Insight input.
READ_ONLY_RECIPES = frozenset(
    {
        "cluster_time_summary",
        "cluster_time_compare_summary",
        "module_statistic",
        "calibrate_npu_gpu",
        "compute_op_sum",
        "freq_analysis",
        "ep_load_balance",
        "computational_op_masking",
        "operator_mfu",
        "communication_group_map",
        "communication_time_sum",
        "communication_matrix_sum",
        "hccl_sum",
        "slow_rank",
        "slow_link",
        "communication_bottleneck",
        "cann_api_sum",
        "mstx_sum",
        "free_analysis",
        "export_summary",
    }
)
MUTATING_RECIPES = frozenset({"mstx2commop", "p2p_pairing", "pp_chart"})
KNOWN_RECIPES = READ_ONLY_RECIPES | MUTATING_RECIPES

# The standard diagnostic set needs no model-specific MSTX ranges and applies
# to every multi-rank Level1+ Ascend PyTorch Profiler capture. It covers the
# official compute, communication, slow-rank, slow-link, and Host dispatch
# drill-down path without altering captured rank databases.
NECESSARY_RECIPES = (
    "cluster_time_summary",
    "compute_op_sum",
    "freq_analysis",
    "communication_group_map",
    "communication_time_sum",
    "communication_matrix_sum",
    "hccl_sum",
    "slow_rank",
    "slow_link",
    "communication_bottleneck",
    "cann_api_sum",
    "free_analysis",
    "export_summary",
)


@dataclass(frozen=True)
class AdvancedRecipePlan:
    """Resolved advanced recipe plan and documented omissions."""

    recipes: tuple[str, ...]
    skipped: dict[str, str]


def parse_recipe_policy(value: str) -> tuple[str, ...] | str:
    """Parse standard policy names or a comma-separated recipe list."""

    normalized = value.strip().lower()
    if normalized in {"none", "necessary", "all"}:
        return normalized
    recipes = tuple(dict.fromkeys(item.strip() for item in value.split(",")))
    unknown = sorted(set(recipes) - KNOWN_RECIPES)
    if not recipes or any(not item for item in recipes) or unknown:
        choices = ", ".join(sorted(KNOWN_RECIPES))
        raise ValueError(
            f"unknown msprof-analyze recipe(s): {unknown}; choices: {choices}"
        )
    return recipes


def resolve_recipe_plan(
    policy: str,
    *,
    topology: ParallelTopology,
    record_shapes: bool,
    mstx_enabled: bool,
    with_flops: bool,
    cluster_summary_baseline: bool,
) -> AdvancedRecipePlan:
    """Resolve a topology- and capture-aware official recipe plan."""

    parsed = parse_recipe_policy(policy)
    skipped: dict[str, str] = {}
    if parsed == "none":
        return AdvancedRecipePlan((), skipped)
    if parsed == "necessary":
        selected = list(NECESSARY_RECIPES)
        # GLM5 is an MoE model in every topology.  The recipe remains useful
        # at EP=1 for expert-token and GroupedMM imbalance; EP>1 additionally
        # exposes cross-rank dispatch and expert placement imbalance.
        selected.append("ep_load_balance")
        if cluster_summary_baseline:
            selected.append("cluster_time_compare_summary")
    elif parsed == "all":
        selected = list(READ_ONLY_RECIPES)
    else:
        selected = list(parsed)

    if "cluster_time_compare_summary" in selected and not cluster_summary_baseline:
        selected.remove("cluster_time_compare_summary")
        skipped["cluster_time_compare_summary"] = (
            "requires --cluster-summary-baseline"
        )
    if "calibrate_npu_gpu" in selected:
        selected.remove("calibrate_npu_gpu")
        skipped["calibrate_npu_gpu"] = (
            "requires a dedicated Nsys GPU baseline and is not the same as "
            "cluster_time_compare_summary"
        )
    if "ep_load_balance" in selected and not record_shapes:
        selected.remove("ep_load_balance")
        skipped["ep_load_balance"] = "requires record_shapes capture data"
    for recipe in ("mstx_sum", "module_statistic"):
        if recipe in selected and not mstx_enabled:
            selected.remove(recipe)
            skipped[recipe] = "requires an MSTX-enabled capture"
    if "operator_mfu" in selected and not (mstx_enabled and with_flops):
        selected.remove("operator_mfu")
        skipped["operator_mfu"] = "requires MSTX and with_flops capture data"

    # Mutating recipes are accepted only when explicitly named. They are never
    # added by necessary/all, because pp_chart also requires TorchTitan PP MSTX
    # ranges and metadata that cannot be inferred from an ordinary capture.
    return AdvancedRecipePlan(tuple(dict.fromkeys(selected)), skipped)


def recipe_arguments(
    recipe: str,
    *,
    rank_id: int | None = None,
    top_num: int = 20,
    export_type: str | None = None,
) -> list[str]:
    """Return official optional arguments for one advanced recipe."""

    if recipe == "communication_bottleneck":
        if rank_id is None:
            raise ValueError("communication_bottleneck requires rank_id")
        arguments = [
            "--rank_id",
            str(rank_id),
            "--top_num",
            str(top_num),
        ]
        if export_type is not None:
            arguments.extend(("--export_type", export_type))
        return arguments
    if recipe == "free_analysis":
        arguments = ["--top_num", str(top_num)]
        if export_type is not None:
            arguments.extend(("--export_type", export_type))
        return arguments
    if recipe in {
        "cluster_time_summary",
        "module_statistic",
        "compute_op_sum",
        "communication_time_sum",
        "communication_matrix_sum",
        "hccl_sum",
        "cann_api_sum",
        "mstx_sum",
    }:
        return ["--export_type", export_type] if export_type is not None else []
    return []
