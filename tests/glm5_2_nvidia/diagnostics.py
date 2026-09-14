"""Deterministic, evidence-linked triage for Nsight Systems CSV reports."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def _number(row: dict[str, str], *names: str) -> float | None:
    for name in names:
        value = row.get(name)
        if value in (None, ""):
            continue
        try:
            return float(value.replace(",", "").rstrip("%"))
        except ValueError:
            continue
    return None


def _rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _top(path: Path, limit: int = 10) -> list[dict[str, Any]]:
    result = []
    for row in _rows(path):
        name = row.get("Name") or row.get("Kernel Name") or row.get("API Name")
        time_pct = _number(row, "Time (%)", "Time(%)", "Percentage")
        total = _number(row, "Total Time (ns)", "Total Time", "Duration (ns)")
        instances = _number(row, "Instances", "Calls", "Count")
        if name:
            result.append(
                {"name": name, "time_percent": time_pct, "total_time": total,
                 "instances": instances}
            )
    return result[:limit]


def diagnose(stats_dir: Path, output_dir: Path) -> dict[str, Any]:
    """Create conservative triage; findings point to evidence, not verdicts."""

    kernels = _top(stats_dir / "cuda_gpu_kern_sum.csv")
    apis = _top(stats_dir / "cuda_api_sum.csv")
    memory = _top(stats_dir / "cuda_gpu_mem_time_sum.csv")
    nvtx = _top(stats_dir / "nvtx_sum.csv")
    findings: list[dict[str, Any]] = []

    sync_names = ("synchronize", "event", "wait")
    sync = [row for row in apis if any(word in row["name"].lower() for word in sync_names)]
    if sync:
        findings.append({
            "category": "host_or_synchronization",
            "severity": "investigate",
            "evidence": sync[:5],
            "next_step": "Inspect CUDA API-to-kernel correlation and GPU gaps in the .nsys-rep; rerun profile=host only if CPU attribution is required.",
        })
    nccl = [row for row in kernels if "nccl" in row["name"].lower()]
    if nccl:
        findings.append({
            "category": "communication",
            "severity": "investigate",
            "evidence": nccl[:5],
            "next_step": "Use profile=communication on every rank, then compare collective arrival and execution intervals; kernel duration alone does not prove a network bottleneck.",
        })
    if memory:
        findings.append({
            "category": "memory_operations",
            "severity": "information",
            "evidence": memory[:5],
            "next_step": "Inspect allocation and memcpy intervals in Nsight Systems; use a PyTorch memory snapshot for allocator blocks, tensors, and fragmentation.",
        })
    if kernels:
        findings.append({
            "category": "kernel_hotspots",
            "severity": "information",
            "evidence": kernels[:5],
            "next_step": "Select a stable hotspot and run kernel_benchmark.py with a narrow kernel/NVTX filter before collecting detailed or full NCU sections.",
        })

    payload = {
        "schema": "torchtitan.glm5_2.nvidia.system_diagnosis",
        "schema_version": 1,
        "interpretation": "triage_only",
        "inputs": {path.name: str(path) for path in sorted(stats_dir.glob("*.csv"))},
        "top_kernels": kernels,
        "top_cuda_apis": apis,
        "top_memory_operations": memory,
        "top_nvtx_ranges": nvtx,
        "findings": findings,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "diagnosis.json"
    markdown_path = output_dir / "diagnosis.md"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# NVIDIA system performance triage", "",
        "This is an automatic triage backed by Nsight Systems statistics; it is not a performance verdict.", "",
    ]
    for finding in findings:
        lines.extend((
            f"## {finding['category']}", "",
            f"- Severity: `{finding['severity']}`",
            f"- Evidence: `{json.dumps(finding['evidence'], ensure_ascii=False)}`",
            f"- Next step: {finding['next_step']}", "",
        ))
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(markdown_path), "findings": findings}
