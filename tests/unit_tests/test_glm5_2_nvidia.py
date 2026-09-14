from argparse import Namespace
import json
from pathlib import Path
import tempfile

from tests.glm5_2_common.topology import standard_topologies
from tests.glm5_2_nvidia.workflow import (
    DEFAULT_STATS,
    _adopt_legacy_experiment_layout,
    _adopt_legacy_outputs,
    _contract,
    _identity_name,
)


def _args() -> Namespace:
    return Namespace(
        module="glm5",
        config="glm5_debugmodel",
        steps=30,
        local_batch_size=8,
        global_batch_size=64,
        sequence_length=128,
        seed=61,
        trace="cuda,nvtx,osrt,cublas,cudnn",
        pytorch="functions-trace-shapes,autograd-nvtx",
        sample="none",
        cuda_memory_usage=True,
        cuda_graph_trace="node",
        delay=None,
        duration=None,
        stats_reports=DEFAULT_STATS,
    )


def test_nvidia_nsys_identity_includes_capture_contract() -> None:
    args = _args()
    topology = standard_topologies()["fsdp8"]
    contract = _contract(args, topology, "NVIDIA Nsight Systems 2026.4.1")
    name = _identity_name(args, topology, contract)

    assert name.startswith("cuda-fsdp8-bf16-s30-l8-b64-seq128-seed61-standard-")
    assert contract["topology"]["data_parallel_shard_degree"] == 8
    assert contract["cuda_memory_usage"] is True


def test_nvidia_nsys_identity_changes_with_trace_policy() -> None:
    args = _args()
    topology = standard_topologies()["ddp2"]
    first = _identity_name(args, topology, _contract(args, topology, "v1"))
    args.trace += ",nccl"
    second = _identity_name(args, topology, _contract(args, topology, "v1"))

    assert first != second


def test_legacy_nsys_payload_is_adopted_by_run_without_reprofiling() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        run = root / "run"
        artifact = root / "artifact"
        artifact.mkdir()
        (artifact / "manifest.json").write_text(
            json.dumps({"capture_status": "completed"}), encoding="utf-8"
        )
        (artifact / "profile.nsys-rep").write_bytes(b"report")
        (artifact / "profile.sqlite").write_bytes(b"sqlite")
        stats = artifact / "stats"
        stats.mkdir()
        (stats / "cuda_api_sum.csv").write_text("Name\n", encoding="utf-8")

        output = _adopt_legacy_outputs(run, artifact)

        assert output == run / "trainer_output" / "profiling" / "nsys"
        assert (output / "profile.nsys-rep").read_bytes() == b"report"
        assert (output / "profile.sqlite").read_bytes() == b"sqlite"
        assert (output / "stats" / "cuda_api_sum.csv").is_file()
        assert not (artifact / "profile.nsys-rep").exists()
        assert (artifact / "manifest.json").is_file()


def test_legacy_nsys_roots_are_adopted_by_nvidia_suite() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        legacy_run = root / "nsys_runs" / "run"
        legacy_artifact = root / "nsys_artifacts" / "artifact"
        legacy_report = root / "nsys_reports" / "report.html"
        legacy_run.mkdir(parents=True)
        legacy_artifact.mkdir(parents=True)
        legacy_report.parent.mkdir(parents=True)
        (legacy_run / "training.log").write_text("complete\n", encoding="utf-8")
        (legacy_artifact / "manifest.json").write_text("{}\n", encoding="utf-8")
        legacy_report.write_text("report\n", encoding="utf-8")
        run = root / "nvidia_runs" / "performance" / "system" / "run"
        artifact = (
            root / "nvidia_artifacts" / "performance" / "system" / "artifact"
        )
        report = (
            root
            / "nvidia_reports"
            / "performance"
            / "system"
            / "report.html"
        )

        _adopt_legacy_experiment_layout(
            run_dir=run,
            artifact_dir=artifact,
            report_path=report,
            legacy_run_dir=legacy_run,
            legacy_artifact_dir=legacy_artifact,
            legacy_report_path=legacy_report,
        )

        assert (run / "training.log").is_file()
        assert (artifact / "manifest.json").is_file()
        assert report.is_file()
        assert not legacy_run.exists()
        assert not legacy_artifact.exists()
        assert not legacy_report.exists()
