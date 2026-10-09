from argparse import Namespace
import json
from pathlib import Path
import tempfile
from unittest import mock

from tests.glm5_2_common.topology import standard_topologies
from tests.glm5_2_nvidia.workflow import (
    DEFAULT_STATS,
    PROFILE_PRESETS,
    _adopt_legacy_experiment_layout,
    _adopt_legacy_diagnosis,
    _adopt_legacy_outputs,
    _contract,
    _identity_name,
    _write_report,
)
from tests.glm5_2_nvidia.diagnostics import diagnose
from tests.glm5_2_nvidia.ncu_workflow import (
    _completed as ncu_completed,
    _contract as ncu_contract,
    _write_report as write_ncu_report,
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
        profile="standard",
        profiler_off=False,
        replicate=0,
        skip_steps=10,
        workload="representative",
        workload_arg=[],
        graph="eager",
        compile_components=("model",),
        compiler_diagnostics=False,
    )


def test_nvidia_nsys_identity_includes_capture_contract() -> None:
    args = _args()
    topology = standard_topologies()["fsdp8"]
    contract = _contract(args, topology, "NVIDIA Nsight Systems 2026.4.1")
    name = _identity_name(args, topology, contract)

    assert name.startswith(
        "cuda-fsdp8-bf16-s30-l8-b64-seq128-seed61-"
        "workload-representative-standard-"
    )
    assert contract["topology"]["data_parallel_shard_degree"] == 8
    assert contract["cuda_memory_usage"] is True


def test_nvidia_nsys_identity_changes_with_trace_policy() -> None:
    args = _args()
    topology = standard_topologies()["ddp2"]
    first = _identity_name(args, topology, _contract(args, topology, "v1"))
    args.trace += ",nccl"
    second = _identity_name(args, topology, _contract(args, topology, "v1"))

    assert first != second


def test_nvidia_identity_exposes_profile_policy() -> None:
    args = _args()
    args.profile = "communication"
    topology = standard_topologies()["ddp2"]
    name = _identity_name(args, topology, _contract(args, topology, "v1"))
    assert "-communication-" in name


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


def test_legacy_nvidia_diagnosis_is_adopted_into_self_scope() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        run = root / "run"
        output = run / "trainer_output" / "profiling" / "nsys"
        legacy = output / "diagnosis"
        legacy.mkdir(parents=True)
        (legacy / "diagnosis.json").write_text("{}\n", encoding="utf-8")
        (legacy / "diagnosis.md").write_text("legacy\n", encoding="utf-8")

        destination = _adopt_legacy_diagnosis(run, output)

        assert destination == run / "diagnosis" / "self"
        assert (destination / "diagnosis.json").read_text() == "{}\n"
        assert (destination / "diagnosis.md").read_text() == "legacy\n"
        assert not legacy.exists()


def test_layered_profiles_preserve_a_light_standard_first_pass() -> None:
    assert "nccl" not in PROFILE_PRESETS["standard"]["trace"]
    assert PROFILE_PRESETS["standard"]["sample"] == "none"
    assert "nccl" in PROFILE_PRESETS["communication"]["trace"]
    assert PROFILE_PRESETS["host"]["sample"] == "process-tree"
    assert PROFILE_PRESETS["memory"]["cuda_memory_usage"] is True


def test_automatic_diagnosis_is_evidence_linked() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        stats = root / "stats"
        stats.mkdir()
        (stats / "cuda_gpu_kern_sum.csv").write_text(
            'Time (%),Total Time (ns),Instances,Name\n60,1200,2,"ncclKernel"\n40,800,1,"gemm"\n',
            encoding="utf-8",
        )
        (stats / "cuda_api_sum.csv").write_text(
            'Time (%),Total Time (ns),Instances,Name\n70,700,2,"cudaDeviceSynchronize"\n',
            encoding="utf-8",
        )

        result = diagnose(
            stats,
            root / "diagnosis" / "self",
            metadata={"topology": "single", "profile": "standard"},
        )

        payload = json.loads(Path(result["json"]).read_text(encoding="utf-8"))
        assert payload["verdict_policy"] == "diagnostic_only_no_pass_fail"
        assert payload["scope"] == "single_topology"
        assert payload["metadata"]["topology"] == "single"
        assert payload["branches"]["communication"]["status"] == "suspect"
        assert {item["category"] for item in payload["findings"]} >= {
            "communication", "host_or_synchronization", "kernel_hotspots"
        }


def test_nvidia_report_embeds_self_diagnosis() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        run = root / "run"
        artifact = root / "artifact"
        artifact.mkdir()
        diagnosis = run / "diagnosis" / "self"
        diagnosis.mkdir(parents=True)
        (diagnosis / "diagnosis.json").write_text(
            json.dumps(
                {
                    "branches": {
                        "communication": {
                            "status": "suspect",
                            "evidence": [{"name": "ncclKernel"}],
                            "next_action": "Inspect collective arrival.",
                        }
                    }
                }
            ),
            encoding="utf-8",
        )
        (artifact / "manifest.json").write_text(
            json.dumps(
                {
                    "statistics": ["cuda_gpu_kern_sum.csv"],
                }
            ),
            encoding="utf-8",
        )
        report = root / "report.html"

        _write_report(
            report,
            topology=standard_topologies()["single"],
            run_dir=run,
            artifact_dir=artifact,
        )

        content = report.read_text(encoding="utf-8")
        assert "GPU 单拓扑性能诊断" in content
        assert "communication" in content
        assert "ncclKernel" in content


def test_nvidia_metrics_use_shared_interactive_report() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        run = root / "run"
        artifact = root / "artifact"
        run.mkdir()
        artifact.mkdir()
        (run / "metrics.jsonl").write_text(
            json.dumps(
                {
                    "step": 1,
                    "metrics": {
                        "time_metrics/end_to_end(s)": 1.0,
                        "throughput(tps)": 100.0,
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )
        (artifact / "manifest.json").write_text(
            json.dumps(
                {
                    "contract": {
                        "profiler_enabled": False,
                        "config": "glm5_debugmodel",
                        "steps": 20,
                        "skip_steps": 10,
                    }
                }
            ),
            encoding="utf-8",
        )
        report = root / "report.html"

        with mock.patch(
            "tests.glm5_2_nvidia.workflow.write_training_metrics_report"
        ) as write_report:
            _write_report(
                report,
                topology=standard_topologies()["single"],
                run_dir=run,
                artifact_dir=artifact,
            )

        write_report.assert_called_once()
        analysis = write_report.call_args.kwargs["analysis"]
        assert analysis["tool_outputs"][0]["type"] == "TorchTitan metrics"


def test_ncu_contract_records_replay_and_selection() -> None:
    args = Namespace(
        module="glm5", config="glm5_debugmodel", steps=30,
        local_batch_size=8, global_batch_size=64, sequence_length=128,
        seed=61, section_set="basic", kernel_name="regex:.*gemm.*",
        nvtx_include=None, launch_skip=2, launch_count=1,
        replay_mode="kernel",
    )
    contract = ncu_contract(args, standard_topologies()["single"], "2026.4")
    assert contract["kernel_name"] == "regex:.*gemm.*"
    assert contract["launch_skip"] == 2
    assert contract["replay_mode"] == "kernel"


def test_ncu_report_explains_targeted_replay_boundary() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        report = root / "kernel-123.ncu-rep"
        report.write_bytes(b"native")
        output = root / "index.html"
        write_ncu_report(
            output,
            contract={
                "kernel_name": "regex:.*gemm.*",
                "nvtx_include": None,
                "section_set": "detailed",
                "replay_mode": "kernel",
                "launch_skip": 2,
                "launch_count": 1,
            },
            reports=[report],
            runtime_log=root / "runtime.log",
        )

        content = output.read_text(encoding="utf-8")
        assert "GPU 定点算子分析" in content
        assert "SpeedOfLight" in content
        assert "不能用于判断跨 rank 原始耗时" in content


def test_ncu_resume_requires_matching_complete_report() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        manifest = root / "manifest.json"
        report_dir = root / "ncu"
        report_dir.mkdir()
        report = report_dir / "profile-1.ncu-rep"
        contract = {"kernel_name": "gemm"}
        manifest.write_text(
            json.dumps({"status": "completed", "contract": contract}),
            encoding="utf-8",
        )
        assert not ncu_completed(manifest, report_dir, contract)
        report.write_bytes(b"ncu")
        assert ncu_completed(manifest, report_dir, contract)
        assert not ncu_completed(manifest, report_dir, {"kernel_name": "attention"})
