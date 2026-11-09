"""Replay an existing Turbo NPU training capture on CUDA without new fixtures."""

import argparse
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tests.glm5_2_common.cli import print_output_path, run_managed_process


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_evidence(path, runtime):
    evidence = json.loads(Path(path).read_text(encoding="utf-8"))
    if (evidence["runtime"] != runtime or evidence["status"] != "passed"
            or evidence["world_size"] != 1 or evidence["mtp"]):
        raise ValueError("Expected a completed single-rank non-MTP capture of " + runtime)
    if evidence["expected_steps"] != 500 or evidence["completed_steps"] != 500:
        raise ValueError("The comparison requires exactly 500 completed steps")
    if evidence["precision"] != {"dtype": "float32", "param_dtype": "bfloat16", "reduce_dtype": "float32"}:
        raise ValueError("Expected FP32 master/reduction and BF16 parameters")
    if not evidence["deterministic"] or evidence["remat"] not in ("selective", "full"):
        raise ValueError("Determinism and activation rematerialization are required")
    if evidence.get("local_compile_regions") != []:
        raise ValueError("This comparison requires the recorded eager configuration")
    rows = evidence["metrics"]
    if [row["step"] for row in rows] != list(range(1, 501)):
        raise ValueError("Missing or duplicated training metrics")
    if not all(math.isfinite(row[key]) for row in rows for key in ("loss", "max_loss", "grad_norm")):
        raise ValueError("Nonfinite metrics")
    return evidence


def validate_reference(root):
    root = Path(root).resolve()
    evidence = read_evidence(root / "steps-500/rank-0.json", "turbo")
    fixture = root / "initial-model.pt"
    inputs = root / "steps-500/inputs"
    if sha256(fixture) != evidence["fixture_sha256"]:
        raise ValueError("Initial fixture checksum mismatch")
    manifest = inputs / "manifest.jsonl"
    if sha256(manifest) != evidence["input_manifest_sha256"]:
        raise ValueError("Input manifest checksum mismatch")
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    if len(rows) != evidence["input_microbatches"] or len(rows) < 500:
        raise ValueError("Incomplete recorded inputs")
    for index, row in enumerate(rows, 1):
        name = f"microbatch-{index:06d}.pt"
        if row["microbatch"] != index or row["file"] != name or sha256(inputs / name) != row["sha256"]:
            raise ValueError("Recorded input identity mismatch: " + name)
    if not (root / "steps-500/rank-0-config.txt").is_file():
        raise ValueError("Reference resolved config is required for source/config review")
    return evidence, fixture, inputs


def compare(reference, gpu):
    if gpu.get("replayed_microbatches") != reference["input_microbatches"]:
        raise ValueError("GPU did not consume exactly all recorded NPU microbatches")
    for key in ("topology", "remat", "mtp", "expected_steps", "sequence_length",
                "microbatch_tokens_per_dp_rank", "seed", "precision", "deterministic",
                "fixture_sha256", "input_manifest_sha256", "input_microbatches"):
        if reference[key] != gpu[key]:
            raise ValueError("GPU/NPU contract mismatch: " + key)
    result = {"steps": 500, "difference": "NPU - GPU", "metrics": {}}
    for key in ("loss", "grad_norm"):
        actual = [row[key] for row in reference["metrics"]]
        expected = [row[key] for row in gpu["metrics"]]
        delta = [a - b for a, b in zip(actual, expected, strict=True)]
        relative = [d / b if b else None for d, b in zip(delta, expected, strict=True)]
        result["metrics"][key] = {
            "first_gpu": expected[0], "first_npu": actual[0],
            "last_gpu": expected[-1], "last_npu": actual[-1],
            "mean_signed_error": statistics.mean(delta),
            "mean_absolute_error": statistics.mean(map(abs, delta)),
            "max_absolute_error": max(map(abs, delta)),
            "worst_step": max(range(500), key=lambda i: abs(delta[i])) + 1,
            "mean_signed_relative_error": statistics.mean(relative) if None not in relative else None,
            "mean_absolute_relative_error": statistics.mean(map(abs, relative)) if None not in relative else None,
        }
    result["verdict"] = "observation_only_single_repeat"
    result["scope"] = "Same checkpoint and recorded inputs; inspect both resolved configs for source/API drift. Not a multi-repeat formal acceptance verdict."
    return result


def validate_resolved_configs(reference_path, gpu_path):
    reference = Path(reference_path).read_text(encoding="utf-8")
    gpu = Path(gpu_path).read_text(encoding="utf-8")
    # The backend attention config is intentionally different. Check training
    # state and numerical knobs separately, without executing repr text.
    for start, end in (("optim=", ", training="), ("training=", ", garbage_collector="),
                       ("activation_remat=", ", profiler="), ("loss=", ", dist_moe_runtime="),
                       ("precision=", ", hf_assets_path=")):
        def section(text):
            if start not in text or end not in text.split(start, 1)[1]:
                raise ValueError("Missing reference config section: " + start)
            return text.split(start, 1)[1].split(end, 1)[0]
        if section(reference) != section(gpu):
            raise ValueError("Resolved config differs: " + start)
    pattern = r"\b(eps|rope_theta|theta|route_scale|route_norm_epsilon|index_topk|index_sources)=([^,)]+)"
    if sorted(re.findall(pattern, reference)) != sorted(re.findall(pattern, gpu)):
        raise ValueError("Model normalization/routing/position config differs")
    sources = r"\bindex_sources=(\([^)]*\))"
    if not re.findall(sources, reference) or re.findall(sources, reference) != re.findall(sources, gpu):
        raise ValueError("Shared indexer source plan differs or is missing")


def run(args):
    evidence, fixture, inputs = validate_reference(args.reference)
    root = args.output
    (root / "experiment.json").write_text(json.dumps({
        "command": sys.argv, "reference": str(args.reference.resolve()),
        "titan": str(args.titan), "device": args.device,
        "fixture_sha256": evidence["fixture_sha256"],
        "input_manifest_sha256": evidence["input_manifest_sha256"],
        "titan_commit": subprocess.check_output(["git", "-C", str(args.titan), "rev-parse", "HEAD"], text=True).strip(),
        "test_commit": subprocess.check_output(["git", "-C", str(Path(__file__).resolve().parents[2]), "rev-parse", "HEAD"], text=True).strip(),
    }, indent=2), encoding="utf-8")
    (root / "README.md").write_text(
        "# Single-card CUDA/NPU 500-step comparison\n\n"
        "Uses the recorded NPU checkpoint and actual CPU microbatches.\n"
        "See comparison.log, experiment.json, steps-*/runtime.log and comparison.json.\n"
        "A single repeat produces observations, not formal multi-repeat acceptance.\n",
        encoding="utf-8")
    print("Reference validated: 500 finite steps; all fixture/input hashes verified")
    environment = {k: v for k, v in os.environ.items() if not k.startswith("GLM_")}
    environment.pop("ASCEND_RT_VISIBLE_DEVICES", None)
    environment.update({
        "CUDA_VISIBLE_DEVICES": args.device, "GLM_RUNTIME": "cuda",
        "GLM_TOPOLOGY": "single", "GLM_REMAT": evidence["remat"],
        "GLM_PRECISION": "bfloat16", "GLM_DETERMINISTIC": "1",
        "GLM_SEQ_LEN": str(evidence["sequence_length"]),
        "GLM_MICROBATCH_TOKENS": str(evidence["microbatch_tokens_per_dp_rank"]),
        "GLM_SEED": str(evidence["seed"]), "GLM_AB_FIXTURE": str(fixture),
        "GLM_INPUT_REPLAY": str(inputs), "GLM_LOCAL_COMPILE_REGIONS": "",
        "OMP_NUM_THREADS": "4", "TORCHTITAN_CONFIG_FIRST_CLI": "1",
    })
    for steps in (2, 500):
        destination = root / f"steps-{steps}"
        destination.mkdir()
        environment.update(GLM_STEPS=str(steps), GLM_OUTPUT=str(destination))
        environment["GLM_REFERENCE_CONFIG"] = str(args.reference.resolve() / "steps-500/rank-0-config.txt")
        command = [sys.executable, "-m", "torch.distributed.run", "--standalone",
                   "--nproc-per-node=1", str(Path(__file__).with_name("worker.py"))]
        print("command:", subprocess.list2cmdline(command), "steps:", steps, flush=True)
        with (destination / "runtime.log").open("w", encoding="utf-8") as log:
            completed = run_managed_process(command, cwd=args.titan, env=environment,
                                            stdout=log, stderr=subprocess.STDOUT)
        print("training_exit_code:", completed.returncode, flush=True)
        if completed.returncode:
            raise RuntimeError(f"Training failed; see {destination / 'runtime.log'}")
    gpu = read_evidence(root / "steps-500/rank-0.json", "cuda")
    validate_resolved_configs(args.reference / "steps-500/rank-0-config.txt",
                              root / "steps-500/rank-0-config.txt")
    result = compare(evidence, gpu)
    (root / "comparison.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (root / "summary.log").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("===== 500 STEP COMPARISON SUMMARY =====")
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True,
                        help="Transferred NPU capture root with initial-model.pt and steps-500")
    parser.add_argument("--output", type=Path, required=True, help="Fresh output directory")
    parser.add_argument("--titan", type=Path, default=Path(__file__).resolve().parents[3] / "torchtitan")
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    if not args.device.isdecimal():
        parser.error("--device must select one physical GPU index")
    args.output = args.output.resolve()
    args.titan = args.titan.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    print_output_path("Comparison log", args.output / "comparison.log")
    with (args.output / "comparison.log").open("w", encoding="utf-8", buffering=1) as log:
        with redirect_stdout(log), redirect_stderr(log):
            try:
                run(args)
            except BaseException:
                import traceback
                traceback.print_exc()
                print("exit_code=1")
                raise
            else:
                print("exit_code=0")


if __name__ == "__main__":
    main()
