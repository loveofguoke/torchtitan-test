"""Run the Ascend GLM recipe on a real process group and persist rank evidence."""

import hashlib
import json
import math
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

from tests.glm5_2_common.topology import ParallelTopology, standard_topologies


def topologies():
    return {
        **standard_topologies(),
        "fsdp2": ParallelTopology("fsdp2", 2, data_parallel_shard_degree=2),
        "tp2-ep2": ParallelTopology(
            "tp2-ep2", 2, tensor_parallel_degree=2, expert_parallel_degree=2
        ),
        "fsdp2-ep2": ParallelTopology(
            "fsdp2-ep2", 2, data_parallel_shard_degree=2, expert_parallel_degree=2
        ),
        "tp2-cp2-ep2": ParallelTopology(
            "tp2-cp2-ep2",
            4,
            tensor_parallel_degree=2,
            context_parallel_degree=2,
            expert_parallel_degree=2,
        ),
        "fsdp2-tp2-cp2-ep2": ParallelTopology(
            "fsdp2-tp2-cp2-ep2",
            8,
            data_parallel_shard_degree=2,
            tensor_parallel_degree=2,
            context_parallel_degree=2,
            expert_parallel_degree=2,
        ),
        "pp2": ParallelTopology("pp2", 2, pipeline_parallel_degree=2),
    }


def main():
    from torchtitan.config.parallelism import ParallelismConfig
    from torchtitan.distributed.activation_remat import FullActivationRematConfig

    runtime = os.environ.get("GLM_RUNTIME", "ascend")
    turbo_status = None
    if runtime == "turbo":
        import torchtitanturbo
        from torchtitan_recipes.tests.models.glm5 import (
            glm5_3_debugmodel_mtp as mtp_factory,
            glm5_debugmodel as base_factory,
        )
        from torchtitanturbo.patch import get_patch_status

        turbo_status = get_patch_status()
    elif runtime == "ascend":
        from torchtitan_ascend import setup
        from torchtitan_ascend.recipes.glm5 import (
            glm5_2_debugmodel_npu as base_factory,
            glm5_3_debugmodel_mtp_npu as mtp_factory,
        )

        setup()
    else:
        raise ValueError(f"Unknown GLM runtime: {runtime}")
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    topology = topologies()[os.environ["GLM_TOPOLOGY"]]
    policy = os.environ.get("GLM_REMAT", "selective")
    use_mtp = os.environ.get("GLM_MTP", "0") == "1"
    if use_mtp and topology.pp > 1:
        raise ValueError("PP+MTP is unsupported; do not silently drop MTP")
    parallelism = ParallelismConfig(
        data_parallel_replicate_degree=topology.dp_replicate,
        data_parallel_shard_degree=topology.dp_shard,
        tensor_parallel_degree=topology.tp,
        context_parallel_degree=topology.cp,
        pipeline_parallel_degree=topology.pp,
        expert_parallel_degree=topology.ep,
        enable_sequence_parallel=True,
    )
    if topology.pp > 1:
        parallelism.pipeline_parallel_schedule = "1F1B"
        parallelism.num_pp_microbatches = topology.pp
    factory = mtp_factory if use_mtp else base_factory
    seq_len = int(os.environ.get("GLM_SEQ_LEN", "32"))
    if seq_len <= 0:
        raise ValueError("GLM_SEQ_LEN must be positive")
    config = factory(seq_len=seq_len, parallelism=parallelism)
    npu_flex = os.environ.get("GLM_NPU_FLEX", "0")
    if npu_flex not in ("0", "1"):
        raise ValueError("GLM_NPU_FLEX must be 0 or 1")
    if policy == "full":
        config.activation_remat = FullActivationRematConfig()
    elif policy != "selective":
        raise ValueError("This validation requires selective or full remat")
    assert config.activation_remat is not None
    root = Path(os.environ["GLM_OUTPUT"])
    root.mkdir(parents=True, exist_ok=True)
    config.dump_folder = str(root / "trainer")
    steps = int(os.environ.get("GLM_STEPS", "2"))
    if steps < 2:
        raise ValueError("GLM_STEPS must be at least 2")
    config.training.steps = steps
    config.metrics.log_freq = 1
    microbatch_tokens = int(os.environ.get("GLM_MICROBATCH_TOKENS", "256"))
    if microbatch_tokens <= 0 or microbatch_tokens % seq_len:
        raise ValueError(
            "GLM_MICROBATCH_TOKENS must be a positive multiple of sequence length"
        )
    config.training.num_tokens_per_microbatch_per_dp_rank = microbatch_tokens
    config.debug.seed = int(os.environ.get("GLM_SEED", "42"))
    precision = os.environ.get("GLM_PRECISION", "bfloat16")
    if precision not in ("bfloat16", "float32"):
        raise ValueError("GLM_PRECISION must be bfloat16 or float32")
    config.precision.param_dtype = precision
    deterministic = os.environ.get("GLM_DETERMINISTIC", "0")
    if deterministic not in ("0", "1"):
        raise ValueError("GLM_DETERMINISTIC must be 0 or 1")
    config.debug.deterministic = deterministic == "1"
    compile_regions = os.environ.get("GLM_LOCAL_COMPILE_REGIONS")
    if compile_regions is not None:
        if runtime != "turbo":
            raise ValueError(
                "Local compile probes require the current TorchTitan runtime"
            )
        config.model.local_compile.regions = [
            name for name in compile_regions.split(",") if name
        ]
    if npu_flex == "1":
        if runtime != "turbo":
            raise ValueError("The explicit GLM Flex transform requires Turbo")
        from torchtitan.config.transform import apply_transforms
        from torchtitanturbo.models.glm5.attention import NpuGlm5FlexTransform

        config = apply_transforms(config, [NpuGlm5FlexTransform()])
    rank = int(os.environ["RANK"])
    evidence = {
        "rank": rank,
        "world_size": int(os.environ["WORLD_SIZE"]),
        "topology": vars(topology),
        "remat": policy,
        "mtp": use_mtp,
        "runtime": runtime,
        "npu_flex_transform": npu_flex == "1",
        "expected_steps": steps,
        "sequence_length": seq_len,
        "microbatch_tokens_per_dp_rank": microbatch_tokens,
        "seed": config.debug.seed,
        "hccl_npu_socket_port_range": os.environ.get("HCCL_NPU_SOCKET_PORT_RANGE"),
        "precision": {
            "dtype": config.precision.dtype,
            "param_dtype": config.precision.param_dtype,
            "reduce_dtype": config.precision.reduce_dtype,
        },
        "deterministic": config.debug.deterministic,
        "local_compile_regions": (
            list(config.model.local_compile.regions) if runtime == "turbo" else None
        ),
        "npu_codegen_loader": os.environ.get("TORCHINDUCTOR_NPU_BACKEND", "default"),
        "turbo_patches": turbo_status,
        "metrics": [],
        "status": "running",
    }
    import torch_npu

    (root / f"rank-{rank}-config.txt").write_text(repr(config), encoding="utf-8")

    npu_kernel = Path(torch_npu.__file__).parent / "_inductor/kernel/flex_attention.py"
    evidence["torch_npu"] = {
        "version": torch_npu.__version__,
        "path": torch_npu.__file__,
        "flex_kernel_sha256": hashlib.sha256(npu_kernel.read_bytes()).hexdigest(),
    }
    trainer = None
    try:
        trainer = config.build()
        fixture_name = os.environ.get("GLM_AB_FIXTURE")
        if fixture_name:
            from torch.distributed.checkpoint.state_dict import (
                get_model_state_dict,
                set_model_state_dict,
                StateDictOptions,
            )

            assert len(trainer.engine.model_parts) == 1
            model = trainer.engine.model_parts[0]
            fixture = Path(fixture_name)
            if not fixture.exists():
                if topology.world_size != 1:
                    raise ValueError("Create the shared fixture on one rank first")
                state = get_model_state_dict(
                    model,
                    options=StateDictOptions(full_state_dict=True, cpu_offload=True),
                )
                torch.save(state, fixture)
            state = torch.load(fixture, map_location="cpu", weights_only=True)
            incompatible = set_model_state_dict(
                model, state, options=StateDictOptions(full_state_dict=True)
            )
            assert not incompatible.missing_keys and not incompatible.unexpected_keys
            evidence["fixture"] = str(fixture)
            evidence["fixture_sha256"] = hashlib.sha256(
                fixture.read_bytes()
            ).hexdigest()
            torch.manual_seed(config.debug.seed)
        original = trainer.metrics_processor.log
        input_capture = os.environ.get("GLM_INPUT_CAPTURE")
        input_replay = os.environ.get("GLM_INPUT_REPLAY")
        if input_capture or input_replay:
            if input_capture and input_replay:
                raise ValueError("Choose either input capture or replay")
            if topology.world_size != 1:
                raise ValueError("This recorded-batch contract is single-rank only")
            from recorded_batches import record_batches, replay_batches

            original_batches = trainer.microbatch_generator
            if input_capture:
                trainer.microbatch_generator = lambda data: record_batches(
                    original_batches(data), input_capture
                )
            else:
                trainer.microbatch_generator = lambda data: original_batches(
                    replay_batches(input_replay)
                )

        def capture(step, global_avg_loss, global_max_loss, grad_norm, *args, **kwargs):
            values = [float(global_avg_loss), float(global_max_loss), float(grad_norm)]
            assert all(math.isfinite(value) for value in values), values
            evidence["metrics"].append(
                {
                    "step": int(step),
                    "loss": values[0],
                    "max_loss": values[1],
                    "grad_norm": values[2],
                }
            )
            evidence["metrics"][-1]["extra_metrics"] = {
                name: float(value)
                for name, value in (kwargs.get("extra_metrics") or {}).items()
                if isinstance(value, (int, float))
                or (isinstance(value, torch.Tensor) and value.numel() == 1)
            }
            with (root / f"rank-{rank}-metrics.jsonl").open(
                "a", encoding="utf-8"
            ) as stream:
                stream.write(json.dumps(evidence["metrics"][-1]) + "\n")
            return original(
                step, global_avg_loss, global_max_loss, grad_norm, *args, **kwargs
            )

        trainer.metrics_processor.log = capture
        trainer.train()
        torch.npu.synchronize()
        evidence["completed_steps"] = trainer.engine.num_completed_steps
        assert evidence["completed_steps"] == steps
        assert [row["step"] for row in evidence["metrics"]] == list(range(1, steps + 1))
        if use_mtp:
            norms = {}
            for part in trainer.engine.model_parts:
                for name, parameter in part.named_parameters():
                    if (
                        "mtp_layers" in name
                        and "eh_proj" in name
                        and parameter.grad is not None
                    ):
                        gradient = parameter.grad
                        if hasattr(gradient, "to_local"):
                            gradient = gradient.to_local()
                        norms[name] = float(gradient.float().norm())
            counts = torch.tensor(
                [
                    sum(value > 0 for value in norms.values()),
                    sum(not math.isfinite(value) for value in norms.values()),
                ],
                device=trainer.engine.device,
            )
            dist.all_reduce(counts)
            evidence["mtp_projection_local_grad_norms"] = norms
            evidence["mtp_nonzero_projection_gradient_shards"] = int(counts[0].item())
            assert counts[0].item() > 0 and counts[1].item() == 0, (
                f"Invalid MTP projection gradients: nonzero={counts[0].item()}, "
                f"nonfinite={counts[1].item()}"
            )
        if fixture_name:
            final_state = get_model_state_dict(
                model, options=StateDictOptions(full_state_dict=True, cpu_offload=True)
            )
            frozen_names = [name for name in state if ".indexer." in name]
            assert frozen_names
            changed = []
            for name in frozen_names:
                before, after = state[name], final_state[name]
                if isinstance(before, DTensor):
                    before = before.full_tensor()
                if isinstance(after, DTensor):
                    after = after.full_tensor()
                if not torch.equal(before.detach().cpu(), after.detach().cpu()):
                    changed.append(name)
            evidence["frozen_indexer_changed_parameters"] = changed
            assert not changed, changed
            evidence["frozen_indexer_parameters_unchanged"] = len(frozen_names)
        evidence["status"] = "passed"
    except BaseException as exc:
        evidence["status"] = "failed"
        evidence["error"] = repr(exc)
        raise
    finally:
        input_directory = os.environ.get("GLM_INPUT_CAPTURE") or os.environ.get(
            "GLM_INPUT_REPLAY"
        )
        if input_directory:
            manifest = Path(input_directory) / "manifest.jsonl"
            if manifest.exists():
                evidence["input_manifest"] = str(manifest)
                evidence["input_manifest_sha256"] = hashlib.sha256(
                    manifest.read_bytes()
                ).hexdigest()
                evidence["input_microbatches"] = len(manifest.read_text().splitlines())
        (root / f"rank-{rank}.json").write_text(
            json.dumps(evidence, indent=2), encoding="utf-8"
        )
        if trainer is not None:
            trainer.close()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
