"""Run the Ascend GLM recipe on a real process group and persist rank evidence."""

import json
import math
import os
from pathlib import Path

import torch
import torch.distributed as dist

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
    from torchtitan_ascend import setup
    from torchtitan_ascend.recipes.glm5 import (
        glm5_2_debugmodel_npu,
        glm5_3_debugmodel_mtp_npu,
    )

    setup()
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
    factory = glm5_3_debugmodel_mtp_npu if use_mtp else glm5_2_debugmodel_npu
    config = factory(seq_len=32, parallelism=parallelism)
    if policy == "full":
        config.activation_remat = FullActivationRematConfig()
    elif policy != "selective":
        raise ValueError("This validation requires selective or full remat")
    assert config.activation_remat is not None
    root = Path(os.environ["GLM_OUTPUT"])
    root.mkdir(parents=True, exist_ok=True)
    config.dump_folder = str(root / "trainer")
    config.training.steps = 2
    config.training.num_tokens_per_microbatch_per_dp_rank = 256
    config.debug.seed = 42
    rank = int(os.environ["RANK"])
    evidence = {
        "rank": rank,
        "world_size": int(os.environ["WORLD_SIZE"]),
        "topology": vars(topology),
        "remat": policy,
        "mtp": use_mtp,
        "metrics": [],
        "status": "running",
    }
    trainer = None
    try:
        trainer = config.build()
        original = trainer.metrics_processor.log

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
            return original(
                step, global_avg_loss, global_max_loss, grad_norm, *args, **kwargs
            )

        trainer.metrics_processor.log = capture
        trainer.train()
        torch.npu.synchronize()
        evidence["completed_steps"] = trainer.engine.num_completed_steps
        assert evidence["completed_steps"] == 2
        assert [row["step"] for row in evidence["metrics"]] == [1, 2]
        evidence["status"] = "passed"
    except BaseException as exc:
        evidence["status"] = "failed"
        evidence["error"] = repr(exc)
        raise
    finally:
        (root / f"rank-{rank}.json").write_text(
            json.dumps(evidence, indent=2), encoding="utf-8"
        )
        if trainer is not None:
            trainer.close()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
