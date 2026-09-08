import unittest

from tests.glm5_2_common.topology import (
    DISTRIBUTED_TOPOLOGY_NAMES,
    select_topologies,
    standard_topologies,
    training_command_args,
)
from tests.glm5_2_performance.config import performance_topologies


class HsdpTopologyTest(unittest.TestCase):
    def test_degrees_and_training_budget(self):
        for name, replicate, shard in (("hsdp2x4", 2, 4), ("hsdp4x2", 4, 2)):
            with self.subTest(name=name):
                topology = standard_topologies()[name]
                self.assertEqual(topology.world_size, 8)
                self.assertEqual(topology.data_parallel_degree, 8)
                self.assertEqual(topology.dp_replicate, replicate)
                self.assertEqual(topology.dp_shard, shard)
                self.assertEqual((topology.tp, topology.cp, topology.pp, topology.ep), (1, 1, 1, 1))
                self.assertIn(f"--parallelism.data_parallel_replicate_degree={replicate}", topology.command_args())
                self.assertIn(f"--parallelism.data_parallel_shard_degree={shard}", topology.command_args())
                args = training_command_args(local_batch_size=8, global_batch_size=64, sequence_length=128, topology=topology)
                self.assertIn("--training.num_tokens_per_microbatch_per_dp_rank=1024", args)
                self.assertIn("--training.num_tokens_per_train_step=8192", args)

    def test_shared_selection(self):
        available = tuple(name for name, topo in standard_topologies().items() if topo.world_size <= 8)
        selected = select_topologies(available=available, topology="all")
        for name in ("hsdp2x4", "hsdp4x2"):
            self.assertIn(name, selected)
            self.assertIn(name, DISTRIBUTED_TOPOLOGY_NAMES)
            self.assertEqual(performance_topologies()[name], standard_topologies()[name])
        self.assertEqual(select_topologies(available=available, topologies="hsdp2x4,hsdp4x2"), ("hsdp2x4", "hsdp4x2"))


if __name__ == "__main__":
    unittest.main()
