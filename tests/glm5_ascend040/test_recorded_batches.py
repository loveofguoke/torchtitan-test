from dataclasses import fields

import pytest
import torch

from recorded_batches import record_batches, replay_batches
from torchtitan.components.data.types import TokenizedTrainingMicrobatch


def batch():
    return TokenizedTrainingMicrobatch(
        input=torch.arange(4),
        labels=torch.arange(1, 5),
        positions=torch.arange(4),
        padding_mask=torch.tensor([False, False, False, True]),
        loss_token_counts=torch.tensor(3),
        routing_token_counts=torch.tensor([3]),
    )


def test_exact_replay_and_refuse_overwrite(tmp_path):
    original = batch()
    directory = tmp_path / "inputs"
    assert list(record_batches(iter([original]), directory))[0] is original
    replayed = list(replay_batches(directory))[0]
    for field in fields(original):
        before, after = getattr(original, field.name), getattr(replayed, field.name)
        if isinstance(before, torch.Tensor):
            torch.testing.assert_close(before, after, rtol=0, atol=0)
        else:
            assert before == after
    with pytest.raises(FileExistsError):
        list(record_batches(iter([original]), directory))


def test_corrupt_input_rejected_before_loading(tmp_path):
    directory = tmp_path / "inputs"
    list(record_batches(iter([batch()]), directory))
    (directory / "microbatch-000001.pt").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum mismatch"):
        list(replay_batches(directory))
