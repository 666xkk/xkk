import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

from open_set import (
    OpenSetExposureHead,
    RelationalPrototypeBank,
    type_preserving_roll,
)


def test_type_preserving_intervention_never_crosses_endpoint_types():
    values = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
    node_types = torch.tensor(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [1.0, 0.0],
        ]
    )
    intervened, valid = type_preserving_roll(values, node_types, 2)
    assert valid.all()
    source_type_by_value = {1: 0, 2: 0, 3: 1, 4: 1, 5: 0}
    for row, expected_type in zip(intervened.int().flatten().tolist(), node_types.argmax(1)):
        assert source_type_by_value[row] == int(expected_type)
    assert not torch.equal(intervened, values)


def test_singleton_endpoint_type_is_not_used_as_virtual_unknown():
    values = torch.tensor([[1.0], [2.0], [3.0]])
    node_types = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    intervened, valid = type_preserving_roll(values, node_types, 2)
    assert valid.tolist() == [True, True, False]
    assert intervened[2].item() == 3.0


def test_prototype_divergence_is_low_for_match_and_high_for_mismatch():
    bank = RelationalPrototypeBank(num_classes=2, momentum=0.0)
    probabilities = torch.tensor([[0.95, 0.05], [0.90, 0.10], [0.05, 0.95]])
    labels = torch.tensor([0, 0, 1])
    bank.update(probabilities, labels)
    match = bank.minimum_symmetric_kl(torch.tensor([[0.925, 0.075]]))
    mismatch = bank.minimum_symmetric_kl(torch.tensor([[0.50, 0.50]]))
    assert match.item() < mismatch.item()
    assert bank.counts.tolist() == [2, 1]
    bank.reset()
    assert bank.counts.tolist() == [0, 0]
    assert torch.count_nonzero(bank.prototypes).item() == 0


def test_open_head_has_one_extra_unknown_class_and_receives_gradients():
    head = OpenSetExposureHead(hidden_dim=4, num_known_classes=3)
    logits = head(torch.randn(5, 4), torch.randn(5, 4))
    assert logits.shape == (5, 4)
    torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1, 2, 3, 3])).backward()
    assert all(parameter.grad is not None for parameter in head.parameters())
