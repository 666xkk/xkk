import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

from objectives import DetectionAlignedObjective, ModelEMA, effective_number_weights
from score_normalization import RobustConditionalNormalizer


def test_disabled_objective_is_original_mean():
    losses = torch.tensor([1.0, 2.0, 6.0])
    labels = torch.eye(3)
    objective = DetectionAlignedObjective(enabled=False)
    assert torch.allclose(objective(losses, labels), losses.mean())


def test_tail_risk_emphasizes_hard_benign_events():
    losses = torch.tensor([1.0, 1.0, 1.0, 9.0])
    labels = torch.tensor([[1.0, 0.0]] * 4)
    objective = DetectionAlignedObjective(
        enabled=True, tail_fraction=0.25, tail_weight=1.0
    )
    assert torch.allclose(objective(losses, labels), torch.tensor(9.0))


def test_effective_number_upweights_rare_present_class():
    weights = effective_number_weights([1000, 10], beta=0.999)
    assert weights[1] > weights[0]


def test_ema_can_be_applied_and_restored():
    model = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(0.0)
    ema = ModelEMA(model, decay=0.5)
    with torch.no_grad():
        model.weight.fill_(2.0)
    ema.update(model)
    with torch.no_grad():
        model.weight.fill_(4.0)
    ema.update(model)
    ema.apply(model)
    assert torch.allclose(model.weight, torch.full_like(model.weight, 3.0))
    ema.restore(model)
    assert torch.allclose(model.weight, torch.full_like(model.weight, 4.0))


def test_conditional_normalization_removes_group_scale():
    normalizer = RobustConditionalNormalizer().fit(
        [1.0, 2.0, 100.0, 200.0], ["common", "common", "rare", "rare"]
    )
    transformed = normalizer.transform([1.5, 150.0], ["common", "rare"])
    assert abs(transformed[0]) < 1e-6
    assert abs(transformed[1]) < 1e-6
