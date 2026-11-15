import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

from trace_views import RunningMoments, TrustedDualMemory, two_of_three_consensus


def test_two_of_three_consensus_rejects_one_noisy_view():
    interaction = torch.tensor([10.0, 3.0])
    entity = torch.tensor([0.0, 4.0])
    temporal = torch.tensor([0.0, 1.0])
    score = two_of_three_consensus(interaction, entity, temporal)
    assert torch.allclose(score, torch.tensor([0.0, 3.0]))


def test_running_moments_matches_population_statistics():
    moments = RunningMoments(min_scale=1e-6)
    moments.update(torch.tensor([1.0, 2.0]))
    moments.update(torch.tensor([3.0, 4.0]))
    assert moments.count.item() == 4
    assert torch.allclose(moments.mean, torch.tensor(2.5))
    assert torch.allclose(moments.scale(), torch.tensor(1.118034), atol=1e-5)


def test_dual_memory_requires_repeated_safe_evidence_and_revokes_on_risk():
    memory = TrustedDualMemory(confirmations=2)
    assert memory.observe("node", 0.1, 0.1, 1.0, 1.0) == "pending"
    assert memory.observe("node", 0.1, 0.1, 1.0, 1.0) == "trusted"
    assert "node" in memory.trusted
    assert memory.observe("node", 2.0, 0.1, 1.0, 1.0) == "quarantine"
    assert "node" not in memory.trusted
