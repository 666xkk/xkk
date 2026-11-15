import os
import sys
import types
from types import SimpleNamespace

import torch

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

# model.py imports these project-wide modules for historical wildcard helpers,
# but Orthrus itself does not use them. Keep this focused unit test independent
# of optional preprocessing packages such as NLTK and psycopg2.
sys.modules.setdefault("provnet_utils", types.ModuleType("provnet_utils"))
sys.modules.setdefault("config", types.ModuleType("config"))

from model import Orthrus
from objectives import DetectionAlignedObjective


class DummyEncoder(torch.nn.Module):
    def forward(self, edge_index, **kwargs):
        events = edge_index.size(1)
        values = torch.zeros(events, 2)
        return values, values


class FixedLossHead(torch.nn.Module):
    def __init__(self, values):
        super().__init__()
        self.register_buffer("values", torch.tensor(values, dtype=torch.float))

    def forward(self, **kwargs):
        return self.values


class FixedEntityView(torch.nn.Module):
    def forward(self, **kwargs):
        return torch.tensor([3.0, 3.0])


class FixedTemporalView(torch.nn.Module):
    def forward(self, **kwargs):
        return torch.tensor([3.0, 3.0])


def test_echo_model_fuses_independent_head_losses_at_inference():
    model = Orthrus(
        encoder=DummyEncoder(),
        decoders=[FixedLossHead([1.0, 2.0]), FixedLossHead([1.0, 4.0])],
        num_nodes=3,
        in_dim=2,
        out_dim=2,
        use_contrastive_learning=False,
        device=torch.device("cpu"),
        graph_reindexer=None,
        training_objective=DetectionAlignedObjective(enabled=False),
        uncertainty_enabled=True,
        bootstrap_keep_probability=1.0,
        disagreement_weight=0.5,
    )
    batch = SimpleNamespace(
        x_src=torch.zeros(2, 2),
        x_dst=torch.zeros(2, 2),
        edge_index=torch.tensor([[0, 1], [1, 2]]),
        t=torch.arange(2),
        msg=torch.zeros(2, 1),
        edge_type=torch.eye(2),
    )
    model.eval()
    score = model(batch, full_data=None, inference=True)
    # Event 0 has no disagreement. Event 1 has mean 3 and std 1.
    assert torch.allclose(score, torch.tensor([1.0, 3.5]))


def test_disabled_echo_preserves_single_head_score():
    model = Orthrus(
        encoder=DummyEncoder(),
        decoders=[FixedLossHead([1.0, 2.0])],
        num_nodes=3,
        in_dim=2,
        out_dim=2,
        use_contrastive_learning=False,
        device=torch.device("cpu"),
        graph_reindexer=None,
        training_objective=DetectionAlignedObjective(enabled=False),
        uncertainty_enabled=False,
    )
    batch = SimpleNamespace(
        x_src=torch.zeros(2, 2),
        x_dst=torch.zeros(2, 2),
        edge_index=torch.tensor([[0, 1], [1, 2]]),
        t=torch.arange(2),
        msg=torch.zeros(2, 1),
        edge_type=torch.eye(2),
    )
    model.eval()
    assert torch.allclose(
        model(batch, full_data=None, inference=True),
        torch.tensor([1.0, 2.0]),
    )
    assert not any(key.startswith("trace_") for key in model.state_dict())


def test_trace_requires_two_view_evidence_and_exports_components():
    model = Orthrus(
        encoder=DummyEncoder(),
        decoders=[FixedLossHead([3.0, 3.0]), FixedLossHead([3.0, 3.0])],
        num_nodes=3,
        in_dim=2,
        out_dim=2,
        use_contrastive_learning=False,
        device=torch.device("cpu"),
        graph_reindexer=None,
        training_objective=DetectionAlignedObjective(enabled=False),
        trace_enabled=True,
        trace_entity_decoder=FixedEntityView(),
        trace_temporal_decoder=FixedTemporalView(),
    )
    for moments in (
        model.trace_interaction_moments,
        model.trace_entity_moments,
        model.trace_temporal_moments,
        model.trace_uncertainty_moments,
    ):
        moments.count.fill_(2)
        moments.mean.zero_()
        moments.m2.fill_(2.0)  # population scale = 1
    batch = SimpleNamespace(
        x_src=torch.zeros(2, 2),
        x_dst=torch.zeros(2, 2),
        edge_index=torch.tensor([[0, 0], [1, 1]]),
        t=torch.tensor([0, 1_000_000_000]),
        msg=torch.zeros(2, 1),
        edge_type=torch.eye(2),
    )
    model.eval()
    score = model(batch, full_data=None, inference=True)
    assert torch.allclose(score, torch.tensor([3.0, 3.0]))
    assert model.last_trace_components["temporal_valid"].tolist() == [False, True]
    assert torch.allclose(
        model.last_trace_components["epistemic_uncertainty"], torch.zeros(2)
    )
