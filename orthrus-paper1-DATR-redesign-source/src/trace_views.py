"""TRACE multi-view evidence and trusted-memory primitives.

TRACE deliberately separates anomaly evidence from epistemic uncertainty.
Uncertainty routes an event to review/quarantine; it is not treated as proof of
malice.  This avoids the common failure mode where benign cold-start entities
become high-confidence alerts merely because the model has not seen them.
"""

from collections import defaultdict, deque

import torch
import torch.nn as nn
import torch.nn.functional as F


class RunningMoments(nn.Module):
    """Streaming population moments stored in a model checkpoint."""

    def __init__(self, min_scale=1e-3):
        super().__init__()
        self.min_scale = float(min_scale)
        self.register_buffer("count", torch.zeros((), dtype=torch.long))
        self.register_buffer("mean", torch.zeros(()))
        self.register_buffer("m2", torch.zeros(()))

    @torch.no_grad()
    def reset(self):
        self.count.zero_()
        self.mean.zero_()
        self.m2.zero_()

    @torch.no_grad()
    def update(self, values):
        values = values.detach().reshape(-1).float()
        if values.numel() == 0:
            return
        batch_count = values.numel()
        batch_mean = values.mean()
        batch_m2 = ((values - batch_mean) ** 2).sum()
        if self.count.item() == 0:
            self.count.fill_(batch_count)
            self.mean.copy_(batch_mean)
            self.m2.copy_(batch_m2)
            return
        total = self.count + batch_count
        delta = batch_mean - self.mean
        self.mean.add_(delta * (batch_count / total.float()))
        self.m2.add_(batch_m2 + delta.square() * self.count.float() * batch_count / total.float())
        self.count.copy_(total)

    def scale(self):
        if self.count.item() < 2:
            return self.mean.new_tensor(self.min_scale)
        return torch.sqrt(self.m2 / self.count.float()).clamp_min(self.min_scale)

    def positive_z(self, values):
        return ((values - self.mean) / self.scale()).clamp_min(0.0)


class EntityCrossViewDecoder(nn.Module):
    """Cross-predict endpoint features from the opposite temporal state."""

    def __init__(self, hidden_dim, feature_dim):
        super().__init__()
        self.src_to_dst = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, feature_dim)
        )
        self.dst_to_src = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, feature_dim)
        )

    def forward(self, h_src, h_dst, x_src, x_dst):
        dst_error = F.smooth_l1_loss(self.src_to_dst(h_src), x_dst, reduction="none").mean(dim=1)
        src_error = F.smooth_l1_loss(self.dst_to_src(h_dst), x_src, reduction="none").mean(dim=1)
        return 0.5 * (src_error + dst_error)


class TemporalGapDecoder(nn.Module):
    """Predict log inter-event time from the two endpoint states."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, h_src, h_dst, log_gap, valid):
        prediction = self.network(torch.cat([h_src, h_dst], dim=1)).squeeze(1)
        error = F.smooth_l1_loss(prediction, log_gap, reduction="none")
        return torch.where(valid, error, torch.zeros_like(error))


def two_of_three_consensus(interaction, entity, temporal):
    """Return the second-largest positive evidence across three views.

    One noisy view cannot create an alert; at least two views must agree.  The
    operation is differentiable almost everywhere but is used only for scoring.
    """
    views = torch.stack([interaction, entity, temporal], dim=0)
    return torch.topk(views, k=2, dim=0, largest=True, sorted=True).values[1]


class TrustedDualMemory:
    """Conservative trusted/quarantine routing for future online adaptation.

    This class never updates a detector.  It only promotes an identity after a
    configurable run of low-evidence, low-uncertainty observations.  High-risk
    observations remain quarantined so a deployment cannot silently train on
    possible attacks.
    """

    def __init__(self, confirmations=5, quarantine_size=10000):
        if confirmations < 1:
            raise ValueError("confirmations must be positive")
        self.confirmations = int(confirmations)
        self.streaks = defaultdict(int)
        self.trusted = set()
        self.quarantine = deque(maxlen=int(quarantine_size))

    def observe(self, identity, evidence, uncertainty, evidence_limit, uncertainty_limit):
        safe = evidence <= evidence_limit and uncertainty <= uncertainty_limit
        if safe:
            self.streaks[identity] += 1
            if self.streaks[identity] >= self.confirmations:
                self.trusted.add(identity)
                return "trusted"
            return "pending"
        self.streaks[identity] = 0
        self.trusted.discard(identity)
        self.quarantine.append((identity, float(evidence), float(uncertainty)))
        return "quarantine"
