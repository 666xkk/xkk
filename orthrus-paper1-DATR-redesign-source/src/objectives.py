"""Detection-aligned training utilities.

The original ORTHRUS objective minimizes the *mean* edge-type cross entropy,
while the detector alarms on the upper tail of edge/node reconstruction loss.
This module makes that mismatch explicit and optional: rare event types are
balanced, the upper loss tail is optimized with a CVaR-style term, and an EMA
checkpoint provides a label-free, pre-specified final model.
"""

import copy
import math

import torch
import torch.nn as nn


def bootstrap_ensemble_risk(head_losses, keep_probability=0.8):
    """Return a bootstrap risk for a stack of per-event head losses.

    Each semantic head sees an independent Bernoulli sub-sample.  The temporal
    encoder remains shared, so this is substantially cheaper than training a
    full deep ensemble while preserving the head diversity needed for an
    epistemic signal.  At least one event is retained for every head.
    """
    if head_losses.ndim != 2 or head_losses.numel() == 0:
        raise ValueError("head_losses must have shape [num_heads, num_events]")
    if not 0.0 < keep_probability <= 1.0:
        raise ValueError("bootstrap_keep_probability must be in (0, 1]")

    risks = []
    for losses in head_losses:
        if keep_probability == 1.0:
            selected = losses
        else:
            mask = torch.rand_like(losses) < keep_probability
            if not mask.any():
                mask[torch.randint(losses.numel(), (1,), device=losses.device)] = True
            selected = losses[mask]
        risks.append(selected.mean())
    return torch.stack(risks).mean()


def uncertainty_fused_score(head_losses, disagreement_weight=0.5):
    """Fuse reconstruction surprisal and epistemic head disagreement.

    The first term is the ordinary ORTHRUS edge reconstruction loss.  The
    second is the population standard deviation across independently
    bootstrapped heads.  Setting ``disagreement_weight=0`` recovers the mean
    ensemble reconstruction score exactly.
    """
    if head_losses.ndim != 2 or head_losses.numel() == 0:
        raise ValueError("head_losses must have shape [num_heads, num_events]")
    if disagreement_weight < 0.0:
        raise ValueError("disagreement_weight must be non-negative")
    mean = head_losses.mean(dim=0)
    disagreement = head_losses.std(dim=0, unbiased=False)
    return mean + float(disagreement_weight) * disagreement


def effective_number_weights(counts, beta=0.999, max_weight=10.0):
    """Return normalized class weights using the effective-number estimator."""
    counts = torch.as_tensor(counts, dtype=torch.float)
    if counts.ndim != 1 or counts.numel() == 0:
        raise ValueError("counts must be a non-empty one-dimensional tensor")
    if not 0.0 <= beta < 1.0:
        raise ValueError("class_balance_beta must be in [0, 1)")

    present = counts > 0
    weights = torch.ones_like(counts)
    if beta > 0 and present.any():
        effective = 1.0 - torch.pow(torch.full_like(counts[present], beta), counts[present])
        weights[present] = (1.0 - beta) / effective.clamp_min(1e-12)
        weights[present] /= weights[present].mean().clamp_min(1e-12)
    weights = weights.clamp(max=float(max_weight))
    return weights


class DetectionAlignedObjective(nn.Module):
    """Blend ordinary risk with upper-tail risk on benign training events."""

    def __init__(self, enabled=False, tail_fraction=0.10, tail_weight=0.50):
        super().__init__()
        if not 0.0 < tail_fraction <= 1.0:
            raise ValueError("tail_fraction must be in (0, 1]")
        if not 0.0 <= tail_weight <= 1.0:
            raise ValueError("tail_weight must be in [0, 1]")
        self.enabled = bool(enabled)
        self.tail_fraction = float(tail_fraction)
        self.tail_weight = float(tail_weight)
        # Training-only statistics. Keeping this non-persistent avoids a shape
        # mismatch when an inference model is constructed before loading a
        # checkpoint fitted with a different number of edge types.
        self.register_buffer("class_weights", torch.ones(1), persistent=False)
        self.last_stats = {}

    def set_class_weights(self, weights):
        weights = torch.as_tensor(weights, dtype=torch.float, device=self.class_weights.device)
        self.class_weights = weights.detach().clone()

    def forward(self, per_event_loss, edge_type):
        if per_event_loss.ndim != 1:
            per_event_loss = per_event_loss.reshape(-1)
        if per_event_loss.numel() == 0:
            raise ValueError("cannot reduce an empty event-loss tensor")

        base_mean = per_event_loss.mean()
        if not self.enabled:
            self.last_stats = {
                "mean": float(base_mean.detach()),
                "tail": float(base_mean.detach()),
            }
            return base_mean

        labels = edge_type.argmax(dim=1) if edge_type.ndim > 1 else edge_type.long()
        if self.class_weights.numel() > 1:
            weights = self.class_weights.to(per_event_loss.device)[labels]
            # Keep the objective scale stable across differently composed batches.
            weighted = per_event_loss * weights / weights.mean().clamp_min(1e-12)
        else:
            weighted = per_event_loss

        mean_risk = weighted.mean()
        k = max(1, int(math.ceil(self.tail_fraction * weighted.numel())))
        tail_risk = torch.topk(weighted, k=k, largest=True, sorted=False).values.mean()
        objective = (1.0 - self.tail_weight) * mean_risk + self.tail_weight * tail_risk
        self.last_stats = {
            "mean": float(mean_risk.detach()),
            "tail": float(tail_risk.detach()),
        }
        return objective


class ModelEMA:
    """Exponential moving average of trainable model state.

    Non-floating buffers (for example counters) are copied directly. The class
    can temporarily install the averaged state for checkpointing without
    altering subsequent training.
    """

    def __init__(self, model, decay=0.995):
        if not 0.0 <= decay < 1.0:
            raise ValueError("ema_decay must be in [0, 1)")
        self.decay = float(decay)
        self.shadow = {
            name: value.detach().clone()
            for name, value in model.state_dict().items()
        }
        self.backup = None
        self.num_updates = 0

    @torch.no_grad()
    def update(self, model):
        state = model.state_dict()
        if self.num_updates == 0:
            # Start from the first eligible trained state rather than from the
            # random initialization; otherwise a high decay produces a large
            # and unnecessary initialization bias.
            for name, value in state.items():
                self.shadow[name].copy_(value.detach())
            self.num_updates = 1
            return
        self.num_updates += 1
        for name, value in state.items():
            if value.is_floating_point():
                self.shadow[name].mul_(self.decay).add_(value.detach(), alpha=1.0 - self.decay)
            else:
                self.shadow[name].copy_(value.detach())

    def apply(self, model):
        if self.backup is not None:
            raise RuntimeError("EMA state is already applied")
        self.backup = copy.deepcopy(model.state_dict())
        model.load_state_dict(self.shadow, strict=True)

    def restore(self, model):
        if self.backup is None:
            raise RuntimeError("EMA state is not applied")
        model.load_state_dict(self.backup, strict=True)
        self.backup = None
