"""Attack-label-free open-set exposure for temporal provenance events.

The module does not synthesize an APT story.  It creates conservative,
type-preserving counterfactual events by pairing an endpoint state with the
same endpoint type from a different event in the current causal batch.  These
counterfactuals teach a dedicated unknown class while a prototype bank records
the *distribution* of normal inter-class predictions.  At inference the two
signals remain separate so uncertainty is never equated with maliciousness.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def type_preserving_roll(values, type_features, num_node_types):
    """Roll values only inside endpoint-type groups containing >=2 events.

    Returns the intervened tensor and a mask identifying events for which the
    counterfactual is genuinely different.  Singleton types are left unchanged
    and excluded from the virtual-unknown loss.
    """
    if values.shape[0] != type_features.shape[0]:
        raise ValueError("values and type_features must have the same batch size")
    if num_node_types < 1 or type_features.shape[1] < num_node_types:
        raise ValueError("node type features are missing or num_node_types is invalid")
    type_ids = type_features[:, :num_node_types].argmax(dim=1)
    result = values.clone()
    valid = torch.zeros(values.shape[0], dtype=torch.bool, device=values.device)
    for type_id in torch.unique(type_ids):
        indices = torch.nonzero(type_ids == type_id, as_tuple=False).flatten()
        if indices.numel() < 2:
            continue
        result[indices] = values[indices.roll(1)]
        valid[indices] = True
    return result, valid


class OpenSetExposureHead(nn.Module):
    """Known-relation classifier with one explicit virtual-unknown class."""

    def __init__(self, hidden_dim, num_known_classes):
        super().__init__()
        self.num_known_classes = int(num_known_classes)
        self.network = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, self.num_known_classes + 1),
        )

    def forward(self, h_src, h_dst):
        return self.network(torch.cat([h_src, h_dst], dim=1))


class RelationalPrototypeBank(nn.Module):
    """EMA prototypes of full known-class prediction distributions."""

    def __init__(self, num_classes, momentum=0.95, eps=1e-6):
        super().__init__()
        if not 0.0 <= momentum < 1.0:
            raise ValueError("prototype momentum must be in [0, 1)")
        self.num_classes = int(num_classes)
        self.momentum = float(momentum)
        self.eps = float(eps)
        self.register_buffer("prototypes", torch.zeros(num_classes, num_classes))
        self.register_buffer("counts", torch.zeros(num_classes, dtype=torch.long))

    @torch.no_grad()
    def reset(self):
        self.prototypes.zero_()
        self.counts.zero_()

    @torch.no_grad()
    def update(self, probabilities, labels):
        probabilities = probabilities.detach()
        labels = labels.detach().long()
        for class_id in torch.unique(labels):
            class_id_int = int(class_id)
            if class_id_int < 0 or class_id_int >= self.num_classes:
                continue
            class_mean = probabilities[labels == class_id].mean(dim=0)
            class_mean = class_mean / class_mean.sum().clamp_min(self.eps)
            if self.counts[class_id_int].item() == 0:
                self.prototypes[class_id_int].copy_(class_mean)
            else:
                self.prototypes[class_id_int].mul_(self.momentum).add_(
                    class_mean, alpha=1.0 - self.momentum
                )
            self.counts[class_id_int].add_(int((labels == class_id).sum()))

    def minimum_symmetric_kl(self, probabilities):
        """Distance to the closest valid normal relation prototype."""
        valid = self.counts > 0
        if not valid.any():
            return probabilities.new_zeros(probabilities.shape[0])
        p = probabilities.clamp_min(self.eps)
        p = p / p.sum(dim=1, keepdim=True)
        q = self.prototypes[valid].clamp_min(self.eps)
        q = q / q.sum(dim=1, keepdim=True)
        p_expanded = p[:, None, :]
        q_expanded = q[None, :, :]
        kl_pq = (p_expanded * (p_expanded.log() - q_expanded.log())).sum(dim=2)
        kl_qp = (q_expanded * (q_expanded.log() - p_expanded.log())).sum(dim=2)
        return (0.5 * (kl_pq + kl_qp)).min(dim=1).values
