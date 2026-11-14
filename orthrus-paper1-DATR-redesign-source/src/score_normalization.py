"""Label-free conditional normalization for reconstruction losses."""

from collections import defaultdict

import numpy as np


class RobustConditionalNormalizer:
    """Median/IQR normalizer fitted only on benign validation telemetry.

    Edge types have intrinsically different reconstruction difficulty. A global
    raw-loss scale therefore confounds semantic rarity with anomalousness. This
    class removes that nuisance factor without reading attack/test labels.
    """

    def __init__(self, min_scale_ratio=0.10, eps=1e-6):
        self.min_scale_ratio = float(min_scale_ratio)
        self.eps = float(eps)
        self.global_location = None
        self.global_scale = None
        self.by_group = {}

    @staticmethod
    def _location_scale(values):
        values = np.asarray(values, dtype=float)
        location = float(np.median(values))
        q25, q75 = np.percentile(values, [25, 75])
        return location, float(q75 - q25)

    def fit(self, losses, groups):
        losses = np.asarray(losses, dtype=float)
        groups = np.asarray(groups)
        if losses.size == 0 or losses.size != groups.size:
            raise ValueError("losses and groups must be non-empty and aligned")
        self.global_location, raw_global_scale = self._location_scale(losses)
        self.global_scale = max(raw_global_scale, self.eps)
        floor = max(self.global_scale * self.min_scale_ratio, self.eps)
        grouped = defaultdict(list)
        for loss, group in zip(losses, groups):
            grouped[str(group)].append(float(loss))
        for group, values in grouped.items():
            location, scale = self._location_scale(values)
            self.by_group[group] = (location, max(scale, floor))
        return self

    def transform_one(self, loss, group):
        if self.global_location is None:
            raise RuntimeError("normalizer must be fitted before transform")
        location, scale = self.by_group.get(
            str(group), (self.global_location, self.global_scale)
        )
        return (float(loss) - location) / scale

    def transform(self, losses, groups):
        return np.asarray([
            self.transform_one(loss, group)
            for loss, group in zip(losses, groups)
        ])
