"""Training objective: negative Pearson correlation as a differentiable
proxy for the Information Coefficient (IC), the standard cross-sectional
ranking metric in quantitative equity research.

MSE/SmoothL1 optimize for absolute return magnitude; IC loss optimizes for
rank ordering within a batch, which is what a long-short strategy actually
needs. See the paper, Section 3.3.
"""

from __future__ import annotations

import torch


def ic_loss(pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Negative Pearson correlation, computed per-batch.

    Batches are shuffled across dates during training, so this acts as a
    cross-sample ranking objective over mixed-date predictions; targets
    are pre-normalized cross-sectionally within each date (see
    data/pipeline.py::make_cross_sectional_targets), which preserves the
    ranking signal even when a batch spans multiple dates.
    """
    pred = pred - pred.mean()
    y = y - y.mean()
    num = (pred * y).sum()
    den = torch.sqrt((pred ** 2).sum() * (y ** 2).sum() + 1e-8)
    return -num / den
