"""Shared building blocks for all three transformer variants (RAT, macro-concat
baseline, and the no-macro baseline). See the paper (assets/paper.pdf),
Section 3.2, for the full architectural description.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class RevIN(nn.Module):
    """Reversible Instance Normalization (Kim et al., 2022).

    Normalizes each 60-day window by its own per-feature mean/std so the
    transformer sees inputs on a comparable scale regardless of a stock's
    price level or the current volatility regime. Includes a learnable
    affine transform to recover scale where useful.
    """

    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True) + self.eps
        x = (x - mean) / std
        return x * self.weight + self.bias


class RegimeEncoder(nn.Module):
    """Compresses the macro snapshot into a regime embedding z.

    12 -> 64 -> 64 -> 32, GELU activations, LayerNorm after the first
    linear layer. Learned end-to-end — no hand-labeled regimes.
    """

    def __init__(self, macro_dim: int = 12, z_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(macro_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Linear(64, z_dim),
        )

    def forward(self, macro: torch.Tensor) -> torch.Tensor:
        # macro: (B, macro_dim) -> z: (B, z_dim)
        return self.net(macro)


class RegimeFeatureGate(nn.Module):
    """Per-feature sigmoid gate conditioned on the regime vector z.

    Lets the regime embedding decide which technical features to amplify
    or suppress *before* the transformer processes the sequence. Because
    each stock has different feature values, the gated output differs
    across stocks even on the same date — this is what keeps gradient
    flowing back into the regime encoder under a rank-based loss (see
    training/losses.py and the paper, Section 3.4).
    """

    def __init__(self, z_dim: int, num_features: int):
        super().__init__()
        self.gate = nn.Linear(z_dim, num_features)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F), z: (B, z_dim)
        gate = torch.sigmoid(self.gate(z))       # (B, F)
        return x * gate.unsqueeze(1)              # broadcast over time


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding over the lookback window."""

    def __init__(self, d_model: int, max_len: int = 60):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]
