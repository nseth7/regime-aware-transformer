"""The Regime-Aware Transformer (RAT) — the paper's proposed model.

Conditions computation on a learned macro regime vector z at two points:
  1. A feature gate, applied before the transformer sees the sequence.
  2. FiLM (Perez et al., 2018) scale-and-shift, applied to the pooled
     sequence representation after the transformer.

Both conditioning points break cross-stock symmetry (unlike naive
concatenation — see baselines.py), which is what lets z receive gradient
under a rank-based training loss. See the paper, Sections 3.2-3.4.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from rat.models.layers import PositionalEncoding, RegimeEncoder, RegimeFeatureGate, RevIN


class RegimeAwareTransformer(nn.Module):
    def __init__(
        self,
        stock_features: int = 17,
        macro_features: int = 12,
        d_model: int = 32,
        nhead: int = 4,
        num_layers: int = 1,
        z_dim: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.revin = RevIN(stock_features)
        self.feature_gate = RegimeFeatureGate(z_dim, stock_features)

        self.input_proj = nn.Linear(stock_features, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        self.input_norm = nn.LayerNorm(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # pre-norm — more stable at this depth
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.regime_encoder = RegimeEncoder(macro_features, z_dim)

        # FiLM: z -> [gamma; beta] applied to the pooled representation
        self.film = nn.Linear(z_dim, 2 * d_model)

        self.head = nn.Sequential(
            nn.Linear(d_model + z_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor, macro: torch.Tensor, return_diagnostics: bool = False):
        """
        x     : (B, T, stock_features)
        macro : (B, macro_features)
        """
        x = self.revin(x)
        z = self.regime_encoder(macro)

        gate = torch.sigmoid(self.feature_gate.gate(z))
        x = x * gate.unsqueeze(1)

        h = self.input_proj(x)
        h = self.pos_enc(h)
        h = self.input_norm(h)
        h = self.transformer(h)

        # Last-token pooling: for a 5-day-ahead target, the most recent
        # timestep carries most of the predictive signal. Mean pooling
        # would dilute it by a factor of T.
        h_last = h[:, -1, :]

        gamma, beta = self.film(z).chunk(2, dim=-1)
        h_cond = h_last * (1 + gamma) + beta

        combined = torch.cat([h_cond, z], dim=-1)
        pred = self.head(combined).squeeze(-1)

        if return_diagnostics:
            return pred, {"z": z, "gate": gate}
        return pred
