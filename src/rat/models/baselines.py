"""Comparison baselines used to isolate what regime conditioning buys you.

BaselineTransformer   — stock features only, no macro input at all.
MacroConcatTransformer — same macro information as RAT, but naively
                          concatenated to every timestep instead of used
                          as a conditioning signal.

Per the paper (Section 3.4): under a rank-based (IC) loss, concatenation
fails because the macro vector is identical across all stocks on a given
date, and rank is shift-invariant to a constant added to every prediction.
The concatenated context therefore receives zero gradient throughout
training — MacroConcatTransformer is included precisely to demonstrate
this failure mode empirically, not as a competitive baseline.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from rat.models.layers import PositionalEncoding, RegimeEncoder, RevIN


class BaselineTransformer(nn.Module):
    """Per-stock technical features only. No macro input."""

    def __init__(
        self,
        stock_features: int = 17,
        d_model: int = 32,
        nhead: int = 4,
        num_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.revin = RevIN(stock_features)
        self.input_proj = nn.Linear(stock_features, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        self.input_norm = nn.LayerNorm(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor, macro: torch.Tensor | None = None) -> torch.Tensor:
        # macro is accepted (and ignored) so this model is a drop-in
        # replacement for RAT in the training/backtest loops.
        x = self.revin(x)
        h = self.input_proj(x)
        h = self.pos_enc(h)
        h = self.input_norm(h)
        h = self.transformer(h)
        h_last = h[:, -1, :]
        return self.head(h_last).squeeze(-1)


class MacroConcatTransformer(nn.Module):
    """Concatenates the regime embedding z to every timestep, then runs a
    plain transformer over the combined sequence. Included to demonstrate
    the concatenation failure mode described above.
    """

    def __init__(
        self,
        stock_features: int = 17,
        macro_features: int = 12,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        z_dim: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.revin = RevIN(stock_features)
        self.regime_encoder = RegimeEncoder(macro_features, z_dim)

        self.input_proj = nn.Linear(stock_features + z_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        self.input_norm = nn.LayerNorm(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor, macro: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        x = self.revin(x)
        z = self.regime_encoder(macro)  # (B, z_dim)
        z_expanded = z.unsqueeze(1).expand(B, T, -1)  # (B, T, z_dim)
        x_combined = torch.cat([x, z_expanded], dim=-1)  # (B, T, F+z_dim)

        h = self.input_proj(x_combined)
        h = self.pos_enc(h)
        h = self.input_norm(h)
        h = self.transformer(h)
        h = h.mean(dim=1)  # mean pooling
        return self.head(h).squeeze(-1)
