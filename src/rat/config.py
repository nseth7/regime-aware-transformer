"""Central configuration for data preparation, model, and training.

Everything that used to be a scattered set of module-level constants in the
original research notebooks lives here as a single typed config object, so a
run can be fully reproduced from one YAML file (see configs/default.yaml).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DataConfig:
    start_date: str = "2018-01-01"
    end_date: str = "2026-01-01"
    train_end: str = "2022-12-31"          # train:  2018-2022
    val_end: str = "2023-12-31"            # val:    2023 | test: 2024-2025
    lookback: int = 60                     # days of history per sample
    horizon: int = 5                       # days-ahead prediction target
    max_ffill_gap: int = 5                 # max consecutive NaNs to forward-fill
    macro_start: str = "2016-01-01"        # earlier start so rolling macro feats warm up
    min_history_days: int = 128            # lookback + horizon + indicator warmup
    # If True, raw price/volume levels (Open, High, Low, Close, Volume) are
    # dropped from the stock feature matrix before windowing, matching the
    # "non-stationary levels excluded to avoid leakage" description in the
    # paper. If False (default, matches the original notebook), raw OHLCV is
    # kept and the model's `stock_features` dim is inferred from the data.
    exclude_price_levels: bool = False
    output_dir: str = "data"


@dataclass
class ModelConfig:
    macro_features: int = 12
    d_model: int = 32
    nhead: int = 4
    num_layers: int = 1
    z_dim: int = 32
    dropout: float = 0.1
    # stock_features is inferred at train time from the prepared dataset
    # (len(feature_names['price_cols'])) rather than hardcoded here.


@dataclass
class TrainConfig:
    model_name: str = "regime"             # "regime" | "baseline" | "macro_concat"
    batch_size: int = 256
    lr: float = 1e-4
    weight_decay: float = 1e-2
    max_epochs: int = 50
    patience: int = 10
    grad_clip_norm: float = 1.0
    checkpoint_dir: str = "checkpoints"
    seed: int = 42


@dataclass
class BacktestConfig:
    horizon_days: int = 5
    trading_days_per_year: int = 252
    top_k_values: tuple = (30, 50, 100)
    rebalance_frequencies: tuple = (5, 10, 21)
    cost_bps_levels: tuple = (0, 5, 10, 20)
    headline_top_k: int = 30
    headline_rebalance: int = 21


MACRO_SOURCES = {
    "VIX": "^VIX",
    "TNX": "^TNX",
    "IRX": "^IRX",
    "DXY": "DX-Y.NYB",
    "Gold": "GC=F",
    "Oil": "CL=F",
    "HYG": "HYG",
    "SPY": "SPY",
}

MACRO_COLS = [
    "VIX", "TNX", "IRX", "TermSpread",
    "VIX_zscore",
    "SPY_ret_5d", "SPY_ret_21d", "SPY_ret_63d",
    "HYG_ret", "Oil_ret", "Gold_ret", "DXY_ret",
]

PRICE_LEVEL_COLS = ["Open", "High", "Low", "Close", "Volume"]

TECHNICAL_COLS = [
    "EMA_10", "EMA_30", "MACD", "ADX",
    "RSI_14", "STOCH_k",
    "BB_width", "ATR_14",
    "OBV",
    "ret_1d", "ret_5d", "ret_21d", "logret", "rvol_21d",
]

# Full feature set as produced by the original pipeline (19 columns).
PRICE_COLS = PRICE_LEVEL_COLS + TECHNICAL_COLS
