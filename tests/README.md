# Regime-Aware Transformer (RAT)

[![CI](https://github.com/nseth7/regime-aware-transformer/actions/workflows/ci.yml/badge.svg)](https://github.com/nseth7/regime-aware-transformer/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/nseth7/regime-aware-transformer/branch/main/graph/badge.svg)](https://codecov.io/gh/nseth7/regime-aware-transformer)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

Cross-sectional equity return prediction that conditions a transformer on a *learned* macro regime embedding — no hand-labeled market regimes, no VIX bins, no discrete state machine.

See [`DESIGN.md`](DESIGN.md) for architecture rationale and [`CONTRIBUTING.md`](CONTRIBUTING.md) for the dev workflow.

Full writeup: [`assets/paper.pdf`](assets/paper.pdf) — *Regime-Aware Transformers for Equity Return Prediction* (Naman Seth, Virginia Tech).

## TL;DR

Predicting stock returns is hard because the relationship between technical indicators and future returns shifts with market conditions — a momentum signal that works in calm markets can reverse entirely during stress. This repo implements and compares three transformer architectures, each motivated by diagnosing the previous one's failure:

| Model | Macro conditioning | Test IC | Net Sharpe (20bps, 21d) |
|---|---|---|---|
| `BaselineTransformer` | none | ≈ 0.000 | — |
| `MacroConcatTransformer` | concatenated to every timestep | ≈ 0.000 | — |
| **`RegimeAwareTransformer` (RAT)** | feature gate + FiLM | **+0.006** | **+2.35** |

The concatenation baseline is not a weak version of RAT — it's included to demonstrate a specific failure mode. Under a rank-based (IC) training loss, a macro vector that's *identical across every stock on a given date* contributes zero gradient, because ranking loss is shift-invariant to a constant added to all predictions. RAT's feature gate and FiLM layers both scale/shift by amounts that differ *per stock* (since each stock has different feature values), which restores gradient flow into the regime encoder. `tests/test_regime_conditioning.py` reproduces this empirically.

## Results

RAT, evaluated on the 2024–2025 held-out test set, long-short top/bottom 30 names, 21-day rebalance:

| Cost | Net Sharpe | Ann. Return | Max Drawdown | \|t\| |
|---|---|---|---|---|
| 0 bps (gross) | 2.68 | +36.9% | −4.7% | 3.00 |
| 10 bps | 2.52 | +34.6% | −4.9% | 2.81 |
| 20 bps | **2.35** | **+32.0%** | −5.1% | 2.63 |

For reference, SPY buy-and-hold over the same period returned +17% annualized (Sharpe 1.61, max drawdown −8.0%). RAT is market-neutral by construction — the return comes entirely from cross-sectional stock selection, not passive market exposure.

Regenerate these numbers with `scripts/backtest.py` / `scripts/compare_models.py` after training (see below); nothing in this repo hardcodes them.

## Architecture

```
Stock Seq (B, 60, F)                    Macro (B, 12)
       │                                      │
     RevIN                              RegimeEncoder
  (per-window norm)                    MLP 12→64→64→32
       │                                      │
  Feature Gate  ◄────────────────────────── z (B, 32)
  σ(W_g · z) ⊙ x                              │
       │                                      │
 Linear + PosEnc + Norm                       │
   (F → d_model=32)                           │
       │                                      │
  TransformerEncoder                          │
   (1 layer, 4 heads,                         │
      pre-norm)                               │
       │                                      │
  Last-token pooling                          │
     h ∈ R^32                                 │
       │                                      │
      FiLM  ◄────────────────────────────────┤
  h·(1+γ(z)) + β(z)                           │
       │                                      │
  Concat [h_cond; z]  ◄─────────────────────┘
      dim = 64
       │
   MLP Head (64→64→1)
       │
      ŷ (return rank)
```

- **RevIN** normalizes each 60-day window by its own mean/std, so the model sees comparable scales regardless of price level or volatility regime.
- **RegimeEncoder** compresses 12 macro indicators (VIX level/z-score, yield curve, cross-asset momentum) into a 32-dim embedding `z`, learned end-to-end.
- **Feature gate** lets `z` suppress/amplify technical features *before* the transformer runs — e.g. dampening momentum features in high-volatility regimes.
- **FiLM** (Perez et al., 2018) applies a `z`-conditioned scale-and-shift to the pooled sequence representation.
- Trained with a **negative Pearson IC loss** (differentiable proxy for the Information Coefficient), not MSE — the task is cross-sectional ranking, not magnitude regression.

Total parameters: 27,476. See `src/rat/models/rat.py`.

## Data

- **Universe**: S&P 500 constituents *as of January 2018* (not today's list — using a point-in-time snapshot avoids survivorship bias; delisted/acquired names like TWX, MON, RAI, RHT are included).
- **Features**: 60-day lookback window of OHLCV + technical indicators (EMA-10/30, MACD, ADX, RSI-14, stochastic %K, Bollinger Band width, ATR-14, OBV, 1d/5d/21d returns, log return, 21-day realized vol) per stock, plus a 12-dim macro snapshot (VIX, 10y/3m yields, term spread, VIX z-score, SPY/HYG/oil/gold/DXY momentum) at prediction time.
- **Target**: 5-day-ahead log return, cross-sectionally z-scored within each trading date (so the model optimizes for *rank*, not absolute magnitude).
- **Split**: chronological, never random — train 2018–2022, val 2023, test 2024–2025.
- **Leakage controls**: macro rolling features (e.g. 252-day VIX z-score) are warmed up starting in 2016, well before the training window; imputation medians are fit on train only and reused for val/test; scalers are fit on train only.

See `src/rat/data/pipeline.py` for the full pipeline and `src/rat/config.py::DataConfig` for every tunable.

> **Note on feature count.** The prepared dataset includes raw OHLCV alongside derived technicals (19 stock-level columns total). The paper's data section also describes excluding non-stationary price/volume levels to avoid leakage; that variant is available via `--exclude-price-levels` on `scripts/prepare_data.py` (`DataConfig.exclude_price_levels`). The model's `stock_features` dimension is inferred from whichever dataset you actually build, so either configuration works out of the box.

## Repo structure

```
regime-aware-transformer/
├── assets/
│   └── paper.pdf                  # full writeup
├── configs/
│   └── default.yaml               # single source of truth for hyperparameters
├── scripts/                       # thin CLI entry points
│   ├── prepare_data.py
│   ├── train.py
│   ├── backtest.py
│   └── compare_models.py
├── src/rat/
│   ├── config.py                  # DataConfig / ModelConfig / TrainConfig / BacktestConfig
│   ├── data/
│   │   ├── download.py            # tickers, OHLCV, macro instruments (yfinance)
│   │   ├── features.py            # technical indicators, imputation
│   │   ├── pipeline.py            # split -> window -> scale -> save
│   │   └── dataset.py             # torch Dataset over the prepared .npz files
│   ├── models/
│   │   ├── layers.py              # RevIN, RegimeEncoder, feature gate, positional encoding
│   │   ├── rat.py                 # RegimeAwareTransformer
│   │   └── baselines.py           # BaselineTransformer, MacroConcatTransformer
│   ├── training/
│   │   ├── losses.py              # ic_loss (negative Pearson correlation)
│   │   └── train.py               # train/eval epoch loops, early stopping, CLI-callable train()
│   └── evaluation/
│       ├── backtest.py            # long-short simulation, Sharpe/IC/drawdown
│       ├── inference.py           # checkpoint loading, prediction assembly
│       └── plots.py               # equity curves, cost sensitivity, vs-benchmark
├── tests/
│   ├── test_models.py             # forward-shape + gradient sanity checks
│   └── test_regime_conditioning.py  # reproduces the concat-fails-to-gradient claim
├── requirements.txt
├── pyproject.toml
└── LICENSE
```

## Setup

```bash
git clone https://github.com/<you>/regime-aware-transformer.git
cd regime-aware-transformer
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Usage

```bash
# 1. Build the dataset (downloads S&P 500 OHLCV + macro data via yfinance; ~15-20 min)
python scripts/prepare_data.py --output-dir data

# 2. Train RAT
python scripts/train.py --model regime --data-dir data --checkpoint-dir checkpoints

# 3. Train the two comparison baselines
python scripts/train.py --model baseline --data-dir data --checkpoint-dir checkpoints
python scripts/train.py --model macro_concat --data-dir data --checkpoint-dir checkpoints

# 4. Backtest a single model
python scripts/backtest.py --model regime --data-dir data --checkpoint-dir checkpoints --out-dir results

# 5. Or run the full three-model comparison + SPY benchmark in one shot
python scripts/compare_models.py --data-dir data --checkpoint-dir checkpoints --out-dir results
```

```bash
# Run tests
pytest
```

### Programmatic use

```python
import torch
from rat.models import RegimeAwareTransformer

model = RegimeAwareTransformer(stock_features=19, macro_features=12)
x = torch.randn(8, 60, 19)   # (batch, lookback, stock_features)
macro = torch.randn(8, 12)   # (batch, macro_features)
pred = model(x, macro)       # (batch,) — cross-sectional return rank score

pred, diagnostics = model(x, macro, return_diagnostics=True)
diagnostics["z"]     # (8, 32) learned regime embedding
diagnostics["gate"]  # (8, 19) per-feature gate values, in (0, 1)
```

## Why IC loss instead of MSE?

The task is cross-sectional ranking — "which stocks will outperform which others today" — not predicting the exact magnitude of tomorrow's return (an essentially unsolvable problem at 5-day horizons with this signal-to-noise ratio). `ic_loss` minimizes the negative Pearson correlation between predictions and targets within each batch, which is a differentiable proxy for the Spearman IC used to evaluate the model. See `src/rat/training/losses.py`.

## Limitations / honest caveats

- Test-set daily IC (+0.006, t≈0.75) is not itself statistically significant; the backtest Sharpe is higher because long-short portfolio construction averages the per-stock signal across 30 positions per side, which reduces variance faster than it reduces the signal. Both numbers are reported — see the paper's Discussion (Section 4.3) for the full explanation.
- Single test period (2024–2025); no walk-forward re-estimation.
- Transaction cost model is a flat per-trade bps charge on turnover, not a market-impact model.
- This is a research prototype, not investment advice.

## Citation

```
@misc{seth2026rat,
  title  = {Regime-Aware Transformers for Equity Return Prediction},
  author = {Seth, Naman},
  year   = {2026},
  note   = {Virginia Tech}
}
```
