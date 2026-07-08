# Design Notes

This document is the engineering counterpart to `assets/paper.pdf` — it
explains *why the code is structured the way it is*, not the research
results themselves. Read this if you're reviewing the repo rather than the
paper.

## 1. Problem framing

Cross-sectional equity return prediction: given each stock's recent
technical history and the day's macro snapshot, predict a forward return
*rank* (not magnitude) relative to the rest of the universe on that date.
The downstream use case is a long-short portfolio, so what matters is
whether the model separates future winners from future losers on a given
day — not whether its point prediction is well-calibrated in absolute
units. That framing drives two decisions that show up throughout the
codebase:

- **Loss function.** `training/losses.py::ic_loss` is negative Pearson
  correlation per batch, not MSE. MSE optimizes magnitude; IC optimizes
  rank ordering, which is what the strategy actually monetizes.
- **Targets.** Targets are cross-sectionally z-scored per date before
  training (`data/pipeline.py::make_cross_sectional_targets`), so a batch
  spanning multiple dates still has a meaningful, comparable target scale.

## 2. Why three models, not one

`models/baselines.py` (`BaselineTransformer`, `MacroConcatTransformer`) and
`models/rat.py` (`RegimeAwareTransformer`) aren't a strong/weak pair — each
is a diagnostic step:

1. **`BaselineTransformer`** — no macro input at all. Establishes the floor.
2. **`MacroConcatTransformer`** — macro vector concatenated at every
   timestep, the "obvious" way to add conditioning. Test IC ≈ 0, indistinguishable
   from the no-macro baseline.
3. **`RegimeAwareTransformer`** — same macro information, injected via a
   feature gate and FiLM instead of concatenation.

The concat baseline's failure is the interesting result, and it's a direct
consequence of the loss function in §1: `ic_loss` is shift-invariant to any
constant added identically to every prediction in a batch. On a given date,
the macro vector *is* identical across every stock, so a term that only
adds a constant offset per date contributes zero gradient to the macro
pathway under this loss — regardless of how informative that macro vector
actually is. `RegimeFeatureGate` and the FiLM layer in `rat.py` both scale
or shift by amounts that vary *per stock* (they're multiplied through
stock-specific feature values), which breaks the symmetry and restores
gradient flow into `RegimeEncoder`. `tests/test_regime_conditioning.py`
verifies this empirically by checking gradient norms into the regime
encoder under both conditioning schemes.

This is also why the repo keeps all three models in `MODEL_REGISTRY`
(`models/__init__.py`) rather than deleting the baselines once RAT worked —
the ablation *is* the contribution.

## 3. Model internals (`models/layers.py`, `models/rat.py`)

- **RevIN** (`layers.py::RevIN`) — per-window instance normalization
  (Kim et al., 2022) so the transformer sees comparable input scale
  regardless of a stock's price level or the current volatility regime.
  Learnable affine transform recovers scale where useful.
- **RegimeEncoder** — macro snapshot (12-dim) compressed to a 32-dim
  regime embedding `z` via a 12→64→64→32 MLP with LayerNorm + GELU.
  Learned end-to-end; no hand-labeled or discretized regimes anywhere.
- **RegimeFeatureGate** — per-feature sigmoid gate, `x * sigmoid(W·z)`,
  applied *before* the transformer sees the sequence. Lets the regime
  decide which technical indicators to amplify or suppress.
- **FiLM** (`rat.py`, Perez et al., 2018) — `h * (1 + gamma) + beta`
  applied to the pooled sequence representation after the transformer,
  with `gamma, beta = Linear(z)`. The `(1 + gamma)` parameterization
  (rather than raw `gamma`) means the layer starts near an identity
  transform at initialization, which matters for training stability
  before `gamma` has learned anything useful.
- **Pooling** — last-token, not mean. For a 5-day-ahead target the most
  recent timestep carries most of the signal; mean pooling would dilute
  it by a factor of the lookback window length.

## 4. Data leakage guards (`config.py`, `data/pipeline.py`)

- `DataConfig.exclude_price_levels` — optionally drops raw OHLCV levels
  from the feature matrix, since raw price levels are non-stationary and
  can leak information through scale rather than shape.
- `macro_start` predates `start_date` so rolling macro indicators (e.g.
  63-day SPY return) are fully warmed up before the first training sample,
  rather than starting from partially-computed rolling windows.
- Train/val/test are strict date ranges (`train_end`, `val_end`), not a
  random split, since a random split across time would leak future
  information into training via overlapping lookback windows.

## 5. Backtest engine (`evaluation/backtest.py`)

- `daily_ic` uses Spearman, not Pearson — matches the rank-based framing
  in §1 and is more robust to outlier return days than Pearson.
- `run_backtest`'s `return_scale` handles the mismatch between a fixed
  prediction horizon (`horizon_days`) and a variable rebalance frequency:
  when they're equal the simulation is a clean non-overlapping backtest;
  when they differ, held-period return is linearly scaled as an
  approximation, matching the convention used in the original research
  notebook. This is a known simplification, not an oversight — see the
  docstring in `backtest.py` for the exact caveat.
- Reported headline config (`BacktestConfig.headline_top_k=30`,
  `headline_rebalance=21`) is the one config, out of the full
  top-k × rebalance × cost-bps grid the repo sweeps, that's reported in
  the paper. `scripts/compare_models.py` runs the full grid so the
  headline number isn't cherry-picked without the surrounding sensitivity
  analysis being visible.

## 6. What's intentionally out of scope

- No live/paper trading execution layer — this is a research + backtest
  repo, not a trading system.
- No hyperparameter sweep infra (Optuna/Ray Tune) — the architecture
  search space is small enough that manual sweeps were sufficient and are
  documented in the paper.
- No distributed training — dataset size doesn't warrant it.
