#!/usr/bin/env python
"""Train (if needed) and compare all three model variants, then benchmark
the best one against SPY buy-and-hold. Reproduces the paper's Table 1/2
and Figure 2 comparisons end to end.

Usage:
    python scripts/compare_models.py --data-dir data
"""

from __future__ import annotations

import argparse
import logging
import os

import joblib
import pandas as pd
import yfinance as yf

from rat.config import BacktestConfig, ModelConfig, TrainConfig
from rat.evaluation.backtest import sweep_backtest
from rat.evaluation.inference import load_checkpoint, build_prediction_frame
from rat.evaluation.plots import plot_model_comparison, plot_vs_benchmark
from rat.training.train import get_device, train

MODELS = ["baseline", "macro_concat", "regime"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--checkpoint-dir", default="checkpoints")
    p.add_argument("--out-dir", default="results")
    p.add_argument("--skip-training", action="store_true",
                    help="Assume checkpoints already exist; only run backtests.")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = get_device()
    bt_cfg = BacktestConfig()

    if not args.skip_training:
        for model_name in MODELS:
            checkpoint_path = os.path.join(args.checkpoint_dir, f"{model_name}_best.pt")
            if os.path.exists(checkpoint_path):
                logging.info("Checkpoint exists for %s, skipping training", model_name)
                continue
            logging.info("Training %s ...", model_name)
            train_cfg = TrainConfig(model_name=model_name, checkpoint_dir=args.checkpoint_dir)
            train(args.data_dir, train_cfg, ModelConfig())

    headline_by_model = {}
    for model_name in MODELS:
        checkpoint_path = os.path.join(args.checkpoint_dir, f"{model_name}_best.pt")
        model, _ = load_checkpoint(checkpoint_path, device)
        df = build_prediction_frame(args.data_dir, model, device, split="test")
        all_results = sweep_backtest(df, bt_cfg)
        key = (bt_cfg.headline_top_k, bt_cfg.headline_rebalance, bt_cfg.cost_bps_levels[-1])
        if key in all_results:
            headline_by_model[model_name] = all_results[key]

    best_model = max(headline_by_model, key=lambda n: headline_by_model[n]["net_sharpe"])
    logging.info("Best model @ headline config: %s (Sharpe=%+.3f)",
                 best_model, headline_by_model[best_model]["net_sharpe"])

    plot_model_comparison(
        headline_by_model, os.path.join(args.out_dir, "compare_equity.png"),
        title=f"Model Comparison (top_k={bt_cfg.headline_top_k}, reb={bt_cfg.headline_rebalance}d, "
              f"cost={bt_cfg.cost_bps_levels[-1]}bps)",
    )

    # Best model vs SPY buy-and-hold
    best_bt = headline_by_model[best_model]["bt"]
    test_dates = pd.to_datetime(sorted(best_bt["date"].unique()))
    spy = yf.download("SPY", start=test_dates.min(), end=test_dates.max() + pd.Timedelta(days=2),
                       progress=False, auto_adjust=True)
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)
    spy = spy[["Close"]].dropna()
    spy["ret"] = spy["Close"].pct_change().fillna(0)
    spy_aligned = spy.reindex(best_bt["date"], method="ffill")
    spy_equity = (1 + spy_aligned["ret"]).cumprod()
    spy_equity = spy_equity / spy_equity.iloc[0]

    plot_vs_benchmark(
        best_bt, spy_equity, os.path.join(args.out_dir, "best_vs_spy.png"),
        strategy_label=best_model,
    )

    joblib.dump(
        {"headline_by_model": headline_by_model, "best_model": best_model},
        os.path.join(args.out_dir, "compare_results.pkl"),
    )
    logging.info("Saved comparison results to %s/", args.out_dir)


if __name__ == "__main__":
    main()
