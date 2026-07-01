#!/usr/bin/env python
"""CLI wrapper: backtest a trained checkpoint on the test split.

Usage:
    python scripts/backtest.py --model regime
"""

from __future__ import annotations

import argparse
import logging
import os

import joblib

from rat.config import BacktestConfig
from rat.evaluation.backtest import headline_result, ic_summary, sweep_backtest
from rat.evaluation.inference import load_checkpoint, build_prediction_frame
from rat.evaluation.plots import plot_equity_curve
from rat.training.train import get_device


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=["regime", "baseline", "macro_concat"], default="regime")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--checkpoint-dir", default="checkpoints")
    p.add_argument("--out-dir", default="results")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = get_device()
    checkpoint_path = os.path.join(args.checkpoint_dir, f"{args.model}_best.pt")
    model, ckpt = load_checkpoint(checkpoint_path, device)
    logging.info("Loaded %s from epoch %d (val_IC=%.4f)", args.model, ckpt["epoch"], ckpt.get("ic", float("nan")))

    df = build_prediction_frame(args.data_dir, model, device, split="test")

    ic = ic_summary(df)
    logging.info("Test IC: mean=%+.4f std=%.4f t=%+.2f (%d days)",
                 ic["ic_mean"], ic["ic_std"], ic["ic_t_stat"], ic["n_days"])

    bt_cfg = BacktestConfig()
    all_results = sweep_backtest(df, bt_cfg)

    print(f"\n{'Top-K':<8}{'Reb':<6}{'Cost(bps)':<11}{'GrossSh':<10}{'NetSh':<10}"
          f"{'AnnRet%':<10}{'MaxDD%':<10}{'t-stat':<8}")
    for (top_k, reb, cost), res in sorted(all_results.items()):
        print(f"{top_k:<8}{reb:<6}{cost:<11}{res['gross_sharpe']:>+8.3f}  "
              f"{res['net_sharpe']:>+8.3f}  {res['net_ann_ret']*100:>+7.2f}  "
              f"{res['net_max_dd']*100:>+7.2f}  {res['net_t_stat']:>+6.2f}")

    headline = headline_result(all_results, bt_cfg)
    if headline:
        print(f"\nHeadline (top_k={bt_cfg.headline_top_k}, reb={bt_cfg.headline_rebalance}d, "
              f"cost={bt_cfg.cost_bps_levels[-1]}bps): "
              f"Sharpe={headline['net_sharpe']:+.2f}  AnnRet={headline['net_ann_ret']*100:+.2f}%  "
              f"MaxDD={headline['net_max_dd']*100:+.2f}%  t={headline['net_t_stat']:+.2f}")

    results_by_cost = {
        cost: res for (top_k, reb, cost), res in all_results.items()
        if top_k == bt_cfg.headline_top_k and reb == bt_cfg.headline_rebalance
    }
    plot_equity_curve(
        results_by_cost, os.path.join(args.out_dir, f"{args.model}_equity.png"),
        title=f"{args.model} — Equity Curves (top_k={bt_cfg.headline_top_k}, reb={bt_cfg.headline_rebalance}d)",
    )

    joblib.dump(
        {"model_name": args.model, "ic": ic, "all_results": all_results},
        os.path.join(args.out_dir, f"{args.model}_backtest.pkl"),
    )
    logging.info("Saved results to %s/", args.out_dir)


if __name__ == "__main__":
    main()
