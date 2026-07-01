"""Long-short backtest engine and IC diagnostics.

Given per-(date, ticker) predictions and the raw forward return, this
builds a top-K / bottom-K long-short portfolio, rebalanced every N days,
with a simple per-trade transaction cost model. See the paper, Section 4,
for the reported headline numbers this reproduces.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from rat.config import BacktestConfig


def daily_ic(df: pd.DataFrame, pred_col: str = "pred", true_col: str = "y_cs") -> pd.Series:
    """Mean daily Spearman rank correlation between predictions and targets."""
    return df.groupby("date").apply(
        lambda g: g[pred_col].corr(g[true_col], method="spearman") if len(g) >= 10 else np.nan
    ).dropna()


def ic_summary(df: pd.DataFrame, pred_col: str = "pred", true_col: str = "y_cs") -> dict:
    ic = daily_ic(df, pred_col, true_col)
    ic_t = ic.mean() / (ic.std() / np.sqrt(len(ic))) if len(ic) > 1 and ic.std() > 0 else float("nan")
    return {
        "ic_mean": ic.mean(), "ic_std": ic.std(), "ic_t_stat": ic_t,
        "pct_positive_days": (ic > 0).mean(), "n_days": len(ic),
    }


def run_backtest(
    df: pd.DataFrame, top_k: int, rebalance_days: int, cost_bps: float, bt_cfg: BacktestConfig
) -> dict | None:
    """Long-short backtest.

    `df` needs columns: date, ticker, pred, y_raw (log return over
    `bt_cfg.horizon_days`).

    Note on return alignment: y_raw is a HORIZON_DAYS-ahead log return.
    For a canonical, non-overlapping simulation, rebalance_days should
    equal horizon_days; other values are treated as an approximation by
    linearly scaling the held-period return (see `return_scale` below),
    matching the original research notebook's convention.
    """
    unique_dates = sorted(df["date"].unique())
    rebalance_dates = unique_dates[::rebalance_days]
    return_scale = rebalance_days / bt_cfg.horizon_days

    rows = []
    prev_long, prev_short = set(), set()

    for rd in rebalance_dates:
        day_df = df[df["date"] == rd]
        if len(day_df) < 2 * top_k:
            continue

        day_df = day_df.sort_values("pred", ascending=False)
        longs, shorts = day_df.head(top_k), day_df.tail(top_k)

        long_simple = np.expm1(longs["y_raw"].values * return_scale).mean()
        short_simple = np.expm1(shorts["y_raw"].values * return_scale).mean()
        gross_return = long_simple - short_simple

        long_set, short_set = set(longs["ticker"]), set(shorts["ticker"])
        if prev_long or prev_short:
            n_changed = len(long_set ^ prev_long) + len(short_set ^ prev_short)
            cost_drag = (n_changed / (2 * top_k)) * (cost_bps / 10000.0)
            turnover = n_changed / (4 * top_k)
        else:
            cost_drag = 2 * (cost_bps / 10000.0)
            turnover = 1.0

        rows.append({
            "date": rd, "gross": gross_return, "net": gross_return - cost_drag,
            "cost": cost_drag, "turnover": turnover,
        })
        prev_long, prev_short = long_set, short_set

    if not rows:
        return None

    bt = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    periods_per_year = bt_cfg.trading_days_per_year / rebalance_days

    def sharpe(rets):
        m, s = rets.mean(), rets.std()
        return (m / s * np.sqrt(periods_per_year)) if s > 0 else 0.0

    def t_stat(rets):
        years = len(rets) / periods_per_year
        return sharpe(rets) * np.sqrt(years)

    def max_drawdown(rets):
        equity = (1 + rets).cumprod()
        return ((equity - equity.cummax()) / equity.cummax()).min()

    return {
        "top_k": top_k, "rebalance_days": rebalance_days, "cost_bps": cost_bps,
        "n_rebalances": len(bt),
        "gross_sharpe": sharpe(bt["gross"]), "net_sharpe": sharpe(bt["net"]),
        "gross_t_stat": t_stat(bt["gross"]), "net_t_stat": t_stat(bt["net"]),
        "gross_ann_ret": bt["gross"].mean() * periods_per_year,
        "net_ann_ret": bt["net"].mean() * periods_per_year,
        "ann_vol": bt["gross"].std() * np.sqrt(periods_per_year),
        "gross_max_dd": max_drawdown(bt["gross"]), "net_max_dd": max_drawdown(bt["net"]),
        "win_rate": (bt["gross"] > 0).mean(),
        "avg_turnover": bt["turnover"].mean(), "avg_cost_drag": bt["cost"].mean(),
        "bt": bt,
    }


def sweep_backtest(df: pd.DataFrame, bt_cfg: BacktestConfig) -> dict:
    """Run `run_backtest` over the full (top_k, rebalance, cost) grid."""
    results = {}
    for top_k in bt_cfg.top_k_values:
        for reb in bt_cfg.rebalance_frequencies:
            for cost in bt_cfg.cost_bps_levels:
                res = run_backtest(df, top_k, reb, cost, bt_cfg)
                if res is not None:
                    results[(top_k, reb, cost)] = res
    return results


def headline_result(all_results: dict, bt_cfg: BacktestConfig) -> dict | None:
    return all_results.get((bt_cfg.headline_top_k, bt_cfg.headline_rebalance, bt_cfg.cost_bps_levels[-1]))
