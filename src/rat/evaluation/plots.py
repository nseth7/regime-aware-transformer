"""Plotting helpers — equity curves, cost sensitivity, model comparison,
and strategy-vs-SPY. Mirrors the plots reported in the paper's Figure 2.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_equity_curve(results_by_cost: dict, out_path: str, title: str = "Equity Curve") -> None:
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for cost, res in sorted(results_by_cost.items()):
        bt = res["bt"].copy()
        bt["equity"] = (1 + bt["net"]).cumprod()
        label = "Gross (0 bps)" if cost == 0 else f"Net @ {cost} bps"
        ax.plot(bt["date"], bt["equity"], linewidth=1.5, label=label)
    ax.axhline(1.0, color="grey", linestyle="--", alpha=0.5)
    ax.set_title(title)
    ax.set_ylabel("Equity (start = 1.0)")
    ax.set_xlabel("Date")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_model_comparison(results_by_model: dict, out_path: str, title: str = "Model Comparison") -> None:
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for name, res in results_by_model.items():
        bt = res["bt"].copy()
        bt["equity"] = (1 + bt["net"]).cumprod()
        ax.plot(bt["date"], bt["equity"], linewidth=1.6,
                label=f"{name} (Sh={res['net_sharpe']:+.2f}, t={res['net_t_stat']:+.2f})")
    ax.axhline(1.0, color="grey", linestyle="--", alpha=0.5)
    ax.set_title(title)
    ax.set_ylabel("Equity (start = 1.0)")
    ax.set_xlabel("Date")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_vs_benchmark(strategy_bt: pd.DataFrame, benchmark_equity: pd.Series, out_path: str,
                       strategy_label: str = "RAT", benchmark_label: str = "SPY Buy & Hold") -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True,
                              gridspec_kw={"height_ratios": [2, 1]})

    strat_equity = (1 + strategy_bt["net"]).cumprod()
    axes[0].plot(strategy_bt["date"], strat_equity, linewidth=1.8, label=strategy_label)
    axes[0].plot(benchmark_equity.index, benchmark_equity.values, linewidth=1.8, label=benchmark_label)
    axes[0].axhline(1.0, color="grey", linestyle="--", alpha=0.5)
    axes[0].set_ylabel("Equity (start = 1.0)")
    axes[0].legend(loc="upper left")
    axes[0].grid(alpha=0.3)
    axes[0].set_title(f"{strategy_label} vs {benchmark_label}")

    strat_eq = strat_equity.values
    strat_dd = (strat_eq - np.maximum.accumulate(strat_eq)) / np.maximum.accumulate(strat_eq)
    bench_dd = (benchmark_equity - benchmark_equity.cummax()) / benchmark_equity.cummax()

    axes[1].plot(strategy_bt["date"], strat_dd * 100, linewidth=1.4, label=strategy_label)
    axes[1].fill_between(strategy_bt["date"], strat_dd * 100, 0, alpha=0.2)
    axes[1].plot(bench_dd.index, bench_dd.values * 100, linewidth=1.4, label=benchmark_label)
    axes[1].fill_between(bench_dd.index, bench_dd.values * 100, 0, alpha=0.2)
    axes[1].set_ylabel("Drawdown %")
    axes[1].set_xlabel("Date")
    axes[1].legend(loc="lower left")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
