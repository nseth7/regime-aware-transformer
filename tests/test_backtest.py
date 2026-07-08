import numpy as np
import pandas as pd

from rat.config import BacktestConfig
from rat.evaluation.backtest import (
    daily_ic,
    headline_result,
    ic_summary,
    run_backtest,
    sweep_backtest,
)


def _synthetic_frame(n_dates=40, n_tickers=20, seed=0, signal_strength=1.0):
    """Predictions correlated with y_raw/y_cs by construction, so backtests
    should show a positive edge (long-short spread > 0)."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=n_dates)
    rows = []
    for d in dates:
        y_raw = rng.normal(0, 0.02, n_tickers)
        noise = rng.normal(0, 0.02, n_tickers)
        pred = signal_strength * y_raw + noise
        y_cs = (y_raw - y_raw.mean()) / (y_raw.std() + 1e-8)
        for i in range(n_tickers):
            rows.append(
                {
                    "date": d,
                    "ticker": f"T{i}",
                    "pred": pred[i],
                    "y_raw": y_raw[i],
                    "y_cs": y_cs[i],
                }
            )
    return pd.DataFrame(rows)


def test_daily_ic_shape_and_range():
    df = _synthetic_frame()
    ic = daily_ic(df)
    assert isinstance(ic, pd.Series)
    assert (ic.abs() <= 1.0 + 1e-8).all()


def test_daily_ic_drops_days_with_too_few_names():
    # fewer than 10 tickers on a date -> that date is dropped (NaN filtered)
    df = _synthetic_frame(n_tickers=5)
    ic = daily_ic(df)
    assert len(ic) == 0


def test_ic_summary_keys_and_finiteness():
    df = _synthetic_frame()
    summary = ic_summary(df)
    for key in ("ic_mean", "ic_std", "ic_t_stat", "pct_positive_days", "n_days"):
        assert key in summary
    assert summary["n_days"] > 0
    assert 0.0 <= summary["pct_positive_days"] <= 1.0


def test_run_backtest_positive_signal_beats_negative_signal():
    bt_cfg = BacktestConfig()
    good = _synthetic_frame(signal_strength=5.0, seed=1)
    bad = _synthetic_frame(signal_strength=-5.0, seed=1)

    res_good = run_backtest(good, top_k=5, rebalance_days=5, cost_bps=0, bt_cfg=bt_cfg)
    res_bad = run_backtest(bad, top_k=5, rebalance_days=5, cost_bps=0, bt_cfg=bt_cfg)

    assert res_good is not None and res_bad is not None
    assert res_good["gross_ann_ret"] > res_bad["gross_ann_ret"]


def test_run_backtest_costs_reduce_net_vs_gross():
    bt_cfg = BacktestConfig()
    df = _synthetic_frame(signal_strength=3.0, seed=2)
    res = run_backtest(df, top_k=5, rebalance_days=5, cost_bps=50, bt_cfg=bt_cfg)
    assert res is not None
    assert res["net_ann_ret"] <= res["gross_ann_ret"]
    assert res["avg_cost_drag"] >= 0


def test_run_backtest_returns_none_when_universe_too_small():
    bt_cfg = BacktestConfig()
    df = _synthetic_frame(n_tickers=4)  # < 2 * top_k for any default top_k
    res = run_backtest(df, top_k=30, rebalance_days=5, cost_bps=0, bt_cfg=bt_cfg)
    assert res is None


def test_sweep_backtest_covers_full_grid():
    bt_cfg = BacktestConfig(top_k_values=(5,), rebalance_frequencies=(5, 10), cost_bps_levels=(0, 10))
    df = _synthetic_frame(n_tickers=20)
    results = sweep_backtest(df, bt_cfg)
    assert set(results.keys()) <= {(5, 5, 0), (5, 5, 10), (5, 10, 0), (5, 10, 10)}
    assert len(results) > 0


def test_headline_result_matches_configured_key():
    bt_cfg = BacktestConfig(
        top_k_values=(5,),
        rebalance_frequencies=(5,),
        cost_bps_levels=(0, 20),
        headline_top_k=5,
        headline_rebalance=5,
    )
    df = _synthetic_frame(n_tickers=20)
    results = sweep_backtest(df, bt_cfg)
    headline = headline_result(results, bt_cfg)
    assert headline is not None
    assert headline["top_k"] == 5 and headline["rebalance_days"] == 5 and headline["cost_bps"] == 20
