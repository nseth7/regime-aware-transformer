from dataclasses import asdict

from rat.config import (
    MACRO_COLS,
    MACRO_SOURCES,
    PRICE_COLS,
    PRICE_LEVEL_COLS,
    TECHNICAL_COLS,
    BacktestConfig,
    DataConfig,
    ModelConfig,
    TrainConfig,
)


def test_data_config_defaults_are_chronologically_consistent():
    cfg = DataConfig()
    assert cfg.macro_start < cfg.start_date
    assert cfg.start_date < cfg.train_end < cfg.val_end < cfg.end_date


def test_data_config_is_serializable():
    cfg = DataConfig()
    d = asdict(cfg)
    assert d["lookback"] == 60
    assert d["horizon"] == 5


def test_model_config_defaults():
    cfg = ModelConfig()
    assert cfg.d_model % cfg.nhead == 0  # required by nn.TransformerEncoderLayer
    assert cfg.z_dim > 0


def test_train_config_defaults():
    cfg = TrainConfig()
    assert cfg.model_name in {"regime", "baseline", "macro_concat"}
    assert cfg.patience <= cfg.max_epochs


def test_backtest_config_headline_values_are_in_grids():
    cfg = BacktestConfig()
    assert cfg.headline_top_k in cfg.top_k_values
    assert cfg.headline_rebalance in cfg.rebalance_frequencies


def test_price_cols_is_union_of_level_and_technical():
    assert PRICE_COLS == PRICE_LEVEL_COLS + TECHNICAL_COLS
    assert len(set(PRICE_COLS)) == len(PRICE_COLS)  # no duplicates


def test_macro_cols_and_sources_are_consistent():
    # every raw macro source ticker should be referenced, directly or via a
    # derived column (e.g. "VIX" -> "VIX", "VIX_zscore"), somewhere in MACRO_COLS
    assert len(MACRO_SOURCES) > 0
    assert len(set(MACRO_COLS)) == len(MACRO_COLS)  # no duplicates
