#!/usr/bin/env python
"""CLI wrapper: build train/val/test datasets from raw market data.

Usage:
    python scripts/prepare_data.py
    python scripts/prepare_data.py --output-dir data --exclude-price-levels
"""

from __future__ import annotations

import argparse
import logging

from rat.config import DataConfig
from rat.data.pipeline import run


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", default="data")
    p.add_argument("--start-date", default="2018-01-01")
    p.add_argument("--end-date", default="2026-01-01")
    p.add_argument("--train-end", default="2022-12-31")
    p.add_argument("--val-end", default="2023-12-31")
    p.add_argument("--lookback", type=int, default=60)
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument(
        "--exclude-price-levels",
        action="store_true",
        help="Drop raw Open/High/Low/Close/Volume from model features "
        "(matches the paper's stated non-stationary-level exclusion).",
    )
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    cfg = DataConfig(
        start_date=args.start_date,
        end_date=args.end_date,
        train_end=args.train_end,
        val_end=args.val_end,
        lookback=args.lookback,
        horizon=args.horizon,
        exclude_price_levels=args.exclude_price_levels,
        output_dir=args.output_dir,
    )
    run(cfg)


if __name__ == "__main__":
    main()
