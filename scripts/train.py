#!/usr/bin/env python
"""CLI wrapper: train RAT or a baseline.

Usage:
    python scripts/train.py --model regime
    python scripts/train.py --model baseline
    python scripts/train.py --model macro_concat
"""

from __future__ import annotations

import argparse
import logging

from rat.config import ModelConfig, TrainConfig
from rat.training.train import train


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=["regime", "baseline", "macro_concat"], default="regime")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--checkpoint-dir", default="checkpoints")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--max-epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    train_cfg = TrainConfig(
        model_name=args.model,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_epochs=args.max_epochs,
        patience=args.patience,
        checkpoint_dir=args.checkpoint_dir,
        seed=args.seed,
    )
    model_cfg = ModelConfig()
    train(args.data_dir, train_cfg, model_cfg)


if __name__ == "__main__":
    main()
