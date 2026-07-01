"""Training loop with early stopping, LR scheduling, and per-epoch IC
tracking on the validation set.

Usage:
    python scripts/train.py --model regime --config configs/default.yaml
"""

from __future__ import annotations

import logging
import os
import time
import warnings

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from rat.config import TrainConfig
from rat.data.dataset import RegimeDataset, collate
from rat.models import MODEL_REGISTRY
from rat.training.losses import ic_loss

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_epoch(model, loader, optimizer, loss_fn, device, grad_clip_norm: float) -> dict:
    model.train()
    total_loss, total_pstd, n_batches = 0.0, 0.0, 0

    for x, macro, y, _, _ in loader:
        x, macro, y = x.to(device), macro.to(device), y.to(device)
        pred = model(x, macro)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        optimizer.step()

        total_loss += loss.item()
        total_pstd += pred.std().item()
        n_batches += 1

    return {"loss": total_loss / n_batches, "pred_std": total_pstd / n_batches}


@torch.no_grad()
def eval_epoch(model, loader, loss_fn, device) -> dict:
    model.eval()
    total_loss = 0.0
    all_preds, all_true, all_dates = [], [], []

    for x, macro, y, dates, _ in loader:
        x, macro, y = x.to(device), macro.to(device), y.to(device)
        pred = model(x, macro)
        total_loss += loss_fn(pred, y).item()
        all_preds.append(pred.cpu())
        all_true.append(y.cpu())
        all_dates.extend(dates)

    preds = torch.cat(all_preds).numpy()
    trues = torch.cat(all_true).numpy()
    mae = np.abs(preds - trues).mean()
    dir_acc = ((preds > 0) == (trues > 0)).mean()

    df = pd.DataFrame({"date": all_dates, "pred": preds, "true": trues})
    daily_ic = df.groupby("date").apply(
        lambda g: g["pred"].corr(g["true"], method="spearman") if len(g) >= 10 else np.nan
    ).dropna()

    return {
        "loss": total_loss / len(loader),
        "mae": mae,
        "dir_acc": dir_acc,
        "ic": daily_ic.mean() if len(daily_ic) else float("nan"),
        "pred_std": preds.std(),
    }


def build_model(model_name: str, n_stock_features: int, n_macro_features: int, model_cfg=None):
    cls = MODEL_REGISTRY[model_name]
    kwargs = {"stock_features": n_stock_features}
    if model_name != "baseline":
        kwargs["macro_features"] = n_macro_features
    if model_cfg is not None:
        if model_name == "regime":
            kwargs.update(
                d_model=model_cfg.d_model, nhead=model_cfg.nhead,
                num_layers=model_cfg.num_layers, z_dim=model_cfg.z_dim,
                dropout=model_cfg.dropout,
            )
    return cls(**kwargs)


def train(data_dir: str, cfg: TrainConfig, model_cfg=None) -> list[dict]:
    torch.manual_seed(cfg.seed)
    device = get_device()
    logger.info("Device: %s", device)

    train_ds = RegimeDataset(os.path.join(data_dir, "train_cs.npz"))
    val_ds = RegimeDataset(os.path.join(data_dir, "val_cs.npz"))

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        collate_fn=collate, num_workers=2, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        collate_fn=collate, num_workers=2, pin_memory=True,
    )
    logger.info("Train: %d samples (%d batches) | Val: %d samples (%d batches)",
                len(train_ds), len(train_loader), len(val_ds), len(val_loader))

    model = build_model(cfg.model_name, train_ds.n_stock_features, train_ds.n_macro_features, model_cfg)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Model: %s (%d parameters)", cfg.model_name, n_params)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    loss_fn = ic_loss

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(cfg.checkpoint_dir, f"{cfg.model_name}_best.pt")

    best_val_loss = float("inf")
    best_val_ic = float("-inf")
    best_ic_epoch = 0
    patience_count = 0
    history = []

    for epoch in range(1, cfg.max_epochs + 1):
        t0 = time.time()
        train_metrics = train_epoch(model, train_loader, optimizer, loss_fn, device, cfg.grad_clip_norm)
        val_metrics = eval_epoch(model, val_loader, loss_fn, device)
        elapsed = time.time() - t0

        scheduler.step(val_metrics["loss"])

        record = {"epoch": epoch, "elapsed_s": elapsed,
                  **{f"train_{k}": v for k, v in train_metrics.items()},
                  **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(record)

        if val_metrics["ic"] > best_val_ic:
            best_val_ic, best_ic_epoch = val_metrics["ic"], epoch

        improved = val_metrics["loss"] < best_val_loss
        logger.info(
            "epoch %3d | train_loss %.4f | val_loss %.4f | dir_acc %5.1f%% | ic %+.4f | %5.1fs %s",
            epoch, train_metrics["loss"], val_metrics["loss"],
            val_metrics["dir_acc"] * 100, val_metrics["ic"], elapsed,
            "*" if improved else "",
        )

        if improved:
            best_val_loss = val_metrics["loss"]
            patience_count = 0
            torch.save({
                "epoch": epoch, "model_name": cfg.model_name,
                "state_dict": model.state_dict(),
                "n_stock_features": train_ds.n_stock_features,
                "n_macro_features": train_ds.n_macro_features,
                "val_loss": val_metrics["loss"], "val_mae": val_metrics["mae"],
                "dir_acc": val_metrics["dir_acc"], "ic": val_metrics["ic"],
                "history": history,
            }, checkpoint_path)
        else:
            patience_count += 1
            if patience_count >= cfg.patience:
                logger.info("Early stopping at epoch %d (no improvement for %d epochs)",
                            epoch, cfg.patience)
                break

    logger.info("Best val loss: %.5f | Best val IC: %+.4f (epoch %d) | Checkpoint: %s",
                best_val_loss, best_val_ic, best_ic_epoch, checkpoint_path)
    return history
