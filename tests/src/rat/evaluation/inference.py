"""Run a trained checkpoint over a split and assemble the (date, ticker,
pred, y_cs, y_raw) dataframe consumed by evaluation/backtest.py.
"""

from __future__ import annotations

import os

import joblib
import numpy as np
import pandas as pd
import torch

from rat.models import MODEL_REGISTRY


def load_checkpoint(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cls = MODEL_REGISTRY[ckpt["model_name"]]
    kwargs = {"stock_features": ckpt["n_stock_features"]}
    if ckpt["model_name"] != "baseline":
        kwargs["macro_features"] = ckpt["n_macro_features"]
    model = cls(**kwargs)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt


@torch.no_grad()
def predict_split(
    model, X: torch.Tensor, macro: torch.Tensor, device: torch.device, batch_size: int = 512
) -> np.ndarray:
    preds = []
    for i in range(0, len(X), batch_size):
        xb = X[i : i + batch_size].to(device)
        mb = macro[i : i + batch_size].to(device)
        out = model(xb, mb)
        if isinstance(out, tuple):
            out = out[0]
        preds.append(out.cpu().numpy())
    return np.concatenate(preds)


def build_prediction_frame(data_dir: str, model, device: torch.device, split: str = "test") -> pd.DataFrame:
    """Predicts on `{split}_cs.npz` (model input) and inverse-transforms the
    raw target from `{split}.npz` via the fitted y-scaler, so backtest P&L
    is computed in real log-return units, not standardized units.
    """
    d_cs = np.load(os.path.join(data_dir, f"{split}_cs.npz"), allow_pickle=True)
    X = torch.tensor(d_cs["X"], dtype=torch.float32)
    macro = torch.tensor(d_cs["macro"], dtype=torch.float32)
    y_cs = d_cs["y"]
    dates, tickers = d_cs["dates"], d_cs["tickers"]

    d_raw = np.load(os.path.join(data_dir, f"{split}.npz"), allow_pickle=True)
    assert (d_raw["dates"] == dates).all(), f"{split}.npz and {split}_cs.npz misaligned by date"
    assert (d_raw["tickers"] == tickers).all(), f"{split}.npz and {split}_cs.npz misaligned by ticker"

    scalers = joblib.load(os.path.join(data_dir, "scalers.pkl"))
    y_raw = scalers["y_scaler"].inverse_transform(d_raw["y"].reshape(-1, 1)).ravel()

    preds = predict_split(model, X, macro, device)

    df = pd.DataFrame(
        {
            "date": pd.to_datetime(dates),
            "ticker": tickers,
            "pred": preds,
            "y_cs": y_cs,
            "y_raw": y_raw,
        }
    )
    return df.sort_values(["date", "ticker"]).reset_index(drop=True)
