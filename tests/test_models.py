import torch

from rat.models import BaselineTransformer, MacroConcatTransformer, RegimeAwareTransformer
from rat.training.losses import ic_loss


def _dummy_batch(batch=8, T=60, F=17, M=12):
    x = torch.randn(batch, T, F)
    macro = torch.randn(batch, M)
    y = torch.randn(batch)
    return x, macro, y


def test_rat_forward_shape():
    x, macro, _ = _dummy_batch()
    model = RegimeAwareTransformer(stock_features=17, macro_features=12)
    out = model(x, macro)
    assert out.shape == (8,)


def test_rat_diagnostics():
    x, macro, _ = _dummy_batch()
    model = RegimeAwareTransformer(stock_features=17, macro_features=12)
    out, diag = model(x, macro, return_diagnostics=True)
    assert out.shape == (8,)
    assert diag["z"].shape == (8, 32)
    assert diag["gate"].shape == (8, 17)


def test_baseline_forward_shape():
    x, macro, _ = _dummy_batch()
    model = BaselineTransformer(stock_features=17)
    out = model(x, macro)  # macro ignored, but accepted
    assert out.shape == (8,)


def test_macro_concat_forward_shape():
    x, macro, _ = _dummy_batch()
    model = MacroConcatTransformer(stock_features=17, macro_features=12)
    out = model(x, macro)
    assert out.shape == (8,)


def test_ic_loss_perfect_correlation_is_minus_one():
    y = torch.tensor([1.0, 2.0, 3.0, 4.0])
    loss = ic_loss(y, y)
    assert torch.isclose(loss, torch.tensor(-1.0), atol=1e-5)


def test_ic_loss_backward():
    x, macro, y = _dummy_batch()
    model = RegimeAwareTransformer(stock_features=17, macro_features=12)
    pred = model(x, macro)
    loss = ic_loss(pred, y)
    loss.backward()
    # Regime encoder must receive gradient (this is the whole point of the
    # feature gate + FiLM conditioning — see the paper, Section 3.4).
    grad_norm = sum(p.grad.abs().sum() for p in model.regime_encoder.parameters() if p.grad is not None)
    assert grad_norm > 0
