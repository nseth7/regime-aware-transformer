import torch

from rat.training.losses import ic_loss


def test_ic_loss_perfect_correlation_is_minus_one():
    y = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    pred = y.clone()
    loss = ic_loss(pred, y)
    assert torch.isclose(loss, torch.tensor(-1.0), atol=1e-5)


def test_ic_loss_perfect_anticorrelation_is_plus_one():
    y = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    pred = -y.clone()
    loss = ic_loss(pred, y)
    assert torch.isclose(loss, torch.tensor(1.0), atol=1e-5)


def test_ic_loss_is_scale_invariant():
    torch.manual_seed(0)
    pred = torch.randn(32)
    y = torch.randn(32)
    loss_a = ic_loss(pred, y)
    loss_b = ic_loss(pred * 10.0, y)
    assert torch.isclose(loss_a, loss_b, atol=1e-5)


def test_ic_loss_is_shift_invariant():
    torch.manual_seed(0)
    pred = torch.randn(32)
    y = torch.randn(32)
    loss_a = ic_loss(pred, y)
    loss_b = ic_loss(pred + 5.0, y)
    assert torch.isclose(loss_a, loss_b, atol=1e-5)


def test_ic_loss_bounded_in_range():
    torch.manual_seed(1)
    for _ in range(10):
        pred = torch.randn(64)
        y = torch.randn(64)
        loss = ic_loss(pred, y)
        assert -1.0 - 1e-4 <= loss.item() <= 1.0 + 1e-4


def test_ic_loss_is_differentiable():
    pred = torch.randn(16, requires_grad=True)
    y = torch.randn(16)
    loss = ic_loss(pred, y)
    loss.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()


def test_ic_loss_zero_variance_prediction_is_finite():
    # constant predictions -> zero variance; the 1e-8 epsilon should
    # prevent a division-by-zero NaN/inf.
    pred = torch.ones(10)
    y = torch.randn(10)
    loss = ic_loss(pred, y)
    assert torch.isfinite(loss)
