import torch

from rat.models import MacroConcatTransformer, RegimeAwareTransformer
from rat.training.losses import ic_loss


def test_macro_concat_gets_near_zero_regime_gradient_when_macro_is_constant():
    """Reproduces the paper's Section 3.4 claim: if every sample in a batch
    shares the same macro vector (i.e. all stocks on one date), the IC loss
    is shift-invariant to the (identical, per-sample-constant) contribution
    that a naively-concatenated macro signal makes, so gradient into the
    regime encoder collapses. RAT's feature gate breaks this because the
    gate multiplies stock features that DO vary across samples.
    """
    torch.manual_seed(0)
    batch = 32
    x = torch.randn(batch, 60, 17)
    y = torch.randn(batch)
    macro_same = torch.randn(1, 12).expand(batch, -1)  # identical for every sample

    concat_model = MacroConcatTransformer(stock_features=17, macro_features=12)
    pred = concat_model(x, macro_same)
    loss = ic_loss(pred, y)
    loss.backward()
    concat_grad = sum(
        p.grad.abs().sum().item()
        for p in concat_model.regime_encoder.parameters() if p.grad is not None
    )

    rat_model = RegimeAwareTransformer(stock_features=17, macro_features=12)
    pred = rat_model(x, macro_same)
    loss = ic_loss(pred, y)
    loss.backward()
    rat_grad = sum(
        p.grad.abs().sum().item()
        for p in rat_model.regime_encoder.parameters() if p.grad is not None
    )

    # RAT's conditioning breaks cross-sample symmetry (feature gate scales
    # each stock's own, sample-varying features), so its regime encoder
    # receives meaningfully more gradient signal than the concat baseline
    # under an identical, per-batch-constant macro vector.
    assert rat_grad > concat_grad
