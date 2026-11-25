import torch

from tutorlib import load_debug

m = load_debug(__file__)


def test_each_token_is_normalised_over_its_own_features():
    torch.manual_seed(0)
    ln = m.LayerNorm1d(32)
    y = ln(torch.randn(4, 6, 32) * 3 + 2)
    assert y.mean(-1).abs().max() < 1e-4, f"per-token feature mean should be 0 (worst {y.mean(-1).abs().max():.3f})"
    assert (y.std(-1) - 1).abs().max() < 0.05, "per-token feature std should be ~1"


def test_examples_do_not_influence_each_other():
    torch.manual_seed(0)
    ln = m.LayerNorm1d(32)
    x = torch.randn(4, 6, 32)
    x2 = x.clone()
    x2[1:] = torch.randn_like(x2[1:]) * 9
    assert torch.allclose(ln(x)[0], ln(x2)[0], atol=1e-5), "example 0's output depends on the other examples in the batch"
