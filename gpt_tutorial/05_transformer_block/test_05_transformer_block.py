import torch
import torch.nn as nn

from tutorlib import load_for_test

M = load_for_test(__file__)
N, H, BLOCK = 32, 4, 8


def rand_x(B=3, T=6, C=N, seed=0):
    torch.manual_seed(seed)
    return torch.randn(B, T, C)


# ---------------------------------------------------------------- FeedForward
def test_ffwd_shape_and_size():
    torch.manual_seed(0)
    f = M.FeedForward(N)
    assert f(rand_x()).shape == (3, 6, N)
    expected = (N * 4 * N + 4 * N) + (4 * N * N + N)
    got = sum(p.numel() for p in f.parameters())
    assert got == expected, f"expected {expected} params for an N->4N->N MLP, found {got}"


def test_ffwd_is_applied_per_token_independently():
    torch.manual_seed(0)
    f = M.FeedForward(N).eval()
    x = rand_x()
    full = f(x)
    single = f(x[:, 2:3])
    assert torch.allclose(full[:, 2:3], single, atol=1e-6), "a token's output depended on other positions; the feed-forward must act on each position alone"


def test_ffwd_has_a_nonlinearity():
    torch.manual_seed(0)
    f = M.FeedForward(N).eval()
    a, b = rand_x(seed=1), rand_x(seed=2)
    zero = torch.zeros_like(a)
    # an affine map satisfies f(a+b) - f(a) - f(b) + f(0) == 0; a real MLP does not
    gap = (f(a + b) - f(a) - f(b) + f(zero)).abs().max()
    assert gap > 1e-3, "the block is purely affine; is there an activation between the two Linear layers?"


# ---------------------------------------------------------------- LayerNorm1d
def test_layernorm_normalises_each_row_over_features():
    ln = M.LayerNorm1d(64)
    torch.manual_seed(0)
    x = torch.randn(32, 64) * 3 + 5
    y = ln(x)
    row_mean, row_std = y.mean(-1), y.std(-1)
    assert row_mean.abs().max() < 1e-4, f"each example's features should have mean 0 (worst row mean {row_mean.abs().max():.3f})"
    assert (row_std - 1).abs().max() < 0.03, f"each example's features should have std ~1 (got range {row_std.min():.3f}..{row_std.max():.3f})"


def test_layernorm_is_per_token_on_3d_input_and_batch_independent():
    ln = M.LayerNorm1d(N)
    x = rand_x() * 2 + 1
    y = ln(x)
    assert y.shape == x.shape
    assert y.mean(-1).abs().max() < 1e-4, "normalisation must be over the last (feature) dimension for every (b,t)"
    x2 = x.clone()
    x2[1:] = torch.randn_like(x2[1:]) * 7
    assert torch.allclose(ln(x2)[0], y[0], atol=1e-5), "example 0's output changed when other examples changed (batch-norm-like behaviour)"


def test_layernorm_has_learnable_gamma_beta():
    ln = M.LayerNorm1d(N)
    params = list(ln.parameters())
    assert sum(p.numel() for p in params) == 2 * N, "expected exactly gamma and beta (N values each) as nn.Parameters"
    with torch.no_grad():
        for p in params:
            if torch.allclose(p, torch.ones(N)):
                p.mul_(2)
            else:
                p.add_(3)
    y = ln(rand_x())
    assert abs(y.mean().item() - 3) < 1e-3, "beta should shift the normalised output"


# ---------------------------------------------------------------- Block
def make_block(p=0.0):
    torch.manual_seed(0)
    return M.Block(N, H, BLOCK, p)


def test_block_shape_and_submodule_names():
    b = make_block()
    assert b(rand_x()).shape == (3, 6, N)
    for name in ("sa", "ffwd", "ln1", "ln2"):
        assert hasattr(b, name), f"Block needs an attribute named `{name}`"
    assert isinstance(b.ln1, nn.LayerNorm) and isinstance(b.ln2, nn.LayerNorm)


def test_block_has_residual_paths():
    b = make_block().eval()
    with torch.no_grad():
        for p in list(b.sa.parameters()) + list(b.ffwd.parameters()):
            p.zero_()
    x = rand_x()
    out = b(x)
    assert torch.allclose(out, x, atol=1e-6), (
        "with the attention and feed-forward weights zeroed, the block should pass its input through unchanged "
        f"(max deviation {(out - x).abs().max().item():.3e}); is the residual connection there for BOTH sub-layers?"
    )


def test_block_is_causal():
    b = make_block().eval()
    x = rand_x(T=8)
    x2 = x.clone()
    x2[:, 5:] = torch.randn_like(x2[:, 5:]) * 5
    assert torch.allclose(b(x)[:, :5], b(x2)[:, :5], atol=1e-5), "block output at positions 0..4 changed when only 5..7 were edited"


def test_every_parameter_receives_a_gradient():
    b = make_block()
    (b(rand_x()) ** 2).sum().backward()
    missing = [n for n, p in b.named_parameters() if p.grad is None]
    assert not missing, f"parameters with no gradient: {missing}"
    assert len(list(b.parameters())) >= 4 * 3 + 2 + 4 + 4, "too few parameters: are all sub-layers registered as submodules?"
