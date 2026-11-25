import torch

from tutorlib import load_for_test, load_module

M = load_for_test(__file__)
ATT = load_module("04_self_attention")  # your (or the reference) module-04 attention
N, H, BLOCK = 32, 4, 8


def make(p=0.0):
    torch.manual_seed(0)
    return M.BatchedMultiHeadAttention(N, H, BLOCK, p)


def x_(B=3, T=6, seed=1):
    torch.manual_seed(seed)
    return torch.randn(B, T, N)


def test_shape_and_param_count():
    m = make()
    assert m(x_()).shape == (3, 6, N)
    assert sum(p.numel() for p in m.parameters()) == 3 * N * N + N * N + N


def test_is_causal_and_batch_independent():
    m = make().eval()
    x = x_(T=8)
    x2 = x.clone()
    x2[:, 5:] = torch.randn_like(x2[:, 5:]) * 5
    assert torch.allclose(m(x)[:, :5], m(x2)[:, :5], atol=1e-5), "future tokens leak into earlier outputs"
    x3 = x.clone()
    x3[1:] = torch.randn_like(x3[1:])
    assert torch.allclose(m(x)[0], m(x3)[0], atol=1e-6), "information crossed the batch dimension"


def test_equals_module_04_attention_when_weights_are_copied():
    torch.manual_seed(0)
    ref = ATT.MultiHeadAttention(N, H, BLOCK).eval()
    m = make().eval()
    hs = N // H
    with torch.no_grad():
        q = torch.cat([h.query.weight for h in ref.heads], 0)
        k = torch.cat([h.key.weight for h in ref.heads], 0)
        v = torch.cat([h.value.weight for h in ref.heads], 0)
        m.qkv.weight.copy_(torch.cat([q, k, v], 0))
        m.proj.weight.copy_(ref.proj.weight)
        m.proj.bias.copy_(ref.proj.bias)
    x = x_()
    assert torch.allclose(m(x), ref(x), atol=1e-5), f"max deviation from the looped implementation: {(m(x) - ref(x)).abs().max().item():.3e}"


def test_dropout_modes():
    m = make(p=0.5)
    x = x_()
    m.eval()
    assert torch.equal(m(x), m(x))
    m.train()
    assert not torch.equal(m(x), m(x))
