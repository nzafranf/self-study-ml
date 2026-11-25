import math

import pytest
import torch
import torch.nn.functional as F

from tutorlib import load_for_test

M = load_for_test(__file__)
N_EMBD, HS, BLOCK = 64, 16, 16


def make_head(p=0.0, seed=0):
    torch.manual_seed(seed)
    return M.Head(N_EMBD, HS, BLOCK, p)


def make_mha(n_head=4, p=0.0, seed=0):
    torch.manual_seed(seed)
    return M.MultiHeadAttention(N_EMBD, n_head, BLOCK, p)


def rand_x(B=3, T=10, seed=1):
    torch.manual_seed(seed)
    return torch.randn(B, T, N_EMBD)


# ---------------------------------------------------------------- Head
@pytest.mark.parametrize("T", [1, 5, BLOCK])
def test_head_output_shape_for_any_T_up_to_block_size(T):
    out = make_head()(rand_x(T=T))
    assert out.shape == (3, T, HS), f"expected (3,{T},{HS}), got {tuple(out.shape)}"


def test_projections_have_no_bias_and_right_shapes():
    h = make_head()
    for name in ("key", "query", "value"):
        lin = getattr(h, name)
        assert lin.bias is None, f"{name} should be bias-free"
        assert tuple(lin.weight.shape) == (HS, N_EMBD)


def test_mask_is_state_but_not_a_parameter():
    h = make_head()
    assert len(list(h.buffers())) >= 1, "the causal mask should be registered as a buffer"
    n_params = sum(p.numel() for p in h.parameters())
    assert n_params == 3 * N_EMBD * HS, f"a head should have exactly 3 weight matrices of learnable params, found {n_params} values"


def test_head_is_causal():
    h = make_head().eval()
    x = rand_x(T=12)
    x2 = x.clone()
    x2[:, 7:] = torch.randn_like(x2[:, 7:]) * 10
    a, b = h(x), h(x2)
    assert torch.allclose(a[:, :7], b[:, :7], atol=1e-5), "outputs at positions 0..6 changed when only positions 7..11 were edited"
    assert not torch.allclose(a[:, 7:], b[:, 7:], atol=1e-5), "editing positions 7..11 should change their own outputs"


def test_head_matches_reference_scaled_dot_product_attention():
    h = make_head().eval()
    x = rand_x()
    with torch.no_grad():
        ref = F.scaled_dot_product_attention(h.query(x), h.key(x), h.value(x), is_causal=True)
        out = h(x)
    assert torch.allclose(out, ref, atol=1e-5), (
        f"max abs deviation from reference causal attention: {(out - ref).abs().max().item():.3e}. "
        "Check the mask, the softmax axis and the scaling."
    )


def test_scores_shape_and_scaling_uses_head_size():
    h = make_head()
    with torch.no_grad():  # unit-variance q and k so only the scaling decides the variance
        for lin in (h.query, h.key):
            lin.weight.normal_(0, 1 / math.sqrt(N_EMBD))
    torch.manual_seed(5)
    x = torch.randn(64, BLOCK, N_EMBD)
    with torch.no_grad():
        s = h.scores(x)
    assert s.shape == (64, BLOCK, BLOCK), f"scores should be (B,T,T), got {tuple(s.shape)}"
    var = s.var().item()
    assert 0.7 < var < 1.4, (
        f"variance of the attention scores at init is {var:.2f}; with unit-variance q,k it should be ~1 "
        f"(unscaled would be ~{HS}, dividing by sqrt(C={N_EMBD}) would give ~{HS / N_EMBD:.2f})"
    )


def test_no_information_crosses_the_batch_dimension():
    h = make_head().eval()
    x = rand_x(B=4)
    x2 = x.clone()
    x2[1:] = torch.randn_like(x2[1:])
    assert torch.allclose(h(x)[0], h(x2)[0], atol=1e-6), "example 0's output changed when other examples in the batch changed"


def test_attention_dropout_only_active_in_train_mode():
    h = make_head(p=0.5)
    x = rand_x()
    h.eval()
    assert torch.equal(h(x), h(x)), "eval mode must be deterministic"
    h.train()
    assert not torch.equal(h(x), h(x)), "train mode with dropout=0.5 should be stochastic"


# ---------------------------------------------------------------- MultiHeadAttention
def test_mha_shape_and_registration():
    m = make_mha(n_head=4)
    assert m(rand_x()).shape == (3, 10, N_EMBD)
    assert len(m.heads) == 4
    expected = 4 * 3 * N_EMBD * (N_EMBD // 4) + N_EMBD * N_EMBD + N_EMBD
    got = sum(p.numel() for p in m.parameters())
    assert got == expected, f"expected {expected} trainable values (heads + projection), found {got}. Are the heads registered as submodules?"


def test_mha_projection_shape():
    m = make_mha()
    assert tuple(m.proj.weight.shape) == (N_EMBD, N_EMBD)


def test_mha_equals_concatenation_of_independent_heads():
    m = make_mha().eval()
    with torch.no_grad():
        m.proj.weight.copy_(torch.eye(N_EMBD))
        m.proj.bias.zero_()
    x = rand_x()
    with torch.no_grad():
        expected = torch.cat([h(x) for h in m.heads], dim=-1)
        out = m(x)
    assert torch.allclose(out, expected, atol=1e-6), "with an identity projection the output must be the heads' outputs concatenated along channels"


def test_mha_heads_are_different():
    m = make_mha()
    w = [h.key.weight for h in m.heads]
    assert not torch.equal(w[0], w[1]), "all heads were initialised identically"


def test_mha_is_causal():
    m = make_mha().eval()
    x = rand_x(T=12)
    x2 = x.clone()
    x2[:, 6:] = torch.randn_like(x2[:, 6:]) * 10
    assert torch.allclose(m(x)[:, :6], m(x2)[:, :6], atol=1e-5), "multi-head attention leaks information from future positions"
