import pytest
import torch

from tutorlib import load_for_test

M = load_for_test(__file__)
FNS = ["bow_loop", "bow_matmul", "bow_softmax"]


def sample_x(B=4, T=8, C=3):
    torch.manual_seed(1337)
    return torch.randn(B, T, C)


def reference(x):
    T = x.shape[1]
    return torch.cumsum(x, dim=1) / torch.arange(1, T + 1).view(1, T, 1)


@pytest.mark.parametrize("name", FNS)
def test_matches_running_mean(name):
    x = sample_x()
    out = getattr(M, name)(x)
    assert out.shape == x.shape, f"{name}: expected shape {tuple(x.shape)}, got {tuple(out.shape)}"
    assert torch.allclose(out, reference(x), atol=1e-5), f"{name}: out[b,t] should be the mean of x[b,0..t]"


@pytest.mark.parametrize("name", FNS)
def test_future_tokens_do_not_influence_earlier_outputs(name):
    fn = getattr(M, name)
    x = sample_x(T=10)
    x2 = x.clone()
    x2[:, 6:] += 100 * torch.randn_like(x2[:, 6:])  # edit positions 6..9 only
    a, b = fn(x), fn(x2)
    assert torch.allclose(a[:, :6], b[:, :6], atol=1e-5), (
        f"{name}: outputs at positions 0..5 changed when only positions 6..9 were edited (information flowed backwards in time)"
    )


@pytest.mark.parametrize("name", FNS)
def test_first_position_only_sees_itself(name):
    x = sample_x()
    out = getattr(M, name)(x)
    assert torch.allclose(out[:, 0], x[:, 0], atol=1e-6), f"{name}: position 0 has no past, so its output should equal its input"


def test_three_versions_agree():
    x = sample_x(B=3, T=12, C=5)
    v1, v2, v3 = M.bow_loop(x), M.bow_matmul(x), M.bow_softmax(x)
    assert torch.allclose(v1, v2, atol=1e-5), "version 1 (loops) and version 2 (matmul) disagree"
    assert torch.allclose(v1, v3, atol=1e-5), "version 1 (loops) and version 3 (softmax) disagree"


@pytest.mark.parametrize("T", [1, 4, 9])
def test_causal_weights_are_lower_triangular_and_normalised(T):
    w = M.causal_weights(T)
    assert w.shape == (T, T)
    assert not torch.isnan(w).any(), "weights contain NaN (check how the masked entries are handled before softmax)"
    assert torch.allclose(w.sum(-1), torch.ones(T), atol=1e-6), "every row of the weight matrix must sum to 1"
    assert torch.equal(torch.triu(w, diagonal=1), torch.zeros(T, T)), "weights above the diagonal must be exactly 0"


def test_causal_weights_are_uniform_over_the_allowed_past():
    w = M.causal_weights(5)
    for t in range(5):
        assert torch.allclose(w[t, : t + 1], torch.full((t + 1,), 1 / (t + 1)), atol=1e-6), f"row {t} should average positions 0..{t} equally"
