import torch

from tutorlib import load_debug

m = load_debug(__file__)


def test_weights_are_lower_triangular_and_normalised():
    w = m.causal_weights(6)
    assert torch.allclose(w.sum(-1), torch.ones(6), atol=1e-6), "rows must sum to 1"
    assert torch.equal(torch.triu(w, diagonal=1), torch.zeros(6, 6)), "a token is allowed to look at positions after itself"


def test_early_outputs_do_not_depend_on_later_tokens():
    torch.manual_seed(0)
    x = torch.randn(2, 8, 4)
    x2 = x.clone()
    x2[:, 5:] += 10 * torch.randn_like(x2[:, 5:])
    assert torch.allclose(m.bow_softmax(x)[:, :5], m.bow_softmax(x2)[:, :5], atol=1e-5), "outputs at positions 0..4 changed when only 5..7 were edited"
