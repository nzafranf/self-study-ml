import torch
import torch.nn.functional as F

from tutorlib import load_debug

m = load_debug(__file__)


def head():
    torch.manual_seed(0)
    return m.Head(32, 8, 8).eval()


def test_head_is_causal():
    h = head()
    x = torch.randn(3, 8, 32)
    x2 = x.clone()
    x2[:, 5:] = torch.randn_like(x2[:, 5:]) * 5
    assert torch.allclose(h(x)[:, :5], h(x2)[:, :5], atol=1e-5), "positions 0..4 changed when only 5..7 were edited"


def test_matches_reference_causal_attention():
    h = head()
    x = torch.randn(3, 8, 32)
    with torch.no_grad():
        ref = F.scaled_dot_product_attention(h.query(x), h.key(x), h.value(x), is_causal=True)
    assert torch.allclose(h(x), ref, atol=1e-5), f"max deviation from reference attention: {(h(x) - ref).abs().max().item():.3e}"
