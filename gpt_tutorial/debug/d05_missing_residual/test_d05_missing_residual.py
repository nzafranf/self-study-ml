import torch

from tutorlib import load_debug

m = load_debug(__file__)


def test_block_is_identity_when_sublayers_are_zeroed():
    torch.manual_seed(0)
    b = m.Block(32, 4, 8).eval()
    with torch.no_grad():
        for p in list(b.sa.parameters()) + list(b.ffwd.parameters()):
            p.zero_()
    x = torch.randn(2, 6, 32)
    assert torch.allclose(b(x), x, atol=1e-6), f"block should pass x straight through when its sub-layers output zero (max deviation {(b(x) - x).abs().max().item():.3e})"
