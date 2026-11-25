import math

import torch

from tutorlib import load_debug

m = load_debug(__file__)
N_EMBD, HS, BLOCK = 64, 16, 16


def test_attention_scores_have_unit_variance_at_init():
    torch.manual_seed(0)
    h = m.Head(N_EMBD, HS, BLOCK)
    with torch.no_grad():
        for lin in (h.query, h.key):
            lin.weight.normal_(0, 1 / math.sqrt(N_EMBD))
    x = torch.randn(64, BLOCK, N_EMBD)
    with torch.no_grad():
        var = h.scores(x).var().item()
    assert 0.7 < var < 1.4, f"score variance at init is {var:.3f}; with unit-variance q and k it should stay ~1"
