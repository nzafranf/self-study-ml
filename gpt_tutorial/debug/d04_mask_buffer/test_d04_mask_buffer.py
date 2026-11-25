import torch

from tutorlib import load_debug

m = load_debug(__file__)


def test_mask_is_registered_module_state():
    h = m.Head(32, 8, 8)
    assert len(list(h.buffers())) >= 1, "the causal mask is not a buffer, so .to(device)/.double() will not move it"
    assert "tril" in h.state_dict(), "the causal mask is not part of state_dict()"
    assert len(list(h.parameters())) == 3, "the mask must not become a trainable parameter either"


def test_mask_follows_dtype_conversion():
    h = m.Head(32, 8, 8).double()
    assert h.tril.dtype == torch.float64, "after .double() the mask should have been converted with the rest of the module"
