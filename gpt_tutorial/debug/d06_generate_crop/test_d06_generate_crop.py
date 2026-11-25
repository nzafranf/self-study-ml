import torch

from tutorlib import load_debug

m = load_debug(__file__)


def test_generate_far_beyond_block_size():
    torch.manual_seed(0)
    model = m.GPT(m.GPTConfig(vocab_size=11, block_size=8, n_embd=32, n_head=4, n_layer=2)).eval()
    out = model.generate(torch.zeros((1, 1), dtype=torch.long), 40)
    assert out.shape == (1, 41)
