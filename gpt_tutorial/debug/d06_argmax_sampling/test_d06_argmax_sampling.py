import torch

from tutorlib import load_debug

m = load_debug(__file__)


def test_generation_depends_on_the_random_seed():
    torch.manual_seed(0)
    model = m.GPT(m.GPTConfig(vocab_size=11, block_size=8, n_embd=32, n_head=4, n_layer=2)).eval()
    start = torch.zeros((1, 1), dtype=torch.long)
    torch.manual_seed(1)
    a = model.generate(start, 40)
    torch.manual_seed(2)
    b = model.generate(start, 40)
    assert not torch.equal(a, b), "different random seeds produced identical text: generation is not sampling"
