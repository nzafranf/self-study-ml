import torch

from tutorlib import load_debug

m = load_debug(__file__)


def test_eval_mode_is_deterministic():
    torch.manual_seed(0)
    model = m.GPT(m.GPTConfig(vocab_size=11, block_size=8, n_embd=32, n_head=4, n_layer=2, dropout=0.3))
    x = torch.randint(11, (4, 8))
    model.eval()
    assert torch.equal(model(x)[0], model(x)[0]), "two eval-mode forward passes on the same input differ"


def test_train_mode_is_stochastic():
    torch.manual_seed(0)
    model = m.GPT(m.GPTConfig(vocab_size=11, block_size=8, n_embd=32, n_head=4, n_layer=2, dropout=0.3))
    x = torch.randint(11, (4, 8))
    model.train()
    assert not torch.equal(model(x)[0], model(x)[0])
