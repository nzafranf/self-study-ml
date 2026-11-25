import torch

from tutorlib import load_debug

m = load_debug(__file__)


def test_same_token_at_different_positions_is_distinguishable():
    torch.manual_seed(0)
    model = m.GPT(m.GPTConfig(vocab_size=11, block_size=8, n_embd=32, n_head=4, n_layer=2)).eval()
    logits, _ = model(torch.full((1, 8), 3))
    assert not torch.allclose(logits[0, 0], logits[0, 1], atol=1e-5), "identical tokens at positions 0 and 1 produced identical logits: position is invisible to the model"


def test_position_embeddings_receive_gradient():
    torch.manual_seed(0)
    model = m.GPT(m.GPTConfig(vocab_size=11, block_size=8, n_embd=32, n_head=4, n_layer=2))
    x = torch.randint(11, (4, 8))
    _, loss = model(x, torch.randint(11, (4, 8)))
    loss.backward()
    g = model.position_embedding_table.weight.grad
    assert g is not None and g.abs().sum() > 0, "position_embedding_table never receives a gradient"
