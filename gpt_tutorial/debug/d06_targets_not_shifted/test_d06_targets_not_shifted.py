import math

import torch

from tutorlib import load_debug

m = load_debug(__file__)


def test_loss_on_unpredictable_data_cannot_beat_chance():
    # tokens are i.i.d. random, so nothing about the past predicts the next token: loss must stay >= ~ln(V)
    V, T = 16, 16
    torch.manual_seed(0)
    train, val = torch.randint(V, (2000,)), torch.randint(V, (2000,))
    model = m.GPT(m.GPTConfig(vocab_size=V, block_size=T, n_embd=32, n_head=4, n_layer=2))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for _ in range(150):
        x, y = m.get_batch(train, 32, T)
        _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        x, y = m.get_batch(val, 256, T)
        val_loss = model(x, y)[1].item()
    assert val_loss > 0.9 * math.log(V), f"validation loss {val_loss:.3f} is below chance ({math.log(V):.3f}) on random data: the task is leaking its own answers"


def test_batches_have_next_token_targets():
    data = torch.arange(100)
    x, y = m.get_batch(data, 4, 8)
    assert torch.equal(y, x + 1), "targets should be the inputs shifted by one position"
