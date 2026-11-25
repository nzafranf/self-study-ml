import torch
import torch.nn.functional as F

from tutorlib import load_debug

m = load_debug(__file__)


def make():
    torch.manual_seed(0)
    return m.GPT(m.GPTConfig(vocab_size=11, block_size=8, n_embd=32, n_head=4, n_layer=2))


def test_reported_loss_matches_cross_entropy_of_returned_logits():
    model = make()
    x, y = torch.randint(11, (4, 8)), torch.randint(11, (4, 8))
    logits, loss = model(x, y)
    ref = F.cross_entropy(logits, y.reshape(-1))
    assert torch.allclose(loss, ref, atol=1e-5), f"loss {loss.item():.4f} != scoring the returned logits against the targets ({ref.item():.4f})"


def test_trained_model_actually_predicts_its_targets():
    model = make()
    x, y = torch.randint(11, (4, 8)), torch.randint(11, (4, 8))
    opt = torch.optim.AdamW(model.parameters(), lr=5e-3)
    for _ in range(200):
        _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        logits, _ = model(x, y)
    acc = (logits.argmax(-1) == y.reshape(-1)).float().mean().item()
    assert acc > 0.9, f"after training on one memorisable batch (final loss {loss.item():.3f}) only {acc:.0%} of positions are predicted correctly"
