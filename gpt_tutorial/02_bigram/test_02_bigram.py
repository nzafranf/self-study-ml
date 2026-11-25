import math

import torch
import torch.nn.functional as F

from tutorlib import load_for_test

M = load_for_test(__file__)
V = 65


def make(vocab=V, seed=1337):
    torch.manual_seed(seed)
    return M.BigramLanguageModel(vocab)


def batch(B=4, T=8, vocab=V, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(vocab, (B, T), generator=g), torch.randint(vocab, (B, T), generator=g)


def test_logits_shape_without_targets():
    m = make()
    x, _ = batch()
    logits, loss = m(x)
    assert logits.shape == (4, 8, V), f"expected (B,T,V)=(4,8,{V}), got {tuple(logits.shape)}"
    assert loss is None, "loss must be None when no targets are given"


def test_logits_flattened_and_scalar_loss_with_targets():
    m = make()
    x, y = batch()
    logits, loss = m(x, y)
    assert logits.shape == (32, V), f"with targets logits should be flattened to (B*T,V)=(32,{V}), got {tuple(logits.shape)}"
    assert loss.ndim == 0, "loss must be a scalar"


def test_initial_loss_is_close_to_uniform_guess():
    m = make()
    x, y = batch(B=32, T=8)
    _, loss = m(x, y)
    expected = math.log(V)
    assert 0.8 * expected < loss.item() < expected + 1.2, (
        f"initial loss {loss.item():.3f} is far from ln(V)={expected:.3f}; "
        "an untrained model should be roughly 'uniformly confused'"
    )


def test_loss_matches_cross_entropy_of_the_right_pairs():
    m = make()
    x, y = batch()
    logits3d, _ = m(x)
    _, loss = m(x, y)
    # per-position reference: logits[b,t] must be scored against y[b,t]
    ref = torch.stack([F.cross_entropy(logits3d[b, t][None], y[b, t][None]) for b in range(4) for t in range(8)]).mean()
    assert torch.allclose(loss, ref, atol=1e-5), (
        f"loss {loss.item():.4f} != per-position cross-entropy {ref.item():.4f}; "
        "logits and targets are probably not flattened consistently"
    )


def test_gradient_reaches_embedding_table():
    m = make()
    x, y = batch()
    _, loss = m(x, y)
    loss.backward()
    g = m.token_embedding_table.weight.grad
    assert g is not None and g.abs().sum() > 0


def test_generate_shape_prefix_and_range():
    m = make()
    idx = torch.tensor([[1, 2, 3], [4, 5, 6]])
    out = m.generate(idx, max_new_tokens=10)
    assert out.shape == (2, 13), f"expected (2, 13), got {tuple(out.shape)}"
    assert torch.equal(out[:, :3], idx), "generate must keep the original context as the prefix"
    assert out.min() >= 0 and out.max() < V, "generated ids must be valid tokens"


def test_generate_is_seed_deterministic_and_actually_samples():
    m = make()
    idx = torch.zeros((1, 1), dtype=torch.long)
    torch.manual_seed(0)
    a = m.generate(idx, 200)
    torch.manual_seed(0)
    b = m.generate(idx, 200)
    torch.manual_seed(1)
    c = m.generate(idx, 200)
    assert torch.equal(a, b), "same seed must give the same generation"
    assert not torch.equal(a, c), "different seeds gave identical text; are you sampling (multinomial)?"
    assert len(set(a[0].tolist())) > 8, "generation is nearly constant; sampling should produce variety"


def test_generate_conditions_on_the_last_token():
    m = make(vocab=10)
    with torch.no_grad():  # force the table so that token i is always followed by (i+1) % 10
        m.token_embedding_table.weight.copy_(50 * torch.roll(torch.eye(10), 1, dims=1))
    out = m.generate(torch.tensor([[0, 1, 2]]), max_new_tokens=5)
    assert out[0].tolist() == [0, 1, 2, 3, 4, 5, 6, 7], (
        f"with a deterministic table the continuation should be 3,4,5,6,7; got {out[0].tolist()[3:]}. "
        "Which time step's logits should predict the next token?"
    )


def test_training_drives_loss_down():
    torch.manual_seed(0)
    m = M.BigramLanguageModel(5)
    data = torch.arange(5).repeat(50)
    xb = torch.stack([data[s : s + 8] for s in range(5)])
    yb = torch.stack([data[s + 1 : s + 9] for s in range(5)])
    opt = torch.optim.AdamW(m.parameters(), lr=5e-2)
    first = None
    for _ in range(150):
        _, loss = m(xb, yb)
        first = loss.item() if first is None else first
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    assert loss.item() < 0.3 * first, f"loss only went {first:.3f} -> {loss.item():.3f} on a perfectly predictable pattern"
