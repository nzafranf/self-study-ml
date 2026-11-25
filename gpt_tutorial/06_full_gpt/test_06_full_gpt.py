import copy
import math

import torch

from tutorlib import SAMPLE_TEXT, load_for_test

M = load_for_test(__file__)
V, BLOCK = 11, 8


def make(dropout=0.0, seed=0, **kw):
    torch.manual_seed(seed)
    cfg = M.GPTConfig(vocab_size=V, block_size=BLOCK, n_embd=32, n_head=4, n_layer=2, dropout=dropout, **kw)
    return M.GPTLanguageModel(cfg)


def tokens(B=4, T=BLOCK, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(V, (B, T), generator=g), torch.randint(V, (B, T), generator=g)


def test_shapes_with_and_without_targets():
    m = make()
    x, y = tokens()
    logits, loss = m(x)
    assert logits.shape == (4, BLOCK, V) and loss is None
    logits, loss = m(x, y)
    assert logits.shape == (4 * BLOCK, V) and loss.ndim == 0
    logits, _ = m(x[:, :3])
    assert logits.shape == (4, 3, V), "shorter-than-block_size inputs must work"


def test_submodule_structure_and_parameter_count():
    m = make()
    assert len(m.blocks) == 2, "blocks should contain n_layer Blocks"
    expected = 26251  # V=11, block=8, n_embd=32, n_head=4, n_layer=2
    got = sum(p.numel() for p in m.parameters())
    assert got == expected, (
        f"parameter count is {got}, expected {expected}. Tally: embeddings, per-block (attention + proj + MLP + 2 layernorms), "
        "final layernorm, lm_head (with bias)."
    )


def test_initial_loss_is_near_ln_vocab():
    m = make()
    x, y = tokens(B=32)
    _, loss = m(x, y)
    assert abs(loss.item() - math.log(V)) < 0.6, f"initial loss {loss.item():.3f} vs ln(V)={math.log(V):.3f}"


def test_reported_loss_matches_scoring_the_returned_logits():
    m = make().eval()
    x, y = tokens()
    logits, loss = m(x, y)
    ref = torch.nn.functional.cross_entropy(logits, y.reshape(-1))
    assert torch.allclose(loss, ref, atol=1e-5), (
        f"loss {loss.item():.4f} differs from scoring the returned logits against the targets ({ref.item():.4f}); "
        "are logits and targets paired position-by-position?"
    )


def test_final_layernorm_feeds_the_output_head():
    m = make().eval()
    seen = []
    m.lm_head.register_forward_hook(lambda mod, inp, out: seen.append(inp[0].detach()))
    x, _ = tokens()
    m(x)
    h = seen[0]
    assert h.mean(-1).abs().max() < 1e-4 and (h.std(-1) - 1).abs().max() < 0.05, (
        "the input to lm_head should be layer-normalised per token (mean ~0, std ~1 at init): "
        "is the final layer norm applied between the blocks and the head?"
    )


def test_position_information_reaches_the_output():
    m = make().eval()
    x = torch.full((1, BLOCK), 3)  # same token everywhere
    logits, _ = m(x)
    assert not torch.allclose(logits[0, 0], logits[0, 1], atol=1e-5), (
        "identical tokens at different positions produced identical outputs: the model cannot tell positions apart"
    )


def test_model_is_causal():
    m = make().eval()
    x, _ = tokens(B=2)
    x2 = x.clone()
    x2[:, 5:] = (x2[:, 5:] + 1) % V
    a, _ = m(x)
    b, _ = m(x2)
    assert torch.allclose(a[:, :5], b[:, :5], atol=1e-5), "logits at positions 0..4 changed when only tokens 5..7 changed"


def test_eval_is_deterministic_train_with_dropout_is_not():
    m = make(dropout=0.3)
    x, _ = tokens()
    m.eval()
    assert torch.equal(m(x)[0], m(x)[0]), "eval mode must be deterministic (dropout should switch off)"
    m.train()
    assert not torch.equal(m(x)[0], m(x)[0]), "train mode with dropout=0.3 should be stochastic"


def test_generate_shape_prefix_and_range():
    m = make().eval()
    idx = torch.tensor([[1, 2, 3]])
    out = m.generate(idx, 5)
    assert out.shape == (1, 8) and torch.equal(out[:, :3], idx)
    assert out.min() >= 0 and out.max() < V


def test_generate_works_far_beyond_block_size_and_only_uses_recent_context():
    m = make().eval()
    idx = torch.randint(V, (2, BLOCK))
    out = m.generate(idx, 3 * BLOCK)  # must not raise
    assert out.shape == (2, 4 * BLOCK)
    long_prompt = torch.randint(V, (1, 3 * BLOCK))
    torch.manual_seed(7)
    a = m.generate(long_prompt, 4)
    torch.manual_seed(7)
    b = m.generate(long_prompt[:, -BLOCK:], 4)
    assert torch.equal(a[:, -4:], b[:, -4:]), "the model should condition on exactly the last block_size tokens, no more"


def test_can_overfit_a_single_batch():
    m = make(seed=1)
    x, y = tokens(B=4, seed=3)
    losses = M.train_model(m, lambda split: (x, y), max_iters=200, learning_rate=5e-3)
    assert len(losses) == 200 and all(isinstance(l, float) for l in losses)
    assert losses[0] > 1.5, f"first loss {losses[0]:.2f} suspiciously low for random labels"
    assert losses[-1] < 0.15, (
        f"after 200 steps on ONE fixed batch the loss is {losses[-1]:.3f}; a correct GPT memorises 32 tokens easily. "
        "If this fails, something in the forward/backward path is broken."
    )


def test_train_model_follows_the_standard_recipe():
    m1 = make(seed=2)
    m2 = copy.deepcopy(m1)
    x, y = tokens(seed=5)
    xs, ys = tokens(B=4, seed=6)

    def fn_factory():
        state = {"i": 0}

        def fn(split):
            state["i"] += 1
            return (x, y) if state["i"] % 2 else (xs, ys)

        return fn

    M.train_model(m1, fn_factory(), max_iters=6, learning_rate=1e-3)
    ref_fn, opt = fn_factory(), torch.optim.AdamW(m2.parameters(), lr=1e-3)
    for _ in range(6):
        xb, yb = ref_fn("train")
        _, loss = m2(xb, yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    for (n, a), (_, b) in zip(m1.named_parameters(), m2.named_parameters()):
        assert torch.allclose(a, b, atol=1e-6), f"parameter {n} differs from a standard AdamW loop (gradients cleared each step? optimizer stepped?)"


def test_loss_decreases_on_real_text():
    chars = sorted(set(SAMPLE_TEXT))
    stoi = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in SAMPLE_TEXT])
    torch.manual_seed(0)
    m = M.GPTLanguageModel(M.GPTConfig(vocab_size=len(chars), block_size=16, n_embd=32, n_head=4, n_layer=2))

    def get_batch(split):
        ix = torch.randint(len(data) - 16, (16,))
        return torch.stack([data[i : i + 16] for i in ix]), torch.stack([data[i + 1 : i + 17] for i in ix])

    losses = M.train_model(m, get_batch, max_iters=80, learning_rate=3e-3)
    assert sum(losses[-10:]) / 10 < 0.8 * sum(losses[:10]) / 10, "training loss barely moved in 80 steps"


def test_estimate_loss_contract():
    m = make(dropout=0.2)
    m.train()
    calls, seen = [], []
    x, y = tokens()

    def fn(split):
        calls.append(split)
        return x, y

    m.register_forward_hook(lambda mod, i, o: seen.append((mod.training, torch.is_grad_enabled())))
    out = M.estimate_loss(m, fn, eval_iters=3)
    assert set(out) == {"train", "val"}
    assert sorted(set(calls)) == ["train", "val"] and len(calls) == 6, "each split must be sampled eval_iters times"
    assert all(training is False for training, _ in seen), "evaluation must run in eval mode (dropout off)"
    assert all(grad is False for _, grad in seen), "evaluation should not track gradients"
    assert m.training is True, "the model must be switched back to train mode afterwards"
    m.eval()
    ref = m(x, y)[1].item()
    assert abs(float(out["train"]) - ref) < 1e-5, "with a fixed batch and dropout off, the estimate must equal that batch's loss"
