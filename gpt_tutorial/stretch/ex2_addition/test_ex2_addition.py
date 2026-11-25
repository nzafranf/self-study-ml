import re

import pytest
import torch

from tutorlib import load_for_test, load_module

M = load_for_test(__file__)


def test_example_format():
    assert M.make_example(12, 34, 2) == "12+34=640"
    assert M.make_example(5, 7, 1) == "5+7=21"  # 12 -> "12" reversed "21"
    assert M.make_example(99, 99, 2) == "99+99=891"  # 198 reversed
    assert M.make_example(0, 0, 3) == "000+000=0000"


def test_answer_roundtrip():
    for a, b in [(0, 0), (12, 34), (99, 99), (7, 5)]:
        n = 2
        assert M.answer_value(M.make_example(a, b, n)) == a + b


def test_dataset_is_reproducible_well_formed_and_correct():
    d1, d2, d3 = M.build_dataset(2, 50, seed=1), M.build_dataset(2, 50, seed=1), M.build_dataset(2, 50, seed=2)
    assert d1 == d2 and d1 != d3
    for s in d1:
        m = re.fullmatch(r"(\d\d)\+(\d\d)=(\d{3})", s)
        assert m, f"badly formed example {s!r}"
        assert M.answer_value(s) == int(m.group(1)) + int(m.group(2))
    assert len({len(s) for s in d1}) == 1, "all examples must have the same length"


@pytest.mark.slow
def test_tiny_gpt_learns_one_digit_addition_table():
    """Optional (pytest stretch -m slow): uses your module 06 to memorise/learn the 1-digit table."""
    G = load_module("06_full_gpt")
    data = sorted(set(M.build_dataset(1, 2000, seed=0)))  # all 100 distinct pairs
    chars = sorted(set("".join(data)))
    stoi = {c: i for i, c in enumerate(chars)}
    ids = torch.tensor([[stoi[c] for c in s] for s in data])  # (N, T)
    T = ids.shape[1]
    torch.manual_seed(0)
    model = G.GPTLanguageModel(G.GPTConfig(vocab_size=len(chars), block_size=T, n_embd=48, n_head=4, n_layer=2))

    def get_batch(split):
        rows = ids[torch.randint(len(ids), (64,))]
        return rows[:, :-1].contiguous(), rows[:, 1:].contiguous()  # .view() in the model needs contiguous tensors

    G.train_model(model, get_batch, max_iters=700, learning_rate=3e-3)
    model.eval()
    prompts = ids[:, :4]  # "a+b="
    out = model.generate(prompts, 2)
    correct = (out[:, 4:] == ids[:, 4:]).all(1).float().mean().item()
    assert correct > 0.9, f"only {correct:.0%} of the 100 additions came out right"
