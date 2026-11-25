import torch

from tutorlib import SAMPLE_TEXT, load_for_test

M = load_for_test(__file__)


def test_vocab_is_sorted_unique_characters():
    chars = M.build_vocab("hello world")
    assert list(chars) == sorted(set("hello world")), (
        f"vocab should be the sorted unique characters, got {chars!r}"
    )


def test_encode_returns_python_ints_in_vocab_range():
    chars = M.build_vocab(SAMPLE_TEXT)
    encode, _ = M.make_codec(chars)
    ids = encode(SAMPLE_TEXT)
    assert isinstance(ids, list) and all(isinstance(i, int) for i in ids), "encode must return list[int]"
    assert 0 <= min(ids) and max(ids) < len(chars), "token ids must lie in [0, vocab_size)"


def test_encode_uses_vocab_order():
    chars = M.build_vocab(SAMPLE_TEXT)
    encode, _ = M.make_codec(chars)
    assert encode("a") == [chars.index("a")]
    assert encode("ab") == [chars.index("a"), chars.index("b")]


def test_encode_decode_roundtrip():
    chars = M.build_vocab(SAMPLE_TEXT)
    encode, decode = M.make_codec(chars)
    assert decode(encode(SAMPLE_TEXT)) == SAMPLE_TEXT, "decode(encode(s)) != s"
    ids = torch.tensor(encode("Shall I compare"))
    assert decode(ids.tolist()) == "Shall I compare"


def test_split_sizes_and_contiguity():
    data = torch.arange(1000)
    train, val = M.split_data(data)
    assert (len(train), len(val)) == (900, 100), f"90/10 split of 1000 should be 900/100, got {len(train)}/{len(val)}"
    assert torch.equal(torch.cat([train, val]), data), "train followed by val must reproduce the data (no shuffle, no overlap)"
    train, val = M.split_data(data, 0.8)
    assert (len(train), len(val)) == (800, 200)


def test_batch_shapes_and_dtype():
    data = torch.arange(500)
    x, y = M.get_batch(data, batch_size=4, block_size=8)
    assert x.shape == (4, 8) and y.shape == (4, 8), f"expected (B,T)=(4,8), got {tuple(x.shape)} and {tuple(y.shape)}"
    assert x.dtype == torch.long and y.dtype == torch.long


def test_targets_are_inputs_shifted_by_one():
    data = torch.arange(1000)  # token value == position, so "next token" == value + 1
    x, y = M.get_batch(data, batch_size=32, block_size=16)
    assert torch.equal(y, x + 1), "y[b,t] must be the token right after x[b,t] in the data"
    assert torch.equal(x[:, 1:] - x[:, :-1], torch.ones(32, 15, dtype=torch.long)), "each row of x must be a contiguous window"


def test_window_never_runs_off_the_end():
    block = 8
    data = torch.arange(block + 1)  # exactly one valid (x, y) window exists
    for _ in range(100):
        x, y = M.get_batch(data, batch_size=8, block_size=block)
        assert x.shape == (8, block) and y.shape == (8, block), "a window was truncated at the end of the data"
        assert torch.equal(x[0], torch.arange(block)) and torch.equal(y[0], torch.arange(1, block + 1))


def test_last_window_is_reachable():
    data = torch.arange(50)
    block = 8
    starts = set()
    g = torch.Generator().manual_seed(0)
    for _ in range(300):
        x, _ = M.get_batch(data, batch_size=16, block_size=block, generator=g)
        starts.update(x[:, 0].tolist())
    assert max(starts) == 50 - block - 1, f"the final valid window starts at {50 - block - 1}; largest start ever sampled was {max(starts)}"
    assert min(starts) == 0, "the first window (start 0) was never sampled"


def test_generator_makes_batches_reproducible_and_batches_vary():
    data = torch.arange(10_000)
    a1, _ = M.get_batch(data, 8, 16, generator=torch.Generator().manual_seed(1))
    a2, _ = M.get_batch(data, 8, 16, generator=torch.Generator().manual_seed(1))
    g = torch.Generator().manual_seed(1)
    b1, _ = M.get_batch(data, 8, 16, generator=g)
    b2, _ = M.get_batch(data, 8, 16, generator=g)
    assert torch.equal(a1, a2), "same generator seed must give the same batch"
    assert not torch.equal(b1, b2), "consecutive batches from one generator should differ"
