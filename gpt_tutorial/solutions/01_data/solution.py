import torch


def build_vocab(text):
    return sorted(set(text))


def make_codec(chars):
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    def encode(s):
        return [stoi[c] for c in s]

    def decode(l):
        return "".join(itos[int(i)] for i in l)

    return encode, decode


def split_data(data, train_frac=0.9):
    n = int(train_frac * len(data))
    return data[:n], data[n:]


def get_batch(data, batch_size, block_size, generator=None):
    ix = torch.randint(len(data) - block_size, (batch_size,), generator=generator)
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x, y
