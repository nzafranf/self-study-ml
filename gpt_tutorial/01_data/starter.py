"""Module 01 - Data  (video 00:07:52 -> 00:22:11)

Implement the four functions below. Run:  scripts/check.sh 01
"""
from __future__ import annotations

from typing import Callable, Optional

import torch


def build_vocab(text: str) -> list[str]:
    """Return the sorted list of unique characters in `text`.

    Example: build_vocab("hello") -> ['e', 'h', 'l', 'o']
    """
    # TODO
    return list(sorted(set(text)))


def make_codec(
    chars: list[str],
) -> tuple[Callable[[str], list[int]], Callable[[list[int]], str]]:
    """Build a character-level tokenizer from the vocabulary.

    Returns (encode, decode):
      encode: str -> list[int]    (each char -> its index in `chars`)
      decode: list[int] -> str    (inverse of encode)
    """
    # TODO
    vocab = build_vocab(chars)

    def encode(string: str) -> list[int]:
        stoi = dict(zip(vocab, range(len(vocab))))
        return [stoi[c] for c in string]

    def decode(string: list[int]) -> str:
            itos = dict(zip(range(len(vocab)), vocab))
            return "".join([itos[c] for c in string])

    return (encode, decode)
    


def split_data(
    data: torch.Tensor, train_frac: float = 0.9
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split a 1-D tensor of token ids into (train, val).

    data:  (N,)   ->   train: (int(train_frac*N),), val: (N - len(train),)
    The split is contiguous (first part train, rest val); nothing is shuffled.
    """
    # TODO
    length = data.shape[0]
    # print(int(length*train_frac))
    train = data[:int(length*train_frac)]
    val = data[int(length*train_frac):]
    # print(f"len train val: {len(train)}, {len(val)}")
    return (train, val)


def get_batch(
    data: torch.Tensor,
    batch_size: int,
    block_size: int,
    generator: Optional[torch.Generator] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a random batch of (input, target) chunks.

    data: (N,) long tensor of token ids.
    Returns x, y, both (batch_size, block_size) long tensors where y is x shifted
    left by one position, i.e. y[b, t] is the token that follows x[b, t] in `data`.

    Use `generator` for the random start offsets so callers can make it reproducible
    (torch.randint accepts generator=...).
    Every valid window must be sampleable, and no window may run past the end of `data`.
    """
    # TODO

    length = len(data)
    batches = []
    for i in range(batch_size):
         start = torch.randint(0, length-block_size, (1, ), generator=generator)[0]
         print(f"start: {start}")
         batches.append(data[start:start+block_size+1])
    # batches = torch.cat(batches, dim=0)
    batches = torch.stack(batches)
    print(f"batches:\n {batches}")
    x = batches[:, :-1]
    y = batches[:, 1:]
    return (x, y)
    
