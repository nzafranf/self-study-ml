"""Module 03 - The mathematical trick in self-attention  (video 00:42:13 -> 01:00:18)

Three ways to compute "for each position t, the average of positions 0..t".
Remember the author's correction (~00:57:00): tokens from the FUTURE cannot communicate.

Run:  scripts/check.sh 03
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def bow_loop(x: torch.Tensor) -> torch.Tensor:
    """Version 1 - explicit loops.

    x: (B, T, C)  ->  out: (B, T, C) with out[b, t] = mean over i<=t of x[b, i].
    """
    # TODO
    B, T, C = x.shape
    bow = torch.zeros_like(x)
    for b in range(B):
        for t in range(T):
            bow[b, t, :] = torch.mean(x[b, :t+1], 0)
            # MISTAKE: previous mean calculation mayy get incorporated, it should've used x. also, it's t+1 not t
            # bow[b, t, :] = torch.mean(bow[b, :t, :], dim=1)
    return bow
    


def bow_matmul(x: torch.Tensor) -> torch.Tensor:
    """Version 2 - one matrix multiply.

    Build a (T, T) weight matrix with torch.tril and normalise each row, then
    (T,T) @ (B,T,C) -> (B,T,C). Must equal bow_loop(x).
    """
    # TODO
    B, T, C = x.shape
    wei = torch.tril(torch.ones(T, T))
    # print(wei.sum(axis=1))
    # # wei /= wei.sum(axis=1)
    # print(wei)
    wei /= torch.sum(wei, 1, keepdim=True)
    # print(f"wei:\n {wei}")
    return wei @ x

def causal_weights(T: int) -> torch.Tensor:
    """Version 3's weight matrix, built the way real attention will build it.

    Start from zeros (T,T) "affinities", mask the positions a token may NOT look at
    with -inf, then softmax over the last dim.
    Returns (T, T): lower-triangular, every row sums to 1 (uniform over allowed positions).
    """
    # TODO
    # PREVIOUS MISTAKE: no diagonal=1, whereas a token can have a context of its own, hence don't mask -inf it
    temp = torch.zeros((T,T)) - torch.triu(torch.full((T, T), torch.inf), diagonal=1)
    # print(temp)
    result = F.softmax(temp, dim=-1)
    # print(result)
    return result


def bow_softmax(x: torch.Tensor) -> torch.Tensor:
    """Version 3 - softmax weights. x: (B,T,C) -> (B,T,C). Must equal bow_loop(x)."""
    # TODO
    return causal_weights(x.shape[1]) @ x
