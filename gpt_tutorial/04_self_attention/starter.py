"""Module 04 - Self-attention  (video 01:02:00 -> 01:24:25)

Author's correction (~01:20:05): normalise the scores by head_size, NOT by C (n_embd).

Run:  scripts/check.sh 04
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn import functional as F

# # Batched 3D Matrix Multiplication
# batch_A = torch.randn(10, 2, 3)
# batch_B = torch.randn(10, 3, 4)

# batch_C = batch_A @ batch_B          # Broadcasts batch dims (shape: 10x2x4)
# batch_C = torch.bmm(batch_A, batch_B) # Strict 3D batched multiply (shape: 10x2x4)

class Head(nn.Module):
    """One head of causal self-attention.

    Attribute names the tests rely on: `key`, `query`, `value` (bias-free nn.Linear
    layers n_embd -> head_size) and a registered (non-learnable) causal mask buffer.
    """

    def __init__(self, n_embd: int, head_size: int, block_size: int, dropout: float = 0.0):
        super().__init__()
        # TODO: key / query / value projections (no bias)
        # MISTAKE, the shape is flipped because it is the behavior of nn.Linear
        # self.key = nn.Linear(n_embd, head_size, bias=False)
        # self.query = nn.Linear(n_embd, head_size, bias=False)
        # self.value = nn.Linear(n_embd, head_size, bias=False)

        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        # TODO: the (block_size, block_size) lower-triangular mask, registered so it is
        #       part of the module's state but is NOT a trainable parameter
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        # TODO: dropout applied to the attention weights
        self.do = nn.Dropout(p=dropout)

    def scores(self, x: torch.Tensor) -> torch.Tensor:
        """Scaled query-key affinities BEFORE masking.

        x: (B, T, n_embd) -> (B, T, T)   scores[b, i, j] = how much token i wants token j.
        """
        # TODO
        B, T, C = x.shape
        head_size, _ = self.key.weight.shape
        K = self.key(x) # x @ K.T --> (B, T, C) @ (C, head_size) = (B, T, head_size)
        Q = self.query(x)
        # flip dim -2 with -1
        # (B, T, head_size) @ (B, head_size, T) = (B, T, T)
        wei = Q @ K.transpose(-2, -1) / head_size**0.5
        return wei

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, n_embd) with T <= block_size  ->  (B, T, head_size).

        scores -> mask the future -> softmax over the keys -> dropout -> weighted sum of values.
        """
        # TODO
        B, T, C = x.shape
        head_size, _ = self.key.weight.shape
        K = self.key(x) # x @ K.T --> (B, T, C) @ (C, head_size) = (B, T, head_size)
        Q = self.query(x)
        # flip dim -2 with -1
        # (B, T, head_size) @ (B, head_size, T) = (B, T, T)
        wei = Q @ K.transpose(-2, -1) / head_size**0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1) # (B, T, T), it's softmaxing the row, and the -1 index is responsible to traversing through the rows of TxT matric in a batch b
        wei = self.do(wei)
        # MISTAKE:
        # V = self.value(wei)

        V = wei @ self.value(x)

        return V


class MultiHeadAttention(nn.Module):
    """`n_head` heads in parallel (each of size n_embd // n_head), concatenated, then projected.

    Attribute names the tests rely on: `heads` (nn.ModuleList of Head) and `proj`
    (nn.Linear back to n_embd).
    """

    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float = 0.0):
        super().__init__()
        # TODO
        self.heads = nn.ModuleList([Head(n_embd, n_embd//n_head, block_size, dropout) for _ in range(n_head)])
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, n_embd) -> (B, T, n_embd)."""
        # TODO
        B, T, C  = x.shape

        out = torch.cat([h(x) for h in self.heads], dim=-1)

        return self.proj(out)
            
