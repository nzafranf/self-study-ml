"""Module 05 - The Transformer block  (video 01:24:25 -> 01:37:49)

Feed-forward layer, residual connections, layer norm. This module builds on YOUR
`04_self_attention/starter.py` (loaded below), so finish module 04 first.

Run:  scripts/check.sh 05
"""
from __future__ import annotations

import torch
import torch.nn as nn

from tutorlib import load_module

_attn = load_module("04_self_attention")  # _attn.MultiHeadAttention(n_embd, n_head, block_size, dropout)


class FeedForward(nn.Module):
    """Per-token MLP: expand to 4*n_embd, non-linearity, project back, dropout.

    Attribute name the tests rely on: none beyond being an nn.Module with a (B,T,C)->(B,T,C) forward.
    """

    def __init__(self, n_embd: int, dropout: float = 0.0):
        super().__init__()
        # TODO
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4*n_embd, n_embd)

        ) 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, n_embd) -> (B, T, n_embd). Each position is processed independently."""
        # TODO
        return self.net.forward(x)


class LayerNorm1d(nn.Module):
    """Layer norm over the LAST dimension, written by hand (it is your BatchNorm1d with the axis changed).

    Learnable per-feature scale `gamma` (init 1) and shift `beta` (init 0), both nn.Parameters.
    Normalise each token's feature vector to zero mean / unit variance, then scale and shift.
    x: (..., dim) -> (..., dim).
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        # TODO
        self.gamma = nn.Parameter(torch.ones(1, dim))
        self.beta = nn.Parameter(torch.zeros(1, dim))
        self.eps = eps
        self.dim = dim
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO
        return ((x - torch.mean(x, dim=-1, keepdim=True))/torch.sqrt(torch.var(x, dim=-1, keepdim=True) + self.eps))*self.gamma + self.beta


class Block(nn.Module):
    """Transformer block: communication (attention) then computation (feed-forward).

    Attribute names the tests rely on: `sa` (MultiHeadAttention), `ffwd` (FeedForward),
    `ln1`, `ln2` (nn.LayerNorm(n_embd)).

    Pre-norm with residual connections (as in the video):
        x = x + sa(ln1(x));   x = x + ffwd(ln2(x))
    """

    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float = 0.0):
        super().__init__()
        # TODO (head_size = n_embd // n_head is handled inside MultiHeadAttention)
        self.ln1 = nn.LayerNorm(n_embd)
        self.sa =  _attn.MultiHeadAttention(n_embd, n_head, block_size, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffwd = FeedForward(n_embd, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, n_embd) -> (B, T, n_embd)."""
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x        
