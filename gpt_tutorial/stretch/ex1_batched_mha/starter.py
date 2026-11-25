"""EX1 - Merge Head and MultiHeadAttention into one batched class  (video 01:46:22 nanoGPT walkthrough)

Instead of a ModuleList of Heads, compute all heads at once by moving the head index into
a batch dimension: (B, T, n_embd) -> (B, n_head, T, head_size).

Parameter layout the tests rely on (so they can copy weights from your module-04 attention):
  self.qkv  : nn.Linear(n_embd, 3*n_embd, bias=False). Output channels are [all queries | all keys | all values];
              within each third, head h owns channels h*head_size : (h+1)*head_size.
  self.proj : nn.Linear(n_embd, n_embd)  (same role as in module 04)

Run:  scripts/check.sh stretch
"""
from __future__ import annotations

import torch
import torch.nn as nn


class BatchedMultiHeadAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float = 0.0):
        super().__init__()
        # TODO: qkv, proj, causal mask buffer, dropout
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, n_embd) -> (B, T, n_embd). Causal. No Python loop over heads."""
        # TODO
        raise NotImplementedError
