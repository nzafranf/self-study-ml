"""Module 02 - Bigram baseline  (video 00:22:11 -> 00:42:13)

Run:  scripts/check.sh 02
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch.nn import functional as F


class BigramLanguageModel(nn.Module):
    """Predicts the next token from the current token only, via a lookup table.

    Attribute names the tests rely on: `token_embedding_table`.
    """

    def __init__(self, vocab_size: int):
        super().__init__()
        # TODO: a learnable table of shape (vocab_size, vocab_size);
        #       row i holds the logits for "what comes after token i".
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)


    def forward(
        self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        idx:     (B, T) long
        targets: (B, T) long, or None

        If targets is None:  return (logits (B, T, V), None)
        Otherwise:           return (logits (B*T, V), loss) where logits are flattened
                             to 2-D (as in the video) and loss is a scalar cross-entropy.
        """
        # TODO
        logits = self.token_embedding_table(idx)

        if targets is None:
            loss = None
        else:
            B, T, V = logits.shape
            # next_token = logits[:, -1, :]
            logits = logits.view(B*T, -1)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
            # PREVOIUS MISTAKE
            # loss = nn.CrossEntropyLoss(logits, targets)
        return logits, loss
        

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """Autoregressively extend `idx` (B, T) by `max_new_tokens` sampled tokens.

        Returns (B, T + max_new_tokens). The original T tokens are kept unchanged.
        Sample from the predicted distribution (do not take the argmax).
        """
        # TODO
        for _ in range(max_new_tokens):
            logits, loss = self(idx)
            B, T, V = logits.shape
            next_token = logits[:, -1, :]
            next_token = next_token.view(B, V)
            next_token_probs = F.softmax(next_token, dim=-1)
            # PREVIOUS MISTAKE: next_token_probs = nn.Softmax(next_token, dim=-1)
            pred = torch.multinomial(next_token_probs, num_samples=1)
            idx = torch.cat((idx, pred), dim=1)
        return idx