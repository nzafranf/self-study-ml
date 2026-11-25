"""Module 06 - The full GPT  (video 01:00:18 positional encoding, 01:37:49 scaling up + dropout, 01:46:22 nanoGPT)

Builds on YOUR modules 04 and 05. Run:  scripts/check.sh 06
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn as nn
from torch.nn import functional as F

from tutorlib import load_module

_block = load_module("05_transformer_block")  # _block.Block(n_embd, n_head, block_size, dropout)


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 32  # max context length T
    n_embd: int = 64
    n_head: int = 4
    n_layer: int = 4
    dropout: float = 0.0


class GPTLanguageModel(nn.Module):
    """Decoder-only Transformer language model.

    Attribute names the tests rely on:
      token_embedding_table, position_embedding_table, blocks (nn.Sequential of Blocks),
      ln_f (final layer norm), lm_head (n_embd -> vocab_size).
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        # TODO
        raise NotImplementedError

    def forward(
        self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        idx: (B, T) long, T <= block_size. targets: (B, T) long or None.

        tokens + positions -> blocks -> final layer norm -> lm_head.
        If targets is None: return (logits (B, T, V), None).
        Else: return (logits (B*T, V), scalar cross-entropy loss)  - same convention as module 02.
        """
        # TODO
        raise NotImplementedError

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """Extend idx (B, T) by max_new_tokens sampled tokens -> (B, T + max_new_tokens).

        The model only has position embeddings for block_size positions: the context fed to
        the model must never be longer than that, however long the generation gets.
        """
        # TODO
        raise NotImplementedError


@torch.no_grad()
def estimate_loss(
    model: nn.Module,
    get_batch_fn: Callable[[str], tuple[torch.Tensor, torch.Tensor]],
    eval_iters: int,
) -> dict[str, torch.Tensor]:
    """Average loss over `eval_iters` batches for each split.

    get_batch_fn("train" | "val") -> (x, y). Returns {"train": loss, "val": loss}.
    Evaluation must run with the model in eval mode, and afterwards the model must be
    back in train mode.
    """
    # TODO
    raise NotImplementedError


def train_model(
    model: nn.Module,
    get_batch_fn: Callable[[str], tuple[torch.Tensor, torch.Tensor]],
    max_iters: int,
    learning_rate: float,
) -> list[float]:
    """Plain training loop with torch.optim.AdamW(lr=learning_rate).

    Each iteration: batch from get_batch_fn("train"), forward, zero grads, backward, step.
    Returns the list of per-iteration training losses (floats), length max_iters.
    """
    # TODO
    raise NotImplementedError
