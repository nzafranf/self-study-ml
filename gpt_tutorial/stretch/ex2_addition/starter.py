"""EX2 - Teach a tiny GPT to add numbers.

The video description suggests training a Transformer on an arithmetic task. Here you build the
*data* side (tested); the training is yours (see README and the optional `-m slow` test).

Format of one training string (n_digits = 2):  make_example(12, 34, 2) == "12+34=640"
  - operands zero-padded to n_digits
  - the sum zero-padded to n_digits+1 digits and written REVERSED (least-significant digit first),
    because that is the order in which a left-to-right model can compute carries.
  46 -> "046" -> reversed "640".

Run:  scripts/check.sh stretch
"""
from __future__ import annotations

import random


def make_example(a: int, b: int, n_digits: int) -> str:
    """One training string as described above. 0 <= a, b < 10**n_digits."""
    # TODO
    raise NotImplementedError


def answer_value(example: str) -> int:
    """Read the (reversed) answer after '=' back into an int. answer_value("12+34=640") == 46."""
    # TODO
    raise NotImplementedError


def build_dataset(n_digits: int, n_examples: int, seed: int = 0) -> list[str]:
    """`n_examples` strings from uniformly random operand pairs, reproducible from `seed`
    (use random.Random(seed), not the global RNG)."""
    # TODO
    raise NotImplementedError
