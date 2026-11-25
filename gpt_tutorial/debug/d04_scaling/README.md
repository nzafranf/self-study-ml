# d04_scaling

**Module:** 04   **Related video section:** 01:16:56 (scaled self-attention) / 01:20:05 correction

## Symptom
Your attention head passes shape checks and causality checks, and the model trains - but the attention weights at initialisation are much more uniform than they should be, and training is slower than a reference implementation. Printing the variance of the raw scores shows a value far below 1.

## Your task
Run `scripts/check.sh debug d04_scaling`. The tests fail. `buggy.py` is a small, self-contained GPT that is correct except for **one** planted bug. Diagnose it (use `tools/diagnostics.py`: `print_shapes`, `check_initial_loss`, `print_grad_norms`, `check_causality`), fix `buggy.py`, and get the tests green. Do not edit the test file.

Write down, before you fix it: what invariant was violated, and what would have caught it earlier?

Stuck? `hints.md` (three tiers). Done? A short post-mortem is waiting in `solutions/debug/d04_scaling/`.
