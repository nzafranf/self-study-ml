# d05_missing_residual

**Module:** 05   **Related video section:** 01:26:48 (residual connections)

## Symptom
Stacking more blocks makes the model train worse instead of better. As a sanity check, zeroing the attention/MLP weights should make a block an identity function, but the block returns something else.

## Your task
Run `scripts/check.sh debug d05_missing_residual`. The tests fail. `buggy.py` is a small, self-contained GPT that is correct except for **one** planted bug. Diagnose it (use `tools/diagnostics.py`: `print_shapes`, `check_initial_loss`, `print_grad_norms`, `check_causality`), fix `buggy.py`, and get the tests green. Do not edit the test file.

Write down, before you fix it: what invariant was violated, and what would have caught it earlier?

Stuck? `hints.md` (three tiers). Done? A short post-mortem is waiting in `solutions/debug/d05_missing_residual/`.
