# d06_targets_not_shifted

**Module:** 06   **Related video section:** 00:14:27 (data loader: batches of chunks)

## Symptom
Training loss races to ~0 within a few dozen steps on random, unpredictable data and validation loss is 'amazing'. Generation from this model is nonsense. The loss looks too good to be true.

## Your task
Run `scripts/check.sh debug d06_targets_not_shifted`. The tests fail. `buggy.py` is a small, self-contained GPT that is correct except for **one** planted bug. Diagnose it (use `tools/diagnostics.py`: `print_shapes`, `check_initial_loss`, `print_grad_norms`, `check_causality`), fix `buggy.py`, and get the tests green. Do not edit the test file.

Write down, before you fix it: what invariant was violated, and what would have caught it earlier?

Stuck? `hints.md` (three tiers). Done? A short post-mortem is waiting in `solutions/debug/d06_targets_not_shifted/`.
