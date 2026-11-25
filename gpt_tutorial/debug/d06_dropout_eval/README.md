# d06_dropout_eval

**Module:** 06   **Related video section:** 01:37:49 (scaling up, dropout)

## Symptom
With dropout configured, validation loss is noisy and two evaluations of the *same* input in eval mode give different numbers, even though you called `model.eval()`.

## Your task
Run `scripts/check.sh debug d06_dropout_eval`. The tests fail. `buggy.py` is a small, self-contained GPT that is correct except for **one** planted bug. Diagnose it (use `tools/diagnostics.py`: `print_shapes`, `check_initial_loss`, `print_grad_norms`, `check_causality`), fix `buggy.py`, and get the tests green. Do not edit the test file.

Write down, before you fix it: what invariant was violated, and what would have caught it earlier?

Stuck? `hints.md` (three tiers). Done? A short post-mortem is waiting in `solutions/debug/d06_dropout_eval/`.
