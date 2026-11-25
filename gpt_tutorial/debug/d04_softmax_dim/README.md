# d04_softmax_dim

**Module:** 04   **Related video section:** 01:02:00 (version 4: self-attention)

## Symptom
A single attention head 'works' and rows look normalised, but the model peeks at the future: editing late tokens changes the output at early positions, and your output doesn't match PyTorch's reference causal attention.

## Your task
Run `scripts/check.sh debug d04_softmax_dim`. The tests fail. `buggy.py` is a small, self-contained GPT that is correct except for **one** planted bug. Diagnose it (use `tools/diagnostics.py`: `print_shapes`, `check_initial_loss`, `print_grad_norms`, `check_causality`), fix `buggy.py`, and get the tests green. Do not edit the test file.

Write down, before you fix it: what invariant was violated, and what would have caught it earlier?

Stuck? `hints.md` (three tiers). Done? A short post-mortem is waiting in `solutions/debug/d04_softmax_dim/`.
