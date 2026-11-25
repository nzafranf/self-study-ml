# d06_no_pos_emb

**Module:** 06   **Related video section:** 01:00:18 (positional encoding)

## Symptom
The model trains and the loss falls a little, but it behaves like a bag of words: generations ignore word order, and the same token at different positions gets exactly the same prediction.

## Your task
Run `scripts/check.sh debug d06_no_pos_emb`. The tests fail. `buggy.py` is a small, self-contained GPT that is correct except for **one** planted bug. Diagnose it (use `tools/diagnostics.py`: `print_shapes`, `check_initial_loss`, `print_grad_norms`, `check_causality`), fix `buggy.py`, and get the tests green. Do not edit the test file.

Write down, before you fix it: what invariant was violated, and what would have caught it earlier?

Stuck? `hints.md` (three tiers). Done? A short post-mortem is waiting in `solutions/debug/d06_no_pos_emb/`.
