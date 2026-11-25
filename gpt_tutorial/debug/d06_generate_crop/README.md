# d06_generate_crop

**Module:** 06   **Related video section:** 01:00:18 (positional encoding) / generation

## Symptom
Generating a short text works. Asking for a longer text crashes partway through with an index-out-of-range error inside the embedding layer.

## Your task
Run `scripts/check.sh debug d06_generate_crop`. The tests fail. `buggy.py` is a small, self-contained GPT that is correct except for **one** planted bug. Diagnose it (use `tools/diagnostics.py`: `print_shapes`, `check_initial_loss`, `print_grad_norms`, `check_causality`), fix `buggy.py`, and get the tests green. Do not edit the test file.

Write down, before you fix it: what invariant was violated, and what would have caught it earlier?

Stuck? `hints.md` (three tiers). Done? A short post-mortem is waiting in `solutions/debug/d06_generate_crop/`.
