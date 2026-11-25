# d04_mask_buffer

**Module:** 04   **Related video section:** 01:02:00 (version 4) - register_buffer remark

## Symptom
Attention works on your laptop. But on a GPU you get a device-mismatch error from the mask, and when you save and reload a checkpoint (or inspect `state_dict()`) the causal mask is missing from the saved state.

## Your task
Run `scripts/check.sh debug d04_mask_buffer`. The tests fail. `buggy.py` is a small, self-contained GPT that is correct except for **one** planted bug. Diagnose it (use `tools/diagnostics.py`: `print_shapes`, `check_initial_loss`, `print_grad_norms`, `check_causality`), fix `buggy.py`, and get the tests green. Do not edit the test file.

Write down, before you fix it: what invariant was violated, and what would have caught it earlier?

Stuck? `hints.md` (three tiers). Done? A short post-mortem is waiting in `solutions/debug/d04_mask_buffer/`.
