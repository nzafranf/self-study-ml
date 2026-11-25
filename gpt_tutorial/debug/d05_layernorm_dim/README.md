# d05_layernorm_dim

**Module:** 05   **Related video section:** 01:32:51 (layernorm)

## Symptom
Your hand-written LayerNorm outputs look normalised on average, but a token's output now changes when *other examples* in the batch change, and per-token feature means are not zero. (It behaves like BatchNorm.)

## Your task
Run `scripts/check.sh debug d05_layernorm_dim`. The tests fail. `buggy.py` is a small, self-contained GPT that is correct except for **one** planted bug. Diagnose it (use `tools/diagnostics.py`: `print_shapes`, `check_initial_loss`, `print_grad_norms`, `check_causality`), fix `buggy.py`, and get the tests green. Do not edit the test file.

Write down, before you fix it: what invariant was violated, and what would have caught it earlier?

Stuck? `hints.md` (three tiers). Done? A short post-mortem is waiting in `solutions/debug/d05_layernorm_dim/`.
