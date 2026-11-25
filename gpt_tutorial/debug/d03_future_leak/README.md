# d03_future_leak

**Module:** 03   **Related video section:** 00:54:42 (version 3: softmax) / 00:57:00 correction

## Symptom
You build the averaging weights with the softmax trick (version 3). The numbers look plausible, but when you edit a *later* token, the output at *earlier* positions changes. In a full model this would show up as a training loss that looks far too good and generations that are garbage.

## Your task
Run `scripts/check.sh debug d03_future_leak`. The tests fail. `buggy.py` is a small, self-contained GPT that is correct except for **one** planted bug. Diagnose it (use `tools/diagnostics.py`: `print_shapes`, `check_initial_loss`, `print_grad_norms`, `check_causality`), fix `buggy.py`, and get the tests green. Do not edit the test file.

Write down, before you fix it: what invariant was violated, and what would have caught it earlier?

Stuck? `hints.md` (three tiers). Done? A short post-mortem is waiting in `solutions/debug/d03_future_leak/`.
