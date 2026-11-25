# d06_loss_reshape

**Module:** 06   **Related video section:** 00:22:11 (loss) / 01:00:18 (full model forward)

## Symptom
Everything runs. The initial loss is about ln(vocab), as expected, and the reported loss even goes down during training. But the model's predictions are useless: scored against the targets, the returned logits are right only at chance level, and the reported loss does not match what you get scoring the returned logits yourself.

## Your task
Run `scripts/check.sh debug d06_loss_reshape`. The tests fail. `buggy.py` is a small, self-contained GPT that is correct except for **one** planted bug. Diagnose it (use `tools/diagnostics.py`: `print_shapes`, `check_initial_loss`, `print_grad_norms`, `check_causality`), fix `buggy.py`, and get the tests green. Do not edit the test file.

Write down, before you fix it: what invariant was violated, and what would have caught it earlier?

Stuck? `hints.md` (three tiers). Done? A short post-mortem is waiting in `solutions/debug/d06_loss_reshape/`.
