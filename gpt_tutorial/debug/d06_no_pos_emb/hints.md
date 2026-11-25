# Hints - d06_no_pos_emb

## Tier 1 - nudge
Feed the same token at every position. What does a model that knows position do differently from one that doesn't? Also print the gradient norm of each parameter.

---

## Tier 2 - stronger
Is every module you create in `__init__` actually used in `forward`? Check which parameters have `grad is None`.

---

## Tier 3 - near-answer
The position embedding is computed but never added to the token embedding.
