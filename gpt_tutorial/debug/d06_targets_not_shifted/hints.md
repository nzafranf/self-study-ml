# Hints - d06_targets_not_shifted

## Tier 1 - nudge
If every token were random, what is the best possible loss? If your loss is far below that, what does the model have access to?

---

## Tier 2 - stronger
Look at how `x` and `y` are built from the same window. What should the relation between them be?

---

## Tier 3 - near-answer
`y` is sliced from the same offset as `x` (no +1 shift), so the model is asked to copy its input.
