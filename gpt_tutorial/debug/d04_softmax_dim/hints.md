# Hints - d04_softmax_dim

## Tier 1 - nudge
The scores tensor has shape (B, T, T). Which of the last two axes should be turned into a probability distribution? Check by printing `wei.sum(-1)` and `wei.sum(1)` after the softmax.

---

## Tier 2 - stronger
Row `i` of the weights holds query `i`'s affinity to every key `j`. A softmax over the wrong axis normalises across *queries*, mixing different tokens' scores together.

---

## Tier 3 - near-answer
The softmax runs over the query axis (dim=1) rather than the key axis (the last dim).
