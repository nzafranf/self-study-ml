# Hints - d03_future_leak

## Tier 1 - nudge
Print `causal_weights(4)`. Which part of the matrix is non-zero? Compare with what a token is allowed to see.

---

## Tier 2 - stronger
Look at how the mask is derived from a triangular matrix, then at which positions get `-inf`. Is the triangle the right way round?

---

## Tier 3 - near-answer
A `triu` is used where a `tril` is needed, so each token attends to itself and the *future* instead of itself and the past.
