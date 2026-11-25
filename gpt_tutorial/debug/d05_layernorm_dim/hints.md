# Hints - d05_layernorm_dim

## Tier 1 - nudge
Which axis does each statistic reduce over? For a `(B, T, C)` input, which axis index is 'features of one token'?

---

## Tier 2 - stronger
LayerNorm normalises every token's *own* feature vector; BatchNorm normalises each feature across the batch. Which one does your reduction axis implement?

---

## Tier 3 - near-answer
`mean` and `var` reduce over dim 0 (batch); they must reduce over the last dim (the features).
