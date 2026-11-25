# Hints - d06_loss_reshape

## Tier 1 - nudge
Take the logits the model returns and score them against the targets yourself. Do you get the same number as the model's loss?

---

## Tier 2 - stronger
`view` reinterprets memory in order; it does not move axes. Which logit does each target end up paired with after the reshape?

---

## Tier 3 - near-answer
The loss reshapes logits to `(B, C, T)` with `view` instead of using `transpose(1, 2)` (or flattening to `(B*T, C)`), so logits and targets get mis-paired.
