# Hints - d04_scaling

## Tier 1 - nudge
Measure `scores(x).var()` for unit-variance inputs. What value do you expect for a correct head, and what do you get?

---

## Tier 2 - stronger
The scale factor should depend on one specific dimension of q/k. Which dimension is `x.shape[-1]` here, and is it the same as the dimension the dot products sum over?

---

## Tier 3 - near-answer
The scores are divided by sqrt(n_embd) (the channel count `C` of `x`) rather than sqrt(head_size). Compare the dimension in the exponent with the dimension of `k`.
