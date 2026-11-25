# Hints - d05_missing_residual

## Tier 1 - nudge
Write the equation of the block for each of its two sub-layers. What should the output be if a sub-layer returns zeros?

---

## Tier 2 - stronger
Compare the two lines in `forward`. They should have the same structure.

---

## Tier 3 - near-answer
The first sub-layer replaces `x` instead of adding to it, so the skip path around attention is missing.
