# Post-mortem - d06_loss_reshape

**Bug:** `logits.view(B, C, T)` used to get the 'channels second' layout cross-entropy accepts. `view` doesn't permute axes, it scrambles which logits belong to which target.

**Why it's nasty:** it runs, returns a scalar, and the *initial* loss is still ~ln(V) because scrambled random pairs are still random. The scrambling is a fixed permutation, so the optimiser can even drive that loss down - but it is optimising the wrong objective, so accuracy against the real targets stays at chance.

**How to catch it:** recompute the loss independently from the returned logits; check accuracy (not just loss) after overfitting one batch.
