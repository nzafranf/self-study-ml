# Post-mortem - d05_layernorm_dim

**Bug:** statistics were taken over dim 0 (the batch) instead of the last dim (features).

**Why it's nasty:** output shape is right and values look roughly normalised, but you've built BatchNorm: tokens now depend on batch-mates, behaviour changes with batch size, and generation with batch size 1 breaks.

**How to catch it:** check per-token mean/std, and test independence of examples.
