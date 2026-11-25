# Post-mortem - d04_softmax_dim

**Bug:** `softmax(dim=1)` instead of `dim=-1`.

**Why it's nasty:** output shape is unchanged and the masked `-inf` entries still become 0, so causality *appears* intact in a glance at the mask. But column normalisation makes each weight depend on later queries, which leaks future information and ruins row-normalisation.

**How to catch it:** assert `wei.sum(-1) == 1`, run the causality perturbation test, and compare against a reference implementation.
