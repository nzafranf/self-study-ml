# Post-mortem - d03_future_leak

**Bug:** the mask was built from `torch.triu` instead of `torch.tril`, so each position averaged itself and everything *after* it.

**Why it's nasty:** shapes are fine, rows still sum to 1, nothing crashes. In a trained model the network simply reads the answer from the future, so training loss collapses while generation fails.

**How to catch it:** the causality test - edit tokens at positions > t and assert outputs at <= t don't move - plus printing the weight matrix on a tiny T. `tools/diagnostics.check_causality` does this.
