# Post-mortem - d05_missing_residual

**Bug:** `x = self.sa(self.ln1(x))` instead of `x = x + self.sa(self.ln1(x))`.

**Why it's nasty:** shapes, causality and even a short training run all look fine. The model just has no gradient highway around attention, so depth hurts rather than helps.

**How to catch it:** zero the sub-layer weights and expect identity; inspect gradient norms across depth (`tools/diagnostics.print_grad_norms`).
