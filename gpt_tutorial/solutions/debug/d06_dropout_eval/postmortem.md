# Post-mortem - d06_dropout_eval

**Bug:** `F.dropout(x, p)` used with its default `training=True` instead of an `nn.Dropout` module (or `training=self.training`).

**Why it's nasty:** everything else respects `eval()`, so only the embeddings are noisy. The effect is subtle: slightly worse, non-reproducible eval metrics.

**How to catch it:** call the model twice in eval mode and compare.
