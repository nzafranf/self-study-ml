# Hints - 05_transformer_block

## Tier 1 - nudges
- FeedForward: a small two-layer network applied to every position. The paper's inner width is a multiple of the model width.
- LayerNorm: compute the statistics *per token over its features* (not per feature over the batch). Then give each feature its own learnable scale and shift. Must they be `nn.Parameter`s?
- Block, residual test failing: if all sub-layer weights are zero, what should the block return? Where does the original `x` have to be added back?
- Block should have exactly the four attributes the docstring lists; the test zeroes `sa` and `ffwd`, not the norms.

---

## Tier 2 - stronger
- `nn.Sequential(Linear(n, 4n), ReLU(), Linear(4n, n), Dropout(p))`.
- LayerNorm: `mean = x.mean(-1, keepdim=True)`, `var = x.var(-1, keepdim=True)`, `xhat = (x - mean) / sqrt(var + eps)`, `out = gamma * xhat + beta`.
- Block: `x = x + self.sa(self.ln1(x))` then the same pattern for the feed-forward (note: add to `x`, normalise only the *input of the sub-layer*).

---

## Tier 3 - near-answer
```
self.sa = MultiHeadAttention(n_embd, n_head, block_size, dropout)   # via the module-04 loader
self.ffwd = FeedForward(n_embd, dropout)
self.ln1, self.ln2 = nn.LayerNorm(n_embd), nn.LayerNorm(n_embd)
forward: x = x + self.sa(self.ln1(x)); x = x + self.ffwd(self.ln2(x)); return x
```
