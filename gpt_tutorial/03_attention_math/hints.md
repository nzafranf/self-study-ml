# Hints - 03_attention_math

## Tier 1 - nudges
- v1: for every `(b, t)` average the slice of positions up to and *including* `t`.
- v2: which `torch` function gives you a lower-triangular matrix of ones? How do you turn rows of ones into rows that sum to 1?
- v3: softmax turns scores into weights. What score should a forbidden position get so its weight is exactly 0? What scores should allowed positions get so the weights come out uniform?
- A `(T, T)` matrix times a `(B, T, C)` tensor: how does PyTorch broadcast this?

---

## Tier 2 - stronger
- v1: `x[b, :t+1].mean(0)` (mind the `+1`).
- v2: `torch.tril(torch.ones(T, T))`, divide by its `sum(1, keepdim=True)`, then `wei @ x`.
- v3 / `causal_weights`: `torch.zeros(T,T).masked_fill(tril == 0, float('-inf'))`, then `softmax(dim=-1)`.
- NaN in your weights? An entire row of `-inf` softmaxes to NaN. Can a row of your mask be entirely forbidden?

---

## Tier 3 - near-answer
- `bow_softmax(x) = causal_weights(x.shape[1]) @ x`.
- Check your own work by hand for `T=3`: row 2 of the weights should be `[1/3, 1/3, 1/3]`, row 0 should be `[1, 0, 0]`.
