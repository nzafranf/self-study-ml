# Hints - 04_self_attention

## Tier 1 - nudges
- Three bias-free linear layers from `n_embd` to `head_size`. What do you need `x` to produce, and in which shapes?
- `scores[b, i, j]` is a dot product between query `i` and key `j`. Which transposition makes a batched matmul do that for all pairs at once?
- The mask has `block_size` rows but your input may be shorter. Slice it.
- A learnable parameter vs a fixed tensor that should still follow `.to(device)` / be saved in `state_dict`: PyTorch has a dedicated registration call for the second kind.
- Softmax over which axis? The row for query `i` must become a distribution over keys `j`.
- Fails the "variance of scores" test: work out what multiplying by `C**-0.5` vs `head_size**-0.5` does to variance when `C != head_size`.

---

## Tier 2 - stronger
- `scores`: `q @ k.transpose(-2, -1) * head_size ** -0.5`. Take `head_size` from `k.shape[-1]`.
- `forward`: scores -> `masked_fill(tril[:T,:T] == 0, -inf)` -> `softmax(dim=-1)` -> dropout -> `@ v`.
- `register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))`.
- MultiHead: `nn.ModuleList` (not a plain list!) of heads; `torch.cat(..., dim=-1)`; `nn.Linear(head_size * n_head, n_embd)`; dropout.

---

## Tier 3 - near-answer
- Head: `k = self.key(x); q = self.query(x); wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5` (that's `scores`); then in `forward` apply the mask and softmax, `out = wei @ self.value(x)`.
- MultiHead forward: `out = torch.cat([h(x) for h in self.heads], dim=-1); return self.dropout(self.proj(out))`.
- Verify yourself: compare against `torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)`.
