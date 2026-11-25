# Hints - 06_full_gpt

## Tier 1 - nudges
- Two embedding tables: *what* the token is (`vocab_size` rows) and *where* it is (`block_size` rows). The positions in a forward pass are `0 .. T-1`.
- The block stack should be an `nn.Sequential` built from `n_layer` `Block`s, followed by a final layer norm and a linear head to the vocabulary.
- Wrong parameter count? Write the tally on paper per component and compare to `sum(p.numel() for p in model.parameters())`; print the per-parameter shapes to find the odd one out.
- Test says the loss "can't overfit"? That is a *whole-pipeline* signal. Shrink the problem (one batch), check initial loss ~ ln(V), check gradients are non-zero for every parameter (`tools/diagnostics.py`).
- `generate` crashes after a while? Which tensor has a fixed maximum length?
- `estimate_loss`: three things must be true during evaluation (mode, gradient tracking, which split) and one afterwards.
- `train_model` mismatch against the reference loop: what has to happen to the gradients *every* iteration?

---

## Tier 2 - stronger
- Embeddings: `nn.Embedding(vocab_size, n_embd)` and `nn.Embedding(block_size, n_embd)`; positions via `torch.arange(T, device=idx.device)`; add the two (broadcasting over batch).
- Loss uses the same flatten convention as module 02.
- Generation: `idx_cond = idx[:, -block_size:]`, then as in the bigram.
- `estimate_loss`: `@torch.no_grad()`, `model.eval()`, loop over splits, `losses[k] = loss.item()`, mean, `model.train()`.
- `train_model`: `AdamW(model.parameters(), lr=...)`, then `zero_grad(set_to_none=True)`, `loss.backward()`, `step()`.

---

## Tier 3 - near-answer
```
x = tok_emb(idx) + pos_emb(arange(T)); x = ln_f(blocks(x)); logits = lm_head(x)
```
`self.blocks = nn.Sequential(*[Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)])`
