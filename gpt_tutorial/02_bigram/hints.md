# Hints - 02_bigram

## Tier 1 - nudges
- An embedding table *is* a lookup table: integer ids in, vectors out. If the vector length equals the vocabulary size, those vectors can be read as logits.
- What shape does the table's output have for `idx` of shape `(B, T)`?
- Test says the loss doesn't match the per-position reference? Print the shapes of `logits` and `targets` right before `cross_entropy`, and ask how the *rows* of each correspond to each other.
- Test says generation doesn't continue the pattern? Which of the `T` time steps contains the prediction for the *next* token?

---

## Tier 2 - stronger
- `nn.Embedding(vocab_size, vocab_size)`; `forward` with no targets just returns the embedding output.
- With targets: `cross_entropy` expects `(N, C)` and `(N,)`. Flatten **both** with the same `B*T` ordering.
- `generate` loop: forward -> keep only the last time step -> `softmax` over the vocab dim -> `torch.multinomial(probs, num_samples=1)` -> `torch.cat` onto `idx` along dim 1.

---

## Tier 3 - near-answer
- `logits = self.token_embedding_table(idx)`; `B, T, C = logits.shape`; `logits.view(B*T, C)`; `targets.view(B*T)`.
- `logits[:, -1, :]` is `(B, C)`; `probs = F.softmax(that, dim=-1)`; `idx = torch.cat((idx, idx_next), dim=1)` with `idx_next` of shape `(B, 1)`.
