# 04 - Self-attention (the crux)

**Watch:** 01:02:00 -> 01:24:25 (v4 self-attention, notes 1-6 incl. "scaled" attention at 01:16:56, inserting a head into the network at 01:19:11, multi-headed attention at 01:21:59)
**Implement:** `starter.py` - `Head` (`scores`, `forward`), `MultiHeadAttention`
**Check:** `scripts/check.sh 04`
**Debug exercises (after you're green):** `debug/d04_scaling`, `debug/d04_softmax_dim`, `debug/d04_mask_buffer`

**Author's correction (~01:20:05):** normalise the scores by **head_size**, not by `C` (n_embd). The tests check this via the variance of the scores.

## Questions to answer in your own notes
1. Say in one sentence what the query, key and value of a token each represent. Which pair of vectors produces the attention weight between two tokens?
2. Derive why `q @ k^T` has variance about `head_size` when `q` and `k` have unit variance, and what happens to the softmax (and to gradients) if you don't rescale.
3. "Attention has no notion of space." What follows for how we must feed in word order (module 06)? And why is there no communication across the batch dimension?
4. What does the final projection after concatenating heads add that plain concatenation lacks? Why split `n_embd` into several small heads instead of one big one?

Stuck? `hints.md`.
