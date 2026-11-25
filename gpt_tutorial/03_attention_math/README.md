# 03 - The mathematical trick in self-attention

**Watch:** 00:42:13 -> 01:00:18 (v1 loops, matmul as weighted aggregation, v2 matmul, v3 softmax, minor cleanup)
**Implement:** `starter.py` - `bow_loop`, `bow_matmul`, `causal_weights`, `bow_softmax`
**Check:** `scripts/check.sh 03`
**Debug exercise (after you're green):** `debug/d03_future_leak`

**Author's correction (~00:57:00):** it is tokens from the *future* that must not communicate with the present (not the past).

## Questions to answer in your own notes
1. Why must a token at position `t` not see positions `> t` when we train a language model to predict the next token? What would the training loss look like if it could?
2. Why does masking with `-inf` *before* softmax work, while zeroing probabilities *after* softmax does not give a valid average?
3. Version 3 starts from a matrix of zeros. In real self-attention what replaces those zeros, and what stays the same?
4. In the lower-triangular matrix `W`, what does row `t` mean, and what does the matmul `W @ x` do to the token embeddings?

Stuck? `hints.md`.
