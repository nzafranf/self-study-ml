# EX1 - Batched multi-head attention

**Video:** 01:46:22 (nanoGPT walkthrough: batched multi-headed self-attention). Description exercise: *"Head and MultiHeadAttention merged into one class that processes all the heads in parallel, treating the heads as another batch dimension."*

Implement `BatchedMultiHeadAttention` in `starter.py` (read its docstring for the parameter layout), run `scripts/check.sh stretch`.

## Questions
1. What is the shape of the score tensor, and why does `softmax(dim=-1)` still do the right thing with an extra head dimension?
2. Which `view`/`transpose` order gives `(B, n_head, T, head_size)`, and why is a plain `.view(B, n_head, T, hs)` of the `(B, T, C)` tensor *wrong*?
3. After attention, how do you merge heads back to `(B, T, C)` and why might `.contiguous()` be needed?

Optional: compare speed with module 04, and with `torch.nn.functional.scaled_dot_product_attention`.
