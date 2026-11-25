# 06 - The full GPT

**Watch:** 01:00:18 (positional encoding), 01:37:49 -> 01:46:22 (scaling up, dropout, encoder vs decoder), 01:46:22 (nanoGPT walkthrough; optional here, see `stretch/` EX1)
**Implement:** `starter.py` - `GPTLanguageModel` (`__init__`, `forward`, `generate`), `estimate_loss`, `train_model`
**Check:** `scripts/check.sh 06`   (uses *your* modules 04 and 05; whole module runs in a few seconds on CPU)
**Debug exercises (after you're green):** `debug/d06_*` (six of them)

Suggested follow-up outside the tests: train on Tiny Shakespeare with the video's config (`n_embd=384, n_head=6, n_layer=6, block_size=256, dropout=0.2`) if you have a GPU, or a CPU-sized config (`n_embd=64, n_head=4, n_layer=4, block_size=32`) and read the samples.

## Questions to answer in your own notes
1. Why can't attention tell token order on its own, and how do the position embeddings fix it? Why *add* them to the token embeddings instead of concatenating?
2. Why does `generate` have to crop the context to the last `block_size` tokens, and which exact line of `forward` would crash without cropping?
3. The video notes that train loss and val loss start to diverge when you scale up. What is that telling you, and what does dropout do about it? Why must it be off at evaluation?
4. This model is a "decoder-only" Transformer. What would you change to make an encoder block, and what extra piece does cross-attention (translation example, ~01:42:39) need?

Stuck? `hints.md`.
