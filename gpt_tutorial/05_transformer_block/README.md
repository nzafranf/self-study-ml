# 05 - The Transformer block

**Watch:** 01:24:25 -> 01:37:49 (feed-forward layers, residual connections 01:26:48, layer norm 01:32:51)
**Implement:** `starter.py` - `FeedForward`, `LayerNorm1d`, `Block`
**Check:** `scripts/check.sh 05`   (uses *your* module 04)
**Debug exercises (after you're green):** `debug/d05_missing_residual`, `debug/d05_layernorm_dim`

## Questions to answer in your own notes
1. The video frames attention as "communication" and the MLP as "computation". Which axis does each operate over (batch, time, channels), and which one mixes information between tokens?
2. Why do residual connections make deep networks trainable? Describe what the gradient does at an addition node, and what "gradient highway" means.
3. LayerNorm is your BatchNorm with one axis changed. Which axis, why is that a better fit for sequences, and why does it have no running mean/variance or train/eval difference?
4. The video uses pre-norm (norm *before* attention/MLP) rather than the original paper's post-norm. What is placed on the residual pathway in each case?

Stuck? `hints.md`.
