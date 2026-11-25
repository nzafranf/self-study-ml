# 02 - Bigram baseline

**Watch:** 00:22:11 -> 00:42:13 (bigram model, loss + generation, training it with AdamW)
**Implement:** `starter.py` - `BigramLanguageModel.__init__`, `forward`, `generate`
**Check:** `scripts/check.sh 02`

The tests train your model themselves, so you don't need to write a training loop here (you will in module 06). But do try one in a scratch file with real text and look at the samples before/after.

## Questions to answer in your own notes
1. Why should the *untrained* loss be about `-ln(1/vocab_size)`? What does it tell you if your initial loss is 10x that, or far below it?
2. `F.cross_entropy` wants `(N, C)` logits and `(N,)` targets. Why does the video flatten `(B, T, C)` to `(B*T, C)`, and what silent bug appears if you reshape logits and targets *differently*?
3. In `generate`, only the last time step's logits are used even though `forward` computed logits for all `T` positions. Why compute the rest? Why does a bigram model gain nothing from a longer context?
4. What changes in the generated text if you take the argmax instead of sampling with `torch.multinomial`?

Stuck? `hints.md`.
