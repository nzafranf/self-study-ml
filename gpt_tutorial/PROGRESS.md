# Progress

Run a module's tests: `scripts/check.sh 01` (etc.). Tick a box when the tests are green AND you can answer the module's README questions in your own words.

## 01_data — 00:07:52 → 00:22:11
- [ ] `build_vocab`, `make_codec` (round-trip)
- [ ] `split_data`
- [ ] `get_batch` (shapes, shifted targets, no off-by-one)
- [ ] README questions answered

## 02_bigram — 00:22:11 → 00:42:13
- [ ] `forward` (logits + loss)
- [ ] `generate`
- [ ] README questions answered

## 03_attention_math — 00:42:13 → 01:00:18
- [ ] `bow_loop` (v1)
- [ ] `bow_matmul` (v2)
- [ ] `causal_weights` / `bow_softmax` (v3)
- [ ] README questions answered
- [ ] Debug: `d03_future_leak`

## 04_self_attention — 01:02:00 → 01:24:25
- [ ] `Head` (scores, mask, softmax, values)
- [ ] `MultiHeadAttention`
- [ ] README questions answered
- [ ] Debug: `d04_scaling`
- [ ] Debug: `d04_softmax_dim`
- [ ] Debug: `d04_mask_buffer`

## 05_transformer_block — 01:24:25 → 01:37:49
- [ ] `FeedForward`
- [ ] `LayerNorm1d`
- [ ] `Block` (residuals + pre-norm)
- [ ] README questions answered
- [ ] Debug: `d05_missing_residual`
- [ ] Debug: `d05_layernorm_dim`

## 06_full_gpt — 01:00:18 + 01:37:49 → 01:46:22
- [ ] `GPTLanguageModel.forward` (token + position embeddings)
- [ ] `generate` (context cropping)
- [ ] `estimate_loss`, `train_model`
- [ ] README questions answered
- [ ] Debug: `d06_no_pos_emb`
- [ ] Debug: `d06_loss_reshape`
- [ ] Debug: `d06_targets_not_shifted`
- [ ] Debug: `d06_generate_crop`
- [ ] Debug: `d06_dropout_eval`
- [ ] Debug: `d06_argmax_sampling`

## Stretch (optional)
- [ ] EX1 batched multi-head attention
- [ ] EX2 tiny GPT adds numbers
- [ ] EX3 pretrain, then fine-tune
- [ ] EX4 one feature from the Transformer paper
