# Debug track

Each exercise is a small self-contained GPT with **one** planted bug and a failing test. Your job is to diagnose it like you would in real life: read the symptom, form a hypothesis, measure, then fix.

```
scripts/check.sh debug d04_scaling        # run one exercise
```

Do the exercises for a module after that module's tests are green. Hints are tiered (`hints.md`); post-mortems are in `solutions/debug/<id>/` - read them afterwards.

## Method
1. Reproduce: run the test, read the failure message (it describes a violated property, not the fix).
2. Shrink: smallest input that shows it (T=4, B=1).
3. Measure: `tools/diagnostics.py`.
4. Hypothesise one cause, change one thing, re-run.
5. Ask: what test would have caught this before it cost me an afternoon?

## Catalog
| exercise | module | symptom (short) |
|---|---|---|
| `d03_future_leak` | 03 | You build the averaging weights with the softmax trick (version 3). |
| `d04_scaling` | 04 | Your attention head passes shape checks and causality checks, and the model trains - but the attention weights at initialisation are much more uniform than they should be, and training is slower than a reference implementation. |
| `d04_softmax_dim` | 04 | A single attention head 'works' and rows look normalised, but the model peeks at the future: editing late tokens changes the output at early positions, and your output doesn't match PyTorch's reference causal attention. |
| `d04_mask_buffer` | 04 | Attention works on your laptop. |
| `d05_missing_residual` | 05 | Stacking more blocks makes the model train worse instead of better. |
| `d05_layernorm_dim` | 05 | Your hand-written LayerNorm outputs look normalised on average, but a token's output now changes when *other examples* in the batch change, and per-token feature means are not zero. |
| `d06_no_pos_emb` | 06 | The model trains and the loss falls a little, but it behaves like a bag of words: generations ignore word order, and the same token at different positions gets exactly the same prediction. |
| `d06_loss_reshape` | 06 | Everything runs. |
| `d06_targets_not_shifted` | 06 | Training loss races to ~0 within a few dozen steps on random, unpredictable data and validation loss is 'amazing'. |
| `d06_generate_crop` | 06 | Generating a short text works. |
| `d06_dropout_eval` | 06 | With dropout configured, validation loss is noisy and two evaluations of the *same* input in eval mode give different numbers, even though you called `model.eval()`. |
| `d06_argmax_sampling` | 06 | Every generation run, whatever the seed, produces the exact same text, and it tends to fall into repetitive loops. |
