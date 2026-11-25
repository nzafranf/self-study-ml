# Post-mortem - d06_argmax_sampling

**Bug:** greedy `argmax` decoding instead of sampling from `softmax(logits)`.

**Why it's nasty:** it is a valid decoding strategy, so nothing errors - but outputs are deterministic and repetitive, hiding the model's actual distribution.

**How to catch it:** generate with two seeds and compare; examine entropy of the samples.
