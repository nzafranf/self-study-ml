# EX4 - Implement a feature from the Transformer paper (open-ended, no tests)

Description exercise: *"Read some transformer papers and implement one additional feature or change that people seem to use. Does it improve the performance of your GPT?"*

Pick **one** and compare with your module-06 baseline using identical data, seed, steps and config; report validation loss (mean over several seeds if you can):

- Sinusoidal positional encodings instead of learned (the original paper, section 3.5)
- Weight tying between the token embedding and `lm_head`
- GELU instead of ReLU in the feed-forward layer
- Learning-rate warmup + cosine decay
- Rotary position embeddings (RoPE) or ALiBi (more advanced)

Write a short note: hypothesis, change, result, whether the difference is bigger than seed-to-seed noise.
