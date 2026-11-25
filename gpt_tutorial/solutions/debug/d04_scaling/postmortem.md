# Post-mortem - d04_scaling

**Bug:** scaling by `C ** -0.5` (n_embd) instead of `head_size ** -0.5`.

**Why it's nasty:** it only changes the *magnitude* of the logits. Everything still runs and trains; the softmax is just too diffuse (var = head_size/C instead of 1), so learning is slower. The video itself had this slip (see the author's correction at ~01:20:05).

**How to catch it:** check the variance of pre-softmax scores at init, or compare against `F.scaled_dot_product_attention`.
