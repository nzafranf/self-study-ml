# Post-mortem - d04_mask_buffer

**Bug:** the mask was stored as a plain tensor attribute instead of a registered buffer.

**Why it's nasty:** on CPU nothing is wrong. On GPU the mask stays on the CPU and `masked_fill` raises a device mismatch; checkpoints silently omit it.

**How to catch it:** inspect `state_dict()` / `buffers()`; move the model to a different dtype/device in a test.
