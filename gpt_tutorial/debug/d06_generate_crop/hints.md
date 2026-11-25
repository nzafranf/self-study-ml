# Hints - d06_generate_crop

## Tier 1 - nudge
Where exactly does the crash happen, and which tensor in that call has a fixed maximum length? What grows each loop iteration?

---

## Tier 2 - stronger
The model only has `block_size` position embeddings. What must be true about the `T` you pass to `forward`?

---

## Tier 3 - near-answer
`generate` passes the whole, ever-growing sequence to the model instead of only the last `block_size` tokens.
