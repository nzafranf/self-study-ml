# Hints - d04_mask_buffer

## Tier 1 - nudge
What does `.to(device)` move? Print `list(head.state_dict().keys())` and `list(head.buffers())`.

---

## Tier 2 - stronger
Parameters are learnable; plain tensor attributes are neither moved by `.to()` nor saved. There's a middle category for fixed tensors that are still part of a module's state.

---

## Tier 3 - near-answer
`tril` is assigned as an ordinary attribute. It needs `register_buffer`.
