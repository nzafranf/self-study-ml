# Hints - 01_data

Read only as far as you need.

## Tier 1 - nudges
- `set` removes duplicates; something has to give the vocabulary a stable order.
- A tokenizer is two dictionaries (one per direction) and two tiny functions closing over them.
- For `get_batch`, think of each row as "pick a random start `i`, slice a window". Then ask what `y` is relative to `x`.
- If a test says a window was truncated or the last window is unreachable, write down the largest legal `i` for `len(data)=N` and a window of `block_size` inputs **plus one extra token** for the final target.

---

## Tier 2 - stronger
- Build `stoi` with `enumerate(chars)`; `decode` needs the inverse mapping. `encode` returns a plain `list[int]`.
- `split_data`: compute the cut index as an `int`, then slice the tensor twice.
- `get_batch`: `torch.randint(high, (batch_size,), generator=generator)` gives the starts; build `x` and `y` per start and combine rows with `torch.stack`.
- `y`'s slice is `x`'s slice moved one to the right.

---

## Tier 3 - near-answer
- starts: `torch.randint(len(data) - block_size, (batch_size,), generator=generator)` (the upper bound is exclusive, so the largest start is `N - block_size - 1`, which leaves exactly one token for the last target).
- `x = torch.stack([data[i:i+block_size] for i in ix])`, `y = torch.stack([data[i+1:i+block_size+1] for i in ix])`.
- decode: `''.join(itos[i] for i in l)`.
