# Hints - d06_argmax_sampling

## Tier 1 - nudge
Change `torch.manual_seed` between two generation calls. What should change if the next token is *sampled*?

---

## Tier 2 - stronger
Look at the line that turns `probs` into the next token. Does it use the probabilities as a distribution or just pick the largest?

---

## Tier 3 - near-answer
`argmax` picks the most likely token every time; use `torch.multinomial(probs, num_samples=1)`.
