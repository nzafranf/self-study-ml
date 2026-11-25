# Hints - d06_dropout_eval

## Tier 1 - nudge
Which dropout layers does `model.eval()` actually switch off? What decides whether a given dropout is active?

---

## Tier 2 - stronger
A module remembers the train/eval flag, but a *function call* doesn't know about the model's mode unless you tell it. Search the forward pass for dropout that isn't an `nn.Dropout` module.

---

## Tier 3 - near-answer
`F.dropout` is called without `training=self.training`, so it stays active in eval mode.
