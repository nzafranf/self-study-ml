# Post-mortem - d06_targets_not_shifted

**Bug:** targets sliced `data[i : i+block]` instead of `data[i+1 : i+block+1]`.

**Why it's nasty:** the model can trivially copy its current token, so the loss falls to ~0. It *feels* like success.

**How to catch it:** whenever loss looks too good, train on i.i.d. random tokens and expect ln(V); assert `y == x + 1` on an arange dataset.
