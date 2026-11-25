# EX2 - Train a tiny GPT to add numbers

Description exercise: *"Train the GPT on an arithmetic task, e.g. addition."* This is the best exercise for feeling how a small Transformer fails and succeeds.

1. Implement `make_example`, `answer_value`, `build_dataset` in `starter.py`; `scripts/check.sh stretch`.
2. Using your module 06, train on 2-digit (then 3-digit) addition. Hold out operand pairs you never train on.
3. Optional slow check of the pipeline on the 1-digit table: `python -m pytest stretch/ex2_addition -m slow`.

## Questions
1. Why is the answer written *reversed*? Try the normal order and compare how fast it learns.
2. Should the loss be computed on the whole string or only on the answer tokens? What does the model waste capacity on otherwise?
3. How does accuracy on held-out pairs change with `n_digits`, model size and training set size? When does it memorise vs generalise?
