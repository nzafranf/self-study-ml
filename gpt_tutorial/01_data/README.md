# 01 - Data

**Watch:** 00:07:52 -> 00:22:11 (reading the data, tokenization + train/val split, the data loader)
**Implement:** `starter.py` - `build_vocab`, `make_codec`, `split_data`, `get_batch`
**Check:** `scripts/check.sh 01`   (no download needed; tests use `data/sample.txt` and synthetic tensors)

Optional: download Tiny Shakespeare (the video's dataset) and try your functions on it:
`https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt`

## Questions to answer in your own notes
1. A chunk of `block_size + 1` tokens contains how many (context, target) training examples, and what is the longest/shortest context in them? Why does the model need to see *all* context lengths from 1 to `block_size`?
2. Why split train/val contiguously rather than shuffling characters? What would go wrong with a random character-level split?
3. What is the trade-off between a character-level vocabulary (65 tokens) and a subword vocabulary (e.g. 50k tokens), in terms of sequence length and embedding-table size?
4. `get_batch` must never produce a window that runs past the end of the data. Work out, on paper, the largest valid start index for `N` tokens and window `block_size`.

Stuck? `hints.md` (three tiers - stop reading when you have enough).
