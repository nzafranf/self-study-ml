# Post-mortem - d06_no_pos_emb

**Bug:** `pos_emb` was computed but not added to `tok_emb`.

**Why it's nasty:** nothing crashes, the loss still drops (unigram/bigram-ish statistics are learnable from token identity alone), and the dead parameter is easy to miss.

**How to catch it:** a permutation/position test, and a gradient check that flags parameters whose `.grad` is None.
