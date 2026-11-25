# Stretch exercises (optional)

These come from the exercises listed in the video description. Do them after module 06.

| exercise | what | tests |
|---|---|---|
| `ex1_batched_mha` | one class that runs all heads in parallel as a batch dimension (the nanoGPT trick, 01:46:22) | yes - `scripts/check.sh stretch` |
| `ex2_addition` | train a tiny GPT to add numbers | dataset tests, plus an optional slow training test: `pytest stretch -m slow` |
| `ex3_pretrain_finetune/` | pretrain on one corpus, then fine-tune on another | no (open-ended; see its README) |
| `ex4_paper_feature/` | implement one feature from "Attention Is All You Need" and compare | no (open-ended; see its README) |
