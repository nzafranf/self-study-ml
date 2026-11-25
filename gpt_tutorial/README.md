# Let's build GPT - test-driven tutorial

A self-paced companion to Andrej Karpathy's [Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY).
You watch a chapter, write the code yourself in `starter.py`, and the tests tell you whether it is right. A separate debug track teaches you to diagnose broken transformer code.

## Setup
Python 3.10+, CPU-only PyTorch is fine. Dependencies: `torch`, `pytest`, `numpy`.

Run everything from inside the `gpt_tutorial` folder.

**Already have torch and pytest installed? Skip all of this** - nothing else is needed; just run `python -m pytest 01_data -v`.
(Check with `python -c "import torch, pytest"`.) If you still want a venv that reuses your installed packages instead of
downloading torch again, create it with `python -m venv --system-site-packages .venv` and do *not* run `pip install`.

Fresh machine instructions:

macOS / Linux / Git Bash:
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt                         # or: uv pip install -r requirements.txt
```
Windows PowerShell:
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1      # if blocked: Set-ExecutionPolicy -Scope Process RemoteSigned
pip install -r requirements.txt
```
(Optional: a smaller CPU-only torch wheel via `pip install torch --index-url https://download.pytorch.org/whl/cpu`. The venv itself is optional if your Python already has torch + pytest.)

## Run checks
```bash
scripts/check.sh 01              # Git Bash / macOS / Linux. One module (prefix is enough): 01 .. 06, or "stretch"
scripts/check.sh 04 -k causal    # extra args go to pytest
scripts/check.sh debug d04_scaling
scripts/check.sh all             # everything (all modules red until you implement them)
```
PowerShell equivalents (no bash needed):
```powershell
python -m pytest 01_data -v
python -m pytest 04_self_attention -v -k causal
python -m pytest debug/d04_scaling -v
python -m pytest                 # everything
```
All tests are CPU-only, seeded, need no downloads, and the whole suite runs in ~10 seconds.

## Recommended workflow
1. Open the module's `README.md`: it gives the **video timestamp range**, what to implement, and a few conceptual questions.
2. Watch that range. Pause and code - don't just copy.
3. Implement the TODOs in `starter.py`; run `scripts/check.sh <module>`. Failure messages describe the *symptom* (which property broke), not the fix.
4. Stuck? Use `tools/diagnostics.py` (shape printer, initial-loss check, gradient norms, causality check) and think about which invariant is violated. Only then open `hints.md` - three tiers (nudge -> stronger -> near-answer); stop reading as soon as you have enough.
5. When green, answer the README questions **in your own notes** (no answers are provided), then tick `PROGRESS.md`.
6. Do the matching `debug/` exercises for that module.

## Map
| module | video | you implement |
|---|---|---|
| `01_data` | 00:07:52 - 00:22:11 | vocab, codec, train/val split, `get_batch` |
| `02_bigram` | 00:22:11 - 00:42:13 | bigram LM: loss + generation |
| `03_attention_math` | 00:42:13 - 01:00:18 | averaging past context: loops -> matmul -> softmax |
| `04_self_attention` | 01:02:00 - 01:24:25 | `Head`, `MultiHeadAttention` |
| `05_transformer_block` | 01:24:25 - 01:37:49 | feed-forward, LayerNorm, residual `Block` |
| `06_full_gpt` | 01:00:18, 01:37:49 - 01:46:22 | full GPT, `generate`, `estimate_loss`, training loop |
| `debug/` | - | 12 planted-bug exercises for modules 03-06 |
| `stretch/` | 01:46:22 + description | batched multi-head attention, addition, pretrain/finetune, paper feature |

Two corrections from the video's author are built into the tests: tokens from the **future** cannot communicate (~00:57:00), and attention scores are normalised by **head_size**, not `C` (~01:20:05).

Modules 05 and 06 build on your own 04 and 05 (loaded automatically), so go in order.

## Layout
```
NN/starter.py  test_*.py  README.md  hints.md     per-module work
debug/dXX_*/   buggy.py  test_*.py  README.md  hints.md
stretch/       optional exercises
tools/diagnostics.py                               debugging helpers
data/sample.txt                                    tiny bundled text (optional: download Tiny Shakespeare)
PROGRESS.md  CLAUDE.md  scripts/check.sh
```
