# EX3 - Pretrain, then fine-tune (open-ended, no tests)

Description exercise: *"Find a dataset that is very large, so large that you can't see a gap between train and val loss. Pretrain the transformer on this data, then initialize with that model and finetune it on tiny Shakespeare with a smaller number of steps and lower learning rate. Can you obtain a lower validation loss by the use of pretraining?"*

Suggested plan (CPU-friendly): pretrain on a larger text of your choice (e.g. several books from Project Gutenberg) with a *shared character vocabulary* that also covers Tiny Shakespeare, save `state_dict`, then load it and fine-tune on Tiny Shakespeare with a ~10x lower learning rate and far fewer steps.

Keep a notes file with: vocab handling, step counts, learning rates, train/val loss curves, and the comparison against training from scratch with the same number of fine-tuning steps.

Questions: What happens if the vocabularies differ? Why a lower learning rate? Where do the benefits show up first - in early steps or in final loss?
