# Post-mortem - d06_generate_crop

**Bug:** no context cropping in `generate`.

**Why it's nasty:** short demos work; the failure only appears once the sequence exceeds `block_size`.

**How to catch it:** always test generation for `>= 2 * block_size` new tokens.
