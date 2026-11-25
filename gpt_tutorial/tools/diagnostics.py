"""Debugging helpers. Use them in a scratch script or the REPL:

    from diagnostics import print_shapes, check_initial_loss, print_grad_norms, check_causality
"""
import math

import torch


def print_shapes(model, *inputs):
    """Run one forward pass and print the output shape of every leaf module."""
    handles = []

    def make_hook(name):
        def hook(mod, inp, out):
            o = out[0] if isinstance(out, tuple) else out
            shp = tuple(o.shape) if torch.is_tensor(o) else type(o).__name__
            print(f"{name:<45} {type(mod).__name__:<18} -> {shp}")
        return hook

    for name, mod in model.named_modules():
        if len(list(mod.children())) == 0:
            handles.append(mod.register_forward_hook(make_hook(name or "<root>")))
    try:
        with torch.no_grad():
            model(*inputs)
    finally:
        for h in handles:
            h.remove()


def check_initial_loss(loss, vocab_size, tol=1.0):
    """A freshly initialised classifier should score about ln(vocab_size)."""
    expected = math.log(vocab_size)
    val = float(loss.detach()) if torch.is_tensor(loss) else float(loss)
    verdict = "OK" if abs(val - expected) <= tol else "SUSPICIOUS"
    print(f"initial loss {val:.3f} vs ln(V) = {expected:.3f}  ->  {verdict}")
    return verdict == "OK"


def print_grad_norms(model):
    """Call after loss.backward(). Zero / None / huge values are all clues."""
    for name, p in model.named_parameters():
        g = "None" if p.grad is None else f"{p.grad.norm().item():.3e}"
        print(f"{name:<50} grad-norm {g:>12}   param-norm {p.norm().item():.3e}")


def check_causality(fn, x, t=None):
    """fn: (B,T,C) -> (B,T,...). Perturb the FUTURE of position t; the output at <= t must not move."""
    t = x.shape[1] // 2 if t is None else t
    x2 = x.clone()
    x2[:, t + 1 :] = torch.randn_like(x2[:, t + 1 :])
    with torch.no_grad():
        a, b = fn(x), fn(x2)
    diff = (a[:, : t + 1] - b[:, : t + 1]).abs().max().item()
    print(f"max change at positions <= {t} after editing the future: {diff:.3e}  ->  {'OK' if diff < 1e-6 else 'LEAK'}")
    return diff < 1e-6
