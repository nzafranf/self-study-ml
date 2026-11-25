import random


def make_example(a, b, n_digits):
    ans = f"{a + b:0{n_digits + 1}d}"[::-1]
    return f"{a:0{n_digits}d}+{b:0{n_digits}d}={ans}"


def answer_value(example):
    return int(example.split("=")[1][::-1])


def build_dataset(n_digits, n_examples, seed=0):
    rng = random.Random(seed)
    hi = 10**n_digits - 1
    return [make_example(rng.randint(0, hi), rng.randint(0, hi), n_digits) for _ in range(n_examples)]
