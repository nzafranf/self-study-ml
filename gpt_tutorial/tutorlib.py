"""Test plumbing. You should not need to edit this file.

Tests load *your* code from each module's `starter.py`. Maintainers can point the
same tests at the reference implementations with `TUTOR_TARGET=solutions`.
"""
import importlib.util
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TARGET = os.environ.get("TUTOR_TARGET", "starter")  # "starter" | "solutions"
SAMPLE_TEXT = (ROOT / "data" / "sample.txt").read_text(encoding="utf-8")


def _import(path: Path, name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_module(dirname: str):
    """Load a module directory's implementation (starter.py, or the reference)."""
    if TARGET == "solutions":
        path = ROOT / "solutions" / dirname / "solution.py"
    else:
        path = ROOT / dirname / "starter.py"
        if not path.exists():  # stretch exercises live in stretch/<dirname>/
            path = ROOT / "stretch" / dirname / "starter.py"
    return _import(path, f"_tutor_{TARGET}_{dirname}")


def load_for_test(test_file: str):
    """Load the module that lives in the same directory as `test_file`."""
    return load_module(Path(test_file).resolve().parent.name)


def load_debug(test_file: str):
    """Debug exercises: the planted-bug file (default) or the fixed reference."""
    ex = Path(test_file).resolve().parent.name
    if TARGET == "solutions":
        path = ROOT / "solutions" / "debug" / ex / "fixed.py"
    else:
        path = ROOT / "debug" / ex / "buggy.py"
    return _import(path, f"_tutor_dbg_{TARGET}_{ex}")
