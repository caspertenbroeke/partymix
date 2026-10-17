"""Frozen-app entry point used by packaging/partymix.spec (both PartyMix.exe and partymix-cli.exe).

It behaves exactly like `python -m partymix ...` (GUI by default, CLI sub-commands; argument
handling lives in partymix/__main__.py + partymix/cli.py), including `--selftest` (implemented
in partymix/cli.py by the lead). Packaging adds one extra mode: `--bundle-selftest` as the first
argument runs packaging/partymix_selftest.py (bundled as a top-level module), a deeper check of
the frozen bundle used by CI.

freeze_support() must run before anything else: the GUI starts worker processes with
multiprocessing, and in a frozen exe every child process re-launches this same file.
"""
import multiprocessing
import runpy
import sys

if __name__ == "__main__":
    multiprocessing.freeze_support()
    # A windowed exe (console=False) has no stdout/stderr; give stray prints somewhere harmless.
    if sys.stdout is None or sys.stderr is None:
        import os
        _null = open(os.devnull, "w", encoding="utf-8")
        sys.stdout = sys.stdout or _null
        sys.stderr = sys.stderr or _null
    if sys.argv[1:2] == ["--bundle-selftest"]:
        import partymix_selftest
        sys.exit(partymix_selftest.main(sys.argv[2:]))
    runpy.run_module("partymix", run_name="__main__", alter_sys=True)
