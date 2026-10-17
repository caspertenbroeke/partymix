"""Entry point: ``python -m partymix`` (GUI) or ``python -m partymix <command> …`` (CLI).

Also the PyInstaller entry script. ``freeze_support()`` must run before anything else so that the
spawned worker processes of a frozen Windows build do not start a second GUI.
"""
import multiprocessing
import sys


def run() -> int:
    multiprocessing.freeze_support()
    from partymix.cli import main
    return main()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    sys.exit(run())
