"""Small helpers shared by the render modules (subprocess with cancel, progress, errors)."""
from __future__ import annotations

import os
import subprocess
import time
from typing import Callable, List, Optional

__all__ = ["RenderCancelled", "RenderError", "run_tool", "check_cancel", "ProgressCb", "sub_progress"]

ProgressCb = Optional[Callable[[float, str], None]]


class RenderCancelled(Exception):
    """Raised when the cancel event is set during rendering."""


class RenderError(RuntimeError):
    pass


def check_cancel(cancel_event) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise RenderCancelled("Geannuleerd")


def run_tool(cmd: List[str], cwd: Optional[str] = None, cancel_event=None, what: str = "") -> None:
    """Run an external tool (list args, no shell, no console window), polling ``cancel_event``."""
    kw = {}
    if os.name == "nt":
        kw["creationflags"] = 0x08000000  # CREATE_NO_WINDOW
    p = subprocess.Popen(cmd, cwd=cwd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                         stderr=subprocess.PIPE, **kw)
    try:
        while True:
            try:
                _, err = p.communicate(timeout=0.25)
                break
            except subprocess.TimeoutExpired:
                if cancel_event is not None and cancel_event.is_set():
                    p.kill()
                    p.communicate()
                    raise RenderCancelled("Geannuleerd")
    except BaseException:
        if p.poll() is None:
            p.kill()
            p.wait()
        raise
    if p.returncode != 0:
        msg = (err or b"").decode("utf-8", "replace").strip()[-800:]
        raise RenderError(f"{what or os.path.basename(cmd[0])} mislukt (code {p.returncode}): {msg}")


def sub_progress(cb: ProgressCb, lo: float, hi: float) -> ProgressCb:
    """Map a 0..1 progress callback onto [lo, hi] of ``cb``."""
    if cb is None:
        return None

    def f(frac: float, msg: str) -> None:
        cb(lo + (hi - lo) * max(0.0, min(1.0, frac)), msg)
    return f


class Timer:
    def __init__(self):
        self.t0 = time.time()

    def __call__(self) -> float:
        return time.time() - self.t0
