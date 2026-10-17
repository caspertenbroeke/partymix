"""Tests for partymix.worker (process pool with progress, cancel and crash isolation)."""
import os
import time

import pytest

from partymix.worker import WorkerPool, parse_progress, dutch_error, CancelToken

HERE = __name__  # module name as importable by the spawned children (tests dir is on sys.path)


# ---- job functions executed in the child ----------------------------------------------------
def job_ok(x, progress=None, cancel=None):
    for i in range(3):
        progress(i / 3, f"stap {i}")
    return x * 2


def job_crash(x, progress=None, cancel=None):
    progress(0.5, "bijna crash")
    os._exit(3)


def job_raise(x, progress=None, cancel=None):
    raise ValueError("kapot audiobestand")


def job_slow(x, progress=None, cancel=None):
    for _ in range(400):
        if cancel.is_set():
            return None
        time.sleep(0.01)
        progress(0.1)
    return x


def job_stubborn(x, progress=None, cancel=None):
    time.sleep(30)  # ignores cancel
    return x


def job_analyze_fake(path, progress=None, cancel=None):
    """Stand-in for analysis.analyze_track used by tests/test_gui_backend.py."""
    if "crash" in path:
        os._exit(7)
    progress(0.5, "Beats zoeken")
    from partymix.gui.demo import analysis_for_path
    return analysis_for_path(path)


# ---- tests ----------------------------------------------------------------------------------
def test_parse_progress_shapes():
    assert parse_progress((0.5,), {}) == (0.5, "")
    assert parse_progress((50,), {}) == (0.5, "")
    assert parse_progress((0.25, "Beats"), {}) == (0.25, "Beats")
    assert parse_progress(("Beats", 0.25), {}) == (0.25, "Beats")
    assert parse_progress((512, 1024), {}) == (0.5, "")
    assert parse_progress((), {"fraction": 0.1, "message": "x"}) == (0.1, "x")
    assert parse_progress(("alleen tekst",), {}) == (None, "alleen tekst")


def test_dutch_error():
    assert "niet gevonden" in dutch_error(FileNotFoundError(2, "No such file", "/a/b.mp3"))
    assert dutch_error(ValueError("boom"), "Analyse") == "Analyse mislukt: boom"
    assert dutch_error(RuntimeError("Bestand is geen audio")) == "Bestand is geen audio"


def test_cancel_token_conventions():
    class Ev:
        v = False
        def is_set(self): return self.v
        def set(self): self.v = True
        def clear(self): self.v = False
        def wait(self, t=None): return self.v
    t = CancelToken(Ev())
    assert bool(t) and not t() and not t.is_set()
    t.set()
    assert t() and t.is_set()


def test_pool_results_errors_and_crash():
    pool = WorkerPool(max_workers=2, kill_timeout=1.0)
    try:
        ids = {
            pool.submit(f"{HERE}:job_ok", 1, tag="a"): "a",
            pool.submit(f"{HERE}:job_crash", 2, tag="b"): "b",
            pool.submit(f"{HERE}:job_raise", 3, tag="c"): "c",
            pool.submit(f"{HERE}:job_ok", 4, tag="d"): "d",
        }
        final, progress = {}, []
        for ev in pool.wait(list(ids), timeout=90):
            if ev.kind == "progress":
                progress.append(ev.tag)
            elif ev.kind in ("done", "error", "cancelled"):
                final[ev.tag] = ev
        assert final["a"].kind == "done" and final["a"].result == 2
        assert final["d"].kind == "done" and final["d"].result == 8
        assert final["b"].kind == "error" and "crash" in final["b"].error
        assert final["c"].kind == "error" and "kapot audiobestand" in final["c"].error
        assert "a" in progress
    finally:
        pool.shutdown()


def test_pool_cancel_pending_running_and_stubborn():
    pool = WorkerPool(max_workers=1, kill_timeout=0.5)
    try:
        j1 = pool.submit(f"{HERE}:job_slow", 1)
        j2 = pool.submit(f"{HERE}:job_ok", 2)          # pending behind j1
        j3 = pool.submit(f"{HERE}:job_stubborn", 3)    # pending
        assert pool.cancel(j2)
        res = {}
        t0 = time.monotonic()
        cancelled_j1 = False
        for ev in pool.wait([j1, j2, j3], timeout=120):
            if ev.kind == "progress" and ev.job_id == j1 and not cancelled_j1:
                pool.cancel(j1)
                cancelled_j1 = True
            if ev.kind == "started" and ev.job_id == j3:
                time.sleep(0.3)
                pool.cancel(j3)
            if ev.kind in ("done", "error", "cancelled"):
                res[ev.job_id] = ev.kind
        assert res == {j1: "cancelled", j2: "cancelled", j3: "cancelled"}
        assert time.monotonic() - t0 < 60
        # pool still usable after killing the stubborn worker
        j4 = pool.submit(f"{HERE}:job_ok", 5)
        done = [ev for ev in pool.wait([j4], timeout=60) if ev.kind == "done"]
        assert done and done[0].result == 10
    finally:
        pool.shutdown()
