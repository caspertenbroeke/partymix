"""Job orchestration: run CPU-heavy work (analysis, planning, rendering) in worker processes.

Design
------
* ``WorkerPool`` owns up to ``max_workers`` long-lived child processes (``spawn`` context, so it
  works the same on Windows, Linux and inside a PyInstaller bundle – ``freeze_support()`` is called
  in ``partymix/__main__.py``).
* Scheduling happens in the parent: a job is only handed to an idle worker, so every job can be
  cancelled individually (pending jobs are simply dropped, running jobs get their worker's cancel
  event set and – if they do not stop within ``kill_timeout`` – the worker is terminated).
* If a worker process dies (segfault, OOM, ``os._exit``) while running a job, that job is reported as
  failed with a Dutch message and a fresh worker takes over the remaining jobs.
* ``poll()`` is non-blocking and returns a list of :class:`JobEvent`. The GUI calls it from a QTimer,
  the CLI from a simple loop (``wait()``).

Job functions are referenced as ``"module:function"`` strings (``BUILTIN_JOBS`` maps short names).
They are called as ``fn(*args, progress=<callable>, cancel=<CancelToken>)`` in the child.
"""
from __future__ import annotations

import itertools
import multiprocessing as mp
import os
import queue
import sys
import time
import traceback
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

__all__ = ["WorkerPool", "JobEvent", "CancelToken", "Cancelled", "dutch_error",
           "default_workers", "BUILTIN_JOBS"]

# --------------------------------------------------------------------------------------------
# helpers usable in both parent and child


class Cancelled(Exception):
    """Raised by job code (or by our progress callback) when the job was cancelled."""


class CancelToken:
    """Wraps a multiprocessing Event so it works for both conventions used by the other modules:
    ``cancel()`` (callable returning bool) and ``cancel.is_set()`` (threading.Event-like)."""

    def __init__(self, event):
        self._event = event

    def is_set(self) -> bool:
        try:
            return self._event.is_set()
        except Exception:  # pragma: no cover - broken pipe during shutdown
            return True

    def __call__(self) -> bool:
        return self.is_set()

    def set(self) -> None:
        self._event.set()

    def clear(self) -> None:
        self._event.clear()

    def wait(self, timeout: Optional[float] = None) -> bool:
        return self._event.wait(timeout)

    def raise_if_set(self) -> None:
        if self.is_set():
            raise Cancelled()

    def __bool__(self) -> bool:  # ``if cancel:`` style code must not think it is cancelled
        return True


def parse_progress(args: tuple, kwargs: dict) -> Tuple[Optional[float], str]:
    """Accept the many shapes a progress callback may be called with.

    ``cb(0.42)``, ``cb(42)`` (percent), ``cb(0.42, "tekst")``, ``cb("tekst", 0.42)``,
    ``cb(done_bytes, total_bytes)``, ``cb(fraction=…, message=…)``.
    """
    nums = [a for a in args if isinstance(a, (int, float)) and not isinstance(a, bool)]
    texts = [a for a in args if isinstance(a, str)]
    for k in ("fraction", "frac", "progress", "value", "p"):
        if k in kwargs and isinstance(kwargs[k], (int, float)):
            nums.insert(0, kwargs[k])
    for k in ("message", "msg", "text", "stage", "step", "status"):
        if k in kwargs and isinstance(kwargs[k], str):
            texts.insert(0, kwargs[k])
    frac: Optional[float] = None
    if len(nums) >= 2 and nums[1] and nums[1] > 1 and nums[0] <= nums[1]:
        frac = float(nums[0]) / float(nums[1])            # done / total
    elif nums:
        frac = float(nums[0])
        if frac > 1.0:
            frac = frac / 100.0                           # percent
    if frac is not None:
        frac = min(max(frac, 0.0), 1.0)
    return frac, (texts[0] if texts else "")


_DUTCH_HINTS = (" niet ", " mislukt", "fout", "bestand", "geen ", "kan ", "wordt", "ongeldig")


def dutch_error(exc: BaseException, what: str = "Analyse") -> str:
    """Short Dutch, user facing error message for an exception."""
    msg = str(exc).strip().splitlines()[0] if str(exc).strip() else ""
    if isinstance(exc, FileNotFoundError):
        name = getattr(exc, "filename", None)
        return f"Bestand niet gevonden: {os.path.basename(str(name))}" if name else (msg or "Bestand niet gevonden")
    if isinstance(exc, PermissionError):
        return "Geen toegang tot het bestand (is het geopend in een ander programma?)"
    if isinstance(exc, MemoryError):
        return f"{what} mislukt: te weinig werkgeheugen"
    if isinstance(exc, Cancelled):
        return "Geannuleerd"
    if msg and any(h in f" {msg.lower()} " for h in _DUTCH_HINTS):
        return msg                                       # already Dutch (from our own modules)
    if msg:
        return f"{what} mislukt: {msg[:200]}"
    return f"{what} mislukt ({type(exc).__name__})"


def default_workers() -> int:
    """madmom + onnxruntime are CPU heavy: at most 2 parallel analyses, 1 on small machines."""
    env = os.environ.get("PARTYMIX_WORKERS")
    if env and env.isdigit() and int(env) > 0:
        return int(env)
    n = os.cpu_count() or 1
    return 2 if n >= 4 else 1


# --------------------------------------------------------------------------------------------
# built-in jobs (run in the child process)

def _job_analyze(path: str, progress=None, cancel=None):
    from partymix.analysis import analyze_track
    return analyze_track(path, progress, cancel)


def _job_order(analyses, pinned_first=None, pinned_last=None, progress=None, cancel=None):
    from partymix.planner import order_tracks
    return order_tracks(analyses, pinned_first, pinned_last)


def _job_plan(analyses_in_order, target_minutes, progress=None, cancel=None):
    """Plan the mix and estimate its length. Returns ``(MixPlan, seconds)``."""
    from partymix.planner import plan_mix
    plan = plan_mix(analyses_in_order, target_minutes)
    seconds = None
    try:
        from partymix.render import estimate_duration
        seconds = float(estimate_duration(plan, analyses_in_order))
    except Exception:
        seconds = None
    return plan, seconds


def _job_mix(analyses_in_order, target_minutes, settings, plan=None, progress=None, cancel=None):
    """(Re)plan if needed and render. Returns ``(MixPlan, [output paths])``."""
    if plan is None:
        if progress:
            progress(0.0, "Mix plannen…")
        from partymix.planner import plan_mix
        plan = plan_mix(analyses_in_order, target_minutes)
    if cancel is not None and cancel.is_set():
        raise Cancelled()
    from partymix.render import render_mix
    outputs = render_mix(plan, analyses_in_order, settings, progress, cancel)
    return plan, list(outputs or [])


def _job_model(progress=None, cancel=None):
    """Download the vocal model if analysis offers it. Returns "unsupported" when it does not
    (analysis will then fetch it itself on first use)."""
    fn = find_model_downloader()
    if fn is None:
        return "unsupported"
    try:
        fn(progress)
    except TypeError:
        fn(progress_cb=progress)
    return "ok"


def find_model_downloader() -> Optional[Callable]:
    """Locate the analysis module's model download function (name not fixed by the contract)."""
    import importlib
    for mod_name in ("partymix.analysis", "partymix.analysis.vocals", "partymix.analysis.model"):
        try:
            mod = importlib.import_module(mod_name)
        except Exception:
            continue
        for name in ("ensure_model", "download_model", "ensure_vocal_model", "download_vocal_model"):
            fn = getattr(mod, name, None)
            if callable(fn):
                return fn
    return None


BUILTIN_JOBS: Dict[str, str] = {
    "analyze": "partymix.worker:_job_analyze",
    "order": "partymix.worker:_job_order",
    "plan": "partymix.worker:_job_plan",
    "mix": "partymix.worker:_job_mix",
    "model": "partymix.worker:_job_model",
}


def _resolve(ref: str) -> Callable:
    import importlib
    ref = BUILTIN_JOBS.get(ref, ref)
    mod_name, _, fn_name = ref.partition(":")
    return getattr(importlib.import_module(mod_name), fn_name)


def _job_label(ref: str) -> str:
    """Dutch name of a job for error messages ("analyze" / "partymix.worker:_job_analyze" → "Analyse")."""
    name = ref.split(":")[-1].lstrip("_")
    name = name[4:] if name.startswith("job_") else name
    for key, label in (("analy", "Analyse"), ("mix", "Mix maken"), ("plan", "Planning"),
                       ("order", "Volgorde bepalen"), ("model", "Download")):
        if name.startswith(key):
            return label
    return "Taak"


# --------------------------------------------------------------------------------------------
# child process main

def _child_main(worker_id: int, task_q, event_q, cancel_ev) -> None:  # pragma: no cover - runs in child
    if sys.stdout is None:                  # windowed PyInstaller app: no console
        sys.stdout = open(os.devnull, "w")
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    token = CancelToken(cancel_ev)
    while True:
        try:
            task = task_q.get()
        except (EOFError, OSError, KeyboardInterrupt):
            return
        if task is None:
            return
        job_id, ref, args, kwargs = task
        last = [0.0]

        def progress(*a, **kw):
            frac, text = parse_progress(a, kw)
            now = time.monotonic()
            # throttle: max ~10 messages/s, but always pass text changes and the end
            if frac is None or frac >= 1.0 or text or now - last[0] > 0.1:
                last[0] = now
                event_q.put(("progress", job_id, frac, text))
            if token.is_set():
                raise Cancelled()

        try:
            from partymix.paths import setup_logging
            setup_logging()
            fn = _resolve(ref)
            result = fn(*args, progress=progress, cancel=token, **kwargs)
            if token.is_set():
                event_q.put(("cancelled", job_id))
            else:
                event_q.put(("done", job_id, result))
        except Cancelled:
            event_q.put(("cancelled", job_id))
        except BaseException as exc:  # noqa: BLE001 - report everything, keep the worker alive
            if token.is_set():
                event_q.put(("cancelled", job_id))
            else:
                what = _job_label(ref)
                try:
                    import logging
                    logging.getLogger("partymix").error("%s mislukt (%s)\n%s", what, args[:1], traceback.format_exc())
                except Exception:
                    pass
                try:
                    event_q.put(("error", job_id, dutch_error(exc, what), traceback.format_exc()))
                except Exception:  # unpicklable exception text, should not happen
                    event_q.put(("error", job_id, f"{what} mislukt", ""))
        finally:
            token.clear()


# --------------------------------------------------------------------------------------------
# parent side

@dataclass
class JobEvent:
    job_id: int
    kind: str                      # started | progress | done | error | cancelled
    fraction: Optional[float] = None
    text: str = ""
    result: Any = None
    error: str = ""                # Dutch user message
    details: str = ""              # traceback
    tag: Any = None                # whatever the submitter attached to the job


@dataclass
class _Job:
    job_id: int
    ref: str
    args: tuple
    kwargs: dict
    tag: Any = None
    started: bool = False


@dataclass
class _Worker:
    wid: int
    proc: Any
    task_q: Any
    cancel_ev: Any
    job: Optional[_Job] = None
    cancel_at: Optional[float] = None


class WorkerPool:
    """A small process pool with per-job cancel and crash isolation (see module docstring)."""

    def __init__(self, max_workers: Optional[int] = None, kill_timeout: float = 5.0, name: str = "pm"):
        self.max_workers = max(1, int(max_workers or default_workers()))
        self.kill_timeout = kill_timeout
        self.name = name
        self._ctx = mp.get_context("spawn")
        self._event_q = self._ctx.Queue()
        self._workers: Dict[int, _Worker] = {}
        self._pending: List[_Job] = []
        self._jobs: Dict[int, _Job] = {}
        self._ids = itertools.count(1)
        self._wids = itertools.count(1)
        self._paused = False
        self._closed = False
        self._late: List[JobEvent] = []      # events generated in the parent (started / dropped)

    # ---- public API -------------------------------------------------------------------------
    def submit(self, ref: str, *args, tag: Any = None, **kwargs) -> int:
        if self._closed:
            raise RuntimeError("WorkerPool is gesloten")
        job = _Job(next(self._ids), ref, args, kwargs, tag)
        self._jobs[job.job_id] = job
        self._pending.append(job)
        self._dispatch()
        return job.job_id

    def cancel(self, job_id: int) -> bool:
        """Cancel one job. A pending job is dropped immediately (a ``cancelled`` event follows on
        the next poll); a running job is asked to stop and killed after ``kill_timeout``."""
        job = self._jobs.get(job_id)
        if job is None:
            return False
        for i, p in enumerate(self._pending):
            if p.job_id == job_id:
                del self._pending[i]
                self._jobs.pop(job_id, None)
                self._late.append(JobEvent(job_id, "cancelled", tag=job.tag))
                return True
        for w in self._workers.values():
            if w.job is not None and w.job.job_id == job_id and w.cancel_at is None:
                w.cancel_ev.set()
                w.cancel_at = time.monotonic()
                return True
        return False

    def cancel_all(self) -> None:
        for jid in [j.job_id for j in self._pending]:
            self.cancel(jid)
        for w in list(self._workers.values()):
            if w.job is not None:
                self.cancel(w.job.job_id)

    def pause(self, paused: bool = True) -> None:
        """While paused, pending jobs are not started (used while the vocal model downloads)."""
        self._paused = paused
        if not paused:
            self._dispatch()

    @property
    def busy(self) -> bool:
        return bool(self._pending) or any(w.job is not None for w in self._workers.values())

    def active_jobs(self) -> List[int]:
        return list(self._jobs)

    def poll(self) -> List[JobEvent]:
        events: List[JobEvent] = list(self._late)
        self._late.clear()
        # 1. drain the event queue
        while True:
            try:
                msg = self._event_q.get_nowait()
            except queue.Empty:
                break
            except (EOFError, OSError):
                break
            ev = self._handle_msg(msg)
            if ev is not None:
                events.append(ev)
        # 2. dead workers / cancel timeouts
        now = time.monotonic()
        for wid, w in list(self._workers.items()):
            if w.job is not None and w.cancel_at is not None and now - w.cancel_at > self.kill_timeout:
                job = w.job
                self._kill(w)
                self._jobs.pop(job.job_id, None)
                events.append(JobEvent(job.job_id, "cancelled", tag=job.tag))
            elif not w.proc.is_alive():
                # drain once more: the job might have finished right before the exit
                while True:
                    try:
                        msg = self._event_q.get_nowait()
                    except (queue.Empty, EOFError, OSError):
                        break
                    ev = self._handle_msg(msg)
                    if ev is not None:
                        events.append(ev)
                job = w.job
                self._workers.pop(wid, None)
                if job is not None and job.job_id in self._jobs:
                    self._jobs.pop(job.job_id, None)
                    if w.cancel_at is not None:
                        events.append(JobEvent(job.job_id, "cancelled", tag=job.tag))
                    else:
                        code = w.proc.exitcode
                        events.append(JobEvent(
                            job.job_id, "error", tag=job.tag,
                            error="Verwerking is onverwacht gestopt (crash"
                                  + (f", code {code}" if code not in (None, 0) else "") + ")",
                        ))
        # 3. hand out work
        self._dispatch()
        return events

    def wait(self, job_ids=None, timeout: Optional[float] = None, interval: float = 0.05):
        """Generator for non-GUI use: yields events until the given (or all) jobs finished."""
        t0 = time.monotonic()
        remaining = set(job_ids) if job_ids is not None else None
        while True:
            for ev in self.poll():
                yield ev
                if remaining is not None and ev.kind in ("done", "error", "cancelled"):
                    remaining.discard(ev.job_id)
            if remaining is not None and not remaining:
                return
            if remaining is None and not self.busy and not self._late:
                return
            if timeout is not None and time.monotonic() - t0 > timeout:
                raise TimeoutError("WorkerPool.wait timeout")
            time.sleep(interval)

    def shutdown(self, wait: float = 2.0) -> None:
        self._closed = True
        self._pending.clear()
        for w in self._workers.values():
            if w.job is not None:
                w.cancel_ev.set()
            try:
                w.task_q.put(None)
            except Exception:
                pass
        deadline = time.monotonic() + wait
        for w in list(self._workers.values()):
            w.proc.join(max(0.0, deadline - time.monotonic()))
            if w.proc.is_alive():
                w.proc.terminate()
                w.proc.join(1.0)
        self._workers.clear()
        try:
            self._event_q.close()
            self._event_q.cancel_join_thread()
        except Exception:
            pass

    # ---- internals --------------------------------------------------------------------------
    def _handle_msg(self, msg) -> Optional[JobEvent]:
        kind, job_id = msg[0], msg[1]
        job = self._jobs.get(job_id)
        if job is None:
            return None
        if kind == "started":
            return JobEvent(job_id, "started", tag=job.tag)
        if kind == "progress":
            return JobEvent(job_id, "progress", fraction=msg[2], text=msg[3] or "", tag=job.tag)
        # terminal
        self._jobs.pop(job_id, None)
        for w in self._workers.values():
            if w.job is not None and w.job.job_id == job_id:
                w.job = None
                w.cancel_at = None
        if kind == "done":
            return JobEvent(job_id, "done", fraction=1.0, result=msg[2], tag=job.tag)
        if kind == "cancelled":
            return JobEvent(job_id, "cancelled", tag=job.tag)
        return JobEvent(job_id, "error", error=msg[2], details=msg[3] if len(msg) > 3 else "", tag=job.tag)

    def _spawn(self) -> _Worker:
        wid = next(self._wids)
        task_q = self._ctx.Queue()
        cancel_ev = self._ctx.Event()
        proc = self._ctx.Process(target=_child_main, args=(wid, task_q, self._event_q, cancel_ev),
                                 name=f"{self.name}-worker-{wid}", daemon=True)
        proc.start()
        w = _Worker(wid, proc, task_q, cancel_ev)
        self._workers[wid] = w
        return w

    def _kill(self, w: _Worker) -> None:
        try:
            w.proc.terminate()
            w.proc.join(2.0)
            if w.proc.is_alive() and hasattr(w.proc, "kill"):
                w.proc.kill()
                w.proc.join(1.0)
        except Exception:
            pass
        self._workers.pop(w.wid, None)

    def _dispatch(self) -> None:
        if self._paused or self._closed:
            return
        while self._pending:
            idle = [w for w in self._workers.values() if w.job is None and w.proc.is_alive()]
            if not idle:
                if len(self._workers) >= self.max_workers:
                    return
                idle = [self._spawn()]
            w = idle[0]
            job = self._pending.pop(0)
            w.job = job
            w.cancel_at = None
            w.cancel_ev.clear()
            job.started = True
            w.task_q.put((job.job_id, job.ref, job.args, job.kwargs))
            self._late.append(JobEvent(job.job_id, "started", tag=job.tag))
