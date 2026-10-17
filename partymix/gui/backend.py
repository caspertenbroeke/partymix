"""Bridge between the GUI and the worker processes.

The window only talks to a :class:`Backend` (Qt signals in, method calls out). ``ProcessBackend`` is the
real implementation on top of :class:`partymix.worker.WorkerPool`; tests and ``--demo`` use
``partymix.gui.demo.FakeBackend``.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from PySide6.QtCore import QObject, QTimer, Signal


class Backend(QObject):
    # analysis (key = absolute file path)
    analysisStarted = Signal(str)
    analysisProgress = Signal(str, float, str)          # key, fraction (-1 = unknown), text
    analysisDone = Signal(str, object)                  # key, TrackAnalysis
    analysisFailed = Signal(str, str)                   # key, Dutch message
    # ordering / planning
    orderDone = Signal(object)                          # list[int]
    orderFailed = Signal(str)
    planDone = Signal(int, object, object)              # request id, MixPlan, seconds | None
    planFailed = Signal(int, str)
    # mix
    mixProgress = Signal(float, str)
    mixDone = Signal(object, object)                    # MixPlan, list[str] outputs
    mixFailed = Signal(str)
    mixCancelled = Signal()
    # first run model download
    modelProgress = Signal(float, str)
    modelDone = Signal(bool)                            # True = downloaded, False = not needed/unsupported
    modelFailed = Signal(str)

    # --- interface ---------------------------------------------------------------------------
    def model_missing(self) -> bool:
        return False

    def download_model(self) -> None:
        self.modelDone.emit(False)

    def analyze(self, key: str, path: str) -> None:
        raise NotImplementedError

    def cancel_analysis(self, key: str) -> None:
        pass

    def order(self, analyses, pinned_first: Optional[int], pinned_last: Optional[int]) -> None:
        raise NotImplementedError

    def plan(self, request_id: int, analyses_in_order, target_minutes: float) -> None:
        raise NotImplementedError

    def mix(self, analyses_in_order, target_minutes: float, settings, plan=None) -> None:
        raise NotImplementedError

    def cancel_mix(self) -> None:
        pass

    def pair_score(self, a, b) -> Optional[Tuple[float, str]]:
        return None

    def shutdown(self) -> None:
        pass


def normalise_score(score: Any, scale: Optional[float] = None) -> float:
    """Map a pair score to 0..1. ``scale`` = maximum of the planner's scale (100 for
    ``partymix.planner.pair_score``); without it, values > 1 are taken as percentages."""
    s = float(score)
    if scale:
        s /= scale
    elif s > 1.0:
        s /= 100.0
    return min(max(s, 0.0), 1.0)


class ProcessBackend(Backend):
    """Runs analysis in ``workers`` processes and order/plan/mix/model jobs in one extra process."""

    def __init__(self, workers: Optional[int] = None, parent=None, job_refs: Optional[Dict[str, str]] = None):
        super().__init__(parent)
        from partymix.worker import BUILTIN_JOBS, WorkerPool
        self._refs = dict(BUILTIN_JOBS, **(job_refs or {}))
        self._analysis = WorkerPool(workers, name="analyse")
        self._tasks = WorkerPool(1, kill_timeout=8.0, name="taak")
        self._by_key: Dict[str, int] = {}
        self._mix_job: Optional[int] = None
        self._pair_fn = None
        self._pair_fn_loaded = False
        self._closed = False
        self._timer = QTimer(self)
        self._timer.setInterval(100)
        self._timer.timeout.connect(self._poll)
        self._timer.start()

    # --- model --------------------------------------------------------------------------------
    def model_missing(self) -> bool:
        try:
            from partymix import paths
            return not paths.model_path().exists()
        except Exception:
            return False

    def download_model(self) -> None:
        if self._closed:
            return
        self._analysis.pause(True)
        self._tasks.submit(self._refs["model"], tag=("model",))

    # --- analysis -----------------------------------------------------------------------------
    def analyze(self, key: str, path: str) -> None:
        if self._closed:
            return
        if key in self._by_key:
            return
        self._by_key[key] = self._analysis.submit(self._refs["analyze"], path, tag=("analyze", key))

    def cancel_analysis(self, key: str) -> None:
        jid = self._by_key.pop(key, None)
        if jid is not None:
            self._analysis.cancel(jid)

    # --- order / plan / mix -------------------------------------------------------------------
    def order(self, analyses, pinned_first, pinned_last) -> None:
        if self._closed:
            return
        self._tasks.submit(self._refs["order"], list(analyses), pinned_first, pinned_last, tag=("order",))

    def plan(self, request_id, analyses_in_order, target_minutes) -> None:
        if self._closed:
            return
        self._tasks.submit(self._refs["plan"], list(analyses_in_order), float(target_minutes), tag=("plan", request_id))

    def mix(self, analyses_in_order, target_minutes, settings, plan=None) -> None:
        if self._closed:
            return
        self._mix_job = self._tasks.submit(self._refs["mix"], list(analyses_in_order), float(target_minutes), settings,
                                           plan, tag=("mix",))

    def cancel_mix(self) -> None:
        if self._mix_job is not None:
            self._tasks.cancel(self._mix_job)

    def pair_score(self, a, b):
        if not self._pair_fn_loaded:
            self._pair_fn_loaded = True
            try:
                from partymix.planner import pair_score
                self._pair_fn = pair_score
            except Exception:
                self._pair_fn = None
        if self._pair_fn is None:
            return None
        try:
            score, reason = self._pair_fn(a, b)
            return normalise_score(score, 100.0), str(reason)   # planner: 0..100
        except Exception:
            return None

    def shutdown(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._timer.stop()
        self._analysis.shutdown()
        self._tasks.shutdown()

    # --- event pump ---------------------------------------------------------------------------
    def _poll(self) -> None:
        for ev in self._analysis.poll():
            kind, key = ev.tag
            if self._by_key.get(key) != ev.job_id:
                continue                      # removed / superseded
            if ev.kind == "started":
                self.analysisStarted.emit(key)
            elif ev.kind == "progress":
                self.analysisProgress.emit(key, -1.0 if ev.fraction is None else ev.fraction, ev.text)
            elif ev.kind == "done":
                self._by_key.pop(key, None)
                self.analysisDone.emit(key, ev.result)
            elif ev.kind == "error":
                self._by_key.pop(key, None)
                self.analysisFailed.emit(key, ev.error)
            elif ev.kind == "cancelled":
                self._by_key.pop(key, None)
        for ev in self._tasks.poll():
            what = ev.tag[0]
            if what == "model":
                if ev.kind == "progress":
                    self.modelProgress.emit(-1.0 if ev.fraction is None else ev.fraction, ev.text)
                elif ev.kind in ("done", "error", "cancelled"):
                    self._analysis.pause(False)
                    if ev.kind == "error":
                        self.modelFailed.emit(ev.error)
                    else:
                        self.modelDone.emit(ev.result != "unsupported")
            elif what == "order":
                if ev.kind == "done":
                    self.orderDone.emit(list(ev.result))
                elif ev.kind == "error":
                    self.orderFailed.emit(ev.error)
            elif what == "plan":
                if ev.kind == "done":
                    plan, secs = ev.result
                    self.planDone.emit(ev.tag[1], plan, secs)
                elif ev.kind == "error":
                    self.planFailed.emit(ev.tag[1], ev.error)
            elif what == "mix":
                if ev.job_id != self._mix_job:
                    continue
                if ev.kind == "progress":
                    self.mixProgress.emit(-1.0 if ev.fraction is None else ev.fraction, ev.text)
                elif ev.kind == "done":
                    self._mix_job = None
                    plan, outputs = ev.result
                    self.mixDone.emit(plan, outputs)
                elif ev.kind == "error":
                    self._mix_job = None
                    self.mixFailed.emit(ev.error)
                elif ev.kind == "cancelled":
                    self._mix_job = None
                    self.mixCancelled.emit()
