"""Mock data + a fake backend: used by the tests, screenshots and ``python -m partymix --demo``.

Nothing here touches audio; analyses are synthesised from a small table (the 14 tracks of the
hand-made reference mix)."""
from __future__ import annotations

import hashlib
import random
from pathlib import Path
from typing import List, Optional

from PySide6.QtCore import QTimer

from partymix.model import MixPlan, PlannedTrack, Section, TrackAnalysis, Transition
from partymix.gui.backend import Backend

DEMO_TRACKS = [
    # artist, title, bpm, key, energy, duration
    ("Swedish House Mafia", "World Hold On", 127.0, "8B", 6.2, 214),
    ("Edward Maya", "Stereo Love", 127.0, "4A", 5.8, 248),
    ("Kesha", "Die Young", 128.0, "1A", 6.8, 213),
    ("Avicii", "Levels", 126.0, "2A", 7.1, 199),
    ("Snollebollekes", "Rampeneren", 125.0, "7B", 6.9, 197),
    ("Lil Kleine", "It's That Time", 130.0, "9A", 7.4, 186),
    ("David Guetta", "Memories (Extended)", 130.0, "10A", 7.0, 278),
    ("Kaolo", "Kaolo Beat", 132.0, "11A", 7.9, 168),
    ("Fuego", "Y Que Fue", 133.0, "12A", 8.1, 191),
    ("CNCO", "La Mamá", 129.7, "6A", 7.3, 173),
    ("Snollebollekes", "Schudden", 135.1, "5B", 8.8, 205),
    ("Brokko", "Brokko Maken", 135.0, "6B", 8.6, 162),
    ("Crazy Frog", "Axel F", 138.0, "3A", 9.2, 166),
    ("Vengaboys", "We Like To Party", 136.0, "6B", 9.0, 222),
]


def demo_paths(folder: str = "C:/Muziek/Feest") -> List[str]:
    return [f"{folder}/{i + 1:02d} - {a} - {t}.mp3" for i, (a, t, *_rest) in enumerate(DEMO_TRACKS)]


def make_analysis(path: str, artist: str, title: str, bpm: float, key: str, energy: float,
                  duration: float) -> TrackAnalysis:
    per = 60.0 / bpm
    n = int((duration - 0.5) / per)
    beats = [round(0.25 + i * per, 4) for i in range(n)]
    bars = (n - 0) // 4
    rnd = random.Random(title)
    sections, b = [], 0
    for kind, length in (("intro", 16), ("verse", 16), ("build", 8), ("drop", 16), ("break", 8),
                         ("build", 8), ("drop", 16), ("outro", 16)):
        if b >= bars:
            break
        sections.append(Section(b, min(bars, b + length), kind))
        b += length
    return TrackAnalysis(
        path=path, file_hash=hashlib.sha1(path.encode()).hexdigest(), title=title, artist=artist,
        duration=float(duration), sample_rate=44100, bpm=bpm, constant_tempo=True, beats=beats,
        first_downbeat_index=0, key=key, key_confidence=0.8, phrase_offset=0,
        bar_bass=[rnd.uniform(-30, -10) for _ in range(bars)], bar_rms=[rnd.uniform(-20, -8) for _ in range(bars)],
        sections=sections, energy=energy, loudness_lufs=-9.0,
    )


def mock_analyses(paths: Optional[List[str]] = None) -> List[TrackAnalysis]:
    paths = paths or demo_paths()
    return [make_analysis(p, a, t, bpm, k, e, d) for p, (a, t, bpm, k, e, d) in zip(paths, DEMO_TRACKS)]


def analysis_for_path(path: str) -> TrackAnalysis:
    stem = Path(path).stem.lower()
    for a, t, bpm, k, e, d in DEMO_TRACKS:
        if t.lower() in stem:
            return make_analysis(path, a, t, bpm, k, e, d)
    rnd = random.Random(path)
    return make_analysis(path, "Onbekend", Path(path).stem, round(rnd.uniform(122, 136), 1),
                         f"{rnd.randint(1, 12)}{rnd.choice('AB')}", round(rnd.uniform(4, 9), 1), rnd.randint(160, 260))


# --- simple heuristics standing in for the real planner ------------------------------------------
def camelot_distance(a: str, b: str) -> int:
    try:
        na, la, nb, lb = int(a[:-1]), a[-1], int(b[:-1]), b[-1]
    except (ValueError, IndexError):
        return 6
    d = min((na - nb) % 12, (nb - na) % 12)
    return d + (0 if la == lb else 1)


def demo_pair_score(a: TrackAnalysis, b: TrackAnalysis):
    tempo = abs(a.bpm - b.bpm) / a.bpm * 100
    kd = camelot_distance(a.key, b.key)
    score = 1.0 - min(tempo / 8, 1) * 0.5 - min(kd / 4, 1) * 0.5
    parts = []
    parts.append(f"tempo {a.bpm:.0f} → {b.bpm:.0f} BPM ({tempo:.1f}%)")
    if kd == 0:
        parts.append(f"zelfde toonsoort ({a.key})")
    elif kd == 1:
        parts.append(f"toonsoort past ({a.key} → {b.key})")
    else:
        parts.append(f"toonsoort botst ({a.key} → {b.key})")
    if b.energy - a.energy > 1.5:
        parts.append("grote energiesprong")
    return max(score, 0.0), "; ".join(parts).capitalize()


def demo_order(analyses, first=None, last=None) -> List[int]:
    idx = list(range(len(analyses)))
    fixed_first = [first] if first is not None else []
    fixed_last = [last] if last is not None and last != first else []
    rest = sorted((i for i in idx if i not in fixed_first + fixed_last), key=lambda i: analyses[i].energy)
    return fixed_first + rest + fixed_last


def demo_plan(analyses, minutes: float):
    tracks, trans = [], []
    for i, a in enumerate(analyses):
        tracks.append(PlannedTrack(a.file_hash, 0, max(8, min(a.n_bars, 72))))
        if i:
            prev = analyses[i - 1]
            trans.append(Transition(16 if i % 2 else 8, round((prev.bpm + a.bpm) / 2, 1), "swap", {"swap": 0.5}))
    plan = MixPlan(tracks, trans, analyses[0].bpm, analyses[-1].bpm, target_minutes=minutes)
    from partymix.cli import tracklist_entries
    ent = tracklist_entries(plan, analyses)
    last = tracks[-1]
    secs = ent[-1][0] + (last.end_bar - last.start_bar) * 240 / analyses[-1].bpm
    return plan, secs


class FakeBackend(Backend):
    """Answers every request with synthetic data. ``delay_ms`` simulates work; with ``auto=False``
    analyses stay pending until :meth:`finish_all` (handy for screenshots of intermediate states)."""

    def __init__(self, delay_ms: int = 0, auto: bool = True, model_missing: bool = False, parent=None):
        super().__init__(parent)
        self.delay_ms = delay_ms
        self.auto = auto
        self._model_missing = model_missing
        self.requests: list = []
        self.pending: List[tuple] = []
        self.mix_settings = None
        self._mix_cancelled = False

    def _later(self, fn, ms=None):
        QTimer.singleShot(self.delay_ms if ms is None else ms, fn)

    def model_missing(self) -> bool:
        return self._model_missing

    def download_model(self) -> None:
        self.requests.append(("model",))
        steps = [0.0, 0.35, 0.7, 1.0]
        for k, f in enumerate(steps):
            self._later(lambda f=f: self.modelProgress.emit(f, "Stemmenmodel downloaden"), k * self.delay_ms)
        self._later(lambda: (setattr(self, "_model_missing", False), self.modelDone.emit(True)),
                    len(steps) * self.delay_ms)

    def analyze(self, key, path):
        self.requests.append(("analyze", key))
        self.pending.append((key, path))
        if self.auto:
            self._later(lambda: self._progress(key, 0.5, "Beats zoeken"))
            self._later(lambda: self._finish(key, path), self.delay_ms * 2)

    def _progress(self, key, f, text):
        self.analysisStarted.emit(key)
        self.analysisProgress.emit(key, f, text)

    def _finish(self, key, path):
        if (key, path) not in self.pending:
            return
        self.pending.remove((key, path))
        if "kapot" in Path(path).name.lower():
            self.analysisFailed.emit(key, "Analyse mislukt: bestand is beschadigd of geen audio")
        else:
            self.analysisDone.emit(key, analysis_for_path(path))

    def finish_all(self):
        for key, path in list(self.pending):
            self._finish(key, path)

    def cancel_analysis(self, key):
        self.pending = [p for p in self.pending if p[0] != key]

    def order(self, analyses, pinned_first, pinned_last):
        self.requests.append(("order", pinned_first, pinned_last))
        self._later(lambda: self.orderDone.emit(demo_order(analyses, pinned_first, pinned_last)))

    def plan(self, request_id, analyses_in_order, target_minutes):
        self.requests.append(("plan", request_id))
        lst = list(analyses_in_order)
        self._later(lambda: self.planDone.emit(request_id, *demo_plan(lst, target_minutes)))

    def mix(self, analyses_in_order, target_minutes, settings, plan=None):
        self.requests.append(("mix",))
        self.mix_settings = settings
        self._mix_cancelled = False
        lst = list(analyses_in_order)
        steps = [(0.05, "Mix plannen…"), (0.3, "Tempo aanpassen (3/14)"), (0.7, "Mixen"), (0.95, "Mastering")]
        for k, (f, t) in enumerate(steps):
            self._later(lambda f=f, t=t: None if self._mix_cancelled else self.mixProgress.emit(f, t),
                        k * self.delay_ms)

        def done():
            if self._mix_cancelled:
                return
            p = plan or demo_plan(lst, target_minutes)[0]
            outs = [str(Path(settings.out_path).with_suffix("." + f)) for f in settings.formats]
            self.mixDone.emit(p, outs)
        self._later(done, len(steps) * self.delay_ms)

    def cancel_mix(self):
        self._mix_cancelled = True
        self._later(self.mixCancelled.emit, 0)

    def pair_score(self, a, b):
        return demo_pair_score(a, b)
