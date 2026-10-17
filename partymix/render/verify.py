"""Optional self-check of a rendered mix (port of ``reference/verify.py``).

madmom downbeat tracking over the output in <= 4-minute chunks, then per track:
percentage of detected beats on the intended output grid (< 35 ms), beats half a beat off,
and detected downbeats on bar 1 (< 50 ms).  madmom is imported lazily.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import soundfile as sf

from .common import check_cancel
from .layout import Layout

__all__ = ["detect_beats", "grid_stats", "verify_mix"]

CHUNK_SEC = 240.0
OVERLAP_SEC = 5.0


def detect_beats(path: str, progress: Optional[Callable[[float, str], None]] = None,
                 cancel_event=None) -> np.ndarray:
    """``[n, 2]`` array (time s, beat position 1..4) over the whole file."""
    from madmom.audio.signal import Signal
    from madmom.features.downbeats import RNNDownBeatProcessor, DBNDownBeatTrackingProcessor
    rnn = RNNDownBeatProcessor()
    dbn = DBNDownBeatTrackingProcessor(beats_per_bar=[4], fps=100, min_bpm=100, max_bpm=160)
    info = sf.info(path)
    sr, total = info.samplerate, info.frames
    step = int((CHUNK_SEC - OVERLAP_SEC) * sr)
    ln = int(CHUNK_SEC * sr)
    out: List[np.ndarray] = []
    starts = list(range(0, max(total - int(OVERLAP_SEC * sr), 1), step))
    with sf.SoundFile(path) as f:
        for ci, s in enumerate(starts):
            check_cancel(cancel_event)
            f.seek(s)
            x = f.read(min(ln, total - s), dtype="float32", always_2d=True).mean(axis=1)
            if len(x) < sr * 2:
                break
            r = dbn(rnn(Signal(x, sample_rate=sr, num_channels=1)))
            if len(r):
                r = np.asarray(r, float).copy()
                r[:, 0] += s / sr
                lo = s / sr + (OVERLAP_SEC if ci else 0.0)
                hi = s / sr + CHUNK_SEC
                out.append(r[(r[:, 0] >= lo) & (r[:, 0] < hi)])
            if progress:
                progress((ci + 1) / len(starts), "Controle: beats detecteren")
    return np.concatenate(out) if out else np.zeros((0, 2))


def _near(a, b):
    if len(b) < 2 or len(a) == 0:
        return np.full(len(a), np.inf)
    i = np.searchsorted(b, a).clip(1, len(b) - 1)
    return np.minimum(np.abs(a - b[i]), np.abs(a - b[i - 1]))


def grid_stats(det: np.ndarray, layout: Layout, titles: Sequence[str], offset: float = 0.0) -> Dict:
    """Compare detected beats (``detect_beats``) with the intended grid of ``layout``.
    ``offset``: output seconds of output bar 0 (0 for a rendered mix)."""
    bt = det[:, 0] if len(det) else np.zeros(0)
    db = bt[det[:, 1] == 1] if len(det) else np.zeros(0)
    exp_beats = layout.T(np.arange(0, layout.total_bars * 4) / 4) + offset
    exp_db = layout.T(np.arange(layout.total_bars)) + offset
    if len(bt) == 0:
        return {"lag_ms": None, "tracks": [], "beats_on_grid_pct": 0.0}
    idx = np.searchsorted(exp_beats, bt[::7]).clip(1, len(exp_beats) - 1)
    near_e = np.where(np.abs(bt[::7] - exp_beats[idx]) < np.abs(bt[::7] - exp_beats[idx - 1]),
                      exp_beats[idx], exp_beats[idx - 1])
    lag = float(np.median(bt[::7] - near_e))
    bt = bt - lag
    db = db - lag
    cues = [layout.track_start(i) + offset for i in range(layout.n)] + [layout.total_sec + offset]
    tracks = []
    on_all = n_all = 0
    for i in range(layout.n):
        t0, t1 = cues[i], cues[i + 1]
        mb = bt[(bt >= t0) & (bt < t1)]
        per = float(np.median(np.diff(mb))) if len(mb) > 2 else 60 / 128
        d = _near(mb, exp_beats)
        mdb = db[(db >= t0) & (db < t1)]
        dd = _near(mdb, exp_db)
        on = float(100 * (d < 0.035).mean()) if len(d) else 0.0
        tracks.append({
            "index": i, "title": titles[i] if i < len(titles) else str(i), "start": t0,
            "beats": int(len(mb)),
            "beats_on_grid_pct": round(on, 1),
            "half_beat_off_pct": round(float(100 * (d > per * 0.3).mean()) if len(d) else 0.0, 1),
            "downbeats_on_bar_pct": round(float(100 * (dd < 0.05).mean()) if len(dd) else 0.0, 1),
        })
        on_all += (d < 0.035).sum()
        n_all += len(d)
    return {"lag_ms": round(lag * 1000, 1), "beats_on_grid_pct": round(100 * on_all / max(n_all, 1), 1),
            "tracks": tracks}


def verify_mix(path: str, layout: Layout, titles: Sequence[str],
               progress: Optional[Callable[[float, str], None]] = None, cancel_event=None) -> Dict:
    det = detect_beats(path, progress, cancel_event)
    res = grid_stats(det, layout, titles)
    res["file"] = path
    worst = min((t["beats_on_grid_pct"] for t in res["tracks"]), default=0.0)
    res["ok"] = bool(worst >= 85.0)
    return res
