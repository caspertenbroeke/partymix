"""Beat-accurate time stretching of one planned track with the rubberband CLI.

For track ``i`` of the plan the source segment ``[bar_time(start_bar) - margin,
bar_time(end_bar + tail))`` of the instrumental and vocal stems is cut out, and a per-beat
timemap (rubberband ``-M``) maps every source beat (``TrackAnalysis.bar_time``) to the output
beat time of the mix tempo curve (:mod:`.layout`).  The ``margin`` seconds before the first
bar are mapped 1:1 so the stretched file starts ``pre`` seconds before output bar
``offsets[i]``.

Fix vs. ``reference/engine.py``: when the first downbeat is at (or slightly before) 0 s
the reference clipped the slice start at 0 and wrote a negative timemap line.  Here the
slice is padded with silence instead, so ``pre`` is always ``margin`` and all map lines are
positive and strictly increasing.
"""
from __future__ import annotations

import os
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import soundfile as sf

from .. import paths
from ..model import MixPlan, PlannedTrack, TrackAnalysis
from .common import RenderCancelled, RenderError, check_cancel, run_tool
from .layout import Layout

__all__ = ["StretchResult", "stem_path", "build_timemap", "stretch_track", "iter_stretched",
           "MARGIN_SEC", "TAIL_BARS", "PARTS"]

MARGIN_SEC = 0.05     # source seconds kept before the first bar (mapped 1:1)
TAIL_BARS = 4         # extra bars stretched after end_bar (echo / filter tails)
PARTS = ("inst", "voc")


@dataclass
class StretchResult:
    index: int
    inst_path: str
    voc_path: str
    pre: float          # seconds in the stretched files before output bar offsets[i]
    frames: int

    def paths(self) -> Tuple[str, str]:
        return self.inst_path, self.voc_path

    def cleanup(self) -> None:
        for p in (self.inst_path, self.voc_path):
            try:
                os.remove(p)
            except OSError:
                pass


def stem_path(analysis: TrackAnalysis, part: str) -> str:
    """Path of the ``inst`` / ``voc`` stem in the analysis cache dir (``.flac``, else ``.wav``)."""
    d = Path(analysis.cache_dir)
    for ext in (".flac", ".wav"):
        p = d / f"{part}{ext}"
        if p.exists():
            return str(p)
    raise RenderError(f"Stem '{part}' ontbreekt in {d} ({analysis.title or analysis.path})")


def _q_samples(t: float, sr: int) -> int:
    """Sample index of time ``t`` quantised like the reference (``ffmpeg -ss/-to "%.5f"``:
    10 us steps, then microseconds -> samples rounded half up).  Keeps renders bit-compatible
    with ``reference/engine.py``; rubberband output is sensitive to 1-sample input shifts."""
    us = int(round(float(f"{t:.5f}") * 1e6))
    num = us * sr
    return (num + 500000) // 1000000 if num >= 0 else -((-num + 500000) // 1000000)


def build_timemap(analysis: TrackAnalysis, start_bar: int, end_bar: int, layout: Layout, index: int,
                  sr: int, src_frames: int, margin: float = MARGIN_SEC, tail_bars: int = TAIL_BARS):
    """Compute the slice and the rubberband timemap for one track.

    Returns ``(a, z, lines, pre)``: slice ``[a, z)`` in source samples (``a`` may be negative
    = leading silence), ``lines`` = list of ``(src_sample, out_sample)`` relative to ``a``
    (strictly increasing in both), ``pre`` = seconds before the first bar.
    """
    s, e = start_bar, end_bar
    t_s = analysis.bar_time(s)
    a_sec = t_s - margin               # may be < 0: the slice is then padded with silence
    a = _q_samples(a_sec, sr)
    z_t = min(analysis.bar_time(e + tail_bars), analysis.beats[-1] + 0.4, src_frames / sr)
    z = min(a + int(round((float(f"{z_t:.5f}") - float(f"{a_sec:.5f}")) * sr)), src_frames)
    if z <= a + sr // 10:
        raise RenderError(f"Segment van {analysis.title or analysis.path} is leeg (maat {s}-{e})")
    pre = t_s - a_sec
    O = layout.offsets[index]
    t0 = layout.t(O)
    lines: List[Tuple[int, int]] = [(0, 0)]
    for q in range(int((e + tail_bars - s) * 4) + 1):
        st = analysis.bar_time(s + q / 4) - a_sec
        if st > z_t - a_sec - 0.01:
            break
        ot = layout.t(O + q / 4) - t0
        li, lo = int(round(st * sr)), int(round((ot + pre) * sr))
        if li > lines[-1][0] and lo > lines[-1][1]:
            lines.append((li, lo))
    return a, z, lines, pre


def _read_slice(path: str, a: int, z: int) -> np.ndarray:
    """Read source samples ``[a, z)`` as float32 stereo; negative ``a`` / past-end -> silence."""
    import re
    with sf.SoundFile(path) as f:
        n = f.frames
        lo, hi = max(a, 0), min(z, n)
        f.seek(lo)
        x = f.read(max(hi - lo, 0), dtype="float32", always_2d=True)
        # cache stems are stored attenuated by a power of two (see audio_io.write_flac); undo it
        m = re.search(r"partymix_gain=([0-9.eE+-]+)", f.comment or "")
    if m:
        x *= np.float32(float(m.group(1)))
    if x.shape[1] == 1:
        x = np.repeat(x, 2, axis=1)
    elif x.shape[1] > 2:
        x = x[:, :2]
    if a < 0:
        x = np.concatenate([np.zeros((-a, 2), np.float32), x])
    return x


def _src_frames(path: str) -> int:
    return sf.info(path).frames


def stretch_track(analysis: TrackAnalysis, planned: PlannedTrack, layout: Layout, index: int,
                  workdir: str, sr: int = 44100, cancel_event=None,
                  stems: Optional[Dict[str, str]] = None) -> StretchResult:
    """Stretch inst and voc stem of track ``index``; files are written into ``workdir``."""
    os.makedirs(workdir, exist_ok=True)
    stems = stems or {p: stem_path(analysis, p) for p in PARTS}
    frames = min(_src_frames(stems[p]) for p in PARTS)
    a, z, lines, pre = build_timemap(analysis, planned.start_bar, planned.end_bar, layout, index, sr, frames)
    outs = {}
    exe = paths.rubberband()
    for part in PARTS:
        check_cancel(cancel_event)
        # ASCII-only relative names + cwd=workdir: rubberband (libsndfile) is not unicode-safe on Windows
        raw, mp, out = f"t{index:03d}_{part}_in.wav", f"t{index:03d}_{part}.map", f"t{index:03d}_{part}.wav"
        x = _read_slice(stems[part], a, z)
        sf.write(os.path.join(workdir, raw), x, sr, subtype="FLOAT")
        inlen = len(x)
        del x
        with open(os.path.join(workdir, mp), "w", newline="\n") as fh:
            fh.write("".join(f"{li} {lo}\n" for li, lo in lines))
        li, lo = lines[-1]
        outlen = lo + int((inlen - li) * (lo / max(li, 1)))
        try:
            run_tool([exe, "-q", "-3", "-M", mp, "-D", f"{outlen / sr:.6f}", raw, out],
                     cwd=workdir, cancel_event=cancel_event, what="rubberband")
        finally:
            for f in (raw, mp):
                try:
                    os.remove(os.path.join(workdir, f))
                except OSError:
                    pass
        outs[part] = os.path.join(workdir, out)
    nfr = min(sf.info(outs[p]).frames for p in PARTS)
    return StretchResult(index=index, inst_path=outs["inst"], voc_path=outs["voc"], pre=pre, frames=nfr)


def iter_stretched(plan: MixPlan, analyses: Sequence[TrackAnalysis], layout: Layout, workdir: str,
                   sr: int = 44100, max_jobs: int = 2, lookahead: int = 1, cancel_event=None,
                   stems: Optional[Sequence[Dict[str, str]]] = None) -> Iterator[StretchResult]:
    """Stretch all tracks with up to ``max_jobs`` parallel rubberband jobs, yielding results in
    track order.  At most ``max_jobs + lookahead`` tracks are stretched ahead of the consumer,
    so temp disk usage stays bounded (consumer should ``cleanup()`` each result).

    ``analyses[i]`` belongs to ``plan.tracks[i]``.
    """
    n = len(plan.tracks)
    ahead = max(1, max_jobs) + max(0, lookahead)
    local_cancel = threading.Event()

    class _Either:
        def is_set(self):
            return local_cancel.is_set() or (cancel_event is not None and cancel_event.is_set())

    ce = _Either()
    ex = ThreadPoolExecutor(max_workers=max(1, max_jobs), thread_name_prefix="stretch")
    pending: deque = deque()
    nxt = 0
    try:
        for i in range(n):
            while nxt < n and nxt < i + ahead:
                st = stems[nxt] if stems is not None else None
                pending.append(ex.submit(stretch_track, analyses[nxt], plan.tracks[nxt], layout, nxt,
                                         workdir, sr, ce, st))
                nxt += 1
            fut: Future = pending.popleft()
            while True:
                try:
                    res = fut.result(timeout=0.25)
                    break
                except TimeoutError:
                    check_cancel(cancel_event)
                except Exception as e:  # concurrent.futures.TimeoutError is TimeoutError on 3.11+
                    if type(e).__name__ == "TimeoutError":
                        check_cancel(cancel_event)
                        continue
                    raise
            yield res
    finally:
        local_cancel.set()
        for f in pending:
            f.cancel()
        ex.shutdown(wait=True)
        for f in pending:
            if f.done() and not f.cancelled() and f.exception() is None:
                f.result().cleanup()
