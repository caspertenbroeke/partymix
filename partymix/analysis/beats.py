"""Beat / downbeat tracking (madmom) and constant-tempo grid fitting.

Pipeline (identical to the hand-made ``reference/grid2.py`` / ``reference/bars.py``):

1. madmom ``RNNDownBeatProcessor`` + ``DBNDownBeatTrackingProcessor(beats_per_bar=[4], fps=100,
   min_bpm=100, max_bpm=160)``.  The RNN runs on ≤ 4-min chunks with 10 s overlap (activations
   cross-faded).  The DBN's Viterbi back-pointer table grows with ``frames × states`` (a 30-min
   file was OOM-killed at 7 GB), so files longer than 7 min are decoded in 5-min windows with
   60 s overlap, stitched in the middle of the overlap with bar labels re-aligned.  Shorter files
   are decoded in one go: DBN windows change the beat phase on ambiguous tracks (Stereo Love
   came out half a beat off when decoded in 4-min pieces).
2. :func:`fit_grid`: local beat index ``cumsum(round(diff / median_period))``, iterative robust
   linear fit (5 rounds, residual < 30 ms), downbeat phase by majority vote of madmom's
   downbeats.  With ≥ 85 % inliers the tempo is constant and the fitted grid is returned
   (extended back to ~0 s and to the end of the file); otherwise the lightly smoothed madmom
   beats are kept (the "La Mamá" case).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np

FPS = 100
MIN_BPM = 100.0
MAX_BPM = 160.0
CHUNK_S = 240.0
OVERLAP_S = 10.0
# The DBN Viterbi table is uint32[frames, ~11.5k states] (≈ 46 MB per 10 s of audio).
DBN_MAX_S = 420.0           # decode in one go up to 7 min (≈ 1.9 GB peak)
DBN_WIN_S = 300.0
DBN_OVERLAP_S = 60.0
INLIER_TOL = 0.030          # s, residual threshold of the robust fit
MIN_INLIERS = 0.85          # fraction of madmom beats on the grid for "constant tempo"

ProgressCb = Optional[Callable[[float], None]]

__all__ = ["BeatResult", "track_beats", "fit_grid", "madmom_beats", "madmom_activations"]


@dataclass
class BeatResult:
    bpm: float
    constant_tempo: bool
    beats: List[float]                  # beat times in seconds
    first_downbeat_index: int           # index into ``beats`` of bar 0
    inlier_ratio: float = 1.0
    residual_ms: float = 0.0            # std of inlier residuals
    raw_beats: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)), repr=False)

    @property
    def first_downbeat(self) -> float:
        return self.beats[self.first_downbeat_index]


_PROCS: dict = {}


def _processors():
    """madmom processors are expensive to build (model loading) – create once per process."""
    if not _PROCS:
        from madmom.features.downbeats import RNNDownBeatProcessor, DBNDownBeatTrackingProcessor
        _PROCS["rnn"] = RNNDownBeatProcessor()
        _PROCS["dbn"] = DBNDownBeatTrackingProcessor(beats_per_bar=[4], fps=FPS,
                                                     min_bpm=MIN_BPM, max_bpm=MAX_BPM)
    return _PROCS["rnn"], _PROCS["dbn"]


def _to_mono(audio: np.ndarray) -> np.ndarray:
    a = np.asarray(audio, dtype=np.float32)
    return a.mean(axis=1, dtype=np.float32) if a.ndim == 2 else a


def _activations(mono: np.ndarray, sr: int) -> np.ndarray:
    from madmom.audio.signal import Signal
    rnn, _ = _processors()
    return np.asarray(rnn(Signal(mono, sample_rate=sr, num_channels=1)), dtype=np.float32)


def _dbn(act: np.ndarray) -> np.ndarray:
    _, dbn = _processors()
    return np.asarray(dbn(act), dtype=np.float64).reshape(-1, 2)


def _stitch(a: np.ndarray, b: np.ndarray, b_start: float, overlap: float) -> np.ndarray:
    """Join two ``[beat_time, position]`` arrays whose time ranges overlap after ``b_start``."""
    if len(a) == 0:
        return b
    if len(b) == 0:
        return a
    # re-label b's bar positions to agree with a (vote over beats matched in the overlap)
    ov_a = a[a[:, 0] >= b_start]
    shifts = []
    for t, pa in ov_a:
        j = int(np.argmin(np.abs(b[:, 0] - t)))
        if abs(b[j, 0] - t) < 0.07:
            shifts.append(int(pa - b[j, 1]) % 4)
    b = b.copy()
    if shifts:
        sh = int(np.argmax(np.bincount(shifts, minlength=4)))
        b[:, 1] = (b[:, 1] - 1 + sh) % 4 + 1
    cut = b_start + overlap / 2
    a_keep, b_keep = a[a[:, 0] < cut], b[b[:, 0] >= cut]
    if len(a_keep) >= 2 and len(b_keep):
        per = float(np.median(np.diff(a_keep[-16:, 0])))
        if b_keep[0, 0] - a_keep[-1, 0] < 0.5 * per:
            b_keep = b_keep[1:]
    return np.vstack([a_keep, b_keep])


def madmom_activations(audio: np.ndarray, sr: int, chunk_s: float = CHUNK_S,
                       overlap_s: float = OVERLAP_S, progress: ProgressCb = None,
                       cancel: Optional[Callable[[], bool]] = None) -> np.ndarray:
    """RNN beat/downbeat activations ``[frames, 2]`` at 100 fps, computed in overlapping chunks
    (linear cross-fade in the overlap) so memory stays bounded for long files."""
    mono = _to_mono(audio)
    n = len(mono)
    hop = sr // FPS
    clen = int(chunk_s * FPS) * hop
    step = int((chunk_s - overlap_s) * FPS) * hop
    ov = clen - step
    starts = [0] if n <= clen else list(range(0, n - ov, step))
    n_frames = int(np.ceil(n / hop))
    act = np.zeros((n_frames, 2), np.float32)
    for k, s in enumerate(starts):
        if cancel is not None and cancel():
            raise InterruptedError("geannuleerd")
        a = _activations(mono[s:s + clen], sr)
        f0 = s // hop
        a = a[: n_frames - f0]
        if k == 0:
            act[f0:f0 + len(a)] = a
        else:
            nov = min(ov // hop, len(a))
            w = np.linspace(0.0, 1.0, nov, dtype=np.float32)[:, None]
            act[f0:f0 + nov] = act[f0:f0 + nov] * (1 - w) + a[:nov] * w
            act[f0 + nov:f0 + len(a)] = a[nov:]
        if progress:
            progress((k + 1) / len(starts))
    return act


def madmom_beats(audio: np.ndarray, sr: int, chunk_s: float = CHUNK_S, overlap_s: float = OVERLAP_S,
                 progress: ProgressCb = None, cancel: Optional[Callable[[], bool]] = None) -> np.ndarray:
    """Run madmom beat+downbeat tracking.

    The RNN runs in ``chunk_s`` chunks; the DBN decodes the whole activation sequence at once
    when it is at most ``DBN_MAX_S`` long (identical to a single madmom call), otherwise in
    ``DBN_WIN_S`` windows with ``DBN_OVERLAP_S`` overlap that are stitched together.

    Returns an array ``[n, 2]`` of (beat time s, bar position 1..4).
    """
    act = madmom_activations(audio, sr, chunk_s, overlap_s,
                             progress=(lambda f: progress(0.8 * f)) if progress else None, cancel=cancel)
    n = len(act)
    if n <= DBN_MAX_S * FPS:
        out = _dbn(act)
    else:
        win, step = int(DBN_WIN_S * FPS), int((DBN_WIN_S - DBN_OVERLAP_S) * FPS)
        out = np.zeros((0, 2))
        for k, f0 in enumerate(range(0, n - (win - step), step)):
            if cancel is not None and cancel():
                raise InterruptedError("geannuleerd")
            r = _dbn(act[f0:f0 + win])
            r[:, 0] += f0 / FPS
            out = _stitch(out, r, f0 / FPS, DBN_OVERLAP_S) if k else r
    if progress:
        progress(1.0)
    return out


def _smooth_beats(b: np.ndarray) -> np.ndarray:
    """Light smoothing of a madmom beat list (reference/bars.py, La Mamá)."""
    if len(b) < 6:
        return b.copy()
    b2 = 0.5 * b + 0.5 * np.convolve(b, np.ones(5) / 5, "same")
    b2[:2] = b[:2]
    b2[-2:] = b[-2:]
    return b2


def fit_grid(raw: np.ndarray, duration: float) -> BeatResult:
    """Fit a constant tempo grid to madmom output ``raw`` (``[n, 2]`` beat time, position)."""
    raw = np.asarray(raw, dtype=np.float64).reshape(-1, 2)
    if len(raw) < 8:
        raise ValueError("te weinig beats gevonden")
    b, pos = raw[:, 0], raw[:, 1].astype(int)
    per0 = float(np.median(np.diff(b)))
    idx = np.concatenate([[0.0], np.cumsum(np.round(np.diff(b) / per0))])
    m = np.ones(len(b), bool)
    A = np.array([per0, b[0]])
    for _ in range(5):
        if m.sum() < 2:
            break
        A = np.polyfit(idx[m], b[m], 1)
        res = b - np.polyval(A, idx)
        m = np.abs(res) < INLIER_TOL
    per, b0 = float(A[0]), float(A[1])
    res = b - np.polyval(A, idx)
    inl = float(m.mean())
    res_ms = float(res[m].std() * 1000) if m.any() else float("nan")
    bpm = 60.0 / per

    if inl >= MIN_INLIERS:
        di = idx[pos == 1].astype(int) % 4
        ph = int(np.argmax(np.bincount(di, minlength=4))) if len(di) else 0
        first_bar = b0 + ph * per
        while first_bar - 4 * per > -0.05:
            first_bar -= 4 * per
        while first_bar <= -0.05:
            first_bar += 4 * per
        start = first_bar
        n_pre = 0
        while start - per > -0.05:
            start -= per
            n_pre += 1
        beats = np.arange(start, duration - 0.01, per)
        return BeatResult(bpm, True, beats.tolist(), n_pre, inl, res_ms, raw)

    # tempo not constant: keep the madmom beats (lightly smoothed), extended to both ends
    b2 = _smooth_beats(b)
    first = int(np.flatnonzero(pos == 1)[0]) if (pos == 1).any() else 0
    p_head = float(np.median(np.diff(b2[:9])))
    p_tail = float(np.median(np.diff(b2[-9:])))
    head = []
    t = b2[0] - p_head
    while t > -0.05:
        head.append(t)
        t -= p_head
    tail = []
    t = b2[-1] + p_tail
    while t < duration - 0.01:
        tail.append(t)
        t += p_tail
    beats = np.concatenate([head[::-1], b2, tail])
    first += len(head)
    # bar 0 = earliest downbeat that still lies inside the file
    while first - 4 >= 0:
        first -= 4
    return BeatResult(bpm, False, beats.tolist(), first, inl, res_ms, raw)


def track_beats(audio: np.ndarray, sr: int, progress: ProgressCb = None,
                cancel: Optional[Callable[[], bool]] = None) -> BeatResult:
    """Full beat analysis of a decoded track (``[n, ch]`` float32)."""
    raw = madmom_beats(audio, sr, progress=progress, cancel=cancel)
    return fit_grid(raw, len(audio) / sr)
