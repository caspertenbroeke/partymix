"""Shared planner helpers: per-bar level maps, Camelot keys, tempo rules and the output layout.

The layout / tempo-curve maths is a 1:1 port of ``reference/engine.py`` so that the planner's
duration estimate equals what the renderer will produce.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..model import MixPlan, TrackAnalysis

MAX_STRETCH = 0.03          # never stretch a track more than 3 % (else echo_cut)
TEMPO_BIAS = 0.45           # transition tempo = a + 0.45 (b - a)  (a bit closer to the outgoing track)


# --------------------------------------------------------------------------------------------
# per-bar levels (0..9 like reference/maps_v2.txt)
# --------------------------------------------------------------------------------------------
def _pad(x: Sequence[float], n: int, fill: float) -> np.ndarray:
    a = np.asarray(list(x)[:n], float)
    if len(a) < n:
        a = np.concatenate([a, np.full(n - len(a), fill)])
    return np.nan_to_num(a, nan=fill, posinf=fill, neginf=fill)


def n_bars_of(t: TrackAnalysis) -> int:
    n = len(t.bar_bass) if t.bar_bass else 0
    return max(n, 0) if n else max(t.n_bars, 0)


def bass_levels(t: TrackAnalysis, n: Optional[int] = None) -> np.ndarray:
    """0..9 per bar, 9 = full bass, one step = 2 dB below the 90th percentile (as maps_v2)."""
    n = n_bars_of(t) if n is None else n
    if not t.bar_bass:
        return np.full(n, 8.0)            # unknown: assume bass everywhere (planner degrades gracefully)
    x = _pad(t.bar_bass, n, -100.0)
    top = np.percentile(x, 90)
    return np.clip((x - top + 18.0) / 2.0, 0.0, 9.0)


def vocal_levels(t: TrackAnalysis, n: Optional[int] = None) -> np.ndarray:
    """0..9 per bar (one step = 2.2 dB, as vox.json). Instrumental tracks stay low because the
    reference level is at least (mix p90 − 10 dB)."""
    n = n_bars_of(t) if n is None else n
    if not t.bar_vocal:
        return np.zeros(n)
    v = _pad(t.bar_vocal, n, -100.0)
    ref = np.percentile(v, 90)
    if t.bar_rms:
        r = _pad(t.bar_rms, n, -100.0)
        ref = max(ref, np.percentile(r, 90) - 10.0)
    return np.clip((v - ref + 20.0) / 2.2, 0.0, 9.0)


def rms_levels(t: TrackAnalysis, n: Optional[int] = None) -> np.ndarray:
    """0..9 per bar, one step = 2 dB below p90 of bar RMS (falls back to bass/mid/high)."""
    n = n_bars_of(t) if n is None else n
    src = t.bar_rms
    if not src and t.bar_mid:
        src = [10 * np.log10(sum(10 ** (v / 10) for v in vals) + 1e-12)
               for vals in zip(t.bar_bass, t.bar_mid, t.bar_high or t.bar_mid)]
    if not src:
        return np.full(n, 8.0)
    x = _pad(src, n, -100.0)
    top = np.percentile(x, 90)
    return np.clip((x - top + 18.0) / 2.0, 0.0, 9.0)


def mean(a: np.ndarray, lo: int, hi: int) -> float:
    lo, hi = max(lo, 0), min(hi, len(a))
    return float(a[lo:hi].mean()) if hi > lo else 0.0


# --------------------------------------------------------------------------------------------
# Camelot
# --------------------------------------------------------------------------------------------
def parse_camelot(code: str) -> Optional[Tuple[int, str]]:
    code = (code or "").strip().upper()
    if len(code) < 2 or code[-1] not in "AB":
        return None
    try:
        num = int(code[:-1])
    except ValueError:
        return None
    if not 1 <= num <= 12:
        return None
    return num, code[-1]


def camelot_relation(a: str, b: str) -> Tuple[str, float]:
    """(relation, cost). relation ∈ same, adjacent, relative, boost, diagonal, clash, unknown."""
    pa, pb = parse_camelot(a), parse_camelot(b)
    if pa is None or pb is None:
        return "unknown", 2.0
    d = (pb[0] - pa[0]) % 12
    dist = min(d, 12 - d)
    if pa[1] == pb[1]:
        if dist == 0:
            return "same", 0.0
        if dist == 1:
            return "adjacent", 1.0
        if d == 2:
            return "boost", 2.5          # +2 "energy boost"
        return "clash", 3.5 if dist <= 4 else 4.5
    if dist == 0:
        return "relative", 1.0
    if dist == 1:
        return "diagonal", 2.5
    return "clash", 3.5 if dist <= 3 else 4.5


def camelot_distance(a: str, b: str) -> int:
    """Steps on the wheel (+1 for a mode change); 99 if unknown."""
    pa, pb = parse_camelot(a), parse_camelot(b)
    if pa is None or pb is None:
        return 99
    d = (pb[0] - pa[0]) % 12
    return min(d, 12 - d) + (1 if pa[1] != pb[1] else 0)


# --------------------------------------------------------------------------------------------
# tempo rules
# --------------------------------------------------------------------------------------------
def round_half(x: float) -> float:
    return round(x * 2.0) / 2.0


def tempo_gap(a: float, b: float) -> float:
    """relative tempo difference |ln(b/a)| (≈ fraction)."""
    return abs(float(np.log(b / a)))


def stretch_feasible(a: float, b: float, limit: float = MAX_STRETCH) -> bool:
    return b * (1 - limit) <= a * (1 + limit) and a * (1 - limit) <= b * (1 + limit)


def transition_tempo(a: float, b: float, limit: float = MAX_STRETCH) -> float:
    """Tempo held during the overlap: between both native tempos, a bit closer to the outgoing (a),
    rounded to 0.5 BPM, and never more than `limit` away from either native tempo when possible."""
    t = round_half(a + TEMPO_BIAS * (b - a))
    lo = max(a, b) * (1 - limit)
    hi = min(a, b) * (1 + limit)
    if lo <= hi:
        if t < lo or t > hi:
            t = min(max(t, lo), hi)
            r = round_half(t)
            t = r if lo <= r <= hi else round(t, 2)
    else:                                   # infeasible (echo_cut): minimise the worst stretch
        t = round(float(np.sqrt(a * b)), 2)
    return float(t)


def stretch_of(tempo: float, native: float) -> float:
    return abs(tempo / native - 1.0)


# --------------------------------------------------------------------------------------------
# output layout (port of reference/engine.py)
# --------------------------------------------------------------------------------------------
@dataclass
class Layout:
    offsets: List[int]            # output bar where each track starts
    total_bars: int
    beat_t: np.ndarray            # output time (s) of each beat boundary
    beat_dur: np.ndarray

    def time_of(self, xbar: float) -> float:
        xb = float(xbar) * 4
        j = int(np.clip(np.floor(xb), 0, len(self.beat_dur) - 1))
        return float(self.beat_t[j] + (xb - j) * self.beat_dur[j])

    @property
    def total_sec(self) -> float:
        return self.time_of(self.total_bars)


def anchors(segs: Sequence[Tuple[int, int]], ovs: Sequence[int], tempos: Sequence[float],
            start_tempo: float, end_tempo: float, tempos_in: Optional[Sequence[Optional[float]]] = None) -> Tuple[List[int], int, np.ndarray, np.ndarray]:
    n = len(segs)
    O = [0]
    for i in range(n - 1):
        O.append(O[i] + (segs[i][1] - segs[i][0]) - ovs[i])
    total = O[-1] + segs[-1][1] - segs[-1][0]
    an = [(0, start_tempo)]
    for i in range(n - 1):
        tin = tempos_in[i] if tempos_in is not None else None
        if ovs[i] == 0 and tin is not None and abs(tin - tempos[i]) > 1e-9:
            # tempo jump exactly at the cut bar: A arrives at `tempo`, B continues from `tempo_in`
            an += [(O[i + 1] - 1e-6, tempos[i]), (O[i + 1], tin)]
        else:
            an += [(O[i + 1], tempos[i]), (O[i + 1] + ovs[i], tempos[i])]
    an.append((total + 8, end_tempo))
    ax = np.array([a for a, _ in an], float)
    ay = np.array([t for _, t in an], float)
    return O, total, ax, ay


def layout_from(segs, ovs, tempos, start_tempo, end_tempo, tempos_in=None) -> Layout:
    O, total, ax, ay = anchors(segs, ovs, tempos, start_tempo, end_tempo, tempos_in)
    nb = max((total + 8) * 4, 1)
    beat_tempo = np.interp((np.arange(nb) + 0.5) / 4, ax, ay)
    beat_dur = 60.0 / beat_tempo
    beat_t = np.concatenate([[0.0], np.cumsum(beat_dur)])
    return Layout(O, total, beat_t, beat_dur)


def layout(plan: MixPlan) -> Layout:
    """Output layout of a MixPlan exactly as reference/engine.py computes it."""
    segs = [(t.start_bar, t.end_bar) for t in plan.tracks]
    ovs = [tr.overlap_bars for tr in plan.transitions]
    tempos = [tr.tempo for tr in plan.transitions]
    return layout_from(segs, ovs, tempos, plan.start_tempo, plan.end_tempo,
                       [tempo_in_of(tr) for tr in plan.transitions])


def tempo_in_of(tr) -> Optional[float]:
    """Tempo of the incoming track right after the transition (options["tempo_in"] for cuts)."""
    v = tr.options.get("tempo_in") if tr.overlap_bars == 0 else None
    return float(v) if v is not None else None


def cut_tempos(a: float, b: float, limit: float = MAX_STRETCH) -> Tuple[float, Optional[float]]:
    """(tempo, tempo_in) for an echo cut: one held tempo when both sides stay within `limit`,
    else A ends at the tempo nearest to B within its limit and B starts likewise (a jump at the cut)."""
    if stretch_feasible(a, b, limit):
        return transition_tempo(a, b, limit), None
    def toward(x, y):
        t = min(max(round_half(y), x * (1 - limit)), x * (1 + limit))
        r = round_half(t)
        return r if abs(r / x - 1) <= limit else round(t, 2)
    return toward(a, b), toward(b, a)


def mix_duration(plan: MixPlan) -> float:
    """Total mix length in seconds (engine.py: T(TOTAL_BARS))."""
    return layout(plan).total_sec


def track_tempo_range(plan: MixPlan, i: int) -> Tuple[float, float]:
    """(min, max) output tempo while track i plays (tempo curve is piecewise linear, so the
    extremes are at the anchors bounding the track)."""
    if i == 0:
        tin = plan.start_tempo
    else:
        prev = plan.transitions[i - 1]
        tin = tempo_in_of(prev) or prev.tempo
    tout = plan.end_tempo if i == len(plan.tracks) - 1 else plan.transitions[i].tempo
    return min(tin, tout), max(tin, tout)


def max_stretch(plan: MixPlan, analyses: Sequence[TrackAnalysis]) -> List[float]:
    out = []
    for i, a in enumerate(analyses):
        lo, hi = track_tempo_range(plan, i)
        out.append(max(stretch_of(lo, a.bpm), stretch_of(hi, a.bpm)))
    return out
