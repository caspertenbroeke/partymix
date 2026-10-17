"""Output layout of a mix: bar offsets, tempo curve and output time of a bar.

Pure functions, same math as ``reference/engine.py``:

* track ``i`` starts at output bar ``offsets[i]``; the next track starts ``overlap`` bars
  before track ``i`` ends;
* tempo anchors: ``(0, start_tempo)``, for every transition ``(O[i+1], tempo)`` and
  ``(O[i+1] + overlap, tempo)`` (tempo held during the overlap; for an overlap-0 cut with
  ``options["tempo_in"]`` the second anchor uses ``tempo_in``, i.e. the curve jumps), and
  ``(total_bars + 8, end_tempo)``; per-beat tempo is linearly interpolated at beat centres;
* ``T(x)`` = output seconds of fractional output bar ``x`` (cumulative beat durations).

Used by the renderer and by the planner/GUI for duration estimates
(:func:`estimate_duration`).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from ..model import MixPlan, TrackAnalysis

__all__ = ["Layout", "compute_layout", "estimate_duration", "END_EXTRA_BARS"]

END_EXTRA_BARS = 8   # the tempo curve is defined 8 bars past the end (echo/fade tails)


@dataclass
class Layout:
    offsets: List[int]          # output bar where each track starts
    lengths: List[int]          # bars played per track (end_bar - start_bar)
    overlaps: List[int]         # per transition
    tempos: List[float]         # per transition
    total_bars: int
    beat_dur: np.ndarray        # seconds per output beat, (total_bars + 8) * 4 beats
    beat_t: np.ndarray          # cumulative start time of each beat, len = len(beat_dur) + 1
    total_sec: float            # T(total_bars)
    start_tempo: float
    end_tempo: float

    @property
    def n(self) -> int:
        return len(self.offsets)

    def T(self, xbar):
        """Output seconds of (fractional, array-like) output bar index."""
        xb = np.asarray(xbar, dtype=float) * 4
        j = np.clip(np.floor(xb).astype(int), 0, len(self.beat_dur) - 1)
        return self.beat_t[j] + (xb - j) * self.beat_dur[j]

    def t(self, xbar: float) -> float:
        """Scalar version of :meth:`T`."""
        return float(self.T(xbar))

    def track_start(self, i: int) -> float:
        """Output seconds where track ``i`` starts (start of its incoming overlap)."""
        return self.t(self.offsets[i])

    def track_end(self, i: int) -> float:
        return self.t(self.offsets[i] + self.lengths[i])

    def tempo_at(self, xbar) -> np.ndarray:
        xb = np.asarray(xbar, dtype=float) * 4
        j = np.clip(np.floor(xb).astype(int), 0, len(self.beat_dur) - 1)
        return 60.0 / self.beat_dur[j]

    def echo_tempo(self, i: int) -> float:
        """Tempo used for the echo delay of track ``i`` (its outgoing transition / end tempo)."""
        return self.tempos[i] if i < len(self.tempos) else self.end_tempo


def compute_layout(plan: MixPlan, analyses: Optional[Sequence[TrackAnalysis]] = None) -> Layout:
    """Compute the output layout of ``plan``. ``analyses`` is accepted for API symmetry (unused)."""
    tr, trs = plan.tracks, plan.transitions
    n = len(tr)
    if n == 0:
        raise ValueError("Plan bevat geen nummers")
    if len(trs) != n - 1:
        raise ValueError(f"Plan heeft {len(trs)} overgangen voor {n} nummers (verwacht {n - 1})")
    lengths = [int(t.end_bar) - int(t.start_bar) for t in tr]
    for i, L in enumerate(lengths):
        if L <= 0:
            raise ValueError(f"Nummer {i + 1}: eindmaat ligt niet na beginmaat")
    overlaps = [int(t.overlap_bars) for t in trs]
    for i, ov in enumerate(overlaps):
        if ov < 0 or ov > lengths[i] or ov > lengths[i + 1]:
            raise ValueError(f"Overgang {i + 1}: overlap {ov} maten past niet in de nummers")
    O = [0]
    for i in range(n - 1):
        O.append(O[i] + lengths[i] - overlaps[i])
    total_bars = O[-1] + lengths[-1]
    anch = [(0.0, float(plan.start_tempo))]
    for i in range(n - 1):
        t_out = float(trs[i].tempo)
        t_in = t_out
        if overlaps[i] == 0 and (trs[i].options or {}).get("tempo_in") is not None:
            t_in = float(trs[i].options["tempo_in"])   # tempo jumps at the cut point
        anch += [(O[i + 1], t_out), (O[i + 1] + overlaps[i], t_in)]
    anch.append((total_bars + END_EXTRA_BARS, float(plan.end_tempo)))
    ax = np.array([a for a, _ in anch], float)
    ay = np.array([t for _, t in anch], float)
    nbeats = (total_bars + END_EXTRA_BARS) * 4
    beat_tempo = np.interp((np.arange(nbeats) + 0.5) / 4, ax, ay)
    beat_dur = 60.0 / beat_tempo
    beat_t = np.concatenate([[0.0], np.cumsum(beat_dur)])
    lay = Layout(offsets=O, lengths=lengths, overlaps=overlaps, tempos=[float(t.tempo) for t in trs],
                 total_bars=total_bars, beat_dur=beat_dur, beat_t=beat_t, total_sec=0.0,
                 start_tempo=float(plan.start_tempo), end_tempo=float(plan.end_tempo))
    lay.total_sec = lay.t(total_bars)
    return lay


def estimate_duration(plan: MixPlan, analyses: Optional[Sequence[TrackAnalysis]] = None) -> float:
    """Duration of the rendered mix in seconds (exact up to the final 3 s tail the renderer adds)."""
    return compute_layout(plan, analyses).total_sec
