"""Running-order optimiser.

Cost of an order = Σ pair costs (tempo distance, Camelot compatibility, tempo going down, same artist)
               + Σ position costs (energy vs. target curve, BPM rising over the set).
Solved with beam search followed by deterministic simulated annealing (swap / 2-opt / relocate).
30 tracks solve well under a second.
"""
from __future__ import annotations

import math
import random
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..model import TrackAnalysis
from .common import camelot_relation, stretch_feasible

W_ENERGY = 6.0
W_BPM_POS = 4.0
W_DOWN = 0.5            # per % tempo going down
P_SAME_ARTIST = 8.0
BEAM = 48

KEY_LABEL = {
    "same": "zelfde toonsoort",
    "adjacent": "buren op de Camelot-cirkel",
    "relative": "relatieve toonsoort (A/B)",
    "boost": "+2 op de cirkel (energie-boost)",
    "diagonal": "diagonaal op de cirkel (matig)",
    "clash": "toonsoorten botsen",
    "unknown": "toonsoort onbekend",
}


def _nl(x: float, d: int = 1) -> str:
    return f"{x:.{d}f}".replace(".", ",")


def energy_target(x: float, curve: str = "party") -> float:
    """Target energy 0..1 at relative position x (0 = first, 1 = last)."""
    x = min(max(x, 0.0), 1.0)
    if curve == "rising":
        return 0.15 + 0.85 * x
    if curve == "flat":
        return 0.6
    # party: warm-up → build → peak around 70–85 % → short finale
    if x <= 0.75:
        u = x / 0.75
        return 0.25 + 0.75 * u * u * (3 - 2 * u)
    if x <= 0.88:
        return 1.0
    return 1.0 - 0.2 * (x - 0.88) / 0.12


def tempo_cost(a: float, b: float) -> Tuple[float, str]:
    d = 100.0 * abs(math.log(b / a))
    if d <= 3.0:
        c, lab = 0.4 * d, "goed"
    elif d <= 6.0:
        c, lab = 1.2 + 1.2 * (d - 3.0), "oké"
    else:
        c, lab = 10.0 + 1.5 * (d - 6.0), "te groot: harde overgang (echo-cut)"
    if not stretch_feasible(a, b) and d <= 6.0:
        lab = "net te groot: harde overgang (echo-cut)"
    if b < a:
        c += W_DOWN * 100.0 * math.log(a / b)
    return c, lab


def _same_artist(a: TrackAnalysis, b: TrackAnalysis) -> bool:
    x, y = (a.artist or "").strip().lower(), (b.artist or "").strip().lower()
    return bool(x) and x == y


def pair_cost(a: TrackAnalysis, b: TrackAnalysis) -> float:
    c, _ = tempo_cost(a.bpm, b.bpm)
    c += camelot_relation(a.key, b.key)[1]
    if _same_artist(a, b):
        c += P_SAME_ARTIST
    return c


def pair_score(a: TrackAnalysis, b: TrackAnalysis) -> Tuple[float, str]:
    """(score 0..100, Dutch explanation) for playing b directly after a."""
    tc, tlab = tempo_cost(a.bpm, b.bpm)
    rel, kc = camelot_relation(a.key, b.key)
    c = tc + kc + (P_SAME_ARTIST if _same_artist(a, b) else 0.0)
    pct = 100.0 * (b.bpm / a.bpm - 1.0)
    sign = "±" if abs(pct) < 0.05 else ("+" if pct > 0 else "−")
    parts = [f"tempo {_nl(a.bpm)} → {_nl(b.bpm)} BPM ({sign}{_nl(abs(pct))} %, {tlab})"]
    k = f"{a.key or '?'} → {b.key or '?'}"
    parts.append(f"toonsoort {k}: {KEY_LABEL[rel]}")
    parts.append(f"energie {_nl(a.energy, 0)} → {_nl(b.energy, 0)}")
    if _same_artist(a, b):
        parts.append("zelfde artiest achter elkaar")
    score = max(0.0, min(100.0, 100.0 - 8.0 * c))
    txt = "; ".join(parts)
    return round(score, 1), txt[:1].upper() + txt[1:]


def _rank01(v: Sequence[float]) -> np.ndarray:
    v = np.asarray(v, float)
    n = len(v)
    if n < 2 or np.ptp(v) < 1e-9:
        return np.full(n, 0.5)
    order = v.argsort(kind="stable")
    r = np.empty(n)
    r[order] = np.arange(n)
    # average ties
    for val in np.unique(v):
        m = v == val
        r[m] = r[m].mean()
    rank = r / (n - 1)
    lin = (v - v.min()) / np.ptp(v)
    return 0.5 * rank + 0.5 * lin


def order_cost_matrices(analyses: Sequence[TrackAnalysis], energy_curve: str = "party"):
    n = len(analyses)
    C = np.zeros((n, n))
    for i, a in enumerate(analyses):
        for j, b in enumerate(analyses):
            if i != j:
                C[i, j] = pair_cost(a, b)
    e = _rank01([t.energy for t in analyses])
    tb = _rank01([t.bpm for t in analyses])
    xs = np.linspace(0, 1, n) if n > 1 else np.zeros(1)
    tgt = np.array([energy_target(x, energy_curve) for x in xs])
    w_e = 0.0 if energy_curve == "none" else W_ENERGY
    Pp = w_e * (e[:, None] - tgt[None, :]) ** 2 + W_BPM_POS * (tb[:, None] - xs[None, :]) ** 2
    return C, Pp


def sequence_cost(seq: Sequence[int], C: np.ndarray, Pp: np.ndarray) -> float:
    c = sum(Pp[t, p] for p, t in enumerate(seq))
    c += sum(C[a, b] for a, b in zip(seq, seq[1:]))
    return float(c)


def order_tracks(analyses: List[TrackAnalysis], pinned_first: Optional[int] = None,
                 pinned_last: Optional[int] = None, energy_curve: str = "party",
                 seed: int = 1234) -> List[int]:
    """Best running order as a list of indices into `analyses`.
    energy_curve: "party" (default), "rising", "flat" or "none"."""
    n = len(analyses)
    if n <= 1:
        return list(range(n))
    if pinned_first is not None and pinned_first == pinned_last:
        pinned_last = None
    C, Pp = order_cost_matrices(analyses, energy_curve)

    # ---- beam search
    free = [i for i in range(n) if i not in (pinned_first, pinned_last)]
    start = [] if pinned_first is None else [pinned_first]
    beam = [(sum(Pp[t, p] for p, t in enumerate(start)), start)]
    slots = n - len(start) - (pinned_last is not None)
    for _ in range(slots):
        cand = []
        for cost, seq in beam:
            used = set(seq)
            p = len(seq)
            for t in free:
                if t in used:
                    continue
                c = cost + Pp[t, p] + (C[seq[-1], t] if seq else 0.0)
                cand.append((c, seq + [t]))
        cand.sort(key=lambda x: x[0])
        beam, seen = [], set()
        for c, s in cand:
            key = (frozenset(s), s[-1])
            if key in seen:
                continue
            seen.add(key)
            beam.append((c, s))
            if len(beam) >= BEAM:
                break
    if pinned_last is not None:
        beam = [(c + Pp[pinned_last, n - 1] + C[s[-1], pinned_last], s + [pinned_last]) for c, s in beam]
    best_c, best = min(((sequence_cost(s, C, Pp), s) for _, s in beam), key=lambda x: x[0])

    # ---- simulated annealing (deterministic)
    rng = random.Random(seed)
    lo = 1 if pinned_first is not None else 0
    hi = n - 2 if pinned_last is not None else n - 1      # inclusive movable range
    if hi - lo >= 1:
        cur, cur_c = list(best), best_c
        iters = 400 * n
        T0, T1 = 2.0, 0.02
        for it in range(iters):
            temp = T0 * (T1 / T0) ** (it / iters)
            i, j = sorted(rng.sample(range(lo, hi + 1), 2))
            mv = rng.random()
            new = cur[:]
            if mv < 0.35:
                new[i], new[j] = new[j], new[i]
            elif mv < 0.7:
                new[i:j + 1] = new[i:j + 1][::-1]
            else:
                t = new.pop(i)
                new.insert(j, t)
            c = sequence_cost(new, C, Pp)
            if c < cur_c or rng.random() < math.exp(-(c - cur_c) / temp):
                cur, cur_c = new, c
                if c < best_c - 1e-12:
                    best, best_c = new[:], c
    return list(best)


def explain_order(analyses: Sequence[TrackAnalysis], order: Sequence[int]) -> List[str]:
    """Dutch one-liners per adjacent pair (for the GUI)."""
    out = []
    for a, b in zip(order, order[1:]):
        s, why = pair_score(analyses[a], analyses[b])
        out.append(f"{analyses[a].title} → {analyses[b].title} ({_nl(s, 0)}/100): {why}")
    return out
