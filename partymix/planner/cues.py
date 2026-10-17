"""Cue planner: for every adjacent pair pick in/out bars, overlap, swap point, preset and tempo,
then fit the total length to the target duration.

Rules (reference/PLAN_NOTES.md):
* all cue points on the 4-bar grid, phrase (8-bar) boundaries preferred;
* incoming window [s, s+ov) ends where the incoming track's first main section starts
  (bass entry / drop / vocal entry over full bass);
* outgoing window [e-ov, e) ends at the end of a section (end of drop, end of break, track end);
* swap (bass + vocal hand-over) = incoming bass entry inside the window
  (0.5 if the intro already has bass, 0.0 if the outgoing has no bass in its window);
* 16-bar overlaps when both sides have long intro/outro material, else 8; 0 + echo_cut on tempo/key jumps;
* transition tempo between both native tempos (closer to the outgoing), stretch ≤ 3 %;
* total length ≈ target: Lagrangian DP over candidate out-points (+ greedy refinement),
  each track plays ≥ ~45 s, whole drops/choruses kept.

Primary signals are the per-bar bass / vocal arrays; `phrase_offset` only sets the grid phase and
the phrase bonus, `sections` are not required (robust to imperfect analysis).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..model import MixPlan, PlannedTrack, TrackAnalysis, Transition
from .common import (MAX_STRETCH, bass_levels, cut_tempos, camelot_distance, layout, max_stretch, mean,
                     n_bars_of, rms_levels, round_half, stretch_feasible, tempo_gap,
                     transition_tempo, vocal_levels)

BASS_ON = 7.5          # 4-bar mean bass level counted as "bass is in" (full bass)
MIN_PLAY_SEC = 45.0
END_FADE = 8
INF = 1e9

# cost weights (dimensionless "quality points")
W_LATE = 0.25          # per 4 bars the incoming main entry is later than the first possible one
W_VOC_PRE = 0.08       # per vocal level of the incoming before the swap (instrumental intro preferred)
W_VOC_POST = 0.04      # per vocal level of the outgoing after the swap
BONUS_OV16 = 0.4
P_NOT_PHRASE = 0.2
P_S_NOT_PHRASE = 0.1
W_BAL = 3.0           # quadratic penalty on relative deviation from the track's fair share of the target


# --------------------------------------------------------------------------------------------
@dataclass
class Profile:
    t: TrackAnalysis
    n: int
    bass: np.ndarray
    vocal: np.ndarray
    rms: np.ndarray
    off: int

    @property
    def base(self) -> int:
        return self.off % 4

    def grid(self, lo: int = 0, hi: Optional[int] = None) -> List[int]:
        hi = self.n if hi is None else hi
        g0 = self.base
        return [g for g in range(g0, hi + 1, 4) if g >= lo]

    def phrase(self, b: int) -> bool:
        return (b - self.off) % 8 == 0

    def snap(self, b: float) -> int:
        """nearest grid bar (ties go to the later one)."""
        return int(self.base + 4 * np.floor((b - self.base) / 4 + 0.5))

    def B(self, a: int, z: int) -> float:
        return mean(self.bass, a, z)

    def V(self, a: int, z: int) -> float:
        return mean(self.vocal, a, z)

    def bass_entry(self, lo: int, hi: int) -> Optional[int]:
        """first bar b in [lo, hi) where the 4-bar mean bass reaches BASS_ON (exact bar)."""
        for b in range(max(lo, 0), min(hi, self.n)):
            if self.B(b, b + 4) >= BASS_ON:
                return b
        return None


def profile(t: TrackAnalysis) -> Profile:
    n = n_bars_of(t)
    return Profile(t, n, bass_levels(t, n), vocal_levels(t, n), rms_levels(t, n), int(t.phrase_offset) % 8)


@dataclass
class InOpt:
    s: int
    ov: int
    W: int            # end of the incoming window = start of its main section
    swap: float       # bass hand-over fraction (before the A-side rule)
    bass_in_intro: bool
    cost: float
    why: str = ""


# --------------------------------------------------------------------------------------------
# incoming options
# --------------------------------------------------------------------------------------------
def _swap_for(p: Profile, s: int, ov: int) -> Tuple[float, bool]:
    W = s + ov
    b = p.bass_entry(s, W + 2)
    if b is None:
        return 1.0, False
    g = min(max(p.snap(b), s), W)
    if g <= s:
        return 0.5, True          # intro already has bass → hand over halfway
    return round((g - s) / ov, 3), False


def incoming_options(p: Profile, ovs=(8, 16)) -> List[InOpt]:
    maxW = int(min(64, max(16, 0.5 * p.n)))
    raw = []
    for W in p.grid(8, maxW):
        post_b, post_v = p.B(W, W + 8), p.V(W, W + 8)
        main = post_b >= 6.0 or (post_b >= 5.0 and post_v >= 5.0)
        if not main:
            continue
        db = p.B(W, W + 4) - p.B(W - 4, W)
        dv = p.V(W, W + 4) - p.V(W - 4, W)
        ent = p.bass_entry(0, W + 2)
        entry_here = ent is not None and ent > 0 and W - 8 <= ent <= W + 1
        if not (abs(db) >= 2.0 or dv >= 2.0 or entry_here):
            continue
        for ov in ovs:
            s = W - ov
            if s < 0:
                continue
            sw, bin_ = _swap_for(p, s, ov)
            sb = int(round(s + sw * ov))
            c = W_VOC_PRE * p.V(s, sb) if sb > s else 0.0
            c += 0.0 if p.phrase(W) else P_NOT_PHRASE
            c += 0.0 if p.phrase(s) else P_S_NOT_PHRASE
            c -= BONUS_OV16 if ov == 16 else 0.0
            raw.append(InOpt(s, ov, W, sw, bin_, c, f"hoofddeel begint op maat {W}"))
    if not raw:
        W = max(8, p.grid(8)[0] if p.grid(8) else 8)
        sw, bin_ = _swap_for(p, W - 8, 8)
        return [InOpt(W - 8, 8, W, sw, bin_, 2.0, "geen duidelijke intro gevonden")]
    strong = [o.W for o in raw if min(p.B(o.W, o.W + 4), p.B(o.W + 4, o.W + 8)) >= BASS_ON]
    W0 = min(strong) if strong else min(o.W for o in raw)
    for o in raw:
        if o.W >= W0:
            o.cost += W_LATE * (o.W - W0) / 4
        else:                                   # earlier, weaker "main" (half bass): allowed but worse
            o.cost += 0.4 * max(BASS_ON - min(p.B(o.W, o.W + 4), p.B(o.W + 4, o.W + 8)), 0.0) + 0.1
    raw.sort(key=lambda o: o.cost)
    return raw[:8]


def cut_in_option(p: Profile) -> InOpt:
    """Hard start after an echo_cut: first non-silent phrase start."""
    for g in p.grid(0, max(p.n - 16, 0)):
        if mean(p.rms, g, g + 4) >= 4.0:
            return InOpt(g, 0, g, 0.0, False, 0.0, "harde start op de 1")
    return InOpt(0, 0, 0, 0.0, False, 0.5, "harde start op de 1")


def first_track_options(p: Profile) -> List[InOpt]:
    """Start of the mix: ~8 bars before the first main section."""
    out = []
    for o in incoming_options(p, ovs=(8,)):
        out.append(InOpt(o.s, 0, o.W, 0.0, False, o.cost - W_VOC_PRE * p.V(o.s, o.W), o.why))
    out.append(InOpt(0, 0, 0, 0.0, False, 0.6, "vanaf het begin"))
    return out


# --------------------------------------------------------------------------------------------
# outgoing candidates
# --------------------------------------------------------------------------------------------
def boundary_strength(p: Profile, e: int) -> float:
    if e >= p.n - 4:
        return 9.0                                  # end of the track
    db4 = abs(p.B(e, e + 4) - p.B(e - 4, e))
    late = 0.8 * abs(p.B(e + 2, e + 6) - p.B(e - 4, e))      # bass stops just after e (short tail)
    dv4 = abs(p.V(e, e + 4) - p.V(e - 4, e))
    return max(db4, late, 0.7 * dv4)


def out_cost(p: Profile, e: int, ov: int, last: bool = False) -> float:
    if e > p.n or e - max(ov, 1) < 0:
        return INF
    c = 0.0 if p.phrase(e) else 0.5
    bs = boundary_strength(p, e)
    c += 0.0 if bs >= 2.0 else (0.5 if bs >= 1.0 else 1.0)
    if ov == 0 and not last:
        c += 0.0 if bs >= 2.0 else 0.6               # echo cut wants a clear section end
    if ov > 0 and not last:
        h = ov // 2
        first, second = p.B(e - ov, e - h), p.B(e - h, e)
        if second - first >= 3.0:
            c += 1.0                                 # A's drop starts inside the window
        elif abs(second - first) >= 3.0:
            c += 0.5
        rise_after = p.B(e, e + 4) - p.B(e - 4, e)
        if rise_after >= 3.0 and p.B(e, e + 8) >= 6.5:
            c += 0.3                                 # leaving right before A's own drop
    if last:
        rest = p.n - e
        c += 0.0 if rest <= 8 else (1.0 if bs >= 2.0 else 1.5)    # finale: let the last track end naturally
    return c


# --------------------------------------------------------------------------------------------
# preset choice
# --------------------------------------------------------------------------------------------
def choose_preset(pa: Profile, pb: Profile, e: int, opt: InOpt, a_bpm: float, b_bpm: float,
                  cut: bool) -> Tuple[str, Dict, str]:
    if cut:
        return "echo_cut", {}, "harde overgang met echo (tempo/toonsoort springt)"
    ov = opt.ov
    aB = pa.B(e - ov, e)
    sw = opt.swap
    if aB < 3.5:
        return ("swap", {"swap": 0.0, "fadein": 0.5, "lpin": True, "lpdur": 0.5},
                "uitgaand nummer zit in een break zonder bas: bas van het nieuwe nummer meteen erin")
    if tempo_gap(a_bpm, b_bpm) > 0.03:
        return "echo_out", {"swap": sw, "fadein": 0.0}, "groot tempoverschil: uitgaand nummer echoot weg"
    if ov == 16 and not opt.bass_in_intro:
        return "xfade", {"swap": sw, "fadein": 0.5, "lpin": True}, "lange intro: rustige crossfade met bas-wissel"
    if sw >= 0.75:
        if pa.V(e - max(ov // 4, 2), e) >= 6.5:
            return "editor", {"swap": sw, "fadein": 0.5}, "bas komt op het eind binnen; zang van A wordt weggefilterd met echo"
        return "swap", {"swap": sw, "fadein": 0.25, "lpin": True}, "bas komt op het eind binnen: bas-wissel op de drop"
    if opt.bass_in_intro:
        return "swap", {"swap": sw, "fadein": 0.25}, "intro heeft al bas: bas-wissel halverwege"
    return "swap_hp", {"swap": sw, "fadein": 0.25, "lpin": True}, "bas-wissel op de bas-entree, A met hoogdoorlaat eruit"


# --------------------------------------------------------------------------------------------
@dataclass
class TransitionNote:
    index: int
    out_window: Tuple[int, int]
    in_window: Tuple[int, int]
    swap_bar: Optional[int]
    reason: str


@dataclass
class PlanReport:
    target_sec: float
    achieved_sec: float
    reachable: bool
    max_stretch: List[float]
    notes: List[TransitionNote] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def achieved_minutes(self) -> float:
        return self.achieved_sec / 60

    def summary_nl(self) -> str:
        m, s = divmod(int(round(self.achieved_sec)), 60)
        tm = self.target_sec / 60
        txt = f"Mixlengte {m}:{s:02d} (doel {tm:.0f} min)"
        if not self.reachable:
            txt += " – doelduur niet haalbaar met deze nummers"
        return txt


def _is_cut(a: TrackAnalysis, b: TrackAnalysis) -> bool:
    if not stretch_feasible(a.bpm, b.bpm):
        return True
    return camelot_distance(a.key, b.key) >= 5 and camelot_distance(a.key, b.key) < 99 \
        and tempo_gap(a.bpm, b.bpm) >= 0.02


def plan_mix(analyses_in_order: Sequence[TrackAnalysis], target_minutes: float = 30.0) -> MixPlan:
    return plan_mix_with_report(analyses_in_order, target_minutes)[0]


def plan_mix_with_report(analyses_in_order: Sequence[TrackAnalysis], target_minutes: float = 30.0,
                         ) -> Tuple[MixPlan, PlanReport]:
    A = list(analyses_in_order)
    n = len(A)
    if n == 0:
        raise ValueError("geen nummers om te plannen")
    P = [profile(t) for t in A]
    target = float(target_minutes) * 60.0

    # ---- tempo plan (depends only on the order)
    cut = [_is_cut(A[i], A[i + 1]) for i in range(n - 1)]
    T, T_in = [], []
    for i in range(n - 1):
        if cut[i]:
            t, ti = cut_tempos(A[i].bpm, A[i + 1].bpm)
        else:
            t, ti = transition_tempo(A[i].bpm, A[i + 1].bpm), None
        T.append(t)
        T_in.append(ti)
    start_tempo = round_half(A[0].bpm) if abs(round_half(A[0].bpm) / A[0].bpm - 1) < 0.004 else round(A[0].bpm, 2)
    if n > 1:
        end_tempo = round_half((T[-1] + A[-1].bpm) / 2)
        if abs(end_tempo / A[-1].bpm - 1) > MAX_STRETCH:
            end_tempo = round(A[-1].bpm, 2)
    else:
        end_tempo = start_tempo
    t_in = [start_tempo] + [ti if ti is not None else t for t, ti in zip(T, T_in)]
    t_out = T + [end_tempo]
    tempo_i = [(t_in[i] + t_out[i]) / 2 for i in range(n)]

    # ---- options
    opts: List[List[InOpt]] = [first_track_options(P[0])]
    for i in range(1, n):
        opts.append([cut_in_option(P[i])] if cut[i - 1] else incoming_options(P[i]))
    ecands = [[e for e in P[i].grid(8, P[i].n)] or [P[i].n] for i in range(n)]

    caps = [float(min(pp.n, float(np.median([q.n for q in P])))) for pp in P]
    share = [target * c_ / sum(caps) for c_ in caps]

    def min_bars(i):
        return MIN_PLAY_SEC * tempo_i[i] / 240.0

    # ---- per-step cost / seconds tensors
    steps = []   # for i < n-1: (C[j,e,k], S[j,e,k]); last: (C[j,e], S[j,e])
    for i in range(n):
        p = P[i]
        J, E = opts[i], ecands[i]
        last = i == n - 1
        K = [None] if last else opts[i + 1]
        C = np.full((len(J), len(E), len(K)), INF)
        S = np.zeros_like(C)
        for ei, e in enumerate(E):
            for ki, k in enumerate(K):
                ovo = 0 if last else k.ov
                oc = out_cost(p, e, END_FADE if last else ovo, last=last)
                if oc >= INF:
                    continue
                jc = 0.0
                if not last and k.ov > 0:
                    sb = int(round(k.s + k.swap * k.ov))
                    aB = p.B(e - k.ov, e)
                    if aB < 3.5 and P[i + 1].B(k.s, max(sb, k.s + 1)) < 3.5:
                        jc += 0.4                              # both without bass: energy hole
                    swap_bar_a = e - k.ov + int(round((0.0 if aB < 3.5 else k.swap) * k.ov))
                    jc += W_VOC_POST * p.V(swap_bar_a, e)
                    jc += k.cost
                elif not last:
                    jc += k.cost
                for ji, j in enumerate(J):
                    ovi = j.ov
                    bars = (e - j.s) - ovo
                    play = (e - j.s) - (ovi + ovo) / 2
                    pen = 0.0
                    if e - ovo < j.W + 8 or e - ovo < j.s + ovi:
                        continue
                    if play < min_bars(i):
                        pen += 3.0 + 0.1 * (min_bars(i) - play)
                    sec_ = bars * 240.0 / tempo_i[i]
                    pen += (W_BAL * (0.3 if last else 1.0)) * ((sec_ - share[i]) / share[i]) ** 2
                    C[ji, ei, ki] = oc + jc + pen
                    S[ji, ei, ki] = sec_
        if not np.isfinite(C).any() or (C >= INF).all():
            # nothing feasible (very short track): allow anything on the grid with a penalty
            for ei, e in enumerate(E):
                for ki, k in enumerate(K):
                    ovo = 0 if last else k.ov
                    for ji, j in enumerate(J):
                        if e - ovo >= j.s + ovi_of(j) and e > j.s:
                            C[ji, ei, ki] = 5.0 + (0 if last else k.cost)
                            S[ji, ei, ki] = ((e - j.s) - ovo) * 240.0 / tempo_i[i]
        steps.append((C, S))

    first_cost = np.array([o.cost for o in opts[0]])

    def solve(lam: float):
        V = first_cost.copy()
        back = []
        for i in range(n):
            C, S = steps[i]
            tot = V[:, None, None] + C - lam * S
            tot[C >= INF] = INF
            if i < n - 1:
                flat = tot.reshape(-1, tot.shape[2])
                arg = flat.argmin(0)
                V = flat[arg, np.arange(tot.shape[2])]
                back.append(arg)
            else:
                flat = tot.reshape(-1)
                a = int(flat.argmin())
                back.append(a)
        # backtrack
        choice_j = [0] * n
        choice_e = [0] * n
        a = back[-1]
        CJ, CE, _ = steps[-1][0].shape
        j, e = divmod(a, CE)
        choice_j[-1], choice_e[-1] = j, e
        for i in range(n - 2, -1, -1):
            k = choice_j[i + 1]
            a = back[i][k]
            _, CE, _ = steps[i][0].shape
            j, e = divmod(int(a), CE)
            choice_j[i], choice_e[i] = j, e
        return choice_j, choice_e

    def evaluate(cj, ce):
        q = float(first_cost[cj[0]])
        sec = 0.0
        for i in range(n):
            C, S = steps[i]
            k = 0 if i == n - 1 else cj[i + 1]
            q += C[cj[i], ce[i], k]
            sec += S[cj[i], ce[i], k]
        return q, sec

    def build(cj, ce) -> Tuple[MixPlan, List[TransitionNote]]:
        tracks, trans, notes = [], [], []
        for i in range(n):
            tracks.append(PlannedTrack(A[i].file_hash, opts[i][cj[i]].s, ecands[i][ce[i]]))
        for i in range(n - 1):
            k = opts[i + 1][cj[i + 1]]
            e = tracks[i].end_bar
            name, o, why = choose_preset(P[i], P[i + 1], e, k, A[i].bpm, A[i + 1].bpm, cut[i])
            if T_in[i] is not None:
                o = dict(o, tempo_in=float(T_in[i]))
                why += f" (tempo springt van {T[i]:.1f} naar {T_in[i]:.1f} BPM)".replace(".", ",")
            trans.append(Transition(overlap_bars=k.ov, tempo=T[i], preset=name, options=o))
            sbar = None if k.ov == 0 else k.s + int(round(o.get("swap", 0.5) * k.ov))
            notes.append(TransitionNote(i, (e - k.ov, e), (k.s, k.s + k.ov), sbar, why))
        # variety: avoid three identical presets in a row
        for i in range(2, len(trans)):
            if trans[i].preset == trans[i - 1].preset == trans[i - 2].preset and trans[i].preset in ("swap_hp", "swap"):
                if trans[i].preset == "swap_hp":
                    trans[i].preset = "swap"
                elif opts[i + 1][cj[i + 1]].ov > 0 and not opts[i + 1][cj[i + 1]].bass_in_intro:
                    trans[i].preset = "swap_hp"
                    trans[i].options.setdefault("lpin", True)
        texts = []
        for nt in notes:
            a_, b_ = A[nt.index].title, A[nt.index + 1].title
            if nt.swap_bar is None:
                texts.append(f"{a_} → {b_}: {why_of(nt)}; uit op maat {nt.out_window[1]}, in op maat {nt.in_window[0]}")
            else:
                texts.append(f"{a_} → {b_}: {why_of(nt)}; uit maat {nt.out_window[0]}–{nt.out_window[1]}, "
                             f"in maat {nt.in_window[0]}–{nt.in_window[1]}, wissel op maat {nt.swap_bar}")
        plan = MixPlan(tracks=tracks, transitions=trans, start_tempo=float(start_tempo), notes=texts,
                       end_tempo=float(end_tempo), end_fade_bars=END_FADE, target_minutes=float(target_minutes))
        return plan, notes

    # ---- Lagrangian search on the length reward
    sols = {}

    def run(lam):
        cj, ce = solve(lam)
        q, sec = evaluate(cj, ce)
        sols[lam] = (q, sec, cj, ce)
        return sec

    def search(goal):
        lo, hi = -0.5, 0.5
        s_lo, s_hi = run(lo), run(hi)
        if goal <= s_lo or goal >= s_hi:
            return
        for _ in range(22):
            mid = (lo + hi) / 2
            if run(mid) < goal:
                lo = mid
            else:
                hi = mid

    search(target)
    # exact duration (engine maths) of the best approximate solution, then correct once
    def pick(goal):
        best = min(sols.values(), key=lambda v: (abs(v[1] - goal) > 0.01 * goal, abs(v[1] - goal) + 2.0 * v[0]))
        return best

    q, sec, cj, ce = pick(target)
    plan, _ = build(cj, ce)
    exact = layout(plan).total_sec
    if sec > 0 and abs(exact - sec) > 1.0:
        goal2 = target * sec / exact
        search(goal2)
        q, sec, cj, ce = pick(goal2)

    # ---- greedy refinement on exact duration
    cj, ce = list(cj), list(ce)

    def exact_of(cj_, ce_):
        return layout(build(cj_, ce_)[0]).total_sec

    cur_q, cur_s = evaluate(cj, ce)
    cur_d = exact_of(cj, ce)
    for _ in range(3 * n):
        err = abs(cur_d - target)
        if err <= 0.01 * target:
            break
        ratio = cur_d / cur_s if cur_s > 0 else 1.0
        best = None
        for i in range(n):
            for ne in range(len(ecands[i])):
                if ne == ce[i]:
                    continue
                ce2 = ce.copy()
                ce2[i] = ne
                q2, s2 = evaluate(cj, ce2)
                if q2 >= INF / 2:
                    continue
                gain = err - abs(s2 * ratio - target)
                if gain <= 0:
                    continue
                score = (q2 - cur_q) - gain / 30.0          # 1 quality point ≈ 30 s
                if best is None or score < best[0]:
                    best = (score, i, ne, q2, s2)
        if best is None or best[0] > 0.5:
            break
        _, i, ne, cur_q, cur_s = best
        ce[i] = ne
        cur_d = exact_of(cj, ce)

    plan, notes = build(cj, ce)
    achieved = layout(plan).total_sec
    st = max_stretch(plan, A)
    warnings = []
    for i, s_ in enumerate(st):
        if s_ > MAX_STRETCH + 1e-6:
            warnings.append(f"'{A[i].title}' wordt {s_ * 100:.1f} % versneld/vertraagd (meer dan 3 %)")
    if cut and any(cut):
        for i, c in enumerate(cut):
            if c and not stretch_feasible(A[i].bpm, A[i + 1].bpm):
                warnings.append(f"Tempoverschil {A[i].title} → {A[i + 1].title} te groot: harde overgang")
    # reachable within ±10 %?
    mins = min(v[1] for v in sols.values())
    maxs = max(v[1] for v in sols.values())
    reachable = abs(achieved - target) <= 0.10 * target
    if not reachable:
        if target > maxs:
            warnings.append("Doelduur te lang voor deze nummers: maximale lengte gebruikt")
        elif target < mins:
            warnings.append("Doelduur te kort voor deze nummers: elk nummer speelt minstens ~45 s")
    report = PlanReport(target, achieved, reachable, st, notes, warnings)
    return plan, report


def why_of(nt: "TransitionNote") -> str:
    return nt.reason


def ovi_of(j: InOpt) -> int:
    return j.ov


def describe_plan(plan: MixPlan) -> str:
    """Short Dutch text summary (for logs / GUI tooltips)."""
    lay = layout(plan)
    m, s = divmod(int(round(lay.total_sec)), 60)
    return f"{len(plan.tracks)} nummers, {lay.total_bars} maten, {m}:{s:02d}"
