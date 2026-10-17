"""Dev tool: compare the automatic plan with the hand-made v2 plan on the reference fixture.

    python -m partymix.planner._evaluate [target_minutes]
"""
from __future__ import annotations

import random
import sys
import time

from ..model import MixPlan, PlannedTrack, Transition
from ._fixtures import load_reference_tracks, reference_plan
from .common import mix_duration
from .cues import plan_mix_with_report
from .order import order_tracks, pair_score


def hand_plan() -> MixPlan:
    H = reference_plan()
    return MixPlan([PlannedTrack("ref-" + k, s, e) for k, s, e in H["set"]],
                   [Transition(o, t, p, dict(op)) for o, t, p, op in H["tr"]],
                   H["start_tempo"], H["end_tempo"], H.get("end_fade", 8))


def compare(target: float = 30.0) -> None:
    tr = load_reference_tracks()
    t0 = time.time()
    plan, rep = plan_mix_with_report(tr, target)
    dt = time.time() - t0
    hp = hand_plan()
    print(f"plan time {dt:.2f}s  auto {rep.achieved_sec / 60:.2f} min  hand {mix_duration(hp) / 60:.2f} min  "
          f"{rep.summary_nl()}  {rep.warnings}")
    for i, t in enumerate(tr):
        a, h = plan.tracks[i], hp.tracks[i]
        print(f"{t.title[:16]:16s} off{t.phrase_offset} play hand {h.start_bar:3d}-{h.end_bar:3d}  auto {a.start_bar:3d}-{a.end_bar:3d}")
        if i < len(tr) - 1:
            T, HT, nt = plan.transitions[i], hp.transitions[i], rep.notes[i]
            hs = HT.options.get("swap", 0.5)
            hsb = None if HT.overlap_bars == 0 else hp.tracks[i + 1].start_bar + int(hs * HT.overlap_bars)
            print(f"   hand ov{HT.overlap_bars:2d} {HT.tempo:5.1f} {HT.preset:8s} out {h.end_bar - HT.overlap_bars}-{h.end_bar} "
                  f"in {hp.tracks[i + 1].start_bar}-{hp.tracks[i + 1].start_bar + HT.overlap_bars} swapbar {hsb}")
            print(f"   auto ov{T.overlap_bars:2d} {T.tempo:5.1f} {T.preset:8s} out {nt.out_window[0]}-{nt.out_window[1]} "
                  f"in {nt.in_window[0]}-{nt.in_window[1]} swapbar {nt.swap_bar}  {T.options}  | {nt.reason}")
    print("tempo", plan.start_tempo, [t.tempo for t in plan.transitions], plan.end_tempo)
    print("max stretch %", [round(x * 100, 1) for x in rep.max_stretch])


def order_demo(seed: int = 7) -> None:
    tr = load_reference_tracks()
    random.Random(seed).shuffle(tr)
    t0 = time.time()
    idx = order_tracks(tr)
    dt = time.time() - t0
    print(f"order time {dt:.2f}s")
    seq = [tr[i] for i in idx]
    for a, b in zip(seq, seq[1:]):
        sc, why = pair_score(a, b)
        print(f"  {a.title[:16]:16s} {a.bpm:6.1f} {a.key:>3s} E{a.energy:4.1f} → {b.title[:16]:16s} {sc:5.1f}  {why}")
    print(f"  {seq[-1].title[:16]:16s} {seq[-1].bpm:6.1f} {seq[-1].key:>3s} E{seq[-1].energy:4.1f}")


if __name__ == "__main__":
    tgt = float(sys.argv[1]) if len(sys.argv) > 1 else 30.0
    compare(tgt)
    if "--order" in sys.argv or len(sys.argv) == 1:
        order_demo()
