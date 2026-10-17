"""Planner tests: order + cue constraints on the 14 real reference tracks (fixture) and synthetic cases."""
import copy
import json
import random
import time

import numpy as np
import pytest

from partymix.model import MixPlan, PlannedTrack, TrackAnalysis, Transition
from partymix.planner import (layout, max_stretch, mix_duration, order_tracks, pair_score,
                              plan_mix, plan_mix_with_report, transition_tempo)
from partymix.planner._fixtures import load_reference_tracks, reference_plan
from partymix.planner.common import MAX_STRETCH, camelot_relation, stretch_feasible
from partymix.planner.order import explain_order

PRESETS = {"xfade", "swap", "swap_hp", "editor", "echo_out", "echo_cut", "double"}


@pytest.fixture(scope="module")
def tracks():
    return load_reference_tracks()


def hand_plan():
    H = reference_plan()
    return MixPlan([PlannedTrack("ref-" + k, s, e) for k, s, e in H["set"]],
                   [Transition(o, t, p, dict(op)) for o, t, p, op in H["tr"]],
                   H["start_tempo"], H["end_tempo"], H.get("end_fade", 8))


def engine_total_sec(P):
    """Literal re-implementation of reference/engine.py's layout lines."""
    SET, TR = P["set"], P["tr"]
    n = len(SET)
    O = [0]
    for i in range(n - 1):
        O.append(O[i] + (SET[i][2] - SET[i][1]) - TR[i][0])
    TOTAL_BARS = O[-1] + SET[-1][2] - SET[-1][1]
    anch = [(0, P["start_tempo"])]
    for i in range(n - 1):
        anch += [(O[i + 1], TR[i][1]), (O[i + 1] + TR[i][0], TR[i][1])]
    anch.append((TOTAL_BARS + 8, P["end_tempo"]))
    ax = np.array([a for a, _ in anch], float)
    ay = np.array([t for _, t in anch], float)
    beat_tempo = np.interp((np.arange((TOTAL_BARS + 8) * 4) + 0.5) / 4, ax, ay)
    beat_dur = 60 / beat_tempo
    beat_t = np.concatenate([[0], np.cumsum(beat_dur)])
    xb = TOTAL_BARS * 4
    j = int(np.clip(np.floor(xb), 0, len(beat_dur) - 1))
    return float(beat_t[j] + (xb - j) * beat_dur[j])


def check_plan(plan: MixPlan, A, require_phrase_frac=0.6):
    n = len(A)
    assert len(plan.tracks) == n and len(plan.transitions) == n - 1
    on_phrase = total = 0
    for i, (pt, a) in enumerate(zip(plan.tracks, A)):
        assert pt.track_hash == a.file_hash
        assert 0 <= pt.start_bar < pt.end_bar <= len(a.bar_bass)
        for b in (pt.start_bar, pt.end_bar):
            assert (b - a.phrase_offset) % 4 == 0, f"{a.title}: bar {b} not on the 4-bar grid"
            total += 1
            on_phrase += (b - a.phrase_offset) % 8 == 0
        ov_in = plan.transitions[i - 1].overlap_bars if i else 0
        ov_out = plan.transitions[i].overlap_bars if i < n - 1 else 0
        # incoming window and outgoing window both inside the segment and not overlapping each other
        assert pt.start_bar + ov_in <= pt.end_bar - ov_out, a.title
    assert on_phrase / total >= require_phrase_frac
    for i, tr in enumerate(plan.transitions):
        assert tr.preset in PRESETS
        assert tr.overlap_bars in (0, 4, 8, 16)
        assert (tr.overlap_bars == 0) == (tr.preset == "echo_cut")
        sw = tr.options.get("swap", 0.5)
        assert 0.0 <= sw <= 1.0
        a, b = A[i].bpm, A[i + 1].bpm
        assert min(a, b) - 0.01 <= tr.tempo <= max(a, b) + 0.01
        if not stretch_feasible(a, b):
            assert tr.preset == "echo_cut"
    lay = layout(plan)
    assert all(x < y for x, y in zip(lay.offsets, lay.offsets[1:])), "layout must be monotonic"
    # stretch ≤ 3 % unless the pair is an echo cut that cannot be stretched
    st = max_stretch(plan, A)
    for i, s in enumerate(st):
        cut_near = (i > 0 and plan.transitions[i - 1].preset == "echo_cut") or \
                   (i < n - 1 and plan.transitions[i].preset == "echo_cut")
        if not cut_near:
            assert s <= MAX_STRETCH + 1e-9, (A[i].title, s)


# --------------------------------------------------------------------------------------------
def test_fixture_loads(tracks):
    assert len(tracks) == 14
    for t in tracks:
        assert len(t.bar_bass) == len(t.bar_vocal) == len(t.bar_rms) > 40
        assert 0 <= t.phrase_offset < 8 and t.sections and t.key
        assert 1 <= t.energy <= 10
        assert TrackAnalysis.from_json(t.to_json()).bar_bass == t.bar_bass


def test_layout_equals_engine():
    P = reference_plan()
    hp = hand_plan()
    assert mix_duration(hp) == pytest.approx(engine_total_sec(P), abs=1e-6)
    assert 29.5 * 60 < mix_duration(hp) < 31.5 * 60          # the v2 mix is ~30 min
    assert layout(hp).total_bars == 996


@pytest.mark.parametrize("minutes", [20, 30, 40])
def test_plan_constraints_and_duration(tracks, minutes):
    plan, rep = plan_mix_with_report(tracks, minutes)
    check_plan(plan, tracks)
    assert rep.reachable
    assert abs(rep.achieved_sec - minutes * 60) <= 0.10 * minutes * 60
    assert rep.achieved_sec == pytest.approx(mix_duration(plan))
    # every track plays at least ~40 s (outside the overlaps counted half)
    lay = layout(plan)
    for i, pt in enumerate(plan.tracks):
        ov_in = plan.transitions[i - 1].overlap_bars if i else 0
        ov_out = plan.transitions[i].overlap_bars if i < len(tracks) - 1 else 0
        a = lay.offsets[i] + ov_in / 2
        z = lay.offsets[i] + (pt.end_bar - pt.start_bar) - ov_out / 2
        assert lay.time_of(z) - lay.time_of(a) >= 40.0, tracks[i].title


def test_v2_comparison_with_hand_plan(tracks):
    """Most incoming windows should end where the hand plan's do (same phrase logic)."""
    plan = plan_mix(tracks, 30)
    hp = hand_plan()
    same_w = same_swap = 0
    for i in range(len(tracks) - 1):
        a, h = plan.transitions[i], hp.transitions[i]
        wa = plan.tracks[i + 1].start_bar + a.overlap_bars
        wh = hp.tracks[i + 1].start_bar + h.overlap_bars
        same_w += abs(wa - wh) <= 4
        if a.overlap_bars and h.overlap_bars:
            sa = plan.tracks[i + 1].start_bar + a.options.get("swap", .5) * a.overlap_bars
            sh = hp.tracks[i + 1].start_bar + h.options.get("swap", .5) * h.overlap_bars
            same_swap += abs(sa - sh) <= 4
    assert same_w >= 10
    assert same_swap >= 8
    # the Brokko Maken → Axel F key+tempo jump is an echo cut, as in the hand plan
    assert plan.transitions[11].preset == "echo_cut"


def test_transition_tempos_like_hand(tracks):
    hp = hand_plan()
    for i, h in enumerate(hp.transitions):
        if h.preset == "echo_cut":
            continue
        t = transition_tempo(tracks[i].bpm, tracks[i + 1].bpm)
        assert abs(t - h.tempo) <= 0.51


def test_unreachable_target_reports(tracks):
    plan, rep = plan_mix_with_report(tracks[:4], 60)
    check_plan(plan, tracks[:4])
    assert not rep.reachable and rep.warnings
    assert rep.achieved_sec < 60 * 60 * 0.9


def test_deterministic_and_json(tracks):
    p1 = plan_mix(tracks, 30).to_json()
    p2 = plan_mix(tracks, 30).to_json()
    assert p1 == p2
    assert MixPlan.from_json(p1).to_json() == p1
    json.loads(p1)


def _synthetic(title, bpm, key, n=96, intro=16, outro=16, phrase=0, vocals=True, artist=""):
    beats = list(np.arange((n + 2) * 4) * 60.0 / bpm)
    bass = [-30.0 if (b < intro or b >= n - outro) else 0.0 for b in range(n)]
    voc = [(-6.0 if (vocals and intro + 8 <= b < n - outro) else -40.0) for b in range(n)]
    return TrackAnalysis(path=title, file_hash="h-" + title, title=title, artist=artist,
                         duration=beats[-1], sample_rate=44100, bpm=bpm, constant_tempo=True,
                         beats=beats, first_downbeat_index=0, key=key, key_confidence=1.0,
                         phrase_offset=phrase, bar_bass=bass, bar_mid=[0.0] * n, bar_high=[0.0] * n,
                         bar_vocal=voc, bar_rms=[-8.0] * n, sections=[], energy=6.0)


def test_big_tempo_jump_is_echo_cut():
    A = [_synthetic("a", 120, "8A"), _synthetic("b", 132, "8A")]
    plan, rep = plan_mix_with_report(A, 6)
    assert plan.transitions[0].preset == "echo_cut" and plan.transitions[0].overlap_bars == 0
    assert any("Tempoverschil" in w for w in rep.warnings)


def test_synthetic_intro_rules():
    A = [_synthetic("a", 126, "8A"), _synthetic("b", 127, "8A", intro=16), _synthetic("c", 128, "9A", intro=8)]
    plan = plan_mix(A, 8)
    check_plan(plan, A)
    # incoming window ends exactly at the bass entry (bar 16 for b, bar 8 for c)
    assert plan.tracks[1].start_bar + plan.transitions[0].overlap_bars == 16
    assert plan.tracks[2].start_bar + plan.transitions[1].overlap_bars == 8
    # swap = bass entry at the window end, or 0 when the outgoing window has no bass (outro)
    e = plan.tracks[0].end_bar
    a_has_bass = max(A[0].bar_bass[e - plan.transitions[0].overlap_bars:e]) > -10
    assert plan.transitions[0].options["swap"] == (1.0 if a_has_bass else 0.0)


def test_robust_to_missing_data():
    A = [_synthetic("a", 126, "", phrase=3), _synthetic("b", 127, "1B", n=40, intro=8, outro=4),
         _synthetic("c", 128, "2A", vocals=False)]
    A[1].bar_vocal = []
    A[2].bar_rms = []
    plan, rep = plan_mix_with_report(A, 5)
    check_plan(plan, A, require_phrase_frac=0.0)


# ---- order ------------------------------------------------------------------------------------
def test_order_shuffled_reference(tracks):
    tr = list(tracks)
    random.Random(5).shuffle(tr)
    o = order_tracks(tr)
    assert sorted(o) == list(range(14))
    bpms = [tr[i].bpm for i in o]
    assert np.corrcoef(np.arange(14), bpms)[0, 1] > 0.7          # BPM rises over the set
    # deterministic, independent of input order
    tr2 = list(tracks)
    random.Random(9).shuffle(tr2)
    assert [tr[i].file_hash for i in o] == [tr2[i].file_hash for i in order_tracks(tr2)]
    # no tempo jump that would need a cut
    assert all(stretch_feasible(a, b) for a, b in zip(bpms, bpms[1:]))
    assert len(explain_order(tr, o)) == 13


def test_order_pins_and_speed():
    base = load_reference_tracks()
    rng = random.Random(3)
    keys = ["1A", "2A", "3B", "4A", "5B", "6A", "7A", "8B", "9A", "10A", "11B", "12A"]
    tr = []
    for i in range(30):
        t = copy.deepcopy(base[i % 14])
        t.file_hash, t.bpm, t.key = f"x{i}", t.bpm * rng.uniform(0.97, 1.04), rng.choice(keys)
        t.artist = "Zelfde" if i < 3 else f"art{i}"
        tr.append(t)
    t0 = time.time()
    o = order_tracks(tr, pinned_first=5, pinned_last=7)
    assert time.time() - t0 < 2.0
    assert o[0] == 5 and o[-1] == 7 and sorted(o) == list(range(30))
    pos = {t: p for p, t in enumerate(o)}
    assert all(abs(pos[a] - pos[b]) > 1 for a in range(3) for b in range(3) if a < b)   # same artist apart
    t0 = time.time()
    plan, rep = plan_mix_with_report([tr[i] for i in o], 45)
    assert time.time() - t0 < 2.0
    assert rep.reachable


def test_pair_score_dutch(tracks):
    s, why = pair_score(tracks[0], tracks[1])
    assert 0 <= s <= 100 and "Tempo" in why and "toonsoort" in why
    s2, why2 = pair_score(tracks[11], tracks[12])
    assert s2 < s
    assert camelot_relation("8A", "8B")[0] == "relative"
    assert camelot_relation("8A", "9A")[0] == "adjacent"
    assert camelot_relation("12A", "1A")[0] == "adjacent"
    assert camelot_relation("8A", "10A")[0] == "boost"
