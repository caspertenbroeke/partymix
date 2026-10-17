"""Fast synthetic tests for render layout math, timemaps, presets, loudness and limiter."""
import numpy as np
import pytest

from partymix.model import MixPlan, PlannedTrack, Transition, TrackAnalysis
from partymix.render.layout import compute_layout, estimate_duration
from partymix.render.stretch import build_timemap
from partymix.render.mixer import preset_envelopes, MasterBuffer
from partymix import render


def _plan(tempos=(126.0, 128.0), ovs=(16, 8), lens=(64, 48, 40), start=124.0, end=130.0):
    tracks = [PlannedTrack(f"h{i}", 0, L) for i, L in enumerate(lens)]
    trs = [Transition(ov, t, "swap", {}) for ov, t in zip(ovs, tempos)]
    return MixPlan(tracks=tracks, transitions=trs, start_tempo=start, end_tempo=end)


def _analysis(bpm=125.0, first=0.3, n_beats=600, key="h0"):
    per = 60 / bpm
    return TrackAnalysis(path=key, file_hash=key, title=key, artist="", duration=n_beats * per + 1,
                         sample_rate=44100, bpm=bpm, constant_tempo=True,
                         beats=list(first + per * np.arange(n_beats)), first_downbeat_index=0,
                         key="", key_confidence=0, phrase_offset=0)


def test_offsets_and_total():
    lay = compute_layout(_plan())
    assert lay.offsets == [0, 48, 88]
    assert lay.total_bars == 128
    assert len(lay.beat_dur) == (128 + 8) * 4


def test_tempo_held_during_overlap_and_ramps():
    lay = compute_layout(_plan())
    # overlap 1: bars 48..64 at 126, overlap 2: bars 88..96 at 128
    for b in (48.1, 55, 63.9):
        assert lay.tempo_at(b) == pytest.approx(126.0)
    for b in (88.1, 95.9):
        assert lay.tempo_at(b) == pytest.approx(128.0)
    assert lay.tempo_at(0.01) == pytest.approx(124.0, abs=0.01)
    # linear ramp between transitions (bar 64 -> 88: 126 -> 128)
    assert lay.tempo_at(76.0 + 0.125) == pytest.approx(127.0, abs=0.02)
    # end ramp to end tempo at total+8
    assert lay.tempo_at(135.99) == pytest.approx(130.0, abs=0.02)


def test_constant_tempo_time_and_duration():
    p = _plan(tempos=(120.0, 120.0), start=120.0, end=120.0)
    lay = compute_layout(p)
    assert lay.t(1) == pytest.approx(2.0)
    assert lay.t(10.5) == pytest.approx(21.0)
    assert lay.total_sec == pytest.approx(128 * 2.0)
    assert estimate_duration(p) == pytest.approx(256.0)
    assert render.estimate_duration(p) == pytest.approx(259.0)  # + 3 s tail written to file


def test_engine_equivalence():
    """Same numbers as reference/engine.py's inline math."""
    p = _plan()
    lay = compute_layout(p)
    SET = [(t.start_bar, t.end_bar) for t in p.tracks]
    TR = [(t.overlap_bars, t.tempo) for t in p.transitions]
    O = [0]
    for i in range(len(SET) - 1):
        O.append(O[i] + (SET[i][1] - SET[i][0]) - TR[i][0])
    TB = O[-1] + SET[-1][1] - SET[-1][0]
    anch = [(0, p.start_tempo)]
    for i in range(len(SET) - 1):
        anch += [(O[i + 1], TR[i][1]), (O[i + 1] + TR[i][0], TR[i][1])]
    anch.append((TB + 8, p.end_tempo))
    ax = np.array([a for a, _ in anch], float); ay = np.array([t for _, t in anch], float)
    bd = 60 / np.interp((np.arange((TB + 8) * 4) + 0.5) / 4, ax, ay)
    bt = np.concatenate([[0], np.cumsum(bd)])
    x = np.linspace(0, TB, 517)
    xb = x * 4; j = np.clip(np.floor(xb).astype(int), 0, len(bd) - 1)
    np.testing.assert_allclose(lay.T(x), bt[j] + (xb - j) * bd[j], rtol=0, atol=1e-12)


def test_invalid_plans():
    with pytest.raises(ValueError):
        compute_layout(_plan(ovs=(80, 8)))
    p = _plan()
    p.transitions = p.transitions[:1]
    with pytest.raises(ValueError):
        compute_layout(p)


def test_single_track_plan():
    p = MixPlan(tracks=[PlannedTrack("h0", 0, 32)], transitions=[], start_tempo=128, end_tempo=128)
    lay = compute_layout(p)
    assert lay.total_bars == 32 and lay.total_sec == pytest.approx(32 * 4 * 60 / 128)


@pytest.mark.parametrize("first", [0.3, 0.0, -0.004, -0.2])
def test_timemap_positive_and_monotonic(first):
    a = _analysis(first=first)
    p = MixPlan(tracks=[PlannedTrack("h0", 0, 32), PlannedTrack("h1", 0, 32)],
                transitions=[Transition(8, 127.0, "swap", {})], start_tempo=125, end_tempo=127)
    lay = compute_layout(p)
    sr = 44100
    lo, hi, lines, pre = build_timemap(a, 0, 32, lay, 0, sr, src_frames=10 ** 8)
    assert pre == pytest.approx(0.05)
    assert lines[0] == (0, 0)
    src = np.array([l[0] for l in lines]); out = np.array([l[1] for l in lines])
    assert (src >= 0).all() and (out >= 0).all()
    assert (np.diff(src) > 0).all() and (np.diff(out) > 0).all()
    assert lo == round((a.bar_time(0) - 0.05) * sr)
    # bar 0 maps to `pre`, bar 16 (beat 64) maps to output T(16) + pre
    assert lines[1] == (round(0.05 * sr), round(0.05 * sr))
    assert lines[65][1] == round((lay.t(16) + pre) * sr)
    assert lines[65][0] == round((a.bar_time(16) - a.bar_time(0) + 0.05) * sr)


def test_timemap_stops_at_source_end():
    a = _analysis(n_beats=140)  # 35 bars only
    p = MixPlan(tracks=[PlannedTrack("h0", 0, 34)], transitions=[], start_tempo=125, end_tempo=125)
    lay = compute_layout(p)
    lo, hi, lines, pre = build_timemap(a, 0, 34, lay, 0, 44100, src_frames=10 ** 8)
    assert lines[-1][0] <= hi - lo


# ---------------------------------------------------------------- presets
U = np.linspace(0, 0.999, 1000)


def test_swap_point_bass_and_vocals():
    for sw in (0.0, 0.5, 1.0):
        va, ba, fa, xa, ga = preset_envelopes("swap", "A", U, {"swap": sw})
        vb, bb, fb, xb, gb = preset_envelopes("swap", "B", U, {"swap": sw, "fadein": 0.25})
        # never both basses at the same time, and exactly one after/before the swap (for sw in (0,1))
        assert not ((ba > 0) & (bb > 0)).any()
        assert not ((ga > 0) & (gb > 0)).any()
        assert (ba + bb == 1).all() or sw == 0.0
        assert (va == 1).all()


def test_swap_no_vocal_handover():
    _, _, _, _, ga = preset_envelopes("swap", "A", U, {"voc": 0})
    _, _, _, _, gb = preset_envelopes("swap", "B", U, {"voc": False})
    assert (ga == 1).all() and (gb == 1).all()


def test_fadein_and_lpin():
    vb, _, fb, _, _ = preset_envelopes("swap", "B", U, {"fadein": 0.5, "lpin": 1})
    assert vb[0] == 0 and vb[U >= 0.5].min() == 1 and (np.diff(vb) >= -1e-12).all()
    assert fb[0] == pytest.approx(-0.65) and fb[-1] == 0
    vb0, _, fb0, _, _ = preset_envelopes("swap", "B", U, {"fadein": 0})
    assert (vb0 == 1).all() and (fb0 == 0).all()


def test_preset_specifics():
    _, _, f, _, _ = preset_envelopes("swap_hp", "A", U, {})
    assert f[U < 0.5].max() == 0 and f[-1] == pytest.approx(0.5, abs=1e-3)
    _, _, f, fx, _ = preset_envelopes("editor", "A", U, {})
    assert f[-1] == pytest.approx(-0.6, abs=1e-3) and fx[U < 0.75].max() == 0 and fx[-1] == pytest.approx(0.45, abs=3e-3)
    v, _, _, _, _ = preset_envelopes("xfade", "A", U, {"swap": 0.5})
    assert (v[U < 0.5] == 1).all() and v[-1] < 0.01
    _, _, _, fx, _ = preset_envelopes("echo_out", "A", U, {})
    assert fx[U < 0.5].max() == 0 and fx[-1] == pytest.approx(0.6, abs=3e-3)
    for name in ("swap", "double"):
        v, b, f, fx, g = preset_envelopes(name, "A", U, {})
        assert (v == 1).all() and (f == 0).all() and (fx == 0).all()


def test_master_buffer_file(tmp_path):
    m = MasterBuffer(1000, str(tmp_path / "m.f32"))
    m.add(-10, np.ones((50, 2)))
    m.add(990, np.ones((50, 2)) * 2)
    m.add(20, np.ones((10, 2)))
    x = m.read(0, 1000)
    assert x[:20].sum() == 40 and x[20:30].sum() == 40 and x[39:41, 0].tolist() == [1, 0]
    assert x[990:].sum() == 40 and x.shape == (1000, 2)


# ---------------------------------------------------------------- loudness / limiter
def test_loudness_matches_pyloudnorm():
    pyln = pytest.importorskip("pyloudnorm")
    from partymix.render.loudness import integrated_loudness, LoudnessMeter
    rng = np.random.default_rng(3)
    x = rng.standard_normal((44100 * 5 + 77, 2)) * np.linspace(0.02, 0.4, 44100 * 5 + 77)[:, None]
    ref = pyln.Meter(44100).integrated_loudness(x)
    assert integrated_loudness(x, 44100) == pytest.approx(ref, abs=1e-9)
    m = LoudnessMeter(44100)
    for s in range(0, len(x), 12345):
        m.feed(x[s:s + 12345])
    assert m.result() == pytest.approx(ref, abs=1e-9)


def test_limiter_true_peak():
    from partymix.render.master import TruePeakLimiter, true_peak_db
    sr = 44100
    rng = np.random.default_rng(0)
    n = sr * 3
    t = np.arange(n) / sr
    x = np.stack([np.sin(2 * np.pi * 11025.3 * t + 0.7), np.sin(2 * np.pi * 60 * t)], 1)
    x += rng.standard_normal((n, 2)) * 0.3
    x[sr: sr + 40] *= 6
    lim = TruePeakLimiter(sr, -1.0)
    P, ch, out = lim.P, sr, []
    for s in range(0, n, ch):
        a, z = s - P, s + ch + P
        xe = x[max(a, 0):z]
        xe = np.concatenate([np.zeros((max(0, -a), 2)), xe, np.zeros((max(0, z - n), 2))])
        out.append(lim.process(xe, min(ch, n - s)))
    y = np.concatenate(out)
    assert true_peak_db(y) <= -0.99
    quiet = 0.3 * np.stack([np.sin(2 * np.pi * 440 * t)] * 2, 1)
    lim2 = TruePeakLimiter(sr, -1.0)
    y2 = lim2.process(np.concatenate([np.zeros((lim2.P, 2)), quiet, np.zeros((lim2.P, 2))]), n)
    np.testing.assert_allclose(y2, quiet)   # below ceiling: untouched


def test_cue_and_tracklist(tmp_path):
    from partymix.render.master import CueEntry, write_cue, write_tracklist
    cues = [CueEntry(0.0, "Één", "A"), CueEntry(75.52, 'Quote "x"', "")]
    p = write_cue(str(tmp_path / "m.cue"), str(tmp_path / "m.flac"), cues)
    txt = open(p, encoding="utf-8").read()
    assert 'FILE "m.flac" WAVE' in txt and "INDEX 01 01:15:39" in txt and "Één" in txt
    t = open(write_tracklist(str(tmp_path / "m.txt"), cues, 100.0), encoding="utf-8").read()
    assert "01:15" in t and "A - Één" in t
