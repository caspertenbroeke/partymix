"""Small end-to-end render with synthetic stems (needs the rubberband CLI; skipped otherwise)."""
import shutil
import threading

import numpy as np
import pytest
import soundfile as sf

from partymix.model import MixPlan, PlannedTrack, Transition, TrackAnalysis, RenderSettings
from partymix import render

pytestmark = pytest.mark.skipif(shutil.which("rubberband") is None, reason="rubberband CLI not found")
SR = 44100


def _track(tmp_path, key, bpm, first, bars=40, freq=200.0):
    per = 60 / bpm
    n = int((first + (bars * 4 + 2) * per) * SR)
    t = np.arange(n) / SR
    beats = first + per * np.arange(bars * 4 + 2)
    inst = 0.05 * np.sin(2 * np.pi * 55 * t)
    for b in beats:  # kick-ish click on every beat
        i = int(round(b * SR))
        if 0 <= i < n:
            seg = np.exp(-np.arange(min(2000, n - i)) / 300.0)
            inst[i:i + len(seg)] += 0.6 * seg * np.sin(2 * np.pi * freq * np.arange(len(seg)) / SR)
    voc = 0.1 * np.sin(2 * np.pi * 880 * t) * (np.sin(2 * np.pi * 0.5 * t) > 0)
    d = tmp_path / key
    d.mkdir()
    sf.write(d / "inst.flac", np.stack([inst, inst], 1), SR, subtype="PCM_24")
    sf.write(d / "voc.flac", np.stack([voc, voc], 1), SR, subtype="PCM_24")
    return TrackAnalysis(path=key + ".mp3", file_hash=key, title=f"Track {key}", artist="Test", duration=n / SR,
                         sample_rate=SR, bpm=bpm, constant_tempo=True, beats=[float(b) for b in beats],
                         first_downbeat_index=0, key="8A", key_confidence=1, phrase_offset=0, cache_dir=str(d))


def _plan():
    return MixPlan(tracks=[PlannedTrack("a", 0, 24), PlannedTrack("b", 0, 24)],
                   transitions=[Transition(8, 123.0, "editor", {"swap": 0.5, "fadein": 0.25, "lpin": True})],
                   start_tempo=121.0, end_tempo=124.0, end_fade_bars=4)


def test_render_small_mix(tmp_path):
    an = [_track(tmp_path, "a", 120.0, -0.01), _track(tmp_path, "b", 125.0, 0.3, freq=300.0)]
    plan = _plan()
    out = tmp_path / "out" / "Mix één.flac"
    msgs = []
    files = render.render_mix(plan, an[::-1], RenderSettings(out_path=str(out), formats=["flac", "wav"]),
                              lambda f, m: msgs.append((f, m)), workdir=str(tmp_path / "work"))
    assert str(out) in files and str(out.with_suffix(".wav")) in files
    assert any(f.endswith(".cue") for f in files) and any(f.endswith(".txt") for f in files)
    info = sf.info(str(out))
    assert info.subtype == "PCM_24"
    assert info.frames == pytest.approx(render.estimate_duration(plan) * SR, abs=2)
    y = sf.read(str(out))[0]
    from partymix.render.master import true_peak_db
    assert true_peak_db(y) <= -0.95
    from partymix.render.loudness import integrated_loudness
    assert abs(integrated_loudness(y, SR) - RenderSettings(out_path="x").target_lufs) <= 0.3
    assert np.abs(y).max() > 0.1
    assert msgs[-1][0] == 1.0 and all(0 <= f <= 1 for f, _ in msgs)
    assert [f for f, _ in msgs] == sorted(f for f, _ in msgs)
    assert not list((tmp_path / "work" / "stretch").glob("*.wav"))   # temp stems cleaned up


def test_render_cancel(tmp_path):
    an = [_track(tmp_path, "a", 120.0, 0.1), _track(tmp_path, "b", 125.0, 0.3)]
    ev = threading.Event()
    ev.set()
    with pytest.raises(render.RenderCancelled):
        render.render_mix(_plan(), an, RenderSettings(out_path=str(tmp_path / "x.flac")), None, ev)
    assert not (tmp_path / "x.flac").exists()
