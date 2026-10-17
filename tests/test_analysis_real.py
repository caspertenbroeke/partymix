"""Real-data regression: beat grid of the 14 tracks of the hand-made "Half uur mix v2".

Compares tempo and first-downbeat phase with ``reference/grid2.json`` (verified correct in the
final mix).  Track 09 (La Mamá) has no constant tempo; there the reference is the smoothed
madmom beat list of ``reference/bars.json`` that the mix actually used.

Skipped when the decoded reference audio (``/home/claude/mix/wav`` or ``$PARTYMIX_REAL_WAV``)
is not available.  Takes ~5 minutes on 2 cores.  Run with ``-s`` to see the table.

Set ``PARTYMIX_REAL_FULL=1`` to additionally run the complete ``analyze_track`` (incl. vocal
separation, ~1.5 min per track) on one track.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from partymix.audio_io import decode
from partymix.analysis import beats as B

WAV = Path(os.environ.get("PARTYMIX_REAL_WAV", "/home/claude/mix/wav"))
REF = Path(__file__).resolve().parent.parent / "reference"
KEYS = ["01", "02", "03", "04", "05", "06", "06m", "07", "08", "09", "10", "11", "12", "13"]

pytestmark = pytest.mark.skipif(not WAV.is_dir(), reason=f"real test audio not found in {WAV}")


@pytest.fixture(scope="module")
def refs():
    return (json.loads((REF / "grid2.json").read_text()), json.loads((REF / "bars.json").read_text()))


_rows: list = []


@pytest.mark.parametrize("k", KEYS)
def test_real_track_grid(k, refs):
    grid, bars = refs
    g = grid[k]
    x, sr = decode(WAV / f"{k}.wav")
    r = B.track_beats(x, sr)
    if k == "09":
        ref_beats = np.array(bars[k]["beats"])
        ref_first = float(ref_beats[bars[k]["first"]])
        ref_downs = ref_beats[bars[k]["first"]::4]
        err = float(np.min(np.abs(ref_downs - r.first_downbeat)))
        # our bar 0 may be an earlier (extrapolated) downbeat: compare with the reference bar grid
        if r.first_downbeat < ref_first - 0.1:
            per = float(np.median(np.diff(ref_beats[:9])))
            err = abs(((ref_first - r.first_downbeat) / (4 * per)) % 1.0)
            err = min(err, 1 - err) * 4 * per
        expect_constant = False
    else:
        bar_len = 4 * 60.0 / g["bpm"]
        d = (r.first_downbeat - g["first_bar"]) % bar_len
        err = min(d, bar_len - d)
        expect_constant = True
    _rows.append((k, g["name"], g["bpm"], r.bpm, g["first_bar"], r.first_downbeat, err * 1000,
                  r.constant_tempo, r.inlier_ratio))
    print(f"\n{k:>3} {g['name'][:16]:16} ref {g['bpm']:8.3f} got {r.bpm:8.3f}  "
          f"first ref {g['first_bar']:6.3f} got {r.first_downbeat:6.3f}  phase err {err * 1000:5.1f} ms  "
          f"const={r.constant_tempo} inliers={r.inlier_ratio:.2f}")
    assert abs(r.bpm - g["bpm"]) < 0.05
    assert err < 0.030
    assert r.constant_tempo == expect_constant


@pytest.mark.skipif(os.environ.get("PARTYMIX_REAL_FULL") != "1", reason="set PARTYMIX_REAL_FULL=1")
def test_real_full_analysis(tmp_path, monkeypatch, refs):
    monkeypatch.setenv("PARTYMIX_CACHE", str(tmp_path))
    from partymix.analysis import analyze_track
    from partymix.model import TrackAnalysis
    ta = analyze_track(WAV / "07.wav")
    assert abs(ta.bpm - refs[0]["07"]["bpm"]) < 0.05
    assert len(ta.bar_bass) == len(ta.bar_vocal) == ta.n_bars
    assert ta.key and 1 <= ta.energy <= 10
    for n in ("analysis.json", "audio.flac", "inst.flac", "voc.flac"):
        assert (Path(ta.cache_dir) / n).exists()
    again = analyze_track(WAV / "07.wav")
    assert TrackAnalysis.to_json(again) == TrackAnalysis.to_json(ta)
