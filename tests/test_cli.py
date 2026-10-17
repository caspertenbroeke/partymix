"""CLI tests with stubbed analysis/planner/render (no audio, no heavy deps)."""
import json
import subprocess
import sys
import types
from pathlib import Path

import pytest

from partymix import cli
from partymix.model import MixPlan, PlannedTrack, Transition, TrackAnalysis

ROOT = Path(__file__).resolve().parent.parent


def fake_analysis(path, bpm=128.0, energy=5.0, title=None):
    import hashlib
    beats = [i * 60 / bpm for i in range(400)]
    return TrackAnalysis(path=str(path), file_hash=hashlib.sha1(str(path).encode()).hexdigest(),
                         title=title or Path(path).stem, artist="Artiest", duration=190.0, sample_rate=44100,
                         bpm=bpm, constant_tempo=True, beats=beats, first_downbeat_index=0, key="8A",
                         key_confidence=0.9, phrase_offset=0, energy=energy)


@pytest.fixture
def stubs(monkeypatch, tmp_path):
    calls = {}

    def analyze_files(files, workers=None, progress=None):
        calls["analyzed"] = list(files)
        res = [fake_analysis(f, 120 + i, energy=10 - i) for i, f in enumerate(files) if "kapot" not in f]
        fails = [(f, "Analyse mislukt: kapot") for f in files if "kapot" in f]
        return res, fails

    planner = types.ModuleType("partymix.planner")

    def order_tracks(analyses, pinned_first=None, pinned_last=None):
        calls["order"] = (len(analyses), pinned_first, pinned_last)
        return sorted(range(len(analyses)), key=lambda i: analyses[i].energy)

    def plan_mix(analyses, target_minutes):
        calls["plan_titles"] = [a.title for a in analyses]
        tracks = [PlannedTrack(a.file_hash, 0, 32) for a in analyses]
        trans = [Transition(8, 128.0, "swap") for _ in analyses[1:]]
        return MixPlan(tracks, trans, 128.0, 128.0, target_minutes=target_minutes)

    planner.order_tracks, planner.plan_mix = order_tracks, plan_mix
    planner.pair_score = lambda a, b: (0.8, "ok")
    render = types.ModuleType("partymix.render")

    def render_mix(plan, analyses, settings, progress_cb, cancel_event):
        calls["render"] = (settings, [a.title for a in analyses])
        progress_cb(0.5, "Mixen")
        return [str(Path(settings.out_path).with_suffix("." + f)) for f in settings.formats]

    render.render_mix = render_mix
    render.estimate_duration = lambda plan, analyses: 600.0
    monkeypatch.setattr(cli, "analyze_files", analyze_files)
    monkeypatch.setitem(sys.modules, "partymix.planner", planner)
    monkeypatch.setitem(sys.modules, "partymix.render", render)
    folder = tmp_path / "muziek"
    (folder / "sub").mkdir(parents=True)
    for name in ["10 - C.mp3", "2 - B.flac", "1 - A.wav", "notes.txt", "sub/3 - D.m4a"]:
        (folder / name).write_bytes(b"")          # names only, never real audio
    return calls, folder


def test_expand_inputs_natural_sort_and_filter(stubs):
    _, folder = stubs
    files = [Path(f).name for f in cli.expand_inputs([str(folder)])]
    assert files == ["1 - A.wav", "2 - B.flac", "10 - C.mp3", "3 - D.m4a"]


def test_guess_title_artist_and_fmt():
    assert cli.guess_title_artist("C:/x/03 - Avicii - Levels.mp3") == ("Levels", "Avicii")
    assert cli.guess_title_artist("Levels.mp3") == ("Levels", "")
    assert cli.fmt_time(65) == "1:05" and cli.fmt_time(3725) == "1:02:05"


def test_tracklist_entries():
    a, b, c = (fake_analysis(f"/{n}.mp3") for n in "abc")
    plan = MixPlan([PlannedTrack(a.file_hash, 0, 64), PlannedTrack(b.file_hash, 8, 40), PlannedTrack(c.file_hash, 0, 32)],
                   [Transition(16, 120.0, "swap"), Transition(0, 120.0, "echo_cut")], 120.0, 120.0)
    ent = cli.tracklist_entries(plan, [a, b, c], exact=False)
    assert [round(t, 3) for t, _ in ent] == [0.0, 48 * 2.0, 48 * 2.0 + 32 * 2.0]
    assert ent[1][1] is b
    lines = cli.tracklist_text(plan, [a, b, c]).splitlines()
    assert len(lines) == 3 and lines[0].startswith(" 0:00   1. Artiest – a")


def test_settings_for():
    s = cli.settings_for("out/mix.flac", mp3=True)
    assert s.formats == ["flac", "mp3"] and s.out_path == "out/mix.flac"
    s = cli.settings_for("mix.wav")
    assert s.formats == ["wav"]
    s = cli.settings_for("mix")
    assert s.out_path.endswith("mix.flac") and s.formats == ["flac"]


def test_all_auto_order(stubs, tmp_path, capsys):
    calls, folder = stubs
    out = tmp_path / "mix.flac"
    rc = cli.main(["all", str(folder), "-o", str(out), "--minutes", "20", "--mp3", "--last", "1",
                   "--plan-out", str(tmp_path / "plan.json")])
    assert rc == 0
    assert calls["order"] == (4, None, 0)
    settings, titles = calls["render"]
    assert settings.formats == ["flac", "mp3"]
    assert titles == calls["plan_titles"]
    assert MixPlan.from_json((tmp_path / "plan.json").read_text()).target_minutes == 20
    text = capsys.readouterr().out
    assert "Tracklijst" in text and "Geschreven:" in text and "10:00" in text


def test_all_given_order(stubs, tmp_path):
    calls, folder = stubs
    cli.main(["all", str(folder), "-o", str(tmp_path / "m.wav"), "--order", "given"])
    assert "order" not in calls
    assert calls["render"][1] == ["1 - A", "2 - B", "10 - C", "3 - D"]


def test_analyze_plan_render_roundtrip(stubs, tmp_path):
    calls, folder = stubs
    aj, pj = tmp_path / "analyses.json", tmp_path / "plan.json"
    assert cli.main(["analyze", str(folder), "-o", str(aj)]) == 0
    data = json.loads(aj.read_text())
    assert len(data) == 4 and data[0]["bpm"] == 120
    calls.clear()
    assert cli.main(["plan", str(aj), "-o", str(pj), "--order", "given"]) == 0
    assert "analyzed" not in calls                       # JSON input: no re-analysis
    assert cli.main(["render", str(pj), str(aj), "-o", str(tmp_path / "x.flac")]) == 0
    assert calls["render"][1] == ["1 - A", "2 - B", "10 - C", "3 - D"]


def test_render_missing_track_fails(stubs, tmp_path):
    calls, folder = stubs
    aj, pj = tmp_path / "a.json", tmp_path / "p.json"
    cli.main(["analyze", str(folder), "-o", str(aj)])
    cli.main(["plan", str(aj), "-o", str(pj)])
    one = tmp_path / "one.json"
    one.write_text(json.dumps(json.loads(aj.read_text())[:1]))
    with pytest.raises(SystemExit, match="ontbreken"):
        cli.main(["render", str(pj), str(one)])


def test_no_args_starts_gui(monkeypatch):
    import partymix.gui as gui
    seen = {}
    monkeypatch.setattr(gui, "main", lambda demo=False, workers=None: seen.setdefault("demo", demo) and 0 or 0)
    assert cli.main([]) == 0 and seen["demo"] is False


def test_module_help_runs():
    r = subprocess.run([sys.executable, "-m", "partymix", "all", "--help"], cwd=ROOT, capture_output=True,
                       text=True, timeout=60)
    assert r.returncode == 0 and "--minutes" in r.stdout
