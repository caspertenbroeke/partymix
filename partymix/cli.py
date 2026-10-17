"""Command line interface.

    python -m partymix                                   # start the GUI
    python -m partymix all <bestanden|map> -o mix.flac --minutes 30 [--mp3] [--order auto|given]
    python -m partymix analyze <bestanden|map> [-o analyses.json]
    python -m partymix plan <bestanden|map|analyses.json> -o plan.json [--minutes 30] [--order auto|given]
    python -m partymix render plan.json <bestanden|map|analyses.json> -o mix.flac [--mp3]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

AUDIO_EXTS = (".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg", ".opus", ".aiff", ".aif")


# ---------------------------------------------------------------------------------------------
# helpers shared with the GUI (no Qt imports here)

def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def expand_inputs(items: Iterable[str]) -> List[str]:
    """Files and folders (recursive) → list of audio/JSON paths; folder content sorted naturally."""
    out: List[str] = []
    seen = set()
    for it in items:
        p = Path(it)
        if p.is_dir():
            files = sorted((f for f in p.rglob("*") if f.is_file() and f.suffix.lower() in AUDIO_EXTS),
                           key=lambda f: natural_key(str(f.relative_to(p))))
        else:
            files = [p]
        for f in files:
            k = os.path.normcase(str(f.resolve())) if f.exists() else str(f)
            if k not in seen:
                seen.add(k)
                out.append(str(f.resolve()) if f.exists() else str(f))
    return out


def is_audio(path: str) -> bool:
    return Path(path).suffix.lower() in AUDIO_EXTS


def guess_title_artist(path: str) -> Tuple[str, str]:
    """Title/artist from a file name like '03 - Artiest - Titel.mp3' (before analysis has tags)."""
    stem = Path(path).stem.replace("_", " ").strip()
    stem = re.sub(r"^\s*\d{1,3}\s*[-._)]\s*", "", stem)          # track numbers
    parts = [x.strip() for x in re.split(r"\s+[-–—]\s+", stem) if x.strip()]
    if len(parts) >= 2:
        return " - ".join(parts[1:]), parts[0]
    return stem, ""


def clean_step(text: str, path: str) -> str:
    """Drop a leading "<file name>: " from analysis progress messages (the UI shows the name already)."""
    text = (text or "").strip()
    name = Path(path).name
    for prefix in (name + ":", Path(path).stem + ":"):
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
    return text[:1].upper() + text[1:] if text else text


def fmt_time(seconds: Optional[float], hours: bool = False) -> str:
    if seconds is None:
        return "–"
    s = int(round(max(seconds, 0)))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h or hours:
        return f"{h}:{m:02d}:{sec:02d}"
    return f"{m}:{sec:02d}"


def tracklist_entries(plan, analyses, exact: bool = True) -> List[Tuple[float, object]]:
    """Start time (s, in the mix) of each planned track → [(seconds, TrackAnalysis|None)].

    With ``exact`` the render module's own layout is used (the moment each track takes over, the same
    times as in the .cue file). Otherwise / as fallback an approximation: track i plays source bars [start_bar, end_bar); the next track starts when track i reaches
    ``end_bar - overlap_bars``. Tempo of a track's solo part ramps between the transition tempos, so the
    average of both ends is used for the bar length (60 * 4 / bpm)."""
    by_hash = {a.file_hash: a for a in analyses}
    if exact:
        try:
            from partymix.render import compute_layout, order_analyses, track_cues
            an = order_analyses(plan, analyses)
            cues = track_cues(plan, compute_layout(plan, an), an)
            return [(float(c.time), a) for c, a in zip(cues, an)]
        except Exception:
            pass
    tr = plan.transitions
    out = []
    t = 0.0
    for i, pt in enumerate(plan.tracks):
        out.append((t, by_hash.get(pt.track_hash)))
        if i == len(plan.tracks) - 1:
            break
        t_in = tr[i - 1].tempo if i > 0 else plan.start_tempo
        t_out = tr[i].tempo if i < len(tr) else plan.end_tempo
        tempo = max((t_in + t_out) / 2.0, 1.0)
        ov = tr[i].overlap_bars if i < len(tr) else 0
        bars = max(pt.end_bar - pt.start_bar - ov, 0)
        t += bars * 240.0 / tempo
    return out


def tracklist_text(plan, analyses, total_seconds: Optional[float] = None) -> str:
    entries = tracklist_entries(plan, analyses)
    hours = bool(total_seconds and total_seconds >= 3600)
    lines = []
    for i, (t, a) in enumerate(entries, 1):
        if a is None:
            name = "?"
        else:
            name = f"{a.artist} – {a.title}" if a.artist else a.title
        lines.append(f"{fmt_time(t, hours):>{8 if hours else 5}}  {i:>2}. {name}")
    return "\n".join(lines)


def settings_for(out_path: str, mp3: bool = False, wav: bool = False, verify: bool = False):
    from partymix.model import RenderSettings
    ext = Path(out_path).suffix.lower().lstrip(".")
    primary = ext if ext in ("flac", "wav", "mp3") else "flac"
    if ext not in ("flac", "wav", "mp3"):
        out_path = str(Path(out_path).with_suffix(".flac"))
    formats = [primary]
    if wav and "wav" not in formats:
        formats.append("wav")
    if mp3 and "mp3" not in formats:
        formats.append("mp3")
    return RenderSettings(out_path=out_path, formats=formats, verify=verify)


def load_analyses_json(path: str):
    from partymix.model import TrackAnalysis
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("tracks", [data])
    return [TrackAnalysis.from_json(json.dumps(d)) for d in data]


def dump_analyses(analyses) -> str:
    return json.dumps([json.loads(a.to_json()) for a in analyses], indent=1)


# ---------------------------------------------------------------------------------------------
# progress output

class Progress:
    def __init__(self, stream=None):
        self.stream = stream or sys.stderr
        self.tty = hasattr(self.stream, "isatty") and self.stream.isatty()
        self._last = ""
        self._last_t = 0.0

    def show(self, frac: Optional[float], text: str) -> None:
        pct = "    " if frac is None or frac < 0 else f"{int(frac * 100):3d}%"
        line = f"[{pct}] {text}"
        if self.tty:
            self.stream.write("\r" + line[:110].ljust(110))
            self.stream.flush()
        else:
            now = time.monotonic()
            key = f"{text}|{int((frac or 0) * 10)}"
            if key != self._last or now - self._last_t > 30:
                self.stream.write(line + "\n")
                self.stream.flush()
                self._last, self._last_t = key, now

    def done(self, text: str = "") -> None:
        if self.tty:
            self.stream.write("\r" + " " * 110 + "\r")
        if text:
            self.stream.write(text + "\n")
        self.stream.flush()


def _say(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------------------------
# steps

def analyze_files(files: Sequence[str], workers: Optional[int] = None, progress: Optional[Progress] = None):
    """Analyse audio files in worker processes (crash-safe). Returns (analyses_in_input_order, failures)."""
    from partymix.worker import WorkerPool
    progress = progress or Progress()
    pool = WorkerPool(workers, name="analyse")
    results: Dict[int, object] = {}
    failures: List[Tuple[str, str]] = []
    fracs: Dict[int, float] = {}
    try:
        ids = {pool.submit("analyze", f, tag=i): i for i, f in enumerate(files)}
        n = len(files)
        for ev in pool.wait(list(ids)):
            i = ev.tag
            name = Path(files[i]).name
            if ev.kind == "progress":
                fracs[i] = ev.fraction if ev.fraction is not None else fracs.get(i, 0.0)
                overall = (len(results) + len(failures) + sum(fracs.values())) / n
                step = clean_step(ev.text, files[i])
                progress.show(overall, f"Analyseren {len(results) + len(failures) + 1}/{n}: {name}"
                              + (f" – {step}" if step else ""))
            elif ev.kind == "done":
                fracs.pop(i, None)
                results[i] = ev.result
            elif ev.kind in ("error", "cancelled"):
                fracs.pop(i, None)
                failures.append((files[i], ev.error or "Geannuleerd"))
                progress.done(f"FOUT  {name}: {ev.error or 'geannuleerd'}")
    except KeyboardInterrupt:
        pool.cancel_all()
        raise
    finally:
        pool.shutdown()
    progress.done()
    return [results[i] for i in sorted(results)], failures


def gather_analyses(inputs: Sequence[str], workers: Optional[int] = None):
    files = expand_inputs(inputs)
    if not files:
        raise SystemExit("Geen audiobestanden gevonden.")
    analyses, audio = [], []
    order: List[Tuple[str, str]] = []                 # keep the given order across json + audio
    for f in files:
        if f.lower().endswith(".json"):
            order.append(("json", f))
        elif is_audio(f):
            order.append(("audio", f))
            audio.append(f)
        else:
            _say(f"Overgeslagen (geen audio): {f}")
    analysed, failures = analyze_files(audio, workers) if audio else ([], [])
    by_path = {os.path.normcase(os.path.abspath(a.path)): a for a in analysed}
    for kind, f in order:
        if kind == "json":
            analyses.extend(load_analyses_json(f))
        else:
            a = by_path.get(os.path.normcase(os.path.abspath(f)))
            if a is None:                                  # analysis might normalise the path differently
                a = next((x for x in analysed if Path(x.path).name == Path(f).name), None)
            if a is not None and a not in analyses:
                analyses.append(a)
    return analyses, failures


def _pick(spec: Optional[str], analyses) -> Optional[int]:
    """--first/--last: 1-based number in the given order, or part of the file name/title."""
    if not spec:
        return None
    if spec.isdigit():
        i = int(spec) - 1
        if 0 <= i < len(analyses):
            return i
        raise SystemExit(f"--first/--last: nummer {spec} bestaat niet (1..{len(analyses)})")
    s = spec.lower()
    for i, a in enumerate(analyses):
        if s in Path(a.path).name.lower() or s in (a.title or "").lower():
            return i
    raise SystemExit(f"--first/--last: geen nummer gevonden voor '{spec}'")


def order_analyses(analyses, how: str, first: Optional[str] = None, last: Optional[str] = None):
    if how == "given" or len(analyses) < 3:
        return list(analyses)
    from partymix.planner import order_tracks
    idx = order_tracks(analyses, _pick(first, analyses), _pick(last, analyses))
    return [analyses[i] for i in idx]


def make_plan(analyses_in_order, minutes: float):
    from partymix.planner import plan_mix
    return plan_mix(analyses_in_order, minutes)


def render(plan, analyses_in_order, settings, progress: Optional[Progress] = None) -> List[str]:
    from partymix.render import render_mix
    progress = progress or Progress()
    cancel = threading.Event()
    try:
        outputs = render_mix(plan, analyses_in_order, settings,
                             lambda *a, **kw: _render_progress(progress, a, kw), cancel)
    except KeyboardInterrupt:
        cancel.set()
        raise
    progress.done()
    return list(outputs or [])


def _render_progress(progress: Progress, a, kw) -> None:
    from partymix.worker import parse_progress
    frac, text = parse_progress(a, kw)
    progress.show(frac, text or "Mix maken…")


def _estimate(plan, analyses) -> Optional[float]:
    try:
        from partymix.render import estimate_duration
        return float(estimate_duration(plan, analyses))
    except Exception:
        return None


def _print_summary(plan, analyses, outputs=None) -> None:
    total = _estimate(plan, analyses)
    print("Tracklijst:")
    print(tracklist_text(plan, analyses, total))
    if total:
        print(f"Geschatte lengte: {fmt_time(total)}")
    for o in outputs or []:
        print(f"Geschreven: {o}")


# ---------------------------------------------------------------------------------------------
# commands

def cmd_analyze(args) -> int:
    analyses, failures = gather_analyses(args.inputs, args.workers)
    text = dump_analyses(analyses)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
        _say(f"{len(analyses)} analyses geschreven naar {args.output}")
    else:
        for a in analyses:
            print(f"{a.bpm:6.1f} BPM  {a.key or '?':>3}  E{a.energy:4.1f}  {fmt_time(a.duration):>6}  "
                  f"{(a.artist + ' – ') if a.artist else ''}{a.title}")
    return 1 if failures and not analyses else 0


def cmd_plan(args) -> int:
    analyses, _ = gather_analyses(args.inputs, args.workers)
    if len(analyses) < 2:
        raise SystemExit("Minstens 2 geanalyseerde nummers nodig.")
    ordered = order_analyses(analyses, args.order, args.first, args.last)
    plan = make_plan(ordered, args.minutes)
    Path(args.output).write_text(plan.to_json(), encoding="utf-8")
    if args.analyses_out:
        Path(args.analyses_out).write_text(dump_analyses(ordered), encoding="utf-8")
    _print_summary(plan, ordered)
    _say(f"Plan geschreven naar {args.output}")
    return 0


def cmd_render(args) -> int:
    from partymix.model import MixPlan
    plan = MixPlan.from_json(Path(args.plan).read_text(encoding="utf-8"))
    analyses, _ = gather_analyses(args.inputs, args.workers)
    by_hash = {a.file_hash: a for a in analyses}
    missing = [t.track_hash for t in plan.tracks if t.track_hash not in by_hash]
    if missing:
        raise SystemExit(f"{len(missing)} nummer(s) uit het plan ontbreken in de invoer.")
    ordered = []
    for t in plan.tracks:
        if by_hash[t.track_hash] not in ordered:
            ordered.append(by_hash[t.track_hash])
    outputs = render(plan, ordered, settings_for(args.output, args.mp3, args.wav, args.verify))
    _print_summary(plan, ordered, outputs)
    return 0


def cmd_all(args) -> int:
    analyses, failures = gather_analyses(args.inputs, args.workers)
    if failures:
        _say(f"{len(failures)} nummer(s) overgeslagen door fouten.")
    if len(analyses) < 2:
        raise SystemExit("Minstens 2 geanalyseerde nummers nodig.")
    ordered = order_analyses(analyses, args.order, args.first, args.last)
    plan = make_plan(ordered, args.minutes)
    if args.plan_out:
        Path(args.plan_out).write_text(plan.to_json(), encoding="utf-8")
    outputs = render(plan, ordered, settings_for(args.output, args.mp3, args.wav, args.verify))
    _print_summary(plan, ordered, outputs)
    return 0


def cmd_gui(args) -> int:
    from partymix.gui import main as gui_main
    return gui_main(demo=getattr(args, "demo", False), workers=getattr(args, "workers", None))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="partymix", description="PartyMix – maak van losse nummers één doorlopende mix.")
    sub = p.add_subparsers(dest="cmd")

    def common(sp, inputs=True):
        if inputs:
            sp.add_argument("inputs", nargs="+", help="audiobestanden, mappen of analyses.json")
        sp.add_argument("--workers", type=int, default=None, help="aantal parallelle analyses (standaard 1–2)")

    def order_opts(sp):
        sp.add_argument("--minutes", type=float, default=30.0, help="doelduur in minuten (standaard 30)")
        sp.add_argument("--order", choices=("auto", "given"), default="auto",
                        help="auto = slimme volgorde, given = volgorde zoals opgegeven")
        sp.add_argument("--first", help="vast eerste nummer (nummer 1..n of deel van de naam)")
        sp.add_argument("--last", help="vast laatste nummer (nummer 1..n of deel van de naam)")

    def out_opts(sp):
        sp.add_argument("-o", "--output", default="mix.flac", help="uitvoerbestand (.flac of .wav)")
        sp.add_argument("--mp3", action="store_true", help="ook een MP3 320 kbps maken")
        sp.add_argument("--wav", action="store_true", help="ook een WAV 24-bit maken")
        sp.add_argument("--verify", action="store_true", help="resultaat controleren (beats op het grid)")

    sp = sub.add_parser("all", help="analyseren, plannen en renderen in één keer")
    common(sp); order_opts(sp); out_opts(sp)
    sp.add_argument("--plan-out", help="plan ook als JSON opslaan")
    sp.set_defaults(func=cmd_all)

    sp = sub.add_parser("analyze", help="nummers analyseren (BPM, toonsoort, energie…)")
    common(sp)
    sp.add_argument("-o", "--output", help="analyses als JSON opslaan")
    sp.set_defaults(func=cmd_analyze)

    sp = sub.add_parser("plan", help="volgorde en overgangen bepalen → plan.json")
    common(sp); order_opts(sp)
    sp.add_argument("-o", "--output", default="plan.json")
    sp.add_argument("--analyses-out", help="analyses (in mixvolgorde) ook als JSON opslaan")
    sp.set_defaults(func=cmd_plan)

    sp = sub.add_parser("render", help="plan.json renderen naar een audiobestand")
    sp.add_argument("plan", help="plan.json")
    common(sp); out_opts(sp)
    sp.set_defaults(func=cmd_render)

    sp = sub.add_parser("gui", help="grafische interface starten")
    sp.add_argument("--demo", action="store_true", help="met voorbeeldnummers (geen echte analyse)")
    sp.add_argument("--workers", type=int, default=None, help="aantal parallelle analyses (standaard 1–2)")
    sp.set_defaults(func=cmd_gui)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv == ["--demo"]:
        from partymix.gui import main as gui_main
        return gui_main(demo=bool(argv))
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 2
    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        _say("\nGeannuleerd.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
