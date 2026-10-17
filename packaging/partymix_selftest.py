"""PartyMix self-test - proves that a (frozen) build is complete and works end to end.

    PartyMix.exe --selftest [--quick] [--log FILE] [--keep DIR]
    partymix-cli.exe --selftest ...          (same, with console output)
    python packaging/partymix_selftest.py    (development checkout)

Steps (each prints "OK"/"FOUT"; exit code 0 only if all pass):
  1. import every partymix module (and numpy/scipy/madmom/onnxruntime/soundfile/pyloudnorm/PySide6)
  2. locate ffmpeg, ffprobe, rubberband via partymix.paths and run them (-version)
  3. load the UVR ONNX model with onnxruntime (CPU) and run one inference on silence
  4. load madmom's downbeat + key networks (they live in madmom/models inside the bundle) and
     track the beats of a 12 s click track
  5. create a QApplication (checks the Qt platform plugin)
  6. (not with --quick) synthesise two short house-style tracks (128 / 125 BPM), run the full
     CLI pipeline `partymix all a.wav b.wav -o mix.flac --minutes 2` and check the result.

Log goes to stdout (if any), to --log FILE, and to $PARTYMIX_SELFTEST_LOG if set (the windowed
PartyMix.exe has no console). The analysis cache is redirected to a temp dir.
Owned by packaging; uses only the public CLI contract of partymix.cli.
"""
from __future__ import annotations

import argparse
import importlib
import os
import pkgutil
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

_LOG_FILES: list = []


def log(msg: str = "") -> None:
    line = msg + "\n"
    for stream in (sys.stdout,):
        try:
            if stream is not None:
                stream.write(line)
                stream.flush()
        except Exception:  # noqa: BLE001
            pass
    for f in _LOG_FILES:
        f.write(line)
        f.flush()


def _run(cmd: list[str]) -> str:
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=60, creationflags=flags)
    if r.returncode != 0:
        raise RuntimeError(f"{cmd[0]} exit {r.returncode}: {(r.stderr or r.stdout)[-400:]}")
    out = (r.stdout or r.stderr).strip().splitlines()
    return out[0] if out else ""


# ------------------------------------------------------------------ steps
def step_imports() -> str:
    import numpy, scipy, soundfile, pyloudnorm, onnxruntime, madmom, PySide6  # noqa: F401,E401
    import partymix
    bad = []
    n = 0
    for mod in pkgutil.walk_packages(partymix.__path__, "partymix."):
        if ".tests" in mod.name:
            continue
        try:
            importlib.import_module(mod.name)
            n += 1
        except Exception as exc:  # noqa: BLE001
            bad.append(f"{mod.name}: {exc!r}")
    if bad:
        raise RuntimeError("import errors:\n    " + "\n    ".join(bad))
    return (f"{n} partymix modules; partymix {partymix.__version__}, numpy {numpy.__version__}, "
            f"scipy {scipy.__version__}, madmom {madmom.__version__}, onnxruntime {onnxruntime.__version__}, "
            f"PySide6 {PySide6.__version__}, libsndfile {soundfile.__libsndfile_version__}")


def step_tools() -> str:
    from partymix import paths
    parts = []
    for name, fn, flag in (("ffmpeg", paths.ffmpeg, "-version"), ("ffprobe", paths.ffprobe, "-version"),
                           ("rubberband", paths.rubberband, "--version")):
        exe = fn()
        parts.append(f"{name}={exe} [{_run([exe, flag])[:60]}]")
    return "; ".join(parts)


def step_model() -> str:
    import onnxruntime as ort
    from partymix import paths
    p = paths.model_path()
    if not p.exists():
        raise FileNotFoundError(f"model not found: {p}")
    sess = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
    i = sess.get_inputs()[0]
    import numpy as np
    shape = [1 if not isinstance(d, int) else d for d in i.shape]
    out = sess.run(None, {i.name: np.zeros(shape, dtype=np.float32)})[0]
    return f"{p} ({p.stat().st_size // 2**20} MB), input {i.name} {i.shape} -> output {list(out.shape)}"


def step_madmom() -> str:
    import numpy as np
    from madmom.audio.signal import Signal
    from madmom.features.downbeats import DBNDownBeatTrackingProcessor, RNNDownBeatProcessor
    from madmom.features.key import CNNKeyRecognitionProcessor
    from madmom.models import DOWNBEATS_BLSTM, KEY_CNN
    if not DOWNBEATS_BLSTM or not KEY_CNN:
        raise FileNotFoundError("madmom model files missing from the bundle (madmom/models)")
    # 12 s of 120 BPM clicks with an accent on beat 1
    sr = 44100
    y = np.zeros(12 * sr, dtype=np.float32)
    click = (np.sin(2 * np.pi * 1000 * np.arange(800) / sr) * np.hanning(800)).astype(np.float32)
    for i in range(24):
        s0 = int(i * 0.5 * sr)
        y[s0:s0 + 800] += click * (1.0 if i % 4 == 0 else 0.5)
    act = RNNDownBeatProcessor()(Signal(y, sample_rate=sr, num_channels=1))
    beats = DBNDownBeatTrackingProcessor(beats_per_bar=[4], fps=100)(act)
    CNNKeyRecognitionProcessor()
    if len(beats) < 8:
        raise RuntimeError(f"madmom found only {len(beats)} beats in a 12 s click track")
    bpm = 60.0 / float(np.median(np.diff(beats[:, 0])))
    return (f"{len(DOWNBEATS_BLSTM)} downbeat nets, {len(KEY_CNN)} key net(s); click track: "
            f"{len(beats)} beats, {bpm:.1f} BPM (expect 120)")


def step_qt() -> str:
    from PySide6.QtWidgets import QApplication, QLabel
    app = QApplication.instance() or QApplication(["partymix-selftest"])
    w = QLabel("PartyMix selftest")
    w.adjustSize()
    w.deleteLater()
    return f"platform {app.platformName()}"


def synth_track(path: Path, bpm: float, bars: int = 40, root_hz: float = 55.0, seed: int = 0, sr: int = 44100) -> None:
    """A simple 4/4 house loop: kick every beat, clap on 2+4, off-beat hats, bass from bar 8, pad."""
    import numpy as np
    import soundfile as sf
    rng = np.random.default_rng(seed)
    beat = 60.0 / bpm
    n = int(round(bars * 4 * beat * sr)) + sr
    y = np.zeros(n)
    t_k = np.arange(int(0.25 * sr)) / sr
    kick = np.sin(2 * np.pi * (45 * t_k + 60 * (1 - np.exp(-t_k * 30)) / 30)) * np.exp(-t_k * 9)
    t_h = np.arange(int(0.05 * sr)) / sr
    hat = np.diff(rng.standard_normal(len(t_h) + 1)) * np.exp(-t_h * 80) * 0.15
    t_c = np.arange(int(0.15 * sr)) / sr
    clap = rng.standard_normal(len(t_c)) * np.exp(-t_c * 25) * 0.3
    for b in range(bars * 4):
        s = int(round(b * beat * sr))
        y[s:s + len(kick)] += kick * 0.9
        h = int(round((b + 0.5) * beat * sr))
        y[h:h + len(hat)] += hat
        if b % 4 in (1, 3) and b >= 16:
            y[s:s + len(clap)] += clap
    t = np.arange(n) / sr
    # bass (bars 8..bars-4), simple saw-ish with an octave pattern, gated per 8th note
    bass_on = (t >= 8 * 4 * beat) & (t < (bars - 4) * 4 * beat)
    eighth = (np.floor(t / (beat / 2)) % 2 == 1)
    bass = (np.sin(2 * np.pi * root_hz * t) + 0.3 * np.sin(4 * np.pi * root_hz * t)) * 0.35
    y += bass * bass_on * eighth
    # minor-chord pad (mid band) in bars 16..bars-8
    pad_on = (t >= 16 * 4 * beat) & (t < (bars - 8) * 4 * beat)
    pad = sum(np.sin(2 * np.pi * root_hz * 4 * r * t) for r in (1.0, 1.1892, 1.4983)) * 0.06
    y += pad * pad_on
    y /= max(1e-9, np.max(np.abs(y))) / 0.8
    stereo = np.stack([y, np.roll(y, 7)], axis=1).astype(np.float32)
    sf.write(str(path), stereo, sr, subtype="PCM_16")


def step_pipeline(work: Path) -> str:
    import soundfile as sf
    from partymix import cli
    a, b = work / "01 - Selftest - Alpha.wav", work / "02 - Selftest - Beta.wav"
    synth_track(a, 128.0, root_hz=55.0, seed=1)            # A
    synth_track(b, 125.0, root_hz=55.0 * 1.4983, seed=2)   # E (Camelot neighbour)
    out = work / "selftest-mix.flac"
    t0 = time.time()
    rc = cli.main(["all", str(a), str(b), "-o", str(out), "--minutes", "2"])
    if rc not in (0, None):
        raise RuntimeError(f"partymix all returned {rc}")
    if not out.exists():
        raise FileNotFoundError(f"no output written: {out}")
    info = sf.info(str(out))
    if info.duration < 30:
        raise RuntimeError(f"mix too short: {info.duration:.1f} s")
    return f"{out.name}: {info.duration:.1f} s, {info.samplerate} Hz, {info.channels} ch, {info.subtype} ({time.time() - t0:.0f} s)"


# ------------------------------------------------------------------ main
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="PartyMix --selftest")
    ap.add_argument("--quick", action="store_true", help="skip the analysis/render pipeline")
    ap.add_argument("--log", help="also write the log to this file")
    ap.add_argument("--keep", help="work directory to keep (default: temp dir, removed afterwards)")
    args = ap.parse_args(argv)

    for p in filter(None, (args.log, os.environ.get("PARTYMIX_SELFTEST_LOG"))):
        Path(p).parent.mkdir(parents=True, exist_ok=True)
        _LOG_FILES.append(open(p, "w", encoding="utf-8"))

    work = Path(args.keep) if args.keep else Path(tempfile.mkdtemp(prefix="partymix-selftest-"))
    work.mkdir(parents=True, exist_ok=True)
    os.environ["PARTYMIX_CACHE"] = str(work / "cache")
    if sys.platform != "win32" and not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    log(f"PartyMix selftest  python {sys.version.split()[0]}  frozen={getattr(sys, 'frozen', False)}  exe={sys.executable}")
    steps = [("imports", step_imports), ("tools", step_tools), ("model", step_model), ("madmom", step_madmom), ("qt", step_qt)]
    if not args.quick:
        steps.append(("pipeline", lambda: step_pipeline(work)))
    failed = 0
    for name, fn in steps:
        t0 = time.time()
        try:
            detail = fn()
            log(f"[OK]   {name:9s} {time.time() - t0:6.1f}s  {detail}")
        except BaseException as exc:  # noqa: BLE001  (SystemExit from the CLI counts as failure too)
            failed += 1
            log(f"[FOUT] {name:9s} {time.time() - t0:6.1f}s  {exc!r}")
            log(traceback.format_exc())
    log("SELFTEST " + ("GESLAAGD" if not failed else f"MISLUKT ({failed} stap(pen))"))
    for f in _LOG_FILES:
        f.close()
    if not args.keep:
        shutil.rmtree(work, ignore_errors=True)
    return 0 if not failed else 1


if __name__ == "__main__":
    # development checkout: make `partymix` importable
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    sys.exit(main())
