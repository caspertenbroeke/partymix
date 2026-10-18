"""`PartyMix --selftest`: end-to-end smoke test for a (frozen) build.

Imports every module, locates the bundled tools and model, analyses two generated
click tracks, plans and renders a short mix. Exit code 0 = everything OK.
"""
from __future__ import annotations
import sys, os, tempfile, time, traceback
import numpy as np


def _click_track(path: str, bpm: float, seconds: float, sr: int = 44100, pitch: float = 220.0) -> None:
    import soundfile as sf
    n = int(seconds * sr)
    t = np.arange(n) / sr
    y = np.zeros(n)
    per = 60.0 / bpm
    k = np.arange(int(0.12 * sr)) / sr
    kick = np.sin(2 * np.pi * 55 * k * (1 + 2 * np.exp(-k * 40))) * np.exp(-k * 18)
    for i, bt in enumerate(np.arange(0.5, seconds - 0.2, per)):
        s = int(bt * sr)
        amp = 1.0 if i % 4 == 0 else 0.6
        e = min(n, s + len(kick))
        y[s:e] += amp * kick[: e - s]
        hs = int((bt + per / 2) * sr)
        if hs + 400 < n:
            y[hs:hs + 400] += 0.15 * np.random.default_rng(i).standard_normal(400) * np.exp(-np.arange(400) / 80)
    y += 0.08 * np.sin(2 * np.pi * pitch * t) * (0.5 + 0.5 * np.sin(2 * np.pi * t / (per * 16)))
    y = 0.5 * y / np.abs(y).max()
    sf.write(path, np.stack([y, y], 1).astype(np.float32), sr, subtype="PCM_24")


def run(verbose: bool = True) -> int:
    log_path = os.environ.get("PARTYMIX_SELFTEST_LOG")
    log = open(log_path, "a", encoding="utf-8") if log_path else None

    def say(m):
        if verbose and sys.stdout is not None:
            try:
                print(m, flush=True)
            except Exception:
                pass
        if log:
            log.write(m + "\n"); log.flush()

    ok = True
    t0 = time.time()
    try:
        say("1/5 modules importeren…")
        import partymix.analysis, partymix.planner, partymix.render, partymix.worker  # noqa
        import madmom, onnxruntime, soundfile, pyloudnorm, scipy  # noqa
        from partymix import paths
        say("2/5 hulpprogramma's zoeken…")
        for name, fn in (("ffmpeg", paths.ffmpeg), ("ffprobe", paths.ffprobe), ("rubberband", paths.rubberband)):
            say(f"   {name}: {fn()}")
        mp = paths.model_path()
        say(f"   model: {mp} ({'aanwezig' if mp.exists() else 'ONTBREEKT'})")
        say(f"   onnxruntime providers: {', '.join(onnxruntime.get_available_providers())}")
        if not mp.exists():
            ok = False
            raise FileNotFoundError("stemscheidingsmodel ontbreekt")
        with tempfile.TemporaryDirectory() as td:
            os.environ.setdefault("PARTYMIX_CACHE", os.path.join(td, "cache"))
            say("3/5 testnummers analyseren…")
            files = [os.path.join(td, "a.wav"), os.path.join(td, "b.wav")]
            _click_track(files[0], 126.0, 50, pitch=220.0)
            _click_track(files[1], 128.0, 50, pitch=330.0)
            from partymix.analysis import analyze_track
            from partymix.analysis.vocals import device_name
            say(f"   stemscheiding op: {device_name()}")
            an = [analyze_track(f) for f in files]
            for a, want in zip(an, (126.0, 128.0)):
                say(f"   {os.path.basename(a.path)}: {a.bpm:.2f} BPM (verwacht {want})")
                if abs(a.bpm - want) > 0.5:
                    ok = False
            say("4/5 mix plannen…")
            from partymix.planner import plan_mix
            plan = plan_mix(an, target_minutes=1.3)
            say("5/5 mix renderen…")
            from partymix.model import RenderSettings
            from partymix.render import render_mix
            out = os.path.join(td, "selftest.flac")
            outs = render_mix(plan, an, RenderSettings(out_path=out, formats=["flac"]), None, None)
            import soundfile as sf
            info = sf.info(outs[0])
            say(f"   resultaat: {info.duration:.1f} s, {info.samplerate} Hz, {info.subtype}")
            if info.duration < 20:
                ok = False
    except Exception:
        ok = False
        tb = traceback.format_exc()
        say(tb)
    say(("SELFTEST OK" if ok else "SELFTEST MISLUKT") + f" ({time.time() - t0:.0f} s)")
    if log:
        log.close()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(run())
