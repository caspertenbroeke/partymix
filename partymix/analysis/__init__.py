"""Per-track analysis with an on-disk cache.

``analyze_track(path)`` decodes the file, tracks beats/downbeats, detects the key, separates
vocals/instrumental and computes per-bar features, sections, energy and loudness.  Results are
cached in ``paths.cache_dir()/<sha1 of file>/``::

    analysis.json   TrackAnalysis (written last – its presence marks a complete analysis)
    audio.flac      decoded source, 44.1 kHz stereo, 24-bit
    inst.flac       instrumental stem  (inst + voc == audio)
    voc.flac        vocal stem
    meta.json       analysis algorithm version + stem gain

FLAC files may be stored attenuated by a power-of-two headroom gain (decoded MP3s often peak
above 0 dBFS); read them with :func:`partymix.audio_io.read_audio` to get the original scale.

The function only uses module-level lazy singletons (madmom processors, onnxruntime session),
so it is safe to call from a worker process (``multiprocessing`` with ``spawn``).
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Optional, Union

import numpy as np

from .. import paths
from ..audio_io import SR, choose_gain, decode, read_audio, read_title_artist, write_flac
from ..model import SCHEMA_VERSION, TrackAnalysis

ANALYSIS_VERSION = 1          # bump when the algorithms change (invalidates caches)

ProgressCb = Optional[Callable[[float, str], None]]
Cancel = Optional[Union[Callable[[], bool], Any]]     # callable or object with is_set()

__all__ = ["analyze_track", "Cancelled", "file_hash", "cached_analysis", "load_stems", "ANALYSIS_VERSION"]


class Cancelled(InterruptedError):
    """Raised when the ``cancel`` callback/event signals cancellation."""


def file_hash(path: Union[str, os.PathLike]) -> str:
    """SHA-1 of the file contents (cache key)."""
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for buf in iter(lambda: f.read(1 << 20), b""):
            h.update(buf)
    return h.hexdigest()


def _cancel_fn(cancel: Cancel) -> Callable[[], bool]:
    if cancel is None:
        return lambda: False
    if hasattr(cancel, "is_set"):
        return cancel.is_set
    return cancel


def _write_json(p: Path, obj_text: str) -> None:
    tmp = p.with_suffix(p.suffix + ".part")
    tmp.write_text(obj_text, encoding="utf-8")
    os.replace(tmp, p)


def _meta(d: Path) -> dict:
    try:
        return json.loads((d / "meta.json").read_text(encoding="utf-8"))
    except Exception:
        return {}


def cached_analysis(path: Union[str, os.PathLike], fhash: Optional[str] = None) -> Optional[TrackAnalysis]:
    """Return the cached analysis of ``path`` if complete and current, else ``None``."""
    fhash = fhash or file_hash(path)
    d = paths.cache_dir() / fhash
    try:
        ta = TrackAnalysis.from_json((d / "analysis.json").read_text(encoding="utf-8"))
    except Exception:
        return None
    if ta.schema != SCHEMA_VERSION or _meta(d).get("analysis_version") != ANALYSIS_VERSION:
        return None
    if not all((d / n).exists() for n in ("audio.flac", "inst.flac", "voc.flac")):
        return None
    ta.path = os.fspath(path)
    ta.cache_dir = str(d)
    return ta


def load_stems(ta: TrackAnalysis):
    """``(audio, inst, voc, sr)`` float32 ``[n, 2]`` arrays from the cache, at original scale."""
    d = Path(ta.cache_dir)
    a, sr = read_audio(d / "audio.flac")
    i, _ = read_audio(d / "inst.flac")
    v, _ = read_audio(d / "voc.flac")
    return a, i, v, sr


def analyze_track(path: Union[str, os.PathLike], progress_cb: ProgressCb = None,
                  cancel: Cancel = None) -> TrackAnalysis:
    """Analyse one audio file (or return the cached result).

    ``progress_cb(fraction 0..1, message_nl)`` reports progress with Dutch messages.
    ``cancel`` is a callable returning True, or an Event-like object with ``is_set()``;
    on cancellation :class:`Cancelled` is raised and no partial ``analysis.json`` is written.
    """
    from . import beats as _beats, features as _feat, key as _key, vocals as _voc

    path = os.fspath(path)
    name = Path(path).name
    is_cancelled = _cancel_fn(cancel)

    def report(frac: float, msg: str) -> None:
        if is_cancelled():
            raise Cancelled("geannuleerd")
        if progress_cb:
            progress_cb(float(min(max(frac, 0.0), 1.0)), msg)

    report(0.0, f"{name}: bestand controleren…")
    fhash = file_hash(path)
    hit = cached_analysis(path, fhash)
    if hit is not None:
        report(1.0, f"{name}: uit cache geladen")
        return hit

    d = paths.cache_dir() / fhash
    d.mkdir(parents=True, exist_ok=True)
    report(0.01, f"{name}: audio decoderen…")
    audio, sr = decode(path, SR)
    duration = len(audio) / sr
    if duration < 10:
        raise ValueError(f"{name}: te kort om te mixen ({duration:.1f} s)")
    title, artist = read_title_artist(path)

    report(0.04, f"{name}: beats en maatsoort zoeken…")
    br = _beats.track_beats(audio, sr, cancel=is_cancelled,
                            progress=lambda f: report(0.04 + 0.14 * f, f"{name}: beats zoeken…"))

    report(0.18, f"{name}: toonsoort bepalen…")
    camelot, kconf, _ = _key.detect_key(audio, sr)

    # stems: reuse from an interrupted earlier run when complete
    meta = _meta(d)
    stems_ok = (meta.get("analysis_version") == ANALYSIS_VERSION and meta.get("stems_done")
                and all((d / n).exists() for n in ("audio.flac", "inst.flac", "voc.flac")))
    if stems_ok:
        report(0.22, f"{name}: stemmen uit cache laden…")
        inst, _ = read_audio(d / "inst.flac")
        voc, _ = read_audio(d / "voc.flac")
    else:
        def sep_prog(f: float) -> None:
            report(0.22 + 0.63 * f, f"{name}: zang en muziek scheiden… {int(f * 100)}%")
        report(0.22, f"{name}: stemscheidingsmodel laden…")
        _voc.ensure_model(lambda f, m: report(0.22, m))
        inst, voc = _voc.separate(audio, progress=sep_prog, cancel=is_cancelled)
        report(0.85, f"{name}: stems opslaan…")
        g = choose_gain(audio, inst, voc)
        write_flac(d / "audio.flac", audio, sr, gain=g)
        write_flac(d / "inst.flac", inst, sr, gain=g)
        write_flac(d / "voc.flac", voc, sr, gain=g)
        _write_json(d / "meta.json", json.dumps({"analysis_version": ANALYSIS_VERSION, "stems_done": True,
                                                 "gain": g, "created": time.time()}))

    report(0.9, f"{name}: energie en structuur analyseren…")
    bounds = _feat.bar_bounds(br.beats, br.first_downbeat_index, duration)
    bf = _feat.bar_features(audio, sr, bounds)
    vdb = _feat.vocal_db(voc, sr, bounds)
    off = _feat.phrase_offset(bf.bass, vdb, bf.mid)
    secs = _feat.sections(bf.bass, bf.rms, bf.high, off)
    lufs = _feat.loudness_lufs(audio, sr)
    energy = _feat.energy_score(lufs, bf.bass, br.bpm, _feat.onset_rate(audio, sr))

    ta = TrackAnalysis(
        path=path, file_hash=fhash, title=title, artist=artist, duration=duration, sample_rate=sr,
        bpm=float(br.bpm), constant_tempo=bool(br.constant_tempo), beats=[float(b) for b in br.beats],
        first_downbeat_index=int(br.first_downbeat_index), key=camelot, key_confidence=kconf,
        phrase_offset=int(off), bar_bass=bf.bass, bar_mid=bf.mid, bar_high=bf.high, bar_vocal=vdb,
        bar_rms=bf.rms, sections=secs, energy=float(energy), loudness_lufs=float(lufs), cache_dir=str(d))
    report(0.99, f"{name}: opslaan…")
    _write_json(d / "analysis.json", ta.to_json())
    report(1.0, f"{name}: klaar")
    return ta
