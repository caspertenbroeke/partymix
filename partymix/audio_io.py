"""Audio decoding / encoding helpers built on the bundled ffmpeg.

* :func:`decode` – any format ffmpeg understands -> float32 ``[n, 2]`` at 44.1 kHz.
* :func:`read_title_artist` – tags via ffprobe, falling back to "Artist - Title" file names.
* :func:`write_flac` / :func:`read_audio` – lossless 24-bit FLAC cache files (with headroom gain).

All subprocess calls pass arguments as a list (no shell), so unicode and special characters
in file names (``La Mamá de la Mamá.mp3``, ``Ke$ha - …``) are safe on every platform.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np

from . import paths

SR = 44100
PathLike = Union[str, os.PathLike]

__all__ = ["SR", "decode", "read_title_artist", "write_flac", "read_audio", "probe_duration",
           "choose_gain"]


def _run(cmd: list, **kw) -> subprocess.CompletedProcess:
    """subprocess.run without a console window on Windows."""
    if os.name == "nt":
        kw.setdefault("creationflags", 0x08000000)  # CREATE_NO_WINDOW
    return subprocess.run(cmd, stdin=subprocess.DEVNULL, **kw)


def decode(path: PathLike, sr: int = SR) -> Tuple[np.ndarray, int]:
    """Decode the first audio stream of ``path`` to float32 stereo.

    Mono files are duplicated to two channels, multichannel files are downmixed by ffmpeg.
    Embedded cover art / video streams are ignored.

    Returns ``(data, sr)`` with ``data.shape == (n_samples, 2)``.
    """
    path = os.fspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    cmd = [paths.ffmpeg(), "-nostdin", "-v", "error", "-i", path, "-map", "0:a:0", "-vn",
           "-ac", "2", "-ar", str(sr), "-f", "f32le", "-acodec", "pcm_f32le", "pipe:1"]
    p = _run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0 or not p.stdout:
        err = p.stderr.decode("utf-8", "replace").strip()
        raise RuntimeError(f"Kan audio niet decoderen: {Path(path).name}: {err[-500:]}")
    data = np.frombuffer(p.stdout, dtype="<f4")
    data = data[: len(data) // 2 * 2].reshape(-1, 2)
    return np.ascontiguousarray(data, dtype=np.float32), sr


def probe_duration(path: PathLike) -> float:
    """Container duration in seconds (0.0 if unknown)."""
    cmd = [paths.ffprobe(), "-v", "error", "-show_entries", "format=duration", "-of", "json", os.fspath(path)]
    try:
        p = _run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return float(json.loads(p.stdout.decode("utf-8", "replace"))["format"]["duration"])
    except Exception:
        return 0.0


_JUNK = re.compile(
    r"\s*[\(\[](?:official\s*(?:music\s*)?(?:video|audio|lyric[s]?\s*video|visualizer)|lyrics?|"
    r"audio|video|hq|hd|4k|clip\s*officiel|videoclip)[\)\]]\s*",
    re.IGNORECASE)


def _clean_title(t: str) -> str:
    t = _JUNK.sub(" ", t)
    return re.sub(r"\s{2,}", " ", t).strip()


def _tags(path: str) -> dict:
    try:
        cmd = [paths.ffprobe(), "-v", "error", "-show_entries", "format_tags:stream=codec_type:stream_tags",
               "-of", "json", path]
        p = _run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        info = json.loads(p.stdout.decode("utf-8", "replace") or "{}")
    except Exception:
        return {}
    tags = {}
    # audio stream tags first (ogg/opus keep them there); never use cover-art stream tags
    for s in info.get("streams", []):
        if s.get("codec_type") == "audio":
            tags.update({k.lower(): v for k, v in (s.get("tags") or {}).items()})
    tags.update({k.lower(): v for k, v in (info.get("format", {}).get("tags") or {}).items()})
    return tags


def read_title_artist(path: PathLike) -> Tuple[str, str]:
    """Return ``(title, artist)``; artist may be ``""``.

    Uses the file's title/artist tags; otherwise parses the file name as "Artist - Title".
    Common YouTube junk such as "(Official Video)" is removed from the title.
    """
    path = os.fspath(path)
    tags = _tags(path)
    title = str(tags.get("title", "")).strip()
    artist = str(tags.get("artist", tags.get("album_artist", ""))).strip()
    stem = Path(path).stem
    f_artist, f_title = "", stem
    if " - " in stem:
        f_artist, f_title = (x.strip() for x in stem.split(" - ", 1))
    if not title or re.search(r"[\\/]|\.(jpe?g|png)$", title, re.IGNORECASE):
        title = f_title
        artist = artist or f_artist
    return _clean_title(title) or stem, artist


def choose_gain(*arrays: np.ndarray) -> float:
    """Power-of-two attenuation (1, 0.5, 0.25 …) that keeps all ``arrays`` below full scale."""
    peak = max((float(np.abs(a).max()) for a in arrays if a.size), default=0.0)
    g = 1.0
    while peak * g >= 1.0 and g > 2.0 ** -8:
        g *= 0.5
    return g


def write_flac(path: PathLike, data: np.ndarray, sr: int = SR, gain: Optional[float] = None) -> float:
    """Write float audio ``[n, ch]`` (or ``[n]``) as lossless 24-bit FLAC.

    FLAC is an integer format, so decoded MP3s (which often peak at 1.2–1.5) would clip.
    The data is therefore stored multiplied by a power-of-two ``gain`` (chosen automatically
    when ``None``; exact in 24 bit) and the gain is recorded in the file's comment tag as
    ``partymix_gain=<1/gain>``.  :func:`read_audio` undoes it; a plain ``soundfile.read``
    sees the attenuated signal.  Returns the gain used.  Written atomically.
    """
    import soundfile as sf
    path = os.fspath(path)
    x = np.asarray(data, dtype=np.float32)
    g = choose_gain(x) if gain is None else float(gain)
    tmp = path + ".part"
    y = np.clip(x * np.float32(g), -1.0, 1.0 - 2.0 ** -23)
    with sf.SoundFile(tmp, "w", samplerate=sr, channels=1 if y.ndim == 1 else y.shape[1],
                      format="FLAC", subtype="PCM_24") as f:
        f.comment = f"partymix_gain={1.0 / g:g}"
        f.write(y)
    os.replace(tmp, path)
    return g


def read_audio(path: PathLike) -> Tuple[np.ndarray, int]:
    """Read a (cache) audio file as float32 ``[n, ch]``, undoing the :func:`write_flac` gain."""
    import soundfile as sf
    with sf.SoundFile(os.fspath(path)) as f:
        sr = f.samplerate
        m = re.search(r"partymix_gain=([0-9.eE+-]+)", f.comment or "")
        x = f.read(dtype="float32", always_2d=True)
    if m:
        x *= np.float32(float(m.group(1)))
    return x, sr
