"""Mastering and output: loudness normalisation, true-peak limiter, file writing, cue sheet.

* integrated loudness (BS.1770, streaming) of the pre-master -> gain to ``target_lufs``;
* look-ahead peak limiter in numpy with 4x oversampled true-peak detection
  (attack ~3 ms as a look-ahead ramp, release ~80 ms exponential in dB);
* outputs: FLAC 24-bit / WAV 24-bit via soundfile, MP3 320 kbit/s via ffmpeg
  (with ID3 chapters), plus ``.cue`` sheet and ``.txt`` tracklist.

Everything works in chunks so a 40-minute mix needs only a few hundred MB of RAM.
"""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import soundfile as sf
from scipy.ndimage import maximum_filter1d
from scipy.signal import firwin, upfirdn

from .. import paths
from .common import RenderError, check_cancel, run_tool
from .loudness import LoudnessMeter

__all__ = ["TruePeakLimiter", "true_peak_db", "master_to_files", "write_cue", "write_tracklist",
           "CueEntry", "MasterReport"]

CHUNK_SEC = 10.0
_OS = 4
_TAPS = 49                  # odd -> integer group delay (24 sub-samples = 6 samples)
_FIR = firwin(_TAPS, 1.0 / _OS, window=("kaiser", 8.0)) * _OS
_DELAY = (_TAPS - 1) // 2   # in oversampled samples


def _true_peak_per_sample(x: np.ndarray) -> np.ndarray:
    """``tp[k]`` = max |signal| over [k, k+1) (4x oversampled) and channels; x: [n, ch].
    The first/last 6 values have incomplete filter context."""
    n = len(x)
    tp = np.abs(x).max(axis=1)
    for c in range(x.shape[1]):
        y = upfirdn(_FIR, x[:, c], up=_OS)
        y = np.abs(y[_DELAY:_DELAY + n * _OS])
        if len(y) < n * _OS:
            y = np.concatenate([y, np.zeros(n * _OS - len(y))])
        np.maximum(tp, y.reshape(n, _OS).max(axis=1), out=tp)
    return tp


def true_peak_db(x: np.ndarray) -> float:
    tp = _true_peak_per_sample(np.asarray(x, dtype=np.float64)).max()
    return float(20 * np.log10(max(tp, 1e-12)))


class TruePeakLimiter:
    """Streaming look-ahead true-peak limiter.

    Feed consecutive chunks via :meth:`process`; the caller provides ``P`` samples of
    context on both sides (``process(x_ext, n)``: ``x_ext`` = ``[P | n samples | P]``).
    Guarantee (up to interpolation between gain samples): output true peak <= ceiling.
    """

    def __init__(self, sr: int, ceiling_db: float = -1.0, attack_ms: float = 3.0, release_ms: float = 80.0):
        self.sr = sr
        self.ceiling_db = ceiling_db
        self.L = max(2, int(round(attack_ms / 1000 * sr)))
        self.loga = -1.0 / (release_ms / 1000 * sr)
        self.P = self.L + 16
        self.d_hist = np.zeros(self.L - 1)       # last L-1 released GR values (dB)
        self.max_gr_db = 0.0
        self._first = True

    def _release(self, h: np.ndarray, prev: float) -> np.ndarray:
        """d[k] = max(h[k], d[k-1] * a) (exponential decay of the GR in dB), vectorised via
        log d[k] = k log a + cummax(log h[j] - j log a)."""
        n = len(h)
        if n == 0:
            return np.zeros(0)
        k = np.arange(n, dtype=np.float64)
        with np.errstate(divide="ignore", over="ignore"):
            v = np.log(np.maximum(h, 0.0)) - k * self.loga
            cm = np.maximum.accumulate(v)
            if prev > 0:
                cm = np.maximum(cm, np.log(prev) + self.loga)
            d = np.exp(k * self.loga + cm)
        d[~np.isfinite(d)] = 0.0
        return d

    def process(self, x_ext: np.ndarray, n: int) -> np.ndarray:
        P, L = self.P, self.L
        tp = _true_peak_per_sample(x_ext)
        with np.errstate(divide="ignore"):
            D = np.maximum(0.0, 20 * np.log10(np.maximum(tp, 1e-12)) - self.ceiling_db)
        # h[k] = max D over [k-2, k+L]
        W = L + 3
        h_ext = maximum_filter1d(D, size=W, origin=2 - W // 2, mode="constant", cval=0.0)
        if self._first:  # GR history before the first sample comes from the leading context
            self.d_hist = self._release(h_ext[P - (L - 1):P], 0.0)
            self._first = False
        h = h_ext[P:P + n]
        prev = self.d_hist[-1] if len(self.d_hist) else 0.0
        d = self._release(h, prev)
        # attack: box average over the last L released values (look-ahead covers the window)
        full = np.concatenate([self.d_hist, d])
        cs = np.concatenate([[0.0], np.cumsum(full)])
        g_db = (cs[L:] - cs[:-L]) / L
        self.d_hist = full[-(L - 1):]
        if len(g_db):
            self.max_gr_db = max(self.max_gr_db, float(g_db.max()))
        g = 10 ** (-g_db / 20)
        return x_ext[P:P + n] * g[:, None]


@dataclass
class CueEntry:
    time: float       # seconds in the output
    title: str
    artist: str = ""


@dataclass
class MasterReport:
    input_lufs: float
    gain_db: float
    output_lufs: float
    output_true_peak_db: float
    max_gain_reduction_db: float
    files: List[str]


def _fmt_ts(t: float) -> str:
    t = max(0.0, t)
    return f"{int(t // 3600):d}:{int(t % 3600 // 60):02d}:{int(t % 60):02d}" if t >= 3600 else \
        f"{int(t // 60):02d}:{int(t % 60):02d}"


def write_cue(path: str, audio_file: str, cues: Sequence[CueEntry], title: str = "PartyMix") -> str:
    ext = Path(audio_file).suffix.lower()
    ftype = {".mp3": "MP3", ".wav": "WAVE", ".flac": "WAVE"}.get(ext, "WAVE")

    def q(s):
        return '"' + str(s).replace('"', "'") + '"'
    lines = [f"TITLE {q(title)}", "PERFORMER \"PartyMix\"", f"FILE {q(Path(audio_file).name)} {ftype}"]
    for i, c in enumerate(cues, 1):
        fr = int(round(max(0.0, c.time) * 75))
        lines += [f"  TRACK {i:02d} AUDIO", f"    TITLE {q(c.title)}"]
        if c.artist:
            lines.append(f"    PERFORMER {q(c.artist)}")
        lines.append(f"    INDEX 01 {fr // 4500:02d}:{fr // 75 % 60:02d}:{fr % 75:02d}")
    with open(path, "w", encoding="utf-8", newline="\r\n") as f:
        f.write("\n".join(lines) + "\n")
    return path


def write_tracklist(path: str, cues: Sequence[CueEntry], total_sec: float, title: str = "PartyMix") -> str:
    lines = [f"{title} ({_fmt_ts(total_sec)})", ""]
    for i, c in enumerate(cues, 1):
        name = f"{c.artist} - {c.title}" if c.artist else c.title
        lines.append(f"{i:2d}. {_fmt_ts(c.time)}  {name}")
    with open(path, "w", encoding="utf-8", newline="\r\n") as f:
        f.write("\n".join(lines) + "\n")
    return path


def _ffmetadata(path: str, cues: Sequence[CueEntry], total_sec: float, title: str) -> None:
    def esc(s):
        return "".join("\\" + ch if ch in "=;#\\\n" else ch for ch in str(s))
    out = [";FFMETADATA1", f"title={esc(title)}", "artist=PartyMix"]
    for i, c in enumerate(cues):
        end = cues[i + 1].time if i + 1 < len(cues) else total_sec
        name = f"{c.artist} - {c.title}" if c.artist else c.title
        out += ["[CHAPTER]", "TIMEBASE=1/1000", f"START={int(max(0, c.time) * 1000)}",
                f"END={int(max(c.time, end) * 1000)}", f"title={esc(name)}"]
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(out) + "\n")


def master_to_files(read_chunk: Callable[[int, int], np.ndarray], frames: int, sr: int, out_path: str,
                    formats: Sequence[str], workdir: str, cues: Sequence[CueEntry] = (),
                    target_lufs: float = -9.5, true_peak_db_: float = -1.0, bit_depth: int = 24,
                    title: str = "PartyMix", progress: Optional[Callable[[float, str], None]] = None,
                    cancel_event=None) -> MasterReport:
    """Master the float pre-mix (accessed via ``read_chunk(start, count) -> [count, 2]``) and
    write all requested formats.  Returns a :class:`MasterReport` with the written files
    (audio files first, then ``.cue`` and ``.txt``)."""
    fmts = [f.lower().lstrip(".") for f in formats] or ["flac"]
    for f in fmts:
        if f not in ("flac", "wav", "mp3"):
            raise RenderError(f"Onbekend uitvoerformaat: {f}")
    chunk = int(CHUNK_SEC * sr)
    prog = progress or (lambda f, m: None)

    # 1. loudness of the pre-master
    meter = LoudnessMeter(sr, 2)
    for s in range(0, frames, chunk):
        check_cancel(cancel_event)
        meter.feed(read_chunk(s, min(chunk, frames - s)))
        prog(0.25 * min(1.0, (s + chunk) / frames), "Luidheid meten")
    lin = meter.result()
    gain_db = float(target_lufs - lin) if np.isfinite(lin) else 0.0
    gain = 10 ** (gain_db / 20)

    # 2. limiter -> float32 temp file (re-run once with a lower ceiling if it overshoots)
    lim_path = os.path.join(workdir, "limited.f32")
    ceiling = true_peak_db_ - 0.05
    for attempt in range(3):
        lim = TruePeakLimiter(sr, ceiling)
        P = lim.P
        tp_out = 0.0
        out_meter = LoudnessMeter(sr, 2)
        prev_tail = None
        with open(lim_path, "wb") as fo:
            for s in range(0, frames, chunk):
                check_cancel(cancel_event)
                n = min(chunk, frames - s)
                a, z = s - P, s + n + P
                x = read_chunk(max(a, 0), z - max(a, 0)).astype(np.float64)
                if a < 0:
                    x = np.concatenate([np.zeros((-a, 2)), x])
                if len(x) < z - a:
                    x = np.concatenate([x, np.zeros((z - a - len(x), 2))])
                x *= gain
                y = lim.process(x, n)
                # output true peak, measured with 16 samples of context from the previous chunk
                ctx = np.concatenate([prev_tail, y]) if prev_tail is not None else y
                tp_out = max(tp_out, float(_true_peak_per_sample(ctx)[:-8].max(initial=0.0)))
                prev_tail = y[-16:]
                out_meter.feed(y)
                y.astype(np.float32).tofile(fo)
                prog(0.25 + 0.45 * min(1.0, (s + n) / frames), "Limiter")
        tp_db = 20 * np.log10(max(tp_out, 1e-12))
        if tp_db <= true_peak_db_ + 0.01:
            break
        ceiling -= (tp_db - true_peak_db_) + 0.05
    out_lufs = out_meter.result()

    def read_lim(start, count):
        with open(lim_path, "rb") as f:
            f.seek(start * 8)
            return np.fromfile(f, dtype=np.float32, count=count * 2).reshape(-1, 2)

    # 3. write files
    base = Path(out_path)
    stem = base.with_suffix("")
    written: List[str] = []
    subtype = {16: "PCM_16", 24: "PCM_24", 32: "PCM_32"}.get(int(bit_depth), "PCM_24")
    lossless = [f for f in fmts if f in ("flac", "wav")]
    tmp_wav = None
    if "mp3" in fmts and not lossless:
        tmp_wav = os.path.join(workdir, "for_mp3.wav")
    targets = [(str(stem) + "." + f, f) for f in lossless]
    if tmp_wav:
        targets.append((tmp_wav, "wav"))
    for k, (path, f) in enumerate(targets):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        part = path + ".part"
        with sf.SoundFile(part, "w", samplerate=sr, channels=2, subtype=subtype,
                          format="FLAC" if f == "flac" else "WAV") as fo:
            for s in range(0, frames, chunk):
                check_cancel(cancel_event)
                y = read_lim(s, min(chunk, frames - s))
                fo.write(np.clip(y, -1.0, 1.0 - 2.0 ** -23))
                prog(0.70 + 0.2 * (k + (s + chunk) / frames) / max(1, len(targets)),
                     "Wegschrijven " + f.upper())
        os.replace(part, path)
        if path != tmp_wav:
            written.append(path)
    total_sec = frames / sr
    if "mp3" in fmts:
        check_cancel(cancel_event)
        prog(0.92, "MP3 coderen")
        src = tmp_wav or written[0]
        meta = os.path.join(workdir, "chapters.txt")
        _ffmetadata(meta, cues, total_sec, title)
        mp3 = str(stem) + ".mp3"
        tmp_mp3 = os.path.join(workdir, "out.mp3")
        run_tool([paths.ffmpeg(), "-nostdin", "-v", "error", "-y", "-i", src, "-i", meta,
                  "-map", "0:a", "-map_metadata", "1", "-map_chapters", "1",
                  "-c:a", "libmp3lame", "-b:a", "320k", "-id3v2_version", "3", tmp_mp3],
                 cancel_event=cancel_event, what="ffmpeg (mp3)")
        shutil.move(tmp_mp3, mp3)
        written.append(mp3)
        if tmp_wav:
            try:
                os.remove(tmp_wav)
            except OSError:
                pass
    try:
        os.remove(lim_path)
    except OSError:
        pass
    if cues:
        written.append(write_cue(str(stem) + ".cue", written[0], cues, title))
        written.append(write_tracklist(str(stem) + ".txt", cues, total_sec, title))
    prog(1.0, "Klaar")
    return MasterReport(input_lufs=float(lin), gain_db=gain_db, output_lufs=float(out_lufs),
                        output_true_peak_db=float(tp_db), max_gain_reduction_db=lim.max_gr_db, files=written)
