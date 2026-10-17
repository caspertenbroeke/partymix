"""Per-bar features, phrase offset, sections, energy score and loudness.

All dB values are mean-square power in dBFS (a full-scale sine reads −3 dB), computed per bar
from bar *i* = ``beats[first + 4i] .. beats[first + 4i + 4]`` (see ``reference/bars.py`` and
``reference/vox.py``).  Bands: bass 20–150 Hz, mid 150–3000 Hz, high 3–11 kHz.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..model import Section

N_FFT = 4096          # 93 ms at 44.1 kHz (same resolution as reference: 2048 @ 22.05 kHz)
HOP = 1024
BANDS = ((20.0, 150.0), (150.0, 3000.0), (3000.0, 11000.0))
FLOOR_DB = -100.0
BASS_ON_DB = 6.0      # bass within 6 dB of the track's 90th percentile = "bass on" (maps level ≥ 6)

__all__ = ["BarFeatures", "band_energies", "bar_bounds", "bar_features", "vocal_db",
           "phrase_offset", "sections", "energy_score", "loudness_lufs", "onset_rate"]


@dataclass
class BarFeatures:
    bass: List[float]
    mid: List[float]
    high: List[float]
    rms: List[float]


def _mono(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, np.float32)
    return x.mean(axis=1, dtype=np.float32) if x.ndim == 2 else x


def _db(p: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore"):
        d = 10 * np.log10(np.maximum(p, 1e-12))
    return np.maximum(np.nan_to_num(d, nan=FLOOR_DB), FLOOR_DB)


def band_energies(mono: np.ndarray, sr: int, block_frames: int = 2048) -> Tuple[np.ndarray, np.ndarray]:
    """Band power per STFT frame (blockwise, bounded memory).

    Returns ``(E [3, frames], flux [frames])`` where ``E`` is mean-square power per band
    (Parseval-scaled) and ``flux`` the positive log-spectral flux (onset strength).
    """
    w = np.hanning(N_FFT).astype(np.float32)
    norm = 2.0 / (N_FFT * float((w ** 2).sum()))
    f = np.fft.rfftfreq(N_FFT, 1.0 / sr)
    masks = [(f >= lo) & (f < hi) for lo, hi in BANDS]
    x = np.pad(mono.astype(np.float32), (N_FFT // 2, N_FFT // 2))
    n_frames = max(0, 1 + (len(x) - N_FFT) // HOP)
    E = np.zeros((3, n_frames), np.float32)
    flux = np.zeros(n_frames, np.float32)
    prev = None
    for s in range(0, n_frames, block_frames):
        nf = min(block_frames, n_frames - s)
        seg = x[s * HOP: (s + nf - 1) * HOP + N_FFT]
        fr = np.lib.stride_tricks.sliding_window_view(seg, N_FFT)[::HOP][:nf]
        P = np.abs(np.fft.rfft(fr * w, axis=-1)).astype(np.float32) ** 2
        for b, m in enumerate(masks):
            E[b, s:s + nf] = P[:, m].sum(axis=1) * norm
        L = np.log1p(1000.0 * np.sqrt(P[:, f < 11000]))
        d = np.diff(L, axis=0, prepend=L[:1] if prev is None else prev[None])
        flux[s:s + nf] = np.maximum(d, 0).mean(axis=1)
        prev = L[-1]
    return E, flux


def bar_bounds(beats: Sequence[float], first: int, duration: float) -> np.ndarray:
    """``[n_bars, 2]`` start/end times (s) of every bar from the first downbeat."""
    b = np.asarray(beats, np.float64)
    n_bars = (len(b) - first) // 4
    if n_bars <= 0:
        return np.zeros((0, 2))
    starts = b[first:first + 4 * n_bars:4]
    ends = np.empty(n_bars)
    ends[:-1] = starts[1:]
    j = first + 4 * n_bars
    ends[-1] = b[j] if j < len(b) else min(duration, starts[-1] + 4 * float(np.median(np.diff(b[-9:]))))
    return np.stack([starts, np.minimum(ends, duration)], axis=1)


def bar_features(audio: np.ndarray, sr: int, bounds: np.ndarray) -> BarFeatures:
    mono = _mono(audio)
    E, _ = band_energies(mono, sr)
    fps = sr / HOP
    out = {k: [] for k in ("bass", "mid", "high", "rms")}
    for t0, t1 in bounds:
        a, z = int(max(t0, 0) * fps), max(int(t1 * fps), int(max(t0, 0) * fps) + 1)
        e = E[:, a:z]
        for k, i in (("bass", 0), ("mid", 1), ("high", 2)):
            out[k].append(float(_db(e[i].mean() if e.shape[1] else 0.0)))
        s0, s1 = int(max(t0, 0) * sr), max(int(t1 * sr), int(max(t0, 0) * sr) + 1)
        seg = mono[s0:s1]
        out["rms"].append(float(_db(float(np.mean(seg.astype(np.float64) ** 2)) if len(seg) else 0.0)))
    return BarFeatures(**out)


def vocal_db(vocals: np.ndarray, sr: int, bounds: np.ndarray) -> List[float]:
    """Per-bar vocal level (dBFS mean square) from the vocal stem (``reference/vox.py``)."""
    v = _mono(vocals).astype(np.float64)
    res = []
    for t0, t1 in bounds:
        seg = v[int(max(t0, 0) * sr): max(int(t1 * sr), int(max(t0, 0) * sr) + 1)]
        res.append(float(_db(float(np.mean(seg ** 2)) if len(seg) else 0.0)))
    return res


def onset_rate(audio: np.ndarray, sr: int) -> float:
    """Onsets per second (peak-picked spectral flux)."""
    _, flux = band_energies(_mono(audio), sr)
    if len(flux) < 8:
        return 0.0
    from scipy.ndimage import maximum_filter1d, uniform_filter1d
    thr = uniform_filter1d(flux, 43) * 1.5 + 1e-4          # ~1 s adaptive threshold
    pk = (flux == maximum_filter1d(flux, 3)) & (flux > thr)
    return float(pk.sum() / (len(flux) * HOP / sr))


# ----------------------------------------------------------------------------- structure
def _rel(x: Sequence[float]) -> np.ndarray:
    a = np.asarray(x, np.float64)
    return a - np.percentile(a, 90) if len(a) else a


def phrase_offset(bass: Sequence[float], vocal: Sequence[float], mid: Optional[Sequence[float]] = None) -> int:
    """Bar index (0..7) where 8-bar phrases start.

    Novelty per bar = |mean(next 2 bars) − mean(previous 2 bars)| of bass, vocal and mid levels
    (each clipped to a 24 dB window below its 90th percentile), summed.  The offset whose
    boundaries collect the most novelty wins; offset 0 is preferred unless another offset is
    clearly (> 15 %) better.
    """
    feats = [bass, vocal] + ([mid] if mid is not None else [])
    n = min(len(f) for f in feats)
    if n < 16:
        return 0
    nov = np.zeros(n)
    for f in feats:
        a = np.clip(_rel(f[:n]), -24, 0)
        for i in range(2, n - 1):
            nov[i] += abs(a[i:i + 2].mean() - a[i - 2:i].mean())
    scores = np.array([nov[o::8].mean() if len(nov[o::8]) else 0 for o in range(8)])
    best = int(np.argmax(scores))
    if best != 0 and scores[best] < 1.15 * scores[0]:
        return 0
    return best


def sections(bass: Sequence[float], rms: Sequence[float], high: Sequence[float],
             offset: int = 0, block: int = 4) -> List[Section]:
    """Rough song structure on a grid of ``block`` bars aligned to the phrase offset.

    * drop  – bass on and loud (chorus / drop)
    * verse – bass on but clearly quieter than the drops
    * build – no/low bass, rising level, directly followed by a drop
    * intro / break / outro – bass off before the first, between, after the last drop
    """
    n = min(len(bass), len(rms), len(high))
    if n == 0:
        return []
    B, R, H = _rel(bass[:n]), _rel(rms[:n]), _rel(high[:n])
    edges = sorted({0, n, *range(offset % block, n, block)})
    blocks = [(a, z) for a, z in zip(edges[:-1], edges[1:]) if z > a]
    cls = []
    for a, z in blocks:
        bass_on = np.mean(B[a:z] >= -BASS_ON_DB) >= 0.5
        loud = np.mean(R[a:z]) >= -3.0
        cls.append("drop" if bass_on and loud else "verse" if bass_on else "low")
    drops = [i for i, c in enumerate(cls) if c == "drop"]
    if not drops:                                    # no clear drop: loudest bass blocks count as drops
        drops = [i for i, c in enumerate(cls) if c == "verse"]
        cls = ["drop" if c == "verse" else c for c in cls]
    first_d, last_d = (drops[0], drops[-1]) if drops else (len(cls), -1)
    kinds = []
    for i, ((a, z), c) in enumerate(zip(blocks, cls)):
        if c != "low":
            kinds.append(c)
            continue
        nxt = cls[i + 1] if i + 1 < len(cls) else None
        rising = z - a >= 2 and (np.polyfit(np.arange(z - a), R[a:z] + H[a:z], 1)[0] > 0.4)
        if nxt == "drop" and rising and i > 0:
            kinds.append("build")
        elif i < first_d:
            kinds.append("intro")
        elif i > last_d:
            kinds.append("outro")
        else:
            kinds.append("break")
    out: List[Section] = []
    for (a, z), k in zip(blocks, kinds):
        if out and out[-1].kind == k:
            out[-1].end_bar = z
        else:
            out.append(Section(a, z, k))
    return out


# ----------------------------------------------------------------------------- loudness / energy
def loudness_lufs(audio: np.ndarray, sr: int) -> float:
    """Integrated loudness of the loud 2/3 of the track (3 s blocks, quietest third dropped),
    exactly like ``reference/mix2.py`` ``lufs()``."""
    import pyloudnorm as pyln
    x = np.asarray(audio, np.float64)
    blk = sr * 3
    segs = [x[i:i + blk] for i in range(0, len(x) - blk, blk)]
    if len(segs) < 3:
        keep = x
    else:
        r = np.array([np.sqrt((s ** 2).mean()) for s in segs])
        keep = np.concatenate([segs[i] for i in np.argsort(r)[len(r) // 3:]])
    if len(keep) < int(0.5 * sr):
        return -70.0
    v = pyln.Meter(sr).integrated_loudness(keep)
    return float(v) if np.isfinite(v) else -70.0


def energy_score(lufs: float, bass: Sequence[float], bpm: float, onsets_per_s: float) -> float:
    """1..10 party-energy score.

    Weighted mix of loudness of the loud part (−16..−6 LUFS), bass density (fraction of bars
    with bass on), tempo (100..150 BPM) and onset density (1..6 onsets/s).
    """
    def u(v: float, lo: float, hi: float) -> float:
        return float(np.clip((v - lo) / (hi - lo), 0.0, 1.0))
    B = _rel(bass)
    dens = float(np.mean(B >= -BASS_ON_DB)) if len(B) else 0.0
    s = 0.30 * u(lufs, -16, -6) + 0.30 * dens + 0.20 * u(bpm, 100, 150) + 0.20 * u(onsets_per_s, 1, 6)
    return round(1.0 + 9.0 * s, 1)
