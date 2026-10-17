"""ITU-R BS.1770-4 integrated loudness, numerically identical to ``pyloudnorm.Meter(sr)``
(RBJ K-weighting biquads, 400 ms blocks, 75 % overlap, -70 LUFS absolute / -10 LU relative
gate), plus a streaming variant for long files that do not fit in RAM.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import lfilter

__all__ = ["k_weighting", "integrated_loudness", "LoudnessMeter", "loudness_of_loud_part"]


def _biquad(G, Q, fc, rate, kind):
    A = 10 ** (G / 40.0)
    w0 = 2.0 * np.pi * (fc / rate)
    alpha = np.sin(w0) / (2.0 * Q)
    c = np.cos(w0)
    if kind == "high_shelf":
        b0 = A * ((A + 1) + (A - 1) * c + 2 * np.sqrt(A) * alpha)
        b1 = -2 * A * ((A - 1) + (A + 1) * c)
        b2 = A * ((A + 1) + (A - 1) * c - 2 * np.sqrt(A) * alpha)
        a0 = (A + 1) - (A - 1) * c + 2 * np.sqrt(A) * alpha
        a1 = 2 * ((A - 1) - (A + 1) * c)
        a2 = (A + 1) - (A - 1) * c - 2 * np.sqrt(A) * alpha
    else:  # high_pass
        b0 = (1 + c) / 2
        b1 = -(1 + c)
        b2 = (1 + c) / 2
        a0 = 1 + alpha
        a1 = -2 * c
        a2 = 1 - alpha
    return np.array([b0, b1, b2]) / a0, np.array([a0, a1, a2]) / a0


def k_weighting(rate: int):
    """The two K-weighting stages ``[(b, a), (b, a)]`` exactly as pyloudnorm builds them."""
    return [_biquad(4.0, 1 / np.sqrt(2), 1500.0, rate, "high_shelf"),
            _biquad(0.0, 0.5, 38.0, rate, "high_pass")]


def _block_bounds(num_samples: int, rate: int, block=0.4, overlap=0.75):
    step = 1.0 - overlap
    T = num_samples / rate
    nb = int(np.round(((T - block) / (block * step)))) + 1
    j = np.arange(nb)
    lo = np.array([int(block * (jj * step) * rate) for jj in j], dtype=np.int64)
    hi = np.array([int(block * (jj * step + 1) * rate) for jj in j], dtype=np.int64)
    return lo, np.minimum(hi, num_samples)


def _gate(z: np.ndarray) -> float:
    """z: (channels, blocks) mean squares -> gated loudness (channel gains 1.0 for L/R)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        lj = -0.691 + 10.0 * np.log10(z.sum(0))
        Jg = lj >= -70.0
        if not Jg.any():
            return float("-inf")
        Gr = -0.691 + 10.0 * np.log10(z[:, Jg].mean(1).sum()) - 10.0
        Jg = (lj > Gr) & (lj > -70.0)
        if not Jg.any():
            return float("-inf")
        return float(-0.691 + 10.0 * np.log10(np.nan_to_num(z[:, Jg].mean(1)).sum()))


def integrated_loudness(x: np.ndarray, rate: int) -> float:
    """Integrated loudness (LUFS) of ``x`` ``[n]`` or ``[n, ch]`` (ch <= 2)."""
    x = np.asarray(x)
    if x.ndim == 1:
        x = x[:, None]
    if len(x) < 0.4 * rate:
        raise ValueError("Audio moet langer zijn dan 0,4 s voor een loudness-meting")
    y = x.astype(np.float64, copy=True)
    for b, a in k_weighting(rate):
        y = lfilter(b, a, y, axis=0)
    if x.dtype != np.float64:          # pyloudnorm stores the filtered signal in the input dtype
        y = y.astype(x.dtype).astype(np.float64)
    lo, hi = _block_bounds(len(y), rate)
    c = np.concatenate([np.zeros((1, y.shape[1])), np.cumsum(y * y, axis=0)])
    z = ((c[hi] - c[lo]) / (0.4 * rate)).T
    return _gate(z)


class LoudnessMeter:
    """Streaming integrated loudness: ``feed()`` consecutive chunks, then ``result()``.

    Gives the same value as :func:`integrated_loudness` on the concatenated signal.
    """

    def __init__(self, rate: int, channels: int = 2):
        self.rate = rate
        self.ch = channels
        self.filt = k_weighting(rate)
        self.zi = [np.zeros((2, channels)) for _ in self.filt]
        self.total = 0
        self.acc = np.zeros(channels)
        self.cum = {0: np.zeros(channels)}     # sample index -> cumulative K-weighted energy
        self._j = [0, 0]                         # next block index for lower / upper bounds

    def _bound(self, which: int, j: int) -> int:
        return int(0.4 * (j * 0.25 + which) * self.rate)

    def feed(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x[:, None]
        if len(x) == 0:
            return
        y = x
        for k, (b, a) in enumerate(self.filt):
            y, self.zi[k] = lfilter(b, a, y, axis=0, zi=self.zi[k])
        c = self.acc + np.cumsum(y * y, axis=0)
        n0, n1 = self.total, self.total + len(y)
        for w in (0, 1):
            while True:
                idx = self._bound(w, self._j[w])
                if idx > n1:
                    break
                if idx > n0:
                    self.cum[idx] = c[idx - n0 - 1].copy()
                self._j[w] += 1
        self.acc = c[-1].copy()
        self.total = n1

    def result(self) -> float:
        if self.total < 0.4 * self.rate:
            raise ValueError("Audio moet langer zijn dan 0,4 s voor een loudness-meting")
        lo, hi = _block_bounds(self.total, self.rate)
        z = np.array([(self._cum_at(h) - self._cum_at(l)) / (0.4 * self.rate) for l, h in zip(lo, hi)]).T
        return _gate(z)

    def _cum_at(self, idx: int) -> np.ndarray:
        # block upper bounds are clipped to the signal length (not a 100 ms boundary)
        if idx >= self.total:
            return self.acc
        c = self.cum.get(int(idx))
        if c is None:
            raise RuntimeError(f"loudness: blokgrens {idx} ontbreekt (intern)")
        return c


def loudness_of_loud_part(x: np.ndarray, rate: int, block_sec: float = 3.0) -> float:
    """Loudness of the loudest 2/3 of 3-s blocks (as ``reference/mix2.py`` ``lufs()``)."""
    blk = int(rate * block_sec)
    segs = [x[i:i + blk] for i in range(0, len(x) - blk, blk)]
    if len(segs) < 2:
        return integrated_loudness(x, rate)
    rms = np.array([np.sqrt((s ** 2).mean()) for s in segs])
    keep = np.concatenate([segs[i] for i in np.argsort(rms)[len(rms) // 3:]])
    return integrated_loudness(keep, rate)
