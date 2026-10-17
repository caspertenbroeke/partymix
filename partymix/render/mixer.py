"""Mixing of the stretched stems into one float32 master — faithful port of ``reference/mix2.py``.

Per track (processed one at a time, added into a disk-backed float32 master):

* loudness match: -14 LUFS measured on the loudest 2/3 of 3-s blocks of the played part;
* envelopes at 512-sample block rate (volume, bass, filter, echo send, vocal gain) built from
  the transition presets, smoothed and interpolated to sample rate;
* vocal hand-over: the vocal stem gain is switched at the swap point;
* bass split at 160 Hz (4th-order Butterworth, zero-phase) with its own envelope;
* time-varying LP (f < 0) / HP (f > 0) 2nd-order filter;
* echo send (3/4-beat feedback delay) with a 2-bar tail after the track ends.

Presets (``Transition.preset``) and options (``Transition.options``):
``swap`` (0..1, default 0.5: bass + vocal hand-over point), ``fadein`` (default 0.25),
``lpin`` (low-pass sweep on the incoming track), ``lpdur``, ``voc`` (vocal hand-over, default on).

* ``swap``     – hard bass/vocal swap at ``swap``
* ``swap_hp``  – + gentle high-pass on A in the second half
* ``xfade``    – + A fades out (equal power) after the swap
* ``editor``   – + A low-pass in the 2nd half and echo in the last quarter
* ``echo_out`` – + echo on A in the second half
* ``echo_cut`` – (overlap 0) A echoes out over its last bar, B starts on the 1
* ``double``   – no extra A effect (both tracks run full volume, bass/vocal swap only);
  identical to ``swap``, as in the reference.
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt, sosfiltfilt

from ..model import MixPlan, Transition
from .layout import Layout
from .loudness import loudness_of_loud_part
from .stretch import StretchResult

__all__ = ["PRESETS", "preset_envelopes", "MasterBuffer", "mix_track", "output_frames",
           "TRACK_LUFS", "BLK", "OUTPUT_TAIL_SEC"]

PRESETS = ("xfade", "swap", "swap_hp", "editor", "echo_out", "echo_cut", "double")
TRACK_LUFS = -14.0
BLK = 512
BASS_HZ = 160.0
OUTPUT_TAIL_SEC = 3.0     # master runs this long past the last bar (fade/echo tails)


def _sm(u):  # smoothstep
    u = np.clip(u, 0, 1)
    return u * u * (3 - 2 * u)


def _rp(u, a, b):
    return np.clip((u - a) / (b - a), 0, 1)


def preset_envelopes(name: str, role: str, u: np.ndarray, opt: Dict) -> Tuple[np.ndarray, ...]:
    """Envelopes ``(vol, bass, filt, fx, voc)`` of one side of a transition.

    ``role`` "A" = outgoing, "B" = incoming; ``u`` = position 0..1 inside the overlap.
    """
    u = np.asarray(u, dtype=float)
    one, zero = np.ones_like(u), np.zeros_like(u)
    sw = opt.get("swap", 0.5)
    if role == "A":
        vol = one.copy()
        bass = (u < sw).astype(float) if sw > 0 else zero.copy()
        filt = zero.copy()
        fx = zero.copy()
        voc = (u < sw).astype(float) if opt.get("voc", 1) else one.copy()
        if name == "swap_hp":
            filt = 0.5 * _sm(_rp(u, 0.5, 1))
        if name == "editor":
            filt = -0.6 * _sm(_rp(u, 0.5, 1))
            fx = 0.45 * _rp(u, 0.75, 1)
        if name == "xfade":
            vol = np.cos(_sm(_rp(u, sw, 1)) * np.pi / 2)
        if name == "echo_out":
            fx = 0.6 * _rp(u, 0.5, 1)
        return vol, bass, filt, fx, voc
    fin = opt.get("fadein", 0.25)
    vol = _sm(_rp(u, 0, fin)) if fin > 0 else one.copy()
    bass = (u >= sw).astype(float)
    voc = (u >= sw).astype(float) if opt.get("voc", 1) else one.copy()
    lpd = opt.get("lpdur", max(sw, 0.25) if sw < 1 else 0.5)
    filt = -0.65 * (1 - _sm(_rp(u, 0, lpd))) if opt.get("lpin") else zero.copy()
    return vol, bass, filt, zero.copy(), voc


@lru_cache(maxsize=4096)
def _tv_sos(f: float, sr: int):
    if f < 0:
        fc = 18000 * (300 / 18000) ** (-f)
        return butter(2, fc, "low", fs=sr, output="sos"), -1
    fc = 30 * (900 / 30) ** f
    return butter(2, fc, "high", fs=sr, output="sos"), 1


def _tv_filter(x: np.ndarray, fb: np.ndarray, blk: int, sr: int) -> np.ndarray:
    y = np.empty_like(x)
    zi = None
    mode = 0
    for b in range(len(fb)):
        seg = x[b * blk:(b + 1) * blk]
        if len(seg) == 0:
            break
        f = float(fb[b])
        if abs(f) < 0.01:
            y[b * blk:b * blk + len(seg)] = seg
            zi = None
            mode = 0
            continue
        sos, m = _tv_sos(f, sr)
        if zi is None or m != mode:
            zi = np.zeros((1, 2, x.shape[1]))
        out, zi = sosfilt(sos, seg, axis=0, zi=zi)
        y[b * blk:b * blk + len(seg)] = out
        mode = m
    return y


def _echo(x: np.ndarray, send: np.ndarray, D: int, fb: float = 0.5) -> np.ndarray:
    inp = x * send[:, None]
    y = np.zeros_like(x)
    for s in range(0, len(x), D):
        e = min(s + D, len(x))
        y[s:e] = inp[s:e] + (fb * y[s - D:e - D] if s >= D else 0)
    return y


def output_frames(layout: Layout, sr: int) -> int:
    return int((layout.total_sec + OUTPUT_TAIL_SEC) * sr)


class MasterBuffer:
    """Float32 stereo master.

    With ``path`` the master lives in a raw float32 file on disk (``frames * 2 * 4`` bytes) and
    ``add()`` does read-modify-write of only the touched region, so RAM use does not grow with
    the mix length.  Without ``path`` it is an in-memory array (tests).
    """

    def __init__(self, frames: int, path: Optional[str] = None, channels: int = 2):
        self.frames = frames
        self.channels = channels
        self.path = path
        self.data = None
        if path:
            with open(path, "wb") as f:
                f.truncate(frames * channels * 4)
        else:
            self.data = np.zeros((frames, channels), np.float32)

    def add(self, start: int, seg: np.ndarray) -> None:
        a = max(0, -start)
        seg = seg[a: max(self.frames - start, 0)]
        start += a
        if not len(seg):
            return
        if self.data is not None:
            self.data[start: start + len(seg)] += seg.astype(np.float32)
            return
        fs = self.channels * 4
        with open(self.path, "r+b") as f:
            f.seek(start * fs)
            cur = np.fromfile(f, dtype=np.float32, count=len(seg) * self.channels).reshape(-1, self.channels)
            cur += seg.astype(np.float32)
            f.seek(start * fs)
            cur.tofile(f)

    def read(self, start: int, count: int) -> np.ndarray:
        count = max(0, min(count, self.frames - start))
        if self.data is not None:
            return np.array(self.data[start:start + count])
        with open(self.path, "rb") as f:
            f.seek(start * self.channels * 4)
            return np.fromfile(f, dtype=np.float32, count=count * self.channels).reshape(-1, self.channels)

    def chunks(self, size: int):
        for s in range(0, self.frames, size):
            yield s, self.read(s, size)

    def remove(self) -> None:
        if self.path:
            try:
                os.remove(self.path)
            except OSError:
                pass


def _read_stem32(path: str) -> np.ndarray:
    x = sf.read(path, dtype="float32", always_2d=True)[0]
    if x.shape[1] == 1:
        x = np.repeat(x, 2, axis=1)
    return np.ascontiguousarray(x[:, :2])


def mix_track(i: int, stretched: StretchResult, plan: MixPlan, layout: Layout, master: MasterBuffer,
              sr: int = 44100, track_lufs: float = TRACK_LUFS) -> float:
    """Process track ``i`` and add it into ``master``. Returns the output start time (s)."""
    n = layout.n
    L = layout.lengths[i]
    trs: List[Transition] = plan.transitions
    xi32 = _read_stem32(stretched.inst_path)
    xv32 = _read_stem32(stretched.voc_path)
    m_ = min(len(xi32), len(xv32))
    pre = stretched.pre
    O = layout.offsets[i]
    t0 = layout.t(O)

    def loc(bar: float) -> float:  # local file seconds of local bar
        return layout.t(O + bar) - t0 + pre

    end_s = loc(L)
    N = min(m_, int(loc(L + 3) * sr))
    xi32, xv32 = xi32[:N], xv32[:N]
    ba, bz = max(0, int(loc(0) * sr)), int(end_s * sr)
    body = xi32[ba:bz].astype(np.float64) + xv32[ba:bz].astype(np.float64)
    lu = loudness_of_loud_part(body, sr)
    del body
    gain = 10 ** ((track_lufs - lu) / 20) if np.isfinite(lu) else 1.0

    nb = N // BLK + 1
    tb = (np.arange(nb) + 0.5) * BLK / sr
    vol = np.ones(nb); bass = np.ones(nb); filt = np.zeros(nb); fx = np.zeros(nb); vg = np.ones(nb)
    vol[(tb < loc(0)) | (tb >= end_s)] = 0

    def apply(ta, tz, name, role, opt):
        m = (tb >= ta) & (tb < tz)
        u = (tb[m] - ta) / (tz - ta)
        v, b_, f_, s_, g_ = preset_envelopes(name, role, u, opt)
        vol[m] *= v; bass[m] *= b_; filt[m] += f_; fx[m] = np.maximum(fx[m], s_); vg[m] *= g_

    if i > 0:
        t = trs[i - 1]
        if t.overlap_bars > 0:
            apply(loc(0), loc(t.overlap_bars), t.preset, "B", t.options)
    if i < n - 1:
        t = trs[i]
        if t.overlap_bars > 0:
            apply(loc(L - t.overlap_bars), end_s, t.preset, "A", t.options)
        elif t.preset == "echo_cut":
            a2 = loc(L - 1)
            m = (tb >= a2) & (tb < end_s)
            u = (tb[m] - a2) / (end_s - a2)
            fx[m] = 1.0
            vol[m] *= 1 - _sm(u)
            bass[m] *= 1 - _sm(_rp(u, 0, 0.5))
    else:
        fb_ = plan.end_fade_bars
        if fb_ > 0:
            a2 = loc(L - fb_)
            m = (tb >= a2) & (tb < end_s)
            vol[m] *= 1 - _sm((tb[m] - a2) / (end_s - a2))

    def up(env, ms=10):
        k = max(1, int(ms / 1000 * sr / BLK))
        env = np.convolve(np.pad(env, (k, k), mode="edge"), np.ones(2 * k + 1) / (2 * k + 1), "valid")
        return np.interp(np.arange(N), (np.arange(len(env)) + 0.5) * BLK, env)

    lp = butter(4, BASS_HZ, "low", fs=sr, output="sos")
    vg_u, bass_u, vol_u = up(vg, 60), up(bass, 12) - 1.0, up(vol, 6)
    do_filt = np.abs(filt).max() > 0.01
    send = tail = None
    D = 0
    if fx.max() > 0:
        D = int(0.75 * 60 / layout.echo_tempo(i) * sr)
        send = up(fx, 20) * up(((tb >= loc(0)) & (tb < end_s)).astype(float), 5)
        tail = 0.55 * np.clip(1 - (np.arange(N) / sr - end_s) / (loc(L + 2) - end_s), 0, 1)
    out32 = np.empty((N, 2), np.float32)
    for c in range(2):  # every step is per channel: halves the float64 working set
        xi = xi32[:, c].astype(np.float64)
        xi *= gain
        xv = xv32[:, c].astype(np.float64)
        xv *= gain
        xv *= vg_u
        xi += xv                          # x = inst + voc * vocal gain
        x = xi
        del xv
        low = sosfiltfilt(lp, x)          # zero-phase bass split: (x - low) + low * bass
        low *= bass_u
        x += low
        del low
        if do_filt:
            x = _tv_filter(x[:, None], filt, BLK, sr)[:, 0]
        x *= vol_u
        if send is not None:
            wet = _echo(x[:, None], send, D)[:, 0]
            wet *= tail
            x += wet
            del wet
        out32[:, c] = x
        del x, xi
    del xi32, xv32
    st = int((t0 - pre) * sr)
    master.add(st, out32)
    return t0
